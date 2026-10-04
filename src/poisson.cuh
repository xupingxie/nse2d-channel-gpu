// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// FFT (x) + batched tridiagonal (y) Poisson solver for the streamfunction.
#pragma once
#include "common.cuh"

__global__ void real_to_complex_kernel(const Real* __restrict__ r, Cx* __restrict__ z, size_t N);
__global__ void complex_to_real_scaled_kernel(const Cx* __restrict__ z, Real* __restrict__ r, size_t N, Real s);
__global__ void fill_kx2_z2z_kernel(Real* kx2, int Nx, Real Lx);
__global__ void precompute_cp_kernel(Real* __restrict__ cp, const Real* __restrict__ kx2, int Nx, int Ny, Real inv_dy2);
__global__ void precompute_cp_mapped_kernel(Real* __restrict__ cp,
    const Real* __restrict__ a_node, const Real* __restrict__ a_edge, const Real* __restrict__ kx2,
    int Nx, int Ny, Real inv_deta2);
__global__ void poisson_tridiag_batched_mapped_kernel(
    const Cx* __restrict__ omega_hat, Cx* __restrict__ psi_hat,
    const Real* __restrict__ cp, const Real* __restrict__ a_node, const Real* __restrict__ a_edge,
    const Real* __restrict__ kx2, int Nx, int Ny, Real inv_deta2, Real psi_bot, Real psi_top);

struct Poisson2D_Z2Z {
    int Nx, Ny;
    cufftHandle fwd{}, inv{};
    Real *d_kx2{nullptr}, *d_cp{nullptr};
    Cx *d_hat_in{nullptr}, *d_hat_out{nullptr};
    bool profile{false};
    float acc_fft_ms{0.0f};
    float acc_tridiag_ms{0.0f};

    // mapped support (external variable coefficients; not owned)
    bool mapped{false};
    bool cp_mapped_ready{false};
    const Real *d_a_node{nullptr}, *d_a_edge{nullptr};
    Real inv_deta2{0.0};

    // Constructor: build FFT plans, kx^2, and cp for a uniform grid
    Poisson2D_Z2Z(const Grid& G, cudaStream_t s=0) : Nx(G.Nx), Ny(G.Ny){
   	    // 1D FFT in x, batched over Ny rows
	    CUFFT_CHECK(cufftPlan1d(&fwd, Nx, CUFFT_Z2Z, Ny));
	    CUFFT_CHECK(cufftPlan1d(&inv, Nx, CUFFT_Z2Z, Ny));
	    CUFFT_CHECK(cufftSetStream(fwd,s));
	    CUFFT_CHECK(cufftSetStream(inv,s));
	    // kx^2 array
	    CUDA_CHECK(cudaMalloc(&d_kx2, (size_t)Nx*sizeof(Real)));
	    int threads = 256, blocks = (Nx + threads - 1)/threads;
	    fill_kx2_z2z_kernel<<<blocks,threads,0,s>>>(d_kx2, Nx, G.Lx);
	    CUDA_CHECK(cudaGetLastError());
            // tri-diagonal forward thomas coefficient cp(k,j), j=1...Ny-2
	    const int Ni = Ny - 2;
	    CUDA_CHECK(cudaMalloc(&d_cp, (size_t)Nx*Ni*sizeof(Real)));

	    // Fallback uniform coefficients using physical dy; for stretched grids we overwrite cp in enable_mapped()
	    precompute_cp_kernel<<<blocks,threads,0,s>>>( d_cp, d_kx2, Nx, Ny, 1.0/(G.dy()*G.dy()));
	    CUDA_CHECK(cudaGetLastError());
            // spectral work buffers
	    CUDA_CHECK(cudaMalloc(&d_hat_in,  (size_t)Nx*Ny*sizeof(Cx)));
	    CUDA_CHECK(cudaMalloc(&d_hat_out, (size_t)Nx*Ny*sizeof(Cx)));
    }
    ~Poisson2D_Z2Z() {
        if (d_hat_out) cudaFree(d_hat_out);
        if (d_hat_in)  cudaFree(d_hat_in);
        if (d_cp)      cudaFree(d_cp);
        if (d_kx2)     cudaFree(d_kx2);
        if (fwd) cufftDestroy(fwd);
        if (inv) cufftDestroy(inv);
    }

    inline void set_profile(bool on){ profile = on; }
    inline void profile_reset(){ acc_fft_ms = 0.0f; acc_tridiag_ms = 0.0f; }
    inline void profile_get(float& fft_ms, float& tridiag_ms){ fft_ms = acc_fft_ms; tridiag_ms = acc_tridiag_ms; }
    
    // attach mapped metrics and precompute cp(k,j) for the mapped laplacian.
    // inv_deta is 1/Delta eta on the computational eta in [-1, 1] grid.
    inline void enable_mapped(const Real* a_node, const Real* a_edge, Real inv_deta, cudaStream_t s=0){
	    mapped = true; d_a_node = a_node; d_a_edge = a_edge; inv_deta2 = inv_deta*inv_deta;

	    int threads = 256, blocks = (Nx + threads - 1)/threads;
	    // Precompute cp(k,j) ONCE for the mapped Laplacian; reused every solve().
	    precompute_cp_mapped_kernel<<<blocks,threads,0,s>>>(d_cp, d_a_node, d_a_edge, d_kx2, Nx, Ny, inv_deta2);
	    CUDA_CHECK(cudaGetLastError());

	    cp_mapped_ready = true;
    }

    // NOTE: psi_bot/top Dirichlet enforced on kx=0 mode in spectral space Solve -∇²ψ = ω with ψ(y=±h) = psi_bot/top (Dirichlet in kx=0 mode).
    void solve(const Real* d_omega, Real* d_psi, Real dy, Real psi_bot, Real psi_top, cudaStream_t s=0){
		// dy is for API compatibility of uniform case, not used in mapped path
        size_t N = (size_t)Nx*Ny;
        int threads=256, blocks=(N+threads-1)/threads;

        cudaEvent_t e0,e1,e2,e3; if (profile){ cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3); }

        // pack real->complex
        real_to_complex_kernel<<<blocks,threads,0,s>>>(d_omega, d_hat_in, N);
        CUDA_CHECK(cudaGetLastError());

        // forward FFT in x
        if (profile) cudaEventRecord(e0, s);
        CUFFT_CHECK(cufftExecZ2Z(fwd, d_hat_in, d_hat_in, CUFFT_FORWARD));
        if (profile){ cudaEventRecord(e1, s); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); acc_fft_ms += ms; }

        // batched tri-diagonal solve in y for each kx
        if (profile) cudaEventRecord(e2, s);
	dim3 gb(Nx), tb(1);
	// cp already precomputed (uniform constructor or enable_mapped());
        // just use the mapped tridiagonal solve
	poisson_tridiag_batched_mapped_kernel<<<gb,tb,0,s>>>(d_hat_in, d_hat_in, d_cp, d_a_node, d_a_edge, d_kx2, Nx, Ny, inv_deta2, psi_bot, psi_top);
        CUDA_CHECK(cudaGetLastError());
        if (profile){ cudaEventRecord(e3, s); cudaEventSynchronize(e3); float ms; cudaEventElapsedTime(&ms, e2, e3); acc_tridiag_ms += ms; }

        // inverse FFT
        if (profile) cudaEventRecord(e2, s);
        CUFFT_CHECK(cufftExecZ2Z(inv, d_hat_in, d_hat_out, CUFFT_INVERSE));
        if (profile){ cudaEventRecord(e3, s); cudaEventSynchronize(e3); float ms; cudaEventElapsedTime(&ms, e2, e3); acc_fft_ms += ms; }

        // unpack and scale by 1/Nx (cuFFT convention)
        complex_to_real_scaled_kernel<<<blocks,threads,0,s>>>(d_hat_out, d_psi, N, (Real)(1.0/(Real)Nx));
        CUDA_CHECK(cudaGetLastError());

        if (profile){ cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3); }
    }
};

