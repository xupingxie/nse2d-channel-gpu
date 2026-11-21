// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Run-time diagnostics: budgets, wall quantities, row moments, CFL maxima, optional 1D spectra.
#pragma once
#include "common.cuh"
#include "ops.cuh"

__global__ void enstrophy_diss_mapped_kernel(const Real* __restrict__ omg,
    int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node,
    const Real* __restrict__ w_node,
    Real* __restrict__ sum_w2, Real* __restrict__ sum_gw2);
void launch_enstrophy_diss_mapped(const Real* omg, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                         const Real* d_a_node, const Real* d_w_node,
                                         Real* d_sum_w2, Real* d_sum_gw2, cudaStream_t s=0);
__global__ void enstrophy_diss_mapped_full_kernel(const Real* __restrict__ omg,
    int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node,
    const Real* __restrict__ w_node,
    Real* __restrict__ sum_w2_full, Real* __restrict__ sum_gw2_full);
void launch_enstrophy_diss_mapped_full(const Real* omg, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                              const Real* d_a_node, const Real* d_w_node,
                                              Real* d_sum_w2_full, Real* d_sum_gw2_full, cudaStream_t s=0);
__global__ void kinetic_energy_mapped_kernel(const Real* __restrict__ psi,
    int Nx, int Ny, Real inv_dx, Real inv_deta, const Real* __restrict__ a_node,
    const Real* __restrict__ w_node, Real* __restrict__ out_sum);
void launch_kinetic_energy_mapped(const Real* psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                         const Real* d_a_node, const Real* d_w_node, Real* d_sum, cudaStream_t s=0);
void launch_energy_dissipation_mapped(
    const Real* d_psi, const Real* d_omega, int Nx, int Ny,
    Real inv_dx, Real inv_deta,
    const Real* d_a_node,  // [Ny]
    const Real* d_a_edge,  // [Ny-1]
    const Real* d_w_node,  // [Ny]
    Real* d_sum_eps,       // device scalar
    cudaStream_t s);
__global__ void power_input_mapped_kernel(const Real* __restrict__ psi,
    int Nx, int Ny, Real inv_deta, Real h, Real F0, int nforce,
    const Real* __restrict__ a_node, const Real* __restrict__ y_node,
    const Real* __restrict__ w_node, Real* __restrict__ sum);
void launch_power_input_mapped(const Real* psi, int Nx, int Ny, Real inv_deta, Real h, Real F0, int nforce,
                                      const Real* d_a_node, const Real* d_y_node, const Real* d_w_node,
                                      Real* d_sum, cudaStream_t s=0);
__global__ void enstrophy_input_mapped_full_kernel(
    const Real* __restrict__ omega,
    int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* __restrict__ y_node,
    const Real* __restrict__ w_node,
    Real* __restrict__ sum);
void launch_enstrophy_input_mapped_full(
    const Real* omega, int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* d_y_node, const Real* d_w_node,
    Real* d_sum, cudaStream_t s=0);
__global__ void enstrophy_input_mapped_kernel(
    const Real* __restrict__ omega, 
    int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* __restrict__ y_node,
    const Real* __restrict__ w_node, 
    Real* __restrict__ sum);
void launch_enstrophy_input_mapped(
    const Real* omega, int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* d_y_node, const Real* d_w_node,
    Real* d_sum, cudaStream_t s=0);
__global__ void row_moments_from_psi_mapped_kernel(
    const Real* __restrict__ psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node,
    Real* __restrict__ sum_u, Real* __restrict__ sum_v, Real* __restrict__ sum_u2, Real* __restrict__ sum_v2, Real* __restrict__ sum_uv);
void launch_row_moments_from_psi_mapped(const Real* psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                               const Real* d_a_node, Real* su, Real* sv, Real* su2, Real* sv2, Real* suv, cudaStream_t s=0);
__global__ void mean_wall_omega_kernel(const Real* __restrict__ omega, int Nx, int Ny, Real* __restrict__ mean_bot, Real* __restrict__ mean_top);
void launch_mean_wall_omega(const Real* omega, int Nx, int Ny, Real* d_mean_bot, Real* d_mean_top, cudaStream_t s=0);
__global__ void mean_wall_omega_omegay_mapped_kernel(
    const Real* __restrict__ omega,
    int Nx, int Ny,
    Real inv_deta,
    const Real* __restrict__ a_node,
    Real* __restrict__ mean_bot,
    Real* __restrict__ mean_top);
void launch_mean_wall_omega_omegay_mapped(
    const Real* omega, int Nx, int Ny,
    Real inv_deta,
    const Real* d_a_node,
    Real* d_mean_bot, Real* d_mean_top,
    cudaStream_t s=0);
void launch_enstrophy_diss_balance_mapped(
    const Real* d_omega,        // [Ny*Nx] vorticity field
    int Nx, int Ny,
    Real inv_dx2, Real inv_deta2,
    const Real* d_a_node,       // mapping metrics
    const Real* d_a_edge,
    const Real* d_w_node,       // quadrature weights in y
    Real* d_sum_w2,             // device scalar
    Real* d_sum_eta,            // device scalar
    cudaStream_t s = 0);
void launch_omega_rhs_balance_mapped(
    const Real* d_omega,
    const Real* d_rhs,
    int Nx, int Ny,
    const Real* d_w_node,
    Real* d_sum_omega_rhs,
    cudaStream_t stream = 0);
__global__ void max_uv_blocks_mapped_kernel( const Real* __restrict__ psi, int Nx, int Ny,
    Real inv_dx, Real inv_deta, const Real* __restrict__ a_node, Real* __restrict__ umax_blk, Real* __restrict__ vmax_blk,
    Real* __restrict__ vmaxm_blk);
__global__ void reduce_max_triple_kernel(const Real* __restrict__ umax_blk, const Real* __restrict__ vmax_blk,
    const Real* __restrict__ vmaxm_blk, int n, Real* __restrict__ umax_out, Real* __restrict__ vmax_out, Real* __restrict__ vmaxm_out);
void launch_max_uv_from_psi_mapped( const Real* d_psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* d_a_node, Real* d_umax, Real* d_vmax, Real* d_vmaxm,       // device scalars (size 1 each)
    cudaStream_t stream = 0);
__global__ void fill_kx_array_kernel(Real* kx, int Nxc, Real Lx);
__global__ void spectra1d_rowReduce_kernel(
    const cufftDoubleComplex* __restrict__ Uhat,
    const cufftDoubleComplex* __restrict__ Vhat,
    int Ny, int Nxc, Real norm1d,
    Real* __restrict__ Eu_tmp, Real* __restrict__ Ev_tmp);
__global__ void vec_add_kernel(Real* acc, const Real* tmp, int n);

// 1D energy spectra accumulator (optional run-time diagnostic)
struct Spectra1D {
    int Nx, Ny, Nxc;
    cufftHandle plan{};
    cufftDoubleComplex *d_Uhat{nullptr}, *d_Vhat{nullptr};
    Real *d_kx{nullptr};
    Real *d_Eu_acc{nullptr}, *d_Ev_acc{nullptr};
    Real *d_Eu_tmp{nullptr}, *d_Ev_tmp{nullptr};
    Real norm1d{0};
    int samples{0};

    void init(const Grid& G){
        Nx = G.Nx; Ny = G.Ny; Nxc = Nx/2 + 1;
        CUFFT_CHECK(cufftPlan1d(&plan, Nx, CUFFT_D2Z, Ny)); // batch Ny
        CUDA_CHECK(cudaMalloc(&d_Uhat, (size_t)Ny*Nxc*sizeof(cufftDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_Vhat, (size_t)Ny*Nxc*sizeof(cufftDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_kx, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_Eu_acc, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_Ev_acc, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_Eu_tmp, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_Ev_tmp, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_Eu_acc, 0, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_Ev_acc, 0, Nxc*sizeof(Real)));
        int tb=256, gb=(Nxc+tb-1)/tb;
        fill_kx_array_kernel<<<gb,tb>>>(d_kx, Nxc, G.Lx);
        CUDA_CHECK(cudaGetLastError());
        // ∫∫ 0.5(u^2+v^2) dx dy = sum_k [ 0.5 * (dx*dy/Nx) * (|Û|^2 + |V̂|^2)_full ]
        // Area-averaged spectra with cuFFT UNnormalized forward DFT (diagnostics §8.5):
        // Eu(km) = wm/(Nx^2*Ny) * sum_j |û_j(km)|^2   (and similarly for Ev)
        // so that ∑_m E(km) = <u^2+v^2>_A = 2K.
        norm1d = 1.0 / ((Real)G.Nx * (Real)G.Nx * (Real)G.Ny);
        samples = 0;
    }
    ~Spectra1D(){
        if (d_Ev_tmp) cudaFree(d_Ev_tmp);
        if (d_Eu_tmp) cudaFree(d_Eu_tmp);
        if (d_Ev_acc) cudaFree(d_Ev_acc);
        if (d_Eu_acc) cudaFree(d_Eu_acc);
        if (d_kx) cudaFree(d_kx);
        if (d_Vhat) cudaFree(d_Vhat);
        if (d_Uhat) cudaFree(d_Uhat);
        if (plan) cufftDestroy(plan);
    }
    void accumulate(Real* d_u, Real* d_v){
        CUFFT_CHECK(cufftExecD2Z(plan, (cufftDoubleReal*)d_u, d_Uhat));
        CUFFT_CHECK(cufftExecD2Z(plan, (cufftDoubleReal*)d_v, d_Vhat));
        int tb=256, gb=(Nxc+tb-1)/tb;
        spectra1d_rowReduce_kernel<<<gb,tb>>>(d_Uhat,d_Vhat,Ny,Nxc,norm1d,d_Eu_tmp,d_Ev_tmp);
        CUDA_CHECK(cudaGetLastError());
        vec_add_kernel<<<gb,tb>>>(d_Eu_acc,d_Eu_tmp,Nxc);
        vec_add_kernel<<<gb,tb>>>(d_Ev_acc,d_Ev_tmp,Nxc);
        samples++;
    }
    void write_csv(const std::string& path){
        std::vector<Real> Hkx(Nxc), Heu(Nxc), Hev(Nxc);
        CUDA_CHECK(cudaMemcpy(Hkx.data(), d_kx, Nxc*sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Heu.data(), d_Eu_acc, Nxc*sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Hev.data(), d_Ev_acc, Nxc*sizeof(Real), cudaMemcpyDeviceToHost));
        if (samples>0){
            for (int k=0;k<Nxc;k++){ Heu[k]/=samples; Hev[k]/=samples; }
        }
        std::ofstream f(path);
        f.setf(std::ios::scientific); f.precision(10);
        f << "kx,E_u,E_v,E_tot\n";
        for (int k=0;k<Nxc;k++){
            Real Etot = Heu[k] + Hev[k];
            f << Hkx[k] << "," << Heu[k] << "," << Hev[k] << "," << Etot << "\n";
        }
    }
};

