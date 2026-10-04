// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// FFT (x) + batched tridiagonal (y) Poisson solver for the streamfunction.
#include "poisson.cuh"


/* -------------------- Z2Z Poisson (x-FFT + y-tridiagonal) -------------------- */
using Cx = cufftDoubleComplex;
__global__ void real_to_complex_kernel(const Real* __restrict__ r, Cx* __restrict__ z, size_t N){
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N){ z[i].x = r[i]; z[i].y = 0.0; }
}
__global__ void complex_to_real_scaled_kernel(const Cx* __restrict__ z, Real* __restrict__ r, size_t N, Real s){
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N){ r[i] = s * z[i].x; }
}
__global__ void fill_kx2_z2z_kernel(Real* kx2, int Nx, Real Lx){
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= Nx) return;
    int kphys_i = (k <= Nx/2 ? k : k - Nx);
    Real kphys = (2.0*CUDART_PI * (Real)kphys_i) / Lx;
    kx2[k] = kphys*kphys;
}
__global__ void precompute_cp_kernel(Real* __restrict__ cp, const Real* __restrict__ kx2, int Nx, int Ny, Real inv_dy2){
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= Nx) return;
    int Ni = Ny-2;
    Real a = inv_dy2, c = inv_dy2, b = -2.0*inv_dy2 - kx2[k];
    Real cp_prev = c / b;
    cp[(size_t)k*Ni + 0] = cp_prev;
    for (int j=1; j<Ni; ++j){
        Real den = b - a*cp_prev;
        Real cpi = c/den;
        cp[(size_t)k*Ni + j] = cpi;
        cp_prev = cpi;
    }
}

__global__ void precompute_cp_mapped_kernel(Real* __restrict__ cp,
    const Real* __restrict__ a_node, const Real* __restrict__ a_edge, const Real* __restrict__ kx2,
    int Nx, int Ny, Real inv_deta2)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= Nx) return;
    int Ni = Ny - 2;
    // Row j=1 (index 0)
    int j = 1;
    Real an = a_node[j];
    Real ae_p = a_edge[j];     // j+1/2
    Real ae_m = a_edge[j-1];   // j-1/2
    Real a_sub = an * ae_m * inv_deta2;
    Real c_sup = an * ae_p * inv_deta2;
    Real b = - (a_sub + c_sup) - kx2[k];
    Real cp_prev = c_sup / b;
    cp[(size_t)k*Ni + 0] = cp_prev;
    for (int r=1; r<Ni; ++r){
        j = r+1;
        an = a_node[j];
        ae_p = a_edge[j];
        ae_m = a_edge[j-1];
        a_sub = an * ae_m * inv_deta2;
        c_sup = an * ae_p * inv_deta2;
        b = - (a_sub + c_sup) - kx2[k];
        Real den = b - a_sub * cp_prev;
        Real cpi = c_sup / den;
        cp[(size_t)k*Ni + r] = cpi;
        cp_prev = cpi;
    }
}

__global__ void poisson_tridiag_batched_mapped_kernel(
    const Cx* __restrict__ omega_hat, Cx* __restrict__ psi_hat,
    const Real* __restrict__ cp, const Real* __restrict__ a_node, const Real* __restrict__ a_edge,
    const Real* __restrict__ kx2, int Nx, int Ny, Real inv_deta2, Real psi_bot, Real psi_top)
{
    int k = blockIdx.x;
    if (k >= Nx) return;
    int Ni = Ny - 2;
    const Cx* wh = omega_hat + k;
    Cx* ph       = psi_hat   + k;

    // Row j=1 (index 0)
    int j = 1;
    Real an = a_node[j];
    Real ae_p = a_edge[j];
    Real ae_m = a_edge[j-1];
    Real a_sub = an * ae_m * inv_deta2;
    Real c_sup = an * ae_p * inv_deta2;
    Real b = - (a_sub + c_sup) - kx2[k];

    Cx dp_prev; dp_prev.x=0.0; dp_prev.y=0.0;
    Cx rhs0 = wh[Nx*j]; rhs0.x = -rhs0.x; rhs0.y = -rhs0.y;
    // *** FIX: wall ψ contribution only for kx = 0 ***
    if (k == 0){
        rhs0.x -= a_sub * psi_bot * (Real)Nx;
    }

    Real den0 = b;
    Cx dp0; dp0.x = (rhs0.x - a_sub*dp_prev.x)/den0; dp0.y = (rhs0.y - a_sub*dp_prev.y)/den0;
    ph[Nx*j] = dp0; dp_prev = dp0;

    for (int r=1; r<Ni; ++r){
        j = r+1;
        an = a_node[j];
        ae_p = a_edge[j];
        ae_m = a_edge[j-1];
        a_sub = an * ae_m * inv_deta2;
        c_sup = an * ae_p * inv_deta2;
        b = - (a_sub + c_sup) - kx2[k];
        Cx rhs = wh[Nx*j]; rhs.x = -rhs.x; rhs.y = -rhs.y;

        // *** FIX: top-wall contribution only for kx = 0, last row ***
        if (k == 0 && r == Ni-1){
            rhs.x -= c_sup * psi_top * (Real)Nx;
        }

        Real cpj = cp[(size_t)k*Ni + r];
        Real den = c_sup / cpj; // equals b - a_sub * cp_prev
        Cx dpj; dpj.x = (rhs.x - a_sub*dp_prev.x)/den; dpj.y = (rhs.y - a_sub*dp_prev.y)/den;
        ph[Nx*j] = dpj; dp_prev = dpj;
    }
    Cx xnext = ph[Nx*(Ni)];
    for (int r=Ni-2; r>=0; --r){
        j = r+1;
        Real cpj = cp[(size_t)k*Ni + r];
        Cx dpj = ph[Nx*j];
        Cx xj; xj.x = dpj.x - cpj*xnext.x; xj.y = dpj.y - cpj*xnext.y;
        ph[Nx*j] = xj;
        xnext = xj;
    }
    Cx zero; zero.x=0.0; zero.y=0.0;
    ph[0] = (k==0 ? Cx{ (Real)Nx*psi_bot, 0.0 } : zero);
    ph[Nx*(Ny-1)] = (k==0 ? Cx{ (Real)Nx*psi_top, 0.0 } : zero);
}

// -------------------- Poisson (FFT in x; tri-diagonal in y) --------------------

