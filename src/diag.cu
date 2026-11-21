// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Run-time diagnostics: budgets, wall quantities, row moments, CFL maxima, optional 1D spectra.
#include "diag.cuh"


__global__ void enstrophy_diss_mapped_kernel(const Real* __restrict__ omg,
    int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node,
    const Real* __restrict__ w_node,
    Real* __restrict__ sum_w2, Real* __restrict__ sum_gw2)
{
    __shared__ Real sbw[256]; __shared__ Real sbg[256];
    size_t tid=threadIdx.x; Real lw2=0.0, lg2=0.0; size_t N=(size_t)Nx*Ny;
    for (size_t idx=blockIdx.x*blockDim.x + tid; idx<N; idx+=gridDim.x*blockDim.x){
        int ix=idx % Nx, iy=idx / Nx; if (iy==0 || iy==Ny-1) continue;
        int xm=(ix==0? Nx-1:ix-1), xp=(ix==Nx-1? 0:ix+1);
        int ym=iy-1, yp=iy+1;
        Real w = omg[idx];
        Real wx = (omg[(size_t)iy*Nx + xp] - omg[(size_t)iy*Nx + xm]) * 0.5 * inv_dx;
        Real wy = a_node[iy] * (omg[(size_t)yp*Nx + ix] - omg[(size_t)ym*Nx + ix]) * 0.5 * inv_deta;
        Real wt = w_node[iy];
        lw2 += wt * (w*w);
        lg2 += wt * (wx*wx + wy*wy);
    }
    sbw[tid]=lw2; sbg[tid]=lg2; __syncthreads();
    for (unsigned s=blockDim.x/2; s>0; s>>=1){ if (tid<s){ sbw[tid]+=sbw[tid+s]; sbg[tid]+=sbg[tid+s]; } __syncthreads(); }
    if (tid==0){ atomicAdd(sum_w2,sbw[0]); atomicAdd(sum_gw2,sbg[0]); }
}
void launch_enstrophy_diss_mapped(const Real* omg, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                         const Real* d_a_node, const Real* d_w_node,
                                         Real* d_sum_w2, Real* d_sum_gw2, cudaStream_t s)
{
    int threads=256; int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum_w2,0,sizeof(Real),s));
    CUDA_CHECK(cudaMemsetAsync(d_sum_gw2,0,sizeof(Real),s));
    enstrophy_diss_mapped_kernel<<<blocks,threads,0,s>>>(omg,Nx,Ny,inv_dx,inv_deta,d_a_node,d_w_node,d_sum_w2,d_sum_gw2);
    CUDA_CHECK(cudaGetLastError());
}
// Full-domain enstrophy and palinstrophy (includes wall rows).
// Uses centered differences in the interior and 2nd-order one-sided in η at the walls.
// ω_x: centered periodic in x everywhere.
// ω_y: ω_y = a_node[j] * ω_η;  ω_η is one-sided at j=0,Ny-1 and centered otherwise.
__global__ void enstrophy_diss_mapped_full_kernel(const Real* __restrict__ omg,
    int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node,
    const Real* __restrict__ w_node,
    Real* __restrict__ sum_w2_full, Real* __restrict__ sum_gw2_full)
{
    __shared__ Real sbw[256]; __shared__ Real sbg[256];
    size_t tid=threadIdx.x; Real lw2=0.0, lg2=0.0; size_t N=(size_t)Nx*Ny;
    for (size_t idx=blockIdx.x*blockDim.x + tid; idx<N; idx+=gridDim.x*blockDim.x){
        int ix=idx % Nx, iy=(int)(idx / Nx);
        int xm=(ix==0? Nx-1:ix-1), xp=(ix==Nx-1? 0:ix+1);

        const Real w = omg[idx];

        // ω_x: centered periodic
        const Real wx = (omg[(size_t)iy*Nx + xp] - omg[(size_t)iy*Nx + xm]) * (Real)0.5 * inv_dx;

        // ω_y in physical y: ω_y = a(j) * ω_η
        Real weta;
        if (iy == 0){
            // 2nd-order one-sided in η at bottom wall (uses j=0,1,2)
            const Real w0 = omg[(size_t)0*Nx + ix];
            const Real w1 = omg[(size_t)1*Nx + ix];
            const Real w2 = omg[(size_t)2*Nx + ix];
            weta = (-3.0*w0 + 4.0*w1 - 1.0*w2) * ((Real)0.5 * inv_deta);
        } else if (iy == Ny-1){
            // 2nd-order one-sided in η at top wall (uses j=Ny-1,Ny-2,Ny-3)
            const Real wN   = omg[(size_t)(Ny-1)*Nx + ix];
            const Real wNm1 = omg[(size_t)(Ny-2)*Nx + ix];
            const Real wNm2 = omg[(size_t)(Ny-3)*Nx + ix];
            weta = ( 3.0*wN - 4.0*wNm1 + 1.0*wNm2) * ((Real)0.5 * inv_deta);
        } else {
            int ym=iy-1, yp=iy+1;
            weta = (omg[(size_t)yp*Nx + ix] - omg[(size_t)ym*Nx + ix]) * (Real)0.5 * inv_deta;
        }
        const Real wy = a_node[iy] * weta;

        const Real wt = w_node[iy];
        lw2 += wt * (w*w);
        lg2 += wt * (wx*wx + wy*wy);
    }
    sbw[tid]=lw2; sbg[tid]=lg2; __syncthreads();
    for (unsigned s=blockDim.x/2; s>0; s>>=1){
        if (tid<s){ sbw[tid]+=sbw[tid+s]; sbg[tid]+=sbg[tid+s]; }
        __syncthreads();
    }
    if (tid==0){ atomicAdd(sum_w2_full,sbw[0]); atomicAdd(sum_gw2_full,sbg[0]); }
}

void launch_enstrophy_diss_mapped_full(const Real* omg, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                              const Real* d_a_node, const Real* d_w_node,
                                              Real* d_sum_w2_full, Real* d_sum_gw2_full, cudaStream_t s)
{
    int threads=256; int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum_w2_full, 0, sizeof(Real), s));
    CUDA_CHECK(cudaMemsetAsync(d_sum_gw2_full, 0, sizeof(Real), s));
    enstrophy_diss_mapped_full_kernel<<<blocks,threads,0,s>>>(omg,Nx,Ny,inv_dx,inv_deta,d_a_node,d_w_node,d_sum_w2_full,d_sum_gw2_full);
    CUDA_CHECK(cudaGetLastError());
}
// Enstrophy dissipation using -ω Δω (wall-balanced, works for no-slip or free-slip)
__global__ void enstrophy_diss_balance_mapped_kernel(
    const Real* __restrict__ omega,      // ω
    const Real* __restrict__ lap_omega,  // ∇²ω with the same stencil as RHS
    int Nx, int Ny,
    const Real* __restrict__ w_node,     // quadrature weights in y
    Real* __restrict__ sum_w2,           // accumulator for ∑ w_node ω²
    Real* __restrict__ sum_eta)          // accumulator for ∑ w_node (-ω Δω)
{
    __shared__ Real sbw[256];
    __shared__ Real sbe[256];

    const int    tid = threadIdx.x;
    const size_t N   = (size_t)Nx * (size_t)Ny;

    Real lw2  = 0.0;
    Real leta = 0.0;

    // grid‑stride loop over all nodes
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
         idx < N;
         idx += (size_t)gridDim.x * blockDim.x)
    {
        int j = (int)(idx / Nx);          // y-index

        // match Ω definition: interior only (j=1..Ny-2)
        if (j == 0 || j == Ny - 1) continue;

        Real w   = omega[idx];
        Real lap = lap_omega[idx];
        Real wt  = w_node[j];

        lw2  += wt * (w * w);
        // + sign: η = ⟨ω Δω⟩
        leta += wt * (w * lap);
    }

    sbw[tid] = lw2;
    sbe[tid] = leta;
    __syncthreads();

    // block reduction
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sbw[tid] += sbw[tid + s];
            sbe[tid] += sbe[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_w2,  sbw[0]);
        atomicAdd(sum_eta, sbe[0]);
    }
}

__global__ void kinetic_energy_mapped_kernel(const Real* __restrict__ psi,
    int Nx, int Ny, Real inv_dx, Real inv_deta, const Real* __restrict__ a_node,
    const Real* __restrict__ w_node, Real* __restrict__ out_sum)
{
    __shared__ Real sb[256];
    size_t tid=threadIdx.x; Real sum=0.0; size_t N=(size_t)Nx*Ny;
    for (size_t idx=blockIdx.x*blockDim.x + tid; idx<N; idx+=gridDim.x*blockDim.x){
        int ix=idx % Nx, iy=idx / Nx;
        // Skip wall rows 
        if (iy == 0 || iy == Ny-1) continue;

        int xm=(ix==0? Nx-1:ix-1), xp=(ix==Nx-1? 0:ix+1);
       	//int ym=(iy==0? 0:iy-1), yp=(iy==Ny-1? Ny-1:iy+1);
        int ym = iy-1, yp = iy + 1;
	Real u=(psi[(size_t)yp*Nx + ix]-psi[(size_t)ym*Nx + ix]) * 0.5 * inv_deta * a_node[iy];
        Real v=-(psi[(size_t)iy*Nx + xp]-psi[(size_t)iy*Nx + xm]) * 0.5 * inv_dx;
	
        sum += 0.5*(u*u + v*v) * w_node[iy];
    }
    sb[tid]=sum; __syncthreads();
    for (unsigned s=blockDim.x/2; s>0; s>>=1){ if (tid<s) sb[tid]+=sb[tid+s]; __syncthreads(); }
    if (tid==0) atomicAdd(out_sum,sb[0]);
}
void launch_kinetic_energy_mapped(const Real* psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                         const Real* d_a_node, const Real* d_w_node, Real* d_sum, cudaStream_t s)
{
    int threads=256; int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum,0,sizeof(Real),s));
    kinetic_energy_mapped_kernel<<<blocks,threads,0,s>>>(psi,Nx,Ny,inv_dx,inv_deta,d_a_node,d_w_node,d_sum);
    CUDA_CHECK(cudaGetLastError());
}
// Index helper
#ifndef IDX
#define IDX(j,i) ((size_t)(j) * (size_t)Nx + (size_t)(i))
#endif
// energy_dissipation_mapped_kernel:
// Computes sum_yx w_node[j] * ( |∇u|^2 + |∇v|^2 ), where
//   u = ψ_y = a(j) * ψ_η,  v = -ψ_x
// with:
//   u_x = a(j) * ∂η(ψ_x)          (cross derivative, j±1 only)
//   v_y = -u_x                    (same cross stencil → discrete incompressibility)
//   v_x = -ψ_xx                   (true second derivative in x)
//   u_y = a(j)*[ a_{j+1/2}(ψ_{j+1}-ψ_j) - a_{j-1/2}(ψ_j-ψ_{j-1}) ] / Δη²   (fully conservative)
//
// Notes:
// - Skips wall rows j=0 and j=Ny-1 (no-slip). If you prefer to include walls,
//   you can add one-sided variants, though contribution is tiny there.
// - Uses dynamic shared memory sized to blockDim.x for reduction.
// - Requires: a_node[j] (∂η/∂y at nodes), a_edge[j] (∂η/∂y at j+1/2), w_node[j] (∼dy trapezoid).
__global__ void energy_dissipation_mapped_kernel(
    const Real* __restrict__ psi,     // [Ny*Nx]
    const Real* __restrict__ omega,   // [Ny*Nx] (needed for wall contribution via u_y|wall = -ω_wall)
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node,  // [Ny]
    const Real* __restrict__ a_edge,  // [Ny-1], where a_edge[j] = a at (j+1/2)
    const Real* __restrict__ w_node,  // [Ny] trapezoid-like y weights(~dy)
    Real* __restrict__ sum_eps)       // scalar accumulator (device)
{
    extern __shared__ Real sb[];      // sb[blockDim.x]
    const int tid = threadIdx.x;
    const size_t N = (size_t)Nx * (size_t)Ny;

    Real loc = 0.0;
    const Real inv_dx2   = inv_dx   * inv_dx;
    const Real inv_deta2 = inv_deta * inv_deta;

    // Grid-stride loop
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
         idx < N;
         idx += (size_t)gridDim.x * blockDim.x)
    {
        const int j = (int)(idx / Nx);
        const int i = (int)(idx - (size_t)j * Nx);

        // Include wall rows in the quadrature using no-slip kinematics:
        // u=v=0 on y=±h ⇒ u_x=v_x=0 and by incompressibility v_y=0, so |∇u|^2 reduces to (u_y)^2.
        // With ω = v_x - u_y and v_x|wall=0 ⇒ u_y|wall = -ω_wall.
        if (j==0 || j==Ny-1){
            const Real uy = -omega[idx];
            loc += (uy*uy) * w_node[j];
            continue;
        }       
        const int im = (i == 0     ? Nx - 1 : i - 1);
        const int ip = (i == Nx-1  ? 0      : i + 1);
        // Neighbor linear indices
        const size_t c     = IDX(j, i);
        const size_t c_ip  = IDX(j, ip);
        const size_t c_im  = IDX(j, im);
        const size_t jp_i  = IDX(j+1, i);
        const size_t jm_i  = IDX(j-1, i);
        const size_t jp_ip = IDX(j+1, ip);
        const size_t jp_im = IDX(j+1, im);
        const size_t jm_ip = IDX(j-1, ip);
        const size_t jm_im = IDX(j-1, im);
        
        // Mapping metrics
        const Real a  = a_node[j];
        const Real ap = a_edge[j];     // j+1/2
        const Real am = a_edge[j-1];   // j-1/2

        // ψ_x at j±1 (periodic in x)
        const Real psi_x_jp = (psi[jp_ip] - psi[jp_im]) * (0.5 * inv_dx);
        const Real psi_x_jm = (psi[jm_ip] - psi[jm_im]) * (0.5 * inv_dx);

        // Cross derivatives: u_x = a * ∂η(ψ_x),  v_y = -u_x
        const Real ux = a * (psi_x_jp - psi_x_jm) * (0.5 * inv_deta);
        const Real vy = -ux;

        // v_x = -ψ_xx (true second derivative in x)
        const Real vx = -(psi[c_ip] - 2.0*psi[c] + psi[c_im]) * inv_dx2;

        // u_y = a * [ ap*(ψ_{j+1}-ψ_j) - am*(ψ_j-ψ_{j-1}) ] / Δη²   (fully conservative)
        const Real uy = a * ( ap * (psi[jp_i] - psi[c]) - am * (psi[c] - psi[jm_i]) ) * inv_deta2;

        // Accumulate contribution with y-quadrature weight
        const Real g2 = ux*ux + uy*uy + vx*vx + vy*vy;
        loc += g2 * w_node[j];
    }

    // block reduction (sum)
    sb[tid] = loc;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sb[tid] += sb[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(sum_eps, sb[0]);
}
void launch_energy_dissipation_mapped(
    const Real* d_psi, const Real* d_omega, int Nx, int Ny,
    Real inv_dx, Real inv_deta,
    const Real* d_a_node,  // [Ny]
    const Real* d_a_edge,  // [Ny-1]
    const Real* d_w_node,  // [Ny]
    Real* d_sum_eps,       // device scalar
    cudaStream_t s)
{
    const int threads = 256;
    const size_t N    = (size_t)Nx * (size_t)Ny;
    const int blocks  = (int)std::min<size_t>(8192, (N + threads - 1) / threads);
    const size_t shmem = threads * sizeof(Real);

    CUDA_CHECK(cudaMemsetAsync(d_sum_eps, 0, sizeof(Real), s));
    energy_dissipation_mapped_kernel<<<blocks, threads, shmem, s>>>(
        d_psi, d_omega, Nx, Ny, inv_dx, inv_deta, d_a_node, d_a_edge, d_w_node, d_sum_eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void power_input_mapped_kernel(const Real* __restrict__ psi,
    int Nx, int Ny, Real inv_deta, Real h, Real F0, int nforce,
    const Real* __restrict__ a_node, const Real* __restrict__ y_node,
    const Real* __restrict__ w_node, Real* __restrict__ sum)
{
    __shared__ Real sb[256]; size_t tid=threadIdx.x; Real loc=0.0; size_t N=(size_t)Nx*Ny;
    for (size_t idx=blockIdx.x*blockDim.x + tid; idx<N; idx+=gridDim.x*blockDim.x){
        int ix=idx % Nx, iy=idx / Nx;
        int ym=(iy==0? 0:iy-1), yp=(iy==Ny-1? Ny-1:iy+1);
        Real u = (psi[(size_t)yp*Nx + ix]-psi[(size_t)ym*Nx + ix]) * 0.5 * inv_deta * a_node[iy];
        Real y = y_node[iy];
        Real Fx = F0 * sin((Real)nforce * CUDART_PI * ((y + h)/(2.0*h)));
        loc += u*Fx * w_node[iy];
    }
    sb[tid]=loc; __syncthreads();
    for (unsigned s=blockDim.x/2; s>0; s>>=1){ if (tid<s) sb[tid]+=sb[tid+s]; __syncthreads(); }
    if (tid==0) atomicAdd(sum,sb[0]);
}
void launch_power_input_mapped(const Real* psi, int Nx, int Ny, Real inv_deta, Real h, Real F0, int nforce,
                                      const Real* d_a_node, const Real* d_y_node, const Real* d_w_node,
                                      Real* d_sum, cudaStream_t s)
{
    int threads=256; int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum,0,sizeof(Real),s));
    power_input_mapped_kernel<<<blocks,threads,0,s>>>(psi,Nx,Ny,inv_deta,h,F0,nforce,d_a_node,d_y_node,d_w_node,d_sum);
    CUDA_CHECK(cudaGetLastError());
}
// Full-domain enstrophy input P_Ω = ⟨ ω f_ω ⟩_A (includes wall rows for consistent area quadrature).
__global__ void enstrophy_input_mapped_full_kernel(
    const Real* __restrict__ omega,
    int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* __restrict__ y_node,
    const Real* __restrict__ w_node,
    Real* __restrict__ sum)
{
    __shared__ Real sb[256];
    size_t tid = threadIdx.x;
    Real loc = 0.0;
    size_t N = (size_t)Nx * (size_t)Ny;

    for (size_t idx = blockIdx.x*blockDim.x + tid; idx < N; idx += gridDim.x*blockDim.x){
        int iy = (int)(idx / Nx);
        const Real w = omega[idx];
        const Real y = y_node[iy];

        // f_ω = -∂F_x/∂y where F_x = F0 sin(nπ(y+h)/(2h))
        const Real arg = (Real)nforce * CUDART_PI * ((y + h)/(2.0*h));
        const Real f_omega = -F0 * ((Real)nforce*CUDART_PI/(2.0*h)) * cos(arg);

        loc += w * f_omega * w_node[iy];
    }

    sb[tid] = loc;
    __syncthreads();
    for (unsigned ssz=blockDim.x/2; ssz>0; ssz>>=1){
        if (tid<ssz) sb[tid] += sb[tid+ssz];
        __syncthreads();
    }
    if (tid==0) atomicAdd(sum, sb[0]);
}

void launch_enstrophy_input_mapped_full(
    const Real* omega, int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* d_y_node, const Real* d_w_node,
    Real* d_sum, cudaStream_t s)
{
    int threads=256;
    int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(Real), s));
    enstrophy_input_mapped_full_kernel<<<blocks,threads,0,s>>>(
        omega, Nx, Ny, h, F0, nforce, d_y_node, d_w_node, d_sum);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void enstrophy_input_mapped_kernel(
    const Real* __restrict__ omega, 
    int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* __restrict__ y_node,
    const Real* __restrict__ w_node, 
    Real* __restrict__ sum)
{
    __shared__ Real sb[256]; 
    size_t tid=threadIdx.x; 
    Real loc=0.0; 
    size_t N=(size_t)Nx*Ny;
    
    for (size_t idx=blockIdx.x*blockDim.x + tid; idx<N; idx+=gridDim.x*blockDim.x){
	int iy=idx / Nx;
	if (iy==0 || iy==Ny-1) continue;  // interior only, matches Ω/η	
        Real w = omega[idx];
        Real y = y_node[iy];
        
        // f_ω = -∂F_x/∂y where F_x = F0 sin(nπ(y+h)/(2h))
        // f_ω = -F0 * (nπ/(2h)) * cos(nπ(y+h)/(2h))
        Real arg = (Real)nforce * CUDART_PI * ((y + h)/(2.0*h));
        Real f_omega = -F0 * ((Real)nforce*CUDART_PI/(2.0*h)) * cos(arg);
        loc += w * f_omega * w_node[iy];
    }
    
    sb[tid]=loc; 
    __syncthreads();
    for (unsigned s=blockDim.x/2; s>0; s>>=1){ 
        if (tid<s) sb[tid]+=sb[tid+s]; 
        __syncthreads(); 
    }
    if (tid==0) atomicAdd(sum, sb[0]);
}
void launch_enstrophy_input_mapped(
    const Real* omega, int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* d_y_node, const Real* d_w_node,
    Real* d_sum, cudaStream_t s)
{
    int threads=256; 
    int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum,0,sizeof(Real),s));
    enstrophy_input_mapped_kernel<<<blocks,threads,0,s>>>(
        omega,Nx,Ny,h,F0,nforce,d_y_node,d_w_node,d_sum);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void row_moments_from_psi_mapped_kernel(
    const Real* __restrict__ psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node,
    Real* __restrict__ sum_u, Real* __restrict__ sum_v, Real* __restrict__ sum_u2, Real* __restrict__ sum_v2, Real* __restrict__ sum_uv)
{
    int iy = blockIdx.x;
    if (iy >= Ny) return;
    extern __shared__ Real s[];
    Real* su=s; Real* sv=s+blockDim.x; Real* su2=s+2*blockDim.x; Real* sv2=s+3*blockDim.x; Real* suv=s+4*blockDim.x;
    size_t tid=threadIdx.x;
    Real lu=0.0, lv=0.0, lu2=0.0, lv2=0.0, luv=0.0;
    for (int ix=threadIdx.x; ix<Nx; ix+=blockDim.x){
        int xm=(ix==0? Nx-1:ix-1), xp=(ix==Nx-1? 0:ix+1);
        int ym=(iy==0? 0:iy-1), yp=(iy==Ny-1? Ny-1:iy+1);
        Real u = (psi[(size_t)yp*Nx + ix] - psi[(size_t)ym*Nx + ix]) * 0.5 * inv_deta * a_node[iy];
        Real v = -(psi[(size_t)iy*Nx + xp]-psi[(size_t)iy*Nx + xm]) * 0.5 * inv_dx;
        lu+=u; lv+=v; lu2+=u*u; lv2+=v*v; luv+=u*v;
    }
    su[tid]=lu; sv[tid]=lv; su2[tid]=lu2; sv2[tid]=lv2; suv[tid]=luv; __syncthreads();
    for (unsigned ssz=blockDim.x/2; ssz>0; ssz>>=1){
        if (tid<ssz){ su[tid]+=su[tid+ssz]; sv[tid]+=sv[tid+ssz]; su2[tid]+=su2[tid+ssz]; sv2[tid]+=sv2[tid+ssz]; suv[tid]+=suv[tid+ssz]; }
        __syncthreads();
    }
    if (tid==0){ sum_u[iy]=su[0]; sum_v[iy]=sv[0]; sum_u2[iy]=su2[0]; sum_v2[iy]=sv2[0]; sum_uv[iy]=suv[0]; }
}
void launch_row_moments_from_psi_mapped(const Real* psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                               const Real* d_a_node, Real* su, Real* sv, Real* su2, Real* sv2, Real* suv, cudaStream_t s)
{
    int threads=256; dim3 gb(Ny); size_t sh=threads*sizeof(Real)*5;
    row_moments_from_psi_mapped_kernel<<<gb,threads,sh,s>>>(psi,Nx,Ny,inv_dx,inv_deta,d_a_node,su,sv,su2,sv2,suv);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void mean_wall_omega_kernel(const Real* __restrict__ omega, int Nx, int Ny, Real* __restrict__ mean_bot, Real* __restrict__ mean_top){
    extern __shared__ Real s[]; Real* sb=s; Real* st=s+blockDim.x;
    size_t tid=threadIdx.x; Real lb=0.0, lt=0.0;
    for (int ix=tid; ix<Nx; ix+=blockDim.x){ lb+=omega[ix]; lt+=omega[(size_t)(Ny-1)*Nx + ix]; }
    sb[tid]=lb; st[tid]=lt; __syncthreads();
    for (unsigned ssz=blockDim.x/2; ssz>0; ssz>>=1){ if (tid<ssz){ sb[tid]+=sb[tid+ssz]; st[tid]+=st[tid+ssz]; } __syncthreads(); }
    if (tid==0){ *mean_bot = sb[0]/(Real)Nx; *mean_top = st[0]/(Real)Nx; }
}
void launch_mean_wall_omega(const Real* omega, int Nx, int Ny, Real* d_mean_bot, Real* d_mean_top, cudaStream_t s){
    int threads=256; size_t sh=threads*sizeof(Real)*2;
    mean_wall_omega_kernel<<<1,threads,sh,s>>>(omega,Nx,Ny,d_mean_bot,d_mean_top);
    CUDA_CHECK(cudaGetLastError());
}
// Mean wall enstrophy-flux integrand ⟨ ω ω_y ⟩_x at bottom/top walls (mapped grid).
// ω_y is computed using 2nd-order one-sided differences in η and the metric: ω_y = a_node[j] * ω_η.
// Outputs mean_bot = (1/Nx) Σ_i ω_bot(i) * ω_y_bot(i), and similarly for mean_top.
__global__ void mean_wall_omega_omegay_mapped_kernel(
    const Real* __restrict__ omega,
    int Nx, int Ny,
    Real inv_deta,
    const Real* __restrict__ a_node,
    Real* __restrict__ mean_bot,
    Real* __restrict__ mean_top)
{
    extern __shared__ Real s[];
    Real* sb = s;
    Real* st = s + blockDim.x;

    const int tid = threadIdx.x;
    Real lb = 0.0;
    Real lt = 0.0;

    for (int ix = tid; ix < Nx; ix += blockDim.x){
        // bottom wall j=0 : ω_η ≈ (-3ω0 + 4ω1 - ω2)/(2Δη)
        const Real w0 = omega[(size_t)0*Nx + ix];
        const Real w1 = omega[(size_t)1*Nx + ix];
        const Real w2 = omega[(size_t)2*Nx + ix];
        const Real weta_bot = (-3.0*w0 + 4.0*w1 - 1.0*w2) * ((Real)0.5 * inv_deta);
        const Real wy_bot   = a_node[0] * weta_bot;
        lb += w0 * wy_bot;

        // top wall j=Ny-1 : ω_η ≈ (3ωN - 4ωN-1 + ωN-2)/(2Δη)
        const size_t jN = (size_t)(Ny-1);
        const size_t j1 = (size_t)(Ny-2);
        const size_t j2 = (size_t)(Ny-3);
        const Real wN   = omega[jN*(size_t)Nx + ix];
        const Real wNm1 = omega[j1*(size_t)Nx + ix];
        const Real wNm2 = omega[j2*(size_t)Nx + ix];
        const Real weta_top = ( 3.0*wN - 4.0*wNm1 + 1.0*wNm2) * ((Real)0.5 * inv_deta);
        const Real wy_top   = a_node[Ny-1] * weta_top;
        lt += wN * wy_top;
    }

    sb[tid] = lb;
    st[tid] = lt;
    __syncthreads();

    for (unsigned ssz = blockDim.x/2; ssz > 0; ssz >>= 1){
        if (tid < ssz){
            sb[tid] += sb[tid + ssz];
            st[tid] += st[tid + ssz];
        }
        __syncthreads();
    }

    if (tid == 0){
        *mean_bot = sb[0] / (Real)Nx;
        *mean_top = st[0] / (Real)Nx;
    }
}

void launch_mean_wall_omega_omegay_mapped(
    const Real* omega, int Nx, int Ny,
    Real inv_deta,
    const Real* d_a_node,
    Real* d_mean_bot, Real* d_mean_top,
    cudaStream_t s)
{
    int threads = 256;
    size_t sh   = (size_t)threads * sizeof(Real) * 2;
    mean_wall_omega_omegay_mapped_kernel<<<1, threads, sh, s>>>(
        omega, Nx, Ny, inv_deta, d_a_node, d_mean_bot, d_mean_top);
    CUDA_CHECK(cudaGetLastError());
}

// Wall‑balanced enstrophy dissipation using -ω Δω
void launch_enstrophy_diss_balance_mapped(
    const Real* d_omega,        // [Ny*Nx] vorticity field
    int Nx, int Ny,
    Real inv_dx2, Real inv_deta2,
    const Real* d_a_node,       // mapping metrics
    const Real* d_a_edge,
    const Real* d_w_node,       // quadrature weights in y
    Real* d_sum_w2,             // device scalar
    Real* d_sum_eta,            // device scalar
    cudaStream_t s)
{
    const size_t N = (size_t)Nx * (size_t)Ny;

    // Lazily allocated scratch for ∇²ω
    static Real*  d_lap     = nullptr;
    static size_t capacity  = 0;
    if (N > capacity) {
        if (d_lap) CUDA_CHECK(cudaFree(d_lap));
        CUDA_CHECK(cudaMalloc(&d_lap, N * sizeof(Real)));
        capacity = N;
    }

    // ∇²ω with the *same* stencil as the PDE (includes mapped y & walls)
    launch_laplacian_fd_mapped(
        d_omega, d_lap, Nx, Ny,
        inv_dx2, inv_deta2,
        d_a_node, d_a_edge, s);

    // Reduction of Ω and η = ν⟨ω Δω⟩
    const int threads = 256;
    const int blocks  =
        (int)std::min<size_t>(8192, (N + threads - 1) / threads);

    CUDA_CHECK(cudaMemsetAsync(d_sum_w2,  0, sizeof(Real), s));
    CUDA_CHECK(cudaMemsetAsync(d_sum_eta, 0, sizeof(Real), s));

    enstrophy_diss_balance_mapped_kernel<<<blocks, threads, 0, s>>>(
        d_omega, d_lap, Nx, Ny, d_w_node, d_sum_w2, d_sum_eta);
    CUDA_CHECK(cudaGetLastError());
}

//compute diffusion test: d Omega/dt = <omega*rhs> 
__global__ void omega_rhs_balance_mapped_kernel(
    const Real* __restrict__ omega,   // size Nx*Ny
    const Real* __restrict__ rhs,     // size Nx*Ny (current RHS)
    int Nx, int Ny,
    const Real* __restrict__ w_node,  // size Ny, y-weights
    Real* __restrict__ sum_omega_rhs) // single scalar accumulator
{
    extern __shared__ Real sdata[];

    const int tid = threadIdx.x;
    Real local_sum = (Real)0;

    // Flat index over the full grid
    for (int idx = blockIdx.x * blockDim.x + tid;
         idx < Nx * Ny;
         idx += blockDim.x * gridDim.x)
    {
        int j = idx / Nx;

        // interior only, like your enstrophy diagnostic
        //if (j == 0 || j == Ny - 1) continue;
        Real wj   = w_node[j];
        Real omg  = omega[idx];
        Real rhsv = rhs[idx];

        // contribution to ⟨ω·rhs⟩ with same y-weights
        local_sum += wj * omg * rhsv;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_omega_rhs, sdata[0]);
    }
}
void launch_omega_rhs_balance_mapped(
    const Real* d_omega,
    const Real* d_rhs,
    int Nx, int Ny,
    const Real* d_w_node,
    Real* d_sum_omega_rhs,
    cudaStream_t stream)
{
    // zero accumulator
    CUDA_CHECK( cudaMemsetAsync(d_sum_omega_rhs, 0, sizeof(Real), stream) );

    const int threads = 256;
    const int blocks  = (Nx * Ny + threads - 1) / threads;
    const size_t shmem = threads * sizeof(Real);

    omega_rhs_balance_mapped_kernel<<<blocks, threads, shmem, stream>>>(
        d_omega, d_rhs, Nx, Ny, d_w_node, d_sum_omega_rhs);

    CUDA_CHECK( cudaGetLastError() );
}


// - temporary block buffers are managed inside the launcher
// =======================================================
#ifndef IDX
#define IDX(j,i) ((size_t)(j) * (size_t)Nx + (size_t)(i))
#endif
// ---------------------------
// Stage 1: per‑block maxima
// ---------------------------
__global__ void max_uv_blocks_mapped_kernel( const Real* __restrict__ psi, int Nx, int Ny,
    Real inv_dx, Real inv_deta, const Real* __restrict__ a_node, Real* __restrict__ umax_blk, Real* __restrict__ vmax_blk,
    Real* __restrict__ vmaxm_blk)
{
    extern __shared__ unsigned char smem[];
    Real* sU = reinterpret_cast<Real*>(smem);
    Real* sV = sU + blockDim.x;
    Real* sVm = sV + blockDim.x;

    const size_t tid = threadIdx.x;
    const size_t N   = (size_t)Nx * (size_t)Ny;

    Real umax = 0.0;
    Real vmax = 0.0;
    Real vmaxm = 0.0; // max|v|*a(j), used for mapped-grid y-CFL in eta
   
    // Grid‑stride loop over all cells
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
         idx < N;
         idx += (size_t)gridDim.x * blockDim.x)
    {
        const int j = (int)(idx / Nx);
        const int i = (int)(idx - (size_t)j * Nx);

        // Skip wall rows for max (no‑slip -> u=v=0 at walls).
        // If you prefer, you can keep walls and use a one‑sided y‑derivative there.
        if (j == 0 || j == Ny - 1) {
            continue;
        }

        const int ip = (i + 1 < Nx) ? (i + 1) : 0;
        const int im = (i > 0)      ? (i - 1) : (Nx - 1);

        // Derivatives of ψ
        const Real dpsideta = (psi[IDX(j+1,i)] - psi[IDX(j-1,i)]) * (0.5 * inv_deta);
        const Real dpsidx   = (psi[IDX(j,ip)] - psi[IDX(j,im)])   * (0.5 * inv_dx);

        // Map to physical u, v
        const Real a = a_node[j];
        const Real u = a * dpsideta;    // ψ_y
        const Real v = -dpsidx;         // -ψ_x

        // Track maxima
        const Real au = fabs(u);
        const Real av = fabs(v);
	const Real avm = av * a;  // v/dy ~ |v|*a/deta (eta-space advective speed)
        if (au > umax) umax = au;
        if (av > vmax) vmax = av;
	if (avm > vmaxm) vmaxm = avm;
    }

    // In‑block reduction
    sU[tid] = umax;
    sV[tid] = vmax;
    sVm[tid] = vmaxm;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (sU[tid + s] > sU[tid]) sU[tid] = sU[tid + s];
            if (sV[tid + s] > sV[tid]) sV[tid] = sV[tid + s];
            if (sVm[tid + s] > sVm[tid]) sVm[tid] = sVm[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        umax_blk[blockIdx.x] = sU[0];
        vmax_blk[blockIdx.x] = sV[0];
        vmaxm_blk[blockIdx.x] = sVm[0];
    }
}

// ---------------------------------------
// Stage 2: reduce block maxima to scalars
// (single‑block reduction; blocks ≤ 8192)
// ---------------------------------------
__global__ void reduce_max_triple_kernel(const Real* __restrict__ umax_blk, const Real* __restrict__ vmax_blk,
    const Real* __restrict__ vmaxm_blk, int n, Real* __restrict__ umax_out, Real* __restrict__ vmax_out, Real* __restrict__ vmaxm_out)
{
    extern __shared__ unsigned char smem[];
    Real* sU = reinterpret_cast<Real*>(smem);
    Real* sV = sU + blockDim.x;
    Real* sVm = sV + blockDim.x;

    Real umax = 0.0, vmax = 0.0, vmaxm = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const Real u = umax_blk[i];
        const Real v = vmax_blk[i];
        const Real vm = vmaxm_blk[i];
        if (u > umax) umax = u;
        if (v > vmax) vmax = v;
        if (vm > vmaxm) vmaxm = vm;
    }

    sU[threadIdx.x] = umax;
    sV[threadIdx.x] = vmax;
    sVm[threadIdx.x] = vmaxm;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sU[threadIdx.x + s] > sU[threadIdx.x]) sU[threadIdx.x] = sU[threadIdx.x + s];
            if (sV[threadIdx.x + s] > sV[threadIdx.x]) sV[threadIdx.x] = sV[threadIdx.x + s];
            if (sVm[threadIdx.x + s] > sVm[threadIdx.x]) sVm[threadIdx.x] = sVm[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *umax_out = sU[0];
        *vmax_out = sV[0];
        *vmaxm_out = sVm[0];
    }
}

// -------------------------------------------------------
// Host launcher: same call signature you already use
// -------------------------------------------------------
void launch_max_uv_from_psi_mapped( const Real* d_psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* d_a_node, Real* d_umax, Real* d_vmax, Real* d_vmaxm,       // device scalars (size 1 each)
    cudaStream_t stream)
{
    // Sanity guard for tiny grids
    if (Nx < 2 || Ny < 3) {
        const Real zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_umax, &zero, sizeof(Real), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_vmax, &zero, sizeof(Real), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_vmaxm, &zero, sizeof(Real), cudaMemcpyHostToDevice, stream));
        return;
    }

    const int threads = 256;
    const size_t N    = (size_t)Nx * (size_t)Ny;

    // Cap block count to keep shared memory small; 8192 works well on all GPUs
    const int blocks  = (int)std::min<size_t>(8192, (N + threads - 1) / threads);
    const size_t shmem_stage1 = 3 * threads * sizeof(Real);
    const size_t shmem_stage2 = 3 * threads * sizeof(Real);

    // Persistent scratch for per‑block maxima
    static int   cap = 0;
    static Real* d_umax_blk = nullptr;
    static Real* d_vmax_blk = nullptr;
    static Real* d_vmaxm_blk = nullptr;
    if (blocks > cap) {
        if (d_umax_blk) CUDA_CHECK(cudaFree(d_umax_blk));
        if (d_vmax_blk) CUDA_CHECK(cudaFree(d_vmax_blk));
        if (d_vmaxm_blk) CUDA_CHECK(cudaFree(d_vmaxm_blk));
        CUDA_CHECK(cudaMalloc(&d_umax_blk, blocks * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_vmax_blk, blocks * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_vmaxm_blk, blocks * sizeof(Real)));
        cap = blocks;
    }

    // Stage 1: compute per‑block maxima over all Nx*Ny cells
    max_uv_blocks_mapped_kernel<<<blocks, threads, shmem_stage1, stream>>>(
        d_psi, Nx, Ny, inv_dx, inv_deta, d_a_node, d_umax_blk, d_vmax_blk, d_vmaxm_blk);
    CUDA_CHECK(cudaGetLastError());

    // Stage 2: reduce per‑block maxima to single scalars
    reduce_max_triple_kernel<<<1, threads, shmem_stage2, stream>>>(
        d_umax_blk, d_vmax_blk, d_vmaxm_blk, blocks, d_umax, d_vmax, d_vmaxm);
    CUDA_CHECK(cudaGetLastError());
}

/* -------------------- 1D spectra in x -------------------- */
__global__ void fill_kx_array_kernel(Real* kx, int Nxc, Real Lx){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Nxc){
        kx[i] = (2.0*CUDART_PI * (Real)i) / Lx; // i = 0..Nx/2
    }
}
__global__ void spectra1d_rowReduce_kernel(
    const cufftDoubleComplex* __restrict__ Uhat,
    const cufftDoubleComplex* __restrict__ Vhat,
    int Ny, int Nxc, Real norm1d,
    Real* __restrict__ Eu_tmp, Real* __restrict__ Ev_tmp)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= Nxc) return;
    double wx = (k>0 && k<Nxc-1)? 2.0 : 1.0; // R2C half-spectrum weight
    double sumU = 0.0, sumV = 0.0;
    for (int j=0;j<Ny;j++){
        size_t idx = (size_t)j*Nxc + k;
        double ur=Uhat[idx].x, ui=Uhat[idx].y;
        double vr=Vhat[idx].x, vi=Vhat[idx].y;
        sumU += (ur*ur + ui*ui);
        sumV += (vr*vr + vi*vi);
    }
    Eu_tmp[k] = norm1d * wx * sumU;
    Ev_tmp[k] = norm1d * wx * sumV;
}
__global__ void vec_add_kernel(Real* acc, const Real* tmp, int n){
    int i=blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) acc[i] += tmp[i];
}


