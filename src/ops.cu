// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Spatial operators: velocities, Laplacian, Arakawa Jacobian, wall closure, RHS, SSPRK updates.
#include "ops.cuh"

/* -------------------- Operators & kernels (numerics preserved) -------------------- */

// given u = p_y, v = -p_x, w = -\Delta p   : uniform Arakawa_J
// this kernel computes advective term u \nabla w = uw_x + vw_y = p_yw_x-p_xw_y=J(w,p)
__global__ void arakawa_J_kernel(const Real* __restrict__ psi, const Real* __restrict__ omg,
    Real* __restrict__ J, int Nx, int Ny, Real inv_dx, Real inv_dy)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    if (iy==0 || iy==Ny-1){ J[(size_t)iy*Nx + ix] = 0.0; return; }

    auto wrap = [&](int j)->int { return (j<0? j+Nx : (j>=Nx? j-Nx : j)); };
    int xm=wrap(ix-1), xp=wrap(ix+1);
    int ym=iy-1, yp=iy+1;
    size_t idx = (size_t)iy*Nx + ix;

    Real psi_ip = psi[(size_t)iy*Nx + xp], psi_im = psi[(size_t)iy*Nx + xm];
    Real psi_jp = psi[(size_t)yp*Nx + ix], psi_jm = psi[(size_t)ym*Nx + ix];
    Real omg_ip = omg[(size_t)iy*Nx + xp], omg_im = omg[(size_t)iy*Nx + xm];
    Real omg_jp = omg[(size_t)yp*Nx + ix], omg_jm = omg[(size_t)ym*Nx + ix];

    Real J1 = ( (psi_ip-psi_im)*(omg_jp-omg_jm) - (psi_jp-psi_jm)*(omg_ip-omg_im) ) * 0.25 * inv_dx * inv_dy;

    Real psi_ip_jp = psi[(size_t)yp*Nx + xp], psi_ip_jm = psi[(size_t)ym*Nx + xp];
    Real psi_im_jp = psi[(size_t)yp*Nx + xm], psi_im_jm = psi[(size_t)ym*Nx + xm];
    Real omg_ip_jp = omg[(size_t)yp*Nx + xp], omg_ip_jm = omg[(size_t)ym*Nx + xp];
    Real omg_im_jp = omg[(size_t)yp*Nx + xm], omg_im_jm = omg[(size_t)ym*Nx + xm];

    Real J2 = (  psi_ip*(omg_ip_jp-omg_ip_jm) - psi_im*(omg_im_jp-omg_im_jm)
               - psi_jp*(omg_ip_jp-omg_im_jp) + psi_jm*(omg_ip_jm-omg_im_jm) ) * 0.25 * inv_dx * inv_dy;

    Real J3 = ( -omg_ip*(psi_ip_jp-psi_ip_jm) + omg_im*(psi_im_jp-psi_im_jm)
               + omg_jp*(psi_ip_jp-psi_im_jp) - omg_jm*(psi_ip_jm-psi_im_jm) ) * 0.25 * inv_dx * inv_dy;

    J[idx] = -(J1 + J2 + J3) / 3.0;
}

// -------------------- y-metrics builder (host-side) --------------------
// Build y, a_node, a_edge, dy_edge, w_node on host; fills G.y* vectors.

// ===================== MAPPED-Y VARIANTS (Phase-1) =====================
__global__ void uv_from_psi_mapped_kernel(const Real* __restrict__ psi,
    Real* __restrict__ u, Real* __restrict__ v,
    int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    auto wrap = [&](int j)->int { return (j<0? j+Nx : (j>=Nx? j-Nx : j)); };
    int xm=wrap(ix-1), xp=wrap(ix+1);
    int ym=(iy==0? 0:iy-1), yp=(iy==Ny-1? Ny-1:iy+1);
    Real dpsideta = (psi[(size_t)yp*Nx + ix] - psi[(size_t)ym*Nx + ix]) * 0.5 * inv_deta;
    Real dpsidx   = (psi[(size_t)iy*Nx + xp] - psi[(size_t)iy*Nx + xm]) * 0.5 * inv_dx;
    Real a = a_node[iy];
    u[(size_t)iy*Nx + ix] = a * dpsideta;
    v[(size_t)iy*Nx + ix] = - dpsidx;
}
void launch_uv_from_psi_mapped(const Real* psi, Real* u, Real* v, int Nx, int Ny,
                                      Real inv_dx, Real inv_deta, const Real* d_a_node, cudaStream_t s)
{
    dim3 tb(32,8), gb = blocksFor2D(Nx,Ny,tb.x,tb.y);
    uv_from_psi_mapped_kernel<<<gb,tb,0,s>>>(psi,u,v,Nx,Ny,inv_dx,inv_deta,d_a_node);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void omega_from_psi_interior_mapped_kernel(Real* __restrict__ w, const Real* __restrict__ psi,
    int Nx, int Ny, Real inv_dx2, Real inv_deta2,
    const Real* __restrict__ a_node,
    const Real* __restrict__ a_edge)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    if (iy==0 || iy==Ny-1) return;
    auto wrap = [&](int j)->int { return (j<0? j+Nx : (j>=Nx? j-Nx : j)); };
    int xm=wrap(ix-1), xp=wrap(ix+1);
    size_t idx = (size_t)iy*Nx + ix;

    Real lapx = psi[(size_t)iy*Nx + xp] - 2.0*psi[idx] + psi[(size_t)iy*Nx + xm];
    // conservative mapped y-Laplacian: a_node[j] * [ a_edge[j+1/2](ψ_{j+1}-ψ_j) - a_edge[j-1/2](ψ_j-ψ_{j-1}) ] / Δη^2
    Real aN = a_node[iy];
    Real ap = a_edge[iy];     // j+1/2
    Real am = a_edge[iy-1];   // j-1/2
    Real lapy = aN * ( ap*(psi[(size_t)(iy+1)*Nx + ix] - psi[idx]) - am*(psi[idx] - psi[(size_t)(iy-1)*Nx + ix]) );

    w[idx] = -(inv_dx2*lapx + inv_deta2*lapy);
}
void launch_omega_from_psi_interior_mapped(Real* w, const Real* psi, int Nx, int Ny,
                                                  Real inv_dx2, Real inv_deta2,
                                                  const Real* d_a_node, const Real* d_a_edge, cudaStream_t s)
{
    dim3 tb(32,8), gb = blocksFor2D(Nx,Ny,tb.x,tb.y);
    omega_from_psi_interior_mapped_kernel<<<gb,tb,0,s>>>(w,psi,Nx,Ny,inv_dx2,inv_deta2,d_a_node,d_a_edge);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void thom_wall_vorticity_mapped_kernel(Real* __restrict__ w, const Real* __restrict__ psi,
    int Nx, int Ny, const Real* __restrict__ dy_edge, Real psi_bot, Real psi_top)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= Nx) return;

#ifdef FREE_SLIP_WALLS
    //free-slip wall
    w[i] = 0;
    w[(size_t)(Ny-1)*Nx + i] = 0;
#else
    // second-order no-slip wall
    // ---------- bottom wall (j = 0) ----------
    {
	    // distances from wall y0 to first two interior nodes
	    Real dy1 = dy_edge[0];     // y1 - y0
	    Real dy2 = dy_edge[1];     // y2 - y1

	    // coefficients for ψ''(y0) ≈ A*(ψ1-ψ0) + B*(ψ2-ψ0)
	    Real A = (Real)2.0 * ( (Real)1.0/(dy1*dy1) + (Real)1.0/(dy1*dy2) );
	    Real B = (Real)(-2.0) * dy1 / ( dy2 * (dy1 + dy2) * (dy1 + dy2) );

	    size_t j1 = (size_t)1 * Nx + i;
	    size_t j2 = (size_t)2 * Nx + i;

	    Real dpsi1 = psi[j1] - psi_bot;
	    Real dpsi2 = psi[j2] - psi_bot;

	    Real d2psi = A * dpsi1 + B * dpsi2;
	    w[i] = -d2psi;  // ω = -ψ_yy at bottom wall
    }

    // ---------- top wall (j = Ny-1) ----------
    {
	    // distances from top wall y_{Ny-1} downwards
	    Real dy1 = dy_edge[Ny-2];  // y_{Ny-1} - y_{Ny-2}
    Real dy2 = dy_edge[Ny-3];  // y_{Ny-2} - y_{Ny-3}

    Real A = (Real)2.0 * ( (Real)1.0/(dy1*dy1) + (Real)1.0/(dy1*dy2) );
    Real B = (Real)(-2.0) * dy1 / ( dy2 * (dy1 + dy2) * (dy1 + dy2) );

    size_t j1 = (size_t)(Ny-2)*Nx + i;
    size_t j2 = (size_t)(Ny-3)*Nx + i;

    Real dpsi1 = psi[j1] - psi_top;
    Real dpsi2 = psi[j2] - psi_top;

    Real d2psi = A * dpsi1 + B * dpsi2;
    w[(size_t)(Ny-1)*Nx + i] = -d2psi;  // ω at top wall
    }

   // Real dyb = dy_edge[0];  //dy_edge[j] = y[j+1] - y[j]
   // Real dyt = dy_edge[Ny-2];
   // w[i] = -2.0/(dyb*dyb) * (psi[Nx + i] - psi_bot);  //first-order wall
   // w[(size_t)(Ny-1)*Nx + i] = -2.0/(dyt*dyt) * (psi[(size_t)(Ny-2)*Nx + i] - psi_top);
#endif
}
/*__global__ void thom_wall_vorticity_mapped_kernel(Real* __restrict__ w,
                                                  const Real* __restrict__ psi,
                                                  int Nx, int Ny,
                                                  const Real* __restrict__ dy_edge,
                                                  Real psi_bot, Real psi_top)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= Nx) return;

#ifdef FREE_SLIP_WALLS
    // Free-slip: ω_wall = 0
    w[i] = (Real)0;
    w[(size_t)(Ny-1)*Nx + i] = (Real)0;
#else
    // ===== 2nd-order no-slip Thom wall on a mapped, non-uniform y-grid =====
    // Use a 4-point one-sided stencil for ψ'' at the wall:
    //
    //   h1 = y1 - y0,  h2 = y2 - y1,  h3 = y3 - y2
    //
    //   ψ''(y0) ≈ a0 ψ0 + a1 ψ1 + a2 ψ2 + a3 ψ3
    //
    //   a0 = 2 (3h1 + 2h2 + h3) / [ h1 (h1^2 + 2 h1 h2 + h1 h3 + h2^2 + h2 h3) ]
    //   a1 = 2 (-2h1 - 2h2 - h3) / [ h1 h2 (h2 + h3) ]
    //   a2 = 2 ( 2h1 +  h2 + h3) / [ h2 h3 (h1 + h2) ]
    //   a3 = 2 (-2h1 -  h2     ) / [ h3 (h1 h2 + h1 h3 + h2^2 + 2 h2 h3 + h3^2) ]
    //
    // For a uniform grid h1=h2=h3=Δy this becomes
    //   ψ''(0) ≈ ( 2ψ0 - 5ψ1 + 4ψ2 - ψ3 ) / Δy^2.

    // ---- bottom wall (j = 0) ----
    if (Ny >= 4) {
        Real h1b = dy_edge[0];
        Real h2b = dy_edge[1];
        Real h3b = dy_edge[2];

        Real h1b2 = h1b*h1b;
        Real h2b2 = h2b*h2b;
        Real h3b2 = h3b*h3b;

        Real denom0b = h1b * ( h1b2 + (Real)2.0*h1b*h2b + h1b*h3b
                             + h2b2 + h2b*h3b );
        Real denom3b = h3b * ( h1b*h2b + h1b*h3b + h2b2
                             + (Real)2.0*h2b*h3b + h3b2 );

        Real a0b = (Real)2.0 * ( (Real)3.0*h1b + (Real)2.0*h2b + h3b ) / denom0b;
        Real a1b = (Real)2.0 * (-( (Real)2.0*h1b + (Real)2.0*h2b + h3b) )
                   / ( h1b*h2b*(h2b + h3b) );
        Real a2b = (Real)2.0 * ( (Real)2.0*h1b + h2b + h3b )
                   / ( h2b*h3b*(h1b + h2b) );
        Real a3b = (Real)2.0 * (-( (Real)2.0*h1b + h2b) ) / denom3b;

        size_t j0b = (size_t)0   * Nx + i;
        size_t j1b = (size_t)1   * Nx + i;
        size_t j2b = (size_t)2   * Nx + i;
        size_t j3b = (size_t)3   * Nx + i;

        Real psi0b = psi[j0b];   // equals psi_bot
        Real psi1b = psi[j1b];
        Real psi2b = psi[j2b];
        Real psi3b = psi[j3b];

        Real d2psib = a0b*psi0b + a1b*psi1b + a2b*psi2b + a3b*psi3b;
        w[i] = -d2psib;          // ω = −ψ_yy at bottom wall
    } else {
        // Tiny Ny fallback: 1st-order Thom to avoid out-of-bounds
        Real dyb = dy_edge[0];
        w[i] = -(Real)2.0/(dyb*dyb) * ( psi[(size_t)1*Nx + i] - psi_bot );
    }

    // ---- top wall (j = Ny-1) ----
    if (Ny >= 4) {
        Real h1t = dy_edge[Ny-2];  // y_{Ny-1} - y_{Ny-2}
        Real h2t = dy_edge[Ny-3];  // y_{Ny-2} - y_{Ny-3}
        Real h3t = dy_edge[Ny-4];  // y_{Ny-3} - y_{Ny-4}

        Real h1t2 = h1t*h1t;
        Real h2t2 = h2t*h2t;
        Real h3t2 = h3t*h3t;

        Real denom0t = h1t * ( h1t2 + (Real)2.0*h1t*h2t + h1t*h3t
                             + h2t2 + h2t*h3t );
        Real denom3t = h3t * ( h1t*h2t + h1t*h3t + h2t2
                             + (Real)2.0*h2t*h3t + h3t2 );

        Real a0t = (Real)2.0 * ( (Real)3.0*h1t + (Real)2.0*h2t + h3t ) / denom0t;
        Real a1t = (Real)2.0 * (-( (Real)2.0*h1t + (Real)2.0*h2t + h3t) )
                   / ( h1t*h2t*(h2t + h3t) );
        Real a2t = (Real)2.0 * ( (Real)2.0*h1t + h2t + h3t )
                   / ( h2t*h3t*(h1t + h2t) );
        Real a3t = (Real)2.0 * (-( (Real)2.0*h1t + h2t) ) / denom3t;

        size_t j0t = (size_t)(Ny-1)*Nx + i;
        size_t j1t = (size_t)(Ny-2)*Nx + i;
        size_t j2t = (size_t)(Ny-3)*Nx + i;
        size_t j3t = (size_t)(Ny-4)*Nx + i;

        Real psi0t = psi[j0t];   // equals psi_top
        Real psi1t = psi[j1t];
        Real psi2t = psi[j2t];
        Real psi3t = psi[j3t];

        Real d2psit = a0t*psi0t + a1t*psi1t + a2t*psi2t + a3t*psi3t;
        w[(size_t)(Ny-1)*Nx + i] = -d2psit;
    } else {
        Real dyt = dy_edge[Ny-2];
        w[(size_t)(Ny-1)*Nx + i] =
            -(Real)2.0/(dyt*dyt) * ( psi[(size_t)(Ny-2)*Nx + i] - psi_top );
    }
#endif
}*/

void launch_thom_wall_vorticity_mapped(Real* w, const Real* psi, int Nx, int Ny,
                                              const Real* d_dy_edge, Real psi_bot, Real psi_top, cudaStream_t s)
{
    dim3 tb(256), gb((Nx+tb.x-1)/tb.x);
    thom_wall_vorticity_mapped_kernel<<<gb,tb,0,s>>>(w,psi,Nx,Ny,d_dy_edge,psi_bot,psi_top);
    CUDA_CHECK(cudaGetLastError());
}
// Returns J_adv = u·∇ω = J(ω,ψ) = -J(ψ,ω)
// psi, omg: Nx*Ny (row-major), periodic in x; J is undefined at walls (we zero j=0,Ny-1)
// inv_dx = 1/Δx; inv_deta = 1/Δη on the uniform computational η-grid
// a_node[j] = ∂η/∂y at node j  (so ∂y = a * ∂η)
// Returns J_adv = u·∇ω = - (J1 + J2 + J3)/3 with u=(ψ_y,-ψ_x), and ∂y = a · ∂η.
// 9-point mapped Arakawa; returns J_adv = u·∇ω = -(J1+J2+J3)/3
__global__ void arakawa_J_mapped_kernel(const Real* __restrict__ psi,
                                        const Real* __restrict__ omg,
                                        Real* __restrict__ J,
                                        int Nx, int Ny,
                                        Real inv_dx, Real inv_deta,
                                        const Real* __restrict__ a_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=Nx || iy>=Ny) return;

    size_t c = (size_t)iy*Nx + ix;
    if (iy==0 || iy==Ny-1){ J[c]=(Real)0; return; }

    auto wrap=[Nx](int i){ return (i<0? i+Nx : (i>=Nx? i-Nx : i)); };
    int xm=wrap(ix-1), xp=wrap(ix+1), ym=iy-1, yp=iy+1;

    size_t cxp=(size_t)iy*Nx+xp, cxm=(size_t)iy*Nx+xm;
    size_t cyp=(size_t)yp*Nx+ix, cym=(size_t)ym*Nx+ix;
    size_t ip_jp=(size_t)yp*Nx+xp, ip_jm=(size_t)ym*Nx+xp;
    size_t im_jp=(size_t)yp*Nx+xm, im_jm=(size_t)ym*Nx+xm;

    const Real a        = a_node[iy];                 // ∂η/∂y at row j
    const Real inv4dxde = (Real)0.25 * inv_dx * inv_deta;

    // J1 = (δxψ)(a·δηω) − (a·δηψ)(δxω)
    Real J1 = ( (psi[cxp]-psi[cxm]) * ( a*(omg[cyp]-omg[cym]) )
              - ( a*(psi[cyp]-psi[cym]) ) * (omg[cxp]-omg[cxm]) ) * inv4dxde;

    // J2 = δx(ψ · a·δηω) − δη(ψ · δxω)    [note: the second leg is multiplied by 'a']
    Real J2 = (  psi[cxp] * ( a*(omg[ip_jp]-omg[ip_jm]) )
              -  psi[cxm] * ( a*(omg[im_jp]-omg[im_jm]) )
              -  a * (  psi[cyp] * (omg[ip_jp]-omg[im_jp])
                     - psi[cym] * (omg[ip_jm]-omg[im_jm]) ) ) * inv4dxde;

    // J3 = δx(ω · a·δηψ) − δη(ω · δxψ)    [FIXED: signs match uniform kernel]
    Real J3 = (- omg[cxp] * ( a*(psi[ip_jp]-psi[ip_jm]) )
              + omg[cxm] * ( a*(psi[im_jp]-psi[im_jm]) )
              + a * (  omg[cyp] * (psi[ip_jp]-psi[im_jp])
                     - omg[cym] * (psi[ip_jm]-psi[im_jm]) ) ) * inv4dxde;

    Real Jbr  = (J1 + J2 + J3) * (Real)(1.0/3.0);
    J[c]      = -Jbr;  // u·∇ω
}

void launch_arakawa_J_mapped(const Real* psi, const Real* omg, Real* J, int Nx, int Ny,
                                    Real inv_dx, Real inv_deta, const Real* d_a_node, cudaStream_t s)
{
    dim3 tb(32,8), gb = blocksFor2D(Nx,Ny,tb.x,tb.y);
    arakawa_J_mapped_kernel<<<gb,tb,0,s>>>(psi,omg,J,Nx,Ny,inv_dx,inv_deta,d_a_node);
    CUDA_CHECK(cudaGetLastError());
}
// ======================================================================
// 1) Direct advective reference from (u,v): J_ref = u·∇ω
//    (uses the SAME mapped first-derivative scales as the solver)
// ======================================================================
__global__ void compute_adv_from_uv_mapped(const Real* __restrict__ u,
                                           const Real* __restrict__ v,
                                           const Real* __restrict__ omg,
                                           Real* __restrict__ Jadv,
                                           int Nx, int Ny,
                                           Real inv_dx, Real inv_deta,
                                           const Real* __restrict__ a_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=Nx || iy>=Ny) return;

    const size_t c=(size_t)iy*Nx+ix;
    if (iy==0 || iy==Ny-1){ Jadv[c]=(Real)0; return; }

    auto wrap=[Nx](int i){ return (i<0? i+Nx : (i>=Nx? i-Nx : i)); };
    int xm=wrap(ix-1), xp=wrap(ix+1), ym=iy-1, yp=iy+1;

    const size_t cxm=(size_t)iy*Nx+xm, cxp=(size_t)iy*Nx+xp;
    const size_t cym=(size_t)ym*Nx+ix, cyp=(size_t)yp*Nx+ix;

    const Real inv2dx=(Real)0.5*inv_dx, inv2de=(Real)0.5*inv_deta;
    const Real a=a_node[iy];

    const Real dx_omg = (omg[cxp]-omg[cxm]) * inv2dx;
    const Real dy_omg = a * (omg[cyp]-omg[cym]) * inv2de;

    // u = ψ_y, v = -ψ_x (already computed upstream)
    const Real uc = u[c];
    const Real vc = v[c];

    Jadv[c] = uc*dx_omg + vc*dy_omg;     // u·∇ω  (this is what the PDE uses)
}
__global__ void compute_adv_flux_mapped(const Real* __restrict__ u,
                                        const Real* __restrict__ v,
                                        const Real* __restrict__ omg,
                                        Real* __restrict__ Jadv,
                                        int Nx, int Ny,
                                        Real inv_dx, Real inv_deta,
                                        const Real* __restrict__ a_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=Nx || iy>=Ny) return;

    const size_t c=(size_t)iy*Nx+ix;
    if (iy==0 || iy==Ny-1){ Jadv[c]=(Real)0; return; }

    auto wrap=[Nx](int i){ return (i<0? i+Nx : (i>=Nx? i-Nx : i)); };
    int xm=wrap(ix-1), xp=wrap(ix+1), ym=iy-1, yp=iy+1;

    size_t cxm=(size_t)iy*Nx+xm, cxp=(size_t)iy*Nx+xp;
    size_t cym=(size_t)ym*Nx+ix, cyp=(size_t)yp*Nx+ix;

    // x–flux at faces i±1/2: Fx = (u_avg)*(ω_avg)
    Real u_ip = (u[cxp] + u[c]) * (Real)0.5;
    Real u_im = (u[c]   + u[cxm]) * (Real)0.5;
    Real w_ip = (omg[cxp] + omg[c]) * (Real)0.5;
    Real w_im = (omg[c]   + omg[cxm]) * (Real)0.5;
    Real dFdx = (u_ip*w_ip - u_im*w_im) * inv_dx;

    // y–flux at faces j±1/2: Fy = (v_avg)*(ω_avg), then ∂y Fy = a * ∂η Fy
    Real v_jp = (v[cyp] + v[c]) * (Real)0.5;
    Real v_jm = (v[c]   + v[cym]) * (Real)0.5;
    Real w_jp = (omg[cyp] + omg[c]) * (Real)0.5;
    Real w_jm = (omg[c]   + omg[cym]) * (Real)0.5;
    Real dFdy = a_node[iy] * ( (v_jp*w_jp - v_jm*w_jm) * ( (Real)0.5*inv_deta*2.0 ) );
    // the last factor simplifies: 0.5*inv_deta*2.0 = inv_deta; written expanded to mirror ∂y=a·∂η.

    Jadv[c] = dFdx + dFdy;                          // conservative u·∇ω
}
// ======================================================================
// 2) Mapped Arakawa A/B/C kernel (ADVECTIVE form), *split* outputs:
//    writes J1, J2, J3 individually and Jadv = u·∇ω = - (J1+J2+J3)/3.
//    Every y-derivative carries the metric a_j = ∂η/∂y.
// ======================================================================
__global__ void arakawa_J_mapped_split_kernel(const Real* __restrict__ psi,
                                              const Real* __restrict__ omg,
                                              Real* __restrict__ J1o,
                                              Real* __restrict__ J2o,
                                              Real* __restrict__ J3o,
                                              Real* __restrict__ Jadv,
                                              int Nx, int Ny,
                                              Real inv_dx, Real inv_deta,
                                              const Real* __restrict__ a_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=Nx || iy>=Ny) return;

    size_t c = (size_t)iy*Nx + ix;
    if (iy==0 || iy==Ny-1){
        if (J1o)  J1o[c]=(Real)0; if (J2o)  J2o[c]=(Real)0;
        if (J3o)  J3o[c]=(Real)0; if (Jadv) Jadv[c]=(Real)0;
        return;
    }

    auto wrap=[Nx](int i){ return (i<0? i+Nx : (i>=Nx? i-Nx : i)); };
    int xm=wrap(ix-1), xp=wrap(ix+1), ym=iy-1, yp=iy+1;

    size_t cxp=(size_t)iy*Nx+xp, cxm=(size_t)iy*Nx+xm;
    size_t cyp=(size_t)yp*Nx+ix, cym=(size_t)ym*Nx+ix;
    size_t ip_jp=(size_t)yp*Nx+xp, ip_jm=(size_t)ym*Nx+xp;
    size_t im_jp=(size_t)yp*Nx+xm, im_jm=(size_t)ym*Nx+xm;

    const Real a        = a_node[iy];
    const Real inv4dxde = (Real)0.25 * inv_dx * inv_deta;

    Real J1 = ( (psi[cxp]-psi[cxm]) * ( a*(omg[cyp]-omg[cym]) )
              - ( a*(psi[cyp]-psi[cym]) ) * (omg[cxp]-omg[cxm]) ) * inv4dxde;

    Real J2 = (  psi[cxp] * ( a*(omg[ip_jp]-omg[ip_jm]) )
              -  psi[cxm] * ( a*(omg[im_jp]-omg[im_jm]) )
              -  a * (  psi[cyp] * (omg[ip_jp]-omg[im_jp])
                     - psi[cym] * (omg[ip_jm]-omg[im_jm]) ) ) * inv4dxde;

    // J3 [FIXED: signs match uniform kernel]
    Real J3 = ( -omg[cxp] * ( a*(psi[ip_jp]-psi[ip_jm]) )
              + omg[cxm] * ( a*(psi[im_jp]-psi[im_jm]) )
              +  a * (  omg[cyp] * (psi[ip_jp]-psi[im_jp])
                     - omg[cym] * (psi[ip_jm]-psi[im_jm]) ) ) * inv4dxde;

    if (J1o)  J1o[c]=J1;
    if (J2o)  J2o[c]=J2;
    if (J3o)  J3o[c]=J3;

    Real Jbr = (J1 + J2 + J3) * (Real)(1.0/3.0);
    if (Jadv) Jadv[c] = -Jbr; // u·∇ω
}

// ============= ADD THIS NEW KERNEL =============
// q_lap = ∇² q  (mapped y, periodic x). Writes interior j=1..Ny-2; sets walls = 0.
__global__ void laplacian_fd_mapped_kernel(const Real* __restrict__ q,
                                           Real* __restrict__ q_lap,
                                           int Nx, int Ny,
                                           Real inv_dx2, Real inv_deta2,
                                           const Real* __restrict__ a_node,  // Ny
                                           const Real* __restrict__ a_edge)  // Ny-1
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=Nx || iy>=Ny) return;

    size_t id = (size_t)iy*Nx + ix;
    if (iy==0 || iy==Ny-1){ q_lap[id] = (Real)0; return; }

    auto wrap=[Nx](int i){ return (i<0? i+Nx : (i>=Nx? i-Nx : i)); };
    int xm = wrap(ix-1), xp = wrap(ix+1);

    // δxx (uniform in x)
    Real dxx = ( q[(size_t)iy*Nx+xp] - 2*q[id] + q[(size_t)iy*Nx+xm] ) * inv_dx2;

    // mapped δyy = a_j * [ a_{j+1/2}(q_{j+1}-q_j) - a_{j-1/2}(q_j-q_{j-1}) ] / Δη²
    Real a   = a_node[iy];
    Real ap  = a_edge[iy];     // j+1/2
    Real am  = a_edge[iy-1];   // j-1/2
    Real dyy = a * ( ap*( q[(size_t)(iy+1)*Nx + ix] - q[id] )
                   - am*( q[id] - q[(size_t)(iy-1)*Nx + ix] ) ) * inv_deta2;

    q_lap[id] = dxx + dyy;
}

void launch_laplacian_fd_mapped(const Real* d_q, Real* d_out,
                                       int Nx, int Ny,
                                       Real inv_dx2, Real inv_deta2,
                                       const Real* d_a_node, const Real* d_a_edge,
                                       cudaStream_t s)
{
    dim3 tb(32,4), gb((Nx+31)/32, (Ny+3)/4);
    laplacian_fd_mapped_kernel<<<gb,tb,0,s>>>(d_q, d_out, Nx, Ny,
                                              inv_dx2, inv_deta2,
                                              d_a_node, d_a_edge);
    CUDA_CHECK(cudaGetLastError());
}

// ============= RHS mapped kernel =============
__global__ void rhs_mapped_kernel(
    const Real* __restrict__ J, const Real* __restrict__ lap, 
    const Real* __restrict__ omg, Real* __restrict__ rhs, 
    int Nx, int Ny, Real nu, Real F0, int nforce, Real h, Real lin_drag,
    const Real* __restrict__ y_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    size_t idx = (size_t)iy*Nx + ix;

    Real f = 0.0;
    if (F0 != 0.0){
        Real y = y_node[iy];  // Use actual grid coordinate
        Real arg = (Real)nforce * CUDART_PI * ((y + h)/(2.0*h));
        f = -F0 * ((Real)nforce*CUDART_PI/(2.0*h)) * cos(arg);
    }
    rhs[idx] = -J[idx] + nu*lap[idx] + f - lin_drag*omg[idx];
    //rhs[idx] = nu*lap[idx];
}

void launch_rhs_mapped(
    const Real* J, const Real* lap, const Real* omg, Real* rhs,
    int Nx, int Ny, Real nu, Real F0, int nforce, Real h, Real lin_drag,
    const Real* d_y_node, cudaStream_t s)
{
    dim3 tb(32,8), gb = blocksFor2D(Nx,Ny,tb.x,tb.y);
    rhs_mapped_kernel<<<gb,tb,0,s>>>(J,lap,omg,rhs,Nx,Ny,nu,F0,nforce,h,lin_drag,d_y_node);
    CUDA_CHECK(cudaGetLastError());
}
// Add manufactured forcing f_omega(x,y,t) to RHS for MMS verification.
//
// Manufactured streamfunction:
//   psi_ex = A(t) * sin(k x) * sin^2(theta),   theta = pi (y+h)/(2h)
// where A(t)=A0*(1 + eps*sin(Om*t)),  k = 2*pi*kx_mode/Lx.
//
// Governing PDE (your solver):
//   omega_t = -J(psi,omega) + nu * Lap(omega) + f_omega - r*omega
//
// MMS forcing definition (continuous):
//   f_omega = omega_t + J(psi,omega) - nu * Lap(omega) + r*omega
//

// This kernel computes f_omega analytically and does: rhs += f_omega.
__global__ void mms_add_forcing_kernel(
    Real* __restrict__ rhs,          // [Ny*Nx] (in/out)
    int Nx, int Ny, Real Lx, Real h, Real t, Real nu, Real r,
    Real A0, Real eps, Real Om, int  kx_mode,
    const Real* __restrict__ y_node) // [Ny] physical y
{
    const size_t N = (size_t)Nx * (size_t)Ny;
    const Real dx = Lx / (Real)Nx;

    // Wavenumbers / constants
    const Real k = (Real)(2.0) * (Real)CUDART_PI * (Real)kx_mode / Lx;
    const Real c = (Real)CUDART_PI / ((Real)2.0 * h);  // theta = c*(y+h)

    // Time dependence
    const Real A      = A0 * ((Real)1.0 + eps * sin(Om * t));
    const Real Aprime = A0 * (eps * Om * cos(Om * t));

    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += (size_t)gridDim.x * blockDim.x)
    {
        const int j = (int)(idx / (size_t)Nx);
        const int i = (int)(idx - (size_t)j * (size_t)Nx);

        const Real y = y_node[j];
        const Real x = (Real)i * dx;

        // theta, 2theta
        const Real theta  = c * (y + h);
        const Real two_th = (Real)2.0 * theta;

        const Real s_th   = sin(theta);
        const Real s      = s_th * s_th;     // sin^2(theta)
        const Real cos2   = cos(two_th);
        const Real sin2   = sin(two_th);

        // g(y) where omega_ex = A * sin(kx) * g(y)
        // For s=sin^2(theta):
        //   s'' = 2 c^2 cos(2theta)
        //   g   = k^2 s - s'' = k^2 s - 2 c^2 cos(2theta)
        const Real c2 = c * c;
        const Real g  = (k*k) * s - (Real)2.0 * c2 * cos2;

        // g'' = 2 c^2 cos(2theta) * (k^2 + 4 c^2)
        const Real gpp = (Real)2.0 * c2 * cos2 * ( (k*k) + (Real)4.0 * c2 );

        // sin(kx), sin(2kx)
        const Real sk   = sin(k * x);
        const Real s2k  = sin((Real)2.0 * k * x);

        // omega_t term: Aprime * sin(kx) * g
        // -nu Lap(omega) term: nu * A * sin(kx) * (k^2 g - g'')
        // +r omega term: r * A * sin(kx) * g
        const Real term_lin = Aprime * g + (nu * A) * ((k*k) * g - gpp) + (r  * A) * g;

        // IMPORTANT:
        //   The solver's Arakawa kernel returns J_adv = u·∇ω, and RHS uses -J_adv.
        //   MMS must therefore add +J_adv to cancel that term.
        //   For this manufactured solution:
        //     J(ψ,ω) = A^2 k c^3 sin(2kx) sin(2θ)
        //     J_adv  = u·∇ω = -J(ψ,ω)
        const Real c3 = c2 * c;
        const Real term_adv = -(A*A) * k * c3 * s2k * sin2;  // u·∇ω

        const Real fomega = sk * term_lin + term_adv;
        rhs[idx] += fomega;
    }
}

void launch_mms_add_forcing(
    Real* d_rhs, int Nx, int Ny, Real Lx, Real h,
    Real t, Real nu, Real r,
    Real A0, Real eps, Real Om, int kx_mode,
    const Real* d_y_node,
    cudaStream_t s)
{
    const size_t N = (size_t)Nx * (size_t)Ny;
    int threads = 256;
    int blocks  = (int)std::min<size_t>(8192, (N + threads - 1) / threads);
    mms_add_forcing_kernel<<<blocks, threads, 0, s>>>(
        d_rhs, Nx, Ny, Lx, h, t, nu, r, A0, eps, Om, kx_mode, d_y_node);
    CUDA_CHECK(cudaGetLastError());
}


__global__ void axpy_inplace_kernel(Real* __restrict__ y, const Real* __restrict__ x, const Real* __restrict__ r, Real dt, size_t N){
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) y[i] = x[i] + dt*r[i];
}
__global__ void ssprk2_combine_kernel(Real* __restrict__ w2, const Real* __restrict__ w, const Real* __restrict__ w1, const Real* __restrict__ rhs, Real dt, size_t N){
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) w2[i] = 0.75*w[i] + 0.25*(w1[i] + dt*rhs[i]);
}
__global__ void ssprk3_combine_kernel(Real* __restrict__ wnp1, const Real* __restrict__ w, const Real* __restrict__ w2, const Real* __restrict__ rhs, Real dt, size_t N){
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) wnp1[i] = (1.0/3.0)*w[i] + (2.0/3.0)*(w2[i] + dt*rhs[i]);
}
void launch_axpy(Real* y, const Real* x, const Real* r, Real dt, size_t N, cudaStream_t s){
    dim3 tb(256), gb((N+tb.x-1)/tb.x);
    axpy_inplace_kernel<<<gb,tb,0,s>>>(y,x,r,dt,N);
    CUDA_CHECK(cudaGetLastError());
}
void launch_ssprk2(Real* w2, const Real* w, const Real* w1, const Real* rhs, Real dt, size_t N, cudaStream_t s){
    dim3 tb(256), gb((N+tb.x-1)/tb.x);
    ssprk2_combine_kernel<<<gb,tb,0,s>>>(w2,w,w1,rhs,dt,N);
    CUDA_CHECK(cudaGetLastError());
}
void launch_ssprk3(Real* wnp1, const Real* w, const Real* w2, const Real* rhs, Real dt, size_t N, cudaStream_t s){
    dim3 tb(256), gb((N+tb.x-1)/tb.x);
    ssprk3_combine_kernel<<<gb,tb,0,s>>>(wnp1,w,w2,rhs,dt,N);
}

// =======================================================
// Max |u|, |v| on a mapped (stretched) grid
// u =  ψ_y = a(j) * ψ_η,   v = -ψ_x
// - x is periodic
// - y derivatives use j±1 (one‑sided at the walls by skipping them)
// - two‑stage reduction (per‑block -> final scalar)
