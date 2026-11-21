// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// C-linkage wrappers for unit-testing individual operators (compiled only with -DEXPORT_TEST_API).
#include "ops.cuh"

#ifdef EXPORT_TEST_API
// C-linkage wrappers around the operator launchers, for external unit tests (build with -DEXPORT_TEST_API).

extern "C" void testapi_uv_from_psi_mapped(
    const Real* d_psi, Real* d_u, Real* d_v,
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,          // scalars first
    const Real* d_a_node,                // pointer last
    cudaStream_t s)
{
    launch_uv_from_psi_mapped(d_psi, d_u, d_v,
                              Nx, Ny,
                              inv_dx, inv_deta, d_a_node, s);
}

extern "C" void testapi_omega_from_psi_interior_mapped(
    Real* d_w, const Real* d_psi,
    int Nx, int Ny,
    Real inv_dx2, Real inv_deta2,     // scalars
    const Real* d_a_node,             // Ny
    const Real* d_a_edge,             // Ny-1
    cudaStream_t s)
{
    launch_omega_from_psi_interior_mapped(d_w, d_psi, Nx, Ny,
                                          inv_dx2, inv_deta2, d_a_node, d_a_edge, s);
}

extern "C" void testapi_thom_wall_vorticity_mapped(
    Real* d_w, const Real* d_psi,
    int Nx, int Ny,
    const Real* d_dy_edge,            // Ny-1 (physical Δy at edges)
    Real psi_bot, Real psi_top,
    cudaStream_t s)
{
    // no inv_deta2 or d_y needed; Thom uses physical Δy near the walls
    launch_thom_wall_vorticity_mapped(d_w, d_psi, Nx, Ny,
                                      d_dy_edge, psi_bot, psi_top, s);
}

extern "C" void testapi_arakawa_J_mapped(
    const Real* d_psi, const Real* d_w, Real* d_J,
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,       // scalars
    const Real* d_a_node,             // Ny
    cudaStream_t s)
{
    launch_arakawa_J_mapped(d_psi, d_w, d_J, Nx, Ny,
                            inv_dx, inv_deta, d_a_node, s);
}

extern "C" void testapi_compute_adv_from_uv_mapped(
    const Real* d_u, const Real* d_v, const Real* d_omg, Real* d_Jref,
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,
    const Real* d_a_node, cudaStream_t s)
{
    dim3 tb(32,4), gb((Nx+31)/32,(Ny+3)/4);
    compute_adv_from_uv_mapped<<<gb,tb,0,s>>>(
        d_u, d_v, d_omg, d_Jref, Nx, Ny, inv_dx, inv_deta, d_a_node);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void testapi_compute_adv_flux_mapped(
    const Real* d_u, const Real* d_v, const Real* d_omg, Real* d_Jref,
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,
    const Real* d_a_node, cudaStream_t s)
{
    dim3 tb(32,4), gb((Nx+31)/32,(Ny+3)/4);
    compute_adv_flux_mapped<<<gb,tb,0,s>>>(
        d_u, d_v, d_omg, d_Jref, Nx, Ny, inv_dx, inv_deta, d_a_node);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void testapi_arakawa_J_mapped_split(
    const Real* d_psi, const Real* d_omg,
    Real* d_J1, Real* d_J2, Real* d_J3, Real* d_Jadv,
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,
    const Real* d_a_node,
    cudaStream_t s)
{
    dim3 tb(32,4), gb((Nx+31)/32,(Ny+3)/4);
    arakawa_J_mapped_split_kernel<<<gb,tb,0,s>>>(
        d_psi, d_omg, d_J1, d_J2, d_J3, d_Jadv,
        Nx, Ny, inv_dx, inv_deta, d_a_node);
    CUDA_CHECK(cudaGetLastError());
}
#endif
