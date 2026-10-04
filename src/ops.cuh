// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Spatial operators: velocities, Laplacian, Arakawa Jacobian, wall closure, RHS, SSPRK updates.
#pragma once
#include "common.cuh"

__global__ void arakawa_J_kernel(const Real* __restrict__ psi, const Real* __restrict__ omg,
    Real* __restrict__ J, int Nx, int Ny, Real inv_dx, Real inv_dy);
__global__ void uv_from_psi_mapped_kernel(const Real* __restrict__ psi,
    Real* __restrict__ u, Real* __restrict__ v,
    int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node);
void launch_uv_from_psi_mapped(const Real* psi, Real* u, Real* v, int Nx, int Ny,
                                      Real inv_dx, Real inv_deta, const Real* d_a_node, cudaStream_t s=0);
__global__ void omega_from_psi_interior_mapped_kernel(Real* __restrict__ w, const Real* __restrict__ psi,
    int Nx, int Ny, Real inv_dx2, Real inv_deta2,
    const Real* __restrict__ a_node,
    const Real* __restrict__ a_edge);
void launch_omega_from_psi_interior_mapped(Real* w, const Real* psi, int Nx, int Ny,
                                                  Real inv_dx2, Real inv_deta2,
                                                  const Real* d_a_node, const Real* d_a_edge, cudaStream_t s=0);
__global__ void thom_wall_vorticity_mapped_kernel(Real* __restrict__ w, const Real* __restrict__ psi,
    int Nx, int Ny, const Real* __restrict__ dy_edge, Real psi_bot, Real psi_top);
void launch_thom_wall_vorticity_mapped(Real* w, const Real* psi, int Nx, int Ny,
                                              const Real* d_dy_edge, Real psi_bot, Real psi_top, cudaStream_t s=0);
__global__ void arakawa_J_mapped_kernel(const Real* __restrict__ psi,
                                        const Real* __restrict__ omg,
                                        Real* __restrict__ J,
                                        int Nx, int Ny,
                                        Real inv_dx, Real inv_deta,
                                        const Real* __restrict__ a_node);
void launch_arakawa_J_mapped(const Real* psi, const Real* omg, Real* J, int Nx, int Ny,
                                    Real inv_dx, Real inv_deta, const Real* d_a_node, cudaStream_t s=0);
__global__ void compute_adv_from_uv_mapped(const Real* __restrict__ u,
                                           const Real* __restrict__ v,
                                           const Real* __restrict__ omg,
                                           Real* __restrict__ Jadv,
                                           int Nx, int Ny,
                                           Real inv_dx, Real inv_deta,
                                           const Real* __restrict__ a_node);
__global__ void compute_adv_flux_mapped(const Real* __restrict__ u,
                                        const Real* __restrict__ v,
                                        const Real* __restrict__ omg,
                                        Real* __restrict__ Jadv,
                                        int Nx, int Ny,
                                        Real inv_dx, Real inv_deta,
                                        const Real* __restrict__ a_node);
__global__ void arakawa_J_mapped_split_kernel(const Real* __restrict__ psi,
                                              const Real* __restrict__ omg,
                                              Real* __restrict__ J1o,
                                              Real* __restrict__ J2o,
                                              Real* __restrict__ J3o,
                                              Real* __restrict__ Jadv,
                                              int Nx, int Ny,
                                              Real inv_dx, Real inv_deta,
                                              const Real* __restrict__ a_node);
void launch_laplacian_fd_mapped(const Real* d_q, Real* d_out,
                                       int Nx, int Ny,
                                       Real inv_dx2, Real inv_deta2,
                                       const Real* d_a_node, const Real* d_a_edge,
                                       cudaStream_t s=0);
__global__ void rhs_mapped_kernel(
    const Real* __restrict__ J, const Real* __restrict__ lap, 
    const Real* __restrict__ omg, Real* __restrict__ rhs, 
    int Nx, int Ny, Real nu, Real F0, int nforce, Real h, Real lin_drag,
    const Real* __restrict__ y_node);
void launch_rhs_mapped(
    const Real* J, const Real* lap, const Real* omg, Real* rhs,
    int Nx, int Ny, Real nu, Real F0, int nforce, Real h, Real lin_drag,
    const Real* d_y_node, cudaStream_t s=0);
void launch_mms_add_forcing(
    Real* d_rhs, int Nx, int Ny, Real Lx, Real h,
    Real t, Real nu, Real r,
    Real A0, Real eps, Real Om, int kx_mode,
    const Real* d_y_node,
    cudaStream_t s=0);
__global__ void axpy_inplace_kernel(Real* __restrict__ y, const Real* __restrict__ x, const Real* __restrict__ r, Real dt, size_t N);
__global__ void ssprk2_combine_kernel(Real* __restrict__ w2, const Real* __restrict__ w, const Real* __restrict__ w1, const Real* __restrict__ rhs, Real dt, size_t N);
__global__ void ssprk3_combine_kernel(Real* __restrict__ wnp1, const Real* __restrict__ w, const Real* __restrict__ w2, const Real* __restrict__ rhs, Real dt, size_t N);
void launch_axpy(Real* y, const Real* x, const Real* r, Real dt, size_t N, cudaStream_t s=0);
void launch_ssprk2(Real* w2, const Real* w, const Real* w1, const Real* rhs, Real dt, size_t N, cudaStream_t s=0);
void launch_ssprk3(Real* wnp1, const Real* w, const Real* w2, const Real* rhs, Real dt, size_t N, cudaStream_t s=0);
