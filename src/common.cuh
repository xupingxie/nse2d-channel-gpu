// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Common types (Grid, Params), error macros, launch helpers.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <stdexcept>
#include <limits>
#include <algorithm>
#if defined(_WIN32)
#include <direct.h>
#endif
#include <cuda_runtime.h>
#include <cufft.h>
#include <math_constants.h>
#ifndef NO_HDF5
#include <hdf5.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

using Real = double;
using Cx   = cufftDoubleComplex;

// ---- grid, parameters, error macros ---------------------------------------------------------
struct Grid {
    int Nx{1024};
    int Ny{1025};
    Real Lx{12.566370612};
    Real h {1.0};
    // --- Moved from YMetrics to here ---
    std::vector<Real> y;       // Ny
    std::vector<Real> a_node;  // Ny (a_j = 1/y_eta at nodes)
    std::vector<Real> a_edge;  // Ny-1 (a at j+1/2 edges)
    std::vector<Real> dy_edge; // Ny-1 (y_{j+1}-y_j)
    std::vector<Real> w_node;  // Ny (trapezoid weights)
    
    __host__ __device__ inline Real dx() const { return Lx / (Real)Nx; }
    __host__ __device__ inline Real dy() const { return (2.0*h) / (Real)(Ny-1); }
    __host__ __device__ inline Real Ly() const { return 2.0*h; }
};
struct Params {
    Real Ub   {1.0};
    // --- y-stretch mapping controls ---
    std::string stretch{"none"}; // "none" or "tanh"
    std::string ytable{"none"}; // "none" or "tanh"
    Real beta{0.0};
    // --- Time integration ---
    std::string integrator{"imex3"};  // "ssprk3" (explicit) or "imex" (2nd order) or "imex3" (3rd order)

    Real Reb  {10000.0};
    Real nu_override{-1}; // <0 means "use 2h*Ub/Reb"; otherwise use this absolute ν
    Real nu(Real h) const { return (nu_override>=0 ? nu_override : (2.0*h*Ub)/Reb); }

    Real dt_init {1e-3};
    Real dt_max  {1e-2};
    Real cfl     {0.5};
    Real cvisc   {0.5};
    bool adapt   {true};
    Real t_end   {200.0};
    Real F0      {0.0};
    int  nforce  {1};
    Real lin_drag{0.0};
    std::string progress_path{}; // path to progress.log so utilities can tee messages without changing signatures
    // NEW: resume controls
    bool resume{false};      // if true, start from snapshot time instead of 0
    Real t0{0.0};            // initial time (filled from snapshot 't' when resume=true)

    // --- Initial condition controls ---
    std::string init{"load"};     // laminar | rand | ts | mix | load
    Real amp{0.5};               // amplitude parameter (see notes above)
    Real alpha{1.2};             // TS streamwise wavenumber (default 2π/Lx if 0)
    int  my{1};                  // TS wall-normal index
    Real phase{0.0};             // TS phase [rad]
    Real rand_amp{0.0};          // legacy alias (if init not given)
    unsigned long long seed{123456789ULL};
    // snapshot loading
    std::string load_path{""};
    bool load_use_psi{false}; // if true, read psi and rebuild omega; else read omega
    std::string snap_fields{"all"}; //which fields to write: "all" "omega" "omega_psi"

    // Controls for 'rand' (turbulent vorticity IC)
    int  rand_nx{48};            // # streamwise modes to sample
    int  rand_mymax{8};          // max wall-normal sine index (>=1)
    Real rand_k0{0.0};           // center kx (rad/unit); 0 => 6*(2π/Lx)
    Real rand_sigma{0.6};        // log bandwidth for kx sampling
    Real rand_amp_abs{0.0};      // absolute vorticity amplitude; if 0, falls back to 'amp'
 // --- Manufactured-solution (MMS) verification controls ---
 // If mms=true, the solver should add f_omega(x,y,t) into the vorticity RHS.
 // Typically you also use --init mms (so ω(t=0) matches ω_exact(t=0)).
    bool mms{false};          // enable MMS forcing f_ω
    Real mms_A0{1e-3};        // base amplitude A0
    Real mms_eps{0.1};        // modulation amplitude ε (0 => steady in time)
    Real mms_Om{1.0};         // modulation frequency Ω
    int  mms_kx{1};           // streamwise mode index (k = 2π*mms_kx/Lx)
};

#define CUDA_CHECK(e) do { cudaError_t err__=(e); if (err__!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); std::abort(); } } while(0)
#define CUFFT_CHECK(e) do { cufftResult res__=(e); if (res__!=CUFFT_SUCCESS){ \
  fprintf(stderr,"cuFFT error %s:%d: code %d\n", __FILE__, __LINE__, int(res__)); std::abort(); } } while(0)

inline dim3 blocksFor2D(int NX, int NY, int tx=32, int ty=8){
    return dim3( (NX + tx - 1)/tx, (NY + ty - 1)/ty );
}

/* -------------------- Progress logger -------------------- */

// flat index (row-major, x fastest)
#ifndef IDX
#define IDX(j,i) ((size_t)(j) * (size_t)Nx + (size_t)(i))
#endif
