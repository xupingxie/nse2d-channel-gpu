# nse2d-channel-gpu

GPU-accelerated direct numerical simulation of two-dimensional incompressible turbulence in a
plane channel, in vorticity–streamfunction form. Single-source CUDA C++ solver with a Python
post-processing package.

**Physics and numerics**

- Vorticity–streamfunction formulation: `ω_t + J(ψ, ω) = ν ∇²ω + f_ω − r ω`, `ω = −∇²ψ`
- Channel periodic in `x`, no-slip walls at `y = ±h` (free-slip available as a build option);
  constant volumetric flux imposed through Dirichlet values of `ψ`
- Wall-normal grid stretching through a smooth `tanh` mapping (or a tabulated mapping)
- Second-order finite differences on the mapped grid; energy- and enstrophy-conserving
  Arakawa Jacobian for the nonlinear term
- Second-order Thom-type wall-vorticity closure valid on non-uniform grids
- SSPRK(3,3) time integration with adaptive step (advective and viscous limits)
- Streamfunction Poisson solve by batched cuFFT in `x` and batched Thomas tridiagonal solves in `y`
  with precomputed elimination coefficients; `O(Nx Ny log Nx)` per step
- Kolmogorov body forcing and linear drag; manufactured-solution and viscous-eigenmode modes for verification
- Run-time diagnostics on the GPU: energy and enstrophy budgets including the wall enstrophy flux,
  wall friction (`u_τ`, `Re_τ`, `C_f`), mean and Reynolds-stress profiles; HDF5 snapshots with restart

## Source layout

```
src/common.cuh   Grid and Params structs, error macros, launch helpers, shared includes
src/grid.cu/.cuh wall-normal metrics: uniform, tanh stretching, tabulated mapping
src/ops.cu/.cuh  operators: velocities from psi, mapped Laplacian, Arakawa Jacobian, Thom wall closure,
                 RHS assembly (forcing, drag), manufactured-solution forcing, SSPRK stage updates
src/poisson.cu/.cuh  cuFFT (x) + batched Thomas (y) Poisson solver with precomputed coefficients
src/diag.cu/.cuh     reductions for budgets, wall quantities, row moments, time-step maxima
src/io.cu/.cuh       run log, output scheduling, HDF5 snapshots and restart, CSV writers
src/init.cu/.cuh     command-line parsing and initial conditions
src/main.cu          time loop: three SSPRK(3,3) stages, each followed by a Poisson solve and wall closure
src/test_api.cu      C-linkage wrappers around the operator launchers for unit tests (-DEXPORT_TEST_API)
Makefile             build rules (walls, HDF5, architecture are make variables)
build_solver.sh      NERSC Perlmutter helper: loads modules, finds libraries, calls make
postproc/            verification and analysis scripts, test plan
```

## Build

Requires CUDA 12 (cuFFT) and optionally HDF5.

```bash
make                       # no-slip walls, HDF5 on      -> bin/nse2d
make WALLS=freeslip        # free-slip walls (ω_wall=0)  -> bin/nse2d_freeslip
make HDF5=0                # without HDF5 (no snapshots, restart, tabulated grids)
make ARCH=sm_90            # other GPU (default sm_80, NVIDIA A100)
make CUDA_HOME=/usr/local/cuda HDF5_ROOT=/usr     # library locations if not on the default paths
```

On NERSC Perlmutter, `bash build_solver.sh [hdf5|bin] [noslip|freeslip]` loads the modules and passes the
library paths to `make`.

## Run

```bash
# forced turbulent channel, Re_b = 2e4, stretched grid
bin/nse2d --Nx 4096 --Ny 2049 --Ub 1 --Reb 20000 --init mix --amp 0.05 \
    --F0 0.05 --nforce 12 --drag 1e-3 --stretch tanh --beta 2 \
    --tend 400 --diag_save 0.1 --snap_save 5 --snap_fields all --outdir out/prod

# laminar Poiseuille check
bin/nse2d --Nx 256 --Ny 257 --Reb 500 --init laminar --stretch tanh --beta 2 --tend 50 --outdir out/poiseuille

# manufactured solution, convergence test
bin/nse2d --Nx 256 --Ny 257 --Ub 0 --nu 1e-2 --init mms --mmsA0 1 --mmsEps 0.1 --mmsOm 1 --mmsKx 2 \
    --stretch tanh --beta 2 --no-adapt --dt 5e-3 --tend 1 --snap_save 1 --outdir out/mms
```

Main options (see `parse_args` in `src/init.cu` for the full list):

| option | meaning |
|---|---|
| `--Nx --Ny --Lx --h` | grid and domain (`y ∈ [−h, h]`, `Ny` odd) |
| `--stretch tanh --beta β` | wall-normal stretching; `--stretch none` for a uniform grid |
| `--Ub --Reb` or `--nu` | bulk velocity and `ν = 2hUb/Re_b`, or `ν` directly |
| `--cfl --cvisc --dtmax --no-adapt --dt` | time-step control |
| `--F0 --nforce --drag` | Kolmogorov forcing amplitude and mode, linear drag |
| `--init` | `laminar`, `ts`, `mix`, `rand`, `load`, `mms`, `diffusion` |
| `--tend --diag_save --snap_save --snap_fields` | run length and output cadence |
| `--statsEvery --profilesWriteEvery` | profile accumulation and output |
| `--profile --timeEvery` | per-step timers (FFT, tridiagonal, total) |
| `--resume --load_path snap.h5` | restart from a snapshot |

Output files in `--outdir`: `diagnostics.csv`, `profiles_timeavg.csv`, `timing.csv` (with `--profile`),
`progress.log`, and `snaps/snap_t*.h5`.

## Post-processing and verification

`postproc/` contains the verification and analysis scripts (eigenmode and manufactured-solution
convergence, Poiseuille check, Arakawa conservation, budget closure, resolution sensitivity, production
statistics, streamwise spectra and cascade fluxes, performance tables) together with the test plan
`postproc/TESTS_TO_RUN.md`. See `postproc/README.md`. Requires `numpy pandas matplotlib h5py`;
`python postproc/selftest.py` checks the installation on synthetic data.

## Citing

A paper describing the method and its verification is in preparation. Until it appears, please cite
this repository.

## Acknowledgments

This research used resources of the National Energy Research Scientific Computing Center (NERSC),
a Department of Energy Office of Science User Facility.

## License

BSD 3-Clause, see `LICENSE`.
