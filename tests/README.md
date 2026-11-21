# Test suite

`run_tests.py` runs the solver on six small problems and checks the results against exact
solutions, exact identities of the scheme, and the reference data in `reference/`.
The whole suite takes a few minutes on one GPU and needs less than 1 GB of GPU memory.

## Requirements

- the solver built with HDF5 (the default): `make` gives `bin/nse2d`;
- for the conservation test, the free-slip build: `make WALLS=freeslip` gives `bin/nse2d_freeslip`
  (without it that test is skipped);
- Python 3.8 or later with `numpy` and `h5py`.

## Run

```bash
make && make WALLS=freeslip
python3 tests/run_tests.py all
```

`all` runs every case into `tests/out/<case>/` and then checks them. The two steps can be run
separately (`run`, `check`), `--only mms channel` restricts both to some tests, `--exe` and
`--exe-freeslip` select other executables, and `list` prints the command line of every case.
The exit code of `check` is 0 if every test passed.

Expected output (the numbers differ in the last digits between GPUs and compilers):

```
PASS  eigenmode_space
PASS  eigenmode_time
PASS  poiseuille
PASS  mms
PASS  conservation
PASS  channel

6 passed, 0 failed, 0 skipped
```

Under each test the script prints the individual checks with the measured values.

## What is tested

| test | problem | check |
|---|---|---|
| `eigenmode_space` | viscous shear eigenmode `omega = -A mu cos(mu (y+h)) exp(-nu mu^2 t)`, `N_y` = 33, 65, 129, uniform and `tanh` grid (`beta = 2`) | error against the exact solution decreases with order 2 |
| `eigenmode_time` | the same mode (`m = 8`) on a fixed grid with `dt` = 0.64, 0.32, 0.16, 0.08, against a run with `dt` = 0.01 | error decreases with order 3 |
| `poiseuille` | laminar Poiseuille flow, `Re_b = 500`, stretched grid | wall stress, pressure gradient, dissipation and kinetic energy equal the exact values within 0.5 % |
| `mms` | manufactured solution `psi = A(t) sin(kx) sin^2(pi (y+h) / 2h)` of the full nonlinear equation, `N` = 64, 128, 256, uniform and stretched grid | error against the exact solution decreases with order 2 |
| `conservation` | inviscid, unforced, free-slip walls, random band-limited vorticity | the discrete Arakawa energy and enstrophy drift by less than `1e-6` and the drift decreases with the time step |
| `channel` | forced channel with no-slip walls, `Re_b = 5000`, `512 x 257`, `t` in `[0, 20]`, run with `--closure_diag` | the scheme-consistent enstrophy identity `D_w = A_w - eta_h + Phi_h + P_w - 2 r Omega_w` holds at every output time; the diagnostics of `t <= 2` equal the reference |

The convergence tests refine the grid with a time step small enough that the temporal error is
negligible, and the temporal test compares runs on the same grid so that the spatial error cancels.

## Reference data

`reference/reference.json` holds the measured errors, deviations and drifts of every test, and
`reference/output/<case>/` holds the command line (`command.txt`), `diagnostics.csv` and
`progress.log` of every case, as produced on an NVIDIA A100 (the platform is recorded in
`reference/platform.txt`). `check` compares the new values with the reference within a relative
tolerance per test (2 % to 5 % for discretisation errors, which are reproducible to many digits;
50 % for the conservation drifts, which are close to round-off; `1e-4` for the early-time channel
diagnostics). The channel flow is chaotic, so its diagnostics are compared only for `t <= 2`.

To regenerate the reference data after a change that is meant to alter the results:

```bash
python3 tests/run_tests.py all --write-reference tests/reference
```

## The `--closure_diag` columns

With `--closure_diag` the solver appends 13 columns to `diagnostics.csv`. They are evaluated at
the same state as the other columns of the row, in the quadrature `wt_j = dxi / a_j` on the
interior rows in which the discrete Laplacian is self-adjoint:

| column | meaning |
|---|---|
| `dOmega_rhs_same`, `dOmega_rhs_same_int` | `<omega N(omega)>` over the full domain and over the interior rows, trapezoid weights |
| `Omega_int`, `eta_int` | interior-row parts of `Omega` and `eta` |
| `Jwall_s1_cum`, `Jwall_s2_cum`, `Jwall_s3_cum` | enstrophy added to the wall rows by the wall-vorticity update after each Runge-Kutta stage, accumulated since the start |
| `Omega_int_w` | `1/2 <omega^2>_wt` |
| `dOmega_same_int_w` | `D_w = <omega N(omega)>_wt` |
| `A_w` | `-<omega (u . grad omega)>_wt` with the Arakawa Jacobian |
| `P_int_w` | `<omega f_omega>_wt` |
| `eta_h`, `Phi_h` | summation-by-parts remainders of the discrete diffusion, `nu <omega Lap_h omega>_wt = -eta_h + Phi_h` |

The option is off by default; without it the output and the cost of a step are unchanged.
