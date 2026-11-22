#!/usr/bin/env python3
"""Test suite of nse2d-channel-gpu: runs the solver on small cases and checks the results.

    python3 tests/run_tests.py run   [--exe bin/nse2d] [--exe-freeslip bin/nse2d_freeslip] [--out tests/out]
    python3 tests/run_tests.py check [--out tests/out] [--reference tests/reference/reference.json]
    python3 tests/run_tests.py all   ...            # run, then check

`run` executes every case into <out>/<case>/ and stores the command line in command.txt.
`check` evaluates each test from the output files and prints PASS or FAIL per test:
  1. against exact solutions and exact identities (no reference data needed);
  2. against the reference values shipped in tests/reference/reference.json (if the file exists).
`check --write-reference DIR` writes reference.json and copies the small text outputs of every
case into DIR/output/ (used by the maintainers to regenerate the reference data).

Requirements: Python 3.8+, numpy, h5py; a solver built with HDF5 (the default build).
The whole suite takes a few minutes on one GPU and needs less than 1 GB of GPU memory.

Tests
  eigenmode_space  viscous shear eigenmode, error against the exact solution on three grids,
                   uniform and stretched: second order in space
  eigenmode_time   same mode with four time steps against a small-step reference: third order
  poiseuille       laminar Poiseuille flow: wall stress, pressure gradient, dissipation, energy
  mms              manufactured solution of the full nonlinear equation on three grids, uniform
                   and stretched: second order
  conservation     inviscid free-slip run: the discrete Arakawa invariants (energy, enstrophy)
                   stay constant up to the time-integration error   [needs the free-slip build]
  channel          forced no-slip channel with --closure_diag: the scheme-consistent enstrophy
                   identity holds at every output time; early-time diagnostics match the reference
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ----------------------------------------------------------------------------------------------
# Case definitions (the single place where the run parameters live)
# ----------------------------------------------------------------------------------------------
LX = 4.0 * math.pi          # solver default --Lx
H = 1.0                     # solver default --h

EIG_NU, EIG_T = 1e-2, 2.0                       # eigenmode_space
EIG_NY, EIG_BETA, EIG_DT = [33, 65, 129], [0, 2], 1e-4
TIM_NU, TIM_T, TIM_M, TIM_AU = 1e-4, 10.24, 8, 0.1   # eigenmode_time
TIM_DT, TIM_DTREF, TIM_NY = [0.64, 0.32, 0.16, 0.08], 0.01, 129
POI_REB, POI_UB, POI_T = 500.0, 1.0, 5.0        # poiseuille
MMS_NU, MMS_T, MMS_A0, MMS_EPS, MMS_OM, MMS_KX = 1e-2, 1.0, 1.0, 0.1, 1.0, 2
MMS_N, MMS_BETA = [64, 128, 256], [0, 2]
CON_DT, CON_T = [4e-3, 2e-3], 2.0               # conservation
CH_REB, CH_F0, CH_NF, CH_DRAG, CH_T = 5000.0, 0.05, 12, 1e-3, 20.0   # channel
CH_DIAG = 0.05              # interval of the diagnostic output
CH_TREF = 2.0               # early-time window compared with the reference


def stretch_args(beta):
    return ["--stretch", "none"] if beta == 0 else ["--stretch", "tanh", "--beta", str(beta)]


def mms_dt(n):
    return 5e-4 * (64.0 / n) ** 2


def cases():
    """List of (name, executable key, argument list)."""
    c = []
    for b in EIG_BETA:
        for ny in EIG_NY:
            c.append((f"eig_space_Ny{ny}_b{b}", "noslip",
                      ["--Nx", "64", "--Ny", str(ny), "--Ub", "0", "--nu", str(EIG_NU), "--init", "diffusion",
                       "--my", "1", "--amp", "1"] + stretch_args(b) +
                      ["--no-adapt", "--dt", str(EIG_DT), "--tend", str(EIG_T), "--snap_save", str(EIG_T),
                       "--snap_fields", "omega_psi", "--diag_save", "0.5"]))
    for dt in TIM_DT + [TIM_DTREF]:
        name = "eig_time_ref" if dt == TIM_DTREF else f"eig_time_dt{dt:g}"
        c.append((name, "noslip",
                  ["--Nx", "64", "--Ny", str(TIM_NY), "--Ub", "0", "--nu", str(TIM_NU), "--init", "diffusion",
                   "--my", str(TIM_M), "--amp", str(TIM_AU), "--stretch", "none", "--no-adapt", "--dt", str(dt),
                   "--dtmax", "1", "--tend", str(TIM_T), "--snap_save", str(TIM_T), "--snap_fields", "omega",
                   "--diag_save", str(TIM_T)]))
    c.append(("poiseuille", "noslip",
              ["--Nx", "64", "--Ny", "129", "--Ub", str(POI_UB), "--Reb", str(POI_REB), "--init", "laminar"] +
              stretch_args(2) + ["--tend", str(POI_T), "--diag_save", "0.5", "--snap_save", str(POI_T),
                                 "--snap_fields", "omega_psi"]))
    for b in MMS_BETA:
        for n in MMS_N:
            c.append((f"mms_N{n}_b{b}", "noslip",
                      ["--Nx", str(n), "--Ny", str(n + 1), "--Ub", "0", "--nu", str(MMS_NU), "--init", "mms",
                       "--mmsA0", str(MMS_A0), "--mmsEps", str(MMS_EPS), "--mmsOm", str(MMS_OM),
                       "--mmsKx", str(MMS_KX)] + stretch_args(b) +
                      ["--no-adapt", "--dt", repr(mms_dt(n)), "--tend", str(MMS_T), "--snap_save", str(MMS_T),
                       "--snap_fields", "omega_psi", "--diag_save", str(MMS_T)]))
    for dt in CON_DT:
        c.append((f"cons_dt{dt:g}", "freeslip",
                  ["--Nx", "256", "--Ny", "257", "--Ub", "0", "--nu", "0", "--init", "rand", "--amp", "0.5",
                   "--seed", "7", "--stretch", "none", "--no-adapt", "--dt", str(dt), "--tend", str(CON_T),
                   "--diag_save", "0.2", "--snap_save", "0.4", "--snap_fields", "omega_psi"]))
    c.append(("channel", "noslip",
              ["--Nx", "512", "--Ny", "257", "--Ub", "1", "--Reb", str(CH_REB), "--init", "mix", "--amp", "0.05",
               "--F0", str(CH_F0), "--nforce", str(CH_NF), "--drag", str(CH_DRAG)] + stretch_args(2) +
              ["--tend", str(CH_T), "--diag_save", str(CH_DIAG), "--snap_save", str(CH_T), "--snap_fields", "omega_psi",
               "--closure_diag"]))
    return c


# ----------------------------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------------------------
def run_all(args):
    exes = {"noslip": args.exe, "freeslip": args.exe_freeslip}
    for key in ("noslip",):
        if not (exes[key] and os.path.isfile(exes[key])):
            sys.exit(f"ERROR: solver executable not found: {exes[key]}  (build it with `make`, or pass --exe)")
    have_fs = bool(exes["freeslip"]) and os.path.isfile(exes["freeslip"])
    if not have_fs:
        print(f"[run] free-slip executable not found ({exes['freeslip']}): the conservation test is skipped "
              f"(build it with `make WALLS=freeslip`)")
    os.makedirs(args.out, exist_ok=True)
    t_all = time.time()
    for name, key, a in cases():
        if args.only and not any(name.startswith(o) for o in args.only):
            continue
        if key == "freeslip" and not have_fs:
            continue
        d = os.path.join(args.out, name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        cmd = [os.path.abspath(exes[key])] + a + ["--outdir", d]
        with open(os.path.join(d, "command.txt"), "w") as f:
            f.write(" ".join([os.path.basename(exes[key])] + a + ["--outdir", "<case directory>"]) + "\n")
        t0 = time.time()
        with open(os.path.join(d, "run.log"), "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"[run] {name:24s} {'ok' if rc == 0 else 'EXIT CODE %d' % rc:14s} {time.time() - t0:7.1f} s", flush=True)
        if rc != 0:
            sys.exit(f"ERROR: case {name} failed; see {os.path.join(d, 'run.log')}")
    print(f"[run] all cases finished in {time.time() - t_all:.0f} s; output in {args.out}")


# ----------------------------------------------------------------------------------------------
# Reading output
# ----------------------------------------------------------------------------------------------
def read_diag(d):
    path = os.path.join(d, "diagnostics.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    a = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(a)


class Snap:
    pass


def read_snaps(d):
    """All snapshots of a run, sorted by time. Fields have shape [Ny, Nx]."""
    import h5py
    out = []
    for fn in sorted(glob.glob(os.path.join(d, "snaps", "snap_t*.h5"))):
        with h5py.File(fn, "r") as f:
            s = Snap()
            s.t = float(f.attrs["t"])
            s.Nx, s.Ny = int(f.attrs["Nx"]), int(f.attrs["Ny"])
            s.Lx, s.h, s.nu = float(f.attrs["Lx"]), float(f.attrs["h"]), float(f.attrs["nu"])
            s.omega = np.array(f["omega"], dtype=float).reshape(s.Ny, s.Nx)
            s.psi = np.array(f["psi"], dtype=float).reshape(s.Ny, s.Nx) if "psi" in f else None
            s.y = (np.array(f["y"], dtype=float).ravel() if "y" in f
                   else np.linspace(-s.h, s.h, s.Ny))
            s.x = np.arange(s.Nx) * (s.Lx / s.Nx)
        out.append(s)
    if not out:
        raise FileNotFoundError(f"no snapshots in {os.path.join(d, 'snaps')} (is the solver built with HDF5?)")
    return sorted(out, key=lambda s: s.t)


def last_snap(d, t_expected):
    s = read_snaps(d)[-1]
    if abs(s.t - t_expected) > 1e-6 * max(1.0, abs(t_expected)):
        raise RuntimeError(f"{d}: last snapshot is at t={s.t}, expected t={t_expected}")
    return s


def trap_weights(y):
    w = np.zeros_like(y)
    w[1:] += 0.5 * np.diff(y)
    w[:-1] += 0.5 * np.diff(y)
    return w


def norms(err, y):
    """Discrete L2 norm with trapezoid weights in y and the mean in x, and the maximum norm."""
    w = trap_weights(y)
    l2 = math.sqrt(float(np.sum(w[:, None] * err ** 2) / (err.shape[1] * np.sum(w))))
    return l2, float(np.max(np.abs(err)))


def order(e_coarse, e_fine, ratio=2.0):
    return math.log(e_coarse / e_fine) / math.log(ratio)


# ----------------------------------------------------------------------------------------------
# Tests. Each returns (metrics, checks): metrics = {name: value} compared with the reference,
# checks = [(description, passed, detail)] evaluated without reference data.
# ----------------------------------------------------------------------------------------------
def test_eigenmode_space(out):
    m, checks = {}, []
    mu = math.pi / H
    for b in EIG_BETA:
        e2 = []
        for ny in EIG_NY:
            s = last_snap(os.path.join(out, f"eig_space_Ny{ny}_b{b}"), EIG_T)
            prof = -mu * np.cos(mu * (s.y + H)) * math.exp(-EIG_NU * mu * mu * s.t)
            l2, li = norms(s.omega - prof[:, None], s.y)
            wall = 0.5 * (s.omega[0].mean() + s.omega[-1].mean())
            wall_ex = -mu * math.exp(-EIG_NU * mu * mu * s.t)
            m[f"E2_Ny{ny}_b{b}"], m[f"Einf_Ny{ny}_b{b}"] = l2, li
            m[f"wall_rel_Ny{ny}_b{b}"] = abs(wall - wall_ex) / abs(wall_ex)
            e2.append(l2)
        p = [order(e2[i], e2[i + 1]) for i in range(len(e2) - 1)]
        checks.append((f"beta={b}: observed order in space {', '.join('%.2f' % q for q in p)} (expected 2)",
                       1.7 <= p[-1] <= 2.3, "E2 = " + ", ".join("%.3e" % e for e in e2)))
    return m, checks


def test_eigenmode_time(out):
    m, checks = {}, []
    ref = last_snap(os.path.join(out, "eig_time_ref"), TIM_T)
    e2 = []
    for dt in TIM_DT:
        s = last_snap(os.path.join(out, f"eig_time_dt{dt:g}"), TIM_T)
        l2, _ = norms(s.omega - ref.omega, s.y)
        m[f"E2_dt{dt:g}"] = l2
        e2.append(l2)
    p = [order(e2[i], e2[i + 1]) for i in range(len(e2) - 1)]
    checks.append((f"observed order in time {', '.join('%.2f' % q for q in p)} (expected 3)",
                   2.5 <= 0.5 * (p[0] + p[1]) <= 3.5, "E2 = " + ", ".join("%.3e" % e for e in e2)))
    return m, checks


def test_poiseuille(out):
    m, checks = {}, []
    r = read_diag(os.path.join(out, "poiseuille"))[-1]
    nu = 2.0 * H * POI_UB / POI_REB
    exact = {"tauw_bot": 3 * nu * POI_UB / H, "tauw_top": -3 * nu * POI_UB / H, "dpdx": -3 * nu * POI_UB / H ** 2,
             "epsilon": 3 * nu * POI_UB ** 2 / H ** 2, "K": 0.6 * POI_UB ** 2}
    for k, ex in exact.items():
        dev = abs(float(r[k]) - ex) / abs(ex)
        m[f"dev_{k}"] = dev
        checks.append((f"{k} = {float(r[k]):+.6e}, exact {ex:+.6e}", dev < 5e-3, f"relative deviation {dev:.2e}"))
    bal = abs(-POI_UB * float(r["dpdx"]) - float(r["epsilon"])) / exact["epsilon"]
    m["energy_balance"] = bal
    checks.append(("energy balance -Ub dp/dx = epsilon", bal < 5e-3, f"relative mismatch {bal:.2e}"))
    return m, checks


def test_mms(out):
    m, checks = {}, []
    k = 2.0 * math.pi * MMS_KX / LX
    c = math.pi / (2.0 * H)
    for b in MMS_BETA:
        e2 = []
        for n in MMS_N:
            s = last_snap(os.path.join(out, f"mms_N{n}_b{b}"), MMS_T)
            amp = MMS_A0 * (1.0 + MMS_EPS * math.sin(MMS_OM * s.t))
            th = c * (s.y + H)
            g = k * k * np.sin(th) ** 2 - 2.0 * c * c * np.cos(2.0 * th)
            ex = amp * g[:, None] * np.sin(k * s.x)[None, :]
            l2, li = norms(s.omega - ex, s.y)
            m[f"E2_N{n}_b{b}"], m[f"Einf_N{n}_b{b}"] = l2, li
            e2.append(l2)
        p = [order(e2[i], e2[i + 1]) for i in range(len(e2) - 1)]
        checks.append((f"beta={b}: observed order {', '.join('%.2f' % q for q in p)} (expected 2)",
                       1.7 <= p[-1] <= 2.3, "E2 = " + ", ".join("%.3e" % e for e in e2)))
    return m, checks


def test_conservation(out):
    m, checks = {}, []
    drift = {}
    for dt in CON_DT:
        snaps = read_snaps(os.path.join(out, f"cons_dt{dt:g}"))
        dy = 2.0 * H / (snaps[0].Ny - 1)
        Z = np.array([0.5 * dy * np.sum(s.omega ** 2) / s.Nx for s in snaps])
        E = np.array([0.5 * dy * np.sum(s.psi * s.omega) / s.Nx for s in snaps])
        dZ, dE = float(np.max(np.abs(Z / Z[0] - 1.0))), float(np.max(np.abs(E / E[0] - 1.0)))
        m[f"drift_enstrophy_dt{dt:g}"], m[f"drift_energy_dt{dt:g}"] = dZ, dE
        drift[dt] = (dE, dZ)
        checks.append((f"dt={dt:g}: relative drift of the discrete energy {dE:.2e} and enstrophy {dZ:.2e} "
                       f"over t in [0,{CON_T:g}]", dE < 1e-6 and dZ < 1e-6, f"{len(snaps)} snapshots"))
    a, b = drift[CON_DT[0]], drift[CON_DT[1]]
    checks.append(("drift decreases when the time step is halved", b[0] < a[0] and b[1] < a[1],
                   f"ratios {a[0] / max(b[0], 1e-300):.1f} (energy), {a[1] / max(b[1], 1e-300):.1f} (enstrophy)"))
    return m, checks


CLOSURE_COLS = ["Omega_int_w", "dOmega_same_int_w", "A_w", "P_int_w", "eta_h", "Phi_h"]


def test_channel(out):
    m, checks = {}, []
    d = read_diag(os.path.join(out, "channel"))
    missing = [c for c in CLOSURE_COLS if c not in d.dtype.names]
    if missing:
        raise RuntimeError(f"diagnostics.csv has no column {missing[0]}: the case must run with --closure_diag")
    ok = np.isfinite(d["eta_h"])
    r = d[ok]
    resid = r["dOmega_same_int_w"] - (r["A_w"] - r["eta_h"] + r["Phi_h"] + r["P_int_w"] - 2.0 * CH_DRAG * r["Omega_int_w"])
    scale = float(np.mean(np.abs(r["eta_h"])))
    worst = float(np.max(np.abs(resid)) / scale)
    m["identity_max_over_eta_h"] = worst
    checks.append((f"scheme-consistent enstrophy identity at {len(r)} output times", worst < 1e-6,
                   f"max |residual| / mean eta_h = {worst:.2e} (the file stores 9 significant digits)"))
    late = r[r["t"] >= 0.5 * CH_T]
    m["eta_h_over_eta_late"] = float(np.mean(late["eta_h"]) / np.mean(late["eta"]))
    m["Phi_h_over_Phi_late"] = float(np.mean(late["Phi_h"]) / np.mean(late["BOmega_wall"]))
    early = d[(d["t"] > 0) & (d["t"] <= CH_TREF + 1e-9)]
    for col in ("K", "Omega", "eta", "epsilon"):
        m[f"early_mean_{col}"] = float(np.mean(early[col]))
    m[f"K_at_t{CH_TREF:g}"] = float(early["K"][-1])
    m[f"Omega_at_t{CH_TREF:g}"] = float(early["Omega"][-1])
    fin = np.all([np.all(np.isfinite(d[c][1:])) for c in ("K", "Omega", "eta", "epsilon", "Re_tau")])
    # the last diagnostic row is written one output interval before the end time
    checks.append((f"run reached t = {float(d['t'][-1]):g} with finite diagnostics",
                   bool(fin) and float(d["t"][-1]) > CH_T - CH_DIAG - 1e-6, f"{len(d)} rows"))
    return m, checks


TESTS = [("eigenmode_space", test_eigenmode_space), ("eigenmode_time", test_eigenmode_time),
         ("poiseuille", test_poiseuille), ("mms", test_mms), ("conservation", test_conservation),
         ("channel", test_channel)]

# Relative tolerance of the comparison with the reference values. Errors, deviations and drifts are small
# differences, so they are compared loosely; the channel values are means of O(1) quantities.
REF_TOL = {"eigenmode_space": 0.02, "eigenmode_time": 0.05, "poiseuille": 0.05, "mms": 0.02,
           "conservation": 0.5, "channel": 1e-4}
REF_SKIP = {"channel": ("identity_max_over_eta_h", "eta_h_over_eta_late", "Phi_h_over_Phi_late")}
REF_ABS_FLOOR = 1e-13      # values below this are round-off and are not compared


def check_all(args):
    ref = None
    if args.reference and os.path.isfile(args.reference) and not args.write_reference:
        with open(args.reference) as f:
            ref = json.load(f)
    results, new_ref, n_fail, n_skip = {}, {}, 0, 0
    for name, fn in TESTS:
        if args.only and not any(name.startswith(o) for o in args.only):
            continue
        if name == "conservation" and not os.path.isdir(os.path.join(args.out, f"cons_dt{CON_DT[0]:g}")):
            print(f"SKIP  {name}: no output (free-slip build not run)")
            n_skip += 1
            continue
        try:
            metrics, checks = fn(args.out)
        except Exception as e:                       # missing files, wrong columns, ...
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            n_fail += 1
            continue
        new_ref[name] = metrics
        lines, passed = [], True
        for desc, ok, detail in checks:
            lines.append(f"      [{'ok' if ok else 'FAILED'}] {desc}   ({detail})")
            passed = passed and ok
        if ref is not None and name in ref:
            tol, bad = REF_TOL[name], []
            for k, v in metrics.items():
                if k in REF_SKIP.get(name, ()) or k not in ref[name]:
                    continue
                r = ref[name][k]
                if max(abs(v), abs(r)) < REF_ABS_FLOOR:
                    continue
                if abs(v - r) > tol * max(abs(r), REF_ABS_FLOOR):
                    bad.append(f"{k}: {v:.6e} vs reference {r:.6e}")
            lines.append(f"      [{'ok' if not bad else 'FAILED'}] {len(metrics)} values against the reference "
                         f"(relative tolerance {tol:g})" + ("" if not bad else "\n          " + "\n          ".join(bad)))
            passed = passed and not bad
        elif not args.write_reference:
            lines.append("      [--] no reference data: only the exact checks were evaluated")
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        print("\n".join(lines))
        n_fail += 0 if passed else 1
        results[name] = passed
    print(f"\n{sum(results.values())} passed, {n_fail} failed, {n_skip} skipped")
    if args.write_reference:
        os.makedirs(args.write_reference, exist_ok=True)
        with open(os.path.join(args.write_reference, "reference.json"), "w") as f:
            json.dump(new_ref, f, indent=1, sort_keys=True)
        for name, _, _ in cases():
            src = os.path.join(args.out, name)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(args.write_reference, "output", name)
            os.makedirs(dst, exist_ok=True)
            for fn in ("command.txt", "diagnostics.csv", "progress.log", "profiles_timeavg.csv"):
                if os.path.isfile(os.path.join(src, fn)):
                    shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))
        print(f"[reference] wrote {os.path.join(args.write_reference, 'reference.json')} and output/")
    return 1 if n_fail else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "check", "all", "list"])
    ap.add_argument("--exe", default=os.path.join(ROOT, "bin", "nse2d"))
    ap.add_argument("--exe-freeslip", default=os.path.join(ROOT, "bin", "nse2d_freeslip"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--reference", default=os.path.join(HERE, "reference", "reference.json"))
    ap.add_argument("--write-reference", metavar="DIR", help="write reference.json and sample outputs to DIR")
    ap.add_argument("--only", nargs="+", help="restrict to tests / cases whose name starts with these strings")
    args = ap.parse_args()
    if args.mode == "list":
        for name, key, a in cases():
            print(f"{name:24s} {'nse2d_freeslip' if key == 'freeslip' else 'nse2d'} {' '.join(a)}")
        return 0
    if args.mode in ("run", "all"):
        run_all(args)
    if args.mode in ("check", "all"):
        return check_all(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
