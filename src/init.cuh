// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Command-line parsing and initial conditions.
#pragma once
#include "common.cuh"

void parse_args(int argc, char** argv, Grid& G, Params& P,
                       int& statsEvery, int& timeEvery, bool& profile,
                       double& snap_save_dt, std::string& snap_fmt_str,
                       double& diag_save_dt,
                       int& spectraEvery,
                       int& profilesWriteEvery,
                       std::string& outdir);
void init_omega_with_IC(const Grid& G, const Params& P, std::vector<Real>& h_w);
