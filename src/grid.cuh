// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Wall-normal grid metrics (uniform, tanh, tabulated).
#pragma once
#include "common.cuh"

void build_y_metrics(const Grid& Gin, const Params& P, Grid& Gout);
