#!/bin/bash
# Build helper for NERSC Perlmutter (A100, sm_80): loads the modules, locates cuFFT/cuSPARSE and HDF5
# inside the NVIDIA HPC SDK / Cray PE, and calls make with those paths.
#
#   bash build_solver.sh                 # HDF5 on, no-slip walls      -> bin/nse2d
#   bash build_solver.sh hdf5 freeslip   # free-slip walls             -> bin/nse2d_freeslip
#   bash build_solver.sh bin             # without HDF5
#   CUDA_ARCH=sm_90 bash build_solver.sh # other architecture
#
# On other systems call make directly, e.g.
#   make CUDA_HOME=/usr/local/cuda HDF5_ROOT=/usr ARCH=sm_86
set -euo pipefail
[[ "${VERBOSE:-0}" == "1" ]] && set -x

mode="${1:-hdf5}"
wall="${2:-noslip}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"

module load PrgEnv-nvidia
module load cudatoolkit
[[ "$mode" == "hdf5" ]] && module load cray-hdf5

NVCC=$(which nvcc) || { echo "ERROR: nvcc not found (module load cudatoolkit?)"; exit 1; }
CUDA_TOP=$(dirname "$(dirname "$NVCC")")                 # .../cuda/12.4
CUDA_VER=$(basename "$CUDA_TOP")
SDK_ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$NVCC")")")")
MATH_LIB="$SDK_ROOT/math_libs/$CUDA_VER/lib64"           # cuFFT & cuSPARSE live here in NVHPC

[[ -e "$MATH_LIB/libcufft.so" || -e "$MATH_LIB/libcufft_static.a" ]] || { echo "cuFFT not in $MATH_LIB"; exit 1; }

MAKEARGS=(CUDA_HOME="$CUDA_TOP" MATH_LIB="$MATH_LIB" ARCH="$CUDA_ARCH")
if [[ "$mode" == "hdf5" ]]; then
  if which h5cc >/dev/null 2>&1; then
    H5ROOT=$(dirname "$(dirname "$(which h5cc)")")
  else
    H5ROOT=$(ls -d /opt/cray/pe/hdf5/*/nvidia/* 2>/dev/null | sort -V | tail -1 || true)
  fi
  [[ -z "${H5ROOT:-}" ]] && { echo "HDF5 not found. Try: module load cray-hdf5"; exit 1; }
  MAKEARGS+=(HDF5=1 HDF5_ROOT="$H5ROOT")
else
  MAKEARGS+=(HDF5=0)
fi
[[ "$wall" == "freeslip" ]] && MAKEARGS+=(WALLS=freeslip)

echo "make ${MAKEARGS[*]}"
make "${MAKEARGS[@]}"
echo "If the runtime loader complains, run:"
echo "  export LD_LIBRARY_PATH=$CUDA_TOP/lib64:$MATH_LIB${H5ROOT:+:$H5ROOT/lib}:\$LD_LIBRARY_PATH"
