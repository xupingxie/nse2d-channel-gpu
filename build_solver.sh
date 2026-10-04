#!/bin/bash
# Simple NVCC builder for Perlmutter (A100, sm_80)
# Default: BIN (no HDF5). Use 'bash build_gpu.sh hdf5' to enable HDF5.
# Optional env: SRC=2dchannel_gpu.cu  OUT=2dchannel_gpu  CUDA_ARCH=sm_80  VERBOSE=1
set -euo pipefail
[[ "${VERBOSE:-0}" == "1" ]] && set -x

mode="${1:-hdf5}"
wall="${2:-noslip}"

SRC="${SRC:-nse2d_ssprk3.cu}"
OUT="${OUT:-${SRC%.cu}}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"

module load PrgEnv-nvidia
module load cudatoolkit
[[ "$mode" == "hdf5" ]] && module load cray-hdf5

NVCC=$(which nvcc)
[[ -z "${NVCC}" ]] && { echo "ERROR: nvcc not found (module load cudatoolkit?)"; exit 1; }

# Example: /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/nvcc
CUDA_TOP=$(dirname "$(dirname "$NVCC")")        # .../cuda/12.4
CUDA_VER=$(basename "$CUDA_TOP")                # <-- 12.4 (NOT 'bin')
SDK_ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$NVCC")")")")  # .../Linux_x86_64/24.5

CUDA_INC="$CUDA_TOP/include"
CUDA_LIB="$CUDA_TOP/lib64"
MATH_LIB="$SDK_ROOT/math_libs/$CUDA_VER/lib64"  # cuFFT & cuSPARSE live here in NVHPC

echo "CUDA include : $CUDA_INC"
echo "CUDA libdir  : $CUDA_LIB"
echo "MATH libdir  : $MATH_LIB"

# quick sanity
[[ -e "$MATH_LIB/libcufft.so"    || -e "$MATH_LIB/libcufft_static.a"   ]] || { echo "cuFFT not in $MATH_LIB"; exit 1; }
[[ -e "$MATH_LIB/libcusparse.so" || -e "$MATH_LIB/libcusparse_static.a" ]] || { echo "cuSPARSE not in $MATH_LIB"; exit 1; }

# Optional HDF5
H5_INC=""; H5_LIBS=""; H5_DEF=""
if [[ "$mode" == "hdf5" ]]; then
  if which h5cc >/dev/null 2>&1; then
    H5ROOT=$(dirname "$(dirname "$(which h5cc)")")
  else
    H5ROOT=$(ls -d /opt/cray/pe/hdf5/*/nvidia/* 2>/dev/null | sort -V | tail -1 || true)
  fi
  [[ -z "${H5ROOT:-}" ]] && { echo "HDF5 not found. Try: module load cray-hdf5"; exit 1; }
  H5_INC="-I${H5ROOT}/include"
  H5_LIBS="-L${H5ROOT}/lib -lhdf5"
  H5_DEF="-DUSE_HDF5"
  echo "HDF5 include : ${H5ROOT}/include"
  echo "HDF5 libdir  : ${H5ROOT}/lib"
fi

NVCC_FLAGS="--extended-lambda -O3 -arch=${CUDA_ARCH} -use_fast_math -lineinfo"

# optional free-slip walls build
if [[ "$wall" == "freeslip" ]]; then
  NVCC_FLAGS="${NVCC_FLAGS} -DFREE_SLIP_WALLS"
  echo "Compiling with FREE_SLIP_WALLS (free-slip walls)"
else
  echo "Compiling with Thom no-slip walls (default)"
fi

INCS="-I${CUDA_INC} ${H5_INC}"
LIBS="-L${CUDA_LIB} -L${MATH_LIB} -lcufft -lcusparse ${H5_LIBS}"

echo "Command:"
echo nvcc ${NVCC_FLAGS} "${SRC}" ${INCS} ${LIBS} ${H5_DEF} -o "${OUT}"
nvcc ${NVCC_FLAGS} "${SRC}" ${INCS} ${LIBS} ${H5_DEF} -o "${OUT}"

echo "Built ./${OUT} (${mode})"
echo "If the runtime loader complains, run:"
echo "  export LD_LIBRARY_PATH=${CUDA_LIB}:${MATH_LIB}${mode:+:${H5ROOT}/lib}:\$LD_LIBRARY_PATH"
