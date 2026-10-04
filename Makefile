# nse2d-channel-gpu — build the solver from the modular sources in src/
#
#   make                 # no-slip walls, HDF5 on          -> bin/nse2d
#   make WALLS=freeslip  # free-slip walls (omega_wall = 0) -> bin/nse2d_freeslip
#   make HDF5=0          # without HDF5 (no snapshots / restart / tabulated grids)
#   make ARCH=sm_90      # other GPU architecture (default sm_80, NVIDIA A100)
#   make TESTAPI=1       # also compile the C-linkage operator wrappers (src/test_api.cu)
#   make clean
#
# Library locations can be given on the command line or in the environment, e.g.
#   make CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4 MATH_LIB=.../math_libs/12.4/lib64 HDF5_ROOT=...
# On NERSC Perlmutter, build_solver.sh loads the modules and fills these in.

NVCC      ?= nvcc
ARCH      ?= sm_80
HDF5      ?= 1
WALLS     ?= noslip
TESTAPI   ?= 0
OPT       ?= -O3 -use_fast_math -lineinfo

SRCDIR    := src
BINDIR    := bin
OBJDIR    := build

SRCS      := common.cuh io.cu grid.cu ops.cu diag.cu poisson.cu init.cu main.cu
CU        := $(filter %.cu,$(SRCS))
ifeq ($(TESTAPI),1)
  CU      += test_api.cu
  DEFS    += -DEXPORT_TEST_API
endif
OBJS      := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU))

NVCCFLAGS := --extended-lambda $(OPT) -arch=$(ARCH) -std=c++17
INCS      :=
LIBS      := -lcufft -lcusparse

ifdef CUDA_HOME
  INCS    += -I$(CUDA_HOME)/include
  LIBS    += -L$(CUDA_HOME)/lib64
endif
ifdef MATH_LIB
  LIBS    += -L$(MATH_LIB)
endif

ifeq ($(HDF5),1)
  DEFS    += -DUSE_HDF5
  ifdef HDF5_ROOT
    INCS  += -I$(HDF5_ROOT)/include
    LIBS  += -L$(HDF5_ROOT)/lib
  endif
  LIBS    += -lhdf5
else
  DEFS    += -DNO_HDF5
endif

TARGET    := $(BINDIR)/nse2d
ifeq ($(WALLS),freeslip)
  DEFS    += -DFREE_SLIP_WALLS
  TARGET  := $(BINDIR)/nse2d_freeslip
  OBJDIR  := build_freeslip
  OBJS    := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU))
endif

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJS) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LIBS) -o $@
	@echo "Built $@  (walls=$(WALLS), hdf5=$(HDF5), arch=$(ARCH))"

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(SRCDIR)/*.cuh | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $(DEFS) $(INCS) -c $< -o $@

$(BINDIR) $(OBJDIR):
	mkdir -p $@

clean:
	rm -rf build build_freeslip bin
