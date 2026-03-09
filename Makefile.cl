# =============================================================================
# Makefile -- OpenCL ECM / cofactorization project
#
# Targets
# -------
#   make              -- build gpu_cofactor (default)
#   make test_ocl     -- build the standalone array-addition smoke test
#   make clean        -- remove build artefacts
#
# Platform detection
# ------------------
# Primary target: MSYS2 MinGW64 on Windows with AMD Adrenalin OpenCL.
#   pacman packages needed:
#     mingw-w64-x86_64-gcc
#     mingw-w64-x86_64-opencl-headers
#     mingw-w64-x86_64-opencl-icd
#     mingw-w64-x86_64-gmp  (or build GMP yourself)
#
# The Makefile also detects Linux automatically so the same file works
# on a Linux dev machine (e.g. with ROCm or Mesa Rusticl OpenCL).
#
# Usage on MSYS2:
#   cd <project_dir>
#   make
#
# Usage on Linux:
#   make
# =============================================================================

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
    PLATFORM   := windows
    EXE_SUFFIX := .exe
else
    UNAME := $(shell uname -s)
    ifeq ($(UNAME),Linux)
        PLATFORM   := linux
        EXE_SUFFIX :=
    else
        PLATFORM   := unknown
        EXE_SUFFIX :=
    endif
endif

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
CC      := gcc
AR      := ar
ARFLAGS := rcs

# ---------------------------------------------------------------------------
# OpenCL paths
#
# MSYS2/MinGW64:  opencl-icd installs headers to /mingw64/include/CL and
#                 the import library to /mingw64/lib/libOpenCL.a
# Linux (system): /usr/include/CL  +  -lOpenCL  (from ocl-icd or vendor)
# Linux (ROCm):   /opt/rocm/include + /opt/rocm/lib
#
# Override on the command line if your layout differs:
#   make OCL_INCLUDE=/my/path/include OCL_LIB=/my/path/lib
# ---------------------------------------------------------------------------
ifeq ($(PLATFORM),windows)
    # MSYS2 MinGW64 default layout
    OCL_INCLUDE ?= /mingw64/include
    OCL_LIB     ?= /mingw64/lib
    OCL_LDFLAGS  = -L$(OCL_LIB) -lOpenCL
else
    # Linux: try ROCm first, fall back to system
    ifneq ($(wildcard /opt/rocm/include/CL/cl.h),)
        OCL_INCLUDE ?= /opt/rocm/include
        OCL_LIB     ?= /opt/rocm/lib
    else
        OCL_INCLUDE ?= /usr/include
        OCL_LIB     ?= /usr/lib
    endif
    OCL_LDFLAGS = -L$(OCL_LIB) -lOpenCL
endif

# ---------------------------------------------------------------------------
# GMP paths
#
# MSYS2: mingw-w64-x86_64-gmp puts gmp.h in /mingw64/include
# Linux: usually /usr/include with -lgmp
# Override: make GMP_INCLUDE=/path GMP_LIB=/path
# ---------------------------------------------------------------------------
ifeq ($(PLATFORM),windows)
    GMP_INCLUDE ?= /mingw64/include
    GMP_LIB     ?= /mingw64/lib
else
    GMP_INCLUDE ?= /usr/include
    GMP_LIB     ?= /usr/lib
endif
GMP_LDFLAGS = -L$(GMP_LIB) -lgmp

# ---------------------------------------------------------------------------
# Project source layout
#
# Assumption: all translated .c files live in the same directory as this
# Makefile.  Headers for stubs (batch_factor.h, microecm.h) are expected
# here too.  Adjust SRC_DIR / INC_DIR if your tree differs.
# ---------------------------------------------------------------------------
SRC_DIR := .
OBJ_DIR := obj

# Source files we own
SRCS := \
    $(SRC_DIR)/ocl_xface.c          \
    $(SRC_DIR)/gpu_cofactorization_cl.c \
    $(SRC_DIR)/batch_factor.c \
    $(SRC_DIR)/ytools/ytools.c \
    $(SRC_DIR)/ytools/threadpool.c \
    $(SRC_DIR)/microecm.c \
    $(SRC_DIR)/tinyecm.c \
    $(SRC_DIR)/monty.c \
    $(SRC_DIR)/cmdOptions.c \
	$(SRC_DIR)/cofactorize_siqs.c \
    $(SRC_DIR)/arith.c \
    $(SRC_DIR)/main_cl.c \
    $(SRC_DIR)/mpz-ull.c \
    $(SRC_DIR)/ysieve/presieve.c \
	$(SRC_DIR)/ysieve/count.c \
	$(SRC_DIR)/ysieve/offsets.c \
	$(SRC_DIR)/ysieve/primes.c \
	$(SRC_DIR)/ysieve/roots.c \
	$(SRC_DIR)/ysieve/linesieve.c \
	$(SRC_DIR)/ysieve/soe.c \
	$(SRC_DIR)/ysieve/tiny.c \
	$(SRC_DIR)/ysieve/worker.c \
	$(SRC_DIR)/ysieve/soe_util.c \
	$(SRC_DIR)/ysieve/wrapper.c \
    $(SRC_DIR)/aprcl/mpz_aprcl.c \
	$(SRC_DIR)/aprcl/tinyprp.c

# The standalone smoke-test (optional, doesn't need GMP)
TEST_SRC := $(SRC_DIR)/opencl_add_arrays.c

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
CFLAGS := \
    -std=c11                        \
    -O2                             \
    -Wall -Wextra                   \
    -Wno-unused-parameter           \
    -DHAVE_CUDA_BATCH_FACTOR        \
    -DULL_NO_UL                     \
    -DBITS_PER_GMP_ULONG=32         \
    -I$(SRC_DIR)                    \
    -I$(SRC_DIR)/ysieve/            \
    -I$(SRC_DIR)/aprcl/             \
	-I$(SRC_DIR)/ytools/            \
    -I$(OCL_INCLUDE)                \
    -I$(GMP_INCLUDE)

# Debug build: make DEBUG=1
ifeq ($(DEBUG),1)
    CFLAGS += -g -O0 -DDEBUG
endif

# ---------------------------------------------------------------------------
# Linker flags
# ---------------------------------------------------------------------------

# note: liblasieve must be listed before lgmp
LDFLAGS := \
    $(OCL_LDFLAGS)  \
	-llasieve \
    $(GMP_LDFLAGS)  \
	-Lmpqs3 \
    -lm

# On Windows, also link the math and C runtime explicitly when using MinGW
ifeq ($(PLATFORM),windows)
    LDFLAGS += -static-libgcc
endif

# ---------------------------------------------------------------------------
# Primary target
# ---------------------------------------------------------------------------
TARGET := gpu_cofactor$(EXE_SUFFIX)

.PHONY: all clean test_ocl

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
	@echo "Built $@"

# ---------------------------------------------------------------------------
# Compilation rules
# ---------------------------------------------------------------------------
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR) \
    mkdir -p $(OBJ_DIR)/ysieve \
    mkdir -p $(OBJ_DIR)/aprcl \
	mkdir -p $(OBJ_DIR)/ytools

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# ---------------------------------------------------------------------------
# Standalone OpenCL smoke test (no GMP, no batch_factor deps)
# ---------------------------------------------------------------------------
test_ocl: $(TEST_SRC)
	$(CC) -std=c11 -O2 -Wall \
	    -I$(OCL_INCLUDE) \
	    -o opencl_add_test$(EXE_SUFFIX) \
	    $(TEST_SRC) \
	    $(OCL_LDFLAGS) -lm
	@echo "Built opencl_add_test$(EXE_SUFFIX)"

# ---------------------------------------------------------------------------
# Dependency tracking (auto-generated .d files)
# ---------------------------------------------------------------------------
DEPS := $(OBJS:.o=.d)
-include $(DEPS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

# 

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -rf $(OBJ_DIR)
	rm -f $(TARGET) opencl_add_test$(EXE_SUFFIX)
	@echo "Cleaned"

# ---------------------------------------------------------------------------
# Help / diagnostics
# ---------------------------------------------------------------------------
.PHONY: info
info:
	@echo "PLATFORM    = $(PLATFORM)"
	@echo "CC          = $(CC)"
	@echo "OCL_INCLUDE = $(OCL_INCLUDE)"
	@echo "OCL_LIB     = $(OCL_LIB)"
	@echo "GMP_INCLUDE = $(GMP_INCLUDE)"
	@echo "GMP_LIB     = $(GMP_LIB)"
	@echo "CFLAGS      = $(CFLAGS)"
	@echo "LDFLAGS     = $(LDFLAGS)"
	@echo "SRCS        = $(SRCS)"
	@echo "OBJS        = $(OBJS)"
