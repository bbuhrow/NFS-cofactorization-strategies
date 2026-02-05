
# ============================================================================
# Makefile
# ============================================================================
CC = gcc
CUDA_PATH = /usr/local/cuda-12
NVCC = $(CUDA_PATH)/bin/nvcc
CFLAGS = -I$(CUDA_PATH)/include -I../gmp-install/6.2.0-gcc/include  -I. -Iytools -Iysieve -Iaprcl -O2 -DHAVE_CUDA -fno-common -mbmi2
LDFLAGS = -L$(CUDA_PATH)/lib64 -L../gmp-install/6.2.0/lib -Lysieve -Lytools -Laprcl -lcudart -lgmp -lm -ldl -lcuda -pthread
SM = 80


ifeq ($(ICELAKE),1)
	CFLAGS += -DUSE_BMI2 -DUSE_AVX2 -DUSE_AVX512F -DUSE_AVX512BW -DSKYLAKEX -DIFMA -march=icelake-client
	SKYLAKEX = 1
else

ifeq ($(SKYLAKEX),1)
	CFLAGS += -DUSE_BMI2 -DUSE_AVX2 -DUSE_AVX512F -DUSE_AVX512BW -DSKYLAKEX -march=skylake-avx512 
endif
	
endif

MAIN_SRC = \
	main.c \
	gpu_cofactorization.c \
	cuda_xface.c \
	util.c \
	monty.c \
	arith.c \
	cmdOptions.c

YSIEVE_SRC = \
	ysieve/presieve.c \
	ysieve/count.c \
	ysieve/offsets.c \
	ysieve/primes.c \
	ysieve/roots.c \
	ysieve/linesieve.c \
	ysieve/soe.c \
	ysieve/tiny.c \
	ysieve/worker.c \
	ysieve/soe_util.c \
	ysieve/wrapper.c \

YTOOLS_SRC = \
	ytools/threadpool.c \
	ytools/ytools.c \
	aprcl/mpz_aprcl.c \
	aprcl/tinyprp.c

BATCHGCD_SRC = \
	batch_factor.c \
	microecm.c \
	tinyecm.c \
	cofactorize_siqs.c

HEADERS = \
	cuda_xface.h \
	util.h \
	aprcl/jacobi_sum.h \
	aprcl/mpz_aprcl.h \
	aprcl/tinyprp.h \
	batch_factor.h \
	cofactorize.h \
	microecm.h \
	tinyecm.h \
	gpu_cofactorization.h \
	ytools/threadpool.h \
	ytools/ytools.h \
	ysieve/soe_impl.h \
	ysieve/soe.h \
	monty.h \
	common.h \
	arith.h \
	cmdOptions.h

YSIEVE_OBJS = $(YSIEVE_SRC:.c=.o)
YTOOLS_OBJS = $(YTOOLS_SRC:.c=.o)
BATCHGCD_OBJS = $(BATCHGCD_SRC:.c=.o)
MAIN_OBJS = $(MAIN_SRC:.c=.o)

all: cuda_3lp

cuda_3lp: $(MAIN_OBJS) $(YSIEVE_OBJS) $(YTOOLS_OBJS) $(BATCHGCD_OBJS) cuda_ecm$(SM).ptx
	$(CC) $(CFLAGS) -o cuda_3lp $(MAIN_OBJS) $(YSIEVE_OBJS) $(YTOOLS_OBJS) \
	$(BATCHGCD_OBJS) $(LDFLAGS)

# build rules

cuda_ecm$(SM).ptx: cuda_ecm64.cu cuda_intrinsics.h
	$(NVCC) -arch sm_$(SM) -ptx -o $@ $<

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o ysieve/*.o ytools/*.o aprcl/*.o cuda_3lp