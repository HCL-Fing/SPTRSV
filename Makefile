#INCLUDECUDA = -I/usr/local/cuda-7.0/samples/common/inc/
# HEADERNVMLAPI = -L/usr/lib64/nvidia -lnvidia-ml -L/usr/lib64 -lcuda -I/usr/include -lpthread

#compilers
#CC=gcc
CC=nvcc

#GLOBAL_PARAMETERS
VALUE_TYPE = double
AMPERE = 0	#1 para usar el reduce en HW 
OLD=0
LOG_FILE =\"NONE\"
TIMER_SOLVERS=0
PRINT_TIME_ANALYSIS=0

#CUDA_PARAMETERS
# NVCC_FLAGS = -g -O3 -w -m64 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30
# NVCC_FLAGS = -O3 -w -m64 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52

#NVCC_FLAGS = -Xcompiler -fopenmp -O3 -w -m64 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 -Xptxas -dlcm=cg
#NVCC_FLAGS = -O3 -w -m64 -arch=sm_30

## Flags para Compilar Optmiziado
NVCC_FLAGS =  -Xcompiler -ftree-vectorize -Xcompiler -fopenmp -O3 -w -m64 -gencode=arch=compute_86,code=sm_86 -Xptxas -dlcm=cg
#NVCC_FLAGS = -Xcompiler -ftree-vectorize -Xcompiler -fopenmp -O3 -w -m64 -gencode=arch=compute_75,code=sm_75 -Xptxas -dlcm=cg #60 para tener warps sincrónicos
## Flags para Compilar y Debuggear
#NVCC_FLAGS = -g  -G -Xcompiler -fopenmp -O3 -w -m64 -gencode=arch=compute_75,code=sm_75 -Xptxas -dlcm=cg #60 para tener warps sincrónicos
#NVCC_FLAGS = -g  -lineinfo -Xcompiler -fopenmp -O3 -w -m64 -gencode=arch=compute_75,code=sm_75 -Xptxas -dlcm=cg #Show line error

##NVCC_FLAGS = -Xcompiler -fopenmp -O3 -w -m64 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -Xptxas -dlcm=cg
#NVCC_FLAGS = -O3 -std=c99 -w -m64

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /usr/local/cuda
#MKLROOT = /opt/intel/mkl
MKLROOT = /home/gpgpu/software/mkl
#includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I./include -I${MKLROOT}/include
        
#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
MKL_LIBS =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -ltbb -lstdc++ -lpthread -lm -ldl
MKL_LIBS = -lpthread -lm# -L/opt/intel/lib/intel64_lin ${MKLROOT}/lib/intel64_lin/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64_lin/libmkl_intel_thread.a ${MKLROOT}/lib/intel64_lin/libmkl_core.a -lstdc++ -lpthread -lm -ldl -liomp5
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib  -lcudart -lcuda -lcusparse -lnvidia-ml # -lgomp
#LIBS = $(MKL_LIBS)
LIBS = $(CUDA_LIBS) $(CLANG_LIBS) $(MKL_LIBS)
#options
#OPTIONS = -std=c99

MAIN=main
#FILES = main.c 
 
FILES = solver/solve_csr_simple.cu \
	solver/solve_csr_multirow.cu \
	solver/solve_csr_multirow_format.cu \
        solver/solve_csr_cusparse.cu \
	solver/solve_csr_order.cu \
	solver/solve_csr_no_lvl.cu \
        analysis_csr.cu \
        $(MAIN).cu \
        nvmlPower.cpp \
        test/test.cu \

#FILES = analysis_csr.cu \
        test.cu \
        main.cu

make:
	$(CC) $(NVCC_FLAGS) $(FILES) --keep-dir local -o sptrsv_$(VALUE_TYPE)_$(MAIN) $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D__$(VALUE_TYPE)__ -D AMPERE=$(AMPERE) -D OLD_ANALYSIS=$(OLD) -D LOG_FILE=$(LOG_FILE) -D TIMERS_SOLVERS=$(TIMER_SOLVERS) 
