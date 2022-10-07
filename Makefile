#compilers
CC=nvcc

#GLOBAL_PARAMETERS
 
#Defines the parameter data type Possible Values: int, float, double
VALUE_TYPE = double
#Value 1 allows for hardware reduccion 
AMPERE = 0

## Optmized Compiler Flags
NVCC_FLAGS = -Xcompiler -fopenmp -O3 -w -m64 -gencode=arch=compute_86,code=sm_86 -Xptxas -dlcm=cg
#NVCC_FLAGS = -Xcompiler -ftree-vectorize -Xcompiler -fopenmp -O3 -w -m64 -gencode=arch=compute_75,code=sm_75 -Xptxas -dlcm=cg #60 para tener warps sincrónicos
## Debugging Compiler Flags
#NVCC_FLAGS = -g  -G -Xcompiler -fopenmp -O3 -w -m64 -gencode=arch=compute_75,code=sm_75 -Xptxas -dlcm=cg #60 para tener warps sincrónicos
#NVCC_FLAGS = -g  -lineinfo -Xcompiler -fopenmp -O3 -w -m64 -gencode=arch=compute_75,code=sm_75 -Xptxas -dlcm=cg #Show line error

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /usr/local/cuda
MKLROOT = /home/gpgpu/software/mkl
#includes
INCLUDES = -I./header -I$(CUDA_INSTALL_PATH)/include -I${MKLROOT}/include
        
#libs

MKL_LIBS =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -ltbb -lstdc++ -lpthread -lm -ldl
MKL_LIBS = -lpthread -lm
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib  -lcudart -lcuda -lcusparse -lnvidia-ml # -lgomp

LIBS = $(CUDA_LIBS) $(CLANG_LIBS) $(MKL_LIBS) 
FILES = solver/solve_csr_simple.cu \
	solver/solve_csr_multirow.cu \
        solver/solve_csr_cusparse.cu \
	solver/solve_csr_order.cu \
        analysis/analysis_csr.cu \
        main.cu \
        source/nvmlPower.cpp \
        test.cu \

make:
	$(CC) $(NVCC_FLAGS) $(FILES) --keep-dir local -o sptrsv_$(VALUE_TYPE) $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D__$(VALUE_TYPE)__ -D AMPERE=$(AMPERE) 
