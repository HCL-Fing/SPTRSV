#ifndef __COMMON__
#define __COMMON__

#define MAX(A,B)        (((A)>(B))?(A):(B))
#define MIN(A,B)        (((A)<(B))?(A):(B))


#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>

#include <sys/time.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#ifdef _MKL_
	#include "mkl.h"
	#include "mkl_blas.h"
	#include "mkl_spblas.h"
	#include "mkl_rci.h"
#endif

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef BENCH_REPEAT
#define BENCH_REPEAT  1 
#endif

#ifndef WARP_SIZE
#define WARP_SIZE   32
#endif


#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK   32
#endif

#ifndef SIZE_SHARED
#define SIZE_SHARED 512
#endif

#ifndef ROWS_PER_THREAD
#define ROWS_PER_THREAD   1
#endif

#define PRINT_TIME_ANALYSIS 1 

#define CUDA_EVTS 1
#define POSIX 2


#define CLK CUDA_EVTS

float clockElapsed(cudaEvent_t evt_start, cudaEvent_t evt_stop);

#if CLK == CUDA_EVTS

	#define CLK_INIT \
			cudaEvent_t evt_start, evt_stop; \
			printf("Usando CUDA EVENTS para medir el tiempo\n"); \
			float t_elap; \
			cudaEventCreate(&evt_start); \
			cudaEventCreate(&evt_stop) ;

	#define CLK_START \
        cudaEventRecord(evt_start, 0);


	#define CLK_STOP \
        cudaEventRecord(evt_stop, 0); 


	#define CLK_ELAPSED \
			t_elap = clockElapsed(evt_start, evt_stop)
#else
	#define CLK_INIT \
			printf("Usando gettimeofday para medir el tiempo\n"); \
			struct timeval t_i, t_f; \
			float t_elap

	#define CLK_START \
			gettimeofday(&t_i,NULL)
	#define CLK_STOP \
			gettimeofday(&t_f,NULL) 
	#define CLK_ELAPSED \
			t_elap = ((double) t_f.tv_sec * 1000.0 + (double) t_f.tv_usec / 1000.0 - \
				 	 ((double) t_i.tv_sec * 1000.0 + (double) t_i.tv_usec / 1000.0))
#endif




#define FULL_MASK 0xffffffff
#define IT_HASH 200

#define BENCH_RUN_SOLVE(f,t,t2,pow,s)                                             					\
	float t, t2;																					\																				
	float pow;  																					\																					
	nvmlAPIRun(s);	 																				\
	t = 0;                                                              				 			\
	for (int i = 0; i < BENCH_REPEAT; ++i){															\
		CLK_START;			                                                        				\
		cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE));                                 				\
		f;                                                                          				\
		CLK_STOP;       		                                                    				\
		CLK_ELAPSED;   			    	                                            				\
		t += t_elap;																				\
		t2 += (1/double(i+1))*(t_elap*t_elap - t2);	                       							\
																									\
	}	                                                                            				\
	t = t / BENCH_REPEAT;																			\
	t2 = sqrt(abs(t2 - t*t));	                      			            						\
	pow = nvmlAPIEnd();                                                             				\
	printf("%20.20s ----- %6.2f ms (stdev=%6.2f ms) ----- %6.2f W ----- ", s, t, t2, pow/1000.0 );  \
	memset(x, 0xFF, n * sizeof(VALUE_TYPE));      	                                 	 \
	cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);              \
	if (validate_x(x, n, s)) { all_passed=0; }

	

#define BENCH_RUN_SOLVE_MKL(f,t,s)                                            		 \
	float t;                                                                     	 \
	CLK_START;			                                                             \
	for (int i = 0; i < BENCH_REPEAT; ++i)                                           \
	{	                                                                     	     \
		memset(x, 0, n * sizeof(VALUE_TYPE));      	                                 \
		f;                                                                           \
	}	                                                                             \
	CLK_STOP;       		                                                         \
	t = CLK_ELAPSED/(double)BENCH_REPEAT;                                 			 \
	printf("%s ------------ %f ms ", s, t );                                         \
	if (validate_x(x, n, s)) { t = -1; }

#define BENCH_RUN_DEPTH(f,t,s)                                                 		 \
	float t;                                                                     	 \
	CLK_START;			                                                             \
	for (int i = 0; i < BENCH_REPEAT; ++i)                                           \
	{	                                                                     	     \
		cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE));                                  \
		f;                                                                           \
	}	                                                                             \
	cudaDeviceSynchronize();                                                         \
	CLK_STOP;       		                                                         \
	t = CLK_ELAPSED/(double)BENCH_REPEAT;                                 			 \
	printf("%s ------------ %f ms\n", s, t );                                             


#define CUSP_CHK(call) print_cusparse_state(call);
#define CUDA_CHK(call) print_cuda_state(call);


#define DEBUG(x) printf(x);printf("\n");fflush(0);



static inline void print_cuda_state(cudaError_t code){

   if (code != cudaSuccess) printf("\ncuda error: %s\n", cudaGetErrorString(code));
   
}

 
static inline void print_cusparse_state(cusparseStatus_t stat){
	if(stat){
		printf("\n");

		switch(stat){
			case CUSPARSE_STATUS_SUCCESS: printf("cusparse stat: SUCCESS"); break;
			case CUSPARSE_STATUS_NOT_INITIALIZED: printf("cusparse stat: NOT_INITIALIZED"); break;
			case CUSPARSE_STATUS_ALLOC_FAILED: printf("cusparse stat: ALLOC_FAILED"); break;
			case CUSPARSE_STATUS_INVALID_VALUE: printf("cusparse stat: INVALID_VALUE"); break;
			case CUSPARSE_STATUS_ARCH_MISMATCH: printf("cusparse stat: ARCH_MISMATCH"); break;
			case CUSPARSE_STATUS_MAPPING_ERROR: printf("cusparse stat: MAPPING_ERROR"); break;
			case CUSPARSE_STATUS_EXECUTION_FAILED: printf("cusparse stat: EXECUTION_FAILED"); break;
			case CUSPARSE_STATUS_INTERNAL_ERROR: printf("cusparse stat: INTERNAL_ERROR"); break;
			case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: printf("cusparse stat: MATRIX_TYPE_NOT_SUPPORTED"); break;
		}
		
		printf("\n");
	}
}


double eval_time();



#endif
