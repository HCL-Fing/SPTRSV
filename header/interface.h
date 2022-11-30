#ifndef __INTERFACE__
#define __INTERFACE__
#include "dfr_syncfree.h"
#include "common.h"

int allocate_memory(int* csrRowPtrL, int* csrColIdxL, VALUE_TYPE* csrValL, int n, sp_mat_t* gpu_L, MODE mode) ;
int run_analysis(dfr_analysis_info_t* info, sp_mat_t* gpu_L);
void run_solver(sp_mat_t* gpu_L, dfr_analysis_info_t* info, VALUE_TYPE* b, VALUE_TYPE* x, int n);
void free_device_analysis(sp_mat_t* gpu_L, dfr_analysis_info_t* info) ;

#endif