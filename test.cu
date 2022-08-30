#include <unistd.h>

#include "matrix_properties.h"
#include "nvmlPower.hpp"
#include "test.h"
#include "common.h"
#include "dfr_syncfree.h"
#include "analysis.h"
#include "solver.h"

void test_solve_L_analysis_multirow(const char* filename, int* csrRowPtrL, int* csrColIdxL, VALUE_TYPE* csrValL, int n) {
    dfr_analysis_info_t* info = (dfr_analysis_info_t*)malloc(sizeof(dfr_analysis_info_t));
    dfr_analysis_info_t* info2 = (dfr_analysis_info_t*)malloc(sizeof(dfr_analysis_info_t));

    sp_mat_t* gpu_L = (sp_mat_t*)malloc(sizeof(sp_mat_t));

    int nnzL = csrRowPtrL[n] - csrRowPtrL[0];

    cudaMalloc((void**)&gpu_L->ia, (n + 1) * sizeof(int));
    cudaMalloc((void**)&gpu_L->ja, nnzL * sizeof(int));
    cudaMalloc((void**)&gpu_L->a, nnzL * sizeof(VALUE_TYPE));

    cudaMemcpy(gpu_L->ia, csrRowPtrL, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_L->ja, csrColIdxL, nnzL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_L->a, csrValL, nnzL * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    gpu_L->nr = n;
    gpu_L->nc = n;
    gpu_L->nnz = nnzL;


    int cusparse_levs;
    int *levelPtr, *levelInd;

    multirow_analysis_base_GPU(&info, gpu_L);


    VALUE_TYPE* b = (VALUE_TYPE*)malloc(sizeof(VALUE_TYPE) * n);
    VALUE_TYPE* x = (VALUE_TYPE*)malloc(sizeof(VALUE_TYPE) * n);

    VALUE_TYPE* d_b;
    VALUE_TYPE* d_x;

    int* is_solved;
    int* is_solved_ptr;

    for (int i = 0; i < n; i++) {
        b[i] = 0;
        for (int j = csrRowPtrL[i]; j < csrRowPtrL[i + 1]; j++) b[i] += csrValL[j];
    }

    int* depths = (int*)malloc(sizeof(int) * n);

    cudaMalloc((void**)&d_b, n * sizeof(VALUE_TYPE));
    cudaMalloc((void**)&d_x, n * sizeof(VALUE_TYPE));

    cudaMalloc((void**)&is_solved, n * sizeof(int));
    cudaMalloc((void**)&is_solved_ptr, sizeof(int));

    cudaMemcpy(d_b, b, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    char uplo = 'L', transa = 'N', diag = 'N';

    int all_passed = 1;
    
    CLK_INIT;
    
    BENCH_RUN_SOLVE(csr_L_solve_simple(gpu_L, d_b, d_x, n, is_solved), t_simple, t_simple_stdev, p_simple, "solve base");
    BENCH_RUN_SOLVE(csr_L_solve_order(gpu_L, info, d_b, d_x, n, 0), t_order, t_order_stdev, p_order, "solve order");
    BENCH_RUN_SOLVE(csr_L_solve_multirow(gpu_L, info, d_b, d_x, n, 0), t_multi, t_multi_stdev, p_multi, "solve multirow");
    
    cudaFree(gpu_L->ia);
    cudaFree(gpu_L->ja);
    cudaFree(gpu_L->a);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(is_solved);

    free(x);
    free(b);
    free(info);
    free(gpu_L);
}
