#include <unistd.h>

#include "analysis.h"
#include "interface.h"
#include "matrix_properties.h"
#include "nvmlPower.hpp"
#include "solver.h"

int allocate_memory(int* csrRowPtrL, int* csrColIdxL, VALUE_TYPE* csrValL, int n, sp_mat_t* gpu_L) {
    gpu_L = (sp_mat_t*)malloc(sizeof(sp_mat_t));
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

    return 0;  // Everything OK
}

int run_analysis(sp_mat_t* gpu_L, MODE mode, dfr_analysis_info_t* info) {
    if (mode == SIMPLE) return -1;  // Does not require analysis

    info = (dfr_analysis_info_t*)malloc(sizeof(dfr_analysis_info_t));
    info->mode = mode;

    int cusparse_levs;
    int *levelPtr, *levelInd;
    multirow_analysis_base_GPU(&info, gpu_L, selectedMode);
    return 0;  // Analysis OK
}

void run_solver(sp_mat_t* gpu_L, dfr_analysis_info_t* info, VALUE_TYPE* b, VALUE_TYPE* x, int n) {
    CLK_INIT;
    
    x = (VALUE_TYPE*)malloc(sizeof(VALUE_TYPE) * n);
    
    VALUE_TYPE* d_x;
    VALUE_TYPE* d_b;

    int* is_solved;
    int* is_solved_ptr;

    int* depths = (int*)malloc(sizeof(int) * n);

    cudaMalloc((void**)&d_b, n * sizeof(VALUE_TYPE));
    cudaMalloc((void**)&d_x, n * sizeof(VALUE_TYPE));

    cudaMalloc((void**)&is_solved, n * sizeof(int));
    cudaMalloc((void**)&is_solved_ptr, sizeof(int));

    printf("MEM COPY A\n");
    cudaMemcpy(d_b, b, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    printf("MEM COPY B\n");
    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    char uplo = 'L', transa = 'N', diag = 'N';

    int all_passed = 1;
    printf("SWITCH\n");
    if (info == NULL) {
        printf("RUNING SOVLER SIMPLE\n");
        BENCH_RUN_SOLVE(
            csr_L_solve_simple(gpu_L, d_b, d_x, n, is_solved), t_simple, t_simple_stdev, p_simple, "solve base");
    } else {
        switch (info->mode) {
            case ORDER:
                printf("RUNING SOVLER ORDER\n");
                BENCH_RUN_SOLVE(
                    csr_L_solve_order(gpu_L, info, d_b, d_x, n, 0), t_order, t_order_stdev, p_order, "solve order");

                break;
            case MULTIROW:
                printf("RUNING SOVLER MULTIROW\n");
                BENCH_RUN_SOLVE(
                    csr_L_solve_multirow(gpu_L, info, d_b, d_x, n, 0), t_multi, t_multi_stdev, p_multi, "solve multirow");
                break;
            default:
                printf("RUNING ALL SOLVER\n");

                break;
        }
    }

    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(is_solved);
}

void free_device_analysis(sp_mat_t* gpu_L, dfr_analysis_info_t* info,VALUE_TYPE* x ) {
    cudaFree(gpu_L->ia);
    cudaFree(gpu_L->ja);
    cudaFree(gpu_L->a);
    free(info);
    free(x);
}