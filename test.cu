#include "interface.h"
#include "test.h"

void test_solve_L_analysis_multirow(const char* filename, int* csrRowPtrL, int* csrColIdxL, VALUE_TYPE* csrValL, int n) {
    dfr_analysis_info_t* info;
    sp_mat_t* gpu_L;
    // CALL ANALYSIS
    allocate_memory(csrRowPtrL, csrColIdxL, csrValL, n, info, gpu_L, ALL);

    // CALL SOLVER
    VALUE_TYPE* b = (VALUE_TYPE*)malloc(sizeof(VALUE_TYPE) * n);
    VALUE_TYPE* x;
    for (int i = 0; i < n; i++) {
        b[i] = 0;
        for (int j = csrRowPtrL[i]; j < csrRowPtrL[i + 1]; j++) b[i] += csrValL[j];
    }

    printf("CALL SOLVER\n");
    run_solver(gpu_L, info, b, x, n);
    printf("FIN CALL SOLVER\n");
    // FREE SOLVER
    printf("FREE SOLVER\n");
    free_device_analysis(gpu_L, info, x);
    printf("FIN FREE SOLVER\n");

    free(b);

    free(info);
    free(gpu_L);
}
