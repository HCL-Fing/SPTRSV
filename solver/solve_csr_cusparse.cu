#include "common.h"
#include "dfr_syncfree.h"

void csr_L_solve_cusparse_v2(sp_mat_t* mat, const VALUE_TYPE* d_b, VALUE_TYPE* d_x, int n, int nnz,
    cusparseHandle_t cusp_handle, cusparseMatDescr_t desc_L, csrsv2Info_t info_L,
    cusparseSolvePolicy_t    policy, void* pBuffer) {


    CLK_INIT
    VALUE_TYPE alpha = 1.0;
    CLK_START
    // cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE));
   
#ifdef __float__
    CUSP_CHK(cusparseScsrsv2_solve(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
        &alpha, desc_L,
        mat->a, mat->ia, mat->ja, info_L,
        d_b, d_x, policy, pBuffer))
#else
    CUSP_CHK(cusparseDcsrsv2_solve(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
        &alpha, desc_L,
        mat->a, mat->ia, mat->ja, info_L,
        d_b, d_x, policy, pBuffer))
#endif
        CLK_STOP;

    float t_anal_cusparse_v2 = CLK_ELAPSED;
    printf("cusparseDcsrsv2_solve :: runtime = %f ms \n", t_anal_cusparse_v2); fflush(0);

   
}
