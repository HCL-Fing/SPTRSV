#ifndef __SOLVER__
#define __SOLVER__

//Solvers
void csr_L_solve_simple(sp_mat_t* mat, const VALUE_TYPE* b, VALUE_TYPE* x, int n, int* is_solved);
void csr_L_solve_order(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n, cudaStream_t stream);
void csr_L_solve_multirow(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n, cudaStream_t stream);
void csr_L_solve_cusparse_v2(sp_mat_t* mat, const VALUE_TYPE* d_b, VALUE_TYPE* d_x, int n, int nnz, cusparseHandle_t cusp_handle, cusparseMatDescr_t desc_L, csrsv2Info_t info_L, cusparseSolvePolicy_t    policy, void* pBuffer);

#endif