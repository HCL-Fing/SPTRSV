#define WARP_PER_BLOCK 28
#define WARP_SIZE 32

typedef struct {
	int nr; //Number of rows
	int nc; //Number of columns
	int nnz; //Non-Zeros of matrix
	int* ia; //Rows -> row_pointer
	int* ja; //Columns -> col_idx

	VALUE_TYPE* a; //Valor del tipo

} sp_mat_t;



typedef struct{
	bool first; 			//True if its the first iteration
	int* cols;				//ELL->col_idx 
	VALUE_TYPE* values;		//Values for main loop		
	VALUE_TYPE* diag;		//Diagonal elements of each row
	int* row_idx;			//original row index of each row
} sp_mat_ana_t;

typedef struct {

	int lev_ctr;
	int nlevs;  //guarda la cantidad maxima de niveles de dependias de todas las posibles fila 

	int* lev_size; //Cuenta filas hay en cada nivel
	int* warp_lev;

	int n_warps;
	int* inv_iorder;
	int* iorder;
	int* ibase_row;
	int* ivect_size_warp;
	int* row_ctr;

//	int n_warps_leq32;
} dfr_analysis_info_t;


//analysis
void csr_L_get_depth(sp_mat_t* gpu_L, int* dfr_analysis_info);
void csr_L_get_depth_test(sp_mat_t* gpu_L, int* dfr_analysis_info);
void multirow_analysis_base(dfr_analysis_info_t** info, sp_mat_t* gpu_L, int mode = 0);
void multirow_analysis_base_parallel(dfr_analysis_info_t** info, sp_mat_t* gpu_L, int mode = 0);
void multirow_analysis_base_GPU(dfr_analysis_info_t** info, sp_mat_t* gpu_L, int mode=0, sp_mat_ana_t * ana_mat = NULL, unsigned int * dfr = NULL);
void multirow_analysis_no_lvl(dfr_analysis_info_t** mat, sp_mat_t* gpu_L, int mode=0);


//void csr_L_solve_cusparse ( sp_mat_t * mat, const VALUE_TYPE * d_b, VALUE_TYPE * d_x, int n, cusparseHandle_t cusp_handle, cusparseMatDescr_t desc_L, cusparseSolveAnalysisInfo_t info_L );
void csr_L_solve_no_lvl(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n, cudaStream_t stream);
void csr_L_solve_cusparse_v2(sp_mat_t* mat, const VALUE_TYPE* d_b, VALUE_TYPE* d_x, int n, int nnz, cusparseHandle_t cusp_handle, cusparseMatDescr_t desc_L, csrsv2Info_t info_L, cusparseSolvePolicy_t    policy, void* pBuffer);
void csr_L_solve_order(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n, cudaStream_t stream);
void csr_L_solve_multirow(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n, cudaStream_t stream);
void csr_L_solve_simple(sp_mat_t* mat, const VALUE_TYPE* b, VALUE_TYPE* x, int n, int* is_solved);
void csr_L_solve_simple_v2(sp_mat_t* mat, const VALUE_TYPE* b, VALUE_TYPE* x, int n, int* is_solved, int* is_solved_ptr);
void csr_L_solve_nan(sp_mat_t* mat, const VALUE_TYPE* b, VALUE_TYPE* x, int n);
void csr_L_solve_nan_hash(const char* filename, sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n);
void csr_L_solve_nan_noshared(sp_mat_t* mat, const VALUE_TYPE* b, VALUE_TYPE* x, int n);
void csr_L_solve_multirow_hash1(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n);
void csr_L_solve_multirow_format(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n, sp_mat_ana_t* mat_ana_info, cudaStream_t stream);
