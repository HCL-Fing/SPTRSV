#ifndef __DFR_SYNCFREE__
#define __DFR_SYNCFREE__

#define WARP_PER_BLOCK 28
#define WARP_SIZE 32

enum MODE {
    SIMPLE,
    ORDER,
    MULTIROW,
	ALL // TESTING
};

typedef struct {
	int nr; //Number of rows
	int nc; //Number of columns
	int nnz; //Non-Zeros of matrix
	int* ia; //Rows -> row_pointer
	int* ja; //Columns -> col_idx

	VALUE_TYPE* a; //Valor del tipo

} sp_mat_t;

typedef struct {
	MODE mode; //Mode 0 = Multirow Analysis; Anything else is Order Analysis

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

} dfr_analysis_info_t;


#endif
