#include "test.h"
#include "matrix_properties.h"
#include "nvmlPower.hpp"
#include <unistd.h>


/*
void test_two_streams(const char * filename, int * csrRowPtrL, int * csrColIdxL, VALUE_TYPE * csrValL, int n){

    dfr_analysis_info_t * info = (dfr_analysis_info_t*) malloc(sizeof(dfr_analysis_info_t));
    sp_mat_t * gpu_L = (sp_mat_t*) malloc(sizeof(sp_mat_t));

    int nnzL = csrRowPtrL[n] - csrRowPtrL[0];

    cudaMalloc((void **)&gpu_L->ia , (n+1) * sizeof(int));
    cudaMalloc((void **)&gpu_L->ja , nnzL  * sizeof(int));
    cudaMalloc((void **)&gpu_L->a  , nnzL  * sizeof(VALUE_TYPE));

    cudaMemcpy(gpu_L->ia, csrRowPtrL, (n+1) * sizeof(int)        ,   cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_L->ja, csrColIdxL, nnzL  * sizeof(int)        ,   cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_L->a , csrValL,    nnzL  * sizeof(VALUE_TYPE) ,   cudaMemcpyHostToDevice);

    gpu_L->nr = n;
    gpu_L->nc = n;
    gpu_L->nnz = nnzL;

    cusparseHandle_t cusp_handle;
    cusparseMatDescr_t desc_L;
    cusparseSolveAnalysisInfo_t info_L;

    //cusparseCreate(&cusp_handle);

    printf("analisis dfr :: start \n"); fflush(0);
    multirow_analysis_base( &info, gpu_L );
    printf("analisis dfr :: done \n" ); fflush(0);

    VALUE_TYPE *b = (VALUE_TYPE *) malloc( sizeof(VALUE_TYPE) * n );
    VALUE_TYPE *x = (VALUE_TYPE *) malloc( sizeof(VALUE_TYPE) * n );

    VALUE_TYPE *d_b1;
    VALUE_TYPE *d_x1;

    VALUE_TYPE *d_b2;
    VALUE_TYPE *d_x2;

    int *is_solved;
    int *is_solved_ptr;

    for (int i = 0; i < n; i++)
    {
        b[i] = 0;
        for (int j = csrRowPtrL[i]; j < csrRowPtrL[i+1]; j++) b[i] += csrValL[j];
    }

    int *depths = (int *) malloc( sizeof(int) * n );

    cudaStream_t gpu_stream[2];

    CUDA_CHK( cudaStreamCreate(&gpu_stream[0]) );
    CUDA_CHK( cudaStreamCreate(&gpu_stream[1]) );

    cudaMalloc((void **)&d_b1, n * sizeof(VALUE_TYPE));
    cudaMalloc((void **)&d_x1, n * sizeof(VALUE_TYPE));
    cudaMalloc((void **)&d_b2, n * sizeof(VALUE_TYPE));
    cudaMalloc((void **)&d_x2, n * sizeof(VALUE_TYPE));

    cudaMemcpyAsync(d_b1, b, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice, gpu_stream[0]);
    csr_L_solve_multirow ( gpu_L, info, d_b1, d_x1, n, gpu_stream[0] );

    // cudaMemcpyAsync(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost, gpu_stream[0]);

    cudaMemcpyAsync(d_b2, b, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice, gpu_stream[1]);
    csr_L_solve_multirow ( gpu_L, info, d_b2, d_x2, n, gpu_stream[1] );


    cudaDeviceSynchronize();
    csr_L_solve_multirow ( gpu_L, info, d_b2, d_x2, n, gpu_stream[1] );
    csr_L_solve_multirow ( gpu_L, info, d_b2, d_x2, n, gpu_stream[1] );
    cudaDeviceSynchronize();
}
*/
void test_solve_L_analysis_multirow(const char* filename, int* csrRowPtrL, int* csrColIdxL, VALUE_TYPE* csrValL, int n) {

    CLK_INIT;

    // CLK_START;
    // usleep(1231435);
    // CLK_STOP;

    // double t_usleep = CLK_ELAPSED;
    // printf("1231435 usec = %f ms\n", t_usleep );



/*
	printf("starting\n");
	//Niveles CPU
	int* niveles = (int*) malloc(n*sizeof(int));
	memset(niveles, 0, n*sizeof(int));
	for(int i=0;i<n;++i){
		//printf("Entering row %d of %d\n",i,n);
		for(int j = csrRowPtrL[i]; j<csrRowPtrL[i+1]-1;j++){
			
			int col = csrColIdxL[j];
			//printf("Row: %d. Elem: %d. Col: %d. \n",i,j,col);


			
			if(niveles[col]+1>niveles[i])
				niveles[i] = niveles[col]+1;
		}
	}
	printf("out for\n");	
	int max=niveles[0];
	for(int i=1; i<n;i++){
		if(max<niveles[i]){
			max=niveles[i];
//			printf("max:%i\n",max);
		}
	}
	printf("El máximo encontrado es %i.\n",max);

*/


    dfr_analysis_info_t* info = (dfr_analysis_info_t*)malloc(sizeof(dfr_analysis_info_t));
    dfr_analysis_info_t* info2 = (dfr_analysis_info_t*)malloc(sizeof(dfr_analysis_info_t));



    sp_mat_t* gpu_L = (sp_mat_t*)malloc(sizeof(sp_mat_t));

    int nnzL = csrRowPtrL[n] - csrRowPtrL[0];

    CLK_START;
    cudaMalloc((void**)&gpu_L->ia, (n + 1) * sizeof(int));
    cudaMalloc((void**)&gpu_L->ja, nnzL * sizeof(int));
    cudaMalloc((void**)&gpu_L->a, nnzL * sizeof(VALUE_TYPE));
    CLK_STOP;
    float t_malloc_L = CLK_ELAPSED;

    CLK_START;
    cudaMemcpy(gpu_L->ia, csrRowPtrL, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_L->ja, csrColIdxL, nnzL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_L->a, csrValL, nnzL * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    CLK_STOP;
    float t_memcpy_L = CLK_ELAPSED;
    printf("Copia datos a GPU:: runtime = %f ms \n", t_memcpy_L); fflush(0);

    gpu_L->nr = n;
    gpu_L->nc = n;
    gpu_L->nnz = nnzL;


    cusparseHandle_t cusp_handle;
    cusparseMatDescr_t desc_L;
    //cusparseSolveAnalysisInfo_t info_L;

    //cusparseCreate(&cusp_handle);

    //CUSP_CHK(cusparseCreateMatDescr(&(desc_L)))
    //CUSP_CHK(  cusparseCreateSolveAnalysisInfo(&(info_L))     )

    //CUSP_CHK(cusparseSetMatIndexBase(desc_L, CUSPARSE_INDEX_BASE_ZERO))
    //CUSP_CHK(cusparseSetMatType(desc_L, CUSPARSE_MATRIX_TYPE_GENERAL))
    // CUSP_CHK(  cusparseSetMatFillMode(desc_L, CUSPARSE_FILL_MODE_LOWER)      )
    // CUSP_CHK(  cusparseSetMatDiagType(desc_L, CUSPARSE_DIAG_TYPE_NON_UNIT)   )


    /*
    CLK_START;
    #ifdef __float__
        CUSP_CHK(  cusparseScsrsv_analysis(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            n, nnzL , desc_L, gpu_L->a, gpu_L->ia, gpu_L->ja, info_L)      )
    #else
    //    CUSP_CHK(  cusparseDcsrsv_analysis(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                       n, nnzL , desc_L, gpu_L->a, gpu_L->ia, gpu_L->ja, info_L)      )
    #endif
    cudaDeviceSynchronize();
    CLK_STOP;
    */

    int cusparse_levs;
    int* levelPtr, * levelInd;

    //CUSP_CHK(  cusparseGetLevelInfo(cusp_handle,
    //                 				info_L,
    //                 				&cusparse_levs,
    //                 				&levelPtr,
    //                 				&levelInd)   )

    //float t_anal_cusparse = CLK_ELAPSED;
    //printf("analisis cusparse :: niveles = %d, runtime = %f ms \n", cusparse_levs, t_anal_cusparse ); fflush(0);



    int csrv2_buffer_size = 0;
    void* csrv2_buffer;
    void* csrv2_buffer_nolev;
    csrsv2Info_t info_L_v2;
    csrsv2Info_t info_L_v2_nolev;

    /*
    CUSP_CHK(cusparseCreateCsrsv2Info(&info_L_v2));
    CUSP_CHK(cusparseCreateCsrsv2Info(&info_L_v2_nolev));

    CLK_START;
    #ifdef __float__
        CUSP_CHK(cusparseScsrsv2_bufferSize(cusp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            n,
            nnzL,
            desc_L,
            gpu_L->a,
            gpu_L->ia,
            gpu_L->ja,
            info_L_v2,
            &csrv2_buffer_size))
    #else
        CUSP_CHK(cusparseDcsrsv2_bufferSize(cusp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            n,
            nnzL,
            desc_L,
            gpu_L->a,
            gpu_L->ia,
            gpu_L->ja,
            info_L_v2,
            &csrv2_buffer_size))
    #endif

            CUDA_CHK(cudaMalloc(&csrv2_buffer, csrv2_buffer_size))

    #ifdef __float__
            CUSP_CHK(cusparseScsrsv2_analysis(cusp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                n,
                nnzL,
                desc_L,
                gpu_L->a,
                gpu_L->ia,
                gpu_L->ja,
                info_L_v2,
                CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                csrv2_buffer))
    #else
            CUSP_CHK(cusparseDcsrsv2_analysis(cusp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                n,
                nnzL,
                desc_L,
                gpu_L->a,
                gpu_L->ia,
                gpu_L->ja,
                info_L_v2,
                CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                csrv2_buffer))
    #endif
        CLK_STOP;

        float t_anal_cusparse_v2 = CLK_ELAPSED;
        printf("analisis cusparse v2 :: niveles = %d, runtime = %f ms \n", cusparse_levs, t_anal_cusparse_v2); fflush(0);



        CLK_START;
    #ifdef __float__
        CUSP_CHK(cusparseScsrsv2_bufferSize(cusp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            n,
            nnzL,
            desc_L,
            gpu_L->a,
            gpu_L->ia,
            gpu_L->ja,
            info_L_v2_nolev,
            &csrv2_buffer_size))
    #else
        CUSP_CHK(cusparseDcsrsv2_bufferSize(cusp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            n,
            nnzL,
            desc_L,
            gpu_L->a,
            gpu_L->ia,
            gpu_L->ja,
            info_L_v2_nolev,
            &csrv2_buffer_size))
    #endif
            CUDA_CHK(cudaMalloc(&csrv2_buffer, csrv2_buffer_size))
    #ifdef __float__
            CUSP_CHK(cusparseScsrsv2_analysis(cusp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                n,
                nnzL,
                desc_L,
                gpu_L->a,
                gpu_L->ia,
                gpu_L->ja,
                info_L_v2_nolev,
                CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                csrv2_buffer))
    #else
            CUSP_CHK(cusparseDcsrsv2_analysis(cusp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                n,
                nnzL,
                desc_L,
                gpu_L->a,
                gpu_L->ia,
                gpu_L->ja,
                info_L_v2_nolev,
                CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                csrv2_buffer))
    #endif
    CLK_STOP;

    float t_anal_cusparse_v2_nolev = CLK_ELAPSED;
    printf("analisis cusparse v2 no_lev :: runtime = %f ms \n", cusparse_levs, t_anal_cusparse_v2_nolev); fflush(0);
    */
    // Printea Datos de la matriz 
   
    float t_anal_dfr = CLK_ELAPSED;
    //for(int p = 1; p <= BENCH_REPEAT ;p++){
   /* 	if(PRINT_TIME_ANALYSIS  && LOG_FILE != "NONE"){

        	FILE* fp;
        	fp = fopen(LOG_FILE, "a+");
        	fprintf(fp,",%d", n);
        	fprintf(fp,",%d", info->nlevs);
        	fprintf(fp,",%d", nnzL);
        	fclose(fp);
    	}
    */
unsigned int *anali1, *anali2;
CUDA_CHK(cudaMalloc((void**)&(anali1), n * sizeof(int)))
CUDA_CHK(cudaMalloc((void**)&(anali2), n * sizeof(int)))
//CUDA_CHK(cudaMemsetAsync(anali1, 0, n * sizeof(int)));
//CUDA_CHK(cudaMemsetAsync(anali2, 0, n * sizeof(int)));

sp_mat_ana_t* ana_mat = (sp_mat_ana_t*) malloc(sizeof(sp_mat_ana_t));
sp_mat_ana_t* ana_mat2 = (sp_mat_ana_t*) malloc(sizeof(sp_mat_ana_t));


        if (!OLD_ANALYSIS) {
            	CLK_START;

	        printf("New gpu\n");
		//multirow_analysis_no_lvl(&info2, gpu_L);
            	multirow_analysis_base_GPU(&info2, gpu_L,5,ana_mat2,anali1);
                //multirow_analysis_base(&info2, gpu_L);
            	CLK_STOP;
        } else {
            printf("Using old analysis\n");
            CLK_START;
            multirow_analysis_base_parallel(&info2, gpu_L);
            CLK_STOP;
        }


        t_anal_dfr = CLK_ELAPSED;
        
    //}
    //t_anal_dfr = t_anal_dfr/BENCH_REPEAT;
	float t2_analy;
	CLK_START;
        multirow_analysis_base_GPU(&info, gpu_L,0,ana_mat, anali2);
	CLK_STOP;
	t2_analy = CLK_ELAPSED;
    printf("analisis dfr :: niveles = %d, runtime = %f ms, rutime2 = %f ms \n", info->nlevs, t2_analy,t_anal_dfr); fflush(0);


	
    if(TIMERS_SOLVERS && LOG_FILE != "NONE"){                                                                                                 
        FILE* fp = fopen(LOG_FILE, "a+");                                                                                                               
        fprintf(fp,",%.2f,%.2f",t2_analy, t_anal_dfr );                                                                                                                               
        fclose(fp);                                                                                                                                                                                                                     
    }



    //Initialize memory of structure matrix
    ana_mat->first = true;
    CUDA_CHK(cudaMalloc((void**) &(ana_mat->values), sizeof(VALUE_TYPE)*(info->n_warps*WARP_SIZE ) )); //values
	
    CUDA_CHK(cudaMalloc((void**) &(ana_mat->diag), sizeof(VALUE_TYPE)*n  )); //diag
  

    CUDA_CHK(cudaMalloc((void**) &(ana_mat->row_idx), sizeof(int)*n )); //row_idx
    CUDA_CHK(cudaMalloc((void**) &(ana_mat->cols), sizeof(int)*(info->n_warps*WARP_SIZE) )); //col








/*
printf("Trying\n");
	int n_warps_h1 = info->n_warps, n_warps_h2 = info2->n_warps;
 

	unsigned int* dfr1,*dfr2;
	dfr1= (unsigned int*) malloc(n*sizeof(unsigned int));
	dfr2= (unsigned int*) malloc(n*sizeof(unsigned int));
        CUDA_CHK(cudaMemcpy(dfr1, anali1, n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(dfr2, anali2, n * sizeof(int), cudaMemcpyDeviceToHost));

	
	int max=0, nmax=0,nmin=0;
	double avg=0;
	printf("Levels\n");
	for(int i=0;i<n; i++){	
		int temp = dfr1[i]- dfr2[i];
		//printf("Temp: %i\n", temp);
		if (temp == max) nmax++;
		else if(temp>max) {
			max = temp;
			nmax=1;
		}
		if(temp==0) nmin++;
		avg+=temp;

		
	}

	avg = avg/n;
	printf("Max = %i. Nmax = %i. Avg = %f. Nmin = %i\n",max,nmax,avg,nmin);
    	if(TIMERS_SOLVERS && LOG_FILE != "NONE"){
        	FILE* fp = fopen(LOG_FILE, "a+");
        	fprintf(fp,",%i,%i,%.2f,%i",max,nmax,avg,nmin );
        	fclose(fp);                                                                                                                                                                                                 
    	}*/
/*
        if(n_warps_h1!=n_warps_h2){printf("Nwarps: %i, Value %i. Wrong\n",n_warps_h1, n_warps_h2); return;}
	

	int* ibase_row_h1, *ibase_row_h2, *ivect_size_warp_h1, *ivect_size_warp_h2,*iorder_h1,*iorder_h2;
	ibase_row_h1 = (int*) malloc((n_warps_h1+1)*sizeof(int));
        ibase_row_h2 = (int*) malloc((n_warps_h2+1)*sizeof(int));        
	CUDA_CHK(cudaMemcpy(ibase_row_h1,info->ibase_row, (n_warps_h1 + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(ibase_row_h2,info2->ibase_row, (n_warps_h1 + 1) * sizeof(int), cudaMemcpyDeviceToHost));	
	ivect_size_warp_h1 = (int*) malloc((n_warps_h1)*sizeof(int));
        ivect_size_warp_h2 = (int*) malloc((n_warps_h1)*sizeof(int));
        CUDA_CHK(cudaMemcpy(ivect_size_warp_h1,info->ivect_size_warp, (n_warps_h1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(ivect_size_warp_h2,info2->ivect_size_warp, (n_warps_h1) * sizeof(int), cudaMemcpyDeviceToHost));
	iorder_h1 = (int*) malloc((n)*sizeof(int));
        iorder_h2 = (int*) malloc((n)*sizeof(int));
	CUDA_CHK(cudaMemcpy(iorder_h1,info->iorder,n*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(iorder_h2,info2->iorder,n*sizeof(int), cudaMemcpyDeviceToHost));

	
	int errores=0;
	for(int i=0; i<n_warps_h1&&errores<10;i++){
		if(ibase_row_h1[i] != ibase_row_h2[i]){ printf("Dif base %i. Calculated: %i (level %u). Real %i (level %u). Sz: %i\n",i, ibase_row_h1[i], dfr1[iorder_h1[ibase_row_h1[i]]], ibase_row_h2[i], dfr2[iorder_h1[ibase_row_h2[i]]],ivect_size_warp_h1[i]);errores++;}
                if(ivect_size_warp_h1[i] != ivect_size_warp_h2[i]){ printf("Dif size in %i. %i vs %i\n",i, ivect_size_warp_h1[i], ivect_size_warp_h2[i]);errores++;} //else printf("Row %i correct. %i %i\n",i, ivect_size_warp_h1[i], ivect_size_warp_h2[i]);
		//if(iorder_h1[i] != iorder_h2[i]) {printf("Dif iorder in %i. Calculated: %i. Real %i\n", i,iorder_h1[i], iorder_h2[i]); errores++;} 
	}
	int n_warps=n_warps_h1;
        if(ibase_row_h1[n_warps] != ibase_row_h2[n_warps]){ printf("Dif last base %i. Calculated: %i. Real %i.\n",n_warps, ibase_row_h1[n_warps], ibase_row_h2[n_warps]);}
errores=0;
        for(int i=0; i<n&&errores<10;i++){
        	if(iorder_h1[i] != iorder_h2[i]) {printf("Dif iorder in %i. Calculated: %i (level %u). Real %i (level %u)\n", i,iorder_h1[i], dfr1[iorder_h1[i]], iorder_h2[i], dfr2[iorder_h2[i]]); errores++;}
	}
	return;
  
*/
    
    /*if(TIMERS_SOLVERS   && LOG_FILE != "NONE"){

        FILE* fp;
        fp = fopen("dimensiones_todas.csv", "a+");
        fprintf(fp,",%d", n);
        fprintf(fp,",%d", info->nlevs);
        fprintf(fp,",%d,$", nnzL);
        fclose(fp);
    }*/
    
    
    // Print tiempo Analisis Multirow
   /* if(TIMERS_SOLVERS && LOG_FILE != "NONE"){													
        FILE* fp = fopen(LOG_FILE, "a+");														
        fprintf(fp,",%.2f", t_anal_dfr );																
        fclose(fp);																											
    }

    // Print tiempo Analisis Order (Por ahora mismo que multirow)
    if(TIMERS_SOLVERS && LOG_FILE != "NONE"){													
            CLK_START;
            multirow_analysis_base_GPU(&info2, gpu_L,1);
            CLK_STOP;

	FILE* fp = fopen(LOG_FILE, "a+");														
        fprintf(fp,",%.2f", CLK_ELAPSED );																
        fclose(fp);																										

    }*/


    VALUE_TYPE* b = (VALUE_TYPE*)malloc(sizeof(VALUE_TYPE) * n);
    VALUE_TYPE* x = (VALUE_TYPE*)malloc(sizeof(VALUE_TYPE) * n);

    VALUE_TYPE* d_b;
    VALUE_TYPE* d_x;

    int* is_solved;
    int* is_solved_ptr;

    for (int i = 0; i < n; i++)
    {
        b[i] = 0;
        for (int j = csrRowPtrL[i]; j < csrRowPtrL[i + 1]; j++) b[i] += csrValL[j];
    }


#ifdef _MKL_
    //prueba mkl
    sparse_matrix_t mklL, mklU;

    printf("max mkl threads = %d ms \n", mkl_get_max_threads()); fflush(0);

    struct matrix_descr desc_mklL;
    desc_mklL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    desc_mklL.mode = SPARSE_FILL_MODE_LOWER;
    desc_mklL.diag = SPARSE_DIAG_NON_UNIT;

#ifdef __float__
    sparse_status_t stat = mkl_sparse_s_create_csr(&mklL, SPARSE_INDEX_BASE_ZERO, n, n, csrRowPtrL, csrRowPtrL + 1, csrColIdxL, csrValL);
#else
    sparse_status_t stat = mkl_sparse_d_create_csr(&mklL, SPARSE_INDEX_BASE_ZERO, n, n, csrRowPtrL, csrRowPtrL + 1, csrColIdxL, csrValL);
#endif

    CLK_START;
    stat = mkl_sparse_set_sv_hint(mklL, SPARSE_OPERATION_TRANSPOSE, desc_mklL, 1);
    CLK_STOP;
    float t_anal_mkl = CLK_ELAPSED;

    printf("analisis mkl :: runtime = %f ms \n", t_anal_mkl); fflush(0);

    if (SPARSE_STATUS_SUCCESS != stat) {
        fprintf(stderr, "Failed to set sv hint\n");
    }

    stat = mkl_sparse_optimize(mklL);

    if (SPARSE_STATUS_SUCCESS != stat) {
        fprintf(stderr, "Failed to sparse optimize\n");
    }

#endif

    int* depths = (int*)malloc(sizeof(int) * n);

    CLK_START;
    cudaMalloc((void**)&d_b, n * sizeof(VALUE_TYPE));
    cudaMalloc((void**)&d_x, n * sizeof(VALUE_TYPE));
    CLK_STOP;
    float t_malloc_bx = CLK_ELAPSED;

    CLK_START;
    cudaMalloc((void**)&is_solved, n * sizeof(int));
    cudaMalloc((void**)&is_solved_ptr, sizeof(int));
    CLK_STOP;
    float t_malloc_ready = CLK_ELAPSED;

    CLK_START;
    cudaMemcpy(d_b, b, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
    CLK_STOP;
    float t_memcpy_bx = CLK_ELAPSED;

    char uplo = 'L', transa = 'N', diag = 'N';

    //     // float t_depth;
    //     BENCH_RUN_DEPTH( csr_L_get_depth( gpu_L, depths ) , t_depth, "solve depths" );

    // #ifdef _MKL_
    //     VALUE_TYPE alpha = 1.0;

    //     mkl_set_num_threads(4);

    //     #ifdef __float__
    //     BENCH_RUN_SOLVE_MKL( mkl_sparse_s_trsv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, mklL, desc_mklL, b, x), t_mkl_4th, "solve mkl" );
    //     #else
    //     BENCH_RUN_SOLVE_MKL( mkl_sparse_d_trsv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, mklL, desc_mklL, b, x), t_mkl_4th, "solve mkl" );
    //     #endif

    //     mkl_set_num_threads(2);

    //     #ifdef __float__
    //     BENCH_RUN_SOLVE_MKL( mkl_sparse_s_trsv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, mklL, desc_mklL, b, x), t_mkl_2th, "solve mkl" );
    //     #else
    //     BENCH_RUN_SOLVE_MKL( mkl_sparse_d_trsv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, mklL, desc_mklL, b, x), t_mkl_2th, "solve mkl" );
    //     #endif

    //     mkl_set_num_threads(1);

    //     #ifdef __float__
    //     BENCH_RUN_SOLVE_MKL( mkl_sparse_s_trsv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, mklL, desc_mklL, b, x), t_mkl_1th, "solve mkl" );
    //     #else
    //     BENCH_RUN_SOLVE_MKL( mkl_sparse_d_trsv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, mklL, desc_mklL, b, x), t_mkl_1th, "solve mkl" );
    //     #endif
    // #endif

    int all_passed = 1;

    //DEPRECATED!! BENCH_RUN_SOLVE( csr_L_solve_cusparse ( gpu_L, d_b, d_x, n, cusp_handle, desc_L, info_L ) , t_cusp, t_cusp_stdev, p_cusp, "solve cusparse" );

    //BENCH_RUN_SOLVE(csr_L_solve_cusparse_v2(gpu_L, d_b, d_x, n, nnzL, cusp_handle, desc_L, info_L_v2, CUSPARSE_SOLVE_POLICY_USE_LEVEL, csrv2_buffer), t_cusp_v2, t_cusp_v2_stdev, p_cusp_v2, "solve cusp_v2 lev");

 	//BENCH_RUN_SOLVE( csr_L_solve_cusparse_v2 ( gpu_L, d_b, d_x, n, nnzL, cusp_handle, desc_L, info_L_v2_nolev, CUSPARSE_SOLVE_POLICY_NO_LEVEL, csrv2_buffer ) , t_cusp_v2_nolev, t_cusp_v2_stdev_nolev, p_cusp_v2_nolev, "solve cusp_v2 no_lev" );

    //BENCH_RUN_SOLVE(csr_L_solve_simple(gpu_L, d_b, d_x, n, is_solved), t_simple, t_simple_stdev, p_simple, "solve base");


    // BENCH_RUN_SOLVE( csr_L_solve_simple_v2 ( gpu_L, d_b, d_x, n, is_solved, is_solved_ptr ) , t_simple2, t_simple_stdev2, p_simple2, "solve base2" );

    //BENCH_RUN_SOLVE( csr_L_solve_nan ( gpu_L, d_b, d_x, n ), t_nan, t_nan_stdev, p_nan, "solve NaN" );

    // BENCH_RUN_SOLVE( csr_L_solve_nan_hash  ( filename, gpu_L, info, d_b, d_x, n ), t_nan_hash, t_nan_stdev_hash, p_nan_hash, "solve nan_hash" );

    //BENCH_RUN_SOLVE( csr_L_solve_order (  gpu_L, info, d_b, d_x, n, 0 ) , t_order, t_order_stdev, p_order, "solve order" );


    BENCH_RUN_SOLVE(csr_L_solve_multirow(gpu_L, info, d_b, d_x, n, 0), t_multi, t_multi_stdev, p_multi, "solve multirow");

  //  BENCH_RUN_SOLVE(csr_L_solve_multirow(gpu_L, info2, d_b, d_x, n, 0), t_multi2, t_multi_stdev2, p_multi2, "solve multirow 2");

    //BENCH_RUN_SOLVE(csr_L_solve_multirow_format(gpu_L, info, d_b, d_x, n, ana_mat, 0), t_format, t_format_stdev, p_format, "solve format mr");

    BENCH_RUN_SOLVE(csr_L_solve_multirow_format(gpu_L, info2, d_b, d_x, n, ana_mat2, 0), t_format2, t_format_stdev2, p_format2, "solve format2 mr");


    //BENCH_RUN_SOLVE(csr_L_solve_no_lvl(gpu_L, info2, d_b, d_x, n, 0), t_no_lvl, t_no_lvl_stdev, p_no_lvl, "solve no level");


    // BENCH_RUN_SOLVE( csr_L_solve_multirow_hash1 ( gpu_L, info, d_b, d_x, n ), t_multi_hash, t_multi_stdev_hash, p_multi_hash, "solve multirow_hash" )



    if(TIMERS_SOLVERS && LOG_FILE != "NONE"){
            if(!all_passed){
                FILE* fp = fopen("fallos.txt", "a+");
        	    fprintf(fp,"%s\n",filename);
        	    fclose(fp);  
            }                                                                                                                                                                                               
    	}
   /* 

    sp_mat_ana_t* ana_mat2 = (sp_mat_ana_t*) malloc(sizeof(sp_mat_ana_t));
    ana_mat2->values = (VALUE_TYPE*) malloc(sizeof(VALUE_TYPE)*(info->n_warps*WARP_SIZE));
    ana_mat2->diag = (VALUE_TYPE*) malloc(sizeof(VALUE_TYPE)*(n ) );
    ana_mat2->row_idx = (int*) malloc(sizeof(int)*n);
    ana_mat2->cols = (int*) malloc(sizeof(int)*(info->n_warps*WARP_SIZE));
  


    cudaMemcpy(ana_mat2->values, ana_mat->values,sizeof(VALUE_TYPE)*info->n_warps*WARP_SIZE,cudaMemcpyDeviceToHost);  
    cudaMemcpy(ana_mat2->diag, ana_mat->diag,sizeof(VALUE_TYPE)*n,cudaMemcpyDeviceToHost);
    cudaMemcpy(ana_mat2->row_idx, ana_mat->row_idx,sizeof(int)*n,cudaMemcpyDeviceToHost);
    cudaMemcpy(ana_mat2->cols, ana_mat->cols,sizeof(int)*(info->n_warps*WARP_SIZE),cudaMemcpyDeviceToHost);

    int* vect_size =(int*) malloc(info->n_warps*sizeof(int));
    cudaMemcpy(vect_size,info->ivect_size_warp,info->n_warps*sizeof(int),cudaMemcpyDeviceToHost);


    printf("\n\n\nSz   Values                                                                                                                                                                Cols\n");

    for(int i=0;i<info->n_warps;i++){
	printf("%i   ",vect_size[i]);
	for(int j=0;j<32;j++) printf(" %.1f",ana_mat2->values[i*32+j]);
	printf("          ");
        for(int j=0;j<32;j++) printf(" %i",ana_mat2->cols[i*32+j]);
	printf("\n");

    } 
    
    printf("\n\n\n D            r\n");
    
    for(int i=0;i<n;i++){
	printf(" %.1f       %i\n",ana_mat2->diag[i],ana_mat2->row_idx[i]);

    }
 


    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
    for(int i=0;i<n;i++){
	printf("X[%i] = %f\n",i,x[i]);
    }
 */






        


    //     if (all_passed){
    //         int * inv_iorder = (int*) malloc(  n * sizeof(int)  );
    //         CUDA_CHK( cudaMemcpy( inv_iorder, info->inv_iorder, n * sizeof(int), cudaMemcpyDeviceToHost ) );

    //         FILE *ftabla;       
    //         ftabla = fopen("resultados.txt","a+");

    //         fprintf(ftabla,"%s & %s & %d & %d & %d & %d \
        // & %4.2f & %4.2f & %4.2f & %d \
        // & %4.2f% & %4.2f% \
        // & %4.2f & %4.2f & %4.2f \
        // & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \
        // & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \
        // & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \
        // \n",
        //                                     filename,
        //                                     #ifdef __float__
        //                                         "float "
        //                                     #else
        //                                         "double"
        //                                     #endif 
        //                                     ,WARP_PER_BLOCK,
        //                                     n, 
        //                                     nnzL, 
        //                                     get_nnz_max( csrRowPtrL,  csrColIdxL, n ), 
        //                                     (float) nnzL / (float) n, 
        //                                     get_nnz_stdev( csrRowPtrL,  csrColIdxL, n ),
        //                                     get_avg_bw( csrRowPtrL,  csrColIdxL, n ), 
        //                                     info->nlevs,
        //                                     get_locality_simple( csrRowPtrL,  csrColIdxL, inv_iorder, WARP_PER_BLOCK, n ) * 100,
        //                                     get_locality_multirow( csrRowPtrL,  csrColIdxL, inv_iorder, WARP_PER_BLOCK, n ) * 100,
        //                                     // t_anal_mkl,
        //                                     t_anal_cusparse,
        //                                     t_anal_dfr,
        //                                     t_depth,
        //                                     // t_mkl_1th,
        //                                     // t_mkl_2th,
        //                                     // t_mkl_4th,
        //                                     t_cusp, 
        //                                     t_simple, 
        //                                     t_nan,
        //                                     t_order, 
        //                                     t_multi,
        //                                     // t_multi_hash,
        //                                     t_cusp_stdev, 
        //                                     t_simple_stdev, 
        //                                     t_nan_stdev,
        //                                     t_order_stdev, 
        //                                     t_multi_stdev,
        //                                     // t_multi_stdev_hash,
        //                                     // t_malloc_L,
        //                                     // t_malloc_bx,
        //                                     // t_malloc_ready,
        //                                     // t_memcpy_L,
        //                                     // t_memcpy_bx 
        //                                     (t_cusp    / 1000.0) * (p_cusp   /1000.0),
        //                                     (t_simple  / 1000.0) * (p_simple /1000.0),
        //                                     (t_nan / 1000.0) * (p_nan/1000.0),
        //                                     (t_order   / 1000.0) * (p_order  /1000.0),
        //                                     (t_multi   / 1000.0) * (p_multi  /1000.0)  ); //,
        //                                     // p_multi_hash );    
        //         fclose(ftabla);
        //     }

            // print_nnz( csrRowPtrL, n );


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

void test_cusparse(const char* filename, int* csrRowPtrL, int* csrColIdxL, VALUE_TYPE* csrValL, int n) {

    CLK_INIT;

    sp_mat_t* gpu_L = (sp_mat_t*)malloc(sizeof(sp_mat_t));

    int nnzL = csrRowPtrL[n] - csrRowPtrL[0];

    CLK_START;
    cudaMalloc((void**)&gpu_L->ia, (n + 1) * sizeof(int));
    cudaMalloc((void**)&gpu_L->ja, nnzL * sizeof(int));
    cudaMalloc((void**)&gpu_L->a, nnzL * sizeof(VALUE_TYPE));
    CLK_STOP;
    float t_malloc_L = CLK_ELAPSED;

    CLK_START;
    cudaMemcpy(gpu_L->ia, csrRowPtrL, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_L->ja, csrColIdxL, nnzL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_L->a, csrValL, nnzL * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    CLK_STOP;
    float t_memcpy_L = CLK_ELAPSED;

    gpu_L->nr = n;
    gpu_L->nc = n;
    gpu_L->nnz = nnzL;

    cusparseHandle_t cusp_handle;
    cusparseMatDescr_t desc_L;

    int* levelPtr, * levelInd;

    int csrv2_buffer_size = 0;
    void* csrv2_buffer;
    void* csrv2_buffer_nolev;
    csrsv2Info_t info_L_v2;
    csrsv2Info_t info_L_v2_nolev;


    CLK_START;
    cusparseCreate(&cusp_handle);
    CUSP_CHK(cusparseCreateMatDescr(&(desc_L)));
    CUSP_CHK(cusparseSetMatIndexBase(desc_L, CUSPARSE_INDEX_BASE_ZERO));
    CUSP_CHK(cusparseSetMatType(desc_L, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSP_CHK(cusparseCreateCsrsv2Info(&info_L_v2));
    CUSP_CHK(cusparseCreateCsrsv2Info(&info_L_v2_nolev));
    CLK_STOP;

    float t_CusparseSet = CLK_ELAPSED;
    printf("Cusparse Create Data: %f ms \n", t_CusparseSet); fflush(0);

    float t_anal_cusparse_v2 = CLK_ELAPSED;
    for(int p = 1; p <= BENCH_REPEAT ;p++){

    CLK_START;
    #ifdef __float__
    CUSP_CHK(cusparseScsrsv2_bufferSize(cusp_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        n,
        nnzL,
        desc_L,
        gpu_L->a,
        gpu_L->ia,
        gpu_L->ja,
        info_L_v2,
        &csrv2_buffer_size))
    #else
    CUSP_CHK(cusparseDcsrsv2_bufferSize(cusp_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        n,
        nnzL,
        desc_L,
        gpu_L->a,
        gpu_L->ia,
        gpu_L->ja,
        info_L_v2,
        &csrv2_buffer_size))
    #endif

        CUDA_CHK(cudaMalloc(&csrv2_buffer, csrv2_buffer_size))

    #ifdef __float__
        CUSP_CHK(cusparseScsrsv2_analysis(cusp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            n,
            nnzL,
            desc_L,
            gpu_L->a,
            gpu_L->ia,
            gpu_L->ja,
            info_L_v2,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            csrv2_buffer))
    #else
        CUSP_CHK(cusparseDcsrsv2_analysis(cusp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            n,
            nnzL,
            desc_L,
            gpu_L->a,
            gpu_L->ia,
            gpu_L->ja,
            info_L_v2,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            csrv2_buffer))
    #endif
        CLK_STOP;

        t_anal_cusparse_v2 += CLK_ELAPSED;
    }

    int cusparse_levs;
    //printf("Falta conseguir cupsare level info\n");
    t_anal_cusparse_v2 = t_anal_cusparse_v2 / BENCH_REPEAT;
    printf("analisis cusparse v2 :: niveles = %d, runtime = %f ms \n", cusparse_levs, t_anal_cusparse_v2); fflush(0);
    if (TIMERS_SOLVERS && !PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
        FILE* fp;
        fp = fopen(LOG_FILE, "a+");
        fprintf(fp, ",%.2f", t_anal_cusparse_v2);
        fclose(fp);
    }
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


#ifdef _MKL_
    printf("MKL DEFINED");
    //prueba mkl
    sparse_matrix_t mklL, mklU;

    printf("max mkl threads = %d ms \n", mkl_get_max_threads()); fflush(0);

    struct matrix_descr desc_mklL;
    desc_mklL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    desc_mklL.mode = SPARSE_FILL_MODE_LOWER;
    desc_mklL.diag = SPARSE_DIAG_NON_UNIT;

#ifdef __float__
    sparse_status_t stat = mkl_sparse_s_create_csr(&mklL, SPARSE_INDEX_BASE_ZERO, n, n, csrRowPtrL, csrRowPtrL + 1, csrColIdxL, csrValL);
#else
    sparse_status_t stat = mkl_sparse_d_create_csr(&mklL, SPARSE_INDEX_BASE_ZERO, n, n, csrRowPtrL, csrRowPtrL + 1, csrColIdxL, csrValL);
#endif

    CLK_START;
    stat = mkl_sparse_set_sv_hint(mklL, SPARSE_OPERATION_TRANSPOSE, desc_mklL, 1);
    CLK_STOP;
    float t_anal_mkl = CLK_ELAPSED;

    printf("analisis mkl :: runtime = %f ms \n", t_anal_mkl); fflush(0);

    if (SPARSE_STATUS_SUCCESS != stat) {
        fprintf(stderr, "Failed to set sv hint\n");
    }

    stat = mkl_sparse_optimize(mklL);

    if (SPARSE_STATUS_SUCCESS != stat) {
        fprintf(stderr, "Failed to sparse optimize\n");
    }
#endif

    int* depths = (int*)malloc(sizeof(int) * n);

    CLK_START;
    cudaMalloc((void**)&d_b, n * sizeof(VALUE_TYPE));
    cudaMalloc((void**)&d_x, n * sizeof(VALUE_TYPE));
    CLK_STOP;
    float t_malloc_bx = CLK_ELAPSED;

    CLK_START;
    cudaMalloc((void**)&is_solved, n * sizeof(int));
    cudaMalloc((void**)&is_solved_ptr, sizeof(int));
    CLK_STOP;
    float t_malloc_ready = CLK_ELAPSED;

    CLK_START;
    cudaMemcpy(d_b, b, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    CLK_STOP;
    float t_memcpy_bx = CLK_ELAPSED;

    char uplo = 'L', transa = 'N', diag = 'N';

    int all_passed = 1;
    BENCH_RUN_SOLVE(csr_L_solve_cusparse_v2(gpu_L, d_b, d_x, n, nnzL, cusp_handle, desc_L, info_L_v2, CUSPARSE_SOLVE_POLICY_USE_LEVEL, csrv2_buffer), t_cusp_v2, t_cusp_v2_stdev, p_cusp_v2, "solve cusp_v2 lev");

    cudaFree(gpu_L->ia);
    cudaFree(gpu_L->ja);
    cudaFree(gpu_L->a);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(is_solved);

    free(x);
    free(b);
    free(gpu_L);
}
