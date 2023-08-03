#include <math.h>
//#include "include/dfr_syncfree.h"
#include "include/mmio.h"
#include "include/common.h"
#include "include/test.h"
//#define WARP_SIZE   32
#define WARP_PER_BLOCK 28
//#define CUDA_CHK(call) print_cuda_state(call);
//#define ROWS_PER_THREAD 1


/*static inline void print_cuda_state(cudaError_t code){

   if (code != cudaSuccess) printf("\ncuda error: %s\n", cudaGetErrorString(code));
   
}*/
/*typedef struct {

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


*/




__global__ void kernel_analysis_L_prueba(const int* __restrict__ row_ptr,
	const int* __restrict__ col_idx,
	volatile int* is_solved, int n,
	unsigned int* dfr_analysis_info) {


	extern volatile __shared__ int s_mem[];



	int* s_is_solved = (int*)&s_mem[0];
	int* s_info = (int*)&s_is_solved[28];

	int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
	int local_warp_id = threadIdx.x / WARP_SIZE;

	int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

	if (wrp >= n) return;
//         if(threadIdx.x%32 ==0)printf("My id: %i. My warp: %i\n",threadIdx.x+blockIdx.x*blockDim.x, wrp);

	int row = row_ptr[wrp];
	int start_row = blockIdx.x * 28;
	int nxt_row = row_ptr[wrp + 1];

	int my_level = 0;


	if (lne == 0) {
		s_is_solved[local_warp_id] = 0;
		s_info[local_warp_id] = 0;
	}

	__syncthreads();

	int off = row + lne;
	int colidx = col_idx[off];

	int myvar = 0;

	while (off < nxt_row - 1)
	{
		colidx = col_idx[off];


		if (!myvar)
		{
			if (colidx > start_row) {
				myvar = s_is_solved[colidx - start_row];

				if (myvar) {
					my_level = max(my_level, s_info[colidx - start_row]);
				}
			} else
			{
				myvar = is_solved[colidx];

				if (myvar) {
					my_level = max(my_level, dfr_analysis_info[colidx]);
				}
			}
		}

		if (__all_sync(__activemask(), myvar)) {

			off += WARP_SIZE;
			myvar = 0;
		}
	}
	__syncwarp();
	// Reduccion
	for (int i = 16; i >= 1; i /= 2) {
		my_level = max(my_level, __shfl_down_sync(__activemask(), my_level, i));
	}

	if (lne == 0) {

		//escribo en el resultado
		s_info[local_warp_id] = 1 + my_level;
		s_is_solved[local_warp_id] = 1;


		dfr_analysis_info[wrp] = 1 + my_level;

		__threadfence();

		is_solved[wrp] = 1;
	}
}

	

float clockElapsed(cudaEvent_t evt_start, cudaEvent_t evt_stop) {
    cudaEventSynchronize(evt_stop);

    float elapsedTime = 0;

    cudaEventElapsedTime(&elapsedTime, evt_start, evt_stop);


    return elapsedTime;
}





int main(int argc, char** argv)
{
    dfr_analysis_info_t* info = (dfr_analysis_info_t*)malloc(sizeof(dfr_analysis_info_t));


    int m, n, nnzA;

    //ex: ./Mayores webbase-1M.mtx [num]
    int argi = 1;


    int* csrRowPtrA;
    int* csrColIdxA;
    double* csrValA;


    char* filename;
    int lim;
    if(argc<2){
	printf("Error: no route to matrix\n");
	return 1;
    }
	
    filename = argv[1];

    if(argc >2)
	lim = atoi(argv[2]);
    else 
	lim = 10000;
         

    printf("-------------- %s --------------\n", filename);

    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE* f;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_complex(matcode))
    {
        printf("Sorry, data type 'COMPLEX' is not supported.\n");
        return -3;
    }

    char* pch, * pch1;
    pch = strtok(filename, "/");
    while (pch != NULL) {
        pch1 = pch;
        pch = strtok(NULL, "/");
    }

    pch = strtok(pch1, ".");


    if (mm_is_pattern(matcode)) { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if (mm_is_real(matcode)) { isReal = 1; /*printf("type = real\n");*/ }
    if (mm_is_integer(matcode)) { isInteger = 1; /*printf("type = integer\n");*/ }
    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric = 1;
        printf("input matrix is symmetric = true\n");
    } else
    {
        printf("input matrix is symmetric = false\n");
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    //printf("Matrix is %i x %i. NNZ = %i\n",m,n,nnzA_mtx_report );
    if (ret_code != 0)
        return -4;

    if (n != m) 
    {
	printf("Matrix is not square.\n");
	return -2;
    }	

    if(n<lim)
    {
        printf("Matrix is not large enough.\n");
        return -1;
    }
  

    //Cargar la matriz


   
    int* csrRowPtrA_counter = (int*)malloc((m + 1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

    int* csrRowIdxA_tmp = (int*)malloc(nnzA_mtx_report * sizeof(int));
    int* csrColIdxA_tmp = (int*)malloc(nnzA_mtx_report * sizeof(int));
    double* csrValA_tmp = (VALUE_TYPE*)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;
        int returnvalue;

        if (isReal)
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        } else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }
        idxi--;
        idxj--;

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }



    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int*)malloc((m + 1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m + 1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

    csrColIdxA = (int*)malloc(nnzA * sizeof(int));
    csrValA = (VALUE_TYPE*)malloc(nnzA * sizeof(VALUE_TYPE));

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            } else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    } else
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }



    printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);


    int nnzL = 0;
    int* csrRowPtrL_tmp = (int*)malloc((m + 1) * sizeof(int));
    int* csrColIdxL_tmp = (int*)malloc(nnzA * sizeof(int));
    double* csrValL_tmp = (VALUE_TYPE*)malloc(nnzA * sizeof(VALUE_TYPE));

    int nnz_pointer = 0;
    csrRowPtrL_tmp[0] = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i + 1]; j++)
        {
            if (csrColIdxA[j] < i)
            {
                csrColIdxL_tmp[nnz_pointer] = csrColIdxA[j];
                csrValL_tmp[nnz_pointer] = 1.0; //csrValA[j];
                nnz_pointer++;
            } else
            {
                break;
            }
        }

        csrColIdxL_tmp[nnz_pointer] = i;
        csrValL_tmp[nnz_pointer] = 1.0;
        nnz_pointer++;

        csrRowPtrL_tmp[i + 1] = nnz_pointer;
    }

    nnzL = csrRowPtrL_tmp[m];
    printf("A's unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzL);

    csrColIdxL_tmp = (int*)realloc(csrColIdxL_tmp, sizeof(int) * nnzL);



    







   //stdev
   
    double nnzfila = 0;
    double uAux = 0;
    for (int i = 0; i < m; i++) {
        nnzfila = csrRowPtrL_tmp[i] - csrRowPtrL_tmp[i + 1];
        uAux += nnzfila;
    }
    double u = uAux / m;
    double dif = 0;
    double sigmaSquereAux = 0;
    for (int i = 0; i < m; i++) {
        nnzfila = csrRowPtrA[i] - csrRowPtrA[i + 1];
        dif = pow(nnzfila - u, 2);
        sigmaSquereAux += dif;
    }
    double sigmaSquere = sigmaSquereAux / m;
    double sigma = sqrt(sigmaSquere);
    FILE* efepe;
//    efepe = fopen("dimensiones_todas.csv", "a+");
//    fprintf(efepe, ",%.2f", sigma);
//    fclose(efepe);





    //Levs




    int* is_solved;
    CUDA_CHK(cudaMalloc((void**)&is_solved, m * sizeof(int)))
    CUDA_CHK( cudaMemset(is_solved, 0, m * sizeof(int)) )


    unsigned int* levels;
    CUDA_CHK(cudaMalloc((void**)&levels, m * sizeof(int)))
    CUDA_CHK( cudaMemset(levels, 0, m * sizeof(int)) )
    




    int* cols,*rows;

    CUDA_CHK(cudaMalloc((void**)&rows, (m+1)* sizeof(int)))
    CUDA_CHK(cudaMalloc((void**)&cols, nnzL * sizeof(int)))

    CUDA_CHK(cudaMemcpy(rows,csrRowPtrL_tmp,(m+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(cols,csrColIdxL_tmp,nnzL *sizeof(int), cudaMemcpyHostToDevice));






    int num_threads = 28 * WARP_SIZE;
    int grid = ceil((double)n * WARP_SIZE / (double)(num_threads * ROWS_PER_THREAD));
    
    int shared_size = 28 * sizeof(double) ;


    /*char* pch, * pch1;
    pch = strtok(filename, "/");
    while (pch != NULL) {
        pch1 = pch;
        pch = strtok(NULL, "/");
    }

    pch = strtok(pch1, ".");
*/
    //test_solve_L_analysis_multirow(pch, csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, n);
	





    kernel_analysis_L_prueba << < grid, num_threads, shared_size >> > (rows,
                cols,
                is_solved,
                m,
                levels);

    cudaDeviceSynchronize();



    unsigned int *levels_h = (unsigned int*) malloc(m*sizeof(unsigned int));
    CUDA_CHK(cudaMemcpy(levels_h, levels, m*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    unsigned int max= levels_h[0];
    for(int i=0; i<m;i++){
//	printf("Level[%i] = %i\n", i, levels_h[i]);
	if (max<levels_h[i]) max= levels_h[i];

    }




        FILE* fp;
        fp = fopen("datos_matrices.csv", "a+");
        fprintf(fp,",%d", n);
        fprintf(fp,",%u", max);
        fprintf(fp,",%d", nnzL);
	fprintf(fp,",%.2f\n",sigma);
        fclose(fp);

    printf("Matrices mayores result is Dim = %i, NNZ = %i, Stdev = %.2f, Levs = %u\n",n,nnzL,sigma,max);
  
        return 0;
   
}
