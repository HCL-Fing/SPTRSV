#include "common.h"
#include "dfr_syncfree.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#define SIMPLE_COOP 0

using namespace cooperative_groups;

template<int tile_size >
__global__ void csr_L_solve_simple_kernel_coop(int* row_ctr,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const VALUE_TYPE* __restrict__ val,
    const VALUE_TYPE* __restrict__ b,
    VALUE_TYPE* x,
    int* is_solved, int n) {
    const int groupsPerWarp = (WARP_SIZE / tile_size);
    
    volatile __shared__ int        s_is_solved[WARP_PER_BLOCK *  groupsPerWarp];
    volatile __shared__ VALUE_TYPE s_x[WARP_PER_BLOCK *  groupsPerWarp];
    __shared__ int s_row;

	thread_block_tile<tile_size>  myTile = tiled_partition<tile_size>(this_thread_block());

    int groupId;                                      // identifica numero del grupo 
    int local_group_id = threadIdx.x / tile_size;   // identifica grupo dentro del bloque
    int lne = myTile.thread_rank();                 // identifica el hilo dentro el grupo


    if (threadIdx.x == 0){
        s_row = atomicAdd(&(row_ctr[0]), 1) * WARP_PER_BLOCK * groupsPerWarp;
    } 

    this_thread_block().sync();
    groupId = s_row + local_group_id;

    if (groupId >= n) return;

    int row = row_ptr[groupId];
    int start_row = s_row; // identificador primer warp del bloque, cuenta cuantos warps antes, identifica la priemra fila calculara por bloque 
    int nxt_row = row_ptr[groupId + 1];
    int lock = 0;

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv = 1 / val[nxt_row - 1];

    if (lne == 0) {
        left_sum = b[groupId];
        s_is_solved[local_group_id] = 0;
    }

    myTile.sync();

    int off = row + lne; // identifica la posicion que le corresponde al thread
    int colidx;
    VALUE_TYPE my_val;
    VALUE_TYPE xx;
    int ready = 0;

    while (off < nxt_row - 1) {
        // Verificar que no se pida varias veces el mismo valor (meter adentro del if)
        my_val = val[off];
        colidx = col_idx[off];

        if (!ready) {
            if (colidx > start_row) { // esto identifica si la fila es procesada por el bloque   
                ready = s_is_solved[colidx - start_row];

                if (ready) {
                    xx = s_x[colidx - start_row];
                }
            } else {
                ready = is_solved[colidx];

                if (ready) {
                    xx = x[colidx];
                }
            }
        }

        if (ready) {
            left_sum -= my_val * xx;

            off += myTile.size();
            ready = 0;
        }
    }
    myTile.sync();
    // Reduccion
    #if __CUDA_ARCH__ >= 800
        VALUE_TYPE res = reduce(myTile, (VALUE_TYPE)left_sum, plus<VALUE_TYPE>());
        left_sum = res;
    #elif __CUDA_ARCH__ < 800 || !defined(__CUDA_ARCH__)   
        printf("Using cuda capability 800-\n");
        for (int i = tile_size/2; i >= 1; i /= 2)
            left_sum += myTile.shfl_down(left_sum, i);;
    #endif

    if (lne == 0) {
        //escribo en el resultado
        s_x[local_group_id] = left_sum * piv;
        s_is_solved[local_group_id] = 1;
        x[groupId] = left_sum * piv;
        __threadfence();
        is_solved[groupId] = 1;
    }

}

template __global__  void csr_L_solve_simple_kernel_coop<2>(int* row_ctr, const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const VALUE_TYPE* __restrict__ val, const VALUE_TYPE* __restrict__ b, VALUE_TYPE* x, int* is_solved, int n);
template __global__  void csr_L_solve_simple_kernel_coop<4>(int* row_ctr, const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const VALUE_TYPE* __restrict__ val, const VALUE_TYPE* __restrict__ b, VALUE_TYPE* x, int* is_solved, int n);
template __global__  void csr_L_solve_simple_kernel_coop<8>(int* row_ctr, const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const VALUE_TYPE* __restrict__ val, const VALUE_TYPE* __restrict__ b, VALUE_TYPE* x, int* is_solved, int n);
template __global__  void csr_L_solve_simple_kernel_coop<16>(int* row_ctr, const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const VALUE_TYPE* __restrict__ val, const VALUE_TYPE* __restrict__ b, VALUE_TYPE* x, int* is_solved, int n);
template __global__  void csr_L_solve_simple_kernel_coop<32>(int* row_ctr, const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const VALUE_TYPE* __restrict__ val, const VALUE_TYPE* __restrict__ b, VALUE_TYPE* x, int* is_solved, int n);

void aux_call_csr_L_solve_simple_kernel_coop(dfr_analysis_info_t* info,sp_mat_t* mat, const VALUE_TYPE* b, VALUE_TYPE* x, int n, int* is_solved,int num_threads, int grid, int average) {
	int* group_id_counter;
	CUDA_CHK(cudaMalloc((void**)&(group_id_counter), sizeof(int)));
	CUDA_CHK(cudaMemsetAsync(group_id_counter, 0, sizeof(int)));
	
	switch (average){
	case 0 ... 2:
        csr_L_solve_simple_kernel_coop<2><<<grid, num_threads >>>(info->row_ctr, mat->ia, mat->ja, mat->a, b, x, is_solved, n);
		break;
	case 3 ... 5:
		csr_L_solve_simple_kernel_coop<4><<<grid, num_threads >>>(info->row_ctr, mat->ia, mat->ja, mat->a, b, x, is_solved, n);
		break;
	case 6 ... 11:
		csr_L_solve_simple_kernel_coop<8><<<grid, num_threads >>>(info->row_ctr, mat->ia, mat->ja, mat->a, b, x, is_solved, n);
		break;
	case 12 ... 23:
		csr_L_solve_simple_kernel_coop<16><<<grid, num_threads >>>(info->row_ctr, mat->ia, mat->ja, mat->a, b, x, is_solved, n);
		break;
	default:
		csr_L_solve_simple_kernel_coop<32><<<grid, num_threads >>>(info->row_ctr, mat->ia, mat->ja, mat->a, b, x, is_solved, n);
		break;
	}
}


void csr_L_solve_simple(sp_mat_t* mat, const VALUE_TYPE* b, VALUE_TYPE* x, int n, int* is_solved) {
    int rows = mat->nr;
    int nnz = mat->nnz;
    int average = nnz/rows;
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int grid = ceil((double)n * WARP_SIZE / (double)num_threads);


    cudaMemset(is_solved, 0, n * sizeof(int));
    dfr_analysis_info_t* info = (dfr_analysis_info_t*)malloc(sizeof(dfr_analysis_info_t));  // Ver si se puede resolver con un puntero a int
    CUDA_CHK(cudaMalloc((void**)&(info->row_ctr), 1 * sizeof(int)));
    cudaStream_t stream = 0;
    CUDA_CHK(cudaMemsetAsync(info->row_ctr, 0, 1 * sizeof(int), stream));

    aux_call_csr_L_solve_simple_kernel_coop(info, mat, b, x, n, is_solved, num_threads, grid, average);

}
