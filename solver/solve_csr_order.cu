#include "common.h"
#include "dfr_syncfree.h"
#include <cooperative_groups.h>

#define ORDER_COOP 0

using namespace cooperative_groups;

template<int tile_size >
__global__ void csr_L_solve_kernel_order_coop(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const VALUE_TYPE* __restrict__ val, const VALUE_TYPE* __restrict__ b, VALUE_TYPE* x, int* iorder, int* row_ctr, int n) {
    __shared__ int s_row;
    const int groupsPerWarp = (WARP_SIZE / tile_size);
    thread_block_tile<tile_size>  myTile = tiled_partition<tile_size>(this_thread_block());


    int groupId;                                       // identifica numero del grupo 
    int local_group_id = threadIdx.x / tile_size;   // identifica grupo dentro del bloque
    int lne = myTile.thread_rank();                 // identifica el hilo dentro el grupo

    if (threadIdx.x == 0) {
        s_row = atomicAdd(&(row_ctr[0]), 1) * WARP_PER_BLOCK * groupsPerWarp;
    }

    this_thread_block().sync();
    groupId = s_row + local_group_id;

    if (groupId >= n) return;

    int row_idx = iorder[groupId];

    int row = row_ptr[row_idx];
    int start_row = s_row;
    int nxt_row = row_ptr[row_idx + 1];

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv = 1 / val[nxt_row - 1];

    if (lne == 0) {
        left_sum = b[row_idx];
    }

    int off = row + lne;
    int colidx;
    VALUE_TYPE my_val;
    VALUE_TYPE xx;

    int ready = 0;

    myTile.sync();

    if(groupId == 0 ){
        printf("A");
    }
    
    while (off < nxt_row - 1) {
        
        if(groupId == 0 ){
            printf("B");
        }
        colidx = col_idx[off];
        my_val = val[off];
        if (!ready) {
            xx = x[colidx];
            ready = __double2hiint(xx) != (int)0xFFFFFFFF;
        }
        if (ready) {
            left_sum -= my_val * xx;
            off += myTile.size();

            ready = 0;
        }
        
    }
    

    // Reduccion
    for (int i = tile_size / 2; i >= 1; i /= 2)
        // Falta optimizar la mascara
        left_sum += myTile.shfl_down(left_sum, i);

    if (lne == 0) {
        x[row_idx] = left_sum * piv;
    }
}


//template __global__  void csr_L_solve_kernel_order_coop<2>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const VALUE_TYPE* __restrict__ val, const VALUE_TYPE* __restrict__ b, VALUE_TYPE* x, int* iorder, int* row_ctr, int n);
//template __global__  void csr_L_solve_kernel_order_coop<4>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const VALUE_TYPE* __restrict__ val, const VALUE_TYPE* __restrict__ b, VALUE_TYPE* x, int* iorder, int* row_ctr, int n);
template __global__  void csr_L_solve_kernel_order_coop<8>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const VALUE_TYPE* __restrict__ val, const VALUE_TYPE* __restrict__ b, VALUE_TYPE* x, int* iorder, int* row_ctr, int n);
template __global__  void csr_L_solve_kernel_order_coop<16>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const VALUE_TYPE* __restrict__ val, const VALUE_TYPE* __restrict__ b, VALUE_TYPE* x, int* iorder, int* row_ctr, int n);
template __global__  void csr_L_solve_kernel_order_coop<32>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const VALUE_TYPE* __restrict__ val, const VALUE_TYPE* __restrict__ b, VALUE_TYPE* x, int* iorder, int* row_ctr, int n);


__global__ void csr_L_solve_kernel_order(const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const VALUE_TYPE* __restrict__ val,
    const VALUE_TYPE* __restrict__ b,
    VALUE_TYPE* x,
    int* iorder,
    int* row_ctr,
    int n) {

    int lne = threadIdx.x & 0x1f;  // identifica el hilo dentro el warp
    int wrp;

    if (lne == 0) wrp = atomicAdd(&(row_ctr[0]), 1);

    wrp = __shfl_sync(__activemask(), wrp, 0);

    if (wrp >= n) return;

    int row_idx = iorder[wrp];

    int row = row_ptr[row_idx];
    int start_row = blockIdx.x * WARP_PER_BLOCK;
    int nxt_row = row_ptr[row_idx + 1];

    int local_warp_id = wrp - start_row;

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv = 1 / val[nxt_row - 1];

    if (lne == 0) {
        left_sum = b[row_idx];
    }

    int off = row + lne;
    int colidx;
    VALUE_TYPE my_val;
    VALUE_TYPE xx;

    int ready = 0;

    __syncwarp();

    while (off < nxt_row - 1) {
        colidx = col_idx[off];
        my_val = val[off];

        if (!ready) {
            xx = x[colidx];
            ready = __double2hiint(xx) != (int)0xFFFFFFFF;
        }

        if (ready) {
            left_sum -= my_val * xx;
            off += WARP_SIZE;

            ready = 0;
        }
    }
    __syncwarp();

    // Reduccion
    for (int i = 16; i >= 1; i /= 2) {
        left_sum += __shfl_down_sync(__activemask(), left_sum, i);
    }

    if (lne == 0) {
        x[row_idx] = left_sum * piv;
    }
}

void aux_call_csr_L_solve_kernel_order_coop(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n,cudaStream_t stream, int grid, int num_threads, int shared_size, int average) {
	int* group_id_counter;
	CUDA_CHK(cudaMalloc((void**)&(group_id_counter), sizeof(int)));
	CUDA_CHK(cudaMemsetAsync(group_id_counter, 0, sizeof(int)));
	printf("Average::::::::: %d\n", average);
	
	switch (average){
	case 0 ... 11:
		csr_L_solve_kernel_order_coop<8><<<grid, num_threads, shared_size, stream>>> (mat->ia, mat->ja, mat->a, b, x, info->iorder, info->row_ctr, n);
		break;
	case 12 ... 23:
		csr_L_solve_kernel_order_coop<16><<<grid, num_threads, shared_size, stream>>> (mat->ia, mat->ja, mat->a, b, x, info->iorder, info->row_ctr, n);
		break;
	default:
		csr_L_solve_kernel_order_coop<32><<<grid, num_threads, shared_size, stream>>> (mat->ia, mat->ja, mat->a, b, x, info->iorder, info->row_ctr, n);
		break;
	}
}

void csr_L_solve_order(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n, cudaStream_t stream = 0) {
    int rows = mat->nr;
    int nnz = mat->nnz;
    int average = nnz/rows;
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;

    int grid = ceil((double)n * WARP_SIZE / (double)num_threads);

    // CUDA_CHK( cudaMemset(is_solved, 0 , n * sizeof(int)  ) )
    CUDA_CHK(cudaMemsetAsync(x, 0xFF, n * sizeof(VALUE_TYPE), stream));
    CUDA_CHK(cudaMemsetAsync(info->row_ctr, 0, sizeof(int), stream));
    int shared_size = WARP_PER_BLOCK * (sizeof(int) + sizeof(VALUE_TYPE));
    if(ORDER_COOP){
        printf("Using Order Coop\n");
        aux_call_csr_L_solve_kernel_order_coop(mat, info, b, x, n, stream, grid, num_threads, shared_size, average);
    }else{
        csr_L_solve_kernel_order<<<grid, num_threads, shared_size, stream>>> (mat->ia, mat->ja, mat->a, b, x, info->iorder, info->row_ctr, n);
    }
    
}
