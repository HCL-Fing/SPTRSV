#include <cooperative_groups.h>

#include "common.h"
#include "dfr_syncfree.h"

using namespace cooperative_groups;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
}
#endif

__global__ void csr_L_solve_kernel_multirow(const int* __restrict__ row_ptr,
                                            const int* __restrict__ col_idx,
                                            const VALUE_TYPE* __restrict__ val,
                                            const VALUE_TYPE* __restrict__ b,
                                            volatile VALUE_TYPE* x,
                                            int* iorder, int* warp_base_idx, int* warp_vect_size,
                                            int* row_ctr, int n, int n_warps) {
    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());

    int wrp;  // warp identifier

    int lne0 = tile32.thread_rank();

    if (lne0 == 0) wrp = atomicAdd(&(row_ctr[0]), 1);
    wrp = __shfl_sync(__activemask(), wrp, 0);

    if (wrp >= n_warps) return;

    int vect_size = warp_vect_size[wrp];  // Number of column the warp needs to process

    int base_idx = warp_base_idx[wrp];  // Starting column

    int n_vects = warp_base_idx[wrp + 1] - base_idx;  // Number of elements to process

    int vect_idx = (vect_size == 0) ? lne0 : lne0 / vect_size;  // Thread starting value

    int row_idx = iorder[base_idx + vect_idx];  // This is the value for the first row corresponeding to the thread

    
    if ((row_idx >= n) || (vect_idx >= n_vects)) return;

    int nxt_row = row_ptr[row_idx + 1];  // Next row starting position

    //The warp that has vect_size = 0 sets x and dies (the whole warp)

    if (vect_size == 0) {
        x[row_idx] = b[row_idx] / val[nxt_row - 1];
        return;
    }

    tile32.sync();

    int vect_off = lne0 % vect_size;  

    int row = row_ptr[row_idx];  // Row value
    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv;

    if (vect_off == 0) {
        piv = 1 / val[nxt_row - 1];
        left_sum = b[row_idx];
    }

    int off = row + vect_off;

    VALUE_TYPE my_val;  // Element corresponding to the thread
    VALUE_TYPE xx;
    int ready = 0;

    tile32.sync();

    int colidx;

    while (off < nxt_row - 1) {
        colidx = col_idx[off];
        my_val = val[off];

        if (!ready) {
            xx = x[colidx];
            ready = __double2hiint(xx) != (int)0xFFFFFFFF;
        }

        if (ready) {                  
            left_sum -= my_val * xx;  // left_sum is the partial sum of the values correpsing to the row
            off += vect_size;

            if (off < nxt_row) {
                ready = 0;
            }
        }
    }

    tile32.sync();

    // Reduction
    for (int i = vect_size / 2; i >= 1; i /= 2) {
        left_sum += __shfl_down_sync(__activemask(), left_sum, i, vect_size);
    }

    if (vect_off == 0) {
        // Write result
        x[row_idx] = left_sum * piv;
    }
}

void csr_L_solve_multirow(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n, cudaStream_t stream = 0) {
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;

    int grid = ceil((double)info->n_warps * WARP_SIZE / (double)num_threads);

    CUDA_CHK(cudaMemsetAsync(x, 0xFF, n * sizeof(VALUE_TYPE), stream));

    CUDA_CHK(cudaMemsetAsync(info->row_ctr, 0, sizeof(int), stream));

    csr_L_solve_kernel_multirow<<<grid, num_threads, 0, stream>>>(mat->ia, mat->ja, mat->a,
                                                                  b, x,
                                                                  info->iorder,
                                                                  info->ibase_row, info->ivect_size_warp,
                                                                  info->row_ctr, n, info->n_warps);
}
