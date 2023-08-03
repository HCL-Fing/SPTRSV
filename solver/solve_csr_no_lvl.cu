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
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
}
#endif

__global__ void csr_L_solve_kernel_no_lvl(const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const VALUE_TYPE* __restrict__ val,
    const VALUE_TYPE* __restrict__ b,
    volatile VALUE_TYPE* x,
    int* iorder, int* warp_base_idx, int* warp_vect_size,
    int* row_ctr, int n, int n_warps) {

    thread_group tile32 = tiled_partition(this_thread_block(), 32);
    __shared__ int lft[48];	//48 es para que los vect_size/2 no se vayan de rango
    int wrp;                                      // identifica numero del warp
  
    int lne0 = tile32.thread_rank();

    if (lne0 == 0) wrp = atomicAdd(&(row_ctr[0]), 1);
    wrp = __shfl_sync(__activemask(), wrp, 0);

    if (wrp >= n_warps) return;

    int vect_size = warp_vect_size[wrp];  // Cantidad de columnas que procesa el warp


    int base_idx = warp_base_idx[wrp];  // En que columna empieza

    int n_vects = warp_base_idx[wrp + 1] - base_idx;  // Cantidad de elementos que tiene que procesar

    int vect_idx = (vect_size == 0) ? lne0 : lne0 / vect_size;  // En cual arranca cada thread

    int row_idx = iorder[base_idx + vect_idx];  // Es la fila que esta en la primera posicion para este thread

    //return;
    // when executed in loop with few blocks to minimize occupancy and enable
    // concurrent execution with streams, the return statement would cause a
    // deadlock because some threads of the warp terminate the execution and
    // are not available in the next iteration. So we replace it with a continue statement.
    if ((row_idx >= n) || (vect_idx >= n_vects)) return;
    else printf("Thread: %i. row: %i. Vect: %i\n",threadIdx.x, row_idx, vect_idx);
    int nxt_row = row_ptr[row_idx + 1];  // Hasta que posicion va la fila




    // El warp que tenga vect_size = 0 setea x y muere (el warp completo)

    if (vect_size == 0) {
        x[row_idx] = b[row_idx] / val[nxt_row - 1];
        return;
    }

    tile32.sync();
    thread_group tileSize = tiled_partition(tile32, vect_size);


    int vect_off = lne0 % vect_size;  // Cual le toca a cada thread (por si el warp es mas grande que la cantidad a procesar)


    int row = row_ptr[row_idx];  // El valor de la fila

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv;

    if (vect_off == 0) {
        piv = 1 / val[nxt_row - 1];
        left_sum = b[row_idx];
    }

    int off = row + vect_off;

    VALUE_TYPE my_val;  // El elemento de la fila que trabaja el thread
    VALUE_TYPE xx;
    int ready = 0;

    tile32.sync();

    int colidx;

    // El problema esta en que en determinado momento deja de entrar en el segundo if, entonces deja de avanzar
    // en la fila, sin haber llegado al final de esta
    while (off < nxt_row - 1) {

        colidx = col_idx[off];
        my_val = val[off];

        if (!ready) {
            xx = x[colidx];
            ready = __double2hiint(xx) != (int)0xFFFFFFFF;

        }

        if (ready) {  // __all es usada para que todos los threads de un warp tomen la misma decision, si hay alguno que no lo cumple, no lo cumple ninguno
            left_sum -= my_val * xx;  // left_sum es la suma parcial del valor del vector b de esa fila
            off += vect_size;

            if (off < nxt_row) {
                ready = 0;
            }
        }
    }

    
	//tile32.sync();
    lft[lne0] = left_sum;

    // Reduccion
    for (int i = vect_size / 2; i >= 1; i /= 2) {
        //left_sum += __shfl_down_sync(__activemask(), left_sum, i, vect_size);
        //left_sum += tileSize.shfl_down(left_sum, i);
	tileSize.sync();
	left_sum += lft[i+lne0];
	tileSize.sync();
	lft[i+lne0] = left_sum;
    }



    if (vect_off == 0) {
        // escribo en el resultado
        x[row_idx] = left_sum * piv;

    }
}

void csr_L_solve_no_lvl(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n, cudaStream_t stream = 0) {

    int num_threads = WARP_PER_BLOCK * WARP_SIZE;

    int grid = ceil((double)info->n_warps * WARP_SIZE / (double)num_threads);

    CUDA_CHK(cudaMemsetAsync(x, 0xFF, n * sizeof(VALUE_TYPE), stream));

    CUDA_CHK(cudaMemsetAsync(info->row_ctr, 0, sizeof(int), stream));
printf("N: %i. num_threads: %i. Grid: %i. N_warps: %i\n",n, num_threads, grid,info->n_warps);
    csr_L_solve_kernel_no_lvl << <grid, num_threads, 0, stream >> > (mat->ia, mat->ja, mat->a,
        b, x,
        info->iorder,
        info->ibase_row, info->ivect_size_warp,
        info->row_ctr, n, info->n_warps);
    


}
