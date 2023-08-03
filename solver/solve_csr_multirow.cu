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

__global__ void csr_L_solve_kernel_multirow(const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const VALUE_TYPE* __restrict__ val,
    const VALUE_TYPE* __restrict__ b,
    volatile VALUE_TYPE* x,
    int* iorder, int* warp_base_idx, int* warp_vect_size,
    int* row_ctr, int n, int n_warps) {

    thread_group tile32 = tiled_partition(this_thread_block(), 32);
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

    int nxt_row = row_ptr[row_idx + 1];  // Hasta que posicion va la fila




    // El warp que tenga vect_size = 0 setea x y muere (el warp completo)

    if (vect_size == 0) {
        x[row_idx] = b[row_idx] / val[nxt_row - 1];
        return;
    }

    tile32.sync();

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


	 //printf("Warp: %d row: %d waiting for col %d\n", wrp,row_idx, colidx);


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

	tile32.sync();


    // Reduccion
    for (int i = vect_size / 2; i >= 1; i /= 2) {
        left_sum += __shfl_down_sync(__activemask(), left_sum, i, vect_size);
    }



    if (vect_off == 0) {
        // escribo en el resultado
        x[row_idx] = left_sum * piv;

    }
}

void csr_L_solve_multirow(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n, cudaStream_t stream = 0) {

    int num_threads = WARP_PER_BLOCK * WARP_SIZE;

    int grid = ceil((double)info->n_warps * WARP_SIZE / (double)num_threads);

    CUDA_CHK(cudaMemsetAsync(x, 0xFF, n * sizeof(VALUE_TYPE), stream));

    CUDA_CHK(cudaMemsetAsync(info->row_ctr, 0, sizeof(int), stream));

    csr_L_solve_kernel_multirow << <grid, num_threads, 0, stream >> > (mat->ia, mat->ja, mat->a,
        b, x,
        info->iorder,
        info->ibase_row, info->ivect_size_warp,
        info->row_ctr, n, info->n_warps);
    
	cudaDeviceSynchronize();

}

/*
__global__ void update_alpha_kernel(VALUE_TYPE* out, VALUE_TYPE* alpha, VALUE_TYPE* rho){

    out[0] = -rho[0] / alpha[0];
}

void update_alpha(VALUE_TYPE* out, VALUE_TYPE* alpha, VALUE_TYPE* rho, cudaStream_t stream){

    update_alpha_kernel<<< 1 , 1, 0, stream>>>(out, alpha, rho);
}


__global__ void reverse_array_inplace_kernel(VALUE_TYPE* arr_in, int n){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid >= n/2) return;

    VALUE_TYPE swp = arr_in[n-tid-1];

    arr_in[n-tid-1] = arr_in[tid];

    arr_in[tid] = swp;
}


void reverse_array_inplace(VALUE_TYPE* arr_in, int n, cudaStream_t stream){

    int num_threads = WARP_PER_BLOCK*WARP_SIZE;
    int grid = ceil ((double)(n/2) / (double)num_threads);

    reverse_array_inplace_kernel<<< grid , num_threads, 0, stream>>>(arr_in, n);
}

/*
__global__ void csr_L_solve_kernel_multirow_hash1(  const int* __restrict__ row_ptr,
                                              const int* __restrict__ col_idx,
                                              const VALUE_TYPE* __restrict__ val,
                                              const VALUE_TYPE* __restrict__ b,
                                              volatile VALUE_TYPE* x,
                                              int * iorder, int * warp_base_idx, int * warp_vect_size,
                                              int * row_ctr, int n, int n_warps ) {


      //  int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;

    // __shared__ int blIdx;
    // if(threadIdx.x==0) blIdx = atomicAdd(&(row_ctr[0]), 1);

    extern __shared__ int shmem[];

    int * sh_ind = &shmem[0];
    VALUE_TYPE * sh_vals = (VALUE_TYPE*) &sh_ind[SIZE_SHARED];

    int lne0 = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp
    // int t1 = clock();
    int wrp;


    if(threadIdx.x < SIZE_SHARED) sh_ind[threadIdx.x]  = -1;

    // while(1)
    for (int i = 0; i < IT_HASH; ++i)
    {

        if(lne0==0) wrp = atomicAdd(&(row_ctr[0]), 1);

        // __syncthreads();

        wrp = __shfl_sync(__activemask(),wrp,0);

        // int wrp = (threadIdx.x + blIdx * blockDim.x) / WARP_SIZE;


        if(wrp >= n_warps) return;

        int vect_size = warp_vect_size[wrp];
        int base_idx  = warp_base_idx [wrp];

        int n_vects = warp_base_idx [wrp+1] - base_idx;

        int vect_idx = (vect_size==0)? lne0 : lne0 / vect_size;

        int row_idx = iorder[base_idx + vect_idx];

        // when executed in loop with few blocks to minimize occupancy and enable
        // concurrent execution with streams, the return statement would cause a
        // deadlock because some threads of the warp terminate the execution and
        // are not available in the next iteration. So we replace it with a continue statement.
        if( (row_idx >= n) || (vect_idx >= n_vects) ) continue;
        // if( (row_idx >= n) || (vect_idx >= n_vects) ) return;

        int nxt_row = row_ptr[row_idx+1] - 1;

        if(vect_size==0){
            x[row_idx] = b[row_idx] / val[nxt_row];
            // iorder[row_idx] = blockIdx.x;

            // return;
            continue;
        }

        int vect_off = lne0 % vect_size;

        int sh_offset = row_idx % 509; // SIZE_SHARED;

        atomicCAS(&sh_ind[sh_offset], -1, row_idx);
        // while ( atomicCAS(&sh_ind[sh_offset], -1, row_idx) != -1 && (row_idx % SIZE_SHARED) != (sh_offset = (sh_offset+1)%SIZE_SHARED)) ;

                // if(vect_off==0){ //por alguna razón demora más...
                // sh_ind[sh_offset]  = row_idx;
                #ifdef __double__
                sh_vals[sh_offset] = __longlong_as_double ( 0xFFFFFFFFFFFFFFFF );
                #else
                sh_vals[sh_offset] = __int_as_float ( 0xFFFFFFFF );
                #endif
            // }
        __syncthreads();

        int row = row_ptr[row_idx];

        VALUE_TYPE left_sum = 0;
        VALUE_TYPE piv;

        if(vect_off==0){
            piv = 1 / val[nxt_row];
            left_sum = b[row_idx];

            //prueba borrar!
            // iorder[row_idx] = blockIdx.x;
        }

        int off = row + vect_off;

        VALUE_TYPE my_val = val[off];
        int colidx = col_idx[off];

        // linear search for the index corresponding to my x entry
        // int sh_colidx = -1;
        // int ii=0;
        // while (ii < blockDim.x && sh_ind[ii]!=colidx) ii++;
        // if (ii < blockDim.x) sh_colidx=ii;

        // look in hash table
        int consumer_off=colidx;
        // while ( sh_ind[consumer_off%SIZE_SHARED]!=colidx && sh_ind[consumer_off%SIZE_SHARED]!=-1  && sh_ind[consumer_off%SIZE_SHARED]%SIZE_SHARED==colidx%SIZE_SHARED) consumer_off++;

        // int sh_colidx = (sh_ind[consumer_off%SIZE_SHARED]!=-1) ? sh_colidx=consumer_off%SIZE_SHARED : -1;
        // int sh_colidx = (sh_ind[colidx%SIZE_SHARED]==colidx) ? sh_colidx=colidx%SIZE_SHARED : -1;
        int sh_colidx = (sh_ind[colidx%509]==colidx) ? sh_colidx=colidx%509 : -1;



        VALUE_TYPE xx;
        volatile VALUE_TYPE * xx_ptr =  &(x[colidx]);

        int ready = 0;

        // int t2 = clock();
        // if(lne0==0) times[wrp].ticks_ini = t2 - t1;

        while(off < nxt_row)
        {
            if(!ready){
                if (sh_ind[sh_colidx]==colidx)
                    xx = sh_vals[sh_colidx];
                else
                {
                    // xx = x[colidx];
                    #ifdef __double__
                    asm volatile ("ld.global.cg.f64 %0, [%1];" : "=d"(xx) : "l"(xx_ptr));
                    #else
                    asm volatile ("ld.global.cg.f32 %0, [%1];" : "=f"(xx) : "l"(xx_ptr));
                    #endif
                }
                ready = __double2hiint ( xx ) != (int) 0xFFFFFFFF;
            }

            if( __all_sync(__activemask(), ready ) ){

                left_sum -= my_val * xx; //x[colidx];

                off += vect_size;

                if(off < nxt_row){
                    colidx = col_idx[off];
                    xx_ptr =  &(x[colidx]);

                    sh_colidx = (sh_ind[colidx%509]==colidx) ? sh_colidx=colidx%509 : -1;
                    consumer_off = colidx;
                    // while ( sh_ind[consumer_off%SIZE_SHARED]!=colidx && sh_ind[consumer_off%SIZE_SHARED]!=-1) consumer_off++;
                    // sh_colidx = (sh_ind[consumer_off%509]!=-1) ? sh_colidx=consumer_off%509 : -1;

                    my_val = val[off];
                    ready=0;
                }
            }
        }

        // t1 = clock();
        // if(lne0==0) times[wrp].ticks_wait = t1 - t2;


        // Reduccion
        for (int i=vect_size/2; i>=1; i/=2){
            left_sum += __shfl_down_sync(__activemask(),left_sum, i, vect_size);
        }

        if(vect_off==0){

            //escribo en el resultado
            x[row_idx]         = left_sum * piv;
            sh_vals[sh_offset] = left_sum * piv;
            // __threadfence();

            // is_solved[row_idx] = 1;
        }

        __syncthreads();
        // t2 = clock();
        // if(lne0==0) times[wrp].ticks_end = t2 - t1;
    }
}
*/

/*
void csr_L_solve_multirow_hash1 (  sp_mat_t * mat, dfr_analysis_info_t * info, const VALUE_TYPE * b, VALUE_TYPE * x, int n  ){


        int num_threads = WARP_PER_BLOCK*WARP_SIZE;

        int grid = ceil ((double) (info->n_warps / IT_HASH+1) * WARP_SIZE / (double) num_threads );

        CUDA_CHK( cudaMemset(x             , 0xFF , n * sizeof(VALUE_TYPE)  ) )
        CUDA_CHK( cudaMemset(info->row_ctr , 0    ,     sizeof(int)         ) )

        csr_L_solve_kernel_multirow_hash1<<< grid , num_threads, SIZE_SHARED * (sizeof(int) + sizeof(VALUE_TYPE)) >>>( mat->ia, mat->ja, mat->a,
                                                               b, x,
                                                               info->iorder,
                                                               info->ibase_row, info->ivect_size_warp,
                                                               info->row_ctr, n, info->n_warps);
}
*/
