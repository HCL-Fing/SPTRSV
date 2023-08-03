#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include "common.h"
#include "dfr_syncfree.h"
#include "cub/util_type.cuh"

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

__global__ void csr_L_solve_kernel_multirow_first(const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const VALUE_TYPE* __restrict__ val,
    const VALUE_TYPE* __restrict__ b,
    volatile VALUE_TYPE* x,
    int* iorder, int* warp_base_idx, int* warp_vect_size,
    int* row_ctr, int n, int n_warps, 
    int* mat_cols, VALUE_TYPE* mat_values, VALUE_TYPE* mat_diag, int* mat_row_idx) {

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



    float v;
    // El warp que tenga vect_size = 0 setea x y muere (el warp completo)
    if (vect_size == 0) {
        v = val[nxt_row - 1];
        x[row_idx] = b[row_idx] / v;
        mat_diag[base_idx+vect_idx] = v;      //Agrego los dos de abajo, chequear
        mat_row_idx[base_idx+vect_idx] = row_idx; 

        return;
    }

    tile32.sync();

    int vect_off = lne0 % vect_size;  // Cual le toca a cada thread (por si el warp es mas grande que la cantidad a procesar)


    int row = row_ptr[row_idx];  // El primer elemento de la fila

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv;

    if (vect_off == 0) {
        v = val[nxt_row - 1];    //Read first para la asignación final
        piv = 1 / v;
        left_sum = b[row_idx];
    }

    int off = row + vect_off;

    VALUE_TYPE my_val;  // El elemento de la fila que trabaja el thread
    VALUE_TYPE xx;
    int ready = 0;

    tile32.sync();

    int colidx;


    if(vect_size<32){


        if(off>=nxt_row-1){ 
            colidx=-1;
        }else{
            colidx = col_idx[off];
        }


        mat_cols[wrp*WARP_SIZE + lne0] = colidx;
        if(off>=nxt_row-1) return;

        my_val = val[off];
        mat_values[wrp*WARP_SIZE + lne0] = my_val;        

        while(!ready){

            xx = x[colidx];
            ready = __double2hiint(xx) != (int)0xFFFFFFFF;  
        }

        left_sum -= my_val * xx;  // left_sum es la suma parcial del valor del vector b de esa fila    


        tile32.sync();
        for (int i = vect_size / 2; i >= 1; i /= 2) {
            left_sum += __shfl_down_sync(__activemask(), left_sum, i, vect_size);
        }

            



        if (vect_off == 0) {
            // escribo en el resultado
            x[row_idx] = left_sum * piv;                        //Ver de hacer coalesced estos accesos (x3)
            mat_diag[base_idx+vect_idx] = v;   
            mat_row_idx[base_idx+vect_idx] = row_idx; 
         }


    }else{
        

        colidx = col_idx[off];
        my_val = val[off];
        if(vect_off == WARP_SIZE-1){                     //Idea del dufre de poner la referencia a la fila en el primero y llenar con datos válidos los demás
            mat_cols[wrp*WARP_SIZE] = row + WARP_SIZE-1;
            mat_values[wrp*WARP_SIZE] = nxt_row;                   //Aca guardo nxt_row para recuperarlo fácil en el otro
        }else{
            mat_cols[wrp*WARP_SIZE+lne0+1] = colidx;
            mat_values[wrp*WARP_SIZE+lne0+1] = my_val;
        }



        while (off < nxt_row - 1){ 

            


            if (!ready) {
                xx = x[colidx];
                ready = __double2hiint(xx) != (int)0xFFFFFFFF;

            }

            if (ready) {  // __all es usada para que todos los threads de un warp tomen la misma decision, si hay alguno que no lo cumple, no lo cumple ninguno
                left_sum -= my_val * xx;  // left_sum es la suma parcial del valor del vector b de esa fila
                off += WARP_SIZE;

                if (off < nxt_row) {
                    ready = 0;
                    colidx = col_idx[off];
                    my_val = val[off];                    
                }
            }


        }
        tile32.sync();

        for (int i = vect_size / 2; i >= 1; i /= 2) {
            left_sum += __shfl_down_sync(__activemask(), left_sum, i, vect_size);
        }



        if (vect_off == 0) {
            // escribo en el resultado
            x[row_idx] = left_sum * piv;
            mat_diag[base_idx] = val[nxt_row-1];    

            mat_row_idx[base_idx] = row_idx;
            //analysis_matrix->row_ptr[(nxt_row - row) + small_rows*WARP_SIZE]
        }

    }
}








__global__ void csr_L_solve_kernel_multirow_rest(/*const int* __restrict__ row_ptr,
    */const int* __restrict__ col_idx,
    const VALUE_TYPE* __restrict__ val,
    const VALUE_TYPE* __restrict__ b,
    volatile VALUE_TYPE* x,
    /*int* iorder, */int* warp_base_idx, int* warp_vect_size,
    int* row_ctr, int n, int n_warps,
    int* mat_cols, VALUE_TYPE* mat_values, VALUE_TYPE* mat_diag, int* mat_row_idx) {



    thread_group tile32 = tiled_partition(this_thread_block(), 32);
    int wrp;                                      // identifica numero del warp
  
    int lne0 = tile32.thread_rank();

    if (lne0 == 0) wrp = atomicAdd(&(row_ctr[0]), 1);
    wrp = __shfl_sync(__activemask(), wrp, 0);

    if (wrp >= n_warps) return;

    int vect_size = warp_vect_size[wrp];

    int base_idx = warp_base_idx[wrp];  
    int vect_idx = (vect_size == 0) ? lne0 : lne0 / vect_size;  
    int n_vects = warp_base_idx[wrp + 1] - base_idx;  // Cantidad de elementos que tiene que procesar
    if(vect_idx>=n_vects) return;
    //    int vect_off = lne0 % vect_size;
    int vect_off = lne0 % vect_size; 
    int row_idx = mat_row_idx[base_idx + vect_idx];
    //if(row_idx>=n) return;



    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv;

    if (vect_size == 0) {
        x[row_idx] = b[row_idx] / mat_diag[base_idx+vect_idx];
        return;
    }
    
    
    VALUE_TYPE my_val;  // El elemento de la fila que trabaja el thread
    VALUE_TYPE xx;
    
    if (vect_off == 0) {
        piv = 1 / mat_diag[base_idx+vect_idx] ;
        left_sum = b[row_idx];
    }

    int ready = 0;

    tile32.sync();

    int colidx;

    int off;
    if(vect_size < 32){
        off = WARP_SIZE*wrp + lne0;


        colidx = mat_cols[off];//col_idx[off];
        if(colidx==-1) return;
        my_val = mat_values[off];//val[off];
        
        while(!ready){
            xx = x[colidx];
            ready = __double2hiint(xx) != (int)0xFFFFFFFF;             
        }
        left_sum -= my_val * xx;    

        tile32.sync();
        for (int i = vect_size / 2; i >= 1; i /= 2) {
            left_sum += __shfl_down_sync(__activemask(), left_sum, i, vect_size);
        }


        if(vect_off==0) x[row_idx] = left_sum * piv;


    }else{

        colidx = mat_cols[wrp*WARP_SIZE+lne0];
        my_val = mat_values[wrp*WARP_SIZE+lne0];

        off = __shfl_sync(__activemask(), colidx, 0);  
        int nxt_row = __shfl_sync(__activemask(), my_val, 0);

        if(lne0 == 0){
            //off+= WARP_SIZE;
            colidx = col_idx[off];
            my_val = val[off];
        }else{
            off+= + lne0 - WARP_SIZE;
        }
        
        while (off < nxt_row - 1){ 

            if (!ready) {
                xx = x[colidx];
                ready = __double2hiint(xx) != (int)0xFFFFFFFF;

            }

            if (ready) {  // __all es usada para que todos los threads de un warp tomen la misma decision, si hay alguno que no lo cumple, no lo cumple ninguno
                left_sum -= my_val * xx;  // left_sum es la suma parcial del valor del vector b de esa fila
                off += WARP_SIZE;

         
                if (off < nxt_row) {
                    ready = 0;
                    colidx = col_idx[off];
                    my_val = val[off];
                }
            }

        }
        tile32.sync();

        for (int i = vect_size / 2; i >= 1; i /= 2) {
            left_sum += __shfl_down_sync(__activemask(), left_sum, i, vect_size);
        }



        if (lne0 == 0) {
            // escribo en el resultado
            x[row_idx] = left_sum * piv;
        }
    }
}








void csr_L_solve_multirow_format(sp_mat_t* mat, dfr_analysis_info_t* info, const VALUE_TYPE* b, VALUE_TYPE* x, int n, sp_mat_ana_t* mat_info, cudaStream_t stream = 0) {

	//printf("Starting Format %i warps\n",info->n_warps);


    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int grid = ceil((double)info->n_warps * WARP_SIZE / (double)num_threads);
    CUDA_CHK(cudaMemsetAsync(x, 0xFF, n * sizeof(VALUE_TYPE), stream));
    CUDA_CHK(cudaMemsetAsync(info->row_ctr, 0, sizeof(int), stream));
 
    if(mat_info->first){
    	//printf("First iteration\n");


        mat_info->first=false;
        CUDA_CHK(cudaMemsetAsync(mat_info->values, 0, sizeof(VALUE_TYPE)*info->n_warps*WARP_SIZE , stream));
     //   CUDA_CHK(cudaMemsetAsync(mat_info->cols, 0, sizeof(int)*info->n_warps*WARP_SIZE , stream));


        csr_L_solve_kernel_multirow_first << <grid, num_threads, 0, stream >> > (mat->ia, mat->ja, mat->a,
        	b, x,
        	info->iorder,
        	info->ibase_row, info->ivect_size_warp,
        	info->row_ctr, n, info->n_warps,  
		mat_info->cols, mat_info->values, mat_info->diag, mat_info->row_idx);


        
    
    }else{
	    //printf("Rest iteration\n");
        csr_L_solve_kernel_multirow_rest << <grid, num_threads, 0, stream >> > (
        	mat->ja, mat->a, b, x,
        	info->ibase_row, info->ivect_size_warp,
        	info->row_ctr, n, info->n_warps, 
		mat_info->cols, mat_info->values, mat_info->diag, mat_info->row_idx);

    }


        cudaDeviceSynchronize();
   	//printf("Exiting format\n") ;
}
