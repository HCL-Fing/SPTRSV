#ifndef _SPTS_SYNCFREE_CUDA_
#define _SPTS_SYNCFREE_CUDA_

#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cusparse_v2.h>

struct spts_times
{
    int ticks_ini;
    int ticks_wait;
    int ticks_end;
};


//__device__ int row_ctr;

#if VALUE_TYPE == double
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
unsigned long long int* address_as_ull =
(unsigned long long int*)address;
unsigned long long int old = *address_as_ull, assumed;
do {
assumed = old;
old = atomicCAS(address_as_ull, assumed,
__double_as_longlong(val +
__longlong_as_double(assumed)));
// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
} while (assumed != old);
}
#endif
#endif


int validate_x(const VALUE_TYPE * x_ref, const VALUE_TYPE * x, int n, const char * func){
    // validate x
    int err_counter = 0;

    for (int i = 0; i < n; i++)
    {
        if (abs(x_ref[i] - x[i]) > 0.01 * abs(x_ref[i])){
            err_counter++;
            printf("Error x[%i]=%f\n",i,x[i] );
        }
    }

    if (!err_counter)
        printf("%s test passed!\n", func);
    else
        printf("%s test failed! %d errors\n", func, err_counter);

    return err_counter;
}



__global__ void forward_csr_L_solve_kernel_multirow(   const int* __restrict__ row_ptr, 
                                              const int* __restrict__ col_idx, 
                                              const VALUE_TYPE* __restrict__ val, 
                                              const VALUE_TYPE* __restrict__ b, 
                                              volatile VALUE_TYPE* x, 
                                              volatile int * is_solved, 
                                              int * iorder, int * warp_base_idx, int * warp_vect_size,
                                              int * row_ctr, int n, int n_warps, struct spts_times * times ) {


    /* 
        int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
    /*/
    __shared__ int blIdx;

    int t1 = clock();

    if(threadIdx.x==0) blIdx = atomicAdd(&(row_ctr[0]), 1);

    __syncthreads();

    int wrp = (threadIdx.x + blIdx * blockDim.x) / WARP_SIZE;
    //*/

    if(wrp >= n_warps) return;

    int vect_size = warp_vect_size[wrp];
    int base_idx  = warp_base_idx [wrp];

    int n_vects = warp_base_idx [wrp+1] - base_idx;

    int lne0 = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp
    int vect_idx = lne0 / vect_size;

    int row_idx = iorder[base_idx + vect_idx];

    if( (row_idx >= n) || (vect_idx >= n_vects) ) return;

    int nxt_row = row_ptr[row_idx+1];

    if(vect_size==1){
        //escribo en el resultado
        x[row_idx] = b[row_idx] / val[nxt_row-1];
        __threadfence();
        is_solved[row_idx] = 1;
        return;
    }
    
    int vect_off = lne0 % vect_size;
    int row = row_ptr[row_idx];

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv;

    if(vect_off==0){
        piv = 1 / val[nxt_row-1];
        left_sum = b[row_idx];
    }

    int off = row + vect_off;

    int colidx = col_idx[off];
    VALUE_TYPE my_val = val[off];

    int ready = 0; 

    int t2 = clock();
    if(lne0==0) times[wrp].ticks_ini = t2 - t1;

    while(off < nxt_row - 1)
    {
        if(!ready) ready = is_solved[colidx];

        if( __all( ready ) ){ 

            left_sum -= my_val * x[colidx];

            off += vect_size;

            if(off < nxt_row - 1){
                colidx = col_idx[off];
                my_val = val[off];
                ready=0;
            }
        }
    }

    t1 = clock();
    if(lne0==0) times[wrp].ticks_wait = t1 - t2;


    // Reduccion
    for (int i=vect_size/2; i>=1; i/=2){
        left_sum += __shfl_down(left_sum, i, vect_size);
    }

    if(vect_off==0){

        //escribo en el resultado
        x[row_idx] = left_sum * piv; 

        __threadfence();

        is_solved[row_idx] = 1;
    }

    // t2 = clock();
    // if(lne0==0) times[wrp].ticks_end = t2 - t1;

}



__global__ void forward_csr_L_solve_kernel(   const int* __restrict__ row_ptr, 
                                              const int* __restrict__ col_idx, 
                                              const VALUE_TYPE* __restrict__ val, 
                                              const VALUE_TYPE* __restrict__ b, 
                                              VALUE_TYPE* x, 
                                              int * is_solved, 
                                              int * iorder, int * inv_iorder,
                                              int n ) {

    extern volatile __shared__ int s_mem[];       

    int WARP_PER_BLOCK = blockDim.x / WARP_SIZE;

    int * s_is_solved = (int *) &s_mem[0];
    VALUE_TYPE * s_x  = (VALUE_TYPE*) &s_is_solved[WARP_PER_BLOCK];

    int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;

    int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

    int row_idx = iorder[wrp];

    if(wrp >= n) return;
    
    int row = row_ptr[row_idx];
    int start_row = blockIdx.x*WARP_PER_BLOCK;
    int nxt_row = row_ptr[row_idx+1];

    int local_warp_id = wrp - start_row ; //threadIdx.x / WARP_SIZE;

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv = 1 / val[nxt_row-1];
 
    if(lne==0){
        left_sum = b[row_idx];
        s_is_solved[local_warp_id] = 0;
    }

    __syncthreads();

    int off = row+lne;
    int colidx = col_idx[off];
    int who = inv_iorder[colidx];

    VALUE_TYPE my_val = val[off];

    int myvar = 0;

    while(off < nxt_row - 1)
    {
        if(!myvar)
        {            
            //if(colidx > start_row){

            // if(who >= start_row  && who < start_row + WARP_PER_BLOCK){
            //     //printf("en sahared col=%d who=%d start=%d wrp=%d\n", colidx, who, start_row, wrp );
            //     //myvar = s_is_solved[colidx-start_row];
            //     myvar = s_is_solved[who-start_row];
            //     if (myvar){
            //         //left_sum -= my_val * s_x[colidx-start_row];
            //         left_sum -= my_val * s_x[who-start_row];
            //     }
            // }
            // else
            {
                myvar = is_solved[colidx];

                if (myvar){
                    left_sum -= my_val * x[colidx];
                }

            }
        } 

        if( __all(myvar) ){

            off+=WARP_SIZE;
            colidx = col_idx[off];
            who = inv_iorder[colidx];
            my_val = val[off];

            myvar=0;
        }
    }
    
    // Reduccion
    for (int i=16; i>=1; i/=2){
        left_sum += __shfl_down(left_sum, i);
    }
     
    if(lne==0){

        //escribo en el resultado
        
        s_x[local_warp_id] = left_sum * piv;
        s_is_solved[local_warp_id] = 1;
        
        x[row_idx] = left_sum * piv;

        __threadfence();

        is_solved[row_idx] = 1;
    }
}

__global__ void forward_csr_L_solve_kernel_2porwarp(   const int* __restrict__ row_ptr, 
                                              const int* __restrict__ col_idx, 
                                              const VALUE_TYPE* __restrict__ val, 
                                              const VALUE_TYPE* __restrict__ b, 
                                              volatile VALUE_TYPE* x, 
                                              volatile int * is_solved, 
                                              int * iorder, int * inv_iorder,
                                              int n ) {
/*
    extern volatile __shared__ int s_mem[];       

    int WARP_PER_BLOCK = blockDim.x / WARP_SIZE;

    int * s_is_solved = (int *) &s_mem[0];
    VALUE_TYPE * s_x  = (VALUE_TYPE*) &s_is_solved[WARP_PER_BLOCK];
*/
    int wrp = ((threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE) * 2;

    int lne0 = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp
    int lne = lne0;

    if (lne >= 16){ 
        lne -= 16;
        wrp++;
    }

//    int row_idx = wrp;
    int row_idx = iorder[wrp];

    if(row_idx >= n) return;
    
    //int start_row = blockIdx.x*WARP_PER_BLOCK;
    int row = row_ptr[row_idx];
    int nxt_row = row_ptr[row_idx+1];

    //int local_warp_id = wrp - start_row ; //threadIdx.x / WARP_SIZE;

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv = 1 / val[nxt_row-1];

    if(lne==0){
        left_sum = b[row_idx];
//        s_is_solved[local_warp_id] = 0;
    }

    //__syncthreads();

    int off = row+lne;
    int colidx = col_idx[off];
    //int who = inv_iorder[colidx];

    VALUE_TYPE my_val = val[off];
 
    int desc = 1;

    if ((lne0 >= 16) && (nxt_row-2 >= row) && (col_idx[nxt_row-2]==wrp)){ 
        desc++; 
    }    

    int myvar = 0;

    while(off < nxt_row - desc)
    {
        if(!myvar) myvar = is_solved[colidx];

        if( __all( myvar ) ){ 

            left_sum -= my_val * x[colidx];

            off+= 16; //WARP_SIZE;
            colidx = col_idx[off];
            //who = inv_iorder[colidx];
            my_val = val[off];

            myvar=0;
        }
    }

    // Reduccion
    for (int i=8; i>=1; i/=2){
        left_sum += __shfl_down(left_sum, i, 16);
    }

    left_sum *= piv;

    VALUE_TYPE x0 = __shfl(left_sum, 0, 32);

    if(lne0==16 && desc==2){
        left_sum -= val[nxt_row-2] * x0 * piv;
    }

    if(lne==0){

        //escribo en el resultado
        
//        s_x[local_warp_id] = left_sum * piv;
//        s_is_solved[local_warp_id] = 1;
        x[row_idx] = left_sum; 

        __threadfence();

        is_solved[row_idx] = 1;
    }

}


__global__ void forward_csr_L_solve_kernel_2porwarp_shared(   const int* __restrict__ row_ptr, 
                                              const int* __restrict__ col_idx, 
                                              const VALUE_TYPE* __restrict__ val, 
                                              const VALUE_TYPE* __restrict__ b, 
                                              volatile VALUE_TYPE* x, 
                                              volatile int * is_solved, 
                                              int * iorder, int * inv_iorder,
                                              int n ) {

    extern volatile __shared__ int s_mem[];       

    int WARP_PER_BLOCK = blockDim.x / WARP_SIZE * 2;

    int * s_is_solved = (int *) &s_mem[0];
    VALUE_TYPE * s_x  = (VALUE_TYPE*) &s_is_solved[WARP_PER_BLOCK];

    int wrp = ((threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE) * 2;

    int lne0 = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp
    int lne = lne0;

    int row_idx = (lne0 < 16) ? wrp:wrp+1;
//    int row_idx = (lne0 < 16) ? iorder[wrp]:iorder[wrp+1];

    if (lne >= 16) lne -= 16;

    if(row_idx >= n) return;
    
    int start_row = blockIdx.x*WARP_PER_BLOCK;
    int row = row_ptr[row_idx];
    int nxt_row = row_ptr[row_idx+1];

    int local_warp_id = row_idx - start_row ; //threadIdx.x / WARP_SIZE;

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv = 1 / val[nxt_row-1];

    if(lne==0){
        left_sum = b[row_idx];
//        s_is_solved[local_warp_id] = 0;
        char * xx = (char *) &s_x[local_warp_id];
        *xx = 0xFFFFFFFF;
    }

    __syncthreads();

    int off = row+lne;
    int colidx = col_idx[off];
    //int who = inv_iorder[colidx];

    VALUE_TYPE my_val = val[off];
 
    int desc = 1;

    if ((lne0 >= 16) && (nxt_row-2 >= row) && (col_idx[nxt_row-2]==wrp)){ 
        desc++; 
    }    

    int myvar = 0;

    while(off < nxt_row - desc)
    {
        if(!myvar){ 

            VALUE_TYPE xx = (colidx > start_row)?s_x[colidx-start_row]:x[colidx];

//             if(colidx > start_row)
//             {
//                 xx = s_x[colidx-start_row];

//                 // unsigned char * xx = (unsigned char *) &(s_x[colidx-start_row]);    
//                 // myvar = (xx[0]!=0xFF || xx[1]!=0xFF);
// //                myvar = s_is_solved[colidx-start_row];
//                 // if (myvar) left_sum -= my_val * s_x[colidx-start_row];
//             }
//             else 
//             {
//                 xx = x[colidx];

//                 // unsigned char * xx = (unsigned char *) &(x[colidx]);    
//                 // myvar = (xx[0]!=0xFF || xx[1]!=0xFF);
//                 //                myvar = is_solved[colidx];
//                 // if (myvar) left_sum -= my_val * x[colidx];
//             }

            myvar = (*((char*)&xx)!=0xFFFFFFFF);

            if (myvar) left_sum -= my_val * xx;
        }

        if( __all( myvar ) ){ 

            off+= 16; //WARP_SIZE;
            colidx = col_idx[off];
            //who = inv_iorder[colidx];
            my_val = val[off];

            myvar=0;
        }
    }

    // Reduccion
    for (int i=8; i>=1; i/=2){
        left_sum += __shfl_down(left_sum, i, 16);
    }

    left_sum *= piv;

    VALUE_TYPE x0 = __shfl(left_sum, 0, 32);

    if(lne0==16 && desc==2){
        left_sum -= val[nxt_row-2] * x0 * piv;
    }

    if(lne==0){

        //escribo en el resultado        
        s_x[local_warp_id] = left_sum;
//        s_is_solved[local_warp_id] = 1;

        x[row_idx] = left_sum; 

        __threadfence();

//        is_solved[row_idx] = 1;
    }
}



__global__ void forward_csr_L_solve_kernel_levels( const int* __restrict__ row_ptr, 
                                              const int* __restrict__ col_idx, 
                                              const VALUE_TYPE* __restrict__ val, 
                                              const VALUE_TYPE* __restrict__ b, 
                                              VALUE_TYPE* x, 
                                              int * iorder, int * depth, int * level_histogram,
                                              int n ) {


    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int row_idx = iorder[tid];

    if(tid >= n) return;
    
    int row = row_ptr[row_idx];
    int nxt_row = row_ptr[row_idx+1];

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv = 1 / val[nxt_row-1];

    left_sum = b[row_idx];

    int my_lev = depth[tid];

    //niveles empiezan en 1 ?
    //while(my_lev>0 && level_histogram[my_lev-1]);

    for (int i = row; i < nxt_row; ++i)
        left_sum -= val[i] * x[col_idx[i]];    
         
    x[row_idx] = left_sum * piv;

    __threadfence();

    atomicSub(&level_histogram[my_lev],1);
}


__global__ void forward_csr_L_solve_kernel2(   const int* __restrict__ row_ptr, 
                                              const int* __restrict__ col_idx, 
                                              const VALUE_TYPE* __restrict__ val, 
                                              const VALUE_TYPE* __restrict__ b, 
                                              VALUE_TYPE* x, 
                                              int * is_solved, 
                                              int * iorder, int * inv_iorder,
                                              int n ) {

    extern __shared__ int s_mem2[];       

    int WARP_PER_BLOCK = blockDim.x / WARP_SIZE;

    // int * s_is_solved = (int *) &s_mem2[0];
    // VALUE_TYPE * s_x  = (VALUE_TYPE*) &s_is_solved[WARP_PER_BLOCK];

    int * is_solved_cache = (int*) &s_mem2[0];
    VALUE_TYPE * x_cache = (VALUE_TYPE*) &is_solved_cache[blockDim.x];

    int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;

    int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

    int row_idx = iorder[wrp];

    int start_row = blockIdx.x*WARP_PER_BLOCK;


    is_solved_cache[threadIdx.x] = 0;

    // if (start_row - blockDim.x + threadIdx.x >= 0 && start_row - blockDim.x + threadIdx.x < n)
    // {
    //     is_solved_cache[threadIdx.x] = is_solved[start_row - blockDim.x + threadIdx.x];
    //     __threadfence();
    //     x_cache[threadIdx.x] = x[start_row - blockDim.x + threadIdx.x];
    // }

    __syncthreads();


    if(wrp >= n) return;

    // {
    //     is_solved_cache[blockDim.x-threadIdx.x-1] = is_solved[ iorder[ wrp - threadIdx.x - 1 ]]; //- blockDim.x + threadIdx.x ] ];
    //     x_cache[blockDim.x-threadIdx.x-1] = x[ iorder[ wrp - threadIdx.x - 1 ]]; //- blockDim.x + threadIdx.x ] ];
    // } 
    
    int row = row_ptr[row_idx];
    int nxt_row = row_ptr[row_idx+1];

    int local_warp_id = wrp - start_row ; //threadIdx.x / WARP_SIZE;

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv = 1 / val[nxt_row-1];
 
    if(lne==0){
        left_sum = b[row_idx];
        //s_is_solved[local_warp_id] = 0;
    }

    int off = row+lne;
    int colidx = col_idx[off];

    //int sh_idx = colidx - start_row + blockDim.x;

    VALUE_TYPE my_val = val[off];

    int myvar = 0;

    while(off < nxt_row - 1)
    {

        int who = inv_iorder[colidx];
        int sh_idx = who - start_row + blockDim.x;

        if(!myvar)
        {  

            if( (sh_idx >= 0) && (sh_idx < blockDim.x) && (is_solved_cache[sh_idx]==1) ){
                myvar = 1;
                left_sum -= my_val * x_cache[sh_idx];
            }
            // else if(who >= start_row  && who < start_row + WARP_PER_BLOCK){
            //     myvar = s_is_solved[who-start_row];
            //     if (myvar){
            //         left_sum -= my_val * s_x[who-start_row];
            //     }
            // }
            else           
            {

                myvar = is_solved[who];

                if (myvar){
                    left_sum -= my_val * x[who];
                }
            }
        } 

        if( __all(myvar) ){

            off+=WARP_SIZE;
            colidx = col_idx[off];
            //who = inv_iorder[colidx];
            my_val = val[off];

            sh_idx = colidx - start_row + blockDim.x;

            myvar=0;
        }
    }
    
    // Reduccion
    for (int i=16; i>=1; i/=2){
        left_sum += __shfl_down(left_sum, i);
    }
     
    if(lne==0){

        //escribo en el resultado
        
        //s_x[local_warp_id] = left_sum * piv;
        //s_is_solved[local_warp_id] = 1;
        
        //x[row_idx] = left_sum * piv;
        x[wrp] = left_sum * piv;

        __threadfence();

        //is_solved[row_idx] = 1;
        is_solved[wrp] = 1;


        // int sh_idx = wrp - start_row + blockDim.x;
        // if( (sh_idx >= 0) && (sh_idx < blockDim.x) && (is_solved_cache[sh_idx]==1) ){
        //     x_cache[sh_idx] = x[wrp];
        //     is_solved_cache[sh_idx] = 1;
        // }
    }
}

__global__ void sptrsv_kernel_opt23(const int* __restrict__ row_ptr, 
                              const int* __restrict__ col_idx, 
                              const VALUE_TYPE* __restrict__ val, 
                              const VALUE_TYPE* __restrict__ b, 
                              VALUE_TYPE* x, 
                              int * is_solved, int n, int * d_while_profiler) {
/*
    volatile __shared__ int        s_is_solved[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_x        [WARP_PER_BLOCK];
*/

    extern volatile __shared__ int s_mem[];       

    int WARP_PER_BLOCK = blockDim.x / WARP_SIZE;

    int * s_is_solved = (int *) &s_mem[0];
    VALUE_TYPE * s_x  = (VALUE_TYPE*) &s_is_solved[WARP_PER_BLOCK];

    int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
    int local_warp_id = threadIdx.x / WARP_SIZE;


    int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

    if(wrp >= n) return;
    
    int row = row_ptr[wrp];
    int start_row = blockIdx.x*WARP_PER_BLOCK;
    int nxt_row = row_ptr[wrp+1];
    int lock = 0;

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv = 1 / val[nxt_row-1];

/*    

    if(!local_warp_id && lne < WARP_PER_BLOCK){

        s_x[lne] = b[start_row+lne];
        s_is_solved[lne] = 0;
    }

*/
    if(lne==0){
        left_sum = b[wrp];
        s_is_solved[local_warp_id] = 0;
    }

    __syncthreads();

    int off = row+lne;
    int colidx = col_idx[off];

    VALUE_TYPE my_val = val[off];
    VALUE_TYPE xx;


    int myvar = 0; //(off >= nxt_row - 1);
    int prefetch = 1; 

    while(off < nxt_row - 1)
    {

        if(!myvar)
        {
            if(colidx > start_row){
                myvar = s_is_solved[colidx-start_row];

                if (myvar){ //prefetch){
                    xx = s_x[colidx-start_row];
                }
            }
            else
            {
                myvar = is_solved[colidx];

                if (myvar){ //prefetch){
                    xx = x[colidx];
                }

            }
        } 

        if(myvar && prefetch){
            left_sum -= my_val * xx; // * x[colidx]
            prefetch = 0;
        }

        lock = __all(myvar);

        if(lock){

            off+=WARP_SIZE;
            colidx = col_idx[off];
            my_val = val[off];

            myvar=0;
            prefetch = 1;
        }
    }
    
    // Reduccion
    for (int i=16; i>=1; i/=2)
        left_sum += __shfl_down(left_sum, i);
        //left_sum += __shfl_xor(left_sum, i, 32);
     
    if(lne==0){

        //left_sum += s_x[local_warp_id];

        //escribo en el resultado
        s_x[local_warp_id] = left_sum * piv;
        s_is_solved[local_warp_id] = 1;

        x[wrp] = left_sum * piv;

        __threadfence();

        is_solved[wrp] = 1;
    }
}







__global__ void sptrsv_kernel_dfr_analysis_solve(const int* __restrict__ row_ptr, 
                                          const int* __restrict__ col_idx, 
                                          const VALUE_TYPE* __restrict__ val, 
                                          const VALUE_TYPE* __restrict__ b, 
                                          VALUE_TYPE* x, 
                                          int * is_solved, int n, 
                                          int * dfr_analysis_info, 
                                          int * d_while_profiler) {
/*
    volatile __shared__ int        s_is_solved[WARP_PER_BLOCK];
    volatile __shared__ int        s_info     [WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_x        [WARP_PER_BLOCK];
*/

    extern volatile __shared__ int s_mem[];       

    int WARP_PER_BLOCK = blockDim.x / WARP_SIZE;

    int * s_is_solved = (int *) &s_mem[0];
    int * s_info      = (int *) &s_is_solved[WARP_PER_BLOCK];
    VALUE_TYPE * s_x  = (VALUE_TYPE*) &s_info[WARP_PER_BLOCK];

    int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
    int local_warp_id = threadIdx.x / WARP_SIZE;

    int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

    if(wrp >= n) return;
    
    int row = row_ptr[wrp];
    int start_row = blockIdx.x*WARP_PER_BLOCK;
    int nxt_row = row_ptr[wrp+1];

    int my_level = 0;

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv = 1 / val[nxt_row-1];

/*    

    if(!local_warp_id && lne < WARP_PER_BLOCK){

        s_x[lne] = b[start_row+lne];
        s_is_solved[lne] = 0;
    }

*/
    if(lne==0){
        left_sum = b[wrp];
        s_is_solved[local_warp_id] = 0;
        s_info[local_warp_id] = 0;
    }

    __syncthreads();

    int off = row+lne;
    int colidx = col_idx[off];

    VALUE_TYPE my_val = val[off];

    int myvar = 0;

    while(off < nxt_row - 1)
    {

        if(!myvar)
        {
            if(colidx > start_row){
                myvar = s_is_solved[colidx-start_row];

                if (myvar){
                    left_sum -= my_val * s_x[colidx-start_row];
                    my_level = max(my_level, s_info[colidx-start_row]);

                }
            }
            else
            {
                myvar = is_solved[colidx];

                if (myvar){
                    left_sum -= my_val * x[colidx];
                    my_level = max(my_level, dfr_analysis_info[colidx]);
                }

            }
        } 

        if( __all(myvar) ){


            off+=WARP_SIZE;
            colidx = col_idx[off];
            my_val = val[off];

            myvar=0;
        }
    }
    
    // Reduccion
    for (int i=16; i>=1; i/=2){
        left_sum += __shfl_down(left_sum, i);
        my_level = max(my_level, __shfl_down(my_level, i));
    }
     
    if(lne==0){

        //escribo en el resultado
        s_x[local_warp_id] = left_sum * piv;
        s_info[local_warp_id] = 1+my_level;
        s_is_solved[local_warp_id] = 1;

        x[wrp] = left_sum * piv;
        dfr_analysis_info[wrp] = 1+my_level;

        __threadfence();

        is_solved[wrp] = 1;
    }
}



__global__ void sptrsv_kernel_dfr_analysis(const int* __restrict__ row_ptr, 
                                          const int* __restrict__ col_idx, 
                                          int * is_solved, int n, 
                                          int * dfr_analysis_info, 
                                          int * d_while_profiler) {

/*
    volatile __shared__ int        s_is_solved[WARP_PER_BLOCK];
    volatile __shared__ int        s_info     [WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_x        [WARP_PER_BLOCK];
*/

    extern volatile __shared__ int s_mem[];       

    int WARP_PER_BLOCK = blockDim.x / WARP_SIZE;

    int * s_is_solved = (int *) &s_mem[0];
    int * s_info      = (int *) &s_is_solved[WARP_PER_BLOCK];

    int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
    int local_warp_id = threadIdx.x / WARP_SIZE;

    int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

    if(wrp >= n) return;
    
    int row = row_ptr[wrp];
    int start_row = blockIdx.x*WARP_PER_BLOCK;
    int nxt_row = row_ptr[wrp+1];

    int my_level = 0;

/*    

    if(!local_warp_id && lne < WARP_PER_BLOCK){

        s_x[lne] = b[start_row+lne];
        s_is_solved[lne] = 0;
    }

*/
    if(lne==0){
        s_is_solved[local_warp_id] = 0;
        s_info[local_warp_id] = 0;
    }

    __syncthreads();

    int off = row+lne;
    int colidx = col_idx[off];

    int myvar = 0;

    while(off < nxt_row - 1)
    {

        if(!myvar)
        {
            if(colidx > start_row){
                myvar = s_is_solved[colidx-start_row];

                if (myvar){
                    my_level = max(my_level, s_info[colidx-start_row]);
                }
            }
            else
            {
                myvar = is_solved[colidx];

                if (myvar){
                    my_level = max(my_level, dfr_analysis_info[colidx]);
                }
            }
        } 

        if( __all(myvar) ){

            off+=WARP_SIZE;
            colidx = col_idx[off];
            myvar=0;
        }
    }
    
    // Reduccion
    for (int i=16; i>=1; i/=2){
        my_level = max(my_level, __shfl_down(my_level, i));
    }
     
    if(lne==0){

        //escribo en el resultado
        s_info[local_warp_id] = 1+my_level;
        s_is_solved[local_warp_id] = 1;


        dfr_analysis_info[wrp] = 1+my_level;

        __threadfence();

        is_solved[wrp] = 1;
    }
}

__global__ void sptrsv_kernel_dfr_analysis2(const int* __restrict__ row_ptr, 
                                          const int* __restrict__ col_idx, 
                                          int * is_solved, int n, 
                                          int * dfr_analysis_info, 
                                          int * row_ctr) {

/*
    volatile __shared__ int        s_is_solved[WARP_PER_BLOCK];
    volatile __shared__ int        s_info     [WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_x        [WARP_PER_BLOCK];
*/

    extern volatile __shared__ int s_mem[];       

    int WARP_PER_BLOCK = blockDim.x / WARP_SIZE;

    int * s_is_solved = (int *) &s_mem[0];
    int * s_info      = (int *) &s_is_solved[WARP_PER_BLOCK];

    int * blIdx = (int *) &s_info[WARP_PER_BLOCK];

    if(threadIdx.x==0) blIdx[0] = atomicAdd(&(row_ctr[0]), 1) ; 

    __syncthreads();

    //printf("blockIdx.x = %d  row_ctr = %d\n", blockIdx.x, blIdx[0]);


//    int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
    int wrp = (threadIdx.x + blIdx[0] * blockDim.x) / WARP_SIZE;
    int local_warp_id = threadIdx.x / WARP_SIZE;

    int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

    if(wrp >= n) return;
    
    int row = row_ptr[wrp];
    //int start_row = blockIdx.x*WARP_PER_BLOCK;
    int start_row = blIdx[0]*WARP_PER_BLOCK;
    int nxt_row = row_ptr[wrp+1];

    int my_level = 0;

/*    

    if(!local_warp_id && lne < WARP_PER_BLOCK){

        s_x[lne] = b[start_row+lne];
        s_is_solved[lne] = 0;
    }

*/
    if(lne==0){
        s_is_solved[local_warp_id] = 0;
        s_info[local_warp_id] = 0;
    }

    __syncthreads();

    int off = row+lne;
    int colidx = col_idx[off];

    int myvar = 0;

    while(off < nxt_row - 1)
    {

        if(!myvar)
        {
            if(colidx > start_row){
                myvar = s_is_solved[colidx-start_row];

                if (myvar){
                    my_level = max(my_level, s_info[colidx-start_row]);
                }
            }
            else
            {
                myvar = is_solved[colidx];

                if (myvar){
                    my_level = max(my_level, dfr_analysis_info[colidx]);
                }
            }
        } 

        if( __all(myvar) ){

            off+=WARP_SIZE;
            colidx = col_idx[off];
            myvar=0;
        }
    }
    
    // Reduccion
    for (int i=16; i>=1; i/=2){
        my_level = max(my_level, __shfl_down(my_level, i));
    }
     
    if(lne==0){

        //escribo en el resultado
        s_info[local_warp_id] = 1+my_level;
        s_is_solved[local_warp_id] = 1;


        dfr_analysis_info[wrp] = 1+my_level;

        __threadfence();

        is_solved[wrp] = 1;
    }
}

__global__
void spts_syncfree_cuda_analyser(const int   *d_cscRowIdx,
                                 const int    m,
                                 const int    nnz,
                                       int   *d_csrRowHisto)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < nnz)
    {
        atomicAdd(&d_csrRowHisto[d_cscRowIdx[global_id]], 1);
    }
}

__global__
void spts_syncfree_cuda_executor_pre(const int   *d_csrRowPtrL,
                                     const int    m,
                                           int   *d_csrRowHisto)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < m)
    {
        d_csrRowHisto[global_id] = d_csrRowPtrL[global_id+1] - d_csrRowPtrL[global_id];
    }
}

__global__
void spts_syncfree_cuda_executor(const int* __restrict__        d_cscColPtr,
                                 const int* __restrict__        d_cscRowIdx,
                                 const VALUE_TYPE* __restrict__ d_cscVal,
                                 const int* __restrict__        d_csrRowPtr,
                                 int*                           d_csrRowHisto,
                                 VALUE_TYPE*                    d_left_sum,
                                 VALUE_TYPE*                    d_partial_sum,
                                 const int                      m,
                                 const int                      nnz,
                                 const VALUE_TYPE* __restrict__ d_b,
                                 VALUE_TYPE*                    d_x,
                                 int*                           d_while_profiler)

{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_x_id = global_id / WARP_SIZE;



/*
    volatile __shared__ int s_csrRowHisto[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK];
*/

    extern volatile __shared__ int s_mem[];       

    int WARP_PER_BLOCK = blockDim.x / WARP_SIZE;

    int * s_csrRowHisto = (int *) &s_mem[0];
    VALUE_TYPE * s_left_sum  = (VALUE_TYPE*) &s_csrRowHisto[WARP_PER_BLOCK];



    if (global_x_id >= m) return;
    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int starting_x = (global_id / (WARP_PER_BLOCK * WARP_SIZE)) * WARP_PER_BLOCK;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    // Prefetch
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[d_cscColPtr[global_x_id]];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    if (threadIdx.x < WARP_PER_BLOCK) { s_csrRowHisto[threadIdx.x] = 1; s_left_sum[threadIdx.x] = 0; }
    __syncthreads();

    clock_t start;
    // Consumer
    do {
        start = clock();
    }
    while (s_csrRowHisto[local_warp_id] != d_csrRowHisto[global_x_id]);
  
    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_csrRowHisto[global_x_id]) :: "memory"); 
    //}
    //while (s_csrRowHisto[local_warp_id] != graphInDegree );

    VALUE_TYPE xi = d_left_sum[global_x_id] + s_left_sum[local_warp_id]; 
    xi = (d_b[global_x_id] - xi) * coef;

    // Producer
    for (int j = d_cscColPtr[global_x_id] + 1 + lane_id; j < d_cscColPtr[global_x_id+1]; j += WARP_SIZE) {   
        int rowIdx = d_cscRowIdx[j];
        if (rowIdx < starting_x + WARP_PER_BLOCK) {
            atomicAdd((VALUE_TYPE *)&s_left_sum[rowIdx - starting_x], xi * d_cscVal[j]);
            atomicAdd((int *)&s_csrRowHisto[rowIdx - starting_x], 1);
        }
        else {
            atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
            atomicSub(&d_csrRowHisto[rowIdx], 1);
        }
    }
    // Finish
    if (!lane_id) d_x[global_x_id] = xi;
}

int spts_syncfree_cuda( int           *csrRowPtrL_tmp,
                          int           *csrColIdxL_tmp,
                          VALUE_TYPE    *csrValL_tmp,
                          const int            m,
                          const int            n,
                          const int            nnzL,
                          const char * filename, const int WARP_PER_BLOCK)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }



    printf("Inicio... M=%i, N=%i, NNZL=%i!\n",m,n,nnzL);

    VALUE_TYPE *x_ref = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);
    for ( int i = 0; i < n; i++)
        x_ref[i] = 1;

    VALUE_TYPE *b = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m);

    for (int i = 0; i < m; i++)
    {
        b[i] = 0;
        for (int j = csrRowPtrL_tmp[i]; j < csrRowPtrL_tmp[i+1]; j++)
            b[i] += csrValL_tmp[j] * x_ref[csrColIdxL_tmp[j]];
    }

    VALUE_TYPE *x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);

    // transpose from csr to csc first
    int *cscRowIdxL = (int *)malloc(nnzL * sizeof(int));
    int *cscColPtrL = (int *)malloc((n+1) * sizeof(int));
    memset(cscColPtrL, 0, (n+1) * sizeof(int));
    VALUE_TYPE *cscValL    = (VALUE_TYPE *)malloc(nnzL * sizeof(VALUE_TYPE));

    matrix_transposition(m, n, nnzL,
                         csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp,
                         cscRowIdxL, cscColPtrL, cscValL);

    // transfer host mem to device mem
    int *d_cscColPtrL;
    int *d_cscRowIdxL;
    VALUE_TYPE *d_cscValL;

    int *d_csrColIdxL;
    int *d_csrRowPtrL;
    VALUE_TYPE *d_csrValL;


    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;

    // Matrix L en CSC
    cudaMalloc((void **)&d_cscColPtrL, (n+1) * sizeof(int));
    cudaMalloc((void **)&d_cscRowIdxL, nnzL  * sizeof(int));
    cudaMalloc((void **)&d_cscValL,    nnzL  * sizeof(VALUE_TYPE));

    cudaMemcpy(d_cscColPtrL, cscColPtrL, (n+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIdxL, cscRowIdxL, nnzL  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscValL,    cscValL,    nnzL  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);

    // Vector b
    cudaMalloc((void **)&d_b, m * sizeof(VALUE_TYPE));
    cudaMemcpy(d_b, b, m * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    // Vector x
    cudaMalloc((void **)&d_x, n  * sizeof(VALUE_TYPE));
    cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE));

    //  - cuda syncfree SpTS analysis start!
    //printf(" - cuda syncfree SpTS analysis start!\n");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    // malloc tmp memory to simulate atomic operations
    int *d_csrRowHisto;
    cudaMalloc((void **)&d_csrRowHisto, sizeof(int) * (m+1));

    // generate row pointer by partial transposition
    //int *d_csrRowPtrL;
    cudaMalloc((void **)&d_csrRowPtrL, (m+1) * sizeof(int));
    thrust::device_ptr<int> d_csrRowPtrL_thrust = thrust::device_pointer_cast(d_csrRowPtrL);
    thrust::device_ptr<int> d_csrRowHisto_thrust = thrust::device_pointer_cast(d_csrRowHisto);

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *d_left_sum;
    cudaMalloc((void **)&d_left_sum, sizeof(VALUE_TYPE) * m);

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *d_partial_sum;
    cudaMalloc((void **)&d_partial_sum, sizeof(VALUE_TYPE) * nnzL);
    //cudaMemset(d_partial_sum, 0, sizeof(VALUE_TYPE) * nnzL);

    int num_threads = WARP_PER_BLOCK*WARP_SIZE; //256;
    int num_blocks = ceil ((double)nnzL / (double)num_threads);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        cudaMemset(d_csrRowHisto, 0, (m+1) * sizeof(int));
 
 //      spts_syncfree_cuda_analyser<<< num_blocks, num_threads >>>
//                                      (d_cscRowIdxL, m, nnzL, d_csrRowHisto);
  
    }
    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);
    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_analysis /= BENCH_REPEAT;

    thrust::exclusive_scan(d_csrRowHisto_thrust, d_csrRowHisto_thrust + m+1, d_csrRowPtrL_thrust);

 //   printf("LIU analysis %4.2f ms\n", time_cuda_analysis);

    // validate csrRowPtrL
    int *csrRowPtrL = (int *)malloc((m+1) * sizeof(int));
    cudaMemcpy(csrRowPtrL, d_csrRowPtrL, (m+1) * sizeof(int), cudaMemcpyDeviceToHost);

    int err_counter = 0;
    for (int i = 0; i <= m; i++)
    {
        //printf("[%i]: csrRowPtrL = %i, csrRowPtrL_tmp = %i\n", i, csrRowPtrL[i], csrRowPtrL_tmp[i]);
        if (csrRowPtrL[i] != csrRowPtrL_tmp[i])
            err_counter++;
    }

    free(csrRowPtrL);

   // if (!err_counter)
    //    printf("LIU analysis test passed!\n");
   // else
  //      printf("LIU analysis test failed!\n");
    
//  - cuda syncfree SpTS solve start!
    // printf(" - cuda syncfree SpTS solve start!\n");

    int *d_while_profiler;
    cudaMalloc((void **)&d_while_profiler, sizeof(int) * n);
    cudaMemset(d_while_profiler, 0, sizeof(int) * n);
    int *while_profiler = (int *)malloc(sizeof(int) * n);

    // step 5: solve L*y = x
    double time_cuda_solve = 0;
/*
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        num_threads = WARP_PER_BLOCK*WARP_SIZE; //256;
        num_blocks = ceil ((double)m / (double)(num_threads));

        spts_syncfree_cuda_executor_pre<<< num_blocks, num_threads >>>
                                          (d_csrRowPtrL, m, d_csrRowHisto);
       
        gettimeofday(&t1, NULL);

        cudaMemset(d_left_sum, 0, sizeof(VALUE_TYPE) * m);

        num_threads = WARP_PER_BLOCK * WARP_SIZE;
        num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));

        spts_syncfree_cuda_executor<<< num_blocks, num_threads, WARP_PER_BLOCK * (sizeof(int)+sizeof(VALUE_TYPE)) >>>
                                   (d_cscColPtrL, d_cscRowIdxL, d_cscValL, 
                                    d_csrRowPtrL, d_csrRowHisto, 
                                    d_left_sum, d_partial_sum,
                                    m, nnzL, d_b, d_x, d_while_profiler);

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    }
    cudaDeviceSynchronize();

    time_cuda_solve /= BENCH_REPEAT;

    printf("LIU solve %4.2f ms, %4.2f gflops\n",
           time_cuda_solve, 2*nnzL/(1e6*time_cuda_solve));

    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    // validate x
    int ec_liu = validate_x(x_ref,x,n, "LIU");
*/
    cudaFree(d_cscColPtrL);
    cudaFree(d_cscRowIdxL);
    cudaFree(d_cscValL);
    cudaFree(d_csrRowHisto);
    cudaFree(d_left_sum);
    cudaFree(d_partial_sum);

//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################

    // Matrix L en CSR
    cudaMalloc((void **)&d_csrRowPtrL, (n+1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdxL, nnzL  * sizeof(int));
    cudaMalloc((void **)&d_csrValL,    nnzL  * sizeof(VALUE_TYPE));

    cudaMemcpy(d_csrRowPtrL, csrRowPtrL_tmp, (n+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdxL, csrColIdxL_tmp, nnzL  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValL,    csrValL_tmp,    nnzL  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);

//step 6: cusparse

    cusparseHandle_t cusp_handle;
    cusparseMatDescr_t desc_L;
    cusparseSolveAnalysisInfo_t info_L;
 
    cusparseCreate(&cusp_handle);

    cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE));

    cusparseCreateMatDescr(&desc_L);
    cusparseCreateSolveAnalysisInfo(&info_L);

    // defino la matriz L como triangular inferior...
    cusparseSetMatIndexBase(desc_L, CUSPARSE_INDEX_BASE_ZERO);
//    cusparseSetMatType(desc_L, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
    cusparseSetMatType(desc_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(desc_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(desc_L, CUSPARSE_DIAG_TYPE_NON_UNIT);

    VALUE_TYPE alpha = 1.0;

// step 6.1: csrvs2

    csrsv2Info_t info = 0;

    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseStatus_t status;



    // step 2: create a empty info structure
    cusparseCreateCsrsv2Info(&info);

    double time_cuda_analysis_cusparse2 = 0;
    double time_cuda_solve_cusparse2_level = 0;
    double time_cuda_solve_cusparse2_nolevel = 0;

    cudaMemset(d_x, 7, n * sizeof(VALUE_TYPE));

    gettimeofday(&t1, NULL);

    // step 3: query how much memory used in csrsv2, and allocate the buffer
    #ifdef __double__
        CUSP_CHK( cusparseDcsrsv2_bufferSize(cusp_handle, trans, n, nnzL, desc_L, d_csrValL, d_csrRowPtrL, d_csrColIdxL, info, &pBufferSize) );
    #else
        CUSP_CHK( cusparseScsrsv2_bufferSize(cusp_handle, trans, n, nnzL, desc_L, d_csrValL, d_csrRowPtrL, d_csrColIdxL, info, &pBufferSize) );
    #endif

    printf("tamaño de pBuffer = %i \n", pBufferSize );

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    CUDA_CHK(cudaMalloc((void**)&pBuffer, pBufferSize));

    // step 4: perform analysis
    #ifdef __double__
        CUSP_CHK( cusparseDcsrsv2_analysis(cusp_handle, trans, n, nnzL, desc_L, d_csrValL, d_csrRowPtrL, d_csrColIdxL, info, policy, pBuffer) );
    #else
        CUSP_CHK( cusparseScsrsv2_analysis(cusp_handle, trans, n, nnzL, desc_L, d_csrValL, d_csrRowPtrL, d_csrColIdxL, info, policy, pBuffer) );
    #endif  

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    time_cuda_analysis_cusparse2 = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    // step 5: solve L*y = x

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        gettimeofday(&t1, NULL);
        #ifdef __double__
            CUSP_CHK( cusparseDcsrsv2_solve(cusp_handle, trans, n, nnzL, &alpha, desc_L,
                                d_csrValL, d_csrRowPtrL, d_csrColIdxL, info,
                                d_b, d_x, policy, pBuffer));
        #else
            CUSP_CHK( cusparseScsrsv2_solve(cusp_handle, trans, n, nnzL, &alpha, desc_L,
                                d_csrValL, d_csrRowPtrL, d_csrColIdxL, info,
                                d_b, d_x, policy, pBuffer));
        #endif      

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve_cusparse2_level += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    time_cuda_solve_cusparse2_level /= BENCH_REPEAT;

    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    // validate x
    int ec_cusp2 = validate_x(x_ref,x,n,"CUSPARSE2");


//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################

    double time_cuda_analysis_cusparse = 0;
    double time_cuda_solve_cusparse = 0;

    gettimeofday(&t1, NULL);

    #ifdef __double__
    cusparseDcsrsv_analysis(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        n, nnzL , desc_L,
                                        d_csrValL, d_csrRowPtrL,
                                        d_csrColIdxL, info_L);
    #else
    cusparseScsrsv_analysis(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        n, nnzL , desc_L,
                                        d_csrValL, d_csrRowPtrL,
                                        d_csrColIdxL, info_L);
    #endif
    gettimeofday(&t2, NULL);
    time_cuda_analysis_cusparse = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

	int nLevs, *levPtr, *levInd;

    cusparseGetLevelInfo(cusp_handle, info_L, &nLevs, &levPtr, &levInd);
/*

	double avg =  nnzL/n;

	double std_dev = 0;
	int row_max = 	csrRowPtrL_tmp[1]-csrRowPtrL_tmp[0] , row_min =	csrRowPtrL_tmp[1]-csrRowPtrL_tmp[0];

	for (int ii = 1; ii <= n;++ii)
	{
		double tmp =	csrRowPtrL_tmp[ii]-csrRowPtrL_tmp[ii-1] ;
		if(row_max<tmp)row_max=tmp;
		if(row_min>tmp)row_min=tmp;
		
		tmp -= avg;
		std_dev += (tmp*tmp)/n;
	}


    printf("LEVINFO: %s %i %i %i %f %f %i %i \n", filename, n, nnzL, nLevs, avg, sqrt( std_dev), row_max, row_min);
*/
	//return 0;

    printf("CUSPARSE analysis %4.2f ms, %4.2f gflops\n", time_cuda_analysis_cusparse, 2*nnzL/(1e6*time_cuda_analysis_cusparse));

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        
        gettimeofday(&t1, NULL);

        #ifdef __double__
        cusparseDcsrsv_solve(cusp_handle , CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                n, &alpha, desc_L,
                                                d_csrValL, d_csrRowPtrL,
                                                d_csrColIdxL, info_L,
                                                d_b, d_x );
        #else
        cusparseScsrsv_solve(cusp_handle , CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                n, &alpha, desc_L,
                                                d_csrValL, d_csrRowPtrL,
                                                d_csrColIdxL, info_L,
                                                d_b, d_x );
        #endif

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve_cusparse += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    }
    cudaDeviceSynchronize();

    time_cuda_solve_cusparse /= BENCH_REPEAT;

    printf("CUSPARSE solve %4.2f ms, %4.2f gflops\n", time_cuda_solve_cusparse, 2*nnzL/(1e6*time_cuda_solve_cusparse));

    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    // validate x
    int ec_cusp = validate_x(x_ref,x,n,"CUSPARSE");

    cusparseDestroySolveAnalysisInfo(info_L);


//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################



//step 7: versión dufre
    double time_cuda_solve_dfr = 0;

    int * d_is_solved;
    int * dfr_analysis_info;
    int * d_dfr_analysis_info;

    cudaMalloc((void **)&d_is_solved, n * sizeof(int));
    cudaMalloc((void **)&d_dfr_analysis_info, n * sizeof(int));
    cudaMemset(d_x, 7, n * sizeof(VALUE_TYPE));

    dfr_analysis_info = (int*) malloc(n * sizeof(int));
    
    for (int i = 0; i < BENCH_REPEAT; i++)
    {

        for (int i = 0; i < n; i++) x[i] = 0;
        
        gettimeofday(&t1, NULL);

        num_threads = WARP_PER_BLOCK*WARP_SIZE;

        int grid = ceil ((double)n*WARP_SIZE / (double)(num_threads*ROWS_PER_THREAD));
        //int grid = ceil ((double)n*WARP_SIZE / (double)(num_threads));
        //num_blocks = n * WARP_SIZE / num_threads + 1;
	
	    cudaMemset(d_is_solved, 0, n * sizeof(int));
        cudaMemset(d_dfr_analysis_info, 0, n * sizeof(int));

//        sptrsv_kernel_opt3<<< grid , num_threads >>>(d_csrRowPtrL,d_csrColIdxL,d_csrValL, d_b, d_x, d_is_solved, n, d_while_profiler);
//        sptrsv_kernel_opt23<<< grid , num_threads >>>(d_csrRowPtrL,d_csrColIdxL,d_csrValL, d_b, d_x, d_is_solved, n, d_while_profiler);
//        sptrsv_kernel_opt22<<< grid , num_threads >>>(d_csrRowPtrL,d_csrColIdxL,d_csrValL, d_b, d_x, d_is_solved, n, d_while_profiler);
//        sptrsv_kernel_opt21<<< grid , num_threads >>>(d_csrRowPtrL,d_csrColIdxL,d_csrValL, d_b, d_x, d_is_solved, n, d_while_profiler);
//        sptrsv_kernel_opt2<<< grid , num_threads >>>(d_csrRowPtrL,d_csrColIdxL,d_csrValL, d_b, d_x, d_is_solved, n, d_while_profiler);
//        sptrsv_kernel_opt<<< grid , num_threads >>>(d_csrRowPtrL,d_csrColIdxL,d_csrValL, d_b, d_x, d_is_solved, n, d_while_profiler);
//        sptrsv_kernel<<< grid , num_threads >>>(d_csrRowPtrL,d_csrColIdxL,d_csrValL, d_b, d_x, d_is_solved, n);

        sptrsv_kernel_dfr_analysis_solve<<< grid , num_threads, WARP_PER_BLOCK * (2*sizeof(int)+sizeof(VALUE_TYPE)) >>>(d_csrRowPtrL,d_csrColIdxL,d_csrValL, d_b, d_x, d_is_solved, n, d_dfr_analysis_info, d_while_profiler);

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve_dfr += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    }
    cudaDeviceSynchronize();
    time_cuda_solve_dfr /= BENCH_REPEAT;

    printf("DUFRE A+S %4.2f ms, %4.2f gflops\n", time_cuda_solve_dfr, 2*nnzL/(1e6*time_cuda_solve_dfr));

    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(dfr_analysis_info, d_dfr_analysis_info, n * sizeof(int), cudaMemcpyDeviceToHost);

    // validate x
    int ec_dfr_as = validate_x(x_ref,x,n,"DUFRE A+S");

    int nLevs_dfr = dfr_analysis_info[0];
    for (int i = 1; i < n; ++i)
    {
        nLevs_dfr = max(nLevs_dfr, dfr_analysis_info[i]);
    }

    printf("CUSPARSE Niveles = %d, Tiempo analysis=%f\n", nLevs, time_cuda_analysis_cusparse);
    printf("DUFRE Niveles = %d \n", nLevs_dfr);

/*
    cudaMemcpy(while_profiler, d_while_profiler, n * sizeof(int), cudaMemcpyDeviceToHost);
    long long unsigned int while_count = 0;
    for (int i = 0; i < n; i++)
    {
        while_count += while_profiler[i];
        //printf("while_profiler[%i] = %i\n", i, while_profiler[i]);
    }
    //printf("\nwhile_count= %llu in total, %llu per row/column\n", while_count, while_count/m);
*/


//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################

    double time_cuda_solve_dfr_a = 0;

    for (int i = 0; i < BENCH_REPEAT; i++)
    {

        for (int i = 0; i < n; i++) x[i] = 0;
        
        gettimeofday(&t1, NULL);

        num_threads = WARP_PER_BLOCK*WARP_SIZE;

        int grid = ceil ((double)n*WARP_SIZE / (double)(num_threads*ROWS_PER_THREAD));
    
        cudaMemset(d_is_solved, 0, n * sizeof(int));
        cudaMemset(d_dfr_analysis_info, 0, n * sizeof(int));


        sptrsv_kernel_dfr_analysis<<< grid , num_threads, WARP_PER_BLOCK * (2*sizeof(int)) >>>(d_csrRowPtrL,d_csrColIdxL, d_is_solved, n, d_dfr_analysis_info, d_while_profiler);

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve_dfr_a += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    }
    cudaDeviceSynchronize();
    time_cuda_solve_dfr_a /= BENCH_REPEAT;


    cudaMemcpy(dfr_analysis_info, d_dfr_analysis_info, n * sizeof(int), cudaMemcpyDeviceToHost);

/*
Given an array idepth of length n containing the depth
of each node, obtain the maximum, which is equal to
the total amount of levels. 
*/
    nLevs_dfr = dfr_analysis_info[0];
    for (int i = 1; i < n; ++i)
    {
        nLevs_dfr = max(nLevs_dfr, dfr_analysis_info[i]);
    }

    printf("DUFRE A %4.2f ms, %4.2f gflops Niveles = %d\n", time_cuda_solve_dfr_a, 2*nnzL/(1e6*time_cuda_solve_dfr_a), nLevs_dfr);

//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################

    memset(dfr_analysis_info, 0, n * sizeof(int));
    nLevs_dfr = 0;
    int izero = 0;
    int * row_ctr;

    cudaMalloc(&row_ctr,sizeof(int));

    double time_cuda_solve_dfr_a2 = 0;

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        for (int i = 0; i < n; i++) x[i] = 0;
        
        gettimeofday(&t1, NULL);

        num_threads = WARP_PER_BLOCK*WARP_SIZE;

        int grid = ceil ((double)n*WARP_SIZE / (double)(num_threads*ROWS_PER_THREAD));
    
        cudaMemset(d_is_solved, 0, n * sizeof(int));
        cudaMemset(d_dfr_analysis_info, 0, n * sizeof(int));
        cudaMemset(row_ctr, 0, sizeof(int));
        //cudaMemcpyToSymbol("row_ctr",&izero,sizeof(int),0,cudaMemcpyHostToDevice);

        sptrsv_kernel_dfr_analysis2<<< grid , num_threads, WARP_PER_BLOCK * (2*sizeof(int)) + sizeof(int) >>>(d_csrRowPtrL,d_csrColIdxL, d_is_solved, n, d_dfr_analysis_info, row_ctr);

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve_dfr_a2 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    }
    cudaDeviceSynchronize();
    time_cuda_solve_dfr_a2 /= BENCH_REPEAT;

    cudaMemcpy(dfr_analysis_info, d_dfr_analysis_info, n * sizeof(int), cudaMemcpyDeviceToHost);

/*
Given an array idepth of length n containing the depth
of each node, obtain the maximum, which is equal to
the total amount of levels. 
*/
    nLevs_dfr = dfr_analysis_info[0];
    for (int i = 1; i < n; ++i)
    {
        nLevs_dfr = max(nLevs_dfr, dfr_analysis_info[i]);
    }

    printf("DUFRE A2 %4.2f ms, %4.2f gflops Niveles = %d\n", time_cuda_solve_dfr_a2, 2*nnzL/(1e6*time_cuda_solve_dfr_a2), nLevs_dfr);



//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################




/*

    int test_analysis[15] = {0,1,2,3,2,0,3,3,4,4,1,2,4,2,3};
    int test_nLevs = 5;

    int * ilevels = (int *) calloc(test_nLevs,sizeof(int));
    int * iorder  = (int *) calloc(15,sizeof(int));

    for(int i = 0; i< 15; i++ )
        ilevels[test_analysis[i]]++;

    printf("Cant en cada nivel\n");
    for(int i = 0; i< test_nLevs; i++ ) printf("%d -> %d\n", i, ilevels[i] );

    exclusive_scan(ilevels, test_nLevs );

    printf("offset\n");
    for(int i = 0; i< test_nLevs; i++ ) printf("%d -> %d\n", i, ilevels[i] );


    for(int i = 0; i< 15; i++ ){
        int idepth = test_analysis[i];
        iorder[ ilevels[ idepth ]++ ] = i;
    }

    printf("Orden de prueba\n");
    for(int i = 0; i< 15; i++ ) printf("%d (%d) -> %d\n", i, test_analysis[ iorder[i] ], iorder[i] );

    int ec_dfr_order = 0;

/*/

// Allocate the vector ilevels and initialize it such that
// ilevels(i) contains the number of nodes in level i.

    int * ilevels = (int *) calloc(6 * nLevs_dfr,sizeof(int));
    int * iorder  = (int *) calloc(n,sizeof(int));
    int * ivect_size  = (int *) calloc(n,sizeof(int));
//    int * inv_iorder  = (int *) calloc(n,sizeof(int));

    for(int i = 0; i< n; i++ ){

        int lev = dfr_analysis_info[i]-1;
        int nnz_row = csrRowPtrL_tmp[i+1]-csrRowPtrL_tmp[i];
        int vect_size;

        if (nnz_row <= 1)
            vect_size = 0;
        else if (nnz_row <= 2)
            vect_size = 1;
        else if (nnz_row <= 4)
            vect_size = 2;
        else if (nnz_row <= 8)
            vect_size = 3;
        else if (nnz_row <= 16)
            vect_size = 4;
        else vect_size = 5;

        ilevels[6*lev+vect_size]++;
    }

    // int * ibase_row, *ivect_size;
    // int * d_ibase_row, *d_ivect_size;



    // int * d_ilevels;
    // CUDA_CHK( cudaMalloc( (void**) &d_ilevels, nLevs_dfr * sizeof(int) ) );
    // CUDA_CHK( cudaMemcpy( d_ilevels, ilevels, nLevs_dfr * sizeof(int), cudaMemcpyHostToDevice ) );

// Performing a scan operation on this vector will yield
// the starting position of each level in the final iorder
// array, which is the final content of ilevels.

    exclusive_scan(ilevels, 6 * nLevs_dfr);


// Maintaining an offset variable for each level, assign
// each node j to the iorder array the following way
// iorder(ilevels(idepth(j)) + offset(idepth(j))) = j
// incrementing the offset by 1 afterwards.

    for(int i = 0; i< n; i++ ){
        
        int idepth = dfr_analysis_info[i]-1;
        int nnz_row = csrRowPtrL_tmp[i+1]-csrRowPtrL_tmp[i];
        int vect_size;

        if (nnz_row <= 1)
            vect_size = 0;
        else if (nnz_row <= 2)
            vect_size = 1;
        else if (nnz_row <= 4)
            vect_size = 2;
        else if (nnz_row <= 8)
            vect_size = 3;
        else if (nnz_row <= 16)
            vect_size = 4;
        else vect_size = 5;

//csrRowPtrL_tmp[ iorder[i]+1 ] - csrRowPtrL_tmp[ iorder[i] ]

        iorder[ ilevels[ 6*idepth+vect_size ] ] = i;             
        ivect_size[ ilevels[ 6*idepth+vect_size ] ] = pow(2,vect_size);        

//        inv_iorder[ i ] = ilevels[ 6*idepth+vect_size ];

        ilevels[ 6*idepth+vect_size ]++;

        // int idepth = dfr_analysis_info[i]-1;
        // iorder[ i ] = i;        
        // inv_iorder[ i ] = i;
    }

    int ii = 0;
    int filas_warp = 0;
    ii++;
    for (int ctr = 0; ctr < n; ++ctr)
    {
        filas_warp++;

        if( dfr_analysis_info[iorder[ctr]]!=dfr_analysis_info[iorder[ctr-1]] ||
            ivect_size[ctr]!=ivect_size[ctr-1] ||
            filas_warp * ivect_size[ctr] > 32 ){

            // ibase_row[ii] = ctr;
            filas_warp = 1;
            ii++;
        }
    }

    int n_warps = ii;
    int * ibase_row = (int*) calloc(n_warps,sizeof(int));
    int * ivect_size_warp = (int*) calloc(n_warps,sizeof(int));

    ii = 1;
    filas_warp = 0;
    ivect_size_warp[0]=ivect_size[0];

    int ctr;
    for (ctr = 0; ctr < n; ++ctr)
    {
        filas_warp++;

        if( dfr_analysis_info[iorder[ctr]]!=dfr_analysis_info[iorder[ctr-1]] ||
            ivect_size[ctr]!=ivect_size[ctr-1] ||
            filas_warp * ivect_size[ctr] > 32 ){

            ibase_row[ii] = ctr;
            ivect_size_warp[ii]=ivect_size[ctr];
            filas_warp = 1;
            ii++;
        }
    }
    ibase_row[ii] = ctr;
    ivect_size_warp[ii]=ivect_size[ctr];


    for (int i = 0; i < n_warps; ++i)
    {
        if(ivect_size_warp[i]<=0){
            printf("ivect_size_warp[%d] <= 0!!!!\n", ivect_size_warp[i]);
            exit(0);
        }

        if(ivect_size_warp[i]>32){
            printf("ivect_size_warp[%d] > 32!!!!\n", ivect_size_warp[i]);
            exit(0);
        }
    }


    printf("Orden primeros 100\n");
    for(int i = 0; i < 760; i++ ) printf("i=%d iorder=%d level=%d nnz_row=%d vect_size=%d ibase_row=%d\n ", i, 
                                                                                                            iorder[i], 
                                                                                                            dfr_analysis_info[iorder[i]], 
                                                                                                            csrRowPtrL_tmp[ iorder[i]+1 ] - csrRowPtrL_tmp[ iorder[i] ], 
                                                                                                            ivect_size[ i ], 
                                                                                                            ibase_row[i]);

    // exit(0);

    // int * d_inv_iorder;
    int * d_iorder;
    int * d_ibase_row;
    int * d_ivect_size_warp;


    //CUDA_CHK( cudaMalloc( (void**) &d_inv_iorder, n * sizeof(int) ) );
    //CUDA_CHK( cudaMemcpy( d_inv_iorder, inv_iorder, n * sizeof(int), cudaMemcpyHostToDevice ) );

    CUDA_CHK( cudaMalloc( (void**) &d_iorder, n * sizeof(int) ) );
    CUDA_CHK( cudaMemcpy( d_iorder, iorder, n * sizeof(int), cudaMemcpyHostToDevice ) );

    CUDA_CHK( cudaMalloc( (void**) &d_ibase_row, n_warps * sizeof(int) ) );
    CUDA_CHK( cudaMemcpy( d_ibase_row, ibase_row, n_warps * sizeof(int), cudaMemcpyHostToDevice ) );

    CUDA_CHK( cudaMalloc( (void**) &d_ivect_size_warp, n_warps * sizeof(int) ) );
    CUDA_CHK( cudaMemcpy( d_ivect_size_warp, ivect_size_warp, n_warps * sizeof(int), cudaMemcpyHostToDevice ) );

    int grid;

    double time_cuda_solve_dfr_order = 0;
    int * d_row_ctr;

    CUDA_CHK( cudaMalloc( (void**) &d_row_ctr, sizeof(int) ) );

    struct spts_times * kernel_times, * d_kernel_times;

    kernel_times  = (struct spts_times *) malloc(n_warps * sizeof(struct spts_times));
    CUDA_CHK( cudaMalloc( (void**) &d_kernel_times, n_warps * sizeof(struct spts_times) ) );


    CUDA_CHK( cudaMalloc( (void**) &d_row_ctr, sizeof(int) ) );
   
    for (int i = 0; i < BENCH_REPEAT; i++)
    {

        gettimeofday(&t1, NULL);

        num_threads = WARP_PER_BLOCK*WARP_SIZE;
        grid = ceil ((double)n_warps*WARP_SIZE / (double) num_threads);
    
        cudaMemset(d_is_solved, 0, n * sizeof(int));
        cudaMemset(d_row_ctr, 0, sizeof(int));

        // CUDA_CHK( cudaMemcpy( d_x, x, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice ) );

        forward_csr_L_solve_kernel_multirow<<< grid , num_threads >>>( d_csrRowPtrL, d_csrColIdxL, d_csrValL, 
                                                                       d_b, d_x, d_is_solved, 
                                                                       d_iorder, d_ibase_row, d_ivect_size_warp, 
                                                                       d_row_ctr, n, n_warps, d_kernel_times);

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve_dfr_order += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    cudaDeviceSynchronize();
    time_cuda_solve_dfr_order /= BENCH_REPEAT;

    printf("DUFRE ORDER %4.2f ms, %4.2f gflops\n", time_cuda_solve_dfr_order, 2*nnzL/(1e6*time_cuda_solve_dfr_order));

    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    cudaMemcpy(kernel_times, d_kernel_times, n_warps * sizeof(struct spts_times), cudaMemcpyDeviceToHost);

    float av_t_ini, av_t_wait, av_t_end;
    for (int i = 0; i < n_warps; ++i)
    {
        av_t_ini += kernel_times[i].ticks_ini / n_warps;
        av_t_wait += kernel_times[i].ticks_wait / n_warps;
        av_t_end += kernel_times[i].ticks_end / n_warps;  
    }

    float total_ticks = av_t_ini + av_t_wait + av_t_end;

    printf("ini %.2f %.2f%\n wait %.2f %.2f%\n end %.2f %.2f%\n", av_t_ini, (av_t_ini/total_ticks)*100,
                                                                  av_t_wait, (av_t_wait/total_ticks)*100,
                                                                  av_t_end, (av_t_end/total_ticks)*100);

    // for (int i = 0; i < 100; ++i)
    // {
    //     printf("x[%d]=%f\n", iorder[i], x[iorder[i]]);
    // }

    // validate x
    int ec_dfr_order = validate_x(x_ref,x,n,"DUFRE ORDER");


//*/


//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################
//########################################################################################################################################################################




    //if(!ec_liu && !ec_cusp && !ec_dfr_as && !ec_cusp2){
    if(!ec_cusp && !ec_dfr_as && !ec_dfr_order){
        printf("ALL tests passed: writing result!\n");
    
        FILE *ftabla;       
        ftabla = fopen("resultados.txt","a+");
/*
        fprintf(ftabla,"%s & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",
                                    filename,
                                    time_cuda_analysis_cusparse, time_cuda_solve_cusparse, 2*nnzL/(1e6*time_cuda_solve_cusparse), 
                                    time_cuda_analysis, time_cuda_solve, 2*nnzL/(1e6*time_cuda_solve), 
                                    time_cuda_solve_dfr,2*nnzL/(1e6*time_cuda_solve_dfr));
                                    */

        fprintf(ftabla,"%s & %i & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",
                                    filename, WARP_PER_BLOCK,
                                    time_cuda_analysis_cusparse, time_cuda_solve_cusparse, time_cuda_analysis_cusparse + time_cuda_solve_cusparse,
                                    time_cuda_analysis_cusparse2, time_cuda_solve_cusparse2_level, time_cuda_solve_cusparse2_nolevel,
                                    time_cuda_analysis, time_cuda_solve, time_cuda_analysis + time_cuda_solve,
                                    time_cuda_solve_dfr_a, time_cuda_solve_dfr);

        fclose(ftabla);
    }






    // step 6: free resources
    free(while_profiler);


    cudaFree(d_csrRowPtrL);
    cudaFree(d_csrColIdxL);
    cudaFree(d_csrValL);
    cudaFree(d_while_profiler);


    cudaFree(d_b);
    cudaFree(d_x);

    return 0;
}

#endif



