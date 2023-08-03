#include "common.h"
#include "dfr_syncfree.h"
#include <cooperative_groups.h>

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
    for (int i = tile_size/2; i >= 1; i /= 2)
        // Falta optimizar la mascara
        left_sum += myTile.shfl_down(left_sum, i);;

    
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

__global__ void csr_L_solve_simple_kernel(int* row_ctr,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const VALUE_TYPE* __restrict__ val,
    const VALUE_TYPE* __restrict__ b,
    VALUE_TYPE* x,
    int* is_solved, int n) {

    volatile __shared__ int        s_is_solved[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_x[WARP_PER_BLOCK];
    __shared__ int s_row;

    int wrp; // identifica numero del warp 
    int local_warp_id = threadIdx.x / WARP_SIZE;  // identifica warp dentro del bloque
    int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp


    if (threadIdx.x == 0) s_row = atomicAdd(&(row_ctr[0]), 1) * WARP_PER_BLOCK;
    __syncthreads();
    wrp = s_row + local_warp_id;

    if (wrp >= n) return;

    int row = row_ptr[wrp];
    int start_row = s_row; // identificador primer warp del bloque, cuenta cuantos warps antes, identifica la priemra fila calculara por bloque 
    int nxt_row = row_ptr[wrp + 1];
    int lock = 0;

    VALUE_TYPE left_sum = 0;
    VALUE_TYPE piv = 1 / val[nxt_row - 1];

    if (lne == 0) {
        left_sum = b[wrp];
        s_is_solved[local_warp_id] = 0;
    }

    __syncwarp();

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

            off += WARP_SIZE;
            ready = 0;
        }
    }
    __syncwarp();
    // Reduccion
    for (int i = 16; i >= 1; i /= 2)
        // Falta optimizar la mascara
        left_sum += __shfl_down_sync(__activemask(), left_sum, i);

    if (lne == 0) {
        //escribo en el resultado
        s_x[local_warp_id] = left_sum * piv;
        s_is_solved[local_warp_id] = 1;
        x[wrp] = left_sum * piv;
        __threadfence();
        is_solved[wrp] = 1;
    }

}


void aux_call_csr_L_solve_simple_kernel_coop(dfr_analysis_info_t* info,sp_mat_t* mat, const VALUE_TYPE* b, VALUE_TYPE* x, int n, int* is_solved,int num_threads, int grid, int average) {
	int* group_id_counter;
	CUDA_CHK(cudaMalloc((void**)&(group_id_counter), sizeof(int)));
	CUDA_CHK(cudaMemsetAsync(group_id_counter, 0, sizeof(int)));
	printf("Average::::::::: %d\n", average);
	
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
    if(SIMPLE_COOP){
    	printf("Using Simple Coop\n");
        aux_call_csr_L_solve_simple_kernel_coop(info, mat, b, x, n, is_solved, num_threads, grid, average);
    }else{
        csr_L_solve_simple_kernel<<< grid, num_threads >>>(info->row_ctr, mat->ia, mat->ja, mat->a, b, x, is_solved, n);

    }
	cudaDeviceSynchronize();
}

/*
__global__ void csr_L_solve_simple_v2_kernel(   const int* __restrict__ row_ptr,
                                              const int* __restrict__ col_idx,
                                              const VALUE_TYPE* __restrict__ val,
                                              const VALUE_TYPE* __restrict__ b,
                                              VALUE_TYPE* x,
                                              int * is_solved, int *is_solved_ptr, int n ) {

    volatile __shared__ int        s_is_solved[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_x        [WARP_PER_BLOCK];

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
    int local_is_solved = 0;
    int local_is_solved_ptr = lne;

    while(off < nxt_row - 1)
    {

        if(!myvar)
        {
            // if(colidx > start_row){
            //     myvar = s_is_solved[colidx-start_row];

            //     if (myvar){ //prefetch){
            //         xx = s_x[colidx-start_row];
            //     }
            // }
            // else
            {
                if (local_is_solved_ptr < *is_solved_ptr){
                    local_is_solved = is_solved[local_is_solved_ptr];
                    local_is_solved_ptr+=WARP_SIZE;
                }

                // myvar = myvar | __any(local_is_solved==colidx);
                // myvar = myvar | __any(local_is_solved == 0);
                myvar=1;

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

    if(lne==0){

        //escribo en el resultado
        // s_x[local_warp_id] = left_sum * piv;
        // s_is_solved[local_warp_id] = 1;

        x[wrp] = left_sum * piv;

        __threadfence();

        // is_solved[wrp] = 1;
        int old=atomicAdd(is_solved_ptr,1);
        is_solved[old] = wrp;
    }
}
*/
/*
__global__ void csr_L_solve_nan_kernel(   const int* __restrict__ row_ptr,
                                              const int* __restrict__ col_idx,
                                              const VALUE_TYPE* __restrict__ val,
                                              const VALUE_TYPE* __restrict__ b,
                                              VALUE_TYPE* x, int n ) {

    extern volatile __shared__ int s_mem[];

    VALUE_TYPE * s_x  = (VALUE_TYPE*) &s_mem[0];

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

    if(lne==0){
        left_sum = b[wrp];
        s_x[local_warp_id] = __longlong_as_double ( 0xFFFFFFFFFFFFFFFF );
    }

    __syncthreads();

    int off = row+lne;
    int colidx = col_idx[off];

    VALUE_TYPE my_val = val[off];
    VALUE_TYPE xx;

    int ready = 0;

    while(off < nxt_row - 1)
    {

        if(!ready)
        {
            if(colidx > start_row)
                xx = s_x[colidx-start_row];
            else{
                // xx = x[colidx];
                const VALUE_TYPE * xx_ptr =  &(x[colidx]);
                #ifdef __double__
                asm volatile ("ld.global.cg.f64 %0, [%1];" : "=d"(xx) : "l"(xx_ptr));
                #else
                asm volatile ("ld.global.cg.f32 %0, [%1];" : "=f"(xx) : "l"(xx_ptr));
                #endif
            }

            ready = __double2hiint ( xx ) != (int) 0xFFFFFFFF;
        }

        if( __all(ready) ){

            left_sum -= my_val * xx;
            off+=WARP_SIZE;
            colidx = col_idx[off];
            my_val = val[off];

            ready=0;
        }
    }

    // Reduccion
    for (int i=16; i>=1; i/=2){
        left_sum += __shfl_down(left_sum, i);
    }

    if(lne==0){

        //escribo en el resultado
        s_x[local_warp_id] = left_sum * piv;
        x[wrp] = left_sum * piv;
    }
}
*/
/*
__global__ void csr_L_solve_nan_hash_kernel(  const int* __restrict__ row_ptr,
                                              const int* __restrict__ col_idx,
                                              const VALUE_TYPE* __restrict__ val,
                                              const VALUE_TYPE* __restrict__ b,
                                              VALUE_TYPE* x, int n, int * row_ctr, unsigned long long int * hit, unsigned long long int * miss ) {

    extern __shared__ int shmem[];

    int * sh_ind = &shmem[0];
    VALUE_TYPE * sh_vals = (VALUE_TYPE*) &sh_ind[SIZE_SHARED];



    int wrp; //  = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
    int local_warp_id = threadIdx.x / WARP_SIZE;

    int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

    // if(wrp==10)
    // printf("Step %d\n", gridDim.x*WARP_PER_BLOCK);


    // if(threadIdx.x==0) s_mem[0] = atomicAdd(&(row_ctr[0]), 1);
    // __syncthreads();
    // wrp =  s_mem[0] * WARP_PER_BLOCK + threadIdx.x / WARP_SIZE;

    for(int i=threadIdx.x; i < SIZE_SHARED; i+=blockDim.x){
        sh_ind[i]=-1;
        sh_vals[i] = __longlong_as_double ( 0xFFFFFFFFFFFFFFFF );
    }

    if(lne==0) wrp = atomicAdd(&(row_ctr[0]), 1);
    __syncthreads();
    wrp = __shfl(wrp,0);

    unsigned long long int l_hits=0, l_misses=0;

    // for (int i = 0; i < IT_HASH; ++i)
    for (; wrp < n;)
    {


        int sh_off = wrp % SIZE_SHARED;
        atomicExch(&sh_ind[sh_off], wrp);


        int row = row_ptr[wrp];
        int start_row = wrp-wrp%WARP_PER_BLOCK;
        int nxt_row = row_ptr[wrp+1];

        int my_level = 0;

        VALUE_TYPE left_sum = 0;
        VALUE_TYPE piv = 1 / val[nxt_row-1];

        if(lne==0) left_sum = b[wrp];

        int off = row+lne;
        int colidx = col_idx[off];

        VALUE_TYPE my_val = val[off];
        VALUE_TYPE xx;

        int ready = 0;

        while(off < nxt_row - 1)
        {

            if(!ready)
            {
                // if (sh_ind[colidx%SIZE_SHARED]==colidx) l_hits++;
                // else l_misses++;

                if (sh_ind[colidx%SIZE_SHARED]!=-1){
                    if (sh_ind[colidx%SIZE_SHARED]==colidx)
                    {
                        l_hits++;
                    }
                    else
                    {
                        l_misses++;
                    }
                }


                if (sh_ind[colidx%SIZE_SHARED]==colidx)
                {
                    xx = sh_vals[colidx%SIZE_SHARED];
                }
                else
                {
                    // xx = x[colidx];
                    const VALUE_TYPE * xx_ptr =  &(x[colidx]);
                    #ifdef __double__
                    asm volatile ("ld.global.cg.f64 %0, [%1];" : "=d"(xx) : "l"(xx_ptr));
                    #else
                    asm volatile ("ld.global.cg.f32 %0, [%1];" : "=f"(xx) : "l"(xx_ptr));
                    #endif
                }

                ready = __double2hiint ( xx ) != (int) 0xFFFFFFFF;

                if (ready){
                    // atomicExch(&sh_ind[colidx%SIZE_SHARED],colidx);
                    // sh_vals[colidx%SIZE_SHARED]=xx;
                    if (atomicCAS(&sh_ind[colidx%SIZE_SHARED], -1, colidx)==-1) sh_vals[colidx%SIZE_SHARED] = xx;
                }
            }


            if( __all_sync(__activemask(),ready) )
            {

                left_sum -= my_val * xx;
                off+=WARP_SIZE;
                colidx = col_idx[off];
                my_val = val[off];

                ready=0;
            }
        }

        // Reduccion
        for (int i=16; i>=1; i/=2){
            left_sum += __shfl_down_sync(__activemask(),left_sum, i);
        }

        if(lne==0){

            //escribo en el resultado
            sh_vals[sh_off] = left_sum * piv;

            // s_x[local_warp_id] = left_sum * piv;
            x[wrp] = left_sum * piv;
        }

        if(lne==0) wrp = atomicAdd(&(row_ctr[0]), 1);
        wrp = __shfl_sync(__activemask(),wrp,0);

    }

    __syncthreads();

    atomicAdd(hit, l_hits);
    atomicAdd(miss, l_misses);
}
*/


/*
__global__ void csr_L_solve_nan_kernel_noshared(   const int* __restrict__ row_ptr,
                                              const int* __restrict__ col_idx,
                                              const VALUE_TYPE* __restrict__ val,
                                              const VALUE_TYPE* __restrict__ b,
                                              VALUE_TYPE* x, int n ) {

    // extern volatile __shared__ int s_mem[];

    // VALUE_TYPE * s_x  = (VALUE_TYPE*) &s_mem[0];

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

    if(lne==0){
        left_sum = b[wrp];
        // s_x[local_warp_id] = __longlong_as_double ( 0xFFFFFFFFFFFFFFFF );
    }

    __syncthreads();

    int off = row+lne;
    int colidx = col_idx[off];

    VALUE_TYPE my_val = val[off];
    VALUE_TYPE xx;

    int ready = 0;

    while(off < nxt_row - 1)
    {

        if(!ready)
        {
            // if(colidx > start_row)
                // xx = s_x[colidx-start_row];
            // else
                xx = x[colidx];

            ready = __double2hiint ( xx ) != (int) 0xFFFFFFFF;
        }

        if( __all(ready) ){

            left_sum -= my_val * xx;
            off+=WARP_SIZE;
            colidx = col_idx[off];
            my_val = val[off];

            ready=0;
        }
    }

    // Reduccion
    for (int i=16; i>=1; i/=2){
        left_sum += __shfl_down(left_sum, i);
    }

    if(lne==0){

        //escribo en el resultado
        // s_x[local_warp_id] = left_sum * piv;
        x[wrp] = left_sum * piv;
    }
}
*/


/*
void csr_L_solve_simple_v2 ( sp_mat_t * mat, const VALUE_TYPE * b, VALUE_TYPE * x, int n, int * is_solved , int * is_solved_ptr ){

        int num_threads = WARP_PER_BLOCK*WARP_SIZE;
        int grid = ceil ( (double) n * WARP_SIZE / (double) num_threads);

        cudaMemset(is_solved, 0, n * sizeof(int));
        cudaMemset(is_solved_ptr, 0, sizeof(int));

        csr_L_solve_simple_v2_kernel<<< grid , num_threads >>>(mat->ia, mat->ja, mat->a, b, x, is_solved, is_solved_ptr, n );
}

void csr_L_solve_nan ( sp_mat_t * mat, const VALUE_TYPE * b, VALUE_TYPE * x, int n ){

        int num_threads = WARP_PER_BLOCK*WARP_SIZE;
        int grid = ceil ((double) n * WARP_SIZE / (double) num_threads);

        cudaMemset(x, 0xFF, n * sizeof(VALUE_TYPE));

        csr_L_solve_nan_kernel<<< grid , num_threads, WARP_PER_BLOCK * sizeof(VALUE_TYPE) >>>(mat->ia, mat->ja, mat->a, b, x, n);
}

void csr_L_solve_nan_hash ( const char * matname, sp_mat_t * mat, dfr_analysis_info_t * info, const VALUE_TYPE * b, VALUE_TYPE * x, int n ){

        // cudaDeviceProp deviceProp;
        // cudaGetDeviceProperties(&deviceProp, 0);
        // int grid = deviceProp.multiProcessorCount * 8;

        // printf("%d\n", grid );

        // int grid = 9000;

    CLK_INIT;

    unsigned long long int *hit,*miss;
    CUDA_CHK( cudaMallocManaged((void**)&hit , sizeof(int) ) )
    CUDA_CHK( cudaMallocManaged((void**)&miss , sizeof(int) ) )

    char filename[50];

    strcpy(filename,"_hash_");
    strcat(filename,matname);



    FILE *ftabla,*ftabla_nan;
    ftabla = fopen(filename,"w");

    strcpy(filename,"_nan_");
    strcat(filename,matname);

    ftabla_nan = fopen(filename,"w");

    // int i =1;
    for (int i = 1; i < n/32; i++)
    {


        // int num_threads = i * WARP_SIZE; //WARP_PER_BLOCK*WARP_SIZE;
        // int grid = ceil ((double) i / (double) WARP_PER_BLOCK);

        // int num_threads = WARP_PER_BLOCK*WARP_SIZE;
        // int grid = ceil ((double) n * WARP_SIZE / (double) num_threads);

        // int i = 1429;
        int num_threads = i * WARP_SIZE; //WARP_PER_BLOCK*WARP_SIZE;
        int grid = ceil ((double) i / (double) WARP_PER_BLOCK);

        CUDA_CHK( cudaMemset(info->row_ctr, 0 , sizeof(int) ) )
        CUDA_CHK( cudaMemset(x, 0xFF, n * sizeof(VALUE_TYPE)) )

        // CUDA_CHK( cudaMemset(hit , 0 , sizeof(unsigned long long int) ) )
        // CUDA_CHK( cudaMemset(miss, 0 , sizeof(unsigned long long int) ) )



        hit[0]=0;
        miss[0]=0;

        // cudaDeviceSynchronize();
        CLK_START;
        csr_L_solve_nan_hash_kernel<<< grid , min(num_threads,WARP_PER_BLOCK*WARP_SIZE), SIZE_SHARED * (sizeof(int) + sizeof(VALUE_TYPE))  >>>(mat->ia, mat->ja, mat->a, b, x, n, info->row_ctr, hit, miss );
        CLK_STOP;

        cudaDeviceSynchronize();

        fprintf(ftabla, "%d;%f\n", i, (double) hit[0] / (double) (miss[0]+hit[0]) );
        // fprintf(ftabla, "%d;%f\n", i, CLK_ELAPSED );

        CUDA_CHK( cudaMemset(x, 0xFF, n * sizeof(VALUE_TYPE)) )

        num_threads = WARP_PER_BLOCK*WARP_SIZE;
        grid = ceil ((double) n * WARP_SIZE / (double) num_threads);

        CLK_START;
        csr_L_solve_nan_kernel<<< grid , num_threads, WARP_PER_BLOCK * sizeof(VALUE_TYPE) >>>(mat->ia, mat->ja, mat->a, b, x, n);
        CLK_STOP;
        fprintf(ftabla_nan, "%d;%f\n", i, CLK_ELAPSED );

    }

    fclose(ftabla);
    fclose(ftabla_nan);

}

void csr_L_solve_nan_noshared ( sp_mat_t * mat, const VALUE_TYPE * b, VALUE_TYPE * x, int n ){

        int num_threads = WARP_PER_BLOCK*WARP_SIZE;
        int grid = ceil ((double) n * WARP_SIZE / (double) num_threads);

        cudaMemset(x, 0xFF, n * sizeof(VALUE_TYPE));

        csr_L_solve_nan_kernel_noshared<<< grid , num_threads, WARP_PER_BLOCK * sizeof(VALUE_TYPE) >>>(mat->ia, mat->ja, mat->a, b, x, n);
}
*/
