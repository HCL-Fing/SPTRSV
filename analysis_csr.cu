#include "common.h"
#include "dfr_syncfree.h"
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define NUM_THREADS_OMP 4
#define ANALYSIS_COOP 1 
using namespace cooperative_groups;

struct vect_size_calculator : public thrust::unary_function<int, int> {
	__device__
		int operator()(int nnz_row) {
		int vect_size;
		if (nnz_row == 0)
			vect_size = 6;
		else if (nnz_row == 1)
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

		return vect_size;
	}
	;
};


struct is32{
__device__
bool operator()(int input){
	return input == 5;	//vect_size == 5 -> n(row)>16

};
};

struct subst_one : public thrust::unary_function<int, int> {
	__device__
		int operator()(int input) {
		return input - 1;
	}
	;
};

struct vect_size_from_index : public thrust::unary_function<int, int> {
	__device__
		int operator()(int input) {
		int vect_size = input % 7;
		return (vect_size == 6) ? 0 : pow(2, vect_size);
	}
	;
};

struct inverse_subst : public thrust::binary_function<int, int, int> {
	__device__
		int operator()(int fst, int snd) {
		return snd - fst - 1;
	};
};

struct get_index_from_size : public thrust::binary_function<int, int, int> {
	__device__
		int operator()(int vect_size, int lev) {
		return 7 * lev + vect_size;
	};
};

struct rows_to_order : public thrust::unary_function<int, int>{
	__device__
		int operator()(int pos){
			return order[pos];
		};
	int* order;
};


struct get_warps_per_bucks : public thrust::binary_function<int, int, int>{
	__device__/*
		int operator()(int ind){
			int size = sizes[ind];
			if(size == -1) size =1;
			return ceil((buckets[ind+1]-buckets[ind])*size/32.0);		
		};
		int* buckets;
		int* sizes;*/
		int operator()(int size, int cant){
			if(size==-1) size=1;
			return ceil(cant*size/32.0);
		}
};

struct circ_max : public thrust::binary_function<int, int, int>{
	__device__
		int operator()(int left, int right){
			if(right == -1)
				return 0;
			else if(left > right)
				return left;
			else
				return right;
		};
};

struct get_size_from_pos : public thrust::unary_function<int, int>{
	__device__
		int operator()(int pos){
			if(pos%7 == 0){
				return 1;
			}else if(pos%7 == 1){
				return 2;
			}else if(pos%7 == 2){
				return 4;
			}else if(pos%7 == 3){
				return 8;
			}else if(pos%7 == 4){
				return 16;
			}else if(pos%7 == 5){
				return 32;
			}else{	//%7 ==6
				return -1;	//Size 0
			}
		};
};



__global__ void two_phase_analysis_preproccessing(const int* __restrict__ col_idx, int* counter, int nnz){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<nnz)
	//printf("Thread %i-> counter[%i]+=1\n",i,col_idx[i]);

    	atomicAdd(&(counter[col_idx[i]]), 1);

}

__global__ void two_phase_analysis_up(const int* __restrict__ row_ptr, 
                                          const int* __restrict__ col_idx, 
                                          volatile int * is_solved, int n, 
                                          unsigned int * dfr_analysis_info/*, int* row_idx*/) {


    extern volatile __shared__ int s_mem[];       

    // int WARP_PER_BLOCK = blockDim.x / WARP_SIZE;

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
	colidx = col_idx[off];
	/*if((colidx == 10975475 && row == 12024060) || (colidx == 10241505 && row== 10975475)) 
	{
	if(colidx == 10975475) printf("elem 2\n"); else printf("elem 1\n");

	}*/


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

        if(__all_sync(__activemask(), myvar) ){

            off+=WARP_SIZE;
    //        colidx = col_idx[off];
            myvar=0;
        }
    }
    
   
    __syncwarp();
    // Reduccion
    for (int i=16; i>=1; i/=2){
        my_level = max(my_level, __shfl_down_sync(__activemask(), my_level, i));
    }
    

    //Create row_idx
/*    off = row+lne;
    colidx = col_idx[off];


    while(off < nxt_row - 1){
        row_idx[colidx] = row;
        off+=WARP_SIZE;
        colidx = col_idx[off];
    }*/

    if(lne==0){

        //escribo en el resultado
        s_info[local_warp_id] = 1+my_level;
        s_is_solved[local_warp_id] = 1;


        dfr_analysis_info[wrp] = 1+my_level;

        __threadfence();

        is_solved[wrp] = 1;
    
	
    }
}

__global__ void two_phase_analysis_down(const int* __restrict__ row_ptr, 
                                          const int* __restrict__ col_idx, 
                                          volatile int * is_solved, int n, 
                                          unsigned int * dfr_analysis_info, volatile int* counter, unsigned int* max_lev){



    /*extern volatile __shared__ int s_mem[];       

    // int WARP_PER_BLOCK = blockDim.x / WARP_SIZE;

    int * s_is_solved = (int *) &s_mem[0];
    int * s_info      = (int *) &s_is_solved[WARP_PER_BLOCK];
	*/
    int wrp =n-1- (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
    //int local_warp_id = threadIdx.x / WARP_SIZE;

    int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

    if(wrp >= n || wrp<0) return; 
  
    int row = row_ptr[wrp];
    int start_row = blockIdx.x*WARP_PER_BLOCK;
    int nxt_row = row_ptr[wrp+1];

    //int my_level = 
    

	//if (lne == 0) printf("Counter[%i]: %i\n",wrp,counter[wrp]);

    while(counter[wrp]>1){
	//if (lne == 0) printf("Counter[%i]: %i\n",wrp,counter[wrp]);
    }

//	__syncwarp();
    __threadfence();
    int off = row+lne;
    int colidx = col_idx[off];

    int my_level = min( dfr_analysis_info[wrp], max_lev[0]);
    if(my_level == max_lev[0] && lne==0)
	dfr_analysis_info[wrp] = my_level;		//Caso que todavía tiene maxint en dfr


	__threadfence();
    while(off < nxt_row - 1){


        atomicMin(&(dfr_analysis_info[colidx]), my_level-1 );

        
        __threadfence();

        atomicSub((int*) &(counter[colidx]),1);
        off+=WARP_SIZE;
        colidx = col_idx[off];
    }
}






template<int tile_size >
__global__ void kernel_analysis_L_coop_groups(const int* __restrict__ row_ptr,
	const int* __restrict__ col_idx,
	volatile int* is_solved, int n,
	unsigned int* dfr_analysis_info,
	int* group_id_counter) {

	//Define the size of a tile and how many tiles per warp
	const int groupsPerWarp = (WARP_SIZE / tile_size);
	thread_block_tile<tile_size>  myTile = tiled_partition<tile_size>(this_thread_block());

	__shared__ int start_row;
	extern volatile __shared__ int s_mem[];
	int* s_is_solved = (int*)&s_mem[0];
	int* s_info = (int*)&s_is_solved[WARP_PER_BLOCK * groupsPerWarp];

	int local_tile_id = threadIdx.x / tile_size;

	if (threadIdx.x == 0) {
		start_row = atomicAdd(group_id_counter, 1) * WARP_PER_BLOCK * groupsPerWarp;
	}
	this_thread_block().sync();
	int groupId = start_row + local_tile_id;

	if (groupId > n) return;
	if (groupId == 0) {
		int x = 0;
	}

	int row = row_ptr[groupId];
	int nxt_row = row_ptr[groupId + 1];

	int lne = myTile.thread_rank();// identifica el hilo dentro el warp

	if (lne == 0) {
		s_is_solved[local_tile_id] = 0;
		s_info[local_tile_id] = 0;
	}

	myTile.sync();

	int off = row + lne;
	int colidx = col_idx[off];

	int ready = 0;
	int my_level = 0;

	while (off < nxt_row - 1) {
		colidx = col_idx[off];
		if (!ready) {
			if (colidx > start_row) {
				ready = s_is_solved[colidx - start_row];

				if (ready) {
					my_level = max(my_level, s_info[colidx - start_row]);
				}
			} else {
				ready = is_solved[colidx];

				if (ready) {
					my_level = max(my_level, dfr_analysis_info[colidx]);
				}
			}
		}

		if (ready) {
			off += myTile.size();
			ready = 0;
		}
	}
	myTile.sync();
	// Reduccion

	
 	#if __CUDA_ARCH__ >= 800
	int aux = reduce(myTile, (int)my_level, greater<int>());
	my_level = aux;

 	#elif __CUDA_ARCH__ < 800 || !defined(__CUDA_ARCH__)   
 	for (int i = tile_size / 2; i >= 1; i /= 2) {
		int aux = myTile.shfl_down(my_level, i);
		my_level = max(my_level, aux);
	}
     #endif
	 


	if (lne == 0) {

		//escribo en el resultado
		s_info[local_tile_id] = 1 + my_level;
		s_is_solved[local_tile_id] = 1;


		dfr_analysis_info[groupId] = 1 + my_level;

		__threadfence(); //Esta al pedo?

		is_solved[groupId] = 1;
	}
}

template __global__ void kernel_analysis_L_coop_groups<1>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved,int n, unsigned int * dfr_analysis_info, int* group_id_counter);
template __global__ void kernel_analysis_L_coop_groups<2>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved,int n, unsigned int * dfr_analysis_info, int* group_id_counter);
template __global__ void kernel_analysis_L_coop_groups<4>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved,int n, unsigned int * dfr_analysis_info, int* group_id_counter);
template __global__ void kernel_analysis_L_coop_groups<8>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved,int n, unsigned int * dfr_analysis_info, int* group_id_counter);
template __global__ void kernel_analysis_L_coop_groups<16>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved,int n, unsigned int* dfr_analysis_info, int* group_id_counter);
template __global__ void kernel_analysis_L_coop_groups<32>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved,int n, unsigned int* dfr_analysis_info, int* group_id_counter);

__global__ void kernel_analysis_L(const int* __restrict__ row_ptr,
	const int* __restrict__ col_idx,
	volatile int* is_solved, int n,
	unsigned int* dfr_analysis_info) {

	/*
		volatile __shared__ int        s_is_solved[WARP_PER_BLOCK];
		volatile __shared__ int        s_info     [WARP_PER_BLOCK];
		volatile __shared__ FLOAT s_x        [WARP_PER_BLOCK];
	*/

	extern volatile __shared__ int s_mem[];

	// int WARP_PER_BLOCK = blockDim.x / WARP_SIZE;
	if(threadIdx.x==0&&blockIdx.x==0) printf("%i\n", WARP_PER_BLOCK);
	int* s_is_solved = (int*)&s_mem[0];
	int* s_info = (int*)&s_is_solved[WARP_PER_BLOCK];

	int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
	int local_warp_id = threadIdx.x / WARP_SIZE;

	int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

	if (wrp >= n) return;

	int row = row_ptr[wrp];
	int start_row = blockIdx.x * WARP_PER_BLOCK;
	int nxt_row = row_ptr[wrp + 1];

	int my_level = 0;

	/*

		if(!local_warp_id && lne < WARP_PER_BLOCK){

			s_x[lne] = b[start_row+lne];
			s_is_solved[lne] = 0;
		}

	*/
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
			//           colidx = col_idx[off];
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

// in-place exclusive scan
void exclusive_scan(int* input, int length)
{
	if (length == 0 || length == 1)
		return;

	int old_val, new_val;

	old_val = input[0];
	input[0] = 0;
	for (int i = 1; i < length; i++)
	{
		new_val = input[i];
		input[i] = old_val + input[i - 1];
		old_val = new_val;
	}
}

void aux_call_kernel_analysis_L_coop_groups(int grid, int num_threads,  sp_mat_t* gpu_L, 
										   int* d_is_solved, int rows, unsigned int* d_dfr_analysis_info, int average) {
	int* group_id_counter;
	CUDA_CHK(cudaMalloc((void**)&(group_id_counter), sizeof(int)));
	CUDA_CHK(cudaMemsetAsync(group_id_counter, 0, sizeof(int)));
	printf("\n");
	printf("Average::::::::: %d\n", average);
	printf("\n");
	
	int shared_size;
	switch (average){
	case 0 ... 1:
		 shared_size= WARP_PER_BLOCK * sizeof(VALUE_TYPE) * (WARP_SIZE);
		kernel_analysis_L_coop_groups<1><<< grid, num_threads, shared_size >>> (gpu_L->ia, gpu_L->ja, d_is_solved, rows, d_dfr_analysis_info,group_id_counter);
		break;
	case 2:
		shared_size = WARP_PER_BLOCK * sizeof(VALUE_TYPE) * (WARP_SIZE / 2);
		kernel_analysis_L_coop_groups<2><<< grid, num_threads, shared_size >>> (gpu_L->ia, gpu_L->ja, d_is_solved, rows, d_dfr_analysis_info,group_id_counter);
		break;
	case 3 ... 5:
		shared_size = WARP_PER_BLOCK * sizeof(VALUE_TYPE) * (WARP_SIZE / 4);
		kernel_analysis_L_coop_groups<4><<< grid, num_threads, shared_size >>> (gpu_L->ia, gpu_L->ja, d_is_solved, rows, d_dfr_analysis_info,group_id_counter);
		break;
	case 6 ... 11:
		shared_size = WARP_PER_BLOCK * sizeof(VALUE_TYPE) * (WARP_SIZE / 8);
		kernel_analysis_L_coop_groups<8><<< grid, num_threads, shared_size >>> (gpu_L->ia, gpu_L->ja, d_is_solved, rows, d_dfr_analysis_info,group_id_counter);
		break;
	case 12 ... 23:
		shared_size = WARP_PER_BLOCK * sizeof(VALUE_TYPE) * (WARP_SIZE / 16);
		kernel_analysis_L_coop_groups<16><<< grid, num_threads, shared_size >>> (gpu_L->ia, gpu_L->ja, d_is_solved, rows, d_dfr_analysis_info,group_id_counter);
		break;
	default:
		shared_size = WARP_PER_BLOCK * sizeof(VALUE_TYPE) * (WARP_SIZE / 32);
		kernel_analysis_L_coop_groups<32><<< grid, num_threads, shared_size >>> (gpu_L->ia, gpu_L->ja, d_is_solved, rows, d_dfr_analysis_info,group_id_counter);
		break;
	}
}

/*void csr_L_get_depth(sp_mat_t* gpu_L, int* dfr_analysis_info) {

	int rows = gpu_L->nr;
	int nnz = gpu_L->nnz;
	int average = rows/nnz;
	
	int* d_dfr_analysis_info;
	int* d_is_solved;

	CUDA_CHK(cudaMalloc((void**)&(d_dfr_analysis_info), rows * sizeof(int)));
	CUDA_CHK(cudaMalloc((void**)&(d_is_solved), rows * sizeof(int)));

	int num_threads = WARP_PER_BLOCK * WARP_SIZE;

	int grid = ceil((double)rows * WARP_SIZE / (double)(num_threads * ROWS_PER_THREAD));

	CUDA_CHK(cudaMemset(d_is_solved, 0, rows * sizeof(int)));
	CUDA_CHK(cudaMemset(d_dfr_analysis_info, 0, rows * sizeof(int)));
	
	if (ANALYSIS_COOP) {
		aux_call_kernel_analysis_L_coop_groups(grid,num_threads,gpu_L,d_is_solved,rows,d_dfr_analysis_info,average);
	} else {
		int shared_size = WARP_PER_BLOCK * sizeof(VALUE_TYPE) * (WARP_SIZE);
		kernel_analysis_L << < grid, num_threads,  shared_size>> > (gpu_L->ia, gpu_L->ja, d_is_solved, rows, d_dfr_analysis_info);
	}


	CUDA_CHK(cudaMemcpy(dfr_analysis_info, d_dfr_analysis_info, rows * sizeof(int), cudaMemcpyDeviceToHost))

		// int nLevs_dfr = dfr_analysis_info[0];
		// for (int i = 1; i < n; ++i)
		// {
		//     nLevs_dfr = MAX(nLevs_dfr, dfr_analysis_info[i]);
		// }

}*/




__global__ void kernel_nnz_row(int* csr_row_ptr, int n, int* nnz_row, int* iorder){
	int x=threadIdx.x+blockIdx.x*blockDim.x;
	extern __shared__ int rows[];

	if(x>=n)		//x==n deberia ser el que trae el csr_row_ptr[n] y luego no hace más nada
		return;
	
	rows[threadIdx.x] = csr_row_ptr[x];
	if(threadIdx.x==blockDim.x-1)
		rows[blockDim.x] = csr_row_ptr[x+1];
	__syncthreads();

	
	//if(threadIdx.x != blockDim.x-1){
		int size = rows[threadIdx.x+1] - rows[threadIdx.x] -1;
		int vect_size;
		if (size == 0)
			vect_size = 6;
		else if (size == 1)
			vect_size = 0;
		else if (size <= 2)
			vect_size = 1;
		else if (size <= 4)
			vect_size = 2;
		else if (size <= 8)
			vect_size = 3;
		else if (size <= 16)
			vect_size = 4;
		else vect_size = 5;

		//printf("nnz_row[%i] = %i\n", x, vect_size);

		nnz_row[x] = vect_size;
		iorder[x] = x;	//Si esto es viable hacer al principio capaz hay más coalesced
		

	//}


}

//Hardcodeado 7
__global__ void kernel_transform(int* row_off, int* warp_off){
	if(threadIdx.x < 8){
		
		if (threadIdx.x==7){
			warp_off[threadIdx.x] =0;
			//printf("Wrp[%i] = 0 = %i\n", threadIdx.x, 0);
		}else {
			int size = (threadIdx.x == 6) ? 1 : pow(2,(threadIdx.x));

			warp_off[threadIdx.x] = ceil(row_off[threadIdx.x] / (double) (32/ size));     
                        //printf("Wrp[%i] = %i/%i =%i\n", threadIdx.x, row_off[threadIdx.x], 32/size   ,warp_off[threadIdx.x]);

		}
	}

}
__global__ void scatter(int* warp_off, int* warp_size){
	warp_size[warp_off[threadIdx.x]] = threadIdx.x;
}

__global__ void kernel_assign_no_lvl(int* warp_off, int* ivect_size_warp, int* row_off,
		int* ibase_row){
		

		int num_warp = blockDim.x * blockIdx.x + threadIdx.x;
		int size = ivect_size_warp[num_warp];



		int inicio_w = warp_off[size];
		int inicio_r = row_off[size];

		//Termino de arreglar el vector size para que quede en el mismo formato
		if(size == 6)
			size = 0;
		else 
			size= pow(2,size);
		ivect_size_warp[num_warp] = size;



		//Calculo las filas que corresponden al warp
		if(size== 0) size =1;
		int off = num_warp-inicio_w;
		ibase_row[num_warp] = inicio_r+ off*size;
}


void multirow_analysis_no_lvl(dfr_analysis_info_t** mat, sp_mat_t* gpu_L, int mode){
	dfr_analysis_info_t* current = *mat;


	//CUDA_CHK(cudaMemcpy(csrRowPtrL_tmp, gpu_L->ia, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost))

        int n = gpu_L->nr;

	/*Call kernel*/
	
	int num_threads = WARP_PER_BLOCK * WARP_SIZE;

	int grid = ceil((float)n  / num_threads );

	int *ivects, *iorder_in,*nnz_row,*vect_size;
	CUDA_CHK(cudaMalloc((void**)&(ivects), (8+3*n) * sizeof(int)))
	iorder_in = ivects+8;
	//iorder=iorder_in+n;
	nnz_row = iorder_in+n;
	vect_size = nnz_row+n;//tamaño = n


	CUDA_CHK(cudaMalloc(&(current->iorder),n*sizeof(int))); 
	kernel_nnz_row << < grid, num_threads, num_threads+1 >> >(gpu_L->ia, n, nnz_row, iorder_in);

	cudaDeviceSynchronize();	
	void*    d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	//num_levs = num_bins + 1 = 7+1 
	//Pongo otro +1 para tener un 0 al final por el exclusive scan 
	cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
    nnz_row, ivects, 9, 0, 8, n);
	CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
    nnz_row, ivects, 9, 0, 8, n);


	CUDA_CHK(cudaFree(d_temp_storage));
	d_temp_storage=NULL;
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
    nnz_row, vect_size, iorder_in, current->iorder, n);
	CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
    nnz_row, vect_size, iorder_in, current->iorder, n);


	int prueba[n];
	cudaMemcpy(&prueba, current->iorder, n*sizeof(int), cudaMemcpyDeviceToHost); 
	for(int i=0;i<n;i++) printf("Iorder[%i] = %i ",i, prueba[i]);
	printf("\n");
	int* warp_off;
	CUDA_CHK(cudaMalloc(&warp_off,8*sizeof(int)));
	kernel_transform << < 1, 8 >> >(ivects, iorder_in+8);


     


	//Genero offsets de donde empieza cada tamaño en las filas y los warps
	CUDA_CHK(cudaFree(d_temp_storage));
	d_temp_storage=NULL;
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, ivects, iorder_in, 8);
	CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, ivects, iorder_in, 8);
       

	//Asumo que n>16 (iorder_in tiene suficientes elementos)
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, iorder_in+8, warp_off, 8);
        

	int num_warps=777;
	;
	CUDA_CHK(cudaMemcpy(&num_warps, &(warp_off[7]), sizeof(int), cudaMemcpyDeviceToHost))
	current->n_warps = num_warps;
        

	int num_rows =77;
        CUDA_CHK(cudaMemcpy(&num_rows, &(iorder_in[7]), sizeof(int), cudaMemcpyDeviceToHost))
	printf("N_rows: %i. N_warps:%i\n", num_rows,num_warps);
	//int num_warps = warp_off[7];
        


	//CUDA_CHK(cudaMalloc((void**)&(current->ibase_row), (num_warps + 1) * sizeof(int)));
	//CUDA_CHK(cudaMalloc((void**)&(current->ivect_size_warp), num_warps * sizeof(int)));
	CUDA_CHK(cudaMalloc(&(current->ibase_row),(2*num_warps + 1)));
        

	current->ivect_size_warp = current->ibase_row+num_warps+1;
        


	//Uso temporalmente ibase_row para almacenar el arreglo agujereado
	cudaMemset(current->ibase_row, 0, num_warps*sizeof(int));
        

	scatter << < 1, 7>> >(iorder_in, current->ibase_row);
	


	//Scan para obtener el arreglo de tamaños completo
	CUDA_CHK(cudaFree(d_temp_storage));
	d_temp_storage=NULL;
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, current->ibase_row, current->ivect_size_warp, num_warps);
	CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, current->ibase_row, current->ivect_size_warp, num_warps);





	
	


	kernel_assign_no_lvl<<< ceil(num_warps/num_threads),num_threads >>> (warp_off, current->ivect_size_warp, iorder_in,
		current->ibase_row);

	CUDA_CHK(cudaMalloc((void**)&(current->row_ctr), sizeof(int)));




	CUDA_CHK(cudaFree(ivects));
	/*kernel_base<<<7*nLevs_dfr,32>>>(thrust::raw_pointer_cast(warps_per_bucks.data()), thrust::raw_pointer_cast(buckets.data()), 
		current->ibase_row, current->ivect_size_warp);
	*/

}




//Calcula warp base y warp size
__global__ void kernel_base(int* buckets_off, int* rows_off, int* warp_base, int* warp_size){

	int off = buckets_off[blockIdx.x], lst=buckets_off[blockIdx.x+1];
	int cant = lst -off;
	int size;

	if(blockIdx.x== gridDim.x-1 && threadIdx.x==0)
		warp_base[lst] = rows_off[gridDim.x];

	if(cant == 0) return;
	if(threadIdx.x > cant) return;
	size = ((blockIdx.x%7) == 6) ? 0 : pow(2,(blockIdx.x%7));     
	int rows = rows_off[blockIdx.x];
	// ivect_size[ivects[7 * idepth + vect_size]] = (vect_size == 6) ? 0 : pow(2, vect_size);

	__syncthreads();

	int x=threadIdx.x;
	int sz = size; if(sz == 0) sz =1;

	while(x+off<lst){
		warp_size[off + x] = size;
		warp_base[off + x] = rows + x*(32/sz);
		
		x += blockDim.x ;
		__syncwarp(__activemask());
	}



 /*	if(blockIdx.x == gridDim.x-1 && x+off == lst){
		warp_base[x+off] = rows_off[gridDim.x];
		printf("warp_base[%i+%i] = rows_off[%i] = %i\n", x, off, gridDim.x, rows_off[gridDim.x]);
	}*/
}







__global__ void kernel_init_matrix(const int* __restrict__ row_ptr,
	const int* __restrict__ col_idx,
	const VALUE_TYPE* __restrict__ val,
	int* iorder, int* warp_base_idx, int* warp_vect_size,
	int* row_ctr, int n, int n_warps, 
	int* mat_cols, VALUE_TYPE* mat_values, VALUE_TYPE* mat_diag, int* mat_row_idx){


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

    if ((row_idx >= n) || (vect_idx >= n_vects)) return;
    int nxt_row = row_ptr[row_idx + 1];  // Hasta que posicion va la fila



    if (vect_size == 0) {
        mat_diag[base_idx+vect_idx] = val[nxt_row - 1];;      //Agrego los dos de abajo, chequear
        mat_row_idx[base_idx+vect_idx] = row_idx; 
        return;
    }


    int vect_off = lne0 % vect_size;  // Cual le toca a cada thread (por si el warp es mas grande que la cantidad a procesar)
    int row = row_ptr[row_idx];  // El primer elemento de la fila
	int off = row + vect_off;

    VALUE_TYPE my_val;  // El elemento de la fila que trabaja el thread


    int colidx;


    if(vect_size<32){


        if(off>=nxt_row-1){ 
            colidx=-1;
        }else{
            colidx = col_idx[off];
        }


        mat_cols[wrp*WARP_SIZE + lne0] = colidx;
        if(off>=nxt_row-1) return;

        mat_values[wrp*WARP_SIZE + lne0] = val[off];        




        if (vect_off == 0) {
            mat_diag[base_idx+vect_idx] = val[nxt_row - 1];;   
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

        if (vect_off == 0) {
            mat_diag[base_idx] = val[nxt_row-1];    
            mat_row_idx[base_idx] = row_idx;
        }

    }

}






void multirow_analysis_base_GPU(dfr_analysis_info_t** mat, sp_mat_t* gpu_L, int mode, sp_mat_ana_t* ana_mat, unsigned int* dfr){
	FILE* fp;
	CLK_INIT;

	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		fp = fopen(LOG_FILE, "a+");
		CLK_START;
	}
	dfr_analysis_info_t* current = *mat;

	//n es el número de filas
	int rows = gpu_L->nr;
	int nnz = gpu_L->nnz;
	int average = nnz/rows;
	unsigned int* d_dfr_analysis_info;
	int* d_is_solved;


if(dfr == NULL)	CUDA_CHK(cudaMalloc((void**)&(d_dfr_analysis_info), rows * sizeof(int)))
else d_dfr_analysis_info= dfr;
	CUDA_CHK(cudaMalloc((void**)&(d_is_solved), rows * sizeof(int)))

	int num_threads = WARP_PER_BLOCK * WARP_SIZE;
	int grid = ceil((double)rows * WARP_SIZE / (double)(num_threads * ROWS_PER_THREAD));

	CUDA_CHK(cudaMemset(d_is_solved, 0, rows * sizeof(int)))
	CUDA_CHK(cudaMemset(d_dfr_analysis_info, 0, rows * sizeof(int)))


	
	
    // printf("Using base kernel\n");
    int shared_size = WARP_PER_BLOCK * sizeof(VALUE_TYPE) * (WARP_SIZE);
    //kernel_analysis_L << < grid, num_threads,  shared_size>> > (gpu_L->ia, gpu_L->ja, d_is_solved, rows, d_dfr_analysis_info);
	

	int* counter;
	if(mode ==5){
	  	CUDA_CHK(cudaMalloc((void**)&(counter), rows * sizeof(int)))
        	CUDA_CHK(cudaMemset(counter, 0, rows * sizeof(int)))

        	two_phase_analysis_preproccessing<<<nnz/num_threads+1, num_threads>>>(gpu_L->ja, counter, nnz);
cudaDeviceSynchronize();
		
		two_phase_analysis_up<<< grid, num_threads, WARP_PER_BLOCK * sizeof(VALUE_TYPE) >>>    (gpu_L->ia, gpu_L->ja, d_is_solved, rows, d_dfr_analysis_info);



		/*kernel_analysis_L << < grid, num_threads, WARP_PER_BLOCK * sizeof(VALUE_TYPE) >> > (gpu_L->ia,
                        gpu_L->ja,
                        d_is_solved,
                        rows,
                        d_dfr_analysis_info);
*/		cudaDeviceSynchronize();

		
		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;
		unsigned int *d_max;
		CUDA_CHK(cudaMalloc((void**) &(d_max),sizeof(int)))
		CUDA_CHK(cudaMemset(d_max, 0, sizeof(int)))
		unsigned int *max = (unsigned int*) malloc(sizeof(int));


		cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_dfr_analysis_info, d_max, rows);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run max-reduction
		cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_dfr_analysis_info, d_max, rows);

		CUDA_CHK(cudaMemcpy(max, d_max, sizeof(int), cudaMemcpyDeviceToHost))		
		printf("Down. Max: %i\n",max[0]);

                CUDA_CHK(cudaMemset(d_dfr_analysis_info, 0xFF, rows * sizeof(int)))
		two_phase_analysis_down<<< grid, num_threads >>> (gpu_L->ia, 
					gpu_L->ja, 
					d_is_solved, rows, 
					d_dfr_analysis_info, 
					counter,
					d_max);
		cudaDeviceSynchronize();

	/*kernel_analysis_L << < grid, num_threads, WARP_PER_BLOCK * sizeof(VALUE_TYPE) >> > (gpu_L->ia,
                gpu_L->ja,
                d_is_solved,
                n,
                d_dfr_analysis_info);
	*/
		cudaFree(counter);
	
	}else{
	//	aux_call_kernel_analysis_L_coop_groups(grid,num_threads,gpu_L,d_is_solved,rows,d_dfr_analysis_info,average);
		kernel_analysis_L << < grid, num_threads, WARP_PER_BLOCK * sizeof(VALUE_TYPE) >> > (gpu_L->ia,
                	gpu_L->ja,
                	d_is_solved,
                	rows,
                	d_dfr_analysis_info);

	}
	cudaDeviceSynchronize();


	//    CUDA_CHK( cudaMemcpy(dfr_analysis_info, d_dfr_analysis_info, rows * sizeof(int), cudaMemcpyDeviceToHost) )

	//    CUDA_CHK( cudaMemcpy(dfr_analysis_info, d_dfr_analysis_info, rows * sizeof(int), cudaMemcpyDeviceToHost) )

	thrust::device_ptr<unsigned int> temp_ptr(d_dfr_analysis_info);

	thrust::device_vector<unsigned int> dfr_analysis_info(temp_ptr, temp_ptr + rows);

	//  gpu_L->ia es csrRowPtrL_tmp de abajo
	//Wrap gpu_L->ia




	
	int nLevs_dfr = *thrust::max_element(dfr_analysis_info.begin(), dfr_analysis_info.end());
	printf("N_levs:%i\n",nLevs_dfr);
	/*int nLevs_dfr = dfr_analysis_info_h[0];
		for (int i = 1; i < rows; ++i){
			nLevs_dfr = MAX(nLevs_dfr, dfr_analysis_info[i]);
	}
	*/
	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		CLK_STOP;
		//Max
		fprintf(fp, ",%.2f", CLK_ELAPSED);
		CLK_START;
	}


	thrust::device_ptr<int> d_ptr(gpu_L->ia);
	//nnz = csr row pointer
	thrust::device_vector<int> nnz_row(d_ptr, d_ptr + rows + 1);


	//int nnz_row = csrRowPtrL_tmp[i+1]-csrRowPtrL_tmp[i]-1;
	inverse_subst sub;
	thrust::device_vector<int> vect_size(rows);
	//vect_size[i] = nnz_row[i-1] 
	thrust::copy(nnz_row.begin() + 1, nnz_row.begin() + rows + 1, vect_size.begin());
	//nnz_row[i] = nnz of row i without diag elem
	thrust::transform(nnz_row.begin(), nnz_row.begin() + rows, vect_size.begin(), nnz_row.begin(), sub);
	
	

	vect_size_calculator vcc;
	//vect_size = ceil(sqrt(nnz)), 6 for rows with only diag element
	thrust::transform(nnz_row.begin(),
		nnz_row.begin() + rows,
		vect_size.begin(),
		vcc);
	
	//Después uso este valor para restar al total
//	current->n_warps_leq32 = thrust::count_if(vect_size.begin(), vect_size.begin()+ rows, is32());


	thrust::device_vector<int> lev(rows);
	subst_one minus_one;
	thrust::transform(dfr_analysis_info.begin(), dfr_analysis_info.begin() + rows, lev.begin(), minus_one);


	get_index_from_size id_ivects;
	//dfr_analysis_info[i] = index of the counter for the pair(lev(row_i), size(row_i))
	// 		       = 7*lev+vect_size
	thrust::transform(vect_size.begin(),
		vect_size.begin() + rows,
		lev.begin(),
		dfr_analysis_info.begin(),
		id_ivects);


	//calculado: .*?[7*lev+vect_size]++
	thrust::device_vector<int> iorder(rows + 1);//aux(rows);
	thrust::device_vector<int> buckets(7 * nLevs_dfr + 1);

	thrust::constant_iterator<int> ones(1);
	//Acums the number of elements with the same lev-size
	thrust::device_vector<int> ivect_size(rows);


	//Comienza buckets
	thrust::device_vector<int> map(7 * nLevs_dfr);
	thrust::copy(dfr_analysis_info.begin(), dfr_analysis_info.begin() + rows, ivect_size.begin());
	thrust::stable_sort(ivect_size.begin(), ivect_size.begin() + rows);
	auto end = thrust::reduce_by_key(ivect_size.begin(), ivect_size.begin() + rows, ones, map.begin(), iorder.begin());

	thrust::scatter(iorder.begin(), end.second, map.begin(), buckets.begin());
	//	for(int i=0; i<iorder.size();i++){
	//		int temp = iorder[i];
	//		printf("I: %i. Iorder: %i\n", i, temp);
	//	}	









	thrust::counting_iterator<int> iter(0);
	thrust::copy(iter, iter + rows, iorder.begin());
	thrust::stable_sort_by_key(dfr_analysis_info.begin(), dfr_analysis_info.begin() + rows, iorder.begin());
	int num_warps; 

if(mode == 1){
	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
                CLK_STOP;
                //T5
                fprintf(fp, ",%.2f", CLK_ELAPSED);
                CLK_START;
        }

}else{
	vect_size_from_index calc_size;
	thrust::transform(dfr_analysis_info.begin(), dfr_analysis_info.begin() + rows, ivect_size.begin(), calc_size);




        if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
                CLK_STOP;
                //T5
                fprintf(fp, ",%.2f", CLK_ELAPSED);
                CLK_START;
        }


	//Termina T5
	dfr_analysis_info.resize(7*nLevs_dfr+1);
	thrust::device_vector<int> warps_per_bucks(7 * nLevs_dfr+1);
	get_size_from_pos sz;
	thrust::transform(iter, iter+7*nLevs_dfr, dfr_analysis_info.begin(),sz);

       

	get_warps_per_bucks wpb;

	thrust::transform(dfr_analysis_info.begin(), dfr_analysis_info.begin()+7*nLevs_dfr,buckets.begin(),warps_per_bucks.begin(),wpb);
	//Calcular buck off
	thrust::exclusive_scan(warps_per_bucks.begin(),warps_per_bucks.begin()+7*nLevs_dfr+1, warps_per_bucks.begin());
	num_warps = warps_per_bucks[7*nLevs_dfr];

	thrust::exclusive_scan(buckets.begin(),buckets.begin()+7*nLevs_dfr+1,buckets.begin());




	
	CUDA_CHK(cudaMalloc((void**)&(current->ibase_row), (num_warps + 1) * sizeof(int)));
	CUDA_CHK(cudaMalloc((void**)&(current->ivect_size_warp), num_warps * sizeof(int)));



	kernel_base<<<7*nLevs_dfr,32>>>(thrust::raw_pointer_cast(warps_per_bucks.data()), thrust::raw_pointer_cast(buckets.data()), current->ibase_row, current->ivect_size_warp);
        cudaDeviceSynchronize();

        warps_per_bucks.clear();
        warps_per_bucks.shrink_to_fit();
	
}



	cudaDeviceSynchronize();
	CUDA_CHK(cudaMalloc((void**)&(current->iorder), rows * sizeof(int)));
	CUDA_CHK(cudaMemcpy(current->iorder, thrust::raw_pointer_cast(iorder.data()), rows * sizeof(int), cudaMemcpyDeviceToDevice));
	CUDA_CHK(cudaMalloc((void**)&(current->row_ctr), sizeof(int)));


	current->nlevs = nLevs_dfr;
	current->n_warps = num_warps;

	


if(dfr==NULL){	dfr_analysis_info.clear();
	dfr_analysis_info.shrink_to_fit();}


	nnz_row.clear();
	nnz_row.shrink_to_fit();
	vect_size.clear();
	vect_size.shrink_to_fit();
	lev.clear();
	lev.shrink_to_fit();
	iorder.clear();
	iorder.shrink_to_fit();
	buckets.clear();
	buckets.shrink_to_fit();
	ivect_size.clear();
	ivect_size.shrink_to_fit();
	map.clear();
	map.shrink_to_fit();
//	warps_per_bucks.clear();
//	warps_per_bucks.shrink_to_fit();	


        if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
                CLK_STOP;
                //Assign
                fprintf(fp, ",%.2f\n", CLK_ELAPSED);
                CLK_START;
        }


		if(ana_mat != NULL){

			CLK_START;
			ana_mat->first = false;
			CUDA_CHK(cudaMalloc((void**) &(ana_mat->values), sizeof(VALUE_TYPE)*(current->n_warps*WARP_SIZE ) )); //values
			
			CUDA_CHK(cudaMalloc((void**) &(ana_mat->diag), sizeof(VALUE_TYPE)*rows  )); //diag
		  
		
			CUDA_CHK(cudaMalloc((void**) &(ana_mat->row_idx), sizeof(int)*rows )); //row_idx
			CUDA_CHK(cudaMalloc((void**) &(ana_mat->cols), sizeof(int)*(current->n_warps*WARP_SIZE) )); //col

			num_threads = WARP_PER_BLOCK * WARP_SIZE;
			grid = ceil((double)current->n_warps * WARP_SIZE / (double)num_threads);




			CUDA_CHK(cudaMemsetAsync(current->row_ctr, 0, sizeof(int), 0));
			CUDA_CHK(cudaMemsetAsync(ana_mat->values, 0, sizeof(VALUE_TYPE)*current->n_warps*WARP_SIZE , 0));

			kernel_init_matrix<<<grid, num_threads>>>(gpu_L->ia,
				gpu_L->ja,
				gpu_L->a,
				current->iorder,current->ibase_row, current->ivect_size_warp,
				current->row_ctr, rows, current->n_warps,  
				ana_mat->cols, ana_mat->values, ana_mat->diag, ana_mat->row_idx);
			cudaDeviceSynchronize();

			CLK_STOP;
			fp = fopen(LOG_FILE, "a+");
			fprintf(fp, ",%.2f", CLK_ELAPSED);
			fclose(fp);
		}
		

		
} 





void multirow_analysis_base_parallel(dfr_analysis_info_t** mat, sp_mat_t* gpu_L, int mode) {
	FILE* fp;
	CLK_INIT;
//float tiempo=0;
	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		fp = fopen(LOG_FILE, "a+");
		CLK_START;
	}
	dfr_analysis_info_t* current = *mat;

	//n es el número de filas
	int rows = gpu_L->nr;
	int nnz = gpu_L->nnz;
	int average = nnz/rows;
	unsigned int* d_dfr_analysis_info;
	int* d_is_solved;

	CUDA_CHK(cudaMalloc((void**)&(d_dfr_analysis_info), rows * sizeof(unsigned int)))
		CUDA_CHK(cudaMalloc((void**)&(d_is_solved), rows * sizeof(int)))

		int num_threads = WARP_PER_BLOCK * WARP_SIZE;
	int grid = ceil((double)rows * WARP_SIZE / (double)(num_threads * ROWS_PER_THREAD));

	CUDA_CHK(cudaMemset(d_is_solved, 0, rows * sizeof(int)))
		CUDA_CHK(cudaMemset(d_dfr_analysis_info, 0, rows * sizeof(int)))


		if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
			CLK_STOP;
			//Init
			fprintf(fp, ",%.2f", CLK_ELAPSED);
			CLK_START;

		}

	
	if (ANALYSIS_COOP) {
		printf("Using coop group kernel\n");
		aux_call_kernel_analysis_L_coop_groups(grid,num_threads,gpu_L,d_is_solved,rows,d_dfr_analysis_info,average);
	} else {
		printf("Using base kernel\n");
		int shared_size = WARP_PER_BLOCK * sizeof(VALUE_TYPE) * (WARP_SIZE);
		kernel_analysis_L << < grid, num_threads,  shared_size>> > (gpu_L->ia, gpu_L->ja, d_is_solved, rows, d_dfr_analysis_info);
	}
	cudaDeviceSynchronize();
	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		CLK_STOP;
		//Kernel analysis
		fprintf(fp, ",%.2f", CLK_ELAPSED);
		CLK_START;
	}
	//    CUDA_CHK( cudaMemcpy(dfr_analysis_info, d_dfr_analysis_info, rows * sizeof(int), cudaMemcpyDeviceToHost) )

	//    CUDA_CHK( cudaMemcpy(dfr_analysis_info, d_dfr_analysis_info, rows * sizeof(int), cudaMemcpyDeviceToHost) )

	thrust::device_ptr<unsigned int> temp_ptr(d_dfr_analysis_info);
	thrust::device_vector<unsigned int> dfr_analysis_info(temp_ptr, temp_ptr + rows);

	//  gpu_L->ia es csrRowPtrL_tmp de abajo
	//Wrap gpu_L->ia



//CLK_START
	unsigned int* dfr_analysis_info_h = (unsigned int*)calloc(rows, sizeof(unsigned int));
	CUDA_CHK(cudaMemcpy(dfr_analysis_info_h, d_dfr_analysis_info, rows * sizeof(unsigned int), cudaMemcpyDeviceToHost));
//CLK_STOP
//tiempo+= CLK_ELAPSED;

	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		CLK_STOP;
		//Dfr info
		fprintf(fp, ",%.2f", CLK_ELAPSED);
		CLK_START;
	}

	int nLevs_dfr = *thrust::max_element(dfr_analysis_info.begin(), dfr_analysis_info.end());
	/*int nLevs_dfr = dfr_analysis_info_h[0];
		for (int i = 1; i < rows; ++i){
			nLevs_dfr = MAX(nLevs_dfr, dfr_analysis_info[i]);
	}
	*/
	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		CLK_STOP;
		//Max
		fprintf(fp, ",%.2f", CLK_ELAPSED);
		CLK_START;
	}

/*	if(DIMENSIONES && LOG_FILE != "NONE"){
        FILE* fp2;
        fp2 = fopen("dimensiones_todas.csv", "a+");
        fprintf(fp2,",%d", gpu_L->nr);
        fprintf(fp2,",%d", nLevs_dfr);
        fprintf(fp2,",%d\n", gpu_L->nnz);
        fclose(fp2);
    }*/

	thrust::device_ptr<int> d_ptr(gpu_L->ia);
	//nnz = csr row pointer
	thrust::device_vector<int> nnz_row(d_ptr, d_ptr + rows + 1);


	//int nnz_row = csrRowPtrL_tmp[i+1]-csrRowPtrL_tmp[i]-1;
	inverse_subst sub;
	thrust::device_vector<int> vect_size(rows);
	//vect_size[i] = nnz_row[i-1] 
	thrust::copy(nnz_row.begin() + 1, nnz_row.begin() + rows + 1, vect_size.begin());
	//nnz_row[i] = nnz of row i without diag elem
	thrust::transform(nnz_row.begin(), nnz_row.begin() + rows, vect_size.begin(), nnz_row.begin(), sub);


	vect_size_calculator vcc;
	//vect_size = ceil(sqrt(nnz)), 6 for rows with only diag element
	thrust::transform(nnz_row.begin(),
		nnz_row.begin() + rows,
		vect_size.begin(),
		vcc);



	thrust::device_vector<int> lev(rows);
	subst_one minus_one;
	thrust::transform(dfr_analysis_info.begin(), dfr_analysis_info.begin() + rows, lev.begin(), minus_one);


	get_index_from_size id_ivects;
	//dfr_analysis_info[i] = index of the counter for the pair(lev(row_i), size(row_i))
	// 		       = 7*lev+vect_size
	thrust::transform(vect_size.begin(),
		vect_size.begin() + rows,
		lev.begin(),
		dfr_analysis_info.begin(),
		id_ivects);




	//ivects[7*lev+vect_size]++
	thrust::device_vector<int> iorder(rows + 1);//aux(rows);
	thrust::device_vector<int> ivects(7 * nLevs_dfr + 1);

	thrust::constant_iterator<int> ones(1);
	//Acums the number of elements with the same lev-size
	int* levels_warp_info;
	thrust::device_vector<int> ivect_size(rows);

	if (mode == 2) {
		thrust::device_vector<int> map(7 * nLevs_dfr);
		thrust::copy(dfr_analysis_info.begin(), dfr_analysis_info.begin() + rows, ivect_size.begin());
		thrust::sort(ivect_size.begin(), ivect_size.begin() + rows);
		auto end = thrust::reduce_by_key(ivect_size.begin(), ivect_size.begin() + rows, ones, map.begin(), iorder.begin());

		thrust::scatter(iorder.begin(), end.second, map.begin(), ivects.begin());
		thrust::exclusive_scan(ivects.begin(), ivects.begin() + 7 * nLevs_dfr, ivects.begin());
		/*for(int i=0;i<7*nLevs_dfr;i++){
			int temp = ivects[i];
			printf("I: %i, Val: %i\n", i, temp);
		}*/
		levels_warp_info = (int*)calloc(7 * nLevs_dfr + 1, sizeof(int));
		
		CUDA_CHK(cudaMemcpy(levels_warp_info, thrust::raw_pointer_cast(ivects.data()), 7 * nLevs_dfr * sizeof(int), cudaMemcpyDeviceToHost));
		levels_warp_info[7 * nLevs_dfr] = rows;
		map.clear();
		map.shrink_to_fit();

	}


	//ilevels[lev]++
	thrust::sort(lev.begin(), lev.begin() + rows);
	//Reuso vector y reuso ivects_aux(7*nLevs_dfr) como ilevels(nLevs_Dfr)
	thrust::reduce_by_key(lev.begin(), lev.begin() + rows, ones, thrust::make_discard_iterator(),/* ilevels*/ivects/*_aux*/.begin()); //ivects_aux.begin());


	//exclusive_scan(ivects, 7 * nLevs_dfr);
	//thrust::exclusive_scan(ivects.begin(),ivects.end(), ivects.begin());

	//iorder[ ivects[ 7*idepth+vect_size ] ] = i;             
	//ivect_size[ ivects[ 7*idepth+vect_size ] ] = ( vect_size == 6)? 0 : pow(2,vect_size);
	//ivects[ 7*idepth+vect_size ]++; 
	//Construyo iorder a partir de acá
	//thrust::device_vector<int> ivect_size(rows);
	thrust::counting_iterator<int> iter(0);
	thrust::copy(iter, iter + rows, iorder.begin());
	thrust::stable_sort_by_key(dfr_analysis_info.begin(), dfr_analysis_info.begin() + rows, iorder.begin());
	vect_size_from_index calc_size;
	thrust::transform(dfr_analysis_info.begin(), dfr_analysis_info.begin() + rows, ivect_size.begin(), calc_size);


	if((!(mode == 0 || mode ==2))&&(!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE")  ){
	//Print aca solo si no va a hacer warp info y no necesita transferir nada
			
					CLK_STOP;
					//Iorder/Size calculation
					fprintf(fp, ",%.2f", CLK_ELAPSED);
					CLK_START;
			

	}


	int* local_ii;
if (mode == 0 || mode ==2) {
		int ii = 1;
		int filas_warp;

//CLK_START
		int* iorder_h = (int*)calloc(rows, sizeof(int));
		CUDA_CHK(cudaMemcpy(iorder_h, thrust::raw_pointer_cast(iorder.data()), rows * sizeof(int), cudaMemcpyDeviceToHost));
		// 	int* dfr_analysis_info_h = (int*) calloc(rows, sizeof(int));
		//	CUDA_CHK( cudaMemcpy(dfr_analysis_info_h, d_dfr_analysis_info, rows * sizeof(int), cudaMemcpyDeviceToHost));	
		int* ivect_size_h = (int*)calloc(rows, sizeof(int));
		CUDA_CHK(cudaMemcpy(ivect_size_h, thrust::raw_pointer_cast(ivect_size.data()), rows * sizeof(int), cudaMemcpyDeviceToHost));
//CLK_STOP
//tiempo+=CLK_ELAPSED;
        	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
                	CLK_STOP;
                	//Iorder/Size calculation
                	fprintf(fp, ",%.2f", CLK_ELAPSED);
                	CLK_START;
        	}

		//        CLK_STOP;
		  //      printf("Middle transfer: %f\n", CLK_ELAPSED );
			//    CLK_START;
		//Enganchar con la otra
		if (mode == 0) {
			ii = 1;
			filas_warp = 1;
			for (int ctr = 1; ctr < rows; ++ctr) {

				if (dfr_analysis_info_h[iorder_h[ctr]] != dfr_analysis_info_h[iorder_h[ctr - 1]] ||
					ivect_size_h[ctr] != ivect_size_h[ctr - 1] ||
					filas_warp * ivect_size_h[ctr] >= 32 ||
					(ivect_size_h[ctr] == 0 && filas_warp == 32)) {

					filas_warp = 1;
					ii++;
				} else {
					filas_warp++;
				}
			}

		} else {	//mode =2
		//levels_warp_info
//CLK_START
			local_ii = (int*)calloc(7 * nLevs_dfr + 1, sizeof(int));
//CLK_STOP
//tiempo+=CLK_ELAPSED;
			ii = 0;
			omp_set_num_threads(NUM_THREADS_OMP);
#pragma omp parallel for private(filas_warp) shared(ivect_size,local_ii) reduction(+: ii)
			for (int i = 0; i < 7 * nLevs_dfr;++i) {
				filas_warp = 1;                                  //Esto puede que vaya afuera del for
				int temp;
				if (levels_warp_info[i] < levels_warp_info[i + 1]) {
					ii += 1;
					//		local_ii[i] =1;
					temp = 1;
				} else {
					temp = 0;
					//		local_ii[i] =0;
				}
				for (int ctr = levels_warp_info[i] + 1; ctr < levels_warp_info[i + 1]; ++ctr) {
					if (//dfr_analysis_info[iorder[ctr]] != dfr_analysis_info[iorder[ctr - 1]] || 	-->asegurado xq cada thread es 1 lvl
										//ivect_size[ctr] != ivect_size[ctr - 1] ||				-->asegurado xq cada thread es 1 vect_size
						filas_warp * ivect_size_h[ctr] >= 32 ||
						(ivect_size_h[ctr] == 0 && filas_warp == 32)) {
						filas_warp = 1;
						//local_ii[i]++;
						temp++;
						ii++;
					} else {
						filas_warp++;
					}
				}



				local_ii[i] = temp;

			}

			//		printf("NWARPS: %i\n",ii);
			exclusive_scan(local_ii, 7 * nLevs_dfr + 1);

		}


//CLK_START
		int n_warps = ii;
		int* ibase_row = (int*)calloc((n_warps + 1), sizeof(int));
		int* ivect_size_warp = (int*)calloc((n_warps + 1), sizeof(int));
		int* iwarp_lev = (int*)calloc((n_warps + 1), sizeof(int));
//CLK_STOP
//tiempo+=CLK_ELAPSED;

		//	int* inv_iorder = (int*)calloc(rows, sizeof(int));

		filas_warp = 1;
		ivect_size_warp[0] = ivect_size[0];
		ibase_row[0] = 0;
		//	inv_iorder[iorder_h[0]] = 0;

			// printf("\n analysis::7 \n"); fflush(0);
		if (mode == 0) {
			ii = 1;
			int ctr;
			for (ctr = 1; ctr < rows; ++ctr) {

				if (dfr_analysis_info_h[iorder_h[ctr]] != dfr_analysis_info_h[iorder_h[ctr - 1]] ||
					ivect_size_h[ctr] != ivect_size_h[ctr - 1] ||
					filas_warp * ivect_size_warp[ii - 1] >= 32 ||
					(ivect_size_warp[ii - 1] == 0 && filas_warp >= 32)) {

					ibase_row[ii] = ctr;
					ivect_size_warp[ii] = ivect_size_h[ctr];

					iwarp_lev[ii] = dfr_analysis_info_h[iorder_h[ctr]] - 1;


					filas_warp = 1;
					ii++;
				} else {
					filas_warp++;
				}

				// here inv_iorder stores which warp processes each row
			//	inv_iorder[iorder_h[ctr]] = ii - 1;
			}
			ibase_row[ii] = ctr;



		} else {	//mode =2




#pragma omp parallel for private(filas_warp,ii) shared(ibase_row,ivect_size_warp,ivect_size,iwarp_lev,dfr_analysis_info,iorder)//schedule(dynamic,1)             //Probar con static, es esperable que sea peor
			for (int i = 0; i < 7 * nLevs_dfr;++i) {                	//Chequear init	
				if (levels_warp_info[i] < levels_warp_info[i + 1]) {
					ii = 1 + local_ii[i];
					filas_warp = 1;                                  //Esto puede que vaya afuera del for
					int ctr = levels_warp_info[i] + 1;
					ivect_size_warp[ii - 1] = ivect_size_h[ctr - 1];


					for (; ctr < levels_warp_info[i + 1]; ++ctr) {
						if (//dfr_analysis_info[iorder[ctr]] != dfr_analysis_info[iorder[ctr - 1]] ||
							//ivect_size[ctr] != ivect_size[ctr - 1] ||
							filas_warp * ivect_size_warp[ii - 1] >= 32 ||
							(ivect_size_warp[ii - 1] == 0 && filas_warp >= 32)) {

							ibase_row[ii] = ctr;
							ivect_size_warp[ii] = ivect_size_h[ctr];

							iwarp_lev[ii] = dfr_analysis_info_h[iorder_h[ctr]] - 1;


							filas_warp = 1;
							ii++;
						} else {
							filas_warp++;
						}
					}
					//if(ii==4) printf("Will write %i\n",ctr);
					ibase_row[ii] = ctr;
				}
			}

			//		for(int i=0;i<7*nLevs_dfr+1;++i) printf("I: %i. level %i\n",i,levels_warp_info[i]);
			//		for(int i=0;i<7*nLevs_dfr+1;++i) printf("I: %i. Local_ii %i\n",i,local_ii[i]);
			//		for(int i=0;i<n_warps+1;i++)	printf("I: %i. Ibase: %i\n",i,ibase_row[i]);	
		}
		// ivect_size_warp[ii]=ivect_size[ctr];
		if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
			CLK_STOP;
			//Warp info
			fprintf(fp, ",%.2f", CLK_ELAPSED);
			CLK_START;
		}
		/*


				for (int i = 0; i < n_warps; ++i) {
					if (ivect_size_warp[i] < 0) {
						printf("ivect_size_warp[%d] < 0!!!!\n", ivect_size_warp[i]);
						exit(0);
					}

					if (ivect_size_warp[i] > 32) {
						printf("ivect_size_warp[%d] > 32!!!!\n", ivect_size_warp[i]);
						exit(0);
					}
				}

				if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
					CLK_STOP;
					//Check
					fprintf(fp, ",%.2f", CLK_ELAPSED);
					CLK_START;
				}

		*/
		/*Memory copy*/

		//CUDA_CHK(cudaMalloc((void**)&(current->inv_iorder), rows * sizeof(int)));
		//CUDA_CHK(cudaMemcpy(current->inv_iorder, inv_iorder, rows * sizeof(int), cudaMemcpyHostToDevice));

		current->n_warps = n_warps;

		// printf("\n analysis::8::1 n=%d\n",rows); fflush(0);



		CUDA_CHK(cudaMalloc((void**)&(current->ibase_row), (n_warps + 1) * sizeof(int)));
//CLK_START
		CUDA_CHK(cudaMemcpy(current->ibase_row, ibase_row, (n_warps + 1) * sizeof(int), cudaMemcpyHostToDevice));
//CLK_STOP
//tiempo+=CLK_ELAPSED;
		// printf("\n analysis::8::3 \n"); fflush(0);

		CUDA_CHK(cudaMalloc((void**)&(current->ivect_size_warp), n_warps * sizeof(int)));
//CLK_START
		CUDA_CHK(cudaMemcpy(current->ivect_size_warp, ivect_size_warp, n_warps * sizeof(int), cudaMemcpyHostToDevice));
//CLK_STOP
//tiempo+=CLK_ELAPSED;
		// printf("\n analysis::8::5 \n"); fflush(0);




		// printf("\n analysis::8::4 \n"); fflush(0);

		CUDA_CHK(cudaMalloc((void**)&(current->warp_lev), n_warps * sizeof(int)));
//CLK_START
		CUDA_CHK(cudaMemcpy(current->warp_lev, iwarp_lev, n_warps * sizeof(int), cudaMemcpyHostToDevice));
//CLK_STOP
//tiempo+=CLK_ELAPSED;


		// printf("\n analysis::9 \n"); fflush(0);


		// vector that stores the number of rows of each level
		CUDA_CHK(cudaMalloc((void**)&(current->lev_size), nLevs_dfr * sizeof(int)));
//CLK_START
		CUDA_CHK(cudaMemcpy(current->lev_size, thrust::raw_pointer_cast(/*ilevels*/ivects/*_aux*/.data()), nLevs_dfr * sizeof(int), cudaMemcpyHostToDevice));
//CLK_STOP
//tiempo+=CLK_ELAPSED;


		// vector of counters
		CUDA_CHK(cudaMalloc((void**)&(current->lev_ctr), nLevs_dfr * sizeof(int)));


		CUDA_CHK(cudaMalloc((void**)&(current->iorder), rows * sizeof(int)));
		//current->iorder = thrust::raw_pointer_cast(iorder.data());
		CUDA_CHK(cudaMemcpy(current->iorder, thrust::raw_pointer_cast(iorder.data()), rows * sizeof(int), cudaMemcpyDeviceToDevice));
		CUDA_CHK(cudaMalloc((void**)&(current->row_ctr), sizeof(int)));

		current->nlevs = nLevs_dfr;
		if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
			CLK_STOP;
			//Copy to device
			fprintf(fp, ",%.2f", CLK_ELAPSED);
			CLK_START;
		}
		//free(ilevels);


		// struct spts_times * kernel_times, * d_kernel_times;
		// kernel_times  = (struct spts_times *) malloc(n_warps * sizeof(struct spts_times));
		// CUDA_CHK( cudaMalloc( (void**) &d_kernel_times, n_warps * sizeof(struct spts_times) ) );

		//free(iorder);
		//free(inv_iorder);
//CLK_START
		free(ibase_row);
		free(ivect_size_warp);
		free(iwarp_lev);
//CLK_STOP
//tiempo+=CLK_ELAPSED;

}else

	if (mode == 1) {
		CUDA_CHK(cudaMalloc((void**)&(current->iorder), rows * sizeof(int)));
		//current->iorder = thrust::raw_pointer_cast(iorder.data());
//CLK_START
		CUDA_CHK(cudaMemcpy(current->iorder, thrust::raw_pointer_cast(iorder.data()), rows * sizeof(int), cudaMemcpyDeviceToDevice));
//CLK_STOP
//tiempo+=CLK_ELAPSED;

		CUDA_CHK(cudaMalloc((void**)&(current->row_ctr), sizeof(int)));

		current->nlevs = nLevs_dfr;
		if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
			CLK_STOP;
			//Copy to device
			fprintf(fp, ",%.2f", CLK_ELAPSED);
			CLK_START;
		}

	}



	CUDA_CHK(cudaFree(d_dfr_analysis_info));
	CUDA_CHK(cudaFree(d_is_solved));
	if (mode == 2) {
//CLK_START
		free(levels_warp_info);
//CLK_STOP
//tiempo+=CLK_ELAPSED;

	}

	//task_list.clear();
	//task_list.shrink_to_fit();
	dfr_analysis_info.clear();
	dfr_analysis_info.shrink_to_fit();
	nnz_row.clear();
	nnz_row.shrink_to_fit();
	vect_size.clear();
	vect_size.shrink_to_fit();
	lev.clear();
	lev.shrink_to_fit();
	//indices.clear();
	//indices.shrink_to_fit();
	ivects.clear();
	ivects.shrink_to_fit();
	//ivects_aux.clear();
	//ivects_aux.shrink_to_fit();

	//ilevels.clear();
	//ilevels.shrink_to_fit();
	iorder.clear();
	iorder.shrink_to_fit();
	ivect_size.clear();
	ivect_size.shrink_to_fit();
	//free(dfr_analysis_info);

	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		CLK_STOP;
		//Free
		fprintf(fp, ",%.2f\n", CLK_ELAPSED);
		fclose(fp);
	}

//printf("Memory time: %f\n",tiempo);
//fp = fopen("t_mem.csv", "a+");
//fprintf(fp, ",%.2f",tiempo);
//fclose(fp);
}


void multirow_analysis_base(dfr_analysis_info_t** mat, sp_mat_t* gpu_L, int mode) {
	CLK_INIT;
	FILE* fp;
	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		fp = fopen(LOG_FILE, "a+");
		CLK_START;
	}

	dfr_analysis_info_t* current = *mat;
	int n = gpu_L->nr;

	// printf("gpu_L->nr = %d\n", n );

	unsigned int* dfr_analysis_info;

	dfr_analysis_info = (unsigned int*)malloc(n * sizeof(int));

	unsigned int* d_dfr_analysis_info;
	int* d_is_solved;

	CUDA_CHK(cudaMalloc((void**)&(d_dfr_analysis_info), n * sizeof(int)))
		CUDA_CHK(cudaMalloc((void**)&(d_is_solved), n * sizeof(int)))

		int num_threads = WARP_PER_BLOCK * WARP_SIZE;
	// int num_threads = GPU_BLK;
	// int WARP_PER_BLOCK = GPU_BLK / WARP_SIZE;

	int grid = ceil((double)n * WARP_SIZE / (double)(num_threads * ROWS_PER_THREAD));

	CUDA_CHK(cudaMemset(d_is_solved, 0, n * sizeof(int)))
	CUDA_CHK(cudaMemset(d_dfr_analysis_info, 0, n * sizeof(int)))

	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		CLK_STOP;
		fprintf(fp, ",%.2f", CLK_ELAPSED);
		CLK_START;
	}



	kernel_analysis_L << < grid, num_threads, WARP_PER_BLOCK * sizeof(VALUE_TYPE) >> > (gpu_L->ia,
		gpu_L->ja,
		d_is_solved,
		n,
		d_dfr_analysis_info);


	cudaDeviceSynchronize();
	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		CLK_STOP;
		fprintf(fp, ",%.2f", CLK_ELAPSED);
		CLK_START;
	}
	CUDA_CHK(cudaMemcpy(dfr_analysis_info, d_dfr_analysis_info, n * sizeof(int), cudaMemcpyDeviceToHost))


		// for (int i = 0; i < 10; ++i)
		// {
		//     printf("dfr_analysis_info[%d] = %d\n", i, dfr_analysis_info[i] );
		// }

		if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
			CLK_STOP;
			fprintf(fp, ",%.2f", CLK_ELAPSED);
			CLK_START;
		}

	int nLevs_dfr = dfr_analysis_info[0];
	for (int i = 1; i < n; ++i) {
		nLevs_dfr = MAX(nLevs_dfr, dfr_analysis_info[i]);
	}

	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		CLK_STOP;
		fprintf(fp, ",%.2f", CLK_ELAPSED);
		CLK_START;
	}
	// printf("\n analysis::2 \n"); fflush(0);

	int* ilevels = (int*)calloc(nLevs_dfr, sizeof(int));
	int* ivects = (int*)calloc(7 * nLevs_dfr, sizeof(int));
	int* iorder = (int*)calloc(n, sizeof(int));
	int* ivect_size = (int*)calloc(n, sizeof(int));
	//	int* inv_iorder = (int*)calloc(n, sizeof(int));

		//CLK_STOP;
		//printf("CPU mallocs: %f\n", CLK_ELAPSED );
		//CLK_START;


	int* csrRowPtrL_tmp = (int*)malloc((n + 1) * sizeof(int));
	CUDA_CHK(cudaMemcpy(csrRowPtrL_tmp, gpu_L->ia, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost))

		// Count the number of rows on each level and each vector size


		for (int i = 0; i < n; i++) {
			int lev = dfr_analysis_info[i] - 1;
			int nnz_row = csrRowPtrL_tmp[i + 1] - csrRowPtrL_tmp[i] - 1;
			int vect_size;

			if (nnz_row == 0)
				vect_size = 6;
			else if (nnz_row == 1)
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

			ivects[7 * lev + vect_size]++;

			ilevels[lev]++;
		}



	// printf("\n analysis::4 \n"); fflush(0);

	// Performing a scan operation on this vector will yield
	// the starting position of each level in the final iorder
	// array, which is the final content of ivects.

	exclusive_scan(ivects, 7 * nLevs_dfr);
	//printf("nLevs_dfr %i\n", nLevs_dfr);
	int* levels_warp_info;

	if (mode == 2) {
		levels_warp_info = (int*)calloc(7 * nLevs_dfr + 1, sizeof(int));
		memcpy((void*)levels_warp_info, (void*)ivects, 7 * nLevs_dfr * sizeof(int));
		levels_warp_info[7 * nLevs_dfr] = n;
		for (int i = 0;i < 7 * nLevs_dfr + 1;++i) printf("I: %i. level %i\n", i, levels_warp_info[i]);

	}
	// Maintaining an offset variable for each level, assign
	// each node j to the iorder array the following way
	// iorder(ivects(idepth(j)) + offset(idepth(j))) = j
	// incrementing the offset by 1 afterwards.

		// printf("\n analysis::5 \n"); fflush(0);

	for (int i = 0; i < n; i++) {

		int idepth = dfr_analysis_info[i] - 1;
		int nnz_row = csrRowPtrL_tmp[i + 1] - csrRowPtrL_tmp[i] - 1;
		int vect_size;

		if (nnz_row == 0)
			vect_size = 6;
		else if (nnz_row == 1)
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

		iorder[ivects[7 * idepth + vect_size]] = i;
		ivect_size[ivects[7 * idepth + vect_size]] = (vect_size == 6) ? 0 : pow(2, vect_size);

		ivects[7 * idepth + vect_size]++;
	}




	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		CLK_STOP;
		fprintf(fp, ",%.2f", CLK_ELAPSED);
		CLK_START;
	}
	int* ibase_row;
	int* ivect_size_warp;
	int* iwarp_lev;
	int* local_ii;
	if (mode == 0 || mode == 2) {
		// printf("\n analysis::5 \n"); fflush(0);

		int ii = 1;
		int filas_warp;
		if (mode == 0) {
			filas_warp = 1;
			for (int ctr = 1; ctr < n; ++ctr) {

				if (dfr_analysis_info[iorder[ctr]] != dfr_analysis_info[iorder[ctr - 1]] ||
					ivect_size[ctr] != ivect_size[ctr - 1] ||
					filas_warp * ivect_size[ctr] >= 32 ||
					(ivect_size[ctr] == 0 && filas_warp == 32)) {

					filas_warp = 1;
					ii++;
				} else {
					filas_warp++;
				}
			}

		} else {	//mode =2
		//levels_warp_info

			local_ii = (int*)calloc(7 * nLevs_dfr + 1, sizeof(int));

			ii = 0;
			omp_set_num_threads(NUM_THREADS_OMP);
#pragma omp parallel for private(filas_warp) shared(ivect_size,local_ii) reduction(+: ii)
			for (int i = 0; i < 7 * nLevs_dfr;++i) {
				filas_warp = 1;                                  //Esto puede que vaya afuera del for
				int temp;

				if (levels_warp_info[i] < levels_warp_info[i + 1]) {

					ii += 1;
					//		local_ii[i] =1;
					temp = 1;
				} else {
					temp = 0;
					//		local_ii[i] =0;
				}
				for (int ctr = levels_warp_info[i] + 1; ctr < levels_warp_info[i + 1]; ++ctr) {
					if (//dfr_analysis_info[iorder[ctr]] != dfr_analysis_info[iorder[ctr - 1]] || 	-->asegurado xq cada thread es 1 lvl
										//ivect_size[ctr] != ivect_size[ctr - 1] ||				-->asegurado xq cada thread es 1 vect_size
						filas_warp * ivect_size[ctr] >= 32 ||
						(ivect_size[ctr] == 0 && filas_warp == 32)) {
						filas_warp = 1;
						//local_ii[i]++;
						temp++;
						ii++;
					} else {
						filas_warp++;
					}
				}
				local_ii[i] = temp;

			}

			//		printf("NWARPS: %i\n",ii);
			exclusive_scan(local_ii, 7 * nLevs_dfr + 1);

		}
		// printf("\n analysis::6 \n"); fflush(0);


		int n_warps = ii;
		ibase_row = (int*)calloc((n_warps + 1), sizeof(int));
		ivect_size_warp = (int*)calloc((n_warps + 1), sizeof(int));
		iwarp_lev = (int*)calloc((n_warps + 1), sizeof(int));


		filas_warp = 1;
		ivect_size_warp[0] = ivect_size[0];
		ibase_row[0] = 0;
		//	inv_iorder[iorder[0]] = 0;

			// printf("\n analysis::7 \n"); fflush(0);

		if (mode == 0) {

			ii = 1;
			int ctr;
			for (ctr = 1; ctr < n; ++ctr) {
				if (dfr_analysis_info[iorder[ctr]] != dfr_analysis_info[iorder[ctr - 1]] ||
					ivect_size[ctr] != ivect_size[ctr - 1] ||
					filas_warp * ivect_size_warp[ii - 1] >= 32 ||
					(ivect_size_warp[ii - 1] == 0 && filas_warp >= 32)) {

					ibase_row[ii] = ctr;
					ivect_size_warp[ii] = ivect_size[ctr];

					iwarp_lev[ii] = dfr_analysis_info[iorder[ctr]] - 1;


					filas_warp = 1;
					ii++;
				} else {
					filas_warp++;
				}

				// here inv_iorder stores which warp processes each row
			//	inv_iorder[iorder[ctr]] = ii - 1;
			}
			ibase_row[ii] = ctr;
		} else {	//mode =2




#pragma omp parallel for private(filas_warp,ii) shared(ibase_row,ivect_size_warp,ivect_size,iwarp_lev,dfr_analysis_info,iorder)//schedule(dynamic,1)             //Probar con static, es esperable que sea peor
			for (int i = 0; i < 7 * nLevs_dfr;++i) {                	//Chequear init	
				if (levels_warp_info[i] < levels_warp_info[i + 1]) {
					ii = 1 + local_ii[i];
					filas_warp = 1;                                  //Esto puede que vaya afuera del for
					int ctr = levels_warp_info[i] + 1;
					ivect_size_warp[ii - 1] = ivect_size[ctr - 1];


					for (; ctr < levels_warp_info[i + 1]; ++ctr) {
						if (//dfr_analysis_info[iorder[ctr]] != dfr_analysis_info[iorder[ctr - 1]] ||
							//ivect_size[ctr] != ivect_size[ctr - 1] ||
							filas_warp * ivect_size_warp[ii - 1] >= 32 ||
							(ivect_size_warp[ii - 1] == 0 && filas_warp >= 32)) {

							ibase_row[ii] = ctr;
							ivect_size_warp[ii] = ivect_size[ctr];

							iwarp_lev[ii] = dfr_analysis_info[iorder[ctr]] - 1;


							filas_warp = 1;
							ii++;
						} else {
							filas_warp++;
						}
					}
					//if(ii==4) printf("Will write %i\n",ctr);
					ibase_row[ii] = ctr;
				}
			}

			//		for(int i=0;i<7*nLevs_dfr+1;++i) printf("I: %i. level %i\n",i,levels_warp_info[i]);
			//		for(int i=0;i</*7*nLevs_dfr+1*/20;++i) printf("I: %i. Local_ii %i\n",i,local_ii[i]);
			//		for(int i=0;i<14;i++)	printf("I: %i. Ibase: %i\n",i,ibase_row[i]);	
		}

		// ivect_size_warp[ii]=ivect_size[ctr];

		if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
			CLK_STOP;
			fprintf(fp, ",%.2f", CLK_ELAPSED);
			CLK_START;
		}

		// printf("\n analysis::8 %d %d \n", n, n_warps); fflush(0);

	/*		for (int i = 0; i < n_warps; ++i) {

			if (ivect_size_warp[i] < 0) {
				printf("ivect_size_warp[%d] < 0!!!!\n", ivect_size_warp[i]);
				exit(0);
			}

			if (ivect_size_warp[i] > 32) {
				printf("ivect_size_warp[%d] > 32!!!!\n", ivect_size_warp[i]);
				exit(0);
			}
		}

		if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
			CLK_STOP;
			fprintf(fp, ",%.2f", CLK_ELAPSED);
			CLK_START;
		}

	*/
		current->n_warps = n_warps;



		CUDA_CHK(cudaMalloc((void**)&(current->iorder), n * sizeof(int)));
		CUDA_CHK(cudaMemcpy(current->iorder, iorder, n * sizeof(int), cudaMemcpyHostToDevice));


		// 	printf("\n analysis::8::2 \n"); fflush(0);


		CUDA_CHK(cudaMalloc((void**)&(current->ibase_row), (n_warps + 1) * sizeof(int)));
		CUDA_CHK(cudaMemcpy(current->ibase_row, ibase_row, (n_warps + 1) * sizeof(int), cudaMemcpyHostToDevice));

		// printf("\n analysis::8::3 \n"); fflush(0);

		CUDA_CHK(cudaMalloc((void**)&(current->ivect_size_warp), n_warps * sizeof(int)));
		CUDA_CHK(cudaMemcpy(current->ivect_size_warp, ivect_size_warp, n_warps * sizeof(int), cudaMemcpyHostToDevice));

		// printf("\n analysis::8::5 \n"); fflush(0);

		//	CUDA_CHK( cudaMalloc( (void**) &(current->row_ctr), sizeof(int) ) );


	 // printf("\n analysis::8::4 \n"); fflush(0);

		CUDA_CHK(cudaMalloc((void**)&(current->warp_lev), n_warps * sizeof(int)));
		CUDA_CHK(cudaMemcpy(current->warp_lev, iwarp_lev, n_warps * sizeof(int), cudaMemcpyHostToDevice));


		// printf("\n analysis::9 \n"); fflush(0);


		// vector that stores the number of rows of each level
		CUDA_CHK(cudaMalloc((void**)&(current->lev_size), nLevs_dfr * sizeof(int)));
		CUDA_CHK(cudaMemcpy(current->lev_size, ilevels, nLevs_dfr * sizeof(int), cudaMemcpyHostToDevice));

		// vector of counters
		CUDA_CHK(cudaMalloc((void**)&(current->lev_ctr), nLevs_dfr * sizeof(int)));

		current->nlevs = nLevs_dfr;
	}

	CUDA_CHK(cudaMalloc((void**)&(current->row_ctr), sizeof(int)));
	CUDA_CHK(cudaMalloc((void**)&(current->iorder), n * sizeof(int)));
	CUDA_CHK(cudaMemcpy(current->iorder, iorder, n * sizeof(int), cudaMemcpyHostToDevice));

	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		CLK_STOP;
		fprintf(fp, ",%.2f", CLK_ELAPSED);
		CLK_START;
	}
	//free(ilevels);


	// struct spts_times * kernel_times, * d_kernel_times;
	// kernel_times  = (struct spts_times *) malloc(n_warps * sizeof(struct spts_times));
	// CUDA_CHK( cudaMalloc( (void**) &d_kernel_times, n_warps * sizeof(struct spts_times) ) );

	// CUDA_CHK( cudaFree(d_dfr_analysis_info) ) 
	// CUDA_CHK( cudaFree(d_is_solved) ) 

	if (mode == 0 || mode == 2) {
		free(ibase_row);
		free(ivect_size_warp);
		free(iwarp_lev);
	}
	if (mode == 2) {
		free(levels_warp_info);
		free(local_ii);
	}

	free(ivect_size);
	free(csrRowPtrL_tmp);
	free(ivects);
	free(dfr_analysis_info);
	free(iorder);
	free(ilevels);
	//free(inv_iorder);
	/*
	free(ivects);
	free(ivects_size);*/
	CUDA_CHK(cudaFree(d_dfr_analysis_info));
	CUDA_CHK(cudaFree(d_is_solved));

	if (!TIMERS_SOLVERS && PRINT_TIME_ANALYSIS && LOG_FILE != "NONE") {
		CLK_STOP;
		fprintf(fp, ",%.2f\n", CLK_ELAPSED);
		fclose(fp);
	}
}

/*
void multirow_analysis_adv(dfr_analysis_info_t** mat, sp_mat_t* gpu_L) {

	dfr_analysis_info_t* current = *mat;

	int n = gpu_L->nr;

	// printf("gpu_L->nr = %d\n", n );

	int* dfr_analysis_info;

	dfr_analysis_info = (int*)malloc(n * sizeof(int));

	int* d_dfr_analysis_info;
	int* d_is_solved;

	CUDA_CHK(cudaMalloc((void**)&(d_dfr_analysis_info), n * sizeof(int)));
	CUDA_CHK(cudaMalloc((void**)&(d_is_solved), n * sizeof(int)));

	int num_threads = WARP_PER_BLOCK * WARP_SIZE;
	// int num_threads = GPU_BLK;
	// int WARP_PER_BLOCK = GPU_BLK / WARP_SIZE;

	int grid = ceil((double)n * WARP_SIZE / (double)(num_threads * ROWS_PER_THREAD));

	CUDA_CHK(cudaMemset(d_is_solved, 0, n * sizeof(int)));
	CUDA_CHK(cudaMemset(d_dfr_analysis_info, 0, n * sizeof(int)));

	kernel_analysis_L << < grid, num_threads, WARP_PER_BLOCK * sizeof(VALUE_TYPE) >> > (gpu_L->ia,
		gpu_L->ja,
		d_is_solved,
		n,
		d_dfr_analysis_info);

	CUDA_CHK(cudaMemcpy(dfr_analysis_info, d_dfr_analysis_info, n * sizeof(int), cudaMemcpyDeviceToHost));


	// for (int i = 0; i < 10; ++i)
	// {
	//     printf("dfr_analysis_info[%d] = %d\n", i, dfr_analysis_info[i] );
	// }

	int nLevs_dfr = dfr_analysis_info[0];
	for (int i = 1; i < n; ++i)
	{
		nLevs_dfr = MAX(nLevs_dfr, dfr_analysis_info[i]);
	}

	int* csrRowPtrL_tmp = (int*)malloc((n + 1) * sizeof(int));

	// printf("\n analysis::1 \n"); fflush(0);

	CUDA_CHK(cudaMemcpy(csrRowPtrL_tmp, gpu_L->ia, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));


	// printf("\n analysis::2 \n"); fflush(0);

	int* ilevels = (int*)calloc(nLevs_dfr, sizeof(int));
	int* ivects = (int*)calloc(7 * nLevs_dfr, sizeof(int));
	int* iorder = (int*)calloc(n, sizeof(int));
	int* ivect_size = (int*)calloc(n, sizeof(int));
	int* inv_iorder = (int*)calloc(n, sizeof(int));


	// Count the number of rows on each level and each vector size

	for (int i = 0; i < n; i++) {

		int lev = dfr_analysis_info[i] - 1;
		int nnz_row = csrRowPtrL_tmp[i + 1] - csrRowPtrL_tmp[i] - 1;
		int vect_size;

		if (nnz_row == 0)
			vect_size = 6;
		else if (nnz_row == 1)
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

		ivects[7 * lev + vect_size]++;

		ilevels[lev]++;
	}

	// printf("\n analysis::4 \n"); fflush(0);

	// Performing a scan operation on this vector will yield
	// the starting position of each level in the final iorder
	// array, which is the final content of ivects.

	exclusive_scan(ivects, 7 * nLevs_dfr);

	// Maintaining an offset variable for each level, assign
	// each node j to the iorder array the following way
	// iorder(ivects(idepth(j)) + offset(idepth(j))) = j
	// incrementing the offset by 1 afterwards.

		// printf("\n analysis::5 \n"); fflush(0);

	for (int i = 0; i < n; i++) {

		int idepth = dfr_analysis_info[i] - 1;
		int nnz_row = csrRowPtrL_tmp[i + 1] - csrRowPtrL_tmp[i] - 1;
		int vect_size;

		if (nnz_row == 0)
			vect_size = 6;
		else if (nnz_row == 1)
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

		iorder[ivects[7 * idepth + vect_size]] = i;
		ivect_size[ivects[7 * idepth + vect_size]] = (vect_size == 6) ? 0 : pow(2, vect_size);

		ivects[7 * idepth + vect_size]++;
	}


	// printf("\n analysis::5 \n"); fflush(0);

	int ii = 1;
	int filas_warp = 1;
	for (int ctr = 1; ctr < n; ++ctr) {

		if (dfr_analysis_info[iorder[ctr]] != dfr_analysis_info[iorder[ctr - 1]] ||
			ivect_size[ctr] != ivect_size[ctr - 1] ||
			filas_warp * ivect_size[ctr] >= 32 ||
			(ivect_size[ctr] == 0 && filas_warp == 32)) {

			filas_warp = 1;
			ii++;
		} else {
			filas_warp++;
		}
	}


	// printf("\n analysis::6 \n"); fflush(0);

	int n_warps = ii;
	int* ibase_row = (int*)calloc((n_warps + 1), sizeof(int));
	int* ivect_size_warp = (int*)calloc((n_warps + 1), sizeof(int));
	int* iwarp_lev = (int*)calloc((n_warps + 1), sizeof(int));

	int* bitmap_prod = (int*)calloc((n_warps + 1), sizeof(int));
	int* bitmap_cons = (int*)calloc((n_warps + 1), sizeof(int));

	filas_warp = 1;
	ivect_size_warp[0] = ivect_size[0];
	ibase_row[0] = 0;
	inv_iorder[iorder[0]] = 0;

	// printf("\n analysis::7 \n"); fflush(0);

	ii = 1;
	int ctr;
	for (ctr = 1; ctr < n; ++ctr) {
		if (dfr_analysis_info[iorder[ctr]] != dfr_analysis_info[iorder[ctr - 1]] ||
			ivect_size[ctr] != ivect_size[ctr - 1] ||
			filas_warp * ivect_size_warp[ii - 1] >= 32 ||
			(ivect_size_warp[ii - 1] == 0 && filas_warp >= 32)) {

			ibase_row[ii] = ctr;
			ivect_size_warp[ii] = ivect_size[ctr];

			iwarp_lev[ii] = dfr_analysis_info[iorder[ctr]] - 1;


			filas_warp = 1;
			ii++;
		} else {
			filas_warp++;
		}

		// here inv_iorder stores which warp processes each row
		inv_iorder[iorder[ctr]] = ii - 1;
	}
	ibase_row[ii] = ctr;
	// ivect_size_warp[ii]=ivect_size[ctr];



	// printf("\n analysis::8 %d %d \n", n, n_warps); fflush(0);

	for (int i = 0; i < n_warps; ++i)
	{

		if (ivect_size_warp[i] < 0) {
			printf("ivect_size_warp[%d] < 0!!!!\n", ivect_size_warp[i]);
			exit(0);
		}

		if (ivect_size_warp[i] > 32) {
			printf("ivect_size_warp[%d] > 32!!!!\n", ivect_size_warp[i]);
			exit(0);
		}
	}

	// printf("n_warps %d\n", n_warps);

	// printf("Orden primeros 100\n");
	// for(int i = 0; i < 100; i++ ){ 
	//     printf("i=%d iorder=%d inv_iorder=%d level=%d nnz_row=%d vect_size=%d ibase_row=%d\n ", i, 
	//                                                                                             iorder[i], inv_iorder[i], 
	//                                                                                             dfr_analysis_info[iorder[i]], 
	//                                                                                             csrRowPtrL_tmp[ iorder[i]+1 ] - csrRowPtrL_tmp[ iorder[i] ], 
	//                                                                                             ivect_size[ i ], 
	//                                                                                             ibase_row[i]);
	// }

	// int vect_idx = 3;
	// int vect_size = 4;

	// int my_mask = pow(2,vect_size)-1; 
	// my_mask <<= vect_size * (32/vect_size - vect_idx - 1);

	// printf("vect_idx= %d, vect_size=%d, my_mask = %x \n", vect_idx, vect_size, my_mask);

	// exit(0);

	CUDA_CHK(cudaMalloc((void**)&(current->inv_iorder), n * sizeof(int)));
	CUDA_CHK(cudaMemcpy(current->inv_iorder, inv_iorder, n * sizeof(int), cudaMemcpyHostToDevice));

	current->n_warps = n_warps;

	// printf("\n analysis::8::1 n=%d\n",n); fflush(0);

	CUDA_CHK(cudaMalloc((void**)&(current->iorder), n * sizeof(int)));
	CUDA_CHK(cudaMemcpy(current->iorder, iorder, n * sizeof(int), cudaMemcpyHostToDevice));


	// printf("\n analysis::8::2 \n"); fflush(0);


	CUDA_CHK(cudaMalloc((void**)&(current->ibase_row), (n_warps + 1) * sizeof(int)));
	CUDA_CHK(cudaMemcpy(current->ibase_row, ibase_row, (n_warps + 1) * sizeof(int), cudaMemcpyHostToDevice));

	// printf("\n analysis::8::3 \n"); fflush(0);

	CUDA_CHK(cudaMalloc((void**)&(current->ivect_size_warp), n_warps * sizeof(int)));
	CUDA_CHK(cudaMemcpy(current->ivect_size_warp, ivect_size_warp, n_warps * sizeof(int), cudaMemcpyHostToDevice));

	// printf("\n analysis::8::5 \n"); fflush(0);

	CUDA_CHK(cudaMalloc((void**)&(current->row_ctr), sizeof(int)));


	// printf("\n analysis::8::4 \n"); fflush(0);

	CUDA_CHK(cudaMalloc((void**)&(current->warp_lev), n_warps * sizeof(int)));
	CUDA_CHK(cudaMemcpy(current->warp_lev, iwarp_lev, n_warps * sizeof(int), cudaMemcpyHostToDevice));


	// printf("\n analysis::9 \n"); fflush(0);


	// vector that stores the number of rows of each level
	CUDA_CHK(cudaMalloc((void**)&(current->lev_size), nLevs_dfr * sizeof(int)));
	CUDA_CHK(cudaMemcpy(current->lev_size, ilevels, nLevs_dfr * sizeof(int), cudaMemcpyHostToDevice));

	// vector of counters
	CUDA_CHK(cudaMalloc((void**)&(current->lev_ctr), nLevs_dfr * sizeof(int)));

	current->nlevs = nLevs_dfr;


	free(ilevels);


	// struct spts_times * kernel_times, * d_kernel_times;
	// kernel_times  = (struct spts_times *) malloc(n_warps * sizeof(struct spts_times));
	// CUDA_CHK( cudaMalloc( (void**) &d_kernel_times, n_warps * sizeof(struct spts_times) ) );

	CUDA_CHK(cudaFree(d_dfr_analysis_info));
	CUDA_CHK(cudaFree(d_is_solved));


	free(iorder);
	free(inv_iorder);
	free(ibase_row);
	free(ivect_size_warp);
	free(dfr_analysis_info);
}*/
