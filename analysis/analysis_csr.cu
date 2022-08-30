#include "common.h"
#include "dfr_syncfree.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <cooperative_groups.h>
#define NUM_THREADS_OMP 4
#define ANALYSIS_COOP 0
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
	__device__
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

template<int tile_size >
__global__ void kernel_analysis_L_coop_groups(const int* __restrict__ row_ptr,
	const int* __restrict__ col_idx,
	volatile int* is_solved, int n,
	int* dfr_analysis_info,
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
	for (int i = tile_size / 2; i >= 1; i /= 2) {
		int aux = myTile.shfl_down(my_level, i);
		my_level = max(my_level, aux);
	}

	if (lne == 0) {
		//escribo en el resultado
		s_info[local_tile_id] = 1 + my_level;
		s_is_solved[local_tile_id] = 1;
		dfr_analysis_info[groupId] = 1 + my_level;
		__threadfence();
		is_solved[groupId] = 1;
	}
}

template __global__ void kernel_analysis_L_coop_groups<1>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved, int n, int* dfr_analysis_info, int* group_id_counter);
template __global__ void kernel_analysis_L_coop_groups<2>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved, int n, int* dfr_analysis_info, int* group_id_counter);
template __global__ void kernel_analysis_L_coop_groups<4>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved, int n, int* dfr_analysis_info, int* group_id_counter);
template __global__ void kernel_analysis_L_coop_groups<8>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved, int n, int* dfr_analysis_info, int* group_id_counter);
template __global__ void kernel_analysis_L_coop_groups<16>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved, int n, int* dfr_analysis_info, int* group_id_counter);
template __global__ void kernel_analysis_L_coop_groups<32>(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, volatile int* is_solved, int n, int* dfr_analysis_info, int* group_id_counter);

// in-place exclusive scan
void exclusive_scan(int* input, int length){
	if (length == 0 || length == 1) return;

	int old_val, new_val;
	old_val = input[0];
	input[0] = 0;
	for (int i = 1; i < length; i++){
		new_val = input[i];
		input[i] = old_val + input[i - 1];
		old_val = new_val;
	}
}

void aux_call_kernel_analysis_L_coop_groups(int grid, int num_threads,  sp_mat_t* gpu_L, 
										   int* d_is_solved, int rows, int* d_dfr_analysis_info, int average) {
	int* group_id_counter;
	CUDA_CHK(cudaMalloc((void**)&(group_id_counter), sizeof(int)));
	CUDA_CHK(cudaMemsetAsync(group_id_counter, 0, sizeof(int)));
	
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

//Calcula warp base y warp size
__global__ void kernel_base(int* buckets_off, int* rows_off, int* warp_base, int* warp_size){
	int off = buckets_off[blockIdx.x], lst=buckets_off[blockIdx.x+1];
	int cant = lst -off;
	int size;

	if(blockIdx.x== gridDim.x-1 && threadIdx.x==0) warp_base[lst] = rows_off[gridDim.x];

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
}

void multirow_analysis_base_GPU(dfr_analysis_info_t** mat, sp_mat_t* gpu_L){
	FILE* fp;

	dfr_analysis_info_t* current = *mat;

	//n es el número de filas
	int rows = gpu_L->nr;
	int nnz = gpu_L->nnz;
	int average = nnz/rows;
	int* d_dfr_analysis_info;
	int* d_is_solved;

	CUDA_CHK(cudaMalloc((void**)&(d_dfr_analysis_info), rows * sizeof(int)));
	CUDA_CHK(cudaMalloc((void**)&(d_is_solved), rows * sizeof(int)));

	int num_threads = WARP_PER_BLOCK * WARP_SIZE;
	int grid = ceil((double)rows * WARP_SIZE / (double)(num_threads * ROWS_PER_THREAD));

	CUDA_CHK(cudaMemset(d_is_solved, 0, rows * sizeof(int)));
	CUDA_CHK(cudaMemset(d_dfr_analysis_info, 0, rows * sizeof(int)));

	int shared_size = WARP_PER_BLOCK * sizeof(VALUE_TYPE) * (WARP_SIZE);
	
	aux_call_kernel_analysis_L_coop_groups(grid,num_threads,gpu_L,d_is_solved,rows,d_dfr_analysis_info,average);
	
	cudaDeviceSynchronize();

	thrust::device_ptr<int> temp_ptr(d_dfr_analysis_info);
	thrust::device_vector<int> dfr_analysis_info(temp_ptr, temp_ptr + rows);

	int* dfr_analysis_info_h = (int*)calloc(rows, sizeof(int));
	CUDA_CHK(cudaMemcpy(dfr_analysis_info_h, d_dfr_analysis_info, rows * sizeof(int), cudaMemcpyDeviceToHost));
	
	int nLevs_dfr = *thrust::max_element(dfr_analysis_info.begin(), dfr_analysis_info.end());

	thrust::device_ptr<int> d_ptr(gpu_L->ia);
	//nnz = csr row pointer
	thrust::device_vector<int> nnz_row(d_ptr, d_ptr + rows + 1);

	inverse_subst sub;
	thrust::device_vector<int> vect_size(rows);
	//vect_size[i] = nnz_row[i-1] 
	thrust::copy(nnz_row.begin() + 1, nnz_row.begin() + rows + 1, vect_size.begin());
	//nnz_row[i] = nnz of row i without diag elem
	thrust::transform(nnz_row.begin(), nnz_row.begin() + rows, vect_size.begin(), nnz_row.begin(), sub);


	vect_size_calculator vcc;
	//vect_size = ceil(sqrt(nnz)), 6 for rows with only diag element
	thrust::transform(nnz_row.begin(), nnz_row.begin() + rows, vect_size.begin(), vcc);

	thrust::device_vector<int> lev(rows);
	subst_one minus_one;
	thrust::transform(dfr_analysis_info.begin(), dfr_analysis_info.begin() + rows, lev.begin(), minus_one);


	get_index_from_size id_ivects;
	//dfr_analysis_info[i] = index of the counter for the pair(lev(row_i), size(row_i)) = 7*lev+vect_size
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

	thrust::counting_iterator<int> iter(0);
	thrust::copy(iter, iter + rows, iorder.begin());
	thrust::stable_sort_by_key(dfr_analysis_info.begin(), dfr_analysis_info.begin() + rows, iorder.begin());
	vect_size_from_index calc_size;
	thrust::transform(dfr_analysis_info.begin(), dfr_analysis_info.begin() + rows, ivect_size.begin(), calc_size);

	//Termina T5
	dfr_analysis_info.resize(7*nLevs_dfr+1);
	thrust::device_vector<int> warps_per_bucks(7 * nLevs_dfr+1);
	//copy iter to warps_per_bucks
	//Dfr_analysis_info tiene size (tamaño de fila) de cada bucket o -1 en caso de que sea 0 
	get_size_from_pos sz;
	thrust::transform(iter, iter+7*nLevs_dfr, dfr_analysis_info.begin(),sz);


	get_warps_per_bucks wpb;

	thrust::transform(dfr_analysis_info.begin(), dfr_analysis_info.begin()+7*nLevs_dfr,buckets.begin(),warps_per_bucks.begin(),wpb);
	//Calcular buck off
	thrust::exclusive_scan(warps_per_bucks.begin(),warps_per_bucks.begin()+7*nLevs_dfr+1, warps_per_bucks.begin());
	int num_warps = warps_per_bucks[7*nLevs_dfr];

	thrust::exclusive_scan(buckets.begin(),buckets.begin()+7*nLevs_dfr+1,buckets.begin());
	
	CUDA_CHK(cudaMalloc((void**)&(current->ibase_row), (num_warps + 1) * sizeof(int)));

	CUDA_CHK(cudaMalloc((void**)&(current->ivect_size_warp), num_warps * sizeof(int)));

	kernel_base<<<7*nLevs_dfr,32>>>(thrust::raw_pointer_cast(warps_per_bucks.data()), thrust::raw_pointer_cast(buckets.data()), current->ibase_row, current->ivect_size_warp);

	cudaDeviceSynchronize();
	CUDA_CHK(cudaMalloc((void**)&(current->iorder), rows * sizeof(int)));
	CUDA_CHK(cudaMemcpy(current->iorder, thrust::raw_pointer_cast(iorder.data()), rows * sizeof(int), cudaMemcpyDeviceToDevice));
	CUDA_CHK(cudaMalloc((void**)&(current->row_ctr), sizeof(int)));

	current->nlevs = nLevs_dfr;
	current->n_warps = num_warps;

	map.clear();
    map.shrink_to_fit();
} 


