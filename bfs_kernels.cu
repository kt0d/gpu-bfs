#include "common.h"
#include "bfs_kernels.cuh"

#include <stdio.h>

__global__ void quadratic_bfs(const int n, const int* row_offset, const int* column_index, int*const distance, const int iteration, bool*const done)
{
	// Calculate corresponding vertex
	const int global_tid = blockIdx.x*blockDim.x + threadIdx.x;

	if(global_tid < n && distance[global_tid] == iteration)
	{
		bool local_done=true;
		for(int offset = row_offset[global_tid]; offset < row_offset[global_tid+1]; offset++)
		{
			int j = column_index[offset];
			if(distance[j] > iteration+1)
			{
				distance[j]=iteration+1;
				local_done=false;
			}
		}
		if(!local_done)
			*done=local_done;
	}
}

__global__ void linear_bfs(const int n, const int* row_offset, const int*const column_index, int*const distance, const int iteration,const int*const in_queue,const int*const in_queue_count, int*const out_queue, int*const out_queue_count)
{
	// Calculate corresponding vertex in queue
	const int global_tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(global_tid < *in_queue_count) 
	{
		// Get vertex from the queue
		int v = in_queue[global_tid];
		for(int offset = row_offset[v]; offset < row_offset[v+1]; offset++)
		{
			int j = column_index[offset];
			if(distance[j] == bfs::infinity)
			{
				distance[j]=iteration+1;
				// Locekd enqueue
				int ind = atomicAdd(out_queue_count,1);
				out_queue[ind]=j;
			}
		}
	}

}

__device__ bool warp_cull(volatile int scratch[WARPS][HASH_RANGE], const int v)
{
	const int hash = v & (HASH_RANGE-1);
	const int warp_id = threadIdx.x / WARP_SIZE;
	//const int index = warp_id*HASH_RANGE + hash;

	// Threads without valid vertex provide -1 as v. They must enter this function, because they are needed for __syncwarp (which is only useful for Volta arch)
	if (v != -1)
		scratch[warp_id][hash]= v;
	__syncwarp();
	const int retrieved = scratch[warp_id][hash];
	if (retrieved == v)
	{
		// Vie to be the only thread in warp inspecting vertex v
		scratch[warp_id][hash] = threadIdx.x;
	}
	__syncwarp();
	if(v == -1)
		return false;
	if (retrieved == v && scratch[warp_id][hash] != threadIdx.x)
	{
		// Some other thread has this vertex
		return true;
	}
	return false;
}

__device__ bool history_cull()
{
	//TODO
	return false;
}

__device__ int2 block_prefix_sum(const int val)
{
	// Heavily inspired/copied from sample "shfl_scan" provied by NVIDIA
	// Block-wide prefix sum using shfl intrinsic
	volatile __shared__ int sums[WARPS];
	int value = val;

	const int lane_id = threadIdx.x % WARP_SIZE;
	const int warp_id = threadIdx.x / WARP_SIZE;

	// Warp-wide prefix sums
#pragma unroll
	for(int i = 1; i <= WARP_SIZE; i <<= 1)
	{
		const unsigned int mask = 0xffffffff;
		const int n = __shfl_up_sync(mask, value, i, WARP_SIZE);
		if (lane_id >= i)
			value += n;
	}

	// Write warp total to shared array
	if (threadIdx.x % WARP_SIZE == WARP_SIZE- 1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();

	// Prefix sum of warp sums
	if (warp_id == 0 && lane_id < WARPS)
	{
		int warp_sum = sums[lane_id];
		const unsigned int mask = (1 << (WARPS)) - 1;
#pragma unroll
		for (int i = 1; i <= WARPS; i <<= 1)
		{
			const int n = __shfl_up_sync(mask, warp_sum, i, WARPS);
			if (lane_id >= i)
				warp_sum += n;
		}

		sums[lane_id] = warp_sum;
	}

	__syncthreads();

	// Add total sum of previous warps to current element
	if (warp_id > 0)
	{
		const int block_sum = sums[warp_id-1];
		value += block_sum;
	}

	int2 result;
	// Subtract value given by thread to get exclusive prefix sum
	result.x = value - val;
	// Get total sum
	result.y = sums[WARPS-1];
	return result; 
}

__device__ bool status_lookup(int * const distance,const cudaSurfaceObject_t bitmask_surf, const int neighbor)
{
	// Just check status directly if bitmask is unavailable
	if (bitmask_surf == 0)
		return distance[neighbor] == bfs::infinity;
	bool not_visited = false;

	const unsigned int neighbor_mask = (1 << (neighbor % (8 * sizeof(unsigned int))));
	unsigned int mask = 0;
	const int count = neighbor / (8 * sizeof(unsigned int));
	surf1Dread(&mask, bitmask_surf, count* 4);
	if(mask & neighbor_mask )
	{
		return false;
	}

	not_visited = distance[neighbor] == bfs::infinity;

	if(not_visited)
	{
		// Update bitmask
		mask |= neighbor_mask;
		surf1Dwrite(mask,bitmask_surf,count * 4);	
	}

	return not_visited;
}

__device__ void block_coarse_grained_gather(const int* const column_index, int* const distance, cudaSurfaceObject_t bitmask_surf, const int iteration, int * const out_queue, int* const out_queue_count,int r, int r_end)
{
	volatile __shared__ int comm[3];
	while(__syncthreads_or(r_end-r))
	{
		// Vie for control of blokc
		if(r_end-r)
			comm[0] = threadIdx.x;
		__syncthreads();
		if(comm[0] == threadIdx.x)
		{
			// If won, share your range to entire block
			comm[1] = r;
			comm[2] = r_end;
			r = r_end;
		}
		__syncthreads();
		int r_gather = comm[1] + threadIdx.x;
		const int r_gather_end = comm[2];
		const int total = comm[2] - comm[1];
		int block_progress = 0;
		while((total - block_progress) > 0)
		{
			int neighbor = -1;
			bool is_valid = false;
			if (r_gather < r_gather_end)
			{
				neighbor = column_index[r_gather];
				// Look up status
				is_valid = status_lookup(distance,bitmask_surf, neighbor);
				if(is_valid)
				{
					// Update label
					distance[neighbor] = iteration + 1;
				}
			}
			// Obtain offset in queue by prefix sum
			const int2 queue_offset = block_prefix_sum(is_valid?1:0);
			volatile __shared__ int base_offset[1];
			// Obtain base enqueue offset
			if(threadIdx.x == 0)
				base_offset[0] = atomicAdd(out_queue_count,queue_offset.y);
			__syncthreads();
			// Write to queue
			if (is_valid)
				out_queue[base_offset[0]+queue_offset.x] = neighbor;

			r_gather += BLOCK_SIZE;
			block_progress+= BLOCK_SIZE;
			__syncthreads();
		}
	}
}

/*
   __device__ void warp_coarse_grained_gather(const int* const column_index, int* const distance, const int iteration, int * const out_queue, int* const out_queue_count,int r, int r_end)
{
	volatile __shared__ int comm[WARPS][3];
	const int thread_id = threadIdx.x;
	const int lane_id = threadIdx.x % WARP_SIZE;
	const int warp_id = threadIdx.x / WARP_SIZE;
	while(__any_sync(r_end-r))
	{
		if(r_end-r)
			comm[warp_id][0] = lane_id;
		__syncwarp();
		if(comm[warp_id][0] == thread_id)
		{
			comm[warp_id][1] = r;
			comm[warp_id][2] = r_end;
			r = r_end;
		}
		__syncwarp();
		int r_gather = comm[warp_id][1] + lane_id;
		const int r_gather_end = comm[warp_id][2];
		int warp_progress = 0;
		const int total = comm[2] - comm[1];
		while(r_gather < r_gather_end)
		{

		}
		while((total - block_progress) > 0)
		{
			int neighbor = -1;
			bool is_valid = false;
			if (r_gather < r_gather_end)
			{
				neighbor = column_index[r_gather];
				// Look up status
				is_valid = status_lookup(distance, neighbor);
				if(is_valid)
				{
					// Update label
					distance[neighbor] = iteration + 1;
				}
			}
			// Prefix sum
			const int2 queue_offset = block_prefix_sum(is_valid?1:0);
			volatile __shared__ int base_offset[1];
			// Obtain base enqueue offset
			if(threadIdx.x == 0)
				base_offset[0] = atomicAdd(out_queue_count,queue_offset.y);
			__syncwarp();
			// Write to queue
			if (is_valid)
				out_queue[base_offset[0]+queue_offset.x] = neighbor;


			r_gather += WARP_SIZE;
			block_progress+= WARP_SIZE;
			__syncwarp();
		}
	}
}
*/


__device__ void fine_grained_gather(const int* const column_index, int* const distance,cudaSurfaceObject_t bitmask_surf, const int iteration, int * const out_queue, int* const out_queue_count,int r, int r_end)
{
	// Fine-grained neigbor-gathering
	const int2 ranks = block_prefix_sum(r_end-r);

	int rsv_rank = ranks.x;
	const int total = ranks.y;

	__shared__ int comm[BLOCK_SIZE];
	int cta_progress = 0;
	int remain;

	while ((remain = total - cta_progress) > 0)
	{
		// Pack shared array with neighbors from adjacency lists
		while((rsv_rank < cta_progress + BLOCK_SIZE) && (r < r_end))
		{
			comm[rsv_rank - cta_progress] = r;
			rsv_rank++;
			r++;
		}
		__syncthreads();
		int neighbor;
		bool is_valid = false;
		if (threadIdx.x < remain)
		{
			neighbor = column_index[comm[threadIdx.x]];
			// Look up status
			is_valid = status_lookup(distance,bitmask_surf, neighbor);
			if(is_valid)
			{
				// Update label
				distance[neighbor] = iteration + 1;
			}
		}
		__syncthreads();
		// Obtain offset in queue by performing prefix sum
		const int2 queue_offset = block_prefix_sum(is_valid?1:0);
		volatile __shared__ int base_offset[1];
		// Obtain base enqueue offset
		if(threadIdx.x == 0)
		{
			base_offset[0] = atomicAdd(out_queue_count,queue_offset.y);
		}
		__syncthreads();
		const int queue_index = base_offset[0] + queue_offset.x;
		// Write to queue
		if (is_valid)
		{
			out_queue[queue_index] = neighbor;
		}

		cta_progress += BLOCK_SIZE;
		__syncthreads();
	}
}

__global__ void expand_contract_bfs(const int n, const int* const row_offset, const int* const column_index, int* const distance, const int iteration,const int* const in_queue,const int* const in_queue_count, int* const out_queue, int* const out_queue_count, cudaSurfaceObject_t bitmask_surf)
{
	const int global_tid = blockIdx.x*blockDim.x + threadIdx.x;
	//if(global_tid >= *in_queue_count) return; // you can't do this

	const int queue_count = *in_queue_count;

	// Get vertex from the queue
	const int v = global_tid < queue_count? in_queue[global_tid]:-1;

	// Local warp-culling
	volatile __shared__ int scratch[WARPS][HASH_RANGE];
	bool is_duplicate =  warp_cull(scratch, v);
	if(v == -1) is_duplicate= true;


	// Local history-culling
	// TODO
	//volatile __shared__ int history[BLOCK_SIZE][2];

	// Load corresponding row-ranges
	int r = is_duplicate?0:row_offset[v];
	int r_end = is_duplicate?0:row_offset[v+1];
	int count = r_end - r;

	// TODO Coarse-grained neighbor-gathering

	int end = count >= BLOCK_SIZE ? r_end: r;
	//block_coarse_grained_gather(column_index, distance,bitmask_surf, iteration, out_queue, out_queue_count, r, r_end);
	//    fine_grained_gather(column_index, distance, bitmask_surf,iteration, out_queue, out_queue_count, r, r_end);
	block_coarse_grained_gather(column_index, distance,bitmask_surf, iteration, out_queue, out_queue_count, r, end);
	__syncthreads();
	end = count < BLOCK_SIZE ? r_end: r;
	fine_grained_gather(column_index, distance,bitmask_surf, iteration, out_queue, out_queue_count, r,end);

}

__device__ void fine_grained_gather(const int* const column_index, int* const distance, int* const out_queue, int* const out_queue_count, int r, int r_end, int rsv_rank, const int total, const int base_offset)
{

	volatile __shared__ int comm[BLOCK_SIZE];
	int cta_progress = 0;
	int remain;
	while ((remain = total - cta_progress) > 0)
	{
		// Pack shared array with neighbors from adjacency lists
		while((rsv_rank < cta_progress + BLOCK_SIZE) && (r < r_end))
		{
			comm[rsv_rank - cta_progress] = r;
			rsv_rank++;
			r++;
		}
		__syncthreads();
		if (threadIdx.x < remain) // && threadIdx.x < BLOCK_SIZE)
		{
			//printf("+");
			const int neighbor = column_index[comm[threadIdx.x]];
			const int queue_index = base_offset+cta_progress + threadIdx.x;
			// Write to queue
			//printf("%d,",queue_index);
			out_queue[queue_index] = neighbor;
		}
		cta_progress += BLOCK_SIZE;
		__syncthreads();
	}
}

__global__ void contract_expand_bfs(const int n, const int* const row_offset, const int* const column_index, int* const distance, const int iteration, const int*const in_queue,const int* const in_queue_count, int* const out_queue, int* const out_queue_count)
{
	const int global_tid = blockIdx.x*blockDim.x + threadIdx.x;
	const int queue_count = *in_queue_count;
	const int v = global_tid < queue_count? in_queue[global_tid]:-1;
	//printf("%d,",v);
	bool is_valid = false;
	if(global_tid < queue_count)
	{
		//printf("%d,",v);
		is_valid = distance[v] == bfs::infinity;
	}
	int r = 0, r_end = 0;
	if(is_valid)
	{
		//printf("%d,",v);
		distance[v] = iteration + 1;
		r = row_offset[v];
		r_end = row_offset[v+1];
		//int base = atomicAdd(out_queue_count,r_end-r);
		//for(int i = r; i < r_end; i++)
		//	out_queue[base+i-r] = column_index[i];
	}

	int2 prescan_result = block_prefix_sum(r_end - r);
	//printf("%d:%d:%d,  ",global_tid,prescan_result.x,r_end-r);
	
	//printf("%d:%d:%d\n",global_tid,prescan_result.x,r_end-r);
	//printf("rsv:%d",prescan_result.x);
	volatile __shared__ int base_offset[1];
	if(threadIdx.x == 0)
		base_offset[0] = atomicAdd(out_queue_count, prescan_result.y);
	__syncthreads();
	/*
	int queue_index = base_offset[0] + prescan_result.x;
		for(int i = r; i < r_end; i++)
		{
			if(queue_index >= *out_queue_count)
				printf("error %d >= %d\n",queue_index,*out_queue_count);
			out_queue[queue_index] = column_index[i];
			queue_index++;
		}
		*/
	
	fine_grained_gather(column_index, distance, out_queue, out_queue_count, r, r_end, prescan_result.x, prescan_result.y, base_offset[0]);
	//if(global_tid == 0)
	//printf("base:%d total:%d\n",base_offset[0],prescan_result.y);
	

}

