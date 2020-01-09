#include "bfs_kernels.cuh"

#include "common.h"
#include <assert.h>

#include <stdio.h>

constexpr unsigned int FULL_MASK = 0xffffffff;

struct prescan_result
{
	int offset, total;
};

__global__ void quadratic_bfs(const int n, const int* row_offset, const int* column_index, int*const distance, const int iteration, bool*const done)
{
	// Compute corresponding vertex.
	const int global_tid = blockIdx.x*blockDim.x + threadIdx.x;

	if(global_tid >= n) return;
	// Inspect only vertices in current frontier.
	if(distance[global_tid] != iteration) return;

	bool local_done=true;
	const int r = row_offset[global_tid];
	const int r_end = row_offset[global_tid+1];
	for(int offset = r; offset < r_end; offset++)
	{
		const int j = column_index[offset];
		if(distance[j] > iteration+1)
		{
			distance[j]=iteration+1;
			local_done=false;
		}
	}
	if(!local_done)
		*done=local_done;
}

__global__ void linear_bfs(const int n, const int* row_offset, const int*const column_index, int*const distance, const int iteration,const int*const in_queue,const int in_queue_count, int*const out_queue, int*const out_queue_count)
{
	// Compute index of corresponding vertex in the queue.
	int global_tid = blockIdx.x*blockDim.x + threadIdx.x;

	do
	{
		if(global_tid >= in_queue_count) continue;
		// Get vertex from the queue.
		const int v = in_queue[global_tid];
		// Load row range of vertex v.
		const int r = row_offset[v];
		const int r_end = row_offset[v+1];
		for(int offset = r; offset < r_end; offset++)
		{
			const int j = column_index[offset];
			if(distance[j] == bfs::infinity)
			{
				distance[j]=iteration+1;
				// Enqueue vertex.
				const int ind = atomicAdd(out_queue_count,1);
				assert(ind < n);
				out_queue[ind]=j;
			}
		}
		global_tid += gridDim.x*blockDim.x;
	} 
	while(__syncthreads_or(global_tid < in_queue_count));
}

 __device__ int warp_cull(volatile int scratch[WARPS][HASH_RANGE], const int v)
{
	//unsigned int active = __ballot_sync(FULL_MASK, v >= 0);
	//if( v == -1) return v;
	const int hash = v & (HASH_RANGE-1);
	const int warp_id = threadIdx.x / WARP_SIZE;
	if(v >= 0)
		scratch[warp_id][hash]= v;
	__syncwarp();
	const int retrieved = v >= 0 ? scratch[warp_id][hash] : v;
	__syncwarp();
	unsigned int active = __ballot_sync(FULL_MASK, retrieved == v);
	if (retrieved == v)
	{
		// Vie to be the only thread in warp inspecting vertex v.
		scratch[warp_id][hash] = threadIdx.x;
		__syncwarp(active);
		// Some other thread has this vertex
		if(scratch[warp_id][hash] != threadIdx.x)
			return -1;
	}
	return v;
}

__forceinline__  __device__ int history_cull(int history[BLOCK_SIZE], const int v)
{
	const int hash = ((BLOCK_SIZE) - 1) & v;
	if(history[hash] == v)
	{
		return -1; // vertex/edge already processed
	}
	else
		history[hash] = v;
	return v;
}

 __device__ prescan_result block_prefix_sum(const int val)
{
	// Heavily inspired/copied from sample "shfl_scan" provided by NVIDIA.
	// Block-wide prefix sum using shfl intrinsic.
	volatile __shared__ int sums[WARPS];
	int value = val;

	const int lane_id = threadIdx.x % WARP_SIZE;
	const int warp_id = threadIdx.x / WARP_SIZE;

	// Warp-wide prefix sums.
#pragma unroll
	for(int i = 1; i <= WARP_SIZE; i <<= 1)
	{
		const int n = __shfl_up_sync(FULL_MASK, value, i, WARP_SIZE);
		if (lane_id >= i)
			value += n;
	}

	// Write warp total to shared array.
	if (lane_id == WARP_SIZE- 1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();

	// Prefix sum of warp sums.
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

	// Add total sum of previous warps to current element.
	if (warp_id > 0)
	{
		const int block_sum = sums[warp_id-1];
		value += block_sum;
	}

	prescan_result result;
	// Subtract value given by thread to get exclusive prefix sum.
	result.offset = value - val;
	// Get total sum.
	result.total = sums[WARPS-1];
	return result; 
}

 __device__ bool status_lookup(int * const distance,const cudaSurfaceObject_t bitmask_surf, const int neighbor)
{
	// Just check status directly if bitmask is unavailable.
	//if (bitmask_surf == 0)
		return distance[neighbor] == bfs::infinity;
	
	//bool not_visited = false;
	/*
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
	*/
}

 __device__ void block_gather(const int* const column_index, int* const distance, cudaSurfaceObject_t bitmask_surf, const int iteration, int * const out_queue, int* const out_queue_count,int r, int r_end)
{
	volatile __shared__ int comm[3];
	while(__syncthreads_or(r < r_end))
	{
		// Vie for control of block.
		if(r < r_end)
			comm[0] = threadIdx.x;
		__syncthreads();
		if(comm[0] == threadIdx.x)
		{
			// If won, share your range to the entire block.
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
				// Look up status of current neighbor.
				is_valid = status_lookup(distance,bitmask_surf, neighbor);
				if(is_valid)
				{
					// Update label.
					distance[neighbor] = iteration + 1;
				}
			}
			// Obtain offset in queue by computing prefix sum
			const prescan_result prescan = block_prefix_sum(is_valid?1:0);
			volatile __shared__ int base_offset[1];

			// Obtain base enqueue offset and share it to whole block.
			if(threadIdx.x == 0)
				base_offset[0] = atomicAdd(out_queue_count,prescan.total);
			__syncthreads();
			// Write vertex to the out queue.
			if (is_valid)
				out_queue[base_offset[0]+prescan.offset] = neighbor;

			r_gather += BLOCK_SIZE;
			block_progress+= BLOCK_SIZE;
			__syncthreads();
		}
	}
}

 __device__ void fine_gather(const int* const column_index, int* const distance,cudaSurfaceObject_t bitmask_surf, const int iteration, int * const out_queue, int* const out_queue_count,int r, int r_end)
{
	prescan_result rank = block_prefix_sum(r_end-r);

	__shared__ int comm[BLOCK_SIZE];
	int cta_progress = 0;

	while ((rank.total - cta_progress) > 0)
	{
		// Pack shared array with neighbors from adjacency lists.
		while((rank.offset < cta_progress + BLOCK_SIZE) && (r < r_end))
		{
			comm[rank.offset - cta_progress] = r;
			rank.offset++;
			r++;
		}
		__syncthreads();
		int neighbor;
		bool is_valid = false;
		if (threadIdx.x < (rank.total - cta_progress))
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
		// Obtain offset in queue by computing prefix sum.
		const prescan_result prescan = block_prefix_sum(is_valid?1:0);
		volatile __shared__ int base_offset[1];
		// Obtain base enqueue offset
		if(threadIdx.x == 0)
		{
			base_offset[0] = atomicAdd(out_queue_count,prescan.total);
		}
		__syncthreads();
		const int queue_index = base_offset[0] + prescan.offset;
		// Write to queue
		if (is_valid)
		{
			out_queue[queue_index] = neighbor;
		}

		cta_progress += BLOCK_SIZE;
		__syncthreads();
	}
}

__global__ void expand_contract_bfs(const int n, const int* const row_offset, const int* const column_index, int* const distance, const int iteration,const int* const in_queue,const int in_queue_count, int* const out_queue, int* const out_queue_count, cudaSurfaceObject_t bitmask_surf)
{
	int global_tid = blockIdx.x*blockDim.x + threadIdx.x;

	do
	{
		// Get vertex from the queue.
		int v = global_tid < in_queue_count? in_queue[global_tid]:-1;

		// Do local warp-culling.
		volatile __shared__ int scratch[WARPS][HASH_RANGE];
		v = warp_cull(scratch, v);

		// Load corresponding row-ranges.
		const int r =  v < 0 ?0:row_offset[v];
		const int r_end = v < 0?0:row_offset[v+1];
		const bool big_list = (r_end - r) >= BLOCK_SIZE;

		// Both expand and contract phases occur in these functions.
		block_gather(column_index, distance,bitmask_surf, iteration, out_queue, out_queue_count, r, big_list ? r_end : r);
		fine_gather(column_index, distance,bitmask_surf, iteration, out_queue, out_queue_count, r, big_list ? r : r_end);

		global_tid += gridDim.x*blockDim.x;
	}
	while(__syncthreads_or(global_tid < in_queue_count));
}

 __device__ void fine_gather(const int* const column_index, int* const out_queue, int r, int r_end, int rsv_rank, const int total, const int base_offset)
{
	volatile __shared__ int comm[BLOCK_SIZE];
	int cta_progress = 0;
	int remain;
	while ((remain = total - cta_progress) > 0)
	{
		// Pack shared array with neighbors from adjacency lists.
		while((rsv_rank < cta_progress + BLOCK_SIZE) && (r < r_end))
		{
			comm[rsv_rank - cta_progress] = r;
			rsv_rank++;
			r++;
		}
		__syncthreads();
		if (threadIdx.x < remain)
		{
			const int neighbor = column_index[comm[threadIdx.x]];
			const int queue_index = base_offset+cta_progress + threadIdx.x;
			// Write to queue.
			out_queue[queue_index] = neighbor;
		}
		cta_progress += BLOCK_SIZE;
		__syncthreads();
	}
}

 __device__ void warp_gather(const int* const column_index, int * const out_queue,int r, const int r_end, int rsv_rank, int base_offset)
{
	volatile __shared__ int comm[WARPS][4];
	const int lane_id = threadIdx.x % WARP_SIZE;
	const int warp_id = threadIdx.x / WARP_SIZE;
	while(__any_sync(FULL_MASK,r < r_end))
	{
		// Vie for control of warp.
		if(r < r_end)
			comm[warp_id][0] = lane_id;
		__syncwarp();
		if(comm[warp_id][0] == lane_id)
		{
			// If won, share your range and enqueue offset to the entire warp.
			comm[warp_id][1] = r;
			comm[warp_id][2] = r_end;
			comm[warp_id][3] = rsv_rank;
			r = r_end;
		}
		__syncwarp();
		int r_gather = comm[warp_id][1] + lane_id;
		const int r_gather_end = comm[warp_id][2];
		int queue_index = base_offset+comm[warp_id][3] + lane_id;
		while(r_gather < r_gather_end)
		{
			const int v = column_index[r_gather];
			out_queue[queue_index] = v;
			r_gather += WARP_SIZE;
			queue_index += WARP_SIZE;
		}
	}
}

__global__ void contract_expand_bfs(const int m, const int* const row_offset, const int* const column_index, int* const distance, const int iteration, const int*const in_queue,const int in_queue_count, int* const out_queue, int* const out_queue_count)
{
	int global_tid = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ int history[BLOCK_SIZE];
	volatile __shared__ int scratch[WARPS][HASH_RANGE];
	history[threadIdx.x] = -1;

	do
	{
		// Get neighbor from the queue.
		int v = global_tid < in_queue_count? in_queue[global_tid]:-1;

		// Contract phase: filter previously visited and duplicate neighbors.
		v = warp_cull(scratch, v);
		v = history_cull(history, v);

		if(v >= 0 &&  distance[v] == bfs::infinity){
			distance[v] = iteration+1;
		}
		else
			v = -1;
		int r = 0, r_end = 0;

		if(v >= 0)
		{
			r = row_offset[v];
			r_end = row_offset[v+1];
		}

		// Expand phase: expand adjacency lists and copy them to the out queue.
		const bool big_list = (r_end - r) >= WARP_SIZE; 
		const prescan_result warp_gather_prescan = block_prefix_sum(big_list ? (r_end - r):0);
		__syncthreads(); // __syncthreads is very much needed because of shared array used in block_prefix_sum
		const prescan_result fine_gather_prescan = block_prefix_sum(big_list ? 0 : (r_end - r));
		//printf("%d of %d, %d of %d\n", warp_gather_prescan.offset, warp_gather_prescan.total, fine_gather_prescan.offset, fine_gather_prescan.total);

		volatile __shared__ int base_offset[1];
		if(threadIdx.x == 0)
		{
			base_offset[0] = atomicAdd(out_queue_count, warp_gather_prescan.total + fine_gather_prescan.total);
			//assert(((base_offset[0]+warp_gather_prescan.total + fine_gather_prescan.total) < m));
		}
		__syncthreads();
		int base = base_offset[0];	
		warp_gather(column_index, out_queue, r, big_list ? r_end : 0, warp_gather_prescan.offset, base);
		base += warp_gather_prescan.total;
		fine_gather(column_index, out_queue, r, big_list ? 0: r_end, fine_gather_prescan.offset, fine_gather_prescan.total, base);

		global_tid += gridDim.x*blockDim.x;
	}
	while(__syncthreads_or(global_tid < in_queue_count));
}

__global__ void two_phase_expand(const int m, const int* const row_offset, const int* const column_index, const int*const in_queue,const int in_queue_count, int* const out_queue, int* const out_queue_count)
{
	int global_tid = blockIdx.x*blockDim.x + threadIdx.x;

	do
	{
		// Get neighbor from the queue.
		int v = global_tid < in_queue_count? in_queue[global_tid]:-1;

		int r = 0, r_end = 0;

		if(v >= 0)
		{
			r = row_offset[v];
			r_end = row_offset[v+1];
		}

		// Expand phase: expand adjacency lists and copy them to the out queue.
		const bool big_list = (r_end - r) >= WARP_SIZE; 
		const prescan_result warp_gather_prescan = block_prefix_sum(big_list ? (r_end - r):0);
		__syncthreads(); // __syncthreads is very much needed because of shared array used in block_prefix_sum
		const prescan_result fine_gather_prescan = block_prefix_sum(big_list ? 0 : (r_end - r));

		volatile __shared__ int base_offset[1];
		if(threadIdx.x == 0)
		{
			base_offset[0] = atomicAdd(out_queue_count, warp_gather_prescan.total + fine_gather_prescan.total);
			assert(((base_offset[0]+warp_gather_prescan.total + fine_gather_prescan.total) < m));
		}
		__syncthreads();
		int base = base_offset[0];	
		warp_gather(column_index, out_queue, r, big_list ? r_end : 0, warp_gather_prescan.offset, base);
		base += warp_gather_prescan.total;
		fine_gather(column_index, out_queue, r, big_list ? 0: r_end, fine_gather_prescan.offset, fine_gather_prescan.total, base);
		global_tid += gridDim.x*blockDim.x;
	}
	while(__syncthreads_or(global_tid < in_queue_count));

}

__global__ void two_phase_contract(const int n, int* const distance, const int iteration, const int*const in_queue,const int in_queue_count, int* const out_queue, int* const out_queue_count)
{
	int global_tid = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ int history[BLOCK_SIZE];
	volatile __shared__ int scratch[WARPS][HASH_RANGE];
	history[threadIdx.x] = -1;

	do
	{
		// Get neighbor from the queue.
		int v = global_tid < in_queue_count? in_queue[global_tid]:-1;

		// Contract phase: filter previously visited and duplicate neighbors.
		v = warp_cull(scratch, v);
		v = history_cull(history, v);

		if(v >= 0 &&  distance[v] == bfs::infinity){
			distance[v] = iteration+1;
		}
		else
			v = -1;
		const prescan_result prescan = block_prefix_sum(v >= 0 ? 1: 0);
		volatile __shared__ int base_offset[1];
		if(threadIdx.x == 0)
		{
			base_offset[0] = atomicAdd(out_queue_count, prescan.total);
			assert( (base_offset[0]+prescan.total) < n);
		}
		__syncthreads();
		if( v >= 0)
		{
			out_queue[base_offset[0]+prescan.offset] = v;
		}
		global_tid += gridDim.x*blockDim.x;
	}
	while(__syncthreads_or(global_tid < in_queue_count));
}
