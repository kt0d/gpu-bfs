#include "common.h"
#include "bfs_kernels.cuh"

__global__ void quadratic_bfs(const int n, const int* row_offset, const int* column_index, int*const distance, const int iteration, bool*const done)
{
    // Calculate corresponding vertex
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if(id < n && distance[id] == iteration)
    {
        bool local_done=true;
        for(int offset = row_offset[id]; offset < row_offset[id+1]; offset++)
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
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id < *in_queue_count) 
    {
        // Get vertex from the queue
        int v = in_queue[id];
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
    if (v != -1)
        scratch[warp_id][hash] = v;
    __syncwarp();
    const int retrieved = scratch[warp_id][hash];
    if (retrieved == v)
    {
        scratch[warp_id][hash] = threadIdx.x;
    }
    __syncwarp();
    if (retrieved == v && scratch[warp_id][hash] != threadIdx.x)
    {
        return true;
    }
    return false;
}

__device__ bool history_cull()
{

    return false;
}

__device__ int2 block_prefix_sum(const int val)
{
    // Heavily inspired/copied from sample "shfl_scan" provied by NVIDIA
    // Block-wide prefix sum using shfl intrinsic
    __shared__ int sums[WARPS];
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
    if (warp_id == 0 && lane_id < (blockDim.x / WARP_SIZE))
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

        mask |= neighbor_mask;
        surf1Dwrite(mask,bitmask_surf,count * 4);	
    }

    return not_visited;
}

__device__ void block_coarse_grained_gather(const int* const column_index, int* const distance, cudaSurfaceObject_t bitmask_surf, const int iteration, int * const out_queue, int* const out_queue_count,int r, int r_end)
{
    volatile __shared__ int comm[3];
    const int thread_id = threadIdx.x;
    while(__syncthreads_or(r_end-r))
    {
        // Vie for control of blokc
        if(r_end-r)
            comm[0] = thread_id;
        __syncthreads();
        if(comm[0] == thread_id)
        {
            // If won, share your range to entire block
            comm[1] = r;
            comm[2] = r_end;
            r = r_end;
        }
        __syncthreads();
        int r_gather = comm[1] + thread_id;
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
            // Prefix sum
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
   int r_gather = comm[1] + lane_id;
   const int r_gather_end = comm[2];
   int warp_progress = 0;
   const int total = comm[2] - comm[1];
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
    // Prefix scan
    int2 ranks = block_prefix_sum(r_end-r);

    int rsv_rank = ranks.x;
    const int total = ranks.y;

    __shared__ int comm[BLOCK_SIZE];
    int cta_progress = 0;
    int remain;

    while ((remain = total - cta_progress) > 0)
    {
        while((rsv_rank < cta_progress + BLOCK_SIZE) && (r < r_end))
        {
            comm[rsv_rank - cta_progress] = r;
            rsv_rank++;
            r++;
        }
        __syncthreads();
        int neighbor;
        bool is_valid = false;
        if (threadIdx.x < remain && threadIdx.x < BLOCK_SIZE)
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
        // Prefix sum
        __syncthreads();
        const int2 queue_offset = block_prefix_sum(is_valid?1:0);
        volatile __shared__ int base_offset[1];
        // Obtain base enqueue offset
        if(threadIdx.x == 0)
{
            base_offset[0] = atomicAdd(out_queue_count,queue_offset.y);
}
        __syncthreads();
        const int queue_index = base_offset[0] + queue_offset.x;
        // Can't write to queue more than n items
        //if(is_valid && queue_index >= n)
        //{
        //}
        // Write to queue
        if (is_valid)
        {
            out_queue[queue_index] = neighbor;
        }

        cta_progress += BLOCK_SIZE;
        __syncthreads();
    }
}

__global__ void expand_contract_bfs(const int n, const int* row_offset, const int* column_index, int* distance, const int iteration,const int* in_queue,const int* in_queue_count, int* out_queue, int* out_queue_count, cudaSurfaceObject_t bitmask_surf)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    //if(tid >= *in_queue_count) return; // you can't do this

    int queue_count = *in_queue_count;

    // Get vertex from the queue
    const int v = tid < queue_count? in_queue[tid]:-1;

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

