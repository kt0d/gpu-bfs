#include "common.h"
#include "bfs.cuh"
#include "csr_matrix.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>

constexpr size_t WARP_SIZE= 32;
constexpr size_t BLOCK_SIZE = 512;
constexpr size_t WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr size_t HASH_RANGE = 128;


// Calculate number of needed blocks
int div_up(int dividend, int divisor)
{
	return (dividend % divisor == 0)?(dividend/divisor):(dividend/divisor+1);
}

__global__ void init_distance(const int n, int*const distance,const int start)
{
	// Calculate corresponding vertex
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	// Fill distance vector
	if(id < n)
		distance[id]=bfs::infinity;
	if(id == start)
		distance[id]=0;
}

void initialize_graph(csr::matrix graph, int*&d_row_offset, int*&d_column_index)
{
	// Allocate device memory
	checkCudaErrors(cudaMalloc((void**)&d_row_offset,(graph.n+1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_column_index,graph.nnz * sizeof(int)));

	// Copy graph to device memory
	checkCudaErrors(cudaMemcpy(d_row_offset, graph.ptr, (graph.n+1) * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_column_index, graph.index, graph.nnz * sizeof(int), cudaMemcpyHostToDevice));
}

void dispose_graph(int*& d_row_offset, int*& d_column_index)
{
	// Free device memory
	checkCudaErrors(cudaFree(d_row_offset));
	checkCudaErrors(cudaFree(d_column_index));
}

void initialize_distance_vector(const int n, const int starting_vertex, int*& d_distance)
{
	// Allocate device memory
	checkCudaErrors(cudaMalloc((void**)&d_distance,n * sizeof(int)));

	// Calculate numbeer of blocks
	int num_of_blocks = div_up(n,BLOCK_SIZE);

	// Run kernel initializng distance vector
	init_distance<<<num_of_blocks,BLOCK_SIZE>>>(n, d_distance,starting_vertex);
}

void dispose_distance_vector(int* d_distance)
{
	// Free device memory
	checkCudaErrors(cudaFree(d_distance));
}

void initialize_vertex_queue(const int n, const int starting_vertex, int*& d_in_queue, int*& d_in_queue_count, int*& h_in_queue_count, int*& d_out_queue, int*& d_out_queue_count, int*& h_out_queue_count)
{
	// Allocate device memory
	checkCudaErrors(cudaMalloc((void**)&d_in_queue,n * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_out_queue,n * sizeof(int)));

	// Allocate host memory and map it to device memory
	checkCudaErrors(cudaHostAlloc((void**)&h_in_queue_count,sizeof(int),cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&d_in_queue_count,(void*)h_in_queue_count,0));
	checkCudaErrors(cudaHostAlloc((void**)&h_out_queue_count,sizeof(int),cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&d_out_queue_count,(void*)h_out_queue_count,0));

	// Insert starting vertex into queue
	checkCudaErrors(cudaMemcpy(d_in_queue, &starting_vertex, sizeof(int), cudaMemcpyHostToDevice));
	*h_in_queue_count=1;
	*h_out_queue_count=0;

}

void dispose_vertex_queue(int*& d_in_queue, int*& h_in_queue_count, int*& d_out_queue, int*& h_out_queue_count)
{
	// Free host memory
	checkCudaErrors(cudaFreeHost(h_in_queue_count));
	checkCudaErrors(cudaFreeHost(h_out_queue_count));

	// Free device memory
	checkCudaErrors(cudaFree(d_in_queue));
	checkCudaErrors(cudaFree(d_out_queue));
}

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

__device__ bool status_lookup(int * const distance, int neighbor)
{
    bool is_valid = false;
     // TODO bitmask
    is_valid = distance[neighbor] == bfs::infinity;
    return is_valid;
}

__device__ void block_coarse_grained_gather(const int* const column_index, int* const distance, const int iteration, int * const out_queue, int* const out_queue_count,int r, int r_end)
{
    volatile __shared__ int comm[3];
    const int thread_id = threadIdx.x;
    //while(block_any(r_end-r))
    while(__syncthreads_or(r_end-r))
    {
        if(r_end-r)
            comm[0] = thread_id;
        __syncthreads();
        if(comm[0] == thread_id)
        {
            comm[1] = r;
            comm[2] = r_end;
            r = r_end;
        }
        __syncthreads();
        int r_gather = comm[1] + thread_id;
        const int r_gather_end = comm[2];
        int block_progress = 0;
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

__device__ void fine_grained_gather(const int* const column_index, int* const distance, const int iteration, int * const out_queue, int* const out_queue_count,int r, int r_end)
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
        __syncthreads();
        // Write to queue
        if (is_valid)
            out_queue[base_offset[0]+queue_offset.x] = neighbor;

        cta_progress += BLOCK_SIZE;
        __syncthreads();
    }
}

__global__ void expand_contract_bfs(const int n, const int* row_offset, const int* column_index, int* distance, const int iteration,const int* in_queue,const int* in_queue_count, int* out_queue, int* out_queue_count)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    //if(tid >= *in_queue_count) return; // you can't do this

    int queue_count = *in_queue_count;

    // Get vertex from the queue
    const int v = tid < queue_count? in_queue[tid]:-1;

    // Local warp-culling
    volatile __shared__ int scratch[WARPS][HASH_RANGE];
    bool is_duplicate =  warp_cull(scratch, v);

    
    // Local history-culling
    // TODO
    //volatile __shared__ int history[BLOCK_SIZE][2];

    // Load corresponding row-ranges
    int r = is_duplicate?0:row_offset[v];
    int r_end = is_duplicate?0:row_offset[v+1];
    int count = r_end - r;

    // TODO Coarse-grained neighbor-gathering

    int end = count >= BLOCK_SIZE ? r_end: r;
    __syncthreads();
   // block_coarse_grained_gather(column_index, distance, iteration, out_queue, out_queue_count, r, r_end);
//    fine_grained_gather(column_index, distance, iteration, out_queue, out_queue_count, r, r_end);
    block_coarse_grained_gather(column_index, distance, iteration, out_queue, out_queue_count, r, end);
    __syncthreads();
    end = count < BLOCK_SIZE ? r_end: r;
    fine_grained_gather(column_index, distance, iteration, out_queue, out_queue_count, r,end);

}

bfs::result run_linear_bfs(const csr::matrix graph, int starting_vertex)
{
	// Allocate device memory for graph and copy it
	int *d_row_offset, *d_column_index;
	initialize_graph(graph,d_row_offset,d_column_index);

	// Allocate and initialize distance vector
	int *d_distance;
	initialize_distance_vector(graph.n, starting_vertex, d_distance);

	// Allocate and initialize queues and queue counters
	int *d_in_queue_count, *d_out_queue_count;
	int *h_in_queue_count, *h_out_queue_count;
	int *d_in_queue, *d_out_queue;
	initialize_vertex_queue(graph.n, starting_vertex, d_in_queue, d_in_queue_count, h_in_queue_count, d_out_queue, d_out_queue_count, h_out_queue_count); 

	// Create events for time measurement
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start time measurement
	cudaEventRecord(start);
	cudaProfilerStart();
	// Algorithm
	*h_in_queue_count=1;
	*h_out_queue_count=0;


	int iteration = 0;
	while(*h_in_queue_count > 0)
	{

		// Empty out queue
		*h_out_queue_count = 0;

		// Calculate number of blocks
		int num_of_blocks = div_up(*h_in_queue_count,BLOCK_SIZE);

		// Run kernel
		linear_bfs<<<num_of_blocks,BLOCK_SIZE>>>(graph.n,d_row_offset,d_column_index,d_distance,iteration, d_in_queue,d_in_queue_count, d_out_queue, d_out_queue_count);
		checkCudaErrors(cudaDeviceSynchronize());

		// Increment iteration counf
		iteration++;
		// Swap queues
		std::swap(d_in_queue,d_out_queue);
		std::swap(h_in_queue_count,h_out_queue_count);
		std::swap(d_in_queue_count,d_out_queue_count);

	}

	cudaProfilerStop();
	// Calculate elapsed time
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);

	// Event cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy distance vector to host memory
	int *h_distance = new int[graph.n];
	checkCudaErrors(cudaMemcpy(h_distance,d_distance,graph.n*sizeof(int),cudaMemcpyDeviceToHost));

	// Free queue memory
	dispose_vertex_queue(d_in_queue, h_in_queue_count, d_out_queue, h_out_queue_count);
	// Free distance vector memory
	dispose_distance_vector(d_distance); 
	// Free graph memory
	dispose_graph(d_row_offset, d_column_index);

	bfs::result result;
	result.distance= h_distance;
	result.total_time = miliseconds;
	return result;
}

bfs::result run_quadratic_bfs(const csr::matrix graph, int starting_vertex)
{
	// Allocate device memory for graph and copy it
	int *d_row_offset, *d_column_index;
	initialize_graph(graph,d_row_offset,d_column_index);

	// Allocate and initialize distance vector
	int *d_distance;
	initialize_distance_vector(graph.n, starting_vertex, d_distance);

	// Allocate and map bool flag, for use in algorithm
	bool *h_done, *d_done;
	int iteration = 0;
	checkCudaErrors(cudaHostAlloc((void**)&h_done,sizeof(bool),cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&d_done,(void*)h_done,0));

	// Create events for time measurement
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start time measurement
	cudaEventRecord(start);

	// Algorithm
	int num_of_blocks = div_up(graph.n, BLOCK_SIZE);
	do
	{
		*h_done=true;
		quadratic_bfs<<<num_of_blocks,BLOCK_SIZE>>>(graph.n,d_row_offset,d_column_index,d_distance,iteration, d_done);
		checkCudaErrors(cudaDeviceSynchronize());
		iteration++;
	} while(!(*h_done));

	// Calculate elapsed time
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);

	// Event cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	// Copy distance vector to host memory
	int *h_distance = new int[graph.n];
	checkCudaErrors(cudaMemcpy(h_distance,d_distance,graph.n*sizeof(int),cudaMemcpyDeviceToHost));

	// Free flag memory
	checkCudaErrors(cudaFreeHost(h_done));
	// Free distance vector memory
	dispose_distance_vector(d_distance); 
	// Free graph memory
	dispose_graph(d_row_offset, d_column_index);

	bfs::result result;
	result.distance= h_distance;
	result.total_time = miliseconds;
	return result;
}

bfs::result run_expand_contract_bfs(csr::matrix graph, int starting_vertex)
{
	// Allocate device memory for graph and copy it
	int *d_row_offset, *d_column_index;
	initialize_graph(graph,d_row_offset,d_column_index);

	// Allocate and initialize distance vector
	int *d_distance;
	initialize_distance_vector(graph.n, starting_vertex, d_distance);

	// Allocate and initialize queues and queue counters
	int *d_in_queue_count, *d_out_queue_count;
	int *h_in_queue_count, *h_out_queue_count;
	int *d_in_queue, *d_out_queue;
	initialize_vertex_queue(graph.n, starting_vertex, d_in_queue, d_in_queue_count, h_in_queue_count, d_out_queue, d_out_queue_count, h_out_queue_count); 

	// Create events for time measurement
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start time measurement
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	cudaProfilerStart();
	// Algorithm
	*h_in_queue_count=1;
	*h_out_queue_count=0;


	int iteration = 0;
	while(*h_in_queue_count > 0)
	{

		// Empty out queue
		*h_out_queue_count = 0;

		// Calculate number of blocks
		int num_of_blocks = div_up(*h_in_queue_count,BLOCK_SIZE);

		// Run kernel
		expand_contract_bfs<<<num_of_blocks,BLOCK_SIZE>>>(graph.n,d_row_offset,d_column_index,d_distance,iteration, d_in_queue,d_in_queue_count, d_out_queue, d_out_queue_count);
		checkCudaErrors(cudaDeviceSynchronize());

		// Increment iteration counf
		iteration++;
		// Swap queues
		std::swap(d_in_queue,d_out_queue);
		std::swap(h_in_queue_count,h_out_queue_count);
		std::swap(d_in_queue_count,d_out_queue_count);

	}

	cudaProfilerStop();
	// Calculate elapsed time
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);

	// Event cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy distance vector to host memory
	int *h_distance = new int[graph.n];
	checkCudaErrors(cudaMemcpy(h_distance,d_distance,graph.n*sizeof(int),cudaMemcpyDeviceToHost));

	// Free queue memory
	dispose_vertex_queue(d_in_queue, h_in_queue_count, d_out_queue, h_out_queue_count);
	// Free distance vector memory
	dispose_distance_vector(d_distance); 
	// Free graph memory
	dispose_graph(d_row_offset, d_column_index);

	bfs::result result;
	result.distance= h_distance;
	result.total_time = miliseconds;
	return result;
}
