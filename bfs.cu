#include "common.h"
#include "bfs.cuh"
#include "csr_matrix.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 512


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

	// Allocate and map host memory
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
	if(id >= *in_queue_count) return;

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

__global__ void expand_contract_bfs(const int n, const int* row_offset, const int* column_index, int* distance, const int iteration,const int* in_queue,const int* in_queue_count, int* out_queue, int* out_queue_count)
{

	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id >= *in_queue_count) return;

	// Get vertex from the queue
	int v = in_queue[id];


	// Local warp-culling
	// Local history-culling
	// Load corresponding row-ranges
	// Coarse-grained neigbor-gathering
	// Fine-grained neigbor-gathering

	// Look up status
	// Update label
	// Prefix sum
	// Obtain base enqueue offset
	// Write to queue
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
	// Intialize queues and queue counters
	checkCudaErrors(cudaMemcpy(d_in_queue, &starting_vertex, sizeof(int), cudaMemcpyHostToDevice));

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
	cudaProfilerStart();
	// Algorithm
	// Intialize queues and queue counters
	checkCudaErrors(cudaMemcpy(d_in_queue, &starting_vertex, sizeof(int), cudaMemcpyHostToDevice));

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
		expand_contract<<<num_of_blocks,BLOCK_SIZE>>>(graph.n,d_row_offset,d_column_index,d_distance,iteration, d_in_queue,d_in_queue_count, d_out_queue, d_out_queue_count);
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
