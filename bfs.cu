#include "bfs.cuh"
#include "bfs_kernels.cuh"
#include "csr_matrix.h"

#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>

int div_up(int dividend, int divisor)
{
	return (dividend % divisor == 0)?(dividend/divisor):(dividend/divisor+1);
}

__global__ void init_distance(const int n, int*const distance,const int start)
{
	// Calculate corresponding vertex
	const int id = blockIdx.x*blockDim.x + threadIdx.x;

	// Fill distance vector
	if(id < n)
		distance[id]=bfs::infinity;
	if(id == start)
		distance[id]=0;
}

__global__ void init_bitmask(const int count, cudaSurfaceObject_t bitmask_surf, const int start)
{
	// Calculate corresponding uint in bitmask
	const int id = blockIdx.x*blockDim.x + threadIdx.x;

	// Fill bitmask
	if(id < count)
	{
		const unsigned int mask = 0;
		surf1Dwrite(mask, bitmask_surf, id*4);
	} 
	if(id == (start / (8 * sizeof(unsigned int))))
	{
		const unsigned int mask = 1 << (start % (8 * sizeof(unsigned int)));
		surf1Dwrite(mask,bitmask_surf, id*4);
	}
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
	const int num_of_blocks = div_up(n,BLOCK_SIZE);

	// Run kernel initializng distance vector
	init_distance<<<num_of_blocks,BLOCK_SIZE>>>(n, d_distance,starting_vertex);
}

void dispose_distance_vector(int* d_distance)
{
	// Free device memory
	checkCudaErrors(cudaFree(d_distance));
}

void initialize_vertex_queue(const int n, const int starting_vertex, int*& d_in_queue, int*& in_queue_count, int*& d_out_queue, int*& out_queue_count)
{
	// Allocate device memory
	checkCudaErrors(cudaMalloc((void**)&d_in_queue,n * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_out_queue,n * sizeof(int)));

	// Allocate counters as unified memory
	checkCudaErrors(cudaMallocManaged((void**)&in_queue_count,sizeof(int)));
	checkCudaErrors(cudaMallocManaged((void**)&out_queue_count,sizeof(int)));

	// Insert starting vertex into queue
	checkCudaErrors(cudaMemcpy(d_in_queue, &starting_vertex, sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize()); // without this you can get bus error sometimes (try -qle kron_g500
	*in_queue_count=1;
	*out_queue_count=0;

}

void initialize_edge_queue(const int m, int*& d_queue, int*& queue_count)
{
	// Allocate device memory
	checkCudaErrors(cudaMalloc((void**)&d_queue, m * sizeof(int)));

	// Allocate counters as unified memory
	checkCudaErrors(cudaMallocManaged((void**)&queue_count,sizeof(int)));
	checkCudaErrors(cudaDeviceSynchronize()); // without this you can get bus error sometimes (try -qle kron_g500
	*queue_count = 0;
}

void initialize_edge_queue(const int m, int*& d_queue, int*& queue_count, const int * const d_column_index, const int r, const int r_end)
{
	initialize_edge_queue(m, d_queue, queue_count);
	checkCudaErrors(cudaMemcpy(d_queue, d_column_index + r, (r_end-r) * sizeof(int), cudaMemcpyDeviceToDevice));
	*queue_count = r_end - r;
}

void dispose_edge_queue(int*& d_queue, int*& queue_count)
{
	// Free unified memory
	checkCudaErrors(cudaFree(queue_count));
	// Free device memory
	checkCudaErrors(cudaFree(d_queue));
}
void dispose_vertex_queue(int*& d_in_queue, int*& in_queue_count, int*& d_out_queue, int*& out_queue_count)
{
	// Free unified memory
	checkCudaErrors(cudaFree(in_queue_count));
	checkCudaErrors(cudaFree(out_queue_count));

	// Free device memory
	checkCudaErrors(cudaFree(d_in_queue));
	checkCudaErrors(cudaFree(d_out_queue));
}

void initialize_bitmask(const int n,cudaSurfaceObject_t& bitmask_surf, int starting_vertex)
{
	bitmask_surf = 0;
	return;
	// problem is surface can be bound only to cudaArray and with maximum width of 65536 bytes
	// make it 2d or sth idk
	/*
	   const int count = div_up(n, 8*sizeof(unsigned int));	
	   cudaResourceDesc res_desc;
	   std::fill_n((volatile char*)&res_desc,sizeof(res_desc),0);

	   cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<unsigned int>();
	   cudaArray
	   cudaArray *bitmask_array;
	   checkCudaErrors(cudaMallocArray(&bitmask_array, &channel_desc,count,0,cudaArraySurfaceLoadStore));
	   res_desc.resType = cudaResourceTypeArray;
	   res_desc.res.array.array= bitmask_array;

	   checkCudaErrors(cudaCreateSurfaceObject(&bitmask_surf, &res_desc));
	   init_bitmask<<<div_up(count,BLOCK_SIZE),BLOCK_SIZE>>>(count, bitmask_surf,starting_vertex);
	 */
}

void dispose_bitmask(cudaSurfaceObject_t bitmask_surf)
{
	/*
	   cudaResourceDesc res_desc;
	   checkCudaErrors(cudaGetSurfaceObjectResourceDesc(&res_desc, bitmask_surf));
	   checkCudaErrors(cudaFreeArray(res_desc.res.array.array));
	   checkCudaErrors(cudaDestroySurfaceObject(bitmask_surf));
	 */
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
	int *in_queue_count, *out_queue_count;
	int *d_in_queue, *d_out_queue;
	initialize_vertex_queue(graph.n, starting_vertex, d_in_queue, in_queue_count,  d_out_queue, out_queue_count); 

	// Create events for time measurement
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start time measurement
	cudaEventRecord(start);
	cudaProfilerStart();
	// Algorithm

	int iteration = 0;
	while(*in_queue_count > 0)
	{

		// Empty out queue
		*out_queue_count = 0;

		// Calculate number of blocks
		const int num_of_blocks = div_up(*in_queue_count,BLOCK_SIZE);

		// Run kernel
		linear_bfs<<<num_of_blocks,BLOCK_SIZE>>>(graph.n,d_row_offset,d_column_index,d_distance,iteration, d_in_queue,in_queue_count, d_out_queue, out_queue_count);
		checkCudaErrors(cudaDeviceSynchronize());

		iteration++;
		// Swap queues
		std::swap(d_in_queue,d_out_queue);
		std::swap(in_queue_count,out_queue_count);

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
	dispose_vertex_queue(d_in_queue, in_queue_count, d_out_queue, out_queue_count);
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
	const int num_of_blocks = div_up(graph.n, BLOCK_SIZE);
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
	int *in_queue_count, *out_queue_count;
	int *d_in_queue, *d_out_queue;
	initialize_vertex_queue(graph.n, starting_vertex, d_in_queue, in_queue_count,  d_out_queue, out_queue_count); 

	// Allocate and initialize bitmask for status lookup
	cudaSurfaceObject_t bitmask_surf = 0;
	initialize_bitmask(graph.n,bitmask_surf,starting_vertex);


	// Create events for time measurement
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start time measurement
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	cudaProfilerStart();
	// Algorithm

	int iteration = 0;
	while(*in_queue_count > 0)
	{
		// Empty out queue
		*out_queue_count = 0;

		// Calculate number of blocks
		const int num_of_blocks = div_up(*in_queue_count,BLOCK_SIZE);

		// Run kernel
		expand_contract_bfs<<<num_of_blocks,BLOCK_SIZE>>>(graph.n,d_row_offset,d_column_index,d_distance,iteration, d_in_queue,in_queue_count, d_out_queue, out_queue_count,bitmask_surf);
		checkCudaErrors(cudaDeviceSynchronize());

		iteration++;
		// Swap queues
		std::swap(d_in_queue,d_out_queue);
		std::swap(in_queue_count,out_queue_count);

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

	// Free bitmask
	dispose_bitmask(bitmask_surf);
	// Free queue memory
	dispose_vertex_queue(d_in_queue, in_queue_count, d_out_queue, out_queue_count);
	// Free distance vector memory
	dispose_distance_vector(d_distance); 
	// Free graph memory
	dispose_graph(d_row_offset, d_column_index);

	bfs::result result;
	result.distance= h_distance;
	result.total_time = miliseconds;
	return result;
}

bfs::result run_contract_expand_bfs(csr::matrix graph, int starting_vertex)
{
	// Allocate device memory for graph and copy it
	int *d_row_offset, *d_column_index;
	initialize_graph(graph,d_row_offset,d_column_index);

	// Allocate and initialize distance vector
	int *d_distance;
	initialize_distance_vector(graph.n, starting_vertex, d_distance);

	// Allocate and initialize queues and queue counters
	int *in_queue_count, *out_queue_count;
	int *d_in_queue, *d_out_queue;
	initialize_edge_queue(graph.nnz, d_in_queue, in_queue_count, d_column_index, graph.ptr[starting_vertex], graph.ptr[starting_vertex+1]);
	initialize_edge_queue(graph.nnz, d_out_queue, out_queue_count);

	// Create events for time measurement
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start time measurement
    cudaEventRecord(start);
	cudaEventSynchronize(start);
	cudaProfilerStart();
	checkCudaErrors(cudaDeviceSynchronize());
	// Algorithm

	int iteration = 0;
	while(*in_queue_count > 0)
	{
		// Empty out queue
		*out_queue_count = 0;

		//checkCudaErrors(cudaMemPrefetchAsync((void*)out_queue_count,sizeof(int),device));
		//checkCudaErrors(cudaMemPrefetchAsync((void*)in_queue_count,sizeof(int),device));
		// Calculate number of blocks
		const int num_of_blocks = div_up(*in_queue_count,BLOCK_SIZE);

		//std::cout << "=====================" << std::endl;
		//std::cout <<"in: " << *in_queue_count << std::endl;
		// Run kernel
		contract_expand_bfs<<<num_of_blocks,BLOCK_SIZE>>>(graph.n, d_row_offset, d_column_index, d_distance, iteration, d_in_queue, in_queue_count, d_out_queue, out_queue_count);
		checkCudaErrors(cudaDeviceSynchronize());
		//std::cout << "out: " << *out_queue_count << std::endl;

		iteration++;
		// Swap queues
		std::swap(d_in_queue,d_out_queue);
		std::swap(in_queue_count,out_queue_count);

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
	dispose_edge_queue(d_in_queue, in_queue_count);
	dispose_edge_queue(d_out_queue, out_queue_count);
	// Free distance vector memory
	dispose_distance_vector(d_distance); 
	// Free graph memory
	dispose_graph(d_row_offset, d_column_index);

	// Fill result struct
	bfs::result result;
	result.distance= h_distance;
	result.total_time = miliseconds;
	return result;
}

