#include "bfs.cuh"

#include "bfs_kernels.cuh"
#include "csr_matrix.h"

#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>

/// HELPER FUNCTIONS

int div_up(int dividend, int divisor)
{
	return (dividend % divisor == 0)?(dividend/divisor):(dividend/divisor+1);
}

/// KERNELS

// Set every label in distance vecto to infinity, except source vertex.
__global__ void init_dist_kernel(const int n, int*const distance,const int source_vertex)
{
	// Compute corresponding vertex.
	const int id = blockIdx.x*blockDim.x + threadIdx.x;

	// Fill distance vector with infinity.
	if(id < n)
		distance[id]=bfs::infinity;
	// Set distance to source_vertexing vertex to 0.
	if(id == source_vertex)
		distance[id]=0;
}

// Set every bit in bitmask to 0, except source vertex.
__global__ void init_bitmask(const int count, cudaSurfaceObject_t bitmask_surf, const int source_vertex)
{
	// Compute index of corresponding uint in bitmask.
	const int id = blockIdx.x*blockDim.x + threadIdx.x;

	// Fill bitmask with zeros.
	if(id < count)
	{
		const unsigned int mask = 0;
		surf1Dwrite(mask, bitmask_surf, id*4);
	} 
	// Set bit corresponding to source_vertexing vertex to 1.
	if(id == (source_vertex / (8 * sizeof(unsigned int))))
	{
		const unsigned int mask = 1 << (source_vertex % (8 * sizeof(unsigned int)));
		surf1Dwrite(mask,bitmask_surf, id*4);
	}
}

/// INITIALIZATIONS AND MEMORY RELEASE

// Copy graph data to global device memory.
void init_graph(csr::matrix graph, int*&d_row_offset, int*&d_column_index)
{
	checkCudaErrors(cudaMalloc((void**)&d_row_offset,(graph.n+1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_column_index,graph.nnz * sizeof(int)));

	// Copy graph to device memory.
	checkCudaErrors(cudaMemcpy(d_row_offset, graph.ptr, (graph.n+1) * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_column_index, graph.index, graph.nnz * sizeof(int), cudaMemcpyHostToDevice));
}

void dispose_graph(int*& d_row_offset, int*& d_column_index)
{
	checkCudaErrors(cudaFree(d_row_offset));
	checkCudaErrors(cudaFree(d_column_index));
}

void init_dist_vector(const int n, const int source_vertex, int*& d_distance)
{
	checkCudaErrors(cudaMalloc((void**)&d_distance,n * sizeof(int)));
	// Compute number of blocks needed to fill distance vector.
	const int num_of_blocks = div_up(n,BLOCK_SIZE);
	init_dist_kernel<<<num_of_blocks,BLOCK_SIZE>>>(n, d_distance,source_vertex);
}

void dispose_distance_vector(int* d_distance)
{
	checkCudaErrors(cudaFree(d_distance));
}

void init_queue(const int n, int *& d_queue)
{
	checkCudaErrors(cudaMalloc((void**)&d_queue,n * sizeof(int)));
}

// Initialize queue and push source vertex into it. Set host-side and device-side queue counters to 1.
void init_queue_with_vertex(const int n, int*& d_queue, int*& d_queue_count, int& h_queue_count,  const int source_vertex)
{
	init_queue(n, d_queue);
	checkCudaErrors(cudaMalloc((void**)&d_queue_count, sizeof(int)));
	// Insert starting vertex into the queue.
	checkCudaErrors(cudaMemcpy(d_queue, &source_vertex, sizeof(int), cudaMemcpyHostToDevice));
	// Set queue count to correct value.
	h_queue_count = 1;
	checkCudaErrors(cudaMemcpy(d_queue_count, &h_queue_count, sizeof(int), cudaMemcpyHostToDevice)); // kinda useless tbh
}

// Initialize queue and copy adjacency list given by range (r, r_end) into it. Set host-side and device-side queue counters to correct value.
void init_queue_with_edges(const int m, int*& d_queue, int*& d_queue_count, int& h_queue_count,  const int * const d_column_index, const int r, const int r_end)
{
	// Initialize empty queue.
	init_queue(m, d_queue);
	checkCudaErrors(cudaMalloc((void**)&d_queue_count, sizeof(int)));
	// Copy neighbors of starting vertex into the queue.
	const int count = r_end - r;
	checkCudaErrors(cudaMemcpy(d_queue, d_column_index + r, count * sizeof(int), cudaMemcpyDeviceToDevice));
	// Set queue count to correct value.
	h_queue_count = count;
	checkCudaErrors(cudaMemcpy(d_queue_count, &h_queue_count, sizeof(int), cudaMemcpyHostToDevice));// not needed
}

void dispose_queue(int* d_queue, int* d_queue_count= nullptr)
{
	checkCudaErrors(cudaFree(d_queue_count));
	if(d_queue_count != nullptr)
	checkCudaErrors(cudaFree(d_queue));
}

void initialize_bitmask(const int n,cudaSurfaceObject_t& bitmask_surf, int source_vertex)
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
	   init_bitmask<<<div_up(count,BLOCK_SIZE),BLOCK_SIZE>>>(count, bitmask_surf,source_vertex);
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

/// BFS FUNCTIONS

bfs::result run_linear_bfs(const csr::matrix graph, int source_vertex)
{
	// Initialize graph, queue and distance vector in device memory.
	int *d_row_offset, *d_column_index;
	init_graph(graph,d_row_offset,d_column_index);

	int *d_distance;
	init_dist_vector(graph.n, source_vertex, d_distance);

	int *d_queue_count, h_queue_count;
	int *d_in_queue, *d_out_queue;
	// Because linear kernel contains no duplicate-eliminating heuristics, to 
	// achieve succesful traversal size of queue should be more than number of 
	// vertices. 
	init_queue_with_vertex(graph.nnz, d_in_queue, d_queue_count, h_queue_count, source_vertex);
	init_queue(graph.nnz, d_out_queue);

	// Create events for time measurement.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start time measurement.
	cudaEventRecord(start);
	// Start profiling.
	cudaProfilerStart();

	// Algorithm
	int iteration = 0;
	while(h_queue_count > 0)
	{
		// Empty queue.
		checkCudaErrors(cudaMemset(d_queue_count,0,sizeof(int)));
		// Compute number of blocks needed.
		const int num_of_blocks = div_up(h_queue_count,
				QUEUE_RATIO_LINEAR * BLOCK_SIZE);
		// Run kernel.
		linear_bfs<<<num_of_blocks,BLOCK_SIZE>>>(graph.nnz,d_row_offset,d_column_index,d_distance,iteration, d_in_queue, h_queue_count,d_out_queue, d_queue_count);
		// Get queue count
		checkCudaErrors(cudaMemcpy(&h_queue_count, d_queue_count,sizeof(int), cudaMemcpyDeviceToHost));
		//std::cout << h_queue_count << std::endl;

		iteration++;
		std::swap(d_in_queue,d_out_queue);
	}

	// Stop profiling.
	cudaProfilerStop();
	// Compute elapsed time.
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);

	// Cleanup events. 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy distance vector to host memory.
	int *h_distance = new int[graph.n];
	checkCudaErrors(cudaMemcpy(h_distance,d_distance,graph.n*sizeof(int),cudaMemcpyDeviceToHost));

	// Cleanup device memory.
	dispose_queue(d_in_queue, d_queue_count);
	dispose_queue(d_out_queue);
	dispose_distance_vector(d_distance); 
	dispose_graph(d_row_offset, d_column_index);

	// Fill and return result struct.
	bfs::result result;
	result.distance= h_distance;
	result.total_time = miliseconds;
    result.depth = iteration - 1;
	return result;
}

bfs::result run_quadratic_bfs(const csr::matrix graph, int source_vertex)
{
	// Initialize graph, distance vector and finish flag in device memory.
	int *d_row_offset, *d_column_index;
	init_graph(graph,d_row_offset,d_column_index);

	int *d_distance;
	init_dist_vector(graph.n, source_vertex, d_distance);

	bool *d_done, h_done;
	checkCudaErrors(cudaMalloc((void**)&d_done,sizeof(bool)));
	checkCudaErrors(cudaDeviceSynchronize());

	// Create events for time measurement.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start time measurement.
	cudaEventRecord(start);

	// Algorithm
	// Compute number of blocks needed so every vertex in graph gets one thread.
	const int num_of_blocks = div_up(graph.n, BLOCK_SIZE);
	int iteration = 0;
	do
	{
		// Set flag to 'true' value.
		checkCudaErrors(cudaMemset(d_done,1,sizeof(bool)));
		// Run kernel.
		quadratic_bfs<<<num_of_blocks,BLOCK_SIZE>>>(graph.n,d_row_offset,d_column_index,d_distance,iteration, d_done);
		// Copy flag from device memory.
		checkCudaErrors(cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost));
		iteration++;
	} while(!h_done);

	// Compute elapsed time.
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);

	// Cleanup events. 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy distance vector to host memory.
	int *h_distance = new int[graph.n];
	checkCudaErrors(cudaMemcpy(h_distance,d_distance,graph.n*sizeof(int),cudaMemcpyDeviceToHost));

	// Cleanup device memory.
	checkCudaErrors(cudaFree(d_done));
	dispose_distance_vector(d_distance); 
	dispose_graph(d_row_offset, d_column_index);

	// Fill and return result struct.
	bfs::result result;
	result.distance= h_distance;
	result.total_time = miliseconds;
    result.depth = iteration - 1;
	return result;
}

bfs::result run_expand_contract_bfs(const csr::matrix graph, int source_vertex)
{
	// Initialize graph, distance vector, vertex queues and bitmask in device memory.
	int *d_row_offset, *d_column_index;
	init_graph(graph,d_row_offset,d_column_index);

	int *d_distance;
	init_dist_vector(graph.n, source_vertex, d_distance);

	int *d_queue_count, h_queue_count;
	int *d_in_queue, *d_out_queue;
	init_queue_with_vertex(graph.n, d_in_queue, d_queue_count, h_queue_count, source_vertex);
	init_queue(graph.n, d_out_queue);

	cudaSurfaceObject_t bitmask_surf;
	initialize_bitmask(graph.n,bitmask_surf,source_vertex);

	// Create events for time measurement.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start time measurement.
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	// Start profiling.
	cudaProfilerStart();

	// Algorithm
	int iteration = 0;
	while(h_queue_count > 0)
	{
		// Empty queue.
		checkCudaErrors(cudaMemset(d_queue_count,0,sizeof(int)));
		// Compute number of blocks needed.
		const int num_of_blocks = div_up(h_queue_count,
				QUEUE_RATIO_EXPAND_CONTRACT * BLOCK_SIZE);

		expand_contract_bfs<<<num_of_blocks,BLOCK_SIZE>>>(graph.n,d_row_offset,d_column_index,d_distance,iteration, d_in_queue,h_queue_count, d_out_queue, d_queue_count,bitmask_surf);
		// Get queue count from device memory.
		checkCudaErrors(cudaMemcpy(&h_queue_count, d_queue_count, sizeof(int), cudaMemcpyDeviceToHost));
		std::swap(d_in_queue,d_out_queue);
		iteration++;
	}

	// Stop profiling.
	cudaProfilerStop();

	// Compute elapsed time.
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);
	// Cleanup events. 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy distance vector to host memory.
	int *h_distance = new int[graph.n];
	checkCudaErrors(cudaMemcpy(h_distance,d_distance,graph.n*sizeof(int),cudaMemcpyDeviceToHost));

	// Cleanup device memory.
	dispose_bitmask(bitmask_surf);
	dispose_queue(d_in_queue, d_queue_count);
	dispose_queue(d_out_queue);
	dispose_distance_vector(d_distance); 
	dispose_graph(d_row_offset, d_column_index);

	// Fill and return result struct.
	bfs::result result;
	result.distance= h_distance;
	result.total_time = miliseconds;
    result.depth = iteration - 1;
	return result;
}

bfs::result run_contract_expand_bfs(const csr::matrix graph, int source_vertex)
{
	// Initialize graph, distance vector and edge queues in device memory.
	int *d_row_offset, *d_column_index;
	init_graph(graph,d_row_offset,d_column_index);

	int *d_distance;
	init_dist_vector(graph.n, source_vertex, d_distance);

	int *d_queue_count, h_queue_count;
	int *d_in_queue, *d_out_queue;
	init_queue_with_edges(graph.nnz, d_in_queue, d_queue_count, h_queue_count, d_column_index, graph.ptr[source_vertex], graph.ptr[source_vertex+1]);
	init_queue(graph.nnz, d_out_queue);

	// Create events for time measurement.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start time measurement.
        cudaEventRecord(start);
	cudaEventSynchronize(start);

	// Start profiling.
	cudaProfilerStart();

	// Algorithm
	int iteration = 0;
	while(h_queue_count > 0)
	{
		// Empty queue.
		checkCudaErrors(cudaMemset(d_queue_count,0,sizeof(int)));
		// Compute number of blocks needed.
		const int num_of_blocks = div_up(h_queue_count,
				QUEUE_RATIO_CONTRACT_EXPAND * BLOCK_SIZE);

//		std::cout << "======== iteration\t" << iteration << " with blocks\t" << num_of_blocks << "=============" << std::endl;
//		std::cout <<"in: " << h_queue_count << std::endl;
		contract_expand_bfs<<<num_of_blocks,BLOCK_SIZE>>>(graph.nnz, d_row_offset, d_column_index, d_distance, iteration, d_in_queue, h_queue_count, d_out_queue, d_queue_count);

		// Get queue count from device memory.
		checkCudaErrors(cudaMemcpy(&h_queue_count, d_queue_count, sizeof(int), cudaMemcpyDeviceToHost));
//		std::cout << "out: " << h_queue_count << std::endl;
		std::swap(d_in_queue,d_out_queue);
		iteration++;
	}

	// Stop profiling.
	cudaProfilerStop();

	// Compute elapsed time.
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);
	// Cleanup events.
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy distance vector to host memory.
	int *h_distance = new int[graph.n];
	checkCudaErrors(cudaMemcpy(h_distance,d_distance,graph.n*sizeof(int),cudaMemcpyDeviceToHost));

	// Cleanup device memory.
	dispose_queue(d_in_queue, d_queue_count);
	dispose_queue(d_out_queue);
	dispose_distance_vector(d_distance); 
	dispose_graph(d_row_offset, d_column_index);

	// Fill result struct.
	bfs::result result;
	result.distance= h_distance;
	result.total_time = miliseconds;
    result.depth = iteration ? (iteration - 1) : 0;
	return result;
}

bfs::result run_two_phase_bfs(const csr::matrix graph, int source_vertex)
{
	// Initialize graph, distance vector, vertex and edge queues in device memory.
	int *d_row_offset, *d_column_index;
	init_graph(graph,d_row_offset,d_column_index);

	int *d_distance;
	init_dist_vector(graph.n, source_vertex, d_distance);

	int *d_queue_count, h_queue_count;
	int *d_edge_queue, *d_vertex_queue;
	init_queue_with_vertex(graph.n, d_vertex_queue, d_queue_count, h_queue_count, source_vertex);
	init_queue(graph.nnz, d_edge_queue);

	// Create events for time measurement.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start time measurement.
        cudaEventRecord(start);
	cudaEventSynchronize(start);

	// Start profiling.
	cudaProfilerStart();

	// Algorithm
	int iteration = 0;
	while(h_queue_count > 0)
	{
		// Empty queue.
		checkCudaErrors(cudaMemset(d_queue_count,0,sizeof(int)));

//		std::cout << "======== iteration\t" << iteration << " with blocks\t" << num_of_blocks << "=============" << std::endl;
		// Process vertex queue.
		// Compute number of blocks needed.
		int num_of_blocks = div_up(h_queue_count,
				QUEUE_RATIO_TWO_PHASE_VERTEX * BLOCK_SIZE);
		two_phase_expand<<<num_of_blocks,BLOCK_SIZE>>>(graph.nnz, d_row_offset, d_column_index, d_vertex_queue, h_queue_count, d_edge_queue, d_queue_count);

		// Get queue count from device memory.
		checkCudaErrors(cudaMemcpy(&h_queue_count, d_queue_count, sizeof(int), cudaMemcpyDeviceToHost));
		// Empty queue.
		checkCudaErrors(cudaMemset(d_queue_count,0,sizeof(int)));
	
		// Process edge queue.
		// Compute number of blocks needed.
		num_of_blocks = div_up(h_queue_count,
				QUEUE_RATIO_TWO_PHASE_EDGE * BLOCK_SIZE);
		two_phase_contract<<<num_of_blocks,BLOCK_SIZE>>>(graph.n,d_distance, iteration, d_edge_queue, h_queue_count, d_vertex_queue, d_queue_count);
		// Get queue count from device memory.
		checkCudaErrors(cudaMemcpy(&h_queue_count, d_queue_count, sizeof(int), cudaMemcpyDeviceToHost));
//		std::cout << "out: " << h_queue_count << std::endl;
		iteration++;
	}

	// Stop profiling.
	cudaProfilerStop();

	// Compute elapsed time.
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);
	// Cleanup events.
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy distance vector to host memory.
	int *h_distance = new int[graph.n];
	checkCudaErrors(cudaMemcpy(h_distance,d_distance,graph.n*sizeof(int),cudaMemcpyDeviceToHost));

	// Cleanup device memory.
	dispose_queue(d_vertex_queue, d_queue_count);
	dispose_queue(d_edge_queue);
	dispose_distance_vector(d_distance); 
	dispose_graph(d_row_offset, d_column_index);

	// Fill result struct.
	bfs::result result;
	result.distance= h_distance;
	result.total_time = miliseconds;
    result.depth = iteration ? (iteration - 1) : 0;
	return result;
}
