#pragma once
constexpr size_t WARP_SIZE = 32;
constexpr size_t BLOCK_SIZE = 512;
constexpr size_t WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr size_t HASH_RANGE = 128;


// By changing these, you can tune how many vertices/edges from the queue will
// be processed by one thread.
constexpr int QUEUE_RATIO_TWO_PHASE_VERTEX 	= 4;
constexpr int QUEUE_RATIO_TWO_PHASE_EDGE 	= 10;
constexpr int QUEUE_RATIO_CONTRACT_EXPAND	= 4;
constexpr int QUEUE_RATIO_EXPAND_CONTRACT	= 1;
constexpr int QUEUE_RATIO_LINEAR		= 4;


__global__ void expand_contract_bfs(const int n, const int* const row_offset, const int* const column_index, int* const distance, const int iteration, const int* in_queue,const int in_queue_count, int* const out_queue, int* const out_queue_count, cudaSurfaceObject_t bitmask_surf);

__global__ void contract_expand_bfs(const int n, const int* const row_offset, const int* const column_index, int* const distance, const int iteration, const int* in_queue,const int in_queue_count, int* const out_queue, int* const out_queue_count);

__global__ void quadratic_bfs(const int n, const int* row_offset, const int* column_index, int*const distance, const int iteration, bool*const done);

__global__ void linear_bfs(const int n, const int* row_offset, const int*const column_index, int*const distance, const int iteration,const int*const in_queue,const int in_queue_count, int*const out_queue, int*const out_queue_count);

__global__ void two_phase_expand(const int m, const int* const row_offset, const int* const column_index, const int*const in_queue,const int in_queue_count, int* const out_queue, int* const out_queue_count);

__global__ void two_phase_contract(const int n, int* const distance, const int iteration, const int*const in_queue,const int in_queue_count, int* const out_queue, int* const out_queue_count);
