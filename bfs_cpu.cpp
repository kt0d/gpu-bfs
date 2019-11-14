#include "common.h"
#include "csr_matrix.h"
#include "bfs_cpu.h"

#include <queue>
#include <chrono>

bfs::result cpu_bfs(const csr::matrix mat, const int starting_vertex)
{
	int* const dist = new int[mat.n];

	const auto start = std::chrono::system_clock::now();

	std::queue<int> q;
	for(int i = 0; i < mat.n; i++)
		dist[i]=bfs::infinity;
	dist[starting_vertex]=0;

	q.push(starting_vertex);
	while(!q.empty())
	{
		const int vertex = q.front();
		q.pop();
		for(int offset = mat.ptr[vertex]; offset < mat.ptr[vertex+1]; offset++)
		{
			const int neighbor = mat.index[offset];
			if(dist[neighbor] == bfs::infinity)
			{
				dist[neighbor] = dist[vertex] + 1;
				q.push(neighbor);
			}
		}
	}


	const auto stop = std::chrono::system_clock::now();
	const std::chrono::duration<float, std::milli> diff = stop - start;

	bfs::result result;
	result.total_time = diff.count();
	result.distance = dist;
	return result;
}
