#include "common.h"
#include "csr_matrix.h"
#include "bfs_cpu.h"

#include <queue>
#include <chrono>

bfs::result cpu_bfs(const csr::matrix mat, int starting_vertex)
{
	int* dist = new int[mat.n];

	auto start = std::chrono::system_clock::now();


	std::queue<int> q;
	for(int i = 0; i < mat.n; i++)
		dist[i]=bfs::infinity;
	dist[starting_vertex]=0;
	q.push(starting_vertex);
	while(!q.empty())
	{
		int i = q.front();
		q.pop();
		for(int offset = mat.ptr[i]; offset < mat.ptr[i+1]; offset++)
		{
			int j = mat.index[offset];
			if(dist[j] == bfs::infinity)
			{
				dist[j] = dist[i] + 1;
				q.push(j);
			}
		}
	}


	auto stop = std::chrono::system_clock::now();
	std::chrono::duration<float, std::milli> diff = stop - start;

	bfs::result result;
	result.total_time = diff.count();
	result.distance = dist;
	return result;
}
