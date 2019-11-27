#pragma once
#include <limits>

namespace bfs
{
	struct result
	{
		int* distance;
		float total_time;
        int depth;
	};

	const int infinity = std::numeric_limits<int>::max();
}
