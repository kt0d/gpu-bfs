#pragma once
#include <limits>

namespace bfs
{
	struct result
	{
		int* distance;
		float total_time;
	};

	constexpr int infinity = std::numeric_limits<int>::max();
}
