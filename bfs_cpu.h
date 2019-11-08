#pragma once
#include "common.h"
#include "csr_matrix.h"

bfs::result cpu_bfs(const csr::matrix mat, int start = 0);
