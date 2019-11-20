#pragma once
#include "common.h"
#include "csr_matrix.h"

bfs::result run_quadratic_bfs(csr::matrix graph, int starting_vertex=0);
bfs::result run_linear_bfs(csr::matrix graph, int starting_vertex=0);
bfs::result run_expand_contract_bfs(csr::matrix graph, int starting_vertex=0);
bfs::result run_contract_expand_bfs(csr::matrix graph, int starting_vertex=0);
