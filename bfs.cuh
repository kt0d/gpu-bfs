#pragma once
#include "common.h"
#include "csr_matrix.h"

bfs::result run_quadratic_bfs(csr::matrix graph, int start=0);
bfs::result run_linear_bfs(csr::matrix graph, int start=0);
bfs::result run_expand_contract_bfs(csr::matrix graph, int start=0);
