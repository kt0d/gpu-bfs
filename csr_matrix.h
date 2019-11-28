#pragma once
#include <iostream>

namespace csr
{
	// Compressed sparse, square, pattern matrix - can represent both compressed sparse column or compressed sparse row format.
	// BFS functions assume compressed sparse row format.
	struct matrix
	{
		int n; // matrix dimension
		int nnz; // number of non-zero entries
		int* ptr; // array of length (n+1) of pointers to indices in index array
		int* index; // array of length nnz of indices
	};

	matrix load_matrix(std::istream& input);
	matrix transpose_matrix(const matrix mat);
	matrix expand_symmetric_matrix(matrix mat);
	void dispose_matrix(matrix& mat);
	void print_matrix(const matrix mat, std::ostream& output = std::cout, bool print_compressed = false, bool print_full = false);
	void print_row(const matrix mat, int v, std::ostream& output = std::cout);

}
