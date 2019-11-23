#include "csr_matrix.h"

#include <iostream>
#include <limits>
#include <string>

// Creates symmetric matrix from lower triangular matrix in compressed sparse column format. Notice that symmetric matrix in compressed sparse column format is the same as in compressed sparse row format. Comments in this function refer to compressed sparse column format.
csr::matrix csr::expand_symmetric_matrix(const csr::matrix mat)
{
	csr::matrix expanded;
	const int n = mat.n;
	expanded.n=n;
	expanded.ptr = new int[expanded.n+1];

	// Intialize column pointers array with zeros.
	for(int i = 0; i <= n; i++)
	{
		expanded.ptr[i] = 0;
	}

	// Count number of non-zero entries in the upper triangular part, excluding main diagonal.
	for(int i = 0; i < n; i++)
	{
		for(int offset = mat.ptr[i]; offset < mat.ptr[i+1]; offset++)
		{
			const int j = mat.index[offset];
			if(i != j)
				expanded.ptr[j]++;
		}
	}

	// Calculate inclusive prefix sum.
	for(int i = 0; i < n; i++)
		expanded.ptr[i+1]+=expanded.ptr[i];

	// Number of non-zero entries is now known.
	expanded.nnz=expanded.ptr[n]+mat.nnz;
	expanded.index = new int[expanded.nnz];

	// Add exlusive prefix sum of counts of non-zero numbers in lower triangular part, including main diagonal.
	for(int i = 0; i <= n; i++)
		expanded.ptr[i]+=mat.ptr[i];
	// Now expanded.ptr[i] points to a place in column range where row indices from lower triangular matrix should be inserted to keep all row indices for given 


	// Insert row indices from lower triangular part 
	for(int i = 0; i < n; i++)
	{
		for(int offset = mat.ptr[i]; offset < mat.ptr[i+1]; offset++)
		{
			const int in_range_offset = offset-mat.ptr[i];
			expanded.index[expanded.ptr[i] + in_range_offset] = mat.index[offset];
		}
	}

	// Insert row indices from upper triangular part, excluding main diagonal. Loop is from (n-1) to 0 to keep row indices in column ranges for given column in ascending order.
	for(int i = n-1; i >= 0; i--)
	{
		for(int offset = mat.ptr[i]; offset < mat.ptr[i+1]; offset++)
		{
			const int j = mat.index[offset];
			if(i != j)
			{
				expanded.ptr[j]--;
				expanded.index[expanded.ptr[j]]=i;
			}
		}
	}
	return expanded;
}

// Accepts only square, pattern matrices in RutherFord-Boeing Format for sparse matrices.
// Rutherford-Boeing format uses compressed sparse column format.
csr::matrix csr::load_matrix(std::istream& input)
{
	csr::matrix mat;
	bool is_symmetric=false;
	std::streamsize inf = std::numeric_limits<std::streamsize>::max();
	// line 1
	input.ignore(inf,'\n');
	// line 2
	input.ignore(inf,'\n');
	// line 3
	std::string matrix_type;
	input >> matrix_type;
	const char c = matrix_type[1];
	if(c == 's' || c == 'h' || c == 'z')
		is_symmetric=true;
	if(matrix_type[2] == 'e')
		throw std::invalid_argument("matrix in elemental form");
	int n,m, entries;
	input >> n;
	input >> m;
	if(m != n)
		throw std::invalid_argument("matrix is not square");
	input >> entries;
	input.ignore(inf,'\n');
	// line 4
	input.ignore(inf,'\n');
	// Fil matrix struct with loaded size data and allocate arrays.
	mat.n=n;
	mat.nnz=entries;
	mat.ptr = new int[n+1];
	mat.index = new int[entries];
	int val;
	for(int j = 0; j < n+1 && input >> val; j++)
	{
		mat.ptr[j]=val-1;
	}
	for(int j = 0; j < entries && input >> val; j++)
	{
		mat.index[j]=val-1;
	}
	if(is_symmetric)
	{
		csr::matrix tmp = csr::expand_symmetric_matrix(mat);
		csr::dispose_matrix(mat);
		mat=tmp;
	}
	else
	{
		csr::matrix tmp = csr::transpose_matrix(mat);
		csr::dispose_matrix(mat);
		mat=tmp;
	}
	return mat;
}

// Comments refer to compressed sparse row format.
csr::matrix csr::transpose_matrix(const csr::matrix mat)
{
	csr::matrix trans;
	trans.n = mat.n;
	trans.nnz=mat.nnz;
	trans.ptr=new int[mat.n+1];
	trans.index = new int[mat.nnz];

	// Intialize column pointers array with zeros.
	for(int i = 0; i < trans.n+1; i++)
		trans.ptr[i]=0;

	// Count number of non-zero elements in every row of transposed matrix.
	for(int i = 0; i < mat.n; i++)
	{
		for(int j = mat.ptr[i]; j < mat.ptr[i+1]; j++)
			trans.ptr[mat.index[j]]++;
	}

	// Calculate inclusive prefix sum to obtain row ranges.
	for(int i = 0; i < trans.n; i++)
		trans.ptr[i+1]+=trans.ptr[i];
	// trans.ptr[i] now points to where range of (i+1)-th row starts.

	// Insert column indices in transposed matrix. Loop is from (n-1) to 0 to keep indices in ascending order.
	for(int i = mat.n-1; i >= 0; i--)
	{
		for(int j = mat.ptr[i]; j < mat.ptr[i+1]; j++)
		{
			trans.ptr[mat.index[j]]--;
			trans.index[trans.ptr[mat.index[j]]]=i;
		}
	}

	return trans;
}

void csr::dispose_matrix(csr::matrix& mat)
{
	delete[] mat.ptr;
	delete[] mat.index;
	mat.ptr=mat.index=nullptr;
	mat.n=mat.nnz=0;
}

void csr::print_matrix(const csr::matrix mat, std::ostream& output, const bool print_compressed, const bool print_full)
{
	// Print size information.
	output << "Matrix of size " << mat.n << " with " << mat.nnz << " non-zero entries" << std::endl;

	if(print_compressed)
	{
		// Print matrix in compressed sparse row/column format.
		output << "Compressed format:" << std::endl;
		for(int i=0;i<mat.n+1;i++)
		{
			output << mat.ptr[i] << ' ';
		}
		output << std::endl;
		for(int i=0;i<mat.n;i++)
		{
			for(int j = mat.ptr[i]; j < mat.ptr[i+1];j++)
				output << mat.index[j] << ' ';
			output << std::endl;
		}
	}

	if(print_full)
	{
		// Print matrix in full form.
		output << "Expanded:" << std::endl;
		for(int i=0; i < mat.n; i++)
		{
			int k = mat.ptr[i];
			for(int j=0; j < mat.n; j++)
			{
				if(k < mat.ptr[i+1])
				{
					if(mat.index[k]<j)
						k++;
					if(mat.index[k]==j)
						output << 1;
					else
						output << '_';
				}
				else
					output << '_';
				output << ' ';
			}
			output << std::endl;
		}
	}
}

