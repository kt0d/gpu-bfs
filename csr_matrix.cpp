#include "csr_matrix.h"

#include <iostream>
#include <limits>
#include <string>


csr::matrix csr::expand_symmetric_matrix(const csr::matrix mat)
{
	csr::matrix expand;
	const int n = mat.n;
	expand.n=n;
	expand.ptr = new int[expand.n+1];
	for(int i = 0; i <= n; i++)
	{
		expand.ptr[i] = 0;
	}
	for(int i = 0; i < n; i++)
	{
		for(int offset = mat.ptr[i]; offset < mat.ptr[i+1]; offset++)
		{
			int j = mat.index[offset];
			if(i != j)
				expand.ptr[j]++;
		}
	}
	for(int i = 0; i < n; i++)
		expand.ptr[i+1]+=expand.ptr[i];
	expand.nnz=expand.ptr[n]+mat.nnz;
	expand.index = new int[expand.nnz];
	for(int i = 0; i <= n; i++)
		expand.ptr[i]+=mat.ptr[i];
	for(int i = 0; i < n; i++)
	{
		for(int offset = mat.ptr[i]; offset < mat.ptr[i+1]; offset++)
		{
			int j = offset-mat.ptr[i];
			expand.index[expand.ptr[i]+j]=mat.index[offset];
		}
	}
	for(int i = n; i >= 0; i--)
	{
		for(int offset = mat.ptr[i]; offset < mat.ptr[i+1]; offset++)
		{
			int j = mat.index[offset];
			if(i != j)
			{
				expand.ptr[j]--;
				expand.index[expand.ptr[j]]=i;
			}
		}
	}
	return expand;

}

// Only for Rutherford-Boeing Format for sparse matrices
// Only square, pattern matrices
// Rutherford-Boeing format uses compressed sparse column format
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
	int n,m, entries;
	input >> matrix_type;
	char c = matrix_type[1];
	if(c == 's' || c == 'h' || c == 'z')
		is_symmetric=true;
	if(matrix_type[2] == 'e')
		throw std::invalid_argument("matrix in elemental form");
	input >> n;
	input >> m;
	if(m != n)
		throw std::invalid_argument("matrix is not square");
	input >> entries;
	input.ignore(inf,'\n');
	// line 4
	input.ignore(inf,'\n');
	// Filling structure with data
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
	csr::matrix tmp;
	if(is_symmetric)
	{
		tmp = csr::expand_symmetric_matrix(mat);
		csr::dispose_matrix(mat);
		mat=tmp;
	}
	else
	{
		tmp = csr::transpose_matrix(mat);
		csr::dispose_matrix(mat);
		mat=tmp;
	}
	return mat;
}

csr::matrix csr::transpose_matrix(const csr::matrix mat)
{
	csr::matrix trans;
	trans.n = mat.n;
	trans.nnz=mat.nnz;
	trans.ptr=new int[mat.n+1];
	trans.index = new int[mat.nnz];
	for(int i = 0; i < trans.n+1; i++)
		trans.ptr[i]=0;
	for(int i = 0; i < trans.nnz; i++)
		trans.index[i]=0;
	for(int i = 0; i < mat.n; i++)
	{
		for(int j = mat.ptr[i]; j < mat.ptr[i+1]; j++)
			trans.ptr[mat.index[j]]++;
	}

	for(int i = 0; i < trans.n; i++)
		trans.ptr[i+1]+=trans.ptr[i];

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


void csr::print_matrix(const csr::matrix mat, std::ostream& output)
{
	output << "Matrix of size " << mat.n << " with " << mat.nnz << " non-zero entries" << std::endl;
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

