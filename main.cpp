#include "csr_matrix.h"
#include "bfs_cpu.h"
#include "bfs.cuh"
#include "common.h"

#include <iostream>
#include <iomanip>
#include <fstream>

#include <unistd.h>

void usage(char *pname)
{
	std::cerr << "USAGE: " << pname << "[-p] [-m] [-l] [-q] [-c] [-e] FILENAME" << std::endl
		<< "-p Print size of matrix" << std::endl
		<< "-m Print matrix" << std:: endl
		<< "-l Run linear-work BFS with blocking queue" << std::endl
		<< "-q Run quadratic-work BFS" << std::endl
		<< "-e Run expand-contract BFS" << std::endl
		<< "-c Run CPU BFS and compare results for corectness" << std::endl
		<< "FILENAME must be square pattern matrix stored in Rutherford Boeing sparse matrix format (*.rb)" << std::endl;
	exit(EXIT_FAILURE);
}

int compare_distance(const int* dist1,const int* dist2,const int n)
{
	int diff = 0;
	int small = 0;
	int inf = 0;

	for(int i = 0; i < n; i++)
	{
		if(dist1[i] != dist2[i])
		{
			diff++;
			if(dist1[i] > dist2[i]) small++;
			if(dist2[i] == bfs::infinity) inf++;
		}
	}
	if(diff)
		std::cout << "NOT OK: " << diff << " OF " << n << " WITH " << small << " SMALLER AND " << inf << " NOT REACHED" << std::endl;
	else
		std::cout << "OK" << std::endl;
	return diff;
}

void print_result(const bfs::result result, const char* method)
{
	std::cout << method << " with time " << std::fixed << std::setprecision(10) << result.total_time << std::endl;
}

int main(int argc, char **argv)
{
	bool run_linear = false, run_quadratic = false, compare = false, print_info = false, print_matrix = false, run_expand_contract = false, run_contract_expand = false;

	char c;
	while ((c = getopt(argc, argv, "lqcpmeE")) != -1)
		switch(c)
		{
			case 'l':
				run_linear = true;
				break;
			case 'q':
				run_quadratic = true;
				break;
			case 'c':
				compare = true;
				break;
			case 'p':
				print_info=true;
				break;
			case 'm':
				print_matrix=true;
				break;
			case 'e':
				run_expand_contract = true;
				break;
			case 'E':
				run_contract_expand = true;
				break;
			default:
				usage(argv[0]);
		}

	if(optind < 2 || optind >= argc)
		usage(argv[0]);
	std::ifstream rb_file;
	rb_file.open(argv[optind]);
	if(!rb_file.good())
		usage(argv[0]);
	csr::matrix mat = csr::load_matrix(rb_file);
	rb_file.close();

	if(print_info)
		std::cout << "Graph with " << mat.n << " vertices and " << mat.nnz << " edges" << std::endl;

	if(print_matrix)
		csr::print_matrix(mat,std::cout);

	bfs::result cpu_result, linear_result, quadratic_result, expand_contract_result, contract_expand_result;
	if(compare)
	{
		cpu_result = cpu_bfs(mat);
		print_result(cpu_result,"CPU");
	}
	if(run_linear)
	{
		linear_result = run_linear_bfs(mat);
		print_result(linear_result,"Linear");
		if(compare)
		{
			compare_distance(cpu_result.distance,linear_result.distance,mat.n);
		}
	}
	if(run_quadratic)
	{
		quadratic_result = run_quadratic_bfs(mat);
		print_result(quadratic_result,"Quadratic");
		if(compare)
		{
			compare_distance(cpu_result.distance,quadratic_result.distance,mat.n);
		}
	}
	if(run_expand_contract)
	{
		expand_contract_result = run_expand_contract_bfs(mat);
		print_result(expand_contract_result,"Expand-contract");
		if(compare)
		{
			compare_distance(cpu_result.distance,expand_contract_result.distance,mat.n);
		}
	}
	if(run_contract_expand)
	{
		contract_expand_result = run_contract_expand_bfs(mat);
		print_result(contract_expand_result,"Contract-expand");
		if(compare)
			compare_distance(cpu_result.distance, contract_expand_result.distance, mat.n);
	}

	if(run_contract_expand)
		delete[] contract_expand_result.distance;
	if(run_expand_contract)
		delete[] expand_contract_result.distance;
	if(run_quadratic)
		delete[] quadratic_result.distance;
	if(run_linear)
		delete[] linear_result.distance;
	if(compare)
		delete[] cpu_result.distance;

	csr::dispose_matrix(mat);
	return EXIT_SUCCESS;
}
