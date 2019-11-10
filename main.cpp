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
	std::cerr << "USAGE: " << pname << "[-p] [-l] [-q] [-c] FILENAME" << std::endl;
	std::cerr << "-p Print size of matrix" << std::endl;
	std::cerr << "-l Run linear-work BFS with blocking queue" << std::endl;
	std::cerr << "-q Run quadratic-work BFS" << std::endl;
	std::cerr << "-c Run CPU BFS and compare results for corectness" << std::endl;
	std::cerr << "FILENAME must be square pattern matrix stored in Rutherford Boeing sparse matrix format (*.rb)" << std::endl;
	exit(EXIT_FAILURE);
}

int compare_distance(const int* dist1,const int* dist2,const int n)
{
	int diff = 0;
	
	for(int i = 0; i < n; i++)
		if(dist1[i] != dist2[i]) diff++;
	if(diff)
		std::cout << "NOT OK: " << diff << " OF " << n << std::endl;
	else
		std::cout << "OK" << std::endl;
	return diff;
}

int main(int argc, char **argv)
{
	bool run_linear = false, run_quadratic = false, compare = false, print_info = false, print_matrix = false, run_expand_contract = false;

	char c;
	while ((c = getopt(argc, argv, "lqcpme")) != -1)
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
		std::cout << "Matrix of size " << mat.n << std::endl;

	if(print_matrix)
		csr::print_matrix(mat,std::cout);

	bfs::result cpu_result, linear_result, quadratic_result, expand_contract_result;
	if(compare)
	{
		cpu_result = cpu_bfs(mat);
		std::cout << "CPU with time " << std::fixed << std::setprecision(10) << cpu_result.total_time << std::endl;
	}
	if(run_linear)
	{
		linear_result = run_linear_bfs(mat);
		std::cout << "Linear with time " << std::fixed << std::setprecision(10) << linear_result.total_time << std::endl;
		if(compare)
		{
			compare_distance(cpu_result.distance,linear_result.distance,mat.n);
		}
	}
	if(run_quadratic)
	{
		quadratic_result = run_quadratic_bfs(mat);
		std::cout << "Quadratic with time " <<  std::fixed << std::setprecision(10) << quadratic_result.total_time << std::endl;
		if(compare)
		{
			compare_distance(cpu_result.distance,quadratic_result.distance,mat.n);
		}
	}
	if(run_expand_contract)
	{
		expand_contract_result = run_expand_contract_bfs(mat);
		std::cout << "Expand contract with time " <<  std::fixed << std::setprecision(10) << expand_contract_result.total_time << std::endl;
		if(compare)
		{
			compare_distance(cpu_result.distance,expand_contract_result.distance,mat.n);
		}
	}

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
