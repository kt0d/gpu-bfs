#include "csr_matrix.h"
#include "bfs_cpu.h"
#include "bfs.cuh"
#include "common.h"

#include <iostream>
#include <iomanip>
#include <fstream>

#include <functional>
#include <list>
#include <utility>
#include <random>

#include <unistd.h>
#include <libgen.h>

void usage(char *pname)
{
	std::cerr << "USAGE: " << pname << "[options] FILENAME" << std::endl
		<< "-p Print size of matrix" << std::endl
		<< "-m Print matrix" << std:: endl
		<< "-L Run linear-work BFS with blocking queue" << std::endl
		<< "-Q Run quadratic-work BFS" << std::endl
		<< "-E Run expand-contract BFS" << std::endl
		<< "-C Run contract-expand BFS" << std::endl
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
		std::cout << ";NOT OK: " << diff << " OF " << n << " WITH " << small << " SMALLER AND " << inf << " NOT REACHED";
	else
		std::cout << ";OK";
	return diff;
}

void print_result(const bfs::result result, const char* method)
{

//	std::cout << method << " with time " << std::fixed << std::setprecision(10) << result.total_time << std::endl;
	//std::cout << method << ";" << 
}

int main(int argc, char **argv)
{
	// Process options
	bool run_linear = false, run_quadratic = false, compare = false, print_info = false, print_matrix = false, run_expand_contract = false, run_contract_expand = false;
	std::list<std::pair<const char*, std::function<bfs::result(csr::matrix, int)>>> kernels_to_run; 

	char c;
	int times = 1;
	int set_source = -1;
	while ((c = getopt(argc, argv, "cpmLQECTn:s:")) != -1)
		switch(c)
		{
			case 'L':
				kernels_to_run.push_back(std::make_pair("Linear",run_linear_bfs));
				run_linear = true;
				break;
			case 'Q':
				kernels_to_run.push_back(std::make_pair("Quadratic", run_quadratic_bfs));
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
			case 'E':
				kernels_to_run.push_back(std::make_pair("Expand-contract", run_expand_contract_bfs));
				run_expand_contract = true;
				break;
			case 'C':
				kernels_to_run.push_back(std::make_pair("Contract-expand", run_contract_expand_bfs));
				run_contract_expand = true;
				break;
			case 'n':
				times = atoi(optarg);
				break;
			case 's':
				set_source = atoi(optarg);
				break;

			default:
				usage(argv[0]);
		}

	if(optind < 2 || optind >= argc)
		usage(argv[0]);

	// Load graph
	const char* graph_name=argv[optind];
	std::ifstream rb_file;
	rb_file.open(graph_name);
	if(!rb_file.good())
		usage(argv[0]);
	csr::matrix graph= csr::load_matrix(rb_file);
	rb_file.close();

	graph_name = basename(argv[optind]);

	if(print_info)
		std::cout << "Graph with " << graph.n << " vertices and " << graph.nnz << " edges" << std::endl;

	if(print_matrix)
		csr::print_matrix(graph,std::cout);

	std::random_device generator;
	std::uniform_int_distribution<int> distribution(0,graph.n);

	// Run kernels
	std::cout << "graph;vertices;edges;source;kernel;time" <<  (compare?";correctness":"") << std::endl;
	for(int i = 0; i < times; i++)
	{
		const int source = set_source >= 0 ? set_source : distribution(generator);
		bfs::result cpu_result;
		if(compare)
		{
			cpu_result = cpu_bfs(graph,source);
			std::cout << graph_name << ";" << graph.n << ";" << graph.nnz << ";" << source << ";CPU;" << cpu_result.total_time << (compare?";OK":"") << std::endl;
		}

		for (auto& it: kernels_to_run)
		{
			auto bfs_func = it.second;
			bfs::result result = bfs_func(graph,source);

			std::cout << graph_name << ";" << graph.n << ";" << graph.nnz << ";" << source << ";" <<  it.first << ";" << result.total_time;
			if(compare)
				compare_distance(cpu_result.distance, result.distance,graph.n);
			std::cout << std::endl;

		delete[] result.distance;
		}

		if(compare)
			delete [] cpu_result.distance;

	}

	/*
	bfs::result cpu_result, linear_result, quadratic_result, expand_contract_result, contract_expand_result;
	if(compare)
	{
		cpu_result = cpu_bfs(mat);
		print_result(cpu_result,"CPU");
	}
	if(run_linear)
	{
		linear_result = run_linear_bfs(mat,0);
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

	// Clean up after yourself
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
	*/

	csr::dispose_matrix(graph);
	return EXIT_SUCCESS;
}
