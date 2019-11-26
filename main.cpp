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
	if(inf)
		std::cout << "NOT TRAVERSED";
	return diff;
}

void print_result(const bfs::result result, const char* method)
{

}

int main(int argc, char **argv)
{
	// Process options
	bool  compare = false, print_info = false, print_matrix = false;
	bool set_source = false;
	std::list<std::pair<const char*, std::function<bfs::result(csr::matrix, int)>>> kernels_to_run; 

	char c;
	int times = 1;
	int set_source_vertex;
	while ((c = getopt(argc, argv, "cpmLQECTn:s:")) != -1)
		switch(c)
		{
			case 'L':
				kernels_to_run.push_back(std::make_pair("Linear",run_linear_bfs));
				break;
			case 'Q':
				kernels_to_run.push_back(std::make_pair("Quadratic", run_quadratic_bfs));
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
				break;
			case 'C':
				kernels_to_run.push_back(std::make_pair("Contract-expand", run_contract_expand_bfs));
				break;
			case 'n':
				times = atoi(optarg);
				break;
			case 's':
				set_source = true;
				set_source_vertex = atoi(optarg);
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
	{
		std::cerr << "Incorrect file" << std::endl;
		usage(argv[0]);
	}
	csr::matrix graph= csr::load_matrix(rb_file);
	rb_file.close();
	
	// If source vertex was chosen explicitly, check if it's correct
	if(set_source && (set_source_vertex < 0 || set_source_vertex >= graph.n))
	{
		std::cerr << "Vertex " << set_source_vertex << " does not belong to loaded graph" << std::endl;
		usage(argv[0]);
	}

	graph_name = basename(argv[optind]);
	if(print_info)
		std::cout << "Graph with " << graph.n << " vertices and " << graph.nnz << " edges" << std::endl;
	if(print_matrix)
		csr::print_matrix(graph,std::cout);

	std::random_device generator;
	std::uniform_int_distribution<int> distribution(0,graph.n);

	const char sep = ';';
	auto print_csv = [=] (const char* name, int n, int m, int s, const char* k, float t)
	{
		std::cout << name << sep << n << sep << m << sep << s << sep << k << sep << t;
	};
	// Run kernels
	// CSV header
	std::cout << "graph;vertices;edges;source;kernel;time" <<  (compare?";correctness":"") << std::endl;
	for(int i = 0; i < times; i++)
	{
		const int source = set_source ? set_source_vertex : distribution(generator);

		bfs::result cpu_result;
		if(compare)
		{
			cpu_result = cpu_bfs(graph,source);
			print_csv( graph_name, graph.n,graph.nnz, source,"CPU" ,cpu_result.total_time);
			if(compare)
				std::cout << ";OK";
			std::cout << std::endl;
		}

		for (auto& it: kernels_to_run)
		{
			auto bfs_func = it.second;
			bfs::result result = bfs_func(graph,source);

			// Print result in CSV format
			print_csv(graph_name, graph.n, graph.nnz, source, it.first, result.total_time);
			if(compare)
				compare_distance(cpu_result.distance, result.distance,graph.n);
			std::cout << std::endl;

		delete[] result.distance;
		}

		if(compare)
			delete[] cpu_result.distance;

	}

	csr::dispose_matrix(graph);
	return EXIT_SUCCESS;
}
