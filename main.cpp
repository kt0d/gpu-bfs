#include "csr_matrix.h"
#include "bfs_cpu.h"
#include "bfs.cuh"
#include "common.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <functional>
#include <list>
#include <utility>
#include <random>

#include <unistd.h>
#include <libgen.h>

const std::string csv_header = "graph;vertices;edges;source;kernel;time;depth";
const std::string csv_header_correct = "graph;vertices;edges;source;kernel;time;depth;correct";

void usage(const char *pname)
{
	std::cerr << "USAGE: " << pname << "[options] FILENAME" << std::endl
		<< "-p Print size of matrix" << std::endl
		<< "-m Print matrix" << std:: endl
		<< "-L Run linear-work BFS with blocking queue" << std::endl
		<< "-Q Run quadratic-work BFS" << std::endl
		<< "-E Run expand-contract BFS" << std::endl
		<< "-C Run contract-expand BFS" << std::endl
		<< "-T Run two-phase BFS" << std::endl
		<< "-c Run CPU BFS and compare results for corectness" << std::endl
		<< "FILENAME must be square pattern matrix stored in Rutherford Boeing sparse matrix format (*.rb)" << std::endl;
	exit(EXIT_FAILURE);
}

struct comparison_result
{
	int diff;
	int small;
	int inf;
};

// Compare dist2 to dist1, counting different values, values in dist2 smaller than in dist1, infinity balues in dist2
comparison_result compare_distance(const int* dist1,const int* dist2,const int n)
{
	comparison_result ret = {0, 0, 0};

	for(int i = 0; i < n; i++)
	{
		if(dist1[i] != dist2[i])
		{
			ret.diff++;
			if(dist1[i] > dist2[i]) ret.small++;
			if(dist2[i] == bfs::infinity) ret.inf++;
		}
	}

	return ret;
}

// Print result in CSV format
void print_csv(std::string name, int n, int m, int s, const std::string k, const bfs::result result, const int* dist_cpu)
{
	const char sep = ';';
	std::basic_ostringstream<char> output;
	output << name << sep << n << sep << m << sep << s << sep << k << sep << result.total_time << sep << result.depth;
	if(dist_cpu != nullptr)
	{
		output << ';';
		comparison_result cmp = compare_distance(dist_cpu,result.distance,n);
		if(cmp.diff == 0)
			output << "OK";
		else
		{
			output << "NOT OK " << cmp.diff << " DIFFERENT WITH " << cmp.small << " SMALLER";
		}
		if(cmp.inf > 0)
			output << " " << cmp.inf << " NOT REACHED";
	}
	output << std::endl;
	std::cout << output.str();
}

struct arguments
{
	bool compare = false, print_info = false, print_matrix = false;
	bool set_source = false;
	std::list<std::pair<std::string, std::function<bfs::result(csr::matrix, int)>>> kernels_to_run; 
	int times = 1;
	int set_source_vertex;
	std::string graph_location;
};

void process_arguments(int argc, char **argv, arguments& args)
{
	char c;
	while ((c = getopt(argc, argv, "cpmLQECTn:s:")) != -1)
		switch(c)
		{
			case 'L':
				args.kernels_to_run.push_back(std::make_pair("Linear",run_linear_bfs));
				break;
			case 'Q':
				args.kernels_to_run.push_back(std::make_pair("Quadratic", run_quadratic_bfs));
				break;
			case 'c':
				args.compare = true;
				break;
			case 'p':
				args.print_info=true;
				break;
			case 'm':
				args.print_matrix=true;
				break;
			case 'E':
				args.kernels_to_run.push_back(std::make_pair("Expand-contract", run_expand_contract_bfs));
				break;
			case 'C':
				args.kernels_to_run.push_back(std::make_pair("Contract-expand", run_contract_expand_bfs));
				break;
			case 'T':
				args.kernels_to_run.push_back(std::make_pair("Two-phase",run_two_phase_bfs));
				break;
			case 'n':
				args.times = atoi(optarg);
				break;
			case 's':
				args.set_source = true;
				args.set_source_vertex = atoi(optarg);
				break;

			default:
				usage(argv[0]);
		}

	if(optind < 2 || optind >= argc)
		usage(argv[0]);
	args.graph_location=argv[optind];
}

int main(int argc, char **argv)
{
	// Process options
	arguments args;
	process_arguments(argc,argv,args);


	// Load graph
	std::ifstream rb_file;
	rb_file.open(args.graph_location);
	if(!rb_file.good())
	{
		std::cerr << "Incorrect file" << std::endl;
		usage(argv[0]);
	}
	csr::matrix graph= csr::load_matrix(rb_file);
	rb_file.close();

	// If source vertex was chosen explicitly, check if it's correct
	if(args.set_source && (args.set_source_vertex < 0 || args.set_source_vertex >= graph.n))
	{
		std::cerr << "Vertex " << args.set_source_vertex << " does not belong to loaded graph" << std::endl;
		csr::dispose_matrix(graph);
		usage(argv[0]);
	}

	// Get filename
	std::string graph_name = basename(&args.graph_location[0]);
	if(args.print_info)
		std::cout << graph_name << " with " << graph.n << " vertices and " << graph.nnz << " edges" << std::endl;
	if(args.print_matrix)
		csr::print_matrix(graph,std::cout,true,false);
	if(args.kernels_to_run.size() == 0)
	{
		csr::dispose_matrix(graph);
		return EXIT_SUCCESS;
	}

	std::random_device generator;
	std::uniform_int_distribution<int> distribution(0,graph.n);

	if(args.compare)
		std::cout << csv_header_correct << std::endl;
	else
		std::cout << csv_header << std::endl;
	// Run kernels
	for(int i = 0; i < args.times; i++)
	{
		const int source = args.set_source ? args.set_source_vertex : distribution(generator);

		bfs::result cpu_result;
		cpu_result.distance = nullptr;
		if(args.compare)
		{
			cpu_result = cpu_bfs(graph,source);
			print_csv(graph_name, graph.n, graph.nnz, source, "CPU", cpu_result, cpu_result.distance);
		}

		for (auto& it: args.kernels_to_run)
		{
			auto bfs_func = it.second;
			bfs::result result = bfs_func(graph,source);

			print_csv(graph_name, graph.n, graph.nnz, source, it.first, result, cpu_result.distance);

			delete[] result.distance;
		}

		if(args.compare)
			delete[] cpu_result.distance;

	}

	csr::dispose_matrix(graph);
	return EXIT_SUCCESS;
}
