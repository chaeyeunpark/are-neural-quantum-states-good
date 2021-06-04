#include <iostream>
#include <fstream>
#include <ios>

#include <cereal/archives/binary.hpp>
#include <tbb/tbb.h>

#include "ParallelMV.hpp"
#include "Basis/TIBasisZ2.hpp"

#include "TITXYZ.hpp"
#include "TITXYZSto.hpp"

#include "ArpackSolver.hpp"

int main(int argc, char* argv[])
{
	int N;

	if(argc != 2)
	{
		printf("Usage: %s [N]\n", argv[0]);
		return 1;
	}
	sscanf(argv[1], "%d", &N);


	char* tbb_num_threads = std::getenv("TBB_NUM_THREADS");
	int num_threads = 0;
	if((tbb_num_threads != nullptr) && (tbb_num_threads[0] != '\0'))
	{
		num_threads = atoi(tbb_num_threads);
	}
	if(num_threads <= 0)
		num_threads = tbb::this_task_arena::max_concurrency();

	std::cerr << "Set number of threads: " << num_threads << std::endl;
	tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);


	std::cout << "#N: " << N << std::endl;
	using UINT = uint64_t;
	using INT = int64_t;

	double J1 = 1.0;

	using Basis = TIBasisZ2<UINT>;
	Basis basis(N, 0, false, 1);

	std::vector<std::pair<double, double>> params;

	for(uint32_t k = 0; k < 11; ++k)
	{
		double a = 0.5 - 0.01*k;
		double b = 2.0 + 0.1*k;
		params.emplace_back(a, b);
	}

	std::cout << "#dimensions: " << basis.getDim() << std::endl;

	for(const auto [a,b]: params)
	{
		TITXYZSto<UINT> ham(basis, a, b);
		const uint32_t dim = basis.getDim();

		ParallelMV mv(dim, ham);
		ArpackSolver<ParallelMV> solver(mv, dim);
		if(solver.solve(2) != ErrorType::NormalExit)
		{
			std::cerr << "error processing (a,b): (" << a << ", " << 
				b << ")" << std::endl;
			return 1;
		}
		auto evals = solver.eigenvalues();
		auto evecs = solver.eigenvectors();
		printf("%f\t%f\t%.12f\t%.12f\t\n", a, b, evals(0), evals(1));
		
		char fileName[255];
		sprintf(fileName, "GS_1DTXYZ_A%03d_B%03d.dat", int(100*a + 0.5), int(100*b + 0.5));
		{
			std::ofstream fout(fileName, std::ios::binary);
			fout.write((const char*)&dim, sizeof(dim));
			fout.write((const char*)evecs.data(), dim*sizeof(double));
			fout.close();
		}

		fflush(stdout);
	}

	return 0;
}
