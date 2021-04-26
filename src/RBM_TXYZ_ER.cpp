#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <nlohmann/json.hpp>

#include "Yannq.hpp"
#include "Hamiltonians/XYZNNN.hpp"
#include "Hamiltonians/XYZSto.hpp"
#include "Hamiltonians/XYZSto2.hpp"

#include "Runners/RunRBMExact.hpp"

using namespace yannq;
using std::ios;

int main(int argc, char** argv)
{
	using namespace yannq;
	using nlohmann::json;

	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	using ValT = std::complex<double>;

	if(argc != 2)
	{
		printf("Usage: %s [param.json]\n", argv[0]);
		return 1;
	}

	json paramIn;
	std::ifstream fin(argv[1]);
	fin >> paramIn;
	fin.close();

	const unsigned int N = paramIn.at("N").get<int>();
	const int alpha = paramIn.at("alpha").get<int>();

	const int useSto = paramIn.at("use_sto").get<int>();

	const double a = paramIn.at("a").get<double>();
	const double b = paramIn.at("b").get<double>();

	const int nIter = paramIn.value("nIter", 2000);

	auto runner = RunRBMExact<ValT>{N, alpha, true, std::cerr};
	runner.setLambda(10.0, 0.9, 1e-3);
	runner.initializeRandom(1e-3);
	runner.setIterParams(nIter, 100); // 10
	runner.setOptimizer(paramIn["optimizer"]);

	json out = runner.getParams();
	
	auto callback = [](int ll, double currE, double nv)
	{
		std::cout << ll << "\t" << currE << "\t" << nv << std::endl;
	};
	if(useSto == 0)
	{
		std::cerr << "Using XYZNNN" << std::endl;
		BasisFull basis(N);
		XYZNNN ham(N, a, b);
		out["hamiltonian"] = ham.params();
		runner.run(callback, std::move(basis), std::move(ham));
	}
	else if(useSto == 1)
	{
		std::cerr << "Using XYZSto" << std::endl;
		BasisFull basis(N);
		XYZNNNSto ham(N, a, b);
		out["hamiltonian"] = ham.params();
		runner.run(callback, std::move(basis), std::move(ham));
	}
	else
	{
		std::cerr << "Using XYZSto2" << std::endl;
		BasisFull basis(N);
		XYZNNNSto2 ham(N, a, b);
		out["hamiltonian"] = ham.params();
		runner.run(callback, std::move(basis), std::move(ham));
	}

	{
		std::ofstream paramOut("paramOut.json");
		paramOut << out << std::endl;
	}

	return 0;
}
