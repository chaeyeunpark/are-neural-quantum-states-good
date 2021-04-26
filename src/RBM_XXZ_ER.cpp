#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <nlohmann/json.hpp>

#include <Yannq.hpp>
#include "Hamiltonians/XXZ.hpp"
#include "Hamiltonians/XXZSto.hpp"

#include "Runners/RunRBMExact.hpp"

using namespace yannq;
using std::ios;

int main(int argc, char** argv)
{
	using namespace yannq;
	using nlohmann::json;

	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(10);

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

	const int N = paramIn.at("N").get<int>();
	const int alpha = paramIn.at("alpha").get<int>();

	const double delta = paramIn.at("delta").get<double>();
	const bool signRule = paramIn.at("sign_rule").get<bool>();
	
	auto runner = RunRBMExact<ValT>{N, alpha, true, std::cerr};

	runner.setLambda(10.0, 0.9, 1e-3);
	runner.setIterParams(3500, 100);
	runner.initializeRandom(1e-3);
	runner.setOptimizer(paramIn["optimizer"]);

	json out = runner.getParams();
	
	auto callback = [](int ll, double currE, double nv)
	{
		std::cout << ll << "\t" << currE << "\t" << nv << std::endl;
	};

	XXZ ham(N, 1.0, delta, signRule);
	out["hamiltonian"] = ham.params();
	BasisJz basis(N, N/2);
	runner.run(callback, std::move(basis), std::move(ham));

	{
		std::ofstream paramOut("paramOut.json");
		paramOut << out << std::endl;
	}

	return 0;
}
