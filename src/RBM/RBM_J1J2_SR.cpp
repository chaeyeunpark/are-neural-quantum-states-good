#include <iostream>
#include <iomanip>
#include <fstream>
#include <ios>
#include <nlohmann/json.hpp>

#include <Yannq.hpp>

#include <Runners/RunRBM.hpp>
#include <Hamiltonians/J1J2.hpp>

using namespace yannq;
using std::ios;

int main(int argc, char** argv)
{
	using namespace yannq;
	using nlohmann::json;

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

	const double J2 = paramIn.at("J2").get<double>();
	const bool signRule = paramIn.at("sign_rule").get<bool>();

	const int nTemp = paramIn.at("number_of_temperatures").get<int>();
	const int nChainsPer = paramIn.at("number_of_chains_per_each").get<int>();
	const int nDownSample = paramIn.value("down_sample", 1);
	
	auto runner = RunRBM<ValT>(N, alpha, true, std::cerr);
	if(signRule)
		runner.setLambda(1.0, 0.9, 1e-3);
	runner.initializeRandom(1e-3);
	runner.setIterParams(3000, 100);
	runner.setOptimizer(paramIn["optimizer"]);
	json out = runner.getParams();

	
	auto callback = [](int iterIdx, double currEnergy, double nv, double cgErr, auto tSampling, auto tSolving)
	{
		std::cout << iterIdx << "\t" << currEnergy << "\t" << nv << "\t" << cgErr
			<< "\t" << tSampling << "\t" << tSolving << std::endl;
	};

	J1J2 ham(N, 1.0, J2, signRule);
	out["hamiltonian"] = ham.params();

	auto dim = runner.getDim();

	auto randomizer = [N](auto& re)
	{
		return randomSigma(N, N/2, re);
	};
	auto sweeper = SwapSweeper{N, nDownSample};
	auto sampler = runner.createSampler(sweeper, nTemp, nChainsPer);
	out["sampler"] = sampler.desc();

	{
		std::ofstream paramOut("paramOut.json");
		paramOut << out << std::endl;
	}


	runner.run(sampler, callback, randomizer, ham, dim, int(0.2*dim));
	return 0;
}
