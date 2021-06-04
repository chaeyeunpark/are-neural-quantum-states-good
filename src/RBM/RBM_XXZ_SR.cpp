#include <iostream>
#include <iomanip>
#include <fstream>
#include <ios>
#include <nlohmann/json.hpp>

#include "Yannq.hpp"

#include "Hamiltonians/XXZ.hpp"
#include "Hamiltonians/XXZSto.hpp"
#include "Runners/RunRBM.hpp"
#include "Serializers/SerializeRBM.hpp"

using namespace yannq;
using std::ios;

int main(int argc, char** argv)
{
	using namespace yannq;
	using nlohmann::json;

	std::cout << std::setprecision(8);

	using ValT = std::complex<double>;

	if(argc != 2)
	{
		printf("Usage: %s [param.json]\n", argv[0]);
		return 1;
	}
	
	json paramIn;
	{
		std::ifstream fin(argv[1]);
		fin >> paramIn;
		fin.close();
	}
	
	const uint32_t N = paramIn.at("N").get<uint32_t>();
	const uint32_t alpha = paramIn.at("alpha").get<uint32_t>();

	const bool signRule = paramIn.at("sign_rule").get<bool>();
	const double delta = paramIn.at("delta").get<double>();
	const int nTemp = paramIn.at("number_of_temperatures").get<int>();
	const int nChainsPer = paramIn.at("number_of_chains_per_each").get<int>();

	std::cout << "#delta: " << delta << std::endl;

	auto callback = [&](int ll, double currE, double nv, double cgErr, long int smp_dur, long int slv_dur)
	{
		std::cout << ll << "\t" << currE << "\t" << nv << "\t" << cgErr
			<< "\t" << smp_dur << "\t" << slv_dur << std::endl;
	};

	auto runner = RunRBM<ValT>{N, alpha, true, std::cerr};

	const auto dim = runner.getDim();
	
	runner.setLambda(10.0, 0.9, 1e-3);
	runner.initializeRandom(1e-3);
	runner.setOptimizer(paramIn["optimizer"]);
	runner.setIterParams(2000, 100);
	json out = runner.getParams();
	
	auto randomizer = [N](auto& re)
	{
		return randomSigma(N, N/2, re);
	};

	XXZ ham(N, 1.0, delta, signRule);
	out["hamiltonian"] = ham.params();

	auto sweeper = SwapSweeper{N};
	auto sampler = runner.createSampler(sweeper, nTemp, nChainsPer);
	out["sampler"] = sampler.desc();

	{
		std::ofstream paramOut("paramOut.json");
		paramOut << out << std::endl;
	}


	runner.run(sampler, callback, randomizer, std::move(ham), dim, int(0.2*dim));

	return 0;
}
