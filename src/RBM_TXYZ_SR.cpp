#include <iostream>
#include <iomanip>
#include <fstream>
#include <ios>
#include <variant>
#include <nlohmann/json.hpp>

#include <Samplers/LocalSweeper.hpp>
#include "Yannq.hpp"

#include "Hamiltonians/XYZNNN.hpp"
#include "Hamiltonians/XYZSto.hpp"
#include "Hamiltonians/XYZSto2.hpp"
#include "Runners/RunRBM.hpp"

using namespace yannq;
using std::ios;
using nlohmann::json;

void printParamOut(const json& out)
{
	std::ofstream paramOut("paramOut.json");
	paramOut << out << std::endl;
}
int main(int argc, char** argv)
{
	using namespace yannq;

	std::cout << std::setprecision(10);

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

	const int useSto = paramIn.at("use_sto").get<int>();
	const double a = paramIn.at("a").get<double>();
	const double b = paramIn.at("b").get<double>();
	const double beta1 = paramIn.value("beta1", 0.0);
	const double beta2 = paramIn.value("beta2", 0.0);

	const int nTemp = paramIn.at("number_of_temperatures").get<int>();
	const int nChainsPer = paramIn.at("number_of_chains_per_each").get<int>();
	const int nDownSample = paramIn.value("down_sample", 1);

	std::cout << "#a: " << a << ", b:" << b << std::endl;

	auto callback = [&](int ll, double currE, double nv, double cgErr, long int smp_dur, long int slv_dur)
	{
		std::cout << ll << "\t" << currE << "\t" << nv << "\t" << cgErr
			<< "\t" << smp_dur << "\t" << slv_dur << std::endl;
	};

	auto runner = RunRBM<ValT>{N, alpha, true, std::cerr};

	auto randomizer = [N](auto& re)
	{
		return randomSigma(N, re);
	};
	runner.setLambda(10.0, 0.9, 1e-3);
	runner.setMomentum(beta1, beta2);
	runner.initializeRandom(1e-3);
	runner.setOptimizer(paramIn["optimizer"]);

	runner.setIterParams(2000, 100);
	
	json out = runner.getParams();

	auto dim = runner.getDim();
	LocalSweeper sweeper{N, nDownSample};

	auto sampler = runner.createSampler(sweeper, nTemp, nChainsPer);
	out["sampler"] = sampler.desc();

	std::variant<std::monostate, XYZNNN, XYZNNNSto, XYZNNNSto2> ham;

	if(useSto == 0)
	{
		std::cerr << "Using XYZNNN" << std::endl;
		ham = XYZNNN{N,a,b};
	}
	else if(useSto == 1)
	{
		std::cerr << "Using XYZSto" << std::endl;
		ham = XYZNNNSto{N, a, b};
	}
	else
	{
		std::cerr << "Using XYZSto2" << std::endl;
		ham = XYZNNNSto2{N, a, b};
	}
	std::visit(
		[&](auto&& arg){
			using T = std::decay_t<decltype(arg)>;
			if constexpr(std::is_same_v<std::monostate, T>)
			{
				std::cerr << "Hamiltonian is not initialized" << std::endl;
			}
			else 
			{
				out["hamiltonian"] = arg.params();
				printParamOut(out);
				runner.run(sampler, callback, randomizer, arg, dim, int(0.2*dim));
			}
		}, ham
	);

	return 0;
}
