#include <iostream>
#include <iomanip>
#include <fstream>
#include <ios>
#include <nlohmann/json.hpp>

#include <Yannq.hpp>
#include <Hamiltonians/J1J2.hpp>

#include <Runners/RunRBMExact.hpp>

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
	std::ifstream fin(argv[1]);
	fin >> paramIn;
	fin.close();

	const uint32_t N = paramIn.at("N").get<uint32_t>();
	const int alpha = paramIn.at("alpha").get<int>();

	const double J2 = paramIn.at("J2").get<double>();
	const bool signRule = paramIn.at("sign_rule").get<bool>();
	
	RunRBMExact<ValT> runner{N, alpha, true, std::cerr};
	if(signRule)
		runner.setLambda(1.0, 0.9, 1e-3);
	runner.initializeRandom(1e-3);
	runner.setIterParams(3000, 100);
	runner.setOptimizer(paramIn["optimizer"]);

	
	auto callback = [](int ll, double currE, double nv)
	{
		std::cout << ll << "\t" << currE << "\t" << nv << std::endl;
	};

	J1J2 ham(N, 1.0, J2, signRule);

	{
		std::ofstream paramOut("paramOut.json");
		json out = runner.getParams();
		out["hamiltonian"] = ham.params();
		paramOut << out << std::endl;
	}
	BasisJz basis(N, N/2);
	runner.run(callback, std::move(basis), std::move(ham));
	return 0;
}
