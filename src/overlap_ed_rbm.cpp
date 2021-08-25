#include <regex>
#include <filesystem>
#include <cstdlib>

#include <Basis/TIBasisZ2.hpp>
#include <Basis/ToOriginalBasis.hpp>

#include <Yannq.hpp>
#include <cereal/archives/binary.hpp>
#include <Serializers/SerializeRBM.hpp>

#include <Eigen/Dense>

int main(int argc, char* argv[])
{
	namespace fs = std::filesystem;
	using Machine = yannq::RBM<std::complex<double>>;


	std::regex n_rgx("N(\\d+)");
	std::cmatch n_match;
	if(argc != 2)
	{
		printf("Usage: %s [result directory]\n", argv[0]);
	}
	if(!std::regex_search(argv[1], n_match, n_rgx))
	{
		fprintf(stderr, "Directory path should contain N(\\d+)\n");
		return 1;
	}

	const int N = std::stoi(n_match.str(1));
	TIBasisZ2<uint64_t> basis(N, 0, true, 1);

	std::regex j2_rgx("J2_(\\d+)");

	yannq::BasisJz basis_u1(N, N/2);
	std::vector<uint32_t> basis_u1_vecs(basis_u1.begin(), basis_u1.end());
	
	for(auto& p: fs::directory_iterator(fs::path(argv[1]) / "RBM"))
	{
		std::string filename = p.path().stem().string();
		std::smatch j2_match;
		if(!std::regex_search(filename, j2_match, j2_rgx))
		{
			continue;
		}
		double J2 = (double)stoi(j2_match.str(1))/100;
	
		//load RBM
		std::unique_ptr<Machine> rbm;
		{
			std::ifstream is(p.path());
			cereal::BinaryInputArchive archive(is);
			archive(rbm);
		}
		//std::cout << J2 << "\t" << rbm->getN() << std::endl;
		
		//load ED
		char ed_filename[255];
		sprintf(ed_filename, "GS_1DJ1J2_J2_%03d.dat", int(J2*100+0.5));
		fs::path p_ed = fs::path(argv[1]) / "ED" / ed_filename;

		if(!fs::exists(p_ed))
		{
			std::cerr << "ED Path " << p_ed << " does not exists; Skipping";
			continue;
		}

		std::vector<double> gs;
		{
			uint32_t dim;
			std::ifstream is(p_ed);
			is.read((char*)&dim, sizeof(uint32_t));

			if(dim != basis.getDim())
			{
				std::cerr << "Saved dimension " << dim << 
					" mismatches with the basis dimension "  << basis.getDim()
					<< std::endl;
				continue;
			}
			gs.resize(dim);
			is.read((char*)gs.data(), sizeof(double)*dim);
		}

		//construct computational basis states
		std::vector<double> ed_state = toOriginalVectorLM(basis, gs.data());

		std::vector<double> ed_state_u1;
		ed_state_u1.reserve(basis_u1_vecs.size());

		for(uint32_t i = 0; i < basis_u1_vecs.size(); ++i)
		{
			ed_state_u1.emplace_back(ed_state[basis_u1_vecs[i]]);
		}

		Eigen::Map<Eigen::VectorXd> ed_map(ed_state_u1.data(), ed_state_u1.size());

		Eigen::VectorXcd rbm_state = yannq::getPsi(*rbm, basis_u1_vecs, true);
		
		std::cout << J2 << "\t" << std::abs(std::complex<double>(rbm_state.transpose() * ed_map)) << std::endl;
	}
	
	return 0;
}
