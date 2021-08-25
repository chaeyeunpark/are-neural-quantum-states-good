#pragma once

#include <string>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <regex>
#include <random>

#include <tbb/tbb.h>

#include <Basis/TIBasisZ2.hpp>
#include "utils.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Dense>

namespace py = pybind11;

void free_int8(void* p)
{
	int8_t* m = reinterpret_cast<int8_t*>(p);
	delete[] m;
}
void free_double(void* p)
{
	double* m = reinterpret_cast<double*>(p);
	delete[] m;
}
void free_uint32(void* p)
{
	uint32_t* m = reinterpret_cast<uint32_t*>(p);
	delete[] m;
}

class Basis1DZ2
{
public:
	const uint32_t N_;

private:
	bool loaded_ = false;
	std::unique_ptr<TIBasisZ2<uint64_t>> basis_;
	std::vector<double> coeffs_;
	std::vector<long double> cumsum_;

	mutable tbb::enumerable_thread_specific<std::default_random_engine> re_;


	static uint32_t extract_size(const std::string& path)
	{
		namespace fs = std::filesystem;

		auto fpath = fs::path(path);
		auto dirname = fpath.parent_path().filename().string();

		std::regex n_rgx("N(\\d+)");

		std::smatch n_match;
		if(!std::regex_search(dirname, n_match, n_rgx))
			throw std::invalid_argument(
					std::string("Path must include N(\\d+). Given: ") + path);

		uint32_t N = stoi(n_match[1].str());
		return N;
	}

public:
	Basis1DZ2(uint32_t N, bool use_U1)
		: N_{N}
	{
		int num_threads = get_num_threads();
		std::cerr << "Constructing Basis using " << num_threads << " threads." << std::endl;
		tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);
		basis_ = std::make_unique<TIBasisZ2<uint64_t> >(N, 0, use_U1, 1);

		std::random_device rd;

		for(auto& r: re_)
			r.seed(rd());

	}

	Basis1DZ2(Basis1DZ2&& rhs) 
		: N_{rhs.N_}, loaded_{rhs.loaded_}, basis_{std::move(rhs.basis_)},
		coeffs_{std::move(rhs.coeffs_)}, cumsum_{std::move(rhs.cumsum_)}
	{
		std::random_device rd;

		for(auto& r: re_)
			r.seed(rd());

	}


	static py::array_t<int8_t> to_bin_array(int64_t nbits, uint64_t number)
	{
		int8_t* res = new int8_t[nbits];
		for(int64_t n = 0; n < nbits; ++n)
		{
			res[n] = int8_t(number & 1);
			number >>= 1;
		}
		py::capsule free_when_done(res, free_int8);
		return py::array_t<int8_t>({nbits}, {sizeof(int8_t)}, res, free_when_done);
	}

	void load_data(const std::string& path)
	{
		int num_threads = get_num_threads();
		std::cerr << "Load data using " << num_threads << " threads." << std::endl;
		tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);
		
		namespace fs = std::filesystem;

		const auto N = extract_size(path);

		if(N != N_)
		{
			throw std::invalid_argument("Size of lattice for basis mismatches"
					"with that of the data.");
		}

		coeffs_.resize(basis_->getDim());

		{
			std::ifstream wf_in(path, std::ios::binary);
			uint32_t dim_in;
			wf_in.read((char*)&dim_in, sizeof(uint32_t));

			if(dim_in != basis_->getDim())
			{
				std::ostringstream error_stream;
				error_stream << 
					"Dimension of the saved wavefunction " << dim_in <<
					" does not match with basis dimension " << basis_->getDim();
				throw std::invalid_argument(error_stream.str());
			}

			wf_in.read((char*)coeffs_.data(), dim_in*sizeof(double));
		}
		count_pos_neg counter;
		tbb::parallel_reduce(
			tbb::blocked_range(coeffs_.cbegin(), coeffs_.cend()),
			counter
		);

		if(counter.n_pos < counter.n_neg)
		{
			tbb::parallel_for_each(coeffs_.begin(), coeffs_.end(),
				[&](double& c)
			{
				c *= -1;
			});
		}

		std::vector<double> probs(basis_->getDim());
		tbb::parallel_for(size_t(0), coeffs_.size(), 
			[&](size_t idx)
		{
			probs[idx] = coeffs_[idx]*coeffs_[idx];
		});
		cumsum_.resize(basis_->getDim());
		std::partial_sum(probs.begin(), probs.end(), cumsum_.begin());
		cumsum_.back() += 1e-6;

		loaded_ = true;
	}

	/*
	 * returns coeff and vector of confs
	 * @param p_tensor a tensor of shape (n_batch) that contains random values from 0 to 1
	 */
	uint32_t sample_from_state_idx(long double p) const
	{
		if(!loaded_)
		{
			throw std::invalid_argument("You should load data first");
		}
		auto it = std::lower_bound(cumsum_.begin(), cumsum_.end(), p);
		return std::distance(cumsum_.begin(), it);
	}

	/*
	 * returns coeff and vector of confs
	 * @param p_tensor a tensor of shape (n_batch) that contains random values from 0 to 1
	 */
	std::pair<double, std::vector<uint64_t>> sample_from_state(double p) const
	{
		if(!loaded_)
		{
			throw std::invalid_argument("You should load data first");
		}
		auto it = std::lower_bound(cumsum_.begin(), cumsum_.end(), p);
		uint32_t idx = std::distance(cumsum_.begin(), it);

		return std::make_pair(coeffs_[idx], basis_vectors(idx));
	}

	std::pair<py::array_t<int8_t>, py::array_t<int8_t>> generate_batch(uint32_t n_batch) const
	{
		if(!loaded_)
		{
			throw std::invalid_argument("You should load data first");
		}

		int8_t* confs = new int8_t[n_batch*N_];
		int8_t* signs = new int8_t[n_batch];
	
		tbb::parallel_for(size_t(0), size_t(n_batch),
			[&](size_t i)
		{
			std::uniform_real_distribution<long double> rdist(0., 1.);
			auto sampled_state = sample_from_state(rdist(re_.local()));

			if (sampled_state.first > 0)
			{
				signs[i] = 1;
			}
			else
			{
				signs[i] = 0;
			}
			std::uniform_int_distribution<> idist(0, sampled_state.second.size()-1);
			uint64_t conf = sampled_state.second[idist(re_.local())];

			for(uint32_t k = 0; k < N_; ++k)
			{
				confs[i*N_ + k] = (conf >> k) & 1;
			}
		});

		py::capsule free_confs_when_done(confs, free_int8);
		py::capsule free_signs_when_done(signs, free_int8);

		return std::make_pair(
			py::array_t<int8_t>({n_batch}, {1}, signs, free_signs_when_done),
			py::array_t<int8_t>({n_batch, N_}, confs, free_confs_when_done)
		);
	}

	std::pair<py::array_t<double>, py::array_t<int8_t>> 
	generate_batch_coeffs(uint32_t n_batch) const
	{
		using std::sqrt;
		if(!loaded_)
		{
			throw std::invalid_argument("You should load data first");
		}
		int8_t* confs = new int8_t[n_batch*N_];
		double* coeffs = new double[n_batch];
	
		tbb::parallel_for(size_t(0), size_t(n_batch),
			[&](size_t i)
		{
			std::uniform_real_distribution<long double> rdist(0., 1.);
			auto sampled_state = sample_from_state(rdist(re_.local()));

			coeffs[i] = sampled_state.first / sqrt(sampled_state.second.size());

			std::uniform_int_distribution<> idist(0, sampled_state.second.size()-1);
			uint64_t conf = sampled_state.second[idist(re_.local())];

			for(uint32_t k = 0; k < N_; ++k)
			{
				confs[i*N_ + k] = (conf >> k) & 1;
			}
		});

		py::capsule free_coeffs_when_done(coeffs, free_double);
		py::capsule free_confs_when_done(confs, free_int8);

		return std::make_pair(
			py::array_t<double>({n_batch}, {sizeof(double)}, coeffs, free_coeffs_when_done),
			py::array_t<int8_t>({n_batch, N_}, confs, free_confs_when_done)
		);
	}


	size_t get_dim() const
	{
		return basis_->getDim();
	}

	double coeff(uint64_t idx) const
	{
		if(!loaded_)
		{
			throw std::invalid_argument("You should load data first");
		}
		return coeffs_[idx];
	}

	py::array_t<uint32_t> basis_degen() const
	{
		if(!loaded_)
		{
			throw std::invalid_argument("You should load data first");
		}
		auto dim = static_cast<long>(get_dim());

		uint32_t* q = new uint32_t[dim];

		for(long idx = 0; idx < dim; ++idx)
		{
			auto bvec = basis_->basisVec(idx);
			q[idx] = bvec.size();
		}

		py::capsule free_when_done(q, free_uint32);
		return py::array_t<uint32_t>({dim}, q, free_when_done);
	}

	py::array_t<double> coeffs_tensor() const
    {
		if(!loaded_)
		{
			throw std::invalid_argument("You should load data first");
		}
		auto dim = static_cast<long>(get_dim());
        return py::array_t<double>({dim}, coeffs_.data());
    }

	py::array_t<int8_t> sample_from_basis_vectors(py::array_t<int32_t> indices)
	{
		typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

        /* Request a buffer descriptor from Python */
        py::buffer_info info = indices.request();

        if (info.format != py::format_descriptor<int32_t>::format())
            throw std::runtime_error("Incompatible format: expected a double array!");

        if (info.ndim != 1)
            throw std::runtime_error("Incompatible buffer dimension!");

		uint32_t n_batch = info.shape[0];

		int8_t* confs = new int8_t[n_batch*N_];

		const int32_t stride = info.strides[0] / sizeof(int32_t);
		tbb::parallel_for(size_t(0), size_t(n_batch), 
				[&](size_t i)
		{
			int32_t idx = *((int32_t*)info.ptr + i*stride);
			
			auto bvecs = basis_vectors(idx);
			std::uniform_int_distribution<> idist(0, bvecs.size()-1);
			auto conf = bvecs[idist(re_.local())];

			for(uint32_t k = 0; k < N_; ++k)
			{
				confs[i*N_ + k] = (conf >> k) & 1;
			}
		});

		py::capsule free_confs_when_done(confs, free_int8);

		return py::array_t<int8_t>({n_batch, N_}, confs, free_confs_when_done);
	}

	std::vector<uint64_t> basis_vectors(uint64_t idx) const
	{
		auto bvec = basis_->basisVec(idx);
		std::vector<uint64_t> confs(bvec.size());

		std::transform(bvec.begin(), bvec.end(), confs.begin(), 
				[&](const std::pair<uint64_t, double>& p)
				{
				return p.first;
				});
		return confs;
	}

};
