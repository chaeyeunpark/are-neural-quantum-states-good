#pragma once
#include "arpackdef.h"

#include <vector>
#include <Eigen/Dense>
#include <random>

extern "C"
{
void dsaupd_(a_int *, char *, a_int *, char *, a_int *, double *,
			   double *, a_int *, double *, a_int *, a_int *, a_int *,
			   double *, double *, a_int *, a_int *);

void dseupd_(a_int* rvec, char* howmny, a_int* select, double* d, double* z,
			   a_int* ldz, double* sigma, char* bmat, a_int* n, char* which, 
			   a_int* nev, double* tol, double* resid, a_int* ncv, double* v,
			   a_int* ldv, a_int* iparam, a_int* inptr, double* workd, double* workl,
			   a_int* lworkl, a_int* info);
}


enum class ErrorType
{
	NormalExit, NotConverged, IncorrectParams, Others
};

template<typename MatrixVectorOp>
class ArpackSolver
{
private:
	MatrixVectorOp& op_;
	const a_int dim_;
	std::vector<double> d_;
	std::vector<double> z_;

public:
	ArpackSolver(MatrixVectorOp& op, a_int dim)
		: op_{op}, dim_{dim}
	{
	}

	ErrorType solve(a_int nev)
	{ 
		a_int dim = dim_;
		if(dim <= 0 || nev <= 0)
		{
			return ErrorType::IncorrectParams;
		}

		a_int ido = 0;
		char bmat[] = "I";
		a_int n = dim;
		char which[] = "SA";
		double tol = 1e-10;
		std::vector<double> resid(dim);

		a_int ncv = 3*nev;
		std::vector<double> v(ncv*dim);

		a_int ldv = dim;
		a_int iparam[11] = {
			1, //ishift
			0, //levec (not used)
			1000000000, //maxiter
			1, //nb
			0, //nconv
			0, //iupd (not used)
			1, //mode (1 is usual eigenvalue problem)
			0, //np
			0, 0, 0 //only for output
		};

		a_int ipntr[14];
		std::vector<double> workd(3*dim, 0.);

		int lworkl = 3*ncv*ncv + 6*ncv;
		std::vector<double> workl(lworkl, 0);

		a_int info = 0;

		//first call
		dsaupd_(&ido, bmat, &n, which, &nev, &tol, resid.data(), &ncv, 
				v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(),
				&lworkl, &info);

		while(ido == -1 || ido == 1)
		{
			op_.perform_op(workd.data() + ipntr[0]-1, workd.data() + ipntr[1]-1);

			dsaupd_(&ido, bmat, &n, which, &nev, &tol, resid.data(), &ncv, 
				v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(),
				&lworkl, &info);
		}

		if(info == 1 || iparam[4] != nev)
		{
			return ErrorType::NotConverged;
		}
		else if(info != 0)
		{
			return ErrorType::IncorrectParams;
		}

		a_int rvec = 1;

		d_.resize(nev);
		z_.resize((dim)*(nev));
		a_int ldz = dim;
		double sigma = 0;

		a_int select[ncv];
		for(int i = 0; i < ncv; i++)
		{
			select[i] = 1;
		}
		char howmny[] = "All";
		dseupd_(&rvec, howmny, select, d_.data(), z_.data(),
				&ldz, &sigma, bmat, &n, which,
				&nev, &tol, resid.data(), &ncv, v.data(), 
				&ldv, iparam, ipntr, workd.data(), workl.data(), &lworkl, &info);

		return ErrorType::NormalExit;
	}

	Eigen::Map<const Eigen::VectorXd> eigenvalues() const
	{
		return Eigen::Map<const Eigen::VectorXd>(d_.data(), d_.size());
	}

	Eigen::Map<const Eigen::MatrixXd> eigenvectors() const
	{
		return Eigen::Map<const Eigen::MatrixXd>(z_.data(), dim_, d_.size());
	}


};
