# Introduction
This repository contains the source code of the paper "[Are neural quantum states good at solving non-stoquastic Hamiltonians](https://arxiv.org/abs/2012.08889)."

Exact diagonalization is implemented using our custom C++ library [ExactDiagonalization](https://github.com/chaeyeunpark/ExactDiagonalization) and Arpack interface [ArpackSolver.hpp](include/ArpackSolver.hpp). It saves the ground stae in a binary format.

RBM codes are written in C++ using [Yannq](https://github.com/chaeyeunpark/Yannq).

For supervised learning, we have used Google JAX. To load the saved ground states, we implemented the python interface to the `Basis1DZ2` class of the ExactDiagonalization library using pybind11.
