# Introduction
Source codes for the paper "[Are neural quantum states good at solving non-stoquastic Hamiltonians](https://arxiv.org/abs/2012.08889)."

Exact diagonalization is implemented using our custom C++ library [ExactDiagonalization](https://github.com/chaeyeunpark/ExactDiagonalization) and Arpack interface [ArpackSolver.cpp](include/ArpackSolver.hpp).
RBM codes is written with [Yannq](https://github.com/chaeyeunpark/Yannq).

For supervised learning, we have used Google JAX. 
