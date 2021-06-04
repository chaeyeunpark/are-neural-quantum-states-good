#include <iostream>
#include <pybind11/pybind11.h>

#include "Basis1DZ2_numpy.hpp"

namespace py = pybind11;


PYBIND11_MODULE(basis_numpy, m) {
	py::class_<Basis1DZ2>(m, "Basis1DZ2")
		.def(py::init<int32_t, bool>(), py::arg("N"), py::arg("use_U1"))
		.def_static("to_bin_array", &Basis1DZ2::to_bin_array)
        .def_readonly("N", &Basis1DZ2::N_)
		.def("load_data", &Basis1DZ2::load_data)
        .def("get_dim", &Basis1DZ2::get_dim)
		.def("sample_from_state", &Basis1DZ2::sample_from_state)
		.def("sample_from_state_idx", &Basis1DZ2::sample_from_state_idx)
		.def("generate_batch", &Basis1DZ2::generate_batch)
		.def("basis_degen", &Basis1DZ2::basis_degen)
		.def("generate_batch_coeffs", &Basis1DZ2::generate_batch_coeffs)
		.def("coeff", &Basis1DZ2::coeff)
        .def("coeffs_tensor", &Basis1DZ2::coeffs_tensor)
		.def("basis_vectors", &Basis1DZ2::basis_vectors);
}
