#pragma once
/// @file ArrayDOF_bind.hpp
/// @brief pybind11 bindings for @ref DNDS::ArrayDof "ArrayDof", mirroring the C++ vector-space
/// operations (norm, dot, AXPY, reductions) into Python.

#include "ArrayDOF.hpp"

#include "Array_bind.hpp"
#include "pybind11/eigen.h"
#include "ArrayDerived/ArrayEigenMatrix_bind.hpp"

namespace DNDS
{
    template <int n_m, int n_n>
    std::string pybind11_ArrayDOF_name()
    {
        return "ArrayDOF" + fmt::format("_{}_{}",
                                        RowSize_To_PySnippet(n_m),
                                        RowSize_To_PySnippet(n_n));
    }

    template <int n_m, int n_n> // ! shared pointer managing
    using tPy_ArrayDOF = py_class_ssp<ArrayDof<n_m, n_n>>;
}

namespace DNDS
{
    template <int n_m, int n_n>
    tPy_ArrayDOF<n_m, n_n>
    pybind11_ArrayDOF_declare(py::module_ &m)
    {
        // inherit from ArrayEigenMatrixPair
        auto ArrayEigenMatrixPair_ = pybind11_ArrayEigenMatrixPair_get_class<n_m, n_n>(m);
        return {m, pybind11_ArrayDOF_name<n_m, n_n>().c_str(), ArrayEigenMatrixPair_};
    }

    template <int n_m, int n_n>
    tPy_ArrayDOF<n_m, n_n>
    pybind11_ArrayDOF_get_class(py::module_ &m)
    {
        return {m.attr(pybind11_ArrayDOF_name<n_m, n_n>().c_str())};
    }

    template <int n_m, int n_n>
    void pybind11_ArrayDOF_define(py::module_ &m)
    {
        using TArr = ArrayDof<n_m, n_n>;
        using t_element_mat = typename TArr::t_element_mat;
        auto Arr_ = pybind11_ArrayDOF_declare<n_m, n_n>(m);

        Arr_
            .def(py::init([]()
                          { return std::make_shared<TArr>(); }));
        // use inherited pair methods
        Arr_
            .def("clone", [](TArr &self)
                 {
                     auto new_pair = std::make_shared<TArr>();
                     new_pair->clone(self); 
                     return new_pair; });
        Arr_
            .def("setConstant", [](TArr &self, real R)
                 { self.setConstant(R); }, py::arg("R"))
            .def("setConstant", [](TArr &self, const Eigen::Ref<const t_element_mat> &R)
                 { self.setConstant(R); }, py::arg("R"));
        Arr_
            .def("__iadd__", [](TArr &self, const TArr &R)
                 { self += R;return self; }, py::arg("R"))
            .def("__iadd__", [](TArr &self, real R)
                 { self += R;return self; }, py::arg("R"))
            .def("__iadd__", [](TArr &self, const Eigen::Ref<const t_element_mat> &R)
                 { self += R;return self; }, py::arg("R"));
        Arr_.def("__isub__", [](TArr &self, const TArr &R)
                 { self -= R;return self; }, py::arg("R"));
        Arr_
            .def("__imul__", [](TArr &self, real R)
                 { self *= R;return self; }, py::arg("R"))
            .def("__imul__", [](TArr &self, const Eigen::Ref<const t_element_mat> &R)
                 { self *= R;return self; }, py::arg("R"))
            .def("__imul__", [](TArr &self, const TArr &R)
                 { self *= R;return self; }, py::arg("R"));

        if constexpr (n_m != 1 || n_n != 1)
        {
            Arr_
                .def("__imul__", [](TArr &self, ArrayDof<1, 1> &R)
                     { self *= R; return self; }, py::arg("R"));
        }
        Arr_
            .def("__idiv__", [](TArr &self, const TArr &R)
                 { self /= R;return self; }, py::arg("R"));

        Arr_
            .def("assign_value", [](TArr &self, const TArr &R)
                 { self = R; }, py::arg("R"))
            .def("addTo", [](TArr &self, const TArr &R, real alpha)
                 { self.addTo(R, alpha); }, py::arg("R"), py::arg("alpha"))
            .def("norm2", [](TArr &self)
                 { return self.norm2(); })
            .def("norm2", [](TArr &self, TArr &R)
                 { return self.norm2(R); })
            .def("min", [](TArr &self)
                 { return self.min(); })
            .def("max", [](TArr &self)
                 { return self.max(); })
            .def("sum", [](TArr &self)
                 { return self.sum(); })
            .def("componentWiseNorm1", [](TArr &self)
                 { return self.componentWiseNorm1(); })
            .def("componentWiseNorm1", [](TArr &self, TArr &R)
                 { return self.componentWiseNorm1(R); })
            .def("dot", [](TArr &self, const TArr &R)
                 { return self.dot(R); }, py::arg("R"));
    }

    template <int n_m, int n_n>
    void _pybind11_ArrayDOF_define_dispatch(py::module_ &m)
    {
        // std::cout << fmt::format("binding ArrayEigenMatrixPair {},{}", _mat_ni, _mat_nj) << std::endl;
        if constexpr (n_m == UnInitRowsize || n_n == UnInitRowsize)
            return;
        else
            return pybind11_ArrayDOF_define<n_m, n_n>(m);
    }
}

namespace DNDS
{
    template <rowsize mat_n, size_t N, std::array<int, N> const &Arr, size_t... Is>
    void pybind11_callBindArrayDOFs_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...> /*unused*/)
    {
        (_pybind11_ArrayDOF_define_dispatch<Arr[Is], mat_n>(m), ...);
        pybind11_ArrayDOF_define<DynamicSize, mat_n>(m);
        pybind11_ArrayDOF_define<NonUniformSize, mat_n>(m);
    }

    template <rowsize mat_n>
    void pybind11_callBindArrayDOF_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = pybind11_arrayRowsizeInstantiationList;
        pybind11_callBindArrayDOFs_rowsizes_sequence<
            mat_n,
            seq.size(),
            seq>(m, std::make_index_sequence<seq.size()>{});
    }

    extern template void pybind11_callBindArrayDOF_rowsizes<1>(py::module_ &m);
    extern template void pybind11_callBindArrayDOF_rowsizes<2>(py::module_ &m);
    extern template void pybind11_callBindArrayDOF_rowsizes<3>(py::module_ &m);
    extern template void pybind11_callBindArrayDOF_rowsizes<4>(py::module_ &m);
    extern template void pybind11_callBindArrayDOF_rowsizes<5>(py::module_ &m);
    extern template void pybind11_callBindArrayDOF_rowsizes<6>(py::module_ &m);
    extern template void pybind11_callBindArrayDOF_rowsizes<7>(py::module_ &m);
    extern template void pybind11_callBindArrayDOF_rowsizes<8>(py::module_ &m);
    extern template void pybind11_callBindArrayDOF_rowsizes<DynamicSize>(py::module_ &m);
    extern template void pybind11_callBindArrayDOF_rowsizes<NonUniformSize>(py::module_ &m);

    void pybind11_bind_ArrayDOF_All(py::module_ &m);
}