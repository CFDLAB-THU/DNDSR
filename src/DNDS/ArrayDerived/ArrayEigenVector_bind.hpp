#pragma once

#include "ArrayEigenVector.hpp"
#include "../Array_bind.hpp"

namespace DNDS
{
    template <rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    std::string pybind11_ArrayEigenVector_name_appends()
    {
        return fmt::format("_{}_{}_{}",
                           RowSize_To_PySnippet(_vec_size),
                           RowSize_To_PySnippet(_row_max),
                           RowSize_To_PySnippet(_align));
    }

    template <rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    std::string pybind11_ArrayEigenVector_name()
    {
        return "ArrayEigenVector" + pybind11_ArrayEigenVector_name_appends<_vec_size, _row_max, _align>();
    }

    template <rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    using tPy_ArrayEigenVector = py::class_<ArrayEigenVector<_vec_size, _row_max, _align>, ssp<ArrayEigenVector<_vec_size, _row_max, _align>>>;
}

namespace DNDS
{
    template <rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    tPy_ArrayEigenVector<_vec_size, _row_max, _align>
    pybind11_ArrayEigenVector_declare(py::module_ &m)
    {
        auto ParArray_ = pybind11_ParArray_get_class<real, _vec_size, _row_max, _align>(m);
        return {m, pybind11_ArrayEigenVector_name<_vec_size, _row_max, _align>().c_str(), ParArray_};
        // std::cout << py::format_descriptor<Eigen::Matrix<double, 3, 3>>().format() << std::endl;
    }

    template <rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    tPy_ArrayEigenVector<_vec_size, _row_max, _align>
    pybind11_ArrayEigenVector_get_class(py::module_ &m)
    {
        return {m.attr(pybind11_ArrayEigenVector_name<_vec_size, _row_max, _align>().c_str())};
    }

    template <rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    void pybind11_ArrayEigenVector_define(py::module_ &m)
    {

        using TArrayEigenVector = ArrayEigenVector<_vec_size, _row_max, _align>;
        auto ArrayEigenVector_ = pybind11_ArrayEigenVector_declare<_vec_size, _row_max, _align>(m);

        // // helper
        // using TArrayEigenVector = ArrayEigenVector<1>;
        // auto ArrayEigenVector_ = pybind11_ArrayEigenVector_declare<1>(m);
        // // helper

        ArrayEigenVector_
            // we only bind the non-default ctor here
            .def(py::init<const MPIInfo &>(), py::arg("nmpi"))
            .def(
                "__getitem__",
                [](TArrayEigenVector &self, index index_)
                {
                    auto row = self[index_];
                    return py::memoryview::from_buffer<real>(
                        row.data(),
                        {row.size()},
                        {sizeof(real)},
                        false);
                },
                py::keep_alive<0, 1>())
            .def(
                "__setitem__",
                [](TArrayEigenVector &self, index index_, py::buffer row)
                {
                    auto row_info = row.request(false);
                    DNDS_assert(row_info.item_type_is_equivalent_to<real>());
                    auto [count, row_style] = py_buffer_get_contigious_size(row_info); // todo: upgrade to accept any 1D array
                    DNDS_assert(self.RowSize(index_) == count);
                    auto row_start_ptr = reinterpret_cast<real *>(row_info.ptr);
                    std::copy(row_start_ptr, row_start_ptr + count, self[index_].data());
                });
    }

    template <rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    void _pybind11_ArrayEigenVector_define_dispatch(py::module_ &m)
    {
        if constexpr (_vec_size == UnInitRowsize)
            return;
        else
            return pybind11_ArrayEigenVector_define<_vec_size, _row_max, _align>(m);
    }
}

namespace DNDS
{
    template <size_t N, std::array<int, N> const &Arr, size_t... Is>
    void __pybind11_callBindArrayEigenVectors_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...>)
    {
        (_pybind11_ArrayEigenVector_define_dispatch<Arr[Is]>(m), ...);
    }

    inline void pybind11_callBindArrayEigenVectors_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = pybind11_arrayRowsizeInstantiationList;
        __pybind11_callBindArrayEigenVectors_rowsizes_sequence<
            seq.size(), seq>(m, std::make_index_sequence<seq.size()>{});
    }

    void pybind11_bind_ArrayEigenVector_All(py::module_ &m);
}