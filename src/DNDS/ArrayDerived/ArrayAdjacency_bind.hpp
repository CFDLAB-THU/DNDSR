#pragma once

#include "ArrayAdjacency.hpp"
#include "../Array_bind.hpp"

namespace DNDS
{
    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    std::string pybind11_ArrayAdjacency_name()
    {
        return "ArrayAdjacency" + pybind11_Array_name_appends<index, _row_size, _row_max, _align>();
    }

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using tPy_ArrayAdjacency = py::class_<ArrayAdjacency<_row_size, _row_max, _align>, ssp<ArrayAdjacency<_row_size, _row_max, _align>>>;
}

namespace DNDS
{
    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_ArrayAdjacency<_row_size, _row_max, _align>
    pybind11_ArrayAdjacency_declare(py::module_ &m)
    {
        auto ParArray_ = pybind11_ParArray_get_class<index, _row_size, _row_max, _align>(m);
        return {m, pybind11_ArrayAdjacency_name<_row_size, _row_max, _align>().c_str(), ParArray_};
        // std::cout << py::format_descriptor<Eigen::Matrix<double, 3, 3>>().format() << std::endl;
    }

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_ArrayAdjacency<_row_size, _row_max, _align>
    pybind11_ArrayAdjacency_get_class(py::module_ &m)
    {
        return {m.attr(pybind11_ArrayAdjacency_name<_row_size, _row_max, _align>().c_str())};
    }

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void pybind11_ArrayAdjacency_define(py::module_ &m)
    {

        using TArrayAdjacency = ArrayAdjacency<_row_size, _row_max, _align>;
        auto ArrayAdjacency_ = pybind11_ArrayAdjacency_declare<_row_size, _row_max, _align>(m);

        // // helper
        // using TArrayAdjacency = ArrayAdjacency<1>;
        // auto ArrayAdjacency_ = pybind11_ArrayAdjacency_declare<1>(m);
        // // helper

        ArrayAdjacency_
            // we only bind the non-default ctor here
            .def(py::init<const MPIInfo &>(), py::arg("nmpi"))
            .def(
                "__getitem__",
                [](TArrayAdjacency &self, std::tuple<index> index_)
                {
                    AdjacencyRow row = self[std::get<0>(index_)];
                    return py::memoryview::from_buffer<index>(
                        row.begin(),
                        {row.size()},
                        {sizeof(index)},
                        false);
                },
                py::keep_alive<0, 1>())
            .def(
                "__setitem__",
                [](TArrayAdjacency &self, std::tuple<index> index_, py::buffer row)
                {
                    auto row_info = row.request(false);
                    DNDS_assert(row_info.item_type_is_equivalent_to<index>());
                    auto [count, row_style] = py_buffer_get_contigious_size(row_info);
                    DNDS_assert(self.RowSize(std::get<0>(index_)) == count);
                    auto row_start_ptr = reinterpret_cast<index *>(row_info.ptr);
                    std::copy(row_start_ptr, row_start_ptr + count, self.rowPtr(std::get<0>(index_)));
                });
    }

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void _pybind11_ArrayAdjacency_define_dispatch(py::module_ &m)
    {
        if constexpr (_row_size == UnInitRowsize)
            return;
        else
            return pybind11_ArrayAdjacency_define<_row_size, _row_max, _align>(m);
    }
}

namespace DNDS
{
    template <size_t N, std::array<int, N> const &Arr, size_t... Is>
    void __pybind11_callBindArrayAdjacencys_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...>)
    {
        (_pybind11_ArrayAdjacency_define_dispatch<Arr[Is]>(m), ...);
    }

    inline void pybind11_callBindArrayAdjacencys_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = pybind11_arrayRowsizeInstantiationList;
        __pybind11_callBindArrayAdjacencys_rowsizes_sequence<
            seq.size(), seq>(m, std::make_index_sequence<seq.size()>{});
    }

    void pybind11_bind_ArrayAdjacency_All(py::module_ &m);
}