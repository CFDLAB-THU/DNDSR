#pragma once

#include "ArrayAdjacency.hpp"
#include "../Array_bind.hpp"

namespace DNDS
{
    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    std::string pybind11_ArrayAdjacency_name_appends()
    {
        return fmt::format("_{}_{}_{}",
                           RowSize_To_PySnippet(_row_size),
                           RowSize_To_PySnippet(_row_max),
                           Align_To_PySnippet(_align));
    }

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    std::string pybind11_ArrayAdjacency_name()
    {
        return "ArrayAdjacency" + pybind11_ArrayAdjacency_name_appends<_row_size, _row_max, _align>();
    }

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    std::string pybind11_ArrayAdjacencyPair_name()
    {
        return "ArrayAdjacencyPair" + pybind11_ArrayAdjacency_name_appends<_row_size, _row_max, _align>();
    }

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign> // ! shared ptr
    using tPy_ArrayAdjacency = py_class_ssp<ArrayAdjacency<_row_size, _row_max, _align>>;

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using tPy_ArrayAdjacencyPair = py_class_ssp<ArrayAdjacencyPair<_row_size, _row_max, _align>>; // ! shared ptr
}

namespace DNDS
{
    template <class TArray = ArrayAdjacency<1>>
    auto pybind11_ArrayAdjacency_setitem(TArray &self, index index_, py::buffer row)
    {
        auto row_info = row.request(false);
        DNDS_assert(row_info.item_type_is_equivalent_to<index>());
        auto [count, row_style] = py_buffer_get_contigious_size(row_info);
        DNDS_assert(self.RowSize(index_) == count);
        auto row_start_ptr = reinterpret_cast<index *>(row_info.ptr);
        std::copy(row_start_ptr, row_start_ptr + count, self.rowPtr(index_));
    }

    template <class TArray = ArrayAdjacency<1>>
    auto pybind11_ArrayAdjacency_getitem(TArray &self, index index_)
    {
        AdjacencyRow row = self[index_];
        return py::memoryview::from_buffer<index>(
            row.begin(),
            {row.size()},
            {sizeof(index)},
            false);
    }
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
                [](TArrayAdjacency &self, index index_)
                {
                    return pybind11_ArrayAdjacency_getitem(self, index_);
                },
                py::keep_alive<0, 1>())
            .def(
                "__setitem__",
                [](TArrayAdjacency &self, index index_, py::buffer row)
                {
                    return pybind11_ArrayAdjacency_setitem(self, index_, row);
                });
        ArrayAdjacency_
            .def("clone", [](TArrayAdjacency &self)
                 {
                auto arr = std::make_shared<TArrayAdjacency>(self);
                return arr; });

        ArrayAdjacency_
            .def("to_device", [](TArrayAdjacency &self, const std::string &backend)
                 { self.to_device(device_backend_name_to_enum(backend)); }, py::arg("backend"))
            .def("to_host", &TArrayAdjacency::to_host);
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
    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_ArrayAdjacencyPair<_row_size, _row_max, _align>
    pybind11_ArrayAdjacencyPair_declare(py::module_ &m)
    {
        return {m, pybind11_ArrayAdjacencyPair_name<_row_size, _row_max, _align>().c_str()};
        // std::cout << py::format_descriptor<Eigen::Matrix<double, 3, 3>>().format() << std::endl;
    }

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_ArrayAdjacencyPair<_row_size, _row_max, _align>
    pybind11_ArrayAdjacencyPair_get_class(py::module_ &m)
    {
        return {m.attr(pybind11_ArrayAdjacencyPair_name<_row_size, _row_max, _align>().c_str())};
    }

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void pybind11_ArrayAdjacencyPair_define(py::module_ &m)
    {

        // using TArray = ParArray<T, _row_size, _row_max, _align>;
        using TPair = ArrayPair<ArrayAdjacency<_row_size, _row_max, _align>>;
        auto Pair_ = pybind11_ArrayAdjacencyPair_declare<_row_size, _row_max, _align>(m);

        // // helper
        // using TArray = ArrayAdjacency<_row_size, _row_max, _align>;
        // using TPair = ArrayPair<ArrayAdjacency<1>>;
        // auto Pair_ = pybind11_ArrayAdjacencyPair_declare<1>(m);
        // // helper

        pybind11_ArrayPairGenericBindBasics<TPair>(Pair_);

        Pair_
            .def("RowSize", [](const TPair &self, index i)
                 { return self.RowSize(i); }, py::arg("i"))
            .def("RowSize", [](const TPair &self)
                 { return self.RowSize(); });
        Pair_
            .def(
                "__getitem__",
                [](TPair &self, index index_)
                {
                    return self.runFunctionAppendedIndex(index_, [&](auto &ar, index iC) //* note the reference here!!!
                                                         { return pybind11_ArrayAdjacency_getitem(ar, iC); });
                },
                py::keep_alive<0, 1>())
            .def(
                "__setitem__",
                [](TPair &self, index index_, py::buffer row)
                {
                    return self.runFunctionAppendedIndex(index_, [&](auto &ar, index iC) //*note the auto&& reference here!!!
                                                         { return pybind11_ArrayAdjacency_setitem(ar, iC, row); });
                });
    }

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void _pybind11_ArrayAdjacencyPair_define_dispatch(py::module_ &m)
    {
        if constexpr (_row_size == UnInitRowsize)
            return;
        else
            return pybind11_ArrayAdjacencyPair_define<_row_size, _row_max, _align>(m);
    }
}

namespace DNDS
{
    template <size_t N, std::array<int, N> const &Arr, size_t... Is>
    void __pybind11_callBindArrayAdjacencys_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...>)
    {
        (_pybind11_ArrayAdjacency_define_dispatch<Arr[Is]>(m), ...);
        (_pybind11_ArrayAdjacencyPair_define_dispatch<Arr[Is]>(m), ...);
    }

    inline void pybind11_callBindArrayAdjacencys_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = pybind11_arrayRowsizeInstantiationList;
        __pybind11_callBindArrayAdjacencys_rowsizes_sequence<
            seq.size(), seq>(m, std::make_index_sequence<seq.size()>{});
    }

    void pybind11_bind_ArrayAdjacency_All(py::module_ &m);
}