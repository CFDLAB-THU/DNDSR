#pragma once

#include "Array.hpp"
#include "ArrayTransformer.hpp"
#include "ArrayPair.hpp"
#include "Defines_bind.hpp"


#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace DNDS
{
    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    std::string pybind11_Array_name_appends()
    {
        static_assert(std::is_arithmetic_v<T>);
        return fmt::format("_{}_{}_{}_{}",
                           py::format_descriptor<T>().format(),
                           RowSize_To_PySnippet(_row_size),
                           RowSize_To_PySnippet(_row_max),
                           RowSize_To_PySnippet(_align));
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    std::string pybind11_Array_name()
    {
        static_assert(std::is_arithmetic_v<T>);
        return "Array" + pybind11_Array_name_appends<T, _row_size, _row_max, _align>();
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    std::string pybind11_ParArray_name()
    {
        return "ParArray" + pybind11_Array_name_appends<T, _row_size, _row_max, _align>();
    }

    template <class TArray>
    std::string pybind11_ArrayTransformer_name()
    {
        return "ArrayTransformer" + pybind11_Array_name_appends<typename TArray::value_type, TArray::rs, TArray::rm, TArray::al>();
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using tPy_Array = py::class_<Array<T, _row_size, _row_max, _align>, ssp<Array<T, _row_size, _row_max, _align>>>;

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using tPy_ParArray = py::class_<ParArray<T, _row_size, _row_max, _align>, ssp<ParArray<T, _row_size, _row_max, _align>>>;

    template <class TArray>
    using tPy_ArrayTransformer = py::class_<ArrayTransformerType_t<TArray>>; // no shared pointer managing
}

namespace DNDS
{

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_Array<T, _row_size, _row_max, _align>
    pybind11_Array_declare(py::module_ &m)
    {
        static_assert(std::is_arithmetic_v<T>);
        return {
            m,
            pybind11_Array_name<T, _row_size, _row_max, _align>().c_str()};
        // std::cout << py::format_descriptor<Eigen::Matrix<double, 3, 3>>().format() << std::endl;
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_Array<T, _row_size, _row_max, _align>
    pybind11_Array_get_class(py::module_ &m)
    {
        static_assert(std::is_arithmetic_v<T>);
        return {m.attr(pybind11_Array_name<T, _row_size, _row_max, _align>().c_str())};
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void pybind11_Array_define(py::module_ &m)
    {

        using TArray = Array<T, _row_size, _row_max, _align>;
        auto Array_ = pybind11_Array_declare<T, _row_size, _row_max, _align>(m);
        // // helper
        // using TArray = Array<real, 1, 1, -1>;
        // auto Array_ = pybind11_Array_declare<real, 1, 1, -1>(m);
        // // helper

        Array_
            .def(py::init<>())
            .def("Size", &TArray::Size)
            .def("Compress", &TArray::Compress)
            .def("Decompress", &TArray::Decompress)
            .def("IfCompressed", &TArray::IfCompressed);
        Array_
            .def(
                "getRowStart",
                [](TArray &self)
                {
                    if (!self.getRowStart())
                        return py::memoryview::from_buffer<index>((index *)(&self), {0}, {sizeof(index)}, true);
                    auto &rs = *self.getRowStart();
                    return py::memoryview::from_buffer<index>(rs.data(), {rs.size()}, {sizeof(index)}, true);
                },
                py::keep_alive<0, 1>() /* remember to keep alive */);

        Array_
            .def(
                "getRowSizes",
                [](TArray &self)
                {
                    if (!self.getRowSizes())
                        return py::memoryview::from_buffer<rowsize>((rowsize *)(&self), {0}, {sizeof(rowsize)}, true);
                    auto &rs = *self.getRowSizes();
                    return py::memoryview::from_buffer<rowsize>(rs.data(), {rs.size()}, {sizeof(rowsize)}, true);
                },
                py::keep_alive<0, 1>() /* remember to keep alive */);

        Array_
            .def(
                "data", [](TArray &self)
                { return py::memoryview::from_buffer<T>(self.data(), {self.DataSize()}, {TArray::sizeof_T}); },
                py::keep_alive<0, 1>() /* remember to keep alive */);

        Array_
            .def("Rowsize", py::overload_cast<index>(&TArray::RowSize, py::const_), py::arg("iRow"));
        Array_
            .def("Rowsize", py::overload_cast<>(&TArray::RowSize, py::const_));
        Array_
            .def("Resize", [](TArray &self, index nRow)
                 { self.Resize(nRow); }, py::arg("nRow"));
        Array_
            .def("Resize", [](TArray &self, index nRow, rowsize nRowsizeDynamic)
                 { self.Resize(nRow, nRowsizeDynamic); }, py::arg("nRow"), py::arg("rowsizeDynamic"));
        if constexpr (TArray::isCSR)
            Array_
                .def(
                    "Resize",
                    [](TArray &self, index nRow, py::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> rowsizes)
                    {
                        DNDS_assert_info(rowsizes.size() >= nRow, fmt::format("rowsizes is of size {}, not enough", rowsizes.size()));
                        self.Resize(nRow, [&](index iRow)
                                    { return rowsizes.at(iRow); });
                    },
                    py::arg("nRow"), py::arg("rowsizesArray"));
        Array_
            .def("ResizeRow", &TArray::ResizeRow, py::arg("iRow"), py::arg("nRowsize"));
        Array_
            .def("__getitem__",
                 [](const TArray &self, std::tuple<index, rowsize> index_)
                 {
                     return self(std::get<0>(index_), std::get<1>(index_));
                 });
        Array_
            .def("__setitem__",
                 [](TArray &self, std::tuple<index, rowsize> index_, const T &value)
                 {
                     self(std::get<0>(index_), std::get<1>(index_)) = value;
                 });
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void _pybind11_Array_define_dispatch(py::module_ &m)
    {
        if constexpr (_row_size == UnInitRowsize)
            return;
        else
            return pybind11_Array_define<T, _row_size, _row_max, _align>(m);
    }
}

namespace DNDS
{

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_ParArray<T, _row_size, _row_max, _align>
    pybind11_ParArray_declare(py::module_ &m)
    {
        static_assert(std::is_arithmetic_v<T>);
        // std::cout << "here1 " << std::endl;
        auto array_ = pybind11_Array_get_class<T, _row_size, _row_max, _align>(m); // same module here
        // std::cout << "here2 " << std::endl;
        return {
            m,
            pybind11_ParArray_name<T, _row_size, _row_max, _align>().c_str(),
            array_};
        // std::cout << py::format_descriptor<Eigen::Matrix<double, 3, 3>>().format() << std::endl;
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_ParArray<T, _row_size, _row_max, _align>
    pybind11_ParArray_get_class(py::module_ &m)
    {
        static_assert(std::is_arithmetic_v<T>);
        return {m.attr(pybind11_ParArray_name<T, _row_size, _row_max, _align>().c_str())};
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void pybind11_ParArray_define(py::module_ &m)
    {
        using TParArray = ParArray<T, _row_size, _row_max, _align>;
        auto ParArray_ = pybind11_ParArray_declare<T, _row_size, _row_max, _align>(m);

        // // helper
        // using TParArray = ParArray<real, 1, 1, -1>;
        // auto ParArray_ = pybind11_ParArray_declare<real, 1, 1, -1>(m);
        // // helper

        ParArray_ // need lambda below to avoid inheritance checking
            .def(py::init([](const MPIInfo &n_mpi)
                          { return std::make_shared<TParArray>(n_mpi); }),
                 py::arg("n_mpi"))
            .def("getMPI", [](const TParArray &self)
                 { return self.getMPI(); })
            .def("setMPI", [](TParArray &self, const MPIInfo &n_mpi)
                 { self.setMPI(n_mpi); }, py::arg("n_mpi"))
            .def("createGlobalMapping", [](TParArray &self)
                 { self.createGlobalMapping(); })
            .def("getLGlobalMapping", [](TParArray &self)
                 { return self.pLGlobalMapping; });
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void _pybind11_ParArray_define_dispatch(py::module_ &m)
    {
        if constexpr (_row_size == UnInitRowsize)
            return;
        else
            return pybind11_ParArray_define<T, _row_size, _row_max, _align>(m);
    }
}

namespace DNDS
{
    template <class TArray>
    tPy_ArrayTransformer<TArray>
    pybind11_ArrayTransformer_declare(py::module_ &m)
    {
        return {m, pybind11_ArrayTransformer_name<TArray>().c_str()};
        // std::cout << py::format_descriptor<Eigen::Matrix<double, 3, 3>>().format() << std::endl;
    }

    template <class TArray>
    tPy_ArrayTransformer<TArray>
    pybind11_ArrayTransformer_get_class(py::module_ &m)
    {
        return {m.attr(pybind11_ArrayTransformer_name<TArray>().c_str())};
    }

    template <class TArray>
    void pybind11_ArrayTransformer_define(py::module_ &m)
    {
        using TArrayTransformer = ArrayTransformerType_t<TArray>;
        auto ArrayTransformer_ = pybind11_ArrayTransformer_declare<TArray>(m);

        // // helper
        // using TArrayTransformer = ArrayTransformer<real, 1>;
        // auto ArrayTransformer_ = pybind11_ArrayTransformer_declare<real, 1>(m);
        // // helper

#define DNDS_pybind11_array_transformer_def_ssp_property(property_name, field_name)                             \
    {                                                                                                           \
        ArrayTransformer_.def_property(                                                                         \
            #property_name,                                                                                     \
            [](TArrayTransformer &self) { return self.field_name; },                                            \
            [](TArrayTransformer &self, decltype(TArrayTransformer::field_name) in) { self.field_name = in; }); \
    }
        DNDS_pybind11_array_transformer_def_ssp_property(LGlobalMapping, pLGlobalMapping);
        DNDS_pybind11_array_transformer_def_ssp_property(LGhostMapping, pLGhostMapping);
        DNDS_pybind11_array_transformer_def_ssp_property(father, father);
        DNDS_pybind11_array_transformer_def_ssp_property(son, son);
        DNDS_pybind11_array_transformer_def_ssp_property(mpi, mpi);

        ArrayTransformer_
            .def(py::init<>())
            .def("setFatherSon", &TArrayTransformer::setFatherSon, py::arg("father"), py::arg("son"))
            .def("createFatherGlobalMapping", &TArrayTransformer::createFatherGlobalMapping)
            .def("createGhostMapping", [](TArrayTransformer &self, std::vector<index> pullIndexGlobal) -> void
                 { self.createGhostMapping(pullIndexGlobal); }, py::arg("pullIndexGlobal"))
            .def("createGhostMapping", [](TArrayTransformer &self, std::vector<index> pushingIndexLocal, std::vector<index> pushingStarts) -> void
                 { self.createGhostMapping(pushingIndexLocal, pushingStarts); }, py::arg("pushingIndexLocal"), py::arg("pushingStarts"));
        ArrayTransformer_
            .def("createMPITypes", &TArrayTransformer::createMPITypes)
            .def("clearMPITypes", &TArrayTransformer::clearMPITypes)
            .def(
                "BorrowGGIndexing",
                [](TArrayTransformer &self, py::object other)
                {
                    auto other_father = other.attr("father");
                    auto other_father_size = other_father.attr("Size")().cast<index>();
                    auto other_pLGhostMapping = other.attr("LGhostMapping").cast<ssp<OffsetAscendIndexMapping>>();
                    auto other_pLGlobalMapping = other.attr("LGlobalMapping").cast<ssp<GlobalOffsetsMapping>>();

                    DNDS_assert(self.father);
                    DNDS_assert(other_father_size == self.father->Size());
                    DNDS_assert(other_pLGhostMapping && other_pLGlobalMapping);

                    self.pLGhostMapping = other_pLGhostMapping;
                    self.pLGlobalMapping = other_pLGlobalMapping;
                    self.father->pLGlobalMapping = self.pLGlobalMapping;
                },
                py::arg("other"));

        ArrayTransformer_
            .def("initPersistentPull", &TArrayTransformer::initPersistentPull)
            .def("initPersistentPush", &TArrayTransformer::initPersistentPush)
            .def("startPersistentPull", &TArrayTransformer::startPersistentPull)
            .def("startPersistentPush", &TArrayTransformer::startPersistentPush)
            .def("waitPersistentPull", &TArrayTransformer::waitPersistentPull)
            .def("waitPersistentPush", &TArrayTransformer::waitPersistentPush)
            .def("clearPersistentPull", &TArrayTransformer::clearPersistentPull)
            .def("clearPersistentPush", &TArrayTransformer::clearPersistentPush)
            .def("pullOnce", &TArrayTransformer::pullOnce)
            .def("pushOnce", &TArrayTransformer::pushOnce);
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void _pybind11_ArrayTransformer_define_dispatch(py::module_ &m)
    {
        if constexpr (_row_size == UnInitRowsize)
            return;
        else
            return pybind11_ArrayTransformer_define<ParArray<T, _row_size, _row_max, _align>>(m);
    }
}

namespace DNDS
{
    constexpr auto _get_pybind11_arrayRowsizeInstantiationList()
    {
        std::array<rowsize, 20> ret{UnInitRowsize};
        for (auto &v : ret)
            v = UnInitRowsize;
        for (int i = 1; i <= 8; i++)
            ret[i] = i;
        return ret;
    }
    static constexpr auto pybind11_arrayRowsizeInstantiationList = _get_pybind11_arrayRowsizeInstantiationList();

    template <class T, size_t N, std::array<int, N> const &Arr, size_t... Is>
    void __pybind11_callBindArrays_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...>)
    {
        (_pybind11_Array_define_dispatch<T, Arr[Is]>(m), ...);
    }

    template <class T>
    void pybind11_callBindArrays_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = pybind11_arrayRowsizeInstantiationList;
        __pybind11_callBindArrays_rowsizes_sequence<
            T, seq.size(), seq>(m, std::make_index_sequence<seq.size()>{});
    }

    template <class T, size_t N, std::array<int, N> const &Arr, size_t... Is>
    void __pybind11_callBindParArrays_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...>)
    {
        (_pybind11_ParArray_define_dispatch<T, Arr[Is]>(m), ...);
    }

    template <class T>
    void pybind11_callBindParArrays_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = pybind11_arrayRowsizeInstantiationList;
        __pybind11_callBindParArrays_rowsizes_sequence<
            T, seq.size(), seq>(m, std::make_index_sequence<seq.size()>{});
    }

    template <class T, size_t N, std::array<int, N> const &Arr, size_t... Is>
    void __pybind11_callBindArrayTransformers_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...>)
    {
        (_pybind11_ArrayTransformer_define_dispatch<T, Arr[Is]>(m), ...);
    }

    template <class T>
    void pybind11_callBindArrayTransformers_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = pybind11_arrayRowsizeInstantiationList;
        __pybind11_callBindArrayTransformers_rowsizes_sequence<
            T, seq.size(), seq>(m, std::make_index_sequence<seq.size()>{});
    }
}

namespace DNDS
{
    void pybind11_bind_Array_All(py::module_ m);
}