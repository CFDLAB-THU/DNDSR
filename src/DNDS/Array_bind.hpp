#pragma once
/// @file Array_bind.hpp
/// @brief pybind11 bindings for @ref DNDS::Array "Array" / @ref DNDS::ParArray "ParArray" / @ref DNDS::ArrayPair "ArrayPair".
///
/// Provides templated helper functions (`pybind11_Array_name_appends`,
/// `pybind11_Array_declare_*`, ...) that mechanically generate Python classes
/// for every instantiated `(T, row_size, row_max)` combination. See the
/// `ArrayPair.hpp` and `ArrayTransformer.hpp` corresponding C++ types.

#include "Array.hpp"
#include "ArrayTransformer.hpp"
#include "ArrayPair.hpp"
#include "Serializer/SerializerBase.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/Device/DeviceStorage.hpp"
#include "Defines_bind.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace DNDS
{
    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    std::string pybind11_Array_name_appends()
    {
        std::string TName;
        if constexpr (std::is_arithmetic_v<T>)
            TName = py::format_descriptor<T>().format();
        else
        {
            TName = T::pybind11_name();
        }
        return fmt::format("_{}_{}_{}_{}",
                           TName,
                           RowSize_To_PySnippet(_row_size),
                           RowSize_To_PySnippet(_row_max),
                           Align_To_PySnippet(_align));
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    std::string pybind11_Array_name()
    {
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
    std::string pybind11_ParArrayPair_name()
    {
        return "ParArrayPair" + pybind11_Array_name_appends<T, _row_size, _row_max, _align>();
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign> // ! shared pointer managing
    using tPy_Array = py_class_ssp<Array<T, _row_size, _row_max, _align>>;

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign> // ! shared pointer managing
    using tPy_ParArray = py_class_ssp<ParArray<T, _row_size, _row_max, _align>>;

    template <class TArray>
    // using tPy_ArrayTransformer = py::class_<ArrayTransformerType_t<TArray>>; // ! unique ptr
    using tPy_ArrayTransformer = py_class_ssp<ArrayTransformerType_t<TArray>>; // ! shared ptr

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    // using tPy_ParArrayPair = py::class_<ArrayPair<ParArray<T, _row_size, _row_max, _align>>>; // ! unique ptr
    using tPy_ParArrayPair = py_class_ssp<ArrayPair<ParArray<T, _row_size, _row_max, _align>>>; // ! shared ptr
}

namespace DNDS // Array
{

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_Array<T, _row_size, _row_max, _align>
    pybind11_Array_declare(py::module_ &m)
    {
        return {
            m,
            pybind11_Array_name<T, _row_size, _row_max, _align>().c_str()};
        // std::cout << py::format_descriptor<Eigen::Matrix<double, 3, 3>>().format() << std::endl;
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_Array<T, _row_size, _row_max, _align>
    pybind11_Array_get_class(py::module_ &m)
    {
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
            .def("Size", &TArray::Size);
        Array_
            .def("clone", [](TArray &self)
                 {
                auto arr = std::make_shared<TArray>();
                arr->clone(self);
                return arr; });
        if constexpr (TArray::GetDataLayoutStatic() == CSR)
            Array_
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
                "data",
                [](TArray &self)
                {
                    if constexpr (std::is_arithmetic_v<T>)
                        return py::memoryview::from_buffer<T>(
                            self.DataSize() ? self.data() : (T *)(&self), // for null buffer
                            {self.DataSize()}, {TArray::sizeof_T});
                    else // todo: determine if have pybind11_buffer_format()
                    {
                        std::string buf_format;
                        buf_format.reserve(32);
                        for (size_t i = 0; i < TArray::sizeof_T; i++)
                            buf_format += "c"; // now we use a untyped byte data
                        return py::memoryview::from_buffer(
                            self.DataSize() ? self.data() : (T *)(&self), // for null buffer
                            TArray::sizeof_T,
                            buf_format.c_str(),
                            {self.DataSize()}, {TArray::sizeof_T});
                    }
                },
                py::keep_alive<0, 1>() /* remember to keep alive */);

        Array_
            .def("Rowsize", py::overload_cast<index>(&TArray::RowSize, py::const_), py::arg("iRow"));
        Array_
            .def("Rowsize", py::overload_cast<>(&TArray::RowSize, py::const_));
        if constexpr (
            TArray::GetDataLayoutStatic() == CSR ||
            TArray::GetDataLayoutStatic() == TABLE_StaticFixed ||
            TArray::GetDataLayoutStatic() == TABLE_StaticMax)
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
        if constexpr (TArray::GetDataLayoutStatic() == CSR ||
                      TArray::GetDataLayoutStatic() == TABLE_Max ||
                      TArray::GetDataLayoutStatic() == TABLE_StaticMax)
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

        Array_
            .def("to_device", [](TArray &self, const std::string &backend)
                 { self.to_device(device_backend_name_to_enum(backend)); }, py::arg("backend"))
            .def("to_host", &TArray::to_host)
            .def("device", &TArray::device);
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

namespace DNDS // ParArray
{

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_ParArray<T, _row_size, _row_max, _align>
    pybind11_ParArray_declare(py::module_ &m)
    {
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
            .def(py::init([](const std::string &name, const MPIInfo &n_mpi)
                          { return make_ssp<TParArray>(ObjName{name}, n_mpi); }),
                 py::arg("name"), py::arg("n_mpi"))
            .def("setObjectName", &TParArray::setObjectName, py::arg("name"))
            .def("getObjectName", &TParArray::getObjectName)
            .def("getMPI", [](const TParArray &self)
                 { return self.getMPI(); })
            .def("setMPI", [](TParArray &self, const MPIInfo &n_mpi)
                 { self.setMPI(n_mpi); }, py::arg("n_mpi"))
            .def("createGlobalMapping", [](TParArray &self)
                 { self.createGlobalMapping(); })
            .def("getLGlobalMapping", [](TParArray &self)
                 { return self.pLGlobalMapping; })
            .def("getTrans", [m](TParArray &self)
                 { return py::type{m.attr(pybind11_ArrayTransformer_name<TParArray>().c_str())}; });

        ParArray_
            .def("clone", [](TParArray &self)
                 {
                auto arr = std::make_shared<TParArray>(self.mpi);
                arr->clone(self);
                return arr; });
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

namespace DNDS // ParArrayPair
{

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_ParArrayPair<T, _row_size, _row_max, _align>
    pybind11_ParArrayPair_declare(py::module_ &m)
    {
        return {
            m,
            pybind11_ParArrayPair_name<T, _row_size, _row_max, _align>().c_str()};
        // std::cout << py::format_descriptor<Eigen::Matrix<double, 3, 3>>().format() << std::endl;
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_ParArrayPair<T, _row_size, _row_max, _align>
    pybind11_ParArrayPair_get_class(py::module_ &m)
    {
        return {m.attr(pybind11_ParArrayPair_name<T, _row_size, _row_max, _align>().c_str())};
    }

    template <class TPair, class TPy_Pair>
    void pybind11_ArrayPairGenericBindBasics(TPy_Pair &Pair_)
    {
        Pair_
            .def(py::init([]()
                          { return std::make_shared<TPair>(); }));
        Pair_
            .def_readwrite("father", &TPair::father)
            .def_readwrite("son", &TPair::son);
        Pair_
            .def_readonly("trans", &TPair::trans, py::return_value_policy::reference_internal);
        Pair_
            .def("clone",
                 [&](TPair &self)
                 {
                     auto new_pair = std::make_shared<TPair>();
                     new_pair->clone(self);
                     return new_pair;
                 });
        Pair_
            .def("InitPair", [](TPair &self, const std::string &name, const MPIInfo &mpi)
                 { self.InitPair(name, mpi); }, py::arg("name"), py::arg("mpi"))
            .def("TransAttach", &TPair::TransAttach)
            .def("hash", &TPair::hash)
            .def("Size", &TPair::Size);
        if constexpr (TPair::t_arr::GetDataLayoutStatic() == CSR)
            Pair_
                .def("CompressBoth", &TPair::CompressBoth);

        Pair_
            .def("to_device", [](TPair &self, const std::string &backend)
                 { self.to_device(device_backend_name_to_enum(backend)); }, py::arg("backend"))
            .def("to_host", &TPair::to_host);

        // Serialization methods
        Pair_
            .def(
                "WriteSerialize",
                [](TPair &self, Serializer::SerializerBaseSSP serializerP,
                   const std::string &name, bool includePIG, bool includeSon)
                {
                    self.WriteSerialize(serializerP, name, includePIG, includeSon);
                },
                py::arg("serializer"), py::arg("name"),
                py::arg("includePIG") = true, py::arg("includeSon") = true)
            .def(
                "WriteSerialize",
                [](TPair &self, Serializer::SerializerBaseSSP serializerP,
                   const std::string &name, std::vector<index> origIndex,
                   bool includePIG, bool includeSon)
                {
                    self.WriteSerialize(serializerP, name, origIndex, includePIG, includeSon);
                },
                py::arg("serializer"), py::arg("name"), py::arg("origIndex"),
                py::arg("includePIG") = true, py::arg("includeSon") = true)
            .def(
                "ReadSerialize",
                [](TPair &self, Serializer::SerializerBaseSSP serializerP,
                   const std::string &name, bool includePIG, bool includeSon)
                {
                    self.ReadSerialize(serializerP, name, includePIG, includeSon);
                },
                py::arg("serializer"), py::arg("name"),
                py::arg("includePIG") = true, py::arg("includeSon") = true)
            .def(
                "ReadSerializeRedistributed",
                [](TPair &self, Serializer::SerializerBaseSSP serializerP,
                   const std::string &name, std::vector<index> newOrigIndex)
                {
                    self.ReadSerializeRedistributed(serializerP, name, newOrigIndex);
                },
                py::arg("serializer"), py::arg("name"), py::arg("newOrigIndex"));
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void pybind11_ParArrayPair_define(py::module_ &m)
    {
        // using TArray = ParArray<T, _row_size, _row_max, _align>;
        using TPair = ArrayPair<ParArray<T, _row_size, _row_max, _align>>;
        auto Pair_ = pybind11_ParArrayPair_declare<T, _row_size, _row_max, _align>(m);

        // // helper
        // using TArray = ParArray<real, 1, 1, -1>;
        // using TPair = ArrayPair<ParArray<real, 1, 1, -1>>;
        // auto Pair_ = pybind11_ParArrayPair_declare<real, 1, 1, -1>(m);
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
                [](const TPair &self, std::tuple<index, rowsize> index_)
                {
                    return self(std::get<0>(index_), std::get<1>(index_));
                })
            .def(
                "__setitem__",
                [](TPair &self, std::tuple<index, rowsize> index_, const T &value)
                {
                    self(std::get<0>(index_), std::get<1>(index_)) = value;
                });
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void _pybind11_ParArrayPair_define_dispatch(py::module_ &m)
    {
        if constexpr (_row_size == UnInitRowsize)
            return;
        else
            return pybind11_ParArrayPair_define<T, _row_size, _row_max, _align>(m);
    }
}

namespace DNDS // ArrayTransformer
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
            .def("createGhostMapping", [](TArrayTransformer &self, py::array_t<index> pullIndexGlobal)
                 {
                    std::vector<index> pullIndexVec;
                    pullIndexVec.reserve(pullIndexGlobal.size());
                    for(ssize_t i = 0; i < pullIndexGlobal.size(); i++)
                        pullIndexVec.push_back(pullIndexGlobal.at(i)); // only 1D
                    self.createGhostMapping(pullIndexVec); }, py::arg("pullIndexGlobal"))
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
            .def("initPersistentPull", &TArrayTransformer::initPersistentPull, py::arg("backend") = DeviceBackend::Unknown)
            .def("initPersistentPush", &TArrayTransformer::initPersistentPush, py::arg("backend") = DeviceBackend::Unknown)
            .def("startPersistentPull", &TArrayTransformer::startPersistentPull, py::arg("backend") = DeviceBackend::Unknown)
            .def("startPersistentPush", &TArrayTransformer::startPersistentPush, py::arg("backend") = DeviceBackend::Unknown)
            .def("waitPersistentPull", &TArrayTransformer::waitPersistentPull, py::arg("backend") = DeviceBackend::Unknown)
            .def("waitPersistentPush", &TArrayTransformer::waitPersistentPush, py::arg("backend") = DeviceBackend::Unknown)
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
    template <int offset = 0>
    constexpr auto _get_pybind11_arrayRowsizeInstantiationList()
    {
        std::array<rowsize, 9> ret{UnInitRowsize};

        for (auto &v : ret)
            v = UnInitRowsize;
        int rs = 8;
        for (int i = 0; i < rs; i++)
            ret[i] = i + 1 + rs * offset;
        return ret;
    }
    static constexpr auto pybind11_arrayRowsizeInstantiationList = _get_pybind11_arrayRowsizeInstantiationList();

    template <class T, size_t N, std::array<int, N> const &Arr, size_t... Is>
    void pybind11_callBindArrays_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...> /*unused*/)
    {
        (_pybind11_Array_define_dispatch<T, Arr[Is]>(m), ...);
    }

    template <class T, int offset = 0>
    void pybind11_callBindArrays_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = _get_pybind11_arrayRowsizeInstantiationList<offset>();
        pybind11_callBindArrays_rowsizes_sequence<
            T, seq.size(), seq>(m, std::make_index_sequence<seq.size()>{});
    }

    template <class T, size_t N, std::array<int, N> const &Arr, size_t... Is>
    void pybind11_callBindParArrays_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...> /*unused*/)
    {
        (_pybind11_ParArray_define_dispatch<T, Arr[Is]>(m), ...);
    }

    template <class T, int offset = 0>
    void pybind11_callBindParArrays_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = _get_pybind11_arrayRowsizeInstantiationList<offset>();
        pybind11_callBindParArrays_rowsizes_sequence<
            T, seq.size(), seq>(m, std::make_index_sequence<seq.size()>{});
    }

    template <class T, size_t N, std::array<int, N> const &Arr, size_t... Is>
    void pybind11_callBindArrayTransformers_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...> /*unused*/)
    {
        (_pybind11_ArrayTransformer_define_dispatch<T, Arr[Is]>(m), ...);
    }

    template <class T, int offset = 0>
    void pybind11_callBindArrayTransformers_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = _get_pybind11_arrayRowsizeInstantiationList<offset>();
        pybind11_callBindArrayTransformers_rowsizes_sequence<
            T, seq.size(), seq>(m, std::make_index_sequence<seq.size()>{});
    }

    template <class T, size_t N, std::array<int, N> const &Arr, size_t... Is>
    void pybind11_callBindParArrayPairs_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...> /*unused*/)
    {
        (_pybind11_ParArrayPair_define_dispatch<T, Arr[Is]>(m), ...);
    }

    template <class T, int offset = 0>
    void pybind11_callBindParArrayPairs_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = _get_pybind11_arrayRowsizeInstantiationList<offset>();
        pybind11_callBindParArrayPairs_rowsizes_sequence<
            T, seq.size(), seq>(m, std::make_index_sequence<seq.size()>{});
    }
}

namespace DNDS
{
    void pybind11_bind_Array_All(py::module_ m);

#define pybind11_bind_Array_All_X_declare(offset) \
    void pybind11_bind_Array_All_##offset(py::module_ m)

#define pybind11_bind_Array_All_X_call(offset, m) \
    pybind11_bind_Array_All_##offset(m)

    pybind11_bind_Array_All_X_declare(1);
    pybind11_bind_Array_All_X_declare(2);
    pybind11_bind_Array_All_X_declare(3);
    pybind11_bind_Array_All_X_declare(4);
    pybind11_bind_Array_All_X_declare(5);
    pybind11_bind_Array_All_X_declare(6);
    pybind11_bind_Array_All_X_declare(7);
    // definitions are offloaded to Array_bind_offset/*.cpp

    inline void pybind11_bind_Array_Offsets(py::module_ m)
    {
        pybind11_bind_Array_All_X_call(1, m);
        pybind11_bind_Array_All_X_call(2, m);
        pybind11_bind_Array_All_X_call(3, m);
        pybind11_bind_Array_All_X_call(4, m);
        pybind11_bind_Array_All_X_call(5, m);
        pybind11_bind_Array_All_X_call(6, m);
        pybind11_bind_Array_All_X_call(7, m);
    }
}