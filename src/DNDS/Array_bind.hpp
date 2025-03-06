#pragma once

#include "Array.hpp"
#include "ArrayTransformer.hpp"
#include "ArrayPair.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

namespace DNDS
{

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using tPy_Array = py::class_<Array<T, _row_size, _row_max, _align>, std::shared_ptr<Array<T, _row_size, _row_max, _align>>>;

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_Array<T, _row_size, _row_max, _align>
    pybind11_array_declare(py::module_ &m)
    {
        static_assert(std::is_arithmetic_v<T>);
        return tPy_Array<T, _row_size, _row_max, _align>(
            m,
            fmt::format("Array_{}_{}_{}_{}",
                        py::format_descriptor<T>().format(),
                        RowSize_To_PySnippet(_row_size),
                        RowSize_To_PySnippet(_row_max),
                        RowSize_To_PySnippet(_align))
                .c_str());
        // std::cout << py::format_descriptor<Eigen::Matrix<double, 3, 3>>().format() << std::endl;
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void pybind11_array_define(py::module_ &m)
    {
        using TArray = Array<T, _row_size, _row_max, _align>;
        auto Array_ = pybind11_array_declare<T, _row_size, _row_max, _align>(m);
        // // helper
        // using TArray = Array<real, 1, 1, -1>;
        // auto Array_ = pybind11_array_declare<real, 1, 1, -1>(m);
        // // helper

        Array_
            .def(py::init<>())
            .def("Size", &TArray::Size)
            .def("Compress", &TArray::Compress)
            .def("Decompress", &TArray::Decompress)
            .def("IfCompressed", &TArray::IfCompressed);
        Array_
            .def("getRowStart",
                 [](TArray &self) -> py::object
                 {
                     if (!self.getRowStart())
                         return py::none();
                     auto &rs = *self.getRowStart();
                     return py::memoryview::from_buffer<index>(rs.data(), {rs.size()}, {sizeof(index)});
                 });

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
}