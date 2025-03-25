#pragma once

#include "ArrayEigenMatrixBatch.hpp"
#include "../Array_bind.hpp"

namespace DNDS
{

    inline std::string pybind11_ArrayEigenMatrixBatch_name()
    {
        return "ArrayEigenMatrixBatch";
    }

    using tPy_ArrayEigenMatrixBatch = py::class_<ArrayEigenMatrixBatch, ssp<ArrayEigenMatrixBatch>>;

}

namespace DNDS
{
    tPy_ArrayEigenMatrixBatch
    pybind11_ArrayEigenMatrixBatch_declare(py::module_ &m)
    {
        auto ParArray_ = pybind11_ParArray_get_class<real, NonUniformSize>(m);
        return {m, pybind11_ArrayEigenMatrixBatch_name().c_str(), ParArray_};
    }

    tPy_ArrayEigenMatrixBatch
    pybind11_ArrayEigenMatrixBatch_get_class(py::module_ &m)
    {
        return {m.attr(pybind11_ArrayEigenMatrixBatch_name().c_str())};
    }

    inline void pybind11_ArrayEigenMatrixBatch_define(py::module_ &m)
    {

        using TArrayEigenMatrixBatch = ArrayEigenMatrixBatch;
        auto ArrayEigenMatrixBatch_ = pybind11_ArrayEigenMatrixBatch_declare(m);

        ArrayEigenMatrixBatch_
            // we only bind the non-default ctor here
            .def(py::init<const MPIInfo &>(), py::arg("nmpi"))
            .def(
                "InitializeWriteRow",
                [](TArrayEigenMatrixBatch &self, index i, const py::list &matList)
                {
                    using tElem = real;
                    using tReadMap = Eigen::Map<
                        const Eigen::Matrix<tElem, Eigen::Dynamic, Eigen::Dynamic, Eigen::DontAlign | Eigen::ColMajor>,
                        Eigen::Unaligned,
                        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
                    std::vector<tReadMap> mat_maps;
                    for (const auto &v : matList)
                    {
                        if (!py::isinstance<py::buffer>(v))
                            throw std::runtime_error("All elements must be buffer-compatible objects.");
                        py::buffer buf = v.cast<py::buffer>();
                        auto buf_info = buf.request(false);
                        DNDS_assert(buf_info.item_type_is_equivalent_to<tElem>());
                        DNDS_assert_info(buf_info.shape.size() == 2, "need to pass a 2-d array");
                        auto buf_start_ptr = reinterpret_cast<tElem *>(buf_info.ptr);
                        DNDS_assert(buf_info.strides.size() == 2);

                        auto c_mat_map = tReadMap(
                            buf_start_ptr,
                            buf_info.shape[0],
                            buf_info.shape[1],
                            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(buf_info.strides[1] / sizeof(tElem) /*col stride*/, buf_info.strides[0] / sizeof(tElem) /*row stride*/));
                        mat_maps.push_back(c_mat_map);
                    }
                    return self.InitializeWriteRow(i, mat_maps);
                },
                py::arg("i"), py::arg("matList"));

        ArrayEigenMatrixBatch_
            .def(
                "__getitem__",
                [](TArrayEigenMatrixBatch &self, index index_)
                {
                    using tElem = real;

                    auto rowBatch = self[index_];
                    py::list ret;
                    for (int iMat = 0; iMat < rowBatch.Size(); iMat++)
                    {
                        auto mat = rowBatch[iMat];
                        ret.append(
                            py::memoryview::from_buffer<tElem>(
                                mat.data(),
                                {mat.rows(), mat.cols()},
                                {sizeof(tElem) * mat.rowStride(), sizeof(tElem) * mat.colStride()},
                                false)); //! warning: deleting list and Array could make the items dangling
                    }
                    return ret;
                },
                py::keep_alive<0, 1>())
            .def(
                "__getitem__",
                [](TArrayEigenMatrixBatch &self, std::tuple<index, rowsize> index_)
                {
                    using tElem = real;

                    auto mat = self(std::get<0>(index_), std::get<1>(index_));
                    return py::memoryview::from_buffer<tElem>(
                        mat.data(),
                        {mat.rows(), mat.cols()},
                        {sizeof(tElem) * mat.rowStride(), sizeof(tElem) * mat.colStride()},
                        false);
                },
                py::keep_alive<0, 1>())
            .def(
                "__setitem__",
                [](TArrayEigenMatrixBatch &self, std::tuple<index, rowsize> index_, py::buffer row)
                {
                    using tElem = real;
                    auto row_info = row.request(false);
                    DNDS_assert(row_info.item_type_is_equivalent_to<tElem>());
                    auto mat = self(std::get<0>(index_), std::get<1>(index_));
                    DNDS_assert_info(row_info.shape.size() == 2, "need to pass a 2-d array");
                    DNDS_assert_info(row_info.shape[0] == mat.rows(), "row size not matching");
                    DNDS_assert_info(row_info.shape[1] == mat.cols(), "col size not matching");

                    auto row_start_ptr = reinterpret_cast<tElem *>(row_info.ptr);
                    DNDS_assert(row_info.strides.size() == 2);
                    auto row_mat_map = Eigen::Map<
                        const Eigen::Matrix<tElem, Eigen::Dynamic, Eigen::Dynamic, Eigen::DontAlign | Eigen::ColMajor>,
                        Eigen::Unaligned,
                        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
                        row_start_ptr,
                        row_info.shape[0],
                        row_info.shape[1],
                        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(row_info.strides[1] / sizeof(tElem) /*col stride*/, row_info.strides[0] / sizeof(tElem) /*row stride*/));
                    mat = row_mat_map;
                });
    }
}

namespace DNDS
{

    inline void pybind11_bind_ArrayEigenMatrixBatch_All(py::module_ &m)
    {
        pybind11_ArrayEigenMatrixBatch_define(m);
    }
}