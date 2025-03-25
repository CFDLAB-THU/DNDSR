#pragma once

#include "ArrayEigenMatrix.hpp"
#include "../Array_bind.hpp"

namespace DNDS
{

    template <rowsize _mat_ni = 1, rowsize _mat_nj = 1,
              rowsize _mat_ni_max = _mat_ni, rowsize _mat_nj_max = _mat_nj, rowsize _align = NoAlign>
    std::string pybind11_ArrayEigenMatrix_name_appends()
    {
        return fmt::format("_{}x{}_{}x{}_{}",
                           RowSize_To_PySnippet(_mat_ni),
                           RowSize_To_PySnippet(_mat_nj),
                           RowSize_To_PySnippet(_mat_ni_max),
                           RowSize_To_PySnippet(_mat_nj_max),
                           RowSize_To_PySnippet(_align));
    }

    template <rowsize _mat_ni = 1, rowsize _mat_nj = 1,
              rowsize _mat_ni_max = _mat_ni, rowsize _mat_nj_max = _mat_nj, rowsize _align = NoAlign>
    std::string pybind11_ArrayEigenMatrix_name()
    {
        return "ArrayEigenMatrix" + pybind11_ArrayEigenMatrix_name_appends<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>();
    }

    template <rowsize _mat_ni = 1, rowsize _mat_nj = 1,
              rowsize _mat_ni_max = _mat_ni, rowsize _mat_nj_max = _mat_nj, rowsize _align = NoAlign>
    using tPy_ArrayEigenMatrix = py::class_<ArrayEigenMatrix<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>, ssp<ArrayEigenMatrix<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>>>;
}

namespace DNDS
{
    template <rowsize _mat_ni = 1, rowsize _mat_nj = 1,
              rowsize _mat_ni_max = _mat_ni, rowsize _mat_nj_max = _mat_nj, rowsize _align = NoAlign>
    tPy_ArrayEigenMatrix<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>
    pybind11_ArrayEigenMatrix_declare(py::module_ &m)
    {
        using TArrayEigenMatrix = ArrayEigenMatrix<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>;
        static constexpr rowsize rs = TArrayEigenMatrix::t_base::rs;
        static constexpr rowsize rm = TArrayEigenMatrix::t_base::rm;
        static constexpr rowsize al = TArrayEigenMatrix::t_base::al;
        auto ParArray_ = pybind11_ParArray_get_class<real, rs, rm, al>(m);
        return {m, pybind11_ArrayEigenMatrix_name<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>().c_str(), ParArray_};
    }

    template <rowsize _mat_ni = 1, rowsize _mat_nj = 1,
              rowsize _mat_ni_max = _mat_ni, rowsize _mat_nj_max = _mat_nj, rowsize _align = NoAlign>
    tPy_ArrayEigenMatrix<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>
    pybind11_ArrayEigenMatrix_get_class(py::module_ &m)
    {
        return {m.attr(pybind11_ArrayEigenMatrix_name<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>().c_str())};
    }

    template <rowsize _mat_ni = 1, rowsize _mat_nj = 1,
              rowsize _mat_ni_max = _mat_ni, rowsize _mat_nj_max = _mat_nj, rowsize _align = NoAlign>
    void pybind11_ArrayEigenMatrix_define(py::module_ &m)
    {

        using TArrayEigenMatrix = ArrayEigenMatrix<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>;
        auto ArrayEigenMatrix_ = pybind11_ArrayEigenMatrix_declare<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>(m);

        // // helper
        // using TArrayEigenMatrix = ArrayEigenMatrix<3, 3>;
        // auto ArrayEigenMatrix_ = pybind11_ArrayEigenMatrix_declare<3, 3>(m);
        // // helper

        ArrayEigenMatrix_
            // we only bind the non-default ctor here
            .def(py::init<const MPIInfo &>(), py::arg("nmpi"))
            .def("Resize", [](TArrayEigenMatrix &self, index nSize, rowsize md, rowsize nd)
                 { return self.Resize(nSize, md, nd); }, py::arg("size"), py::arg("m_size"), py::arg("n_size"));
        ArrayEigenMatrix_
            .def("MatRowSize", [](const TArrayEigenMatrix &self, index iMat)
                 { return self.MatRowSize(iMat); }, py::arg("iMat") = 0)
            .def("MatColSize", [](const TArrayEigenMatrix &self, index iMat)
                 { return self.MatColSize(iMat); }, py::arg("iMat") = 0);

        ArrayEigenMatrix_
            .def("ResizeMat", [](TArrayEigenMatrix &self, index iMat, rowsize nm, rowsize nn)
                 { return self.ResizeMat(iMat, nm, nn); }, py::arg("iMat"), py::arg("nm"), py::arg("nn"))
            .def("ResizeRow", [](TArrayEigenMatrix &self, index iMat, rowsize nm, rowsize nn)
                 { return self.ResizeRow(iMat, nm, nn); }, py::arg("iMat"), py::arg("nm"), py::arg("nn"));

        ArrayEigenMatrix_
            .def(
                "__getitem__",
                [](TArrayEigenMatrix &self, index index_)
                {
                    using tElem = real;
                    auto mat = self[index_];
                    return py::memoryview::from_buffer<tElem>(
                        mat.data(),
                        {mat.rows(), mat.cols()},
                        {sizeof(tElem) * mat.rowStride(), sizeof(tElem) * mat.colStride()},
                        false);
                },
                py::keep_alive<0, 1>())
            .def(
                "__setitem__",
                [](TArrayEigenMatrix &self, index index_, py::buffer row)
                {
                    using tElem = real;
                    auto row_info = row.request(false);
                    DNDS_assert(row_info.item_type_is_equivalent_to<tElem>());
                    auto iRow = index_;
                    auto mat = self[index_];
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

    template <rowsize _mat_ni = 1, rowsize _mat_nj = 1,
              rowsize _mat_ni_max = _mat_ni, rowsize _mat_nj_max = _mat_nj, rowsize _align = NoAlign>
    void _pybind11_ArrayEigenMatrix_define_dispatch(py::module_ &m)
    {
        // std::cout << fmt::format("binding ArrayEigenMatrix {},{}", _mat_ni, _mat_nj) << std::endl;
        if constexpr (_mat_ni == UnInitRowsize || _mat_nj == UnInitRowsize)
            return;
        else
            return pybind11_ArrayEigenMatrix_define<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>(m);
    }
}

namespace DNDS
{
    template <rowsize mat_n, size_t N, std::array<int, N> const &Arr, size_t... Is>
    void __pybind11_callBindArrayEigenMatrixs_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...>)
    {
        (_pybind11_ArrayEigenMatrix_define_dispatch<Arr[Is], mat_n>(m), ...);
    }

    template <rowsize mat_n>
    void pybind11_callBindArrayEigenMatrixs_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = pybind11_arrayRowsizeInstantiationList;
        __pybind11_callBindArrayEigenMatrixs_rowsizes_sequence<
            mat_n,
            seq.size(),
            seq>(m, std::make_index_sequence<seq.size()>{});
    }

    extern template void pybind11_callBindArrayEigenMatrixs_rowsizes<1>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenMatrixs_rowsizes<2>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenMatrixs_rowsizes<3>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenMatrixs_rowsizes<4>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenMatrixs_rowsizes<5>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenMatrixs_rowsizes<6>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenMatrixs_rowsizes<7>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenMatrixs_rowsizes<8>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenMatrixs_rowsizes<DynamicSize>(py::module_ &m);

    void pybind11_bind_ArrayEigenMatrix_All(py::module_ &m);
}