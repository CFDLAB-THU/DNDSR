#pragma once

#include "ArrayEigenUniMatrixBatch.hpp"
#include "../Array_bind.hpp"

namespace DNDS
{

    template <int _n_row, int _n_col>
    std::string pybind11_ArrayEigenUniMatrixBatch_name()
    {
        return "ArrayEigenUniMatrixBatch" + fmt::format("_{}x{}",
                                                        RowSize_To_PySnippet(_n_row),
                                                        RowSize_To_PySnippet(_n_col));
    }

    template <int _n_row, int _n_col>
    std::string pybind11_ArrayEigenUniMatrixBatchPair_name()
    {
        return "ArrayEigenUniMatrixBatchPair" + fmt::format("_{}x{}",
                                                            RowSize_To_PySnippet(_n_row),
                                                            RowSize_To_PySnippet(_n_col));
    }

    template <int _n_row, int _n_col> // ! shared ptr
    using tPy_ArrayEigenUniMatrixBatch = py::class_<ArrayEigenUniMatrixBatch<_n_row, _n_col>, ssp<ArrayEigenUniMatrixBatch<_n_row, _n_col>>>;

    template <int _n_row, int _n_col> // ! unique ptr
    using tPy_ArrayEigenUniMatrixBatchPair = py::class_<ArrayEigenUniMatrixBatchPair<_n_row, _n_col>>;
}

namespace DNDS
{
    template <class TArrayEigenUniMatrixBatch = ArrayEigenUniMatrixBatch<3, 3>>
    auto pybind11_ArrayEigenUniMatrixBatch_getitem(TArrayEigenUniMatrixBatch &self, std::tuple<index, rowsize> index_)
    {
        using tElem = real;
        auto mat = self(std::get<0>(index_), std::get<1>(index_));
        return py::memoryview::from_buffer<tElem>(
            mat.data(),
            {mat.rows(), mat.cols()},
            {sizeof(tElem) * mat.rowStride(), sizeof(tElem) * mat.colStride()},
            false);
    }

    template <class TArrayEigenUniMatrixBatch = ArrayEigenUniMatrixBatch<3, 3>>
    auto pybind11_ArrayEigenUniMatrixBatch_setitem(TArrayEigenUniMatrixBatch &self, std::tuple<index, rowsize> index_, py::buffer row)
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
    }

    template <class TArrayEigenUniMatrixBatch = ArrayEigenUniMatrixBatch<3, 3>>
    auto pybind11_ArrayEigenUniMatrixBatch_getitem_row(TArrayEigenUniMatrixBatch &self, index index_)
    {
        using tElem = real;
        auto mat0 = self(index_, 0);
        //! warning, assume mat0.rowStride() applies to following matrices
        return py::memoryview::from_buffer<tElem>(
            mat0.data(),
            std::array<index, 3>{self.BatchSize(index_), mat0.rows(), mat0.cols()},
            std::array<index, 3>{static_cast<index>(sizeof(tElem)) * self.MSize(), static_cast<index>(sizeof(tElem)) * mat0.rowStride(), static_cast<index>(sizeof(tElem)) * mat0.colStride()},
            false);
    }

    template <class TArrayEigenUniMatrixBatch = ArrayEigenUniMatrixBatch<3, 3>>
    auto pybind11_ArrayEigenUniMatrixBatch_setitem_row(TArrayEigenUniMatrixBatch &self, index index_, py::buffer row)
    {
        using tElem = real;
        auto row_info = row.request(false);
        DNDS_assert(row_info.item_type_is_equivalent_to<tElem>());

        DNDS_assert_info(row_info.shape.size() == 3, "need to pass a 2-d array");
        DNDS_assert_info(row_info.shape[0] == self.BatchSize(index_), "batch size not matching");
        DNDS_assert_info(row_info.shape[1] == self.Rows(), "row size not matching");
        DNDS_assert_info(row_info.shape[2] == self.Cols(), "col size not matching");
        auto row_start_ptr = reinterpret_cast<tElem *>(row_info.ptr);
        DNDS_assert(row_info.strides.size() == 3);
        for (index iB = 0; iB < row_info.shape[0]; iB++)
        {
            auto mat = self(index_, iB);
            for (index iM = 0; iM < row_info.shape[1]; iM++)
                for (index iN = 0; iN < row_info.shape[2]; iN++)
                    mat(iM, iN) = *reinterpret_cast<tElem *>(
                        reinterpret_cast<uint8_t *>(row_info.ptr) + row_info.strides[0] * iB + row_info.strides[1] * iM + row_info.strides[2] * iN);
        }
    }
}

namespace DNDS
{
    template <int _n_row, int _n_col>
    tPy_ArrayEigenUniMatrixBatch<_n_row, _n_col>
    pybind11_ArrayEigenUniMatrixBatch_declare(py::module_ &m)
    {
        auto ParArray_ = pybind11_ParArray_get_class<real, NonUniformSize>(m);
        return {m, pybind11_ArrayEigenUniMatrixBatch_name<_n_row, _n_col>().c_str(), ParArray_};
        // std::cout << py::format_descriptor<Eigen::Matrix<double, 3, 3>>().format() << std::endl;
    }

    template <int _n_row, int _n_col>
    tPy_ArrayEigenUniMatrixBatch<_n_row, _n_col>
    pybind11_ArrayEigenUniMatrixBatch_get_class(py::module_ &m)
    {
        return {m.attr(pybind11_ArrayEigenUniMatrixBatch_name<_n_row, _n_col>().c_str())};
    }

    template <int _n_row, int _n_col>
    void pybind11_ArrayEigenUniMatrixBatch_define(py::module_ &m)
    {

        using TArrayEigenUniMatrixBatch = ArrayEigenUniMatrixBatch<_n_row, _n_col>;
        auto ArrayEigenUniMatrixBatch_ = pybind11_ArrayEigenUniMatrixBatch_declare<_n_row, _n_col>(m);

        // // helper
        // using TArrayEigenUniMatrixBatch = ArrayEigenUniMatrixBatch<3, 3>;
        // auto ArrayEigenUniMatrixBatch_ = pybind11_ArrayEigenUniMatrixBatch_declare<3, 3>(m);
        // // helper

        ArrayEigenUniMatrixBatch_
            // we only bind the non-default ctor here
            .def(py::init<const MPIInfo &>(), py::arg("nmpi"))
            .def("Resize", [](TArrayEigenUniMatrixBatch &self, index size)
                 { return self.Resize(size); }, py::arg("size"))
            .def("Resize", [](TArrayEigenUniMatrixBatch &self, index size, int r, int c)
                 { return self.Resize(size, r, c); }, py::arg("size"), py::arg("r"), py::arg("c"));
        ArrayEigenUniMatrixBatch_ // the once for all resize
            .def("Resize", [](TArrayEigenUniMatrixBatch &self, index size, int r, int c, py::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> batchSizes)
                 { return self.Resize(size, r, c, [&](index i)
                                      { return batchSizes.at(i); }); }, py::arg("size"), py::arg("r"), py::arg("c"), py::arg("batchSizes"));
        ArrayEigenUniMatrixBatch_
            .def("ResizeMatrix", [](TArrayEigenUniMatrixBatch &self, index r, index c)
                 { return self.ResizeMatrix(r, c); }, py::arg("r") = -1, py::arg("c") = -1);
        ArrayEigenUniMatrixBatch_
            .def_property_readonly("rows", [](const TArrayEigenUniMatrixBatch &self)
                                   { return self.Rows(); })
            .def_property_readonly("cols", [](const TArrayEigenUniMatrixBatch &self)
                                   { return self.Cols(); })
            .def_property_readonly("msize", [](const TArrayEigenUniMatrixBatch &self)
                                   { return self.MSize(); });
        ArrayEigenUniMatrixBatch_
            .def("ResizeBatch", [](TArrayEigenUniMatrixBatch &self, index i, rowsize b_size)
                 { return self.ResizeBatch(i, b_size); }, py::arg("i"), py::arg("b_size"))
            .def("ResizeRow", [](TArrayEigenUniMatrixBatch &self, index i, rowsize b_size)
                 { return self.ResizeRow(i, b_size); }, py::arg("i"), py::arg("b_size"));
        ArrayEigenUniMatrixBatch_
            .def("RowSize", [](const TArrayEigenUniMatrixBatch &self, index i)
                 { return self.RowSize(i); }, py::arg("i"))
            .def("BatchSize", [](const TArrayEigenUniMatrixBatch &self, index i)
                 { return self.BatchSize(i); }, py::arg("i"));
        ArrayEigenUniMatrixBatch_
            .def(
                "__getitem__",
                [](TArrayEigenUniMatrixBatch &self, std::tuple<index, rowsize> index_)
                {
                    return pybind11_ArrayEigenUniMatrixBatch_getitem(self, index_);
                },
                py::keep_alive<0, 1>())
            .def(
                "__setitem__",
                [](TArrayEigenUniMatrixBatch &self, std::tuple<index, rowsize> index_, py::buffer row)
                {
                    return pybind11_ArrayEigenUniMatrixBatch_setitem(self, index_, row);
                })
            .def(
                "__getitem__",
                [](TArrayEigenUniMatrixBatch &self, index index_)
                {
                    return pybind11_ArrayEigenUniMatrixBatch_getitem_row(self, index_);
                },
                py::keep_alive<0, 1>())
            .def(
                "__setitem__",
                [](TArrayEigenUniMatrixBatch &self, index index_, py::buffer row)
                {
                    return pybind11_ArrayEigenUniMatrixBatch_setitem_row(self, index_, row);
                });
    }

    template <int _n_row, int _n_col>
    void _pybind11_ArrayEigenUniMatrixBatch_define_dispatch(py::module_ &m)
    {
        if constexpr (_n_row == UnInitRowsize || _n_col == UnInitRowsize)
            return;
        else
            return pybind11_ArrayEigenUniMatrixBatch_define<_n_row, _n_col>(m);
    }

}

namespace DNDS
{
    template <int _n_row, int _n_col>
    tPy_ArrayEigenUniMatrixBatchPair<_n_row, _n_col>
    pybind11_ArrayEigenUniMatrixBatchPair_declare(py::module_ &m)
    {
        return {m, pybind11_ArrayEigenUniMatrixBatchPair_name<_n_row, _n_col>().c_str()};
    }

    template <int _n_row, int _n_col>
    tPy_ArrayEigenUniMatrixBatchPair<_n_row, _n_col>
    pybind11_ArrayEigenUniMatrixBatchPair_get_class(py::module_ &m)
    {
        return {m.attr(pybind11_ArrayEigenUniMatrixBatchPair_name<_n_row, _n_col>().c_str())};
    }

    template <int _n_row, int _n_col>
    void pybind11_ArrayEigenUniMatrixBatchPair_define(py::module_ &m)
    {

        using TPair = ArrayEigenUniMatrixBatchPair<_n_row, _n_col>;
        auto Pair_ = pybind11_ArrayEigenUniMatrixBatchPair_declare<_n_row, _n_col>(m);

        // // helper
        // using TPair = ArrayEigenUniMatrixBatchPair<3, 3>;
        // auto Pair_ = pybind11_ArrayEigenUniMatrixBatchPair_declare<3, 3>(m);
        // // helper

        pybind11_ArrayPairGenericBindBasics<TPair>(Pair_);

        Pair_
            .def("RowSize", [](const TPair &self, index i)
                 { return self.RowSize(i); }, py::arg("i"));
        Pair_
            .def(
                "__getitem__",
                [](TPair &self, std::tuple<index, rowsize> index_)
                {
                    return self.runFunctionAppendedIndex(std::get<0>(index_), [&](auto ar, index iC)
                                                         { return pybind11_ArrayEigenUniMatrixBatch_getitem(ar, std::make_tuple(iC, std::get<1>(index_))); });
                },
                py::keep_alive<0, 1>())
            .def(
                "__setitem__",
                [](TPair &self, std::tuple<index, rowsize> index_, py::buffer row)
                {
                    return self.runFunctionAppendedIndex(std::get<0>(index_), [&](auto ar, index iC)
                                                         { return pybind11_ArrayEigenUniMatrixBatch_setitem(ar, std::make_tuple(iC, std::get<1>(index_)), row); });
                })
            .def(
                "__getitem__",
                [](TPair &self, index index_)
                {
                    return self.runFunctionAppendedIndex(index_, [&](auto ar, index iC)
                                                         { return pybind11_ArrayEigenUniMatrixBatch_getitem_row(ar, iC); });
                },
                py::keep_alive<0, 1>())
            .def(
                "__setitem__",
                [](TPair &self, index index_, py::buffer row)
                {
                    return self.runFunctionAppendedIndex(index_, [&](auto ar, index iC)
                                                         { return pybind11_ArrayEigenUniMatrixBatch_setitem_row(ar, iC, row); });
                });
    }

    template <int _n_row, int _n_col>
    void _pybind11_ArrayEigenUniMatrixBatchPair_define_dispatch(py::module_ &m)
    {
        if constexpr (_n_row == UnInitRowsize || _n_col == UnInitRowsize)
            return;
        else
            return pybind11_ArrayEigenUniMatrixBatchPair_define<_n_row, _n_col>(m);
    }

}

namespace DNDS
{

    template <rowsize mat_n, size_t N, std::array<int, N> const &Arr, size_t... Is>
    void __pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes_sequence(py::module_ &m, std::index_sequence<Is...>)
    {
        (_pybind11_ArrayEigenUniMatrixBatch_define_dispatch<Arr[Is], mat_n>(m), ...);
        (_pybind11_ArrayEigenUniMatrixBatchPair_define_dispatch<Arr[Is], mat_n>(m), ...);
    }

    template <rowsize mat_n>
    void pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes(py::module_ &m)
    {
        static constexpr auto seq = pybind11_arrayRowsizeInstantiationList;
        __pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes_sequence<
            mat_n,
            seq.size(),
            seq>(m, std::make_index_sequence<seq.size()>{});
    }

    extern template void pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes<1>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes<2>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes<3>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes<4>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes<5>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes<6>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes<7>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes<8>(py::module_ &m);
    extern template void pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes<DynamicSize>(py::module_ &m);

    void pybind11_bind_ArrayEigenUniMatrixBatch_All(py::module_ &m);
}