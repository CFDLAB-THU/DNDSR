#pragma once
/// @file ArrayEigenMatrixBatch.hpp
/// @brief Batch of variable-sized Eigen matrices stored in CSR layout.
/// @par Unit Test Coverage (test_ArrayDerived.cpp, MPI np=1,2,4)
/// - Resize, InitializeWriteRow, Compress, BatchSize, operator()(i,j)
/// - Ghost communication via ArrayEigenMatrixBatchPair
/// @par Not Yet Tested
/// - WriteSerializer / ReadSerializer, device views

#include "../ArrayTransformer.hpp"
#include "DNDS/Defines.hpp"
#include "ArrayEigenMatrixBatch_DeviceView.hpp"
#include "DNDS/Device/DeviceStorage.hpp"

namespace DNDS
{
    /**
     * @brief CSR array storing a variable-sized batch of Eigen matrices per row.
     *
     * @details Unlike @ref DNDS::ArrayEigenUniMatrixBatch "ArrayEigenUniMatrixBatch" the matrices in a batch may have
     * different shapes, so each row carries a self-describing size prefix (see
     * @ref DNDS::MatrixBatch "MatrixBatch" in VectorUtils.hpp). Used where a cell's quadrature rule /
     * basis count is not known a priori at compile time.
     *
     * Populate rows with @ref InitializeWriteRow given a vector of concrete Eigen
     * matrices. After that, #operator()(i,j) yields an `Eigen::Map` view onto
     * the `j`-th matrix of row `i`.
     */
    // has to use non uniform?
    class ArrayEigenMatrixBatch : public ParArray<real, NonUniformSize>
    {
    public:
        using t_self = ArrayEigenMatrixBatch;
        using t_base = ParArray<real, NonUniformSize>;
        using t_base::t_base;

        using t_matrix = typename MatrixBatch<real>::t_matrix;
        using t_map = typename MatrixBatch<real>::t_map;

    private:
        using t_base::ResizeRow;

    public:
        // default copy
        ArrayEigenMatrixBatch(const t_self &R) = default;
        t_self &operator=(const t_self &R) = default;
        // Rule-of-five closure: all value-semantic members.
        ArrayEigenMatrixBatch(t_self &&) noexcept = default;
        t_self &operator=(t_self &&) noexcept = default;
        ~ArrayEigenMatrixBatch() = default;
        // operator= handled automatically

        void clone(const t_self &R)
        {
            this->operator=(R);
        }

        template <class t_matrices_elem>
        void InitializeWriteRow(index i, const std::vector<t_matrices_elem> &matrices)
        {
            this->ResizeRow(i, MatrixBatch<real>::getBufSize(matrices));
            MatrixBatch batch(this->t_base::operator[](i), this->RowSize(i));
            batch.CompressIn(matrices);
        }

        MatrixBatch<real> operator[](index i) // todo: add const version
        {
            return {this->t_base::operator[](i), this->RowSize(i)};
        }

        index BatchSize(index i)
        {
            return this->operator[](i).Size();
        }

        t_map operator()(index i, rowsize j)
        {
            return this->operator[](i)[j];
        }

        using t_base::ReadSerializer;
        using t_base::WriteSerializer; //! because no extra data than Array<>

        template <DeviceBackend B>
        using t_deviceView = ArrayEigenMatrixBatchDeviceView<B, real>;

        template <DeviceBackend B>
        using t_deviceViewConst = ArrayEigenMatrixBatchDeviceView<B, const real>;

        template <DeviceBackend B>
        auto deviceView()
        {
            return t_deviceView<B>{this->t_base::template deviceView<B>()}; // CTOR
        }

        template <DeviceBackend B>
        auto deviceView() const
        {
            return t_deviceViewConst<B>{this->t_base::template deviceView<B>()}; // CTOR
        }

        using t_base::to_device;
        using t_base::to_host;

        /// @brief Element iterator for ArrayEigenMatrixBatch, yielding MatrixBatch per row.
        template <DeviceBackend B>
        class iterator : public ArrayIteratorBase<iterator<B>>
        {
        public:
            using view_type = t_deviceView<B>;
            using t_base_iter = ArrayIteratorBase<iterator<B>>;
            using typename t_base_iter::difference_type;
            using reference = MatrixBatch<real>;
            using iterator_category = std::random_access_iterator_tag;

        protected:
            view_type view;

        public:
            auto getView() const { return view; }
            DNDS_DEVICE_CALLABLE iterator(const iterator &) = default;
            DNDS_DEVICE_CALLABLE iterator &operator=(const iterator &) = default;
            DNDS_DEVICE_CALLABLE iterator(iterator &&) noexcept = default;
            DNDS_DEVICE_CALLABLE iterator &operator=(iterator &&) noexcept = default;
            DNDS_DEVICE_CALLABLE ~iterator() = default;
            DNDS_DEVICE_CALLABLE iterator(const view_type &n_view, index n_iRow) : view(n_view), t_base_iter(n_iRow)
            {
            }

            DNDS_DEVICE_CALLABLE reference operator*() { return view.operator[](this->iRow); }
        };

        template <DeviceBackend B>
        iterator<B> begin()
        {
            return {deviceView<B>(), 0};
        }

        template <DeviceBackend B>
        iterator<B> end()
        {
            return {deviceView<B>(), this->Size()};
        }
    };

}