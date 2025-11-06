#pragma once

#include "../ArrayTransformer.hpp"
#include "DNDS/Defines.hpp"
#include "ArrayEigenMatrixBatch_DeviceView.hpp"
#include "DNDS/DeviceStorage.hpp"

namespace DNDS
{
    // has to use non uniform?
    class ArrayEigenMatrixBatch : public ParArray<real, NonUniformSize>
    {
    public:
        using t_base = ParArray<real, NonUniformSize>;
        using t_base::t_base;

        using t_matrix = typename MatrixBatch<real>::t_matrix;
        using t_map = typename MatrixBatch<real>::t_map;

    private:
        using t_base::ResizeRow;

    public:
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
    };

}