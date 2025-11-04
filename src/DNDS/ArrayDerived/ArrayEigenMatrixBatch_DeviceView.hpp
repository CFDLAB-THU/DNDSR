#pragma once

#include "../DeviceView.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include <cstddef>

namespace DNDS
{
    class MatrixBatch
    {
    public:
        struct UInt32PairIn64
        {
            uint64_t data;
            DNDS_DEVICE_CALLABLE [[nodiscard]] uint32_t getM() const { return uint32_t(data & 0x00000000FFFFFFFFULL); }
            DNDS_DEVICE_CALLABLE [[nodiscard]] uint32_t getN() const { return uint32_t(data >> 32); }
            DNDS_DEVICE_CALLABLE void setM(uint32_t v) { data = (data & 0xFFFFFFFF00000000ULL) | uint64_t(v); }
            DNDS_DEVICE_CALLABLE void setN(uint32_t v) { data = (data & 0x00000000FFFFFFFFULL) | (uint64_t(v) << 32); }
        };
        static_assert(sizeof(UInt32PairIn64) == 8);

        struct UInt16QuadIn64
        {
            uint64_t data;
            DNDS_DEVICE_CALLABLE [[nodiscard]] uint16_t getA() const { return uint16_t((data & 0x000000000000FFFFULL) >> 0); }
            DNDS_DEVICE_CALLABLE [[nodiscard]] uint16_t getB() const { return uint16_t((data & 0x00000000FFFF0000ULL) >> 16); }
            DNDS_DEVICE_CALLABLE [[nodiscard]] uint16_t getC() const { return uint16_t((data & 0x0000FFFF00000000ULL) >> 32); }
            DNDS_DEVICE_CALLABLE [[nodiscard]] uint16_t getD() const { return uint16_t((data & 0xFFFF000000000000ULL) >> 48); }

            DNDS_DEVICE_CALLABLE void setA(uint16_t v) { data = (data & (~0x000000000000FFFFULL)) | (uint64_t(v) << 0); }
            DNDS_DEVICE_CALLABLE void setB(uint16_t v) { data = (data & (~0x00000000FFFF0000ULL)) | (uint64_t(v) << 16); }
            DNDS_DEVICE_CALLABLE void setC(uint16_t v) { data = (data & (~0x0000FFFF00000000ULL)) | (uint64_t(v) << 32); }
            DNDS_DEVICE_CALLABLE void setD(uint16_t v) { data = (data & (~0xFFFF000000000000ULL)) | (uint64_t(v) << 48); }
        };
        static_assert(sizeof(UInt32PairIn64) == 8 && sizeof(UInt16QuadIn64) == 8);

        using t_matrix = MatrixXR;
        using t_map = Eigen::Map<t_matrix, Eigen::Unaligned>;

        template <class t_matrices_elem>
        static rowsize getBufSize(const std::vector<t_matrices_elem> &matrices)
        {
            DNDS_assert(matrices.size() < DNDS_ROWSIZE_MAX);
            rowsize bufSiz = matrices.size() + 1;
            for (const auto &i : matrices)
            {
                Eigen::Index mSiz = i.rows() * i.cols();
                static_assert(std::numeric_limits<Eigen::Index>::digits > std::numeric_limits<uint32_t>::digits);
                DNDS_assert((mSiz + bufSiz) < DNDS_ROWSIZE_MAX && i.rows() <= UINT16_MAX && i.cols() <= UINT16_MAX);
                bufSiz += mSiz;
            }
            return bufSiz;
        }

    private:
        real *_buf;
        rowsize _buf_size;
        static_assert(sizeof(real) == 8 || sizeof(real) == 4);
        static const ptrdiff_t n_real_in_64 = 8 / sizeof(real);

        DNDS_DEVICE_CALLABLE real *get_kth_64_meta_block(rowsize k) { return _buf + (k + 1) * n_real_in_64; }
        DNDS_DEVICE_CALLABLE [[nodiscard]] const real *get_kth_64_meta_block(rowsize k) const
        {
            return (const_cast<MatrixBatch *>(this))->get_kth_64_meta_block(k);
        }

    public:
        DNDS_DEVICE_CALLABLE MatrixBatch(real *n_buf, rowsize new_size) : _buf(n_buf), _buf_size(new_size)
        {
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] uint64_t &Size() const
        {
            DNDS_assert(_buf_size > 0);
            return *(uint64_t *)(_buf);
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] uint16_t getNRow(rowsize k) const
        {
            DNDS_assert(k < _buf_size - 1);
            return ((UInt16QuadIn64 *)get_kth_64_meta_block(k))->getA();
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] uint16_t getNCol(rowsize k) const
        {
            DNDS_assert(k < _buf_size - 1);
            return ((UInt16QuadIn64 *)get_kth_64_meta_block(k))->getB();
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] uint32_t getOffset(rowsize k) const
        {
            DNDS_assert(k < _buf_size - 1);
            return ((UInt32PairIn64 *)get_kth_64_meta_block(k))->getN();
        }

        DNDS_DEVICE_CALLABLE void setNRow(rowsize k, uint16_t v)
        {
            DNDS_assert(k < _buf_size - 1);
            ((UInt16QuadIn64 *)get_kth_64_meta_block(k))->setA(v);
        }

        DNDS_DEVICE_CALLABLE void setNCol(rowsize k, uint16_t v)
        {
            DNDS_assert(k < _buf_size - 1);
            ((UInt16QuadIn64 *)get_kth_64_meta_block(k))->setB(v);
        }

        DNDS_DEVICE_CALLABLE void setOffset(rowsize k, uint32_t v)
        {
            DNDS_assert(k < _buf_size - 1);
            ((UInt32PairIn64 *)get_kth_64_meta_block(k))->setN(v);
        }

        template <class t_matrices_elem>
        void CompressIn(const std::vector<t_matrices_elem> &matrices)
        {
            DNDS_assert(getBufSize(matrices) <= _buf_size);
            this->Size() = uint64_t(matrices.size()); // assuming could fit
            // std::cout << "Size: " << this->Size() << std::endl;
            uint32_t curOffset = uint32_t(this->Size()) + 1;
            for (size_t i = 0; i < matrices.size(); i++)
            {
                DNDS_assert(matrices[i].rows() <= Eigen::Index(UINT16_MAX));
                DNDS_assert(matrices[i].cols() <= Eigen::Index(UINT16_MAX));
                this->setNRow(rowsize(i), uint16_t(matrices[i].rows()));
                this->setNCol(rowsize(i), uint16_t(matrices[i].cols()));
                this->setOffset(rowsize(i), curOffset);
                this->operator[](i) = matrices[i];
                // std::cout << "SET: " << this->operator[](i) << std::endl;
                static_assert(std::numeric_limits<Eigen::Index>::digits > std::numeric_limits<uint32_t>::digits);
                DNDS_assert(matrices[i].size() <= Eigen::Index(UINT32_MAX - curOffset)); // overflow check
                curOffset += matrices[i].size();
            }
        }

        DNDS_DEVICE_CALLABLE t_map operator[](rowsize k) // todo: add const version
        {
            DNDS_assert(k < this->Size());
            auto n_row = getNRow(k);
            auto n_col = getNCol(k);
            auto offset = getOffset(k);
            return {_buf + offset, n_row, n_col};
        }
    };

    template <DeviceBackend B>
    class ArrayEigenMatrixBatchDeviceView : public ArrayDeviceView<B, real, NonUniformSize>
    {
    public:
        using t_base = ArrayDeviceView<B, real, NonUniformSize>;
        using t_base::t_base;

        using t_matrix = MatrixBatch::t_matrix;
        using t_map = MatrixBatch::t_map;

        using t_self = ArrayEigenMatrixBatchDeviceView<B>;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ArrayEigenMatrixBatchDeviceView, t_self)

        DNDS_DEVICE_CALLABLE ArrayEigenMatrixBatchDeviceView(const t_base &base_view) : t_base(base_view) {}

        DNDS_DEVICE_CALLABLE MatrixBatch operator[](index i) // todo: add const version
        {
            return {this->t_base::operator[](i), this->RowSize(i)};
        }

        DNDS_DEVICE_CALLABLE index BatchSize(index i)
        {
            return this->operator[](i).Size();
        }

        DNDS_DEVICE_CALLABLE t_map operator()(index i, rowsize j)
        {
            return this->operator[](i)[j];
        }
    };

}