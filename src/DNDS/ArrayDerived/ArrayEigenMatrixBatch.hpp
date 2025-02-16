#pragma once

#include "../ArrayTransformer.hpp"

namespace DNDS
{
    class MatrixBatch
    {
    public:
        struct UInt32PairIn64
        {
            uint64_t data;
            [[nodiscard]] uint32_t getM() const { return uint32_t(data & 0x00000000FFFFFFFFULL); }
            [[nodiscard]] uint32_t getN() const { return uint32_t(data >> 32); }
            void setM(uint32_t v) { data = (data & 0xFFFFFFFF00000000ULL) | uint64_t(v); }
            void setN(uint32_t v) { data = (data & 0x00000000FFFFFFFFULL) | (uint64_t(v) << 32); }
        };
        static_assert(sizeof(UInt32PairIn64) == 8);

        struct UInt16QuadIn64
        {
            uint64_t data;
            [[nodiscard]] uint16_t getA() const { return uint16_t((data & 0x000000000000FFFFULL) >> 0); }
            [[nodiscard]] uint16_t getB() const { return uint16_t((data & 0x00000000FFFF0000ULL) >> 16); }
            [[nodiscard]] uint16_t getC() const { return uint16_t((data & 0x0000FFFF00000000ULL) >> 32); }
            [[nodiscard]] uint16_t getD() const { return uint16_t((data & 0xFFFF000000000000ULL) >> 48); }

            void setA(uint16_t v) { data = (data & (~0x000000000000FFFFULL)) | (uint64_t(v) << 0); }
            void setB(uint16_t v) { data = (data & (~0x00000000FFFF0000ULL)) | (uint64_t(v) << 16); }
            void setC(uint16_t v) { data = (data & (~0x0000FFFF00000000ULL)) | (uint64_t(v) << 32); }
            void setD(uint16_t v) { data = (data & (~0xFFFF000000000000ULL)) | (uint64_t(v) << 48); }
        };
        static_assert(sizeof(UInt32PairIn64) == 8 && sizeof(UInt16QuadIn64) == 8);

        using t_matrix = MatrixXR;
        using t_map = Eigen::Map<t_matrix>;

        static rowsize getBufSize(const std::vector<t_matrix> &matrices)
        {
            DNDS_assert(matrices.size() < DNDS_ROWSIZE_MAX);
            rowsize bufSiz = matrices.size() + 1;
            for (const auto &i : matrices)
            {
                Eigen::Index mSiz = i.rows() * i.cols();
                DNDS_assert(mSiz + bufSiz < DNDS_ROWSIZE_MAX && i.rows() < UINT16_MAX && i.cols() < UINT16_MAX);
                bufSiz += mSiz;
            }
            return bufSiz;
        }

    private:
        real *_buf;
        rowsize _buf_size;
        static_assert(sizeof(real) == 8);

    public:
        MatrixBatch(real *n_buf, rowsize new_size) : _buf(n_buf), _buf_size(new_size)
        {
        }

        uint64_t &Size()
        {
            DNDS_assert(_buf_size > 0);
            return *(uint64_t *)(_buf);
        }

        uint16_t getNRow(rowsize k)
        {
            DNDS_assert(k < _buf_size - 1);
            return ((UInt16QuadIn64 *)(_buf + k + 1))->getA();
        }

        uint16_t getNCol(rowsize k)
        {
            DNDS_assert(k < _buf_size - 1);
            return ((UInt16QuadIn64 *)(_buf + k + 1))->getB();
        }

        uint32_t getOffset(rowsize k)
        {
            DNDS_assert(k < _buf_size - 1);
            return ((UInt32PairIn64 *)(_buf + k + 1))->getN();
        }

        void setNRow(rowsize k, uint16_t v)
        {
            DNDS_assert(k < _buf_size - 1);
            return ((UInt16QuadIn64 *)(_buf + k + 1))->setA(v);
        }

        void setNCol(rowsize k, uint16_t v)
        {
            DNDS_assert(k < _buf_size - 1);
            return ((UInt16QuadIn64 *)(_buf + k + 1))->setB(v);
        }

        void setOffset(rowsize k, uint32_t v)
        {
            DNDS_assert(k < _buf_size - 1);
            return ((UInt32PairIn64 *)(_buf + k + 1))->setN(v);
        }

        void CompressIn(const std::vector<t_matrix> &matrices)
        {
            DNDS_assert(getBufSize(matrices) <= _buf_size);
            this->Size() = uint64_t(matrices.size()); // assuming could fit
            // std::cout << "Size: " << this->Size() << std::endl;
            uint32_t curOffset = uint32_t(this->Size()) + 1;
            for (size_t i = 0; i < matrices.size(); i++)
            {
                this->setNRow(rowsize(i), uint16_t(matrices[i].rows()));
                this->setNCol(rowsize(i), uint16_t(matrices[i].cols()));
                this->setOffset(rowsize(i), curOffset);
                this->operator[](i) = matrices[i];
                // std::cout << "SET: " << this->operator[](i) << std::endl;
                curOffset += matrices[i].size();
            }
        }

        t_map operator[](rowsize k)
        {
            DNDS_assert(k < this->Size());
            auto n_row = getNRow(k);
            auto n_col = getNCol(k);
            auto offset = getOffset(k);
            return {_buf + offset, n_row, n_col};
        }
    };

    // has to use non uniform?
    class ArrayEigenMatrixBatch : public ParArray<real, NonUniformSize>
    {
    public:
        using t_base = ParArray<real, NonUniformSize>;
        using t_base::t_base;

        using t_matrix = MatrixBatch::t_matrix;
        using t_map = MatrixBatch::t_map;

    private:
        using t_base::ResizeRow;

    public:
        void InitializeWriteRow(index i, const std::vector<t_matrix> &matrices)
        {
            this->ResizeRow(i, MatrixBatch::getBufSize(matrices));
            MatrixBatch batch(this->t_base::operator[](i), this->RowSize(i));
            batch.CompressIn(matrices);
        }

        MatrixBatch operator[](index i)
        {
            return {this->t_base::operator[](i), this->RowSize(i)};
        }

        t_map operator()(index i, rowsize j)
        {
            return this->operator[](i)[j];
        }

        using t_base::ReadSerializer;
        using t_base::WriteSerializer; //! because no extra data than Array<>
    };

}