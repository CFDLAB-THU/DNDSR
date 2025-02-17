#pragma once

#include "../ArrayTransformer.hpp"

namespace DNDS
{
    template <int a, int b>
    inline constexpr rowsize EigenSize_Mul_RowSize()
    {
        if constexpr (a >= 0 && b >= 0)
        {
            return a * b;
        }
        if constexpr (a == Eigen::Dynamic || b == Eigen::Dynamic)
        {
            return DynamicSize;
        }
        return DNDS_ROWSIZE_MIN;
    }

    template <int _n_row, int _n_col>
    class ArrayEigenUniMatrixBatch : public ParArray<real, NonUniformSize> // use CSR array
    {
        static_assert(_n_row >= 0 || _n_row == Eigen::Dynamic, "invalid _n_row");
        static_assert(_n_col >= 0 || _n_col == Eigen::Dynamic, "invalid _n_col");

    public:
        using t_base = ParArray<real, NonUniformSize>;
        using t_base::t_base;

        using t_EigenMatrix = Eigen::Matrix<real, _n_row, _n_col>;
        using t_EigenMap = Eigen::Map<t_EigenMatrix>; // default no buffer align and stride
        using t_EigenMap_const = Eigen::Map<const t_EigenMatrix>; // default no buffer align and stride

    private:
        int _row_dynamic = _n_row > 0 ? _n_row : 0;
        int _col_dynamic = _n_col > 0 ? _n_col : 0;
        int _m_size = this->Rows() * this->Cols(); //! extra data!

    private:
        using t_base::Resize;
        using t_base::ResizeRow; // privatize basic resizing
        using t_base::operator();
        using t_base::RowSize;
        // void Resize(index, rowsize) = delete;
        // void ResizeRow(index iRow, rowsize nRowSize) = delete;
    public:
        /**
         * @brief resizes all matrices to be used;
         * -1 means no change
         * @param r
         * @param c
         */
        void ResizeMatrix(int r = -1, int c = -1)
        {
            if constexpr (_n_row >= 0)
                DNDS_assert(r == -1 || r == _n_row);
            if constexpr (_n_col >= 0)
                DNDS_assert(c == -1 || c == _n_col);
            if (r >= 0)
                _row_dynamic = r;
            if (c >= 0)
                _col_dynamic = c;
            // TODO: multiplication overflow detect
            _m_size = this->Rows() * this->Cols();
            this->t_base::Resize(0);
        }

        void Resize(index n_size, int r, int c)
        {
            this->ResizeMatrix(r, c);
            this->t_base::Resize(n_size);
        }

        void Resize(index n_size)
        {
            if constexpr (_n_row > 0 && _n_col > 0)
                this->Resize(n_size, -1, -1);
            else
                DNDS_assert_info(false, "invalid call");
        }

    public:
        int Rows() const { return _n_row > 0 ? _n_row : _row_dynamic; }
        int Cols() const { return _n_col > 0 ? _n_col : _col_dynamic; }
        int MSize() const
        {
            if constexpr (_n_row >= 0 && _n_col >= 0)
                return _n_row * _n_col;
            else
                return _m_size;
        }

        void ResizeBatch(index i, rowsize b_size)
        {
            this->t_base::ResizeRow(i, b_size * MSize());
        }

        void ResizeRow(index i, rowsize b_size)
        {
            this->t_base::ResizeRow(i, b_size * MSize());
        }

        rowsize BatchSize(index i) const
        {
            return this->RowSize(i);
        }

        rowsize RowSize(index i) const
        {
            rowsize row_size_c = this->t_base::RowSize(i);
            DNDS_assert(row_size_c % MSize() == 0);
            return row_size_c / MSize();
        }

        auto operator()(index i, rowsize j)
        {
            DNDS_assert(j >= 0 && j < this->RowSize(i));
            // if constexpr (_n_row >= 0 && _n_col >= 0)
            return t_EigenMap(this->t_base::operator[](i) + MSize() * j, Rows(), Cols());
        }

        auto operator()(index i, rowsize j) const
        {
            DNDS_assert(j >= 0 && j < this->RowSize(i));
            // if constexpr (_n_row >= 0 && _n_col >= 0)
            return t_EigenMap_const(this->t_base::operator[](i) + MSize() * j, Rows(), Cols());
        }

        std::vector<t_EigenMap> operator[](index i)
        {
            std::vector<t_EigenMap> ret;
            ret.reserve(this->BatchSize(i));
            for (rowsize j = 0; j < this->BatchSize(i); j++)
                ret.emplace_back(this->t_base::operator[](i) + MSize() * j, Rows(), Cols());
            return ret;
        }

        // TODO: getting sub matrix ?

        static std::string GetDerivedArraySignature()
        {
            return "ArrayEigenUniMatrixBatch__" + std::to_string(_n_row) +
                   "_" + std::to_string(_n_col);
        }

        void WriteSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name, Serializer::ArrayGlobalOffset offset)
        {
            auto cwd = serializerP->GetCurrentPath();
            serializerP->CreatePath(name);
            serializerP->GoToPath(name);

            serializerP->WriteString("DerivedType", this->GetDerivedArraySignature());
            serializerP->WriteInt("row_dynamic", _row_dynamic);
            serializerP->WriteInt("col_dynamic", _col_dynamic);
            serializerP->WriteInt("m_size", _m_size);
            this->t_base::WriteSerializer(serializerP, "array", offset);

            serializerP->GoToPath(cwd);
        }

        void ReadSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name, Serializer::ArrayGlobalOffset &offset)
        {
            auto cwd = serializerP->GetCurrentPath();
            // serializerP->CreatePath(name); //!remember no create path
            serializerP->GoToPath(name);

            std::string readDerivedType;
            serializerP->ReadString("DerivedType", readDerivedType);
            DNDS_assert(readDerivedType == this->GetDerivedArraySignature());
            serializerP->ReadInt("row_dynamic", _row_dynamic);
            serializerP->ReadInt("col_dynamic", _col_dynamic);
            serializerP->ReadInt("m_size", _m_size);
            this->t_base::ReadSerializer(serializerP, "array", offset);

            serializerP->GoToPath(cwd);
        }
    };
}
