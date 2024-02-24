#pragma once

#include "../ArrayTransformer.hpp"

namespace DNDS
{
    template <rowsize _mat_ni, rowsize _mat_nj>
    inline constexpr rowsize __OneMatGetRowSize()
    {
        if constexpr (_mat_ni >= 0 && _mat_nj >= 0)
        {
            return _mat_ni * _mat_nj;
        }
        else if constexpr (_mat_ni == NonUniformSize || _mat_nj == NonUniformSize)
        {
            return NonUniformSize;
        }
        else
        {
            return DynamicSize;
        }
    }

    template <rowsize _mat_ni = 1, rowsize _mat_nj = 1,
              rowsize _mat_ni_max = _mat_ni, rowsize _mat_nj_max = _mat_nj, rowsize _align = NoAlign>
    class ArrayEigenMatrix : public ParArray<real,
                                             __OneMatGetRowSize<_mat_ni, _mat_nj>(),
                                             __OneMatGetRowSize<_mat_ni_max, _mat_nj_max>(),
                                             _align>
    {
    public:
        static const rowsize _row_size = __OneMatGetRowSize<_mat_ni, _mat_nj>();
        static const rowsize _row_size_max = __OneMatGetRowSize<_mat_ni_max, _mat_nj_max>();
        using t_base = ParArray<real,
                                __OneMatGetRowSize<_mat_ni, _mat_nj>(),
                                __OneMatGetRowSize<_mat_ni_max, _mat_nj_max>(),
                                _align>;
        using t_base::t_base;
        using t_pRowSizes = typename t_base::t_pRowSizes;

        using t_EigenMatrix = Eigen::Matrix<real, RowSize_To_EigenSize(_mat_ni), RowSize_To_EigenSize(_mat_nj)>;
        using t_EigenMap = Eigen::Map<t_EigenMatrix>; // default no buffer align and stride

        using t_copy = t_EigenMatrix;

    private:
        using t_base::Resize;
        using t_base::ResizeRow;
        using t_base::operator();
        t_pRowSizes _mat_nRows;        //! extra data
        rowsize _mat_nRow_dynamic = 0; //! extra data

    public:
        void Resize(index nSize, rowsize nSizeRowDynamic, rowsize nSizeColDynamic)
        {
            if constexpr (_mat_ni >= 0)
                DNDS_assert(nSizeRowDynamic == _mat_ni);
            if constexpr (_mat_nj >= 0)
                DNDS_assert(nSizeColDynamic == _mat_nj);
            if constexpr (_mat_ni_max >= 0)
                DNDS_assert(nSizeRowDynamic <= _mat_ni_max);
            if constexpr (_mat_nj_max >= 0)
                DNDS_assert(nSizeColDynamic <= _mat_nj_max);

            if constexpr (_mat_ni == NonUniformSize)
                DNDS_MAKE_SSP(_mat_nRows, nSize);
            else if constexpr (_mat_ni == DynamicSize)
                _mat_nRow_dynamic = nSizeRowDynamic;

            this->t_base::Resize(nSize, nSizeRowDynamic * nSizeColDynamic);
        }

        void ResizeMat(index iMat, rowsize nSizeRow, rowsize nSizeCol)
        {
            this->ResizeRow(iMat, nSizeRow, nSizeCol);
        }

        void ResizeRow(index iMat, rowsize nSizeRow, rowsize nSizeCol)
        {
            if constexpr (_mat_ni == NonUniformSize)
                this->t_base::ResizeRow(iMat, nSizeRow * nSizeCol), (*_mat_nRows)[iMat] = nSizeRow;
            else if constexpr (_mat_ni == DynamicSize)
                DNDS_assert_info(false, "Invalid call");
        }

        void operator()(index i, rowsize j)
        {
            // just don't call
        }

        t_EigenMap
        operator[](index i)
        {
            rowsize c_nRow;
            if constexpr (_mat_ni == NonUniformSize)
                c_nRow = (*_mat_nRows)[i];
            else if constexpr (_mat_ni == DynamicSize)
                c_nRow = _mat_nRow_dynamic;
            else
                c_nRow = _mat_ni;

            return t_EigenMap(t_base::operator[](i), c_nRow, t_base::RowSize(i) / c_nRow); // need static dispatch?
        }

        static std::string GetDerivedArraySignature()
        {
            return "ArrayEigenMatrix__" + std::to_string(_mat_ni) +
                   "_" + std::to_string(_mat_nj) +
                   "_" + std::to_string(_mat_ni_max) +
                   "_" + std::to_string(_mat_nj_max);
        }

        void WriteSerializer(SerializerBase *serializer, const std::string &name)
        {
            auto cwd = serializer->GetCurrentPath();
            serializer->CreatePath(name);
            serializer->GoToPath(name);

            serializer->WriteString("DerivedType", this->GetDerivedArraySignature());
            serializer->WriteInt("mat_nRow_dynamic", _mat_nRow_dynamic);
            if constexpr (_mat_ni == NonUniformSize)
                serializer->WriteSharedRowsizeVector("mat_nRows", _mat_nRows);
            this->t_base::WriteSerializer(serializer, "array");

            serializer->GoToPath(cwd);
        }

        void ReadSerializer(SerializerBase *serializer, const std::string &name)
        {
            auto cwd = serializer->GetCurrentPath();
            // serializer->CreatePath(name); //!remember no create path
            serializer->GoToPath(name);

            std::string readDerivedType;
            serializer->ReadString("DerivedType", readDerivedType);
            DNDS_assert(readDerivedType == this->GetDerivedArraySignature());
            serializer->ReadInt("mat_nRow_dynamic", _mat_nRow_dynamic);
            if constexpr (_mat_ni == NonUniformSize)
                serializer->ReadSharedRowsizeVector("mat_nRows", _mat_nRows);
            this->t_base::ReadSerializer(serializer, "array");

            serializer->GoToPath(cwd);
        }
    };
}

