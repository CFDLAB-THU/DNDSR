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
                DNDS_MAKE_SSP(_mat_nRows, nSize, nSizeRowDynamic);
            else if constexpr (_mat_ni == DynamicSize)
                _mat_nRow_dynamic = nSizeRowDynamic;

            this->t_base::Resize(nSize, nSizeRowDynamic * nSizeColDynamic);
        }

        rowsize MatRowSize(index iMat = 0)
        {
            if constexpr (_mat_ni >= 0)
                return _mat_ni;
            if constexpr (_mat_ni == NonUniformSize)
                return _mat_nRows->at(iMat);
            if constexpr (_mat_ni == DynamicSize)
                return _mat_nRow_dynamic;
        }

        rowsize MatColSize(index iMat = 0)
        {
            if constexpr (_mat_nj >= 0)
                return _mat_nj;
            if constexpr (_mat_nj == NonUniformSize)
                return this->t_base::RowSize(iMat) / this->MatRowSize(iMat);
            if constexpr (_mat_ni == DynamicSize)
                return this->t_base::RowSize(iMat) / this->MatRowSize(iMat);
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
            // std::cout << c_nRow << "  " << t_base::RowSize(i) << std::endl;

            return t_EigenMap(t_base::operator[](i), c_nRow, t_base::RowSize(i) / c_nRow); // need static dispatch?
        }

        static std::string GetDerivedArraySignature()
        {
            char buf[1024];
            std::sprintf(buf, "ArrayEigenMatrix__%d_%d_%d_%d", _mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max);
            return buf;
        }

        static std::tuple<int, int, int, int> GetDerivedArraySignatureInts(const std::string &v)
        { // TODO: check here!
            auto strings = splitSStringClean(v, '_');
            DNDS_assert(strings.size() == 5 || strings.size() == 6);
            auto sz = strings.size();
            return std::make_tuple(std::stoi(strings[sz - 4]), std::stoi(strings[sz - 3]), std::stoi(strings[sz - 2]), std::stoi(strings[sz - 1]));
        }

        bool SignatureIsCompatible(const std::string &v)
        { // TODO: check here!
            auto [v_mat_ni, v_mat_nj, v_mat_ni_max, v_mat_nj_max] = GetDerivedArraySignatureInts(v);
            // std::cout << fmt::format(" {} {} {} {}", v_mat_ni, v_mat_nj, v_mat_ni_max, v_mat_nj_max) << std::endl;
            // std::cout << fmt::format(" {} {} {} {}", _mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max) << std::endl;
            if (v_mat_ni >= 0 && _mat_ni >= 0 && v_mat_ni != _mat_ni)
                return false;
            if (v_mat_nj >= 0 && _mat_nj >= 0 && v_mat_nj != _mat_nj)
                return false;
            if (v_mat_ni_max >= 0 && _mat_ni_max >= 0 && v_mat_ni_max != _mat_ni_max)
                return false;
            if (v_mat_nj_max >= 0 && _mat_nj_max >= 0 && v_mat_nj_max != _mat_nj_max)
                return false;
            return true;
        }

        void WriteSerializer(SerializerBase *serializer, const std::string &name)
        {
            auto cwd = serializer->GetCurrentPath();
            serializer->CreatePath(name);
            serializer->GoToPath(name);

            this->t_base::WriteSerializer(serializer, "array");
            serializer->WriteString("DerivedType", this->GetDerivedArraySignature());
            serializer->WriteInt("mat_nRow_dynamic", _mat_nRow_dynamic);
            if constexpr (_mat_ni == NonUniformSize)
                serializer->WriteSharedRowsizeVector("mat_nRows", _mat_nRows);

            serializer->GoToPath(cwd);
        }

        void ReadSerializer(SerializerBase *serializer, const std::string &name)
        {
            auto cwd = serializer->GetCurrentPath();
            // serializer->CreatePath(name); //!remember no create path
            serializer->GoToPath(name);

            this->t_base::ReadSerializer(serializer, "array");

            std::string readDerivedType;
            serializer->ReadString("DerivedType", readDerivedType);
            auto [v_mat_ni, v_mat_nj, v_mat_ni_max, v_mat_nj_max] = GetDerivedArraySignatureInts(readDerivedType);
            DNDS_assert_info(readDerivedType == this->GetDerivedArraySignature() || SignatureIsCompatible(readDerivedType),
                             readDerivedType + ", i am: " + this->GetDerivedArraySignature() + fmt::format(" {} {} {} {}", v_mat_ni, v_mat_nj, v_mat_ni_max, v_mat_nj_max));
            serializer->ReadInt("mat_nRow_dynamic", _mat_nRow_dynamic);
            if (_mat_ni == DynamicSize && v_mat_ni >= 0)
                _mat_nRow_dynamic = v_mat_ni;
            if (v_mat_ni == NonUniformSize) // TODO: complete here!
            {
                if constexpr (_mat_ni == NonUniformSize)
                    serializer->ReadSharedRowsizeVector("mat_nRows", _mat_nRows);
                else
                {
                    t_pRowSizes v_mat_nRows;
                    serializer->ReadSharedRowsizeVector("mat_nRows", v_mat_nRows);
                    int c_mat_nRow_dynamic = v_mat_nRows->size() ? 0 : v_mat_nRows->at(0);
                    for (auto i = 0; i < v_mat_nRows->size(); ++i)
                        DNDS_assert(v_mat_nRows->operator[](i) == c_mat_nRow_dynamic);
                    _mat_nRow_dynamic = c_mat_nRow_dynamic;
                }
            }
            else // TODO: complete here!
            {
                if constexpr (_mat_ni == NonUniformSize)
                    DNDS_MAKE_SSP(_mat_nRows, this->Size(), v_mat_ni >= 0 ? v_mat_ni : _mat_nRow_dynamic);
            }

            serializer->GoToPath(cwd);
        }
    };
}
