#pragma once

#include "../ArrayTransformer.hpp"
#include "ArrayEigenMatrix_DeviceView.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"

namespace DNDS
{

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
        using t_self = ArrayEigenMatrix<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>;
        using t_base = ParArray<real,
                                __OneMatGetRowSize<_mat_ni, _mat_nj>(),
                                __OneMatGetRowSize<_mat_ni_max, _mat_nj_max>(),
                                _align>;
        using t_base::t_base;
        // using t_pRowSizes = typename t_base::t_pRowSizes;
        using t_pRowSizes = ssp<host_device_vector<rowsize>>;

        using t_EigenMatrix = Eigen::Matrix<real, RowSize_To_EigenSize(_mat_ni), RowSize_To_EigenSize(_mat_nj)>;
        using t_EigenMap = Eigen::Map<t_EigenMatrix, Eigen::Unaligned>;             // default no buffer align and stride
        using t_EigenMap_const = Eigen::Map<const t_EigenMatrix, Eigen::Unaligned>; // default no buffer align and stride

        using t_copy = t_EigenMatrix;

    private:
        using t_base::Resize;
        using t_base::ResizeRow;
        using t_base::operator();
        t_pRowSizes _mat_nRows;        //! extra data
        rowsize _mat_nRow_dynamic = 0; //! extra data

    public:
        size_t FullSizeBytes() const
        {
            size_t b = this->t_base::FullSizeBytes();
            if (_mat_nRows)
                b += _mat_nRows->size() * sizeof(rowsize);
            return b;
        }

        // default copy
        ArrayEigenMatrix(const t_self &R) = default;
        t_self &operator=(const t_self &R) = default;
        // operator= handled automatically

        void clone(const t_self &R)
        {
            this->operator=(R);
        }

        void Resize(index nSize, rowsize nSizeRowDynamic, rowsize nSizeColDynamic)
        {
            if constexpr (_mat_ni >= 0)
                DNDS_check_throw(nSizeRowDynamic == _mat_ni);
            if constexpr (_mat_nj >= 0)
                DNDS_check_throw(nSizeColDynamic == _mat_nj);
            if constexpr (_mat_ni_max >= 0)
                DNDS_check_throw(nSizeRowDynamic <= _mat_ni_max);
            if constexpr (_mat_nj_max >= 0)
                DNDS_check_throw(nSizeColDynamic <= _mat_nj_max);

            if constexpr (_mat_ni == NonUniformSize)
                DNDS_MAKE_SSP(_mat_nRows, nSize, nSizeRowDynamic);
            else if constexpr (_mat_ni == DynamicSize)
                _mat_nRow_dynamic = nSizeRowDynamic;

            this->t_base::Resize(nSize, nSizeRowDynamic * nSizeColDynamic);
        }

        [[nodiscard]] rowsize MatRowSize(index iMat = 0) const
        {
            if constexpr (_mat_ni >= 0)
                return _mat_ni;
            if constexpr (_mat_ni == NonUniformSize)
                return _mat_nRows->at(iMat);
            if constexpr (_mat_ni == DynamicSize)
                return _mat_nRow_dynamic;
            return UnInitRowsize; // invalid branch
        }

        [[nodiscard]] rowsize MatColSize(index iMat = 0) const
        {
            if constexpr (_mat_nj >= 0)
                return _mat_nj;
            if constexpr (_mat_nj == NonUniformSize)
                return this->t_base::RowSize(iMat) / this->MatRowSize(iMat);
            if constexpr (_mat_nj == DynamicSize)
                return this->t_base::RowSize(iMat) / this->MatRowSize(iMat);
            return UnInitRowsize; // invalid branch
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
                DNDS_check_throw_info(false, "Invalid call");
        }

        std::conditional_t<_mat_ni == 1 && _mat_nj == 1,
                           real &, void>
        operator()(index iRow)
        {
            if constexpr (_mat_ni == 1 && _mat_nj == 1)
                return *t_base::operator[](iRow);
        }

        std::conditional_t<_mat_ni == 1 && _mat_nj == 1,
                           real, void>
        operator()(index iRow) const
        {
            if constexpr (_mat_ni == 1 && _mat_nj == 1)
                return *t_base::operator[](iRow);
        }

        std::conditional_t<_mat_ni == 1 || _mat_nj == 1,
                           real &, void>
        operator()(index iRow, rowsize j)
        {
            if constexpr (_mat_ni == 1 || _mat_nj == 1)
                return t_base::operator()(iRow, j);
        }

        std::conditional_t<_mat_ni == 1 || _mat_nj == 1,
                           real, void>
        operator()(index iRow, rowsize j) const
        {
            if constexpr (_mat_ni == 1 || _mat_nj == 1)
                return t_base::operator()(iRow, j);
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

        t_EigenMap_const
        operator[](index i) const
        {
            rowsize c_nRow;
            if constexpr (_mat_ni == NonUniformSize)
                c_nRow = (*_mat_nRows)[i];
            else if constexpr (_mat_ni == DynamicSize)
                c_nRow = _mat_nRow_dynamic;
            else
                c_nRow = _mat_ni;
            // std::cout << c_nRow << "  " << t_base::RowSize(i) << std::endl;

            return t_EigenMap_const(t_base::operator[](i), c_nRow, t_base::RowSize(i) / c_nRow); // need static dispatch?
        }

        static std::string GetDerivedArraySignature()
        {
            std::array<char, 1024> buf;
            std::sprintf(buf.data(), "ArrayEigenMatrix__%d_%d_%d_%d", _mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max);
            return buf.data();
        }

        static std::tuple<int, int, int, int> GetDerivedArraySignatureInts(const std::string &v)
        { // TODO: check here!
            auto strings = splitSStringClean(v, '_');
            DNDS_check_throw(strings.size() == 5 || strings.size() == 6);
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

        /**
         * @brief
         *
         * @param serializerP
         * @param name
         * @warning need to take care of createGlobalMapping if resized
         */
        void WriteSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name, Serializer::ArrayGlobalOffset offset)
        {
            auto cwd = serializerP->GetCurrentPath();
            serializerP->CreatePath(name);
            serializerP->GoToPath(name);

            this->t_base::WriteSerializer(serializerP, "array", offset);
            serializerP->WriteString("DerivedType", GetDerivedArraySignature());
            serializerP->WriteInt("mat_nRow_dynamic", _mat_nRow_dynamic);
            if constexpr (_mat_ni == NonUniformSize)
                serializerP->WriteSharedRowsizeVector("mat_nRows", _mat_nRows, offset);

            serializerP->GoToPath(cwd);
        }

        void ReadSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name, Serializer::ArrayGlobalOffset &offset)
        {
            auto cwd = serializerP->GetCurrentPath();
            // serializerP->CreatePath(name); //!remember no create path
            serializerP->GoToPath(name);

            this->t_base::ReadSerializer(serializerP, "array", offset);

            std::string readDerivedType;
            serializerP->ReadString("DerivedType", readDerivedType);
            auto [v_mat_ni, v_mat_nj, v_mat_ni_max, v_mat_nj_max] = GetDerivedArraySignatureInts(readDerivedType);
            DNDS_check_throw_info(readDerivedType == this->GetDerivedArraySignature() || SignatureIsCompatible(readDerivedType),
                                  readDerivedType + ", i am: " + this->GetDerivedArraySignature() + fmt::format(" {} {} {} {}", v_mat_ni, v_mat_nj, v_mat_ni_max, v_mat_nj_max));
            serializerP->ReadInt("mat_nRow_dynamic", _mat_nRow_dynamic);
            if (_mat_ni == DynamicSize && v_mat_ni >= 0)
                _mat_nRow_dynamic = v_mat_ni;
            if (v_mat_ni == NonUniformSize) // TODO: complete here!
            {

                if constexpr (_mat_ni == NonUniformSize)
                    serializerP->ReadSharedRowsizeVector("mat_nRows", _mat_nRows, offset); // TODO: multiple write to offset, check?
                else
                {
                    t_pRowSizes v_mat_nRows;
                    serializerP->ReadSharedRowsizeVector("mat_nRows", v_mat_nRows, offset);
                    int c_mat_nRow_dynamic = v_mat_nRows->size() ? 0 : v_mat_nRows->at(0);
                    for (auto i = 0; i < v_mat_nRows->size(); ++i)
                        DNDS_check_throw(v_mat_nRows->operator[](i) == c_mat_nRow_dynamic);
                    _mat_nRow_dynamic = c_mat_nRow_dynamic;
                }
            }
            else // TODO: complete here!
            {
                if constexpr (_mat_ni == NonUniformSize)
                    DNDS_MAKE_SSP(_mat_nRows, this->Size(), v_mat_ni >= 0 ? v_mat_ni : _mat_nRow_dynamic);
            }

            serializerP->GoToPath(cwd);
        }

        template <DeviceBackend B>
        using t_deviceView = ArrayEigenMatrixDeviceView<B, real, _mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>;
        template <DeviceBackend B>
        using t_deviceViewConst = ArrayEigenMatrixDeviceView<B, const real, _mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>;

        template <DeviceBackend B>
        auto deviceView()
        {
            auto base_view = t_base::template deviceView<B>();
            return t_deviceView<B>(base_view,
                                   // do more delicate dispatching for extensions?
                                   _mat_nRows ? (B == DeviceBackend::Host ? _mat_nRows->data() : _mat_nRows->dataDevice())
                                              : nullptr,
                                   _mat_nRow_dynamic);
        }

        template <DeviceBackend B>
        auto deviceView() const
        {
            auto base_view = t_base::template deviceView<B>();
            return t_deviceViewConst<B>(base_view,
                                        // do more delicate dispatching for extensions?
                                        _mat_nRows ? (B == DeviceBackend::Host ? _mat_nRows->data() : _mat_nRows->dataDevice())
                                                   : nullptr,
                                        _mat_nRow_dynamic);
        }

        using t_base::to_host; // we only copy primary data back (no structure modification allowed from device)

        void to_device(DeviceBackend backend)
        {
            this->t_base::to_device(backend);
            if (_mat_nRows)
                _mat_nRows->to_device(backend);
        }

        template <DeviceBackend B, bool is_const = false>
        class iterator : public ArrayIteratorBase<iterator<B>>
        {
        public:
            using view_type = std::conditional_t<is_const, t_deviceViewConst<B>, t_deviceView<B>>;
            using t_base_iter = ArrayIteratorBase<iterator<B>>;
            using typename t_base_iter::difference_type;
            using typename t_base_iter::pointer;

            using value_type = typename view_type::t_EigenView;
            using reference = typename view_type::t_EigenView;
            static_assert(std::is_signed_v<typename iterator::difference_type>);

        protected:
            view_type view;

        public:
            DNDS_DEVICE_CALLABLE auto getView() const { return view; }

            DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(iterator, iterator)

            DNDS_DEVICE_CALLABLE iterator(const view_type &n_view, index n_iRow) : t_base_iter(n_iRow), view(n_view)
            {
                DNDS_HD_assert(this->iRow >= -1 && this->iRow <= getView().Size());
            }

            DNDS_DEVICE_CALLABLE reference operator*() { return view.MatView(this->iRow); }

            DNDS_DEVICE_CALLABLE reference operator*() const { return const_cast<iterator *>(this)->view.MatView(this->iRow); }

            // DNDS_DEVICE_CALLABLE reference operator[](index n) { return view.MatView(this->iRow + n); }

            // DNDS_DEVICE_CALLABLE reference operator[](index n) const { return const_cast<iterator *>(this)->view.MatView(this->iRow + n); }
            std::string to_string()
            {
                return fmt::format("ArrayEigenMatrix::iterator<> iRow[{}] Size[{}]", this->iRow, this->view.Size());
            }
        };

        template <DeviceBackend B>
        iterator<B> begin()
        {
            return iterator<B>{deviceView<B>(), 0};
        }

        template <DeviceBackend B>
        iterator<B> end()
        {
            return iterator<B>{deviceView<B>(), this->Size()};
        }
    };
}
