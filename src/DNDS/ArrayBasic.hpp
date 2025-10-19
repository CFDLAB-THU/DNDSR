#pragma once

#include "Defines.hpp"
#include "Errors.hpp"

namespace DNDS
{
    static const rowsize NoAlign = -1;

    enum DataLayout
    {
        ErrorLayout,
        TABLE_StaticFixed,
        TABLE_Fixed,
        TABLE_Max,
        TABLE_StaticMax,
        CSR,
    };

    constexpr bool isTABLE(DataLayout lo)
    {
        return lo == TABLE_StaticFixed || lo == TABLE_Fixed || lo == TABLE_Max || lo == TABLE_StaticMax;
    }
    constexpr bool isTABLE_Fixed(DataLayout lo)
    {
        return lo == TABLE_StaticFixed || lo == TABLE_Fixed;
    }
    constexpr bool isTABLE_Max(DataLayout lo)
    {
        return lo == TABLE_Max || lo == TABLE_StaticMax;
    }
    constexpr bool isTABLE_Static(DataLayout lo)
    {
        return lo == TABLE_StaticFixed || lo == TABLE_StaticMax;
    }
    constexpr bool isTABLE_Dynamic(DataLayout lo)
    {
        return lo == TABLE_Fixed || lo == TABLE_Max;
    }

    template <class T>
    constexpr bool array_comp_acceptable()
    {
        return std::is_trivially_copyable_v<T> || Meta::is_fixed_data_real_eigen_matrix_v<T>;
    }

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayLayout
    {
    public:
        using value_type = T;
        static const rowsize al = _align;
        static const rowsize rs = _row_size;
        static const rowsize rm = _row_max;
        static const size_t sizeof_T = sizeof(value_type);

        static_assert(sizeof_T <= (1024ULL * 1024ULL * 1024ULL), "Row size larger than 1G");
        static_assert(array_comp_acceptable<T>(), "Do not put in a non trivially copyable type ");

        static_assert(al == NoAlign || al >= 1, "Align bad");

        static const rowsize s_T = al == NoAlign ? sizeof_T : (sizeof_T / al + 1) * al;
        static_assert(s_T >= sizeof_T && s_T - sizeof_T < (al == NoAlign ? 1 : al), "I1");

        static constexpr DataLayout _GetDataLayout()
        {
            if constexpr (rs != DynamicSize && rs != NonUniformSize && rs >= 0)
                return TABLE_StaticFixed;
            else if constexpr (rs == DynamicSize)
                return TABLE_Fixed;
            else if constexpr (rs == NonUniformSize)
            {
                if constexpr (rm == NonUniformSize)
                    return CSR;
                else if constexpr (rm == DynamicSize)
                    return TABLE_Max;
                else if constexpr (rm >= 0)
                    return TABLE_StaticMax;
                else
                    return ErrorLayout;
            }
            else
                return ErrorLayout;
        }
        static const DataLayout _dataLayout = _GetDataLayout();
        static_assert(_dataLayout != ErrorLayout, "Layout Error");
        static const bool isCSR = _dataLayout == CSR;

        static std::string GetArrayName()
        {
            std::string Layout;
            if constexpr (_dataLayout == CSR)
                Layout = "CSR";
            if constexpr (_dataLayout == TABLE_StaticFixed)
                Layout = "TABLE_StaticFixed";
            if constexpr (_dataLayout == TABLE_Fixed)
                Layout = "TABLE_Fixed";
            if constexpr (_dataLayout == TABLE_StaticMax)
                Layout = "TABLE_StaticMax";
            if constexpr (_dataLayout == TABLE_Max)
                Layout = "TABLE_Max";
            return Layout + "__" +
                   typeid(T).name() + "_" + std::to_string(sizeof_T) + "_" + RowSize_To_PySnippet(_row_size) +
                   "_" + RowSize_To_PySnippet(_row_max) +
                   "_" + RowSize_To_PySnippet(_align);
        }

        static std::string GetArraySignature()
        {
            std::string Layout;
            if constexpr (_dataLayout == CSR)
                Layout = "CSR";
            if constexpr (_dataLayout == TABLE_StaticFixed)
                Layout = "TABLE_StaticFixed";
            if constexpr (_dataLayout == TABLE_Fixed)
                Layout = "TABLE_Fixed";
            if constexpr (_dataLayout == TABLE_StaticMax)
                Layout = "TABLE_StaticMax";
            if constexpr (_dataLayout == TABLE_Max)
                Layout = "TABLE_Max";
            return Layout + "__" + std::to_string(sizeof_T) + "_" + std::to_string(_row_size) +
                   "_" + std::to_string(_row_max) +
                   "_" + std::to_string(_align);
        }

        static std::string GetArraySignatureRelaxed()
        {
            std::string Layout;
            if constexpr (_dataLayout == CSR)
                Layout = "CSR";
            if constexpr (_dataLayout == TABLE_StaticFixed)
                Layout = "TABLE_StaticFixed";
            if constexpr (_dataLayout == TABLE_Fixed)
                Layout = "TABLE_Fixed";
            if constexpr (_dataLayout == TABLE_StaticMax)
                Layout = "TABLE_StaticMax";
            if constexpr (_dataLayout == TABLE_Max)
                Layout = "TABLE_Max";
            return Layout + "__" + std::to_string(sizeof_T) + "_" + std::to_string(-1) +
                   "_" + std::to_string(-1) +
                   "_" + std::to_string(_align);
        }

        static std::tuple<int, int, int, int> ParseArraySignatureTuple(const std::string &v)
        {
            auto strings = splitSStringClean(v, '_');
            DNDS_HD_assert(strings.size() == 5 || strings.size() == 6);
            auto sz = strings.size();
            return std::make_tuple(std::stoi(strings[sz - 4]), std::stoi(strings[sz - 3]), std::stoi(strings[sz - 2]), std::stoi(strings[sz - 1]));
        }

        static bool ArraySignatureIsCompatible(const std::string &v)
        {
            auto [sz, rs, rm, align] = ParseArraySignatureTuple(v);
            if (sz != sizeof_T)
                return false;
            if (rs >= 0 && _row_size >= 0 && rs != _row_size)
                return false;
            if (rm >= 0 && _row_max >= 0 && rm != _row_max)
                return false;
            return true;
        }
    };

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayView : public ArrayLayout<T, _row_size, _row_max, _align>
    {
    protected:
        using self_type = ArrayView<T, _row_size, _row_max, _align>;
        using t_Layout = ArrayLayout<T, _row_size, _row_max, _align>;
        using t_Layout::al,
            t_Layout::rs,
            t_Layout::rm,
            t_Layout::sizeof_T,
            t_Layout::s_T,
            t_Layout::_GetDataLayout,
            t_Layout::_dataLayout,
            t_Layout::isCSR;
        using t_Layout::GetArrayName,
            t_Layout::GetArraySignature,
            t_Layout::GetArraySignatureRelaxed,
            t_Layout::ParseArraySignatureTuple,
            t_Layout::ArraySignatureIsCompatible;
        using typename t_Layout::value_type;

        index _size;
        T *_data = nullptr;
        index _data_size = 0;
        index *_rowstart = nullptr;
        index _rowstart_size = 0;
        rowsize *_rowsizes = nullptr;
        index _rowsizes_size = 0;

        index _row_size_dynamic = 0;

        bool _isCompressed = true;

        using t_dataUncompressed = std::vector<std::vector<T>>;
        t_dataUncompressed *_p_dataUncompressed;

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ArrayView, self_type)

        DNDS_DEVICE_CALLABLE ArrayView(index n_size, T *n_data, index n_data_size,
                                       index *n_rowstart, index n_rowstart_size,
                                       rowsize *n_rowsizes, index n_rowsizes_size,
                                       index n_row_size_dynamic,
                                       bool n_isCompressed, t_dataUncompressed *n_p_dataUncompressed)
            : _size(n_size),
              _data(n_data), _data_size(n_data_size),
              _rowstart(n_rowstart), _rowstart_size(n_rowstart_size),
              _rowsizes(n_rowsizes), _rowsizes_size(n_rowsizes_size),
              _row_size_dynamic(n_row_size_dynamic),
              _isCompressed(n_isCompressed), _p_dataUncompressed(n_p_dataUncompressed)
        {
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] index Size() const { return _size; }

        // to be device-callable
        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize RowSize() const
        {
            if constexpr (_dataLayout == TABLE_Fixed)
                return _row_size_dynamic;
            else if constexpr (_dataLayout == TABLE_StaticFixed)
                return rs;
            else
            {
                DNDS_HD_assert_infof(false, "invalid call");
                return -1;
            }
        }

    protected:
        // to be device-callable
        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize RowSize_Compressed(index iRow) const
        {
            DNDS_HD_assert(_isCompressed);
            if constexpr (_dataLayout == TABLE_Fixed)
                return _row_size_dynamic;
            else if constexpr (_dataLayout == TABLE_StaticFixed)
                return rs;
            DNDS_HD_assert_infof(iRow < _size && iRow >= 0, "query position out of range");
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax) // TABLE with Max
            {
                DNDS_HD_assert_infof(_rowsizes, "_rowsizes invalid"); // TABLE with Max must have a RowSizes
                DNDS_HD_assert_infof(iRow >= 0 && iRow < _rowsizes_size,
                                     "iRow invalid: %lld / %lld", iRow, _rowsizes_size);
                return _rowsizes[iRow]; //! unsafe
            }
            else if constexpr (_dataLayout == CSR)
            {
                DNDS_HD_assert_infof(_rowstart, "_rowstart invalid");
                DNDS_HD_assert_infof(iRow >= 0 && iRow + 1 < _rowstart_size,
                                     "iRow invalid: %lld / %lld", iRow, _rowstart_size);
                index pDiff = _rowstart[iRow + 1] - _rowstart[iRow]; //! unsafe
                DNDS_HD_assert(pDiff < INT32_MAX);                   // overflow
                return static_cast<rowsize>(pDiff);
            }
        }

    public:
        // to be device-callable
        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize RowSize(index iRow) const
        {
            if constexpr (_dataLayout == CSR)
            {
                if (_isCompressed)
                    return this->RowSize_Compressed(iRow);
                else
                {
                    auto rs_cur = (*_p_dataUncompressed).at(iRow).size(); //! unsafe
                    return static_cast<rowsize>(rs_cur);
                }
            }
            else
                return this->RowSize_Compressed(iRow);
        }

        // to be device-callable
        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize RowSizeMax() const
        {
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
                return _dataLayout == TABLE_Max ? _row_size_dynamic : rm;
            else
                DNDS_HD_assert_infof(false, "invalid call");
        }

        // to be device-callable
        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize RowSizeField() const
        {
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
                return this->RowSizeMax();
            else if constexpr (_dataLayout == TABLE_Fixed || _dataLayout == TABLE_StaticFixed)
                return this->RowSize();
            else
                DNDS_HD_assert_infof(false, "invalid call");
        }

        // to be device-callable
        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize RowSizeField(index iRow) const
        {
            if constexpr (_dataLayout == CSR)
                return this->RowSize(iRow);
            else
                DNDS_HD_assert_infof(false, "invalid call");
        }

    protected:
        DNDS_DEVICE_CALLABLE const T &at_compressed(index iRow, rowsize iCol) const
        {
            DNDS_HD_assert(_isCompressed);
            DNDS_HD_assert_infof(iRow < _size && iRow >= 0,
                                 "query position i[%lld] out of range [0, %lld)",
                                 iRow, _size);
            DNDS_HD_assert_infof(iCol < RowSize(iRow) && iCol >= 0,
                                 "query position j[%lld] out of range [0, %lld)",
                                 iCol, RowSize(iRow));
            index pos = -1;
            if constexpr (_dataLayout == TABLE_StaticFixed)
                pos = iRow * rs + iCol;
            else if constexpr (_dataLayout == TABLE_StaticMax)
                pos = iRow * rm + iCol;
            else if constexpr (_dataLayout == TABLE_Fixed)
                pos = iRow * _row_size_dynamic + iCol;
            else if constexpr (_dataLayout == TABLE_Max)
                pos = iRow * _row_size_dynamic + iCol;
            else if constexpr (_dataLayout == CSR)
            {
                DNDS_HD_assert_infof(0 <= iRow && iRow + 1 < _rowstart_size,
                                     "iRow invalid, %lld / %lld", iRow, _rowstart_size);
                pos = _rowstart[iRow] + iCol; //! unsafe
            }
            else
                DNDS_HD_assert_infof(false, "invalid call");
            DNDS_HD_assert_infof(0 <= pos && pos < _data_size,
                                 "pos in data not valid, %lld / %lld", pos, _data_size);
            return _data[pos]; //! unsafe
        }

    public:
        // not device callable
        const T &at(index iRow, rowsize iCol) const
        {
            if constexpr (_dataLayout == CSR)
            {
                if (_isCompressed)
                    return this->at_compressed(iRow, iCol);
                else
                    return (*_p_dataUncompressed).at(iRow).at(iCol); //! unsafe
            }
            else
                return this->at_compressed(iRow, iCol);
        }

        T &operator()(index iRow, rowsize iCol = 0)
        {
            return const_cast<T &>(at(iRow, iCol));
        }

        const T &operator()(index iRow, rowsize iCol = 0) const
        {
            return at(iRow, iCol);
        }

    protected:
        DNDS_DEVICE_CALLABLE T *get_rowstart_pointer_compressed(index iRow)
        {
            DNDS_HD_assert(_isCompressed);
            DNDS_HD_assert_infof(iRow <= _size && iRow >= 0, "query position i out of range");
            if constexpr (_dataLayout == TABLE_StaticFixed)
                return _data + iRow * rs;
            else if constexpr (_dataLayout == TABLE_StaticMax)
                return _data + iRow * rm;
            else if constexpr (_dataLayout == TABLE_Fixed)
                return _data + iRow * _row_size_dynamic;
            else if constexpr (_dataLayout == TABLE_Max)
                return _data + iRow * _row_size_dynamic;
            else if constexpr (_dataLayout == CSR)
            {
                DNDS_HD_assert_infof(0 <= iRow && iRow < _rowstart_size,
                                     "iRow invalid, %lld / %lld", iRow, _rowstart_size);
                return _data + _rowstart[iRow]; //! unsafe
            }
            else
            {
                DNDS_assert_info(false, "invalid call");
                return nullptr;
            }
        }

    public:
        /**
         * @brief iRow could be past-the-end to query past-the-end position pointer
         *
         * @param iRow
         * @return T*
         */
        T *operator[](index iRow)
        {
            if constexpr (_dataLayout == CSR)
            {
                if (_isCompressed)
                    return this->get_rowstart_pointer_compressed(iRow);
                else if (_size == 0)
                {
                    static_assert(((T *)(NULL) - (T *)(NULL)) == 0);
                    return (T *)(NULL); // used for past-the-end inquiry of size 0 array
                }
                else
                {
                    DNDS_HD_assert_infof(iRow < _size, "past-the-end query forbidden for CSR uncompressed");
                    return (*_p_dataUncompressed).at(iRow).data(); //! unsafe
                }
            }
            else
                return this->get_rowstart_pointer_compressed(iRow);
        }

        const T *operator[](index iRow) const
        {
            return static_cast<const T *>(const_cast<self_type *>(this)->operator[](iRow));
        }
    };
}