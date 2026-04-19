#pragma once
/// @file Array.hpp
/// @brief Core 2D variable-length array container with five data layouts.
/// @par Unit Test Coverage (test_Array.cpp)
/// - All 5 layouts: TABLE_StaticFixed, TABLE_Fixed, TABLE_StaticMax, TABLE_Max, CSR
/// - Resize, ResizeRow, Size, RowSize, RowSizeMax, GetDataLayout
/// - Element access: operator(), operator[]
/// - DataSize, DataSizeBytes
/// - Compress / Decompress round-trip (CSR)
/// - clone, CopyData, copy constructor, SwapData
/// - WriteSerializer / ReadSerializer (JSON, 3 layouts)
/// - GetArraySignature, ParseArraySignatureTuple, ArraySignatureIsCompatible
/// - hash
/// @par Not Yet Tested
/// - ReserveRow, RawDataVector, view, FullSizeBytes, operator<<
/// - WriteSerializer / ReadSerializer with TABLE_StaticMax, TABLE_Max
/// - to_host / to_device / clear_device / deviceView (CUDA)
/// - Move semantics
#include <cassert>
#include <memory>
#ifndef DNDS_ARRAY_HPP
#    define DNDS_ARRAY_HPP

#    include <vector>
#    include <iostream>
#    include <typeinfo>
#    include <utility>

#    include <fmt/core.h>

#    include "Defines.hpp"
#    include "ArrayBasic.hpp"
#    include "DeviceStorage.hpp"
#    include "DeviceView.hpp"
#    include "SerializerBase.hpp"
#    include "SerializerJSON.hpp"
#    include "Vector.hpp"

namespace DNDS
{

    /**
     * @brief Core 2D variable-length array container, the storage foundation of DNDSR.
     *
     * @details
     * @ref DNDS::Array "Array" is a single template that unifies five distinct storage layouts
     * used throughout the CFD code base (cell volumes, conservative variables,
     * mesh connectivity, reconstruction coefficients, etc.). The layout is
     * chosen at compile time by combining `_row_size` and `_row_max`:
     *
     * ## Array layouts
     * |                         | _row_size>=0                         | _row_size==DynamicSize              | _row_size==NonUniformSize |
     * | ---                     |          ---                         |                    ---              |                       --- |
     * |_row_max>=0              |  TABLE_StaticFixed                   |  TABLE_Fixed                        |   TABLE_StaticMax         |
     * |_row_max==DynamicSize    |  TABLE_StaticFixed _row_max ignored  |  TABLE_Fixed  _row_max ignored      |   TABLE_Max               |
     * |_row_max==NonUniformSize |  TABLE_StaticFixed _row_max ignored  |  TABLE_Fixed  _row_max ignored      |   CSR                     |
     *
     * Concrete semantics per layout:
     * - **TABLE_StaticFixed**: every row has `_row_size` elements (compile-time
     *   constant). Used for Euler state vectors (5 reals), cell volumes (1),
     *   coordinates (3), etc.
     * - **TABLE_Fixed**: every row has the same runtime-determined width
     *   (`_row_size_dynamic`). Used when the width depends on solver settings
     *   (e.g., reconstruction polynomial order).
     * - **TABLE_StaticMax / TABLE_Max**: rows are padded to a maximum width
     *   (`_row_max` compile-time or runtime), with an auxiliary `_pRowSizes`
     *   vector giving the actual used width per row. Offers O(1) random access
     *   at the cost of wasted space for short rows.
     * - **CSR**: flat buffer plus `_pRowStart[n+1]`. No wasted space but needs
     *   pointer indirection. Two internal sub-states: *compressed* (flat) and
     *   *decompressed* (`vector<vector<T>>`, used during incremental row growth).
     *   `Compress()` must be called before MPI communication or serialization.
     *
     * The class inherits from @ref DNDS::ObjectNaming "ObjectNaming" so each instance may carry a
     * human-readable name (e.g. `"coords"`, `"cell2node"`) shown in assertions.
     *
     * @tparam T         Element type. Typically #real (double) or #index (int64_t).
     *                   Must be trivially copyable for CUDA / MPI paths.
     * @tparam _row_size Row width:
     *                   - `>=0`   fixed width at compile time;
     *                   - @ref DynamicSize    uniform width set at Resize();
     *                   - @ref NonUniformSize variable per row (combined with `_row_max`).
     * @tparam _row_max  Only relevant when `_row_size == NonUniformSize`:
     *                   - `>=0`   `TABLE_StaticMax` layout;
     *                   - @ref DynamicSize    `TABLE_Max` layout;
     *                   - @ref NonUniformSize `CSR` layout.
     * @tparam _align    Alignment hint (currently only @ref NoAlign is used).
     *
     * @sa ArrayBasic.hpp for the layout enum and signature parsing.
     * @sa ArrayTransformer for adding MPI ghost communication.
     * @sa ArrayPair for the typical father/son bundle.
     * @sa docs/architecture/arrays.md, docs/guides/array_usage.md.
     * @todo Implement the `_align` feature (currently ignored).
     */
    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class Array : public ArrayLayout<T, _row_size, _row_max, _align>, public ObjectNaming
    {
    public:
        using self_type = Array<T, _row_size, _row_max, _align>;

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

        using t_View = ArrayView<T, _row_size, _row_max, _align>;

        static constexpr bool IsCSR() { return isCSR; }

    public:
        //* compressed data
        using t_Data = host_device_vector<value_type>;
        //* uncompressed data (only for CSR)
        using t_DataUncompressed = std::vector<std::vector<value_type>>;

        //* non uniform data: CSR
        using t_RowStart = host_device_vector<index>;
        using t_pRowStart = ssp<t_RowStart>;

        //* non uniform-with max data: TABLE
        using t_RowSizes = host_device_vector<rowsize>;
        using t_pRowSizes = ssp<t_RowSizes>;

    protected:
        t_pRowStart _pRowStart; // CSR   in number of T
        t_pRowSizes _pRowSizes; // TABLE in number of T
        t_Data _data;
        DeviceBackend deviceBackend = DeviceBackend::Unknown;

        t_DataUncompressed _dataUncompressed;

        index _size = 0;               // in number of T
        rowsize _row_size_dynamic = 0; // in number of T
    public:
        /// @brief Shared pointer to the row-start index (CSR layout only).
        /// @details `_pRowStart->at(i)` gives the flat-buffer offset of row `i`.
        /// Size is `Size()+1`; the sentinel at the end equals `DataSize()`.
        t_pRowStart getRowStart() { return _pRowStart; }
        /// @brief Shared pointer to the per-row size vector (TABLE_Max / TABLE_StaticMax).
        /// @details For padded layouts, records the number of "used" columns in each row.
        t_pRowSizes getRowSizes() { return _pRowSizes; }

    public:
        /// @brief Default-constructed array: empty, no storage.
        Array() = default;

        /// @brief Named constructor: sets the object name for tracing/debugging.
        /// Delegates to the default constructor, then sets the name.
        explicit Array(ObjName objName)
        {
            this->setObjectName(std::move(objName.name));
        }

        // TODO: constructors
        // TODO: A indexer-copying build method:
        // TODO: for CSR: c->c, u->u
        // TODO: for intertype: CSR->Max, Max->CSR ...

        /// @brief Number of rows currently stored. O(1).
        [[nodiscard]] index Size() const { return _size; }

        /// @brief Uniform row width for fixed layouts (no row index needed).
        /// @details Valid only for `TABLE_Fixed` and `TABLE_StaticFixed`; asserts otherwise.
        /// @return Width in number of `T` elements.
        [[nodiscard]] rowsize RowSize() const // iRow is actually dummy here
        {
            if constexpr (_dataLayout == TABLE_Fixed)
                return _row_size_dynamic;
            else if constexpr (_dataLayout == TABLE_StaticFixed)
                return rs;
            else
            {
                DNDS_assert_info(false, "invalid call");
                return -1;
            }
        }

        /// @brief Number of T elements per row in flat storage (data stride).
        /// For TABLE_StaticMax/TABLE_Max, this is _row_max (padded), not RowSize (used).
        /// Not valid for CSR (variable stride per row).
        [[nodiscard]] rowsize DataStride() const
        {
            if constexpr (_dataLayout == TABLE_StaticFixed)
                return rs;
            else if constexpr (_dataLayout == TABLE_StaticMax)
                return rm;
            else if constexpr (_dataLayout == TABLE_Fixed || _dataLayout == TABLE_Max)
                return _row_size_dynamic;
            else
            {
                DNDS_assert(false);
                return 0;
            }
        }

        /// @brief Width used by row `iRow` in number of `T` elements.
        /// @details Works for every layout:
        /// - `TABLE_*Fixed`: returns the uniform row width (iRow ignored for value);
        /// - `TABLE_*Max`:    returns `_pRowSizes->at(iRow)`;
        /// - `CSR`:           returns `pRowStart[iRow+1] - pRowStart[iRow]` when
        ///                    compressed, else the nested vector size.
        /// @param iRow Row index in `[0, Size())`. Asserted in non-fixed layouts.
        [[nodiscard]] rowsize RowSize(index iRow) const
        {
            if constexpr (_dataLayout == TABLE_Fixed)
                return _row_size_dynamic;
            else if constexpr (_dataLayout == TABLE_StaticFixed)
                return rs;
            DNDS_assert_info(iRow < _size && iRow >= 0, "query position out of range");
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax) // TABLE with Max
            {
                DNDS_assert_info(_pRowSizes, "_pRowSizes invalid"); // TABLE with Max must have a RowSizes
                return _pRowSizes->at(iRow);
            }
            else if constexpr (_dataLayout == CSR)
            {
                if (IfCompressed())
                {
                    index pDiff = _pRowStart->at(iRow + 1) - _pRowStart->at(iRow);
                    DNDS_assert(pDiff < INT32_MAX); // overflow
                    return static_cast<rowsize>(pDiff);
                }
                else
                {
                    auto rs_cur = _dataUncompressed.at(iRow).size();
                    return static_cast<rowsize>(rs_cur);
                }
            }
        }

        /// @brief Maximum allowed row width for `TABLE_Max` / `TABLE_StaticMax`.
        /// @details Returns the compile-time `rm` for `TABLE_StaticMax` or the
        /// dynamic max (`_row_size_dynamic`) for `TABLE_Max`. Asserts for other layouts.
        [[nodiscard]] rowsize RowSizeMax() const
        {
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
                return _dataLayout == TABLE_Max ? _row_size_dynamic : rm;
            else
                DNDS_assert_info(false, "invalid call");
        }

        /// @brief "Logical" row-field width used by derived (Eigen) arrays: max for
        /// padded layouts, uniform width for fixed layouts. Not valid for CSR.
        [[nodiscard]] rowsize RowSizeField() const
        {
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
                return this->RowSizeMax();
            else if constexpr (_dataLayout == TABLE_Fixed || _dataLayout == TABLE_StaticFixed)
                return this->RowSize();
            else
                DNDS_assert_info(false, "invalid call");
        }

        /// @brief Per-row "field" size for CSR (= actual row width). Invalid elsewhere.
        [[nodiscard]] rowsize RowSizeField(index iRow) const
        {
            if constexpr (_dataLayout == CSR)
                return this->RowSize(iRow);
            else
                DNDS_assert_info(false, "invalid call");
        }

    protected:
        [[nodiscard]] bool IfCompressed_() const
        {
            if constexpr (_dataLayout == CSR)
            {
                if (_size > 0)
                {
                    return bool(_pRowStart);
                }
                return bool(_pRowStart); // size-0 array is not always considered compressed
            }
            return true; // non-CSR ones are always compressed
        }

    public:
        /// @brief (CSR only) Whether the array is in packed / flat form.
        /// @details `true` means data sits in a single `_data` buffer plus `_pRowStart`;
        /// `false` means rows live in a `vector<vector<T>>` (`_dataUncompressed`), which
        /// allows per-row `ResizeRow()`. MPI / serialization / CUDA require the
        /// compressed form -- call @ref Compress() first.
        [[nodiscard]] bool IfCompressed() const
        {
            static_assert(_dataLayout == CSR, "invalid call");
            return IfCompressed_();
        }

        /// @brief (CSR only) Switch to the uncompressed (nested vector) representation.
        /// @details Copies each row out of the flat buffer into an entry of
        /// `_dataUncompressed`, then clears the flat buffer. No-op if already uncompressed.
        /// After this, @ref ResizeRow and @ref ReserveRow may be used to reshape individual rows.
        void CSRDecompress()
        {
            if (!IfCompressed())
                return;
            _dataUncompressed.resize(_size);
            for (index i = 0; i < _size; i++)
            {
                // _dataUncompressed[i].resize( - _pRowStart->at(i));
                auto iterStart = _data.begin() + _pRowStart->at(i);
                auto iterEnd = _data.begin() + _pRowStart->at(i + 1);
                _dataUncompressed[i].assign(iterStart, iterEnd);
            }
            _data.clear();
            _pRowStart.reset();
        }

        /// @brief (CSR only) Pack the nested-vector representation into a flat
        /// buffer plus `_pRowStart`. No-op if already compressed.
        /// @details Row widths are frozen; further per-row resizing requires another
        /// @ref Decompress. Mandatory before MPI ghost exchange, CUDA transfer, or serialization.
        void CSRCompress()
        {
            if (IfCompressed())
                return;
            _pRowStart = std::make_shared<
                typename decltype(_pRowStart)::element_type>(_size + 1, 0);
            _pRowStart->at(0) = 0;
            for (index i = 0; i < _size; i++)
            {
                index rsI = _pRowStart->at(i);
                index rsIP = rsI + static_cast<index>(_dataUncompressed.at(i).size());
                DNDS_check_throw(rsIP >= rsI);
                _pRowStart->at(i + 1) = rsIP;
            }
            _data.resize(_pRowStart->at(_size));
            for (index i = 0; i < _size; i++)
            {
                // // _dataUncompressed[i].resize( - _pRowStart->at(i));
                // auto iterStart = _data.begin() + _pRowStart->at(i);
                // auto iterEnd = _data.begin() + _pRowStart->at(i + 1);
                // _dataUncompressed[i].assign(iterStart, iterEnd);
                // _data.push_back(data)
                memcpy(_data.data() + _pRowStart->at(i), _dataUncompressed[i].data(),
                       sizeof(T) * _dataUncompressed[i].size());
                DNDS_check_throw(_pRowStart->at(i) + _dataUncompressed[i].size() <= _data.size());
                //! any better way?
            }
            _dataUncompressed.clear();
        }

        /// @brief Layout-polymorphic compress: no-op for non-CSR, calls @ref CSRCompress for CSR.
        void Compress()
        {
            if constexpr (_dataLayout == CSR)
                CSRCompress();
        }
        /// @brief Layout-polymorphic decompress: no-op for non-CSR, calls @ref CSRDecompress for CSR.
        void Decompress()
        {
            if constexpr (_dataLayout == CSR)
                CSRDecompress();
        }

        /// @brief Access to the underlying flat buffer (`host_device_vector<T>`).
        /// @details For CSR, asserts that the array is compressed. Mutating the buffer
        /// bypasses all row-size bookkeeping -- use with care.
        t_Data &RawDataVector()
        {
            if constexpr (_dataLayout == CSR)
                DNDS_check_throw(IfCompressed());
            return _data;
        }

        /**
         * @brief Resize the array, setting a uniform or maximum row width.
         *
         * @details
         * Invalidates all existing data and resets row-size metadata.
         *
         * Layout-specific semantics:
         * - `TABLE_StaticFixed`: `nRow_size_dynamic` must equal `rs`.
         * - `TABLE_StaticMax`:   `nRow_size_dynamic` must equal `rm`.
         * - `TABLE_Fixed`:       sets the runtime uniform row width.
         * - `TABLE_Max`:         sets the runtime maximum row width (per-row sizes start at 0).
         * - `CSR`:               requires the array to be **decompressed**; allocates
         *                        `nSize` empty rows (nested vectors sized to `nRow_size_dynamic`).
         *
         * @param nSize               New number of rows.
         * @param nRow_size_dynamic   Row width (uniform layouts) or max width
         *                            (padded layouts) or initial row length (CSR).
         */
        void Resize(index nSize, rowsize nRow_size_dynamic)
        {
            if constexpr (_dataLayout == CSR) // to un compressed
            {
                DNDS_check_throw_info(!IfCompressed(), "Need to decompress before auto resizing");
                _size = nSize;
                // _dataUncompressed.resize(nSize, typename decltype(_dataUncompressed)::value_type(nRow_size_dynamic));
                // _dataUncompressed.resize(nSize);
                _dataUncompressed.assign(nSize, typename decltype(_dataUncompressed)::value_type(nRow_size_dynamic));
            }
            else
            {
                _size = nSize;
                if constexpr (_dataLayout == TABLE_Fixed || _dataLayout == TABLE_Max)
                    _data.resize(nSize * nRow_size_dynamic), _row_size_dynamic = nRow_size_dynamic;
                else if constexpr (_dataLayout == TABLE_StaticFixed)
                    _data.resize(nSize * rs), DNDS_check_throw(nRow_size_dynamic == rs);
                else if constexpr (_dataLayout == TABLE_StaticMax)
                    _data.resize(nSize * rm), DNDS_check_throw(nRow_size_dynamic == rm);

                if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
                {
                    if (_pRowSizes.use_count() == 1)
                        _pRowSizes->resize(nSize, 0);
                    else
                        _pRowSizes = std::make_shared<
                            typename decltype(_pRowSizes)::element_type>(nSize, 0);
                }
            }
        }

        /// @brief Resize using only the row count (layouts with an implicit row width).
        /// @details Valid for:
        /// - `TABLE_StaticFixed` / `TABLE_StaticMax` (width comes from template params);
        /// - `CSR` decompressed (rows start empty, grow via @ref ResizeRow).
        /// Asserts for other layouts -- use the two-argument overload instead.
        /// @param nSize New number of rows.
        void Resize(index nSize)
        {
            if constexpr (_dataLayout == CSR)
            {
                DNDS_check_throw_info(!IfCompressed(), "Need to decompress before auto resizing");
                _size = nSize;
                _dataUncompressed.resize(nSize);
            }
            else if constexpr (_dataLayout == TABLE_StaticFixed)
            {
                _size = nSize;
                _data.resize(nSize * rs);
            }
            else if constexpr (_dataLayout == TABLE_StaticMax)
            {
                _size = nSize;
                _data.resize(nSize * rm);
                if (_pRowSizes.use_count() == 1)
                    _pRowSizes->resize(nSize, 0);
                else
                    _pRowSizes = std::make_shared<
                        typename decltype(_pRowSizes)::element_type>(nSize, 0);
            }
            else
            {
                static_assert(_dataLayout == CSR ||
                                  _dataLayout == TABLE_StaticFixed ||
                                  _dataLayout == TABLE_StaticMax,
                              "Resize(index nSize) is invalid call");
                DNDS_check_throw_info(false, "invalid call");
            }
        }

        /**
         * @brief Resize a CSR array directly to the compressed form via a width functor.
         *
         * @details CSR-only. Allocates `nSize+1` `_pRowStart` entries populated from
         * the prefix sum of `FRowSize(i)`, then sizes the flat buffer to match.
         * No nested-vector intermediate step; the array is compressed on return.
         *
         * @tparam TFRowSize Callable with signature `rowsize(index)`.
         * @param nSize      New number of rows.
         * @param FRowSize   Width-of-row-i functor.
         */
        template <class TFRowSize>
        void Resize(index nSize, TFRowSize &&FRowSize)
        {
            if constexpr (_dataLayout == CSR)
            {
                _size = nSize;
                _pRowSizes.reset(), _dataUncompressed.clear(); //*directly to compressed
                _pRowStart = std::make_shared<typename decltype(_pRowStart)::element_type>(nSize + 1);
                _pRowStart->operator[](0) = 0;
                for (index i = 0; i < nSize; i++)
                    (*_pRowStart)[i + 1] = (*_pRowStart)[i] + FRowSize(i);
                _data.resize(_pRowStart->at(nSize));
            }
            static_assert(_dataLayout == CSR, "Only Non Uniform, CSR for now");
            static_assert(std::is_invocable_r_v<rowsize, TFRowSize, index>, "Call invalid");
        }

        /**
         * @brief Change the width of a single row.
         *
         * @details Valid for `CSR` (decompressed only) and `TABLE_*Max`.
         * For `TABLE_*Max`, the new size must not exceed the configured maximum and
         * the per-row-size vector is copied-on-write if shared with another array.
         * For CSR, the array must be uncompressed; call @ref Decompress first.
         *
         * @param iRow      Row index in `[0, Size())`.
         * @param nRowSize  New width in `T` elements.
         */
        void ResizeRow(index iRow, rowsize nRowSize)
        {
            if constexpr (_dataLayout == CSR)
            {
                DNDS_check_throw_info(!IfCompressed(), "Need to decompress before auto resizing row");
                DNDS_check_throw_info(iRow < _size && iRow >= 0, "query position out of range");
                _dataUncompressed.at(iRow).resize(nRowSize);
            }
            else if constexpr (_dataLayout == TABLE_Max ||
                               _dataLayout == TABLE_StaticMax)
            {
                DNDS_check_throw(nRowSize <= (_dataLayout == TABLE_Max ? _row_size_dynamic : rm)); //_row_size_dynamic is max now
                DNDS_check_throw_info(iRow < _size && iRow >= 0, "query position out of range");
                DNDS_check_throw_info(_pRowSizes, "_pRowSizes invalid");
                if (_pRowSizes.use_count() > 1) // shared
                    _pRowSizes = std::make_shared<
                        typename decltype(_pRowSizes)::element_type>(*_pRowSizes); // copy the row sizes
                _pRowSizes->at(iRow) = nRowSize;                                   // change
            }
            else
            {
                static_assert(_dataLayout == CSR ||
                                  _dataLayout == TABLE_Max ||
                                  _dataLayout == TABLE_StaticMax,
                              "invalid call");
                DNDS_check_throw_info(false, "invalid call");
            }
        }

        /// @brief Reserve capacity for a CSR decompressed row without changing its size.
        /// @details Analogous to `std::vector::reserve` for the nested row buffer.
        /// CSR-only, decompressed-only.
        void ReserveRow(index iRow, rowsize nRowSize)
        {
            if constexpr (_dataLayout == CSR)
            {
                DNDS_check_throw_info(!IfCompressed(), "Need to decompress before auto resizing row");
                DNDS_check_throw_info(iRow < _size && iRow >= 0, "query position out of range");
                _dataUncompressed.at(iRow).reserve(nRowSize);
            }
            else
            {
                DNDS_check_throw_info(false, "invalid call");
                static_assert(_dataLayout == CSR, "invalid call");
            }
        }

        // TODO: Data reference query method and pointer query method
        // TODO: ? same-size compress for non-uniforms

        /// @brief Produce a lightweight, device-agnostic view onto the array.
        /// @details The returned @ref DNDS::ArrayView "ArrayView" captures pointers and sizes but does
        /// not own any storage. It is the type that implements actual `operator[]`
        /// indexing for all layouts; it is also host/device-callable and is the
        /// building block for @ref DNDS::ArrayDeviceView "ArrayDeviceView" on CUDA.
        t_View view()
        {
            return t_View(_size, _data.data(), _data.size(),
                          _pRowStart ? _pRowStart->data() : nullptr, _pRowStart ? _pRowStart->size() : 0,
                          _pRowSizes ? _pRowSizes->data() : nullptr, _pRowSizes ? _pRowSizes->size() : 0,
                          _row_size_dynamic,
                          IfCompressed_(), IfCompressed_() ? nullptr : &_dataUncompressed);
        }

        /// @brief Bounds-checked element access.
        /// @details Asserts that `iRow` and `iCol` are in range (taking the used
        /// row size into account, not just the stride). Works for every layout.
        /// @param iRow Row index in `[0, Size())`.
        /// @param iCol Column index in `[0, RowSize(iRow))`.
        const T &at(index iRow, rowsize iCol) const
        {
            DNDS_assert_info(iRow < _size && iRow >= 0,
                             fmt::format(
                                 "query position i[{}] out of range [0, {}), array: {}",
                                 iRow, _size, this->getObjectIdentity(GetArrayName())));
            DNDS_assert_info(iCol < RowSize(iRow) && iCol >= 0,
                             fmt::format(
                                 "query position j[{}] out of range [0, {}), array: {}",
                                 iCol, RowSize(iRow), this->getObjectIdentity(GetArrayName())));
            if constexpr (_dataLayout == TABLE_StaticFixed)
                return _data.at(iRow * rs + iCol);
            else if constexpr (_dataLayout == TABLE_StaticMax)
                return _data.at(iRow * rm + iCol);
            else if constexpr (_dataLayout == TABLE_Fixed)
                return _data.at(iRow * _row_size_dynamic + iCol);
            else if constexpr (_dataLayout == TABLE_Max)
                return _data.at(iRow * _row_size_dynamic + iCol);
            else if constexpr (_dataLayout == CSR)
            {
                if (IfCompressed())
                    return _data.at(_pRowStart->at(iRow) + iCol);
                else
                    return _dataUncompressed.at(iRow).at(iCol);
            }
            else
            {
                DNDS_assert_info(false, "invalid call");
            }
        }

        /// @brief Bounds-checked 2D element access (writable).
        /// @details Convenience wrapper around #at. `iCol` defaults to 0 so `arr(i)`
        /// accesses the first column, useful for single-column layouts.
        T &operator()(index iRow, rowsize iCol = 0)
        {
            return const_cast<T &>(at(iRow, iCol));
        }

        /// @brief Bounds-checked 2D element access (read-only).
        const T &operator()(index iRow, rowsize iCol = 0) const
        {
            return at(iRow, iCol);
        }

        /**
         * @brief Return a raw pointer to the start of row `iRow`.
         *
         * @details Fast, untyped access used by stencil loops. Derived classes
         * (e.g. @ref DNDS::ArrayEigenVector "ArrayEigenVector", @ref DNDS::ArrayEigenMatrix "ArrayEigenMatrix") override this to return typed
         * Eigen maps instead of `T*`.
         *
         * @param iRow Row index. For `CSR` compressed, `iRow == Size()` is allowed
         *             and returns the past-the-end pointer, useful for computing
         *             the flat buffer end in sweeps.
         * @return Pointer to the first element of row `iRow`.
         */
        T *operator[](index iRow)
        {
            return this->view().operator[](iRow);
            // DNDS_assert_info(iRow <= _size && iRow >= 0, "query position i out of range");
            // if constexpr (_dataLayout == TABLE_StaticFixed)
            //     return _data.data() + iRow * rs;
            // else if constexpr (_dataLayout == TABLE_StaticMax)
            //     return _data.data() + iRow * rm;
            // else if constexpr (_dataLayout == TABLE_Fixed)
            //     return _data.data() + iRow * _row_size_dynamic;
            // else if constexpr (_dataLayout == TABLE_Max)
            //     return _data.data() + iRow * _row_size_dynamic;
            // else if constexpr (_dataLayout == CSR)
            // {
            //     if (IfCompressed())
            //         return _data.data() + _pRowStart->at(iRow);
            //     else if (this->Size() == 0)
            //     {
            //         static_assert(((T *)(NULL) - (T *)(NULL)) == 0);
            //         return (T *)(NULL); // used for past-the-end inquiry of size 0 array
            //     }
            //     else
            //     {
            //         DNDS_assert_info(iRow < _size, "past-the-end query forbidden for CSR uncompressed");
            //         return _dataUncompressed.at(iRow).data();
            //     }
            // }
            // else
            // {
            //     DNDS_assert_info(false, "invalid call");
            // }
        }

        /// @brief Const row pointer, see the non-const overload.
        const T *operator[](index iRow) const
        {
            return static_cast<const T *>(const_cast<self_type *>(this)->operator[](iRow));
        }

        /// @brief Raw pointer to the flat data buffer.
        /// @param B Target device. `DeviceBackend::Unknown` (default) returns the host
        ///          pointer; otherwise returns the device pointer (must match the
        ///          array's current device).
        T *data(DeviceBackend B = DeviceBackend::Unknown)
        {
            if constexpr (_dataLayout == CSR)
                DNDS_assert_info(this->IfCompressed(), "CSR must be compressed to get data pointer");
            if (B == DeviceBackend::Unknown)
                return _data.data();
            else
            {
                DNDS_assert(_data.device() == B);
                return _data.dataDevice();
            }
        }

        /// @brief Total number of `T` elements currently stored in the flat buffer.
        /// @details For CSR, requires the array to be compressed.
        size_t DataSize() const
        {
            if (this->Size() == 0)
                return 0;
            if constexpr (_dataLayout == CSR)
                DNDS_assert_info(this->IfCompressed(), "CSR must be compressed to get DataSize()");
            return _data.size();
        }

        /// @brief Flat buffer size in bytes (= `DataSize() * sizeof(T)`).
        size_t DataSizeBytes() const
        {
            return this->DataSize() * sizeof_T;
        }

        /// @brief Copy raw row data from another Array of the same type.
        /// Works for all layouts (StaticFixed, Fixed, CSR, etc.) and is not hidden
        /// by derived types that make operator()/operator[] private.
        void CopyRowFrom(index dstRow, const self_type &src, index srcRow)
        {
            auto rs = src.RowSize(srcRow);
            DNDS_assert(rs == this->RowSize(dstRow));
            const T *srcPtr = const_cast<self_type &>(src)[srcRow];
            T *dstPtr = (*this)[dstRow];
            if constexpr (std::is_trivially_copyable_v<T>)
                std::memcpy(dstPtr, srcPtr, rs * sizeof(T));
            else
                for (rowsize j = 0; j < rs; j++)
                    dstPtr[j] = srcPtr[j];
        }

        /// @brief Set per-row sizes from a source, applying an index mapping, then compress.
        /// For CSR layout only. rowSizeFunc(i) returns the desired row size for row i.
        /// Not hidden by derived types that make ResizeRow private.
        template <class TRowSizeFunc>
        void ResizeRowsAndCompress(TRowSizeFunc &&rowSizeFunc)
        {
            static_assert(_dataLayout == CSR, "ResizeRowsAndCompress only for CSR");
            for (index i = 0; i < this->Size(); i++)
                this->ResizeRow(i, rowSizeFunc(i));
            this->Compress();
        }

        /// @brief Total footprint in bytes including structural arrays.
        /// @details Sums the flat data buffer, `_pRowStart` (if any), and
        /// `_pRowSizes` (if any). Approximate because shared-ownership of row
        /// structures is not deduplicated.
        size_t FullSizeBytes() const
        {
            size_t b = this->DataSize() * sizeof_T;
            if (_pRowStart)
                b += _pRowStart->size() * sizeof(index);
            if (_pRowSizes)
                b += _pRowSizes->size() * sizeof(rowsize);
            return b;
        }

        /// @brief Combined hash of size, structural arrays, and data.
        /// @details Byte-hashes non-hashable elements. Intended for testing /
        /// equality diagnostics; not guaranteed cryptographically strong.
        std::size_t hash()
        {
            std::size_t hashData;
            if constexpr (_dataLayout == CSR)
            {
                if (IfCompressed())
                    hashData = vector_hash<T>()(_data.begin(), _data.end());
                else
                    hashData = vector_hash<std::vector<T>>()(_dataUncompressed);
            }
            else
                hashData = vector_hash<T>()(_data.begin(), _data.end());
            std::size_t hashSize = 0;
            if (_pRowSizes)
                hashSize = vector_hash<rowsize>()(_pRowSizes->begin(), _pRowSizes->end());
            if (_pRowStart)
                hashSize = vector_hash<index>()(_pRowSizes->begin(), _pRowSizes->end());
            return array_hash<std::size_t, 3>()(std::array<std::size_t, 3>{std::size_t(_size), hashSize, hashData});
        }

        /// @brief Pretty-print rows, one per line, tab-separated.
        friend std::ostream &operator<<(std::ostream &o, const Array<T, _row_size, _row_max, _align> &A)
        {
            for (index i = 0; i < A._size; i++)
            {
                for (index j = 0; j < A.RowSize(i); j++)
                    o << A(i, j) << "\t";
                o << std::endl;
            }
            return o;
        }

        /// @brief Compile-time layout tag (one of `TABLE_StaticFixed`, `TABLE_Fixed`,
        /// `TABLE_StaticMax`, `TABLE_Max`, `CSR`).
        static constexpr DataLayout GetDataLayoutStatic() { return _dataLayout; }
        /// @brief Runtime accessor for the layout tag (constexpr-folded).
        constexpr DataLayout GetDataLayout() { return _dataLayout; }

        /// @brief Shallow clone: copies all metadata and shares structural/data storage.
        /// @details Copies `_size`, `_row_size_dynamic`, device backend, and the
        /// nested-vector storage. The `_data`, `_pRowStart`, `_pRowSizes` members
        /// are `host_device_vector` / `shared_ptr`, so they share ownership with
        /// the source; subsequent modifications to one may affect the other.
        void clone(const self_type &R)
        {
            this->_size = R._size;
            this->_data = R._data;
            this->_pRowSizes = R._pRowSizes;
            this->_pRowStart = R._pRowStart;
            this->_dataUncompressed = R._dataUncompressed;
            this->_row_size_dynamic = R._row_size_dynamic;

            this->deviceBackend = R.deviceBackend;
        }

        /// @brief Deep copy alias. Currently delegates to #clone; kept for API
        /// compatibility and to allow a future true deep-copy implementation.
        void CopyData(const self_type &R)
        {
            this->clone(R);
            // non-trivial copy call: unique_ptr
        }

        /// @brief Copy-assignment; implemented via #clone with self-assign guard.
        self_type &operator=(const self_type &R)
        {
            if (this == &R)
                return *this;
            this->clone(R);
            return *this;
        }

        /// @brief Copy constructor (same semantics as #clone).
        Array(const self_type &R)
        {
            this->clone(R);
        }

        /// @brief Swap the storage of two arrays in-place.
        /// @details Both arrays must already have identical logical size and
        /// flat-buffer size. Swaps only what the current layout uses (flat buffer
        /// plus structural pointers, or the nested vectors for CSR decompressed).
        // TODO: SwapData on device?
        void SwapData(self_type &R)
        {
            DNDS_check_throw(R.Size() == this->Size());
            DNDS_check_throw(R._data.size() == _data.size());
            if constexpr (_dataLayout == CSR)
            {
                if (IfCompressed())
                {
                    _data.swap(R._data);
                    _pRowStart.swap(R._pRowStart);
                }
                else
                {
                    _dataUncompressed.swap(R._dataUncompressed);
                    _pRowSizes.swap(R._pRowSizes);
                }
            }
            else
            {
                _data.swap(R._data);
            }
            {
                std::swap(deviceBackend, R.deviceBackend);
            }
        }

        void __WriteSerializerData(const Serializer::SerializerBaseSSP &serializerP, Serializer::ArrayGlobalOffset offset)
        {
            auto treatAsBytes = [&]()
            { serializerP->WriteUint8Array("data", (uint8_t *)_data.data(), _data.size() * sizeof_T, offset * sizeof_T); };
            if constexpr (std::is_same_v<T, index>)
            {
                // TODO: OPTIMIZE serializer pass a range
                serializerP->WriteIndexVector("data", std::vector<index>(_data), offset);
            }
            else if constexpr (std::is_same_v<T, real>)
            {
                if (!std::dynamic_pointer_cast<Serializer::SerializerJSON>(serializerP))
                { // TODO: OPTIMIZE serializer pass a range
                    serializerP->WriteRealVector("data", std::vector<real>(_data), offset);
                }
                else
                    treatAsBytes();
            }
            else
                treatAsBytes();
        }

        void __ReadSerializerData(const Serializer::SerializerBaseSSP &serializerP, Serializer::ArrayGlobalOffset &offset)
        {
            auto treatAsBytes = [&]()
            {
                Serializer::ArrayGlobalOffset localOffset = offset;
                if (localOffset.isDist())
                {
                    localOffset = localOffset * index(sizeof_T);
                }
                index bufferSize{0};
                serializerP->ReadUint8Array("data", nullptr, bufferSize, localOffset);
                DNDS_check_throw(bufferSize % sizeof_T == 0);
                _data.resize(bufferSize / sizeof_T);
                uint8_t dummy{};
                serializerP->ReadUint8Array("data", bufferSize == 0 ? &dummy : (uint8_t *)_data.data(), bufferSize, localOffset);
                localOffset.CheckMultipleOf(sizeof_T);
                offset = localOffset / sizeof_T;
            };

            if constexpr (std::is_same_v<T, index>)
            { // TODO: OPTIMIZE host_device_vector accept rvalue std::vector
                std::vector<index> data_tmp;
                serializerP->ReadIndexVector("data", data_tmp, offset);
                _data = data_tmp;
            }
            else if constexpr (std::is_same_v<T, real>)
            {
                if (!std::dynamic_pointer_cast<Serializer::SerializerJSON>(serializerP))
                { // TODO: OPTIMIZE host_device_vector accept rvalue std::vector
                    std::vector<real> data_tmp;
                    serializerP->ReadRealVector("data", data_tmp, offset);
                    _data = data_tmp;
                }
                else
                    treatAsBytes();
            }
            else
                treatAsBytes();
        }

        /// @brief Serialize (write) array data to a serializer.
        ///
        /// Writes metadata (array_sig, array_type, size, row_size_dynamic), structural
        /// data (pRowStart for CSR, pRowSizes for TABLE_Max/StaticMax), and the flat
        /// data buffer into a sub-path `name` under the serializer's current path.
        ///
        /// This method is called by ParArray::WriteSerializer. Users should call the
        /// ParArray version, which handles MPI coordination and CSR global offsets.
        ///
        /// @param serializerP  Serializer instance (JSON per-rank or H5 collective).
        /// @param name         Sub-path name under which the array is stored.
        /// @param offset       [in] Row-level partitioning offset. Typically Parts
        ///                     (serializer computes per-rank offsets automatically).
        /// @param dataOffset   [in] Element-level data range for this rank.
        ///                     - Per-rank or non-CSR: Unknown (default). Array writes
        ///                       pRowStart in local coordinates.
        ///                     - Collective CSR: must be isDist() = {localDataCount,
        ///                       globalDataStart}, computed by ParArray via MPI_Scan.
        ///                       Array skips pRowStart (ParArray writes it separately
        ///                       in global coordinates). Asserted for collective CSR.
        void WriteSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name,
                             Serializer::ArrayGlobalOffset offset,
                             Serializer::ArrayGlobalOffset dataOffset = Serializer::ArrayGlobalOffset_Unknown)
        {
            auto cwd = serializerP->GetCurrentPath();
            serializerP->CreatePath(name);
            serializerP->GoToPath(name);

            serializerP->WriteString("array_sig", GetArraySignature());
            serializerP->WriteString("array_type", typeid(self_type).name());
            if (serializerP->IsPerRank())
                serializerP->WriteIndex("size", _size);
            else
            {
                std::vector<index> _size_vv;
                _size_vv.push_back(_size);
                serializerP->WriteIndexVectorPerRank("size", _size_vv);
            }
            serializerP->WriteInt("row_size_dynamic", _row_size_dynamic);
            // if (_size == 0) //! cannot do this, collective calls!
            //     return;
            if constexpr (_dataLayout == CSR)
            {
                if (!this->IfCompressed())
                    this->Compress();
                // For collective serializers, dataOffset must be isDist() (set by ParArray).
                // ParArray writes pRowStart in global coordinates; Array only writes for per-rank.
                DNDS_assert_info(serializerP->IsPerRank() || dataOffset.isDist(),
                                 "CSR collective write requires isDist dataOffset from ParArray");
                if (dataOffset.isDist())
                {
                    // ParArray handles pRowStart write in global coords
                }
                else
                {
                    serializerP->WriteSharedIndexVector("pRowStart", _pRowStart, offset);
                }
            }
            else if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
            {
                serializerP->WriteSharedRowsizeVector("pRowSizes", _pRowSizes, offset);
            }
            else // fixed
            {
            }
            // doing data
            this->__WriteSerializerData(serializerP, offset);

            serializerP->GoToPath(cwd);
        }

        /// @brief Convenience overload that discards the dataOffset output.
        /// @see ReadSerializer(SerializerBaseSSP, const std::string&, ArrayGlobalOffset&, ArrayGlobalOffset&)
        void ReadSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name,
                            Serializer::ArrayGlobalOffset &offset)
        {
            Serializer::ArrayGlobalOffset dataOffset = Serializer::ArrayGlobalOffset_Unknown;
            ReadSerializer(serializerP, name, offset, dataOffset);
        }

        /// @brief Deserialize (read) array data from a serializer.
        ///
        /// Reads metadata (array_sig, size, row_size_dynamic), structural data
        /// (pRowStart for CSR, pRowSizes for TABLE_Max/StaticMax), and the flat data
        /// buffer from a sub-path `name` under the serializer's current path.
        ///
        /// This method is called by ParArray::ReadSerializer. Users should call the
        /// ParArray version, which handles EvenSplit resolution and CSR offset computation.
        ///
        /// Input `offset` must NOT be EvenSplit; ParArray resolves that before calling.
        ///
        /// @param serializerP  Serializer instance (JSON per-rank or H5 collective).
        /// @param name         Sub-path name for this array.
        /// @param offset       [in/out] Row-level offset.
        ///                     - In: Unknown (same-np, serializer uses ::rank_offsets)
        ///                       or isDist({localRows, globalRowStart}).
        ///                     - Out: updated to the resolved row-level position after
        ///                       reading (derived from dataOffset / DataStride for non-CSR).
        /// @param dataOffset   [out] Element-level data offset resolved during the read.
        ///                     - Per-rank or Unknown offset: stays Unknown.
        ///                     - Collective isDist non-CSR: {localRows * DataStride(),
        ///                       globalRowStart * DataStride()}.
        ///                     - Collective CSR: {localDataCount, globalDataStart},
        ///                       derived from the stored global pRowStart.
        void ReadSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name,
                            Serializer::ArrayGlobalOffset &offset,
                            Serializer::ArrayGlobalOffset &dataOffset)
        {
            DNDS_assert_info(!(offset == Serializer::ArrayGlobalOffset_EvenSplit),
                             "Array::ReadSerializer must not receive EvenSplit; ParArray should resolve it first");

            auto cwd = serializerP->GetCurrentPath();
            serializerP->GoToPath(name);

            // --- Phase 1: Read metadata ---
            std::string array_sigRead;
            serializerP->ReadString("array_sig", array_sigRead);
            DNDS_check_throw_info(array_sigRead == this->GetArraySignature() || ArraySignatureIsCompatible(array_sigRead),
                                  array_sigRead + ", i am : " + this->GetArraySignature());
            auto [szR, rsR, rmR, align] = ParseArraySignatureTuple(array_sigRead);
            if (_row_size != rsR)
            {
                if (_row_size == NonUniformSize || rsR == NonUniformSize)
                    DNDS_check_throw_info(false, "can't handle here");
            }
            if (serializerP->IsPerRank())
                serializerP->ReadIndex("size", _size);
            else if (offset.isDist())
                _size = offset.size();
            else
            {
                std::vector<index> _size_vv;
                Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_Unknown;
                serializerP->ReadIndexVector("size", _size_vv, offsetV);
                DNDS_check_throw(_size_vv.size() == 1);
                _size = _size_vv.front();
            }
            serializerP->ReadInt("row_size_dynamic", _row_size_dynamic);
            if (_row_size >= 0) // TODO: fix this! need full conversion check (maybe just a casting)
            {
                if (rsR == DynamicSize)
                    DNDS_check_throw(_row_size_dynamic == _row_size), _row_size_dynamic = 0;
                else if (rsR >= 0)
                    DNDS_check_throw(rsR == _row_size), _row_size_dynamic = 0;
            }
            if (_row_size == DynamicSize && rsR >= 0)
                _row_size_dynamic = rsR;
            if (_row_max == DynamicSize && rmR >= 0)
                _row_size_dynamic = rmR; // TODO: fix this! need a _row_max_dynamic ?

            // --- Phase 2: Read structural data and resolve dataOffset ---
            __ReadSerializerStructuralAndResolveDataOffset(serializerP, offset, dataOffset);

            // --- Phase 3: Read flat data and propagate offsets ---
            __ReadSerializerDataAndPropagateOffset(serializerP, offset, dataOffset);
            // TODO: check data validity

            serializerP->GoToPath(cwd);
        }

        /// @brief Result type for ReadSerializerMeta.
        ///
        /// Derived array types can extend this struct to carry additional
        /// metadata fields (e.g., mat_nRow_dynamic from ArrayEigenMatrix).
        struct ReadSerializerMetaResult
        {
            std::string array_sig;
            rowsize row_size_dynamic{0};
            index size{0};
        };

        /// @brief Reads only metadata from a serialized array without reading data.
        ///
        /// Navigates to sub-path `name` and reads array_sig, row_size_dynamic, and
        /// size. Does NOT read structural data (pRowStart, pRowSizes) or the flat
        /// data buffer. The array's internal state is not modified.
        ///
        /// Derived types (ArrayEigenMatrix, ArrayEigenUniMatrixBatch) override this
        /// method to also read their own extra metadata and to call the base version
        /// with the "array" sub-path they wrap around the base Array serialization.
        ///
        /// @param serializerP  Serializer instance.
        /// @param name         Sub-path name for this array.
        /// @return Metadata: array_sig, row_size_dynamic, size (local size for
        ///         per-rank; 0 for collective without ParArray).
        ReadSerializerMetaResult ReadSerializerMeta(Serializer::SerializerBaseSSP serializerP, const std::string &name)
        {
            auto cwd = serializerP->GetCurrentPath();
            serializerP->GoToPath(name);

            ReadSerializerMetaResult result;
            serializerP->ReadString("array_sig", result.array_sig);
            serializerP->ReadInt("row_size_dynamic", result.row_size_dynamic);
            if (serializerP->IsPerRank())
                serializerP->ReadIndex("size", result.size);

            serializerP->GoToPath(cwd);
            return result;
        }

        /// @brief Reads structural data (pRowStart / pRowSizes) and resolves dataOffset.
        ///
        /// After metadata has been read and _size / _row_size_dynamic set, this method
        /// reads layout-specific structural data from the serializer and computes the
        /// element-level dataOffset from the row-level offset.
        ///
        /// @param serializerP  Serializer instance (already at the array's sub-path).
        /// @param offset       [in/out] Row-level offset; updated for non-CSR if
        ///                     dataOffset could be derived.
        /// @param dataOffset   [out] Element-level data offset resolved from structural
        ///                     data (CSR: from global pRowStart; non-CSR: offset * DataStride).
        void __ReadSerializerStructuralAndResolveDataOffset(
            const Serializer::SerializerBaseSSP &serializerP,
            Serializer::ArrayGlobalOffset &offset,
            Serializer::ArrayGlobalOffset &dataOffset)
        {
            if constexpr (_dataLayout == CSR)
            {
                DNDS_assert_info(serializerP->IsPerRank() || offset.isDist(),
                                 "CSR collective read requires isDist offset from ParArray");
                if (offset.isDist())
                {
                    auto prsOffset = Serializer::ArrayGlobalOffset{_size + 1, offset.offset()};
                    serializerP->ReadSharedIndexVector("pRowStart", _pRowStart, prsOffset);
                    index globalDataStart = _pRowStart->at(0);
                    for (index i = _size; i >= 0; i--)
                        _pRowStart->at(i) -= globalDataStart;
                    dataOffset = Serializer::ArrayGlobalOffset{_pRowStart->at(_size), globalDataStart};
                }
                else
                {
                    serializerP->ReadSharedIndexVector("pRowStart", _pRowStart, offset);
                }
            }
            else if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
            {
                serializerP->ReadSharedRowsizeVector("pRowSizes", _pRowSizes, offset);
            }
            else // TABLE_StaticFixed, TABLE_Fixed
            {
            }

            // CSR dataOffset is resolved above from pRowStart.
            // For non-CSR, derive from row offset * DataStride.
            if constexpr (_dataLayout != CSR)
            {
                if (offset.isDist())
                    dataOffset = offset * this->DataStride();
            }
        }

        /// @brief Reads flat data and propagates resolved offsets back to the caller.
        ///
        /// @param serializerP  Serializer instance (already at the array's sub-path).
        /// @param offset       [in/out] Row-level offset; updated from dataOffset for non-CSR.
        /// @param dataOffset   [in/out] Element-level data offset; may be updated from
        ///                     Unknown to Parts-resolved by __ReadSerializerData.
        void __ReadSerializerDataAndPropagateOffset(
            const Serializer::SerializerBaseSSP &serializerP,
            Serializer::ArrayGlobalOffset &offset,
            Serializer::ArrayGlobalOffset &dataOffset)
        {
            Serializer::ArrayGlobalOffset dataReadOffset = Serializer::ArrayGlobalOffset_Unknown;
            if (dataOffset.isDist())
                dataReadOffset = dataOffset;
            this->__ReadSerializerData(serializerP, dataReadOffset);
            dataOffset = dataReadOffset;
            if constexpr (_dataLayout != CSR)
            {
                if (dataOffset.isDist())
                {
                    dataOffset.CheckMultipleOf(this->DataStride());
                    offset = dataOffset / this->DataStride();
                }
            }
        }

    public:
        /// @brief Mirror the flat/structural buffers back to host memory.
        /// @details CSR arrays must be compressed before calling. After this the
        /// array still has a device mirror unless #clear_device is also called.
        void to_host()
        {
            if constexpr (_dataLayout == CSR)
                DNDS_check_throw_info(IfCompressed(),
                                      fmt::format("CSR need compressing before to_host: {}",
                                                  this->getObjectIdentity(GetArrayName())));
            _data.to_host();
            deviceBackend = DeviceBackend::Unknown;
        }

        /// @brief Mirror the flat/structural buffers to a target device (e.g. CUDA).
        /// @details CSR arrays must be compressed. `backend` must match a supported
        /// backend from @ref DNDS::DeviceBackend "DeviceBackend"; see @ref DNDS::DeviceStorage "DeviceStorage".hpp.
        void to_device(DeviceBackend backend = DeviceBackend::Host)
        {
            if constexpr (_dataLayout == CSR)
                DNDS_check_throw_info(IfCompressed(),
                                      fmt::format("CSR need compressing before to_device: {}",
                                                  this->getObjectIdentity(GetArrayName())));
            _data.to_device(backend);
            if (_pRowStart)
                _pRowStart->to_device(backend);
            if (_pRowSizes)
                _pRowSizes->to_device(backend);
            deviceBackend = _data.device();
        }

        /// @brief Release any device-side mirror of this array's buffers.
        void clear_device()
        {
            _data.clear_device();
            if (_pRowStart)
                _pRowStart->clear_device();
            if (_pRowSizes)
                _pRowSizes->clear_device();

            deviceBackend = DeviceBackend::Unknown;
        }

        template <DeviceBackend B>
        using t_deviceView = ArrayDeviceView<B, T, _row_size, _row_max, _align>;

        template <DeviceBackend B>
        using t_deviceViewConst = ArrayDeviceView<B, const T, _row_size, _row_max, _align>;

        /// @brief Mutable device-callable view (`Eigen::Map`-style row access on GPU).
        /// @tparam B Device backend; must either match the array's current device
        ///           or be `DeviceBackend::Host` (which yields a host-backed view).
        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            DNDS_check_throw_info((this->deviceBackend == B &&
                                   B != DeviceBackend::Unknown) ||
                                      (B == DeviceBackend::Host),
                                  fmt::format("{}: not on device {}, currently on {}",
                                              this->getObjectIdentity(GetArrayName()),
                                              device_backend_name(B),
                                              device_backend_name(this->deviceBackend)));

            return ArrayDeviceView_build<B, T, _row_size, _row_max, _align>(
                _size, _data.data(), _data.size(),
                _pRowStart ? _pRowStart->data() : nullptr, _pRowStart ? _pRowStart->size() : 0,
                _pRowSizes ? _pRowSizes->data() : nullptr, _pRowSizes ? _pRowSizes->size() : 0,
                _row_size_dynamic,
                _data.dataDevice(),
                _pRowStart ? _pRowStart->dataDevice() : nullptr,
                _pRowSizes ? _pRowSizes->dataDevice() : nullptr);
        }

        /// @brief Const device-callable view. See non-const overload.
        template <DeviceBackend B>
        t_deviceViewConst<B> deviceView() const
        {
            DNDS_check_throw_info((this->deviceBackend == B &&
                                   B != DeviceBackend::Unknown) ||
                                      (B == DeviceBackend::Host),
                                  fmt::format("{}: not on device {}, currently on {}",
                                              this->getObjectIdentity(GetArrayName()),
                                              device_backend_name(B),
                                              device_backend_name(this->deviceBackend)));

            return ArrayDeviceView_build<B, const T, _row_size, _row_max, _align>(
                _size, _data.data(), _data.size(),
                _pRowStart ? _pRowStart->data() : nullptr, _pRowStart ? _pRowStart->size() : 0,
                _pRowSizes ? _pRowSizes->data() : nullptr, _pRowSizes ? _pRowSizes->size() : 0,
                _row_size_dynamic,
                _data.dataDevice(),
                _pRowStart ? _pRowStart->dataDevice() : nullptr,
                _pRowSizes ? _pRowSizes->dataDevice() : nullptr);
        }

        /// @brief Current device backend the data is mirrored on, or @ref Unknown if host-only.
        [[nodiscard]] DeviceBackend device() const
        {
            return this->deviceBackend;
        }

        /// @brief Random-access iterator over rows for a given device backend.
        /// @details `operator*` yields a @ref DNDS::RowView "RowView" `{pointer, rowSize}`. Used by
        /// `std::transform` / CUDA kernels that sweep over rows.
        template <DeviceBackend B>
        class iterator : public ArrayIteratorBase<iterator<B>>
        {
        public:
            using view_type = t_deviceView<B>;
            using t_base_iter = ArrayIteratorBase<iterator<B>>;
            using typename t_base_iter::difference_type;
            using reference = typename view_type::RowView;
            using iterator_category = std::random_access_iterator_tag;

        protected:
            view_type view;

        public:
            auto getView() const { return view; }
            DNDS_DEVICE_CALLABLE iterator(const iterator &) = default;
            DNDS_DEVICE_CALLABLE ~iterator() = default;
            DNDS_DEVICE_CALLABLE iterator(const view_type &n_view, index n_iRow) : view(n_view), t_base_iter(n_iRow)
            {
            }

            DNDS_DEVICE_CALLABLE reference operator*() const { return {view.operator[](this->iRow), view.RowSize(this->iRow)}; }
            DNDS_DEVICE_CALLABLE reference operator*() { return {view.operator[](this->iRow), view.RowSize(this->iRow)}; }
        };

        /// @brief Iterator to the first row, viewed on device backend `B`.
        template <DeviceBackend B>
        iterator<B> begin()
        {
            return {deviceView<B>(), 0};
        }

        /// @brief Iterator one past the last row, viewed on device backend `B`.
        template <DeviceBackend B>
        iterator<B> end()
        {
            return {deviceView<B>(), this->Size()};
        }
    };

}

#endif
