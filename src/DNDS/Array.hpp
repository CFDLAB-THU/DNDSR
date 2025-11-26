#pragma once
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

namespace DNDS
{

    /**
     * @brief 2D var-len data container template
     * @details
     * ## Array's types
     * |                         | _row_size>=0                         | _row_size==DynamicSize              | _row_size==NonUniformSize |
     * | ---                     |          ---                         |                    ---              |                       --- |
     * |_row_max>=0              |  TABLE_StaticFixed                   |  TABLE_Fixed                        |   TABLE_StaticMax         |
     * |_row_max==DynamicSize    |  TABLE_StaticFixed _row_max ignored  |  TABLE_Fixed  _row_max ignored      |   TABLE_Max               |
     * |_row_max==NonUniformSize |  TABLE_StaticFixed _row_max ignored  |  TABLE_Fixed  _row_max ignored      |   CSR                     |
     *
     * @todo //TODO implement align feature
     *
     *
     * @tparam T
     * @tparam _row_size
     * @tparam _row_max
     * @tparam _align
     */
    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class Array : public ArrayLayout<T, _row_size, _row_max, _align>
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
        t_pRowStart getRowStart() { return _pRowStart; }
        t_pRowSizes getRowSizes() { return _pRowSizes; }

    public:
        // default constructor using default:
        Array() = default;

        // TODO: constructors
        // TODO: A indexer-copying build method:
        // TODO: for CSR: c->c, u->u
        // TODO: for intertype: CSR->Max, Max->CSR ...

        // read size
        [[nodiscard]] index Size() const { return _size; }

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

        [[nodiscard]] rowsize RowSizeMax() const
        {
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
                return _dataLayout == TABLE_Max ? _row_size_dynamic : rm;
            else
                DNDS_assert_info(false, "invalid call");
        }

        [[nodiscard]] rowsize RowSizeField() const
        {
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
                return this->RowSizeMax();
            else if constexpr (_dataLayout == TABLE_Fixed || _dataLayout == TABLE_StaticFixed)
                return this->RowSize();
            else
                DNDS_assert_info(false, "invalid call");
        }

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
        [[nodiscard]] bool IfCompressed() const
        {
            static_assert(_dataLayout == CSR, "invalid call");
            return IfCompressed_();
        }

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

        void Compress()
        {
            if constexpr (_dataLayout == CSR)
                CSRCompress();
        }
        void Decompress()
        {
            if constexpr (_dataLayout == CSR)
                CSRDecompress();
        }

        t_Data &RawDataVector()
        {
            if constexpr (_dataLayout == CSR)
                DNDS_check_throw(IfCompressed());
            return _data;
        }

        /**
         * @brief resize invalidates all data and aux, and resets the sizes info to 0 for max
         *
         * @param nSize
         * @param nRow_size_dynamic
         * @return std::enable_if_t<
         * _dataLayout == TABLE_Fixed ||
         * _dataLayout == TABLE_StaticFixed ||
         * _dataLayout == TABLE_Max ||
         * _dataLayout == TABLE_StaticMax,
         * void>
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
         * @brief resize one row
         * valid only for non-uniform
         *
         * @param iRow
         * @param nRowSize
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

        t_View view()
        {
            return t_View(_size, _data.data(), _data.size(),
                          _pRowStart ? _pRowStart->data() : nullptr, _pRowStart ? _pRowStart->size() : 0,
                          _pRowSizes ? _pRowSizes->data() : nullptr, _pRowSizes ? _pRowSizes->size() : 0,
                          _row_size_dynamic,
                          IfCompressed_(), IfCompressed_() ? nullptr : &_dataUncompressed);
        }

        const T &at(index iRow, rowsize iCol) const
        {
            DNDS_assert_info(iRow < _size && iRow >= 0,
                             fmt::format(
                                 "query position i[{}] out of range [0, {}), sig--{}",
                                 iRow, _size, GetArrayName()));
            DNDS_assert_info(iCol < RowSize(iRow) && iCol >= 0,
                             fmt::format(
                                 "query position j[{}] out of range [0, {}), sig--{}",
                                 iCol, RowSize(iRow), GetArrayName()));
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

        T &operator()(index iRow, rowsize iCol = 0)
        {
            return const_cast<T &>(at(iRow, iCol));
        }

        const T &operator()(index iRow, rowsize iCol = 0) const
        {
            return at(iRow, iCol);
        }

        /**
         * @brief iRow could be past-the-end to query past-the-end position pointer
         *
         * @param iRow
         * @return T*
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

        const T *operator[](index iRow) const
        {
            return static_cast<const T *>(const_cast<self_type *>(this)->operator[](iRow));
        }

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

        size_t DataSize() const
        {
            if (this->Size() == 0)
                return 0;
            if constexpr (_dataLayout == CSR)
                DNDS_assert_info(this->IfCompressed(), "CSR must be compressed to get DataSize()");
            return _data.size();
        }

        size_t DataSizeBytes() const
        {
            return this->DataSize() * sizeof_T;
        }

        size_t FullSizeBytes() const
        {
            size_t b = this->DataSize() * sizeof_T;
            if (_pRowStart)
                b += _pRowStart->size() * sizeof(index);
            if (_pRowSizes)
                b += _pRowSizes->size() * sizeof(rowsize);
            return b;
        }

        std::size_t hash()
        {
            std::size_t hashData;
            if constexpr (_dataLayout == CSR)
            {
                if (IfCompressed())
                    hashData = vector_hash<T>()(_data);
                else
                    hashData = vector_hash<std::vector<T>>()(_dataUncompressed);
            }
            else
                hashData = vector_hash<T>()(_data);
            std::size_t hashSize = 0;
            if (_pRowSizes)
                hashSize = vector_hash<rowsize>()(*_pRowSizes);
            if (_pRowStart)
                hashSize = vector_hash<index>()(*_pRowStart);
            return array_hash<std::size_t, 3>()(std::array<std::size_t, 3>{std::size_t(_size), hashSize, hashData});
        }

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

        static constexpr DataLayout GetDataLayoutStatic() { return _dataLayout; }
        constexpr DataLayout GetDataLayout() { return _dataLayout; }

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

        void CopyData(const self_type &R)
        {
            this->clone(R);
            // non-trivial copy call: unique_ptr
        }

        self_type &operator=(const self_type &R)
        {
            if (this == &R)
                return *this;
            this->clone(R);
            return *this;
        }

        // copy constructor
        Array(const self_type &R)
        {
            this->clone(R);
        }

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
                serializerP->WriteIndexVector("data", _data, offset);
            else if constexpr (std::is_same_v<T, real>)
            {
                if (!std::dynamic_pointer_cast<Serializer::SerializerJSON>(serializerP))
                    serializerP->WriteRealVector("data", _data, offset);
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
                index bufferSize{0};
                serializerP->ReadUint8Array("data", nullptr, bufferSize, offset);
                DNDS_check_throw(bufferSize % sizeof_T == 0);
                _data.resize(bufferSize / sizeof_T);
                serializerP->ReadUint8Array("data", (uint8_t *)_data.data(), bufferSize, offset);
                offset.CheckMultipleOf(sizeof_T);
                offset = offset / sizeof_T;
            };

            if constexpr (std::is_same_v<T, index>)
                serializerP->ReadIndexVector("data", _data, offset);
            else if constexpr (std::is_same_v<T, real>)
            {
                if (!std::dynamic_pointer_cast<Serializer::SerializerJSON>(serializerP))
                    serializerP->ReadRealVector("data", _data, offset);
                else
                    treatAsBytes();
            }
            else
                treatAsBytes();
        }

        void WriteSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name, Serializer::ArrayGlobalOffset offset)
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
                serializerP->WriteSharedIndexVector("pRowStart", _pRowStart, offset);
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

        void ReadSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name, Serializer::ArrayGlobalOffset &offset)
        {
            auto cwd = serializerP->GetCurrentPath();
            // serializerP->CreatePath(name); //! if you create, all data will be erased
            serializerP->GoToPath(name);

            std::string array_sigRead;
            serializerP->ReadString("array_sig", array_sigRead);
            //! TODO: parse the sizes and correctly handle dynamic reading
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
            // if (_size == 0) //! cannot do this, collective calls!
            //     return;
            if constexpr (_dataLayout == CSR)
            {
                serializerP->ReadSharedIndexVector("pRowStart", _pRowStart, offset);
                // todo: make the data inside the file correspond to full disp? is that necessary?
            }
            else if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
            {
                serializerP->ReadSharedRowsizeVector("pRowSizes", _pRowSizes, offset);
            }
            else // fixed
            {
            }
            // doing data
            {
                Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_Unknown; // todo: utilize results from input offset (non CSR) or pRowStart data
                this->__ReadSerializerData(serializerP, offsetV);
                if constexpr (_dataLayout == TABLE_StaticFixed || _dataLayout == TABLE_Fixed)
                {
                    offsetV.CheckMultipleOf(this->RowSize());
                    offset = offsetV / this->RowSize(); // to make sure offset in the output is valid (on the elements)
                }
            }
            // TODO: check data validity

            serializerP->GoToPath(cwd);
        }

    public:
        void to_host()
        {
            if constexpr (_dataLayout == CSR)
                DNDS_check_throw_info(IfCompressed(), "CSR need compressing before to_host");
            _data.to_host();
            deviceBackend = DeviceBackend::Unknown;
        }

        void to_device(DeviceBackend backend = DeviceBackend::Host)
        {
            if constexpr (_dataLayout == CSR)
                DNDS_check_throw_info(IfCompressed(), "CSR need compressing before to_device");
            _data.to_device(backend);
            if (_pRowStart)
                _pRowStart->to_device(backend);
            if (_pRowSizes)
                _pRowSizes->to_device(backend);
            deviceBackend = _data.deviceStorage->backend();
        }

        void clear_device()
        {
            _data.deviceStorage.reset();
            if (_pRowStart)
                _pRowStart->deviceStorage.reset();
            if (_pRowSizes)
                _pRowSizes->deviceStorage.reset();

            deviceBackend = DeviceBackend::Unknown;
        }

        template <DeviceBackend B>
        using t_deviceView = ArrayDeviceView<B, T, _row_size, _row_max, _align>;

        template <DeviceBackend B>
        using t_deviceViewConst = ArrayDeviceView<B, const T, _row_size, _row_max, _align>;

        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            DNDS_check_throw_info((this->deviceBackend == B &&
                                   B != DeviceBackend::Unknown) ||
                                      (B == DeviceBackend::Host),
                                  "not on this device: " + std::string(device_backend_name(B)));

            return ArrayDeviceView_build<B, T, _row_size, _row_max, _align>(
                _size, _data.data(), _data.size(),
                _pRowStart ? _pRowStart->data() : nullptr, _pRowStart ? _pRowStart->size() : 0,
                _pRowSizes ? _pRowSizes->data() : nullptr, _pRowSizes ? _pRowSizes->size() : 0,
                _row_size_dynamic,
                _data.dataDevice(),
                _pRowStart ? _pRowStart->dataDevice() : nullptr,
                _pRowSizes ? _pRowSizes->dataDevice() : nullptr);
        }

        template <DeviceBackend B>
        t_deviceViewConst<B> deviceView() const
        {
            DNDS_check_throw_info((this->deviceBackend == B &&
                                   B != DeviceBackend::Unknown) ||
                                      (B == DeviceBackend::Host),
                                  "not on this device");

            return ArrayDeviceView_build<B, const T, _row_size, _row_max, _align>(
                _size, _data.data(), _data.size(),
                _pRowStart ? _pRowStart->data() : nullptr, _pRowStart ? _pRowStart->size() : 0,
                _pRowSizes ? _pRowSizes->data() : nullptr, _pRowSizes ? _pRowSizes->size() : 0,
                _row_size_dynamic,
                _data.dataDevice(),
                _pRowStart ? _pRowStart->dataDevice() : nullptr,
                _pRowSizes ? _pRowSizes->dataDevice() : nullptr);
        }

        [[nodiscard]] DeviceBackend device() const
        {
            return this->deviceBackend;
        }

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

#endif
