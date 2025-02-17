#pragma once

#include "Defines.hpp"
#include "MPI.hpp"
namespace DNDS::Serializer
{
    static const index Offset_Parts = -1;
    static const index Offset_One = -2;
    static const index Offset_Unkown = UnInitIndex;

    class ArrayGlobalOffset
    {
        index _size{0};
        index _offset{0};

    public:
        static_assert(UnInitIndex < 0);
        ArrayGlobalOffset(index __size, index __offset) : _size(__size), _offset(__offset) {}

        [[nodiscard]] index size() const { return _size; }
        [[nodiscard]] index offset() const { return _offset; }

        ArrayGlobalOffset operator*(index R) const
        {
            if (_offset >= 0)
                return ArrayGlobalOffset{_size * R, _offset * R};
            else
                return ArrayGlobalOffset{_size * R, _offset};
            // todo: check on overflow in multiplication
        }

        ArrayGlobalOffset operator/(index R) const
        {
            if (_offset >= 0)
                return ArrayGlobalOffset{_size / R, _offset / R};
            else
                return ArrayGlobalOffset{_size / R, _offset};
        }

        void CheckMultipleOf(index R) const
        {
            if (_offset >= 0)
            {
                DNDS_assert_info(_size % R == 0, fmt::format("_size [{}] must be multiple of R [{}]", _size, R));
                DNDS_assert_info(_offset % R == 0, fmt::format("_offset [{}] must be multiple of R [{}]", _offset, R));
            }
        }

        bool operator==(const ArrayGlobalOffset &other) const
        {
            if (_offset >= 0)
                return _size == other._size && _offset == other._offset;
            else
                return _offset == other._offset;
        }

        operator std::string() const
        {
            return fmt::format("ArrayGlobalOffset{{size: {}, offset: {}}}", _size, _offset);
        }

        [[nodiscard]] bool isDist() const
        {
            return _offset >= 0;
        }

        friend std::ostream &operator<<(std::ostream &o, const ArrayGlobalOffset &v)
        {
            o << (std::string)(v);
            return o;
        }
    };

    static const ArrayGlobalOffset ArrayGlobalOffset_Unknown = ArrayGlobalOffset{0, Offset_Unkown};
    static const ArrayGlobalOffset ArrayGlobalOffset_One = ArrayGlobalOffset{0, Offset_One};
    static const ArrayGlobalOffset ArrayGlobalOffset_Parts = ArrayGlobalOffset{0, Offset_Parts};

    class SerializerBase
    {

    public:
        virtual ~SerializerBase() = default;
        virtual void OpenFile(const std::string &fName, bool read) = 0;
        virtual void CloseFile() = 0;
        virtual void CreatePath(const std::string &p) = 0;
        virtual void GoToPath(const std::string &p) = 0;
        virtual bool IsPerRank() = 0;
        virtual std::string GetCurrentPath() = 0;

        virtual void WriteInt(const std::string &name, int v) = 0;
        virtual void WriteIndex(const std::string &name, index v) = 0;
        virtual void WriteReal(const std::string &name, real v) = 0;
        virtual void WriteString(const std::string &name, const std::string &v) = 0;

        virtual void WriteIndexVector(const std::string &name, const std::vector<index> &v, ArrayGlobalOffset offset) = 0;
        virtual void WriteRowsizeVector(const std::string &name, const std::vector<rowsize> &v, ArrayGlobalOffset offset) = 0;
        virtual void WriteRealVector(const std::string &name, const std::vector<real> &v, ArrayGlobalOffset offset) = 0;
        virtual void WriteSharedIndexVector(const std::string &name, const ssp<std::vector<index>> &v, ArrayGlobalOffset offset) = 0;
        virtual void WriteSharedRowsizeVector(const std::string &name, const ssp<std::vector<rowsize>> &v, ArrayGlobalOffset offset) = 0;
        virtual void WriteUint8Array(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset) = 0;

        /**
         * @brief size of v need to be identical across ranks
         *
         * @param name
         * @param v
         */
        virtual void WriteIndexVectorPerRank(const std::string &name, const std::vector<index> &v) = 0;
        // virtual void WriteIndexVectorParallel(const std::string &name, const std::vector<index> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteRowsizeVectorParallel(const std::string &name, const std::vector<rowsize> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteRealVectorParallel(const std::string &name, const std::vector<real> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteSharedIndexVectorParallel(const std::string &name, const ssp<std::vector<index>> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteSharedRowsizeVectorParallel(const std::string &name, const ssp<std::vector<rowsize>> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteUint8ArrayParallel(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset) = 0;

        virtual void ReadInt(const std::string &name, int &v) = 0;
        virtual void ReadIndex(const std::string &name, index &v) = 0;
        virtual void ReadReal(const std::string &name, real &v) = 0;
        virtual void ReadString(const std::string &name, std::string &v) = 0;

        virtual void ReadIndexVector(const std::string &name, std::vector<index> &v, ArrayGlobalOffset &offset) = 0;
        virtual void ReadRowsizeVector(const std::string &name, std::vector<rowsize> &v, ArrayGlobalOffset &offset) = 0;
        virtual void ReadRealVector(const std::string &name, std::vector<real> &v, ArrayGlobalOffset &offset) = 0;
        virtual void ReadSharedIndexVector(const std::string &name, ssp<std::vector<index>> &v, ArrayGlobalOffset &offset) = 0;
        virtual void ReadSharedRowsizeVector(const std::string &name, ssp<std::vector<rowsize>> &v, ArrayGlobalOffset &offset) = 0;
        /**
         * @brief
         * @param data if data == nullptr, only get the size not reading any data
         */
        virtual void ReadUint8Array(const std::string &name, uint8_t *data, index &size, ArrayGlobalOffset &offset) = 0;

        // virtual void ReadIndexVectorParallel(const std::string &name, const std::vector<index> &v, ArrayGlobalOffset offset) = 0;
        // virtual void ReadRowsizeVectorParallel(const std::string &name, const std::vector<rowsize> &v, ArrayGlobalOffset offset) = 0;
        // virtual void ReadRealVectorParallel(const std::string &name, const std::vector<real> &v, ArrayGlobalOffset offset) = 0;
        // virtual void ReadSharedIndexVectorParallel(const std::string &name, const ssp<std::vector<index>> &v, ArrayGlobalOffset offset) = 0;
        // virtual void ReadSharedRowsizeVectorParallel(const std::string &name, const ssp<std::vector<rowsize>> &v, ArrayGlobalOffset offset) = 0;
        // virtual void ReadUint8ArrayParallel(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset) = 0;
    };

    using SerializerBaseSSP = ssp<SerializerBase>;
}
