#pragma once
/// @file SerializerBase.hpp
/// @brief Base types and abstract interface for array serialization.

#include "Defines.hpp"
#include "MPI.hpp"
#include <set>
#include "DeviceStorage.hpp"
#include "Vector.hpp"

namespace DNDS::Serializer
{
    static const index Offset_Parts = -1;
    static const index Offset_One = -2;
    static const index Offset_EvenSplit = -3;
    static const index Offset_Unkown = UnInitIndex;

    /// @brief Describes a rank's portion of a globally-distributed array (local size + global offset).
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
    /// @brief Even-split read: each rank reads ~N_global/nRanks rows starting at rank * N_global / nRanks.
    static const ArrayGlobalOffset ArrayGlobalOffset_EvenSplit = ArrayGlobalOffset{0, Offset_EvenSplit};

    /// @brief Abstract interface for reading/writing scalars, vectors, and byte arrays.
    ///
    /// ## Collective semantics (SerializerH5)
    ///
    /// For the HDF5 implementation, all Read/Write vector and array methods are
    /// **MPI-collective**: every rank must call the same method in the same order,
    /// even if a rank reads/writes 0 elements. Failing to participate causes a
    /// hang because HDF5 collective I/O synchronizes across the communicator.
    ///
    /// ## Zero-size reads
    ///
    /// Some ranks may legitimately have 0 elements to read (e.g., when
    /// `nGlobal < nRanks` in an even-split read). The Read*Vector and
    /// ReadShared*Vector implementations handle this by passing a dummy
    /// non-null pointer to the internal HDF5 read when the output container
    /// is empty, so that `std::vector<>::data()` returning nullptr does not
    /// cause the rank to skip the collective H5Dread call.
    ///
    /// @warning **ReadUint8Array** uses an explicit two-pass pattern: the
    /// caller first calls with `data == nullptr` to query the size, then
    /// calls again with a buffer. When the queried size is 0, the caller
    /// must still pass a non-null `data` pointer on the second call so
    /// that the collective H5Dread is not skipped. Use a stack dummy:
    /// @code
    ///   uint8_t dummy;
    ///   ser->ReadUint8Array(name, bufferSize == 0 ? &dummy : buf, bufferSize, offset);
    /// @endcode
    class SerializerBase
    {

    public:
        virtual ~SerializerBase(); // define in CPP
        virtual void OpenFile(const std::string &fName, bool read) = 0;
        virtual void CloseFile() = 0;
        virtual void CreatePath(const std::string &p) = 0;
        virtual void GoToPath(const std::string &p) = 0;
        virtual bool IsPerRank() = 0;
        virtual std::string GetCurrentPath() = 0;
        virtual std::set<std::string> ListCurrentPath() = 0;
        virtual int GetMPIRank() = 0;
        virtual int GetMPISize() = 0;
        virtual const MPIInfo& getMPI() = 0;

        virtual void WriteInt(const std::string &name, int v) = 0;
        virtual void WriteIndex(const std::string &name, index v) = 0;
        virtual void WriteReal(const std::string &name, real v) = 0;
        virtual void WriteString(const std::string &name, const std::string &v) = 0;

        virtual void WriteIndexVector(const std::string &name, const std::vector<index> &v, ArrayGlobalOffset offset) = 0;
        virtual void WriteRowsizeVector(const std::string &name, const std::vector<rowsize> &v, ArrayGlobalOffset offset) = 0;
        virtual void WriteRealVector(const std::string &name, const std::vector<real> &v, ArrayGlobalOffset offset) = 0;
        virtual void WriteSharedIndexVector(const std::string &name, const ssp<host_device_vector<index>> &v, ArrayGlobalOffset offset) = 0;
        virtual void WriteSharedRowsizeVector(const std::string &name, const ssp<host_device_vector<rowsize>> &v, ArrayGlobalOffset offset) = 0;
        virtual void WriteUint8Array(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset) = 0;

        /**
         * @brief size of v need to be identical across ranks
         *
         * @param name
         * @param v
         */
        virtual void WriteIndexVectorPerRank(const std::string &name, const std::vector<index> &v) = 0;
        // virtual void WriteIndexVectorParallel(const std::string &name, const host_device_vector<index> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteRowsizeVectorParallel(const std::string &name, const host_device_vector<rowsize> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteRealVectorParallel(const std::string &name, const host_device_vector<real> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteSharedIndexVectorParallel(const std::string &name, const ssp<host_device_vector<index>> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteSharedRowsizeVectorParallel(const std::string &name, const ssp<host_device_vector<rowsize>> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteUint8ArrayParallel(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset) = 0;

        virtual void ReadInt(const std::string &name, int &v) = 0;
        virtual void ReadIndex(const std::string &name, index &v) = 0;
        virtual void ReadReal(const std::string &name, real &v) = 0;
        virtual void ReadString(const std::string &name, std::string &v) = 0;

        /// Read methods resize the output container and populate it.
        /// Internally these use a two-pass HDF5 pattern (size query, then data read).
        /// Both passes are collective. When the local size is 0, a dummy non-null
        /// pointer is passed to the second pass so the rank participates in H5Dread.
        /// @{
        virtual void ReadIndexVector(const std::string &name, std::vector<index> &v, ArrayGlobalOffset &offset) = 0;
        virtual void ReadRowsizeVector(const std::string &name, std::vector<rowsize> &v, ArrayGlobalOffset &offset) = 0;
        virtual void ReadRealVector(const std::string &name, std::vector<real> &v, ArrayGlobalOffset &offset) = 0;
        virtual void ReadSharedIndexVector(const std::string &name, ssp<host_device_vector<index>> &v, ArrayGlobalOffset &offset) = 0;
        virtual void ReadSharedRowsizeVector(const std::string &name, ssp<host_device_vector<rowsize>> &v, ArrayGlobalOffset &offset) = 0;
        /// @}

        /// @brief Two-pass byte array read.
        ///
        /// Pass 1 (data == nullptr): queries the local element count into `size`
        /// and resolves `offset`.
        /// Pass 2 (data != nullptr): reads `size` bytes into `data`.
        ///
        /// @warning When `size` is 0 after pass 1, the caller **must** still pass
        /// a non-null `data` pointer on pass 2 so the rank participates in the
        /// collective HDF5 read. Use a stack dummy (`uint8_t dummy; ... &dummy`).
        virtual void ReadUint8Array(const std::string &name, uint8_t *data, index &size, ArrayGlobalOffset &offset) = 0;

        // virtual void ReadIndexVectorParallel(const std::string &name, const host_device_vector<index> &v, ArrayGlobalOffset offset) = 0;
        // virtual void ReadRowsizeVectorParallel(const std::string &name, const host_device_vector<rowsize> &v, ArrayGlobalOffset offset) = 0;
        // virtual void ReadRealVectorParallel(const std::string &name, const host_device_vector<real> &v, ArrayGlobalOffset offset) = 0;
        // virtual void ReadSharedIndexVectorParallel(const std::string &name, const ssp<host_device_vector<index>> v, ArrayGlobalOffset offset) = 0;
        // virtual void ReadSharedRowsizeVectorParallel(const std::string &name, const ssp<host_device_vector<rowsize>> v, ArrayGlobalOffset offset) = 0;
        // virtual void ReadUint8ArrayParallel(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset) = 0;
    };

    using SerializerBaseSSP = ssp<SerializerBase>;
}
