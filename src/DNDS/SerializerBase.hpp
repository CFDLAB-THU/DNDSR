#pragma once
/// @file SerializerBase.hpp
/// @brief Base types and abstract interface for array serialization.

#include "Defines.hpp"
#include "MPI.hpp"
#include <limits>
#include <set>
#include "DeviceStorage.hpp"
#include "Vector.hpp"

namespace DNDS::Serializer
{
    /// @brief Sentinel: "offset = Parts", indicating each rank writes its own slab.
    static const index Offset_Parts = -1;
    /// @brief Sentinel: "offset = One", indicating the first rank owns the whole dataset.
    static const index Offset_One = -2;
    /// @brief Sentinel: "even-split read" (see @ref ArrayGlobalOffset_EvenSplit).
    static const index Offset_EvenSplit = -3;
    /// @brief Sentinel: offset unknown / uninitialised.
    static const index Offset_Unknown = UnInitIndex;

    /**
     * @brief Describes one rank's window into a globally-distributed dataset.
     *
     * @details Stores a `(localSize, globalOffset)` pair. The offset is
     * overloaded to carry the special sentinels @ref Offset_Parts / @ref Offset_One /
     * @ref Offset_EvenSplit / @ref Offset_Unknown (negative values). Multiplication /
     * division are defined so that byte offsets can be derived from element
     * offsets (`offset * sizeof(T)`) with overflow guards.
     */
    class ArrayGlobalOffset
    {
        index _size{0};
        index _offset{0};

    public:
        static_assert(UnInitIndex < 0);
        /// @brief Construct with explicit local size and global offset.
        ArrayGlobalOffset(index __size, index __offset) : _size(__size), _offset(__offset) {}

        /// @brief Local size this rank owns (in element units of the caller's choosing).
        [[nodiscard]] index size() const { return _size; }
        /// @brief Global offset of this rank's data (or a sentinel value, see
        /// @ref Offset_Parts etc.).
        [[nodiscard]] index offset() const { return _offset; }

        /// @brief Scale the descriptor's element count by `R` (and the offset
        /// too if it is a real offset, not a sentinel). Used for byte-level
        /// offsets: `elemOffset * sizeof(T)`.
        ArrayGlobalOffset operator*(index R) const
        {
            if (_offset >= 0)
            {
                DNDS_assert_info(R == 0 || _size <= std::numeric_limits<index>::max() / R,
                                 "Overflow in ArrayGlobalOffset size multiplication");
                DNDS_assert_info(R == 0 || _offset <= std::numeric_limits<index>::max() / R,
                                 "Overflow in ArrayGlobalOffset offset multiplication");
                return ArrayGlobalOffset{_size * R, _offset * R};
            }
            else
                return ArrayGlobalOffset{_size * R, _offset};
        }

        /// @brief Inverse of #operator*. Real-offset descriptors scale both
        /// size and offset; sentinel descriptors only scale the size.
        ArrayGlobalOffset operator/(index R) const
        {
            if (_offset >= 0)
                return ArrayGlobalOffset{_size / R, _offset / R};
            else
                return ArrayGlobalOffset{_size / R, _offset};
        }

        /// @brief Assert that both size and offset are multiples of `R`.
        /// @details Used when translating between element and byte offsets.
        void CheckMultipleOf(index R) const
        {
            if (_offset >= 0)
            {
                DNDS_assert_info(_size % R == 0, fmt::format("_size [{}] must be multiple of R [{}]", _size, R));
                DNDS_assert_info(_offset % R == 0, fmt::format("_offset [{}] must be multiple of R [{}]", _offset, R));
            }
        }

        /// @brief Equality: sentinel offsets compare only by offset,
        /// real-offset descriptors compare by the full `(size, offset)` pair.
        bool operator==(const ArrayGlobalOffset &other) const
        {
            if (_offset >= 0)
                return _size == other._size && _offset == other._offset;
            else
                return _offset == other._offset;
        }

        /// @brief Pretty-printed representation for logging.
        operator std::string() const
        {
            return fmt::format("ArrayGlobalOffset{{size: {}, offset: {}}}", _size, _offset);
        }

        /// @brief Whether this descriptor carries a real distributed offset
        /// (rather than a sentinel like @ref Offset_Parts).
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

    /// @brief Sentinel: offset unknown / uninitialised.
    static const ArrayGlobalOffset ArrayGlobalOffset_Unknown = ArrayGlobalOffset{0, Offset_Unknown};
    /// @brief Sentinel: rank 0 owns the whole dataset.
    static const ArrayGlobalOffset ArrayGlobalOffset_One = ArrayGlobalOffset{0, Offset_One};
    /// @brief Sentinel: each rank writes its slab; serializer infers offsets via MPI_Scan.
    static const ArrayGlobalOffset ArrayGlobalOffset_Parts = ArrayGlobalOffset{0, Offset_Parts};
    /// @brief Sentinel: even-split read (each rank reads ~N_global/nRanks rows
    /// starting at `rank * N_global / nRanks`). Used during repartitioned restarts.
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
    protected:
        /// Shared-pointer deduplication map: raw pointer -> (kept-alive shared_ptr, H5/JSON path).
        /// By storing the shared_ptr itself we prevent the pointed-to memory from being
        /// freed and its address reused, which would cause false dedup hits.
        std::map<void *, std::pair<std::shared_ptr<void>, std::string>> ptr_2_pth;

        /// Reverse map for read-side dedup: path -> raw pointer to the ssp local variable.
        std::map<std::string, void *> pth_2_ssp;

        /// Check if a shared pointer was already written; if so return its path.
        template <class T>
        bool dedupLookup(const ssp<T> &v, std::string &outPath)
        {
            auto it = ptr_2_pth.find(v.get());
            if (it != ptr_2_pth.end())
            {
                outPath = it->second.second;
                return true;
            }
            return false;
        }

        /// Register a shared pointer after writing its data.
        template <class T>
        void dedupRegister(const ssp<T> &v, const std::string &path)
        {
            ptr_2_pth[v.get()] = {std::static_pointer_cast<void>(v), path};
        }

        /// Clear all dedup state (call on CloseFile).
        void dedupClear()
        {
            ptr_2_pth.clear();
            pth_2_ssp.clear();
        }

    public:
        virtual ~SerializerBase(); // define in CPP
        /// @brief Open a backing file (H5 file or JSON file depending on subclass).
        /// @param read `true` for reading, `false` for writing.
        virtual void OpenFile(const std::string &fName, bool read) = 0;
        /// @brief Close the backing file, flushing buffers.
        virtual void CloseFile() = 0;
        /// @brief Create a sub-path (H5 group / JSON object) at the current location.
        virtual void CreatePath(const std::string &p) = 0;
        /// @brief Navigate to an existing path. Supports `/` -separated segments.
        virtual void GoToPath(const std::string &p) = 0;
        /// @brief Whether this serializer is per-rank (JSON-file-per-rank) or
        /// collective (shared H5 file). Controls which API entry points are used.
        virtual bool IsPerRank() = 0;
        /// @brief String form of the current path.
        virtual std::string GetCurrentPath() = 0;
        /// @brief Names of direct children of the current path.
        virtual std::set<std::string> ListCurrentPath() = 0;
        /// @brief Rank index cached by the serializer (relevant for collective I/O).
        virtual int GetMPIRank() = 0;
        /// @brief Rank count cached by the serializer.
        virtual int GetMPISize() = 0;
        /// @brief MPI context the serializer was opened with.
        virtual const MPIInfo& getMPI() = 0;

        /// @brief Write a scalar int under `name` at the current path.
        virtual void WriteInt(const std::string &name, int v) = 0;
        /// @brief Write a scalar #index under `name`.
        virtual void WriteIndex(const std::string &name, index v) = 0;
        /// @brief Write a scalar #real under `name`.
        virtual void WriteReal(const std::string &name, real v) = 0;
        /// @brief Write a UTF-8 string under `name`.
        virtual void WriteString(const std::string &name, const std::string &v) = 0;

        /// @brief Write an index vector (collective for H5). `offset` carries the
        /// distribution mode (@ref Parts, explicit offset, etc.).
        virtual void WriteIndexVector(const std::string &name, const std::vector<index> &v, ArrayGlobalOffset offset) = 0;
        /// @brief Write a rowsize vector (collective for H5).
        virtual void WriteRowsizeVector(const std::string &name, const std::vector<rowsize> &v, ArrayGlobalOffset offset) = 0;
        /// @brief Write a real vector (collective for H5).
        virtual void WriteRealVector(const std::string &name, const std::vector<real> &v, ArrayGlobalOffset offset) = 0;
        /// @brief Write a shared index vector; deduplicated across multiple writes
        /// that share the same `shared_ptr`.
        virtual void WriteSharedIndexVector(const std::string &name, const ssp<host_device_vector<index>> &v, ArrayGlobalOffset offset) = 0;
        /// @brief Write a shared rowsize vector; deduplicated across multiple writes.
        virtual void WriteSharedRowsizeVector(const std::string &name, const ssp<host_device_vector<rowsize>> &v, ArrayGlobalOffset offset) = 0;
        /// @brief Write a raw byte buffer under `name`. `offset.isDist()` = true
        /// means the caller provides the exact per-rank slab; otherwise the
        /// buffer is treated according to the offset's sentinel.
        virtual void WriteUint8Array(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset) = 0;

        /**
         * @brief Write a per-rank index vector (replicated name, independent values).
         * @details Every rank writes its own vector under `name`; in the H5 case
         * each rank's slab is placed in a separate dataset.
         * @param name  Dataset name (identical on every rank).
         * @param v     Rank-local vector; size may differ between ranks.
         */
        virtual void WriteIndexVectorPerRank(const std::string &name, const std::vector<index> &v) = 0;
        // virtual void WriteIndexVectorParallel(const std::string &name, const host_device_vector<index> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteRowsizeVectorParallel(const std::string &name, const host_device_vector<rowsize> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteRealVectorParallel(const std::string &name, const host_device_vector<real> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteSharedIndexVectorParallel(const std::string &name, const ssp<host_device_vector<index>> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteSharedRowsizeVectorParallel(const std::string &name, const ssp<host_device_vector<rowsize>> &v, ArrayGlobalOffset offset) = 0;
        // virtual void WriteUint8ArrayParallel(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset) = 0;

        /// @brief Read a scalar int into `v`.
        virtual void ReadInt(const std::string &name, int &v) = 0;
        /// @brief Read a scalar #index into `v`.
        virtual void ReadIndex(const std::string &name, index &v) = 0;
        /// @brief Read a scalar #real into `v`.
        virtual void ReadReal(const std::string &name, real &v) = 0;
        /// @brief Read a UTF-8 string into `v`.
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
