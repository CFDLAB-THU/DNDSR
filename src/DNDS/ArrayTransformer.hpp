#pragma once
/// @file ArrayTransformer.hpp
/// @brief ParArray (MPI-aware array) and ArrayTransformer (ghost/halo communication).
/// @par Unit Test Coverage (test_ArrayTransformer.cpp, MPI np=1,2,4)
/// - ParArray: setMPI, Resize, createGlobalMapping, globalSize, AssertConsistent
/// - ArrayTransformer pull-based ghost: setFatherSon, createFatherGlobalMapping,
///   createGhostMapping (pull), createMPITypes, pullOnce
///   -- layouts: TABLE_StaticFixed, TABLE_Fixed, CSR, std::array compound type
/// - Persistent pull: initPersistentPull, startPersistentPull, waitPersistentPull,
///   clearPersistentPull (with father data update between pulls)
/// - BorrowGGIndexing: second array shares ghost mapping of first
/// - pushOnce: write to son, push back to father
/// @par Not Yet Tested
/// - Push-based createGhostMapping(pushingIndexLocal, pushStarts)
/// - Persistent push (initPersistentPush, startPersistentPush, etc.)
/// - clearMPITypes, clearGlobalMapping, clearGhostMapping (independent)
/// - reInitPersistentPullPush
/// - getFatherSonData(DeviceBackend), AssertDataType, setDataType

#include "Array.hpp"
#include "DNDS/Device/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
#include "IndexMapping.hpp"
#include "Profiling.hpp"
#include <utility>
#include "VectorUtils.hpp"

namespace DNDS
{
    /// @brief Shared pointer to a @ref DNDS::GlobalOffsetsMapping "GlobalOffsetsMapping" (globally replicated).
    using t_pLGlobalMapping = ssp<GlobalOffsetsMapping>;
    /// @brief Shared pointer to an @ref DNDS::OffsetAscendIndexMapping "OffsetAscendIndexMapping" (per-rank ghost layout).
    using t_pLGhostMapping = ssp<OffsetAscendIndexMapping>; // TODO: change to unique_ptr and modify corresponding copy constructor/assigner

    /**
     * @brief MPI-aware @ref DNDS::Array "Array": adds a communicator, rank, and global index mapping.
     *
     * @details Inherits everything from @ref DNDS::Array "Array" and layers on:
     *  - an @ref DNDS::MPIInfo "MPIInfo" `mpi` context;
     *  - a @ref DNDS::GlobalOffsetsMapping "GlobalOffsetsMapping" `pLGlobalMapping` that maps local row indices
     *    to the global index space (populated by #createGlobalMapping);
     *  - collective serialization I/O that coordinates writes / reads across ranks.
     *
     * Typical usage:
     * ```cpp
     * auto father = std::make_shared<ParArray<real, 5>>(mpi);
     * father->Resize(nLocal);
     * father->createGlobalMapping();         // collective
     * index nGlobal = father->globalSize();  // total rows across ranks
     * ```
     *
     * Ghost (halo) data is not managed here; pair with @ref DNDS::ArrayTransformer "ArrayTransformer" or
     * wrap in an @ref DNDS::ArrayPair "ArrayPair" for that.
     *
     * @sa ArrayTransformer, ArrayPair, docs/architecture/array_infrastructure.md.
     */
    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ParArray : public Array<T, _row_size, _row_max, _align>
    {
    public:
        using TArray = Array<T, _row_size, _row_max, _align>;
        using t_self = ParArray<T, _row_size, _row_max, _align>;
        using t_pArray = ssp<TArray>;
        static const DataLayout _dataLayout = TArray::_dataLayout;

        using TArray::Array;
        // TODO: privatize these
        /// @brief Shared pointer to the global-offsets table. Populated by
        /// #createGlobalMapping; may be pointed at an existing table to skip
        /// the collective setup.
        t_pLGlobalMapping pLGlobalMapping;
        /// @brief MPI context associated with this array (must be set before collectives).
        MPIInfo mpi;
        using t_pRowSizes = typename TArray::t_pRowSizes;

    public:
        // default copy
        ParArray(const t_self &R) = default;
        t_self &operator=(const t_self &R) = default;
        // Rule-of-five closure: base + added members are all value-semantic,
        // so default move/dtor do a shallow move of the shared storage.
        ParArray(t_self &&) noexcept = default;
        t_self &operator=(t_self &&) noexcept = default;
        ~ParArray() = default;

        // operator= handled automatically

        /// @brief Copy-assign from another ParArray. Shallow copy semantics
        /// (mirrors @ref DNDS::Array "Array"::clone): shares structural/data buffers.
        void clone(const t_self &R)
        {
            this->operator=(R);
        }

    public:
        /// @brief Serialize (write) the parallel array with MPI-aware metadata.
        ///
        /// Delegates to Array::WriteSerializer for metadata, structure, and data.
        /// Additionally for collective (H5) serializers:
        /// - Writes `sizeGlobal` (sum of all ranks' _size) as a scalar attribute.
        /// - For CSR: computes global data offsets via MPI_Scan and writes pRowStart
        ///   in global data coordinates as a contiguous (nRowsGlobal+1) dataset.
        ///   Non-last ranks write nRows entries (dropping the redundant tail),
        ///   last rank writes nRows+1 (including the global data total).
        ///
        /// Asserts MPI context consistency with the serializer.
        ///
        /// @param serializerP  Serializer instance.
        /// @param name         Sub-path name for this array.
        /// @param offset       [in] Row-level partitioning (typically ArrayGlobalOffset_Parts).
        void WriteSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name, Serializer::ArrayGlobalOffset offset)
        {
            if (!serializerP->IsPerRank())
            {
                DNDS_check_throw_info(
                    mpi == serializerP->getMPI(),
                    fmt::format("ParArray MPI context (rank={}, size={}) doesn't match serializer (rank={}, size={})",
                                mpi.rank, mpi.size, serializerP->GetMPIRank(), serializerP->GetMPISize()));
            }

            // For collective CSR, compute global data offset and pass to Array
            // so it skips its own pRowStart write.
            Serializer::ArrayGlobalOffset dataOffset = Serializer::ArrayGlobalOffset_Unknown;
            if constexpr (_dataLayout == CSR)
            {
                if (!this->IfCompressed())
                    this->Compress();
                if (!serializerP->IsPerRank() && this->_pRowStart)
                {
                    index localDataCount = this->_pRowStart->at(this->_size);
                    index globalDataEnd = 0;
                    MPI::Scan(&localDataCount, &globalDataEnd, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
                    index globalDataStart = globalDataEnd - localDataCount;
                    dataOffset = Serializer::ArrayGlobalOffset{localDataCount, globalDataStart};
                }
            }

            TArray::WriteSerializer(serializerP, name, offset, dataOffset);

            if (!serializerP->IsPerRank())
            {
                auto cwd = serializerP->GetCurrentPath();
                serializerP->GoToPath(name);

                // Write sizeGlobal
                index sizeGlobal = 0;
                MPI::Allreduce(&this->_size, &sizeGlobal, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
                serializerP->WriteIndex("sizeGlobal", sizeGlobal);

                // For CSR, write pRowStart in global data coordinates.
                // Non-last ranks write nRows entries; last rank writes nRows+1.
                // Total = nRowsGlobal + 1 (no overlap, contiguous).
                if constexpr (_dataLayout == CSR)
                {
                    if (dataOffset.isDist())
                    {
                        index globalDataStart = dataOffset.offset();
                        index nWrite = (mpi.rank == mpi.size - 1) ? (this->_size + 1) : this->_size;
                        auto prsGlobal = std::make_shared<host_device_vector<index>>(nWrite);
                        for (index i = 0; i < nWrite; i++)
                            prsGlobal->at(i) = this->_pRowStart->at(i) + globalDataStart;
                        serializerP->WriteSharedIndexVector("pRowStart", prsGlobal,
                                                            Serializer::ArrayGlobalOffset_Parts);
                    }
                }

                serializerP->GoToPath(cwd);
            }
        }

        /// @brief Deserialize (read) the parallel array with MPI-aware metadata.
        ///
        /// Resolves the input `offset` before delegating to Array::ReadSerializer:
        /// - EvenSplit: reads `sizeGlobal`, computes even-split range, resolves to
        ///   isDist({localRows, globalRowStart}).
        /// - CSR with collective serializer: reads per-rank size, computes row offset
        ///   via MPI_Scan, resolves to isDist. This is required because CSR pRowStart
        ///   is stored in global coordinates.
        /// - Otherwise: passes offset through unchanged.
        ///
        /// Asserts MPI context consistency with the serializer.
        ///
        /// @param serializerP  Serializer instance.
        /// @param name         Sub-path name for this array.
        /// @param offset       [in/out] Row-level offset. EvenSplit is resolved here.
        ///                     After return, reflects the resolved row-level position.
        void ReadSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name, Serializer::ArrayGlobalOffset &offset)
        {
            if (!serializerP->IsPerRank())
            {
                DNDS_check_throw_info(
                    mpi == serializerP->getMPI(),
                    fmt::format("ParArray MPI context (rank={}, size={}) doesn't match serializer (rank={}, size={})",
                                mpi.rank, mpi.size, serializerP->GetMPIRank(), serializerP->GetMPISize()));
            }

            if (!serializerP->IsPerRank() && !offset.isDist())
            {
                if (offset == Serializer::ArrayGlobalOffset_EvenSplit)
                {
                    // Read sizeGlobal, compute even-split range
                    auto cwd = serializerP->GetCurrentPath();
                    serializerP->GoToPath(name);
                    index sizeGlobal = 0;
                    serializerP->ReadIndex("sizeGlobal", sizeGlobal);
                    serializerP->GoToPath(cwd);

                    auto [start, end] = EvenSplitRange(mpi.rank, mpi.size, sizeGlobal);
                    offset = Serializer::ArrayGlobalOffset{end - start, start};
                }
                else if constexpr (_dataLayout == CSR)
                {
                    // For CSR with collective serializer, pRowStart is stored in global
                    // coordinates (nRowsGlobal+1 contiguous entries). We must always
                    // resolve to isDist offset so Array reads the correct slice.
                    // Read this rank's _size from the per-rank size dataset, compute
                    // row offset via MPI_Scan, then set isDist offset.
                    auto cwd = serializerP->GetCurrentPath();
                    serializerP->GoToPath(name);
                    std::vector<index> _size_vv;
                    Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_Unknown;
                    serializerP->ReadIndexVector("size", _size_vv, offsetV);
                    DNDS_check_throw(_size_vv.size() == 1);
                    index localSize = _size_vv.front();
                    serializerP->GoToPath(cwd);

                    index globalEnd = 0;
                    MPI::Scan(&localSize, &globalEnd, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
                    index globalStart = globalEnd - localSize;
                    offset = Serializer::ArrayGlobalOffset{localSize, globalStart};
                }
            }

            TArray::ReadSerializer(serializerP, name, offset);
        }

    private:
        MPI_Datatype dataType = BasicType_To_MPIIntType<T>().first;
        MPI_int typeMult = BasicType_To_MPIIntType<T>().second;

    public:
        /// @brief MPI element datatype used for ghost exchange (deduced from `T`).
        MPI_Datatype getDataType() { return dataType; }
        /// @brief Per-element count multiplier that goes with #getDataType.
        MPI_int getTypeMult() { return typeMult; }

    public:
        /**
         * @brief Install the MPI context after default construction.
         * @details Calls @ref AssertDataType to verify the deduced datatype / multiplier
         * match `sizeof(T)`.
         */
        void setMPI(const MPIInfo &n_mpi)
        {
            mpi = n_mpi;
            AssertDataType();
        }

        /// @brief Mutable MPI context accessor.
        MPIInfo &getMPI()
        {
            return mpi;
        }

        /// @brief Read-only MPI context accessor.
        [[nodiscard]] const MPIInfo &getMPI() const
        {
            return mpi;
        }

        /// @brief Override the deduced MPI datatype and element multiplier
        /// (advanced; needed for custom compound element types).
        void setDataType(MPI_Datatype n_dType, MPI_int n_TypeMult)
        {
            dataType = n_dType;
            typeMult = n_TypeMult;
        }

        /// @brief Default-construct an uninitialised ParArray; call #setMPI and @ref Resize later.
        ParArray() = default;

        /// @brief Construct a ParArray bound to the given MPI context.
        ParArray(const MPIInfo &n_mpi) : mpi(n_mpi)
        {
            AssertDataType();
        }
        /// @brief Construct with a custom (MPI datatype, multiplier) pair.
        /// @details Useful for element types whose in-memory layout differs from
        /// the default `BasicType_To_MPIIntType<T>()` deduction.
        ParArray(MPI_Datatype n_dType, MPI_int n_TypeMult, const MPIInfo &n_mpi)
            : mpi(n_mpi), dataType(n_dType), typeMult(n_TypeMult)
        {
            AssertDataType();
        }

        /// @brief Named constructor: sets the object name for tracing/debugging.
        /// All existing constructor overloads are supported via perfect forwarding.
        /// Inherited by derived classes (ArrayAdjacency, ArrayEigenVector, etc.)
        /// through `using t_base::t_base`.
        ///
        /// Usage:
        ///   ParArray<index> arr(ObjName{"cell2node"}, mpi);
        ///   ArrayAdjacency<> adj(ObjName{"cell2cell"}, mpi);
        template <typename... Args>
        ParArray(ObjName objName, Args &&...args)
            : ParArray(std::forward<Args>(args)...)
        {
            this->setObjectName(std::move(objName.name));
        }

        /// @brief Assert the MPI datatype matches `sizeof(T)` exactly.
        /// @details Called from constructors / #setMPI / #setDataType. Guards
        /// against size mismatches that would silently corrupt comms.
        void AssertDataType()
        {
            DNDS_check_throw(dataType != MPI_DATATYPE_NULL);
            MPI_Aint lb = 0;
            MPI_Aint extent = 0;
            MPI_Type_get_extent(dataType, &lb, &extent);
            DNDS_check_throw(lb == 0 && extent * typeMult == sizeof(T));
        }

        /**
         * @brief Check array consistency across all ranks.
         *
         * @details Uses `MPI_Allgather` to verify that row sizes (for `TABLE_Fixed`
         * and `TABLE_Max` layouts) and MPI type multipliers are the same on every
         * rank. Intended as a post-setup sanity check before entering ghost exchange.
         *
         * @warning Must be called collectively. O(nRanks) memory and communication.
         * @return true Always (failures are reported via @ref DNDS_check_throw_info).
         */
        bool AssertConsistent()
        {
            DNDS_check_throw(mpi.comm != MPI_COMM_NULL);
            MPI::Barrier(mpi.comm); // must be globally existent
            if constexpr (_dataLayout == TABLE_Max ||
                          _dataLayout == TABLE_Fixed) // must have the same dynamic size
            {
                // checking if is uniform across all procs
                t_RowsizeVec uniformSizes(mpi.size);
                MPI_int rowsizeC = this->RowSizeField();
                static_assert(sizeof(MPI_int) == sizeof(rowsize));
                MPI::Allgather(&rowsizeC, 1, MPI_INT, uniformSizes.data(), 1, MPI_INT, mpi.comm);
                for (auto i : uniformSizes)
                    DNDS_check_throw_info(i == rowsizeC, "sizes not uniform across procs");
            }

            std::vector<MPI_int> uniform_typeMult(mpi.size);
            MPI::Allgather(&typeMult, 1, MPI_INT, uniform_typeMult.data(), 1, MPI_INT, mpi.comm);
            for (auto i : uniform_typeMult)
                DNDS_check_throw_info(i == typeMult, "typeMults not uniform across procs");

            return true; // currently all errors aborts inside
        }

        /**
         * @brief Collective: build the global offsets table.
         *
         * @details Every rank broadcasts its local `Size()`; after the call,
         * #pLGlobalMapping holds the full #GlobalOffsetsMapping::RLengths /
         * #GlobalOffsetsMapping::ROffsets on every
         * rank. Must be invoked before #globalSize, @ref DNDS::ArrayTransformer "ArrayTransformer"::createFatherGlobalMapping,
         * or any CSR collective serialization.
         *
         * @warning Must be called collectively on `mpi.comm`.
         */
        void createGlobalMapping() // collective;
        {
            DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "MPI unset");
            // phase1.1: create localGlobal mapping (broadcast)
            pLGlobalMapping = std::make_shared<GlobalOffsetsMapping>();
            pLGlobalMapping->setMPIAlignBcast(mpi, this->Size());
        }

        /**
         * @brief Returns the total global size (sum of sizes across all ranks).
         *
         * @note This method was previously collective (using MPI_Allreduce) but is now
         *       non-collective. It requires that the global mapping has been created
         *       first (which is done via the collective createGlobalMapping() method
         *       on the underlying ParArray).
         *
         * @pre createGlobalMapping() must have been called on the underlying array.
         * @return index The global size (cached from global mapping).
         */
        [[nodiscard]] index globalSize() const
        {
            DNDS_assert_info(pLGlobalMapping,
                             "globalSize() requires global mapping. "
                             "Ensure createGlobalMapping() was called first (typically via ArrayPair operations).");
            return pLGlobalMapping->globalSize();
        }
    };
    /********************************************************************************************************/

    /********************************************************************************************************/

    /**
     * @brief Ghost-communication engine for a father / son @ref DNDS::ParArray "ParArray" pair.
     *
     * @details Distributed-mesh stencil schemes need data from cells owned by
     * other ranks. @ref DNDS::ArrayTransformer "ArrayTransformer" stores two @ref DNDS::ParArray "ParArray" pointers -- the
     * *father* (owned rows) and the *son* (incoming ghost rows) -- plus the
     * MPI machinery to move data between them.
     *
     * ## Setup (done once)
     * 1. #setFatherSon            -- attach the two arrays.
     * 2. #createFatherGlobalMapping -- collective; populate global offsets.
     * 3. #createGhostMapping      -- specify which global rows this rank needs as ghosts.
     * 4. #createMPITypes          -- build `MPI_Type_create_hindexed` derived
     *                                types (or in-situ pack buffers) for send/recv.
     *
     * ## Communication
     * - One-shot: #pullOnce / #pushOnce (short-lived `MPI_Isend`/`MPI_Irecv` pair).
     * - Persistent: #initPersistentPull -> repeated #startPersistentPull /
     *   #waitPersistentPull cycles -> #clearPersistentPull when done.
     *   Persistent requests avoid re-posting sends and receives on every
     *   iteration, saving per-step overhead.
     *
     * ## Reuse
     * When multiple arrays share the same ghost pattern (e.g. the DOF array
     * and the gradient array both use the `cell2cell` partition), call
     * @ref BorrowGGIndexing on the secondary transformer to copy the mapping
     * without redoing collective setup -- only #createMPITypes must be redone,
     * because the element size differs.
     *
     * @sa ArrayPair, ArrayDof, docs/architecture/array_infrastructure.md.
     */
    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    // template <class TArray>
    class ArrayTransformer
    {
        MPI::CommStrategy::ArrayCommType commTypeCurrent = MPI::CommStrategy::UnknownArrayCommType;

    public:
        using TArray = ParArray<T, _row_size, _row_max, _align>;
        using TSelf = ArrayTransformer<T, _row_size, _row_max, _align>;
        // using T = TArray::value_type;
        // static const rowsize _align = TArray::al;
        // static const rowsize _row_size = TArray::rs;
        // static const rowsize _row_max = TArray::rm;

        using t_pArray = ssp<TArray>;
        static const DataLayout _dataLayout = TArray::_dataLayout;

        /*********************************/
        /*          MEMBER               */
        /*********************************/

        /// @brief MPI context; copied from the attached father array.
        MPIInfo mpi;
        /// @brief Ghost index mapping (rank-local layout). Populated by #createGhostMapping.
        t_pLGhostMapping pLGhostMapping;
        /// @brief The "owned" side of the father/son pair.
        t_pArray father;
        /// @brief The "ghost" side of the father/son pair (receives from other ranks).
        t_pArray son;

        /// @brief Shared pointer to the global offsets table (shared with father).
        t_pLGlobalMapping pLGlobalMapping; // reference from father

        /// @brief Cached `(count, MPI_Datatype)` pairs for push (son -> father).
        ssp<tMPI_typePairVec> pPushTypeVec;
        /// @brief Cached `(count, MPI_Datatype)` pairs for pull (father -> son).
        ssp<tMPI_typePairVec> pPullTypeVec;

        // ** comm aux info: comm running structures **
        // TODO: make these aux info (sized) shared and thread-safe
        /// @brief Persistent request handles for push.
        ssp<MPIReqHolder> PushReqVec;
        /// @brief Persistent request handles for pull.
        ssp<MPIReqHolder> PullReqVec;
        /// @brief Device currently holding push buffers (@ref Unknown if not initialised).
        DeviceBackend pushDevice = DeviceBackend::Unknown;
        /// @brief Device currently holding pull buffers (@ref Unknown if not initialised).
        DeviceBackend pullDevice = DeviceBackend::Unknown;
        /// @brief Number of receive requests in #PushReqVec (the rest are sends).
        MPI_int nRecvPushReq{-1};
        /// @brief Number of receive requests in #PullReqVec.
        MPI_int nRecvPullReq{-1};
        /// @brief Status buffer for push completion.
        tMPI_statVec PushStatVec;
        /// @brief Status buffer for pull completion.
        tMPI_statVec PullStatVec;
        /// @brief Total bytes sent per push call (for buffer sizing).
        MPI_Aint pushSendSize;
        /// @brief Total bytes sent per pull call.
        MPI_Aint pullSendSize;

        tMPI_intVec pushingSizes;                 ///< temp: per-peer count for #createMPITypes.
        tMPI_AintVec pushingDisps;                ///< temp: per-peer byte displacements for #createMPITypes.
        std::vector<index> pushingIndexLocal;     ///< for InSituPack strategy
        std::vector<std::vector<T>> inSituBuffer; ///< for InSituPack strategy

        /*********************************/
        /*          MEMBER               */
        /*********************************/

        /// @brief Copy-assign the transformer state. Persistent requests are
        /// re-created rather than shared because they point to different
        /// memory than the source object.
        TSelf &operator=(const TSelf &R)
        {
            if (this == &R)
                return *this;
            // must have commTypeCurrent copied as a result of createMPITypes()
            commTypeCurrent = R.commTypeCurrent;

            mpi = R.mpi;
            pLGhostMapping = R.pLGhostMapping;
            father = R.father;
            son = R.son;

            pLGlobalMapping = R.pLGlobalMapping;

            // these are shared as results of createMPITypes()
            pPushTypeVec = R.pPushTypeVec;
            pPullTypeVec = R.pPullTypeVec;

            // ** comm aux info: comm running structures **
            // PushReqVec;
            // PullReqVec;
            // pushDevice;
            // pullDevice;
            // nRecvPushReq{-1};
            // nRecvPullReq{-1};
            // PushStatVec;
            // PullStatVec;
            // pushSendSize;
            // pullSendSize;
            // ! check comm aux info status and correctly duplicate them
            // ! cannot share because point to different data
            if (R.PullReqVec)
                this->initPersistentPull();
            if (R.PushReqVec)
                this->initPersistentPush();

            // these are createMPITypes() temporaries,
            // TODO: maybe remove from member?
            // pushingSizes;
            // inSituBuffer;

            // comm aux info but created in createMPITypes()
            // TODO (remove from createMPITypes() maybe?)
            pushingIndexLocal = R.pushingIndexLocal;
            inSituBuffer = R.inSituBuffer;
            return *this;
        }

        /// @brief Default-construct an empty transformer; attach arrays later via #setFatherSon.
        ArrayTransformer() = default;

        /// @brief Copy-construct via operator=.
        ArrayTransformer(const TSelf &R)
        {
            // initial-safe operator= call
            this->operator=(R);
        }

        /// @brief Rule-of-five closure. All non-trivial members are held via
        /// `shared_ptr` (`pLGhostMapping`, `father`, `son`, `pPushTypeVec`,
        /// `pPullTypeVec`, `PushReqVec`, `PullReqVec`), so default move /
        /// destructor are correct — they shallow-move the handles.
        ArrayTransformer(TSelf &&) noexcept = default;
        TSelf &operator=(TSelf &&) noexcept = default;
        ~ArrayTransformer() = default;

        /**
         * @brief Attach father and son arrays. First setup step.
         *
         * @details Both arrays must share the same MPI context and element
         * MPI datatype. The transformer cannot be used until the remaining
         * setup calls (#createFatherGlobalMapping, #createGhostMapping,
         * #createMPITypes) have run.
         *
         * @param n_father Owned-side array (must not be null).
         * @param n_son    Ghost-side array (must not be null).
         */
        void setFatherSon(const t_pArray &n_father, const t_pArray &n_son)
        {
            DNDS_check_throw(n_father && n_son);
            father = n_father;
            son = n_son;
            mpi = father->getMPI();
            DNDS_check_throw_info(son->getMPI() == father->getMPI(), "MPI inconsistent between father & son");
            DNDS_check_throw_info(father->getDataType() == son->getDataType(), "MPI datatype inconsistent between father & son");
            DNDS_check_throw_info(father->getTypeMult() == son->getTypeMult(), "MPI datatype multiplication inconsistent between father & son");
            DNDS_check_throw_info(father->getDataType() != MPI_DATATYPE_NULL, "MPI datatype invalid");
            DNDS_check_throw_info(father->getTypeMult() > 0, "MPI datatype multiplication invalid");
            pLGhostMapping.reset();
            pLGlobalMapping.reset();
            pLGlobalMapping = father->pLGlobalMapping;
        }

        /**
         * @brief Borrow the ghost and global mapping from another transformer.
         *
         * @details Intended for the common case where several arrays share the
         * same partition (e.g., the DOF array and the gradient array both
         * live on the same cell partitioning). Copies the shared pointers --
         * no collective work is performed. After this call, #createMPITypes
         * still needs to be invoked because the element size differs.
         *
         * @tparam TRArrayTrans A compatible transformer type; must expose
         *                     `father`, `pLGhostMapping`, `pLGlobalMapping`.
         */
        template <class TRArrayTrans>
        void BorrowGGIndexing(const TRArrayTrans &RArrayTrans)
        {
            // DNDS_check_throw(father && Rarray.father); // Rarray's father is not visible...
            // DNDS_check_throw(father->obtainTotalSize() == Rarray.father->obtainTotalSize());
            DNDS_check_throw(RArrayTrans.father && father);
            DNDS_check_throw(RArrayTrans.pLGhostMapping && RArrayTrans.pLGlobalMapping);
            DNDS_check_throw(RArrayTrans.father->Size() == father->Size());
            pLGhostMapping = RArrayTrans.pLGhostMapping;
            pLGlobalMapping = RArrayTrans.pLGlobalMapping;
            father->pLGlobalMapping = RArrayTrans.pLGlobalMapping;
        }

        /// @brief Collective: build the global offsets table on the father array.
        /// @details Thin wrapper over `father->createGlobalMapping()` that also
        /// caches the pointer in this transformer. Second setup step.
        void createFatherGlobalMapping()
        {
            father->createGlobalMapping();
            pLGlobalMapping = father->pLGlobalMapping;
        }

        /** @brief create ghost by pulling data
         * @details
         * pulling data indicates the data put in son (received in pulling operation)
         * pullingIndexGlobal is the global indices in son
         * pullingIndexGlobal should be mutually different, otherwise behavior undefined
         *
         * @warning pullingIndexGlobal is **sorted and deduplicated in-place** by
         * OffsetAscendIndexMapping. After this call the input vector's element order
         * is destroyed. If you need to keep the original order (e.g., for a
         * redistribution mapping), save a copy before calling this method.
         * The son array after pullOnce() will contain data in the sorted order
         * of pullingIndexGlobal, NOT in the original input order.
         */
        template <class TPullSet>
        void createGhostMapping(TPullSet &&pullingIndexGlobal) // collective;
        {
            DNDS_check_throw(bool(father) && bool(son));
            DNDS_check_throw_info(bool(father->pLGlobalMapping), "Father needs to createGlobalMapping");
            pLGlobalMapping = father->pLGlobalMapping;
            // phase1.2: count how many to pull and allocate the localGhost mapping, fill the mapping
            // counting could overflow
            // tMPI_intVec ghostSizes(mpi.size, 0); // == pulling sizes
            pLGhostMapping = std::make_shared<OffsetAscendIndexMapping>(
                (*pLGlobalMapping)(mpi.rank, 0), father->Size(),
                std::forward<TPullSet>(pullingIndexGlobal),
                *pLGlobalMapping,
                mpi);
        }

        /**
         * @brief Create the ghost mapping from a *push* specification. Collective.
         *
         * @details Each rank supplies, grouped per receiver, the local indices it
         * will push to that receiver. Row `i` of this rank's father will be sent
         * to every rank listed for `i` across the CSR `(pushingIndexLocal, pushStarts)`.
         * The son array is resized to hold the incoming entries on return from
         * #createMPITypes.
         *
         * @param pushingIndexLocal Flat vector of local indices to push, grouped
         *                          by receiver in ascending rank order.
         * @param pushStarts        Prefix-sum offsets into `pushingIndexLocal`,
         *                          size `mpi.size + 1`.
         *
         * @warning Each local index must appear at most once across the entire
         * CSR, otherwise the resulting ghost layout is undefined.
         */
        template <class TPushSet, class TPushStart>
        void createGhostMapping(TPushSet &&pushingIndexLocal, TPushStart &&pushStarts) // collective;
        {
            DNDS_check_throw(bool(father) && bool(son));
            DNDS_check_throw_info(bool(father->pLGlobalMapping), "Father needs to createGlobalMapping");
            pLGlobalMapping = father->pLGlobalMapping;
            // phase1.2: calculate over pushing
            // counting could overflow
            pLGhostMapping = std::make_shared<OffsetAscendIndexMapping>(
                (*pLGlobalMapping)(mpi.rank, 0), father->Size(),
                std::forward<TPushSet>(pushingIndexLocal),
                std::forward<TPushStart>(pushStarts),
                *pLGlobalMapping,
                mpi);
        }
        /**
         * @brief Collective: build the MPI derived datatypes (or in-situ buffers)
         * that describe the ghost send/recv layout. Resizes the son array.
         *
         * @details Fourth (and final) setup step. Consumes the per-rank push
         * sizes and the ghost mapping produced by #createGhostMapping, then:
         *  - for @ref HIndexed: builds `MPI_Type_create_hindexed` types that
         *    describe the scattered memory layout of the rows being sent and
         *    received;
         *  - for @ref InSituPack: allocates contiguous pack buffers.
         *
         * Also resizes `son` to hold exactly the incoming ghost rows.
         *
         * @pre `father`, `son`, #pLGlobalMapping, #pLGhostMapping are set.
         * @post `pPullTypeVec` and `pPushTypeVec` (or the in-situ buffers) are
         *       valid; son has been resized.
         */
        void createMPITypes() // collective;
        {
            DNDS_check_throw(bool(father) && bool(son));
            DNDS_check_throw(pLGlobalMapping && pLGhostMapping);
            commTypeCurrent = MPI::CommStrategy::Instance().GetArrayStrategy();
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
                father->Compress(); //! assure CSR is in compressed form
            // TODO: support comm for uncompressed: add in-situ packaging working mode
            // TODO: support actual MAX arrays' size communicating: append comm types: ? needed?
            // TODO: add manual packaging mode

            /*********************************************/ // starts to deal with actual byte sizes

            //*phase2.1: build push sizes and push disps
            index nSend = pLGhostMapping->pushingIndexGlobal.size();
            pushingSizes.resize(nSend); // pushing sizes  xx in bytes xx now in num of remove_all_extents_t<T>

            pushingDisps.resize(nSend); // pushing disps in bytes

            DNDS_check_throw(nSend < DNDS_INDEX_MAX);
            auto fatherDataStart = father->operator[](0);
            if (commTypeCurrent == MPI::CommStrategy::InSituPack)
                pushingIndexLocal.resize(nSend);
            for (index i = 0; i < nSend; i++)
            {
                MPI_int rank = -1;
                index loc = -1;
                bool found = pLGhostMapping->search(pLGhostMapping->pushingIndexGlobal[i], rank, loc);
                DNDS_check_throw_info(found && rank == -1, "must be at local main");             // must be at local main
                pushingDisps[i] = (father->operator[](loc) - father->operator[](0)) * sizeof(T); //* in bytes
                if constexpr (_dataLayout == CSR)
                    pushingSizes[i] = father->RowSizeField(loc) * father->getTypeMult();
                if constexpr (isTABLE_Max(_dataLayout)) //! init sizes
                    pushingSizes[i] = father->RowSize(loc) * father->getTypeMult();
                if constexpr (isTABLE_Fixed(_dataLayout))
                    pushingSizes[i] = father->RowSizeField() * father->getTypeMult();

                if (commTypeCurrent == MPI::CommStrategy::InSituPack)
                    pushingIndexLocal[i] = loc;
            }
            // PrintVec(pushingSizes, std::cout);
            // std::cout << std::endl;

            //*phase2.2: be informed of pulled sub-indexer
            // equals to: building pullingSizes and pullingDisps, bytes size and disps of ghost
            // - legacy: indexer.buildAsGhostAlltoall(father->indexer, pushingSizes, *pLGhostMapping, mpi); // cascade from father
            auto do_son_resizing = [&]()
            {
                auto &LGhostMapping = *pLGhostMapping;
                index ghostArraySiz = LGhostMapping.ghostStart[LGhostMapping.ghostStart.size() - 1];
                DNDS_check_throw(mpi.size == LGhostMapping.ghostStart.size() - 1);
                if constexpr (_dataLayout == TABLE_StaticFixed)
                {
                    son->Resize(ghostArraySiz);
                    return;
                }
                if constexpr (_dataLayout == TABLE_Fixed)
                {
                    son->Resize(ghostArraySiz, father->RowSize()); // using father's row size
                    return;
                }
                if constexpr (_dataLayout == TABLE_Max)
                {
                    son->Resize(ghostArraySiz, father->RowSizeMax());
                    // and go on for non-uniform resizing
                }
                if constexpr (_dataLayout == TABLE_StaticMax)
                {
                    son->Resize(ghostArraySiz);
                    // and go on for non-uniform resizing
                }

                // obtain pulling sizes with pushing sizes
                tMPI_intVec pullingSizes(ghostArraySiz);
                MPI_Alltoallv(pushingSizes.data(), LGhostMapping.pushIndexSizes.data(), LGhostMapping.pushIndexStarts.data(), MPI_INT,
                              pullingSizes.data(), LGhostMapping.ghostSizes.data(), LGhostMapping.ghostStart.data(), MPI_INT,
                              mpi.comm);

                // std::cout << LGhostMapping.gStarts().size() << std::endl;
                if constexpr (_dataLayout == CSR)
                    son->Resize(ghostArraySiz, [&](index i)
                                { return pullingSizes[i] / father->getTypeMult(); });
                if constexpr (_dataLayout == TABLE_Max)
                {
                    son->Resize(ghostArraySiz, father->RowSizeMax());
                    for (index i = 0; i < son->Size(); i++)
                        son->ResizeRow(i, pullingSizes[i] / father->getTypeMult());
                }
                if constexpr (_dataLayout == TABLE_StaticMax)
                {
                    son->Resize(ghostArraySiz);
                    for (index i = 0; i < son->Size(); i++)
                        son->ResizeRow(i, pullingSizes[i] / father->getTypeMult());
                }
                // is actually pulling disps, but is contiguous anyway

                // DNDS_MPI_InsertCheck(mpi);
                // std::cout << mpi.rank << " VEC ";
                // PrintVec(pullingSizes, std::cout);
                // std::cout << std::endl;
                // DNDS_MPI_InsertCheck(mpi);

                // note that Rowstart and pullingSizes are in bytes
                // pullingSizes is actual but Rowstart is before indexModder(), use indexModder[] to invert
            };
            do_son_resizing();

            // phase3: create and register MPI types of pushing and pulling
            if constexpr (isTABLE_Max(_dataLayout)) // convert back to full pushing sizes
            {
                for (auto &i : pushingSizes)
                    i = son->RowSizeField() * father->getTypeMult();
            }

            if (commTypeCurrent == MPI::CommStrategy::HIndexed) // record types
            {
                pPushTypeVec = MPITypePairHolder::create();
                pPullTypeVec = MPITypePairHolder::create();
                for (MPI_int r = 0; r < mpi.size; r++)
                {
                    /************************************************************/
                    // push
                    MPI_int pushNumber = pLGhostMapping->pushIndexSizes[r];
                    // std::cout << "PN" << pushNumber << std::endl;
                    MPI_Aint *pPushDisps = pushingDisps.data() + pLGhostMapping->pushIndexStarts[r];
                    MPI_int *pPushSizes = pushingSizes.data() + pLGhostMapping->pushIndexStarts[r];
                    index sumPushSizes = 0; // using upgraded integer
                    for (MPI_int i = 0; i < pushNumber; i++)
                        sumPushSizes += pPushSizes[i];
                    if (sumPushSizes > 0) // if no actuall data is to be sent
                    {
                        // std::cout <<mpi.rank<< " pushSlice " << pPushDisps[0] << outputDelim << pPushSizes[0] << std::endl;

                        // if (mpi.rank == 0)
                        // {
                        //     std::cout << "pushing to " << r << "  size" << pushNumber << "\n";
                        //     for (int i = 0; i < pushNumber; i++)
                        //         std::cout << "b[" << i << "] = " << pPushSizes[i] << std::endl;
                        //     for (int i = 0; i < pushNumber; i++)
                        //         std::cout << "d[" << i << "] = " << pPushDisps[i] << std::endl;
                        // }
                        // std::cout << "=== PUSH TYPE : " << mpi.rank << " from " << r << std::endl;

                        MPI_Datatype dtype = MPI_DATATYPE_NULL;
                        int sizeof_T = MPI_UNDEFINED;
                        MPI_Type_size(father->getDataType(), &sizeof_T);
                        DNDS_check_throw(sizeof_T != MPI_UNDEFINED);
                        auto [n_number, new_Sizes, new_Disps] =
                            optimize_hindexed_layout(pushNumber, pPushSizes, pPushDisps, sizeof_T);
                        MPI_Type_create_hindexed(n_number, new_Sizes.data(), new_Disps.data(), father->getDataType(), &dtype);
                        // MPI_Type_create_hindexed(PushDispsMPI.size(), PushSizesMPI.data(), PushDispsMPI.data(), father->getDataType(), &dtype);

                        MPI_Type_commit(&dtype);
                        pPushTypeVec->push_back(std::make_pair(r, dtype));
                        // OPT: could use MPI_Type_create_hindexed_block to save some space
                    }
                    /************************************************************/
                    // pull
                    std::array<MPI_Aint, 1> pullDisp;

                    std::array<MPI_int, 1> pullSizes; // same as pushSizes
                    auto gRPtr = son->operator[](index(pLGhostMapping->ghostStart[r + 1]));
                    auto gLPtr = son->operator[](index(pLGhostMapping->ghostStart[r]));
                    auto gStartPtr = son->operator[](index(0));
                    auto ghostSpan = gRPtr - gLPtr;
                    auto ghostStart = gLPtr - gStartPtr;
                    DNDS_check_throw(ghostSpan < INT_MAX && ghostStart < INT_MAX);
                    pullSizes[0] = MPI_int(ghostSpan) * father->getTypeMult();
                    pullDisp[0] = ghostStart * sizeof(T);
                    if (pullSizes[0] > 0)
                    {
                        // std::cout << "=== PULL TYPE : " << mpi.rank << " from " << r << std::endl;
                        MPI_Datatype dtype = MPI_DATATYPE_NULL;

                        MPI_Type_create_hindexed(1, pullSizes.data(), pullDisp.data(), father->getDataType(), &dtype);

                        // std::cout << mpi.rank << " pullSlice " << pullDisp[0] << outputDelim << pullBytes[0] << std::endl;
                        MPI_Type_commit(&dtype);
                        pPullTypeVec->push_back(std::make_pair(r, dtype));
                    }
                }
                pPullTypeVec->shrink_to_fit();
                pPushTypeVec->shrink_to_fit();

                pushingDisps.clear();
                pushingSizes.clear(); // no need
            }
            else if (commTypeCurrent == MPI::CommStrategy::CommStrategy::InSituPack)
            {
                // could simplify some info on sparse comm?
                pushingDisps.clear();
            }
            else
            {
                DNDS_check_throw(false);
            }
        }
        /******************************************************************************************************************************/

        auto getFatherSonData(DeviceBackend B)
        {
            T *fatherData = nullptr;
            T *sonData = nullptr;
            if (B == DeviceBackend::Unknown)
            {
                fatherData = father->data();
                sonData = son->data();
            }
            else
            {
                DNDS_check_throw(B == father->device());
                DNDS_check_throw(B == son->device());
                switch (B)
                {
                case DeviceBackend::Host:
                {
                    fatherData = father->data(B);
                    sonData = son->data(B);
                }
                break;
#ifdef DNDS_USE_CUDA
                case DeviceBackend::CUDA:
                {
                    //!
                    DNDS_check_throw_info(MPI::isCudaAware(), "we require CUDA-aware MPI here");
                    fatherData = father->data(B);
                    sonData = son->data(B);
                }
                break;
#endif
                default:
                {
                    DNDS_check_throw(false);
                }
                }
            }
            return std::make_pair(fatherData, sonData);
        }

        /******************************************************************************************************************************/
        /**
         * @brief Initialise persistent, non-blocking, non-buffered MPI requests
         * for the push direction (son -> father).
         *
         * @details Once persistent requests are created, many push cycles may
         * be run via #startPersistentPush / #waitPersistentPush without
         * re-posting sends and receives.
         *
         * @pre #createMPITypes has been called; #pPullTypeVec and #pPushTypeVec are valid.
         * @post @ref PushReqVec is populated with `MPI_Send_init` / `MPI_Recv_init` requests.
         *
         * @param B Device backend for the send/recv buffers
         *          (`DeviceBackend::Unknown` to use host; requires CUDA-aware
         *          MPI for non-host backends).
         * @warning After init, the raw data pointers of both father and son
         * must remain stable until #clearPersistentPush is called.
         */
        void initPersistentPush(DeviceBackend B = DeviceBackend::Unknown) // collective;
        {
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                pushDevice = B;
                auto [fatherData, sonData] = getFatherSonData(B);
                // DNDS_check_throw(pPullTypeVec && pPushTypeVec);
                DNDS_check_throw(pPullTypeVec.use_count() > 0 && pPushTypeVec.use_count() > 0);
                pushSendSize = 0;
                auto nReqs = pPullTypeVec->size() + pPushTypeVec->size();
                // DNDS_check_throw(nReqs > 0);
                PushReqVec = MPIReqHolder::create();
                PushReqVec->resize(nReqs, (MPI_REQUEST_NULL)), PushStatVec.resize(nReqs);
                nRecvPushReq = 0;
                for (auto ip = 0; ip < pPushTypeVec->size(); ip++)
                {
                    auto dtypeInfo = (*pPushTypeVec)[ip];
                    MPI_int rankOther = dtypeInfo.first;
                    MPI_int tag = rankOther + mpi.rank;
                    MPI_Recv_init(fatherData, 1, dtypeInfo.second, rankOther, tag, mpi.comm, PushReqVec->data() + pPullTypeVec->size() + ip);
                    // cascade from father
                    nRecvPushReq++;
                }
                for (auto ip = 0; ip < pPullTypeVec->size(); ip++)
                {
                    auto dtypeInfo = (*pPullTypeVec)[ip];
                    MPI_int rankOther = dtypeInfo.first;
                    MPI_int tag = rankOther + mpi.rank;

#ifndef ARRAY_COMM_USE_BUFFERED_SEND
                    // MPI_Ssend_init
                    MPI_Send_init
#else
                    MPI_Bsend_init
#endif
                        (sonData, 1, dtypeInfo.second, rankOther, tag, mpi.comm, PushReqVec->data() + ip);

                    // cascade from father

                    // // buffer calculate //!deprecated because of size limit
                    // MPI_Aint csize;
                    // MPI_Pack_external_size(1, dtypeInfo.second, mpi.comm, &csize);
                    // csize += MPI_BSEND_OVERHEAD;
                    // DNDS_check_throw(MAX_MPI_Aint - pushSendSize >= csize && csize > 0);
                    // pushSendSize += csize * 2;
                }
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
                // MPIBufferHandler::Instance().claim(pushSendSize, mpi.rank);
#endif
            }
            else if (commTypeCurrent == MPI::CommStrategy::CommStrategy::InSituPack)
            {
                // could simplify some info on sparse comm?
                PushReqVec = MPIReqHolder::create();
            }
            else
            {
                DNDS_check_throw(false);
            }
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * @brief Initialise persistent, non-blocking MPI requests for the pull
         * direction (father -> son). Counterpart to #initPersistentPush.
         *
         * @pre #createMPITypes has been called; #pPullTypeVec and #pPushTypeVec are valid.
         * @post @ref PullReqVec is populated.
         *
         * @param B Device backend for the send/recv buffers.
         * @warning Raw data pointers for both father and son must remain stable
         * until #clearPersistentPull.
         */
        void initPersistentPull(DeviceBackend B = DeviceBackend::Unknown) // collective;
        {
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                pullDevice = B;
                auto [fatherData, sonData] = getFatherSonData(B);
                // DNDS_check_throw(pPullTypeVec && pPushTypeVec);
                DNDS_check_throw(pPullTypeVec.use_count() > 0 && pPushTypeVec.use_count() > 0);
                auto nReqs = pPullTypeVec->size() + pPushTypeVec->size();
                pullSendSize = 0;
                // DNDS_check_throw(nReqs > 0);
                PullReqVec = MPIReqHolder::create();
                PullReqVec->resize(nReqs, (MPI_REQUEST_NULL)), PullStatVec.resize(nReqs);
                nRecvPullReq = 0;
                for (typename decltype(pPullTypeVec)::element_type::size_type ip = 0; ip < pPullTypeVec->size(); ip++)
                {
                    auto dtypeInfo = (*pPullTypeVec)[ip];
                    MPI_int rankOther = dtypeInfo.first;
                    MPI_int tag = rankOther + mpi.rank; //! receives a lot of messages, this distinguishes them
                    // std::cout << mpi.rank << " Recv " << rankOther << std::endl;
                    MPI_Recv_init(sonData, 1, dtypeInfo.second, rankOther, tag, mpi.comm, PullReqVec->data() + ip);
                    nRecvPullReq++;
                    // std::cout << *(real *)(dataGhost.data() + 8 * 0) << std::endl;
                    // cascade from father
                }
                for (typename decltype(pPullTypeVec)::element_type::size_type ip = 0; ip < pPushTypeVec->size(); ip++)
                {
                    auto dtypeInfo = (*pPushTypeVec)[ip];
                    MPI_int rankOther = dtypeInfo.first;
                    MPI_int tag = rankOther + mpi.rank;
                    // std::cout << mpi.rank << " Send " << rankOther << std::endl;
#ifndef ARRAY_COMM_USE_BUFFERED_SEND
                    // MPI_Ssend_init
                    MPI_Send_init
#else
                    MPI_Bsend_init
#endif
                        (fatherData, 1, dtypeInfo.second, rankOther, tag, mpi.comm, PullReqVec->data() + pPullTypeVec->size() + ip);
                    // std::cout << *(real *)(data.data() + 8 * 1) << std::endl;
                    // cascade from father

                    // // buffer calculate //!deprecated because of size limit
                    // MPI_Aint csize;
                    // MPI_Pack_external_size(1, dtypeInfo.second, mpi.comm, &csize);
                    // csize += MPI_BSEND_OVERHEAD * 8;
                    // DNDS_check_throw(MAX_MPI_Aint - pullSendSize >= csize && csize > 0);
                    // pullSendSize += csize * 2;
                }
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
                // MPIBufferHandler::Instance().claim(pullSendSize, mpi.rank);
#endif
            }
            else if (commTypeCurrent == MPI::CommStrategy::CommStrategy::InSituPack)
            {
                // could simplify some info on sparse comm?
                PullReqVec = MPIReqHolder::create();
            }
            else
            {
                DNDS_check_throw(false);
            }
        }
        /******************************************************************************************************************************/

        void InSituPackStartPush(DeviceBackend B)
        {
            if (B != DeviceBackend::Unknown)
                DNDS_check_throw_info(false, "in-situ pack not yet implemented for device");
            nRecvPushReq = 0;
            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // push
                MPI_int pushNumber = pLGhostMapping->pushIndexSizes[r];
                // std::cout << "PN" << pushNumber << std::endl;
                if (pushNumber > 0)
                {
                    index nPushData{0};
                    for (index i = 0; i < pushNumber; i++)
                    {
                        auto loc = pushingIndexLocal.at(pLGhostMapping->pushIndexStarts[r] + i);
                        index nPush = 0;
                        if constexpr (_dataLayout == CSR)
                            nPush = father->RowSizeField(loc);
                        if constexpr (isTABLE_Max(_dataLayout)) //! init sizes
                            nPush = father->RowSize(loc);
                        if constexpr (isTABLE_Fixed(_dataLayout))
                            nPush = father->RowSizeField();
                        nPushData += nPush;
                    }
                    inSituBuffer.emplace_back(nPushData);
                    PushReqVec->emplace_back(MPI_REQUEST_NULL);
                    MPI_Irecv(inSituBuffer.back().data(), nPushData * father->getTypeMult(), father->getDataType(),
                              r, mpi.rank + r, mpi.comm, &PushReqVec->back());
                    nRecvPushReq++;
                }
            }
            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // pull
                MPI_Aint pullDisp = 0;
                MPI_int pullSize = 0; // same as pushSizes
                auto gRPtr = son->operator[](index(pLGhostMapping->ghostStart[r + 1]));
                auto gLPtr = son->operator[](index(pLGhostMapping->ghostStart[r]));
                auto ghostSpan = gRPtr - gLPtr;
                pullSize = MPI_int(ghostSpan);

                if (pullSize > 0)
                {
                    PushReqVec->emplace_back(MPI_REQUEST_NULL);
                    MPI_Issend(gLPtr, pullSize * father->getTypeMult(), father->getDataType(), r, r + mpi.rank, mpi.comm, &PushReqVec->back());
                }
            }
        }

        /// @brief Start all persistent push requests (@ref MPI_Startall).
        /// @param B Device backend; must match the one passed to #initPersistentPush.
        void startPersistentPush(DeviceBackend B = DeviceBackend::Unknown) // collective;
        {
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                DNDS_check_throw(B == pushDevice);
                // req already ready
                DNDS_check_throw(nRecvPushReq <= PushReqVec->size());
                if (!PushReqVec->empty())
                {
                    if (MPI::CommStrategy::Instance().GetUseAsyncOneByOne())
                    {
                    }
                    else
                        MPI_Startall(PushReqVec->size(), PushReqVec->data());
                }
            }
            else if (commTypeCurrent == MPI::CommStrategy::InSituPack)
            {
                InSituPackStartPush(B);
            }
            else
            {
                DNDS_check_throw(false);
            }
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().claim(pushSendSize, mpi.rank);
#endif

            PerformanceTimer::Instance().StopTimer(PerformanceTimer::TimerType::Comm);
        }

        void InSituPackStartPull(DeviceBackend B)
        {
            if (B != DeviceBackend::Unknown)
                DNDS_check_throw_info(false, "in-situ pack not yet implemented for device");
            nRecvPullReq = 0;
            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // pull
                MPI_Aint pullDisp = 0;
                MPI_int pullSize = 0; // same as pushSizes
                auto gRPtr = son->operator[](index(pLGhostMapping->ghostStart[r + 1]));
                auto gLPtr = son->operator[](index(pLGhostMapping->ghostStart[r]));
                auto ghostSpan = gRPtr - gLPtr;
                pullSize = MPI_int(ghostSpan);

                if (pullSize > 0)
                {
                    PullReqVec->emplace_back(MPI_REQUEST_NULL);
                    MPI_Irecv(gLPtr, pullSize * father->getTypeMult(), father->getDataType(), r, r + mpi.rank, mpi.comm, &PullReqVec->back());
                    nRecvPullReq++;
                }
            }
            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // push
                MPI_int pushNumber = pLGhostMapping->pushIndexSizes[r];
                // std::cout << "PN" << pushNumber << std::endl;
                if (pushNumber > 0)
                {
                    index nPushData{0};
                    for (index i = 0; i < pushNumber; i++)
                    {
                        auto loc = pushingIndexLocal.at(pLGhostMapping->pushIndexStarts[r] + i);
                        index nPush = 0;
                        if constexpr (_dataLayout == CSR)
                            nPush = father->RowSizeField(loc);
                        if constexpr (isTABLE_Max(_dataLayout)) //! init sizes
                            nPush = father->RowSize(loc);
                        if constexpr (isTABLE_Fixed(_dataLayout))
                            nPush = father->RowSizeField();
                        nPushData += nPush;
                    }
                    inSituBuffer.emplace_back(nPushData);
                    nPushData = 0;
                    for (index i = 0; i < pushNumber; i++)
                    {
                        auto loc = pushingIndexLocal.at(pLGhostMapping->pushIndexStarts[r] + i);
                        index nPush = 0;
                        if constexpr (_dataLayout == CSR)
                            nPush = father->RowSizeField(loc);
                        if constexpr (isTABLE_Max(_dataLayout)) //! init sizes
                            nPush = father->RowSize(loc);
                        if constexpr (isTABLE_Fixed(_dataLayout))
                            nPush = father->RowSizeField();
                        std::copy((*father)[loc], (*father)[loc] + nPush, inSituBuffer.back().begin() + nPushData);
                        nPushData += nPush;
                    }
                    PullReqVec->emplace_back(MPI_REQUEST_NULL);
                    MPI_Issend(inSituBuffer.back().data(), nPushData * father->getTypeMult(), father->getDataType(),
                               r, mpi.rank + r, mpi.comm, &PullReqVec->back());
                }
            }
        }

        /// @brief Start all persistent pull requests (@ref MPI_Startall).
        /// @details After this call the sends/recvs are in flight; call
        /// #waitPersistentPull to consume the incoming ghost data.
        /// @param B Device backend; must match the one passed to #initPersistentPull.
        void startPersistentPull(DeviceBackend B = DeviceBackend::Unknown) // collective;
        {
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                DNDS_check_throw(B == pullDevice);
                DNDS_check_throw(nRecvPullReq <= PullReqVec->size());
                // req already ready
                if (!PullReqVec->empty())
                {
                    if (MPI::CommStrategy::Instance().GetUseAsyncOneByOne())
                    {
                    }
                    else
                        MPI_Startall(int(PullReqVec->size()), PullReqVec->data());
                }
            }
            else if (commTypeCurrent == MPI::CommStrategy::InSituPack)
            {
                InSituPackStartPull(B);
            }
            else
            {
                DNDS_check_throw(false);
            }
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().claim(pullSendSize, mpi.rank);
#endif

            PerformanceTimer::Instance().StopTimer(PerformanceTimer::TimerType::Comm);
        }

        /// @brief Wait for all outstanding persistent push requests to complete.
        void waitPersistentPush(DeviceBackend B = DeviceBackend::Unknown) // collective;
        {
            if (MPI::CommStrategy::Instance().GetUseStrongSyncWait())
                MPI::Barrier(mpi.comm);
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
            PushStatVec.resize(PushReqVec->size());
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().unclaim(pushSendSize);
#endif
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                // data alright
                if (!PushReqVec->empty())
                {
                    DNDS_check_throw(nRecvPushReq <= PushReqVec->size());
                    if (MPI::CommStrategy::Instance().GetUseAsyncOneByOne())
                    {
                        MPI_Startall(nRecvPushReq, PushReqVec->data());
                        for (int iReq = nRecvPushReq; iReq < PushReqVec->size(); iReq++)
                        {
                            MPI_Start(&PushReqVec->operator[](iReq));
                            MPI_Wait(&PushReqVec->operator[](iReq), MPI_STATUS_IGNORE);
                        }
                        MPI::WaitallAuto(nRecvPushReq, PushReqVec->data(), MPI_STATUSES_IGNORE);
                    }
                    else
                        MPI::WaitallAuto(PushReqVec->size(), PushReqVec->data(), MPI_STATUSES_IGNORE);
                }
            }
            else if (commTypeCurrent == MPI::CommStrategy::InSituPack)
            {
                if (B != DeviceBackend::Unknown)
                    DNDS_check_throw_info(false, "in-situ pack not yet implemented for device");
                if (!PushReqVec->empty())
                    MPI::WaitallAuto(PushReqVec->size(), PushReqVec->data(), PushStatVec.data());
                auto bufferVec = inSituBuffer.begin();
                for (MPI_int r = 0; r < mpi.size; r++)
                {
                    // push
                    DNDS_check_throw(bufferVec < inSituBuffer.end());
                    MPI_int pushNumber = pLGhostMapping->pushIndexSizes[r];
                    // std::cout << "PN" << pushNumber << std::endl;
                    if (pushNumber > 0)
                    {
                        index nPushData = 0;
                        for (index i = 0; i < pushNumber; i++)
                        {
                            auto loc = pushingIndexLocal.at(pLGhostMapping->pushIndexStarts[r] + i);
                            index nPush = 0;
                            if constexpr (_dataLayout == CSR)
                                nPush = father->RowSizeField(loc);
                            if constexpr (isTABLE_Max(_dataLayout)) //! init sizes
                                nPush = father->RowSize(loc);
                            if constexpr (isTABLE_Fixed(_dataLayout))
                                nPush = father->RowSizeField();
                            std::copy(bufferVec->begin() + nPushData, bufferVec->begin() + nPushData + nPush, (*father)[loc]);
                            nPushData += nPush;
                        }
                        bufferVec++;
                    }
                }
                inSituBuffer.clear();
                PushReqVec->clear();
            }
            else
            {
                DNDS_check_throw(false);
            }
            PerformanceTimer::Instance().StopTimer(PerformanceTimer::TimerType::Comm);
            if (MPI::CommStrategy::Instance().GetUseStrongSyncWait())
                MPI::Barrier(mpi.comm);
        }
        /// @brief Wait for all outstanding persistent pull requests. After this
        /// returns, the son array holds fresh ghost data.
        void waitPersistentPull(DeviceBackend B = DeviceBackend::Unknown) // collective;
        {
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
            PullStatVec.resize(PullReqVec->size());

#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().unclaim(pullSendSize);
#endif
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                // data alright
                if (!PullReqVec->empty())
                {
                    DNDS_check_throw(nRecvPullReq <= PullReqVec->size());
                    if (MPI::CommStrategy::Instance().GetUseAsyncOneByOne())
                    {
                        MPI_Startall(nRecvPullReq, PullReqVec->data());
                        for (int iReq = nRecvPullReq; iReq < PullReqVec->size(); iReq++)
                        {
                            MPI_Start(&PullReqVec->operator[](iReq));
                            MPI_Wait(&PullReqVec->operator[](iReq), MPI_STATUS_IGNORE);
                            // if (mpi.rank == 0)
                            //     log() << "waited a req" << std::endl;
                        }
                        MPI::WaitallAuto(nRecvPullReq, PullReqVec->data(), MPI_STATUSES_IGNORE);
                    }
                    else
                    {
                        MPI::WaitallAuto(PullReqVec->size(), PullReqVec->data(), MPI_STATUSES_IGNORE);
                    }
                }
            }
            else if (commTypeCurrent == MPI::CommStrategy::InSituPack)
            {
                if (B != DeviceBackend::Unknown)
                    DNDS_check_throw_info(false, "in-situ pack not yet implemented for device");
                if (!PullReqVec->empty())
                    MPI::WaitallAuto(PullReqVec->size(), PullReqVec->data(), PullStatVec.data());
                // std::cout << "waiting DONE" << std::endl;
                inSituBuffer.clear();
                PullReqVec->clear();
            }
            else
            {
                DNDS_check_throw(false);
            }
            PerformanceTimer::Instance().StopTimer(PerformanceTimer::TimerType::Comm);
        }

        /// @brief Wait on outstanding push requests then free them.
        void clearPersistentPush() // collective;
        {
            waitPersistentPush();
            PushReqVec->clear(); // stat vec is left untouched here
        }
        /// @brief Wait on outstanding pull requests then free them.
        void clearPersistentPull() // collective;
        {
            waitPersistentPull();
            PullReqVec->clear();
        }

        /// @brief Release the MPI derived datatypes built by #createMPITypes.
        /// Rebuild with another call if you wish to continue using the transformer.
        void clearMPITypes() // collective;
        {
            pPullTypeVec.reset();
            pPushTypeVec.reset();
        }

        /// @brief Drop the shared pointer to the global offsets table.
        void clearGlobalMapping() // collective;
        {
            pLGlobalMapping.reset();
        }

        /// @brief Drop the ghost mapping (#pLGhostMapping).
        void clearGhostMapping() // collective;
        {
            pLGhostMapping.reset();
        }

        /// @brief Convenience: init + start + wait + clear a single pull.
        /// @details Suitable when ghosts are updated only once (e.g., post-restart);
        /// use the persistent API in hot loops.
        void pullOnce() // collective;
        {
            initPersistentPull();
            startPersistentPull();
            waitPersistentPull();
            clearPersistentPull();
        }

        /// @brief Convenience: init + start + wait + clear a single push.
        void pushOnce() // collective;
        {
            initPersistentPush();
            startPersistentPush();
            waitPersistentPush();
            clearPersistentPush();
        }

        /// @brief Re-initialise persistent requests; useful after rebuilding MPI
        /// types but wanting to resume persistent comms. Idempotent w.r.t.
        /// whichever direction(s) were previously initialised.
        void reInitPersistentPullPush()
        {
            bool clearedPull{false}, clearedPush{false};
            if (!PullReqVec->empty())
            {
                clearedPull = true;
                waitPersistentPull();
                clearPersistentPull();
            }
            if (!PushReqVec->empty())
            {
                clearedPush = true;
                waitPersistentPush();
                clearPersistentPush();
            }
            if (clearedPull)
                initPersistentPull();
            if (clearedPush)
                initPersistentPush();
        }
    };

    /// @brief Type trait computing the ArrayTransformer type for a given Array type.
    template <class TArray>
    struct ArrayTransformerType
    {
        using Type = ArrayTransformer<typename TArray::value_type, TArray::rs, TArray::rm, TArray::al>;
    };

    template <class TArray>
    using ArrayTransformerType_t = typename ArrayTransformerType<TArray>::Type;

}
