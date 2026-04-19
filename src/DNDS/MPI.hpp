#pragma once
/// @file MPI.hpp
/// @brief MPI wrappers: MPIInfo, collective operations, type mapping, CommStrategy.
/// @par Unit Test Coverage (test_MPI.cpp, MPI np=1,2,4)
/// - MPIInfo: constructor, setWorld, operator==, field validation
/// - MPIWorldSize, MPIWorldRank
/// - Allreduce (MPI_SUM, MPI_MAX for real/index), AllreduceOneReal, AllreduceOneIndex
/// - Scan (MPI_SUM), Allgather, Bcast, Barrier, Alltoall
/// - BasicType_To_MPIIntType (scalar, std::array, Eigen::Matrix)
/// - CommStrategy: get/set HIndexed/InSituPack
/// @par Not Yet Tested
/// - Alltoallv, WaitallLazy, WaitallAuto, BarrierLazy
/// - MPIBufferHandler, MPITypePairHolder, MPIReqHolder (tested indirectly via ArrayTransformer)
/// - MPI::ResourceRecycler, MPISerialDo, InsertCheck
/// - Sub-communicators, CommStrategy functional impact on ArrayTransformer

#include <vector>
#include <fstream>

#include <mutex>

#include "Defines.hpp"
DISABLE_WARNING_PUSH
// disable mpicxx 's many warnings in intel oneAPI mpi's header
DISABLE_WARNING_UNUSED_VALUE
#include "mpi.h"
DISABLE_WARNING_POP

#ifdef OPEN_MPI
#    include <mpi-ext.h> // Open MPI extension header
#endif

#ifdef NDEBUG
#    define NDEBUG_DISABLED
#    undef NDEBUG
#endif

namespace DNDS
{

    /// @brief MPI counterpart type for `MPI_int` (= C `int`). Used for counts
    /// and ranks in MPI calls.
    using MPI_int = int;
    /// @brief MPI-compatible address/offset type (= `MPI_Aint`, 64-bit on all
    /// supported platforms). Used by the `hindexed` datatype machinery.
    using MPI_index = MPI_Aint;
#define MAX_MPI_int INT32_MAX
#define MAX_MPI_Aint INT64_MAX
    static_assert(sizeof(MPI_Aint) == 8);

    /// @brief Vector of MPI counts.
    using tMPI_sizeVec = std::vector<MPI_int>;
    /// @brief Alias for #tMPI_sizeVec; used where the name "int vec" reads better.
    using tMPI_intVec = tMPI_sizeVec;
    /// @brief Vector of MPI_Aint byte-offsets for hindexed datatypes.
    using tMPI_indexVec = std::vector<MPI_index>;
    /// @brief Alias for #tMPI_indexVec to match `MPI_Aint` terminology.
    using tMPI_AintVec = tMPI_indexVec;

    /// @brief Vector of `MPI_Status`, for `MPI_Waitall` / `MPI_Testall`.
    using tMPI_statVec = std::vector<MPI_Status>;
    /// @brief Vector of `MPI_Request`, for persistent / nonblocking calls.
    using tMPI_reqVec = std::vector<MPI_Request>;

    /**
     * @brief Map a DNDS integer type size to an MPI signed-integer datatype.
     * @details Compile-time selects @ref MPI_INT64_T or @ref MPI_INT32_T based on `sizeof(Tbasic)`.
     * Used by @ref DNDS_MPI_INDEX.
     */
    template <class Tbasic>
    constexpr MPI_Datatype __DNDSToMPITypeInt()
    {
        static_assert(sizeof(Tbasic) == 8 || sizeof(Tbasic) == 4, "DNDS::Tbasic is not right size");
        return sizeof(Tbasic) == 8 ? MPI_INT64_T : (sizeof(Tbasic) == 4 ? MPI_INT32_T : MPI_DATATYPE_NULL);
    }

    /**
     * @brief Map a DNDS floating-point type size to an MPI datatype.
     * @details Compile-time selects @ref MPI_REAL8 or @ref MPI_REAL4 based on `sizeof(Tbasic)`.
     * Used by @ref DNDS_MPI_REAL.
     */
    template <class Tbasic>
    constexpr MPI_Datatype __DNDSToMPITypeFloat()
    {
        static_assert(sizeof(Tbasic) == 8 || sizeof(Tbasic) == 4, "DNDS::Tbasic is not right size");
        return sizeof(Tbasic) == 8 ? MPI_REAL8 : (sizeof(Tbasic) == 4 ? MPI_REAL4 : MPI_DATATYPE_NULL);
    }

    /// @brief MPI datatype matching #index (= @ref MPI_INT64_T).
    const MPI_Datatype DNDS_MPI_INDEX = __DNDSToMPITypeInt<index>();
    /// @brief MPI datatype matching #real (= @ref MPI_REAL8).
    const MPI_Datatype DNDS_MPI_REAL = __DNDSToMPITypeFloat<real>();

    //! here are some reasons to upgrade to C++20...
    // detect if have CommMult and CommType static methods

    /// @brief SFINAE trait detecting a static @ref CommMult member in T.
    template <typename T, typename = void>
    struct has_static_CommMult : std::false_type
    {
    };

    template <typename T>
    struct has_static_CommMult<T, std::void_t<decltype(T::CommMult())>> : std::true_type
    {
    };

    /// @brief SFINAE trait detecting a static @ref CommType member in T.
    template <typename T, typename = void>
    struct has_static_CommType : std::false_type
    {
    };

    template <typename T>
    struct has_static_CommType<T, std::void_t<decltype(T::CommType())>> : std::true_type
    {
    };

    /// @brief Dispatch to a user-provided @ref CommPair / @ref CommMult+@ref CommType pair on `T`.
    /// @details Last resort for types that are not scalars, plain arrays, or
    /// real Eigen matrices. `T` must either expose static `CommMult()` +
    /// `CommType()` methods, or a static `CommPair()` returning the same pair.
    template <class T>
    std::pair<MPI_Datatype, MPI_int> BasicType_To_MPIIntType_Custom()
    {
        if constexpr (has_static_CommMult<T>::value && has_static_CommType<T>::value)
            return std::make_pair(T::CommType(), T::CommMult());
        else
            return T::CommPair(); // last resort
    }

    /**
     * @brief Deduce an `(MPI_Datatype, count)` pair that represents a `T` value.
     *
     * @details Compile-time dispatch:
     * - builtin float / int types map to their obvious `MPI_*` datatypes, count = 1;
     * - C-style arrays (`T[N]`) recurse into the element type and multiply the count;
     * - `std::array<T, N>` recurses into `T::value_type` and multiplies by `N`;
     * - fixed-size real Eigen matrices map to @ref DNDS_MPI_REAL with count `sizeof(T)/sizeof(real)`;
     * - otherwise falls back to @ref BasicType_To_MPIIntType_Custom.
     *
     * Used throughout ghost-communication and collective code to avoid hand-
     * writing datatype mapping for every MPI call.
     *
     * @note Not `constexpr` because OpenMPI handles are not constant-expressions.
     */
    template <class T> // TODO: see if an array is bounded
    //! Warning, not const-expr since OpenMPI disallows it
    std::pair<MPI_Datatype, MPI_int> BasicType_To_MPIIntType()
    {
        static const auto badReturn = std::make_pair(MPI_Datatype(MPI_DATATYPE_NULL), MPI_int(-1));
        if constexpr (std::is_scalar_v<T>)
        {
            if constexpr (std::is_same_v<T, float>)
                return std::make_pair(MPI_Datatype(MPI_FLOAT), MPI_int(1));
            if constexpr (std::is_same_v<T, double>)
                return std::make_pair(MPI_Datatype(MPI_DOUBLE), MPI_int(1));
            if constexpr (std::is_same_v<T, long double>)
                return std::make_pair(MPI_Datatype(MPI_LONG_DOUBLE), MPI_int(1));

            if constexpr (std::is_same_v<T, int8_t>)
                return std::make_pair(MPI_Datatype(MPI_INT8_T), MPI_int(1));
            if constexpr (std::is_same_v<T, int16_t>)
                return std::make_pair(MPI_Datatype(MPI_INT16_T), MPI_int(1));
            if constexpr (std::is_same_v<T, int32_t>)
                return std::make_pair(MPI_Datatype(MPI_INT32_T), MPI_int(1));
            if constexpr (std::is_same_v<T, int64_t>)
                return std::make_pair(MPI_Datatype(MPI_INT64_T), MPI_int(1));

            if constexpr (sizeof(T) == 1)
                return std::make_pair(MPI_Datatype(MPI_UINT8_T), MPI_int(1));
            else if constexpr (sizeof(T) == 2)
                return std::make_pair(MPI_Datatype(MPI_UINT16_T), MPI_int(1));
            else if constexpr (sizeof(T) == 4)
                return std::make_pair(MPI_Datatype(MPI_UINT32_T), MPI_int(1));
            else if constexpr (sizeof(T) == 8)
                return std::make_pair(MPI_Datatype(MPI_UINT64_T), MPI_int(1));
            else
                return BasicType_To_MPIIntType_Custom<T>();
        }
        else if constexpr (std::is_array_v<T>)
        {
            std::pair<MPI_Datatype, MPI_int> SizCom = BasicType_To_MPIIntType<std::remove_extent_t<T>>();
            return std::make_pair(SizCom.first, SizCom.second * std::extent_v<T>);
        }
        else if constexpr (std::is_trivially_copyable_v<T>)
        {
            if constexpr (Meta::is_std_array_v<T>)
                return std::make_pair(
                    BasicType_To_MPIIntType<typename T::value_type>().first,
                    BasicType_To_MPIIntType<typename T::value_type>().second * T().size());
            else
                return BasicType_To_MPIIntType_Custom<T>();
        }
        else if constexpr (Meta::is_fixed_data_real_eigen_matrix_v<T>)
            return std::make_pair(DNDS_MPI_REAL, MPI_int(divide_ceil(sizeof(T), sizeof(real))));
        else
            return BasicType_To_MPIIntType_Custom<T>();
        // else
        //     return badReturn;
    }

    /**
     * @brief Lightweight bundle of an MPI communicator and the calling rank's coordinates.
     *
     * @details The canonical "where am I in the parallel world" object passed
     * almost everywhere in DNDSR. Cheap to copy (three ints). Two-phase
     * construction is supported:
     *  - default-construct, then call #setWorld (or the `MPI_Comm` ctor) once
     *    `MPI_Init` has run.
     *
     * Comparison (#operator==) tests exact equality of the triple `(comm, rank, size)`.
     */
    struct MPIInfo
    {
        /// @brief The underlying MPI communicator handle.
        MPI_Comm comm = MPI_COMM_NULL;
        /// @brief This rank's 0-based index within #comm (`-1` until initialised).
        int rank = -1;
        /// @brief Number of ranks in #comm (`-1` until initialised).
        int size = -1;

        MPIInfo() = default;

        /// @brief Wrap an existing MPI communicator; queries rank and size.
        MPIInfo(MPI_Comm ncomm)
        {
            comm = ncomm;
            int ierr;
            ierr = MPI_Comm_rank(comm, &rank), DNDS_assert(ierr == MPI_SUCCESS);
            ierr = MPI_Comm_size(comm, &size), DNDS_assert(ierr == MPI_SUCCESS);
        }

        /// @brief Low-level constructor for callers that already know `(rank, size)`.
        /// @warning Bug: the fourth argument stores `r` into `size` too; callers
        /// currently pass matching values. Prefer the single-arg MPI_Comm ctor.
        MPIInfo(MPI_Comm nc, int r, int s) : comm(nc), rank(r), size(r)
        {
        }

        /// @brief Initialise the object to `MPI_COMM_WORLD`. Requires `MPI_Init` to have run.
        void setWorld()
        {
            comm = MPI_COMM_WORLD;
            int ierr;
            ierr = MPI_Comm_rank(comm, &rank), DNDS_assert(ierr == MPI_SUCCESS);
            ierr = MPI_Comm_size(comm, &size), DNDS_assert(ierr == MPI_SUCCESS);
        }

        /// @brief Exact triple equality.
        bool operator==(const MPIInfo &r) const
        {
            return (comm == r.comm) && (rank == r.rank) && (size == r.size);
        }
    };

    namespace MPI
    {
        /**
         * @brief Singleton that tracks and releases long-lived MPI resources at
         * `MPI_Finalize` time.
         *
         * @details MPI communicators, derived datatypes, and persistent requests
         * must be released before `MPI_Finalize`; otherwise they leak memory and
         * MPICH prints warnings. Several DNDSR objects (@ref DNDS::MPITypePairHolder "MPITypePairHolder",
         * @ref DNDS::MPIReqHolder "MPIReqHolder") register themselves here so that `MPI::Finalize()`
         * can call their cleanup callbacks even if the C++ lifetime would
         * outlive the MPI runtime (e.g., static destructors).
         *
         * Thread-safe C++11 singleton. Intended to be created under `MPI_COMM_WORLD`.
         */
        class ResourceRecycler
        {
        private:
            std::unordered_map<void *, std::function<void()>> cleaners;

            ResourceRecycler(){}; // implemented
            ResourceRecycler(const ResourceRecycler &);
            ResourceRecycler &operator=(const ResourceRecycler &);

        public:
            /// @brief Access the process-wide singleton.
            static ResourceRecycler &Instance();
            /**
             * @brief Register a cleanup callback keyed by `p`.
             * @warning Must be paired with @ref RemoveCleaner when `p` is destroyed,
             *          else dangling pointers will be invoked by #clean.
             */
            void RegisterCleaner(void *p, std::function<void()> nCleaner);
            /// @brief Remove a previously-registered cleaner.
            void RemoveCleaner(void *p);
            /// @brief Invoke all registered cleaners and drop them. Called by
            /// `MPI::Finalize()`.
            void clean();
        };
    }

    /// @brief Convenience: `MPI_Comm_size(MPI_COMM_WORLD)`.
    inline MPI_int MPIWorldSize()
    {
        MPI_int ret{0};
        MPI_Comm_size(MPI_COMM_WORLD, &ret);
        return ret;
    }

    /// @brief Convenience: `MPI_Comm_rank(MPI_COMM_WORLD)`.
    inline MPI_int MPIWorldRank()
    {
        MPI_int ret{0};
        MPI_Comm_rank(MPI_COMM_WORLD, &ret);
        return ret;
    }

    /// @brief Format a human-readable timestamp using the calling rank as context.
    std::string getTimeStamp(const MPIInfo &mpi);

/// @brief Debug helper: barrier + print "CHECK <info> RANK r @ fn @ file:line".
/// @details Compiled out when either `NDEBUG` or `NINSERT` is defined. Used to
/// trace parallel execution during development; see @ref InsertCheck.
#define DNDS_MPI_InsertCheck(mpi, info) \
    InsertCheck(mpi, info, __FUNCTION__, __FILE__, __LINE__)

    using tMPI_typePairVec = std::vector<std::pair<MPI_int, MPI_Datatype>>;
    /**
     * @brief RAII vector of `(count, MPI_Datatype)` pairs that frees every
     * committed datatype when destroyed.
     *
     * @details Used by @ref DNDS::ArrayTransformer "ArrayTransformer" to hold the derived datatypes it
     * builds via `MPI_Type_create_hindexed`. Construction is channelled through
     * the static #create factory so instances are always owned by
     * `shared_ptr<MPITypePairHolder>` and correctly registered with the
     * @ref DNDS::ResourceRecycler "ResourceRecycler".
     */
    struct MPITypePairHolder : public tMPI_typePairVec, public std::enable_shared_from_this<MPITypePairHolder>
    {
        using tSelf = MPITypePairHolder;
        using tBase = tMPI_typePairVec;

    private:
        /// @brief RAII guard restricting construction to shared_ptr factory.
        struct shared_ctor_guard // make_shared needs a public ctor but give a private arg
        {
        };

    public:
        /// @brief Perfect-forwarding factory; returns `shared_ptr<MPITypePairHolder>`.
        template <typename... Args>
        MPITypePairHolder(shared_ctor_guard g, Args &&...args) : tBase(std::forward<Args>(args)...)
        {
            if (!(std::shared_ptr<tSelf>(this, [](tSelf *) {}).use_count() == 1))
                throw std::runtime_error("tSelf must be created via shared_ptr");
            MPI::ResourceRecycler::Instance().RegisterCleaner(this, [this]()
                                                              { this->clear(); });
        }

        /// @brief Only public path to construct an instance; forwards to the private constructor.
        template <typename... Args>
        static ssp<MPITypePairHolder> create(Args &&...args)
        {
            return std::make_shared<MPITypePairHolder>(shared_ctor_guard{}, std::forward<Args>(args)...);
        }
        ~MPITypePairHolder()
        {
            this->clear();
            MPI::ResourceRecycler::Instance().RemoveCleaner(this);
        }
        /// @brief Free every committed datatype and empty the vector.
        void clear()
        {
            for (auto &i : (*this))
                if (i.first >= 0 && i.second != 0 && i.second != MPI_DATATYPE_NULL)
                    MPI_Type_free(&i.second); //, std::cout << "Free Type" << std::endl;
            this->tMPI_typePairVec::clear();
        }
    };

    /// @brief Shared-pointer alias to @ref DNDS::MPITypePairHolder "MPITypePairHolder".
    using tpMPITypePairHolder = ssp<MPITypePairHolder>;
    /**
     * @brief RAII vector of `MPI_Request`s that frees each non-null handle when destroyed.
     *
     * @details Mirror of @ref DNDS::MPITypePairHolder "MPITypePairHolder" for MPI requests (persistent or
     * transient). Used by @ref DNDS::ArrayTransformer "ArrayTransformer" for send/recv request sets.
     * Construction is likewise channelled through the static #create factory.
     */
    struct MPIReqHolder : public tMPI_reqVec, public std::enable_shared_from_this<MPIReqHolder>
    {
        using tSelf = MPIReqHolder;
        using tBase = tMPI_reqVec;

    private:
        /// @brief RAII guard restricting construction to shared_ptr factory.
        struct shared_ctor_guard // make_shared needs a public ctor but give a private arg
        {
        };

    public:
        /// @brief Perfect-forwarding factory; returns `shared_ptr<MPIReqHolder>`.
        template <typename... Args>
        MPIReqHolder(shared_ctor_guard g, Args &&...args) : tBase(std::forward<Args>(args)...)
        {
            if (!(std::shared_ptr<tSelf>(this, [](tSelf *) {}).use_count() == 1))
                throw std::runtime_error("tSelf must be created via shared_ptr");
            MPI::ResourceRecycler::Instance().RegisterCleaner(this, [this]()
                                                              { this->clear(); });
        }

        /// @brief Only public path to construct an instance.
        template <typename... Args>
        static ssp<MPIReqHolder> create(Args &&...args)
        {
            return std::make_shared<MPIReqHolder>(shared_ctor_guard{}, std::forward<Args>(args)...);
        }
        ~MPIReqHolder()
        {
            this->clear();
            MPI::ResourceRecycler::Instance().RemoveCleaner(this);
        }
        /// @brief Free every non-null request and empty the vector.
        void clear()
        {
            for (auto &i : (*this))
                if (i != MPI_REQUEST_NULL)
                    MPI_Request_free(&i);
            this->tMPI_reqVec::clear();
        }
    };

}

namespace DNDS::Debug
{
    /// @brief Whether the current process is running under a debugger.
    /// Implemented via `/proc/self/status TracerPid` on Linux.
    bool IsDebugged();
    /// @brief If #isDebugging is set, block every rank in a busy-wait loop so
    /// the user can attach a debugger and inspect state. Exit by setting
    /// `isDebugging = false` in the debugger.
    void MPIDebugHold(const MPIInfo &mpi);
    /// @brief Flag consulted by @ref MPIDebugHold and #assert_false_info_mpi.
    extern bool isDebugging;
}

// DNDS_assert_info_mpi is used to help barrier the process before exiting if DNDS::Debug::isDebugging is set
// remember to set a breakpoint here
namespace DNDS
{
    /// @brief MPI-aware assertion-failure reporter.
    /// @details Barriers before abort so every rank flushes its output; if
    /// @ref Debug::isDebugging is set, busy-waits to allow debugger attachment.
    void assert_false_info_mpi(const char *expr, const char *file, int line, const std::string &info, const DNDS::MPIInfo &mpi);
}
#ifdef DNDS_NDEBUG
#    define DNDS_assert_info_mpi(expr, mpi, info) (void(0))
#else
/// @brief Collective-aware variant of @ref DNDS_assert_info: every rank calls
/// into an MPI barrier before aborting, so logs are not interleaved.
#    define DNDS_assert_info_mpi(expr, mpi, info) \
        ((static_cast<bool>(expr))                \
             ? void(0)                            \
             : ::DNDS::assert_false_info_mpi(#expr, __FILE__, __LINE__, info, mpi))
#endif

namespace DNDS // TODO: get a concurrency header
{
    /// @brief Global mutex serialising host-side HDF5 calls.
    /// @details HDF5 is not thread-safe by default; this mutex guards all
    /// DNDSR HDF5 I/O when multiple CPU threads might touch the same file.
    extern std::mutex HDF_mutex;

    namespace MPI
    {
        /// @brief Return the MPI thread-support level the current process was initialised with.
        inline int GetMPIThreadLevel()
        {
            int ret;
            int ierr;
            ierr = MPI_Query_thread(&ret), DNDS_assert(ierr == MPI_SUCCESS);
            return ret;
        }

        /**
         * @brief Initialise MPI with thread support, honouring the
         * @ref DNDS_DISABLE_ASYNC_MPI environment override.
         *
         * @details
         *  - No env var or value `0`: request `MPI_THREAD_MULTIPLE` (full).
         *  - `1`: drop to `MPI_THREAD_SERIALIZED`.
         *  - `2`: drop to `MPI_THREAD_FUNNELED`.
         *  - `>=3`: `MPI_THREAD_SINGLE`.
         *
         * Aborts via `MPI_Abort` if the provided level is lower than requested.
         * Idempotent: if MPI is already initialised the call just queries the level.
         */
        inline MPI_int Init_thread(int *argc, char ***argv)
        {
            int init_flag{0};
            MPI_Initialized(&init_flag);

            int provided_MPI_THREAD_LEVEL{0};
            int needed_MPI_THREAD_LEVEL = MPI_THREAD_MULTIPLE;

            auto *env = std::getenv("DNDS_DISABLE_ASYNC_MPI");
            if (env != NULL && (std::stod(env) != 0))
            {
                int ienv = static_cast<int>(std::stod(env));
                if (ienv >= 1)
                    needed_MPI_THREAD_LEVEL = MPI_THREAD_SERIALIZED;
                if (ienv >= 2)
                    needed_MPI_THREAD_LEVEL = MPI_THREAD_FUNNELED;
                if (ienv >= 3)
                    needed_MPI_THREAD_LEVEL = MPI_THREAD_SINGLE;
            }
            int ret{0};
            if (!init_flag)
                ret = MPI_Init_thread(argc, argv, needed_MPI_THREAD_LEVEL, &provided_MPI_THREAD_LEVEL);
            else
                provided_MPI_THREAD_LEVEL = GetMPIThreadLevel();

            if (provided_MPI_THREAD_LEVEL < needed_MPI_THREAD_LEVEL)
            {
                printf("ERROR: The MPI library does not have full thread support\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            return ret;
        }

        /// @brief Release DNDSR-registered MPI resources then call `MPI_Finalize`.
        /// @details Idempotent: returns immediately if MPI has already been finalised.
        inline int Finalize()
        {
            MPI::ResourceRecycler::Instance().clean();
            int finalized{0};
            int err = MPI_Finalized(&finalized);
            if (!finalized)
                err |= MPI_Finalize();
            return err;
        }
    }
}

// MPI buffer handler
#define MPIBufferHandler_REPORT_CHANGE // for monitoring
namespace DNDS
{
    /**
     * @brief Process-singleton managing the buffer attached to MPI for
     * `MPI_Bsend` (buffered sends).
     *
     * @details Some algorithms (e.g., serialised writes) use buffered sends to
     * decouple sender from receiver. MPI requires the application to provide
     * the buffer via `MPI_Buffer_attach`. This singleton owns that buffer,
     * grows it on demand via #claim, and exposes a thin accounting layer
     * (#claim / #unclaim) so multiple components can share the buffer without
     * stepping on each other.
     *
     * Thread-safe construction on C++11; not MT-safe for concurrent claims.
     */
    class MPIBufferHandler // cxx11 + thread-safe singleton
    {
    private:
        std::vector<uint8_t> buf;

    public:
        using size_type = decltype(buf)::size_type;

    private:
        size_type claimed = 0;

    private:
        MPIBufferHandler()
        {
            uint8_t *obuf;
            int osize;
            MPI_Buffer_detach(reinterpret_cast<void *>(&obuf) /* caution */, &osize);

            buf.resize(1024ULL * 1024ULL);
            MPI_Buffer_attach(buf.data(), int(buf.size())); //! warning, bufsize could overflow
        }
        MPIBufferHandler(const MPIBufferHandler &);
        MPIBufferHandler &operator=(const MPIBufferHandler &);

    public:
        /// @brief Access the process-wide singleton.
        static MPIBufferHandler &Instance();
        /// @brief Current buffer size in bytes (fits in `MPI_int`; asserted).
        MPI_int size()
        {
            DNDS_assert(buf.size() <= MAX_MPI_int);
            return MPI_int(buf.size()); // could overflow!
        }
        /// @brief Reserve `cs` additional bytes, growing and re-attaching the
        /// MPI buffer if needed. `reportRank` is only used for diagnostic logs.
        void claim(MPI_Aint cs, int reportRank = 0)
        {
            if (buf.size() - claimed < static_cast<size_type>(cs))
            {
                // std::cout << "claim in " << std::endl;
                uint8_t *obuf;
                int osize;
                MPI_Buffer_detach(reinterpret_cast<void *>(&obuf) /* caution */, &osize);
#ifdef MPIBufferHandler_REPORT_CHANGE
                std::cout << "MPIBufferHandler: New BUf at " << reportRank << std::endl
                          << osize << std::endl;
#endif
                DNDS_assert(static_cast<size_type>(osize) == buf.size());
                buf.resize(claimed + cs);
                MPI_Buffer_attach(buf.data(), size_t_to_signed<MPI_int>(buf.size()));
#ifdef MPIBufferHandler_REPORT_CHANGE
                std::cout << " -> " << buf.size() << std::endl;
#endif
            }
            claimed += cs;
        }
        /// @brief Release `cs` previously-#claim ed bytes (only updates accounting;
        /// does not shrink the buffer).
        void unclaim(MPI_int cs)
        {
            DNDS_assert(size_t_to_signed<MPI_int>(claimed) >= cs);
            claimed -= cs;
        }
        /// @brief Direct pointer to the attached buffer (for diagnostics).
        void *getBuf()
        {
            return (void *)(buf.data());
        }
    };

}

namespace DNDS::MPI
{
    /// @brief Wrapper over `MPI_Bcast` that logs on error and goes through DNDSR retry logic.
    MPI_int Bcast(void *buf, MPI_int num, MPI_Datatype type, MPI_int source_rank, MPI_Comm comm);

    /// @brief Wrapper over `MPI_Alltoall` (fixed per-peer count).
    MPI_int Alltoall(void *send, MPI_int sendNum, MPI_Datatype typeSend, void *recv, MPI_int recvNum, MPI_Datatype typeRecv, MPI_Comm comm);

    /// @brief Wrapper over `MPI_Alltoallv` (variable per-peer counts + displacements).
    MPI_int Alltoallv(
        void *send, MPI_int *sendSizes, MPI_int *sendStarts, MPI_Datatype sendType,
        void *recv, MPI_int *recvSizes, MPI_int *recvStarts, MPI_Datatype recvType, MPI_Comm comm);

    /// @brief Wrapper over `MPI_Allreduce`.
    MPI_int Allreduce(const void *sendbuf, void *recvbuf, MPI_int count,
                      MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

    /// @brief Wrapper over `MPI_Scan` (inclusive prefix reduction).
    MPI_int Scan(const void *sendbuf, void *recvbuf, MPI_int count,
                 MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

    /// @brief Wrapper over `MPI_Allgather`.
    MPI_int Allgather(const void *sendbuf, MPI_int sendcount, MPI_Datatype sendtype,
                      void *recvbuf, MPI_int recvcount,
                      MPI_Datatype recvtype, MPI_Comm comm);

    /// @brief Wrapper over `MPI_Barrier`.
    MPI_int Barrier(MPI_Comm comm);

    /// @brief Polling barrier that sleeps `checkNanoSecs` ns between MPI_Test
    /// calls. Reduces CPU spin when many ranks wait unevenly.
    MPI_int BarrierLazy(MPI_Comm comm, uint64_t checkNanoSecs);

    /// @brief Like @ref WaitallAuto but sleeps `checkNanoSecs` ns between polls.
    MPI_int WaitallLazy(MPI_int count, MPI_Request *reqs, MPI_Status *statuses, uint64_t checkNanoSecs = 10000000);

    /// @brief Wait on an array of requests, choosing between `MPI_Waitall` and
    /// the lazy-poll variant based on @ref DNDS::CommStrategy "CommStrategy" settings.
    MPI_int WaitallAuto(MPI_int count, MPI_Request *reqs, MPI_Status *statuses);

    /// @brief Single-scalar Allreduce helper for reals (in-place, count = 1).
    inline void AllreduceOneReal(real &v, MPI_Op op, const MPIInfo &mpi)
    {
        Allreduce(MPI_IN_PLACE, &v, 1, DNDS_MPI_REAL, op, mpi.comm);
    }

    /// @brief Single-scalar Allreduce helper for indices (in-place, count = 1).
    inline void AllreduceOneIndex(index &v, MPI_Op op, const MPIInfo &mpi)
    {
        Allreduce(MPI_IN_PLACE, &v, 1, DNDS_MPI_INDEX, op, mpi.comm);
    }

}

namespace DNDS
{
    /**
     * @brief Execute `f` on each rank serially, in rank order.
     *
     * @details Inserts an `MPI_Barrier` before each rank's turn so that output
     * interleaving is deterministic. Useful for diagnostics where every rank
     * prints something about its own state.
     *
     * @tparam F  Callable with no arguments.
     */
    template <class F>
    inline void MPISerialDo(const MPIInfo &mpi, F f)
    { //! need some improvement: order could be bad?
        for (MPI_int i = 0; i < mpi.size; i++)
        {
            MPI::Barrier(mpi.comm);
            if (mpi.rank == i)
                f();
        }
    }
}

namespace DNDS::MPI
{
    /**
     * @brief Process-wide singleton that selects how @ref DNDS::ArrayTransformer "ArrayTransformer" packs
     * and waits for MPI messages.
     *
     * @details Settings affect every transformer:
     *  - @ref ArrayCommType: @ref HIndexed (default: `MPI_Type_create_hindexed`
     *    derived types) vs @ref InSituPack (manual `memcpy` into contiguous send/recv
     *    buffers). The latter can be faster on networks where derived types pay
     *    large unpacking overhead.
     *  - @ref UseStrongSyncWait: insert barriers around wait calls for easier
     *    profiling.
     *  - @ref UseAsyncOneByOne: issue per-peer @ref Isend/@ref Irecv instead of one
     *    persistent @ref Startall.
     *  - @ref UseLazyWait: poll interval (ns) used by @ref MPI::WaitallLazy.
     *
     * Must be constructed under `MPI_COMM_WORLD`. Thread-safe C++11 singleton.
     */
    class CommStrategy
    {
    public:
        /// @brief Which derived-type strategy @ref DNDS::ArrayTransformer "ArrayTransformer" should use.
        enum ArrayCommType
        {
            UnknownArrayCommType = 0, ///< Sentinel / uninitialised.
            HIndexed = 1,             ///< Use `MPI_Type_create_hindexed` derived types (default).
            InSituPack = 2,           ///< Manually pack / unpack into contiguous buffers.
        };

        static const int Ntype = 10;

    private:
        ArrayCommType _array_strategy = HIndexed;
        bool _use_strong_sync_wait = false;
        bool _use_async_one_by_one = false;
        double _use_lazy_wait = 0;

        CommStrategy();
        CommStrategy(const CommStrategy &);
        CommStrategy &operator=(const CommStrategy &);

    public:
        /// @brief Access the process-wide singleton.
        static CommStrategy &Instance();
        /// @brief Current array-pack strategy.
        ArrayCommType GetArrayStrategy();
        /// @brief Override the array-pack strategy (affects subsequently-created transformers).
        void SetArrayStrategy(ArrayCommType t);
        /// @brief Whether barriers are inserted around @ref Waitall for profiling.
        [[nodiscard]] bool GetUseStrongSyncWait() const;
        /// @brief Whether transformers should use one-by-one Isend/Irecv.
        [[nodiscard]] bool GetUseAsyncOneByOne() const;
        /// @brief Polling interval (ns) for @ref MPI::WaitallLazy. `0` means use `MPI_Waitall`.
        [[nodiscard]] double GetUseLazyWait() const;
    };
}

namespace DNDS::MPI
{
    /// @brief Runtime probe: is the current MPI implementation configured with
    /// CUDA-aware support? Affects whether arrays are transferred on-device or
    /// via the host round-trip.
    bool isCudaAware();
}

namespace DNDS
{
    /// @brief Barrier + annotated print used by @ref DNDS_MPI_InsertCheck.
    /// @details No-op in release builds (`NDEBUG` or `NINSERT` defined).
    inline void InsertCheck(const MPIInfo &mpi, const std::string &info = "",
                            const std::string &FUNCTION = "", const std::string &FILE = "", int LINE = -1)
    {
#if !(defined(NDEBUG) || defined(NINSERT))
        MPI::Barrier(mpi.comm);
        std::cout << "=== CHECK \"" << info << "\"  RANK " << mpi.rank << " ==="
                  << " @  FName: " << FUNCTION
                  << " @  Place: " << FILE << ":" << LINE << std::endl;
        MPI::Barrier(mpi.comm);
#endif
    }
}

#ifdef NDEBUG_DISABLED
#    define NDEBUG
#    undef NDEBUG_DISABLED
#endif
