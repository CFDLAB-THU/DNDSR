/// @file MPI.cpp
/// @brief Implementations of the MPI wrapper functions declared in @ref MPI.hpp:
/// retry-aware @ref Bcast/@ref Alltoall/@ref Alltoallv/@ref Allreduce/@ref Allgather/@ref Barrier
/// variants, lazy waits, singleton definitions, CUDA-aware probe.

#include <ctime>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <chrono>
#include <thread>
#include "MPI.hpp"
#include "Profiling.hpp"

#ifdef DNDS_UNIX_LIKE
#    include <sys/ptrace.h>
#    include <unistd.h>
#    include <sys/stat.h>
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
#    define NOMINMAX
#    include <Windows.h>
#    include <process.h>
#endif

#ifdef NDEBUG
#    define NDEBUG_DISABLED
#    undef NDEBUG
#endif

namespace DNDS::Debug
{
    bool IsDebugged()
    {

#ifdef DNDS_UNIX_LIKE
        std::ifstream fin("/proc/self/status"); // able to detect gdb
        std::string buf;
        int tpid = 0;
        while (!fin.eof())
        {
            fin >> buf;
            if (buf == "TracerPid:")
            {
                fin >> tpid;
                break;
            }
        }
        fin.close();
        return tpid != 0;
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
        return IsDebuggerPresent();
#endif
    }

    void MPIDebugHold(const MPIInfo &mpi)
    {
#ifdef DNDS_UNIX_LIKE
        MPISerialDo(mpi, [&]
                    { log() << "Rank " << mpi.rank << " PID: " << getpid() << std::endl; });
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
        MPISerialDo(mpi, [&]
                    { log() << "Rank " << mpi.rank << " PID: " << _getpid() << std::endl; });
#endif
        int holdFlag = 1;
        while (holdFlag)
        {
            for (MPI_int ir = 0; ir < mpi.size; ir++)
            {
                int newDebugFlag = 0;
                if (mpi.rank == ir)
                {
                    newDebugFlag = int(IsDebugged());
                    MPI_Bcast(&newDebugFlag, 1, MPI_INT, ir, mpi.comm);
                }
                else
                    MPI_Bcast(&newDebugFlag, 1, MPI_INT, ir, mpi.comm);

                // std::cout << "DBG " << newDebugFlag;
                if (newDebugFlag)
                    holdFlag = 0;
            }
        }
    }

    bool isDebugging = false;
}

namespace DNDS
{
    void assert_false_info_mpi(const char *expr, const char *file, int line, const std::string &info, const DNDS::MPIInfo &mpi)
    {
        std::cerr << ::DNDS::getTraceString() << "\n";
        std::cerr << "\033[91m DNDS_assertion failed\033[39m: \"" << expr << "\"  at [  " << file << ":" << line << "  ]\n"
                  << info << std::endl;
        if (DNDS::Debug::isDebugging)
            MPI_Barrier(mpi.comm);
        std::abort();
    }
}
namespace DNDS
{
    MPIBufferHandler &MPIBufferHandler::Instance()
    {
        static MPIBufferHandler instance;
        return instance;
    }
}

namespace DNDS
{
    std::string getTimeStamp(const MPIInfo &mpi)
    {
        auto result = static_cast<int64_t>(std::time(nullptr));
        std::array<char, 512> bufTime{};
        std::array<char, 512 + 32> buf{};
        int64_t pid = 0;
#ifdef DNDS_UNIX_LIKE
        // pid = Debug::getpid();
        pid = getpid();
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
        // pid = Debug::GetCurrentProcessId();
        pid = GetCurrentProcessId();
#endif
        MPI_Bcast(&result, 1, MPI_INT64_T, 0, mpi.comm);
        MPI_Bcast(&pid, 1, MPI_INT64_T, 0, mpi.comm);

        auto time_result = static_cast<time_t>(result);

        std::strftime(bufTime.data(), 512, "%F_%H-%M-%S", std::localtime(&time_result));

        long pidc = static_cast<long>(pid);
        std::sprintf(buf.data(), "%s_%ld", bufTime.data(), pidc);

        return {buf.data()};
    }
}

namespace DNDS::MPI
{

#define start_timer PerformanceTimer::Instance().StartTimer(PerformanceTimer::Comm)
#define stop_timer PerformanceTimer::Instance().StopTimer(PerformanceTimer::Comm)
    /// @brief dumb wrapper
    MPI_int Bcast(void *buf, MPI_int num, MPI_Datatype type, MPI_int source_rank, MPI_Comm comm)
    {
        int ret{0};
        start_timer;
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            ret = MPI_Bcast(buf, num, type, source_rank, comm);
        else
        {
            MPI_Request req{MPI_REQUEST_NULL};
            ret = MPI_Ibcast(buf, num, type, source_rank, comm, &req);
            ret = MPI::WaitallLazy(1, &req, MPI_STATUSES_IGNORE, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
        }
        stop_timer;
        return ret;
    }

    MPI_int Alltoall(void *send, MPI_int sendNum, MPI_Datatype typeSend, void *recv, MPI_int recvNum, MPI_Datatype typeRecv, MPI_Comm comm)
    {
        int ret{0};
        start_timer;
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            ret = MPI_Alltoall(send, sendNum, typeSend, recv, recvNum, typeRecv, comm);
        else
        {
            MPI_Request req{MPI_REQUEST_NULL};
            ret = MPI_Ialltoall(send, sendNum, typeSend, recv, recvNum, typeRecv, comm, &req);
            ret = MPI::WaitallLazy(1, &req, MPI_STATUSES_IGNORE, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
        }
        stop_timer;
        return ret;
    }

    MPI_int Alltoallv(
        void *send, MPI_int *sendSizes, MPI_int *sendStarts, MPI_Datatype sendType,
        void *recv, MPI_int *recvSizes, MPI_int *recvStarts, MPI_Datatype recvType, MPI_Comm comm)
    {
        int ret{0};
        start_timer;
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            ret = MPI_Alltoallv(
                send, sendSizes, sendStarts, sendType,
                recv, recvSizes, recvStarts, recvType, comm);
        else
        {
            MPI_Request req{MPI_REQUEST_NULL};
            ret = MPI_Ialltoallv(send, sendSizes, sendStarts, sendType,
                                 recv, recvSizes, recvStarts, recvType, comm, &req);
            ret = MPI::WaitallLazy(1, &req, MPI_STATUSES_IGNORE, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
        }
        stop_timer;
        return ret;
    }

    MPI_int Allreduce(const void *sendbuf, void *recvbuf, MPI_int count,
                      MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    {
        int ret{0};
        start_timer;
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            ret = MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        else
        {
            MPI_Request req{MPI_REQUEST_NULL};
            ret = MPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, &req);
            ret = MPI::WaitallLazy(1, &req, MPI_STATUSES_IGNORE, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
        }
        stop_timer;
        return ret;
    }

    MPI_int Scan(const void *sendbuf, void *recvbuf, MPI_int count,
                 MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    {
        int ret{0}; // todo: add wait lazy?
        start_timer;
        ret = MPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);
        stop_timer;
        return ret;
    }

    MPI_int Allgather(const void *sendbuf, MPI_int sendcount, MPI_Datatype sendtype,
                      void *recvbuf, MPI_int recvcount,
                      MPI_Datatype recvtype, MPI_Comm comm)
    {
        int ret{0};
        start_timer;
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            ret = MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        else
        {
            MPI_Request req{MPI_REQUEST_NULL};
            ret = MPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, &req);
            ret = MPI::WaitallLazy(1, &req, MPI_STATUSES_IGNORE, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
        }
        stop_timer;
        return ret;
    }

    MPI_int Barrier(MPI_Comm comm)
    {
        int ret{0};
        start_timer;
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            ret = MPI_Barrier(comm);
        else
            ret = MPI::BarrierLazy(comm, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
        stop_timer;
        return ret;
    }

    MPI_int BarrierLazy(MPI_Comm comm, uint64_t checkNanoSecs)
    {
        MPI_Request req{MPI_REQUEST_NULL};
        MPI_Status stat;
        MPI_Ibarrier(comm, &req);
        MPI_int ret = MPI::WaitallLazy(1, &req, &stat, checkNanoSecs);
        if (req != MPI_REQUEST_NULL)
            MPI_Request_free(&req);
        return ret;
    }

    MPI_int WaitallLazy(MPI_int count, MPI_Request *reqs, MPI_Status *statuses, uint64_t checkNanoSecs)
    {
        MPI_int flag = 0;
        MPI_int ret = 0;
        while (!flag)
        {
            ret = MPI_Testall(count, reqs, &flag, statuses);
            std::this_thread::sleep_for(std::chrono::nanoseconds(checkNanoSecs));
        }
        return ret;
    }

    MPI_int WaitallAuto(MPI_int count, MPI_Request *reqs, MPI_Status *statuses)
    {
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            return MPI_Waitall(count, reqs, statuses);
        else
            return MPI::WaitallLazy(count, reqs, statuses, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
    }

#undef start_timer
#undef stop_timer

}

namespace DNDS::MPI
{
    bool isCudaAware()
    {
#ifdef OPEN_MPI
        return 1 == MPIX_Query_cuda_support();
#else
        return false;
#endif
    }
}

namespace DNDS::MPI
{
    ResourceRecycler &ResourceRecycler::Instance()
    {
        static ResourceRecycler recycler;
        return recycler;
    }

    void ResourceRecycler::RegisterCleaner(void *p, std::function<void()> nCleaner)
    {
        DNDS_assert(cleaners.count(p) == 0);
        cleaners.emplace(std::make_pair(p, std::move(nCleaner)));
    }

    void ResourceRecycler::RemoveCleaner(void *p)
    {
        DNDS_assert(cleaners.count(p) == 1);
        cleaners.erase(p);
    }

    void ResourceRecycler::clean()
    {
        for (auto &[k, f] : cleaners)
            f();
    }
}

namespace DNDS::MPI
{
    CommStrategy::CommStrategy()
    {
        try
        {
            auto *ret = std::getenv("DNDS_USE_LAZY_WAIT");
            if (ret != nullptr && (std::stod(ret) != 0))
            {
                _use_lazy_wait = std::stod(ret);
                auto mpi = MPIInfo();
                mpi.setWorld();
                // std::cout << mpi.rank << std::endl;
                if (mpi.rank == 0)
                    log() << "Detected DNDS_USE_LAZY_WAIT, setting to " << _use_lazy_wait << std::endl;
                MPI::BarrierLazy(mpi.comm, static_cast<uint64_t>(_use_lazy_wait));
            }
        }
        catch (...)
        {
        }
        try
        {
            auto *ret = std::getenv("DNDS_ARRAY_STRATEGY_USE_IN_SITU");
            if (ret != nullptr && (std::stoi(ret) != 0))
            {
                _array_strategy = InSituPack;
                auto mpi = MPIInfo();
                mpi.setWorld();
                if (mpi.rank == 0)
                    log() << "Detected DNDS_ARRAY_STRATEGY_USE_IN_SITU, setting" << std::endl;
                if (_use_lazy_wait)
                    MPI::BarrierLazy(mpi.comm, static_cast<uint64_t>(_use_lazy_wait));
                else
                    MPI_Barrier(mpi.comm);
            }
        }
        catch (...)
        {
        }
        try
        {
            auto *ret = std::getenv("DNDS_USE_STRONG_SYNC_WAIT");
            if (ret != nullptr && (std::stoi(ret) != 0))
            {
                _use_strong_sync_wait = true;
                auto mpi = MPIInfo();
                mpi.setWorld();
                if (mpi.rank == 0)
                    log() << "Detected DNDS_USE_STRONG_SYNC_WAIT, setting" << std::endl;
                if (_use_lazy_wait)
                    MPI::BarrierLazy(mpi.comm, static_cast<uint64_t>(_use_lazy_wait));
                else
                    MPI_Barrier(mpi.comm);
            }
        }
        catch (...)
        {
        }
        try
        {
            auto *ret = std::getenv("DNDS_USE_ASYNC_ONE_BY_ONE");
            if (ret != nullptr && (std::stoi(ret) != 0))
            {
                _use_async_one_by_one = true;
                auto mpi = MPIInfo();
                mpi.setWorld();
                if (mpi.rank == 0)
                    log() << "Detected DNDS_USE_ASYNC_ONE_BY_ONE, setting" << std::endl;
                if (bool(_use_lazy_wait))
                    MPI::BarrierLazy(mpi.comm, static_cast<uint64_t>(_use_lazy_wait));
                else
                    MPI_Barrier(mpi.comm);
            }
        }
        catch (...)
        {
        }
    }

    CommStrategy &CommStrategy::Instance()
    {
        static CommStrategy strategy;
        return strategy;
    }

    CommStrategy::ArrayCommType CommStrategy::GetArrayStrategy()
    {
        return _array_strategy;
    }

    void CommStrategy::SetArrayStrategy(CommStrategy::ArrayCommType t)
    {
        _array_strategy = t;
    }

    bool CommStrategy::GetUseStrongSyncWait() const
    {
        return _use_strong_sync_wait;
    }

    bool CommStrategy::GetUseAsyncOneByOne() const
    {
        return _use_async_one_by_one;
    }

    double CommStrategy::GetUseLazyWait() const
    {
        return _use_lazy_wait;
    }
}

namespace DNDS // TODO: get a concurrency header
{
    std::mutex HDF_mutex;
}

#ifdef NDEBUG_DISABLED
#    define NDEBUG
#    undef NDEBUG_DISABLED
#endif
