
#include <ctime>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <chrono>
#include <thread>

#if defined(linux) || defined(_UNIX) || defined(__linux__)
#include <sys/ptrace.h>
#include <unistd.h>
#include <sys/stat.h>
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
#define NOMINMAX
#include <Windows.h>
#include <process.h>
#endif

#include "MPI.hpp"
#include "Profiling.hpp"

#ifdef NDEBUG
#define NDEBUG_DISABLED
#undef NDEBUG
#endif

namespace DNDS
{

    namespace Debug
    {
        bool IsDebugged()
        {

#if defined(linux) || defined(_UNIX) || defined(__linux__)
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
#if defined(linux) || defined(_UNIX) || defined(__linux__)
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
                    int newDebugFlag;
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
        int64_t result = static_cast<int64_t>(std::time(nullptr));
        char bufTime[512];
        char buf[512 + 32];
        int64_t pid = 0;
#if defined(linux) || defined(_UNIX) || defined(__linux__)
        // pid = Debug::getpid();
        pid = getpid();
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
        // pid = Debug::GetCurrentProcessId();
        pid = GetCurrentProcessId();
#endif
        MPI_Bcast(&result, 1, MPI_INT64_T, 0, mpi.comm);
        MPI_Bcast(&pid, 1, MPI_INT64_T, 0, mpi.comm);

        time_t time_result = static_cast<time_t>(result);

        std::strftime(bufTime, 512, "%F_%H-%M-%S", std::localtime(&time_result));

        long pidc = static_cast<long>(pid);
        std::sprintf(buf, "%s_%ld", bufTime, pidc);

        return std::string(buf);
    }
}

namespace DNDS::MPI
{

#define __start_timer PerformanceTimer::Instance().StartTimer(PerformanceTimer::Comm)
#define __stop_timer PerformanceTimer::Instance().StopTimer(PerformanceTimer::Comm)
    /// @brief dumb wrapper
    MPI_int Bcast(void *buf, MPI_int num, MPI_Datatype type, MPI_int source_rank, MPI_Comm comm)
    {
        int ret{0};
        __start_timer;
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            ret = MPI_Bcast(buf, num, type, source_rank, comm);
        else
        {
            MPI_Request req{MPI_REQUEST_NULL};
            ret = MPI_Ibcast(buf, num, type, source_rank, comm, &req);
            ret = MPI::WaitallLazy(1, &req, MPI_STATUSES_IGNORE, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
        }
        __stop_timer;
        return ret;
    }

    MPI_int Alltoall(void *send, MPI_int sendNum, MPI_Datatype typeSend, void *recv, MPI_int recvNum, MPI_Datatype typeRecv, MPI_Comm comm)
    {
        int ret{0};
        __start_timer;
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            ret = MPI_Alltoall(send, sendNum, typeSend, recv, recvNum, typeRecv, comm);
        else
        {
            MPI_Request req{MPI_REQUEST_NULL};
            ret = MPI_Ialltoall(send, sendNum, typeSend, recv, recvNum, typeRecv, comm, &req);
            ret = MPI::WaitallLazy(1, &req, MPI_STATUSES_IGNORE, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
        }
        __stop_timer;
        return ret;
    }

    MPI_int Alltoallv(
        void *send, MPI_int *sendSizes, MPI_int *sendStarts, MPI_Datatype sendType,
        void *recv, MPI_int *recvSizes, MPI_int *recvStarts, MPI_Datatype recvType, MPI_Comm comm)
    {
        int ret{0};
        __start_timer;
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
        __stop_timer;
        return ret;
    }

    MPI_int Allreduce(const void *sendbuf, void *recvbuf, MPI_int count,
                      MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    {
        int ret{0};
        __start_timer;
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            ret = MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        else
        {
            MPI_Request req{MPI_REQUEST_NULL};
            ret = MPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, &req);
            ret = MPI::WaitallLazy(1, &req, MPI_STATUSES_IGNORE, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
        }
        __stop_timer;
        return ret;
    }

    MPI_int Scan(const void *sendbuf, void *recvbuf, MPI_int count,
                 MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    {
        int ret{0}; //todo: add wait lazy?
        __start_timer;
        ret = MPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);
        __stop_timer;
        return ret;
    }

    MPI_int Allgather(const void *sendbuf, MPI_int sendcount, MPI_Datatype sendtype,
                      void *recvbuf, MPI_int recvcount,
                      MPI_Datatype recvtype, MPI_Comm comm)
    {
        int ret{0};
        __start_timer;
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            ret = MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        else
        {
            MPI_Request req{MPI_REQUEST_NULL};
            ret = MPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, &req);
            ret = MPI::WaitallLazy(1, &req, MPI_STATUSES_IGNORE, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
        }
        __stop_timer;
        return ret;
    }

    MPI_int Barrier(MPI_Comm comm)
    {
        int ret{0};
        __start_timer;
        if (MPI::CommStrategy::Instance().GetUseLazyWait() == 0)
            ret = MPI_Barrier(comm);
        else
            ret = MPI::BarrierLazy(comm, static_cast<uint64_t>(MPI::CommStrategy::Instance().GetUseLazyWait()));
        __stop_timer;
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
        MPI_int ret;
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

#undef __start_timer
#undef __stop_timer

}

namespace DNDS::MPI
{
    CommStrategy::CommStrategy()
    {
        try
        {
            auto ret = std::getenv("DNDS_USE_LAZY_WAIT");
            if (ret != NULL && (std::stod(ret) != 0))
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
            auto ret = std::getenv("DNDS_ARRAY_STRATEGY_USE_IN_SITU");
            if (ret != NULL && (std::stoi(ret) != 0))
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
            auto ret = std::getenv("DNDS_USE_STRONG_SYNC_WAIT");
            if (ret != NULL && (std::stoi(ret) != 0))
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
            auto ret = std::getenv("DNDS_USE_ASYNC_ONE_BY_ONE");
            if (ret != NULL && (std::stoi(ret) != 0))
            {
                _use_async_one_by_one = true;
                auto mpi = MPIInfo();
                mpi.setWorld();
                if (mpi.rank == 0)
                    log() << "Detected DNDS_USE_ASYNC_ONE_BY_ONE, setting" << std::endl;
                if (_use_lazy_wait)
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

    bool CommStrategy::GetUseStrongSyncWait()
    {
        return _use_strong_sync_wait;
    }

    bool CommStrategy::GetUseAsyncOneByOne()
    {
        return _use_async_one_by_one;
    }

    double CommStrategy::GetUseLazyWait()
    {
        return _use_lazy_wait;
    }
}

namespace DNDS // TODO: get a concurrency header
{
    std::mutex HDF_mutex;
}

#ifdef NDEBUG_DISABLED
#define NDEBUG
#undef NDEBUG_DISABLED
#endif
