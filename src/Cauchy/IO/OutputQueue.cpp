#include "Cauchy/IO/OutputMeasurement.hpp"
#include "Cauchy/IO/OutputQueue.hpp"
#include "DNDS/Core/Errors.hpp"
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace DNDS::Cauchy
{
    struct OutputQueue::Data
    {
        Failure terminal;
        unsigned capacity;
        std::thread::id controller = std::this_thread::get_id();
        std::mutex mutex;
        std::condition_variable changed;
        std::deque<Action> pending;
        std::exception_ptr failure;
        bool running = false, closing = false, preparing = false;
        std::thread worker;
        Data(Failure handler, unsigned limit) : terminal(std::move(handler)), capacity(limit)
        {
            DNDS_check_throw_info(terminal && capacity > 0, "output queue needs a terminal failure handler and positive capacity");
            worker = std::thread([this]
                                 { Run(); });
        }
        void CheckController() const
        {
            DNDS_check_throw_info(std::this_thread::get_id() == controller, "output queue control belongs to its creating thread");
        }
        void Run() noexcept
        {
            for (;;)
            {
                Action action;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&]
                                 { return closing || !pending.empty(); });
                    if (pending.empty())
                        return;
                    action = std::move(pending.front());
                    pending.pop_front();
                    running = true;
                }
                try
                {
                    action();
                }
                catch (...)
                {
                    const auto caught = std::current_exception();
                    // No collective agreement/recovery is attempted here.
                    // The production handler aborts; a coordinated UT handler
                    // may return solely to inspect the retained failure.
                    terminal(caught);
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        failure = caught;
                        pending.clear();
                        running = false;
                        closing = true;
                    }
                    changed.notify_all();
                    return;
                }
                action = {}; // Drop snapshot allocations before releasing capacity.
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    running = false;
                }
                changed.notify_all();
            }
        }
    };
    OutputQueue::OutputQueue(Failure terminalFailure, unsigned capacity)
        : _data(std::make_unique<Data>(std::move(terminalFailure), capacity)) {}
    OutputQueue::~OutputQueue()
    {
        try
        {
            Close();
        }
        catch (...)
        {
            std::terminate();
        }
    }
    void OutputQueue::Enqueue(Prepare prepare)
    {
        auto &data = *_data;
        data.CheckController();
        DNDS_check_throw_info(!data.preparing, "output snapshot preparation cannot reenter its queue");
        DNDS_check_throw_info(prepare, "output queue has no snapshot producer");
        {
            std::unique_lock<std::mutex> lock(data.mutex);
            if (!data.closing && data.pending.size() + unsigned(data.running) >= data.capacity)
            {
                const OutputMeasurement wait("queue_full_wait");
                data.changed.wait(lock, [&]
                                  { return data.closing || data.pending.size() + unsigned(data.running) < data.capacity; });
            }
            if (data.failure)
                std::rethrow_exception(data.failure);
            DNDS_check_throw_info(!data.closing, "output queue is closed");
        }
        struct PreparationScope
        {
            bool &active;
            explicit PreparationScope(bool &flag) : active(flag) { active = true; }
            ~PreparationScope() { active = false; }
        } preparation(data.preparing);
        Action action;
        {
            const OutputMeasurement capture("capture");
            action = prepare();
        }
        DNDS_check_throw_info(action, "output snapshot producer returned no writer");
        {
            std::lock_guard<std::mutex> lock(data.mutex);
            if (data.failure)
                std::rethrow_exception(data.failure);
            DNDS_check_throw_info(!data.closing, "output queue closed during preparation");
            data.pending.push_back(std::move(action));
        }
        data.changed.notify_all();
    }
    void OutputQueue::Drain()
    {
        auto &data = *_data;
        data.CheckController();
        DNDS_check_throw_info(!data.preparing, "output snapshot preparation cannot drain its queue");
        const OutputMeasurement drain("drain");
        std::unique_lock<std::mutex> lock(data.mutex);
        data.changed.wait(lock, [&]
                          { return !data.running && data.pending.empty(); });
        if (data.failure)
            std::rethrow_exception(data.failure);
    }
    void OutputQueue::Close()
    {
        auto &data = *_data;
        data.CheckController();
        DNDS_check_throw_info(!data.preparing, "output snapshot preparation cannot close its queue");
        {
            std::lock_guard<std::mutex> lock(data.mutex);
            data.closing = true;
        }
        data.changed.notify_all();
        if (data.worker.joinable())
        {
            const OutputMeasurement drain("shutdown_drain");
            data.worker.join();
        }
        // The explicit observation reports failures. A second Close (including
        // destruction after a UT caught it) only joins and releases resources.
    }
}
