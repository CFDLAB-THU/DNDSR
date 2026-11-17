#pragma once

#include "DNDS/Runtime/Execution/Device.hpp"
#include <atomic>
#include <exception>

namespace DNDS::Runtime
{
    /**
     * @brief Shared backend resource behind one ordered execution lane.
     * @details Contexts and leases retain this object, never a native stream
     * pointer. It does not own an engine or confer submission permission.
     * Engine shutdown first drains accepted work, then publishes retirement.
     * A late lease fence is then a no-op; backend resources remain alive until
     * the last context/operation releases ownership. CPU and device providers
     * implement the same boundary; native handles belong to backend adapters.
     */
    class StreamResource
    {
        std::atomic<bool> _retired{false};
        std::exception_ptr _terminalError;

    protected:
        virtual void SynchronizeBackend() const = 0;

    public:
        virtual ~StreamResource() = default;
        virtual Device GetDevice() const noexcept = 0;
        bool IsRetired() const noexcept { return _retired.load(std::memory_order_acquire); }
        void Synchronize() const
        {
            if (!IsRetired())
                SynchronizeBackend();
            else if (_terminalError)
                std::rethrow_exception(_terminalError);
        }
        /// Provider/controller only: all accepted work and retirement callbacks
        /// must have completed before this publication. This does not wait.
        void RetireAfterDrain(std::exception_ptr error = {}) noexcept
        {
            _terminalError = std::move(error);
            _retired.store(true, std::memory_order_release);
        }
    };
}
