#pragma once

#include "DNDS/Core/OMP.hpp"

#include <chrono>
#include <exception>
#include <functional>
#include <memory>
#include <string>

namespace DNDS::Diagnostics
{
    enum class SetupProgressState
    {
        Starting,
        Done,
        Failed
    };

    struct SetupProgressEvent
    {
        std::string phase, detail;
        SetupProgressState state = SetupProgressState::Starting;
        real seconds = 0;
    };

    /// Explicit, copyable setup observer. Empty instances have zero effect.
    /// Modules report their phase boundaries; the frontend owns formatting.
    class SetupProgress
    {
        using Callback = std::function<void(const SetupProgressEvent &)>;
        std::shared_ptr<const Callback> _callback;
        std::string _prefix;

    public:
        SetupProgress() = default;
        explicit SetupProgress(Callback callback) : _callback(std::make_shared<const Callback>(std::move(callback))) {}

        SetupProgress WithPrefix(std::string prefix) const
        {
            auto result = *this;
            result._prefix += std::move(prefix);
            return result;
        }
        explicit operator bool() const { return bool(_callback); }

        class Scope
        {
            std::shared_ptr<const Callback> _callback;
            std::string _prefix;
            std::string _phase, _detail;
            std::chrono::steady_clock::time_point _started;
            int _exceptions = 0;
            bool _active = false;

        public:
            Scope() = default;
            Scope(std::shared_ptr<const Callback> callback, std::string prefix,
                  std::string phase, std::string detail = {})
                : _callback(std::move(callback)), _prefix(std::move(prefix)),
                  _phase(std::move(phase)), _detail(std::move(detail)),
                  _started(std::chrono::steady_clock::now()), _exceptions(std::uncaught_exceptions()),
                  _active(bool(_callback))
            {
                if (_active)
                    Emit(SetupProgressState::Starting, 0);
            }
            Scope(Scope &&other) noexcept
                : _callback(std::move(other._callback)), _prefix(std::move(other._prefix)),
                  _phase(std::move(other._phase)),
                  _detail(std::move(other._detail)), _started(other._started),
                  _exceptions(other._exceptions), _active(other._active)
            {
                other._active = false;
            }
            Scope(const Scope &) = delete;
            Scope &operator=(const Scope &) = delete;
            Scope &operator=(Scope &&) = delete;
            void Emit(SetupProgressState state, real seconds) const noexcept
            {
                if (!_callback)
                    return;
                try
                {
                    (*_callback)({_prefix + _phase, _detail, state, seconds});
                }
                catch (...)
                {
                }
            }
            ~Scope()
            {
                if (!_active)
                    return;
                const auto seconds = std::chrono::duration<real>(
                                         std::chrono::steady_clock::now() - _started)
                                         .count();
                Emit(std::uncaught_exceptions() > _exceptions
                         ? SetupProgressState::Failed
                         : SetupProgressState::Done,
                     seconds);
            }
        };

        Scope Stage(std::string phase, std::string detail = {}) const
        {
            return Scope(_callback, _prefix, std::move(phase), std::move(detail));
        }

        /// One bounded team formation used only for setup reporting.
        static int ObservedWorkers()
        {
            const int requested = OMP::DistributedThreads();
            int observed = 1;
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel num_threads(requested)
            {
#    pragma omp single
                observed = omp_get_num_threads();
            }
#endif
            return observed;
        }
    };
}
