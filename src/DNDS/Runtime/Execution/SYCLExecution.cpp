#include "DNDS/Runtime/Execution/SYCLQueue.hpp"
#include "DNDS/Runtime/Storage/AllocationResource.hpp"
#include <map>
#include <mutex>
#include <utility>

namespace DNDS::Runtime
{
    namespace
    {
        // Device indices identify a process-local catalog. Contexts and queues
        // are created lazily only for an explicitly selected USM device.
        const auto &Catalog()
        {
            static const auto devices = sycl::device::get_devices();
            return devices;
        }

        sycl::queue NewQueue(const sycl::context &context, const sycl::device &device)
        {
            return {context, device, [](sycl::exception_list errors)
                    { for (const auto &error : errors) std::rethrow_exception(error); },
                    sycl::property::queue::in_order{}};
        }

        sycl::queue DefaultQueue(Device device)
        {
            DNDS_check_throw_info(device.space == MemorySpace::SYCL && device.ordinal >= 0 &&
                                      size_t(device.ordinal) < Catalog().size(),
                                  "unknown SYCL device");
            static std::mutex mutex;
            static std::map<int, sycl::queue> queues;
            std::lock_guard<std::mutex> lock(mutex);
            auto found = queues.find(device.ordinal);
            if (found == queues.end())
            {
                const auto &selected = Catalog()[device.ordinal];
                DNDS_check_throw_info(selected.has(sycl::aspect::usm_device_allocations), "SYCL device lacks device USM");
                found = queues.emplace(device.ordinal, NewQueue(sycl::context(selected), selected)).first;
            }
            return found->second;
        }

        class QueueResource final : public StreamResource
        {
            Device _device;
            sycl::queue _queue;
            void SynchronizeBackend() const override
            {
                auto queue = _queue;
                queue.wait_and_throw();
            }

        public:
            QueueResource(Device device, sycl::queue queue) : _device(device), _queue(std::move(queue)) {}
            Device GetDevice() const noexcept override { return _device; }
            sycl::queue Queue() const { return _queue; }
        };

        class USMResource final : public AllocationResource
        {
            Device _device;
            sycl::queue _queue;
            size_t _bytes;
            void *_base = nullptr;

        public:
            USMResource(size_t bytes, size_t alignment, Device device)
                : _device(device), _queue(DefaultQueue(device)), _bytes(bytes)
            {
                if (!bytes)
                    return;
                _base = sycl::aligned_alloc_device(alignment, bytes, _queue);
                DNDS_check_throw_info(_base, "SYCL device USM allocation failed");
                try
                {
                    _queue.memset(_base, 0, bytes).wait_and_throw();
                }
                catch (...)
                {
                    Release(nullptr);
                    throw;
                }
            }
            ~USMResource() override { Release(nullptr); }
            void *Data() const noexcept override { return _base; }
            size_t Bytes() const noexcept override { return _bytes; }
            Device GetDevice() const noexcept override { return _device; }
            bool ExternalProvider() const noexcept override { return false; }
            void Release(const ExecutionContext *) noexcept override
            {
                if (auto *p = std::exchange(_base, nullptr))
                    sycl::free(p, _queue.get_context());
            }
        };
    }
    Device SelectSYCLDevice(SYCLTarget target, int ordinal)
    {
        DNDS_check_throw_info(ordinal >= 0, "negative SYCL device ordinal");
        const auto &devices = Catalog();
        for (size_t i = 0; i < devices.size(); ++i)
            if ((target == SYCLTarget::CPU ? devices[i].is_cpu() : devices[i].get_backend() == sycl::backend::ext_oneapi_cuda) &&
                devices[i].has(sycl::aspect::usm_device_allocations) && ordinal-- == 0)
                return Device::SYCL(int(i));
        DNDS_check_throw_info(false, "requested SYCL backend/device with device USM is unavailable");
        return Device::Unspecified();
    }
    ExecutionContext CreateSYCLQueue(Device device)
    {
        auto parent = DefaultQueue(device);
        return {device, std::make_shared<QueueResource>(device, NewQueue(parent.get_context(), parent.get_device()))};
    }
    sycl::queue SYCLQueue(const ExecutionContext &context)
    {
        context.Validate();
        if (!context.stream)
            return DefaultQueue(context.device);
        auto lane = std::dynamic_pointer_cast<QueueResource>(context.stream);
        DNDS_check_throw_info(lane && !lane->IsRetired(), "invalid or retired SYCL queue");
        return lane->Queue();
    }
    namespace SYCLDetail
    {
        class Event
        {
        public:
            sycl::event prefix;
            explicit Event(sycl::event e) : prefix(std::move(e)) {}
        };
        std::shared_ptr<Event> Record(const ExecutionContext &context)
        {
            return std::make_shared<Event>(SYCLQueue(context).ext_oneapi_submit_barrier());
        }
        void Wait(const Event &event)
        {
            auto prefix = event.prefix;
            prefix.wait_and_throw();
        }
        bool Ready(const Event &event)
        {
            if (event.prefix.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete)
                return false;
            Wait(event);
            return true;
        }
        void Synchronize(const ExecutionContext &context) { SYCLQueue(context).wait_and_throw(); }
        std::unique_ptr<AllocationResource> Allocate(size_t bytes, size_t alignment, Device device)
        {
            return std::make_unique<USMResource>(bytes, alignment, device);
        }
        void Copy(void *destination, const void *source, size_t bytes, const ExecutionContext &context)
        {
            if (bytes)
                SYCLQueue(context).memcpy(destination, source, bytes);
        }
    }
}
