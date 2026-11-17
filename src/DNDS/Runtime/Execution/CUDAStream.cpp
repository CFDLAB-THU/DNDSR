#include "DNDS/Runtime/Execution/CUDAStream.hpp"

#ifdef DNDS_USE_CUDA
namespace DNDS::Runtime
{
    namespace
    {
        void CheckStream(cudaError_t code) { DNDS_check_throw_info(code == cudaSuccess, cudaGetErrorString(code)); }
        class DeviceScope
        {
            int _previous = -1;

        public:
            explicit DeviceScope(int ordinal)
            {
                CheckStream(cudaGetDevice(&_previous));
                CheckStream(cudaSetDevice(ordinal));
            }
            ~DeviceScope()
            {
                if (cudaSetDevice(_previous) != cudaSuccess)
                    std::terminate();
            }
        };
        void ValidateNativeStream(int ordinal, cudaStream_t stream)
        {
            DeviceScope scope(ordinal);
#    if CUDART_VERSION >= 13000
            int actual = -1;
            CheckStream(cudaStreamGetDevice(stream, &actual));
            DNDS_check_throw_info(actual == ordinal, "CUDA stream belongs to another device");
#    else
            unsigned flags = 0;
            CheckStream(cudaStreamGetFlags(stream, &flags));
#    endif
        }
        class CUDAStreamResource final : public StreamResource
        {
            Device _device;
            cudaStream_t _native;
            bool _owned;
            std::shared_ptr<const void> _owner;

            void SynchronizeBackend() const override
            {
                int previous = -1;
                CheckStream(cudaGetDevice(&previous));
                CheckStream(cudaSetDevice(_device.ordinal));
                const auto result = cudaStreamSynchronize(_native);
                const auto restored = cudaSetDevice(previous);
                CheckStream(result);
                CheckStream(restored);
            }

        public:
            CUDAStreamResource(int ordinal, cudaStream_t stream, bool owned, std::shared_ptr<const void> owner = {})
                : _device(Device::CUDA(ordinal)), _native(stream), _owned(owned), _owner(std::move(owner)) {}
            ~CUDAStreamResource() override
            {
                if (!_owned)
                    return;
                int previous = -1;
                if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(_device.ordinal) != cudaSuccess ||
                    cudaStreamDestroy(_native) != cudaSuccess || cudaSetDevice(previous) != cudaSuccess)
                    std::terminate();
            }
            Device GetDevice() const noexcept override { return _device; }
            cudaStream_t Native() const { return _native; }
        };
    }
    cudaStream_t CUDAStream(const ExecutionContext &context)
    {
        context.Validate();
        DNDS_check_throw_info(context.device.space == MemorySpace::CUDA, "native CUDA stream requires CUDA placement");
        if (!context.stream)
            return nullptr;
        return CUDAStream(*context.stream);
    }
    cudaStream_t CUDAStream(const StreamResource &stream)
    {
        DNDS_check_throw_info(!stream.IsRetired(), "execution stream has retired");
        const auto *resource = dynamic_cast<const CUDAStreamResource *>(&stream);
        DNDS_check_throw_info(resource, "execution stream is not a CUDA resource");
        return resource->Native();
    }
    std::shared_ptr<StreamResource> CreateCUDAStream(int ordinal)
    {
        DeviceScope scope(ordinal);
        cudaStream_t stream = nullptr;
        CheckStream(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try
        {
            return AdoptCUDAStream(ordinal, stream);
        }
        catch (...)
        {
            cudaStreamDestroy(stream);
            throw;
        }
    }
    std::shared_ptr<StreamResource> AdoptCUDAStream(int ordinal, cudaStream_t stream)
    {
        DNDS_check_throw_info(stream && stream != cudaStreamLegacy && stream != cudaStreamPerThread,
                              "adopting a CUDA default stream is invalid");
        ValidateNativeStream(ordinal, stream);
        return std::make_shared<CUDAStreamResource>(ordinal, stream, true);
    }
    std::shared_ptr<StreamResource> ImportCUDAStream(int ordinal, cudaStream_t stream, std::shared_ptr<const void> owner)
    {
        DNDS_check_throw_info(owner, "importing a CUDA stream requires its real external owner");
        ValidateNativeStream(ordinal, stream);
        return std::make_shared<CUDAStreamResource>(ordinal, stream, false, std::move(owner));
    }
    std::shared_ptr<StreamResource> BorrowExternalCUDAStream(int ordinal, cudaStream_t stream)
    {
        ValidateNativeStream(ordinal, stream);
        return std::make_shared<CUDAStreamResource>(ordinal, stream, false);
    }
}
#endif
