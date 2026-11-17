#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"

namespace DNDS::Runtime
{
    /// Explicit logical format conversion. Unlike CopyFrom, this can change the
    /// physical format/pitch without changing row sizes. It allocates the final
    /// destination once, copies only logical values, and synchronizes once at
    /// publication. Intended for setup/interop; repeated numerical operations
    /// use native stride-aware kernels rather than per-element DMA submission.
    template <class T>
    Array<T> ConvertLayout(const Array<T> &source, RowLayout destination,
                           Device device, ExecutionContext context)
    {
        const auto shape = source.Layout();
        DNDS_check_throw_info(destination.RowCount() == shape.RowCount() &&
                                  (destination.IsVariable() || shape.IsVariable() || destination.Width() == shape.Width()),
                              "Array format conversion changes logical shape");
        for (index iRow = 0; iRow < shape.RowCount(); ++iRow)
            DNDS_check_throw_info(shape.RowSize(iRow) == destination.RowSize(iRow), "Array format conversion changes row width");
        const auto inputDevice = source.GetDevice();
        DNDS_check_throw_info((inputDevice != Device::Host() || device != Device::Host() || context.device == Device::Host()) &&
                                  (inputDevice == Device::Host() || inputDevice == context.device) &&
                                  (device == Device::Host() || device == context.device),
                              "Array format conversion context does not cover both placements");
        Array<T> result(destination, device);
        auto read = source.Read({inputDevice, inputDevice == Device::Host() ? nullptr : context.stream});
        auto write = result.Write({device, device == Device::Host() ? nullptr : context.stream});
        const auto input = read.View();
        const auto output = write.View();
        try
        {
            for (index iRow = 0; iRow < shape.RowCount(); ++iRow)
            {
                const auto count = shape.RowSize(iRow);
                const bool contiguous = shape.IsContiguousRows() && destination.IsContiguousRows();
                for (index iComp = 0; iComp < count; iComp += contiguous ? count : 1)
                    // Address selection runs on the host. CSR offset/size
                    // pointers in a CUDA view are device addresses; use the
                    // retained authoritative host layout instead of reading
                    // device metadata through view[iRow].
                    Detail::EnqueueCopyBytes(input.data + shape.ElementOffset(iRow, iComp), inputDevice,
                                             output.data + destination.ElementOffset(iRow, iComp), device,
                                             size_t(contiguous ? count : 1) * sizeof(T), context);
            }
            const auto done = Completion::Record(context);
            if (inputDevice.space == MemorySpace::CUDA)
                read.Complete(done);
            if (device.space == MemorySpace::CUDA)
                write.Complete(done);
            // Host DMA endpoints remain retained until the same copy prefix is
            // observed. This is one boundary, not one event per copied element.
            done.Wait();
            read.Wait();
            write.Wait();
        }
        catch (...)
        {
            Completion::Record(context).Wait();
            write.Fail();
            throw;
        }
        return result;
    }
    template <class T>
    Array<T> ConvertLayout(const Array<T> &source, RowLayout destination)
    {
        const auto device = source.GetDevice();
        return ConvertLayout(source, std::move(destination), device, {device, nullptr});
    }
}
