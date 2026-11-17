#include "DNDS/Runtime/Interop/Array_bind.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Runtime/Interop/third_party/dlpack/dlpack.h"
#include <pybind11/stl.h>
#include <array>

namespace DNDS::Runtime
{
    namespace py = pybind11;
    namespace
    {
        DLDataType Dtype(const PythonArray &array)
        {
            const auto kind = py::str(array.Dtype().attr("kind")).cast<std::string>();
            const auto bytes = array.Dtype().itemsize();
            DNDS_check_throw_info(!array.Element() &&
                                      ((kind == "f" && (bytes == 4 || bytes == 8)) ||
                                       ((kind == "i" || kind == "u") && (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8))),
                                  "DLPack requires a supported numeric dtype, not a POD codec");
            return {uint8_t(kind == "f" ? kDLFloat : kind == "i" ? kDLInt
                                                                 : kDLUInt),
                    uint8_t(bytes * 8), 1};
        }

        // The consumer owns this record through DLManagedTensor.deleter. No
        // Python objects are retained; storage includes immutable row metadata.
        struct Export
        {
            ArrayStorage::Shared storage;
            WriteAccess access;
            std::array<int64_t, 3> shape{}, strides{};
            DLManagedTensor tensor{};
            Export(const PythonArray &array, ExecutionContext context, DLDataType dtype)
                : storage(array.SharedStorage()), access(storage->Write(context))
            {
                auto layout = storage->Layout();
                int dimensions = 2;
                shape = {int64_t(layout.RowCount()), int64_t(layout.Width()), 0};
                strides = layout.Kind() == RowLayoutKind::ColMajorTable ? std::array<int64_t, 3>{1, int64_t(layout.Stride()), 0} : std::array<int64_t, 3>{int64_t(layout.Stride()), 1, 0};
                if (layout.Kind() == RowLayoutKind::TiledColMajorTable)
                {
                    dimensions = 3;
                    const auto tile = layout.Stride();
                    shape = {int64_t(layout.RowCount() / tile + (layout.RowCount() % tile != 0)), int64_t(layout.Width()), int64_t(tile)};
                    strides = {int64_t(layout.Width() * tile), int64_t(tile), 1};
                }
                tensor.dl_tensor = {access.View<uint8_t>().data,
                                    {context.device.space == MemorySpace::Host ? kDLCPU : kDLCUDA,
                                     context.device.space == MemorySpace::Host ? 0 : context.device.ordinal},
                                    dimensions,
                                    dtype,
                                    shape.data(),
                                    strides.data(),
                                    0};
                tensor.manager_ctx = this;
                tensor.deleter = [](DLManagedTensor *value)
                { delete static_cast<Export *>(value->manager_ctx); };
            }
        };

        py::capsule ExportArray(const PythonArray &array, py::object stream, py::object copy, py::object dlDevice)
        {
            const auto layout = array.Layout();
            DNDS_check_throw_info(!layout.IsVariable(), "DLPack export requires table layout");
            DNDS_check_throw_info(layout.RowCount() <= INT64_MAX && layout.Width() <= INT64_MAX && layout.Stride() <= INT64_MAX,
                                  "DLPack shape exceeds int64");
            const auto dtype = Dtype(array);
            if (!copy.is_none() && py::cast<bool>(copy))
                throw py::value_error("DLPack adapter only supports zero-copy sharing");
            auto device = array.Storage().GetDevice();
            auto deviceTuple = py::make_tuple(device.space == MemorySpace::Host ? 1 : 2,
                                              device.space == MemorySpace::Host ? 0 : device.ordinal);
            if (!dlDevice.is_none() && !deviceTuple.equal(dlDevice))
                throw py::value_error("DLPack adapter does not move data between devices");
            ExecutionContext context{device, nullptr};
            if (device.space == MemorySpace::Host)
            {
                if (!stream.is_none())
                    throw py::value_error("CPU DLPack stream must be None");
            }
            else if (!stream.is_none())
            {
                const auto native = py::cast<int64_t>(stream);
                if (native <= 0)
                    throw py::value_error("CUDA DLPack stream must be a positive handle (1 legacy, 2 per-thread)");
#ifdef DNDS_USE_CUDA
                context.stream = BorrowExternalCUDAStream(device.ordinal, reinterpret_cast<cudaStream_t>(uintptr_t(native)));
#else
                DNDS_check_throw_info(false, "CUDA DLPack stream is unavailable in this build");
#endif
            }
            auto owner = std::make_unique<Export>(array, context, dtype);
            py::capsule capsule(&owner->tensor, "dltensor", [](PyObject *value)
                                {
                if (PyCapsule_IsValid(value, "dltensor"))
                {
                    auto *tensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(value, "dltensor"));
                    if (tensor->deleter) tensor->deleter(tensor);
                } });
            owner.release();
            return capsule;
        }

        PythonArray ImportArray(py::object producer)
        {
            // Request the legacy capsule, supported by both pre-1.0 and current
            // producers. This ABI has no readonly flag: readonly producers must
            // reject export. Versioned capsules are rejected, never misread.
            auto descriptor = producer.attr("__dlpack_device__")().cast<std::pair<int, int>>();
            DNDS_check_throw_info(descriptor.first == kDLCPU || descriptor.first == kDLCUDA,
                                  "unsupported DLPack device");
            Device device = descriptor.first == kDLCPU ? Device::Host() : Device::CUDA(descriptor.second);
            auto capsuleObject = device.space == MemorySpace::Host ? producer.attr("__dlpack__")() : producer.attr("__dlpack__")(py::arg("stream") = 1);
            DNDS_check_throw_info(PyCapsule_IsValid(capsuleObject.ptr(), "dltensor"), "expected an unconsumed legacy DLPack capsule");
            auto *managed = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsuleObject.ptr(), "dltensor"));
            const auto &tensor = managed->dl_tensor;
            DNDS_check_throw_info(tensor.device.device_type == descriptor.first && tensor.device.device_id == descriptor.second,
                                  "DLPack producer device mismatch");
            DNDS_check_throw_info((tensor.ndim == 1 || tensor.ndim == 2) && tensor.shape, "DLPack import requires 1D or 2D shape");
            const auto type = tensor.dtype;
            DNDS_check_throw_info(type.lanes == 1 &&
                                      ((type.code == kDLFloat && (type.bits == 32 || type.bits == 64)) ||
                                       ((type.code == kDLInt || type.code == kDLUInt) &&
                                        (type.bits == 8 || type.bits == 16 || type.bits == 32 || type.bits == 64))),
                                  "unsupported DLPack numeric dtype");
            for (int i = 0; i < tensor.ndim; ++i)
                DNDS_check_throw_info(tensor.shape[i] >= 0, "negative DLPack shape");
            const size_t rows = tensor.shape[0], width = tensor.ndim == 1 ? 1 : tensor.shape[1];
            if (tensor.strides && rows && width)
            {
                DNDS_check_throw_info((rows <= 1 || tensor.strides[0] == int64_t(width)) &&
                                          (tensor.ndim == 1 || width <= 1 || tensor.strides[1] == 1),
                                      "DLPack import requires contiguous rows; materialize the producer first");
            }
            auto layout = RowLayout::Dense(rows, width);
            const size_t bytes = CheckedSize::Multiply(layout.Elements(), size_t(type.bits / 8));
            const auto address = reinterpret_cast<uintptr_t>(tensor.data);
            DNDS_check_throw_info(tensor.byte_offset <= UINTPTR_MAX - address && (!bytes || tensor.data), "invalid DLPack data range");
            auto *pointer = reinterpret_cast<void *>(address + tensor.byte_offset);
            DNDS_check_throw_info(reinterpret_cast<uintptr_t>(pointer) % (type.bits / 8) == 0, "unaligned DLPack data");
            // Import publishes ready storage. Producer establishes legacy-stream
            // readiness; explicitly observe it before granting immediate access.
            {
                py::gil_scoped_release release;
                Completion::Record({device, nullptr}).Wait();
            }
            if (PyCapsule_SetName(capsuleObject.ptr(), "used_dltensor") != 0)
                throw py::error_already_set();
            auto owner = std::shared_ptr<DLManagedTensor>(managed, [](DLManagedTensor *value)
                                                          {
                py::gil_scoped_acquire gil;
                if (value->deleter) value->deleter(value); });
            auto allocation = Allocation::Import(std::shared_ptr<void>(owner, pointer), bytes, device);
            auto buffer = Buffer::FromAllocation(allocation, 0, bytes);
            auto dtype = py::dtype(std::string(type.code == kDLFloat ? "float" : type.code == kDLInt ? "int"
                                                                                                     : "uint") +
                                   std::to_string(type.bits));
            return PythonArray(ArrayStorage::FromBuffer(std::move(layout), std::move(buffer), type.bits / 8), dtype);
        }
    }

    void BindDLPackPython(py::module_ &m)
    {
        auto array = py::reinterpret_borrow<py::class_<PythonArray>>(m.attr("Array"));
        array.def("__dlpack_device__", [](const PythonArray &a)
                  { auto d = a.Storage().GetDevice(); return py::make_tuple(d.space == MemorySpace::Host ? 1 : 2, d.space == MemorySpace::Host ? 0 : d.ordinal); })
            .def("__dlpack__", [](const PythonArray &a, py::object stream, py::object, py::object device, py::object copy)
                 { return ExportArray(a, std::move(stream), std::move(copy), std::move(device)); }, py::arg("stream") = py::none(), py::kw_only(), py::arg("max_version") = py::none(), py::arg("dl_device") = py::none(), py::arg("copy") = py::none(), "Zero-copy numeric table export: logical2D for row/column tables; physical3D (tiles,components,tile) including tails for tiled storage. Logical row count remains array.layout.row_count. Holds an exclusive write lease until the consumer's last alias dies; CUDA release fences the passed stream. Finish DNDS producers first; join all other foreign streams into this stream and keep it alive through release. No autograd contract.");
        m.def("from_dlpack", &ImportArray,
              "Import a writable contiguous 1D/2D numeric producer with shared ownership. CUDA import observes producer readiness on the legacy stream. Import each foreign allocation once and derive DNDS aliases; external owners must obey DNDS leases and explicitly synchronize future foreign work. The producer descriptor is trusted for capacity; no alias discovery or autograd is provided.");
    }
}
