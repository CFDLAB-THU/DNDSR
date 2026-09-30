#pragma once
/// @file Defines_bind.hpp
/// @brief Shared pybind11 plumbing used by every `*_bind.hpp` in DNDS
/// (buffer-protocol type check, ssp-based class alias, ostream redirect guard,
/// top-level `pybind11_bind_defines` entry point).

#include "Defines.hpp"
#ifdef DNDS_USE_OMP
#    include <omp.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

namespace DNDS
{
    /// Private Python buffer exporter. No back-reference to Array or Python:
    /// slices/NumPy views retain only this allocation and stable buffer metadata.
    struct PythonBufferOwner
    {
        std::shared_ptr<const void> allocation;
        void *ptr = nullptr;
        py::ssize_t itemsize = 0;
        std::string format;
        std::vector<py::ssize_t> shape, strides;
        bool readonly = false;
        alignas(std::max_align_t) unsigned char empty = 0;

        py::buffer_info buffer()
        {
            return {ptr ? ptr : &empty, itemsize, format,
                    static_cast<py::ssize_t>(shape.size()), shape, strides, readonly};
        }
    };

    inline py::memoryview py_owned_buffer(std::shared_ptr<const void> allocation,
                                          void *ptr, py::ssize_t itemsize, std::string format,
                                          py::detail::any_container<py::ssize_t> shape,
                                          py::detail::any_container<py::ssize_t> strides,
                                          bool readonly = false)
    {
        auto owner = std::make_shared<PythonBufferOwner>();
        owner->allocation = std::move(allocation);
        owner->ptr = ptr;
        owner->itemsize = itemsize;
        owner->format = std::move(format);
        owner->shape = std::move(*shape);
        owner->strides = std::move(*strides);
        owner->readonly = readonly;
        return py::memoryview(py::cast(std::move(owner)));
    }

    template <class T>
    py::memoryview py_owned_buffer(std::shared_ptr<const void> allocation, T *ptr,
                                   py::detail::any_container<py::ssize_t> shape,
                                   py::detail::any_container<py::ssize_t> strides,
                                   bool readonly = false)
    {
        return py_owned_buffer(std::move(allocation), const_cast<std::remove_const_t<T> *>(ptr),
                               sizeof(T), py::format_descriptor<std::remove_const_t<T>>::format(),
                               std::move(shape), std::move(strides), readonly || std::is_const_v<T>);
    }

#define DNDS_PYBIND11_OSTREAM_GUARD py::call_guard<py::scoped_ostream_redirect, \
                                                   py::scoped_estream_redirect>()

    template <class T>
    using py_class_ssp = py::classh<T>;

    template <class T>
    bool py_buffer_contains_T(const py::buffer_info &info)
    {
        // return info.format == py::format_descriptor<T>::format(); // this could cause misjudge like long vs long long
        return info.item_type_is_equivalent_to<T>();
    }

    inline bool py_buffer_is_contigious_c(const py::buffer_info &info)
    {
        bool is_contiguous = true;
        ssize_t stride = info.itemsize;
        for (int i = info.ndim - 1; i >= 0; --i)
        {
            if (info.strides[i] != stride)
            {
                is_contiguous = false;
                break;
            }
            stride *= info.shape[i];
        }
        return is_contiguous;
    }

    inline bool py_buffer_is_contigious_f(const py::buffer_info &info)
    {
        bool is_contiguous = true;
        ssize_t stride = info.itemsize;
        for (int i = 0; i < info.ndim; ++i)
        {
            if (info.strides[i] != stride)
            {
                is_contiguous = false;
                break;
            }
            stride *= info.shape[i];
        }
        return is_contiguous;
    }

    inline std::tuple<ssize_t, char> py_buffer_get_contigious_size(const py::buffer_info &info)
    {
        if (info.ndim == 0)
            return {1, 'A'};
        char style = 'N';
        if (py_buffer_is_contigious_c(info))
            style = 'C';
        else if (py_buffer_is_contigious_f(info))
            style = 'F';
        else
            DNDS_assert_info(false, "the data layout is neither C or F contigious");
        return {info.size, style};
    }

    template <typename T>
    py::memoryview py_vector_as_memory_view(std::vector<T> &vec, bool readonly)
    {
        return py::memoryview::from_buffer<T>(
            vec.data(),
            {vec.size()},
            {sizeof(T)},
            true);
    }

    inline void pybind11_bind_defines(py::module_ &m)
    {
        py::class_<PythonBufferOwner, std::shared_ptr<PythonBufferOwner>>(m, "_BufferOwner", py::buffer_protocol())
            .def_buffer(&PythonBufferOwner::buffer);
        m
            .def("_get_UnInitReal", []()
                 { return UnInitReal; })
            .def("_get_UnInitIndex", []()
                 { return UnInitIndex; })
            .def("_get_UnInitRowsize", []()
                 { return UnInitRowsize; });

        m.attr("UnInitReal") = py::float_(UnInitReal);
        m.attr("UnInitIndex") = py::int_(UnInitIndex);
        m.attr("UnInitRowsize") = py::int_(UnInitRowsize);

        m.def("setLogFile", &setLogFile, py::arg("path"),
              "Redirect log() output to a file while duplicating to stdout.");
        m.def("setLogStreamCout", &setLogStreamCout,
              "Restore log() output to stdout only.");

#ifdef DNDS_USE_OMP
        m.def("omp_set_num_threads", [](int n)
              { omp_set_num_threads(n); });
#endif
    }
}
