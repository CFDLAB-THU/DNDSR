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

#ifdef DNDS_USE_OMP
        m.def("omp_set_num_threads", [](int n)
              { omp_set_num_threads(n); });
#endif
    }
}