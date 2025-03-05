#pragma once

#include "Defines.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace DNDS
{
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
}