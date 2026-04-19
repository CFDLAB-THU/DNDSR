#pragma once
/// @file DeviceView.hpp
/// @brief Non-owning device-side views of Array objects for host and CUDA backends.

#include "ArrayBasic.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"

namespace DNDS
{
    // template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    // class ArrayDeviceViewBase : public ArrayView<T, _row_size, _row_max, _align>
    // {
    // public:
    //     virtual DeviceBackend backend() = 0;
    //     virtual ~ArrayDeviceViewBase() = default;
    // };

    /**
     * @brief Non-owning device-callable view of an #Array, specialised per #DeviceBackend.
     *
     * @details The primary template is deleted; concrete specialisations are
     * provided for each supported backend (Host always, CUDA when
     * #DNDS_USE_CUDA is on). A device view captures host and/or device raw
     * pointers; on the CUDA specialisation the element accessors are marked
     * `__device__` so kernels can use them directly.
     *
     * Produced by `Array::deviceView<B>()`.
     *
     * @tparam B         Device backend.
     * @tparam T         Element type (may be `const T` for read-only views).
     * @tparam _row_size Mirrors #Array.
     * @tparam _row_max  Mirrors #Array.
     * @tparam _align    Mirrors #Array.
     */
    template <DeviceBackend B,
              class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayDeviceView : public ArrayView<T, _row_size, _row_max, _align>
    {
    public:
        ArrayDeviceView() = delete;
    };

    /// @brief Host specialisation of #ArrayDeviceView; identical to the host-side #ArrayView.
    template <class T, rowsize _row_size, rowsize _row_max, rowsize _align>
    class ArrayDeviceView<
        DeviceBackend::Host, T, _row_size, _row_max, _align>
        : public ArrayView<T, _row_size, _row_max, _align>
    {
    public:
        using t_base = ArrayView<T, _row_size, _row_max, _align>;
        using t_base::t_base;
        using typename t_base::RowView;

        static DeviceBackend backend() { return DeviceBackend::Host; }
        // everything uses inherited
    };

#ifdef DNDS_USE_CUDA
    /// @brief CUDA specialisation of #ArrayDeviceView with `__device__`
    /// element accessors suitable for use inside kernels.
    template <class T, rowsize _row_size, rowsize _row_max, rowsize _align>
    class ArrayDeviceView<
        DeviceBackend::CUDA, T, _row_size, _row_max, _align>
        : public ArrayView<T, _row_size, _row_max, _align>
    {
    public:
        using t_base = ArrayView<T, _row_size, _row_max, _align>;
        using t_base::t_base;
        using typename t_base::RowView;

        using self_type = ArrayDeviceView<
            DeviceBackend::CUDA, T, _row_size, _row_max, _align>;

        static DeviceBackend backend() { return DeviceBackend::CUDA; }

        /// @brief Device-side mutable element access. Uses the compressed-row path.
        DNDS_DEVICE_CALLABLE T &operator()(index iRow, rowsize iCol = 0)
        {
            return const_cast<T &>(this->at_compressed(iRow, iCol));
        }

        /// @brief Device-side const element access.
        DNDS_DEVICE_CALLABLE const T &operator()(index iRow, rowsize iCol = 0) const
        {
            return this->at_compressed(iRow, iCol);
        }

        /// @brief Device-side row pointer.
        DNDS_DEVICE_CALLABLE T *operator[](index iRow)
        {
            return this->get_rowstart_pointer_compressed(iRow);
        }

        DNDS_DEVICE_CALLABLE const T *operator[](index iRow) const
        {
            return static_cast<const T *>(const_cast<self_type *>(this)->operator[](iRow));
        }
    };
#endif

    /// @brief Factory for #ArrayDeviceView: selects host or device pointers
    /// based on `B` and forwards everything else.
    /// @details Used by `Array::deviceView<B>()`; rarely called directly.
    template <DeviceBackend B, class T, rowsize _row_size, rowsize _row_max, rowsize _align>
    auto ArrayDeviceView_build(
        index n_size, T *n_data, index n_data_size,
        const index *n_rowstart, index n_rowstart_size,
        const rowsize *n_rowsizes, index n_rowsizes_size,
        rowsize n_row_size_dynamic,
        T *n_data_device,
        const index *n_rowstart_device,
        const rowsize *n_rowsizes_device)
    {
        DNDS_assert(B != DeviceBackend::Unknown);
        DNDS_assert_info(B != DeviceBackend::Custom1, "unimplemented");
        if constexpr (B == DeviceBackend::Host)
            return ArrayDeviceView<DeviceBackend::Host, T, _row_size, _row_max, _align>(
                n_size, n_data, n_data_size,
                n_rowstart, n_rowstart_size,
                n_rowsizes, n_rowsizes_size,
                n_row_size_dynamic,
                true, nullptr);
#ifdef DNDS_USE_CUDA
        else if constexpr (B == DeviceBackend::CUDA)
            return ArrayDeviceView<DeviceBackend::CUDA, T, _row_size, _row_max, _align>(
                n_size, n_data_device, n_data_size,
                n_rowstart_device, n_rowstart_size,
                n_rowsizes_device, n_rowsizes_size,
                n_row_size_dynamic,
                true, nullptr);
#endif
        else
            return ArrayDeviceView<B, T, _row_size, _row_max, _align>(
                n_size, n_data, n_data_size,
                n_rowstart, n_rowstart_size,
                n_rowsizes, n_rowsizes_size,
                n_row_size_dynamic,
                true, nullptr);
    };
}