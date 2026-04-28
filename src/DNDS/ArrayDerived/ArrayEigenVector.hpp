#pragma once
/// @file ArrayEigenVector.hpp
/// @brief Eigen-vector array: each row is an Eigen::Map over contiguous real storage.
/// @par Unit Test Coverage (test_ArrayDerived.cpp, MPI np=1,2,4)
/// - Static size (ArrayEigenVector<5>) and dynamic size (ArrayEigenVector<DynamicSize>)
/// - Resize, Size, RowSize, operator[] returning Eigen::Map
/// - Ghost communication: pull verifying vector element values
/// @par Not Yet Tested
/// - Device views, WriteSerializer / ReadSerializer

#include "../ArrayTransformer.hpp"
#include "ArrayEigenVector_DeviceView.hpp"
#include "DNDS/Device/DeviceStorage.hpp"

namespace DNDS
{
    /**
     * @brief `ParArray<real, N>` whose `operator[]` returns an `Eigen::Map<Vector>`.
     *
     * @details Each row stores an `N`-component vector of `real`. `operator[]`
     * produces an `Eigen::Map` so user code can write natural linear-algebra
     * expressions directly on array storage without copies:
     *
     * ```cpp
     * ArrayEigenVector<3> coords;
     * coords.Resize(nLocal);
     * coords[iNode] << x, y, z;       // write
     * real mag = coords[iNode].norm(); // read
     * ```
     *
     * Used chiefly for node coordinates (`ArrayEigenVectorPair<3>`) and per-cell
     * fluxes.
     *
     * @tparam _vec_size Vector length. `1`, small fixed size, or `DynamicSize`.
     */
    template <rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    class ArrayEigenVector : public ParArray<real, _vec_size, _row_max, _align>
    {
    public:
        using t_self = ArrayEigenVector<_vec_size, _row_max, _align>;
        using t_base = ParArray<real, _vec_size, _row_max, _align>;
        using t_base::t_base;

        template <DeviceBackend B>
        using t_deviceView = ArrayEigenVectorDeviceView<B, real, _vec_size, _row_max, _align>;
        template <DeviceBackend B>
        using t_deviceViewConst = ArrayEigenVectorDeviceView<B, const real, _vec_size, _row_max, _align>;

        /// @brief Owning Eigen vector matching the row shape.
        using t_EigenVector = typename t_deviceView<DeviceBackend::Host>::t_EigenVector;
        /// @brief Mutable Eigen map view onto a row.
        using t_EigenMap = typename t_deviceView<DeviceBackend::Host>::t_EigenMap;
        /// @brief Const Eigen map view onto a row.
        using t_EigenMap_Const = typename t_deviceView<DeviceBackend::Host>::t_EigenMap_Const;

        /// @brief Canonical "snapshot" type produced by value-returning helpers.
        using t_copy = t_EigenVector;

    public:
        // default copy
        ArrayEigenVector(const t_self &R) = default;
        t_self &operator=(const t_self &R) = default;
        // Rule-of-five closure: all value-semantic members.
        ArrayEigenVector(t_self &&) noexcept = default;
        t_self &operator=(t_self &&) noexcept = default;
        ~ArrayEigenVector() = default;
        // operator= handled automatically

        /// @brief Shallow clone (same semantics as assignment).
        void clone(const t_self &R)
        {
            this->operator=(R);
        }

        /// @brief Mutable row-as-Eigen-map accessor.
        t_EigenMap operator[](index i)
        {
            return {t_base::operator[](i), t_base::RowSize(i)}; // need static dispatch?
        }

        /// @brief Const row-as-Eigen-map accessor.
        t_EigenMap_Const operator[](index i) const
        {
            return {t_base::operator[](i), t_base::RowSize(i)};
        }

        using t_base::ReadSerializer;
        using t_base::WriteSerializer; //! because no extra data than Array<>

        /// @brief Mutable device view (Eigen::Map rows on the given backend).
        template <DeviceBackend B>
        auto deviceView()
        {
            return t_deviceView<B>{this->t_base::template deviceView<B>()};
        }

        /// @brief Const device view.
        template <DeviceBackend B>
        [[nodiscard]] auto deviceView() const
        {
            return t_deviceViewConst<B>{this->t_base::template deviceView<B>()};
        }

        /// @brief Element iterator for ArrayEigenVector, yielding Eigen::Map per row.
        template <DeviceBackend B>
        class iterator : public ArrayIteratorBase<iterator<B>>
        {
        public:
            using view_type = t_deviceView<B>;
            using t_base_iter = ArrayIteratorBase<iterator<B>>;
            using typename t_base_iter::difference_type;
            using reference = t_EigenMap;
            using iterator_category = std::random_access_iterator_tag;

        protected:
            view_type view;

        public:
            auto getView() const { return view; }
            DNDS_DEVICE_CALLABLE iterator(const iterator &) = default;
            DNDS_DEVICE_CALLABLE iterator &operator=(const iterator &) = default;
            DNDS_DEVICE_CALLABLE iterator(iterator &&) noexcept = default;
            DNDS_DEVICE_CALLABLE iterator &operator=(iterator &&) noexcept = default;
            DNDS_DEVICE_CALLABLE ~iterator() = default;
            DNDS_DEVICE_CALLABLE explicit iterator(const view_type &n_view, index n_iRow) : view(n_view), t_base_iter(n_iRow)
            {
            }

            DNDS_DEVICE_CALLABLE reference operator*() { return view.operator[](this->iRow); }
        };

        template <DeviceBackend B>
        iterator<B> begin()
        {
            return {deviceView<B>(), 0};
        }

        template <DeviceBackend B>
        iterator<B> end()
        {
            return {deviceView<B>(), this->Size()};
        }
    };
}
