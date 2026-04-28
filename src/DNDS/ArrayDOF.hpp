#pragma once
/// @file ArrayDOF.hpp
/// @brief Degree-of-freedom array with vector-space operations (MPI-collective).
/// @par Unit Test Coverage (test_ArrayDOF.cpp, MPI np=1,2,4)
/// - setConstant (scalar and Eigen::Matrix)
/// - operator+= (scalar, ArrayDof, Eigen::Matrix), operator-=, operator*= (scalar,
///   ArrayDof element-wise, Eigen::Matrix), operator/=
/// - addTo(other, scale)
/// - norm2, norm2(other) (L2 distance), dot (MPI-collective)
/// - min, max, sum (MPI-collective reductions)
/// - componentWiseNorm1, componentWiseNorm1(other)
/// - operator= (value copy), clone (deep copy)
/// - Scalar-array multiply: ArrayDof<N,1> *= ArrayDof<1,1>
/// - Mathematical identity: dot(x,x) == norm2(x)^2
/// @par Not Yet Tested
/// - Multi-column DOFs (n_n > 1, i.e. matrix-valued DOFs)
/// - Ghost communication through ArrayDof (all tests use son->Resize(0))
/// - ArrayDofOp<DeviceBackend::CUDA>
/// - ArrayDofSinglePack::BuildResizeFatherSon

#include "ArrayDerived/ArrayEigenMatrix.hpp"

#include "ArrayDerived/ArrayEigenVector.hpp"

#include "ArrayPair.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/Device/DeviceStorage.hpp"

namespace DNDS
{
    /// @brief Mutable device view of an ArrayDof father/son pair.
    template <DeviceBackend B, int n_m, int n_n>
    class ArrayDofDeviceView : public ArrayPairDeviceView<B, ArrayEigenMatrix<n_m, n_n>>
    {
    public:
        using t_base = ArrayPairDeviceView<B, ArrayEigenMatrix<n_m, n_n>>;
        using t_base::t_base;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ArrayDofDeviceView, ArrayDofDeviceView)

        DNDS_DEVICE_CALLABLE ArrayDofDeviceView(const t_base &base_view) : t_base(base_view) {}
        DNDS_DEVICE_CALLABLE ArrayDofDeviceView(t_base &&base_view) : t_base(base_view) {}
    };

    /// @brief Const device view of an ArrayDof father/son pair.
    template <DeviceBackend B, int n_m, int n_n>
    class ArrayDofDeviceViewConst : public ArrayPairDeviceViewConst<B, ArrayEigenMatrix<n_m, n_n>>
    {
    public:
        using t_base = ArrayPairDeviceViewConst<B, ArrayEigenMatrix<n_m, n_n>>;
        using t_base::t_base;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ArrayDofDeviceViewConst, ArrayDofDeviceViewConst)

        DNDS_DEVICE_CALLABLE ArrayDofDeviceViewConst(const t_base &base_view) : t_base(base_view) {}
        DNDS_DEVICE_CALLABLE ArrayDofDeviceViewConst(t_base &&base_view) : t_base(base_view) {}
    };

    template <int n_m, int n_n>
    class ArrayDof;

    template <DeviceBackend B, int n_m, int n_n>
    class ArrayDofOp;

#define DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n)
// NOLINTBEGIN(bugprone-macro-parentheses)
// Rationale: `spec` is a C++ storage-class specifier token (e.g. `static`)
// placed at the start of a function declaration; it cannot be parenthesized.
#define DNDS_ARRAY_DOF_OP_FUNC_LIST(B, n_m, n_n, spec)                                                                                            \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) setConstant(t_self &self, real R);                                                   \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) setConstant(t_self &self, const Eigen::Ref<const t_element_mat> &R);                 \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_plus_assign(t_self &self, const t_self &R);                                 \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_plus_assign(t_self &self, real R);                                          \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_plus_assign(t_self &self, const Eigen::Ref<const t_element_mat> &R);        \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_minus_assign(t_self &self, const t_self &R);                                \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_mult_assign(t_self &self, real R);                                          \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_mult_assign_scalar_arr(t_self &self, const ArrayDof<1, 1> &R);              \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_mult_assign(t_self &self, const Eigen::Ref<const t_element_mat> &R);        \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_mult_assign(t_self &self, const t_self &R);                                 \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_div_assign(t_self &self, const t_self &R);                                  \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_assign(t_self &self, const t_self &R);                                      \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) addTo(t_self &self, const t_self &R, real r);                                        \
    spec real DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) norm2(t_self &self);                                                                 \
    spec real DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) norm2(t_self &self, const t_self &R);                                                \
    spec real DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) reduction(t_self &self, const std::string &op);                                      \
    spec ArrayDofOp<B, n_m, n_n>::t_element_mat DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) componentWiseNorm1(t_self &self);                  \
    spec ArrayDofOp<B, n_m, n_n>::t_element_mat DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) componentWiseNorm1(t_self &self, const t_self &R); \
    spec real DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) dot(t_self &self, const t_self &R);
    // NOLINTEND(bugprone-macro-parentheses)

    /**
     * @brief Host-side static dispatcher: implements every vector-space operation
     * declared in @ref DNDS_ARRAY_DOF_OP_FUNC_LIST for CPU execution.
     *
     * @details Each member is a `static` free-function-style routine that takes
     * the target @ref DNDS::ArrayDof "ArrayDof" by reference. Explicit instantiations for the
     * supported `(n_m, n_n)` combinations live in `ArrayDOF_inst/<...>.cpp`.
     * The host version uses OpenMP where profitable; the CUDA specialisation
     * (below) is a parallel mirror that dispatches to kernels.
     */
    template <int n_m, int n_n>
    class ArrayDofOp<DeviceBackend::Host, n_m, n_n>
    {
    public:
        using t_self = ArrayDof<n_m, n_n>;
        using t_element_mat = Eigen::Matrix<real, RowSize_To_EigenSize(n_m), RowSize_To_EigenSize(n_n)>;
        DNDS_ARRAY_DOF_OP_FUNC_LIST(DeviceBackend::Host, n_m, n_n, static)
    };

#ifdef DNDS_USE_CUDA
    /**
     * @brief CUDA-side static dispatcher. Same interface as the host version;
     * implemented in `ArrayDOF_inst/<...>.cu`.
     */
    template <int n_m, int n_n>
    class ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>
    {
    public:
        using t_self = ArrayDof<n_m, n_n>;
        using t_element_mat = Eigen::Matrix<real, RowSize_To_EigenSize(n_m), RowSize_To_EigenSize(n_n)>;
        DNDS_ARRAY_DOF_OP_FUNC_LIST(DeviceBackend::CUDA, n_m, n_n, static)
    };
#endif

#ifdef DNDS_USE_CUDA
#    define DNDS_ARRAY_OP_SWITCH_CUDA_CASE(expr)   \
    case DeviceBackend::CUDA:                      \
    {                                              \
        return (t_ops<DeviceBackend::CUDA>::expr); \
    }
#else
#    define DNDS_ARRAY_OP_SWITCH_CUDA_CASE(expr)
#endif

#define DNDS_ARRAY_OP_SWITCHER(Backend, expr)      \
    switch (Backend)                               \
    {                                              \
    case DeviceBackend::Host:                      \
    case DeviceBackend::Unknown:                   \
    {                                              \
        return (t_ops<DeviceBackend::Host>::expr); \
    }                                              \
        DNDS_ARRAY_OP_SWITCH_CUDA_CASE(expr)       \
    default:                                       \
        DNDS_assert(false);                        \
        return (t_ops<DeviceBackend::Host>::expr); \
    }

    /**
     * @brief Primary solver state container: an @ref DNDS::ArrayEigenMatrix "ArrayEigenMatrix" pair with
     * MPI-collective vector-space operations.
     *
     * @details `ArrayDof<n_m, n_n>` inherits everything from
     * @ref ArrayEigenMatrixPair<n_m, n_n> (father, son, transformer, typed row
     * access as `Eigen::Map<Matrix<real, n_m, n_n>>`) and adds:
     *  - entry-wise updates: `+= real`, `+= ArrayDof`, `*= real`, `*= matrix`, etc.;
     *  - AXPY: #addTo;
     *  - **MPI-collective** reductions: #norm2, #dot, #min, #max, #sum, #componentWiseNorm1.
     *
     * Host and CUDA backends are both supported -- the methods dispatch based
     * on `father->device()` at the call site.
     *
     * CFV convenience aliases (in `CFV/VRDefines.hpp`):
     *  - `tUDof<N>   = ArrayDof<N, 1>`              (per-cell state vector).
     *  - `tURec<N>   = ArrayDof<DynamicSize, N>`    (reconstruction coefficients).
     *  - `tUGrad<N,d>= ArrayDof<d, N>`              (gradients).
     *
     * @tparam n_m  Row count per cell (1 for state vectors).
     * @tparam n_n  Column count per cell.
     *
     * @sa ArrayEigenMatrixPair, docs/architecture/array_infrastructure.md,
     * docs/guides/array_usage.md.
     */
    template <int n_m, int n_n>
    class ArrayDof : public ArrayEigenMatrixPair<n_m, n_n>
    {
    public:
        using t_base = ArrayEigenMatrixPair<n_m, n_n>;
        using t_base::t_base;

        /// @brief Mutable device view alias.
        template <DeviceBackend B>
        using t_deviceView = ArrayDofDeviceView<B, n_m, n_n>;
        /// @brief Const device view alias.
        template <DeviceBackend B>
        using t_deviceViewConst = ArrayDofDeviceViewConst<B, n_m, n_n>;

        /// @brief Build a mutable device view (wraps the base-class implementation).
        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            return {t_base::template deviceView<B>()};
        }

        /// @brief Build a const device view.
        template <DeviceBackend B>
        t_deviceViewConst<B> deviceView() const
        {
            return {t_base::template deviceView<B>()};
        }

        using t_base::to_device;
        using t_base::to_host;

        using t_self = ArrayDof<n_m, n_n>;

        /// @brief Static dispatcher alias selecting host / CUDA implementation.
        template <DeviceBackend B>
        using t_ops = ArrayDofOp<B, n_m, n_n>;

        /// @brief Shape of one DOF row as an Eigen matrix.
        using t_element_mat = Eigen::Matrix<real, RowSize_To_EigenSize(n_m), RowSize_To_EigenSize(n_n)>;

        /// @brief Deep copy from another ArrayDof. Delegates to the base-class clone.
        void clone(const t_self &R)
        {
            // ! no using operator= here
            this->t_base::clone(R);
            // no extra data
        }

        /// @brief Set every entry of every (father+son) row to the scalar `R`.
        void setConstant(real R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), setConstant(*this, R));
        }

        /// @brief Set every row to the matrix `R` (must have shape `n_m x n_n`).
        void setConstant(const Eigen::Ref<const t_element_mat> &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), setConstant(*this, R));
        }

        /// @brief In-place element-wise add: `this += R`.
        void operator+=(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_plus_assign(*this, R));
        }

        /// @brief Add the scalar `R` to every entry.
        void operator+=(real R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_plus_assign(*this, R));
        }

        /// @brief Add a per-row matrix `R` (same to every row).
        void operator+=(const Eigen::Ref<const t_element_mat> &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_plus_assign(*this, R));
        }

        /// @brief In-place element-wise subtract: `this -= R`.
        void operator-=(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_minus_assign(*this, R));
        }

        /// @brief Scalar multiply in place.
        void operator*=(real R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_mult_assign(*this, R));
        }

        /// @brief Scale each row by a corresponding scalar stored in `R` (a `1x1` ArrayDof).
        /// @details Typical use: multiply state DOFs by per-cell values such as
        /// inverse mass. Only enabled for non-scalar DOF shapes.
        template <int n_m_T = n_m>
        std::enable_if_t<!(n_m_T == 1 && n_n == 1)>
        operator*=(const ArrayDof<1, 1> &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_mult_assign_scalar_arr(*this, R));
        }

        /// @brief In-place multiplication by a small fixed matrix (same applied to every row).
        void operator*=(const Eigen::Ref<const t_element_mat> &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_mult_assign(*this, R));
        }

        /// @brief Element-wise multiply: `this *= R` (Hadamard).
        void operator*=(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_mult_assign(*this, R));
        }

        /// @brief Element-wise divide: `this /= R`.
        void operator/=(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_div_assign(*this, R));
        }

        /// @brief Value-copy assignment from another ArrayDof of identical layout.
        void operator=(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_assign(*this, R));
        }

        /// @brief Rule-of-five closure. The class's custom `operator=(const t_self&)`
        /// suppresses the implicit move operations; re-enable the compiler-
        /// synthesised ones (shallow move of `shared_ptr` members in the base).
        ArrayDof(const ArrayDof &) = default;
        ArrayDof(ArrayDof &&) noexcept = default;
        ArrayDof &operator=(ArrayDof &&) noexcept = default;
        ~ArrayDof() = default;

        /// @brief AXPY: `this += r * R`. One of the hot-path solver primitives.
        void addTo(const t_self &R, real r)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), addTo(*this, R, r));
        }

        /// @brief Global L2 norm (MPI-collective). `sqrt(sum_i sum_j x_ij^2)`.
        real norm2()
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), norm2(*this));
        }

        /// @brief Global L2 distance between `this` and `R` (collective).
        real norm2(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), norm2(*this, R));
        }

        /// @brief Global minimum across all entries (collective).
        real min()
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), reduction(*this, "min"));
        }

        /// @brief Global maximum across all entries (collective).
        real max()
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), reduction(*this, "max"));
        }

        /// @brief Global sum of all entries (collective).
        real sum()
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), reduction(*this, "sum"));
        }

        /// @brief Per-component global L1 norm, returned as an `n_m x n_n` matrix (collective).
        t_element_mat componentWiseNorm1()
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), componentWiseNorm1(*this));
        }

        /// @brief Per-component global L1 distance between `this` and `R` (collective).
        t_element_mat componentWiseNorm1(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), componentWiseNorm1(*this, R));
        }

        /// @brief Global inner product: `sum_i sum_j x_ij * R_ij` (collective).
        real dot(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), dot(*this, R));
        }
    };
}
//! the host side operators are provided as implemented
#include "ArrayDOF_op.hxx"

namespace DNDS
{
#undef DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE
#define DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) ArrayDofOp<B, n_m, n_n>::

#define DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(B, offset, exttmp)             \
    DNDS_ARRAY_DOF_OP_FUNC_LIST(B, 1, 1 + (offset), exttmp);           \
    DNDS_ARRAY_DOF_OP_FUNC_LIST(B, 2, 1 + (offset), exttmp);           \
    DNDS_ARRAY_DOF_OP_FUNC_LIST(B, 3, 1 + (offset), exttmp);           \
    DNDS_ARRAY_DOF_OP_FUNC_LIST(B, 4, 1 + (offset), exttmp);           \
    DNDS_ARRAY_DOF_OP_FUNC_LIST(B, 5, 1 + (offset), exttmp);           \
    DNDS_ARRAY_DOF_OP_FUNC_LIST(B, 6, 1 + (offset), exttmp);           \
    DNDS_ARRAY_DOF_OP_FUNC_LIST(B, 7, 1 + (offset), exttmp);           \
    DNDS_ARRAY_DOF_OP_FUNC_LIST(B, 8, 1 + (offset), exttmp);           \
    DNDS_ARRAY_DOF_OP_FUNC_LIST(B, DynamicSize, 1 + (offset), exttmp); \
    DNDS_ARRAY_DOF_OP_FUNC_LIST(B, NonUniformSize, 1 + (offset), exttmp);

    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, 0, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, 1, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, 2, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, 3, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, 4, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, 5, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, 6, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, 7, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, DynamicSize - 1, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, NonUniformSize - 1, extern template)

#ifdef DNDS_USE_CUDA
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, 0, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, 1, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, 2, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, 3, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, 4, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, 5, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, 6, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, 7, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, DynamicSize - 1, extern template)
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, NonUniformSize - 1, extern template)
#endif
}

namespace DNDS
{
    /*
#define DNDS_ARRAYDOF_DEVICEVIEW(B, n_m, n_n) ArrayDof<n_m, n_n>::template t_deviceView<B>

#define DNDS_ARRAYDOF_DEVICEVIEW_INST_DELETER_AND_FACTORY(B, n_m, n_n, ext)           \
    DNDS_DEVICE_STORAGE_BASE_DELETER_INST(DNDS_ARRAYDOF_DEVICEVIEW(B, n_m, n_n), ext) \
    DNDS_DEVICE_STORAGE_INST(DNDS_ARRAYDOF_DEVICEVIEW(B, n_m, n_n), B, ext)

#define DNDS_ARRAYDOF_INST_STORAGE(B, offset, ext)                                        \
    DNDS_ARRAYDOF_DEVICEVIEW_INST_DELETER_AND_FACTORY(B, 1, 1 + (offset), ext);           \
    DNDS_ARRAYDOF_DEVICEVIEW_INST_DELETER_AND_FACTORY(B, 2, 1 + (offset), ext);           \
    DNDS_ARRAYDOF_DEVICEVIEW_INST_DELETER_AND_FACTORY(B, 3, 1 + (offset), ext);           \
    DNDS_ARRAYDOF_DEVICEVIEW_INST_DELETER_AND_FACTORY(B, 4, 1 + (offset), ext);           \
    DNDS_ARRAYDOF_DEVICEVIEW_INST_DELETER_AND_FACTORY(B, 5, 1 + (offset), ext);           \
    DNDS_ARRAYDOF_DEVICEVIEW_INST_DELETER_AND_FACTORY(B, 6, 1 + (offset), ext);           \
    DNDS_ARRAYDOF_DEVICEVIEW_INST_DELETER_AND_FACTORY(B, 7, 1 + (offset), ext);           \
    DNDS_ARRAYDOF_DEVICEVIEW_INST_DELETER_AND_FACTORY(B, 8, 1 + (offset), ext);           \
    DNDS_ARRAYDOF_DEVICEVIEW_INST_DELETER_AND_FACTORY(B, DynamicSize, 1 + (offset), ext); \
    DNDS_ARRAYDOF_DEVICEVIEW_INST_DELETER_AND_FACTORY(B, NonUniformSize, 1 + (offset), ext);

    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, 0, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, 1, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, 2, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, 3, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, 4, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, 5, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, 6, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, 7, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, DynamicSize - 1, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, NonUniformSize - 1, extern)

#ifdef DNDS_USE_CUDA
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::CUDA, 0, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::CUDA, 1, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::CUDA, 2, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::CUDA, 3, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::CUDA, 4, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::CUDA, 5, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::CUDA, 6, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::CUDA, 7, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::CUDA, DynamicSize - 1, extern)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::CUDA, NonUniformSize - 1, extern)
#endif
*/
}
