#pragma once

#include "ArrayDerived/ArrayEigenMatrix.hpp"

#include "ArrayDerived/ArrayEigenVector.hpp"

#include "ArrayPair.hpp"

namespace DNDS
{
    template <DeviceBackend B, int n_m, int n_n>
    class ArrayDofDeviceView : public ArrayPairDeviceView<B, ArrayEigenMatrix<n_m, n_n>>
    {
    public:
        using t_base = ArrayPairDeviceView<B, ArrayEigenMatrix<n_m, n_n>>;
        using t_base::t_base;

        ArrayDofDeviceView(const t_base &base_view) : t_base(base_view) {}
        ArrayDofDeviceView(t_base &&base_view) : t_base(base_view) {}
    };

    template <int n_m, int n_n>
    class ArrayDof;

    template <DeviceBackend B, int n_m, int n_n>
    class ArrayDofOp;

#define DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n)
#define DNDS_ARRAY_DOF_OP_FUNC_LIST(B, n_m, n_n, spec)                                                                                     \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) setConstant(t_self &self, real R);                                            \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) setConstant(t_self &self, const Eigen::Ref<const t_element_mat> &R);          \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_plus_assign(t_self &self, const t_self &R);                          \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_plus_assign(t_self &self, real R);                                   \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_plus_assign(t_self &self, const Eigen::Ref<const t_element_mat> &R); \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_minus_assign(t_self &self, const t_self &R);                         \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_mult_assign(t_self &self, real R);                                   \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_mult_assign_scalar_arr(t_self &self, const ArrayDof<1, 1> &R);       \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_mult_assign(t_self &self, const Eigen::Ref<const t_element_mat> &R); \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_mult_assign(t_self &self, const t_self &R);                          \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_div_assign(t_self &self, const t_self &R);                           \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) operator_assign(t_self &self, const t_self &R);                               \
    spec void DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) addTo(t_self &self, const t_self &R, real r);                                 \
    spec real DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) norm2(t_self &self);                                                          \
    spec ArrayDofOp<B, n_m, n_n>::t_element_mat DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) componentWiseNorm1(t_self &self);           \
    spec real DNDS_ARRAY_DOF_OP_FUNC_LIST_SCOPE(B, n_m, n_n) dot(t_self &self, const t_self &R);

    template <int n_m, int n_n>
    class ArrayDofOp<DeviceBackend::Host, n_m, n_n>
    {
    public:
        using t_self = ArrayDof<n_m, n_n>;
        using t_element_mat = Eigen::Matrix<real, RowSize_To_EigenSize(n_m), RowSize_To_EigenSize(n_n)>;
        DNDS_ARRAY_DOF_OP_FUNC_LIST(DeviceBackend::Host, n_m, n_n, static)
    };

#ifdef DNDS_USE_CUDA
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

    template <int n_m, int n_n>
    class ArrayDof : public ArrayEigenMatrixPair<n_m, n_n>
    {
    public:
        using t_base = ArrayEigenMatrixPair<n_m, n_n>;
        using t_base::t_base;

        template <DeviceBackend B>
        using t_deviceView = ArrayDofDeviceView<B, n_m, n_n>;

        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            return {t_base::template deviceView<B>()};
        }

        using t_base::to_device;
        using t_base::to_host;

        using t_self = ArrayDof<n_m, n_n>;

        template <DeviceBackend B>
        using t_ops = ArrayDofOp<B, n_m, n_n>;

        using t_element_mat = Eigen::Matrix<real, RowSize_To_EigenSize(n_m), RowSize_To_EigenSize(n_n)>;

        void setConstant(real R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), setConstant(*this, R));
        }

        void setConstant(const Eigen::Ref<const t_element_mat> &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), setConstant(*this, R));
        }

        void operator+=(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_plus_assign(*this, R));
        }

        void operator+=(real R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_plus_assign(*this, R));
        }

        void operator+=(const Eigen::Ref<const t_element_mat> &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_plus_assign(*this, R));
        }

        void operator-=(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_minus_assign(*this, R));
        }

        void operator*=(real R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_mult_assign(*this, R));
        }

        template <int n_m_T = n_m>
        std::enable_if_t<!(n_m_T == 1 && n_n == 1)>
        operator*=(ArrayDof<1, 1> &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_mult_assign_scalar_arr(*this, R));
        }

        void operator*=(const Eigen::Ref<const t_element_mat> &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_mult_assign(*this, R));
        }

        void operator*=(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_mult_assign(*this, R));
        }

        void operator/=(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_div_assign(*this, R));
        }

        void operator=(const t_self &R)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), operator_assign(*this, R));
        }

        void addTo(const t_self &R, real r)
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), addTo(*this, R, r));
        }

        real norm2()
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), norm2(*this));
        }

        t_element_mat componentWiseNorm1()
        {
            DNDS_ARRAY_OP_SWITCHER(this->father->device(), componentWiseNorm1(*this));
        }

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
