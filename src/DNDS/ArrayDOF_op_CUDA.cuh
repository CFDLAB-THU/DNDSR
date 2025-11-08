#pragma once

#include "ArrayDOF.hpp"
#include "DNDS/Defines.hpp"

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>

namespace DNDS
{
    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::setConstant(t_self &self, real R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();

        thrust::device_ptr<real> d_father(self_view.father.data());
        index father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_son(self_view.son.data());
        index son_d_size = size_t_to_signed<index>(self_view.son.DataSize());
        auto set_value = [R] DNDS_DEVICE_CALLABLE(real v)
        {
            return R;
        };
        thrust::transform(thrust::device, d_father, d_father + father_d_size, d_father, set_value);
        thrust::transform(thrust::device, d_son, d_son + son_d_size, d_son, set_value);
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::setConstant(t_self &self, const Eigen::Ref<const t_element_mat> &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return;
        //! host-side overwriting
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i) = R;

        thrust::device_ptr<real> d_father(self_view.father.data());
        index father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        DNDS_assert(self_view.father.DataSize() == self.father->DataSize());
        thrust::device_ptr<real> d_son(self_view.son.data());
        index son_d_size = size_t_to_signed<index>(self_view.son.DataSize());
        DNDS_assert(self_view.son.DataSize() == self.son->DataSize());
        thrust::copy(self.father->data(), self.father->data() + father_d_size, d_father);
        thrust::copy(self.son->data(), self.son->data() + son_d_size, d_son);
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_plus_assign(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return;

        DNDS_assert(R.father && R.son);
        DNDS_assert(R.father->device() == DeviceBackend::CUDA);
        DNDS_assert(R.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, n_m, n_n> R_view = R.template deviceView<DeviceBackend::CUDA>();

        thrust::device_ptr<real> d_self_father(self_view.father.data());
        index self_father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_self_son(self_view.son.data());
        index self_son_d_size = size_t_to_signed<index>(self_view.son.DataSize());

        thrust::device_ptr<const real> d_R_father(R_view.father.data());
        index R_father_d_size = size_t_to_signed<index>(R_view.father.DataSize());
        thrust::device_ptr<const real> d_R_son(R_view.son.data());
        index R_son_d_size = size_t_to_signed<index>(R_view.son.DataSize());

        DNDS_assert(R_father_d_size == self_father_d_size);
        DNDS_assert(R_son_d_size == self_son_d_size);

        // thrust::transform(
        //     thrust::device,
        //     thrust::make_zip_iterator(d_self_father, d_R_father),
        //     thrust::make_zip_iterator(d_self_father + self_father_d_size, d_R_father + R_father_d_size),
        //     d_self_father,
        //     [](){}
        // );
        thrust::transform(
            thrust::device,
            d_self_father, d_self_father + self_father_d_size, // A
            d_R_father,                                        // B
            d_self_father,                                     // C
            thrust::plus<real>());

        thrust::transform(
            thrust::device,
            d_self_son, d_self_son + self_son_d_size, // A
            d_R_son,                                  // B
            d_self_son,                               // C
            thrust::plus<real>());
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_plus_assign(t_self &self, real R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();

        thrust::device_ptr<real> d_father(self_view.father.data());
        index father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_son(self_view.son.data());
        index son_d_size = size_t_to_signed<index>(self_view.son.DataSize());
        auto set_value = [R] DNDS_DEVICE_CALLABLE(real v)
        {
            return R + v;
        };
        thrust::transform(thrust::device, d_father, d_father + father_d_size, d_father, set_value);
        thrust::transform(thrust::device, d_son, d_son + son_d_size, d_son, set_value);
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_plus_assign(t_self &self, const Eigen::Ref<const t_element_mat> &R)
    {
        DNDS_assert_info(false, "not implemented");
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i) += R;
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_minus_assign(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return;

        DNDS_assert(R.father && R.son);
        DNDS_assert(R.father->device() == DeviceBackend::CUDA);
        DNDS_assert(R.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, n_m, n_n> R_view = R.template deviceView<DeviceBackend::CUDA>();

        thrust::device_ptr<real> d_self_father(self_view.father.data());
        index self_father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_self_son(self_view.son.data());
        index self_son_d_size = size_t_to_signed<index>(self_view.son.DataSize());

        thrust::device_ptr<const real> d_R_father(R_view.father.data());
        index R_father_d_size = size_t_to_signed<index>(R_view.father.DataSize());
        thrust::device_ptr<const real> d_R_son(R_view.son.data());
        index R_son_d_size = size_t_to_signed<index>(R_view.son.DataSize());

        DNDS_assert(R_father_d_size == self_father_d_size);
        DNDS_assert(R_son_d_size == self_son_d_size);

        thrust::transform(
            thrust::device,
            d_self_father, d_self_father + self_father_d_size, // A
            d_R_father,                                        // B
            d_self_father,                                     // C
            thrust::minus<real>());

        thrust::transform(
            thrust::device,
            d_self_son, d_self_son + self_son_d_size, // A
            d_R_son,                                  // B
            d_self_son,                               // C
            thrust::minus<real>());
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_mult_assign(t_self &self, real R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();

        thrust::device_ptr<real> d_father(self_view.father.data());
        index father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_son(self_view.son.data());
        index son_d_size = size_t_to_signed<index>(self_view.son.DataSize());
        auto set_value = [R] DNDS_DEVICE_CALLABLE(real v)
        {
            return R * v;
        };
        thrust::transform(thrust::device, d_father, d_father + father_d_size, d_father, set_value);
        thrust::transform(thrust::device, d_son, d_son + son_d_size, d_son, set_value);
    }

    template <int n_m, int n_n>
    DNDS_GLOBAL static void cuda_operator_mult_assign_scalar_arr(
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self,
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, 1, 1> R)
    {
        index N = self.Size();
        index tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid > N)
            return;
        self[tid] *= R[tid](0);
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_mult_assign_scalar_arr(t_self &self, const ArrayDof<1, 1> &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return;

        DNDS_assert(R.father && R.son);
        DNDS_assert(R.father->device() == DeviceBackend::CUDA);
        DNDS_assert(R.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, 1, 1> R_view = R.template deviceView<DeviceBackend::CUDA>();

        index threadsPerBlock = 32;

        index blocksPerGrid = (self.Size() + threadsPerBlock - 1) / threadsPerBlock;

        cuda_operator_mult_assign_scalar_arr<<<blocksPerGrid, threadsPerBlock>>>(
            self_view, R_view);
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_mult_assign(t_self &self, const Eigen::Ref<const t_element_mat> &R)
    {
        DNDS_assert_info(false, "not implemented");
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i).array() *= R.array();
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_mult_assign(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return;

        DNDS_assert(R.father && R.son);
        DNDS_assert(R.father->device() == DeviceBackend::CUDA);
        DNDS_assert(R.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, n_m, n_n> R_view = R.template deviceView<DeviceBackend::CUDA>();

        thrust::device_ptr<real> d_self_father(self_view.father.data());
        index self_father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_self_son(self_view.son.data());
        index self_son_d_size = size_t_to_signed<index>(self_view.son.DataSize());

        thrust::device_ptr<const real> d_R_father(R_view.father.data());
        index R_father_d_size = size_t_to_signed<index>(R_view.father.DataSize());
        thrust::device_ptr<const real> d_R_son(R_view.son.data());
        index R_son_d_size = size_t_to_signed<index>(R_view.son.DataSize());

        DNDS_assert(R_father_d_size == self_father_d_size);
        DNDS_assert(R_son_d_size == self_son_d_size);

        thrust::transform(
            thrust::device,
            d_self_father, d_self_father + self_father_d_size, // A
            d_R_father,                                        // B
            d_self_father,                                     // C
            thrust::multiplies<real>());

        thrust::transform(
            thrust::device,
            d_self_son, d_self_son + self_son_d_size, // A
            d_R_son,                                  // B
            d_self_son,                               // C
            thrust::multiplies<real>());
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_div_assign(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return;

        DNDS_assert(R.father && R.son);
        DNDS_assert(R.father->device() == DeviceBackend::CUDA);
        DNDS_assert(R.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, n_m, n_n> R_view = R.template deviceView<DeviceBackend::CUDA>();

        thrust::device_ptr<real> d_self_father(self_view.father.data());
        index self_father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_self_son(self_view.son.data());
        index self_son_d_size = size_t_to_signed<index>(self_view.son.DataSize());

        thrust::device_ptr<const real> d_R_father(R_view.father.data());
        index R_father_d_size = size_t_to_signed<index>(R_view.father.DataSize());
        thrust::device_ptr<const real> d_R_son(R_view.son.data());
        index R_son_d_size = size_t_to_signed<index>(R_view.son.DataSize());

        DNDS_assert(R_father_d_size == self_father_d_size);
        DNDS_assert(R_son_d_size == self_son_d_size);

        thrust::transform(
            thrust::device,
            d_self_father, d_self_father + self_father_d_size, // A
            d_R_father,                                        // B
            d_self_father,                                     // C
            thrust::divides<real>());

        thrust::transform(
            thrust::device,
            d_self_son, d_self_son + self_son_d_size, // A
            d_R_son,                                  // B
            d_self_son,                               // C
            thrust::divides<real>());
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_assign(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return;

        DNDS_assert(R.father && R.son);
        DNDS_assert(R.father->device() == DeviceBackend::CUDA);
        DNDS_assert(R.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, n_m, n_n> R_view = R.template deviceView<DeviceBackend::CUDA>();

        thrust::device_ptr<real> d_self_father(self_view.father.data());
        index self_father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_self_son(self_view.son.data());
        index self_son_d_size = size_t_to_signed<index>(self_view.son.DataSize());

        thrust::device_ptr<const real> d_R_father(R_view.father.data());
        index R_father_d_size = size_t_to_signed<index>(R_view.father.DataSize());
        thrust::device_ptr<const real> d_R_son(R_view.son.data());
        index R_son_d_size = size_t_to_signed<index>(R_view.son.DataSize());

        DNDS_assert(R_father_d_size == self_father_d_size);
        DNDS_assert(R_son_d_size == self_son_d_size);

        thrust::copy(d_R_father, d_R_father + R_father_d_size, d_self_father);
        thrust::copy(d_R_son, d_R_son + R_son_d_size, d_self_son);
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::addTo(t_self &self, const t_self &R, real r)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return;

        DNDS_assert(R.father && R.son);
        DNDS_assert(R.father->device() == DeviceBackend::CUDA);
        DNDS_assert(R.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, n_m, n_n> R_view = R.template deviceView<DeviceBackend::CUDA>();

        thrust::device_ptr<real> d_self_father(self_view.father.data());
        index self_father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_self_son(self_view.son.data());
        index self_son_d_size = size_t_to_signed<index>(self_view.son.DataSize());

        thrust::device_ptr<const real> d_R_father(R_view.father.data());
        index R_father_d_size = size_t_to_signed<index>(R_view.father.DataSize());
        thrust::device_ptr<const real> d_R_son(R_view.son.data());
        index R_son_d_size = size_t_to_signed<index>(R_view.son.DataSize());

        DNDS_assert(R_father_d_size == self_father_d_size);
        DNDS_assert(R_son_d_size == self_son_d_size);

        auto multadd = [r] DNDS_DEVICE_CALLABLE(real a, real b)
        {
            return a + b * r;
        };

        thrust::transform(
            thrust::device,
            d_self_father, d_self_father + self_father_d_size, // A
            d_R_father,                                        // B
            d_self_father,                                     // C
            multadd);

        thrust::transform(
            thrust::device,
            d_self_son, d_self_son + self_son_d_size, // A
            d_R_son,                                  // B
            d_self_son,                               // C
            multadd);
    }

    template <int n_m, int n_n>
    real ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::norm2(t_self &self)
    {
        real sqrSum{0}, sqrSumAll{0};

        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return 0.0;

        thrust::device_ptr<real> d_self_father(self_view.father.data());
        index self_father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_self_son(self_view.son.data());
        index self_son_d_size = size_t_to_signed<index>(self_view.son.DataSize());

        // std::cout << "HERE BOONORM " << std::endl;
        // std::cout << d_self_father << std::endl;
        // std::cout << d_self_father + 1 << std::endl;
        // std::cout << self_father_d_size << std::endl;
        // double v{-1};
        // thrust::copy_n(d_self_father + self_father_d_size - 1, 1, &v);
        // std::cout << v << std::endl;

        sqrSum = thrust::transform_reduce(
            d_self_father, d_self_father + self_father_d_size,
            thrust::square<real>(),
            0.0,
            thrust::plus<real>());

        MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
        return std::sqrt(sqrSumAll);
    }

    template <int n_m, int n_n>
    real ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::norm2(t_self &self, const t_self &R)
    {
        real sqrSum{0}, sqrSumAll{0};

        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return 0.0;

        DNDS_assert(R.father && R.son);
        DNDS_assert(R.father->device() == DeviceBackend::CUDA);
        DNDS_assert(R.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, n_m, n_n> R_view = R.template deviceView<DeviceBackend::CUDA>();

        thrust::device_ptr<real> d_self_father(self_view.father.data());
        index self_father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_self_son(self_view.son.data());
        index self_son_d_size = size_t_to_signed<index>(self_view.son.DataSize());

        thrust::device_ptr<const real> d_R_father(R_view.father.data());
        index R_father_d_size = size_t_to_signed<index>(R_view.father.DataSize());
        thrust::device_ptr<const real> d_R_son(R_view.son.data());
        index R_son_d_size = size_t_to_signed<index>(R_view.son.DataSize());

        DNDS_assert(R_father_d_size == self_father_d_size);
        DNDS_assert(R_son_d_size == self_son_d_size);

        sqrSum = thrust::transform_reduce(
            thrust::device,
            thrust::make_zip_iterator(
                thrust::make_tuple(d_self_father, d_R_father)),
            thrust::make_zip_iterator(
                thrust::make_tuple(d_self_father + self_father_d_size, d_R_father + R_father_d_size)),
            [] DNDS_DEVICE_CALLABLE(const thrust::tuple<real, real> &v)
            {
                return sqr(v.get<0>() - v.get<1>());
            },
            0.0,
            thrust::plus<real>());

        MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
        return std::sqrt(sqrSumAll);
    }

    template <int n_m, int n_n>
    typename ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::t_element_mat
    ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::componentWiseNorm1(t_self &self)
    {
        DNDS_assert_info(false, "not implemented");
        t_element_mat minLocal, min;
        //! let it fail if size not compatible
        minLocal.resize(self.father->MatRowSize(0), self.father->MatColSize(0));
        minLocal.setConstant(0);
        min = minLocal;

        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return min;

        thrust::device_ptr<real> d_self_father(self_view.father.data());
        index self_father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_self_son(self_view.son.data());
        index self_son_d_size = size_t_to_signed<index>(self_view.son.DataSize());

        auto self_father_begin = self.father->template begin<DeviceBackend::CUDA>();
        auto self_father_end = self.father->template end<DeviceBackend::CUDA>();

        MPI::Allreduce(minLocal.data(), min.data(), minLocal.size(), DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        return min;
    }

    template <int n_m, int n_n>
    typename ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::t_element_mat
    ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::componentWiseNorm1(t_self &self, const t_self &R)
    {
        DNDS_assert_info(false, "not implemented");
        t_element_mat minLocal, min;
        return min;
    }

    template <int n_m, int n_n>
    real ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::dot(t_self &self, const t_self &R)
    {
        real sqrSum{0}, sqrSumAll{0};

        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return 0.0;

        DNDS_assert(R.father && R.son);
        DNDS_assert(R.father->device() == DeviceBackend::CUDA);
        DNDS_assert(R.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, n_m, n_n> R_view = R.template deviceView<DeviceBackend::CUDA>();

        thrust::device_ptr<real> d_self_father(self_view.father.data());
        index self_father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_self_son(self_view.son.data());
        index self_son_d_size = size_t_to_signed<index>(self_view.son.DataSize());

        thrust::device_ptr<const real> d_R_father(R_view.father.data());
        index R_father_d_size = size_t_to_signed<index>(R_view.father.DataSize());
        thrust::device_ptr<const real> d_R_son(R_view.son.data());
        index R_son_d_size = size_t_to_signed<index>(R_view.son.DataSize());

        DNDS_assert(R_father_d_size == self_father_d_size);
        DNDS_assert(R_son_d_size == self_son_d_size);

        sqrSum = thrust::transform_reduce(
            thrust::device,
            thrust::make_zip_iterator(
                thrust::make_tuple(d_self_father, d_R_father)),
            thrust::make_zip_iterator(
                thrust::make_tuple(d_self_father + self_father_d_size, d_R_father + R_father_d_size)),
            [] DNDS_DEVICE_CALLABLE(const thrust::tuple<real, real> &v)
            {
                return v.get<0>() * v.get<1>();
            },
            0.0,
            thrust::plus<real>());

        MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
        return std::sqrt(sqrSumAll);
    }
}