#pragma once

#include "ArrayDOF.hpp"
// #include "DNDS/Device/CUDA_Utils.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/Errors.hpp"
#include "mpi.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace DNDS
{
    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::setConstant(t_self &self, real R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return; // can skip because non-collective

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
            return; // can skip because non-collective
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
            return; // can skip because non-collective

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
        if (self.Size() == 0)
            return; // can skip because non-collective

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

    template <int n_m, int n_n, typename TF>
    DNDS_GLOBAL static void cuda_ArrayEigenMatrix_uniform_element_mat_op(
        ArrayEigenMatrixDeviceView<DeviceBackend::CUDA, real, n_m, n_n> self,
        EigenMatrixView<DeviceBackend::CUDA, real, RowSize_To_EigenSize(n_m), RowSize_To_EigenSize(n_n)> mat,
        TF F_real_binary,
        rowsize elem_size)
    {
        // index N = self.Size();
        index N_d = self.DataSize();

        index tid = blockIdx.x * blockDim.x + threadIdx.x;
        index iData = tid;
        index iRow = iData / elem_size;
        rowsize i_inElem = iData % elem_size;
        if (iData > N_d)
            return;
        self.data()[iData] = F_real_binary(self.data()[iData], mat.map()(i_inElem));
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_plus_assign(t_self &self, const Eigen::Ref<const t_element_mat> &R)
    {

        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return; // can skip because non-collective

        HostDeviceEigenMatrix<real, RowSize_To_EigenSize(n_m), RowSize_To_EigenSize(n_n)> hd_R;
        hd_R.resize(R.rows(), R.cols());
        hd_R.map() = R;
        hd_R.to_device(DeviceBackend::CUDA);
        using t_MatrixDeviceView = EigenMatrixView<DeviceBackend::CUDA, real, RowSize_To_EigenSize(n_m), RowSize_To_EigenSize(n_n)>;
        t_MatrixDeviceView R_view =
            hd_R.template deviceView<DeviceBackend::CUDA>();

        if constexpr (t_self::IsCSR()) // applies to all layouts but slower
        {
            // iterator version
            auto self_father_begin = self.father->template begin<DeviceBackend::CUDA>();
            auto self_father_end = self.father->template end<DeviceBackend::CUDA>();
            // std::cout << "HERE" << std::endl;
            // // std::cout << self_father_begin != self_father_end << std::endl;
            // std::cout << typeid(self_father_end).name() << std::endl;
            // std::cout << self_father_end - self_father_begin << std::endl;
            // std::cout << self_father_begin.to_string() << std::endl;
            // std::cout << self_father_end.to_string() << std::endl;

            // std::cout << self_father_begin[1].cols() << std::endl;
            // std::cout << self_father_begin[1].data() << std::endl;

            // const auto b = self_father_begin;
            // std::cout << b[1].cols() << std::endl;
            // std::cout << b[1].data() << std::endl;

            // std::cout << (*self_father_begin).cols() << std::endl;
            // std::cout << (*(self.father->template begin<DeviceBackend::Host>())).cols() << std::endl;

            thrust::for_each(
                thrust::device,
                self_father_begin, self_father_end,
                [R_view, self_father_begin] DNDS_DEVICE_CALLABLE(t_MatrixDeviceView m)
                {
                    // printf("xb %d, %d, %p\n", self_father_begin[1].rows(), self_father_begin[1].cols(), self_father_begin[1].data());
                    // printf("xc %d, %d, %p\n", R_view.rows(), R_view.cols(), R_view.data());
                    // printf("xx %d, %d, %p\n", m.rows(), m.cols(), m.data());
                    // double m00 = m.map()(0);
                    m.map() += R_view.map();
                    // printf("xx %g %g\n", m00, m.map()(0));
                });

            thrust::for_each(
                thrust::device,
                self.son->template begin<DeviceBackend::CUDA>(),
                self.son->template end<DeviceBackend::CUDA>(),
                [R_view] DNDS_DEVICE_CALLABLE(t_MatrixDeviceView m)
                {
                    m.map() += R_view.map();
                });
        }
        else
        {
            rowsize elem_size_father = self.father->MatRowSize(0) * self.father->MatColSize(0);
            rowsize elem_size_son = self.son->MatRowSize(0) * self.son->MatColSize(0);
            DNDS_assert(elem_size_father == elem_size_son);

            index threadsPerBlock = 128;
            index blocksPerGridFather = (self.father->DataSize() + threadsPerBlock - 1) / threadsPerBlock;
            index blocksPerGridSon = (self.son->DataSize() + threadsPerBlock - 1) / threadsPerBlock;
            if (blocksPerGridFather)
                cuda_ArrayEigenMatrix_uniform_element_mat_op<<<blocksPerGridFather, threadsPerBlock>>>(
                    self_view.father, R_view,
                    thrust::plus<real>(),
                    elem_size_father);
            if (blocksPerGridSon)
                cuda_ArrayEigenMatrix_uniform_element_mat_op<<<blocksPerGridSon, threadsPerBlock>>>(
                    self_view.son, R_view,
                    thrust::plus<real>(),
                    elem_size_son);
        }
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
        if (self.Size() == 0)
            return; // can skip because non-collective

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
    DNDS_GLOBAL static void cuda_ArrayEigenMatrix_operator_mult_assign_scalar_arr_uniform(
        ArrayEigenMatrixDeviceView<DeviceBackend::CUDA, real, n_m, n_n> self,
        ArrayEigenMatrixDeviceView<DeviceBackend::CUDA, const real, 1, 1> R,
        rowsize elem_size)
    {
        // index N = self.Size();
        index N_d = self.DataSize();

        index tid = blockIdx.x * blockDim.x + threadIdx.x;
        index iData = tid;
        index iRow = iData / elem_size;
        if (iData > N_d)
            return;

        self.data()[iData] *= R[iRow](0);
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
            return; // can skip because non-collective

        DNDS_assert(R.father && R.son);
        DNDS_assert(R.father->device() == DeviceBackend::CUDA);
        DNDS_assert(R.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, 1, 1> R_view = R.template deviceView<DeviceBackend::CUDA>();

        if constexpr (t_self::IsCSR())
        {
            index threadsPerBlock = 32;
            index blocksPerGrid = (self.Size() + threadsPerBlock - 1) / threadsPerBlock;
            cuda_operator_mult_assign_scalar_arr<<<blocksPerGrid, threadsPerBlock>>>(
                self_view, R_view);
        }
        else
        {
            rowsize elem_size_father = self.father->MatRowSize(0) * self.father->MatColSize(0);
            rowsize elem_size_son = self.son->MatRowSize(0) * self.son->MatColSize(0);
            DNDS_assert(elem_size_father == elem_size_son);

            index threadsPerBlock = 128;
            index blocksPerGridFather = (self.father->DataSize() + threadsPerBlock - 1) / threadsPerBlock;
            index blocksPerGridSon = (self.son->DataSize() + threadsPerBlock - 1) / threadsPerBlock;
            if (blocksPerGridFather)
                cuda_ArrayEigenMatrix_operator_mult_assign_scalar_arr_uniform<<<blocksPerGridFather, threadsPerBlock>>>(
                    self_view.father, R_view.father, elem_size_father);
            if (blocksPerGridSon)
                cuda_ArrayEigenMatrix_operator_mult_assign_scalar_arr_uniform<<<blocksPerGridSon, threadsPerBlock>>>(
                    self_view.son, R_view.son, elem_size_son);
        }
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_mult_assign(t_self &self, const Eigen::Ref<const t_element_mat> &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return; // can skip because non-collective

        HostDeviceEigenMatrix<real, RowSize_To_EigenSize(n_m), RowSize_To_EigenSize(n_n)> hd_R;
        hd_R.resize(R.rows(), R.cols());
        hd_R.map() = R;
        hd_R.to_device(DeviceBackend::CUDA);
        using t_MatrixDeviceView = EigenMatrixView<DeviceBackend::CUDA, real, RowSize_To_EigenSize(n_m), RowSize_To_EigenSize(n_n)>;
        t_MatrixDeviceView R_view =
            hd_R.template deviceView<DeviceBackend::CUDA>();

        if constexpr (t_self::IsCSR()) // applies to all layouts but slower
        {
            // iterator version
            auto self_father_begin = self.father->template begin<DeviceBackend::CUDA>();
            auto self_father_end = self.father->template end<DeviceBackend::CUDA>();

            thrust::for_each(
                thrust::device,
                self.father->template begin<DeviceBackend::CUDA>(), self.father->template end<DeviceBackend::CUDA>(),
                [R_view, self_father_begin] DNDS_DEVICE_CALLABLE(t_MatrixDeviceView m)
                {
                    m.map().array() *= R_view.map().array();
                });

            thrust::for_each(
                thrust::device,
                self.son->template begin<DeviceBackend::CUDA>(), self.son->template end<DeviceBackend::CUDA>(),
                [R_view] DNDS_DEVICE_CALLABLE(t_MatrixDeviceView m)
                {
                    m.map().array() *= R_view.map().array();
                });
        }
        else
        {
            rowsize elem_size_father = self.father->MatRowSize(0) * self.father->MatColSize(0);
            rowsize elem_size_son = self.son->MatRowSize(0) * self.son->MatColSize(0);
            DNDS_assert(elem_size_father == elem_size_son);

            index threadsPerBlock = 128;
            index blocksPerGridFather = (self.father->DataSize() + threadsPerBlock - 1) / threadsPerBlock;
            index blocksPerGridSon = (self.son->DataSize() + threadsPerBlock - 1) / threadsPerBlock;
            if (blocksPerGridFather)
                cuda_ArrayEigenMatrix_uniform_element_mat_op<<<blocksPerGridFather, threadsPerBlock>>>(
                    self_view.father, R_view,
                    thrust::multiplies<real>(),
                    elem_size_father);
            if (blocksPerGridSon)
                cuda_ArrayEigenMatrix_uniform_element_mat_op<<<blocksPerGridSon, threadsPerBlock>>>(
                    self_view.son, R_view,
                    thrust::multiplies<real>(),
                    elem_size_son);
        }
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::operator_mult_assign(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        if (self.Size() == 0)
            return; // can skip because non-collective

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
            return; // can skip because non-collective

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
            return; // can skip because non-collective

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
            return; // can skip because non-collective

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
                return sqr(thrust::get<0>(v) - thrust::get<1>(v));
            },
            0.0,
            thrust::plus<real>());

        MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        return std::sqrt(sqrSumAll);
    }

    template <int n_m, int n_n>
    real ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::reduction(t_self &self, const std::string &op)
    {
        real sqrSum{UnInitReal}, sqrSumAll{UnInitReal};

        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        // int device_id;
        // DNDS_CUDA_CHECKED(cudaGetDevice(&device_id));
        // std::cout
        //     << "device_id " << device_id << std::endl;
        thrust::device_ptr<real> d_self_father(self_view.father.data());
        index self_father_d_size = size_t_to_signed<index>(self_view.father.DataSize());
        thrust::device_ptr<real> d_self_son(self_view.son.data());
        index self_son_d_size = size_t_to_signed<index>(self_view.son.DataSize());
        // std::cout << "got device_ptr" << std::endl;

        auto execute = [&](real init, auto binary_op)
        {
            sqrSum = thrust::reduce(
                d_self_father, d_self_father + self_father_d_size,
                init,
                binary_op);
        };

        MPI_Op mpi_op = MPI_OP_NULL;
        if (op == "min")
        {
            mpi_op = MPI_MIN;
            sqrSum = sqrSumAll = std::numeric_limits<real>::max();
            // std::cout << sqrSum << std::endl;
            // thrust::fill_n(d_self_father, self_father_d_size, 1.0);
            // sqrSum = thrust::reduce(
            //     d_self_father, d_self_father + self_father_d_size,
            //     sqrSumAll,
            //     [] DNDS_DEVICE_CALLABLE(real a, real b)
            //     { return a < b ? a : b; });
            // auto v = thrust::min_element(thrust::device, d_self_father, d_self_father + self_father_d_size);
            // double vv;
            // thrust::copy(v, v + 1, &vv);
            // std::cout << vv << ", " << sqrSum << "," << self_father_d_size << std::endl;
            // thrust::host_vector<real> h_vec(self_father_d_size);
            // thrust::copy_n(d_self_father, self_father_d_size, h_vec.begin());

            auto result = thrust::min_element(thrust::device, d_self_father, d_self_father + self_father_d_size);
            thrust::copy_n(result, 1, &sqrSum);
        }
        else if (op == "max")
        {
            mpi_op = MPI_MAX;
            sqrSum = sqrSumAll = std::numeric_limits<real>::lowest();
            // execute(sqrSumAll, [] DNDS_DEVICE_CALLABLE(real a, real b)
            //         { return thrust::max(a, b); });
            auto result = thrust::max_element(thrust::device, d_self_father, d_self_father + self_father_d_size);
            thrust::copy_n(result, 1, &sqrSum);
        }
        else if (op == "sum")
        {
            mpi_op = MPI_SUM;
            sqrSum = sqrSumAll = 0.0;
            execute(sqrSumAll, thrust::plus<real>());
            // thrust::max_element()
        }
        else
            DNDS_assert_info(false, op);

        MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, mpi_op, self.father->getMPI().comm);
        // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
        return sqrSumAll;
    }

    template <typename TFTrans, typename TF, int block_size_max = 256>
    DNDS_GLOBAL static void cuda_ArrayEigenMatrix_bare_uniform_element_reduction(
        const real *v,
        real *v_out,
        real init,
        TFTrans F_real_trans,
        TF F_real_binary,
        index N,
        rowsize elem_size)
    {

        uint32_t tid = threadIdx.x;
        uint32_t block_size = blockDim.x;
        uint32_t i_Col = blockIdx.x % elem_size;
        uint32_t i_block_on_Row = blockIdx.x / elem_size;
        uint32_t i_Row = i_block_on_Row * block_size + tid;

        using BlockReduce = cub::BlockReduce<real, block_size_max>;
        __shared__ typename BlockReduce::TempStorage tempStorage;

        real v_cur = i_Row >= N ? init : F_real_trans(v[i_Row * elem_size + i_Col]);
        real block_sum = BlockReduce(tempStorage).Reduce(v_cur, F_real_binary);

        if (tid == 0)
            v_out[i_block_on_Row * elem_size + i_Col] = block_sum;
    }

    template <typename TFTrans, typename TF, int block_size_max = 256>
    DNDS_GLOBAL static void cuda_ArrayEigenMatrix_bare_uniform_element_reduction_binary(
        const real *v,
        const real *v1,
        real *v_out,
        real init,
        TFTrans F_real_trans,
        TF F_real_binary,
        index N,
        rowsize elem_size)
    {

        uint32_t tid = threadIdx.x;
        uint32_t block_size = blockDim.x;
        uint32_t i_Col = blockIdx.x % elem_size;
        uint32_t i_block_on_Row = blockIdx.x / elem_size;
        uint32_t i_Row = i_block_on_Row * block_size + tid;

        using BlockReduce = cub::BlockReduce<real, block_size_max>;
        __shared__ typename BlockReduce::TempStorage tempStorage;

        real v_cur = i_Row >= N ? init : F_real_trans(v[i_Row * elem_size + i_Col], v1[i_Row * elem_size + i_Col]);
        real block_sum = BlockReduce(tempStorage).Reduce(v_cur, F_real_binary);

        if (tid == 0)
            v_out[i_block_on_Row * elem_size + i_Col] = block_sum;
    }

    template <int n_m, int n_n>
    typename ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::t_element_mat
    ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::componentWiseNorm1(t_self &self)
    {
        t_element_mat minLocal, min;
        //! let it fail if size not compatible
        minLocal.resize(self.father->MatRowSize(0), self.father->MatColSize(0));
        minLocal.setConstant(0);
        min = minLocal;

        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();

        if (self.father->Size())
        {
            if constexpr (t_self::IsCSR())
            {
                DNDS_assert_info(false, "not implemented");
            }
            else
            {
                rowsize elem_size_father = self.father->MatRowSize(0) * self.father->MatColSize(0);
                rowsize elem_size_son = self.son->MatRowSize(0) * self.son->MatColSize(0);
                DNDS_assert(elem_size_father == elem_size_son);

                uint32_t threadsPerBlock = 256; // must be a power of 2!

                uint32_t blocks_on_row = (self.father->Size() + threadsPerBlock - 1) / threadsPerBlock;
                uint32_t n_blocks = blocks_on_row * elem_size_father;

                thrust::device_vector<real> buffer(n_blocks, 0.0);

                cuda_ArrayEigenMatrix_bare_uniform_element_reduction<<<n_blocks, threadsPerBlock>>>(
                    self_view.father.data(),
                    thrust::raw_pointer_cast(buffer.data()),
                    0.0,
                    [] DNDS_DEVICE_CALLABLE(real v)
                    { return std::abs(v); },
                    thrust::plus<real>(),
                    self.father->Size(),
                    elem_size_father);

                uint32_t cur_size = blocks_on_row;

                blocks_on_row = (cur_size + threadsPerBlock - 1) / threadsPerBlock;
                n_blocks = blocks_on_row * elem_size_father;
                thrust::device_vector<real> buffer1;
                if (cur_size >= threadsPerBlock)
                    buffer1.resize(n_blocks, 0.0);

                while (cur_size >= threadsPerBlock)
                {
                    std::swap(buffer1, buffer);
                    cuda_ArrayEigenMatrix_bare_uniform_element_reduction<<<n_blocks, threadsPerBlock>>>(
                        thrust::raw_pointer_cast(buffer1.data()),
                        thrust::raw_pointer_cast(buffer.data()),
                        0.0,
                        [] DNDS_DEVICE_CALLABLE(real v)
                        { return std::abs(v); },
                        thrust::plus<real>(),
                        cur_size,
                        elem_size_father);
                    cur_size = blocks_on_row;
                    blocks_on_row = (cur_size + threadsPerBlock - 1) / threadsPerBlock;
                    n_blocks = blocks_on_row * elem_size_father;
                }

                thrust::host_vector<real> buffer_h(cur_size * elem_size_father);
                thrust::copy_n(buffer.begin(), buffer_h.size(), buffer_h.begin());
                for (index i = 0; i < cur_size; i++)
                    for (rowsize j = 0; j < elem_size_father; j++)
                        minLocal(j) += buffer_h[i * elem_size_father + j];
            }
        }

        MPI::Allreduce(minLocal.data(), min.data(), minLocal.size(), DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        return min;
    }

    template <int n_m, int n_n>
    typename ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::t_element_mat
    ArrayDofOp<DeviceBackend::CUDA, n_m, n_n>::componentWiseNorm1(t_self &self, const t_self &R)
    {
        t_element_mat minLocal, min;
        //! let it fail if size not compatible
        minLocal.resize(self.father->MatRowSize(0), self.father->MatColSize(0));
        minLocal.setConstant(0);
        min = minLocal;

        DNDS_assert(self.father && self.son);
        DNDS_assert(self.father->device() == DeviceBackend::CUDA);
        DNDS_assert(self.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceView<DeviceBackend::CUDA, n_m, n_n> self_view = self.template deviceView<DeviceBackend::CUDA>();
        DNDS_assert(R.father && R.son);
        DNDS_assert(R.father->device() == DeviceBackend::CUDA);
        DNDS_assert(R.son->device() == DeviceBackend::CUDA);
        ArrayDofDeviceViewConst<DeviceBackend::CUDA, n_m, n_n> R_view = R.template deviceView<DeviceBackend::CUDA>();

        if (self.father->Size())
        {
            if constexpr (t_self::IsCSR())
            {
                DNDS_assert_info(false, "not implemented");
            }
            else
            {
                rowsize elem_size_father = self.father->MatRowSize(0) * self.father->MatColSize(0);
                rowsize elem_size_son = self.son->MatRowSize(0) * self.son->MatColSize(0);
                DNDS_assert(elem_size_father == elem_size_son);

                uint32_t threadsPerBlock = 256; // must be a power of 2!

                uint32_t blocks_on_row = (self.father->Size() + threadsPerBlock - 1) / threadsPerBlock;
                uint32_t n_blocks = blocks_on_row * elem_size_father;

                thrust::device_vector<real> buffer(n_blocks, 0.0);

                cuda_ArrayEigenMatrix_bare_uniform_element_reduction_binary<<<n_blocks, threadsPerBlock>>>(
                    self_view.father.data(),
                    R_view.father.data(),
                    thrust::raw_pointer_cast(buffer.data()),
                    0.0,
                    [] DNDS_DEVICE_CALLABLE(real v, real v1)
                    { return std::abs(v - v1); },
                    thrust::plus<real>(),
                    self.father->Size(),
                    elem_size_father);

                uint32_t cur_size = blocks_on_row;

                blocks_on_row = (cur_size + threadsPerBlock - 1) / threadsPerBlock;
                n_blocks = blocks_on_row * elem_size_father;
                thrust::device_vector<real> buffer1;
                if (cur_size >= threadsPerBlock)
                    buffer1.resize(n_blocks, 0.0);

                while (cur_size >= threadsPerBlock)
                {
                    std::swap(buffer1, buffer);
                    cuda_ArrayEigenMatrix_bare_uniform_element_reduction<<<n_blocks, threadsPerBlock>>>(
                        thrust::raw_pointer_cast(buffer1.data()),
                        thrust::raw_pointer_cast(buffer.data()),
                        0.0,
                        [] DNDS_DEVICE_CALLABLE(real v)
                        { return std::abs(v); },
                        thrust::plus<real>(),
                        cur_size,
                        elem_size_father);
                    cur_size = blocks_on_row;
                    blocks_on_row = (cur_size + threadsPerBlock - 1) / threadsPerBlock;
                    n_blocks = blocks_on_row * elem_size_father;
                }

                thrust::host_vector<real> buffer_h(cur_size * elem_size_father);
                thrust::copy_n(buffer.begin(), buffer_h.size(), buffer_h.begin());
                for (index i = 0; i < cur_size; i++)
                    for (rowsize j = 0; j < elem_size_father; j++)
                        minLocal(j) += buffer_h[i * elem_size_father + j];
            }
        }

        MPI::Allreduce(minLocal.data(), min.data(), minLocal.size(), DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
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
                return thrust::get<0>(v) * thrust::get<1>(v);
            },
            0.0,
            thrust::plus<real>());

        MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
        return std::sqrt(sqrSumAll);
    }
}