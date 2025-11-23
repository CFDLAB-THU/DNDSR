#pragma once

#include "ArrayDOF.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/Errors.hpp"

namespace DNDS
{
    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::setConstant(t_self &self, real R)
    {
        DNDS_assert(self.father && self.son);
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i).setConstant(R);
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::setConstant(t_self &self, const Eigen::Ref<const t_element_mat> &R)
    {
        DNDS_assert(self.father && self.son);
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i) = R;
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::operator_plus_assign(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(R.father && R.son);
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i) += R.operator[](i);
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::operator_plus_assign(t_self &self, real R)
    {
        DNDS_assert(self.father && self.son);
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i).array() += R;
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::operator_plus_assign(t_self &self, const Eigen::Ref<const t_element_mat> &R)
    {
        DNDS_assert(self.father && self.son);
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i) += R;
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::operator_minus_assign(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(R.father && R.son);
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i) -= R.operator[](i);
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::operator_mult_assign(t_self &self, real R)
    {
        DNDS_assert(self.father && self.son);
        if (R == 1)
            return;
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i) *= R;
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::operator_mult_assign_scalar_arr(t_self &self, const ArrayDof<1, 1> &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(R.father && R.son);
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i).array() *= R[i](0);
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::operator_mult_assign(t_self &self, const Eigen::Ref<const t_element_mat> &R)
    {
        DNDS_assert(self.father && self.son);
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i).array() *= R.array();
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::operator_mult_assign(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(R.father && R.son);
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i).array() *= R.operator[](i).array();
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::operator_div_assign(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(R.father && R.son);
        index iTop = self.Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index i = 0; i < iTop; i++)
            self.operator[](i).array() /= R.operator[](i).array();
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::operator_assign(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(R.father && R.son);
        // TODO: OMP
        // for (index i = 0; i < this->Size(); i++)
        //     this->operator[](i) = R.operator[](i);
        DNDS_assert(R.father->RawDataVector().size() == self.father->RawDataVector().size());
        DNDS_assert(R.son->RawDataVector().size() == self.son->RawDataVector().size());

        // #if defined(DNDS_DIST_MT_USE_OMP)
        //             {
        //                 size_t part_size = R.father->RawDataVector().size() / omp_get_max_threads();
        // #pragma omp parallel for schedule(static)
        //                 for (int iT = 0; iT < omp_get_max_threads(); iT++)
        //                     std::copy(R.father->RawDataVector().begin() + part_size * iT,
        //                               (iT == omp_get_max_threads() - 1)
        //                                   ? R.father->RawDataVector().end()
        //                                   : R.father->RawDataVector().begin() + part_size * (iT + 1),
        //                               this->father->RawDataVector().begin() + part_size * iT);
        //             }
        //             {
        //                 size_t part_size = R.son->RawDataVector().size() / omp_get_max_threads();
        // #pragma omp parallel for schedule(static)
        //                 for (int iT = 0; iT < omp_get_max_threads(); iT++)
        //                     std::copy(R.son->RawDataVector().begin() + part_size * iT,
        //                               (iT == omp_get_max_threads() - 1)
        //                                   ? R.son->RawDataVector().end()
        //                                   : R.son->RawDataVector().begin() + part_size * (iT + 1),
        //                               this->son->RawDataVector().begin() + part_size * iT);
        //             }
        // #else
        std::copy(R.father->RawDataVector().begin(), R.father->RawDataVector().end(), self.father->RawDataVector().begin());
        std::copy(R.son->RawDataVector().begin(), R.son->RawDataVector().end(), self.son->RawDataVector().begin());
        // #endif
    }

    template <int n_m, int n_n>
    void ArrayDofOp<DeviceBackend::Host, n_m, n_n>::addTo(t_self &self, const t_self &R, real r)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(R.father && R.son);
        // for (index i = 0; i < this->Size(); i++)
        //     this->operator[](i) += R.operator[](i) * r;
        DNDS_assert(R.father->RawDataVector().size() == self.father->RawDataVector().size());
        auto &RVF = R.father->RawDataVector();
        auto &TVF = self.father->RawDataVector();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < RVF.size(); i++)
            TVF[i] += r * RVF[i];
        DNDS_assert(R.son->RawDataVector().size() == self.son->RawDataVector().size());
        auto &RVS = R.son->RawDataVector();
        auto &TVS = self.son->RawDataVector();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < RVS.size(); i++)
            TVS[i] += r * RVS[i];
    }

    template <int n_m, int n_n>
    real ArrayDofOp<DeviceBackend::Host, n_m, n_n>::norm2(t_self &self)
    {
        DNDS_assert(self.father && self.son);
        real sqrSum{0}, sqrSumAll{0};
        index iTop = self.father->Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) reduction(+ : sqrSum)
#endif
        for (index i = 0; i < iTop; i++) //*note that only father is included
            sqrSum += self.father->operator[](i).squaredNorm();
        MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
        return std::sqrt(sqrSumAll);
    }

    template <int n_m, int n_n>
    real ArrayDofOp<DeviceBackend::Host, n_m, n_n>::norm2(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        real sqrSum{0}, sqrSumAll{0};
        index iTop = self.father->Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) reduction(+ : sqrSum)
#endif
        for (index i = 0; i < iTop; i++) //*note that only father is included
            sqrSum += (self.father->operator[](i) - R.father->operator[](i)).squaredNorm();
        MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
        return std::sqrt(sqrSumAll);
    }

    template <int n_m, int n_n>
    real ArrayDofOp<DeviceBackend::Host, n_m, n_n>::reduction(t_self &self, const std::string &op)
    {
        DNDS_assert(self.father && self.son);
        real sqrSum{0}, sqrSumAll{0};
        index iTop = self.father->DataSize();
        if (op == "min")
        {
            sqrSum = sqrSumAll = veryLargeReal;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) reduction(min : sqrSum)
#endif
            for (index i = 0; i < iTop; i++) //*note that only father is included
                sqrSum = std::min(sqrSum, self.father->data()[i]);
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_MIN, self.father->getMPI().comm);
        }
        else if (op == "max")
        {
            sqrSum = sqrSumAll = -veryLargeReal;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) reduction(max : sqrSum)
#endif
            for (index i = 0; i < iTop; i++) //*note that only father is included
                sqrSum = std::max(sqrSum, self.father->data()[i]);
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_MAX, self.father->getMPI().comm);
        }
        else if (op == "sum")
        {
            sqrSum = sqrSumAll = 0.0;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) reduction(+ : sqrSum)
#endif
            for (index i = 0; i < iTop; i++) //*note that only father is included
                sqrSum = sqrSum + self.father->data()[i];
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        }
        else
            DNDS_assert_info(false, op);
        // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
        return std::sqrt(sqrSumAll);
    }

    template <int n_m, int n_n>
    typename ArrayDofOp<DeviceBackend::Host, n_m, n_n>::t_element_mat
    ArrayDofOp<DeviceBackend::Host, n_m, n_n>::componentWiseNorm1(t_self &self)
    {
        DNDS_assert(self.father && self.son);
        t_element_mat minLocal, min;
        //! let it fail if size not compatible
        if (self.father->Size() || (n_m >= 0 && n_n >= 0) || (self.son->Size() == 0))
            minLocal.resize(self.father->MatRowSize(0), self.father->MatColSize(0));
        else if (self.son->Size())
            minLocal.resize(self.son->MatRowSize(0), self.son->MatColSize(0));
        minLocal.setConstant(0);
        min = minLocal;
        index iTop = self.father->Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(EigenVecAdd:t_element_mat : omp_out += omp_in) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(EigenVecAdd : minLocal)
#endif
        for (index i = 0; i < iTop; i++) //*note that only father is included
            minLocal += (self.operator[](i).array().abs()).matrix();
        MPI::Allreduce(minLocal.data(), min.data(), minLocal.size(), DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        return min;
    }

    template <int n_m, int n_n>
    typename ArrayDofOp<DeviceBackend::Host, n_m, n_n>::t_element_mat
    ArrayDofOp<DeviceBackend::Host, n_m, n_n>::componentWiseNorm1(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        t_element_mat minLocal, min;
        //! let it fail if size not compatible
        if (self.father->Size() || (n_m >= 0 && n_n >= 0))
            minLocal.resize(self.father->MatRowSize(0), self.father->MatColSize(0));
        else if (self.son->Size())
            minLocal.resize(self.son->MatRowSize(0), self.son->MatColSize(0));
        minLocal.setConstant(0);
        min = minLocal;
        index iTop = self.father->Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(EigenVecAdd:t_element_mat : omp_out += omp_in) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(EigenVecAdd : minLocal)
#endif
        for (index i = 0; i < iTop; i++) //*note that only father is included
            minLocal += ((self.operator[](i) - R.operator[](i)).array().abs()).matrix();
        MPI::Allreduce(minLocal.data(), min.data(), minLocal.size(), DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        return min;
    }

    template <int n_m, int n_n>
    real ArrayDofOp<DeviceBackend::Host, n_m, n_n>::dot(t_self &self, const t_self &R)
    {
        DNDS_assert(self.father && self.son);
        DNDS_assert(R.father && R.son);
        real sqrSum{0}, sqrSumAll;
        index iTop = self.father->Size();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) reduction(+ : sqrSum)
#endif
        for (index i = 0; i < iTop; i++) //*note that only father is included
            sqrSum += (self.operator[](i).array() * (R.operator[](i)).array()).sum();
        MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, self.father->getMPI().comm);
        return sqrSumAll;
    }
}