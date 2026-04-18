#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/EigenUtil.hpp"
#include "DNDS/ConfigEnum.hpp"
#include "CFV/VRDefines.hpp"
#include "DNDS/Errors.hpp"

namespace DNDS::Euler
{
#define DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS                              \
    static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);   \
    static const auto Seq12 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>);    \
    static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);       \
    static const auto Seq23 = Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>);        \
    static const auto Seq234 = Eigen::seq(Eigen::fix<2>, Eigen::fix<dim + 1>);   \
    static const auto Seq34 = Eigen::seq(Eigen::fix<3>, Eigen::fix<dim + 1>);    \
    static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>); \
    static const auto SeqG012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<gDim - 1>); \
    static const auto SeqI52Last = Eigen::seq(Eigen::fix<I4 + 1>, EigenLast);    \
    static const auto I4 = dim + 1;

    template <int nVarsFixed>
    class ArrayDOFV : public CFV::tUDof<nVarsFixed>
    {
    public:
        using t_self = ArrayDOFV<nVarsFixed>;
        using t_base = CFV::tUDof<nVarsFixed>;
        using t_element_mat = typename t_base::t_element_mat;

        void setConstant(real R)
        {
            this->t_base::setConstant(R);
        }
        void setConstant(const Eigen::Ref<const t_element_mat> &R )
        {
            this->t_base::setConstant(R);
        }
        void operator+=(const t_self &R)
        {
            this->t_base::operator+=(R);
        }
        void operator-=(const t_self &R)
        {
            this->t_base::operator-=(R);
        }
        void operator*=(real R)
        {
            this->t_base::operator*=(R);
        }
        void operator=(const t_self &R)
        {
            this->t_base::operator=(R);
        }

        void addTo(const t_self &R, real r)
        {
            this->t_base::addTo(R, r);
        }

        void operator*=(const std::vector<real> &R)
        {
            DNDS_assert(R.size() >= this->father->Size());
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->father->Size(); i++)
                this->operator[](i) *= R[i];
        }
        template <int nVarsFixed_T = nVarsFixed>
        std::enable_if_t<!(nVarsFixed_T == 1)>
        operator*=(const ArrayDOFV<1> &R)
        {
            this->t_base::operator*=(R);
        }

        void operator+=(const Eigen::Vector<real, nVarsFixed> &R)
        {
            this->t_base::operator+=(R);
        }

        void operator+=(real R)
        {
            this->t_base::operator+=(R);
        }

        void operator*=(const Eigen::Vector<real, nVarsFixed> &R)
        {
            this->t_base::operator*=(R);
        }

        void operator*=(const t_self &R)
        {
            this->t_base::operator*=(R);
        }

        void operator/=(const t_self &R)
        {
            this->t_base::operator/=(R);
        }

        real norm2()
        {
            return this->t_base::norm2();
        }

        Eigen::Vector<real, nVarsFixed> componentWiseNorm1()
        {
            return this->t_base::componentWiseNorm1();
        }

        Eigen::Vector<real, nVarsFixed> min()
        {
            Eigen::Vector<real, nVarsFixed> minLocal, min;
            minLocal.resize(this->RowSize());
            minLocal.setConstant(veryLargeReal);
            min = minLocal;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(EigenVecMin : Eigen::Vector<real, nVarsFixed> : omp_out = omp_out.array().min(omp_in.array())) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(EigenVecMin : minLocal)
#endif
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                minLocal = minLocal.array().min(this->operator[](i).array());
            MPI::Allreduce(minLocal.data(), min.data(), minLocal.size(), DNDS_MPI_REAL, MPI_MIN, this->father->getMPI().comm);
            return min;
        }

        real dot(const t_self &R)
        {
            return this->t_base::dot(R);
        }

        template <class TMultL, class TMultR>
        real dot(const t_self &R, TMultL &&mL, TMultR &&mR)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
            real sqrSum{0}, sqrSumAll;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) reduction(+ : sqrSum)
#endif
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += (this->operator[](i).array() * mL).matrix().dot((R.operator[](i).array() * mR).matrix());
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return sqrSumAll;
        }

        template <class TR>
        void setMaxWith(TR R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() = this->operator[](i).array().max(R);
        }
    };

    ///@todo://TODO add operators
    template <int nVarsFixed>
    class ArrayRECV : public CFV::tURec<nVarsFixed>
    {
    public:
        using t_self = ArrayRECV<nVarsFixed>;
        using t_base = CFV::tURec<nVarsFixed>;
        void setConstant(real R)
        {
            this->t_base::setConstant(R);
        }
        template <class TR>
        void setConstant(const TR &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) = R;
        }
        void operator+=(const t_self &R)
        {
            this->t_base::operator+=(R);
        }
        void operator-=(const t_self &R)
        {
            this->t_base::operator-=(R);
        }
        void operator*=(real R)
        {
            this->t_base::operator*=(R);
        }
        void operator*=(const std::vector<real> &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
            DNDS_assert(R.size() >= this->father->Size());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->father->Size(); i++)
                this->operator[](i) *= R[i];
        }
        void operator*=(const ArrayDOFV<1> &R)
        {
            this->t_base::operator*=(R);
        }
        void operator*=(const Eigen::Array<real, 1, nVarsFixed> &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array().rowwise() *= R;
        }
        void operator=(const t_self &R)
        {
            this->t_base::operator=(R);
        }

        void addTo(const t_self &R, const Eigen::Array<real, 1, nVarsFixed> &r)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() += R.operator[](i).array().rowwise() * r;
        }

        void addTo(const t_self &R, real r)
        {
            this->t_base::addTo(R, r);
        }

        real norm2()
        {
            return this->t_base::norm2();
        }

        real dot(const t_self &R)
        {
            return this->t_base::dot(R);
        }

        auto dotV(const t_self &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
            Eigen::RowVector<real, nVarsFixed> sqrSum, sqrSumAll;
            sqrSum.resize(this->father->MatColSize());
            sqrSumAll.resizeLike(sqrSum);
            sqrSum.setZero();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(EigenVecSum : Eigen::RowVector<real, nVarsFixed> : omp_out += omp_in) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(EigenVecSum : sqrSum)
#endif
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += (this->operator[](i).array() * R.operator[](i).array()).colwise().sum().matrix();
            MPI::Allreduce(sqrSum.data(), sqrSumAll.data(), sqrSum.size(), DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return sqrSumAll;
        }
    };

    template <int nVarsFixed, int gDim>
    class ArrayGRADV : public CFV::tUGrad<nVarsFixed, gDim>
    {
    public:
        using t_self = ArrayGRADV<nVarsFixed, gDim>;
        using t_base = CFV::tUGrad<nVarsFixed, gDim>;
        void setConstant(real R)
        {
            this->t_base::setConstant(R);
        }
        template <class TR>
        void setConstant(const TR &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) = R;
        }

        void operator+=(t_self &R)
        {
            this->t_base::operator+=(R);
        }
        void operator-=(t_self &R)
        {
            this->t_base::operator-=(R);
        }
        void operator*=(real R)
        {
            this->t_base::operator*=(R);
        }
        void operator*=(std::vector<real> &R)
        {
            DNDS_assert(R.size() >= this->father->Size());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->father->Size(); i++)
                this->operator[](i) *= R[i];
        }
        void operator*=(ArrayDOFV<1> &R)
        {
            this->t_base::operator*=(R);
        }
        void operator*=(const Eigen::Array<real, 1, nVarsFixed> &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array().rowwise() *= R;
        }
        void operator=(t_self &R)
        {
            this->t_base::operator=(R);
        }
    };

    template <int nVarsFixed>
    class JacobianValue
    {
    public:
        enum Type
        {
            Diagonal = 0,
            DiagonalBlock = 1,
            Full = 2,
        };
        ArrayDOFV<nVarsFixed> diag, diagInv;
        ArrayEigenMatrix<nVarsFixed, nVarsFixed> diagBlock, diagBlockInv;
        ArrayRECV<nVarsFixed> offDiagBlock;

        void SetDiagonal(ArrayDOFV<nVarsFixed> &uDof)
        {
            type = Diagonal;
            // todo ! allocate square blocks!
        }

        void SetDiagonalBlock(ArrayDOFV<nVarsFixed> &uDof)
        {
            type = DiagonalBlock;
            // todo ! allocate square blocks!
        }

        void SetFull(ArrayDOFV<nVarsFixed> &uDof, Geom::tAdjPair &cell2cell)
        {
            type = Full;
            // todo ! allocate with adjacency!
        }

        void InverseDiag()
        {
            // todo get inverse!
        }

    private:
        Type type = Diagonal;
    };

    enum EulerModel
    {
        NS = 0,
        NS_SA = 1,
        NS_2D = 2,
        NS_3D = 3,
        NS_SA_3D = 4,
        NS_2EQ = 5,
        NS_2EQ_3D = 6,
        NS_EX = 101,
        NS_EX_3D = 102,
    };

    enum RANSModel
    {
        RANS_Unknown = 0,
        RANS_None,
        RANS_SA,
        RANS_KOWilcox,
        RANS_KOSST,
        RANS_RKE,
    };

    DNDS_DEFINE_ENUM_JSON(
        RANSModel,
        {
            {RANS_Unknown, nullptr},
            {RANS_None, "RANS_None"},
            {RANS_SA, "RANS_SA"},
            {RANS_KOWilcox, "RANS_KOWilcox"},
            {RANS_KOSST, "RANS_KOSST"},
            {RANS_RKE, "RANS_RKE"},
        })

    constexpr static inline int getnVarsFixed(const EulerModel model)
    {
        if (model == NS || model == NS_3D)
            return 5;
        else if (model == NS_SA)
            return 6;
        else if (model == NS_SA_3D)
            return 6;
        else if (model == NS_2D)
            return 4;
        else if (model == NS_2EQ || model == NS_2EQ_3D)
            return 7;
        return Eigen::Dynamic;
    }

    constexpr static inline int getNVars(EulerModel model)
    {
        int nVars = getnVarsFixed(model);
        if (nVars < 0)
        {
            if (model == NS || model == NS_3D)
                return 5;
            else if (model == NS_SA)
                return 6;
            else if (model == NS_SA_3D)
                return 6;
            else if (model == NS_2D)
                return 4;
            else if (model == NS_2EQ || model == NS_2EQ_3D)
                return 7;
            // *** handle variable nVars
        }
        return nVars;
    }

    constexpr static inline int getDim_Fixed(const EulerModel model)
    {
        if (model == NS || model == NS_3D)
            return 3;
        else if (model == NS_SA)
            return 3;
        else if (model == NS_SA_3D)
            return 3;
        else if (model == NS_2D)
            return 2;
        else if (model == NS_2EQ || model == NS_2EQ_3D)
            return 3;
        else if (model == NS_EX || model == NS_EX_3D)
            return 3;
        return Eigen::Dynamic;
    }

    constexpr static inline int getGeomDim_Fixed(const EulerModel model)
    {
        if (model == NS)
            return 2;
        else if (model == NS_SA)
            return 2;
        else if (model == NS_2D)
            return 2;
        else if (model == NS_3D)
            return 3;
        else if (model == NS_SA_3D)
            return 3;
        else if (model == NS_2EQ)
            return 2;
        else if (model == NS_2EQ_3D)
            return 3;
        else if (model == NS_EX)
            return 2;
        else if (model == NS_EX_3D)
            return 3;
        return Eigen::Dynamic;
    }

    // constexpr static inline bool ifFixedNvars(EulerModel model)
    // {
    //     return (
    //         model == NS ||
    //         model == NS_SA);
    // } // use +/- is ok

    /**
     * @brief Compile-time traits for EulerModel variants.
     *
     * Replaces repeated `if constexpr (model == NS_SA || model == NS_SA_3D)` patterns
     * with named boolean traits. All members are static constexpr.
     *
     * Future models (reactive, multi-species) should add traits here rather than
     * adding new if-constexpr chains.
     */
    template <EulerModel model>
    struct EulerModelTraits
    {
        /// Number of fixed conservative variables (Eigen::Dynamic for NS_EX).
        static constexpr int nVarsFixed = getnVarsFixed(model);
        /// Physics spatial dimension (2 for NS_2D, 3 for all others).
        static constexpr int dim = getDim_Fixed(model);
        /// Geometry (mesh) spatial dimension.
        static constexpr int gDim = getGeomDim_Fixed(model);

        /// True for Spalart-Allmaras models (NS_SA, NS_SA_3D).
        static constexpr bool hasSA = (model == NS_SA || model == NS_SA_3D);
        /// True for 2-equation RANS models (NS_2EQ, NS_2EQ_3D).
        static constexpr bool has2EQ = (model == NS_2EQ || model == NS_2EQ_3D);
        /// True for any RANS turbulence model (SA or 2-equation).
        static constexpr bool hasRANS = hasSA || has2EQ;
        /// Number of extra RANS transport variables (0, 1, or 2).
        static constexpr int nRANSVars = hasSA ? 1 : (has2EQ ? 2 : 0);

        /// True for extended/dynamic models (NS_EX, NS_EX_3D).
        static constexpr bool isExtended = (model == NS_EX || model == NS_EX_3D);
        /// True for plain NS without turbulence or extensions.
        static constexpr bool isPlainNS = !hasRANS && !isExtended;
        /// True for 2D geometry models.
        static constexpr bool isGeom2D = (gDim == 2);
        /// True for 3D geometry models.
        static constexpr bool isGeom3D = (gDim == 3);
    };

    template <int nVarsFixed, int mul>
    constexpr static inline int nvarsFixedMultiply()
    {
        return nVarsFixed != Eigen::Dynamic ? nVarsFixed * mul : Eigen::Dynamic;
    }
}