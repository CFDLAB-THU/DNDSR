#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/EigenUtil.hpp"
#include "CFV/VRDefines.hpp"

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
    static const auto I4 = dim + 1;

    template <int nVarsFixed>
    class ArrayDOFV : public CFV::tUDof<nVarsFixed>
    {
    public:
        using t_self = ArrayDOFV<nVarsFixed>;
        void setConstant(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).setConstant(R);
        }
        template <class TR>
        void setConstant(const TR &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) = R;
        }
        void operator+=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) += R.operator[](i);
        }
        void operator-=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) -= R.operator[](i);
        }
        void operator*=(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) *= R;
        }
        void operator=(t_self &R)
        {
            // for (index i = 0; i < this->Size(); i++)
            //     this->operator[](i) = R.operator[](i);
            DNDS_assert(R.father->RawDataVector().size() == this->father->RawDataVector().size());
            std::copy(R.father->RawDataVector().begin(), R.father->RawDataVector().end(), this->father->RawDataVector().begin());
            DNDS_assert(R.son->RawDataVector().size() == this->son->RawDataVector().size());
            std::copy(R.son->RawDataVector().begin(), R.son->RawDataVector().end(), this->son->RawDataVector().begin());
        }

        void addTo(t_self &R, real r)
        {
            // for (index i = 0; i < this->Size(); i++)
            //     this->operator[](i) += R.operator[](i) * r;
            DNDS_assert(R.father->RawDataVector().size() == this->father->RawDataVector().size());
            auto &RVF = R.father->RawDataVector();
            auto &TVF = this->father->RawDataVector();
            for (size_t i = 0; i < RVF.size(); i++)
                TVF[i] += r * RVF[i];
            DNDS_assert(R.son->RawDataVector().size() == this->son->RawDataVector().size());
            auto &RVS = R.son->RawDataVector();
            auto &TVS = this->son->RawDataVector();
            for (size_t i = 0; i < RVS.size(); i++)
                TVS[i] += r * RVS[i];
        }

        void operator*=(std::vector<real> &R)
        {
            DNDS_assert(R.size() >= this->father->Size());
            for (index i = 0; i < this->father->Size(); i++)
                this->operator[](i) *= R[i];
        }

        void operator*=(std::conditional_t<nVarsFixed == 1, ArrayDOFV<2>, ArrayDOFV<1>> &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() *= R[i](0);
        }

        void operator+=(const Eigen::Vector<real, nVarsFixed> &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) += R;
        }

        void operator+=(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() += R;
        }

        void operator*=(const Eigen::Vector<real, nVarsFixed> &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() *= R.array();
        }

        void operator*=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() *= R.operator[](i).array();
        }

        void operator/=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() /= R.operator[](i).array();
        }

        void setAbs()
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() = this->operator[](i).array().abs();
        }

        template <class TR>
        void setMaxWith(TR R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() = this->operator[](i).array().max(R);
        }

        template <class TR>
        void setMinWith(TR R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() = this->operator[](i).array().min(R);
        }

        Eigen::Vector<real, nVarsFixed> normInc()
        {
            Eigen::Vector<real, nVarsFixed> ret, retAll;
            ret.resize(this->RowSize());
            retAll.resize(this->RowSize());
            ret.setZero();
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                ret += this->operator[](i).array().abs();
            MPI::Allreduce(ret.data(), retAll.data(), this->RowSize(), DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return retAll;
        }

        real norm2()
        {
            real sqrSum{0}, sqrSumAll{0};
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += this->operator[](i).squaredNorm();
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
            return std::sqrt(sqrSumAll);
        }

        Eigen::Vector<real, nVarsFixed> componentWiseNorm1()
        {
            Eigen::Vector<real, nVarsFixed> minLocal, min;
            minLocal.resize(this->RowSize());
            minLocal.setConstant(0);
            min = minLocal;
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                minLocal += (this->operator[](i).array().abs()).matrix();
            MPI::Allreduce(minLocal.data(), min.data(), minLocal.size(), DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return min;
        }

        Eigen::Vector<real, nVarsFixed> min()
        {
            Eigen::Vector<real, nVarsFixed> minLocal, min;
            minLocal.resize(this->RowSize());
            minLocal.setConstant(veryLargeReal);
            min = minLocal;
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                minLocal = minLocal.array().min(this->operator[](i).array());
            MPI::Allreduce(minLocal.data(), min.data(), minLocal.size(), DNDS_MPI_REAL, MPI_MIN, this->father->getMPI().comm);
            return min;
        }

        real dot(const t_self &R)
        {
            real sqrSum{0}, sqrSumAll;
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += this->operator[](i).dot(R.operator[](i));
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return sqrSumAll;
        }

        template <class TMultL, class TMultR>
        real dot(const t_self &R, TMultL &&mL, TMultR &&mR)
        {
            real sqrSum{0}, sqrSumAll;
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += (this->operator[](i).array() * mL).matrix().dot((R.operator[](i).array() * mR).matrix());
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return sqrSumAll;
        }
    };

    ///@todo://TODO add operators
    template <int nVarsFixed>
    class ArrayRECV : public CFV::tURec<nVarsFixed>
    {
    public:
        using t_self = ArrayRECV<nVarsFixed>;
        void setConstant(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).setConstant(R);
        }
        template <class TR>
        void setConstant(const TR &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) = R;
        }
        void operator+=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) += R.operator[](i);
        }
        void operator-=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) -= R.operator[](i);
        }
        void operator*=(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) *= R;
        }
        void operator*=(std::vector<real> &R)
        {
            DNDS_assert(R.size() >= this->father->Size());
            for (index i = 0; i < this->father->Size(); i++)
                this->operator[](i) *= R[i];
        }
        void operator*=(ArrayDOFV<1> &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) *= R[i](0);
        }
        void operator*=(const Eigen::Array<real, 1, nVarsFixed> &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array().rowwise() *= R;
        }
        void operator=(t_self &R)
        {
            // for (index i = 0; i < this->Size(); i++)
            //     this->operator[](i) = R.operator[](i);
            DNDS_assert(R.father->RawDataVector().size() == this->father->RawDataVector().size());
            std::copy(R.father->RawDataVector().begin(), R.father->RawDataVector().end(), this->father->RawDataVector().begin());
            DNDS_assert(R.son->RawDataVector().size() == this->son->RawDataVector().size());
            std::copy(R.son->RawDataVector().begin(), R.son->RawDataVector().end(), this->son->RawDataVector().begin());
        }

        void addTo(t_self &R, const Eigen::Array<real, 1, nVarsFixed> &r)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() += R.operator[](i).array().rowwise() * r;
        }

        void addTo(t_self &R, real r)
        {
            // for (index i = 0; i < this->Size(); i++)
            //     this->operator[](i) += R.operator[](i) * r;
            DNDS_assert(R.father->RawDataVector().size() == this->father->RawDataVector().size());
            auto &RVF = R.father->RawDataVector();
            auto &TVF = this->father->RawDataVector();
            for (size_t i = 0; i < RVF.size(); i++)
                TVF[i] += r * RVF[i];
            DNDS_assert(R.son->RawDataVector().size() == this->son->RawDataVector().size());
            auto &RVS = R.son->RawDataVector();
            auto &TVS = this->son->RawDataVector();
            for (size_t i = 0; i < RVS.size(); i++)
                TVS[i] += r * RVS[i];
        }

        real norm2()
        {
            real sqrSum{0}, sqrSumAll{0};
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += this->operator[](i).squaredNorm();
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
            return std::sqrt(sqrSumAll);
        }

        real dot(const t_self &R)
        {
            real sqrSum{0}, sqrSumAll;
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += (this->operator[](i).array() * R.operator[](i).array()).sum();
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return sqrSumAll;
        }

        auto dotV(const t_self &R)
        {
            Eigen::RowVector<real, nVarsFixed> sqrSum, sqrSumAll;
            sqrSum.resize(this->father->MatColSize());
            sqrSumAll.resizeLike(sqrSum);
            sqrSum.setZero();
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
        void setConstant(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).setConstant(R);
        }
        template <class TR>
        void setConstant(const TR &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) = R;
        }

        void operator+=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) += R.operator[](i);
        }
        void operator-=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) -= R.operator[](i);
        }
        void operator*=(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) *= R;
        }
        void operator*=(std::vector<real> &R)
        {
            DNDS_assert(R.size() >= this->father->Size());
            for (index i = 0; i < this->father->Size(); i++)
                this->operator[](i) *= R[i];
        }
        void operator*=(ArrayDOFV<1> &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) *= R[i](0);
        }
        void operator*=(const Eigen::Array<real, 1, nVarsFixed> &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array().rowwise() *= R;
        }
        void operator=(t_self &R)
        {
            // for (index i = 0; i < this->Size(); i++)
            //     this->operator[](i) = R.operator[](i);
            DNDS_assert(R.father->RawDataVector().size() == this->father->RawDataVector().size());
            std::copy(R.father->RawDataVector().begin(), R.father->RawDataVector().end(), this->father->RawDataVector().begin());
            DNDS_assert(R.son->RawDataVector().size() == this->son->RawDataVector().size());
            std::copy(R.son->RawDataVector().begin(), R.son->RawDataVector().end(), this->son->RawDataVector().begin());
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

    NLOHMANN_JSON_SERIALIZE_ENUM(
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
        return Eigen::Dynamic;
    }

    // constexpr static inline bool ifFixedNvars(EulerModel model)
    // {
    //     return (
    //         model == NS ||
    //         model == NS_SA);
    // } // use +/- is ok

    template <int nVarsFixed, int mul>
    constexpr static inline int nvarsFixedMultiply()
    {
        return nVarsFixed != Eigen::Dynamic ? nVarsFixed * mul : Eigen::Dynamic;
    }
}