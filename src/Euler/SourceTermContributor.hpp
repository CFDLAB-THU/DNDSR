/**
 * @file SourceTermContributor.hpp
 * @brief Composable, non-virtual source term contributors for the Euler/NS implicit solver.
 *
 * Uses variant-based dispatch. Each contributor is a plain struct holding its config.
 * For fixed-size EulerModels (NS, NS_SA, etc.) the existing if-constexpr path in
 * source() is preserved unchanged — contributors are only used when model == NS_EX
 * or NS_EX_3D (i.e. Traits::isExtended).
 */

#pragma once

#include "Euler.hpp"
#include "Gas.hpp"
#include "RANS_ke.hpp"
#include "Chemistry/ChemicalSource.hpp"

#include <variant>
#include <vector>

namespace DNDS::Euler
{

    // For readability: map EulerModel to matrix types from Euler.hpp
    template <EulerModel M>
    using SrcTU = TU<M>;
    template <EulerModel M>
    using SrcTJac = TJacobianU<M>;
    template <EulerModel M>
    using SrcTDiffU = TDiffU<M>;

    /// Static dim for EX models (both NS_EX and NS_EX_3D have dim=3).
    constexpr int ExDim = 3;

    /**
     * @brief Per-quadrature-point auxiliary data needed by source term contributors.
     */
    struct SourceCellAux
    {
        real dWallC = 0;
        real hMax = 0;
        real muf = 0;
        real T = 300;
        real p = 101325;     // code pressure
        real pPhys = 101325; // physical pressure [Pa] for Cantera
        real gamma = 1.4;    // EOS gamma at this point (from phys_)
    };

    // ============================================================================
    // Shared free functions
    // ============================================================================

    template <int dim, class TMassForce, class TRet, class TJac>
    inline void evalSourceBodyForce(const TMassForce &massForce, TRet &ret, TJac &jac,
                                    const TRet &U, int Mode)
    {
        auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        auto I4 = dim + 1;
        if (Mode == 0)
        {
            ret(Seq123) += massForce(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>)) * U(0);
            ret(I4) += massForce(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>)).dot(U(Seq123));
        }
        if (Mode == 2)
            jac(I4, Seq123) -= massForce(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>));
    }

    template <int dim, class TFrame, class TRet, class TJac>
    inline void evalSourceRotatingFrame(const TFrame &frame, const Geom::tPoint &pPhy,
                                        TRet &ret, TJac &jac, const TRet &U, int Mode)
    {
        using TVec = Eigen::VectorFMTSafe<real, dim>;
        using TMat = Eigen::MatrixFMTSafe<real, dim, dim>;
        auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        auto I4 = dim + 1;
        Geom::tPoint radi = pPhy - frame.center;
        Geom::tPoint radiR = radi - frame.axis * (frame.axis.dot(radi));
        TVec mvolForce = (radiR * sqr(frame.Omega()) * U(0))(Seq012);
        mvolForce += -2.0 * frame.vOmega().cross(Geom::ToThreeDim<dim>(U(Seq123)))(Seq012);
        if (Mode == 0)
        {
            ret(Seq123) += mvolForce;
            ret(I4) += mvolForce.dot(U(Seq123)) / U(0);
        }
        if (Mode == 2)
        {
            TMat dmvolForceDrhov = Geom::CrossVecToMat(-2 * frame.vOmega())(Seq012, Seq012);
            jac(Seq123, Seq123) -= dmvolForceDrhov;
            jac(I4, Seq123) -= mvolForce + dmvolForceDrhov.transpose() * U(Seq123) / U(0);
            jac(I4, 0) -= -mvolForce.dot(U(Seq123)) / sqr(U(0));
        }
    }

    template <class TRet>
    inline void evalSourceAxisymmetric(real gamma, const Geom::tPoint &pPhy,
                                       TRet &ret, const TRet &U, int Mode)
    {
        auto I4 = 4; // dim=3 -> I4=4
        if (Mode == 0)
        {
            TRet uPrim;
            uPrim.resizeLike(U);
            Gas::IdealGasThermalConservative2Primitive(U, uPrim, gamma);
            ret(2) += uPrim(I4) / std::max(verySmallReal, pPhy(1));
        }
    }

    // ============================================================================
    // Contributor structs — evaluate() is templated on the settings types so it
    // works with any model (NS_EX, NS_EX_3D, etc.) without nested-type mismatch.
    // ============================================================================

    struct BodyForceContributor
    {
        Eigen::Vector<real, 3> force{0, 0, 0};

        template <class TRet, class TJac, class TDerivedU, class TGasProp>
        void evaluate(TRet &ret, TJac &jac, const TRet &U, const TDerivedU &,
                      const Geom::tPoint &, const SourceCellAux &,
                      const TGasProp &, index, index, int Mode) const
        {
            if (force.isZero(0))
                return;
            evalSourceBodyForce<ExDim>(force, ret, jac, U, Mode);
        }
    };

    struct RotatingFrameContributor
    {
        // Store only the data we need; the exact FrameConstRotation type is
        // model-dependent, so we use a generic config subset.
        bool enabled = false;
        Geom::tPoint axis{0, 0, 1};
        Geom::tPoint center{0, 0, 0};
        real rpm = 0;
        real Omega() const { return rpm * (2 * pi / 60.); }
        Geom::tPoint vOmega() const { return axis * Omega(); }

        RotatingFrameContributor() = default;
        template <class TFrame>
        explicit RotatingFrameContributor(const TFrame &f)
            : enabled(f.enabled), axis(f.axis), center(f.center), rpm(f.rpm) {}

        template <class TRet, class TJac, class TDerivedU, class TGasProp>
        void evaluate(TRet &ret, TJac &jac, const TRet &U, const TDerivedU &,
                      const Geom::tPoint &pPhy, const SourceCellAux &aux,
                      const TGasProp &, index, index, int Mode) const
        {
            if (!enabled)
                return;
            evalSourceRotatingFrame<ExDim>(*this, pPhy, ret, jac, U, Mode);
        }
    };

    struct AxisymmetricContributor
    {
        bool active = false;

        template <class TRet, class TJac, class TDerivedU, class TGasProp>
        void evaluate(TRet &ret, TJac &, const TRet &U, const TDerivedU &,
                      const Geom::tPoint &pPhy, const SourceCellAux &aux,
                      const TGasProp &, index, index, int Mode) const
        {
            if (!active)
                return;
            evalSourceAxisymmetric(aux.gamma, pPhy, ret, U, Mode);
        }
    };

    struct SASourceContributor
    {
        real muGas = 1;
        real SADESScale = veryLargeReal;
        int SADESMode = 1;
        int SAVersion = 0;
        int ransSARotCorrection = 1;

        template <class TRet, class TJac, class TDerivedU, class TGasProp>
        void evaluate(TRet &ret, TJac &jac, const TRet &U, const TDerivedU &GradU,
                      const Geom::tPoint &, const SourceCellAux &aux,
                      const TGasProp &, index iCell, index, int Mode) const
        {
            TRet retInc;
            retInc.setZero(U.size());
            real d = std::min(aux.dWallC, std::pow(veryLargeReal, 1. / 6.));
            real lLES = aux.hMax * SADESScale;
            real cWall = SADESScale > 100.0 ? 1.0 : 0.15;
            lLES = std::min(lLES, std::max({d * cWall, aux.hMax * cWall}));
            auto call = [&](int mode)
            {
                RANS::GetSource_SA<ExDim>(U, GradU, muGas, aux.muf, aux.gamma,
                                          d, lLES, aux.hMax, SADESMode,
                                          retInc, ransSARotCorrection, mode, SAVersion);
            };
            if (Mode == 0)
                call(0);
            else if (Mode == 1)
                call(1);
            else if (Mode == 2)
            {
                call(1);
                jac += retInc.asDiagonal();
            }
            ret += retInc;
        }
    };

    struct SSTSourceContributor
    {
        real muGas = 1;
        real SADESScale = veryLargeReal;

        template <class TRet, class TJac, class TDerivedU, class TGasProp>
        void evaluate(TRet &ret, TJac &jac, const TRet &U, const TDerivedU &GradU,
                      const Geom::tPoint &, const SourceCellAux &aux,
                      const TGasProp &, index iCell, index, int Mode) const
        {
            TRet retInc;
            retInc.setZero(U.size());
            auto call = [&](int mode)
            {
                RANS::GetSource_SST<ExDim>(U, GradU, aux.muf, aux.dWallC,
                                           aux.hMax * SADESScale, retInc, mode);
            };
            if (Mode == 0)
                call(0);
            else if (Mode == 1)
                call(1);
            else if (Mode == 2)
            {
                call(1);
                jac += retInc.asDiagonal();
            }
            ret += retInc;
        }
    };

    struct WilcoxSourceContributor
    {
        template <class TRet, class TJac, class TDerivedU, class TGasProp>
        void evaluate(TRet &ret, TJac &jac, const TRet &U, const TDerivedU &GradU,
                      const Geom::tPoint &, const SourceCellAux &aux,
                      const TGasProp &, index, index, int Mode) const
        {
            TRet retInc;
            retInc.setZero(U.size());
            auto call = [&](int mode)
            {
                RANS::GetSource_KOWilcox<ExDim>(U, GradU, aux.muf, aux.dWallC, retInc, mode);
            };
            if (Mode == 0)
                call(0);
            else if (Mode == 1)
                call(1);
            else if (Mode == 2)
            {
                call(1);
                jac += retInc.asDiagonal();
            }
            ret += retInc;
        }
    };

    struct RKESourceContributor
    {
        template <class TRet, class TJac, class TDerivedU, class TGasProp>
        void evaluate(TRet &ret, TJac &jac, const TRet &U, const TDerivedU &GradU,
                      const Geom::tPoint &, const SourceCellAux &aux,
                      const TGasProp &, index, index, int Mode) const
        {
            TRet retInc;
            retInc.setZero(U.size());
            auto call = [&](int mode)
            {
                RANS::GetSource_RealizableKe<ExDim>(U, GradU, aux.muf, aux.dWallC, retInc, mode);
            };
            if (Mode == 0)
                call(0);
            else if (Mode == 1)
                call(1);
            else if (Mode == 2)
            {
                call(1);
                jac += retInc.asDiagonal();
            }
            ret += retInc;
        }
    };

    struct ChemicalContributor
    {
        std::shared_ptr<Chemistry::ChemicalSource> chem;

        // Pre-allocated buffers — allocated once, reused every evaluate() call.
        // Thread-unsafe (callers serialise via the SGS sweep over cells).
        mutable std::vector<double> bufY;
        mutable std::vector<double> bufOmega;
        mutable std::vector<double> bufJ;
        mutable std::vector<double> bufH; // per-species enthalpies [J/kg]

        ChemicalContributor() = default;
        explicit ChemicalContributor(std::shared_ptr<Chemistry::ChemicalSource> c)
            : chem(std::move(c))
        {
            if (chem)
            {
                int Ns = chem->nSpecies();
                int nVars = 5 + Ns - 1;
                bufY.resize(Ns);
                bufOmega.resize(Ns);
                bufJ.resize(Ns * nVars);
                bufH.resize(Ns);
            }
        }

        template <class TRet, class TJac, class TDerivedU, class TGasProp>
        void evaluate(TRet &ret, TJac &jac, const TRet &U, const TDerivedU &,
                      const Geom::tPoint &, const SourceCellAux &aux,
                      const TGasProp &, index, index, int Mode) const
        {
            if (!chem)
                return;
            int Ns = chem->nSpecies();
            int Ns1 = Ns - 1;
            int nVars = static_cast<int>(ret.size());
            int Isp = nVars - Ns1; // species start (= 5 + nRANS)

            double rho = U[0];
            double rhoInv = 1.0 / std::max(rho, 1e-60);

            for (int k = 0; k < Ns1; ++k)
                bufY[k] = U[Isp + k] * rhoInv;
            double sumY = 0;
            for (int k = 0; k < Ns1; ++k)
                sumY += bufY[k];
            bufY[Ns1] = 1.0 - sumY;

            for (int k = 0; k < Ns; ++k)
            {
                if (bufY[k] < 0)
                    bufY[k] = 0;
                if (bufY[k] > 1)
                    bufY[k] = 1;
            }
            double ySum = 0;
            for (int k = 0; k < Ns; ++k)
                ySum += bufY[k];
            if (ySum > 0)
                for (int k = 0; k < Ns; ++k)
                    bufY[k] /= ySum;

            DNDS_assert(std::isfinite(aux.T) && aux.T > 0);
            DNDS_assert(std::isfinite(aux.p) && aux.p > 0);
            DNDS_assert(std::isfinite(rho) && rho > 0);
            double Tcantera = std::max(aux.T, 200.0); // NASA poly lower bound

            Chemistry::ConstSpeciesBufferView Yv{bufY.data(), Ns};
            Chemistry::SpeciesBufferView omegav{bufOmega.data(), Ns};

            if (Mode == 0)
            {
                chem->productionRates(Tcantera, aux.pPhys, Yv, omegav);
                for (int k = 0; k < Ns1; ++k)
                    ret[Isp + k] += bufOmega[k] * chem->molecularWeights()[k];
            }
            else if (Mode == 2)
            {
                Chemistry::JacobianBufferView Jv{bufJ.data(), Ns, nVars, Ns};
                chem->productionRatesAndJacobian(Tcantera, aux.pPhys, rho, Yv, omegav, Jv);
                for (int k = 0; k < Ns1; ++k)
                    ret[Isp + k] += bufOmega[k] * chem->molecularWeights()[k];
                for (int k = 0; k < Ns1; ++k)
                {
                    double Mk = chem->molecularWeights()[k];
                    int iRow = Isp + k;
                    for (int j = 0; j < nVars; ++j)
                    {
                        double val = Mk * Jv(k, j);
                        if (!std::isfinite(val))
                        {
                            fprintf(stderr, "[chem-jac] NaN at row=%d col=%d Jv=%g Mk=%g T=%.1f\n",
                                    iRow, j, Jv(k, j), Mk, (double)aux.T);
                            val = 0;
                        }
                        jac(iRow, j) -= val;
                    }
                }
            }
        }
    };

    // ============================================================================
    // Variant + builder + visitor
    // ============================================================================

    using SourceTermVariant = std::variant<
        BodyForceContributor,
        RotatingFrameContributor,
        AxisymmetricContributor,
        SASourceContributor,
        SSTSourceContributor,
        WilcoxSourceContributor,
        RKESourceContributor,
        ChemicalContributor>;

    template <EulerModel model>
    inline std::vector<SourceTermVariant> buildSourceContributors(
        const EulerEvaluatorSettings<model> &settings, int axisSymmetric)
    {
        using Traits = EulerModelTraits<model>;
        if (!Traits::isExtended)
            return {};

        std::vector<SourceTermVariant> contribs;
        if (settings.constMassForce.norm() > 0)
            contribs.push_back(BodyForceContributor{settings.constMassForce});
        if (settings.frameConstRotation.enabled)
            contribs.push_back(RotatingFrameContributor{settings.frameConstRotation});
        if (axisSymmetric)
            contribs.push_back(AxisymmetricContributor{true});

        switch (settings.ransModel)
        {
        case RANS_SA:
            contribs.push_back(SASourceContributor{
                settings.idealGasProperty.muGas,
                settings.SADESScale,
                settings.SADESMode, settings.SAVersion, settings.ransSARotCorrection});
            break;
        case RANS_KOSST:
            contribs.push_back(SSTSourceContributor{
                settings.idealGasProperty.muGas,
                settings.SADESScale});
            break;
        case RANS_KOWilcox:
            contribs.push_back(WilcoxSourceContributor{});
            break;
        case RANS_RKE:
            contribs.push_back(RKESourceContributor{});
            break;
        default:
            break;
        }
        if (settings.reactiveFlow.enabled)
        {
            auto chemSrc = std::make_shared<Chemistry::ChemicalSource>(
                settings.reactiveFlow.mechanismFile);
            contribs.push_back(ChemicalContributor{std::move(chemSrc)});
        }
        return contribs;
    }

    /**
     * @brief Visitor dispatching a single contributor evaluation via std::visit.
     *
     * Templated on the per-model types (TRet/TJac/TDerivedU/TGasProp) so that it
     * works with NS_EX, NS_EX_3D, or any future extended model without type mismatch.
     */
    template <class TRet, class TJac, class TDerivedU, class TGasProp>
    struct SourceTermVisitor
    {
        TRet &ret;
        TJac &jac;
        const TRet &U;
        const TDerivedU &GradU;
        const Geom::tPoint &pPhy;
        const SourceCellAux &aux;
        const TGasProp &gasProp;
        index iCell, ig;
        int Mode;

        template <typename TContrib>
        void operator()(TContrib &c) const
        {
            c.evaluate(ret, jac, U, GradU, pPhy, aux, gasProp, iCell, ig, Mode);
        }
    };

    // Deduction guide for CTAD (C++17)
    template <class TRet, class TJac, class TDerivedU, class TGasProp>
    SourceTermVisitor(TRet &, TJac &, const TRet &, const TDerivedU &,
                      const Geom::tPoint &, const SourceCellAux &,
                      const TGasProp &, index, index, int)
        -> SourceTermVisitor<TRet, TJac, TDerivedU, TGasProp>;

} // namespace DNDS::Euler
