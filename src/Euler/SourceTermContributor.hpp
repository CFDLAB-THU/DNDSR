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
#include "EulerEvaluatorSettings.hpp"
#include "Physics/PhysicsProperties.hpp"
#include "DNDS/EnvReader.hpp"
#include <Eigen/Eigenvalues>
#include <filesystem>
#ifdef DNDS_DIST_MT_USE_OMP
#    include <omp.h>
#endif

#include <variant>
#include <vector>

namespace DNDS::Euler
{

    // For readability: map EulerModel to matrix types from EulerModelTraits
    template <EulerModel M>
    using SrcTU = typename EulerModelTraits<M>::TU;
    template <EulerModel M>
    using SrcTJac = typename EulerModelTraits<M>::TJacobianU;
    template <EulerModel M>
    using SrcTDiffU = typename EulerModelTraits<M>::TDiffU;

    /**
     * @brief Per-quadrature-point auxiliary data needed by source term contributors.
     */
    struct SourceCellAux
    {
        real dWallC = 0;
        real hMax = 0;
        real muf = 0;
        real T = 300;
        real p = 101325;     // default pressure (code=phys when scaling defaults are 1)
        real pPhys = 101325; // physical pressure [Pa] for Cantera (same as p with default scaling)
        real gammaEq = 1.4;  // pressure/energy closure gammaEq at this point
        real rhoE_base = 0;  // volumetric base energy (0 when no chemistry)
    };

    /**
     * @brief Filter which source contributors to evaluate.
     */
    enum class SourceFilter
    {
        All,            ///< All source contributors (default).
        ReactiveOnly,   ///< Only reactive (chemical) source.
        NonReactiveOnly ///< Everything except reactive source.
    };

    // ============================================================================
    // Shared free functions — model-typed via traits
    // ============================================================================

    template <EulerModel model, int dim, class TMassForce>
    inline void evalSourceBodyForce(const TMassForce &massForce,
                                    typename EulerModelTraits<model>::TU &ret,
                                    typename EulerModelTraits<model>::TJacobianU &jac,
                                    const typename EulerModelTraits<model>::TU &U, int Mode)
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

    template <EulerModel model, int dim, class TFrame>
    inline void evalSourceRotatingFrame(const TFrame &frame, const Geom::tPoint &pPhy,
                                        typename EulerModelTraits<model>::TU &ret,
                                        typename EulerModelTraits<model>::TJacobianU &jac,
                                        const typename EulerModelTraits<model>::TU &U, int Mode)
    {
        using Traits = EulerModelTraits<model>;
        using TVec = typename Traits::TVec;
        using TMat = typename Traits::TMat;
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

    template <EulerModel model>
    inline void evalSourceAxisymmetric(real gammaEq, const Geom::tPoint &pPhy,
                                       typename EulerModelTraits<model>::TU &ret,
                                       const typename EulerModelTraits<model>::TU &U, int Mode,
                                       real rhoE_base = 0)
    {
        constexpr auto I4 = EulerModelTraits<model>::dim + 1;
        if (Mode == 0)
        {
            typename EulerModelTraits<model>::TU uPrim;
            uPrim.resizeLike(U);
            Gas::IdealGasThermalConservative2Primitive(U, uPrim, gammaEq, rhoE_base);
            ret(2) += uPrim(I4) / std::max(verySmallReal, pPhy(1));
        }
    }

    // ============================================================================
    // Contributor structs — templated on EulerModel, typed via EulerModelTraits.
    // ============================================================================

    template <EulerModel model>
    struct BodyForceContributor
    {
        using Traits = EulerModelTraits<model>;
        using TU = typename Traits::TU;
        using TJac = typename Traits::TJacobianU;
        using TDiffU = typename Traits::TDiffU;

        Eigen::Vector<real, 3> force{0, 0, 0};

        void evaluate(TU &ret, TJac &jac, const TU &U, const TDiffU &,
                      const Geom::tPoint &, const SourceCellAux &,
                      index, index, int Mode) const
        {
            if (force.isZero(0))
                return;
            evalSourceBodyForce<model, Traits::dim>(force, ret, jac, U, Mode);
        }
    };

    template <EulerModel model>
    struct RotatingFrameContributor
    {
        using Traits = EulerModelTraits<model>;
        using TU = typename Traits::TU;
        using TJac = typename Traits::TJacobianU;
        using TDiffU = typename Traits::TDiffU;

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

        void evaluate(TU &ret, TJac &jac, const TU &U, const TDiffU &,
                      const Geom::tPoint &pPhy, const SourceCellAux &,
                      index, index, int Mode) const
        {
            if (!enabled)
                return;
            evalSourceRotatingFrame<model, Traits::dim>(*this, pPhy, ret, jac, U, Mode);
        }
    };

    template <EulerModel model>
    struct AxisymmetricContributor
    {
        using Traits = EulerModelTraits<model>;
        using TU = typename Traits::TU;
        using TJac = typename Traits::TJacobianU;
        using TDiffU = typename Traits::TDiffU;

        bool active = false;

        void evaluate(TU &ret, TJac &, const TU &U, const TDiffU &,
                      const Geom::tPoint &pPhy, const SourceCellAux &aux,
                      index, index, int Mode) const
        {
            if (!active)
                return;
            evalSourceAxisymmetric<model>(aux.gammaEq, pPhy, ret, U, Mode, aux.rhoE_base);
        }
    };

    template <EulerModel model>
    struct SASourceContributor
    {
        using Traits = EulerModelTraits<model>;
        using TU = typename Traits::TU;
        using TJac = typename Traits::TJacobianU;
        using TDiffU = typename Traits::TDiffU;

        real muGas = 1;
        real SADESScale = veryLargeReal;
        int SADESMode = 1;
        int SAVersion = 0;
        int ransSARotCorrection = 1;

        void evaluate(TU &ret, TJac &jac, const TU &U, const TDiffU &GradU,
                      const Geom::tPoint &, const SourceCellAux &aux,
                      index iCell, index, int Mode) const
        {
            TU retInc;
            retInc.setZero(U.size());
            real d = std::min(aux.dWallC, std::pow(veryLargeReal, 1. / 6.));
            real lLES = aux.hMax * SADESScale;
            real cWall = SADESScale > 100.0 ? 1.0 : 0.15;
            lLES = std::min(lLES, std::max({d * cWall, aux.hMax * cWall}));
            auto call = [&](int mode)
            {
                RANS::GetSource_SA<Traits::dim>(U, GradU, muGas, aux.muf, aux.gammaEq,
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

    template <EulerModel model>
    struct SSTSourceContributor
    {
        using Traits = EulerModelTraits<model>;
        using TU = typename Traits::TU;
        using TJac = typename Traits::TJacobianU;
        using TDiffU = typename Traits::TDiffU;

        real muGas = 1;
        real SADESScale = veryLargeReal;

        void evaluate(TU &ret, TJac &jac, const TU &U, const TDiffU &GradU,
                      const Geom::tPoint &, const SourceCellAux &aux,
                      index iCell, index, int Mode) const
        {
            TU retInc;
            retInc.setZero(U.size());
            auto call = [&](int mode)
            {
                RANS::GetSource_SST<Traits::dim>(U, GradU, aux.muf, aux.dWallC,
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

    template <EulerModel model>
    struct WilcoxSourceContributor
    {
        using Traits = EulerModelTraits<model>;
        using TU = typename Traits::TU;
        using TJac = typename Traits::TJacobianU;
        using TDiffU = typename Traits::TDiffU;

        void evaluate(TU &ret, TJac &jac, const TU &U, const TDiffU &GradU,
                      const Geom::tPoint &, const SourceCellAux &aux,
                      index, index, int Mode) const
        {
            TU retInc;
            retInc.setZero(U.size());
            auto call = [&](int mode)
            {
                RANS::GetSource_KOWilcox<Traits::dim>(U, GradU, aux.muf, aux.dWallC, retInc, mode);
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

    template <EulerModel model>
    struct RKESourceContributor
    {
        using Traits = EulerModelTraits<model>;
        using TU = typename Traits::TU;
        using TJac = typename Traits::TJacobianU;
        using TDiffU = typename Traits::TDiffU;

        void evaluate(TU &ret, TJac &jac, const TU &U, const TDiffU &GradU,
                      const Geom::tPoint &, const SourceCellAux &aux,
                      index, index, int Mode) const
        {
            TU retInc;
            retInc.setZero(U.size());
            auto call = [&](int mode)
            {
                RANS::GetSource_RealizableKe<Traits::dim>(U, GradU, aux.muf, aux.dWallC, retInc, mode);
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

    template <EulerModel model>
    struct ChemicalContributor
    {
        using Traits = EulerModelTraits<model>;
        using TU = typename Traits::TU;
        using TJac = typename Traits::TJacobianU;
        using TDiffU = typename Traits::TDiffU;
        using ChemPool = std::shared_ptr<std::vector<Chemistry::ChemicalSource>>;
        ChemPool pool_;
        typename EulerEvaluatorSettings<model>::IdealGasProperty igProp_;
        real sourceScale_ = 1.0;
        int filterReactiveJacobianSpectrum_ = 1;

        // Per-thread work buffers (one set per OMP thread)
        mutable std::vector<std::vector<double>> bufOmega_;
        mutable std::vector<std::vector<double>> bufJ_;
        mutable std::vector<std::vector<double>> bufY_;
        mutable std::vector<Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>> bufDSdu_;

        int threadIdx() const
        {
            DNDS_assert(pool_);
#ifdef DNDS_DIST_MT_USE_OMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            DNDS_check_throw_info(tid < static_cast<int>(pool_->size()),
                                  fmt::format("ChemicalSource pool has {} entries but OpenMP thread {} requested chemistry source evaluation",
                                              pool_->size(), tid));
            return tid;
        }

        ChemicalContributor() = default;
        explicit ChemicalContributor(ChemPool pool, typename EulerEvaluatorSettings<model>::IdealGasProperty igProp,
                                     real sourceScale, int nVars)
            : pool_(std::move(pool)), igProp_(std::move(igProp)), sourceScale_(sourceScale)
        {
            if (pool_ && pool_->size() > 0)
            {
                int nT = static_cast<int>(pool_->size());
                auto &c0 = (*pool_)[0];
                int Ns = c0.nSpecies();
                bufOmega_.resize(nT);
                bufJ_.resize(nT);
                bufY_.resize(nT);
                bufDSdu_.resize(nT);
                for (int t = 0; t < nT; ++t)
                {
                    bufOmega_[t].resize(Ns);
                    bufJ_[t].resize(Ns * nVars);
                    bufY_[t].resize(Ns);
                    bufDSdu_[t].setZero(nVars, nVars);
                }
            }
        }

        void ensureBuffers(int nVars) const
        {
            // Buffers are eagerly initialized in the constructor; this method
            // exists only as a no-op backward-compatibility shim.  The
            // constructor sizes bufOmega_, bufJ_, bufY_, and bufDSdu_ for
            // every thread based on the pool size and nSpecies / nVars.
            (void)nVars;
        }

        void evaluate(TU &ret, TJac &jac, const TU &U, const TDiffU &,
                      const Geom::tPoint &, const SourceCellAux &aux,
                      index, index, int Mode) const
        {
            if (!pool_)
                return;
            if (sourceScale_ == 0.0)
                return;
            int tid = threadIdx();
            auto &c = (*pool_)[tid];
            int Ns = c.nSpecies();
            int Ns1 = Ns - 1;
            int nVars = static_cast<int>(ret.size());
            int Isp = nVars - Ns1;                                       // species start
            int I4 = static_cast<int>(EulerModelTraits<model>::dim) + 1; // energy index, not Isp-1 (wrong with RANS)

            double rho = U[0];
            double rhoInv = 1.0 / std::max(rho, 1e-60);

            std::vector<double> &Ybuf = bufY_[tid];
            Chemistry::SpeciesBufferView Yv{Ybuf.data(), Ns};
            c.massFractions(rho, {&U[Isp], Ns1}, Yv);
            Chemistry::ConstSpeciesBufferView Yc{Ybuf.data(), Ns};
            auto &bufOmega = bufOmega_[tid];

            DNDS_assert(std::isfinite(aux.T) && aux.T > 0);
            DNDS_assert(std::isfinite(aux.p) && aux.p > 0);
            DNDS_assert(std::isfinite(rho) && rho > 0);
            // aux.T is code-scaled; Cantera needs physical T [K]
            double Tphys = igProp_.T0 > 0 ? aux.T * igProp_.T0 : aux.T;
            double Tcantera = std::max(Tphys, c.baseTemperature());
            double pCantera = aux.pPhys;
            if (Tcantera != Tphys)
                pCantera = rho * igProp_.rho0 * c.mixtureR(Yc) * Tcantera;

            Chemistry::SpeciesBufferView omegav{bufOmega.data(), Ns};

            // Source rate scale: S0 = rho0 * U0 / L0  [kg/(m³·s)]
            // Physical source omega*MW [kg/(m³·s)] -> code source = omega*MW / S0
            double invS0 = igProp_.L0 / (igProp_.rho0 * igProp_.U0);

            if (Mode == 0)
            {
                c.productionRates(Tcantera, pCantera, Yc, omegav);
                for (int k = 0; k < Ns1; ++k)
                    ret[Isp + k] += sourceScale_ * bufOmega[k] * c.molecularWeights()[k] * invS0;
            }
            else if (Mode == 1)
            {
                DNDS_assert_info(false, "ChemicalContributor: diagonal-Jacobian mode not implemented");
            }
            else if (Mode == 2)
            {
                auto &bufJ = bufJ_[tid];
                Chemistry::JacobianBufferView Jv{bufJ.data(), Ns, nVars, Ns};
                double uM1 = (I4 >= 2) ? U[1] : 0; // ρu (I4=dim+1≥3, guard always true)
                double uM2 = (I4 >= 3) ? U[2] : 0; // ρv (I4≥3 always for dim≥2)
                double uM3 = (I4 >= 4) ? U[3] : 0; // ρw (present only for 3D, 0 for 2D)
                c.productionRatesAndJacobian(Tcantera, pCantera, rho, U[I4],
                                             uM1, uM2, uM3, I4, Yc, omegav, Jv,
                                             Chemistry::ChemicalSource::JAC_DEFAULT);
                for (int k = 0; k < Ns1; ++k)
                    ret[Isp + k] += sourceScale_ * bufOmega[k] * c.molecularWeights()[k] * invS0;

                auto &dSdu = bufDSdu_[tid];
                dSdu.setZero(nVars, nVars);
                for (int k = 0; k < Ns1; ++k)
                {
                    double Mk = c.molecularWeights()[k];
                    int iRow = Isp + k;
                    for (int j = 0; j < nVars; ++j)
                    {
                        double val = sourceScale_ * Mk * Jv(k, j) * invS0;
                        if (!std::isfinite(val))
                        {
                            fprintf(stderr, "[chem-jac] NaN at row=%d col=%d Jv=%g Mk=%g T=%.1f\n",
                                    iRow, j, Jv(k, j), Mk, (double)aux.T);
                            val = 0;
                        }
                        dSdu(iRow, j) = val;
                    }
                }
                if (filterReactiveJacobianSpectrum_ == 1)
                {
                    // kDebugEigenFilter — per-cell reactive Jacobian eigenvalue
                    // filtering switch.  The chemical source Jacobian dSdu is
                    // assembled in species rows only (Isp .. Isp+Ns1-1); the
                    // filter ensures that its eigenvalues have non-positive real
                    // parts so the source term does not destabilise the
                    // time-integration scheme.
                    //
                    // Mode 0 — no filtering (default).
                    //   The chemical Jacobian is already diagonally dominant and
                    //   stable (negative-diagonal Z-matrix).  Skipping the filter
                    //   is both the fastest per-iteration *and* converges
                    //   correctly (tested on 1-D H2/O2 detonation, 5000 cells).
                    //
                    // Mode 1 — Gershgorin circle filter (O(Ns1^2)).
                    //   For each species row i, compute radius = sum_{j!=i}|dSdu(i,j)|.
                    //   If center+radius > 0 the diagonal is shifted so the
                    //   Gershgorin disc lies in the left half-plane.  Per-iteration
                    //   cost is close to mode 0, but the filter is *too conservative*
                    //   — it over-suppresses cross-species coupling, stalling
                    //   species residual convergence (tested on detonation).
                    //   Keep available for stiff chemistry regimes where it may help.
                    //
                    // Mode 2 — full ComplexEigenSolver (O(Ns1^3)).
                    //   Original code.  Clips the real part of every eigenvalue
                    //   to ≤ 0 then reconstructs the filtered matrix.  Converges
                    //   correctly but ~2× slower per iteration than modes 0/1
                    //   (7 % of total cycles in ComplexSchur::reduceToTriangularForm).
                    //   Preserved as a reference for validation.
                    //
                    static const int kDebugEigenFilter = 0;
                    switch (kDebugEigenFilter)
                    {
                    case 0:
                        break;
                    case 1:
                    {
                        for (int k = 0; k < Ns1; ++k)
                        {
                            int iRow = Isp + k;
                            real center = dSdu(iRow, iRow);
                            real radius = real(0);
                            for (int j = Isp; j < Isp + Ns1; ++j)
                                if (j != iRow)
                                    radius += std::abs(dSdu(iRow, j));
                            if (center + radius > 0)
                                dSdu(iRow, iRow) -= (center + radius);
                        }
                        break;
                    }
                    case 2:
                    {
                        Eigen::ComplexEigenSolver<Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>> eig(dSdu);
                        if (eig.info() == Eigen::Success)
                        {
                            Eigen::Vector<std::complex<real>, Eigen::Dynamic> lambda = eig.eigenvalues();
                            for (int i = 0; i < lambda.size(); ++i)
                                lambda(i) = std::complex<real>(std::min(lambda(i).real(), real(0)), lambda(i).imag());
                            Eigen::Matrix<std::complex<real>, Eigen::Dynamic, Eigen::Dynamic> dSduFiltered =
                                eig.eigenvectors() * lambda.asDiagonal() * eig.eigenvectors().inverse();
                            dSdu = dSduFiltered.real();
                        }
                        break;
                    }
                    default:
                        break;
                    }
                }
                jac -= dSdu;
            }
        }
    };

    // ============================================================================
    // Variant + builder + visitor
    // ============================================================================

    template <EulerModel model>
    using SourceTermVariant = std::variant<
        BodyForceContributor<model>,
        RotatingFrameContributor<model>,
        AxisymmetricContributor<model>,
        SASourceContributor<model>,
        SSTSourceContributor<model>,
        WilcoxSourceContributor<model>,
        RKESourceContributor<model>,
        ChemicalContributor<model>>;

    template <EulerModel model>
    inline std::vector<SourceTermVariant<model>> buildSourceContributors(
        const EulerEvaluatorSettings<model> &settings, const PhysicsProperties<model> &phys,
        int nVars, int axisSymmetric)
    {
        using Traits = EulerModelTraits<model>;
        if (!Traits::isExtended)
            return {};

        std::vector<SourceTermVariant<model>> contribs;
        if (settings.constMassForce.norm() > 0)
            contribs.push_back(BodyForceContributor<model>{settings.constMassForce});
        if (settings.frameConstRotation.enabled)
            contribs.push_back(RotatingFrameContributor<model>{settings.frameConstRotation});
        if (axisSymmetric)
            contribs.push_back(AxisymmetricContributor<model>{true});

        real muGasCode = phys.muRef();
        switch (settings.ransModel)
        {
        case RANS_SA:
            contribs.push_back(SASourceContributor<model>{
                muGasCode,
                settings.SADESScale,
                settings.SADESMode, settings.SAVersion, settings.ransSARotCorrection});
            break;
        case RANS_KOSST:
            contribs.push_back(SSTSourceContributor<model>{
                muGasCode,
                settings.SADESScale});
            break;
        case RANS_KOWilcox:
            contribs.push_back(WilcoxSourceContributor<model>{});
            break;
        case RANS_RKE:
            contribs.push_back(RKESourceContributor<model>{});
            break;
        default:
            break;
        }
        if (settings.reactiveFlow.enabled)
        {
            int nThreads = 1;
#ifdef DNDS_DIST_MT_USE_OMP
            nThreads = omp_get_max_threads(); // note: OMP generally uses get_max not get_num
#endif
            auto pool = std::make_shared<std::vector<Chemistry::ChemicalSource>>();
            pool->reserve(nThreads);
            {
                std::string mechPath = GetEnvString("DNDS_MECH_PATH", "");
                const std::string &mechFile = settings.reactiveFlow.mechanismFile;
                std::filesystem::path mechFSPath(mechFile);
                std::string resolvedMechFile = mechFile;
                if (!mechFile.empty() && mechFSPath.is_relative() && !std::filesystem::exists(mechFSPath) && !mechPath.empty())
                    resolvedMechFile = (std::filesystem::path(mechPath) / mechFSPath).string();
                pool->emplace_back(resolvedMechFile, "", settings.idealGasProperty.U0, settings.idealGasProperty.rho0,
                                   settings.reactiveFlow.TBase, settings.reactiveFlow.transportModel);
            }
            for (int t = 1; t < nThreads; ++t)
                pool->push_back(std::move(*pool->at(0).clone()));
            contribs.push_back(ChemicalContributor<model>{std::move(pool), settings.idealGasProperty,
                                                          settings.reactiveSourceScale, nVars});
        }
        return contribs;
    }

    /**
     * @brief Visitor dispatching a single contributor evaluation via std::visit.
     */
    template <EulerModel model>
    struct SourceTermVisitor
    {
        using Traits = EulerModelTraits<model>;
        using TU_t = typename Traits::TU;
        using TJac_t = typename Traits::TJacobianU;
        using TDiffU_t = typename Traits::TDiffU;

        TU_t &ret;
        TJac_t &jac;
        const TU_t &U;
        const TDiffU_t &GradU;
        const Geom::tPoint &pPhy;
        const SourceCellAux &aux;
        index iCell, ig;
        int Mode;
        SourceFilter filter = SourceFilter::All;

        template <typename TContrib>
        void operator()(TContrib &c) const
        {
            constexpr bool isReactive = std::is_same_v<std::decay_t<TContrib>, ChemicalContributor<model>>;
            if (filter == SourceFilter::ReactiveOnly && !isReactive)
                return;
            if (filter == SourceFilter::NonReactiveOnly && isReactive)
                return;
            c.evaluate(ret, jac, U, GradU, pPhy, aux, iCell, ig, Mode);
        }
    };

} // namespace DNDS::Euler
