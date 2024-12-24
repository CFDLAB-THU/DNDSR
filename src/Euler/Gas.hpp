#pragma once
#include "DNDS/Defines.hpp"

#include <json.hpp>
#include <fmt/core.h>

namespace DNDS::Euler::Gas
{
    typedef Eigen::Vector3d tVec;
    typedef Eigen::Vector2d tVec2;

    static const int MaxBatch = 16;

    enum RiemannSolverType
    {
        UnknownRS = 0,
        Roe = 1,
        HLLC = 2,
        HLLEP = 3,
        HLLEP_V1 = 21,
        Roe_M1 = 11,
        Roe_M2 = 12,
        Roe_M3 = 13,
        Roe_M4 = 14,
        Roe_M5 = 15,
    };

    NLOHMANN_JSON_SERIALIZE_ENUM(
        RiemannSolverType,
        {
            {UnknownRS, "UnknownRS"},
            {Roe, "Roe"},
            {HLLC, "HLLC"},
            {HLLEP, "HLLEP"},
            {HLLEP_V1, "HLLEP_V1"},
            {Roe_M1, "Roe_M1"},
            {Roe_M2, "Roe_M2"},
            {Roe_M3, "Roe_M3"},
            {Roe_M4, "Roe_M4"},
            {Roe_M5, "Roe_M5"},
        })

    /**
     * @brief 3D only warning: ReV should be already initialized
     *
     */
    template <int dim = 3, class TVec, class TeV>
    inline void EulerGasRightEigenVector(const TVec &velo, real Vsqr, real H, real a, TeV &ReV)
    {
        ReV.setZero();
        ReV(0, {0, 1, dim + 1}).setConstant(1);
        ReV(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>), Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>))
            .diagonal()
            .setConstant(1);

        ReV(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), {0, 1, dim + 1}).colwise() = velo;
        ReV(1, 0) -= a;
        ReV(1, dim + 1) += a;

        // Last Row
        ReV(dim + 1, 0) = H - velo(0) * a;
        ReV(dim + 1, dim + 1) = H + velo(0) * a;
        ReV(dim + 1, 1) = 0.5 * Vsqr;

        ReV(dim + 1, Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
            velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>));
    }

    /**
     * @brief 3D only warning: LeV should be already initialized
     *
     */
    template <int dim = 3, class TVec, class TeV>
    inline void EulerGasLeftEigenVector(const TVec &velo, real Vsqr, real H, real a, real gamma, TeV &LeV)
    {
        LeV.setZero();
        real gammaBar = gamma - 1;
        LeV(0, 0) = H + a / gammaBar * (velo(0) - a);
        LeV(0, 1) = -velo(0) - a / gammaBar;
        LeV(0, Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
            -velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>));
        LeV(0, dim + 1) = 1;

        LeV(1, 0) = -2 * H + 4 / gammaBar * (a * a);
        LeV(1, Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) = velo.transpose() * 2;
        LeV(1, dim + 1) = -2;

        LeV(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>), 0) =
            velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>)) * (-2 * (a * a) / gammaBar);
        LeV(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>), Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>))
            .diagonal()
            .setConstant(2 * (a * a) / gammaBar);

        LeV(dim + 1, 0) = H - a / gammaBar * (velo(0) + a);
        LeV(dim + 1, 1) = -velo(0) + a / gammaBar;
        LeV(dim + 1, Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
            -velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>));
        LeV(dim + 1, dim + 1) = 1;

        LeV *= gammaBar / (2 * a * a);
    }

    inline void IdealGasThermal(
        real E, real rho, real vSqr, real gamma, real &p, real &asqr, real &H)
    {
        p = (gamma - 1) * (E - rho * 0.5 * vSqr);
        asqr = gamma * p / rho;
        H = (E + p) / rho;
    }

    template <int dim = 3, class TCons, class TPrim>
    inline void IdealGasThermalConservative2Primitive(
        const TCons &U, TPrim &prim, real gamma)
    {
        prim = U / U(0);
        real vSqr = (U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) / U(0)).squaredNorm();
        real rho = U(0);
        real E = U(1 + dim);
        real p = (gamma - 1) * (E - rho * 0.5 * vSqr);
        prim(0) = rho;
        prim(1 + dim) = p;
        DNDS_assert(rho > 0);
    }

    template <int dim = 3, class TCons, class TPrim>
    inline void IdealGasThermalPrimitive2Conservative(
        const TPrim &prim, TCons &U, real gamma)
    {
        U = prim * prim(0);
        real vSqr = prim(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).squaredNorm();
        real rho = prim(0);
        real p = prim(dim + 1);
        real E = p / (gamma - 1) + rho * 0.5 * vSqr;
        U(0) = rho;
        U(dim + 1) = E;
        DNDS_assert(rho > 0);
    }

    template <int dim = 3, class TPrim>
    std::tuple<real, real> IdealGasThermalPrimitiveGetP0T0(
        const TPrim &prim, real gamma, real rg)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto I4 = dim + 1;
        real T = prim(I4) / (prim(0) * rg + verySmallReal);
        real vsqr = prim(Seq123).squaredNorm();
        real asqr = gamma * prim(I4) / prim(0);
        real Msqr = vsqr / (asqr + verySmallReal);
        real p0 = std::pow(1 + (gamma - 1) * 0.5 * Msqr, gamma / (gamma - 1)) * prim(I4);
        real T0 = (1 + (gamma - 1) * 0.5 * Msqr) * T;
        return std::make_tuple(p0, T0);
    }

    /**
     * @brief calculates Inviscid Flux for x direction
     *
     */
    template <int dim = 3, typename TU, typename TF, class TVec, class TVecVG>
    inline void GasInviscidFlux(const TU &U, const TVec &velo, const TVecVG &vg, real p, TF &F)
    {
        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = U(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * (velo(0) - vg(0)); // note that additional flux are unattended!
        F(1) += p;
        F(dim + 1) += velo(0) * p;
        // original form: F(dim + 1) += (velo(0) - vg(0)) * p + vg(0) * p;
    }

    template <int dim = 3, typename TU, typename TF, class TVec, class TVecN, class TVecVG>
    inline void GasInviscidFlux_XY(const TU &U, const TVec &velo, const TVecVG &vg, const TVecN &n, real p, TF &F)
    {
        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = U(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * (velo - vg).dot(n); // note that additional flux are unattended!
        F(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) += p * n;
        F(dim + 1) += velo.dot(n) * p;
        // original form: F(dim + 1) += (velo(0) - vg(0)) * p + vg(0) * p;
    }

    template <int dim = 3, typename TU, typename TF, class TVec, class TVecVG, class TP>
    inline void GasInviscidFlux_Batch(const TU &U, const TVec &velo, const TVecVG &vg, TP &&p, TF &F)
    {
        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), Eigen::all) = U(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), Eigen::all).array().rowwise() * (velo(0, Eigen::all) - vg(0, Eigen::all)).array(); // note that additional flux are unattended!
        F(1, Eigen::all).array() += p.array();
        F(dim + 1, Eigen::all).array() += velo(0, Eigen::all).array() * p.array();
        // original form: F(dim + 1) += (velo(0) - vg(0)) * p + vg(0) * p;
    }

    template <int dim = 3, typename TU, typename TF, class TVec, class TVecVG, class TVecN, class TP>
    inline void GasInviscidFlux_XY_Batch(const TU &U, const TVec &velo, const TVecVG &vg, const TVecN &n, TP &&p, TF &F)
    {
        auto vn = (velo.array() * n.array()).colwise().sum();
        auto vgn = (vg.array() * n.array()).colwise().sum();
        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), Eigen::all) = U(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), Eigen::all).array().rowwise() * (vn - vgn).array(); // note that additional flux are unattended!
        F(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), Eigen::all).array() += n.array().rowwise() * p.array();
        F(dim + 1, Eigen::all).array() += vn * p.array();
        // original form: F(dim + 1) += (velo(0) - vg(0)) * p + vg(0) * p;
    }

    template <int dim = 3, typename TU, class TVec>
    inline void IdealGasUIncrement(const TU &U, const TU &dU, const TVec &velo, real gamma, TVec &dVelo, real &dp)
    {
        dVelo = (dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) -
                 U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) / U(0) * dU(0)) /
                U(0);
        dp = (gamma - 1) * (dU(dim + 1) -
                            0.5 * (dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(velo) +
                                   U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(dVelo)));
    } // For Lax-Flux jacobian

    template <int dim = 3, typename TU, typename TF, class TVec, class TVecVG>
    inline void GasInviscidFluxFacialIncrement(const TU &U, const TU &dU,
                                               const TVec &unitNorm,
                                               const TVecVG &velo, const TVec &dVelo, const TVec &vg,
                                               real dp, real p,
                                               TF &F)
    {
        real vn = velo.dot(unitNorm);
        real dvn = dVelo.dot(unitNorm);
        F(0) = dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(unitNorm);
        F(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) =
            dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) * vn +
            U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) * dvn + unitNorm * dp;
        F(dim + 1) = (dU(dim + 1) + dp) * vn + (U(dim + 1) + p) * dvn;
        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -= dU(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * vg.dot(unitNorm);
    }

    template <int dim = 3, typename TU>
    inline auto IdealGas_EulerGasRightEigenVector(const TU &U, real gamma)
    {
        DNDS_assert(U(0) > 0);
        Eigen::Matrix<real, dim + 2, dim + 2> ReV;
        Eigen::Vector<real, dim> velo =
            (U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / U(0)).matrix();
        real vsqr = velo.squaredNorm();
        real asqr, p, H;
        IdealGasThermal(U(dim + 1), U(0), vsqr, gamma, p, asqr, H);
        DNDS_assert(asqr >= 0);
        EulerGasRightEigenVector<dim>(velo, vsqr, H, std::sqrt(asqr), ReV);
        return ReV;
    }

    template <int dim = 3, typename TU>
    inline auto IdealGas_EulerGasLeftEigenVector(const TU &U, real gamma)
    {
        DNDS_assert(U(0) > 0);
        Eigen::Matrix<real, dim + 2, dim + 2> LeV;
        Eigen::Vector<real, dim> velo =
            (U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / U(0)).matrix();
        real vsqr = velo.squaredNorm();
        real asqr, p, H;
        IdealGasThermal(U(dim + 1), U(0), vsqr, gamma, p, asqr, H);
        DNDS_assert(asqr >= 0);
        EulerGasLeftEigenVector<dim>(velo, vsqr, H, std::sqrt(asqr), gamma, LeV);
        return LeV;
    }

    // #define DNDS_GAS_HLLEP_USE_V1
    template <int dim = 3, int type = 0,
              typename TUL, typename TUR,
              typename TULm, typename TURm, typename TVecVG, typename TVecN,
              typename TF, typename TFdumpInfo>
    void HLLEPFlux_IdealGas(const TUL &UL, const TUR &UR, const TULm &ULm, const TURm &URm,
                            const TVecVG &vg, const TVecN &n,
                            real gamma, TF &F, real dLambda,
                            const TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
    {
        static real scaleHartenYee = 0.05;
        using TVec = Eigen::Vector<real, dim>;

        if (!(UL(0) > 0 && UR(0) > 0))
        {
            dumpInfo();
        }
        DNDS_assert(UL(0) > 0 && UR(0) > 0);
        TVec veloL = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UL(0)).matrix();
        TVec veloR = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UR(0)).matrix();
        TVec veloLm = (ULm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / ULm(0)).matrix();
        TVec veloRm = (URm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / URm(0)).matrix();

        real asqrL, asqrR, pL, pR, HL, HR;
        real vsqrL = veloL.squaredNorm();
        real vsqrR = veloR.squaredNorm();
        IdealGasThermal(UL(dim + 1), UL(0), vsqrL, gamma, pL, asqrL, HL);
        IdealGasThermal(UR(dim + 1), UR(0), vsqrR, gamma, pR, asqrR, HR);
        real asqrLm, asqrRm, pLm, pRm, HLm, HRm;
        real vsqrLm = veloLm.squaredNorm();
        real vsqrRm = veloRm.squaredNorm();
        IdealGasThermal(ULm(dim + 1), ULm(0), vsqrLm, gamma, pLm, asqrLm, HLm);
        IdealGasThermal(URm(dim + 1), URm(0), vsqrRm, gamma, pRm, asqrRm, HRm);

        real sqrtRhoLm = std::sqrt(ULm(0));
        real sqrtRhoRm = std::sqrt(URm(0));

        TVec veloRoe = (sqrtRhoLm * veloLm + sqrtRhoRm * veloRm) / (sqrtRhoLm + sqrtRhoRm);
        real vsqrRoe = veloRoe.squaredNorm();
        real HRoe = (sqrtRhoLm * HLm + sqrtRhoRm * HRm) / (sqrtRhoLm + sqrtRhoRm);
        real asqrRoe = (gamma - 1) * (HRoe - 0.5 * vsqrRoe);
        real rhoRoe = sqrtRhoLm * sqrtRhoRm;

        Eigen::Vector<real, dim + 2> FL, FR;
        GasInviscidFlux_XY<dim>(UL, veloL, vg, n, pL, FL);
        GasInviscidFlux_XY<dim>(UR, veloR, vg, n, pR, FR);

        if (!(asqrRoe > 0))
        {
            dumpInfo();
        }
        DNDS_assert(asqrRoe > 0);
        real aRoe = std::sqrt(asqrRoe);
        real veloRoeN = veloRoe.dot(n);
        real vgN = vg.dot(n);
        real veloRoe0 = veloRoeN - vgN;
        lam0 = std::abs(veloRoe0 - aRoe);
        lam123 = std::abs(veloRoe0);
        lam4 = std::abs(veloRoe0 + aRoe);
        real veloLm0 = (veloLm - vg).dot(n);
        real veloRm0 = (veloRm - vg).dot(n);

        Eigen::Vector<real, dim + 2> incU =
            UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
            UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)); //! not using m, for this is accuracy-limited!
        real incU123N = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(n);

        TVec alpha23V = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) - incU(0) * veloRoe;
        TVec alpha23VT = alpha23V - n * alpha23V.dot(n);
        real incU4b = incU(dim + 1) - alpha23VT.dot(veloRoe);
        real alpha1 = (gamma - 1) / asqrRoe *
                      (incU(0) * (HRoe - veloRoeN * veloRoeN) +
                       veloRoeN * incU123N - incU4b);
        real alpha0 = (incU(0) * (veloRoeN + aRoe) - incU123N - aRoe * alpha1) / (2 * aRoe);
        real alpha4 = incU(0) - (alpha0 + alpha1);

        // std::cout << alpha.transpose() << std::endl;
        // std::cout << std::endl;

        real SL = std::min(lam0, veloLm0 - std::sqrt(asqrL));
        real SR = std::max(lam4, veloRm0 + std::sqrt(asqrR));
        real UU = std::abs(veloRoe0);
        real dfix = aRoe / (aRoe + UU);
        real SP = std::max(SR, 0.0);
        real SM = std::min(SL, 0.0);

        if constexpr (type != 1) // ? HLLEP
        {
            Eigen::Vector<real, dim + 2> ReV1;
            {
                ReV1(0) = 1;
                ReV1(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) = veloRoe;
                ReV1(dim + 1) = vsqrRoe * 0.5;
            }
            real div = SP - SM;
            div += signP(div) * verySmallReal;
            // F = (SP * FL - SM * FR) / div + (SP * SM / div) * (UR - UL - dfix * ReVRoe(Eigen::all, {1, 2, 3}) * alpha({1, 2, 3}));
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                (SP * FL - SM * FR) / div +
                (SP * SM / div) *
                    (UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
                     UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
                     dfix * ReV1 * alpha1);
        }
        else
        {
            // ? HLLEP_V1
            real delta1 = aRoe / (std::abs(veloRoe0) + aRoe + verySmallReal);
            real delta2 = 0.0;
            real delta3 = 0.0;

            lam123 = ((SP + SM) * (veloRoe0)-2.0 * (1.0 - delta1) * (SP * SM)) / (SP - SM);
            lam4 = ((SP + SM) * (veloRoe0 + aRoe) - 2.0 * (1.0 - delta2) * (SP * SM)) / (SP - SM);
            lam0 = ((SP + SM) * (veloRoe0 - aRoe) - 2.0 * (1.0 - delta3) * (SP * SM)) / (SP - SM);

            alpha0 *= lam0;
            alpha1 *= lam123;
            alpha23VT *= lam123;
            alpha4 *= lam4; // here becomes alpha_i * lam_i

            Eigen::Vector<real, dim + 2> incF;
            incF(0) = alpha0 + alpha1 + alpha4;
            incF(dim + 1) = (HRoe - veloRoeN * aRoe) * alpha0 + 0.5 * vsqrRoe * alpha1 +
                            (HRoe + veloRoeN * aRoe) * alpha4 + alpha23VT.dot(veloRoe);
            incF(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) =
                (veloRoe - aRoe * n) * alpha0 + (veloRoe + aRoe * n) * alpha4 +
                veloRoe * alpha1 + alpha23VT;

            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = (FL + FR) * 0.5 - 0.5 * incF;
        }
    }

    template <int dim = 3, typename TUL, typename TUR, typename TULm, typename TURm,
              typename TVecVG, typename TVecN,
              typename TF, typename TFdumpInfo>
    void HLLCFlux_IdealGas_HartenYee(const TUL &UL, const TUR &UR, const TULm &ULm, const TURm &URm,
                                     const TVecVG &vg, const TVecN &n,
                                     real gamma, TF &F, real dLambda,
                                     const TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
    {
        //! warning: has accuracy issue (see IV test)
        static real scaleHartenYee = 0.05;
        using TVec = Eigen::Vector<real, dim>;

        if (!(UL(0) > 0 && UR(0) > 0))
        {
            dumpInfo();
        }
        DNDS_assert(UL(0) > 0 && UR(0) > 0);
        TVec veloL = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UL(0)).matrix();
        TVec veloR = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UR(0)).matrix();
        TVec veloLm = (ULm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / ULm(0)).matrix();
        TVec veloRm = (URm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / URm(0)).matrix();

        real asqrL, asqrR, pL, pR, HL, HR;
        real vsqrL = veloL.squaredNorm();
        real vsqrR = veloR.squaredNorm();
        IdealGasThermal(UL(dim + 1), UL(0), vsqrL, gamma, pL, asqrL, HL);
        IdealGasThermal(UR(dim + 1), UR(0), vsqrR, gamma, pR, asqrR, HR);
        real asqrLm, asqrRm, pLm, pRm, HLm, HRm;
        real vsqrLm = veloLm.squaredNorm();
        real vsqrRm = veloRm.squaredNorm();
        IdealGasThermal(ULm(dim + 1), ULm(0), vsqrLm, gamma, pLm, asqrLm, HLm);
        IdealGasThermal(URm(dim + 1), URm(0), vsqrRm, gamma, pRm, asqrRm, HRm);

        real sqrtRhoLm = std::sqrt(ULm(0));
        real sqrtRhoRm = std::sqrt(URm(0));

        TVec veloRoe = (sqrtRhoLm * veloLm + sqrtRhoRm * veloRm) / (sqrtRhoLm + sqrtRhoRm);
        real vsqrRoe = veloRoe.squaredNorm();
        real HRoe = (sqrtRhoLm * HLm + sqrtRhoRm * HRm) / (sqrtRhoLm + sqrtRhoRm);
        real asqrRoe = (gamma - 1) * (HRoe - 0.5 * vsqrRoe);
        real rhoRoe = sqrtRhoLm * sqrtRhoRm;

        Eigen::Vector<real, dim + 2> FL, FR;
        GasInviscidFlux_XY<dim>(UL, veloL, vg, n, pL, FL);
        GasInviscidFlux_XY<dim>(UR, veloR, vg, n, pR, FR);

        if (!(asqrRoe > 0))
        {
            dumpInfo();
        }
        DNDS_assert(asqrRoe > 0);
        real aRoe = std::sqrt(asqrRoe);
        real veloRoeN = veloRoe.dot(n);
        real vgN = vg.dot(n);
        real veloRoe0 = veloRoeN - vgN;
        lam0 = std::abs(veloRoe0 - aRoe);
        lam123 = std::abs(veloRoe0);
        lam4 = std::abs(veloRoe0 + aRoe);
        real veloLm0 = (veloLm - vg).dot(n);
        real veloRm0 = (veloRm - vg).dot(n);

        auto HLLCq = [&](real p, real pS)
        {
            real q = std::sqrt(1 + (gamma + 1) / 2 / gamma * (pS / p - 1));
            if (pS <= p)
                q = 1;
            return q;
        };
        real pS = 0.5 * (pLm + pRm) - 0.5 * (veloLm0 - veloRm0) * rhoRoe * aRoe;
        pS = std::max(0.0, pS);
        real SL = veloLm0 - std::sqrt(asqrLm) * HLLCq(pLm, pS);
        real SR = veloRm0 + std::sqrt(asqrRm) * HLLCq(pRm, pS);

        dLambda += verySmallReal;
        dLambda *= 2.0;

        // * E-Fix
        // SL += sign(SL) * std::exp(-std::abs(SL) / dLambda) * dLambda;
        // SR += sign(SR) * std::exp(-std::abs(SR) / dLambda) * dLambda;

        if (0 <= SL)
        {
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FL;
            return;
        }
        if (SR <= 0)
        {
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FR;
            return;
        }
        real SS = 0;
        real div = (UL(0) * (SL - veloLm0) - UR(0) * (SR - veloRm0));
        if (std::abs(div) > verySmallReal)
            SS = (pR - pL +
                  (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(n) - vgN * UL(0)) * (SL - veloLm0) -
                  (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(n) - vgN * UR(0)) * (SR - veloRm0)) /
                 div;
        //! is this right for moving mesh?
        Eigen::Vector<real, dim + 2> DS;
        DS.setZero();
        DS(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) = n;
        DS(dim + 1) = SS;
        // SS += sign(SS) * std::exp(-std::abs(SS) / dLambda) * dLambda;
        if (SS >= 0)
        {
            real div = SL - SS;
            if (std::abs(div) < verySmallReal)
                F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FL;
            else
                F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                    ((UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * SL - FL) * SS +
                     DS * ((pL + UL(0) * (SL - veloLm0) * (SS - veloLm0)) * SL)) /
                    div;
        }
        else
        {
            real div = SR - SS;
            if (std::abs(div) < verySmallReal)
                F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FR;
            else
                F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                    ((UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * SR - FR) * SS +
                     DS * ((pR + UR(0) * (SR - veloRm0) * (SS - veloRm0)) * SR)) /
                    div;
        }
    }

    template <int dim = 3, int eigScheme = 0,
              typename TUL, typename TUR,
              typename TULm, typename TURm,
              typename TVecVG, typename TVecN,
              typename TF, typename TFdumpInfo>
    void RoeFlux_IdealGas_HartenYee(const TUL &UL, const TUR &UR,
                                    const TULm &ULm, const TURm &URm,
                                    const TVecVG &vg, const TVecN &n,
                                    real gamma, TF &F, real dLambda,
                                    const TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
    {
        static real scaleHartenYee = 0.05;
        static real scaleLD = 0.2;
        using TVec = Eigen::Vector<real, dim>;

        if (!(UL(0) > 0 && UR(0) > 0))
        {
            dumpInfo();
        }
        DNDS_assert(UL(0) > 0 && UR(0) > 0);
        TVec veloL = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UL(0)).matrix();
        TVec veloR = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UR(0)).matrix();
        TVec veloLm = (ULm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / ULm(0)).matrix();
        TVec veloRm = (URm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / URm(0)).matrix();

        real asqrL, asqrR, pL, pR, HL, HR;
        real vsqrL = veloL.squaredNorm();
        real vsqrR = veloR.squaredNorm();
        IdealGasThermal(UL(dim + 1), UL(0), vsqrL, gamma, pL, asqrL, HL);
        IdealGasThermal(UR(dim + 1), UR(0), vsqrR, gamma, pR, asqrR, HR);
        real asqrLm, asqrRm, pLm, pRm, HLm, HRm;
        real vsqrLm = veloLm.squaredNorm();
        real vsqrRm = veloRm.squaredNorm();
        IdealGasThermal(ULm(dim + 1), ULm(0), vsqrLm, gamma, pLm, asqrLm, HLm);
        IdealGasThermal(URm(dim + 1), URm(0), vsqrRm, gamma, pRm, asqrRm, HRm);

        real sqrtRhoLm = std::sqrt(ULm(0));
        real sqrtRhoRm = std::sqrt(URm(0));

        TVec veloRoe = (sqrtRhoLm * veloLm + sqrtRhoRm * veloRm) / (sqrtRhoLm + sqrtRhoRm);
        real vsqrRoe = veloRoe.squaredNorm();
        real HRoe = (sqrtRhoLm * HLm + sqrtRhoRm * HRm) / (sqrtRhoLm + sqrtRhoRm);
        real asqrRoe = (gamma - 1) * (HRoe - 0.5 * vsqrRoe);
        real rhoRoe = sqrtRhoLm * sqrtRhoRm;

        Eigen::Vector<real, dim + 2> FL, FR;
        GasInviscidFlux_XY<dim>(UL, veloL, vg, n, pL, FL);
        GasInviscidFlux_XY<dim>(UR, veloR, vg, n, pR, FR);

        if (!(asqrRoe > 0))
        {
            dumpInfo();
        }
        DNDS_assert(asqrRoe > 0);
        real aRoe = std::sqrt(asqrRoe);
        real veloRoeN = veloRoe.dot(n);
        real vgN = vg.dot(n);
        real veloRoe0 = veloRoeN - vgN;
        lam0 = std::abs(veloRoe0 - aRoe);
        lam123 = std::abs(veloRoe0);
        lam4 = std::abs(veloRoe0 + aRoe);
        real veloLm0 = (veloLm - vg).dot(n);
        real veloRm0 = (veloRm - vg).dot(n);

        if constexpr (eigScheme == 0)
        {
            //*HY
            real thresholdHartenYee = std::max(scaleHartenYee * (std::sqrt(vsqrRoe) + aRoe), dLambda);
            real thresholdHartenYeeS = thresholdHartenYee * thresholdHartenYee;
            if (lam0 < thresholdHartenYee)
                lam0 = (sqr(lam0) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            if (lam4 < thresholdHartenYee)
                lam4 = (sqr(lam4) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            //*HY
        }
        else if constexpr (eigScheme == 1)
        {
            //* cLLF
            real aLm = std::sqrt(asqrLm);
            real aRm = std::sqrt(asqrRm);
            real uLm = signTol(veloLm0, aLm * smallReal) * std::max(std::abs(veloLm0), aLm * scaleLD);
            real uRm = signTol(veloRm0, aRm * smallReal) * std::max(std::abs(veloRm0), aRm * scaleLD);
            lam0 = std::max(std::abs(uLm - aLm), std::abs(uRm - aRm));
            lam123 = std::max(std::abs(uLm), std::abs(uRm));
            lam4 = std::max(std::abs(uLm + aLm), std::abs(uRm + aRm));
        }
        else if constexpr (eigScheme == 2)
        {
            // *vanilla Lax
            // lam0 = lam123 = lam4 = std::max({lam0, lam123, lam4});
            lam0 = lam123 = lam4 = std::max(std::abs(veloLm0) + std::sqrt(asqrLm), std::abs(veloRm0) + std::sqrt(asqrRm));
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                (FL + FR) * 0.5 -
                0.5 * lam0 * (UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) - UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)));
            return; //* early exit
        }
        else if constexpr (eigScheme == 3)
        {
            //*LD, Roe_M
            /**
             * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
             */
            real LDthreshold = std::abs(veloRoe0) / scaleLD;
            real aRoeStar = std::min(LDthreshold, aRoe);
            lam0 = std::abs(veloRoe0 - aRoeStar);
            lam4 = std::abs(veloRoe0 + aRoeStar);
        }
        else if constexpr (eigScheme == 4)
        {
            //*ID, Roe_M
            /**
             * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
             */
#ifdef USE_SIGN_MINUS_AT_ROE_M4_FLUX
            real uStar = signM(veloRoe0) * std::max(aRoe * scaleLD, std::abs(veloRoe0));
#else
            real uStar = signTol(veloRoe0, aRoe * smallReal) * std::max(aRoe * scaleLD, std::abs(veloRoe0)); //! why signM here?
#endif
            lam0 = std::abs(uStar - aRoe);
            lam123 = std::abs(uStar);
            lam4 = std::abs(uStar + aRoe);
            // std::cout << "here" << std::endl;

            // real thresholdHartenYee = std::max(scaleLD * (std::sqrt(vsqrRoe) + aRoe), 0);
            // real thresholdHartenYeeS = thresholdHartenYee * thresholdHartenYee;
            // if (lam0 < thresholdHartenYee)
            //     lam0 = (sqr(lam0) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            // if (lam4 < thresholdHartenYee)
            //     lam4 = (sqr(lam4) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            // if (lam123 < thresholdHartenYee)
            //     lam123 = (sqr(lam123) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
        }
        else if constexpr (eigScheme == 5)
        {
            //*LD, cLLF_M
            /**
             * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
             */
            real aLm = std::min(std::sqrt(asqrLm), std::abs(veloLm0) / scaleLD);
            real aRm = std::min(std::sqrt(asqrRm), std::abs(veloRm0) / scaleLD);
            lam0 = std::max(std::abs(veloLm0 - aLm), std::abs(veloRm0 - aRm));
            lam123 = std::max(std::abs(veloLm0), std::abs(veloRm0));
            lam4 = std::max(std::abs(veloLm0 + aLm), std::abs(veloRm0 + aRm));
            //*LD, cLLF_M
        }
        else
        {
            DNDS_assert(false);
        }

        Eigen::Vector<real, dim + 2> incU =
            UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
            UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)); //! not using m, for this is accuracy-limited!
        real incU123N = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(n);

        TVec alpha23V = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) - incU(0) * veloRoe;
        TVec alpha23VT = alpha23V - n * alpha23V.dot(n);
        real incU4b = incU(dim + 1) - alpha23VT.dot(veloRoe);
        real alpha1 = (gamma - 1) / asqrRoe *
                      (incU(0) * (HRoe - veloRoeN * veloRoeN) +
                       veloRoeN * incU123N - incU4b);
        real alpha0 = (incU(0) * (veloRoeN + aRoe) - incU123N - aRoe * alpha1) / (2 * aRoe);
        real alpha4 = incU(0) - (alpha0 + alpha1);

        alpha0 *= lam0;
        alpha1 *= lam123;
        alpha23VT *= lam123;
        alpha4 *= lam4; // here becomes alpha_i * lam_i

        Eigen::Vector<real, dim + 2> incF;
        incF(0) = alpha0 + alpha1 + alpha4;
        incF(dim + 1) = (HRoe - veloRoeN * aRoe) * alpha0 + 0.5 * vsqrRoe * alpha1 +
                        (HRoe + veloRoeN * aRoe) * alpha4 + alpha23VT.dot(veloRoe);
        incF(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) =
            (veloRoe - aRoe * n) * alpha0 + (veloRoe + aRoe * n) * alpha4 +
            veloRoe * alpha1 + alpha23VT;

        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = (FL + FR) * 0.5 - 0.5 * incF;
    }

    template <int dim = 3, typename TUL, typename TUR, typename TVecV>
    void GetRoeAverage(const TUL &UL, const TUR &UR, real gamma,
                       TVecV &veloRoe, real &vsqrRoe, real &aRoe, real &asqrRoe, real &HRoe)
    {
        using TVec = Eigen::Vector<real, dim>;
        TVec veloLm = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UL(0)).matrix();
        TVec veloRm = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UR(0)).matrix();
        real asqrLm, asqrRm, pLm, pRm, HLm, HRm;
        real vsqrLm = veloLm.squaredNorm();
        real vsqrRm = veloRm.squaredNorm();
        IdealGasThermal(UL(dim + 1), UL(0), vsqrLm, gamma, pLm, asqrLm, HLm);
        IdealGasThermal(UR(dim + 1), UR(0), vsqrRm, gamma, pRm, asqrRm, HRm);
        DNDS_assert(UL(0) >= 0 && UR(0) >= 0);
        real sqrtRhoLm = std::sqrt(UL(0));
        real sqrtRhoRm = std::sqrt(UR(0));

        veloRoe = (sqrtRhoLm * veloLm + sqrtRhoRm * veloRm) / (sqrtRhoLm + sqrtRhoRm);
        vsqrRoe = veloRoe.squaredNorm();
        HRoe = (sqrtRhoLm * HLm + sqrtRhoRm * HRm) / (sqrtRhoLm + sqrtRhoRm);
        asqrRoe = (gamma - 1) * (HRoe - 0.5 * vsqrRoe);
        DNDS_assert(asqrRoe >= 0);
        aRoe = std::sqrt(asqrRoe);
    }

    template <int dim = 3, typename TDU, typename TDF, typename TVecV, typename TVecN>
    void RoeFluxIncFDiff(const TDU &incU, const TVecN &n, const TVecV &veloRoe,
                         real vsqrRoe, real aRoe, real asqrRoe, real HRoe,
                         real lam0, real lam123, real lam4, real gamma,
                         TDF &incF)
    {
        using TVec = Eigen::Vector<real, dim>;
        real veloRoeN = veloRoe.dot(n);

        real incU123N = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(n);

        TVec alpha23V = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) - incU(0) * veloRoe;
        TVec alpha23VT = alpha23V - n * alpha23V.dot(n);
        real incU4b = incU(dim + 1) - alpha23VT.dot(veloRoe);
        real alpha1 = (gamma - 1) / asqrRoe *
                      (incU(0) * (HRoe - veloRoeN * veloRoeN) +
                       veloRoeN * incU123N - incU4b);
        real alpha0 = (incU(0) * (veloRoeN + aRoe) - incU123N - aRoe * alpha1) / (2 * aRoe);
        real alpha4 = incU(0) - (alpha0 + alpha1);

        alpha0 *= lam0;
        alpha1 *= lam123;
        alpha23VT *= lam123;
        alpha4 *= lam4; // here becomes alpha_i * lam_i

        incF(0) += alpha0 + alpha1 + alpha4;
        incF(dim + 1) += (HRoe - veloRoeN * aRoe) * alpha0 + 0.5 * vsqrRoe * alpha1 +
                         (HRoe + veloRoeN * aRoe) * alpha4 + alpha23VT.dot(veloRoe);
        incF(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) +=
            (veloRoe - aRoe * n) * alpha0 + (veloRoe + aRoe * n) * alpha4 +
            veloRoe * alpha1 + alpha23VT;
    }

    template <int dim = 3, int eigScheme = 0,
              typename TUL, typename TUR,
              typename TULm, typename TURm,
              typename TVecVG, typename TVecVGm,
              typename TVecN, typename TVecNm,
              typename TF, typename TFdumpInfo>
    void RoeFlux_IdealGas_HartenYee_Batch(const TUL &UL, const TUR &UR,
                                          const TULm &ULm, const TURm &URm,
                                          const TVecVG &vg, const TVecVGm &vgm,
                                          const TVecN &n, const TVecNm &nm,
                                          real gamma, TF &F, real dLambda,
                                          const TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
    {
        static real scaleHartenYee = 0.05;
        static real scaleLD = 0.2;
        using TVec = Eigen::Vector<real, dim>;
        using TVec_Batch = Eigen::Matrix<real, dim, -1, Eigen::ColMajor, dim, MaxBatch>;
        using TReal_Batch = Eigen::Matrix<real, 1, -1, Eigen::RowMajor, 1, MaxBatch>;
        using TU5_Batch = Eigen::Matrix<real, dim + 2, -1, Eigen::ColMajor, dim + 2, MaxBatch>;

        int nB = UL.cols();
        for (int iB = 0; iB < nB; iB++)
            if (!(UL(0, iB) > 0 && UR(0, iB) > 0))
            {
                dumpInfo();
                DNDS_assert(false);
            }
        TVec_Batch veloL = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), Eigen::all).array().rowwise() / UL(0, Eigen::all).array()).matrix();
        TVec_Batch veloR = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), Eigen::all).array().rowwise() / UR(0, Eigen::all).array()).matrix();
        TVec veloLm = (ULm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / ULm(0)).matrix();
        TVec veloRm = (URm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / URm(0)).matrix();

        TReal_Batch pL, pR;
        pL.resize(nB), pR.resize(nB);
        for (int iB = 0; iB < nB; iB++)
        {
            real asqrL, asqrR, HL, HR;
            real vsqrL = veloL(Eigen::all, iB).squaredNorm();
            real vsqrR = veloR(Eigen::all, iB).squaredNorm();
            IdealGasThermal(UL(dim + 1, iB), UL(0, iB), vsqrL, gamma, pL(iB), asqrL, HL);
            IdealGasThermal(UR(dim + 1, iB), UR(0, iB), vsqrR, gamma, pR(iB), asqrR, HR);
        }

        real asqrLm, asqrRm, pLm, pRm, HLm, HRm;
        real vsqrLm = veloLm.squaredNorm();
        real vsqrRm = veloRm.squaredNorm();
        IdealGasThermal(ULm(dim + 1), ULm(0), vsqrLm, gamma, pLm, asqrLm, HLm);
        IdealGasThermal(URm(dim + 1), URm(0), vsqrRm, gamma, pRm, asqrRm, HRm);

        real sqrtRhoLm = std::sqrt(ULm(0));
        real sqrtRhoRm = std::sqrt(URm(0));

        TVec veloRoe = (sqrtRhoLm * veloLm + sqrtRhoRm * veloRm) / (sqrtRhoLm + sqrtRhoRm);
        real vsqrRoe = veloRoe.squaredNorm();
        real HRoe = (sqrtRhoLm * HLm + sqrtRhoRm * HRm) / (sqrtRhoLm + sqrtRhoRm);
        real asqrRoe = (gamma - 1) * (HRoe - 0.5 * vsqrRoe);
        real rhoRoe = sqrtRhoLm * sqrtRhoRm;
        real veloRoeN = veloRoe.dot(nm);
        real vgmN = vgm.dot(nm);
        real veloRoeRN = veloRoeN - vgmN;
        real veloLm0 = (veloLm - vgm).dot(nm);
        real veloRm0 = (veloRm - vgm).dot(nm);

        TU5_Batch FL, FR;
        FL.resize(Eigen::NoChange, UL.cols());
        FR.resize(Eigen::NoChange, UL.cols());
        GasInviscidFlux_XY_Batch<dim>(UL, veloL, vg, n, pL, FL);
        GasInviscidFlux_XY_Batch<dim>(UR, veloR, vg, n, pR, FR);

        if (!(asqrRoe > 0))
        {
            dumpInfo();
        }
        DNDS_assert(asqrRoe > 0);
        real aRoe = std::sqrt(asqrRoe);

        lam0 = std::abs(veloRoeRN - aRoe);
        lam123 = std::abs(veloRoeRN);
        lam4 = std::abs(veloRoeRN + aRoe);

        if constexpr (eigScheme == 0)
        {
            //*HY
            real thresholdHartenYee = std::max(scaleHartenYee * (std::sqrt(vsqrRoe) + aRoe), dLambda);
            real thresholdHartenYeeS = thresholdHartenYee * thresholdHartenYee;
            if (lam0 < thresholdHartenYee)
                lam0 = (sqr(lam0) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            if (lam4 < thresholdHartenYee)
                lam4 = (sqr(lam4) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            //*HY
        }
        else if constexpr (eigScheme == 1)
        {
            //* cLLF
            real aLm = std::sqrt(asqrLm);
            real aRm = std::sqrt(asqrRm);
            lam0 = std::max(std::abs(veloLm0 - aLm), std::abs(veloRm0 - aRm));
            lam123 = std::max(std::abs(veloLm0), std::abs(veloRm0));
            lam4 = std::max(std::abs(veloLm0 + aLm), std::abs(veloRm0 + aRm));
        }
        else if constexpr (eigScheme == 2)
        {
            // *vanilla Lax
            // lam0 = lam123 = lam4 = std::max({lam0, lam123, lam4});
            lam0 = lam123 = lam4 = std::max(std::abs(veloLm0) + std::sqrt(asqrLm), std::abs(veloRm0) + std::sqrt(asqrRm));
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), Eigen::all) =
                (FL + FR) * 0.5 -
                0.5 * lam0 * (UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), Eigen::all) - UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), Eigen::all));
            return; //* early exit
        }
        else if constexpr (eigScheme == 3)
        {
            //*LD, Roe_M
            /**
             * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
             */
            real LDthreshold = std::abs(veloRoeRN) / scaleLD;
            real aRoeStar = std::min(LDthreshold, aRoe);
            lam0 = std::abs(veloRoeRN - aRoeStar);
            lam4 = std::abs(veloRoeRN + aRoeStar);
        }
        else if constexpr (eigScheme == 4)
        {
            //*ID, Roe_M
            /**
             * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
             */
#ifdef USE_SIGN_MINUS_AT_ROE_M4_FLUX
            real uStar = signM(veloRoeRN) * std::max(aRoe * scaleLD, std::abs(veloRoeRN));
#else
            real uStar = signTol(veloRoeRN, aRoe * smallReal) * std::max(aRoe * scaleLD, std::abs(veloRoeRN)); //! why signM here?
#endif
            lam0 = std::abs(uStar - aRoe);
            lam123 = std::abs(uStar);
            lam4 = std::abs(uStar + aRoe);
        }
        else if constexpr (eigScheme == 5)
        {
            //*LD, cLLF_M
            /**
             * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
             */
            real aLm = std::min(std::sqrt(asqrLm), std::abs(veloLm0) / scaleLD);
            real aRm = std::min(std::sqrt(asqrRm), std::abs(veloRm0) / scaleLD);
            lam0 = std::max(std::abs(veloLm0 - aLm), std::abs(veloRm0 - aRm));
            lam123 = std::max(std::abs(veloLm0), std::abs(veloRm0));
            lam4 = std::max(std::abs(veloLm0 + aLm), std::abs(veloRm0 + aRm));
            //*LD, cLLF_M
        }
        else
        {
            DNDS_assert(false);
        }

        TU5_Batch incU =
            UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), Eigen::all) -
            UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), Eigen::all); //! not using m, for this is accuracy-limited!
        TReal_Batch incU123N =
            (incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), Eigen::all).array() * n.array()).colwise().sum();
        TVec_Batch alpha23V = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), Eigen::all) - veloRoe * incU(0, Eigen::all);
        TVec_Batch alpha23VT = alpha23V.array() - n.array().rowwise() * (alpha23V.array() * n.array()).colwise().sum();
        TReal_Batch incU4b =
            incU(dim + 1, Eigen::all) -
            veloRoe.transpose() * alpha23VT;
        TReal_Batch alpha1 =
            (gamma - 1) / asqrRoe *
            (incU(0, Eigen::all) * (HRoe - veloRoeN * veloRoeN) +
             veloRoeN * incU123N - incU4b);
        TReal_Batch alpha0 =
            (incU(0, Eigen::all) * (veloRoeN + aRoe) - incU123N - aRoe * alpha1) / (2 * aRoe);
        TReal_Batch alpha4 =
            incU(0, Eigen::all) - (alpha0 + alpha1);

        alpha0 *= lam0;
        alpha1 *= lam123;
        alpha23VT *= lam123;
        alpha4 *= lam4; // here becomes alpha_i * lam_i

        TU5_Batch incF;
        incF.resize(Eigen::NoChange, UL.cols());
        incF(0, Eigen::all) = alpha0 + alpha1 + alpha4;
        incF(dim + 1, Eigen::all) = (HRoe - veloRoeN * aRoe) * alpha0 + 0.5 * vsqrRoe * alpha1 +
                                    (HRoe + veloRoeN * aRoe) * alpha4 + veloRoe.transpose() * alpha23VT;
        incF(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), Eigen::all) =
            ((-aRoe * n).array().colwise() + veloRoe.array()).rowwise() * alpha0.array() +
            ((aRoe * n).array().colwise() + veloRoe.array()).rowwise() * alpha4.array() +
            alpha23VT.array() +
            (veloRoe * alpha1).array();
        // incF(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), Eigen::all) =
        //     (veloRoe.array() - (aRoe * n).array().colwise()) * alpha0 * (veloRoe.array() + (aRoe * n).array().colwise()) * alpha4;

        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), Eigen::all) = (FL + FR) * 0.5 - 0.5 * incF;
    }

    template <int dim = 3,
              typename TUL, typename TUR,
              typename TULm, typename TURm,
              typename TVecVG, typename TVecN,
              typename TF, typename TFdumpInfo>
    void InviscidFlux_IdealGas_Dispatcher(
        RiemannSolverType type,
        TUL &&UL, TUR &&UR, TULm &&ULm, TURm &&URm,
        TVecVG &&vg, TVecN &&n, real gamma, TF &&F,
        real dLambda,
        TFdumpInfo &&dumpInfo, real &lam0, real &lam123, real &lam4)
    {
        if (type == Roe)
            RoeFlux_IdealGas_HartenYee<dim>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M1)
            RoeFlux_IdealGas_HartenYee<dim, 1>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M2)
            RoeFlux_IdealGas_HartenYee<dim, 2>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M3)
            RoeFlux_IdealGas_HartenYee<dim, 3>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M4)
            RoeFlux_IdealGas_HartenYee<dim, 4>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M5)
            RoeFlux_IdealGas_HartenYee<dim, 5>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == HLLEP)
            HLLEPFlux_IdealGas<dim, 0>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == HLLEP_V1)
            HLLEPFlux_IdealGas<dim, 1>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == HLLC)
            HLLCFlux_IdealGas_HartenYee<dim>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else
            DNDS_assert_info(false, "the rs type is invalid");
    }

    template <int dim = 3,
              typename TUL, typename TUR,
              typename TULm, typename TURm,
              typename TVecVG, typename TVecVGm,
              typename TVecN, typename TVecNm,
              typename TF, typename TFdumpInfo>
    void InviscidFlux_IdealGas_Batch_Dispatcher(
        RiemannSolverType type,
        TUL &&UL, TUR &&UR,
        TULm &&ULm, TURm &&URm,
        TVecVG &&vg, TVecVGm &&vgm,
        TVecN &&n, TVecNm &&nm,
        real gamma, TF &&F, real dLambda,
        TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
    {
        if (type == Roe)
            RoeFlux_IdealGas_HartenYee_Batch<dim>(
                UL, UR, ULm, URm, vg, vgm, n, nm, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M1)
            RoeFlux_IdealGas_HartenYee_Batch<dim, 1>(
                UL, UR, ULm, URm, vg, vgm, n, nm, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M2)
            RoeFlux_IdealGas_HartenYee_Batch<dim, 2>(
                UL, UR, ULm, URm, vg, vgm, n, nm, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M3)
            RoeFlux_IdealGas_HartenYee_Batch<dim, 3>(
                UL, UR, ULm, URm, vg, vgm, n, nm, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M4)
            RoeFlux_IdealGas_HartenYee_Batch<dim, 4>(
                UL, UR, ULm, URm, vg, vgm, n, nm, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M5)
            RoeFlux_IdealGas_HartenYee_Batch<dim, 5>(
                UL, UR, ULm, URm, vg, vgm, n, nm, gamma, F, dLambda,
                dumpInfo, lam0, lam123, lam4);
        else
            DNDS_assert_info(false, "the rs type is invalid (for batch version)");
    }

    /**
     * @brief 3x5 TGradU and TFlux
     * GradU is grad of conservatives
     *
     */
    template <int dim = 3, typename TU, typename TGradU, typename TFlux, typename TNorm>
    void ViscousFlux_IdealGas(const TU &U, const TGradU &GradUPrim, TNorm norm, bool adiabatic, real gamma,
                              real mu, real mutRatio, bool mutQCRFix, real k, real Cp, TFlux &Flux)
    {
        static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);

        Eigen::Vector<real, dim> velo = U(Seq123) / U(0);
        static const real lambda = -2. / 3.;
        Eigen::Matrix<real, dim, dim> diffVelo = GradUPrim(Seq012, Seq123); // dU_j/dx_i
        Eigen::Vector<real, dim> GradP = GradUPrim(Seq012, dim + 1);
        real vSqr = velo.squaredNorm();
        real p = (gamma - 1) * (U(dim + 1) - U(0) * 0.5 * vSqr);
        Eigen::Vector<real, dim> GradT = (gamma / ((gamma - 1) * Cp * U(0) * U(0))) *
                                         (U(0) * GradP - p * GradUPrim(Seq012, 0)); // GradU(:,0) is grad rho no matter prim or not

        if (adiabatic) //! is this fix reasonable?
            GradT -= GradT.dot(norm) * norm;

        Eigen::Matrix<real, dim, dim> vStress = (diffVelo + diffVelo.transpose()) * mu +
                                                Eigen::Matrix<real, dim, dim>::Identity() * (lambda * mu * diffVelo.trace());
        if (mutQCRFix)
        {
            real b = std::sqrt((diffVelo.array() * diffVelo.array()).sum());
            Eigen::Matrix<real, dim, dim> O = (diffVelo.transpose() - diffVelo) / (b + verySmallReal); // dU_i/dx_j-dU_j/dx_i
            real ccr1 = 0.3;
            Eigen::Matrix<real, dim, dim> vStressQCRFix;
            vStressQCRFix.setZero();
            vStressQCRFix.diagonal() = (vStress.array() * O.array()).rowwise().sum();
            vStressQCRFix(0, 1) = O(0, 1) * (vStress(1, 1) - vStress(0, 0));
            if (dim == 3)
            {
                vStressQCRFix(0, 1) += O(1, 2) * vStress(0, 2) + O(0, 2) * vStress(1, 2);
                vStressQCRFix(0, 2) = O(0, 2) * (vStress(2, 2) - vStress(0, 0)) + O(0, 1) * vStress(2, 1) + O(2, 1) * vStress(0, 1);
                vStressQCRFix(1, 2) = O(1, 2) * (vStress(2, 2) - vStress(1, 1)) + O(1, 0) * vStress(2, 0) + O(2, 0) * vStress(1, 0);
            }
            Eigen::Matrix<real, dim, dim> vStressQCRFixFull = vStressQCRFix + vStressQCRFix.transpose();
            vStress -= ccr1 * mutRatio * vStressQCRFixFull;
        }
        Flux(0) = 0;
        Flux(Seq123) = vStress * norm;
        Flux(dim + 1) = (vStress * velo + k * GradT).dot(norm);
    }

    /**
     * @brief 3x5 TGradU
     * GradU is grad of conservatives
     *
     */
    template <int dim = 3, typename TU, typename TGradU, typename TGradUPrim>
    void GradientCons2Prim_IdealGas(const TU &U, const TGradU &GradU, TGradUPrim &GradUPrim, real gamma)
    {
        static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto I4 = dim + 1;

        Eigen::Vector<real, dim> velo = U(Seq123) / U(0);
        GradUPrim = GradU;

        GradUPrim(Seq012, Seq123) = (1.0 / sqr(U(0))) *
                                    (U(0) * GradU(Seq012, Seq123) -
                                     GradU(Seq012, 0) * Eigen::RowVector<real, dim>(U(Seq123))); // dU_j/dx_i
        GradUPrim(Seq012, I4) = (gamma - 1) *
                                (GradU(Seq012, dim + 1) -
                                 0.5 *
                                     (GradU(Seq012, Seq123) * velo +
                                      GradUPrim(Seq012, Seq123) * Eigen::Vector<real, dim>(U(Seq123))));
        GradUPrim(Seq012, Eigen::seq(Eigen::fix<I4 + 1>, Eigen::last)) -= GradU(Seq012, 0) * U(Eigen::seq(Eigen::fix<I4 + 1>, Eigen::last)).transpose() / U(0);
        GradUPrim(Seq012, Eigen::seq(Eigen::fix<I4 + 1>, Eigen::last)) /= U(0);
    }

    template <int dim, typename TU, typename TGradU>
    auto GetGradVelo(const TU &U, const TGradU &GradU)
    {
        static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        Eigen::Matrix<real, dim, dim> diffVelo = (1.0 / sqr(U(0))) *
                                                 (U(0) * GradU(Seq012, Seq123) -
                                                  GradU(Seq012, 0) * Eigen::RowVector<real, dim>(U(Seq123))); // dU_j/dx_i
        return diffVelo;
    }

    /**
     * TODO: vectorize
     * newrhoEinteralNew is the desired fixed-to-positive e = p / (gamma -1)
     */
    template <int dim = 3, int scheme = 0, int nVarsFixed, typename TU, typename TUInc>
    real IdealGasGetCompressionRatioPressure(const TU &u, const TUInc &uInc, real newrhoEinteralNew)
    {
        static const real safetyRatio = 1 - 1e-5;
        static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto I4 = dim + 1;

        Eigen::Vector<real, nVarsFixed> ret = uInc;
        Eigen::Vector<real, nVarsFixed> uNew = u + uInc;
        real rhoEOld = u(I4) - u(Seq123).squaredNorm() / (u(0) + verySmallReal) * 0.5;
        newrhoEinteralNew = std::max(smallReal * rhoEOld, newrhoEinteralNew);
        real rhoENew = uNew(I4) - uNew(Seq123).squaredNorm() / (uNew(0) + verySmallReal) * 0.5;
        real alphaEst1 = (rhoEOld - newrhoEinteralNew) / std::max(-rhoENew + rhoEOld, verySmallReal);
        if (rhoENew > rhoEOld)
            alphaEst1 = 1;
        alphaEst1 = std::min(alphaEst1, 1.);
        alphaEst1 = std::max(alphaEst1, 0.);
        real alpha = alphaEst1; //! using convex estimation

        real alphaL, alphaR, c0, c1, c2;
        alphaL = alphaR = c0 = c1 = c2 = 0;
        if constexpr (scheme == 0)
        {
            c0 = 2 * u(I4) * u(0) - u(Seq123).squaredNorm() - 2 * u(0) * newrhoEinteralNew;
            c1 = 2 * u(I4) * ret(0) + 2 * u(0) * ret(I4) - 2 * u(Seq123).dot(ret(Seq123)) - 2 * ret(0) * newrhoEinteralNew;
            c2 = 2 * ret(I4) * ret(0) - ret(Seq123).squaredNorm();
            c2 += signP(c2) * verySmallReal;
            real deltaC = sqr(c1) - 4 * c0 * c2;
            if (deltaC <= -sqr(c0) * smallReal)
            {
                std::cout << std::scientific << std::setprecision(5);
                std::cout << u.transpose() << std::endl;
                std::cout << uInc.transpose() << std::endl;
                std::cout << newrhoEinteralNew << std::endl;
                std::cout << fmt::format("{} {} {}", c0, c1, c2) << std::endl;

                DNDS_assert(false);
            }
            deltaC = std::max(0., deltaC);
            real alphaL = (-std::sqrt(deltaC) - c1) / (2 * c2);
            real alphaR = (std::sqrt(deltaC) - c1) / (2 * c2);
            // if (c2 > 0)
            //     DNDS_assert(alphaL > 0);
            // DNDS_assert(alphaR > 0);
            // DNDS_assert(alphaL < 1);
            // if (c2 < 0)
            //     DNDS_assert(alphaR < 1);

            if (std::abs(c2) < 1e-10 * c0)
            {
                if (std::abs(c1) < 1e-10 * c0)
                {
                    alpha = 0;
                }
                else
                {
                    alpha = std::min(-c0 / c1, 1.);
                }
            }
            else
            {
                alpha = std::min((c2 > 0 ? alphaL : alphaL), 1.);
            }
            alpha = std::max(0., alpha);
            alpha *= safetyRatio;
            if (alpha < smallReal)
                alpha = 0;
        }
        else if constexpr (scheme == 1)
        {
            // has used convex
            alpha *= safetyRatio;
            if (alpha < smallReal)
                alpha = 0;
        }

        ret *= alpha;

        // Eigen::Vector<real, nVarsFixed> uNew = u + ret;
        // real eNew = uNew(I4) - 0.5 * uNew(Seq123).squaredNorm() / uNew(0);

        real decay = 1 - 1e-2;
        int iter;
        for (iter = 0; iter < 1000; iter++)
        {
            real ek = 0.5 * (u(Seq123) + ret(Seq123)).squaredNorm() / (u(0) + ret(0) + verySmallReal);
            if (ret(I4) + u(I4) - ek < newrhoEinteralNew)
            {

                ret *= decay, alpha *= decay;
            }
            else
                break;
        }
        if (iter >= 1000)
        {
            real ek = 0.5 * (u(Seq123) + ret(Seq123)).squaredNorm() / (u(0) + ret(0) + verySmallReal);
            {
                std::cout << std::scientific << std::setprecision(5);
                std::cout << fmt::format("alphas: {}, {}, {}\n", alpha, alphaL, alphaR);
                std::cout << fmt::format("ABC: {}, {}, {}\n", c2, c1, c0);
                std::cout << u.transpose() << std::endl;
                std::cout << uInc.transpose() << std::endl;
                std::cout << fmt::format("eks: {} {}\n", ret(I4) + u(I4) - ek, newrhoEinteralNew);
                DNDS_assert(false);
            }
        }
        // std::cout << fmt::format("{} {} {} {} {}", c0, c1, c2, alphaL, alphaR) << std::endl;

        return alpha;
    }
}