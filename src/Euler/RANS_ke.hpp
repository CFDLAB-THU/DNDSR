#pragma once

#include "Euler.hpp"

#define KE_LIMIT_MUT 1

#define KW_WILCOX_VER 1
#define KW_WILCOX_PROD_LIMITS 1
#define KW_WILCOX_LIMIT_MUT 1

#define KW_SST_LIMIT_MUT 1
#define KW_SST_PROD_LIMITS 1
#define KW_SST_PROD_OMEGA_VERSION 1

namespace DNDS::Euler::RANS
{
    template <int dim, class TU, class TDiffU>
    real GetMut_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);
        real cmu = 0.09;
        real phi = 2. / 3.;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal_4;
        real epsilon = UMeanXy(I4 + 2) / rho + verySmallReal_4;
        real Ret = rho * sqr(k) / (muf * epsilon) + smallReal;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real fmu = (1 - std::exp(-0.01 * Ret)) / (1 - exp(-std::sqrt(Ret))) * std::max(1., std::sqrt(2 / Ret));
        real mut = cmu * fmu * rho * sqr(k) / epsilon;
        mut = std::min(mut, phi * rho * k / S);
#if KE_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif
        if (std::isnan(mut) || !std::isfinite(mut))
        {
            std::cerr << k << " " << epsilon << " " << Ret << " " << S << "\n";
            std::cerr << fmu << "\n";
            std::cerr << SS << std::endl;
            DNDS_assert(false);
        }
        return mut;
    }

    template <int dim, class TU, class TDiffU, class TVFlux>
    void GetVisFlux_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real mut, real d, real muPhy, TVFlux &vFlux)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        real sigK = 1.;
        real sigE = 1.3;
        Eigen::Matrix<real, dim, 2> diffRhoKe = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, 2> diffKe = (diffRhoKe - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        vFlux(Seq012, {I4 + 1}) = diffKe(Seq012, 0) * (muPhy + mut / sigK);
        vFlux(Seq012, {I4 + 2}) = diffKe(Seq012, 1) * (muPhy + mut / sigE);
    }

    template <int dim, class TU, class TDiffU, class TSource>
    void GetSource_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d, TSource &source, int mode)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real cmu = 0.09;
        real phi = 2. / 3.;
        real ce1 = 1.44;
        real ce2 = 1.92;
        real AE = 0.3;
        real pphi = 1.065;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, verySmallReal_4);
        real epsilon = std::max(UMeanXy(I4 + 1) / rho, verySmallReal_4);
        real Ret = rho * sqr(k) / (muf * epsilon) + verySmallReal_4;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real fmu = (1 - std::exp(-0.01 * Ret)) / std::max(1 - exp(-std::sqrt(Ret)), verySmallReal_4) * std::max(1., std::sqrt(2 / Ret));
        real mut = cmu * fmu * rho * sqr(k) / epsilon;
        mut = std::min(mut, phi * rho * k / S);

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * diffU.array()).sum();
        Pk = std::min(Pk, pphi * cmu * sqr(UMeanXy(I4 + 1)) / mut);
        real zeta = std::sqrt(Ret / 2);
        real Tt = k / epsilon * std::max(1., 1. / zeta) + smallReal;

        Eigen::Matrix<real, dim, 2> diffRhoKe = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKe = (diffRhoKe - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffTau = diffKe(Seq012, 0) / epsilon - k / sqr(epsilon) * diffKe(Seq012, 1);
        real Psi = std::max(0., diffKe(Seq012, 0).dot(diffTau));
        real E = AE * rho * std::sqrt(epsilon * Tt) * Psi * std::max(std::sqrt(k), std::pow(muf * epsilon / rho, 0.25));

        if (mode == 0)
        {
            source(I4 + 1) = Pk - UMeanXy(I4 + 2);
            source(I4 + 2) = (ce1 * Pk - ce2 * UMeanXy(I4 + 2) + E) / Tt;
        }
        else
        {
            source(I4 + 1) = 0;
            source(I4 + 2) = (ce2) / Tt;
        }
        if (!source.allFinite() || source.hasNaN())
        {
            std::cerr << source.transpose() << "\n";
            std::cerr << UMeanXy.transpose() << "\n";
            std::cerr << DiffUxy << "\n";
            std::cerr << S << "\n";
            std::cerr << mut << "\n";

            std::cout << std::endl;

            DNDS_assert(false);
        }
    }

    template <int dim, class TU>
    std::tuple<real, real> SolveZeroGradEquilibrium(TU &u, real muPhy) // this is tested to be not applicable!
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        Eigen::Matrix<real, dim, 7> diffU;
        diffU.setZero();

        Eigen::Vector<real, Eigen::Dynamic> src = u;
        Eigen::Vector<real, Eigen::Dynamic> uc = u;
        src.setZero();

        real distEps = std::sqrt(smallReal);
        real distV = 1 + distEps;

        auto getDE = [&](real re) -> real
        {
            uc(I4 + 2) = re;
            GetSource_RealizableKe<dim>(uc, diffU, muPhy, src);
            return src(I4 + 1);
        };
        std::cout << "Mu " << muPhy << std::endl;

        real re = u(I4 + 2);
        real rhs0{0}, rhs{0};
        for (int i = 0; i < 1000; i++)
        {
            rhs = getDE(re);
            real dRhsDRe = (getDE(re * distV) - rhs) / (re * distEps);
            real ren = re - rhs / (dRhsDRe + verySmallReal);
            re = std::max(ren, verySmallReal);
            std::cout << rhs << std::endl;
            if (i == 0)
                rhs0 = rhs;
        }
        u(I4 + 2) = re;
        return std::make_tuple(rhs0, rhs);
    }

    template <int dim, class TU, class TDiffU, class TSource>
    void GetSourceJacobianDiag_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, TSource &source)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;

        real cmu = 0.09;
        real phi = 2. / 3.;
        real ce1 = 1.44;
        real ce2 = 1.92;
        real AE = 0.3;
        real pphi = 1.065;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal;
        real epsilon = UMeanXy(I4 + 2) / rho + verySmallReal;
        real Ret = rho * sqr(k) / (muf * epsilon) + verySmallReal;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal;
        real fmu = (1 - std::exp(-0.01 * Ret)) / (1 - exp(-std::sqrt(Ret))) * std::max(1., std::sqrt(2 / Ret));
        real mut = cmu * fmu * rho * sqr(k) / epsilon;
        mut = std::min(mut, phi * rho * k / S);

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * diffU.array()).sum();
        Pk = std::min(Pk, pphi * cmu * sqr(UMeanXy(I4 + 1)) / mut);
        real zeta = std::sqrt(Ret / 2);
        real Tt = k / epsilon * std::max(1., 1. / zeta) + smallReal;

        Eigen::Matrix<real, dim, 2> diffRhoKe = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKe = (diffRhoKe - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffTau = diffKe(Seq012, 0) / epsilon - k / sqr(epsilon) * diffKe(Seq012, 1);
        real Psi = std::max(0., diffKe(Seq012, 0).dot(diffTau));
        real E = AE * rho * std::sqrt(epsilon * Tt) * Psi * std::max(std::sqrt(k), std::pow(muf * epsilon / rho, 0.25));

        real dPk = -((Eigen::Matrix<real, dim, dim>::Identity() * (2. / 3.)).array() * diffU.array()).sum();

        source(I4 + 1) = -Pk / (UMeanXy(I4 + 1) + verySmallReal);
        source(I4 + 2) = (ce2) / Tt;
    }

    template <int dim, class TU, class TDiffU, class TSource>
    void GetSourceJacobianDiag_RealizableKe_ND(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, TSource &source)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        Eigen::VectorXd u0 = UMeanXy;
        Eigen::VectorXd u1 = UMeanXy;
        Eigen::VectorXd sb = source;
        Eigen::VectorXd s0 = source;
        Eigen::VectorXd s1 = source;
        real epsRk = (u0(I4 + 1) + smallReal) * std::sqrt(smallReal);
        real epsRe = (u0(I4 + 2) + smallReal) * std::sqrt(smallReal);
        u0(I4 + 1) += epsRk;
        u1(I4 + 2) += epsRe;
        GetSource_RealizableKe<dim>(UMeanXy, DiffUxy, muf, sb);
        GetSource_RealizableKe<dim>(u0, DiffUxy, muf, s0);
        GetSource_RealizableKe<dim>(u1, DiffUxy, muf, s1);
        source(I4 + 1) = -(s0(I4 + 1) - sb(I4 + 1)) / epsRk;
        source(I4 + 2) = -(s1(I4 + 2) - sb(I4 + 2)) / epsRe;
        source(I4 + 1) = std::max(source(I4 + 1), 0.) * 10;
        source(I4 + 2) = std::max(source(I4 + 2), 0.) * 10;
    }

    template <int dim, class TU, class TDiffU>
    real GetMut_SST(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real a1 = 0.31;
        real betaStar = 0.09;
        real sigOmega2 = 0.856;
        real cmu = 0.09;
        real phi = 2. / 3.;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();                                                  // 2 times strain rate
        Eigen::Matrix<real, dim, dim> SS = SR2 - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity(); // 2 times shear strain rate
        Eigen::Matrix<real, dim, dim> OmegaM = (diffU.transpose() - diffU) * 0.5;
        real OmegaMag = OmegaM.norm() * std::sqrt(2);
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, verySmallReal_4);
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real nuPhy = muf / rho;
        real F2 = std::tanh(sqr(std::max(2 * std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa))));
        // F2 = 0;
        real mut = a1 * k / std::max(OmegaMag * F2, a1 * omegaaa) * rho;
#if KW_SST_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif

        if (std::isnan(mut) || !std::isfinite(mut))
        {
            std::cerr << k << " " << omegaaa << " " << mut << " " << S << "\n";
            std::cerr << SS << std::endl;
            DNDS_assert(false);
        }
        return mut;
    }

    template <int dim, class TU, class TDiffU, class TVFlux>
    void GetVisFlux_SST(TU &&UMeanXy, TDiffU &&DiffUxy, real mutIn, real d, real muf, TVFlux &vFlux)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real a1 = 0.31;
        real betaStar = 0.09;
        real sigK1 = 0.85;
        real sigK2 = 1;
        real sigO1 = 0.5;
        real sigO2 = 0.856;
        real beta1 = 0.075;
        real beta2 = 0.0828;
        real kap = 0.41;
        real gamma1 = beta1 / betaStar - sigO1 * sqr(kap) / std::sqrt(betaStar);
        real gamma2 = beta2 / betaStar - sigO2 * sqr(kap) / std::sqrt(betaStar);
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();                                                  // 2 times strain rate
        Eigen::Matrix<real, dim, dim> SS = SR2 - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity(); // 2 times shear strain rate
        Eigen::Matrix<real, dim, 2> diffRhoKO = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKO = (diffRhoKO - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> OmegaM = (diffU.transpose() - diffU) * 0.5;
        real OmegaMag = OmegaM.norm() * std::sqrt(2);
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, verySmallReal_4);
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real nuPhy = muf / rho;
        real CDKW = std::max(2 * rho * sigO2 / omegaaa * diffKO(Eigen::all, 0).dot(diffKO(Eigen::all, 1)), 1e-10);
        real F1 = std::tanh(std::pow(
            std::min(std::max(std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa)),
                     4 * rho * sigO2 * k / (CDKW * sqr(d))),
            4));
        real F2 = std::tanh(sqr(std::max(2 * std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa))));
        // F2 = 0;
        real mut = a1 * k / std::max(OmegaMag * F2, a1 * omegaaa) * rho;
#if KW_SST_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif

        real sigK = sigK1 * F1 + sigK2 * (1 - F1);
        real sigO = sigO1 * F1 + sigO2 * (1 - F1);

        vFlux(Seq012, {I4 + 1}) = diffKO(Seq012, 0) * (muf + mutIn * sigK);
        vFlux(Seq012, {I4 + 2}) = diffKO(Seq012, 1) * (muf + mutIn * sigO);

        if (!vFlux.allFinite() || vFlux.hasNaN())
        {
            std::cerr << vFlux << "\n";
            std::cerr << sigK << " " << sigO << "\n";
            std::cerr << F1 << " " << F2 << "\n";
            std::cerr << CDKW << "\n";
            std::cerr << k << " " << omegaaa << "\n";
            std::cerr << muf << " " << mut << "\n";
            std::cerr << std::endl;
        }
    }

    template <int dim, class TU, class TDiffU, class TSource>
    void GetSource_SST(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d, TSource &source, int mode)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real a1 = 0.31;
        real betaStar = 0.09;
        real sigK1 = 0.85;
        real sigK2 = 1;
        real sigO1 = 0.5;
        real sigO2 = 0.856;
        real beta1 = 0.075;
        real beta2 = 0.0828;
        real kap = 0.41;
        real gamma1 = beta1 / betaStar - sigO1 * sqr(kap) / std::sqrt(betaStar);
        real gamma2 = beta2 / betaStar - sigO2 * sqr(kap) / std::sqrt(betaStar);
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();                                                  // 2 times strain rate
        Eigen::Matrix<real, dim, dim> SS = SR2 - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity(); // 2 times shear strain rate
        Eigen::Matrix<real, dim, 2> diffRhoKO = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKO = (diffRhoKO - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> OmegaM = (diffU.transpose() - diffU) * 0.5;
        real OmegaMag = OmegaM.norm() * std::sqrt(2);
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, verySmallReal_4);
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real nuPhy = muf / rho;
        real CDKW = std::max(2 * rho * sigO2 / omegaaa * diffKO(Eigen::all, 0).dot(diffKO(Eigen::all, 1)), 1e-10);
        real F1 = std::tanh(std::pow(
            std::min(std::max(std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa)),
                     4 * rho * sigO2 * k / (CDKW * sqr(d))),
            4));
        real F2 = std::tanh(sqr(std::max(2 * std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa))));
        // F2 = 0;
        real mut = a1 * k / std::max(OmegaMag * F2, a1 * omegaaa) * rho; // use S/OmegaMag for SST: S: CFD++, OmegaMag: Turbulence Modeling Validation, Testing, and Developmen
#if KW_SST_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif
        real nutHat = std::max(mut / rho, 1e-8);

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * diffU.array()).sum();
        real PkTilde = Pk;
#if KW_SST_PROD_LIMITS == 1
        PkTilde = std::max(PkTilde, verySmallReal);
        PkTilde = std::min(Pk, 20 * betaStar * rho * k * omegaaa); // CFD++'s limiting: 10 times
#endif

        real gammaC = gamma1 * F1 + gamma1 * (1 - F1);
        real sigK = sigK1 * F1 + sigK2 * (1 - F1);
        real sigO = sigO1 * F1 + sigO2 * (1 - F1);
        real beta = beta1 * F1 + beta2 * (1 - F1);
        real POmega = gammaC / nutHat * Pk;
#if KW_SST_PROD_OMEGA_VERSION == 1
        POmega =
            0.5 * gammaC * rho * ((SR2 - SR2.trace() / 3. * Eigen::Matrix<real, dim, dim>::Identity()).array() * SR2.array()).sum();
#elif KW_SST_PROD_OMEGA_VERSION == 2
        POmega =
            gammaC * rho * sqr(OmegaMag);
#endif
        if (mode == 0)
        {
            source(I4 + 1) = PkTilde - betaStar * rho * k * omegaaa;
            source(I4 + 2) = POmega - beta * rho * sqr(omegaaa) +
                             2 * (1 - F1) * rho * sigO2 / omegaaa * diffKO(Eigen::all, 0).dot(diffKO(Eigen::all, 1));
            // source(I4 + 2) = POmega - beta * rho * sqr(omegaaa) +
            //                  2 * (1 - F1) * rho * sigO2 / omegaaa * diffKO(Eigen::all, 0).dot(diffKO(Eigen::all, 1));
        }
        else
        {
            source(I4 + 1) = betaStar * omegaaa;
            source(I4 + 2) = 2 * beta * omegaaa;
        }
    }

    template <int dim, class TU, class TDiffU>
    real GetMut_KOWilcox(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real CLim = 7. / 8.;
        real betaS = 0.09;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, sqr(verySmallReal_4));
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
#if KW_WILCOX_VER == 0
        real omegaaaTut = omegaaa;
#else
        real omegaaaTut = std::max(omegaaa, CLim * std::sqrt(0.5 * SR2.squaredNorm() / betaS));
#endif
        real mut = k / omegaaaTut * rho;
#if KW_WILCOX_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif

        if (std::isnan(mut) || !std::isfinite(mut))
        {
            std::cerr << k << " " << omegaaa << " " << mut << "\n";
            DNDS_assert(false);
        }
        return mut;
    }

    template <int dim, class TU, class TDiffU, class TVFlux>
    void GetVisFlux_KOWilcox(TU &&UMeanXy, TDiffU &&DiffUxy, real mutIn, real d, real muf, TVFlux &vFlux)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real alpha = 13. / 25.;
        real betaS = 0.09;
        real sigK = 0.5;
        real sigO = 0.5;
        real Prt = 0.9;
        real CLim = 7. / 8.;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();                                                  // 2 times strain rate
        Eigen::Matrix<real, dim, dim> SS = SR2 - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity(); // 2 times shear strain rate
        Eigen::Matrix<real, dim, 2> diffRhoKO = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKO = (diffRhoKO - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, sqr(verySmallReal_4));
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
#if KW_WILCOX_VER == 0
        real omegaaaTut = omegaaa;
#else
        real omegaaaTut = std::max(omegaaa, CLim * std::sqrt(0.5 * SR2.squaredNorm() / betaS));
#endif
        real mut = k / omegaaaTut * rho;
#if KW_WILCOX_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif

        vFlux(Seq012, {I4 + 1}) = diffKO(Seq012, 0) * (muf + mut * sigK);
        vFlux(Seq012, {I4 + 2}) = diffKO(Seq012, 1) * (muf + mut * sigO);
    }

    template <int dim, class TU, class TDiffU, class TSource>
    void GetSource_KOWilcox(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d, TSource &source, int mode)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);
#if KW_WILCOX_VER == 0
        real alpha = 5. / 9.;
#else
        real alpha = 13. / 25.;
#endif
        real betaS = 0.09;
        real sigK = 0.5;
        real sigO = 0.5;
        real Prt = 0.9;
        real CLim = 7. / 8.;
        real betaO = 0.0708;
        real kappa = 0.41;

        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();                                                  // 2 times strain rate
        Eigen::Matrix<real, dim, dim> OmegaM2 = diffU.transpose() - diffU;                                              // 2 times rotation
        Eigen::Matrix<real, dim, dim> SS = SR2 - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity(); // 2 times shear strain rate
        Eigen::Matrix<real, dim, 2> diffRhoKO = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKO = (diffRhoKO - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, sqr(verySmallReal_4)); // make nu -> 0 when k,O->0
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
#if KW_WILCOX_VER == 0 || KW_WILCOX_VER == 2
        real omegaaaTut = omegaaa;
#else
        real omegaaaTut = std::max(omegaaa, CLim * std::sqrt(0.5 * SR2.squaredNorm() / betaS));
#endif
        real mut = k / omegaaaTut * rho;
#if KW_WILCOX_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif

        real ChiOmega = std::abs(((OmegaM2 * OmegaM2).array() * SR2.array()).sum() * 0.125 / cube(betaS * omegaaa));
        real fBeta = (1 + 85 * ChiOmega) / (1 + 100 * ChiOmega);
#if KW_WILCOX_VER == 0
        real beta = 3. / 40.;
#elif KW_WILCOX_VER == 1
        real beta = fBeta * betaO;
#else
        real beta = 0.075;
#endif
        real crossDiff = diffKO(Eigen::all, 0).dot(diffKO(Eigen::all, 1));
#if KW_WILCOX_VER == 0 || KW_WILCOX_VER == 2
        real SigD = 0;
#else
        real SigD = crossDiff > 0 ? 0.125 : 0;
#endif

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * diffU.array()).sum();
        real POmega = alpha * omegaaa / k * Pk;

#if KW_WILCOX_VER == 2
        Pk = OmegaM2.squaredNorm() * 0.5 * mut;
        real gam = beta / betaS - sqr(kappa) / (sigO * std::sqrt(betaS));
        POmega = gam * rho * OmegaM2.squaredNorm() * 0.5; // CFL3D ???
#endif

#if KW_WILCOX_PROD_LIMITS == 1
        // Pk = std::min(Pk, OmegaM2.squaredNorm() * 0.5 * mut); // compare with CFL3D approx

        Pk = std::max(Pk, verySmallReal);
        Pk = std::min(Pk, betaS * rho * k * omegaaa * 20); // CFL3D
#endif

        // POmega = std::min(POmega, beta * rho * sqr(omegaaa) * 20); // CFL3D

        if (mode == 0)
        {
            source(I4 + 1) = Pk - betaS * rho * k * omegaaa;
            source(I4 + 2) = POmega - beta * rho * sqr(omegaaa) + SigD / omegaaa * crossDiff;
        }
        else
        {
            source(I4 + 1) = betaS * omegaaa;
            source(I4 + 2) = 2 * beta * omegaaa;
        }
    }
}