#pragma once

#include "Euler.hpp"

namespace DNDS::Euler::RANS
{
    template <int dim, class TU, class TDiffU>
    real GetMut_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real muf)
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
    void GetVisFlux_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real mut, real muPhy, TVFlux &vFlux)
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
    void GetSource_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, TSource &source)
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

        source(I4 + 1) = Pk - UMeanXy(I4 + 2);
        source(I4 + 2) = (ce1 * Pk - ce2 * UMeanXy(I4 + 2) + E) / Tt;
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
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal_4;
        real omegaaa = UMeanXy(I4 + 2) / rho + verySmallReal_4;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real nuPhy = muf / rho;
        real F2 = std::tanh(sqr(std::max(2 * std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa))));
        real mut = a1 * k / std::max(S * F2, a1 * omegaaa) * rho;

        if (std::isnan(mut) || !std::isfinite(mut))
        {
            std::cerr << k << " " << omegaaa << " " << mut << " " << S << "\n";
            std::cerr << SS << std::endl;
            DNDS_assert(false);
        }
        return mut;
    }

    template <int dim, class TU, class TDiffU, class TVFlux>
    void GetVisFlux_SST(TU &&UMeanXy, TDiffU &&DiffUxy, real mut, real d, real muf, TVFlux &vFlux)
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
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        Eigen::Matrix<real, dim, 2> diffRhoKO = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKO = (diffRhoKO - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal_4;
        real omegaaa = UMeanXy(I4 + 2) / rho + verySmallReal_4;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real nuPhy = muf / rho;
        real CDKW = std::max(2 * rho * sigO2 / omegaaa * diffKO(Eigen::all, 0).dot(diffKO(Eigen::all, 1)), 1e-10);
        real F1 = std::tanh(std::pow(
            std::min(std::max(std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa)),
                     4 * rho * sigO2 * k / (CDKW * sqr(d))),
            4));
        real F2 = std::tanh(sqr(std::max(2 * std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa))));
        // real mut = a1 * k / std::max(S * F2, a1 * omegaaa) * rho;

        real sigK = sigK1 * F1 + sigK2 * (1 - F1);
        real sigO = sigO1 * F1 + sigO2 * (1 - F1);

        vFlux(Seq012, {I4 + 1}) = diffKO(Seq012, 0) * (muf + mut * sigK);
        vFlux(Seq012, {I4 + 2}) = diffKO(Seq012, 1) * (muf + mut * sigO);
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
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        Eigen::Matrix<real, dim, 2> diffRhoKO = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKO = (diffRhoKO - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal_4;
        real omegaaa = UMeanXy(I4 + 2) / rho + verySmallReal_4;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real nuPhy = muf / rho;
        real CDKW = std::max(2 * rho * sigO2 / omegaaa * diffKO(Eigen::all, 0).dot(diffKO(Eigen::all, 1)), 1e-10);
        real F1 = std::tanh(std::pow(
            std::min(std::max(std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa)),
                     4 * rho * sigO2 * k / (CDKW * sqr(d))),
            4));
        real F2 = std::tanh(sqr(std::max(2 * std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa))));
        real mut = a1 * k / std::max(S * F2, a1 * omegaaa) * rho;
        real nutHat = std::max(mut / rho, 1e-8);

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * diffU.array()).sum();
        real PkTilde = std::min(Pk, 10 * betaStar * rho * k * omegaaa);

        real gammaC = gamma1 * F1 + gamma1 * (1 - F1);
        real sigK = sigK1 * F1 + sigK2 * (1 - F1);
        real sigO = sigO1 * F1 + sigO2 * (1 - F1);
        real beta = beta1 * F1 + beta2 * (1 - F1);

        if (mode == 0)
        {
            source(I4 + 1) = PkTilde - betaStar * rho * k * omegaaa;
            source(I4 + 2) = gammaC / nutHat * Pk - beta * rho * sqr(omegaaa) +
                             2 * (1 - F1) * rho * sigO2 / omegaaa * diffKO(Eigen::all, 0).dot(diffKO(Eigen::all, 1));
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

        real alpha = 5. / 9.;
        real beta = 3. / 40.;
        real betaS = 0.09;
        real sigK = 0.5;
        real sigO = 0.5;
        real Prt = 0.9;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        Eigen::Matrix<real, dim, 2> diffRhoKO = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKO = (diffRhoKO - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal_4;
        real omegaaa = UMeanXy(I4 + 2) / rho + verySmallReal_4;
        real mut = k / omegaaa * rho;

        if (std::isnan(mut) || !std::isfinite(mut))
        {
            std::cerr << k << " " << omegaaa << " " << mut << "\n";
            std::cerr << SS << std::endl;
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

        real alpha = 5. / 9.;
        real beta = 3. / 40.;
        real betaS = 0.09;
        real sigK = 0.5;
        real sigO = 0.5;
        real Prt = 0.9;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        Eigen::Matrix<real, dim, 2> diffRhoKO = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKO = (diffRhoKO - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal_4;
        real omegaaa = UMeanXy(I4 + 2) / rho + verySmallReal_4;
        real mut = k / omegaaa * rho;
        

        vFlux(Seq012, {I4 + 1}) = diffKO(Seq012, 0) * (muf + mutIn * sigK);
        vFlux(Seq012, {I4 + 2}) = diffKO(Seq012, 1) * (muf + mutIn * sigO);
    }

    template <int dim, class TU, class TDiffU, class TSource>
    void GetSource_KOWilcox(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d, TSource &source, int mode)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real alpha = 5./9.;
        real beta = 3./40.;
        real betaS = 0.09;
        real sigK = 0.5;
        real sigO = 0.5;
        real Prt = 0.9;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        Eigen::Matrix<real, dim, 2> diffRhoKO = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKO = (diffRhoKO - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal_4;
        real omegaaa = UMeanXy(I4 + 2) / rho + verySmallReal_4;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real nuPhy = muf / rho;

        real mut = k / omegaaa * rho;
        real nutHat = std::max(mut / rho, 1e-8);

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * SS.array()).sum() * 0.5;


        if (mode == 0)
        {
            source(I4 + 1) = Pk - betaS * rho * k * omegaaa;
            source(I4 + 2) = alpha * omegaaa / k * Pk - beta * rho * sqr(omegaaa);
        }
        else
        {
            source(I4 + 1) = betaS * omegaaa;
            source(I4 + 2) = 2 * beta * omegaaa;
        }
    }
}