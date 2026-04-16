#pragma once
#include "DNDS/Defines.hpp"

namespace DNDS::CFV
{
    /**
     * @brief Computes the weighted polynomial squared norm of each column of theta.
     *
     * The weights follow from the L2 inner product structure of the polynomial
     * basis: pure-order terms get weight 1, mixed terms get reduced weights
     * proportional to the combinatorial mixing of coordinate directions.
     *
     * For dim=2:
     *   nRows=2: theta(0)^2 + theta(1)^2
     *   nRows=3: theta(0)^2 + theta(1)^2 * 0.5 + theta(2)^2
     *   nRows=4: theta(0)^2 + theta(1)^2 * (1/3) + theta(2)^2 * (1/3) + theta(3)^2
     *
     * For dim=3:
     *   nRows=3:  uniform weight 1
     *   nRows=6:  first 3 weight 1, next 3 weight 0.5
     *   nRows=10: first 3 weight 1, next 6 weight 1/3, last 1 weight 1/6
     *
     * @tparam dim  Spatial dimension (2 or 3)
     * @param theta  Array of shape (nRows, nCols), normalized polynomial coefficients
     * @return Eigen::ArrayXd of shape (nCols) containing the weighted squared norms
     */
    template <int dim, typename ThetaArray>
    inline Eigen::ArrayXd PolynomialSquaredNorm(const ThetaArray &theta)
    {
        Eigen::ArrayXd result;
        if constexpr (dim == 2)
        {
            switch (theta.rows())
            {
            case 2:
                result =
                    theta(0, EigenAll).square() +
                    theta(1, EigenAll).square();
                break;
            case 3:
                result =
                    theta(0, EigenAll).square() +
                    theta(1, EigenAll).square() * 0.5 +
                    theta(2, EigenAll).square();
                break;
            case 4:
                result =
                    theta(0, EigenAll).square() +
                    theta(1, EigenAll).square() * (1. / 3.) +
                    theta(2, EigenAll).square() * (1. / 3.) +
                    theta(3, EigenAll).square();
                break;
            default:
                DNDS_assert(false);
                break;
            }
        }
        else
        {
            switch (theta.rows())
            {
            case 3:
                result =
                    theta(0, EigenAll).square() +
                    theta(1, EigenAll).square() +
                    theta(2, EigenAll).square();
                break;
            case 6:
                result =
                    theta(0, EigenAll).square() +
                    theta(1, EigenAll).square() +
                    theta(2, EigenAll).square() +
                    theta(3, EigenAll).square() * 0.5 +
                    theta(4, EigenAll).square() * 0.5 +
                    theta(5, EigenAll).square() * 0.5;
                break;
            case 10:
                result =
                    theta(0, EigenAll).square() +
                    theta(1, EigenAll).square() +
                    theta(2, EigenAll).square() +
                    theta(3, EigenAll).square() * (1. / 3.) +
                    theta(4, EigenAll).square() * (1. / 3.) +
                    theta(5, EigenAll).square() * (1. / 3.) +
                    theta(6, EigenAll).square() * (1. / 3.) +
                    theta(7, EigenAll).square() * (1. / 3.) +
                    theta(8, EigenAll).square() * (1. / 3.) +
                    theta(9, EigenAll).square() * (1. / 6.);
                break;
            default:
                DNDS_assert(false);
                break;
            }
        }
        return result;
    }

    /**
     * @brief Computes a weighted polynomial dot product between theta1 and theta2.
     *
     * Uses the same weight structure as PolynomialSquaredNorm<dim>.
     * Only used for dim=2 in FMEMM_Multiway_Polynomial2D.
     */
    template <int dim, typename ThetaArray1, typename ThetaArray2>
    inline Eigen::ArrayXd PolynomialDotProduct(const ThetaArray1 &theta1, const ThetaArray2 &theta2)
    {
        Eigen::ArrayXd result;
        static_assert(dim == 2, "PolynomialDotProduct only implemented for dim=2");
        switch (theta1.rows())
        {
        case 2:
            result =
                theta1(0, EigenAll) * theta2(0, EigenAll) +
                theta1(1, EigenAll) * theta2(1, EigenAll);
            break;
        case 3:
            result =
                theta1(0, EigenAll) * theta2(0, EigenAll) +
                theta1(1, EigenAll) * theta2(1, EigenAll) * 0.5 +
                theta1(2, EigenAll) * theta2(2, EigenAll);
            break;
        case 4:
            result =
                theta1(0, EigenAll) * theta2(0, EigenAll) +
                theta1(1, EigenAll) * theta2(1, EigenAll) * (1. / 3.) +
                theta1(2, EigenAll) * theta2(2, EigenAll) * (1. / 3.) +
                theta1(3, EigenAll) * theta2(3, EigenAll);
            break;
        default:
            DNDS_assert(false);
            break;
        }
        return result;
    }
    /**
     * @brief input vector<Eigen::Array-like>
     */
    template <typename TinOthers, typename Tout>
    static inline void FWBAP_L2_Multiway_Polynomial2D(const TinOthers &uOthers, int Nother, Tout &uOut, real n1 = 1)
    {
        using namespace DNDS;
        static const int p = 4;
        static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);

        Eigen::ArrayXXd uUp; //* copy!
        uUp.resizeLike(uOthers[0]);
        uUp.setZero();
        Eigen::ArrayXd uDown;
        uDown.resize(uOthers[0].cols());
        uDown.setZero();
        Eigen::ArrayXXd uMax = uUp + verySmallReal;
        for (int iOther = 0; iOther < Nother; iOther++)
            uMax = uMax.max(uOthers[iOther].abs());
        uMax.rowwise() = uMax.colwise().maxCoeff();
        uOut = uMax;

        for (int iOther = 0; iOther < Nother; iOther++)
        {
            Eigen::ArrayXd thetaNorm;
            Eigen::ArrayXXd theta = uOthers[iOther] / uMax;
            thetaNorm = PolynomialSquaredNorm<2>(theta);
            thetaNorm += verySmallReal_pDiP;
            thetaNorm = thetaNorm.pow(-p / 2);

            real exn = iOther ? 1.0 : n1;
            uDown += thetaNorm * exn;
            uUp += theta.rowwise() * thetaNorm.transpose() * exn;
        }

        uOut *= uUp.rowwise() / (uDown.transpose() + verySmallReal);

        // * Do cut off
        // for (int iOther = 0; iOther < Nother; iOther++)
        // {
        //     Eigen::ArrayXd uDotuOut;
        //     switch (uOut.rows())
        //     {
        //     case 2:
        //         uDotuOut =
        //             uOthers[iOther](0, EigenAll) * uOut(0, EigenAll) +
        //             uOthers[iOther](1, EigenAll) * uOut(1, EigenAll);
        //         break;
        //     case 3:
        //         uDotuOut =
        //             uOthers[iOther](0, EigenAll) * uOut(0, EigenAll) +
        //             uOthers[iOther](1, EigenAll) * uOut(1, EigenAll) * 0.5 +
        //             uOthers[iOther](2, EigenAll) * uOut(2, EigenAll);
        //         break;
        //     case 4:
        //         uDotuOut =
        //             uOthers[iOther](0, EigenAll) * uOut(0, EigenAll) +
        //             uOthers[iOther](1, EigenAll) * uOut(1, EigenAll) * (1. / 3.) +
        //             uOthers[iOther](2, EigenAll) * uOut(2, EigenAll) * (1. / 3.) +
        //             uOthers[iOther](3, EigenAll) * uOut(3, EigenAll);
        //         break;

        //     default:
        //         DNDS_assert(false);
        //         break;
        //     }

        //     uOut.rowwise() *= 0.5 * (uDotuOut.sign().transpose() + 1);
        // }
        // // * Do cut off

        if (uOut.hasNaN())
        {
            std::cout << "Limiter FWBAP_L2_Multiway Failed" << std::endl;
            std::cout << uMax.transpose() << std::endl;
            std::cout << uUp.transpose() << std::endl;
            std::cout << uDown.transpose() << std::endl;
            abort();
        }
    }

    /**
     * @brief input vector<Eigen::Array-like>
     */
    template <typename Tcenter, typename TinOthers, typename Tout>
    static inline void FMEMM_Multiway_Polynomial2D(const Tcenter &u, const TinOthers &uOthers, int Nother, Tout &uOut, real n1 = 1)
    {
        using namespace DNDS;
        static const int p = 4;

        Eigen::ArrayXXd umax = u.abs();
        umax.rowwise() = umax.colwise().maxCoeff() + verySmallReal;

        Eigen::ArrayXXd theta0 = u / umax;
        Eigen::ArrayXXd thetaMinNorm = theta0;

        for (int iOther = 0; iOther < Nother; iOther++)
        {
            Eigen::ArrayXd thetaMinNormNorm;
            Eigen::ArrayXd thetaNorm;
            Eigen::ArrayXXd theta = uOthers[iOther] / umax;
            Eigen::ArrayXd theta0DotTheta;
            thetaNorm = PolynomialSquaredNorm<2>(theta);
            thetaMinNormNorm = PolynomialSquaredNorm<2>(thetaMinNorm);
            theta0DotTheta = PolynomialDotProduct<2>(theta, theta0);
            Eigen::ArrayXd selection = (thetaNorm - thetaMinNormNorm).sign() * 0.5 + 0.5; //! need eliminate one side?
            thetaMinNorm = theta.rowwise() * (1 - selection).transpose() +
                           thetaMinNorm.rowwise() * selection.transpose();
            // //! cutting
            // theta0 = theta0.rowwise() * (theta0DotTheta.sign() + 1).transpose() * 0.5;
        }
        Eigen::ArrayXd thetaNorm;
        Eigen::ArrayXd thetaMinNormNorm;
        thetaNorm = PolynomialSquaredNorm<2>(theta0);
        thetaMinNormNorm = PolynomialSquaredNorm<2>(thetaMinNorm);
        Eigen::ArrayXd replaceLoc = ((thetaNorm / (thetaMinNormNorm + verySmallReal)).sqrt() - 1).max(verySmallReal);
        // Eigen::ArrayXd replaceFactor = 2 - (-replaceLoc).exp();
        // Eigen::ArrayXd replaceFactor = 2 - (replaceLoc * p + 1).pow(-1. / p);
        Eigen::ArrayXd replaceFactor = replaceLoc * 0 + 1;
        // Eigen::ArrayXd replaceFactor = 1+(1 - (replaceLoc * p + 1).pow(-1. / p)) / (replaceLoc/10+1 );

        replaceFactor = (replaceFactor - 1) / replaceLoc;

        // !safety?
        Eigen::ArrayXd ifReplace = (thetaNorm - thetaMinNormNorm).sign() * 0.5 + 0.5;
        replaceFactor = ifReplace * replaceFactor + (1 - ifReplace);

        uOut = u.rowwise() * replaceFactor.transpose() + (thetaMinNorm * umax).rowwise() * (1 - replaceFactor).transpose();

        if (uOut.hasNaN())
        {
            std::cout << "Limiter FMEMM_L2_Multiway Failed" << std::endl;
            std::cout << umax.transpose() << std::endl;
            std::cout << uOut.transpose() << std::endl;
            std::cout << replaceFactor << std::endl;
            std::cout << replaceLoc << std::endl;
            abort();
        }
    }

    /**
     * @brief input vector<Eigen::Array-like>
     */
    template <typename TinOthers, typename Tout>
    static inline void FWBAP_L2_Multiway_PolynomialOrth(const TinOthers &uOthers, int Nother, Tout &uOut, real n1 = 1)
    {
        using namespace DNDS;
        static const int p = 4;
        static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);

        Eigen::ArrayXXd uUp; //* copy!
        uUp.resizeLike(uOthers[0]);
        uUp.setZero();
        Eigen::ArrayXd uDown;
        uDown.resize(uOthers[0].cols());
        uDown.setZero();
        Eigen::ArrayXXd uMax = uUp + verySmallReal;
        for (int iOther = 0; iOther < Nother; iOther++)
            uMax = uMax.max(uOthers[iOther].abs());
        uMax.rowwise() = uMax.colwise().maxCoeff();
        uOut = uMax;

        for (int iOther = 0; iOther < Nother; iOther++)
        {
            Eigen::ArrayXd thetaNorm;
            Eigen::ArrayXXd theta = uOthers[iOther] / uMax;
            thetaNorm = (theta * theta).colwise().sum();
            thetaNorm += verySmallReal_pDiP;
            thetaNorm = thetaNorm.pow(-p / 2);

            real exn = iOther ? 1.0 : n1;
            uDown += thetaNorm * exn;
            uUp += theta.rowwise() * thetaNorm.transpose() * exn;
        }
        // std::cout << uUp << std::endl;
        // std::cout << uDown << std::endl;
        uOut *= uUp.rowwise() / (uDown.transpose() + verySmallReal);
        if (uOut.hasNaN())
        {
            std::cout << "Limiter FWBAP_L2_Multiway Failed" << std::endl;
            std::cout << uMax.transpose() << std::endl;
            std::cout << uUp.transpose() << std::endl;
            std::cout << uDown.transpose() << std::endl;
            abort();
        }
    }

    namespace _Limiters_Internal
    {
        inline real W12n1(real u1, real u2)
        {
            real n = 1.0;
            real p = 4.0;

            real theta1 = std::pow(u1 / (u2 + signP(u2) * 1e-12), p - 1.0);
            real theta2 = std::pow(u1 / (u2 + signP(u2) * 1e-12), p);

            return u1 * (n + theta1) / (n + theta2);
        }

        inline real W12center(real *u, const int J, real n)
        {
            std::vector<real> thetaBuf(J);
            real *theta = thetaBuf.data();
            for (int ii = 0; ii < J; ++ii)
            {
                theta[ii] = (u[0] + signP(u[0]) * 1e-12) / (u[ii] + signP(u[ii]) * 1e-12);
            }

            real p = 4.0;
            real sumLocal1 = n;
            real sumLocal2 = n;
            for (int ii = 0; ii < J; ++ii)
            {
                sumLocal1 += std::pow(theta[ii], (p - 1.0));
                sumLocal2 += std::pow(theta[ii], p);
            }

            return u[0] * sumLocal1 / (sumLocal2 + 1e-12);
        }
    }

    /**
     * @brief input vector<Eigen::Array-like>
     */
    template <typename TinOthers, typename Tout>
    inline void FWBAP_L2_Multiway(const TinOthers &uOthers, int Nother, Tout &uOut, real n1 = 1.0)
    {
        // PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterA);
        static const int p = 4;
        static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);

        Tout uUp; //* copy!
        uUp.resizeLike(uOthers[0]);
        uUp.setZero();
        Tout uDown = uUp; //* copy!
        Tout uCent = uOthers[0];
        // Tout uMax = uDown + verySmallReal;

        // for (int iOther = 0; iOther < Nother; iOther++)
        //     uMax = uMax.max(uOthers[iOther].abs());
        // uOut = uMax;

        for (int iOther = 1; iOther < Nother; iOther++)
        {
            Tout thetaInverse = uOthers[0] / (uOthers[iOther].abs() + verySmallReal_pDiP) * uOthers[iOther].sign();
            uDown += thetaInverse.square() * thetaInverse.square();
            uUp += thetaInverse.cube();
            uCent *= uOthers[iOther].sign().abs();
        }
        uOut = uCent * (uUp + n1) / (uDown + n1); // currently fast version

        // if (uOut.hasNaN())
        // {
        //     std::cout << "Limiter FWBAP_L2_Multiway Failed" << std::endl;
        //     std::cout << uMax.transpose() << std::endl;
        //     std::cout << uUp.transpose() << std::endl;
        //     std::cout << uDown.transpose() << std::endl;
        //     abort();
        // }

        /************************/ //! safe version
        // uOut.resizeLike(uOthers[0]);
        // static const int maxNeighbour = 7;
        // DNDS_assert(uOthers.size() <= maxNeighbour);
        // real theta[maxNeighbour];

        // for (int idof = 0; idof < uOthers[0].cols(); idof++)
        //     for (int irec = 0; irec < uOthers[0].rows(); irec++)
        //     {
        //         real u0 = uOthers[0](irec, idof);
        //         for (int ii = 0; ii < uOthers.size(); ++ii)
        //         {
        //             real uother = uOthers[ii](irec, idof);
        //             theta[ii] = (u0 + signM(u0) * 1e-12) /
        //                         (uother + signM(uother) * 1e-12);
        //         }

        //         static const real p = 4.0;
        //         real sumLocal1 = n1;
        //         real sumLocal2 = n1;
        //         for (int ii = 1; ii < uOthers.size(); ++ii)
        //         {
        //             sumLocal1 += std::pow(theta[ii], (p - 1.0));
        //             sumLocal2 += std::pow(theta[ii], p);
        //         }

        //         uOut(irec, idof) = u0 * sumLocal1 / (sumLocal2 + 1e-12);
        //     }

        // PerformanceTimer::Instance().StopTimer(PerformanceTimer::LimiterA);
    }

    /**
     * @brief input eigen arrays
     */
    template <typename Tin1, typename Tin2, typename Tout>
    inline void FWBAP_L2_Biway(const Tin1 &u1, const Tin2 &u2, Tout &uOut, real n)
    {
        // PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterA);
        static const int p = 4;
        static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);
        // // static const real n = 10.0;
        // static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);
        // auto uMax = u1.abs().max(u2.abs()) + verySmallReal_pDiP;
        // auto u1p = (u1 / uMax).pow(p);
        // auto u2p = (u2 / uMax).pow(p);
        // // std::cout << u1 << std::endl;

        // uOut = (u1p * u2 + n * u2p * u1) / ((u1p + n * u2p) + verySmallReal);
        // // uOut *= (u1.sign() + u2.sign()).abs() * 0.5; //! cutting below zero!!!
        // // std::cout << u2 << std::endl;

        ///////////
        Tout frac = (u1 / (u2.abs() + verySmallReal_pDiP) * u2.sign());

        // for (index i = 0; i < frac.size(); i++)
        // {
        //     auto &v = frac(i);
        //     if (v == 0.)
        //         v = 1. / verySmallReal_pDiP;
        // }

        // auto theta1 = frac.pow(p - 1);
        // auto theta2 = frac.pow(p);

        auto theta1 = frac.cube();
        auto theta2 = frac.square() * frac.square();

        uOut = u1 * (n + theta1) / (n + theta2); // currently fast version
        uOut *= u2.sign().abs();
        ///////////
        /************************/ //! safe version
        // uOut.resizeLike(u1);
        // for (int idof = 0; idof < u1.cols(); idof++)
        //     for (int irec = 0; irec < u1.rows(); irec++)
        //     {
        //         real u1c = u1(irec, idof);
        //         real u2c = u2(irec, idof);
        //         real frac = (u1c) / (u2c + signM(u2c) * 1e-12);
        //         real theta1 = std::pow(frac, p - 1);
        //         real theta2 = std::pow(frac, p);
        //         uOut(irec, idof) = u1c * (n + theta1) / (n + theta2);
        //     }
        // PerformanceTimer::Instance().StopTimer(PerformanceTimer::LimiterA);
    }

    template <typename Tin1, typename Tin2, typename Tout>
    inline void FWBAP_L2_Cut_Biway(const Tin1 &u1, const Tin2 &u2, Tout &uOut, real n)
    {
        // PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterA);
        static const int p = 4;
        static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);

        Tout frac = (u1 / (u2.abs() + verySmallReal_pDiP) * u2.sign());

        auto theta1 = frac.cube();
        auto theta2 = frac.square() * frac.square();

        uOut = u1 * (n + theta1) / (n + theta2); // currently fast version
        uOut *= u2.sign().abs() * (0.5 * (u1.sign() + u2.sign()).abs());
        ///////////
    }

    /**
     * @brief input eigen arrays
     */
    template <typename Tin1, typename Tin2, typename Tout>
    inline void FMINMOD_Biway(const Tin1 &u1, const Tin2 &u2, Tout &uOut, real n)
    {
        uOut = u1.abs().min(u2.abs()) * (u1.sign() + u2.sign()) * 0.5;
    }

    /**
     * @brief input eigen arrays
     */
    template <typename Tin1, typename Tin2, typename Tout>
    inline void FVanLeer_Biway(const Tin1 &u1, const Tin2 &u2, Tout &uOut, real n)
    {
        uOut = (u1.sign() + u2.sign()) * (u1 * u2).abs() / ((u1 + u2).abs() + verySmallReal);
    }

    template <int dim, int nVarsFixed, typename Tin1, typename Tin2, typename Tout>
    inline void FWBAP_L2_Biway_PolynomialNorm(const Tin1 &u1, const Tin2 &u2, Tout &uOut, real n)
    {
        static const int p = 4;
        // static const real n = 10.0;
        static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);
        static const int nDofMax = dim == 2 ? 6 : 10; //! hard code max 4thorder
        Eigen::Array<real, Eigen::Dynamic, nVarsFixed, Eigen::AutoAlign, nDofMax> uMax = u1.abs().max(u2.abs()) + verySmallReal_pDiP;
        uMax.rowwise() = uMax.colwise().maxCoeff();
        Eigen::ArrayXd u1p, u2p;
        Eigen::Array<real, Eigen::Dynamic, nVarsFixed, Eigen::AutoAlign, nDofMax> theta1 = u1 / uMax;
        Eigen::Array<real, Eigen::Dynamic, nVarsFixed, Eigen::AutoAlign, nDofMax> theta2 = u2 / uMax;
        u1p = PolynomialSquaredNorm<dim>(theta1);
        u2p = PolynomialSquaredNorm<dim>(theta2);
        // u1p = u1p.pow(p / 2);
        // u2p = u2p.pow(p / 2);

        u1p = u1p.square();
        u2p = u2p.square();

        uOut = (u2.rowwise() * u1p.transpose() + n * (u1.rowwise() * u2p.transpose())).rowwise() / ((u1p + n * u2p) + verySmallReal).transpose();
    }

    template <int dim, int nVarsFixed, typename Tin1, typename Tin2, typename Tout>
    inline void FMEMM_Biway_PolynomialNorm(const Tin1 &u1, const Tin2 &u2, Tout &uOut, real n)
    {
        static const int p = 4;
        // static const real n = 10.0;
        static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);
        static const int nDofMax = dim == 2 ? 6 : 10; //! hard code max 4thorder
        Eigen::Array<real, Eigen::Dynamic, nVarsFixed, Eigen::AutoAlign, nDofMax> uMax = u1.abs().max(u2.abs()) + verySmallReal_pDiP;
        uMax.rowwise() = uMax.colwise().maxCoeff();
        Eigen::ArrayXd u1p, u2p;
        Eigen::Array<real, Eigen::Dynamic, nVarsFixed, Eigen::AutoAlign, nDofMax> theta1 = u1 / uMax;
        Eigen::Array<real, Eigen::Dynamic, nVarsFixed, Eigen::AutoAlign, nDofMax> theta2 = u2 / uMax;
        u1p = PolynomialSquaredNorm<dim>(theta1);
        u2p = PolynomialSquaredNorm<dim>(theta2);
        u1p = u1p.sqrt();
        u2p = u2p.sqrt();

        Eigen::ArrayXd replaceLoc = (u1p / (u2p + verySmallReal) - 1).max(verySmallReal);
        // Eigen::ArrayXd replaceFactor = 2 - (-replaceLoc).exp();
        // Eigen::ArrayXd replaceFactor = 2 - (replaceLoc * p + 1).pow(-1. / p);
        Eigen::ArrayXd replaceFactor = replaceLoc * 0 + 1;
        // Eigen::ArrayXd replaceFactor = 1+(1 - (replaceLoc * p + 1).pow(-1. / p)) / (replaceLoc/10+1 );

        replaceFactor = (replaceFactor - 1) / replaceLoc;

        // !safety?
        Eigen::ArrayXd ifReplace = (u1p - u2p).sign() * 0.5 + 0.5;
        // replaceFactor = ifReplace * replaceFactor + (1 - ifReplace);
        // ! fast version:
        replaceFactor = (1 - ifReplace);

        uOut = u1.rowwise() * replaceFactor.transpose() + u2.rowwise() * (1 - replaceFactor).transpose();
        // //! cutting
        // uOut = uOut.rowwise() * (u1u2.sign() + 1).transpose() * 0.5;
    }

    template <typename Tin1, typename Tin2, typename Tout>
    inline void FWBAP_L2_Biway_PolynomialOrth(const Tin1 &u1, const Tin2 &u2, Tout &uOut, real n)
    {
        static const int p = 4;
        // static const real n = 10.0;
        static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);
        Eigen::ArrayXXd uMax = u1.abs().max(u2.abs()) + verySmallReal_pDiP;
        uMax.rowwise() = uMax.colwise().maxCoeff();
        Eigen::ArrayXd u1p, u2p;
        Eigen::ArrayXXd theta1 = u1 / uMax;
        Eigen::ArrayXXd theta2 = u2 / uMax;
        u1p = (theta1 * theta1).colwise().sum();
        u2p = (theta2 * theta2).colwise().sum();
        u1p = u1p.pow(p / 2);
        u2p = u2p.pow(p / 2);

        uOut = (u2.rowwise() * u1p.transpose() + n * (u1.rowwise() * u2p.transpose())).rowwise() / ((u1p + n * u2p) + verySmallReal).transpose();
    }

}
