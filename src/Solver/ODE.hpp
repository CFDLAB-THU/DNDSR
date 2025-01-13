#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Scalar.hpp"

namespace DNDS::ODE
{
    template <class TDATA, class TDTAU>
    class ImplicitDualTimeStep
    {
    public:
        using Frhs = std::function<void(TDATA &, TDATA &, TDTAU &, int, real, int)>;
        using Fdt = std::function<void(TDATA &, TDTAU &, real, int)>;
        // x, res, resOther(=res-alpha*rhs), dTau, dt, alpha, xinc, iter, stage
        using Fsolve = std::function<void(TDATA &, TDATA &, TDATA &, TDTAU &, real, real, TDATA &, int, int)>;
        using Fstop = std::function<bool(int, TDATA &, int)>;
        using Fincrement = std::function<void(TDATA &, TDATA &, real, int)>;

        virtual void Step(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                          int maxIter, const Fstop &fstop, const Fincrement &fincrement, real dt) = 0;
        virtual ~ImplicitDualTimeStep() {}

        virtual TDATA &getLatestRHS() = 0;
    };

    template <class TDATA, class TDTAU>
    class ImplicitEulerDualTimeStep : public ImplicitDualTimeStep<TDATA, TDTAU>
    {
    public:
        using tBase = ImplicitDualTimeStep<TDATA, TDTAU>;
        using Frhs = typename tBase::Frhs;
        using Fdt = typename tBase::Fdt;
        using Fsolve = typename tBase::Fsolve;
        using Fstop = typename tBase::Fstop;
        using Fincrement = typename tBase::Fincrement;

        TDTAU dTau;
        std::vector<TDATA> rhsbuf;
        TDATA rhs;
        TDATA resOther;
        TDATA xLast;
        TDATA xInc;
        index DOF;

        /**
         * @brief mind that NDOF is the dof of dt
         * finit(TDATA& data)
         */
        template <class Finit, class FinitDtau>
        ImplicitEulerDualTimeStep(
            index NDOF, Finit &&finit = [](TDATA &) {}, FinitDtau &&finitDtau = [](TDTAU &) {}) : DOF(NDOF)
        {
            rhsbuf.resize(1);
            for (auto &i : rhsbuf)
                finit(i);
            finit(rhs);
            finit(resOther);
            finit(xLast);
            finit(xInc);
            finitDtau(dTau);
        }

        /**
         * @brief
         * frhs(TDATA &rhs, TDATA &x)
         * fdt(TDTAU& dTau)
         * fsolve(TDATA &x, TDATA &rhs, TDTAU& dTau, real dt, real alphaDiag, TDATA &xinc)
         * bool fstop(int iter, TDATA &xinc, int iInternal)
         */
        virtual void Step(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                          int maxIter, const Fstop &fstop, const Fincrement &fincrement, real dt) override
        {
            xLast = x;
            for (int iter = 1; iter <= maxIter; iter++)
            {
                fdt(x, dTau, 1, 0);

                frhs(rhsbuf[0], x, dTau, iter, 1, 0);
                rhs = xLast;
                rhs -= x;
                rhs *= 1.0 / dt;
                resOther = rhs;
                rhs += rhsbuf[0]; // crhs = rhs + (x_i - x_j) / dt

                fsolve(x, rhs, resOther, dTau, dt, 1.0, xinc, iter, 0);
                fincrement(x, xinc, 1.0, 0);

                if (fstop(iter, rhs, 1))
                    break;
            }
        }

        virtual ~ImplicitEulerDualTimeStep() {}

        virtual TDATA &getLatestRHS() override
        {
            return rhsbuf[0];
        }
    };

    template <class TDATA, class TDTAU>
    class ImplicitSDIRK4DualTimeStep : public ImplicitDualTimeStep<TDATA, TDTAU>
    {

        Eigen::Matrix<real, -1, -1> butcherA;
        Eigen::Vector<real, -1> butcherC;
        Eigen::RowVector<real, -1> butcherB;
        int nInnerStage = 3;
        int schemeC = 0;
        int hasLastEndPointR = 0;
        int latestStage = 0;

    public:
        using tBase = ImplicitDualTimeStep<TDATA, TDTAU>;
        using Frhs = typename tBase::Frhs;
        using Fdt = typename tBase::Fdt;
        using Fsolve = typename tBase::Fsolve;
        using Fstop = typename tBase::Fstop;
        using Fincrement = typename tBase::Fincrement;

        TDTAU dTau;
        std::vector<TDATA> rhsbuf;
        TDATA rhs;
        TDATA resOther;
        TDATA xLast;
        TDATA xIncPrev;
        index DOF;

        /**
         * @brief mind that NDOF is the dof of dt
         * finit(TDATA& data)
         */
        template <class Finit, class FinitDtau>
        ImplicitSDIRK4DualTimeStep(
            index NDOF, Finit &&finit = [](TDATA &) {}, FinitDtau &&finitDtau = [](TDTAU &) {}, int schemeCode = 0) : DOF(NDOF)
        {

            schemeC = schemeCode;

            if (schemeCode == 0)
            {
#define _zeta 0.128886400515
                nInnerStage = 3;
                butcherA.resize(nInnerStage, nInnerStage);
                butcherC.resize(nInnerStage);
                butcherB.resize(nInnerStage);

                butcherA << _zeta, 0, 0,
                    0.5 - _zeta, _zeta, 0,
                    2 * _zeta, 1 - 4 * _zeta, _zeta;
                butcherC << _zeta, 0.5, 1 - _zeta;
                butcherB << 1. / (6 * sqr(2 * _zeta - 1)),
                    (4 * sqr(_zeta) - 4 * _zeta + 2. / 3.) / sqr(2 * _zeta - 1),
                    1. / (6 * sqr(2 * _zeta - 1));
#undef _zeta
            }
            else if (schemeCode == 1)
            {
                nInnerStage = 6;
                butcherA.resize(nInnerStage, nInnerStage);
                butcherC.resize(nInnerStage);
                butcherB.resize(nInnerStage);

                butcherA << verySmallReal, 0, 0, 0, 0, 0,
                    0.25, 0.25, 0, 0, 0, 0,
                    0.137776, -0.055776, 0.25, 0, 0, 0,
                    0.1446368660269822, -0.2239319076133447, 0.4492950415863626, 0.25, 0, 0,
                    0.09825878328356477, -0.5915442428196704, 0.8101210205756877, 0.283164405707806, 0.25, 0,
                    0.1579162951616714, 0, 0.1867589405240008, 0.6805652953093346, -0.2752405309950067, 0.25;
                butcherB = butcherA(Eigen::last, Eigen::all);
                butcherC << 0, 0.5, 0.332, 0.62, 0.849999966747388, 1;
            }
            else
            {
                DNDS_assert(false);
            }

            rhsbuf.resize(nInnerStage);
            for (auto &i : rhsbuf)
                finit(i);
            finit(rhs);
            finit(resOther);
            finit(xLast);
            finit(xIncPrev);
            finitDtau(dTau);
        }

        /**
         * @brief
         * frhs(TDATA &rhs, TDATA &x)
         * fdt(TDTAU& dTau)
         * fsolve(TDATA &x, TDATA &rhs, TDTAU& dTau, real dt, real alphaDiag, TDATA &xinc)
         * bool fstop(int iter, TDATA &xinc, int iInternal)
         */
        virtual void Step(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                          int maxIter, const Fstop &fstop, const Fincrement &fincrement, real dt) override
        {
            xLast = x;
            for (int iB = 0; iB < nInnerStage; iB++)
            {
                x = xLast;
                xIncPrev.setConstant(0.0);
                int iter = 1;
                for (; iter <= maxIter; iter++)
                {
                    if (schemeC == 1 && iB == 0) // for esdirk first frhs evaluation
                    {
                        if (hasLastEndPointR)
                            rhsbuf[0] = rhsbuf[nInnerStage - 1];
                        else
                            frhs(rhsbuf[0], x, dTau, INT_MAX, butcherC(0), 0), latestStage = 0;
                        break;
                    }
                    fdt(x, dTau, butcherA(iB, iB), 0);

                    frhs(rhsbuf[iB], x, dTau, iter, butcherC(iB), 0), latestStage = iB;

                    // //!test explicit
                    // rhs = rhsbuf[iB];
                    // rhs *= dTau;
                    // xinc = rhs;
                    // x += xinc;
                    // if (fstop(iter, xinc, iB + 1))
                    //     break;
                    // continue;
                    // //! test explicit

                    // rhsbuf[0] = rhs;
                    rhs = xLast;
                    rhs -= x;
                    rhs *= 1.0 / dt;
                    for (int jB = 0; jB < iB; jB++)
                        rhs.addTo(rhsbuf[jB], butcherA(iB, jB)); // crhs = rhs + (x_i - x_j) / dt
                    resOther = rhs;
                    rhs.addTo(rhsbuf[iB], butcherA(iB, iB));

                    fsolve(x, rhs, resOther, dTau, dt, butcherA(iB, iB), xinc, iter, 0);
                    // x += xinc;
                    fincrement(x, xinc, 1.0, 0);
                    // x.addTo(xIncPrev, -0.5);

                    xIncPrev = xinc;

                    if (fstop(iter, rhs, iB + 1))
                        break;
                    if (schemeC == 1 && iB == 0) // for esdirk
                        break;

                    // TODO: add time dependent rhs
                }
                if (iter > maxIter)
                    fstop(iter, rhs, iB + 1);
            }
            if (schemeC == 1) // for esdirk
            {
                hasLastEndPointR = 1;
                return;
            }
            x = xLast;
            // for (int jB = 0; jB < nInnerStage; jB++)
            //     fincrement(x, rhsbuf[jB], butcherB(jB) * dt, 0); //!bad here
            for (int jB = 0; jB < nInnerStage; jB++)
                x.addTo(rhsbuf[jB], butcherB(jB) * dt);
        }

        virtual TDATA &getLatestRHS() override
        {
            return rhsbuf[latestStage];
        }

        virtual ~ImplicitSDIRK4DualTimeStep() {}
    };

    template <class TDATA, class TDTAU>
    class ImplicitBDFDualTimeStep : public ImplicitDualTimeStep<TDATA, TDTAU>
    {

        static const Eigen::Matrix<real, 4, 5> BDFCoefs;

    public:
        using tBase = ImplicitDualTimeStep<TDATA, TDTAU>;
        using Frhs = typename tBase::Frhs;
        using Fdt = typename tBase::Fdt;
        using Fsolve = typename tBase::Fsolve;
        using Fstop = typename tBase::Fstop;
        using Fincrement = typename tBase::Fincrement;

        TDTAU dTau;
        std::vector<TDATA> xPrevs;
        Eigen::VectorXd dtPrevs;
        std::vector<TDATA> rhsbuf;
        TDATA rhs;
        TDATA resOther;
        TDATA xLast;
        TDATA xIncPrev;
        TDATA resInc;

        index DOF;
        index cnPrev;
        index prevStart;
        index kBDF;

        /**
         * @brief mind that NDOF is the dof of dt
         * finit(TDATA& data)
         */
        template <class Finit, class FinitDtau>
        ImplicitBDFDualTimeStep(
            index NDOF, Finit &&finit = [](TDATA &) {}, FinitDtau &&finitDtau = [](TDTAU &) {},
            index k = 2) : DOF(NDOF), cnPrev(0), prevStart(k - 2), kBDF(k)
        {
            assert(k > 0 && k <= 4);

            xPrevs.resize(k - 1);
            dtPrevs.resize(k - 1);
            for (auto &i : xPrevs)
                finit(i);
            rhsbuf.resize(1);
            finit(rhsbuf[0]);
            finit(rhs);
            finit(resOther);
            finit(resInc);
            finit(xLast);
            finit(xIncPrev);
            finitDtau(dTau);
        }

        /**
         * @brief
         * frhs(TDATA &rhs, TDATA &x)
         * fdt(TDTAU& dTau)
         * fsolve(TDATA &x, TDATA &rhs, TDTAU& dTau, real dt, real alphaDiag, TDATA &xinc)
         * bool fstop(int iter, TDATA &xinc, int iInternal)
         */
        virtual void Step(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                          int maxIter, const Fstop &fstop, const Fincrement &fincrement, real dt) override
        {
            index kCurrent = cnPrev + 1;
            index prevSiz = kBDF - 1;
            for (index iPrev = 0; iPrev < cnPrev; iPrev++)
                assert(prevSiz && std::abs(dtPrevs[mod(iPrev + prevStart, prevSiz)] - dt) < dt * 1e-8);

            xLast = x;
            // x = xLast;
            xIncPrev.setConstant(0.0);
            int iter = 1;
            for (; iter <= maxIter; iter++)
            {
                fdt(x, dTau, BDFCoefs(kCurrent - 1, 0), 0);
                frhs(rhsbuf[0], x, dTau, iter, 1.0, 0);

                rhs.setConstant(0.0);
                rhs.addTo(x, -1. / dt);
                rhs.addTo(xLast, BDFCoefs(kCurrent - 1, 1) / dt);
                // std::cout << "add " << BDFCoefs(kCurrent - 1, 1) << " " << "last" << std::endl;
                if (prevSiz)
                    for (index iPrev = 0; iPrev < cnPrev; iPrev++)
                    {
                        // std::cout << "add " << BDFCoefs(kCurrent - 1, 2 + iPrev) <<" " << mod(iPrev + prevStart, prevSiz) << std::endl;
                        rhs.addTo(xPrevs[mod(iPrev + prevStart, prevSiz)], BDFCoefs(kCurrent - 1, 2 + iPrev) / dt);
                    }
                resOther = rhs;
                rhs.addTo(rhsbuf[0], BDFCoefs(kCurrent - 1, 0));
                fsolve(x, rhs, resOther, dTau, dt, BDFCoefs(kCurrent - 1, 0), xinc, iter, 0);
                //* xinc = (I/dtau-A*alphaDiag)\rhs

                // std::cout << "BDF::\n";
                // std::cout << kCurrent << " " << cnPrev<<" " << BDFCoefs(kCurrent - 1, 0) << std::endl;

                // x += xinc;
                fincrement(x, xinc, 1.0, 0);
                // x.addTo(xIncPrev, -0.5);

                xIncPrev = xinc;

                if (fstop(iter, rhs, 1))
                    break;
            }
            if (iter > maxIter)
                fstop(iter, rhs, 1);
            if (prevSiz)
            {
                prevStart = mod(prevStart - 1, prevSiz);
                // std::cout << dtPrevs.size() << " " << prevStart << std::endl;
                xPrevs[prevStart] = xLast;
                dtPrevs[prevStart] = dt;
                cnPrev = std::min(cnPrev + 1, prevSiz);
            }
        }

        using FresidualIncPP = std::function<void(
            TDATA &,                       // cx
            TDATA &,                       // xPrev
            TDATA &,                       // crhs
            TDATA &,                       // rhsIncPart,
            const std::function<void()> &, // renewRhsIncPart
            real,                          // ct
            int                            // uPos
            )>;
        using FalphaLimSource = std::function<void(
            TDATA &, // source
            int      // uPos
            )>;

        void StepPP(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                    int maxIter, const Fstop &fstop, const Fincrement &fincrement, real dt,
                    const FalphaLimSource &falphaLimSource,
                    const FresidualIncPP &fresidualIncPP)
        {
            index kCurrent = cnPrev + 1;
            index prevSiz = kBDF - 1;
            for (index iPrev = 0; iPrev < cnPrev; iPrev++)
                assert(prevSiz && std::abs(dtPrevs[mod(iPrev + prevStart, prevSiz)] - dt) < dt * 1e-8);

            xLast = x;
            // x = xLast;
            xIncPrev.setConstant(0.0);
            int iter = 1;
            for (; iter <= maxIter; iter++)
            {
                fdt(x, dTau, BDFCoefs(kCurrent - 1, 0), 0);

                frhs(rhsbuf[0], x, dTau, iter, 1.0, 0);
                fresidualIncPP(
                    x, xLast, rhsbuf[0], resInc,
                    [&]()
                    {
                        resInc.setConstant(0.0);
                        // resInc.addTo(x, -1. / dt);
                        resInc.addTo(xLast, (BDFCoefs(kCurrent - 1, 1) - 1) / dt);
                        // std::cout << "add " << BDFCoefs(kCurrent - 1, 1) << " " << "last" << std::endl;
                        if (prevSiz)
                            for (index iPrev = 0; iPrev < cnPrev; iPrev++)
                            {
                                // std::cout << "add " << BDFCoefs(kCurrent - 1, 2 + iPrev) <<" " << mod(iPrev + prevStart, prevSiz) << std::endl;
                                resInc.addTo(xPrevs[mod(iPrev + prevStart, prevSiz)], BDFCoefs(kCurrent - 1, 2 + iPrev) / dt);
                            }
                        falphaLimSource(resInc, 0); // non-rhs part of residual, fixed with alpha too
                        resInc.addTo(rhsbuf[0], BDFCoefs(kCurrent - 1, 0));
                        resInc *= dt; // so that equation is resInc == x - xLast
                    },
                    1.0,
                    0);

                rhs.setConstant(0.0);
                // rhsbuf .addTo(x, -1. / dt);
                rhs.addTo(xLast, (BDFCoefs(kCurrent - 1, 1) - 1) / dt);
                // std::cout << "add " << BDFCoefs(kCurrent - 1, 1) << " " << "last" << std::endl;
                if (prevSiz)
                    for (index iPrev = 0; iPrev < cnPrev; iPrev++)
                    {
                        // std::cout << "add " << BDFCoefs(kCurrent - 1, 2 + iPrev) <<" " << mod(iPrev + prevStart, prevSiz) << std::endl;
                        rhs.addTo(xPrevs[mod(iPrev + prevStart, prevSiz)], BDFCoefs(kCurrent - 1, 2 + iPrev) / dt);
                    }
                falphaLimSource(rhs, 0); // non-rhs part of residual, fixed with alpha too
                rhs.addTo(x, -1. / dt);
                rhs.addTo(xLast, 1 / dt);
                resOther = rhs;
                rhs.addTo(rhs, BDFCoefs(kCurrent - 1, 0));
                fsolve(x, rhs, resOther, dTau, dt, BDFCoefs(kCurrent - 1, 0), xinc, iter, 0);
                //* xinc = (I/dtau-A*alphaDiag)\rhs

                // std::cout << "BDF::\n";
                // std::cout << kCurrent << " " << cnPrev<<" " << BDFCoefs(kCurrent - 1, 0) << std::endl;

                // x += xinc;
                fincrement(x, xinc, 1.0, 0);
                // x.addTo(xIncPrev, -0.5);

                xIncPrev = xinc;

                if (fstop(iter, rhs, 1))
                    break;
            }
            if (iter > maxIter)
                fstop(iter, rhs, 1);
            if (prevSiz)
            {
                prevStart = mod(prevStart - 1, prevSiz);
                // std::cout << dtPrevs.size() << " " << prevStart << std::endl;
                xPrevs[prevStart] = xLast;
                dtPrevs[prevStart] = dt;
                cnPrev = std::min(cnPrev + 1, prevSiz);
            }
        }

        virtual TDATA &getLatestRHS() override
        {
            return rhsbuf[0];
        }

        virtual ~ImplicitBDFDualTimeStep() {}
    };

    template <class TDATA, class TDTAU>
    const Eigen::Matrix<real, 4, 5> ImplicitBDFDualTimeStep<TDATA, TDTAU>::BDFCoefs{
        {1. / 1., 1. / 1., std::nan("1"), std::nan("1"), std::nan("1")},
        {2. / 3., 4. / 3., -1. / 3., std::nan("1"), std::nan("1")},
        {6. / 11., 18. / 11., -9. / 11., 2. / 11., std::nan("1")},
        {12. / 25., 48. / 25., -36. / 25., 16. / 25., -3. / 25.}};

    template <class TDATA, class TDTAU>
    class ImplicitVBDFDualTimeStep : public ImplicitDualTimeStep<TDATA, TDTAU>
    {

    public:
        using tBase = ImplicitDualTimeStep<TDATA, TDTAU>;
        using Frhs = typename tBase::Frhs;
        using Fdt = typename tBase::Fdt;
        using Fsolve = typename tBase::Fsolve;
        using Fstop = typename tBase::Fstop;
        using Fincrement = typename tBase::Fincrement;

        TDTAU dTau;
        std::vector<TDATA> xPrevs;
        Eigen::VectorXd dtPrevs;
        std::vector<TDATA> rhsbuf;
        TDATA rhs;
        TDATA resOther;
        TDATA xLast;
        TDATA xBase;
        TDATA xIncPrev;
        TDATA resInc;

        index DOF;
        index cnPrev;
        index prevStart;
        index kBDF;

    private:
        index kCurrent = 1;
        index prevSiz = 0;
        Eigen::Vector<real, Eigen::Dynamic> BDFCoefs;

    public:
        /**
         * @brief mind that NDOF is the dof of dt
         * finit(TDATA& data)
         */
        template <class Finit, class FinitDtau>
        ImplicitVBDFDualTimeStep(
            index NDOF, Finit &&finit = [](TDATA &) {}, FinitDtau &&finitDtau = [](TDTAU &) {},
            index k = 2) : DOF(NDOF), cnPrev(0), prevStart(k - 2), kBDF(k)
        {
            assert(k > 0 && k <= 4);

            xPrevs.resize(k - 1);
            dtPrevs.resize(k - 1);
            for (auto &i : xPrevs)
                finit(i);
            rhsbuf.resize(1);
            finit(rhsbuf[0]);
            finit(rhs);
            finit(resOther);
            finit(resInc);
            finit(xLast);
            finit(xBase);
            finit(xIncPrev);
            finitDtau(dTau);
            DNDS_assert(k <= 2);
        }

        void VBDFFrontMatters(real dt)
        {
            kCurrent = cnPrev + 1;
            prevSiz = kBDF - 1;
            // for (index iPrev = 0; iPrev < cnPrev; iPrev++)
            //     assert(prevSiz && std::abs(dtPrevs[mod(iPrev + prevStart, prevSiz)] - dt) < dt * 1e-8);
            BDFCoefs.resize(kCurrent + 1);

            switch (kCurrent)
            {
            case 1:
                BDFCoefs(0) = BDFCoefs(1) = 1;
                break;
            case 2:
            {
                real phi = dt / (dtPrevs[mod(0 + prevStart, prevSiz)] + dt);
                real Rt = dt / dtPrevs[mod(0 + prevStart, prevSiz)];
                BDFCoefs(0) = 1 / (1 + phi);            // C
                BDFCoefs(1) = 1 + Rt * phi / (1 + phi); // -A
                BDFCoefs(2) = -Rt * phi / (1 + phi);    // -B
            }
            break;

            default:
                DNDS_assert(false);
                break;
            }
        }

        /**
         * @brief
         * frhs(TDATA &rhs, TDATA &x)
         * fdt(std::vector<real>& dTau)
         * fsolve(TDATA &x, TDATA &rhs, std::vector<real>& dTau, real dt, real alphaDiag, TDATA &xinc)
         * bool fstop(int iter, TDATA &xinc, int iInternal)
         */
        virtual void Step(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                          int maxIter, const Fstop &fstop, const Fincrement &fincrement, real dt) override
        {
            VBDFFrontMatters(dt);
            xLast = x;
            // x = xLast;
            xIncPrev.setConstant(0.0);
            int iter = 1;
            for (; iter <= maxIter; iter++)
            {
                fdt(x, dTau, BDFCoefs(0), 0);

                frhs(rhsbuf[0], x, dTau, iter, 1.0, 0);

                rhs.setConstant(0.0);
                rhs.addTo(x, -1. / dt);
                rhs.addTo(xLast, BDFCoefs(1) / dt);
                // std::cout << "add " << BDFCoefs(1) << " " << "last" << std::endl;
                if (prevSiz)
                    for (index iPrev = 0; iPrev < cnPrev; iPrev++)
                    {
                        // std::cout << "add " << BDFCoefs(2 + iPrev) <<" " << mod(iPrev + prevStart, prevSiz) << std::endl;
                        rhs.addTo(xPrevs[mod(iPrev + prevStart, prevSiz)], BDFCoefs(2 + iPrev) / dt);
                    }
                resOther = rhs;
                rhs.addTo(rhsbuf[0], BDFCoefs(0));
                fsolve(x, rhs, resOther, dTau, dt, BDFCoefs(0), xinc, iter, 0);
                //* xinc = (I/dtau-A*alphaDiag)\rhs

                // std::cout << "BDF::\n";
                // std::cout << kCurrent << " " << cnPrev<<" " << BDFCoefs(0) << std::endl;

                // x += xinc;
                fincrement(x, xinc, 1.0, 0);
                // x.addTo(xIncPrev, -0.5);

                xIncPrev = xinc;

                if (fstop(iter, rhs, 1))
                    break;
            }
            if (iter > maxIter)
                fstop(iter, rhs, 1);
            if (prevSiz)
            {
                prevStart = mod(prevStart - 1, prevSiz);
                // std::cout << dtPrevs.size() << " " << prevStart << std::endl;
                xPrevs[prevStart] = xLast;
                dtPrevs[prevStart] = dt;
                cnPrev = std::min(cnPrev + 1, prevSiz);
            }
        }

        using FresidualIncPP = std::function<void(
            TDATA &,                       // cx
            TDATA &,                       // xPrev
            TDATA &,                       // crhs
            TDATA &,                       // rhsIncPart,
            const std::function<void()> &, // renewRhsIncPart
            real,                          // ct
            int                            // uPos
            )>;
        using FalphaLimSource = std::function<void(
            TDATA &, // source
            int      // uPos
            )>;
        using FlimitDtBDF = std::function<real(
            TDATA &, // base u,
            TDATA &  // to be limited incu
            )>;

        void LimitDt_StepPPV2(TDATA &xIn, const FlimitDtBDF &flimitDtBDF, real &dtOut, real maxIncrease = 2)
        {
            VBDFFrontMatters(dtOut); // using a wished dt
            switch (kCurrent)
            {
            case 1:
                // nothing
                break;
            case 2:
            {
                DNDS_assert(prevSiz == 1);
                xLast = xIn;
                xLast.addTo(xPrevs[mod(0 + prevStart, prevSiz)], -1.0);
                xLast *= -BDFCoefs(2); // max value
                real limitingV = flimitDtBDF(xIn, xLast);
                // std::cout << fmt::format("limitingV {}, B {}", limitingV, -BDFCoefs(2)) << std::endl;
                if (limitingV >= 1)
                    break;
                DNDS_assert(limitingV >= 0);
                real dtm1 = dtPrevs[mod(0 + prevStart, prevSiz)];
                auto fB = [=](real dt)
                {
                    real Rt = dt / dtm1;
                    real phi = dt / (dt + dtm1);
                    return Rt * phi / (1 + phi);
                };
                real targetB = limitingV * -BDFCoefs(2);
                real dtNew = dtm1;
                if (targetB >= fB(dtm1 * maxIncrease))
                    dtNew = dtm1 * maxIncrease; // max increase value
                else
                    dtNew = Scalar::BisectSolveLower(fB, 0., dtm1 * maxIncrease, targetB * 0.99, 20);

                dtOut = std::min(dtNew, dtOut);
            }
            break;

            default:
                DNDS_assert(false);
                break;
            }
        }

        void StepPP(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                    int maxIter, const Fstop &fstop, const Fincrement &fincrement, real dt,
                    const FalphaLimSource &falphaLimSource,
                    const FresidualIncPP &fresidualIncPP)
        {
            this->VBDFFrontMatters(dt);

            xLast = x;
            // x = xLast;
            xIncPrev.setConstant(0.0);
            int iter = 1;
            for (; iter <= maxIter; iter++)
            {
                fdt(x, dTau, BDFCoefs(0), 0);

                frhs(rhs, x, dTau, iter, 1.0, 0);

                xBase.setConstant(0.0);
                xBase.addTo(xLast, (BDFCoefs(1) - 0) / dt); // 0 because xBase includes xLast/dt part
                if (prevSiz)
                    for (index iPrev = 0; iPrev < cnPrev; iPrev++)
                    {
                        // std::cout << "add " << BDFCoefs(2 + iPrev) <<" " << mod(iPrev + prevStart, prevSiz) << std::endl;
                        xBase.addTo(xPrevs[mod(iPrev + prevStart, prevSiz)], BDFCoefs(2 + iPrev) / dt);
                    }
                fresidualIncPP(
                    x, xBase, rhsbuf[0], resInc,
                    [&]()
                    {
                        resInc.setConstant(0.0);
                        // ***** excluded with VBDF's ts limiting
                        // // resInc.addTo(x, -1. / dt);
                        // resInc.addTo(xLast, (BDFCoefs(1) - 1) / dt);
                        // // std::cout << "add " << BDFCoefs( 1) << " " << "last" << std::endl;
                        // if (prevSiz)
                        //     for (index iPrev = 0; iPrev < cnPrev; iPrev++)
                        //     {
                        //         // std::cout << "add " << BDFCoefs(2 + iPrev) <<" " << mod(iPrev + prevStart, prevSiz) << std::endl;
                        //         resInc.addTo(xPrevs[mod(iPrev + prevStart, prevSiz)], BDFCoefs(2 + iPrev) / dt);
                        //     }
                        // falphaLimSource(resInc, 0); // non-rhs part of residual, fixed with alpha too
                        // *****
                        resInc.addTo(rhsbuf[0], BDFCoefs(0));
                        resInc *= dt; // so that equation is resInc == x - xLast
                    },
                    1.0,
                    0);

                rhs = xBase;
                rhs.addTo(x, -1. / dt);
                // rhs.addTo(xLast, 1 / dt); // 0 because xBase includes xLast/dt part
                resOther = rhs;
                rhs.addTo(rhsbuf[0], BDFCoefs(0));

                fsolve(x, rhs, resOther, dTau, dt, BDFCoefs(0), xinc, iter, 0);
                //* xinc = (I/dtau-A*alphaDiag)\rhs

                // std::cout << "BDF::\n";
                // std::cout << kCurrent << " " << cnPrev<<" " << BDFCoefs(0) << std::endl;

                // x += xinc;
                fincrement(x, xinc, 1.0, 0);
                // x.addTo(xIncPrev, -0.5);

                xIncPrev = xinc;

                if (fstop(iter, rhs, 1))
                    break;
            }
            if (iter > maxIter)
                fstop(iter, rhs, 1);
            if (prevSiz)
            {
                prevStart = mod(prevStart - 1, prevSiz);
                // std::cout << dtPrevs.size() << " " << prevStart << std::endl;
                xPrevs[prevStart] = xLast;
                dtPrevs[prevStart] = dt;
                cnPrev = std::min(cnPrev + 1, prevSiz);
            }
        }

        virtual TDATA &getLatestRHS() override
        {
            return rhs;
        }

        virtual ~ImplicitVBDFDualTimeStep() {}
    };

    /*******************************************************************************************/
    /*                                                                                         */
    /*                                                                                         */
    /*                                                                                         */
    /*******************************************************************************************/

    template <class TDATA, class TDTAU>
    class ImplicitHermite3SimpleJacobianDualStep : public ImplicitDualTimeStep<TDATA, TDTAU>
    {
        int hasLastEndPointR = 0;

    public:
        using tBase = ImplicitDualTimeStep<TDATA, TDTAU>;
        using Frhs = typename tBase::Frhs;
        using Fdt = typename tBase::Fdt;
        using Fsolve = typename tBase::Fsolve;
        using Fstop = typename tBase::Fstop;
        using Fincrement = typename tBase::Fincrement;
        using FsolveNest = std::function<void(
            TDATA &, TDATA &, TDATA &,
            TDTAU &, const std::vector<real> &,
            real, real, TDATA &, int, int)>;

        TDTAU dTau;
        TDATA xMid, rhsMid, rhsFull, resOther;
        std::vector<TDATA> rhsbuf;
        TDATA xLast;
        TDATA xIncPrev;
        index DOF;
        index cnPrev;
        TDATA xIncDamper;
        TDATA xIncDamper2;

        Eigen::Vector<real, 4> cInter;
        Eigen::Vector<real, 3> wInteg;
        int curSolveMethod = 0;
        int nStartIter = 0;
        real thetaM1 = 0.9146;
        real alphaHM3 = 0.5;
        int maskHM3 = 0;
        int maskHM3Exe = 0;

        TDATA xPrev;
        real dtPrev = 0;
        int prevSize = 0;

        /**
         * @brief mind that NDOF is the dof of dt
         * finit(TDATA& data)
         */
        template <class Finit, class FinitDtau>
        ImplicitHermite3SimpleJacobianDualStep(
            index NDOF, Finit &&finit = [](TDATA &) {}, FinitDtau &&finitDtau = [](TDTAU &) {},
            real alpha = 0.55, int nCurSolveMethod = 0, int nnStartIter = 0, real thetaM1n = 0.9146, int mask = 0)
            : DOF(NDOF),
              cnPrev(0),
              curSolveMethod(nCurSolveMethod),
              nStartIter(nnStartIter),
              thetaM1(thetaM1n),
              alphaHM3(alpha),
              maskHM3(mask)
        {

            rhsbuf.resize(3);
            finit(rhsbuf[0]);
            finit(rhsbuf[1]);
            finit(rhsbuf[2]);
            finit(xMid);
            finit(rhsMid);
            finit(rhsFull);
            finit(resOther);
            finit(xLast);
            finit(xIncPrev);
            finit(xIncDamper);
            finit(xIncDamper2);
            finit(xPrev);
            finitDtau(dTau);

            DNDS_assert_info(mask == 0 || mask == 1 || mask == 2, "mask not supported");
            SetCoefs(1);
        }

        void SetCoefs(real hR1 = 1)
        {
            real alpha = alphaHM3;
            int mask = maskHM3;
            maskHM3Exe = mask;
            assert(alpha > 0 && alpha < 1);
            cInter.setZero();
            switch (mask)
            {
            case 0: // U2R2
            {
                cInter[0] = (alpha * alpha) * -3.0 + (alpha * alpha * alpha) * 2.0 + 1.0;
                cInter[1] = (alpha * alpha) * 3.0 - (alpha * alpha * alpha) * 2.0;
                cInter[2] = alpha - (alpha * alpha) * 2.0 + alpha * alpha * alpha;
                cInter[3] = -alpha * alpha + alpha * alpha * alpha;
            }
            break;
            case 1: // U2R1
            {
                cInter[0] = alpha * -2.0 + alpha * alpha + 1.0;
                cInter[1] = alpha * 2.0 - alpha * alpha;
                cInter[2] = 0;
                cInter[3] = -alpha + alpha * alpha;
            }
            break;
            case 2: // U3R1
            {
                if (prevSize == 1 && (hR1 < 1e2 && hR1 > 1e-2))
                {
                    // note that the meaning of cInter[2] is changed!
                    cInter[2] = -(alpha * pow(alpha - 1.0, 2.0) * 1.0 / pow(hR1 + 1.0, 2.0)) / hR1;
                    cInter[0] = ((alpha + hR1) * pow(alpha - 1.0, 2.0)) / hR1;
                    cInter[1] = alpha * 1.0 / pow(hR1 + 1.0, 2.0) * (alpha * 3.0 + hR1 * 3.0 - alpha * (hR1 * hR1) - (alpha * alpha) * hR1 - (alpha * alpha) * 2.0 + (hR1 * hR1) * 2.0);
                    cInter[3] = (alpha * (alpha + hR1) * (alpha - 1.0)) / (hR1 + 1.0);
                }
                else
                {
                    maskHM3Exe = 1;
                    alpha = 0.25;
                    cInter[0] = alpha * -2.0 + alpha * alpha + 1.0;
                    cInter[1] = alpha * 2.0 - alpha * alpha;
                    cInter[2] = 0;
                    cInter[3] = -alpha + alpha * alpha;
                }
            }
            break;
            default:
                DNDS_assert(false);
                break;
            }

            wInteg[0] = (-1.0 / 6.0) / alpha + 1.0 / 2.0;
            wInteg[1] = (-1.0 / 6.0) / (alpha * (alpha - 1.0));
            wInteg[2] = 1.0 / (alpha * 6.0 - 6.0) + 1.0 / 2.0;
        }

        /**
         * @brief
         * frhs(TDATA &rhs, TDATA &x)
         * fdt(TDTAU& dTau)
         * fsolve(TDATA &x, TDATA &rhs, TDTAU& dTau, real dt, real alphaDiag, TDATA &xinc)
         * bool fstop(int iter, TDATA &xinc, int iInternal)
         */
        virtual void Step(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                          int maxIter, const Fstop &fstop, const Fincrement &fincrement, real dt) override
        {
            SetCoefs(dtPrev / (dt + verySmallReal));
            xLast = x;
            xMid = x;
            fdt(xLast, dTau, 1.0, 0);

            if (hasLastEndPointR)
                rhsbuf[0] = rhsbuf[1];
            else
                frhs(rhsbuf[0], xLast, dTau, INT_MAX, 0.0, 0);
            rhsbuf[1] = rhsbuf[0];
            rhsbuf[2] = rhsbuf[0];

            xIncPrev.setConstant(0.0);
            int iter = 1;

            int method = curSolveMethod;

            for (; iter <= maxIter; iter++)
            {

                if (iter < nStartIter)
                {
                    fdt(x, dTau, 1.0, 0);
                    frhs(rhsbuf[1], x, dTau, iter, 1.0, 0);
                    rhsFull = xLast;
                    rhsFull -= x;
                    rhsFull *= 1.0 / dt;
                    resOther = rhsFull;
                    rhsFull += rhsbuf[1];
                    fsolve(x, rhsFull, resOther, dTau, dt, 1.0, xinc, iter, 0);
                }
                else
                {
                    if (method == 0)
                    {
                        rhsMid.setConstant(0.0);
                        real thetaCur = thetaM1;
                        {
                            if (prevSize >= 1 && maskHM3 == 2) // U3R1, cInter[2] is reused for xPrev
                            {
                                rhsMid.addTo(xMid, -1. / dt);
                                rhsMid.addTo(xLast, (cInter(0) + thetaCur) / dt);
                                rhsMid.addTo(x, (cInter(1) - thetaCur) / dt);
                                rhsMid.addTo(xPrev, cInter(2) / dt);
                                rhsMid.addTo(rhsbuf[0], 0 + thetaCur * wInteg(0));
                                rhsMid.addTo(rhsbuf[1], cInter(3) + thetaCur * wInteg(2));
                                resOther = rhsMid;
                                rhsMid.addTo(rhsbuf[2], thetaCur * wInteg(1));
                            }
                            else
                            {
                                if (maskHM3Exe == 1 && maskHM3 == 2)
                                    thetaCur = 1; // for U2R1 filling
                                rhsMid.addTo(xMid, -1. / dt);
                                rhsMid.addTo(xLast, (cInter(0) + thetaCur) / dt);
                                rhsMid.addTo(x, (cInter(1) - thetaCur) / dt);
                                rhsMid.addTo(rhsbuf[0], cInter(2) + thetaCur * wInteg(0));
                                rhsMid.addTo(rhsbuf[1], cInter(3) + thetaCur * wInteg(2));
                                resOther = rhsMid;
                                rhsMid.addTo(rhsbuf[2], thetaCur * wInteg(1));
                            }
                            fdt(xMid, dTau, 1.0, 1);
                            fsolve(xMid, rhsMid, resOther, dTau, dt, std::abs(thetaCur * wInteg(1)), xinc, iter, 1);
                        }

                        fincrement(xMid, xinc, 1.0, 1);

                        frhs(rhsbuf[2], xMid, dTau, iter, alphaHM3, 1);

                        rhsFull.setConstant(0.0);
                        {
                            rhsFull.addTo(xLast, 1. / dt);
                            rhsFull.addTo(x, -1. / dt);
                            rhsFull.addTo(rhsbuf[0], wInteg(0));
                            rhsFull.addTo(rhsbuf[2], wInteg(1));
                            resOther = rhsFull;
                            rhsFull.addTo(rhsbuf[1], wInteg(2));
                            fdt(x, dTau, 1.0, 0);
                            fsolve(x, rhsFull, resOther, dTau, dt, wInteg(2), xinc, iter, 0);
                        }

                        fincrement(x, xinc, 1.0, 0);

                        frhs(rhsbuf[1], x, dTau, iter, 1.0, 0);
                    }
                    else
                    {
                        DNDS_assert(false);
                    }
                }

                xIncPrev = xinc;

                if (fstop(iter, method == 0 ? rhsMid : rhsFull, 1))
                    if (iter >= nStartIter)
                        break;
            }
            if (iter > maxIter)
                fstop(iter, method == 0 ? rhsMid : rhsFull, 1);

            hasLastEndPointR = 1;

            prevSize = 1;
            xPrev = xLast;
            dtPrev = dt;
        }

        void StepNested(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve, const FsolveNest &fsolveN,
                        int maxIter, const Fstop &fstop, const Fincrement &fincrement, real dt)
        {
            xLast = x;
            frhs(rhsbuf[0], x, dTau, 0, 1.0, 0);

            xIncPrev.setConstant(0.0);
            int iter = 1;
            for (; iter <= maxIter; iter++)
            {

                if (iter < nStartIter)
                {
                    fdt(x, dTau, 1.0, 0);
                    frhs(rhsbuf[1], x, dTau, iter, 1.0, 0);
                    rhsFull = xLast;
                    rhsFull -= x;
                    rhsFull *= 1.0 / dt;
                    resOther = rhsFull;
                    rhsFull += rhsbuf[1];
                    fsolve(x, rhsFull, resOther, dTau, dt, 1.0, xinc, iter, 0);
                }
                else
                {
                    fdt(x, dTau, 1.0, 0);

                    frhs(rhsbuf[1], x, dTau, iter, 1.0, 0);
                    xMid.setConstant(0.0);
                    xMid.addTo(xLast, cInter[0]);
                    xMid.addTo(x, cInter[1]);
                    fincrement(xMid, rhsbuf[0], cInter[2] * dt, 1);
                    fincrement(xMid, rhsbuf[1], cInter[3] * dt, 1);
                    frhs(rhsMid, xMid, dTau, iter, 1.0, 1);
                    rhsFull.setConstant(0.0);
                    rhsFull.addTo(rhsbuf[0], wInteg[0]);
                    rhsFull.addTo(rhsMid, wInteg[1]);
                    rhsFull.addTo(rhsbuf[1], wInteg[2]);
                    rhsFull.addTo(x, -1. / dt);
                    rhsFull.addTo(xLast, 1. / dt);

                    {
                        // damping
                        // xIncDamper = xLast;
                        // xIncDamper.addTo(x, -1.);
                        // xIncDamper.setAbs();
                        // xIncDamper += 1e-100;
                        // xIncDamper2 = xIncPrev;
                        // xIncDamper2.setAbs();
                        // xIncDamper2 += xIncDamper;
                        // xIncDamper /= xIncDamper2;
                        // rhsFull *= xIncDamper;
                    }
                    {
                        // fdt(x, dTau, 1.0); // TODO: use "update spectral radius" procedure? or force update in fsolve
                        // for(auto &v : dTau)
                        //     v *= -cInter[3] / cInter[1];
                        // fsolve(x, rhsFull, dTau, -dt * cInter[3] / cInter[1],
                        //        1.0, xinc, iter);
                        // rhsFull = xinc;
                        // fdt(xMid, dTau, 1.0);
                        // for (auto &v : dTau)
                        //     v = veryLargeReal;
                        // fsolve(xMid, rhsFull, dTau, -dt * cInter[3] * wInteg[1] / wInteg[2],
                        //        1.0, xinc, iter);
                        // xinc *= -1. / (2 * dt * cInter[3] * wInteg[1]);
                        fsolveN(x, xMid, rhsFull, dTau,
                                std::vector<real>{
                                    //    cInter[1] * wInteg[2] / (cInter[3] * dt),
                                    //    -(cInter[3] * wInteg[1]),
                                    //    -cInter[3] * wInteg[1] / wInteg[2],
                                    //    -cInter[3] / cInter[1],
                                    0. / dt,
                                    1,
                                    1,
                                    1,
                                },
                                dt, 1.0, xinc, iter, 0);
                    }
                }

                //**    xinc = (I/dtau-A*alphaDiag)\rhs

                // x += xinc;
                {
                    // xIncDamper = xLast;
                    // xIncDamper.addTo(x, -1.);
                    // xIncDamper.setAbs();
                    // xIncDamper += 1e-100;
                    // xIncDamper2 = xIncPrev;
                    // xIncDamper2.setAbs();
                    // xIncDamper2 += xIncDamper;
                    // xIncDamper /= xIncDamper2;
                    // xinc *= xIncDamper;
                }

                fincrement(x, xinc, 1.0, 0);
                // x.addTo(xIncPrev, -0.5);

                xIncPrev = xinc;

                if (fstop(iter, xinc, 1))
                    if (iter >= nStartIter)
                        break;
            }
            if (iter > maxIter)
                fstop(iter, xinc, 1);
        }

        virtual TDATA &getLatestRHS() override
        {
            return rhsbuf[1];
        }

        virtual ~ImplicitHermite3SimpleJacobianDualStep() {}
    };

    /*******************************************************************************************/
    /*                                                                                         */
    /*                                                                                         */
    /*                                                                                         */
    /*******************************************************************************************/

    template <class TDATA, class TDTAU>
    class ExplicitSSPRK3TimeStepAsImplicitDualTimeStep : public ImplicitDualTimeStep<TDATA, TDTAU>
    {
    public:
        using tBase = ImplicitDualTimeStep<TDATA, TDTAU>;
        using Frhs = typename tBase::Frhs;
        using Fdt = typename tBase::Fdt;
        using Fsolve = typename tBase::Fsolve;
        using Fstop = typename tBase::Fstop;
        using Fincrement = typename tBase::Fincrement;

        TDTAU dTau;
        std::vector<TDATA> rhsbuf;
        TDATA rhs;
        TDATA xLast;
        TDATA xInc;
        index DOF;
        bool localDtStepping{false};

        template <class Finit, class FinitDtau>
        ExplicitSSPRK3TimeStepAsImplicitDualTimeStep(
            index NDOF, Finit &&finit = [](TDATA &) {}, FinitDtau &&finitDtau = [](TDTAU &) {},
            bool nLocalDtStepping = false)
            : DOF(NDOF), localDtStepping(nLocalDtStepping)
        {

            rhsbuf.resize(3);
            for (auto &i : rhsbuf)
                finit(i);
            finit(rhs);
            finit(xLast);
            finit(xInc);
            finitDtau(dTau);
        }

        /*!


        @brief fsolve, maxIter, fstop are omitted here
        */
        virtual void Step(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                          int maxIter, const Fstop &fstop, const Fincrement &fincrement, real dt) override
        {

            fdt(x, dTau, 1.0, 0); // always gets dTau for CFL evaluation
            xLast = x;
            // MPI::Barrier(MPI_COMM_WORLD);
            // std::cout << "fucked" << std::endl;

            frhs(rhs, x, dTau, 1, 0.5, 0);
            rhsbuf[0] = rhs;
            if (localDtStepping)
                rhs *= dTau;
            else
                rhs *= dt;

            // x += rhs;
            fincrement(x, rhs, 1.0, 0);

            frhs(rhs, x, dTau, 1, 1, 0);
            rhsbuf[1] = rhs;
            if (localDtStepping)
                rhs *= dTau;
            else
                rhs *= dt;
            x *= 0.25;
            x.addTo(xLast, 0.75);
            // x.addTo(rhs, 0.25);
            fincrement(x, rhs, 0.25, 0);

            frhs(rhs, x, dTau, 1, 0.25, 0);
            rhsbuf[2] = rhs;
            if (localDtStepping)
                rhs *= dTau;
            else
                rhs *= dt;
            x *= 2. / 3.;
            x.addTo(xLast, 1. / 3.);
            // x.addTo(rhs, 2. / 3.);
            fincrement(x, rhs, 2. / 3., 0);
        }

        virtual TDATA &getLatestRHS() override
        {
            return rhsbuf[0];
        }
    };
}