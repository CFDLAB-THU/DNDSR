#pragma once

#include <nlohmann/json.hpp>
#include <set>
#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "DNDS/JsonUtil.hpp"
#include "Geom/Geometric.hpp"

namespace DNDS::Euler
{

    struct CLDriverSettings
    {
        real AOAInit = 0.0;
        std::string AOAAxis = "z";
        std::string CL0Axis = "y";
        std::string CD0Axis = "x";
        real refArea = 1.0;
        real refDynamicPressure = 0.5;
        real targetCL = 0.0;
        real CLIncrementRelax = 0.25;    // reduce each alpha increment
        real thresholdTargetRatio = 0.5; // reduce CL convergence threshold when close to the target CL

        index nIterStartDrive = INT32_MAX;
        index nIterConvergeMin = 50;
        real CLconvergeThreshold = 1e-3;
        index CLconvergeWindow = 10;

        index CLconvergeLongWindow = 100;
        index CLconvergeLongThreshold = 1e-4;

        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
            CLDriverSettings,
            AOAInit,
            AOAAxis, CL0Axis, CD0Axis,
            refArea, refDynamicPressure, targetCL,
            CLIncrementRelax, thresholdTargetRatio,
            nIterStartDrive, nIterConvergeMin, CLconvergeThreshold, CLconvergeWindow,
            CLconvergeLongThreshold, CLconvergeLongWindow)
    };

    class CLDriver
    {
        CLDriverSettings settings;
        real lastCL{veryLargeReal};
        real lastAOA{veryLargeReal};
        Eigen::VectorXd CLHistory;
        index CLHistorySize = 0;
        index CLHistoryHead = 0;
        index CLAtTargetAcc = 0;
        void _PushCL(real CL)
        {
            CLHistoryHead = mod<index>(CLHistoryHead + 1, CLHistory.size());
            CLHistory(CLHistoryHead) = CL;
            CLHistorySize++;
        }
        void _ClearCL()
        {
            CLHistory.setConstant(veryLargeReal);
            CLHistorySize = 0;
            CLHistoryHead = 0;
        }

        real AOA{0.0};

    public:
        CLDriver(const CLDriverSettings &settingsIn) : settings(settingsIn)
        {
            auto assertOnAxisString = [](const std::string &ax)
            {
                DNDS_assert(ax.size() == 1);
                DNDS_assert(std::set<std::string>({"x", "y", "z"}).count(ax) == 1);
            };
            assertOnAxisString(settings.AOAAxis);
            assertOnAxisString(settings.CL0Axis);
            assertOnAxisString(settings.CD0Axis);

            DNDS_assert(settings.CLconvergeWindow >= 2);
            AOA = settings.AOAInit;
            CLHistory.resize(settings.CLconvergeWindow);
            CLHistory.setConstant(veryLargeReal);
        }

        real GetAOA()
        {
            return AOA;
        }

        void Update(index iter, real CL, const MPIInfo &mpi)
        {
            _PushCL(CL);
            real curCLErr = std::abs(settings.targetCL - CL);
            if (curCLErr <= settings.CLconvergeLongThreshold)
                CLAtTargetAcc++;
            else
                CLAtTargetAcc = 0;

            if (iter < settings.nIterStartDrive)
                return;
            if (CLHistorySize >= CLHistory.size() && CLHistorySize >= settings.nIterConvergeMin)
            {
                real curCL = CLHistory.mean();
                real currentCLThreshold = settings.CLconvergeThreshold;
                currentCLThreshold = std::min(currentCLThreshold, std::abs(settings.targetCL - lastCL) * settings.thresholdTargetRatio);
                real maxCLDeviation = std::max(CLHistory.maxCoeff() - curCL, curCL - CLHistory.minCoeff());
                if (maxCLDeviation <= currentCLThreshold) // do AoA Update
                {
                    real CLSlope = (lastCL - CL) / (lastAOA - AOA);
                    real CLSlopeStandard = sqr(pi) / 90.;
                    if (lastCL == veryLargeReal || lastAOA == veryLargeReal)
                        CLSlope = CLSlopeStandard;

                    if (CLSlope < 0)
                        CLSlope = CLSlopeStandard; //! warning, assuming positive CLSlope now
                    // if (std::abs(CLSlope) > 10 * CLSlopeStandard)
                    //     CLSlope = CLSlopeStandard;
                    // if (std::abs(CLSlope) < 0.9 * CLSlopeStandard)
                    //     CLSlope = CLSlopeStandard;
                    CLSlope = std::min(CLSlope, 10 * CLSlopeStandard);
                    CLSlope = std::max(CLSlope, 0.9 * CLSlopeStandard);

                    real AOANew = AOA + (settings.targetCL - CL) / CLSlope * settings.CLIncrementRelax;

                    lastAOA = AOA;
                    lastCL = curCL;
                    AOA = AOANew;
                    CLAtTargetAcc = 0;
                    _ClearCL();

                    if (mpi.rank == 0)
                        log() << fmt::format("=== CLDriver at iter [{}], CL converged = [{}+-{:.1e}], CLSlope = [{}], newAOA [{}]",
                                             iter, curCL, maxCLDeviation, CLSlope, AOA)
                              << std::endl;
                }
            }
        }

        bool ConvergedAtTarget()
        {
            return CLAtTargetAcc >= settings.CLconvergeLongWindow;
        }

        /**
         * \brief rotates inflow from AOA=0 to current AOA
         */
        Geom::tGPoint GetAOARotation()
        {
            if (settings.AOAAxis == "z")
            {
                return Geom::RotZ(AOA);
            }
            else if (settings.AOAAxis == "y")
            {
                return Geom::RotY(-AOA);
            }
            else
            {
                DNDS_assert_info(false, "AOAAxis not supported");
                return Geom::RotY(-AOA);
            }
        }

        Geom::tPoint GetCL0Direction()
        {
            if (settings.CL0Axis == "y")
                return Geom::tPoint{0, 1, 0};
            else if (settings.CL0Axis == "z")
                return Geom::tPoint{0, 0, 1};
            else
            {
                DNDS_assert_info(false, "CL0Axis not supported");
                return Geom::tPoint{0, 0, 1};
            }
        }

        Geom::tPoint GetCD0Direction()
        {
            if (settings.CD0Axis == "x")
                return Geom::tPoint{1, 0, 0};
            else
            {
                DNDS_assert_info(false, "CD0Axis not supported");
                return Geom::tPoint{1, 0, 0};
            }
        }

        real GetForce2CoeffRatio()
        {
            return 1. / (settings.refArea * settings.refDynamicPressure);
        }
    };

}