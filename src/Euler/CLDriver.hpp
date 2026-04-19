/** @file CLDriver.hpp
 *  @brief Iterative angle-of-attack driver for targeting a desired lift coefficient.
 *
 *  Provides the CLDriverSettings configuration struct and the CLDriver controller
 *  class. During a CFD simulation the CLDriver monitors the running lift coefficient,
 *  detects convergence over a sliding window, estimates the CL-vs-AoA slope, and
 *  adjusts the angle of attack to drive CL toward a user-specified target.
 */
#pragma once

#include "DNDS/JsonUtil.hpp"
#include "DNDS/ConfigParam.hpp"
#include <set>
#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Geom/Geometric.hpp"

namespace DNDS::Euler
{

    /**
     * @brief JSON-configurable settings for the CL (lift coefficient) driver.
     *
     * Controls the iterative angle-of-attack (AoA) adjustment loop that seeks
     * to match a target lift coefficient. Parameters govern the AoA rotation
     * axis, force reference directions, normalization areas/pressures,
     * under-relaxation, and both short-window and long-window convergence
     * criteria.
     */
    struct CLDriverSettings
    {
        real AOAInit = 0.0;              ///< Initial angle of attack in degrees.
        std::string AOAAxis = "z";       ///< Coordinate axis about which AoA rotation is applied ("x", "y", or "z").
        std::string CL0Axis = "y";       ///< Coordinate axis defining the zero-AoA lift direction ("x", "y", or "z").
        std::string CD0Axis = "x";       ///< Coordinate axis defining the zero-AoA drag direction ("x", "y", or "z").
        real refArea = 1.0;              ///< Reference area for aerodynamic coefficient normalization.
        real refDynamicPressure = 0.5;   ///< Reference dynamic pressure (0.5 * rho_inf * V_inf^2) for coefficient normalization.
        real targetCL = 0.0;             ///< Target lift coefficient that the driver attempts to achieve.
        real CLIncrementRelax = 0.25;    ///< Under-relaxation factor applied to each AoA increment (0,1]. // reduce each alpha increment
        real thresholdTargetRatio = 0.5; ///< Fraction of |targetCL - lastCL| used to tighten the convergence threshold near the target. // reduce CL convergence threshold when close to the target CL

        index nIterStartDrive = INT32_MAX;  ///< Solver iteration at which the CL driver becomes active.
        index nIterConvergeMin = 50;        ///< Minimum number of iterations before the CL convergence window is evaluated.
        real CLconvergeThreshold = 1e-3;    ///< Maximum deviation within the sliding window for CL to be considered converged.
        index CLconvergeWindow = 10;        ///< Number of most-recent CL samples in the sliding convergence window.

        index CLconvergeLongWindow = 100;       ///< Number of consecutive iterations within tolerance required for final (long-window) convergence. // for converged-at-target exit of main iteration loop
        real CLconvergeLongThreshold = 1e-4;    ///< CL error tolerance for the long-window converged-at-target check.
        bool CLconvergeLongStrictAoA = false;   ///< If true, reset the long-window counter whenever the AoA is updated.

        DNDS_DECLARE_CONFIG(CLDriverSettings)
        {
            DNDS_FIELD(AOAInit,                   "Initial angle of attack (degrees)");
            DNDS_FIELD(AOAAxis,                   "Rotation axis for AoA",
                       DNDS::Config::enum_values({"x", "y", "z"}));
            DNDS_FIELD(CL0Axis,                   "Lift force reference axis",
                       DNDS::Config::enum_values({"x", "y", "z"}));
            DNDS_FIELD(CD0Axis,                   "Drag force reference axis",
                       DNDS::Config::enum_values({"x", "y", "z"}));
            DNDS_FIELD(refArea,                   "Reference area for force coefficients",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(refDynamicPressure,        "Reference dynamic pressure",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(targetCL,                  "Target lift coefficient");
            DNDS_FIELD(CLIncrementRelax,          "Under-relaxation for AoA increment",
                       DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(thresholdTargetRatio,      "Reduce CL threshold when near target",
                       DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(nIterStartDrive,           "Iteration to start CL driver",
                       DNDS::Config::range(0));
            DNDS_FIELD(nIterConvergeMin,          "Min iterations before CL convergence check",
                       DNDS::Config::range(1));
            DNDS_FIELD(CLconvergeThreshold,       "CL convergence window tolerance",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(CLconvergeWindow,          "Number of samples in CL convergence window",
                       DNDS::Config::range(2));
            DNDS_FIELD(CLconvergeLongWindow,      "Long window for converged-at-target exit",
                       DNDS::Config::range(1));
            DNDS_FIELD(CLconvergeLongThreshold,   "Long-window CL tolerance",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(CLconvergeLongStrictAoA,   "Reset long counter on AoA update");
        }
    };

    /**
     * @brief Lift-coefficient driver controller.
     *
     * Called once per solver iteration with the current CL value. Maintains a
     * circular sliding window of recent CL samples. When the window is full and
     * the maximum deviation from the mean falls below a (possibly tightened)
     * threshold, the driver estimates the local CL slope dCL/dAoA from the
     * previous and current converged CL values, clamps it to [0.9, 10] times
     * the thin-airfoil theoretical slope (2*pi per radian ≈ pi^2/90 per degree),
     * and computes a new angle of attack with under-relaxation.
     *
     * A separate long-window counter tracks how many consecutive iterations the
     * CL error stays within CLconvergeLongThreshold; when the counter reaches
     * CLconvergeLongWindow the solver may terminate.
     */
    class CLDriver
    {
        CLDriverSettings settings;       ///< Configuration parameters for this driver instance.
        real lastCL{veryLargeReal};      ///< CL value from the previous converged window (sentinel = not yet set).
        real lastAOA{veryLargeReal};     ///< AoA corresponding to @ref lastCL (sentinel = not yet set).
        Eigen::VectorXd CLHistory;       ///< Circular buffer holding the most recent CL samples.
        index CLHistorySize = 0;         ///< Total number of CL samples pushed (may exceed window size).
        index CLHistoryHead = 0;         ///< Current write position (head) in the circular buffer.
        index CLAtTargetAcc = 0;         ///< Consecutive-iteration counter for the long-window convergence check.

        /**
         * @brief Push a new CL sample into the circular history buffer.
         * @param CL The lift coefficient value for the current iteration.
         */
        void PushCL_(real CL)
        {
            CLHistoryHead = mod<index>(CLHistoryHead + 1, CLHistory.size());
            CLHistory(CLHistoryHead) = CL;
            CLHistorySize++;
        }
        /**
         * @brief Reset the CL history buffer, clearing all stored samples.
         */
        void ClearCL_()
        {
            CLHistory.setConstant(veryLargeReal);
            CLHistorySize = 0;
            CLHistoryHead = 0;
        }

        real AOA{0.0}; ///< Current angle of attack in degrees.

    public:
        /**
         * @brief Construct a CLDriver from the given settings.
         *
         * Validates axis strings, allocates the CL history window, and sets the
         * initial angle of attack from CLDriverSettings::AOAInit.
         *
         * @param settingsIn Fully populated CLDriverSettings instance.
         */
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

        /**
         * @brief Get the current angle of attack.
         * @return Current AoA in degrees.
         */
        real GetAOA()
        {
            return AOA;
        }

        /**
         * @brief Main driver update — call once per solver iteration.
         *
         * Pushes the current CL into the sliding window, updates the long-window
         * convergence counter, and (if the driver is active and the window has
         * converged) computes a new AoA based on the estimated dCL/dAoA slope.
         *
         * The slope is clamped to [0.9, 10] times the thin-airfoil theoretical
         * value (pi^2/90 per degree). The AoA increment is under-relaxed by
         * CLDriverSettings::CLIncrementRelax.
         *
         * @param iter  Current solver iteration number.
         * @param CL    Lift coefficient computed at this iteration.
         * @param mpi   MPI communicator info (rank 0 prints log messages).
         */
        void Update(index iter, real CL, const MPIInfo &mpi)
        {
            PushCL_(CL);
            real curCLErr = std::abs(settings.targetCL - CL);
            // if (mpi.rank == 0)
            //     std::cout << fmt::format("curCLErr {}, n = {}", curCLErr, CLAtTargetAcc) << std::endl;
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
                    if (settings.CLconvergeLongStrictAoA)
                        CLAtTargetAcc = 0;
                    ClearCL_();

                    if (mpi.rank == 0)
                        log() << fmt::format("=== CLDriver at iter [{}], CL converged = [{}+-{:.1e}], CLSlope = [{}], newAOA [{}]",
                                             iter, curCL, maxCLDeviation, CLSlope, AOA)
                              << std::endl;
                }
            }
        }

        /**
         * @brief Check whether the long-window convergence criterion is satisfied.
         *
         * Returns true when the CL error has remained within
         * CLDriverSettings::CLconvergeLongThreshold for at least
         * CLDriverSettings::CLconvergeLongWindow consecutive iterations.
         * The solver may use this to terminate the main iteration loop.
         *
         * @return True if converged at the target CL for sufficiently many iterations.
         */
        bool ConvergedAtTarget()
        {
            return CLAtTargetAcc >= settings.CLconvergeLongWindow;
        }

        /**
         * @brief Compute the rotation matrix that rotates the freestream from AoA = 0 to the current AoA.
         *
         * Supports rotation about the z-axis (standard 2-D convention) and the
         * y-axis. The sign convention follows the right-hand rule for z and a
         * negated angle for y so that positive AoA corresponds to positive lift
         * in the standard aerodynamic frame.
         *
         * @return 3×3 rotation matrix (Geom::tGPoint) encoding the current AoA.
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

        /**
         * @brief Get the unit vector for the zero-AoA lift direction.
         *
         * Returns a unit vector along CLDriverSettings::CL0Axis.
         * Currently supports "y" and "z".
         *
         * @return 3-D unit vector (Geom::tPoint) in the lift direction.
         */
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

        /**
         * @brief Get the unit vector for the zero-AoA drag direction.
         *
         * Returns a unit vector along CLDriverSettings::CD0Axis.
         * Currently supports "x" only.
         *
         * @return 3-D unit vector (Geom::tPoint) in the drag direction.
         */
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

        /**
         * @brief Compute the multiplicative factor that converts an integrated force to
         *        a non-dimensional aerodynamic coefficient.
         *
         * The factor is 1 / (refArea * refDynamicPressure).
         *
         * @return Force-to-coefficient ratio (dimensionless).
         */
        real GetForce2CoeffRatio()
        {
            return 1. / (settings.refArea * settings.refDynamicPressure);
        }
    };

}