#pragma once

#include "Cauchy/Riemann/RiemannFluxMath.hpp"
#include <type_traits>

namespace DNDS::Cauchy::Riemann
{
    namespace ComputeDetail
    {
        template <class Scalar>
        DNDS_DEVICE_CALLABLE inline AcousticStateT<Scalar> Convert(const AcousticState &input) noexcept
        {
            AcousticStateT<Scalar> result;
            for (int j = 0; j < 5; ++j)
                result.flow.conservative[j] = Scalar(input.flow.conservative[j]);
            result.flow.pressure = Scalar(input.flow.pressure);
            result.soundSpeed = Scalar(input.soundSpeed);
            return result;
        }

        /** @brief Point-local owner of converted optional numerical inputs.
         * @details Input() must be called after this owner reaches its final
         * stack location. No embedded pointer survives a copy/move, allocation
         * or kernel boundary. Missing optional inputs remain absent, with the
         * same outer validation contract as Math::Evaluate.
         */
        template <class Scalar>
        struct InputOwner
        {
            AcousticStateT<Scalar> left, right, leftMean{}, rightMean{};
            RoeDataT<Scalar> roe{};
            WaveBoundsT<Scalar> waves{};
            AUSMDataT<Scalar> ausm{};
            Vector3T<Scalar> normal{}, grid{};
            bool hasLeftMean, hasRightMean, hasRoe, hasWaves, hasAUSM;

            DNDS_DEVICE_CALLABLE explicit InputOwner(const FluxInput &input) noexcept
                : left(Convert<Scalar>(input.left)), right(Convert<Scalar>(input.right)),
                  hasLeftMean(input.leftMean != nullptr), hasRightMean(input.rightMean != nullptr),
                  hasRoe(input.roe != nullptr), hasWaves(input.outerWaves != nullptr), hasAUSM(input.ausm != nullptr)
            {
                for (int d = 0; d < 3; ++d)
                {
                    normal[d] = Scalar(input.normal[d]);
                    grid[d] = Scalar(input.gridVelocity[d]);
                }
                if (hasLeftMean)
                    leftMean = Convert<Scalar>(*input.leftMean);
                if (hasRightMean)
                    rightMean = Convert<Scalar>(*input.rightMean);
                if (hasRoe)
                {
                    for (int d = 0; d < 3; ++d)
                        roe.velocity[d] = Scalar(input.roe->velocity[d]);
                    for (int j = 0; j < 5; ++j)
                        roe.contactProjection[j] = Scalar(input.roe->contactProjection[j]);
                    roe.soundSpeed = Scalar(input.roe->soundSpeed);
                    roe.enthalpy = Scalar(input.roe->enthalpy);
                    roe.contactEnergy = Scalar(input.roe->contactEnergy);
                    roe.contactEnergyJump = Scalar(input.roe->contactEnergyJump);
                }
                if (hasWaves)
                    waves = {Scalar(input.outerWaves->left), Scalar(input.outerWaves->right)};
                if (hasAUSM)
                    ausm.faceSoundSpeed = Scalar(input.ausm->faceSoundSpeed);
            }
            DNDS_DEVICE_CALLABLE FluxInputT<Scalar> Input() const noexcept
            {
                return {left, right, normal, grid, hasLeftMean ? &leftMean : nullptr, hasRightMean ? &rightMean : nullptr,
                        hasRoe ? &roe : nullptr, hasWaves ? &waves : nullptr, hasAUSM ? &ausm : nullptr};
            }
        };

        template <class Scalar>
        DNDS_DEVICE_CALLABLE inline FluxSettingsT<Scalar> Convert(const FluxSettings &settings) noexcept
        {
            // pressureStencilMin=double::max is the inactive stencil sentinel.
            // Keep it finite in a narrower exponent range, rather than create
            // infinity merely by converting the canonical default.
            const real maximum = DNDS::Math::ScalarCast<real>(DNDS::Math::ScalarMaxFinite<Scalar>());
            return {Scalar(settings.dLambda), Scalar(settings.fixScale), Scalar(settings.incrementScale),
                    Scalar(settings.referenceMach), Scalar(DNDS::Math::ScalarMin(settings.pressureStencilMin, maximum))};
        }
        template <class Scalar>
        DNDS_DEVICE_CALLABLE inline FluxResult Convert(const FluxResultT<Scalar> &input) noexcept
        {
            FluxResult result;
            for (int j = 0; j < 5; ++j)
                result.flux[j] = DNDS::Math::ScalarCast<real>(input.flux[j]);
            result.acousticMinus = DNDS::Math::ScalarCast<real>(input.acousticMinus);
            result.contact = DNDS::Math::ScalarCast<real>(input.contact);
            result.acousticPlus = DNDS::Math::ScalarCast<real>(input.acousticPlus);
            return result;
        }
    }

    /** @brief Evaluate the gas-agnostic Riemann arithmetic in an explicit local scalar type.
     * @details Physical inputs and persistent results remain FP64. The owner
     * converts all present point-local optional data and borrows it only during
     * this call. The FP64 specialization bypasses conversion completely. This
     * changes arithmetic precision, not EOS preparation or state storage.
     * Host dispatch validates scheme requirements. The numerical caller must
     * check the result for overflow in the selected compute exponent range;
     * this pure arithmetic helper does not inspect an entire device field.
     */
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline FluxResult EvaluateRiemannCompute(Solver solver, const FluxInput &input,
                                                                  const FluxSettings &settings = {}) noexcept
    {
        if constexpr (std::is_same_v<Scalar, real>)
            return Math::Evaluate(solver, input, settings);
        else
        {
            const ComputeDetail::InputOwner<Scalar> owner(input);
            return ComputeDetail::Convert(Math::Evaluate(solver, owner.Input(), ComputeDetail::Convert<Scalar>(settings)));
        }
    }

    /** @brief Roe-family reduced dissipation with an FP64 central physical flux.
     * @details Host dispatch permits Roe through Roe_M9, including scalar-wave
     * Roe_M2, and rejects other families before kernel launch. Pressure/energy
     * in the central flux are never narrowed. Wave decomposition, entropy fix
     * and dissipation reuse Math::RoeDissipation; there is no duplicate formula
     * or EOS interpretation. The returned diagnostics describe the dissipation
     * arithmetic. FP64 calls bypass conversion and preserve the canonical path.
     */
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline FluxResult EvaluateRoeDissipationCompute(Solver solver, const FluxInput &input,
                                                                         const FluxSettings &settings = {}) noexcept
    {
        if constexpr (std::is_same_v<Scalar, real>)
            return Math::Evaluate(solver, input, settings);
        else
        {
            const ComputeDetail::InputOwner<Scalar> owner(input);
            FluxResultT<Scalar> magnitudes{};
            const auto dissipation = Math::RoeDissipation(solver, owner.Input(), ComputeDetail::Convert<Scalar>(settings), magnitudes);
            auto result = ComputeDetail::Convert(magnitudes);
            const auto left = PhysicalFlux(input.left.flow, input.normal, input.gridVelocity);
            const auto right = PhysicalFlux(input.right.flow, input.normal, input.gridVelocity);
            for (int j = 0; j < 5; ++j)
                result.flux[j] = .5 * (left[j] + right[j] - DNDS::Math::ScalarCast<real>(dissipation[j]));
            return result;
        }
    }
}
