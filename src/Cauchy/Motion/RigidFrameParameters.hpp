#pragma once

#include "Cauchy/Physics/FlowState.hpp"
#include <type_traits>

namespace DNDS::Cauchy
{
    /// Plain stage-time kinematics, with column-major local-to-world rotation.
    /// The host motion/pose owner prepares this record once. It contains no
    /// mesh, geometry service or allocation and can be passed by value to CUDA.
    /// Momentum remains absolute velocity in local axes; energy stays absolute
    /// total energy. Grid velocity affects flux transport, not stored energy.
    struct RigidFrameParameters
    {
        std::array<real, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
        Vector3 translation{}, angularVelocity{}, centerVelocity{}, center{};
        bool moving = false;

        DNDS_DEVICE_CALLABLE Vector3 GridVelocity(const Vector3 &point) const noexcept
        {
            if (!moving)
                return {};
            const Vector3 x{point[0] - center[0], point[1] - center[1], point[2] - center[2]};
            const auto &w = angularVelocity;
            return {centerVelocity[0] + (w[1] * x[2] - w[2] * x[1]),
                    centerVelocity[1] + (w[2] * x[0] - w[0] * x[2]),
                    centerVelocity[2] + (w[0] * x[1] - w[1] * x[0])};
        }
        DNDS_DEVICE_CALLABLE Vector3 WorldVector(const Vector3 &v) const noexcept
        {
            if (!moving)
                return v;
            Vector3 result{};
            for (int i = 0; i < 3; ++i)
                result[i] = rotation[i] * v[0] + rotation[i + 3] * v[1] + rotation[i + 6] * v[2];
            return result;
        }
        DNDS_DEVICE_CALLABLE Vector3 LocalVector(const Vector3 &v) const noexcept
        {
            if (!moving)
                return v;
            Vector3 result{};
            for (int i = 0; i < 3; ++i)
                result[i] = rotation[i * 3] * v[0] + rotation[1 + i * 3] * v[1] + rotation[2 + i * 3] * v[2];
            return result;
        }
        DNDS_DEVICE_CALLABLE Vector3 WorldPoint(const Vector3 &point) const noexcept
        {
            if (!moving)
                return point;
            auto result = WorldVector(point);
            for (int d = 0; d < 3; ++d)
                result[d] += translation[d];
            return result;
        }
        DNDS_DEVICE_CALLABLE FlowState WorldState(FlowState state) const noexcept
        {
            const auto momentum = WorldVector({state[1], state[2], state[3]});
            for (int d = 0; d < 3; ++d)
                state[1 + d] = momentum[d];
            return state;
        }
        DNDS_DEVICE_CALLABLE FlowState LocalState(FlowState state) const noexcept
        {
            const auto momentum = LocalVector({state[1], state[2], state[3]});
            for (int d = 0; d < 3; ++d)
                state[1 + d] = momentum[d];
            return state;
        }
        /// Components of minus Omega cross momentum in the local Cartesian axes.
        DNDS_DEVICE_CALLABLE Vector3 MomentumSource(const Vector3 &momentum) const noexcept
        {
            if (!moving)
                return {};
            const auto &w = angularVelocity;
            return {w[2] * momentum[1] - w[1] * momentum[2],
                    w[0] * momentum[2] - w[2] * momentum[0],
                    w[1] * momentum[0] - w[0] * momentum[1]};
        }
        /// Absolute-velocity component-axis source, -Omega cross rho*u.
        /// It contributes no mass or energy work.
        DNDS_DEVICE_CALLABLE FlowState RotationSource(const FlowState &state) const noexcept
        {
            const auto source = MomentumSource({state[1], state[2], state[3]});
            return {0, source[0], source[1], source[2], 0};
        }
        /// Column-major negative derivative; volume/time scaling stays outside.
        DNDS_DEVICE_CALLABLE std::array<real, 25> NegativeSourceJacobian() const noexcept
        {
            std::array<real, 25> matrix{};
            const auto &w = angularVelocity;
            matrix[1 + 5 * 2] = -w[2];
            matrix[1 + 5 * 3] = w[1];
            matrix[2 + 5 * 1] = w[2];
            matrix[2 + 5 * 3] = -w[0];
            matrix[3 + 5 * 1] = -w[1];
            matrix[3 + 5 * 2] = w[0];
            return matrix;
        }
    };
    static_assert(std::is_trivially_copyable_v<RigidFrameParameters>);
}
