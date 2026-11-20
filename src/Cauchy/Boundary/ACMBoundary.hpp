#pragma once

#include "Cauchy/Boundary/BoundaryKind.hpp"
#include "Cauchy/Physics/ACMFlux.hpp"
#include "Cauchy/Motion/RigidFrameParameters.hpp"

namespace DNDS::Cauchy::ACM
{
    /// Reference mechanical boundary data in solver units, uploaded once
    /// during placement and shared by spatial and implicit device kernels.
    struct BoundaryParameters
    {
        BoundaryKind kind = BoundaryKind::Interior;
        std::array<real, 6> acmState{};
        std::array<real, 3> velocity{};
        bool wallMovesWithMesh = true;
    };
    /// Generate the ACM Riemann exterior in the current local axes. Slip and
    /// Dirichlet data are reflected across the trace. Farfield imports only
    /// incoming modes of Gamma^-1 (A_n - w_n M) evaluated at the interior point.
    template <class State, class Prescribed, class Wall, class Normal, class Output>
    DNDS_DEVICE_CALLABLE inline bool Exterior(const Parameters &p, BoundaryKind kind, const State &inside,
                                              const Prescribed &prescribed, const Wall &wall, const Normal &normal,
                                              Output &out, int width, real wallOmega = 0, real gridNormal = 0) noexcept
    {
        const real un = NormalSpeed(inside, normal);
        for (int j = 0; j < width; ++j)
            out[j] = inside[j];
        switch (kind)
        {
        case BoundaryKind::Extrapolate:
            break;
        case BoundaryKind::SlipWall:
            for (int d = 0; d < 3; ++d)
                out[d] -= 2 * (un - NormalSpeed(wall, normal)) * normal[d];
            break;
        case BoundaryKind::NoSlipWall:
            for (int d = 0; d < 3; ++d)
                out[d] = 2 * wall[d] - inside[d];
            if (width > 4)
                out[4] = -inside[4];
            if (width > 5)
                out[5] = 2 * wallOmega - inside[5];
            break;
        case BoundaryKind::VelocityInlet:
            for (int j = 0; j < width; ++j)
                if (j != 3)
                    out[j] = 2 * prescribed[j] - inside[j];
            break;
        case BoundaryKind::PressureOutlet:
            out[3] = 2 * prescribed[3] - inside[3];
            break;
        case BoundaryKind::Farfield:
        {
            real jump[4], incoming[4];
            for (int j = 0; j < 4; ++j)
                jump[j] = prescribed[j] - inside[j];
            ApplyWaveFunction(p, inside, normal, jump, IncomingWave{}, incoming, gridNormal);
            for (int j = 0; j < 4; ++j)
                out[j] += incoming[j];
            if (un < gridNormal)
                for (int j = 4; j < width; ++j)
                    out[j] = prescribed[j];
            break;
        }
        default:
            return false;
        }
        return true;
    }
    /// Inertial inlet/farfield velocities are expressed in the current local axes.
    /// A wall speed specified in mesh axes is relative to the rigid wall motion.
    template <class Condition, class State, class Normal, class Point, class Output>
    DNDS_DEVICE_CALLABLE inline bool ExteriorAt(const Parameters &p, BoundaryKind kind, const State &inside,
                                                const Condition &bc, const Normal &normal, const Point &point,
                                                const RigidFrameParameters &frame, Output &out, int width, real omega) noexcept
    {
        if (!frame.moving)
            return Exterior(p, kind, inside, bc.acmState, bc.velocity, normal, out, width, omega);
        const auto grid = frame.GridVelocity({point[0], point[1], point[2]});
        auto prescribed = bc.acmState;
        const auto velocity = frame.LocalVector({prescribed[0], prescribed[1], prescribed[2]});
        auto wall = frame.LocalVector(bc.velocity);
        for (int d = 0; d < 3; ++d)
        {
            prescribed[d] = velocity[d];
            if (bc.wallMovesWithMesh)
                wall[d] = bc.velocity[d] + grid[d];
        }
        return Exterior(p, kind, inside, prescribed, wall, normal, out, width, omega, NormalSpeed(grid, normal));
    }

}
