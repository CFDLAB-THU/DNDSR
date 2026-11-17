#pragma once

#include "Cauchy/Boundary/BoundaryKind.hpp"
#include "Cauchy/Physics/ACMFlux.hpp"

namespace DNDS::Cauchy::ACM
{
    /// Stationary mechanical boundary data in solver units, uploaded once
    /// during placement and shared by spatial and implicit device kernels.
    struct BoundaryParameters
    {
        BoundaryKind kind = BoundaryKind::Interior;
        std::array<real, 6> acmState{};
        std::array<real, 3> velocity{};
    };
    /// Generate the stationary ACM Riemann exterior. Prescribed slip and
    /// Dirichlet data are reflected across the trace. Farfield imports only
    /// incoming modes of Gamma^-1 A evaluated at the interior point.
    template <class State, class Prescribed, class Wall, class Normal, class Output>
    DNDS_DEVICE_CALLABLE inline bool Exterior(const Parameters &p, BoundaryKind kind, const State &inside,
                                              const Prescribed &prescribed, const Wall &wall, const Normal &normal,
                                              Output &out, int width, real wallOmega = 0) noexcept
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
            ApplyWaveFunction(p, inside, normal, jump, IncomingWave{}, incoming);
            for (int j = 0; j < 4; ++j)
                out[j] += incoming[j];
            if (un < 0)
                for (int j = 4; j < width; ++j)
                    out[j] = prescribed[j];
            break;
        }
        default:
            return false;
        }
        return true;
    }
}
