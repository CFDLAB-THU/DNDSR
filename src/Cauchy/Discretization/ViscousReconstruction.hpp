#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    /// Select the pre-filter derivatives for polynomial viscous fluxes.
    /// LVR changes the reconstruction equations themselves: its sweep has
    /// a single solution, even when unlimited viscous gradients are requested.
    inline bool UsesUnlimitedViscousReconstruction(const DomainSettings &settings)
    {
        const auto &n = settings.numerics;
        if (n.viscousLimited || n.viscousGradient != "polynomial")
            return false;
        if (n.reconstruction == "gradient")
            return n.limiter != "none" || settings.GradientPositivity();
        return n.reconstruction == "variational" && (n.limiter == "WBAP" || n.UsesRANSGradientBarth() || n.positivity);
    }
}
