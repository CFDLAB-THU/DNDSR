#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    /// Select a separate unlimited derivative for viscous fluxes. Modal RANS
    /// P1 forcing updates both coefficient roles, so Barth alone only splits
    /// the viscous input for direct physical-gradient reconstruction.
    inline bool UsesUnlimitedViscousReconstruction(const DomainSettings &settings)
    {
        const auto &n = settings.numerics;
        if (n.viscousLimited || n.viscousGradient != "polynomial")
            return false;
        if (n.reconstruction == "gradient")
            return n.limiter != "none" || settings.GradientPositivity();
        const bool lvr = n.limiter == "LVR" && (n.limitedVR.alphaMax > 0 || n.limitedVR.contactAlphaMax > 0);
        return n.reconstruction == "variational" && (n.limiter == "WBAP" || lvr || n.positivity);
    }
}
