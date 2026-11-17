#pragma once

#include "Physics/RANS/RANSModels.hpp"
#include <Eigen/Core>
#include <array>
#include <type_traits>

namespace DNDS::Physics::RANS
{
    enum class ModelKind
    {
        None,
        SA,
        KOmegaWilcox,
        KOmegaSST,
        RealizableKE
    };

    /// Bounded point scratch, in one Cartesian frame and in code units.
    /// State is [rho,rhou,rhov,rhow,rhoE,RANS0,RANS1]; SA leaves RANS1 zero.
    /// Gradient is (direction, conservative component). The caller gathers
    /// arrays before evaluation; no array, worker or model ownership is retained.
    struct Point
    {
        Eigen::Matrix<real, 7, 1> state = Eigen::Matrix<real, 7, 1>::Zero();
        Eigen::Matrix<real, 3, 7> gradient = Eigen::Matrix<real, 3, 7>::Zero();
        real molecularViscosity = 0, wallDistance = 1, cellLength = 0;
        real cp = 1, gamma = 1.4;
        /// Mechanical closure with normalized unit density and no thermal or species flux.
        bool incompressible = false;
    };
    struct TurbulentTransport
    {
        real viscosity = 0, conductivity = 0, speciesDiffusivity = 0;
    };

    /// Setup-validated, owning-by-value coefficients for shared CPU/CUDA math.
    /// No string, pointer, allocation, registry or configurable callback crosses
    /// the kernel boundary. RealizableKE is deliberately unsupported by these
    /// point kernels; its existing CPU implementation remains available.
    struct PointParameters
    {
        ModelKind kind = ModelKind::None;
        real viscosityReference = 1, turbulentPrandtl = .9, turbulentSchmidt = .9;
        real desScale = veryLargeReal;
        int desMode = 1, saVersion = 0;
        bool saRotationCorrection = true;
        SAConfig sa;
        KOmegaConfig kOmega;
        KOmegaSSTConfig sst;

        DNDS_DEVICE_CALLABLE bool Supported() const noexcept
        {
            return kind == ModelKind::None || kind == ModelKind::SA ||
                   kind == ModelKind::KOmegaWilcox || kind == ModelKind::KOmegaSST;
        }
        DNDS_DEVICE_CALLABLE int ComponentNumber() const noexcept
        {
            return kind == ModelKind::None ? 0 : kind == ModelKind::SA ? 1
                                                                       : 2;
        }
    };
    static_assert(std::is_trivially_copyable_v<PointParameters>);
}
