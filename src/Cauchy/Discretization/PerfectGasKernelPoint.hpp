#pragma once

#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "Cauchy/Physics/PerfectGasViscousFlux.hpp"
#include "Physics/RANS/PointKernels.hpp"

namespace DNDS::Cauchy
{
    class EquationSystem;
    /// Setup-validated immutable parameters for the combined-AoS spatial
    /// specializations. Kept in a resident metadata buffer, not duplicated in
    /// every mesh row. No host model, registry or chemistry worker is borrowed.
    struct PerfectGasSpatialParameters
    {
        PerfectGas gas;
        Physics::RANS::PointParameters rans;
        int dimension = 3;
        bool viscous = false, useQCR = false;
        /// Resolve physics once on the controller. Mixture or unsupported RANS
        /// profiles fail explicitly before any device publication or launch.
        static PerfectGasSpatialParameters FromEquation(const EquationSystem &equation, int dimension);
        void Validate(int components) const
        {
            gas.Validate();
            DNDS_check_throw_info((dimension == 2 || dimension == 3) && rans.Supported() &&
                                      components == 5 + rans.ComponentNumber(),
                                  "unsupported perfect-gas spatial specialization");
        }
    };
    static_assert(std::is_trivially_copyable_v<PerfectGasSpatialParameters>);

    /// Bounded point scratch shared by CUDA spatial leaves. Gradients are
    /// conservative and Cartesian, [component][direction]; inactive directions
    /// are zero. This type has no ownership, allocation, lease or fallback.
    template <int Components>
    struct PerfectGasKernelPoint
    {
        static_assert(Components >= 5 && Components <= 7);
        std::array<real, Components> values{};
        std::array<Vector3, Components> gradients{};
        DNDS_DEVICE_CALLABLE FlowState Flow() const noexcept
        {
            return {values[0], values[1], values[2], values[3], values[4]};
        }
        template <class Values, class Gradients>
        DNDS_DEVICE_CALLABLE void Load(index iCell, Values mean, Gradients gradient, int dimension, bool withGradient) noexcept
        {
            const auto row = mean[iCell];
            for (int j = 0; j < Components; ++j)
            {
                values[j] = row[j];
                for (int d = 0; d < 3; ++d)
                    gradients[j][d] = withGradient && d < dimension ? gradient(iCell, j * dimension + d) : 0;
            }
        }
        /// R maps donor Cartesian axes to receiver axes. Scalar gradients
        /// transform on the derivative axis; momentum also transforms on its
        /// equation axis. Callers skip this entirely for known identity frames.
        DNDS_DEVICE_CALLABLE void Rotate(const std::array<real, 9> &rotation) noexcept
        {
            const auto oldValues = values;
            const auto oldGradients = gradients;
            for (int j = 0; j < Components; ++j)
                for (int d = 0; d < 3; ++d)
                {
                    gradients[j][d] = 0;
                    for (int e = 0; e < 3; ++e)
                        gradients[j][d] += rotation[d + 3 * e] * oldGradients[j][e];
                }
            const auto spatiallyRotated = gradients;
            for (int j = 0; j < 3; ++j)
            {
                values[1 + j] = 0;
                for (int k = 0; k < 3; ++k)
                    values[1 + j] += rotation[j + 3 * k] * oldValues[1 + k];
                for (int d = 0; d < 3; ++d)
                {
                    gradients[1 + j][d] = 0;
                    for (int k = 0; k < 3; ++k)
                        gradients[1 + j][d] += rotation[j + 3 * k] * spatiallyRotated[1 + k][d];
                }
            }
        }
        DNDS_DEVICE_CALLABLE Physics::RANS::Point RANSPoint(const PerfectGasPoint &properties, real wallDistance, real cellLength = 0) const noexcept
        {
            Physics::RANS::Point point;
            for (int j = 0; j < Components; ++j)
            {
                point.state[j] = values[j];
                for (int d = 0; d < 3; ++d)
                    point.gradient(d, j) = gradients[j][d];
            }
            point.molecularViscosity = properties.transport.viscosity;
            point.cp = properties.thermo.cp;
            point.gamma = properties.thermo.cp / properties.thermo.cv;
            point.wallDistance = wallDistance;
            point.cellLength = cellLength;
            return point;
        }
        /// [pressure, sound speed, diffusion bound, eddy viscosity], matching the generic CPU
        /// auxiliary producer. RANS diffusion uses the same mean/gradient and
        /// wall-distance convention; it does not add an implicit state repair.
        DNDS_DEVICE_CALLABLE NumericalResult<std::array<real, 4>> Auxiliary(const PerfectGasSpatialParameters &parameters, real wallDistance) const noexcept
        {
            const auto properties = TryPerfectGasPoint(parameters.gas, Flow(), parameters.viscous);
            if (!properties)
                return properties.status;
            real bound = properties.value.diffusionBound, eddy = 0;
            if constexpr (Components > 5)
            {
                const auto transport = Physics::RANS::TryTransport(parameters.rans, RANSPoint(properties.value, wallDistance));
                if (!transport)
                    return transport.status;
                eddy = transport.value.viscosity;
                const auto &p = properties.value;
                bound = std::max(bound, std::max(real(4. / 3.) * (p.transport.viscosity + transport.value.viscosity) / values[0],
                                                 (p.transport.conductivity + transport.value.conductivity) / (values[0] * p.thermo.cv)));
            }
            return std::array<real, 4>{properties.value.thermo.pressure, properties.value.thermo.soundSpeed, bound, eddy};
        }
        /// Face point already includes the viscous jump/gradient correction.
        /// No scalar repair occurs here: wall-reflected gradients are deliberate.
        DNDS_DEVICE_CALLABLE NumericalResult<std::array<real, Components>> Diffusion(
            const PerfectGasSpatialParameters &parameters, const Vector3 &normal, real wallDistance, bool adiabatic,
            real preparedViscosity = -1) const noexcept
        {
            const auto properties = TryPerfectGasPoint(parameters.gas, Flow(), true);
            if (!properties)
                return properties.status;
            std::array<real, Components> result{};
            real turbulentViscosity = 0, turbulentConductivity = 0;
            if constexpr (Components > 5)
            {
                const auto point = RANSPoint(properties.value, wallDistance);
                const auto transport = Physics::RANS::TryTransport(parameters.rans, point, preparedViscosity);
                if (!transport)
                    return transport.status;
                turbulentViscosity = transport.value.viscosity;
                turbulentConductivity = transport.value.conductivity;
                const auto flux = Physics::RANS::TryDiffusionFlux(parameters.rans, point, Eigen::Vector3d{normal[0], normal[1], normal[2]}, turbulentViscosity);
                if (!flux)
                    return flux.status;
                for (int j = 0; j < Components - 5; ++j)
                    result[5 + j] = flux.value[j];
            }
            FlowGradient gradient;
            for (int j = 0; j < 5; ++j)
                gradient[j] = gradients[j];
            const auto flux = PreparePerfectGasViscousFlux(Flow(), gradient, properties.value, normal,
                                                           turbulentViscosity, turbulentConductivity, adiabatic, parameters.useQCR);
            for (int j = 0; j < 5; ++j)
                result[j] = flux[j];
            return result;
        }
    };
}
