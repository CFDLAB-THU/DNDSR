#include "Physics/RANS/Model.hpp"
#include "Physics/RANS/PointKernels.hpp"
#include "Physics/Thermodynamics/ConservativeConversion.hpp"

namespace DNDS::Physics::RANS
{
    Model::Model(ModelSettings settings, SSTProduction sstProduction) : _settings(std::move(settings)), _kind(ModelKind::None)
    {
        if (_settings.model == "SA")
            _kind = ModelKind::SA;
        else if (_settings.model == "KOmegaWilcox")
            _kind = ModelKind::KOmegaWilcox;
        else if (_settings.model == "KOmegaSST")
            _kind = ModelKind::KOmegaSST;
        else if (_settings.model == "RealizableKE")
            _kind = ModelKind::RealizableKE;
        else
            DNDS_check_throw_info(_settings.model == "none", "unknown RANS model: " + _settings.model);
        DNDS_check_throw_info(_kind != ModelKind::None || !_settings.useQCR, "QCR requires a RANS closure");
        const auto positive = [](real value)
        { return std::isfinite(value) && value > 0; };
        const auto nonnegative = [](real value)
        { return std::isfinite(value) && value >= 0; };
        DNDS_check_throw_info(positive(_settings.viscosityReference) && positive(_settings.turbulentPrandtl) &&
                                  positive(_settings.turbulentSchmidt) && positive(_settings.desScale),
                              "invalid RANS scales");
        DNDS_check_throw_info(_settings.desMode >= 0 && _settings.desMode <= 3 &&
                                  (_settings.saVersion == 0 || _settings.saVersion == 1 || _settings.saVersion == 9001),
                              "invalid SA variant or DES mode");
        DNDS_check_throw_info(nonnegative(_settings.sa.productionLimit) && nonnegative(_settings.kOmega.productionLimit) &&
                                  nonnegative(_settings.kOmega.turbulentViscosityLimit) && nonnegative(_settings.sst.productionLimit) &&
                                  nonnegative(_settings.sst.turbulentViscosityLimit) && nonnegative(_settings.realizableKE.turbulentViscosityLimit),
                              "invalid RANS kernel limit");
        _parameters = {_kind, _settings.viscosityReference, _settings.turbulentPrandtl, _settings.turbulentSchmidt,
                       _settings.desScale, _settings.desMode, _settings.saVersion, _settings.saRotationCorrection,
                       _settings.sa, _settings.kOmega, _settings.sst, sstProduction};
    }
    std::array<const char *, 2> Model::ComponentNames() const
    {
        switch (_kind)
        {
        case ModelKind::SA:
            return {"rhoNuTilde", nullptr};
        case ModelKind::KOmegaSST:
        case ModelKind::KOmegaWilcox:
            return {"rhoK", "rhoOmega"};
        case ModelKind::RealizableKE:
            return {"rhoK", "rhoEpsilon"};
        default:
            return {nullptr, nullptr};
        }
    }
    TurbulentTransport Model::Transport(const Point &point) const
    {
        real viscosity = 0;
        switch (_kind)
        {
        case ModelKind::SA:
            viscosity = GetMut_SA<3>(point.state, _settings.viscosityReference, point.molecularViscosity);
            break;
        case ModelKind::KOmegaWilcox:
            viscosity = GetMut_KOWilcox<3>(point.state, point.gradient, point.molecularViscosity, point.wallDistance, _settings.kOmega);
            break;
        case ModelKind::KOmegaSST:
            viscosity = GetMut_SST<3>(point.state, point.gradient, point.molecularViscosity, point.wallDistance, _settings.sst);
            break;
        case ModelKind::RealizableKE:
            viscosity = GetMut_RealizableKe<3>(point.state, point.gradient, point.molecularViscosity, point.wallDistance, _settings.realizableKE);
            break;
        default:
            return {};
        }
        return {viscosity, point.cp * viscosity / _settings.turbulentPrandtl,
                viscosity / std::max(point.state[0] * _settings.turbulentSchmidt, verySmallReal)};
    }

    NumericalResult<TurbulentTransport> Model::TryTransport(const Point &point) const noexcept
    {
        if (_parameters.Supported())
            return RANS::TryTransport(_parameters, point);
        const auto result = Transport(point); // CPU-only legacy RealizableKE.
        if (!std::isfinite(result.viscosity) || !std::isfinite(result.conductivity) || !std::isfinite(result.speciesDiffusivity))
            return NumericalStatus{NumericalError::InvalidState};
        return result;
    }

    NumericalResult<std::array<real, 2>> Model::TryDiffusionFlux(const Point &point, const Eigen::Vector3d &normal, real turbulentViscosity) const noexcept
    {
        if (_parameters.Supported())
            return RANS::TryDiffusionFlux(_parameters, point, normal, turbulentViscosity);
        const auto result = DiffusionFlux(point, normal, turbulentViscosity);
        if (!std::isfinite(result[0]) || !std::isfinite(result[1]))
            return NumericalStatus{NumericalError::InvalidState};
        return result;
    }

    NumericalResult<std::array<real, 2>> Model::TrySource(const Point &point, int mode) const noexcept
    {
        if (_parameters.Supported())
            return RANS::TrySource(_parameters, point, mode);
        if (mode != 0 && mode != 1)
            return NumericalStatus{NumericalError::InvalidInput};
        // RealizableKE does not enter the DES/SST cellLength throw path.
        const auto result = EvaluateSource(point, mode);
        if (!std::isfinite(result[0]) || !std::isfinite(result[1]))
            return NumericalStatus{NumericalError::InvalidState};
        return result;
    }
    std::array<real, 2> Model::DiffusionFlux(const Point &point, const Eigen::Vector3d &normal, real turbulentViscosity) const
    {
        if (_kind == ModelKind::None)
            return {};
        // Only density, velocity and RANS gradients enter these kernels. The
        // energy column is deliberately not a pretend temperature/pressure.
        Eigen::Matrix<real, 3, 7> gradient = Eigen::Matrix<real, 3, 7>::Zero();
        gradient.col(0) = point.gradient.col(0);
        for (int j : {1, 2, 3, 5, 6})
            for (int d = 0; d < 3; ++d)
                gradient(d, j) = Thermodynamics::ConservativeSpecificDerivative(point.state[0], point.state[j] / point.state[0],
                                                                                point.gradient(d, 0), point.gradient(d, j));
        Eigen::Matrix<real, 7, 1> flux = Eigen::Matrix<real, 7, 1>::Zero();
        switch (_kind)
        {
        case ModelKind::SA:
            GetVisFlux_SA<3>(point.state, gradient, normal, _settings.viscosityReference, point.molecularViscosity, flux);
            break;
        case ModelKind::KOmegaWilcox:
            GetVisFlux_KOWilcox<3>(point.state, gradient, normal, turbulentViscosity, point.wallDistance, point.molecularViscosity, flux, _settings.kOmega);
            break;
        case ModelKind::KOmegaSST:
            GetVisFlux_SST<3>(point.state, gradient, normal, turbulentViscosity, point.wallDistance, point.molecularViscosity, flux, _settings.sst);
            break;
        case ModelKind::RealizableKE:
            GetVisFlux_RealizableKe<3>(point.state, gradient, normal, turbulentViscosity, point.wallDistance, point.molecularViscosity, flux, _settings.realizableKE);
            break;
        default:
            break;
        }
        return {flux[5], flux[6]};
    }
    std::array<real, 2> Model::EvaluateSource(const Point &point, int mode) const
    {
        if ((_kind == ModelKind::SA && _settings.desScale <= 100) || _kind == ModelKind::KOmegaSST)
            DNDS_check_throw_info(std::isfinite(point.cellLength) && point.cellLength > 0,
                                  "DES/SST source requires an explicit positive cell length");
        Eigen::Matrix<real, 7, 1> source = Eigen::Matrix<real, 7, 1>::Zero();
        switch (_kind)
        {
        case ModelKind::SA:
            GetSource_SA<3>(point.state, point.gradient, _settings.viscosityReference, point.molecularViscosity, point.gamma,
                            std::min(point.wallDistance, std::pow(veryLargeReal, 1. / 6.)), _settings.desScale, point.cellLength, _settings.desMode,
                            source, _settings.saRotationCorrection, mode, _settings.sa, _settings.saVersion);
            break;
        case ModelKind::KOmegaWilcox:
            GetSource_KOWilcox<3>(point.state, point.gradient, point.molecularViscosity, point.wallDistance, source, mode, _settings.kOmega);
            break;
        case ModelKind::KOmegaSST:
            GetSource_SST<3>(point.state, point.gradient, point.molecularViscosity, point.wallDistance,
                             point.cellLength * _settings.desScale, source, mode, _settings.sst, _parameters.sstProduction);
            break;
        case ModelKind::RealizableKE:
            GetSource_RealizableKe<3>(point.state, point.gradient, point.molecularViscosity, point.wallDistance, source, mode, _settings.realizableKE);
            break;
        default:
            break;
        }
        return {source[5], source[6]};
    }
}
