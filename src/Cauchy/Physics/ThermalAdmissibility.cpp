#include "Cauchy/Physics/ThermalAdmissibility.hpp"

namespace DNDS::Cauchy
{
    ThermalPolicy::ThermalPolicy(const EquationSystem &equation, real densityFloor, real temperatureFloor,
                                 real sensibleEnergyFloor, real pressureFloor, Physics::Chemistry::ChemicalWorkerPool::Lease worker)
    {
        if (equation.IsACM())
        {
            _view.flowFirst = _view.speciesFirst = -1;
            for (const auto &block : equation.Schema().blocks)
                _view.components += index(block.components.size());
            return;
        }
        DNDS_check_throw_info(std::isfinite(densityFloor) && densityFloor > 0 && std::isfinite(temperatureFloor) && temperatureFloor > 0 &&
                                  std::isfinite(sensibleEnergyFloor) && sensibleEnergyFloor >= 0 && std::isfinite(pressureFloor) && pressureFloor > 0,
                              "invalid equation thermal bounds");
        _view.densityFloor = densityFloor;
        _view.flowFirst = _view.speciesFirst = -1;
        for (const auto &block : equation.Schema().blocks)
        {
            if (block.name == "flow")
            {
                DNDS_check_throw_info(block.components.size() == 5, "thermal policy requires the compressible flow5 profile");
                _view.flowFirst = _view.components;
            }
            if (block.name == "species")
                _view.speciesFirst = _view.components;
            _view.components += index(block.components.size());
        }
        DNDS_check_throw_info(_view.flowFirst >= 0, "thermal policy requires a flow block");
        if (equation.HasMixture())
        {
            const auto &material = equation.Mixture();
            _energies[0] = Runtime::MetadataBuffer<real>(material.BaseSpecificInternalEnergies());
            _energies[1] = Runtime::MetadataBuffer<real>(material.SpecificInternalEnergies(std::max(temperatureFloor, material.BaseTemperature()), std::move(worker)));
            _view.independent = material.SpeciesNumber() - 1;
            _view.floors = {sensibleEnergyFloor, 0};
        }
        else
        {
            DNDS_check_throw_info(!worker, "perfect gas thermal policy requires no chemistry worker");
            const auto &gas = equation.Gas();
            _view.densityCoefficients = {0, gas.gasConstant * temperatureFloor / (gas.gamma - 1)};
            _view.floors = {pressureFloor / (gas.gamma - 1), 0};
        }
    }

    AdmissibilityPolicy::AdmissibilityPolicy(
        const EquationSystem &equation, real densityFloor,
        real temperatureFloor, real sensibleEnergyFloor, real pressureFloor,
        RANSStateView rans,
        Physics::Chemistry::ChemicalWorkerPool::Lease worker)
        : _thermal(equation, densityFloor, temperatureFloor,
                   sensibleEnergyFloor, pressureFloor, std::move(worker)),
          _property(equation.PropertyKey()), _rans(rans)
    {
        index first = 0;
        for (const auto &block : equation.Schema().blocks)
        {
            if (block.name == "rans")
                _ransFirst = first;
            first += index(block.components.size());
        }
        DNDS_check_throw_info(
            equation.HasRANS() == bool(_rans.components),
            "equation and admissibility RANS policies differ");
        if (equation.HasRANS())
            DNDS_check_throw_info(
                _ransFirst >= 0 &&
                    _rans.components == equation.RANS().ComponentNumber() &&
                    _rans.kind == equation.RANS().Kind(),
                "RANS admissibility policy differs from the equation schema");
        else
            DNDS_check_throw_info(_ransFirst < 0,
                                  "non-RANS equation contains a RANS block");
    }
}
