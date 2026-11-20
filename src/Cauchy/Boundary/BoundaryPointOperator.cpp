#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/Boundary/ACMBoundary.hpp"

#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Boundary/CPU/RANSWallKernels.hpp"

#include <cmath>

namespace DNDS::Cauchy
{
    BoundaryPointOperator::BoundaryPointOperator(EquationSystem equation)
        : _equation(std::move(equation))
    {
        for (const auto &block : _equation.Schema().blocks)
        {
            if (block.name == "flow")
                _flowFirst = _components;
            if (block.name == "species")
                _speciesFirst = _components;
            if (block.name == "rans")
                _ransFirst = _components;
            _components += index(block.components.size());
        }
        DNDS_check_throw_info(_flowFirst >= 0 &&
                                  _flowFirst <= _components - (_equation.IsACM() ? 4 : 5),
                              "boundary point operator requires the equation's flow block");
        if (_equation.HasMixture())
        {
            _independentSpecies = _equation.Mixture().SpeciesNumber() - 1;
            if (_independentSpecies)
                DNDS_check_throw_info(
                    _speciesFirst >= 0 &&
                        _speciesFirst <= _components - _independentSpecies,
                    "mixture boundary point operator lacks its species block");
        }
        if (_equation.HasRANS())
            DNDS_check_throw_info(
                _ransFirst >= 0 &&
                    _ransFirst <=
                        _components - _equation.RANS().ComponentNumber(),
                "RANS boundary point operator lacks its model block");
    }

    void BoundaryPointOperator::Evaluate(
        std::vector<real> &point, const BoundaryCondition &condition,
        index interfaceRow, const Vector3 &normal, const Vector3 &location,
        PointEvaluator &evaluator, const RigidFrame &frame, real time,
        BoundaryPointStateKind stateKind,
        OptionalRef<const std::vector<real>> mean, real meanWallDistance,
        InterfaceBoundaryRows interfaceValues) const
    {
        const auto status = TryEvaluate(point, condition, interfaceRow, normal, location, evaluator, frame, time, stateKind, mean, meanWallDistance, interfaceValues);
        DNDS_check_throw_info(status, "boundary point operation failed");
    }
    NumericalStatus BoundaryPointOperator::TryEvaluate(
        std::vector<real> &point, const BoundaryCondition &condition,
        index interfaceRow, const Vector3 &normal, const Vector3 &location,
        PointEvaluator &evaluator, const RigidFrame &frame, real time,
        BoundaryPointStateKind stateKind,
        OptionalRef<const std::vector<real>> mean, real meanWallDistance,
        InterfaceBoundaryRows interfaceValues) const noexcept
    {
        if (!(index(point.size()) == _components))
            return NumericalStatus{NumericalError::InvalidState, 1, -1};
        if (condition.kind == BoundaryKind::Interface)
        {
            if (!(
                    interfaceValues.data && interfaceRow >= 0 &&
                    interfaceRow < interfaceValues.RowCount() &&
                    interfaceValues.Width() == _components))
                return NumericalStatus{NumericalError::InvalidState, 2, -1};
            for (index j = 0; j < _components; ++j)
                point[size_t(j)] = interfaceValues(interfaceRow, j);
            return {};
        }

        const auto kind = condition.KindAt(location, OptionalRef<const RigidFrame>(frame));
        const int ransComponents = _equation.HasRANS()
                                       ? _equation.RANS().ComponentNumber()
                                       : 0;
        const bool twoEquationWall =
            ransComponents == 2 && kind == BoundaryKind::NoSlipWall;
        if (!(
                !twoEquationWall ||
                (mean && index(mean->size()) == _components &&
                 std::isfinite(meanWallDistance) && meanWallDistance > 0)))
            return NumericalStatus{NumericalError::InvalidState, 3, -1};

        if (_equation.IsACM())
        {
            const auto &params = _equation.ACMParameters();
            const real omega = twoEquationWall ? params.kinematicViscosity * Physics::RANS::kWallOmegaCoeff / sqr(meanWallDistance) : 0;
            std::array<real, 6> exterior{};
            if (!ACM::ExteriorAt(params, kind, point, condition, normal, location, frame.KernelParameters(), exterior, int(_components), omega))
                return {NumericalError::UnsupportedRequest};
            for (int j = 0; j < _components; ++j)
                point[j] = stateKind == BoundaryPointStateKind::ExteriorState ? exterior[j] : 0.5 * (point[j] + exterior[j]);
            return {};
        }
        FlowState inside{};
        for (int j = 0; j < 5; ++j)
            inside[j] = point[size_t(_flowFirst + j)];
        const Physics::Chemistry::SpeciesBufferView rhoY{
            _independentSpecies ? point.data() + _speciesFirst : nullptr,
            _independentSpecies};
        const bool exterior = stateKind == BoundaryPointStateKind::ExteriorState;
        const auto flowResult = exterior
                                    ? condition.TryExterior(evaluator, inside, rhoY, normal,
                                                            location, time,
                                                            OptionalRef<const RigidFrame>(frame))
                                    : condition.TryFaceValue(evaluator, inside, rhoY, normal,
                                                             location, time,
                                                             OptionalRef<const RigidFrame>(frame));
        if (!flowResult)
            return flowResult.status;
        const auto &flow = flowResult.value;
        for (int j = 0; j < 5; ++j)
            point[size_t(_flowFirst + j)] = flow[j];

        if (!ransComponents)
            return {};
        if (kind == BoundaryKind::TotalConditionInlet)
        {
            for (int j = 0; j < ransComponents; ++j)
                point[size_t(_ransFirst + j)] = flow[0] * condition.totalRANSPrimitive[j];
            return {};
        }
        if (kind == BoundaryKind::NoSlipWall)
        {
            std::array<real, 2> target{};
            if (twoEquationWall)
            {
                FlowState viscosityFlow{};
                for (int j = 0; j < 5; ++j)
                    viscosityFlow[j] = (*mean)[size_t(_flowFirst + j)];
                Physics::Chemistry::ConstSpeciesBufferView viscositySpecies{
                    _independentSpecies
                        ? mean->data() + _speciesFirst
                        : nullptr,
                    _independentSpecies};
                if (condition.isothermal)
                {
                    viscosityFlow = flow;
                    viscositySpecies = {rhoY.data, rhoY.nSpecies};
                }
                PointProperties properties;
                if (const auto status = evaluator.TryEvaluate(
                        viscosityFlow, viscositySpecies, 0, {true}, properties);
                    !status)
                    return status;
                Physics::RANS::Point meanPoint;
                for (int j = 0; j < 5; ++j)
                    meanPoint.state[j] = (*mean)[size_t(_flowFirst + j)];
                for (int j = 0; j < ransComponents; ++j)
                    meanPoint.state[5 + j] = (*mean)[size_t(_ransFirst + j)];
                meanPoint.molecularViscosity = properties.transport.viscosity;
                meanPoint.wallDistance = meanWallDistance;
                target = CPU::RANSWallTarget(_equation.RANS(), meanPoint, inside[0]);
            }
            for (int j = 0; j < ransComponents; ++j)
            {
                real reflected = point[size_t(_ransFirst + j)];
                if (condition.isothermal && (ransComponents == 1 || j == 0))
                    reflected *= flow[0] / inside[0];
                point[size_t(_ransFirst + j)] =
                    exterior ? 2 * target[j] - reflected : target[j];
            }
            return {};
        }
        if (kind == BoundaryKind::Farfield || kind == BoundaryKind::Analytic)
        {
            const auto grid = frame.GridVelocity(location);
            real speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += (inside[1 + d] / inside[0] - grid[d]) * normal[d];
            if (kind == BoundaryKind::Analytic || speed <= 0)
                for (int j = 0; j < ransComponents; ++j)
                    point[size_t(_ransFirst + j)] = condition.prescribedRANS[j];
        }
        return {};
    }

    void BoundaryPointOperator::ApproximateExteriorJacobian(
        Eigen::Ref<Eigen::MatrixXd> output,
        const BoundaryCondition &condition, const std::vector<real> &inside,
        const Vector3 &normal, const Vector3 &location,
        const RigidFrame &frame, real time) const
    {
        const auto status = TryApproximateExteriorJacobian(output, condition, inside, normal, location, frame, time);
        DNDS_check_throw_info(status, "boundary point operation failed");
    }
    NumericalStatus BoundaryPointOperator::TryApproximateExteriorJacobian(
        Eigen::Ref<Eigen::MatrixXd> output,
        const BoundaryCondition &condition, const std::vector<real> &inside,
        const Vector3 &normal, const Vector3 &location,
        const RigidFrame &frame, real time) const noexcept
    {
        if (!(
                index(inside.size()) == _components &&
                output.rows() == _components && output.cols() == _components))
            return NumericalStatus{NumericalError::InvalidState, 1, -1};
        output.setZero();
        FlowState flow{};
        for (int j = 0; j < 5; ++j)
            flow[j] = inside[size_t(_flowFirst + j)];
        const auto derivative = condition.ApproximateExteriorJacobian(
            flow, normal, location, time, OptionalRef<const RigidFrame>(frame));
        for (int column = 0; column < 5; ++column)
            for (int row = 0; row < 5; ++row)
                output(_flowFirst + row, _flowFirst + column) =
                    derivative[size_t(row + 5 * column)];
        const auto kind = condition.KindAt(location, OptionalRef<const RigidFrame>(frame));
        const bool wall = kind == BoundaryKind::SlipWall || kind == BoundaryKind::NoSlipWall;
        if (!wall)
            return {};
        for (index j = 0; j < _components; ++j)
            if (j < _flowFirst || j >= _flowFirst + 5)
                output(j, j) = 1;
        if (kind == BoundaryKind::NoSlipWall &&
            _equation.HasRANS())
            for (int j = 0; j < _equation.RANS().ComponentNumber(); ++j)
                output(_ransFirst + j, _ransFirst + j) = -1;
        // The optional SA wall-gradient correction belongs to the viscous
        // reconstruction leaf. It must not alter this exterior-state
        // derivative: the centralized wall law reflects every transported
        // RANS scalar as 2*target-interior, hence its diagonal is exactly -1.
        return {};
    }

}
