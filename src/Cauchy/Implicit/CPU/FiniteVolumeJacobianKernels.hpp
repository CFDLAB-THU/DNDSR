#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/Implicit/JacobianGeometry.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSampleKernels.hpp"
#include "Cauchy/Implicit/CPU/JacobianKernels.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "DNDS/Algebra/BlockMatrixStorage.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"

namespace DNDS::Cauchy::CPU
{
    using JacobianInput = Eigen::Ref<const Eigen::VectorXd, 0, Eigen::InnerStride<Eigen::Dynamic>>;
    using JacobianScalarView = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    struct EquationJacobianView
    {
        JacobianGeometryView geometry;
        JacobianScalarView means, pressureDerivative, pressure, radius;
        SpanView<const uint8_t> active;
        RigidFrame frame;
        real scale = 1;
        index components = 0, flowFirst = -1, speciesFirst = -1;
        bool allComponents = true, sixComponentProduct = false, sameComponentFrames = true;
    };
    /// Add the frozen directional flux derivative without acquiring storage.
    /// Selection acts in the stored equation frame, after periodic transforms.
    template <class Means, class Output>
    void AddEquationJacobianPoint(const EquationJacobianView &snapshot,
                                  const JacobianPointValue &point, Output &&output,
                                  const Means &values, bool weighted = true, bool selectComponents = true)
    {
        FlowState flow{};
        for (int j = 0; j < 5; ++j)
            flow[j] = values(point.other, snapshot.flowFirst + j);
        if (point.changesFrame)
            flow = FiniteVolumeDetail::RotateState(flow, point.rotation);
        const auto grid = snapshot.frame.GridVelocity(point.location);
        real speed = 0;
        for (int d = 0; d < 3; ++d)
            speed += (flow[1 + d] / flow[0] - grid[d]) * point.normal[d];
        const real factor = weighted ? .5 * snapshot.scale * point.weight : 1;
        const real lambda = snapshot.radius(point.auxiliary, 0) + snapshot.radius(point.auxiliary, 1);
        for (index column = 0; column < snapshot.components; ++column)
        {
            if (selectComponents && !snapshot.active[column])
                continue;
            FlowState direction{};
            if (column >= snapshot.flowFirst && column < snapshot.flowFirst + 5)
            {
                const auto j = column - snapshot.flowFirst;
                if (point.changesFrame && j >= 1 && j <= 3)
                    for (int d = 0; d < 3; ++d)
                        direction[1 + d] = point.rotation(d, j - 1);
                else
                    direction[j] = 1;
            }
            // dp/dU is frozen in the stored neighbor's snapshot.frame. Applying
            // its rotated direction to the flux directly composes J*R.
            auto derivative = PrepareCompressibleFluxDerivative(flow, direction, snapshot.pressure(point.other, 0), snapshot.pressureDerivative(point.other, column), point.normal, grid);
            // No derivative of the former isotropic 2*rho*k/3 stress or energy work.
            real deltaSpeed = 0;
            for (int d = 0; d < 3; ++d)
                deltaSpeed += (direction[1 + d] - flow[1 + d] / flow[0] * direction[0]) / flow[0] * point.normal[d];
            for (index row = 0; row < snapshot.components; ++row)
            {
                if (selectComponents && !snapshot.active[row])
                    continue;
                real value;
                if (row >= snapshot.flowFirst && row < snapshot.flowFirst + 5)
                {
                    const auto j = row - snapshot.flowFirst;
                    value = derivative[j] - lambda * direction[j];
                }
                else
                    value = PrepareAdvectedScalarFluxDerivative(values(point.other, row), real(row == column), speed, deltaSpeed) - lambda * real(row == column);
                output(row, column) += factor * value;
            }
        }
    }
    /// Closed-form identity-frame convection, retaining all EOS couplings.
    template <class Means, class Output>
    void AddFrozenEquationJacobianPoint(const EquationJacobianView &snapshot,
                                        const JacobianPointValue &point, Output &&output,
                                        const Means &values, bool weighted = true, bool selectComponents = true)
    {
        FlowState flow{};
        for (int j = 0; j < 5; ++j)
            flow[j] = values(point.other, snapshot.flowFirst + j);
        // Component selection does not rotate a snapshot.frame. Keep projection
        // in the generic block path, but skip exact identity transforms.
        const bool sameFrame = !point.changesFrame;
        if (!sameFrame)
        {
            const Geom::tPoint momentum{flow[1], flow[2], flow[3]};
            const auto rotated = (point.rotation * momentum).eval();
            for (int d = 0; d < 3; ++d)
                flow[1 + d] = rotated[d];
        }
        const auto grid = snapshot.frame.GridVelocity(point.location);
        const real inverseDensity = 1 / flow[0];
        real normalVelocity = 0, gridNormalVelocity = 0, scalarSpeed = 0;
        for (int d = 0; d < 3; ++d)
        {
            const real velocity = flow[1 + d] / flow[0];
            normalVelocity += velocity * point.normal[d];
            gridNormalVelocity += grid[d] * point.normal[d];
            scalarSpeed += (velocity - grid[d]) * point.normal[d];
        }
        const real relativeVelocity = normalVelocity - gridNormalVelocity;
        FlowState speedDerivative{};
        speedDerivative[0] = -normalVelocity * inverseDensity;
        for (int j = 0; j < 3; ++j)
            if (sameFrame)
                speedDerivative[1 + j] = point.normal[j] * inverseDensity;
            else
            {
                for (int d = 0; d < 3; ++d)
                    speedDerivative[1 + j] += point.rotation(d, j) * point.normal[d];
                speedDerivative[1 + j] *= inverseDensity;
            }
        const real factor = weighted ? .5 * snapshot.scale * point.weight : 1;
        const real lambda = snapshot.radius(point.auxiliary, 0) + snapshot.radius(point.auxiliary, 1);
        // With no component rotation/projection, J = D + q h^T + t g^T:
        // h=d(v.n)/dU and g is the prepared conservative EOS snapshot.pressure row.
        // Keep distinct flow/passive relative speeds, moving-grid work and
        // arbitrary logical block offsets. No gamma-equivalent closure.
        // Exact identity is intentional: a small physical rotation must
        // still take the fully transformed path below.
        if (sameFrame && (snapshot.allComponents || !selectComponents))
        {
            for (index column = 0; column < snapshot.components; ++column)
            {
                const real deltaSpeed = column >= snapshot.flowFirst && column < snapshot.flowFirst + 5
                                            ? speedDerivative[column - snapshot.flowFirst]
                                            : 0;
                const real dp = snapshot.pressureDerivative(point.other, column);
                output(snapshot.flowFirst, column) += factor * (flow[0] * deltaSpeed + (column == snapshot.flowFirst ? relativeVelocity - lambda : 0));
                for (int d = 0; d < 3; ++d)
                    output(snapshot.flowFirst + 1 + d, column) += factor * (flow[1 + d] * deltaSpeed + dp * point.normal[d] +
                                                                            (column == snapshot.flowFirst + 1 + d ? relativeVelocity - lambda : 0));
                output(snapshot.flowFirst + 4, column) += factor * ((flow[4] + snapshot.pressure(point.other, 0)) * deltaSpeed + dp * normalVelocity +
                                                                    (column == snapshot.flowFirst + 4 ? relativeVelocity - lambda : 0));
                const auto passive = [&](index begin, index end)
                {
                    for (index row = begin; row < end; ++row)
                        output(row, column) += factor * (values(point.other, row) * deltaSpeed +
                                                         (row == column ? scalarSpeed - lambda : 0));
                };
                passive(0, snapshot.flowFirst);
                passive(snapshot.flowFirst + 5, snapshot.components);
            }
            return;
        }
        for (index column = 0; column < snapshot.components; ++column)
        {
            if (selectComponents && !snapshot.active[column])
                continue;
            FlowState direction{};
            real deltaSpeed = 0;
            if (column >= snapshot.flowFirst && column < snapshot.flowFirst + 5)
            {
                const index j = column - snapshot.flowFirst;
                deltaSpeed = speedDerivative[j];
                if (point.changesFrame && j >= 1 && j <= 3)
                    for (int d = 0; d < 3; ++d)
                        direction[1 + d] = point.rotation(d, j - 1);
                else
                    direction[j] = 1;
            }
            const real dp = snapshot.pressureDerivative(point.other, column);
            FlowState derivative{};
            derivative[0] = direction[0] * relativeVelocity + flow[0] * deltaSpeed;
            for (int d = 0; d < 3; ++d)
                derivative[1 + d] = direction[1 + d] * relativeVelocity + flow[1 + d] * deltaSpeed + dp * point.normal[d];
            derivative[4] = direction[4] * relativeVelocity + (flow[4] + snapshot.pressure(point.other, 0)) * deltaSpeed + dp * normalVelocity;
            for (index row = 0; row < snapshot.components; ++row)
            {
                if (selectComponents && !snapshot.active[row])
                    continue;
                const real value = row >= snapshot.flowFirst && row < snapshot.flowFirst + 5
                                       ? derivative[row - snapshot.flowFirst] - lambda * direction[row - snapshot.flowFirst]
                                       : PrepareAdvectedScalarFluxDerivative(values(point.other, row), real(row == column), scalarSpeed, deltaSpeed) - lambda * real(row == column);
                output(row, column) += factor * value;
            }
        }
    }
    void EvaluateEquationJacobianBlock(const EquationJacobianView &, index iRow, index iBlock, Eigen::Ref<Eigen::MatrixXd> output);
    void ApplyEquationJacobianBlock(const EquationJacobianView &, index iRow, index iBlock, JacobianInput input, Eigen::Ref<Eigen::VectorXd> output);

    struct FlowJacobianView
    {
        JacobianGeometryView geometry;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<5>> means;
        JacobianScalarView radius, magnitudes;
        RigidFrame frame;
        real scale = 1;
        bool roe = false, selected = false;
        std::array<bool, 5> active{};
    };
    struct FlowJacobianPreparation
    {
        FiniteVolumeStencilView stencil;
        CFV::FiniteVolumeHostView metric;
        SpanView<const BoundaryCondition> conditions;
        RHSFlowFieldView mean;
        RHSAuxiliaryViews packet;
        RHSScalarView radius, inverseTau, incoming;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> diagonal;
        std::optional<Algebra::BlockMatrixView<real>> blocks;
        std::optional<Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>>> magnitudes;
        RigidFrame frame;
        Riemann::FluxSettings fluxSettings;
        real time = 0, physicalMass = 0, scale = 1;
        bool fullBlock = true, roe = false, evaluated = false, selected = false;
        std::array<bool, 5> active{};
        bool rowMajorBlocks = false;
    };
    void EvaluateFlowJacobianBlock(const FlowJacobianView &, const PerfectGas &, index iRow, index iBlock, Eigen::Ref<Eigen::MatrixXd> output);
    void ApplyFlowJacobianBlock(const FlowJacobianView &, const PerfectGas &, index iRow, index iBlock, JacobianInput input, Eigen::Ref<Eigen::VectorXd> output);
    NumericalStatus PrepareCompressibleFluxJacobian(const FlowJacobianPreparation &, const PerfectGas &, const Riemann::RiemannFlux &) noexcept;
    void FreezeFlowJacobian(RHSFlowFieldView mean, RHSScalarView radius,
                            Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<5>> frozenMean,
                            Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> frozenRadius);

    struct EquationJacobianPreparation
    {
        EquationJacobianView snapshot;
        FiniteVolumeStencilView stencil;
        CFV::FiniteVolumeHostView metric;
        SpanView<const BoundaryCondition> conditions;
        SpanView<const EquationBlockReadView> blocks;
        Algebra::DofComponentView<const real> input;
        RHSScalarView radius, tau, incoming;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> diagonal;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> frozenRadius, boundaryMeans;
        std::optional<Algebra::BlockMatrixView<real>> offDiagonal;
        EquationFVDetail::MeanLinearizationOutput linearization;
        WallDistanceView walls;
        real time = 0, scale = 1, physicalMass = 0;
        index owned = 0, all = 0;
        int independent = 0;
        bool fullBlock = true, directChemicalSource = false, hasWalls = false;
        bool linearizeBoundary = true;
        bool rowMajorBlocks = false;
    };
    /// Initialize mass/rotation and freeze face signals before source addition.
    NumericalStatus InitializeCompressibleFluxJacobian(const EquationJacobianPreparation &) noexcept;
    /// Operator-created private thermodynamics and dense point scratch. No
    /// chemistry worker is acquired, nor scratch resized, inside an OMP team.
    struct EquationJacobianWorker
    {
        PointEvaluator evaluator;
        EquationFVDetail::PointScratch point, cellMean;
        Eigen::MatrixXd own, other, boundaryDerivative;
        EquationJacobianWorker(PointEvaluator properties, index components)
            : evaluator(std::move(properties)), point(components), cellMean(components),
              own(components, components), other(components, components), boundaryDerivative(components, components) {}
    };
    /// Prepare the remaining EOS rows, boundary derivatives and face blocks.
    /// ownedPropertiesPrepared skips only means already emitted by the source kernel.
    NumericalStatus PrepareCompressibleFluxJacobian(const EquationJacobianPreparation &, const BoundaryPointOperator &,
                                                    SpanView<EquationJacobianWorker>, bool ownedPropertiesPrepared) noexcept;
}
