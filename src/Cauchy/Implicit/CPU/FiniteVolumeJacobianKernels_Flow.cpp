#include "Cauchy/Implicit/CPU/FiniteVolumeJacobianKernels.hpp"
#include "DNDS/Core/OMP.hpp"
#include "Cauchy/Physics/PerfectGasCharacteristics.hpp"
#include "Cauchy/Physics/RiemannPreparation.hpp"
namespace DNDS::Cauchy::CPU
{
    using Matrix = Eigen::Matrix<real, 5, 5>;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;
    namespace
    {
        // Same perfect-gas ALE derivative as the model facade, with setup's
        // validated POD passed directly into the numerical leaf. No capability
        // check or throwing model convenience occurs inside a cell scan.
        std::array<real, 25> FlowFluxJacobian(const PerfectGas &gas, const FlowState &state,
                                              const Vector3 &normal, const Vector3 &gridVelocity) noexcept
        {
            auto result = gas.FluxJacobian(state, normal);
            real speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += gridVelocity[d] * normal[d];
            for (int j = 0; j < 5; ++j)
                result[j + 5 * j] -= speed;
            return result;
        }
    }
    void EvaluateFlowJacobianBlock(const FlowJacobianView &snapshot, const PerfectGas &gas, index iRow, index iBlock, Eigen::Ref<Eigen::MatrixXd> output)
    {
        output.setZero();
        for (const auto &record : snapshot.geometry.Points(iRow, iBlock))
        {
            const auto point = ExpandJacobianPoint(snapshot.geometry.Point(iRow, record));
            const auto right = point.changesFrame ? FiniteVolumeDetail::RotateState(FiniteVolumeDetail::Load(snapshot.means, point.other), point.rotation) : FiniteVolumeDetail::Load(snapshot.means, point.other);
            Matrix dissipation = Matrix::Identity() * (snapshot.radius(point.auxiliary, 0) + snapshot.radius(point.auxiliary, 1));
            if (snapshot.roe)
            {
                const auto checked = TryPerfectGasCharacteristics(gas, FiniteVolumeDetail::Load(snapshot.means, iRow), right, point.normal);
                // The producer validated this immutable linearization. A broken
                // snapshot is fatal here; block actions cannot recover by
                // throwing or silently substituting a new eigensystem.
                DNDS_assert(bool(checked));
                const auto &basis = checked.value;
                Eigen::Matrix<real, 5, 1> lambda;
                const auto signal = snapshot.magnitudes[point.sample];
                lambda << signal[0], signal[0], signal[0], signal[1], signal[2];
                dissipation = Eigen::Map<const Matrix>(basis.inverse.data()) * lambda.asDiagonal() * Eigen::Map<const Matrix>(basis.forward.data());
                dissipation.diagonal().array() += snapshot.radius(point.auxiliary, 1);
            }
            const auto derivative = FlowFluxJacobian(gas, right, point.normal, snapshot.frame.GridVelocity(point.location));
            const Matrix other = (.5 * snapshot.scale * point.weight) * (Eigen::Map<const Matrix>(derivative.data()) - dissipation);
            if (!point.changesFrame)
                output += other;
            else
            {
                Matrix rotation = Matrix::Identity();
                rotation.block<3, 3>(1, 1) = point.rotation;
                output.noalias() += other * rotation;
            }
        }
        // Project after the periodic rotation: snapshot.selected components refer
        // to the stored, mesh-bound equation snapshot.frame on both endpoints.
        if (snapshot.selected)
            for (int j = 0; j < 5; ++j)
                for (int k = 0; k < 5; ++k)
                    if (!snapshot.active[j] || !snapshot.active[k])
                        output(j, k) = 0;
    }
    void ApplyFlowJacobianBlock(const FlowJacobianView &snapshot, const PerfectGas &gas, index iRow, index iBlock, JacobianInput input, Eigen::Ref<Eigen::VectorXd> output)
    {
        output.setZero();
        for (const auto &record : snapshot.geometry.Points(iRow, iBlock))
        {
            const auto point = ExpandJacobianPoint(snapshot.geometry.Point(iRow, record));
            const auto right = point.changesFrame ? FiniteVolumeDetail::RotateState(FiniteVolumeDetail::Load(snapshot.means, point.other), point.rotation) : FiniteVolumeDetail::Load(snapshot.means, point.other);
            FlowState direction{};
            for (int j = 0; j < 5; ++j)
                direction[j] = snapshot.active[j] ? input[j] : 0;
            if (point.changesFrame)
                direction = FiniteVolumeDetail::RotateState(direction, point.rotation);
            real dp = direction[4];
            real speedSquared = 0;
            for (int d = 0; d < 3; ++d)
            {
                const real velocity = right[1 + d] / right[0];
                speedSquared += velocity * velocity;
                dp -= velocity * direction[1 + d];
            }
            dp = (gas.gamma - 1) * (dp + .5 * speedSquared * direction[0]);
            const auto derivative = PrepareCompressibleFluxDerivative(right, direction, gas.Pressure(right), dp,
                                                                      point.normal, snapshot.frame.GridVelocity(point.location));
            const Eigen::Map<const Eigen::Matrix<real, 5, 1>> delta(direction.data());
            Eigen::Matrix<real, 5, 1> dissipation = (snapshot.radius(point.auxiliary, 0) + snapshot.radius(point.auxiliary, 1)) * delta;
            if (snapshot.roe)
            {
                const auto checked = TryPerfectGasCharacteristics(gas, FiniteVolumeDetail::Load(snapshot.means, iRow), right, point.normal);
                // The producer validated this immutable linearization. A broken
                // snapshot is fatal here; block actions cannot recover by
                // throwing or silently substituting a new eigensystem.
                DNDS_assert(bool(checked));
                const auto &basis = checked.value;
                auto characteristic = (Eigen::Map<const Matrix>(basis.forward.data()) * delta).eval();
                const auto signal = snapshot.magnitudes[point.sample];
                for (int j = 0; j < 5; ++j)
                    characteristic[j] *= signal[j < 3 ? 0 : j - 2];
                dissipation.noalias() = Eigen::Map<const Matrix>(basis.inverse.data()) * characteristic;
                dissipation += snapshot.radius(point.auxiliary, 1) * delta;
            }
            const real factor = .5 * snapshot.scale * point.weight;
            for (int j = 0; j < 5; ++j)
                if (snapshot.active[j])
                    output[j] += factor * (derivative[j] - dissipation[j]);
        }
    }
    void FreezeFlowJacobian(RHSFlowFieldView mean, RHSScalarView radius,
                            Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<5>> frozenMean,
                            Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> frozenRadius)
    {
        OMP::ForRows(0, frozenMean.RowCount(), [&](index iCell)
                     {
            for (int j = 0; j < 5; ++j)
                frozenMean(iCell, j) = mean(iCell, j); });
        OMP::ForRows(0, frozenRadius.RowCount(), [&](index iFace)
                     {
            for (int j = 0; j < 2; ++j)
                frozenRadius(iFace, j) = radius(iFace, j); });
    }
    NumericalStatus PrepareCompressibleFluxJacobian(const FlowJacobianPreparation &request, const PerfectGas &gas, const Riemann::RiemannFlux &flux, Runtime::RowSelection rows) noexcept
    {
        const auto &packet = request.packet;
        const auto &mean = request.mean;
        const auto &frame = request.frame;
        const auto &metric = request.metric;
        const auto &blocks = request.blocks;
        const auto radius = request.radius, inverseTau = request.inverseTau, incoming = request.incoming;
        const auto d = request.diagonal;
        const auto &active = request.active;
        const real time = request.time;
        const auto sourceDerivative = frame.NegativeSourceJacobian();
        const real rotationBound = Geom::tPoint{frame.angularVelocity[0], frame.angularVelocity[1], frame.angularVelocity[2]}.norm();
        const auto prepared = [&](index iCell, const FlowState &u) -> NumericalResult<Riemann::AcousticState>
        {
            if (!gas.Admissible(u))
                return NumericalStatus{NumericalError::InvalidState, 5, iCell};
            const real pressure = packet.meanThermodynamics ? (*packet.meanThermodynamics)(iCell, 0) : gas.Pressure(u);
            const real sound = packet.meanThermodynamics ? (*packet.meanThermodynamics)(iCell, 1) : std::sqrt(gas.gamma * pressure / u[0]);
            if (!(std::isfinite(pressure) && pressure > 0 && std::isfinite(sound) && sound > 0))
                return NumericalStatus{NumericalError::InvalidState, 5, iCell};
            return Riemann::AcousticState{{u, pressure}, sound};
        };
        return OMP::ForRowsStatus(0, rows.RowCount(), [&](index slot) -> NumericalStatus
                                  {
            const index iCell = rows.RowAt(slot);
            const real inverseStep = inverseTau(iCell, 0);
            if (!(std::isfinite(inverseStep) && inverseStep >= 0))
                return NumericalStatus{NumericalError::InvalidState, 1, iCell};
            Matrix own = Matrix::Identity() * (metric.GetCellVol(iCell) * (inverseStep + request.physicalMass));
            if (frame.moving)
            {
                if (request.fullBlock)
                    own += (request.scale * metric.GetCellVol(iCell)) * Eigen::Map<const Matrix>(sourceDerivative.data());
                else
                    // Scalar entries cannot represent the skew rotation block.
                    // Retain an explicit norm bound as a preconditioning term.
                    own.diagonal().array() += request.scale * metric.GetCellVol(iCell) * rotationBound;
            }
            if (blocks)
                for (index j = 0; j < blocks->BlockNumber(iCell); ++j)
                    (*blocks)(iCell, j).setZero();
            const auto left = Load(mean, iCell);
            for (index entry = request.stencil.jacobianOffsets(iCell, 0); entry < request.stencil.jacobianOffsets(iCell + 1, 0); ++entry)
            {
                const auto sample = FiniteVolumeDetail::ExpandSample(request.stencil.JacobianSample(iCell, entry), request.conditions);
                const auto iAux = sample.auxiliaryFace;
                const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]}, point{sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]};
                const auto gridVelocity = frame.GridVelocity(point);
                const real convection = radius(iAux, 0), diffusion = radius(iAux, 1);
                if (!(std::isfinite(convection) && convection >= 0 && std::isfinite(diffusion) && diffusion >= 0))
                    return NumericalStatus{NumericalError::InvalidState, 2, iCell};
                const real factor = .5 * request.scale * metric.GetFaceArea(sample.face);
                if (!request.roe && sample.other == UnInitIndex)
                {
                    own.diagonal().array() += factor * (convection + diffusion);
                    continue; // Euler's scalar diagonal does not linearize BCs.
                }
                const auto rightResult = sample.other == UnInitIndex ? FiniteVolumeDetail::BoundaryExterior(sample, gas, left, time, frame, incoming)
                                                                     : NumericalResult<FlowState>{!sample.changesFrame ? Load(mean, sample.other) : RotateState(Load(mean, sample.other), sample.rotation)};
                if (!rightResult)
                    return rightResult.status.At(iCell, sample.point);
                const auto &right = rightResult.value;
                Matrix dissipation = Matrix::Identity() * (convection + diffusion);
                if (request.roe)
                {
                    const auto checked = TryPerfectGasCharacteristics(gas, left, right, normal);
                    if (!checked)
                        return checked.status.At(iCell, sample.point);
                    const auto &characteristic = checked.value;
                    // A componentwise limiter fallback is not a Roe matrix.
                    // Retain the physical failure and identify the mean states
                    // before the application aborts the distributed solve.
                    if (characteristic.componentwiseFallback)
                        return NumericalStatus{NumericalError::InvalidState, 3, iCell};
                    const auto leftPrepared = prepared(iCell, left);
                    const real rightPressure = sample.other == UnInitIndex ? gas.Pressure(right) : 0;
                    const auto rightPrepared = sample.other == UnInitIndex
                        ? NumericalResult<Riemann::AcousticState>{Riemann::AcousticState{{right, rightPressure}, std::sqrt(gas.gamma * rightPressure / right[0])}}
                        : prepared(sample.other, right);
                    if (!leftPrepared) return leftPrepared.status.At(iCell, sample.point);
                    if (!rightPrepared) return rightPrepared.status.At(iCell, sample.point);
                    const auto &leftData = leftPrepared.value, &rightData = rightPrepared.value;
                    const auto model = PreparePerfectGasRiemann(gas, flux.Kind(), leftData, rightData, leftData, rightData, normal, gridVelocity);
                    auto settings = request.fluxSettings;
                    if (packet.cellWaveJump)
                    {
                        real jump = (*packet.cellWaveJump)(iCell, 0);
                        if (sample.other != UnInitIndex)
                            jump = std::max(jump, (*packet.cellWaveJump)(sample.other, 0));
                        settings.dLambda = std::max(settings.dLambda, jump);
                    }
                    if (packet.cellPressureMinimum)
                    {
                        real minimum = (*packet.cellPressureMinimum)(iCell, 0);
                        if (sample.other != UnInitIndex)
                            minimum = std::min(minimum, (*packet.cellPressureMinimum)(sample.other, 0));
                        settings.pressureStencilMin = std::min(settings.pressureStencilMin, minimum);
                    }
                    const auto signals = flux.Evaluate(model.Input(leftData, rightData, normal, gridVelocity), settings);
                    if (request.magnitudes)
                    {
                        (*request.magnitudes)(entry, 0) = signals.contact;
                        (*request.magnitudes)(entry, 1) = signals.acousticPlus;
                        (*request.magnitudes)(entry, 2) = signals.acousticMinus;
                    }
                    Eigen::Matrix<real, 5, 1> magnitudes;
                    magnitudes << signals.contact, signals.contact, signals.contact, signals.acousticPlus, signals.acousticMinus;
                    dissipation = Eigen::Map<const Matrix>(characteristic.inverse.data()) * magnitudes.asDiagonal() * Eigen::Map<const Matrix>(characteristic.forward.data());
                    dissipation.diagonal().array() += diffusion;
                }
                own += factor * dissipation;
                if (request.evaluated && sample.other != UnInitIndex && sample.other != iCell)
                    continue; // Physical derivative is regenerated by the evaluator.
                const auto derivative = FlowFluxJacobian(gas, right, normal, gridVelocity);
                Matrix other = factor * (Eigen::Map<const Matrix>(derivative.data()) - dissipation);
                if (sample.other == UnInitIndex)
                {
                    if (request.roe)
                    {
                        const auto boundary = sample.boundary.ApproximateExteriorJacobian(
                            gas, left, normal, point, time,
                            OptionalRef<const RigidFrame>(frame));
                        own.noalias() += other * Eigen::Map<const Matrix>(boundary.data());
                    }
                }
                else
                {
                    if (sample.changesFrame)
                    {
                        Matrix rotation = Matrix::Identity();
                        rotation.block<3, 3>(1, 1) = sample.rotation;
                        other = (other * rotation).eval();
                    }
                    if (sample.other == iCell)
                        own += other;
                    else if (blocks)
                    {
                        DNDS_assert(sample.jacobianColumn != UnInitIndex);
                        (*blocks)(iCell, sample.jacobianColumn) += other;
                    }
                }
            }
            if (!(own.allFinite()))
                return NumericalStatus{NumericalError::InvalidState, 4, iCell};
            if (request.selected)
            {
                for (int j = 0; j < 5; ++j)
                    for (int k = 0; k < 5; ++k)
                        if (!active[j] || !active[k])
                            own(j, k) = j == k ? 1 : 0;
                if (blocks)
                    for (index entry = 0; entry < blocks->BlockNumber(iCell); ++entry)
                        for (int j = 0; j < 5; ++j)
                            for (int k = 0; k < 5; ++k)
                                if (!active[j] || !active[k])
                                    (*blocks)(iCell, entry)(j, k) = 0;
            }
            if (request.fullBlock)
                for (int j = 0; j < 25; ++j)
                    d(iCell, request.rowMajorBlocks ? (j % 5) * 5 + j / 5 : j) = own.data()[j];
            else
                for (int j = 0; j < 5; ++j)
                    d(iCell, j) = own(j, j);
            return {}; });
    }
}
