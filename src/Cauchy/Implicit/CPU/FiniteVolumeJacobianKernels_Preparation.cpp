#include "Cauchy/Implicit/CPU/FiniteVolumeJacobianKernels.hpp"
#include "DNDS/Algebra/EigenView.hpp"
#include "DNDS/Core/OMP.hpp"
namespace DNDS::Cauchy::CPU
{
    template <int C>
    static NumericalStatus InitializeCompressibleFluxJacobianRows(const EquationJacobianPreparation &request) noexcept
    {
        const auto &snapshot = request.snapshot;
        const auto metric = request.metric;
        const auto tau = request.tau, radius = request.radius;
        const auto d = request.diagonal;
        const index owned = request.owned, nv = C ? C : snapshot.components;
        const auto rotation = snapshot.frame.NegativeSourceJacobian();
        const real rotationBound = Geom::tPoint{snapshot.frame.angularVelocity[0], snapshot.frame.angularVelocity[1], snapshot.frame.angularVelocity[2]}.norm();
        const auto faceStatus = OMP::ForRowsStatus(0, radius.RowCount(), [&](index iFace) -> NumericalStatus
                                                   {
            for (int j = 0; j < 2; ++j)
            {
                if (!std::isfinite(radius(iFace, j)) || radius(iFace, j) < 0)
                    return NumericalStatus{NumericalError::InvalidState}.At(-1, iFace);
                request.frozenRadius(iFace, j) = radius(iFace, j);
            }
            return {}; });
        if (!faceStatus)
            return faceStatus;
        const auto timeStatus = OMP::ForRowsStatus(0, owned, [&](index iCell) -> NumericalStatus
                                                   {
            if (!std::isfinite(tau(iCell, 0)) || tau(iCell, 0) < 0)
                return NumericalStatus{NumericalError::InvalidState}.At(iCell);
            return {}; });
        if (!timeStatus)
            return timeStatus;
        // Initialize before the additive source entrance. Mean EOS work
        // can then be shared with source linearization, without retaining
        // borrowed mixture properties or adding a persistent field cache.
        OMP::ForRows(0, owned, [&](index iCell)
                     {
            const real mass = metric.GetCellVol(iCell) * (tau(iCell, 0) + request.physicalMass);
            // Write the strided row directly: no shared dense scratch, and no
            // per-cell allocation when OMP partitions the independent rows.
            if (request.fullBlock)
                for (index j = 0; j < nv; ++j)
                    for (index i = 0; i < nv; ++i)
                        d(iCell, request.rowMajorBlocks ? i * nv + j : i + nv * j) = i == j ? mass : 0;
            else
                for (index j = 0; j < nv; ++j) d(iCell, j) = mass;
            if (snapshot.frame.moving)
            {
                if (request.fullBlock)
                {
                    for (int j = 0; j < 5; ++j)
                        for (int i = 0; i < 5; ++i)
                            d(iCell, request.rowMajorBlocks ? (snapshot.flowFirst + i) * nv + snapshot.flowFirst + j : snapshot.flowFirst + i + nv * (snapshot.flowFirst + j)) += request.scale * metric.GetCellVol(iCell) * rotation[i + 5 * j];
                }
                else
                    for (int i = 0; i < 5; ++i)
                        d(iCell, snapshot.flowFirst + i) += request.scale * metric.GetCellVol(iCell) * rotationBound;
            } });
        return {};
    }
    template <int C>
    static NumericalStatus PrepareCompressibleFluxJacobianRows(const EquationJacobianPreparation &request, const BoundaryPointOperator &boundaryOperator,
                                                               SpanView<EquationJacobianWorker> workers, bool ownedPropertiesPrepared, const JacobianRowBatches &batches) noexcept
    {
        DNDS_assert(!workers.empty());
        const auto &snapshot = request.snapshot;
        const auto &blocks = request.blocks;
        const auto &linearization = request.linearization;
        const auto &metric = request.metric;
        const auto &inputValues = request.input;
        const auto &wallValues = request.walls;
        const auto incomingFace = request.incoming;
        const auto d = request.diagonal;
        const auto boundaryMeans = request.boundaryMeans;
        const auto b = request.offDiagonal;
        const auto radius = request.radius;
        const auto time = request.time;
        const index owned = request.owned, all = request.all, nv = C ? C : snapshot.components;
        const bool sixComponentShape = b && !b->RowMajorBlocks() && !request.rowMajorBlocks && b->ElementStride() == 1 && d.ElementStride() == 1 && nv == 6 && snapshot.flowFirst == 0 &&
                                       snapshot.allComponents;
        const auto ownedTable = sixComponentShape ? Algebra::TryOwnedTable(inputValues.OwnedPart()) : std::nullopt;
        const auto ghostTable = sixComponentShape ? Algebra::TryGhostTable(inputValues.GhostPart()) : std::nullopt;
        const bool useSixComponentKernel = ownedTable && ghostTable;
        const auto rowRange = [&](index part, index begin, index end)
        {
            const index rows = end - begin;
            const index first = begin + part * (rows / workers.size()) + std::min(part, rows % workers.size());
            return std::pair<index, index>{first, first + rows / workers.size() + (part < rows % workers.size())};
        };
        // Only O(N) scalar thermodynamic linearization data are frozen per
        // cell. Never cache a complete N*N flux matrix per cell/face.
        // A source-free/no-sampling workspace returns false: its owned EOS
        // outputs must still be initialized. Ghost means always need EOS.
        const auto prepareMeanRows = [&](index first, index last)
        {
            if (first == last)
                return NumericalStatus{};
            return OMP::ForRowsStatus(0, workers.size(), [&](index part) -> NumericalStatus
                                      {
        auto &point = workers[part].point;
        auto &evaluator = workers[part].evaluator;
        const auto [begin, end] = rowRange(part, first, last);
        for (index iCell = begin; iCell < end; ++iCell)
        {
            EquationFVDetail::Gather(blocks, iCell, point, false);
            const bool source = request.directChemicalSource && iCell < owned && request.independent > 0;
            PointProperties properties;
            const auto status = evaluator.TryEvaluate(point.Flow(snapshot.flowFirst), point.Species(snapshot.speciesFirst, request.independent), 0, {false, false, true, source, source}, properties);
            if (!status)
                return status.At(iCell);
            const auto stored = linearization.FillRow(iCell, point, properties);
            if (!stored)
                return stored;
            if (source)
                for (int j = 0; j < 5 + request.independent; ++j)
                {
                    const index column = j < 5 ? snapshot.flowFirst + j : snapshot.speciesFirst + j - 5;
                    for (int k = 0; k < request.independent; ++k)
                    {
                        if (!std::isfinite(properties.sourceJacobian(k, j)))
                            return NumericalStatus{NumericalError::InvalidState}.At(iCell);
                        const index row = snapshot.speciesFirst + k;
                        if (request.fullBlock || row == column)
                            d(iCell, request.fullBlock ? (request.rowMajorBlocks ? row * nv + column : row + nv * column) : row) -=
                                request.scale * metric.GetCellVol(iCell) * properties.sourceJacobian(k, j);
                    }
                }
        }
        return {}; });
        };
        // Core matrix rows can be completed using only the owned EOS derivatives.
        // Centered source preparation may already have supplied that owned part.
        const auto meanStatus = prepareMeanRows(ownedPropertiesPrepared ? owned : 0, batches.Split() ? owned : all);
        if (!meanStatus)
            return meanStatus;
        // Prepare the independently sampled face-center exterior rows used
        // by the approximate boundary Jacobian. Interface rows are frozen
        // packet values; physical rows use the same centralized equation
        // boundary operator as RHS/reconstruction. The complete equation
        // state and EOS derivative are stored only for this preparation.
        const EquationFVDetail::MeanLinearizationOutput boundaryLinearization{
            snapshot.flowFirst, snapshot.speciesFirst,
            request.linearization.pressureDerivative, request.linearization.pressure, std::nullopt};
        const auto boundaryStatus = OMP::ForRowsStatus(0, workers.size(), [&](index part) -> NumericalStatus
                                                       {
        auto &point = workers[part].point;
        auto &cellMean = workers[part].cellMean;
        auto &evaluator = workers[part].evaluator;
        const auto [begin, end] = rowRange(part, 0, request.linearizeBoundary ? owned : 0);
        for (index iCell = begin; iCell < end; ++iCell)
        for (index entry = request.stencil.jacobianOffsets(iCell, 0); entry < request.stencil.jacobianOffsets(iCell + 1, 0); ++entry)
        {
            const auto sample = FiniteVolumeDetail::ExpandSample(request.stencil.JacobianSample(iCell, entry), request.conditions);
            if (sample.other != UnInitIndex)
                continue;
            EquationFVDetail::Gather(blocks, sample.cell, point, false);
            cellMean.values = point.values;
            {
                if (const auto status = boundaryOperator.TryEvaluate(
                        point.values, sample.boundary, sample.boundaryPoint,
                        {sample.normal[0], sample.normal[1], sample.normal[2]},
                        {sample.ownPoint[0], sample.ownPoint[1],
                         sample.ownPoint[2]},
                        evaluator, snapshot.frame, time,
                        BoundaryPointStateKind::ExteriorState,
                        OptionalRef<const std::vector<real>>(cellMean.values),
                        request.hasWalls ? wallValues.cellMean(sample.cell, 0) : 0,
                        incomingFace);
                    !status)
                    return status;
            }
            for (index j = 0; j < nv; ++j)
                boundaryMeans(entry, j) = point.values[size_t(j)];
            PointProperties properties;
            const auto status = evaluator.TryEvaluate(
                point.Flow(snapshot.flowFirst),
                point.Species(snapshot.speciesFirst,
                              request.independent),
                0, {false, false, true, false, false}, properties);
            if (!status)
                return status.At(sample.cell, sample.point);
            const auto stored = boundaryLinearization.FillRow(all + entry, point, properties);
            if (!stored)
                return stored.At(sample.cell, sample.point);
        }
        return {}; });
        if (!boundaryStatus)
            return boundaryStatus;
        struct PreparedValues
        {
            const decltype(inputValues) &cells;
            const Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> &boundary;
            index firstBoundary;
            real operator()(index iRow, index j) const
            {
                return iRow < firstBoundary
                           ? cells(iRow, j)
                           : boundary(iRow - firstBoundary, j);
            }
        } preparedValues{inputValues, boundaryMeans, all};
        // A block-shaped output is already the row's dense workspace.
        // Keep its write lease and use the centralized borrowed adapter;
        // diagonal-line storage retains the existing projection scratch.
        const auto addFaceBlock = [&](const FiniteVolumeDetail::SampleValue &sample, real weight, auto &block)
        {
            // The immutable incidence already records an exact frame test.
            // Its common six-component path needs no copied Point/rotation;
            // retain the full point representation for the general kernel.
            if (useSixComponentKernel && !sample.changesFrame)
            {
                const auto row = sample.other < owned ? (*ownedTable)[sample.other] : (*ghostTable)[sample.other - owned];
                const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                const Vector3 location{sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]};
                AddFrozenFluxBlock6({row.data(), 6}, {snapshot.pressureDerivative[sample.other].data(), 6},
                                    snapshot.pressure(sample.other, 0), normal, snapshot.frame.GridVelocity(location),
                                    .5 * snapshot.scale * weight, snapshot.radius(sample.auxiliaryFace, 0) + snapshot.radius(sample.auxiliaryFace, 1),
                                    {block.data(), 36});
                return;
            }
            const JacobianPointValue point{sample.other, sample.auxiliaryFace, UnInitIndex, {sample.normal[0], sample.normal[1], sample.normal[2]}, {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, sample.rotation, weight, sample.changesFrame};
            AddFrozenEquationJacobianPoint(snapshot, point, block, inputValues);
        };
        // Each batch reads completed EOS and boundary rows. Matrix rows use private dense
        // boundary/diagonal scratch and unchanged per-row incidence order.
        for (auto batch = batches.begin(); batch != batches.end(); ++batch)
        {
            if (batch != batches.begin())
            {
                const auto ghostStatus = prepareMeanRows(owned, all);
                if (!ghostStatus)
                    return ghostStatus;
            }
            const auto &rows = *batch;
            const auto batchStatus = OMP::ForRowsStatus(0, workers.size(), [&](index part) -> NumericalStatus
                                                        {
        auto &own = workers[part].own;
        auto &other = workers[part].other;
        auto &boundaryDerivative = workers[part].boundaryDerivative;
        auto &cellMean = workers[part].cellMean;
        const auto [begin, end] = rowRange(part, 0, rows.RowCount());
        const auto assembleFaceRow = [&](index iCell, auto &own) -> NumericalStatus
        {
            if (b)
                for (index j = 0; j < b->BlockNumber(iCell); ++j)
                    (*b)(iCell, j).setZero();
            // First-order mean-state linearization: exactly one action
            // per face incidence, regardless of polynomial degree or
            // face curvature. Accurate geometry was integrated at setup.
            for (index entry = request.stencil.jacobianOffsets(iCell, 0); entry < request.stencil.jacobianOffsets(iCell + 1, 0); ++entry)
            {
                const auto sample = FiniteVolumeDetail::ExpandSample(request.stencil.JacobianSample(iCell, entry), request.conditions);
                const real weight = metric.GetFaceArea(sample.face), lambda = radius(sample.auxiliaryFace, 0) + radius(sample.auxiliaryFace, 1);
                own.diagonal().array() += .5 * request.scale * weight * lambda;
                if (sample.other == UnInitIndex)
                {
                    // Euler's non-Roe diagonal uses only the spectral term
                    // above, for flow and transported components alike. Keep
                    // the physical chain rule as the default; the drop-in
                    // eulerScalar policy isolates this implicit approximation
                    // without changing reconstruction, RHS or source terms.
                    if (!request.linearizeBoundary)
                        continue;
                    // The frozen exterior flux derivative is composed with
                    // dU_exterior/dU_inside. Interface/prescribed targets
                    // return zero; walls retain homogeneous flow reflection,
                    // transported scalars and the -I no-slip RANS rule.
                    other.setZero();
                    const JacobianPointValue boundaryPoint{
                        all + entry, sample.auxiliaryFace, UnInitIndex, {sample.normal[0], sample.normal[1], sample.normal[2]}, {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, Eigen::Matrix3d::Identity(), weight, false};
                    AddFrozenEquationJacobianPoint(snapshot,
                                                   boundaryPoint, other, preparedValues, true, false);
                    for (index j = 0; j < nv; ++j)
                        cellMean.values[size_t(j)] =
                            inputValues(sample.cell, j);
                    {
                        if (const auto status = boundaryOperator.TryApproximateExteriorJacobian(
                                boundaryDerivative, sample.boundary,
                                cellMean.values,
                                {sample.normal[0], sample.normal[1],
                                 sample.normal[2]},
                                {sample.ownPoint[0], sample.ownPoint[1],
                                 sample.ownPoint[2]},
                                snapshot.frame, time);
                            !status)
                            return status;
                    }
                    own.noalias() += other * boundaryDerivative;
                    continue;
                }
                if (!b && sample.other != iCell)
                    continue;
                if (sample.other == iCell)
                {
                    other.setZero();
                    addFaceBlock(sample, weight, other);
                    own += other;
                }
                else
                {
                    if (sample.jacobianColumn == UnInitIndex)
                        return NumericalStatus{NumericalError::InvalidInput}.At(iCell, sample.point);
                    auto destination = (*b)(iCell, sample.jacobianColumn);
                    addFaceBlock(sample, weight, destination);
                }
            }
            for (index j = 0; j < nv; ++j)
                for (index i = 0; i < nv; ++i)
                    if (!snapshot.active[i] || !snapshot.active[j])
                        own(i, j) = real(i == j);
            if (!own.allFinite())
                return NumericalStatus{NumericalError::InvalidState}.At(iCell);
            return {};
        };
        if (request.fullBlock)
        {
            const auto matrices = Algebra::MakeEigenRowMatrices(d, nv, nv, Runtime::Device::Host(), request.rowMajorBlocks);
            for (index slot = begin; slot < end; ++slot)
            {
                const index iCell = rows.RowAt(slot);
                auto block = matrices[iCell];
                const auto status = assembleFaceRow(iCell, block);
                if (!status)
                    return status;
            }
        }
        else
            for (index slot = begin; slot < end; ++slot)
            {
                const index iCell = rows.RowAt(slot);
                own.setZero();
                for (index j = 0; j < nv; ++j)
                    own(j, j) = d(iCell, j);
                const auto status = assembleFaceRow(iCell, own);
                if (!status)
                    return status;
                for (index j = 0; j < nv; ++j)
                    d(iCell, j) = own(j, j);
            }
        return {}; });
            if (!batchStatus)
                return batchStatus;
        }
        return {};
    }
    // Static component counts are valid only for an exact compact table. A
    // segmented/strided or extended mixture uses the same row implementation.
    template <class Kernel>
    static NumericalStatus WithJacobianShape(const EquationJacobianPreparation &request, Kernel kernel) noexcept
    {
        const index width = request.snapshot.components;
        const auto owned = Algebra::TryOwnedTable(request.input.OwnedPart());
        const auto ghost = Algebra::TryGhostTable(request.input.GhostPart());
        if (owned && ghost && owned->stride == width && ghost->stride == width)
        {
            switch (width)
            {
            case 5:
                return kernel(std::integral_constant<int, 5>{});
            case 6:
                return kernel(std::integral_constant<int, 6>{});
            case 7:
                return kernel(std::integral_constant<int, 7>{});
            }
        }
        return kernel(std::integral_constant<int, 0>{});
    }

    NumericalStatus InitializeCompressibleFluxJacobian(const EquationJacobianPreparation &request) noexcept
    {
        return WithJacobianShape(request, [&](auto width)
                                 { return InitializeCompressibleFluxJacobianRows<decltype(width)::value>(request); });
    }

    NumericalStatus PrepareCompressibleFluxJacobian(const EquationJacobianPreparation &request, const BoundaryPointOperator &boundaryOperator,
                                                    SpanView<EquationJacobianWorker> workers, bool ownedPropertiesPrepared, const JacobianRowBatches &batches) noexcept
    {
        return WithJacobianShape(request, [&](auto width)
                                 { return PrepareCompressibleFluxJacobianRows<decltype(width)::value>(request, boundaryOperator, workers, ownedPropertiesPrepared, batches); });
    }

}
