#include "CFV/CPU/ReconstructionKernels.hpp"
#include "CFV/Math/PolynomialReconstruction.hpp"
#include "CFV/Math/ReconstructionMatrixAccess.hpp"
#include "DNDS/Algebra/EigenView.hpp"
#include "DNDS/Core/OMP.hpp"
#include <optional>

namespace DNDS::CFV::CPU
{
    using Matrix = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
    using ConstMap = Eigen::Map<const Matrix>;

    template <class MatrixRows, class VectorRows, class InjectionRows>
    NumericalStatus ReconstructionSweepImpl(const ReconstructionSweepView &view, const ReconstructionWorkSelection &work,
                                            MatrixRows matrices, VectorRows vectors, InjectionRows injection)
    {
        if (view.limited)
        {
            const auto status = OMP::ForRowsStatus(0, work.owned, [&](index iCell) -> NumericalStatus
                                                   {
                if (!std::isfinite(view.penalty(iCell, 0)) || view.penalty(iCell, 0) < 0 || view.penalty(iCell, 0) > 1)
                    return NumericalStatus{NumericalError::InvalidInput}.At(iCell);
                return {}; });
            if (!status)
                return status;
        }
        const index n = view.modes, nv = view.components;
        const real relax = view.relaxation;
        const auto next = view.output;
        const auto current = view.current, ghost = view.ghost, targetValues = view.target, penalty = view.penalty;
        const auto u = view.means;
        const auto bValues = view.boundary;
        const auto offsets = view.offsets, neighbors = view.neighbors, bOffsets = view.boundaryOffsets;
        const auto matrixRows = Algebra::MakeEigenRowMatrices(matrices, n, n);
        const auto vectorRows = Algebra::MakeEigenRowMatrices(vectors, n, 1);
        const auto injectionRows = Algebra::MakeEigenRowMatrices(injection, n, 1);
        const auto rotations = view.rotations;
        const auto rotationActive = view.rotationActive;
        const auto nextMatrices = Algebra::MakeEigenRowMatrices(next, n, nv);
        const auto currentMatrices = Algebra::MakeEigenRowMatrices(current, n, nv);
        std::optional<decltype(Algebra::MakeEigenRowMatrices(current, n, nv))> targetMatrices;
        if (view.limited)
            targetMatrices.emplace(Algebra::MakeEigenRowMatrices(targetValues, n, nv));
        const R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows> nextRead{
            next.data, next.offsets, next.rows, next.width, next.stride, next.sizes, next.kind, next.device};
        const auto visitedMatrices = Algebra::MakeEigenRowMatrices(nextRead, n, nv);
        const auto ghostMatrices = Algebra::MakeEigenRowMatrices(ghost, n, nv);
        const bool coreFirst = work.coreFirst;
        const auto position = [&](index iCell)
        { return coreFirst ? work.coreFirstPosition[iCell] : iCell; };
        auto compute = [&](index iCell, WorkerRowRanges worker)
        {
            Math::SmallMatrix rotatedRec;
            auto output = nextMatrices[iCell];
            output = currentMatrices[iCell] * (1 - relax);
            const real alphaCell = view.limited ? penalty(iCell, 0) : 0;
            const real scale = relax * (1 - alphaCell);
            if (alphaCell > 0)
                output.noalias() += (relax * alphaCell) * (*targetMatrices)[iCell];
            if (scale == 0)
                return;
            Math::SmallMatrix rawRight;
            if (view.raw)
                rawRight.setZero(n, nv);
            for (index iEntry = offsets(iCell, 0); iEntry < offsets(iCell + 1, 0); ++iEntry)
            {
                const index other = neighbors(iEntry, 0);
                if (other != UnInitIndex)
                {
                    const bool rotatedEntry = view.rotated && rotationActive(iEntry, 0);
                    const index matrixRow = view.faceCoupling ? view.faces(iEntry, 0) : iEntry;
                    const bool transpose = view.faceCoupling && view.faceSides(iEntry, 0);
                    const auto block = matrices[matrixRow];
                    // SOR reads completed predecessors from a separate output,
                    // leaving the whole input immutable for direct MPI sends.
                    const auto otherCoefficients = other < work.owned
                                                       ? (view.sor && worker.Contains(other) && position(other) < position(iCell) ? visitedMatrices[other] : currentMatrices[other])
                                                       : ghostMatrices[other - work.owned];
                    // Component rotation is independent of spatial derivative
                    // pullback, already included in the prepared B block.
                    // Apply scalar columns directly, then correct vector blocks.
                    if (view.raw)
                    {
                        // A same-frame periodic self-neighbor was subtracted
                        // from A at setup. Its shared face B must not reappear.
                        if (other != iCell || rotatedEntry)
                            for (index variable = 0; variable < nv; ++variable)
                                for (index i = 0; i < n; ++i)
                                {
                                    real product = 0;
                                    for (index j = 0; j < n; ++j)
                                        product += block[transpose ? j + n * i : i + n * j] * otherCoefficients(j, variable);
                                    rawRight(i, variable) += product;
                                }
                    }
                    else
                        output.noalias() += scale * matrixRows[iEntry] * otherCoefficients;
                    if (rotatedEntry)
                        for (auto first : work.vectorStarts)
                        {
                            const Eigen::Map<const Geom::tGPoint> rotation(rotations[iEntry].data());
                            rotatedRec = otherCoefficients.middleCols(first, 3) * (rotation.transpose() - Geom::tGPoint::Identity());
                            if (view.raw)
                                for (index variable = 0; variable < 3; ++variable)
                                    for (index i = 0; i < n; ++i)
                                    {
                                        real product = 0;
                                        for (index j = 0; j < n; ++j)
                                            product += block[transpose ? j + n * i : i + n * j] * rotatedRec(j, variable);
                                        rawRight(i, first + variable) += product;
                                    }
                            else
                                output.middleCols(first, 3).noalias() += scale * matrixRows[iEntry] * rotatedRec;
                        }
                    for (index variable = 0; variable < nv; ++variable)
                    {
                        real value = u(other, variable);
                        if (rotatedEntry)
                            for (auto first : work.vectorStarts)
                                if (variable >= first && variable < first + 3)
                                {
                                    const Eigen::Map<const Geom::tGPoint> rotation(rotations[iEntry].data());
                                    value = 0;
                                    for (int d = 0; d < 3; ++d)
                                        value += rotation(variable - first, d) * u(other, first + d);
                                }
                        if (view.raw)
                            rawRight.col(variable).noalias() += vectorRows[iEntry] * (value - u(iCell, variable));
                        else
                            output.col(variable).noalias() += vectorRows[iEntry] * (scale * (value - u(iCell, variable)));
                    }
                }
                else
                    for (index iQ = bOffsets(iEntry, 0); iQ < bOffsets(iEntry + 1, 0); ++iQ)
                        for (index variable = 0; variable < nv; ++variable)
                            if (view.raw)
                                rawRight.col(variable).noalias() += injectionRows[iQ] * (bValues(iQ, variable) - u(iCell, variable));
                            else
                                output.col(variable).noalias() += injectionRows[iQ] * (scale * (bValues(iQ, variable) - u(iCell, variable)));
            }
            if (view.raw)
            {
                const auto inverse = view.inverse[iCell];
                for (index variable = 0; variable < nv; ++variable)
                    for (index i = 0; i < n; ++i)
                    {
                        real value = 0;
                        for (index j = 0; j < n; ++j)
                            value += inverse[view.inverseTriangle ? Math::SymmetricTriangleEntry(i, j) : i + n * j] * rawRight(j, variable);
                        output(i, variable) += scale * value;
                    }
            }
        };
        const index owned = work.owned;
        if (!view.sor)
        {
            work.rows.Visit([&](auto rowAt)
                            { OMP::ForRows(0, work.rows.RowCount(), [&](index iRow)
                                           { compute(rowAt(iRow), {{0, owned}, {}}); }); });
        }
        else
        {
            // OMP=1 is the canonical serial sweep. With multiple threads, each
            // mesh subpartition is one ordered block; cross-block reads remain
            // frozen input. A split call preserves core-first logical ordering.
            const auto sweep = [&](WorkerRowRanges worker)
            {
                const index coreCount = index(work.coreCount);
                const auto run = [&](bool core)
                {
                    for (auto range : {worker.first, worker.second})
                        for (index cell = range.begin; cell < range.end; ++cell)
                            if (!coreFirst || (work.coreFirstPosition[cell] < coreCount) == core)
                                compute(cell, worker);
                };
                if (!coreFirst || work.all || work.core)
                    run(true);
                if (coreFirst && (work.all || !work.core))
                    run(false);
            };
            if (OMP::DistributedThreads() == 1)
                sweep({{0, owned}, {}});
            else
                OMP::ForRows(0, index(work.workerRanges.size()), [&](index part)
                             { sweep(work.workerRanges[part]); });
        }
        return {};
    }
    NumericalStatus ReconstructionSweep(const ReconstructionSweepView &view, const ReconstructionWorkSelection &work)
    {
        // One physical-layout dispatch outside every cell/product loop. The
        // canonical AoS path keeps a compile-time unit Eigen inner stride.
        if (view.matrices.kind == R::RowLayoutKind::Table && view.vectors.kind == R::RowLayoutKind::Table &&
            view.injection.kind == R::RowLayoutKind::Table)
        {
            const auto table = [](ReconstructionReadRows v) -> ReconstructionTable
            { return {v.data, v.offsets, v.rows, v.width, v.stride, v.sizes, v.kind, v.device}; };
            return ReconstructionSweepImpl(view, work, table(view.matrices), table(view.vectors), table(view.injection));
        }
        return ReconstructionSweepImpl(view, work, view.matrices, view.vectors, view.injection);
    }
    NumericalStatus ReconstructionLimiterWBAP(const ReconstructionLimiterView &view, const ReconstructionWorkSelection &work)
    {
        if (view.hasIndicator)
        {
            const auto status = OMP::ForRowsStatus(0, work.owned, [&](index iCell) -> NumericalStatus
                                                   {
                if (!std::isfinite(view.indicator(iCell, 0)))
                    return NumericalStatus{NumericalError::InvalidInput}.At(iCell);
                return {}; });
            if (!status)
                return status;
        }
        const index n = view.modes, nv = view.components, begin = view.begin, modes = view.orderModes;
        const real threshold = view.threshold;
        const auto scratch = view.scratch;
        const auto output = view.output;
        const auto rec = view.coefficients;
        const auto offsets = view.offsets, neighbors = view.neighbors, faces = view.faces;
        const auto secondary = view.secondary, forwardView = view.forward, inverseView = view.inverse;
        const auto rotations = view.rotations;
        const auto rotationActive = view.rotationActive;
        const auto activation = view.indicator;
        const auto scratchMatrices = Algebra::MakeEigenRowMatrices(scratch, modes, nv);
        const auto outputMatrices = Algebra::MakeEigenRowMatrices(output, n, nv);
        const auto recMatrices = Algebra::MakeEigenRowMatrices(rec, n, nv);
        const real epsilon = std::pow(verySmallReal, .25);
        auto compute = [&](index iCell, index worker)
        {
            auto center = scratchMatrices[6 * worker], other = scratchMatrices[6 * worker + 1], limited = scratchMatrices[6 * worker + 2];
            auto up = scratchMatrices[6 * worker + 3], down = scratchMatrices[6 * worker + 4], valid = scratchMatrices[6 * worker + 5];
            auto out = outputMatrices[iCell];
            out = recMatrices[iCell];
            if (view.hasIndicator && activation(iCell, 0) < threshold)
                return;
            up.setConstant(view.centralWeight), down.setConstant(view.centralWeight), valid.setOnes();
            for (index iEntry = offsets(iCell, 0); iEntry < offsets(iCell + 1, 0); ++iEntry)
            {
                const index neighbor = neighbors(iEntry, 0);
                if (neighbor == UnInitIndex)
                    continue;
                center = out.middleRows(begin, modes);
                other.noalias() = ConstMap(secondary[iEntry].data(), n, n).middleRows(begin, modes) * recMatrices[neighbor];
                const auto iFace = faces(iEntry, 0);
                if (view.rotated && rotationActive(iEntry, 0))
                    for (auto first : work.vectorStarts)
                    {
                        const Eigen::Map<const Geom::tGPoint> rotation(rotations[iEntry].data());
                        other.middleCols(first, 3) = (other.middleCols(first, 3) * rotation.transpose()).eval();
                    }
                if (view.characteristicCount)
                {
                    const index start = view.characteristicFirst, count = view.characteristicCount;
                    const ConstMap left(forwardView[iFace].data(), count, count);
                    // A reusable output scratch avoids Eigen alias temporaries.
                    limited.leftCols(count).noalias() = center.middleCols(start, count) * left.transpose();
                    center.middleCols(start, count) = limited.leftCols(count);
                    limited.leftCols(count).noalias() = other.middleCols(start, count) * left.transpose();
                    other.middleCols(start, count) = limited.leftCols(count);
                }
                for (index v = 0; v < nv; ++v)
                    for (int mode = 0; mode < modes; ++mode)
                        limited(mode, v) = Math::WBAPBiway(center(mode, v), other(mode, v));
                if (view.characteristicCount)
                {
                    const index start = view.characteristicFirst, count = view.characteristicCount;
                    const ConstMap right(inverseView[iFace].data(), count, count);
                    other.leftCols(count).noalias() = limited.middleCols(start, count) * right.transpose();
                    limited.middleCols(start, count) = other.leftCols(count);
                }
                for (index v = 0; v < nv; ++v)
                    for (int mode = 0; mode < modes; ++mode)
                    {
                        const real value = limited(mode, v);
                        const real ratio = out(begin + mode, v) / (std::abs(value) + epsilon) * ((value > 0) - (value < 0));
                        up(mode, v) += ratio * ratio * ratio;
                        down(mode, v) += sqr(sqr(ratio));
                        if (value == 0)
                            valid(mode, v) = 0;
                    }
            }
            for (index v = 0; v < nv; ++v)
                for (int mode = 0; mode < modes; ++mode)
                    out(begin + mode, v) *= valid(mode, v) * up(mode, v) / down(mode, v);
        };
        const auto rows = work.rows;
        const index count = rows.RowCount();
        const index workers = work.workers;
        OMP::ForRows(0, workers, [&](index worker)
                     {
            // Explicit worker slots own six scratch matrices. This remains
            // valid when a nested caller executes the slots serially.
            const index quotient = count / workers, remainder = count % workers;
            const index begin = worker * quotient + std::min(worker, remainder);
            const index end = begin + quotient + (worker < remainder);
            for (index iRow = begin; iRow < end; ++iRow)
                compute(rows.RowAt(iRow), worker); });
        return {};
    }
    void CommitReconstruction(ReconstructionReadRows source, ReconstructionWriteRows destination)
    {
        OMP::ForRows(0, destination.RowCount(), [&](index iCell)
                     { std::copy(source[iCell].begin(), source[iCell].end(), destination[iCell].begin()); });
    }
}
