#include "CFV/CPU/ReconstructionKernels.hpp"
#include "DNDS/Algebra/EigenView.hpp"
#include "DNDS/Core/OMP.hpp"
#include "Geom/Geometry/Geometric.hpp"

namespace DNDS::CFV::CPU
{
    using Matrix = Eigen::MatrixXd;
    using ConstMap = Eigen::Map<const Matrix>;

    void ApplyReconstructionLinearOperator(const ReconstructionLinearView &view)
    {
        R::ValidateViewDevices(R::Device::Host(), view.offsets, view.neighbors, view.matrices,
                               view.diagonal, view.rotations, view.rotationActive, view.input, view.output, view.scratch);
        const index n = view.modes, components = view.components;
        const auto input = Algebra::MakeEigenRowMatrices(view.input, n, components);
        const auto output = Algebra::MakeEigenRowMatrices(view.output, n, components);
        const auto scratch = Algebra::MakeEigenRowMatrices(view.scratch, n, components);
        const index workers = view.scratch.RowCount() / 2;
        DNDS_assert(workers > 0 && view.scratch.RowCount() == 2 * workers);
        OMP::ForRows(0, workers, [&](index worker)
                     {
            auto local = scratch[2 * worker], other = scratch[2 * worker + 1];
            const index rows = view.output.RowCount(), quotient = rows / workers, remainder = rows % workers;
            const index begin = worker * quotient + std::min(worker, remainder);
            const index end = begin + quotient + (worker < remainder);
            for (index iCell = begin; iCell < end; ++iCell)
            {
                local.noalias() = ConstMap(view.diagonal[iCell].data(), n, n) * input[iCell];
                for (index iEntry = view.offsets(iCell, 0); iEntry < view.offsets(iCell + 1, 0); ++iEntry)
                {
                    const index neighbor = view.neighbors(iEntry, 0);
                    if (neighbor == UnInitIndex)
                        continue;
                    other = input[neighbor];
                    if (view.rotated && view.rotationActive(iEntry, 0))
                        for (auto first : view.vectorStarts)
                        {
                            const Eigen::Map<const Geom::tGPoint> rotation(view.rotations[iEntry].data());
                            // Vector triples are validated sorted/disjoint by
                            // the operator. Read the frozen original directly
                            // instead of allocating an alias-protection eval.
                            other.middleCols(first, 3).noalias() = input[neighbor].middleCols(first, 3) * rotation.transpose();
                        }
                    local.noalias() -= ConstMap(view.matrices[iEntry].data(), n, n) * other;
                }
                output[iCell] = local;
            } });
    }

    void BuildReconstructionRightHandSide(const ReconstructionLinearView &view)
    {
        R::ValidateViewDevices(R::Device::Host(), view.offsets, view.neighbors, view.boundaryOffsets,
                               view.vectors, view.injection, view.boundary, view.rotations, view.rotationActive,
                               view.input, view.output, view.scratch);
        const index n = view.modes, components = view.components;
        const auto output = Algebra::MakeEigenRowMatrices(view.output, n, components);
        const auto scratch = Algebra::MakeEigenRowMatrices(view.scratch, n, components);
        const index workers = view.scratch.RowCount() / 2;
        DNDS_assert(workers > 0 && view.scratch.RowCount() == 2 * workers);
        OMP::ForRows(0, workers, [&](index worker)
                     {
            auto local = scratch[2 * worker];
            const index rows = view.output.RowCount(), quotient = rows / workers, remainder = rows % workers;
            const index begin = worker * quotient + std::min(worker, remainder);
            const index end = begin + quotient + (worker < remainder);
            for (index iCell = begin; iCell < end; ++iCell)
            {
                local.setZero();
                for (index iEntry = view.offsets(iCell, 0); iEntry < view.offsets(iCell + 1, 0); ++iEntry)
                {
                    const index neighbor = view.neighbors(iEntry, 0);
                    const bool rotatedEntry = neighbor != UnInitIndex && view.rotated && view.rotationActive(iEntry, 0);
                    for (index component = 0; component < components; ++component)
                        if (neighbor != UnInitIndex)
                        {
                            real value = view.input(neighbor, component);
                            if (rotatedEntry)
                                for (auto first : view.vectorStarts)
                                    if (component >= first && component < first + 3)
                                    {
                                        const Eigen::Map<const Geom::tGPoint> rotation(view.rotations[iEntry].data());
                                        value = 0;
                                        for (int d = 0; d < 3; ++d)
                                            value += rotation(component - first, d) * view.input(neighbor, first + d);
                                    }
                            local.col(component).noalias() += Eigen::Map<const Eigen::VectorXd>(view.vectors[iEntry].data(), n) *
                                (value - view.input(iCell, component));
                        }
                        else
                            for (index iPoint = view.boundaryOffsets(iEntry, 0); iPoint < view.boundaryOffsets(iEntry + 1, 0); ++iPoint)
                                local.col(component).noalias() += Eigen::Map<const Eigen::VectorXd>(view.injection[iPoint].data(), n) *
                                    (view.boundary(iPoint, component) - view.input(iCell, component));
                }
                output[iCell] = local;
            } });
    }

    void ApplyReconstructionLocalInverse(ReconstructionTable inverse, ReconstructionReadRows input,
                                         ReconstructionWriteRows output, index modes)
    {
        R::ValidateViewDevices(R::Device::Host(), inverse, input, output);
        const auto x = Algebra::MakeEigenRowMatrices(input, modes, input.Width() / modes);
        const auto y = Algebra::MakeEigenRowMatrices(output, modes, input.Width() / modes);
        OMP::ForRows(0, output.RowCount(), [&](index iCell)
                     { y[iCell].noalias() = ConstMap(inverse[iCell].data(), modes, modes) * x[iCell]; });
    }

    void FormReconstructionResidual(ReconstructionReadRows rhs, ReconstructionWriteRows output)
    {
        R::ValidateViewDevices(R::Device::Host(), rhs, output);
        OMP::ForRows(0, output.RowCount(), [&](index iCell)
                     {
            for (index mode = 0; mode < output.Width(); ++mode)
                output(iCell, mode) = rhs(iCell, mode) - output(iCell, mode); });
    }
}
