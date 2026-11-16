#include "CFV/Detail/ReconstructionData.hpp"
#include "CFV/CPU/ReconstructionKernels.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;

    void VariationalReconstructionPlan::ApplyLinearOperator(const FVField &coefficients, FVField &output,
                                                            const std::vector<index> &periodicVectorStarts) const
    {
        Require(ReconstructionOperation::LinearSystem);
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        const index n = NRecDOF();
        DNDS_check_throw_info(coefficients.Width() > 0 && coefficients.Width() % n == 0, "linear reconstruction width is not a mode multiple");
        const index components = coefficients.Width() / n;
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        Metrics().ValidateField(coefficients, n * components), Metrics().ValidateField(output, n * components);
        auto owned = coefficients.Owned().Read(), ghost = coefficients.Ghost().Read();
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> x{owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
        auto write = output.Owned().Write();
        const auto y = write.View<R::UniformTableRows>();
        auto offsetRead = _data->offsets.Read(), neighborRead = _data->neighbors.Read();
        auto rotationRead = _data->neighborRotation.Read();
        auto rotationActiveRead = _data->neighborRotationActive.Read();
        const auto rotations = rotationRead.View<R::FixedRows<9>>();
        const auto rotationActive = rotationActiveRead.View<R::FixedRows<1>>();
        auto matrixRead = _data->rawMatrixB->Read(), diagonalRead = _data->localMatrix->Read();
        const auto offsets = offsetRead.View<R::FixedRows<1>>(), neighbors = neighborRead.View<R::FixedRows<1>>();
        const auto matrices = matrixRead.View<R::TableRows>(), diagonals = diagonalRead.View<R::TableRows>();
        const index workers = std::min<index>(OMP::DistributedThreads(), std::max<index>(1, Metrics().NumCell()));
        R::Array<real> scratch(R::RowLayout::Dense(2 * workers, n * components));
        scratch.SetName("CFV/reconstruction/linearActionScratch", DNDS_ALLOCATION_SITE());
        auto scratchWrite = scratch.Write();
        CPU::ReconstructionLinearView request;
        request.offsets = offsets, request.neighbors = neighbors;
        request.matrices = matrices, request.diagonal = diagonals;
        request.rotations = rotations, request.rotationActive = rotationActive;
        request.input = x, request.output = y;
        request.scratch = scratchWrite.View<R::TableRows>();
        request.vectorStarts = {periodicVectorStarts.data(), index(periodicVectorStarts.size())};
        request.modes = n, request.components = components;
        request.rotated = _data->hasRotatedPeriodicity;
        CPU::ApplyReconstructionLinearOperator(request);
    }
    void VariationalReconstructionPlan::BuildRightHandSide(const FVField &mean, const ReconstructionBoundaryValues &boundary,
                                                           FVField &output, const std::vector<index> &periodicVectorStarts) const
    {
        Require(ReconstructionOperation::LinearSystem);
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        const index n = NRecDOF(), components = mean.Width();
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        Metrics().ValidateField(mean, components), Metrics().ValidateField(output, n * components);
        DNDS_check_throw_info(boundary.plan == _identity && boundary.values.GetDevice() == R::Device::Host() &&
                                  boundary.values.Layout().Kind() == R::RowLayoutKind::Table && boundary.values.RowCount() == NumBoundaryQuadrature() &&
                                  boundary.values.Layout().Width() == components,
                              "linear boundary shape/plan mismatch");
        auto owned = mean.Owned().Read(), ghost = mean.Ghost().Read(), boundaryRead = boundary.values.Read();
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> u{owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
        const auto bc = boundaryRead.View<R::TableRows>();
        auto write = output.Owned().Write();
        const auto y = write.View<R::UniformTableRows>();
        auto offsetRead = _data->offsets.Read(), neighborRead = _data->neighbors.Read(), boundaryOffsetRead = _data->boundaryOffsets.Read();
        auto rotationRead = _data->neighborRotation.Read();
        auto rotationActiveRead = _data->neighborRotationActive.Read();
        const auto rotations = rotationRead.View<R::FixedRows<9>>();
        const auto rotationActive = rotationActiveRead.View<R::FixedRows<1>>();
        auto vectorRead = _data->rawVectorB->Read(), injectionRead = _data->rawBoundaryInjection->Read();
        const auto offsets = offsetRead.View<R::FixedRows<1>>(), neighbors = neighborRead.View<R::FixedRows<1>>(), boundaryOffsets = boundaryOffsetRead.View<R::FixedRows<1>>();
        const auto vectors = vectorRead.View<R::TableRows>(), injections = injectionRead.View<R::TableRows>();
        const index workers = std::min<index>(OMP::DistributedThreads(), std::max<index>(1, Metrics().NumCell()));
        R::Array<real> scratch(R::RowLayout::Dense(2 * workers, n * components));
        scratch.SetName("CFV/reconstruction/linearRHSScratch", DNDS_ALLOCATION_SITE());
        auto scratchWrite = scratch.Write();
        CPU::ReconstructionLinearView request;
        request.offsets = offsets, request.neighbors = neighbors, request.boundaryOffsets = boundaryOffsets;
        request.vectors = vectors, request.injection = injections, request.boundary = bc;
        request.rotations = rotations, request.rotationActive = rotationActive;
        request.input = u, request.output = y;
        request.scratch = scratchWrite.View<R::TableRows>();
        request.vectorStarts = {periodicVectorStarts.data(), index(periodicVectorStarts.size())};
        request.modes = n, request.components = components;
        request.rotated = _data->hasRotatedPeriodicity;
        CPU::BuildReconstructionRightHandSide(request);
    }
    void VariationalReconstructionPlan::ComputeResidual(const FVField &rhs, const FVField &coefficients, FVField &output,
                                                        const std::vector<index> &periodicVectorStarts) const
    {
        Require(ReconstructionOperation::LinearSystem);
        Metrics().ValidateField(rhs, coefficients.Width());
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(rhs.Owned().Storage(), output.Owned().Storage()), "residual output aliases RHS");
        ApplyLinearOperator(coefficients, output, periodicVectorStarts);
        auto read = rhs.Owned().Read();
        auto write = output.Owned().Write();
        const auto b = read.View<R::UniformTableRows>();
        const auto y = write.View<R::UniformTableRows>();
        CPU::FormReconstructionResidual(b, y);
    }
    void VariationalReconstructionPlan::ApplyLocalPreconditioner(const FVField &input, FVField &output) const
    {
        Require(ReconstructionOperation::LocalPreconditioner);
        auto geometry = Metrics().AcquireRead();
        Validate();
        const index n = NRecDOF();
        DNDS_check_throw_info(input.Width() > 0 && input.Width() % n == 0, "preconditioner width is not a mode multiple");
        Metrics().ValidateField(input, input.Width()), Metrics().ValidateField(output, input.Width());
        auto read = input.Owned().Read(), matrixRead = _data->localInverse->Read();
        auto write = output.Owned().Write();
        const auto x = read.View<R::UniformTableRows>();
        const auto matrix = matrixRead.View<R::TableRows>();
        const auto y = write.View<R::UniformTableRows>();
        CPU::ApplyReconstructionLocalInverse(matrix, x, y, n);
    }
}
