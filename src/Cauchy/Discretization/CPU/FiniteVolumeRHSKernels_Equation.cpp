#include "Cauchy/Discretization/CPU/FiniteVolumeRHSKernels_Equation_Impl.cpp.inc"
#include "FiniteVolumeCPUFluxCatalog.hpp"

namespace DNDS::Cauchy::CPU
{
    namespace
    {
        NumericalResult<index> GenericFluxRows(const FluxRowsRequest &a) noexcept
        {
            return DispatchCPUEquationLayouts([&](auto owned, auto remote, auto output, auto low)
                                              { return EvaluateEquationFluxRows<EquationPhysicsProfile::Generic, ReconstructionForm::Generic, -1, -1, -1>(
                                                    a.request, a.equation, a.boundary, a.flux, a.worker, owned, remote, output, low,
                                                    a.region, a.scratch, a.first, a.last, a.faces); }, a.owned, a.remote, a.destination, a.low);
        }

        FluxRowsKernel SelectFlux(const EquationRHSRequest &r, const EquationSystem &equation,
                                  EquationLayoutView<const real> owned, EquationLayoutView<const real> remote,
                                  EquationLayoutView<real> dest, EquationLayoutView<real> low)
        {
            const auto form = r.firstOrder || (r.direct && !r.linear) ? ReconstructionForm::MeanOnly : r.direct ? ReconstructionForm::DirectGradient
                                                                                                                : ReconstructionForm::Modal;
            if (!r.genericFluxOracle && owned.CompactHostTable(r.width) && remote.CompactHostTable(r.width) &&
                dest.CompactHostTable(r.width) && low.CompactHostTable(r.width) && CompactFluxOperands(r, form))
                if (const auto run = FindCPUFluxShape(ResolveEquationPhysicsProfile(equation), form, r.width, r.dimension,
                                                      form == ReconstructionForm::Modal ? r.modes : 0))
                    return run;
            return GenericFluxRows;
        }
        index EquationRowCount(const EquationRHSRequest &request, CFV::ReconstructionRegion region, bool face)
        {
            if (region == CFV::ReconstructionRegion::All)
                return face ? request.stencil.stencilFaces.RowCount() : request.ownedCells;
            if (face)
                return (region == CFV::ReconstructionRegion::Core ? request.stencil.rhsCoreFaces : request.stencil.rhsDependentFaces).RowCount();
            return (region == CFV::ReconstructionRegion::Core ? request.stencil.coreCells : request.stencil.dependentCells).RowCount();
        }
    }
    NumericalResult<index> EvaluateEquationFlux(const EquationRHSRequest &request, const EquationSystem &equation,
                                                const BoundaryPointOperator &boundary, const Riemann::RiemannFlux &flux,
                                                PointEvaluator &worker, EquationLayoutView<const real> owned,
                                                EquationLayoutView<const real> remote, EquationLayoutView<real> destination,
                                                EquationLayoutView<real> lowDestination, CFV::ReconstructionRegion region) noexcept
    {
        EquationRHSPointScratch scratch(request.width, request.species, request.produceLow);
        const auto run = SelectFlux(request, equation, owned, remote, destination, lowDestination);
        const auto phase = [&](bool faces)
        {
            return run({request, equation, boundary, flux, worker, owned, remote, destination, lowDestination,
                        region, scratch, 0, EquationRowCount(request, region, faces), faces});
        };
        const auto face = phase(true);
        if (!face)
            return face.status;
        const auto cells = phase(false);
        return cells ? face : cells;
    }
    NumericalResult<index> EvaluateEquationFlux(const EquationRHSRequest &request, const EquationSystem &equation,
                                                const BoundaryPointOperator &boundary, const Riemann::RiemannFlux &flux,
                                                SpanView<EquationRHSWorkerScratch> workers, EquationLayoutView<const real> owned,
                                                EquationLayoutView<const real> remote, EquationLayoutView<real> destination,
                                                EquationLayoutView<real> lowDestination, CFV::ReconstructionRegion region) noexcept
    {
        if (workers.empty())
            return NumericalStatus{NumericalError::InvalidInput};
        const auto run = SelectFlux(request, equation, owned, remote, destination, lowDestination);
        index fallbacks = 0;
        // A face value may serve cells in several worker ranges. Complete
        // the face pass before any gather reads it; no scatter atomics.
        for (const bool faces : {true, false})
        {
            const index count = EquationRowCount(request, region, faces);
            OMP::ForRows(0, workers.size(), [&](index iWorker)
                         {
                    const index quotient = count / workers.size(), remainder = count % workers.size();
                    const index first = iWorker * quotient + std::min(iWorker, remainder);
                    auto &worker = workers[iWorker];
                    worker.result = run({request, equation, boundary, flux, worker.evaluator, owned, remote,
                                         destination, lowDestination, region, worker.scratch, first,
                                         first + quotient + (iWorker < remainder), faces}); });
            // Fixed worker order preserves deterministic error selection and
            // diagnostic integer accumulation. FP sums remain cell-local.
            for (const auto &worker : workers)
            {
                if (!worker.result)
                    return worker.result.status;
                fallbacks += worker.result.value;
            }
        }
        return fallbacks;
    }

}
