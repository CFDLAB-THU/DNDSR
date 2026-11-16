#include "Cauchy/Discretization/FiniteVolumeAuxiliaryInternal.hpp"
#include "Cauchy/Physics/PointWorkerContexts.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::SubmitMeanGradientBoundary(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state,
        MeanGradients &gradients, real time, std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
        OptionalRef<const R::Array<real>> incoming) const
    {
        DNDS_check_throw_info(gradients._data && gradients._data->owner.lock() == _data &&
                                  engine.Owns(stream) && stream.Context().device == Device(),
                              "mean-gradient boundary workspace/stream differs");
        auto &work = *gradients._data;
        work.binding.Check(state);
        DNDS_check_throw_info(!_data->hasLocalInterfaceRows || incoming,
                              "mean gradient requires its face-center interface section");
        if (incoming)
            DNDS_check_throw_info(incoming->RowCount() == _data->boundaryFaces.RowCount() &&
                                      incoming->Layout().Width() == state.ComponentNumber() &&
                                      incoming->GetDevice() == Device() && !incoming->Layout().IsVariable(),
                                  "mean-gradient interface center section shape or placement differs");
        if (!_data->boundaryFaces.RowCount())
            return;
        if (Device().space == R::MemorySpace::CUDA)
        {
#ifdef DNDS_USE_CUDA
            DNDS_check_throw_info(work.boundary.size() == 1 && work.deviceBoundaryStatus, "device mean-gradient boundary requires combined fields/status");
            auto &errors = *work.deviceBoundaryStatus;
            SubmitBoundaryValues(engine, stream, state, {}, work.boundary.front(), errors.rows,
                                 {BoundaryTarget::Gradient, false, false, time}, incoming);
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto rows, auto empty, auto partial, auto output)
                            { CUDA::ReduceNumericalStatus(context, rows, empty, partial, output); }, R::ReadAs<R::FixedRows<1>>(errors.rows), R::ReadAs<R::FixedRows<1>>(errors.empty), R::WriteAs<R::FixedRows<1>>(errors.partial), R::WriteAs<R::FixedRows<1>>(errors.result));
#else
            DNDS_check_throw_info(false, "CUDA mean-gradient boundary is not built");
#endif
            return;
        }
        DNDS_check_throw_info(Device() == R::Device::Host(), "CPU mean-gradient boundary placement differs");
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            FiniteVolumeStencilStorage::ReadAccess stencil;
            R::Array<BoundaryFaceRange>::ReadLease faces;
            bool walls = false;
            std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
            std::vector<EquationBlockLocation> locations;
            std::vector<index> logicalFirst;
            index components = 0, ransFirst = -1, incoming = -1;
            Inputs(std::shared_ptr<const Data> data, std::optional<Physics::Chemistry::ChemicalWorkerPool> pool)
                : owner(std::move(data)),
                  stencil(owner->stencil.Read()), faces(owner->boundaryFaces.Read()),
                  workers(std::move(pool))
            {
                if (owner->equation.HasRANS() && owner->equation.RANS().ComponentNumber() == 2)
                    walls = true;
            }
        };
        auto pin = std::make_shared<Inputs>(_data, std::move(workers));
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
        {
            pin->logicalFirst.push_back(pin->components);
            pin->locations.push_back(state.LocateBlock(iBlock));
            if (state.Schema().blocks[iBlock].name == "rans")
                pin->ransFirst = pin->components;
            pin->components += pin->locations.back().componentNumber;
        }
        std::vector<R::ArrayOperand<real, false, R::UniformTableRows>> inputs;
        std::vector<R::ArrayOperand<real, true, R::UniformTableRows>> outputs;
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            inputs.push_back(R::ReadAs<R::UniformTableRows>(state.PhysicalField(iField).Owned()));
            outputs.push_back(R::WriteAs<R::UniformTableRows>(work.boundary[iField]));
        }
        if (incoming)
        {
            pin->incoming = index(inputs.size());
            inputs.push_back(R::ReadAs<R::UniformTableRows>(*incoming));
        }
        for (const auto &output : outputs)
            for (const auto &input : inputs)
                DNDS_check_throw_info(!output.storage->Values().Overlaps(input.storage->Values()), "mean-gradient boundary output aliases input");
        R::SubmitArrayBatch(engine, stream, [pin, time](R::ExecutionContext, auto inputs, auto outputs, auto wallMean)
                            {
            // Acquire worker state at execution, never while earlier commands
            // may need the same pool. It is released before the next launcher.
            auto workers = MakeCPUPointWorkers<CPU::BoundaryTraceWorkerScratch>(pin->owner->equation,
                pin->workers, {true}, pin->components);
            std::vector<EquationBlockReadView> means;
            std::vector<EquationBlockArrayView<real>> targets;
            for (const auto &location : pin->locations)
            {
                means.push_back({{inputs[location.physicalField], location.firstComponent, location.componentNumber}, {{}, location.firstComponent, location.componentNumber}});
                targets.push_back({outputs[location.physicalField], location.firstComponent, location.componentNumber});
            }
            CPU::BoundaryTraceRequest request;
            request.stencil = pin->stencil.View();
            request.faces = pin->faces.View<R::FixedRows<1>>();
            request.centers = true;
            request.repairRANS = false;
            request.means = {means.data(), index(means.size())};
            request.output = {targets.data(), index(targets.size())};
            request.logicalFirst = {pin->logicalFirst.data(), index(pin->logicalFirst.size())};
            request.componentNumber = pin->components;
            request.ransFirst = pin->ransFirst;
            if (pin->owner->equation.HasRANS())
                request.rans = {pin->owner->equation.RANS().Kind(), pin->owner->equation.RANS().ComponentNumber()};
            if (pin->walls)
                request.wallDistance.cellMean = {wallMean.data, wallMean.offsets, wallMean.rows, wallMean.width,
                                                wallMean.stride, wallMean.sizes, wallMean.kind, wallMean.device};
            if (pin->incoming >= 0) request.incoming = inputs[pin->incoming];
            request.frame = pin->owner->motion.At(time);
            request.time = time;
            RequireNumericalSuccess(CPU::FillBoundaryTraces(request,
                {pin->owner->boundaryConditions.data(), index(pin->owner->boundaryConditions.size())},
                pin->owner->boundaryPoint, SpanView<CPU::BoundaryTraceWorkerScratch>{workers.data(), index(workers.size())}),
                "queued mean-gradient boundary"); }, std::move(inputs), std::move(outputs), std::make_tuple(R::ReadAs<R::RowLayout>(pin->walls ? _data->wallDistance->CellMeans() : _data->deviceEmpty)));
    }
}
