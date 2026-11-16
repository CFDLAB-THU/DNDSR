#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeLimitedVRKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeLimitedVRKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;

    namespace
    {
        /// CUDA specialization admits only combined AoS. Uniform CPU views
        /// retain general field layouts; conversion here is wrapper-local.
        template <class T>
        auto AoS(R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows> view)
        {
            return R::ArrayView<T, R::AccessBoundsChecks, R::TableRows>{view.data, view.offsets, view.rows, view.width, view.stride, view.sizes, view.kind, view.device};
        }
    }

    void FiniteVolume::LimitedVRWorkspace::Build(const Data &data, const CFV::FVField &mean, const CFV::FVField &selectedGradients,
                                                 R::Array<real> &previous,
                                                 CFV::FaceCharacteristicTransformRef transform,
                                                 R::EagerEngine &engine,
                                                 const R::ExecutionStream &stream,
                                                 const R::ExecutionStream &communication)
    {
        const auto &geometry = *data.limitedVRGeometry;
        const auto &settings = data.settings.numerics.limitedVR;
        const auto &controls = data.settings.numerics;
        const auto &gas = data.equation.Gas();
        const auto bounds = data.FixedFlowThermalBounds();
        const auto placement = data.metrics.Device();
        const bool cuda = placement != R::Device::Host();
        const bool overlap = data.settings.execution.overlap;
        const int dimension = data.metrics.Dimension();
        // The immutable shared publication pins topology, metric fields and
        // compact periodic maps through all CPU worker/device submissions.
        const auto stencil = std::make_shared<FiniteVolumeStencilStorage::ReadAccess>(data.stencil.Read({placement, nullptr}));
        // Admission belongs before queued kernels and matched transfers. A
        // retained external history reader is a local access error, not a
        // numerical worker failure that should poison an otherwise reusable
        // checked-test stream or let peers enter another transfer generation.
        // Production LocalStage still aborts on failure; only coordinated UTs
        // may catch this rejection. The domain session and caller's exclusive
        // mutation contract exclude new external accesses after admission,
        // just as for the other FV output buffers.
        data.metrics.Communication()->LocalStage([&]
                                                 {
            DNDS_check_throw_info(engine.Owns(stream) && engine.Owns(communication) &&
                                      stream.Context().device == placement && communication.Context().device == placement &&
                                      mean.Device() == placement && previous.GetDevice() == placement,
                                  "LVR stream or field placement differs");
            if (cuda)
                DNDS_check_throw_info(mean.Owned().Layout().Kind() == R::RowLayoutKind::Table &&
                                          mean.Ghost().Layout().Kind() == R::RowLayoutKind::Table,
                                      "CUDA LVR requires combined AoS");
            auto probe = previous.Write({placement, nullptr});
            probe.CancelUnmodified(); });
        OperatorSchedule schedule(engine, stream, communication, overlap);
        if (cuda)
        {
#ifdef DNDS_USE_CUDA
            if (!device)
            {
                const index cells = data.metrics.NumCell(), ghosts = data.metrics.NumCellProc() - cells, faces = data.stencil.stencilFaces.RowCount();
                device.emplace(DeviceStorage{
                    R::Array<real>(R::RowLayout::Dense(faces, 3), placement),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(cells, 1), placement),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(ghosts, 1), placement),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(faces, 1), placement),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(std::max({cells, ghosts, faces}) + 1), 1), placement),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), placement),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(std::min<index>((cells + 127) / 128, 1024), 1), placement)});
                device->sensors.SetName("Cauchy/LVR/faceSensors", DNDS_ALLOCATION_SITE());
                device->targetStatus.SetName("Cauchy/LVR/targetBlockStatus", DNDS_ALLOCATION_SITE());
                device->ownedStatus.SetName("Cauchy/LVR/ownedStatus", DNDS_ALLOCATION_SITE());
                device->ghostStatus.SetName("Cauchy/LVR/ghostStatus", DNDS_ALLOCATION_SITE());
                device->faceStatus.SetName("Cauchy/LVR/faceStatus", DNDS_ALLOCATION_SITE());
                device->partialStatus.SetName("Cauchy/LVR/statusPartials", DNDS_ALLOCATION_SITE());
                device->resultStatus.SetName("Cauchy/LVR/status", DNDS_ALLOCATION_SITE());
            }
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto status)
                            { CUDA::ClearNumericalStatus(context, status); }, R::WriteAs<R::FixedRows<1>>(device->resultStatus));
#else
            DNDS_check_throw_info(false, "CUDA LVR is not built");
#endif
        }
        const auto accumulateStatus = [&](const R::Array<NumericalStatus> &rows)
        {
#ifdef DNDS_USE_CUDA
            if (cuda)
                R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto rows, auto partial, auto result)
                                { CUDA::ReduceNumericalStatus(context, rows, result.ConstView(), partial, result); }, R::ReadAs<R::FixedRows<1>>(rows), R::WriteAs<R::FixedRows<1>>(device->partialStatus), R::ReadWriteAs<R::FixedRows<1>>(device->resultStatus));
#endif
        };
        data.gradient.ConvertToCoefficients(engine, stream, selectedGradients, *data.reconstruction, targetCoefficients);
        {
            auto transfer = schedule.BeginAfterCompute(targetHalo);
            if (overlap)
                data.reconstruction->SubmitLimiterWBAP_3OrderTo(engine, stream, targetCoefficients, target, 1, transform,
                                                                CFV::ReconstructionRegion::Core);
            schedule.Await(transfer);
            data.reconstruction->SubmitLimiterWBAP_3OrderTo(
                engine, stream, targetCoefficients, target, 1, transform,
                overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);
            data.reconstruction->SubmitCommitReconstruction(engine, stream, targetCoefficients, target);
        }
        // PP POLICY -- the accepted O2 target is guarded at face centers,
        // independently of the high-order final modal PP point set. Sampling,
        // scaling and final stored-state acceptance are fused in each cell.
        auto &ownedStatus = cuda ? device->ownedStatus : emptyStatus;
        R::SubmitArrays(engine, stream, [stencil, dimension, bounds, power = controls.positivityBetaPower, cutoff = controls.positivityBetaCutoff](R::ExecutionContext context, auto means, auto coefficients, auto status)
                        {
                            if (context.device == R::Device::Host())
                                RequireNumericalSuccess(CPU::LimitLVRTarget(stencil->View(), means, coefficients, dimension, bounds, power, cutoff), "LimitLVRTarget");
#ifdef DNDS_USE_CUDA
                            else
                                CUDA::LimitLVRTarget(context, stencil->KernelDescriptor(), AoS(means), AoS(coefficients), dimension, bounds, power, cutoff, status);
#endif
                        },
                        R::ReadAs<R::UniformTableRows>(mean.Owned()), R::ReadWriteAs<R::UniformTableRows>(targetCoefficients.Owned()), R::WriteAs<R::FixedRows<1>>(cuda ? device->targetStatus : emptyStatus));
        accumulateStatus(cuda ? device->targetStatus : emptyStatus);
        {
            auto transfer = schedule.BeginAfterCompute(targetHalo);
            // The mean halo is already observed. Derived mean scalars need no
            // extra communication, including their ghost rows.
            const auto fill = [&](const R::Array<real> &input, R::Array<real> &output, R::Array<NumericalStatus> &status)
            {
                R::SubmitArrays(engine, stream, [gas](R::ExecutionContext context, auto input, auto output, auto status)
                                {
                                    if (context.device == R::Device::Host())
                                        CPU::FillLVRScalars(input, output, gas);
#ifdef DNDS_USE_CUDA
                                    else
                                        CUDA::FillLVRScalars(context, AoS(input), AoS(output), gas, status);
#endif
                                },
                                R::ReadAs<R::UniformTableRows>(input), R::WriteAs<R::UniformTableRows>(output), R::WriteAs<R::FixedRows<1>>(status));
                accumulateStatus(status);
            };
            fill(mean.Owned(), scalars.Owned(), ownedStatus);
            fill(mean.Ghost(), scalars.Ghost(), cuda ? device->ghostStatus : emptyStatus);
            schedule.Await(transfer);
        }
        if (!cuda)
        {
            if (!cpuSensors)
            {
                cpuSensors.emplace(R::RowLayout::Dense(data.stencil.stencilFaces.RowCount(), 3));
                cpuSensors->SetName("Cauchy/LVR/faceSensors", DNDS_ALLOCATION_SITE());
            }
            R::SubmitArrays(engine, stream, [stencil, dimension, gas, bounds, settings](R::ExecutionContext, auto own, auto ghost, auto targetOwn, auto targetGhost, auto sensors)
                            { RequireNumericalSuccess(CPU::BuildLVRFaceSensors(stencil->View(), {own, ghost}, {targetOwn, targetGhost}, dimension, gas, bounds, settings, sensors), "BuildLVRFaceSensors"); }, R::ReadAs<R::UniformTableRows>(mean.Owned()), R::ReadAs<R::UniformTableRows>(mean.Ghost()), R::ReadAs<R::UniformTableRows>(targetCoefficients.Owned()), R::ReadAs<R::UniformTableRows>(targetCoefficients.Ghost()), R::WriteAs<R::UniformTableRows>(*cpuSensors));
            R::SubmitArrays(engine, stream, [stencil](R::ExecutionContext, auto sensors, auto branches, auto base)
                            { CPU::GatherLVRBranches(stencil->View(), sensors, branches, base); }, R::ReadAs<R::UniformTableRows>(*cpuSensors), R::WriteAs<R::UniformTableRows>(branches.Owned()), R::WriteAs<R::FixedRows<2>>(base));
        }
#ifdef DNDS_USE_CUDA
        else
        {
            R::SubmitArrays(engine, stream, [stencil, dimension, gas, bounds, settings = CFV::LimitedVRKernelParameters::FromSettings(settings)](R::ExecutionContext context, auto own, auto ghost, auto targetOwn, auto targetGhost, auto sensors, auto status)
                            { CUDA::BuildLVRFaceSensors(context, stencil->KernelDescriptor(), {own, ghost}, {targetOwn, targetGhost}, dimension, gas, bounds, settings, sensors, status); }, R::ReadAs<R::TableRows>(mean.Owned()), R::ReadAs<R::TableRows>(mean.Ghost()), R::ReadAs<R::TableRows>(targetCoefficients.Owned()), R::ReadAs<R::TableRows>(targetCoefficients.Ghost()), R::WriteAs<R::TableRows>(device->sensors), R::WriteAs<R::FixedRows<1>>(device->faceStatus));
            accumulateStatus(device->faceStatus);
            R::SubmitArrays(engine, stream, [stencil](R::ExecutionContext context, auto sensors, auto faceStatus, auto branches, auto base, auto status)
                            { CUDA::GatherLVRBranches(context, stencil->KernelDescriptor(), sensors, faceStatus, branches, base, status); }, R::ReadAs<R::TableRows>(device->sensors), R::ReadAs<R::FixedRows<1>>(device->faceStatus), R::WriteAs<R::TableRows>(branches.Owned()), R::WriteAs<R::FixedRows<2>>(base), R::WriteAs<R::FixedRows<1>>(ownedStatus));
            accumulateStatus(ownedStatus);
        }
#endif
        {
            auto transfer = schedule.BeginAfterCompute(branchHalo);
            if (overlap)
                geometry.gates.BuildExtremumSeeds(engine, stream, scalars, branches, previous, seeds, settings, CFV::ReconstructionRegion::Core);
            schedule.Await(transfer);
            geometry.gates.BuildExtremumSeeds(engine, stream, scalars, branches, previous, seeds, settings, schedule.DependentRegion());
        }
        const auto maximum = [&](CFV::FVField &input, R::TransferBinding &halo, std::array<bool, 3> active, bool excludeBoundary, bool saturated)
        {
            auto transfer = schedule.BeginAfterCompute(halo);
            if (overlap)
                geometry.gates.NeighborMaximum(engine, stream, input, temporary, active, excludeBoundary, saturated,
                                               {settings.alphaMax, settings.contactAlphaMax, 1}, CFV::ReconstructionRegion::Core);
            schedule.Await(transfer);
            geometry.gates.NeighborMaximum(engine, stream, input, temporary, active, excludeBoundary, saturated,
                                           {settings.alphaMax, settings.contactAlphaMax, 1}, schedule.DependentRegion());
            Algebra::Copy(engine, stream, input, temporary);
        };
        const auto merge = [&](int column)
        {
            R::SubmitArrays(engine, stream, [column](R::ExecutionContext context, auto base, auto seeds)
                            {
                                if (context.device == R::Device::Host())
                                    CPU::MergeLVRBase(base, seeds, column);
#ifdef DNDS_USE_CUDA
                                else
                                    CUDA::MergeLVRBase(context, base, AoS(seeds), column);
#endif
                            },
                            R::ReadAs<R::FixedRows<2>>(base), R::ReadWriteAs<R::UniformTableRows>(seeds.Owned()));
        };
        const int pressureLayers = settings.alphaMax > 0 ? 1 : 0;
        const int contactLayers = settings.variant == "v3" && settings.contactAlphaMax > 0 ? settings.contactLayers : 0;
        if (!pressureLayers)
            merge(0);
        if (!contactLayers)
            merge(1);
        for (int layer = 0; layer < std::max(pressureLayers, contactLayers); ++layer)
        {
            maximum(seeds, seedHalo, {layer < pressureLayers, layer < contactLayers, false}, true, false);
            if (layer + 1 == pressureLayers)
                merge(0);
            if (layer + 1 == contactLayers)
                merge(1);
        }
        Algebra::Copy(engine, stream, branches, seeds);
        for (int layer = 0; layer < settings.neighborLayers; ++layer)
            maximum(branches, branchHalo, {true, false, false}, false, settings.saturatedOnly);
        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto gates, auto history, auto penalty)
                        {
                            if (context.device == R::Device::Host())
                                CPU::PublishLVRPenalty(gates, history, penalty);
#ifdef DNDS_USE_CUDA
                            else
                                CUDA::PublishLVRPenalty(context, AoS(gates), history, penalty);
#endif
                        },
                        R::ReadAs<R::UniformTableRows>(branches.Owned()), R::WriteAs<R::FixedRows<2>>(previous), R::WriteAs<R::FixedRows<1>>(alpha));
        // Synchronous compound-operator return. Fixed neighborhood layers use
        // producer-prefix exchanges; no engine drain separates their kernels.
        schedule.Join();
        if (cuda)
        {
            const auto status = device->resultStatus.To(R::Device::Host());
            RequireNumericalSuccess(status.Read().View()(0, 0), "limited VR preparation");
        }
    }
}
