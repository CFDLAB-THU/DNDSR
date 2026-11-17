#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Implicit/JacobianGeometry.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/Implicit/CPU/FiniteVolumeJacobianKernels.hpp"
#include "Cauchy/Implicit/CPU/ACMJacobianKernels.hpp"
#include "Cauchy/Physics/PointWorkerContexts.hpp"
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    /// Immutable physical snapshot. Evaluated blocks own no MPI, worker, live
    /// input or FV owner. Geometry is retained only for evaluated storage;
    /// assembled filling derives centered points from canonical FV incidences.
    /// TODO(large-species Jacobian): an explicit approximation may omit
    /// flow5<->scalar derivatives and diagonalize extended diagonal blocks.
    /// Its off-diagonal storage can then be flow5 dense plus scalar entries;
    /// no EOS/source coupling may be discarded merely to fit that format.
    /// Keep coupled diagonals as an option, but compressed ILU factors then
    /// generally need projection (not algebraically equivalent). See the
    /// large-species TODO in docs/architecture/CauchyImplicit.md.
    struct FiniteVolume::JacobianData final : OffDiagonalEvaluator, CPU::EquationJacobianView
    {
        struct Geometry
        {
            std::weak_ptr<const Data> owner;
            std::shared_ptr<const Algebra::DofSpace::Partition> partition;
            std::shared_ptr<const JacobianGeometryStorage> storage;
            std::optional<JacobianGeometryStorage::ReadAccess> read;
        };
        std::shared_ptr<const Geometry> geometryOwner;
        R::MetadataBuffer<uint8_t> selection;
        R::Array<real> meanStorage, derivativeStorage, pressureStorage, radiusStorage;
        int independent;
        /// Read scopes pin the frozen publication, not live equation inputs.
        std::optional<R::Array<real>::ReadLease> meanRead, derivativeRead, pressureRead, radiusRead;
        struct WriteAccess
        {
            R::Array<real>::WriteLease means, derivative, pressure, radius;
            explicit WriteAccess(JacobianData &data)
                : means(data.meanStorage.Write()), derivative(data.derivativeStorage.Write()),
                  pressure(data.pressureStorage.Write()), radius(data.radiusStorage.Write())
            {
                data.means = means.View<R::TableRows>().ConstView();
                data.pressureDerivative = derivative.View<R::TableRows>().ConstView();
                data.pressure = pressure.View<R::TableRows>().ConstView();
                data.radius = radius.View<R::TableRows>().ConstView();
            }
        };
        JacobianData(const FiniteVolume &fv, const EquationDofs &state,
                     LinearProblemContext &output, JacobianPreparation controls, real time)
            : meanStorage(R::RowLayout::Dense(output.HasAssembledOffDiagonal() ? 0 : fv.Metrics().NumCellProc(), state.ComponentNumber())),
              derivativeStorage(R::RowLayout::Dense(fv.Metrics().NumCellProc() + fv._data->stencil.IncidenceNumber(), state.ComponentNumber())),
              pressureStorage(R::RowLayout::Dense(fv.Metrics().NumCellProc() + fv._data->stencil.IncidenceNumber(), 1)),
              radiusStorage(R::RowLayout::Dense(fv._data->stencil.stencilFaces.RowCount(), 2)),
              independent(fv._data->equation.HasMixture() ? fv._data->equation.Mixture().SpeciesNumber() - 1 : 0)
        {
            components = state.ComponentNumber();
            frame = fv._data->motion.At(time);
            scale = controls.residualScale;
            const Algebra::ComponentSelection selected(components, controls.updateComponents);
            std::vector<uint8_t> flags(components);
            for (index j = 0; j < components; ++j)
            {
                flags[j] = selected.Includes(j);
                allComponents = allComponents && flags[j];
            }
            selection = R::MetadataBuffer<uint8_t>(std::move(flags));
            active = selection.HostView();
            index first = 0;
            for (const auto &block : state.Schema().blocks)
            {
                if (block.name == "flow")
                    flowFirst = first;
                if (block.name == "species")
                    speciesFirst = first;
                if (block.name == "rans" && fv._data->equation.RANS().ComponentNumber() == 2)
                    kineticFirst = first;
                first += index(block.components.size());
            }
            sixComponentProduct = components == 6 && flowFirst == 0 && kineticFirst < 0 && allComponents;
            if (output.HasAssembledOffDiagonal())
                return;
            const auto old = std::dynamic_pointer_cast<const JacobianData>(output.PreparedOffDiagonalEvaluator());
            if (old && old->geometryOwner->owner.lock() == fv._data)
                geometryOwner = old->geometryOwner;
            else
            {
                auto prepared = std::make_shared<Geometry>();
                prepared->owner = fv._data;
                prepared->partition = output.Space().Pin();
                const auto stencil = fv._data->stencil.Read();
                prepared->storage = std::make_shared<JacobianGeometryStorage>(stencil, output.Columns());
                prepared->read.emplace(prepared->storage->Read());
                geometryOwner = std::move(prepared);
            }
            geometry = geometryOwner->read->View();
            sameComponentFrames = geometryOwner->storage->sameComponentFrames;
        }
        void Publish()
        {
            meanRead.emplace(meanStorage.Read());
            derivativeRead.emplace(derivativeStorage.Read());
            pressureRead.emplace(pressureStorage.Read());
            radiusRead.emplace(radiusStorage.Read());
            means = meanRead->View<R::TableRows>();
            pressureDerivative = derivativeRead->View<R::TableRows>();
            pressure = pressureRead->View<R::TableRows>();
            radius = radiusRead->View<R::TableRows>();
        }
        index Components() const override { return components; }
        auto RowSpaceIdentity() const -> std::shared_ptr<const Algebra::DofSpace::Partition> override { return geometryOwner->partition; }
        std::vector<std::vector<index>> Columns() const override { return geometryOwner->storage->CopyColumns(); }
        index RowNumber() const override { return geometry.columns.RowCount(); }
        index BlockNumber(index iRow) const override { return geometry.columns.RowSize(iRow); }
        SpanView<const index> ColumnRow(index iRow) const override
        {
            const auto row = geometry.columns[iRow];
            return {row.data(), row.size()};
        }
        void EvaluateBlock(index iRow, index iBlock, Eigen::Ref<Eigen::MatrixXd> output) const override
        {
            CPU::EvaluateEquationJacobianBlock(*this, iRow, iBlock, output);
        }
        void ApplyBlock(index iRow, index iBlock, InputVector input, Eigen::Ref<Eigen::VectorXd> output) const override
        {
            CPU::ApplyEquationJacobianBlock(*this, iRow, iBlock, input, output);
        }
    };

    std::shared_ptr<LinearProblemContext> FiniteVolume::BuildLinearProblem(const EquationDofs &prototype, DiagonalStorage diagonal, OffDiagonalStorage offDiagonal, LinearMatrixOptions storage) const
    {
        Metrics().Validate();
        _data->partition.Validate();
        prototype.RequireSchema(_data->equation.Schema());
        if (_data->equation.IsACM())
            DNDS_check_throw_info(diagonal == DiagonalStorage::Block && offDiagonal == OffDiagonalStorage::Assembled,
                                  "ACM requires a full cell block and assembled off-diagonal blocks");
        if (prototype.Device() == R::Device::Host())
            _data->equation.ResolveKernelProfile(prototype.Device());
        else if (!_data->equation.IsACM())
            PerfectGasSpatialParameters::FromEquation(_data->equation, Metrics().Dimension()).Validate(int(prototype.ComponentNumber()));
        DNDS_check_throw_info(prototype.Space().Same(Metrics().Space()) && prototype.Device() == Device(),
                              "equation linear prototype row space or placement differs");
        LinearSweepPartitions partitions;
        if (prototype.Device() == R::Device::Host() && Mesh().NLocalParts() > 0)
        {
            for (int part = 0; part < Mesh().NLocalParts(); ++part)
                partitions.starts.push_back(Mesh().LocalPartStart(part));
            partitions.starts.push_back(Mesh().LocalPartEnd(Mesh().NLocalParts() - 1));
        }
        return std::make_shared<LinearProblemContext>(prototype.Space(), prototype.ComponentNumber(),
                                                      _data->jacobianColumns->CopyColumns(), _data->partition.Core(), diagonal, offDiagonal, prototype.Device(), std::move(partitions), storage);
    }
    void FiniteVolume::PrepareJacobian(const EquationDofs &state, const AuxiliaryValues &auxiliary, const R::Array<real> &inversePseudoTime,
                                       LinearProblemContext &output, JacobianPreparation controls, real time, std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                       std::optional<JacobianContext> context) const
    {
        const auto *sourceWorkspace = context ? &context->workspace.get() : nullptr;
        const TimelineScope timeline("Cauchy/Linear/PrepareJacobianAllBackends");
        ValidateAuxValues(state, auxiliary, time);
        if (state.Device() != R::Device::Host())
        {
            DNDS_check_throw_info(context && !workers, "device Jacobian requires its workspace/lane context and no CPU chemical workers");
            PrepareDeviceJacobian(state, auxiliary, inversePseudoTime, output, controls, time, *context);
            return;
        }
        const bool acm = _data->equation.IsACM();
        if (!acm && controls.roeDissipation && sourceWorkspace && HasRoeJacobianKernel(*sourceWorkspace))
        {
            DNDS_check_throw_info(!workers && state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1,
                                  "flow5 Roe Jacobian specialization received an extended physical state");
            PrepareFlow5Jacobian(state.CoupledField(), auxiliary, inversePseudoTime, output, controls, time);
            return;
        }
        output.Validate();
        const index nv = state.ComponentNumber(), owned = Metrics().NumCell(), all = Metrics().NumCellProc();
        DNDS_check_throw_info(output.Space().Same(state.Space()) && output.Components() == nv && (acm || !controls.roeDissipation),
                              "equation Jacobian space/components differ or Roe profile is unavailable");
        DNDS_check_throw_info(std::isfinite(controls.physicalMass) && controls.physicalMass >= 0 && std::isfinite(controls.residualScale) && controls.residualScale >= 0, "invalid integrated equation Jacobian controls");
        DNDS_check_throw_info(inversePseudoTime.GetDevice() == R::Device::Host() && inversePseudoTime.RowCount() == owned && !inversePseudoTime.Layout().IsVariable() && inversePseudoTime.Layout().Width() == 1, "equation inverse pseudo-time shape/backend differs");
        DNDS_check_throw_info(auxiliary.faceSpectralRadius && _data->equation.HasMixture() == bool(workers), "equation Jacobian requires face signals and a matching worker backend");
        output.ValidateColumns(_data->jacobianColumns);
        const auto checkAlias = [&](const R::Array<real> &input)
        {
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(input.Storage(), output.Diagonal().Storage()) &&
                                      (!output.HasAssembledOffDiagonal() || !Algebra::Detail::Overlaps(input.Storage(), output.OffDiagonal().Values().Storage())),
                                  "equation Jacobian output aliases input");
        };
        checkAlias(inversePseudoTime);
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            for (const auto *input : {&state.PhysicalField(iField).Owned(), &state.PhysicalField(iField).Ghost()})
                checkAlias(*input);
        for (const auto *input : {auxiliary.meanThermodynamics ? &*auxiliary.meanThermodynamics : nullptr,
                                  auxiliary.faceSpectralRadius ? &*auxiliary.faceSpectralRadius : nullptr, auxiliary.faceWaveJump ? &*auxiliary.faceWaveJump : nullptr,
                                  auxiliary.facePressureMinimum ? &*auxiliary.facePressureMinimum : nullptr,
                                  auxiliary.cellSpectralSum ? &auxiliary.cellSpectralSum->Owned() : nullptr, auxiliary.cellWaveJump ? &auxiliary.cellWaveJump->Owned() : nullptr,
                                  auxiliary.cellPressureMinimum ? &auxiliary.cellPressureMinimum->Owned() : nullptr,
                                  auxiliary.derivedOwned ? &auxiliary.derivedOwned->Values() : nullptr, auxiliary.derivedGhost ? &auxiliary.derivedGhost->Values() : nullptr})
            if (input)
                checkAlias(*input);
        if (acm)
        {
            DNDS_check_throw_info(output.DiagonalMode() == DiagonalStorage::Block && output.HasAssembledOffDiagonal(),
                                  "ACM matrix requires full diagonal and off-diagonal blocks");
            DNDS_check_throw_info(nv >= 4 && nv <= 6 && !HasInterfaces() && !_data->motion.At(time).moving,
                                  "ACM Jacobian requires stationary single-domain flow4, flow4+SA or flow4+2EQ");
            const auto input = state.ReadWithGhost();
            const auto layout = input.ComponentLayout();
            const auto stencil = _data->stencil.Read();
            const auto tau = inversePseudoTime.Read(), radius = auxiliary.faceSpectralRadius->Read();
            auto physics = _data->equation.ACMParameters();
            physics.roe = _data->settings.numerics.flux == "Roe";
            const Algebra::ComponentSelection selection(nv, controls.updateComponents);
            std::array<bool, 6> active{};
            for (index j = 0; j < nv; ++j)
                active[j] = selection.Includes(j);
            std::optional<WallDistanceRead> walls;
            std::optional<R::Array<real>::ReadLease> lengths;
            Physics::RANS::PointParameters rans;
            if (nv > 4)
            {
                DNDS_check_throw_info(_data->wallDistance, "ACM RANS Jacobian requires wall distances");
                walls.emplace(_data->wallDistance->AcquireRead());
                lengths.emplace(_data->ransCellLength.Read());
                rans = _data->equation.RANS().KernelParameters();
            }
            auto diagonal = output.Diagonal().Write();
            std::optional<Algebra::BlockMatrixStorage::WriteAccess> off;
            try
            {
                off.emplace(output.OffDiagonal().Write());
                RequireNumericalSuccess(CPU::PrepareACMJacobian(
                                            {physics, stencil.View(), {_data->boundaryConditions.data(), index(_data->boundaryConditions.size())}, layout.View(), tau.View<R::UniformTableRows>(), radius.View<R::UniformTableRows>(), walls ? walls->HostView().cellMean : decltype(WallDistanceView::cellMean){}, lengths ? lengths->View<R::FixedRows<1>>() : decltype(lengths->View<R::FixedRows<1>>()){}, rans, diagonal.View<R::UniformTableRows>(), off->ArrayView(), controls.physicalMass, controls.residualScale, active}),
                                        "assemble ACM Jacobian");
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("Cauchy ACM Jacobian failed");
                if (off)
                {
                    off->Fail();
                    diagonal.Fail();
                }
                else
                    diagonal.CancelUnmodified();
                throw;
            }
            return;
        }
        const bool chemical = _data->settings.equation.source == "chemical";
        const bool rans = _data->equation.HasRANS();
        DNDS_check_throw_info(_data->settings.equation.source == "none" || (chemical && _data->equation.HasMixture()), "unsupported equation Jacobian source");
        if (chemical || rans)
        {
            DNDS_check_throw_info(sourceWorkspace || (!rans && (_data->settings.numerics.reconstruction == "constant" || _data->equation.Mixture().SpeciesNumber() == 1)),
                                  "RANS or nonconstant chemical Jacobian requires current RHS source reconstruction");
            if (sourceWorkspace)
                ValidateEquationSourceJacobian(state, *context, time);
        }
        auto pointWorkers = MakeCPUPointWorkers<CPU::EquationJacobianWorker>(_data->equation, workers,
                                                                             {rans, false, true, chemical, chemical}, nv);
        auto snapshot = std::make_shared<JacobianData>(*this, state, output, controls, time);
        std::optional<JacobianData::WriteAccess> snapshotWrite(std::in_place, *snapshot);
        const auto input = state.ReadWithGhost();
        const auto blocks = EquationFVDetail::BlockViews(state, input);
        const auto inputLayout = input.ComponentLayout();
        const auto inputValues = inputLayout.View();
        std::optional<R::Array<real>::ReadLease> incomingFaceRead;
        std::optional<BoundaryInputs> boundaryInputs;
        InterfaceBoundaryRows incomingFace;
        if (HasInterfaces())
        {
            if (sourceWorkspace)
                boundaryInputs = PreparedBoundaryInputs(*sourceWorkspace);
            DNDS_check_throw_info(
                boundaryInputs,
                "equation interface Jacobian needs the current boundary publication");
            ValidateBoundaryInputs(*boundaryInputs, time);
            incomingFaceRead.emplace(
                boundaryInputs->FaceCenterValues()
                    .CoupledField()
                    .Owned()
                    .Read());
            incomingFace =
                incomingFaceRead->View<R::UniformTableRows>();
        }
        auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        const auto faceRead = auxiliary.faceSpectralRadius->Read(), pseudo = inversePseudoTime.Read();
        const auto radius = faceRead.View<R::UniformTableRows>(), tau = pseudo.View<R::UniformTableRows>();
        std::optional<R::Array<real>::WriteLease> diagonal(output.Diagonal().Write());
        std::optional<Algebra::BlockMatrixStorage::WriteAccess> off;
        try
        {
            if (output.HasAssembledOffDiagonal())
                off.emplace(output.OffDiagonal().Write());
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy equation Jacobian failed");
            diagonal->CancelUnmodified();
            throw;
        }
        const auto d = diagonal->View<R::UniformTableRows>();
        std::optional<Algebra::BlockMatrixView<real>> b;
        if (off)
            b.emplace(off->View());
        R::Array<real> boundaryMeans(R::RowLayout::Dense(_data->stencil.IncidenceNumber(), nv));
        auto boundaryMeansWrite = boundaryMeans.Write();
        const auto stencilRead = _data->stencil.Read();
        const R::MetadataBuffer<EquationBlockReadView> blockMetadata(blocks);
        std::optional<WallDistanceRead> wallRead;
        if (rans && _data->equation.RANS().ComponentNumber() == 2)
        {
            DNDS_check_throw_info(
                _data->wallDistance,
                "two-equation boundary Jacobian needs wall-distance samples");
            wallRead.emplace(_data->wallDistance->AcquireRead());
        }
        const auto wallValues = wallRead ? wallRead->HostView()
                                         : WallDistanceView{};
        CPU::EquationJacobianPreparation request{
            *snapshot, stencilRead.View(), metric, {_data->boundaryConditions.data(), index(_data->boundaryConditions.size())}, blockMetadata.HostView(), inputValues, radius, tau, incomingFace, d, snapshotWrite->radius.View<R::TableRows>(), boundaryMeansWrite.View<R::TableRows>(), b, {snapshot->flowFirst, snapshot->speciesFirst, snapshotWrite->derivative.View<R::TableRows>(), snapshotWrite->pressure.View<R::TableRows>(), output.HasAssembledOffDiagonal() ? std::nullopt : std::optional(snapshotWrite->means.View<R::TableRows>())}, wallValues, time, controls.residualScale, controls.physicalMass, owned, all, snapshot->independent, output.DiagonalMode() == DiagonalStorage::Block, chemical && !sourceWorkspace, bool(wallRead)};
        request.linearizeBoundary = controls.scalarBoundaryDerivative;
        request.rowMajorBlocks = output.RowMajorBlocks();
        try
        {
            RequireNumericalSuccess(CPU::InitializeEquationJacobian(request), "initialize equation Jacobian");
            bool ownedPropertiesPrepared = false;
            if ((chemical || rans) && sourceWorkspace)
                ownedPropertiesPrepared = AddEquationSourceJacobian(state, *context, *diagonal,
                                                                    output.DiagonalMode(), controls.residualScale, pointWorkers.front().evaluator, std::cref(request.linearization), output.RowMajorBlocks());
            RequireNumericalSuccess(CPU::CompleteEquationJacobian(request, _data->boundaryPoint,
                                                                  {pointWorkers.data(), index(pointWorkers.size())}, ownedPropertiesPrepared),
                                    "complete equation Jacobian");
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy equation Jacobian failed");
            diagonal->Fail();
            if (off)
                off->Fail();
            throw;
        }
        diagonal.reset();
        off.reset();
        snapshotWrite.reset();
        if (!output.HasAssembledOffDiagonal())
        {
            snapshot->Publish();
            output.PublishOffDiagonalEvaluator(std::move(snapshot));
        }
    }
}
