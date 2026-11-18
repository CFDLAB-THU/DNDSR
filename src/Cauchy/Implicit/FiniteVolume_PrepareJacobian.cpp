#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Implicit/JacobianGeometry.hpp"
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Implicit/CPU/FiniteVolumeJacobianKernels.hpp"
#include <Eigen/Core>
#include <optional>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;
    using Matrix = Eigen::Matrix<real, 5, 5>;

    /// Frozen physics with reusable geometry, independent of the FV owner's
    /// communicator lifetime. Only the weak owner key is used to reuse geometry
    /// at the next preparation. Block evaluation needs no mesh/Array leases.
    struct FiniteVolume::MatrixFreeJacobian final : OffDiagonalEvaluator, CPU::FlowJacobianView
    {
        struct Geometry
        {
            std::weak_ptr<const Data> owner;
            std::shared_ptr<const Algebra::DofSpace::Partition> partition;
            std::shared_ptr<const JacobianGeometryStorage> storage;
            std::optional<JacobianGeometryStorage::ReadAccess> read;
        };
        std::shared_ptr<const Geometry> geometryOwner;
        PerfectGas gas;
        R::Array<real> meanStorage, radiusStorage, magnitudeStorage;
        std::optional<R::Array<real>::ReadLease> meanRead, radiusRead, magnitudeRead;
        MatrixFreeJacobian(const FiniteVolume &fv, const LinearProblemContext &output, JacobianPreparation controls, real time)
            : gas(fv._data->equation.Gas()),
              meanStorage(R::RowLayout::Dense(fv.Metrics().NumCellProc(), 5)),
              radiusStorage(R::RowLayout::Dense(fv._data->stencil.stencilFaces.RowCount(), 2)),
              magnitudeStorage(R::RowLayout::Dense(controls.roeDissipation ? fv._data->stencil.IncidenceNumber() : 0, 3))
        {
            frame = fv._data->motion.At(time);
            scale = controls.residualScale;
            roe = controls.roeDissipation;
            const Algebra::ComponentSelection selection(5, controls.updateComponents);
            selected = !selection.All();
            for (int j = 0; j < 5; ++j)
                active[j] = selection.Includes(j);
            auto old = std::dynamic_pointer_cast<const MatrixFreeJacobian>(output.PreparedOffDiagonalEvaluator());
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
        }
        void Publish()
        {
            meanRead.emplace(meanStorage.Read());
            radiusRead.emplace(radiusStorage.Read());
            magnitudeRead.emplace(magnitudeStorage.Read());
            means = meanRead->View<R::FixedRows<5>>();
            radius = radiusRead->View<R::TableRows>();
            magnitudes = magnitudeRead->View<R::TableRows>();
        }
        index Components() const override { return 5; }
        std::shared_ptr<const Algebra::DofSpace::Partition> RowSpaceIdentity() const override { return geometryOwner->partition; }
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
            CPU::EvaluateFlowJacobianBlock(*this, gas, iRow, iBlock, output);
        }
        void ApplyBlock(index iRow, index iBlock, InputVector input, Eigen::Ref<Eigen::VectorXd> output) const override
        {
            CPU::ApplyFlowJacobianBlock(*this, gas, iRow, iBlock, input, output);
        }
    };

    void FiniteVolume::PrepareFlow5Jacobian(const CFV::FVField &state, const AuxiliaryValues &auxiliary,
                                            const R::Array<real> &inversePseudoTime, LinearProblemContext &output,
                                            JacobianPreparation controls, real time) const
    {
        const Algebra::ComponentSelection selection(5, controls.updateComponents);
        std::array<bool, 5> active{};
        for (int j = 0; j < 5; ++j)
            active[j] = selection.Includes(j);
        output.Validate();
        DNDS_check_throw_info(output.Space().Same(state.Space()) && output.Components() == 5 &&
                                  (!controls.roeDissipation || output.DiagonalMode() == DiagonalStorage::Block),
                              "Jacobian space/components differ or Roe diagonal is not a block");
        DNDS_check_throw_info(std::isfinite(controls.physicalMass) && controls.physicalMass >= 0 &&
                                  std::isfinite(controls.residualScale) && controls.residualScale >= 0,
                              "invalid integrated Jacobian coefficients");
        DNDS_check_throw_info(inversePseudoTime.GetDevice() == R::Device::Host() && inversePseudoTime.RowCount() == Metrics().NumCell() &&
                                  !inversePseudoTime.Layout().IsVariable() && inversePseudoTime.Layout().Width() == 1,
                              "inverse pseudo-time must be an owned CPU scalar table");
        DNDS_check_throw_info(auxiliary.faceSpectralRadius, "Jacobian requires prepared face spectral radii");
        const auto &faceArray = *auxiliary.faceSpectralRadius;
        const auto checkAlias = [&](const R::Array<real> &source)
        {
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(source.Storage(), output.Diagonal().Storage()) &&
                                      (!output.HasAssembledOffDiagonal() || !Algebra::Detail::Overlaps(source.Storage(), output.OffDiagonal().Values().Storage())),
                                  "Jacobian output aliases a frozen input");
        };
        checkAlias(state.Owned());
        checkAlias(state.Ghost());
        checkAlias(faceArray);
        checkAlias(inversePseudoTime);
        const auto boundary = PreparedBoundaryInputs(auxiliary);
        if (boundary)
        {
            for (const auto *packet : {&boundary->QuadratureValues(),
                                       &boundary->FaceCenterValues()})
                for (index iField = 0; iField < packet->PhysicalFieldNumber();
                     ++iField)
                    checkAlias(packet->PhysicalField(iField).Owned());
        }
        if (auxiliary.meanThermodynamics)
            checkAlias(*auxiliary.meanThermodynamics);
        for (const auto *derived : {&auxiliary.derivedOwned, &auxiliary.derivedGhost})
            if (*derived)
                checkAlias((*derived)->Values());
        for (const auto *field : {&auxiliary.cellWaveJump, &auxiliary.cellPressureMinimum})
            if (*field)
            {
                checkAlias((*field)->Owned());
                checkAlias((*field)->Ghost());
            }
        auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        const auto frame = _data->motion.At(time);
        const auto owned = state.Owned().Read(), ghost = state.Ghost().Read(), face = faceArray.Read(), pseudo = inversePseudoTime.Read();
        const RHSFlowFieldView mean{owned.View<RHSFlowRows>(), ghost.View<RHSFlowRows>()};
        RHSAuxiliaryViews packet;
        std::array<std::optional<R::Array<real>::ReadLease>, 5> retained;
        std::optional<DerivedFields::ReadScope> derivedOwned, derivedGhost;
        std::optional<R::Array<real>::ReadLease> boundaryRead;
        RHSScalarView incoming;
        if (boundary)
        {
            boundaryRead.emplace(
                boundary->FaceCenterValues().CoupledField().Owned().Read());
            incoming = boundaryRead->View<R::UniformTableRows>();
        }
        if (auxiliary.meanThermodynamics)
        {
            retained[0].emplace(auxiliary.meanThermodynamics->Read());
            packet.meanThermodynamics = retained[0]->View<R::UniformTableRows>();
        }
        else if (auxiliary.derivedOwned && auxiliary.derivedGhost &&
                 auxiliary.derivedOwned->Request().Has(DerivedQuantity::Pressure) &&
                 auxiliary.derivedOwned->Request().Has(DerivedQuantity::SoundSpeed) &&
                 auxiliary.derivedOwned->Request().Has(DerivedQuantity::DiffusionBound))
        {
            const auto typed = EquationDofs::Coupled(_data->equation.Schema(), state);
            derivedOwned.emplace(auxiliary.derivedOwned->ReadCurrent(typed, _data->equation));
            derivedGhost.emplace(auxiliary.derivedGhost->ReadCurrent(typed, _data->equation));
            packet.meanThermodynamics = MeanThermodynamicView::Derived(
                *derivedOwned,
                OptionalRef<const DerivedFields::ReadScope>(*derivedGhost));
        }
        const auto bind = [&](const auto &field, int slot, auto &view)
        {
            if (!field)
                return;
            retained[slot].emplace(field->Owned().Read());
            retained[slot + 1].emplace(field->Ghost().Read());
            view = RHSFieldView{retained[slot]->View<R::UniformTableRows>(), retained[slot + 1]->View<R::UniformTableRows>()};
        };
        bind(auxiliary.cellWaveJump, 1, packet.cellWaveJump);
        bind(auxiliary.cellPressureMinimum, 3, packet.cellPressureMinimum);
        DNDS_check_throw_info(!controls.roeDissipation || !_data->settings.numerics.riemann.useStencil ||
                                  ((!_data->flux.UsesTransverseSignal() || packet.cellWaveJump) && (!_data->flux.UsesPressureStencil() || packet.cellPressureMinimum)),
                              "Roe Jacobian requires the configured flux's prepared stencil signals");
        const auto radius = face.View<R::UniformTableRows>(), inverseTau = pseudo.View<R::UniformTableRows>();
        DNDS_check_throw_info(radius.RowCount() == _data->stencil.stencilFaces.RowCount() && radius.Width() == 2, "Jacobian face spectral shape differs");
        std::shared_ptr<MatrixFreeJacobian> evaluated;
        if (!output.HasAssembledOffDiagonal())
        {
            evaluated = std::make_shared<MatrixFreeJacobian>(*this, output, controls, time);
            auto meanWrite = evaluated->meanStorage.Write(), radiusWrite = evaluated->radiusStorage.Write();
            CPU::FreezeFlowJacobian(mean, radius, meanWrite.View<R::FixedRows<5>>(), radiusWrite.View<R::TableRows>());
        }
        std::optional<R::Array<real>::WriteLease> diagonal(output.Diagonal().Write());
        std::optional<Algebra::BlockMatrixStorage::WriteAccess> offDiagonal;
        try
        {
            if (output.HasAssembledOffDiagonal())
                offDiagonal.emplace(output.OffDiagonal().Write());
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy Jacobian preparation failed");
            diagonal->CancelUnmodified();
            throw;
        }
        const auto d = diagonal->View<R::UniformTableRows>();
        const auto blocks = offDiagonal ? std::optional(offDiagonal->View()) : std::nullopt;
        const auto stencilRead = _data->stencil.Read();
        std::optional<R::Array<real>::WriteLease> magnitudeWrite;
        if (evaluated && controls.roeDissipation)
            magnitudeWrite.emplace(evaluated->magnitudeStorage.Write());
        CPU::FlowJacobianPreparation request;
        request.stencil = stencilRead.View();
        request.metric = metric;
        request.conditions = {_data->boundaryConditions.data(), index(_data->boundaryConditions.size())};
        request.mean = mean;
        request.packet = packet;
        request.radius = radius;
        request.inverseTau = inverseTau;
        request.incoming = incoming;
        request.diagonal = d;
        request.blocks = blocks;
        if (magnitudeWrite)
            request.magnitudes = magnitudeWrite->View<R::FixedRows<3>>();
        request.frame = frame;
        request.fluxSettings = _data->settings.numerics.riemann.KernelSettings();
        request.time = time;
        request.physicalMass = controls.physicalMass;
        request.scale = controls.residualScale;
        request.fullBlock = output.DiagonalMode() == DiagonalStorage::Block;
        request.rowMajorBlocks = output.RowMajorBlocks();
        request.roe = controls.roeDissipation;
        request.evaluated = bool(evaluated);
        request.selected = !selection.All();
        request.active = active;
        RequireNumericalSuccess(CPU::PrepareCompressibleFluxJacobian(request, _data->equation.Gas(), _data->flux), "PrepareCompressibleFluxJacobian");
        magnitudeWrite.reset();
        diagonal.reset();
        if (evaluated)
        {
            evaluated->Publish();
            output.PublishOffDiagonalEvaluator(std::move(evaluated));
        }
    }
}
