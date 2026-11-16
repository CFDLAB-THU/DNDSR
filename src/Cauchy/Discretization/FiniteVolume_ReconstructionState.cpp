#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeReconstructionKernels.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/ViscousFlux.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationFVDetail;
    bool FiniteVolume::ReconstructionState::FinalizedCurrent() const
    {
        if (!coefficients)
            return true; // No polynomial publication exists for direct P0/P1.
        const auto current = [](const EquationDofOperations::Binding &binding, const EquationDofs &values)
        {
            if (binding.snapshot.PhysicalFieldNumber() != values.PhysicalFieldNumber())
                return false;
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
            {
                const auto &saved = binding.snapshot.PhysicalField(iField).Owned();
                const auto &value = values.PhysicalField(iField).Owned();
                if (saved.SharedStorage() != value.SharedStorage() ||
                    binding.versions[size_t(iField)] != value.Version())
                    return false;
            }
            return true;
        };
        if (!finalizedFrom || !finalizedPublication)
            return false;
        return current(*finalizedFrom, *coefficients) &&
               current(*finalizedPublication, FinalizedValues());
    }
    void FiniteVolume::ReconstructionState::PublishFinalized()
    {
        if (!coefficients)
            return;
        finalizedFrom.emplace(*coefficients);
        finalizedPublication.emplace(FinalizedValues());
    }
    const EquationDofs &FiniteVolume::ReconstructionState::WarmValues() const
    {
        DNDS_check_throw_info(coefficients, "direct P0/P1 has no modal coefficients");
        return *coefficients;
    }
    EquationDofs &FiniteVolume::ReconstructionState::WarmValues()
    {
        DNDS_check_throw_info(coefficients, "direct P0/P1 has no modal coefficients");
        return *coefficients;
    }
    const EquationDofs &FiniteVolume::ReconstructionState::FinalizedValues() const
    {
        return distinctFinalized ? *fluxCoefficients : WarmValues();
    }
    EquationDofs &FiniteVolume::ReconstructionState::FinalizedValues()
    {
        return distinctFinalized ? *fluxCoefficients : WarmValues();
    }
    EquationDofs &FiniteVolume::ReconstructionState::MutableFinalized()
    {
        DNDS_check_throw_info(distinctFinalized,
                              "unfiltered reconstruction has no mutable finalized scratch");
        return *fluxCoefficients;
    }
    std::vector<R::TransferBinding> &FiniteVolume::ReconstructionState::FinalizedHalo()
    {
        return distinctFinalized ? fluxHalo : coefficientHalo;
    }

    EquationDofs FiniteVolume::ReconstructionState::BuildCoefficients(const Data &data, const EquationDofs &state)
    {
        DNDS_check_throw_info(data.reconstruction,
                              "coefficient allocation requires a polynomial plan");
        Algebra::DofBundle fields(state.Space());
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            const auto &physical = state.PhysicalField(iField);
            fields.Add(state.Fields().Name(iField), data.reconstruction->BuildCoefficients(
                                                        index(physical.Width()),
                                                        FiniteVolumeDetail::FieldLayout(physical.Owned().Layout(), data.metrics.NumCell(),
                                                                                        index(physical.Width()) * data.reconstruction->NRecDOF())));
        }
        return FiniteVolumeDetail::BindFieldSchema(state, std::move(fields), data.reconstruction->NRecDOF());
    }
    FiniteVolume::ReconstructionState::ReconstructionState(std::shared_ptr<const Data> data, EquationDofs &state)
        : owner(std::move(data)), binding(state),
          distinctFinalized(owner->reconstruction && ((owner->settings.numerics.limiter == "WBAP" &&
                                                       !(owner->settings.numerics.reconstruction == "gradient" &&
                                                         !owner->settings.numerics.positivity)) ||
                                                      bool(owner->limitedVRGeometry) || owner->settings.numerics.positivity))
    {
        if (owner->reconstruction)
        {
            coefficients.emplace(BuildCoefficients(*owner, state));
            coefficients->Fields().SetName("Cauchy/reconstruction/warm", DNDS_ALLOCATION_SITE());
        }
        if (distinctFinalized)
        {
            fluxCoefficients.emplace(BuildCoefficients(*owner, state));
            fluxCoefficients->Fields().SetName("Cauchy/reconstruction/finalized", DNDS_ALLOCATION_SITE());
        }
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            meanHalo.push_back(owner->BindTrackedHalo(state.PhysicalField(iField)));
            if (owner->reconstruction)
            {
                Algebra::Fill(coefficients->PhysicalField(iField), 0.);
                coefficientHalo.push_back(owner->BindHalo(coefficients->PhysicalField(iField)));
            }
            if (distinctFinalized)
            {
                Algebra::Fill(fluxCoefficients->PhysicalField(iField), 0.);
                fluxHalo.push_back(owner->BindHalo(fluxCoefficients->PhysicalField(iField)));
            }
        }
        if (owner->limitedVRGeometry)
        {
            limitedVRHistory.emplace(R::RowLayout::Dense(owner->metrics.NumCell(), 2), owner->metrics.Device());
            limitedVRHistory->SetName("Cauchy/reconstruction/limitedVRHistory", DNDS_ALLOCATION_SITE());
            // Construction-time initialization is synchronous and follows
            // the known allocation backend; history must start at zero on
            // CUDA too, before any LVR recurrence reads it.
            const R::ExecutionContext context{owner->metrics.Device(), nullptr};
            auto write = limitedVRHistory->Write(context);
            Algebra::ApplyRowViews(context, Algebra::PointwiseOperation::Fill, write.DynamicView(), {}, {}, 0);
            write.Complete(R::Completion::Record(context));
        }
        PublishFinalized();
    }
    FiniteVolume::FinalizedReconstruction::FinalizedReconstruction(
        std::shared_ptr<const Data> owner, const EquationDofs &source,
        const EquationDofs &values, bool distinct)
        : _owner(std::move(owner)), _source(source.NumericAlias()),
          _values(values.NumericAlias()), _distinct(distinct)
    {
        for (index iField = 0; iField < _source.PhysicalFieldNumber(); ++iField)
            _sourceVersions.push_back(_source.PhysicalField(iField).Owned().Version());
        for (index iField = 0; iField < _values.PhysicalFieldNumber(); ++iField)
            _valueVersions.push_back(_values.PhysicalField(iField).Owned().Version());
    }
    void FiniteVolume::FinalizedReconstruction::Validate() const
    {
        DNDS_check_throw_info(_owner &&
                                  _sourceVersions.size() == size_t(_source.PhysicalFieldNumber()) &&
                                  _valueVersions.size() == size_t(_values.PhysicalFieldNumber()),
                              "invalid finalized reconstruction publication");
        for (index iField = 0; iField < _source.PhysicalFieldNumber(); ++iField)
            DNDS_check_throw_info(_sourceVersions[size_t(iField)] ==
                                      _source.PhysicalField(iField).Owned().Version(),
                                  "warm reconstruction changed after finalization");
        for (index iField = 0; iField < _values.PhysicalFieldNumber(); ++iField)
            DNDS_check_throw_info(_valueVersions[size_t(iField)] ==
                                      _values.PhysicalField(iField).Owned().Version(),
                                  "finalized reconstruction changed after publication");
    }
    const EquationDofs &FiniteVolume::FinalizedReconstruction::Values() const
    {
        Validate();
        return _values;
    }
    const EquationDofs &FiniteVolume::ReconstructionGuess(const ReconstructionState &state) const
    {
        DNDS_check_throw_info(state.owner == _data,
                              "foreign equation reconstruction state");
        DNDS_check_throw_info(_data->reconstruction,
                              "direct P0/P1 has no modal reconstruction guess");
        return state.WarmValues();
    }
    FiniteVolume::ReconstructionStorageInfo FiniteVolume::InspectReconstructionStorage(
        const ReconstructionState &state) const
    {
        DNDS_check_throw_info(state.owner == _data,
                              "foreign equation reconstruction state");
        ReconstructionStorageInfo result;
        result.polynomialPlan = bool(_data->reconstruction);
        result.distinctFinalized = state.distinctFinalized;
        if (result.polynomialPlan)
        {
            result.coefficientFields = state.WarmValues().PhysicalFieldNumber() *
                                       (state.distinctFinalized ? 2 : 1);
            result.coefficientTransferBindings = index(state.coefficientHalo.size() +
                                                       state.fluxHalo.size());
            const auto accumulate = [&](const EquationDofs &values)
            {
                for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                {
                    result.coefficientOwnedBytes = CheckedSize::Add(
                        result.coefficientOwnedBytes,
                        values.PhysicalField(iField).Owned().Storage().Bytes());
                    result.coefficientGhostBytes = CheckedSize::Add(
                        result.coefficientGhostBytes,
                        values.PhysicalField(iField).Ghost().Storage().Bytes());
                }
            };
            accumulate(state.WarmValues());
            if (state.distinctFinalized)
                accumulate(state.FinalizedValues());
        }
        return result;
    }
    void InterfaceDonorPublication::Validate() const
    {
        DNDS_check_throw_info(_owner &&
                                  _meanVersions.size() ==
                                      size_t(_means.PhysicalFieldNumber()),
                              "invalid interface donor publication");
        _means.RequireSchema(_owner->equation.Schema());
        for (index iField = 0; iField < _means.PhysicalFieldNumber(); ++iField)
        {
            _owner->metrics.ValidateField(
                _means.PhysicalField(iField),
                _means.PhysicalField(iField).Width());
            DNDS_check_throw_info(
                _meanVersions[size_t(iField)] ==
                    _means.PhysicalField(iField).Owned().Version(),
                "interface donor mean changed after publication");
        }
        if (_kind == Kind::Variational)
        {
            DNDS_check_throw_info(_owner->reconstruction && _coefficients &&
                                      _reconstructionVersions.size() ==
                                          size_t(_coefficients->PhysicalFieldNumber()),
                                  "variational interface donor is incomplete");
            for (index iField = 0; iField < _coefficients->PhysicalFieldNumber();
                 ++iField)
            {
                DNDS_check_throw_info(
                    _coefficients->Fields().Name(iField) ==
                            _means.Fields().Name(iField) &&
                        _coefficients->PhysicalField(iField).Width() ==
                            _means.PhysicalField(iField).Width() *
                                _owner->reconstruction->NRecDOF(),
                    "variational interface donor field grouping differs");
                _owner->metrics.ValidateField(
                    _coefficients->PhysicalField(iField),
                    _coefficients->PhysicalField(iField).Width());
                DNDS_check_throw_info(
                    _reconstructionVersions[size_t(iField)] ==
                        _coefficients->PhysicalField(iField).Owned().Version(),
                    "interface donor coefficients changed after publication");
            }
        }
        if (_kind == Kind::Gradient)
        {
            DNDS_check_throw_info(
                _gradients.size() == size_t(_means.PhysicalFieldNumber()) &&
                    _reconstructionVersions.size() == _gradients.size(),
                "gradient interface donor is incomplete");
            for (index iField = 0; iField < index(_gradients.size()); ++iField)
            {
                _owner->metrics.ValidateField(
                    _gradients[size_t(iField)],
                    _means.PhysicalField(iField).Width() *
                        _owner->metrics.Dimension());
                DNDS_check_throw_info(
                    _reconstructionVersions[size_t(iField)] ==
                        _gradients[size_t(iField)].Owned().Version(),
                    "interface donor gradient changed after publication");
            }
        }
    }
    InterfaceDonorPublication FiniteVolume::PublishInterfaceDonor(
        const EquationDofs &state, const ReconstructionState &reconstruction,
        const Workspace &work, bool prepared) const
    {
        ValidateWorkspace(state, work);
        DNDS_check_throw_info(reconstruction.owner == _data,
                              "foreign interface reconstruction state");
        reconstruction.binding.Check(state);
        InterfaceDonorPublication result(_data, state.NumericAlias());
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            result._meanVersions.push_back(
                state.PhysicalField(iField).Owned().Version());
        if (_data->settings.numerics.reconstruction == "variational")
        {
            const auto &values = prepared ? reconstruction.FinalizedValues()
                                          : reconstruction.WarmValues();
            if (prepared)
                DNDS_check_throw_info(reconstruction.FinalizedCurrent(),
                                      "interface donor reconstruction is stale");
            result._kind = InterfaceDonorPublication::Kind::Variational;
            result._coefficients = values.NumericAlias();
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                result._reconstructionVersions.push_back(
                    values.PhysicalField(iField).Owned().Version());
        }
        else if (prepared &&
                 _data->settings.numerics.reconstruction == "gradient")
        {
            DNDS_check_throw_info(
                work.nativeDirectGradient &&
                    work.directGradients.size() ==
                        size_t(state.PhysicalFieldNumber()) &&
                    work.directGradientPublications.size() ==
                        work.directGradients.size(),
                "interface donor gradient publication is unavailable");
            result._kind = InterfaceDonorPublication::Kind::Gradient;
            result._gradients = work.directGradients;
            for (index iField = 0; iField < index(work.directGradients.size());
                 ++iField)
            {
                const auto &field = work.directGradients[size_t(iField)].Owned();
                const auto &publication =
                    work.directGradientPublications[size_t(iField)][0];
                DNDS_check_throw_info(
                    publication.Current(
                        *_data, state.PhysicalField(iField).Owned(),
                        state.PhysicalField(iField).Owned(), field,
                        work.incoming, work.rhsTime),
                    "interface donor gradient publication is stale");
                result._reconstructionVersions.push_back(field.Version());
            }
        }
        result.Validate();
        return result;
    }
    InterfaceDonorPublication
    FiniteVolume::PublishVariationalInterfaceDonor(
        const EquationDofs &state, const EquationDofs &coefficients) const
    {
        state.RequireSchema(_data->equation.Schema());
        DNDS_check_throw_info(
            _data->reconstruction && state.Space().Same(Metrics().Space()) &&
                coefficients.Space().Same(state.Space()) &&
                coefficients.Device() == state.Device() &&
                coefficients.PhysicalFieldNumber() == state.PhysicalFieldNumber(),
            "checkpoint interface donor needs compatible variational fields");
        InterfaceDonorPublication result(_data, state.NumericAlias());
        result._kind = InterfaceDonorPublication::Kind::Variational;
        result._coefficients = coefficients.NumericAlias();
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            result._meanVersions.push_back(
                state.PhysicalField(iField).Owned().Version());
            result._reconstructionVersions.push_back(
                coefficients.PhysicalField(iField).Owned().Version());
        }
        result.Validate();
        return result;
    }
    void FiniteVolume::ValidateWorkspace(const EquationDofs &state, const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        state.RequireSchema(_data->equation.Schema());
        DNDS_check_throw_info(state.Space().Same(Metrics().Space()) &&
                                  state.Device() == Device() &&
                                  state.ComponentNumber() == work.componentNumber &&
                                  state.PhysicalFieldNumber() == work.physicalFieldNumber,
                              "equation workspace schema, row space or device differs");
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            DNDS_check_throw_info(state.PhysicalField(iField).Width() == work.fields[iField].boundaryValues.Layout().Width(),
                                  "equation workspace physical field width differs");
    }
    bool FiniteVolume::HasRoeJacobianKernel(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        // Roe assembly is an operator-local flow5 leaf. It consumes the
        // canonical state/auxiliary views directly and does not require the
        // archived whole-flow reconstruction workspace.
        return !_data->equation.HasMixture() && !_data->equation.HasRANS() &&
               work.componentNumber == 5 && work.physicalFieldNumber == 1;
    }
    index FiniteVolume::PrepareFaceCharacteristicMatrices(const EquationDofs &state,
                                                          R::Array<real> &forwardMatrices,
                                                          R::Array<real> &inverseMatrices,
                                                          OptionalRef<const AuxiliaryValues> auxiliary, real time) const
    {
        state.RequireSchema(_data->equation.Schema());
        // Cached mixture properties depend on composition as well as flow.
        // Retain all source fields while validating and consuming the cache.
        const auto means = state.ReadWithGhost();
        const auto flow = state.LocateBlock("flow");
        DNDS_check_throw_info(flow.componentNumber == 5, "characteristic flow block must contain five components");
        const auto &field = state.PhysicalField(flow.physicalField);
        std::optional<PerfectGas> gas;
        std::optional<MeanThermodynamicView> thermal;
        std::optional<R::Array<real>::ReadLease> thermalRead;
        std::optional<DerivedFields::ReadScope> derivedOwned, derivedGhost;
        if (_data->equation.HasMixture())
        {
            DNDS_check_throw_info(auxiliary, "mixture characteristics require prepared cell thermodynamics");
            ValidateAuxValues(state, *auxiliary, time);
            if (auxiliary->meanThermodynamics)
            {
                thermalRead.emplace(auxiliary->meanThermodynamics->Read());
                thermal.emplace(thermalRead->View<R::UniformTableRows>());
            }
            else
            {
                DNDS_check_throw_info(auxiliary->derivedOwned && auxiliary->derivedGhost,
                                      "mixture characteristic derived properties require both row parts");
                derivedOwned.emplace(auxiliary->derivedOwned->ReadCurrent(state, _data->equation));
                derivedGhost.emplace(auxiliary->derivedGhost->ReadCurrent(state, _data->equation));
                thermal.emplace(MeanThermodynamicView::Derived(*derivedOwned, OptionalRef<const DerivedFields::ReadScope>(*derivedGhost)));
            }
            DNDS_check_throw_info(thermal->HasCharacteristics() && thermal->RowCount() == Metrics().NumCellProc(),
                                  "mixture characteristic p/a/H/beta projection missing or wrong-sized");
        }
        else
            gas.emplace(_data->equation.Gas());
        index fallbacks = 0;
        {
            const auto flowViews = means.Block("flow");
            const RHSFieldView u{flowViews.owned.parent, flowViews.ghost.parent};
            DNDS_check_throw_info(flow.firstComponent >= 0 && flow.firstComponent + 5 <= field.Width(),
                                  "flow characteristic range differs from its physical field");
            auto forward = forwardMatrices.Write(), inverse = inverseMatrices.Write();
            const auto stencilRead = _data->stencil.Read();
            R::ValidateViewDevices(R::Device::Host(), u, forward.View<R::TableRows>(), inverse.View<R::TableRows>());
            if (thermal)
                thermal->values.ValidateDevice(R::Device::Host());
            fallbacks = RequireNumericalSuccess(CPU::PrepareFaceCharacteristicMatrices(stencilRead.View(), u, flow.firstComponent,
                                                                                       forward.View<R::TableRows>(), inverse.View<R::TableRows>(), gas, thermal),
                                                "PrepareFaceCharacteristicMatrices");
        }
        return fallbacks;
    }
    index FiniteVolume::PrepareFaceCharacteristics(const EquationDofs &state,
                                                   CFV::FaceCharacteristicTransform &transform,
                                                   OptionalRef<const AuxiliaryValues> auxiliary, real time) const
    {
        DNDS_check_throw_info(transform.first == state.LocateBlock("flow").firstComponent && transform.count == 5,
                              "flow characteristic range differs from its physical field");
        const auto fallbacks = PrepareFaceCharacteristicMatrices(
            state, transform.forward, transform.inverse, auxiliary, time);
        DNDS_check_throw_info(_data->reconstruction,
                              "modal characteristic transform requires variational reconstruction");
        _data->reconstruction->FinalizeFaceTransform(transform, _data->settings.numerics.characteristicDiagnostics);
        return fallbacks;
    }
    index FiniteVolume::PrepareFaceCharacteristics(
        const EquationDofs &state, CFV::GradientFaceCharacteristicTransform &transform,
        OptionalRef<const AuxiliaryValues> auxiliary, real time) const
    {
        DNDS_check_throw_info(transform.first == state.LocateBlock("flow").firstComponent && transform.count == 5,
                              "flow gradient characteristic range differs from its physical field");
        const auto fallbacks = PrepareFaceCharacteristicMatrices(
            state, transform.forward, transform.inverse, auxiliary, time);
        _data->gradient.FinalizeFaceTransform(transform, _data->settings.numerics.characteristicDiagnostics);
        return fallbacks;
    }
    FiniteVolume::FinalizedReconstruction FiniteVolume::AcquireFinalizedReconstruction(
        const ReconstructionState &state) const
    {
        DNDS_check_throw_info(_data->reconstruction,
                              "direct P0/P1 has no modal reconstruction publication");
        DNDS_check_throw_info(state.owner == _data, "foreign equation reconstruction state");
        DNDS_check_throw_info(state.FinalizedCurrent(), "reconstruction has no current finalized publication");
        return {_data, state.WarmValues(), state.FinalizedValues(), state.distinctFinalized};
    }
    const EquationDofs &FiniteVolume::FluxReconstruction(const ReconstructionState &state, const Workspace &work) const
    {
        DNDS_check_throw_info(state.owner == _data && work.owner == _data, "foreign equation reconstruction state or workspace");
        DNDS_check_throw_info(_data->reconstruction,
                              "direct P0/P1 has no modal flux reconstruction");
        DNDS_check_throw_info(state.FinalizedCurrent(), "reconstruction has no current finalized publication");
        return state.FinalizedValues();
    }
    int FiniteVolume::LastReconstructionIterations(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.iterations;
    }
    real FiniteVolume::LastReconstructionUpdate(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.relativeUpdate;
    }
    void FiniteVolume::LimitReconstructionPP(const EquationDofs &state, ReconstructionState &reconstruction, Workspace &work) const
    {
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        Metrics().Communication()->LocalStage([&]
                                              {
            ValidateWorkspace(state, work);
            DNDS_check_throw_info(work.positivity, "equation reconstruction PP was not requested");
            DNDS_check_throw_info(reconstruction.owner == _data, "foreign equation reconstruction state");
            work.positivity->Apply(state, reconstruction.WarmValues(), reconstruction.MutableFinalized(), true);
            reconstruction.PublishFinalized(); });
    }
    std::pair<index, real> FiniteVolume::PositivityReconstructionStats(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        auto result = work.positivity
                          ? work.positivity->LocalStatistics()
                          : std::pair<index, real>{work.directPPLimited,
                                                   work.directPPMinimum};
        result.first = MPI::AllreduceValue<MPI::Reduction::Sum>(result.first, Metrics().Communicator());
        result.second = MPI::AllreduceValue<MPI::Reduction::Min>(result.second, Metrics().Communicator());
        return result;
    }
}
