#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Workspace::BeginNumericalStatus(R::EagerEngine &engine, const R::ExecutionStream &stream, bool withRHS)
    {
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device(),
                              "numerical status segment stream or placement differs");
        if (owner->metrics.Device() == R::Device::Host())
            return;
        ResetPreparationStatus(engine, stream);
        if (withRHS)
            ResetDeviceFluxStatus(engine, stream);
    }
    void FiniteVolume::Workspace::CheckNumericalStatus(bool withRHS)
    {
        if (owner->metrics.Device() == R::Device::Host())
            return;
        CheckPreparationStatus();
        if (withRHS)
            CheckDeviceFluxStatus();
        for (const auto &field : fields)
        {
            if (field.reconstruction)
                owner->reconstruction->CheckStatus(*field.reconstruction);
            if (field.gradientExecution)
                owner->gradient.CheckStatus(*field.gradientExecution);
        }
    }
    void FiniteVolume::SubmitBoundaryValues(
        Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
        const EquationDofs &state, OptionalRef<const EquationDofs> coefficients,
        Runtime::Array<real> &output, Runtime::Array<NumericalStatus> &status,
        BoundarySubmission options, OptionalRef<const Runtime::Array<real>> incoming) const
    {
#ifdef DNDS_USE_CUDA
        namespace R = Runtime;
        DNDS_check_throw_info(Device().space == R::MemorySpace::CUDA && engine.Owns(stream) &&
                                  stream.Context().device == Device() && state.Device() == Device(),
                              "queued boundary backend/stream differs from its spatial plan");
        state.RequireSchema(_data->equation.Schema());
        DNDS_check_throw_info(state.PhysicalFieldNumber() == 1 && state.Space().Same(Metrics().Space()) && std::isfinite(options.time),
                              "queued boundary needs combined state in its prepared row space");
        const bool mapped = options.target != BoundaryTarget::Quadrature || !_data->stencil.quadratureBoundaryValues;
        const bool primitive = options.target == BoundaryTarget::PrimitiveGradient;
        const index rows = mapped ? _data->boundaryFaces.RowCount() : _data->BoundaryValueRows();
        DNDS_check_throw_info(output.RowCount() == rows && output.Layout().Kind() == R::RowLayoutKind::Table &&
                                  output.Layout().Width() == (primitive ? 5 : state.ComponentNumber()) && status.RowCount() == rows &&
                                  status.Layout().Kind() == R::RowLayoutKind::Table && status.Layout().Width() == 1 &&
                                  output.GetDevice() == Device() && status.GetDevice() == Device(),
                              "queued boundary output/status shape or placement differs");
        DNDS_check_throw_info(!coefficients || (!primitive && _data->reconstruction),
                              "polynomial boundary values require a reconstruction plan");
        const auto &mean = state.CoupledField().Owned();
        const auto &rec = coefficients ? coefficients->CoupledField().Owned() : _data->deviceEmpty;
        const auto &packet = incoming ? *incoming : _data->deviceEmpty;
        DNDS_check_throw_info(!(_data->hasLocalInterfaceRows && !incoming),
                              "boundary producer requires its selected center/quadrature interface section");
        if (incoming)
            DNDS_check_throw_info(packet.RowCount() == rows && packet.Layout().Width() == state.ComponentNumber() &&
                                      packet.GetDevice() == Device() && !packet.Layout().IsVariable(),
                                  "boundary interface section shape or placement differs from selected location");
        if (coefficients)
            DNDS_check_throw_info(coefficients->PhysicalFieldNumber() == 1 && coefficients->Space().Same(state.Space()) &&
                                      rec.RowCount() == mean.RowCount() && rec.GetDevice() == Device() &&
                                      rec.Layout().Width() == state.ComponentNumber() * _data->reconstruction->NRecDOF(),
                                  "queued boundary coefficients differ from its combined polynomial layout");
        for (const auto *input : {&mean, &rec, &packet})
            DNDS_check_throw_info(!output.Storage().Overlaps(input->Storage()) && !status.Storage().Overlaps(input->Storage()),
                                  "queued boundary output/status aliases an input");
        DNDS_check_throw_info(!output.Storage().Overlaps(status.Storage()), "queued boundary output aliases status");

        // Retain the geometry generation and immutable metadata owner. Array
        // payloads below all use scheduled reads, including stencil geometry.
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            FiniteVolumeStencilStorage::ReadAccess stencil;
            std::optional<CFV::ReconstructionResidentRead> basis;
            bool hasWall;
            Inputs(std::shared_ptr<const Data> data, bool polynomial, bool primitive)
                : owner(std::move(data)), stencil(owner->stencil.Read({owner->metrics.Device(), nullptr})),
                  hasWall(!primitive && owner->equation.HasRANS() && owner->equation.RANS().ComponentNumber() == 2)
            {
                if (polynomial)
                    basis.emplace(owner->reconstruction->AcquireResidentRead());
                if (hasWall)
                    DNDS_check_throw_info(owner->wallDistance, "two-equation boundary requires wall distance");
            }
        };
        const auto pin = std::make_shared<Inputs>(_data, bool(coefficients), primitive);
        const auto frame = _data->motion.At(options.time).KernelParameters();
        const int components = int(state.ComponentNumber());
        const bool hasIncoming = bool(incoming);
        R::SubmitArrays(engine, stream, [pin, options, frame, components, mapped, primitive, hasIncoming](R::ExecutionContext context, auto mean, auto rec, auto incoming, auto target, auto status, auto faces, auto wall)
                        {
                CUDA::BoundaryTraceRequest request;
                if (pin->owner->equation.IsACM())
                {
                    request.acm = R::MetadataArrayView(pin->owner->deviceACMPhysics, context.device);
                    request.acmBC = R::MetadataArrayView(pin->owner->deviceACMBoundaryConditions,context.device);
                }
                else
                {
                    request.parameters = R::MetadataArrayView(pin->owner->deviceBoundaryPolicy, context.device);
                    request.conditions = R::MetadataArrayView(pin->owner->deviceBoundaryConditions, context.device);
                }
                request.stencil = pin->stencil.KernelDescriptor();
                request.faces = faces;
                request.means = mean; request.coefficients = rec; request.incoming = incoming;
                request.output = target; request.status = status;
                request.time = options.time; request.frame = frame;
                request.components = components; request.centers = mapped;
                request.exterior = options.exterior; request.repairRANS = options.repairRANS;
                request.forcePointFallback = options.forcePointFallback;
                request.hasIncoming = hasIncoming; request.polynomial = bool(pin->basis);
                if (pin->basis)
                {
                    request.basis = pin->basis->PointBasisView();
                    request.modes = request.basis.polynomial.size - 1;
                }
                if (pin->hasWall)
                {
                    request.wallMean = {wall.data, wall.offsets, wall.rows, wall.width, wall.stride, wall.sizes, wall.kind, wall.device};
                }
                if (primitive) CUDA::FillPrimitiveBoundary(context, request);
                else CUDA::FillBoundaryTraces(context, request); }, R::ReadAs<R::TableRows>(mean), R::ReadAs<R::TableRows>(rec), R::ReadAs<R::TableRows>(packet), R::WriteAs<R::TableRows>(output), R::WriteAs<R::FixedRows<1>>(status), R::ReadAs<R::FixedRows<1>>(_data->boundaryFaces), R::ReadAs<R::TableRows>(pin->hasWall ? _data->wallDistance->CellMeans() : _data->deviceEmpty));
#else
        DNDS_check_throw_info(false, "CUDA boundary execution is not built");
#endif
    }
    void FiniteVolume::Workspace::BuildPreparationStatus()
    {
        if (owner->metrics.Device() == Runtime::Device::Host())
            return;
#ifdef DNDS_USE_CUDA
        const auto device = owner->metrics.Device();
        const index quadrature = owner->BoundaryValueRows();
        const bool centered = primitiveBoundary || std::any_of(fields.begin(), fields.end(), [](const auto &field)
                                                               { return field.gradientBoundary && field.gradientBoundary->SharedStorage() != field.boundaryValues.SharedStorage(); });
        const index primitive = centered ? owner->boundaryFaces.RowCount() : 0;
        const bool gradientPP = nativeDirectGradient && owner->settings.numerics.positivity;
        const index owned = primitiveGradient || owner->equation.HasRANS() ? owner->metrics.NumCell() : 0;
        const index ghost = primitiveGradient ? owner->metrics.NumCellProc() - owner->metrics.NumCell() : 0;
        preparationStatus.emplace(PreparationStatusStorage{
            R::Array<NumericalStatus>(R::RowLayout::Dense(quadrature, 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(primitive, 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(owned, 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(ghost, 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(std::max({quadrature, primitive, owned, ghost}) + 1), 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), device)});
        preparationStatus->quadrature.SetName("Cauchy/preparationStatus", DNDS_ALLOCATION_SITE());
        preparationStatus->primitive.SetName("Cauchy/centerBoundaryStatus", DNDS_ALLOCATION_SITE());
        preparationStatus->partial.SetName("Cauchy/preparationStatusPartials", DNDS_ALLOCATION_SITE());
        preparationStatus->result.SetName("Cauchy/preparationStatusResult", DNDS_ALLOCATION_SITE());
        preparationStatus->meanOwned.SetName("Cauchy/preparationOwnedStatus", DNDS_ALLOCATION_SITE());
        preparationStatus->meanGhost.SetName("Cauchy/preparationGhostStatus", DNDS_ALLOCATION_SITE());
#else
        DNDS_check_throw_info(false, "CUDA boundary execution is not built");
#endif
    }
    void FiniteVolume::Workspace::ResetPreparationStatus(R::EagerEngine &engine, const R::ExecutionStream &stream)
    {
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(preparationStatus && !preparationStatus->pending, "preparation status workspace is unavailable or still pending");
        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto result)
                        { CUDA::ClearNumericalStatus(context, result); }, R::WriteAs<R::FixedRows<1>>(preparationStatus->result));
        preparationStatus->pending = true;
#else
        DNDS_check_throw_info(false, "CUDA boundary execution is not built");
#endif
    }
    void FiniteVolume::Workspace::CheckPreparationStatus()
    {
        DNDS_check_throw_info(preparationStatus && preparationStatus->pending && preparationStatus->result.Storage().IsIdle(),
                              "preparation status requires a completed numerical segment");
        const auto result = preparationStatus->result.To(R::Device::Host());
        preparationStatus->pending = false;
        RequireNumericalSuccess(result.Read().View()(0, 0), "queued spatial preparation");
    }
    void FiniteVolume::Workspace::AppendPreparationStatus(R::EagerEngine &engine, const R::ExecutionStream &stream, const R::Array<NumericalStatus> &rows)
    {
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(preparationStatus && preparationStatus->pending, "preparation status segment is not initialized");
        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto rows, auto partial, auto result)
                        { CUDA::ReduceNumericalStatus(context, rows, result.ConstView(), partial, result); }, R::ReadAs<R::FixedRows<1>>(rows), R::WriteAs<R::FixedRows<1>>(preparationStatus->partial), R::ReadWriteAs<R::FixedRows<1>>(preparationStatus->result));
#else
        DNDS_check_throw_info(false, "CUDA preparation status is not built");
#endif
    }
    void FiniteVolume::Workspace::SubmitDeviceBoundary(
        R::EagerEngine &engine, const R::ExecutionStream &stream,
        const EquationDofs &state, bool polynomial, real time,
        OptionalRef<const R::Array<real>> incoming,
        OptionalRef<const R::Array<real>> incomingCenters)
    {
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(preparationStatus && preparationStatus->pending && fields.size() == 1,
                              "queued boundary needs a combined device workspace and an initialized status segment");
        const FiniteVolume plan(owner);
        const bool mirrored = owner->settings.numerics.reconstruction == "variational" &&
                              owner->settings.numerics.vrBoundaryValue == "physicalMirrored";
        auto &field = fields.front();
        const auto coefficients = polynomial ? OptionalRef<const EquationDofs>(Coefficients()) : OptionalRef<const EquationDofs>{};
        plan.SubmitBoundaryValues(engine, stream, state, coefficients,
                                  field.boundaryValues, preparationStatus->quadrature,
                                  {BoundaryTarget::Quadrature, mirrored, true, time}, incoming);
        AppendPreparationStatus(engine, stream, preparationStatus->quadrature);
        if (field.gradientBoundary && field.gradientBoundary->SharedStorage() != field.boundaryValues.SharedStorage())
        {
            plan.SubmitBoundaryValues(engine, stream, state, coefficients,
                                      *field.gradientBoundary, preparationStatus->primitive,
                                      {BoundaryTarget::Gradient, false, true, time}, incomingCenters);
            AppendPreparationStatus(engine, stream, preparationStatus->primitive);
        }
        if (primitiveBoundary)
        {
            plan.SubmitBoundaryValues(engine, stream, state, {}, *primitiveBoundary, preparationStatus->primitive,
                                      {BoundaryTarget::PrimitiveGradient, false, false, time}, incomingCenters);
            AppendPreparationStatus(engine, stream, preparationStatus->primitive);
        }
#else
        DNDS_check_throw_info(false, "CUDA boundary execution is not built");
#endif
    }
}
