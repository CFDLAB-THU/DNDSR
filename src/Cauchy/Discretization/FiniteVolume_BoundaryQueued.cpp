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
        const bool mapped = options.target != BoundaryTarget::Quadrature;
        const bool primitive = options.target == BoundaryTarget::PrimitiveGradient;
        const index rows = mapped ? _data->gradientBoundaryRows.RowCount() : _data->boundaryTraceGeometry.RowCount();
        DNDS_check_throw_info(output.RowCount() == rows && output.Layout().Kind() == R::RowLayoutKind::Table &&
                                  output.Layout().Width() == (primitive ? 5 : state.ComponentNumber()) && status.RowCount() == rows &&
                                  status.Layout().Kind() == R::RowLayoutKind::Table && status.Layout().Width() == 1 &&
                                  output.GetDevice() == Device() && status.GetDevice() == Device(),
                              "queued boundary output/status shape or placement differs");
        DNDS_check_throw_info(!coefficients || (!mapped && _data->reconstruction),
                              "polynomial boundary values require the quadrature reconstruction target");
        const auto &mean = state.CoupledField().Owned();
        const auto &rec = coefficients ? coefficients->CoupledField().Owned() : _data->deviceEmpty;
        const auto &packet = incoming ? *incoming : _data->deviceEmpty;
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
            Geom::MeshReadAccess mesh;
            std::optional<CFV::ReconstructionResidentRead> basis;
            bool hasWall;
            Inputs(std::shared_ptr<const Data> data, bool polynomial, bool primitive)
                : owner(std::move(data)), mesh(owner->metrics.AcquireMeshRead(bool(owner->wallDistance))),
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
        R::SubmitArrays(engine, stream, [pin, options, frame, components, mapped, primitive, hasIncoming](R::ExecutionContext context, auto mean, auto rec, auto incoming, auto target, auto status, auto points, auto rows, auto wall)
                        {
                CUDA::BoundaryTraceRequest request;
                request.parameters = R::MetadataArrayView(pin->owner->deviceBoundaryPolicy, context.device);
                request.conditions = R::MetadataArrayView(pin->owner->deviceBoundaryConditions, context.device);
                request.points = points;
                request.sourceRows = rows;
                request.means = mean; request.coefficients = rec; request.incoming = incoming;
                request.output = target; request.status = status;
                request.time = options.time; request.frame = frame;
                request.components = components; request.useSourceRows = mapped;
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
                else CUDA::FillBoundaryTraces(context, request); }, R::ReadAs<R::TableRows>(mean), R::ReadAs<R::TableRows>(rec), R::ReadAs<R::TableRows>(packet), R::WriteAs<R::TableRows>(output), R::WriteAs<R::FixedRows<1>>(status), R::ReadAs<R::FixedRows<1>>(_data->boundaryTraceGeometry), R::ReadAs<R::FixedRows<1>>(_data->gradientBoundaryRows), R::ReadAs<R::TableRows>(pin->hasWall ? _data->wallDistance->CellMeans() : _data->deviceEmpty));
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
        const index quadrature = owner->boundaryTraceGeometry.RowCount();
        const index primitive = primitiveBoundary ? primitiveBoundary->RowCount() : 0;
        const bool gradientPP = nativeDirectGradient && owner->settings.numerics.positivity;
        const index owned = primitiveGradient || gradientPP || owner->equation.HasRANS() ? owner->metrics.NumCell() : 0;
        const index ghost = primitiveGradient ? owner->metrics.NumCellProc() - owner->metrics.NumCell() : 0;
        preparationStatus.emplace(PreparationStatusStorage{
            R::Array<NumericalStatus>(R::RowLayout::Dense(quadrature, 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(primitive, 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(owned, 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(ghost, 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(std::max({quadrature, primitive, owned, ghost}) + 1), 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), device),
            R::Array<real>(R::RowLayout::Dense(gradientPP ? owned : 0, 1), device)});
        preparationStatus->quadrature.SetName("Cauchy/preparationStatus", DNDS_ALLOCATION_SITE());
        preparationStatus->primitive.SetName("Cauchy/primitiveBoundaryStatus", DNDS_ALLOCATION_SITE());
        preparationStatus->partial.SetName("Cauchy/preparationStatusPartials", DNDS_ALLOCATION_SITE());
        preparationStatus->result.SetName("Cauchy/preparationStatusResult", DNDS_ALLOCATION_SITE());
        preparationStatus->meanOwned.SetName("Cauchy/preparationOwnedStatus", DNDS_ALLOCATION_SITE());
        preparationStatus->meanGhost.SetName("Cauchy/preparationGhostStatus", DNDS_ALLOCATION_SITE());
        preparationStatus->gradientBeta.SetName("Cauchy/gradientBeta", DNDS_ALLOCATION_SITE());
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
        OptionalRef<const R::Array<real>> incoming)
    {
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(preparationStatus && preparationStatus->pending && fields.size() == 1,
                              "queued boundary needs a combined device workspace and an initialized status segment");
        const FiniteVolume plan(owner);
        const bool mirrored = owner->settings.numerics.reconstruction == "variational" &&
                              owner->settings.numerics.vrBoundaryValue == "physicalMirrored";
        auto &field = fields.front();
        const bool gradientBoundary = bool(field.gradientBoundary);
        const auto fill = [&](bool exterior)
        {
            plan.SubmitBoundaryValues(engine, stream, state,
                                      polynomial ? OptionalRef<const EquationDofs>(Coefficients()) : OptionalRef<const EquationDofs>{},
                                      field.boundaryValues, preparationStatus->quadrature,
                                      {BoundaryTarget::Quadrature, exterior, true, time}, incoming);
            AppendPreparationStatus(engine, stream, preparationStatus->quadrature);
        };
        fill(mirrored && !gradientBoundary);
        if (gradientBoundary)
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto source, auto output, auto rows)
                            { CUDA::CopyBoundaryRows(context, source, output, rows); }, R::ReadAs<R::TableRows>(field.boundaryValues), R::WriteAs<R::TableRows>(*field.gradientBoundary), R::ReadAs<R::FixedRows<1>>(owner->gradientBoundaryRows));
        if (mirrored && gradientBoundary)
            fill(true);
        if (primitiveBoundary)
        {
            plan.SubmitBoundaryValues(engine, stream, state, {}, *primitiveBoundary, preparationStatus->primitive,
                                      {BoundaryTarget::PrimitiveGradient, false, false, time}, incoming);
            AppendPreparationStatus(engine, stream, preparationStatus->primitive);
        }
#else
        DNDS_check_throw_info(false, "CUDA boundary execution is not built");
#endif
    }
}
