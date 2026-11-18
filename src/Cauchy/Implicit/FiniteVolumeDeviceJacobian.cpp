#include "Cauchy/Implicit/FiniteVolumeDeviceJacobian.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeSourceKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    FiniteVolume::Workspace::DeviceJacobianData::DeviceJacobianData(std::shared_ptr<const Data> data, index width, R::ExecutionContext context)
        : owner(std::move(data)), metric(owner->metrics.AcquireRead({context.device, nullptr})), stencil(owner->stencil.Read({context.device, nullptr})),
          components(width)
    {
#ifdef DNDS_USE_CUDA
        const auto device = context.device;
        const index owned = owner->metrics.NumCell(), all = owner->metrics.NumCellProc();
        if (!owner->equation.IsACM())
        {
            physics = owner->devicePhysics.Read({context.device, nullptr});
            conditions = owner->deviceBoundaryConditions.Read({context.device, nullptr});
            if (owner->settings.execution.kernels.stateInputLayout == "aosoa32")
            {
                inputOwned.emplace(R::RowLayout::Dense(owned, components), device, "Cauchy/Jacobian/tiledInput");
                inputGhost.emplace(R::RowLayout::Dense(all - owned, components), device, "Cauchy/Jacobian/tiledGhost");
            }
            // Symbolic setup borrows the authoritative CPU topology in the same
            // local numbering. Only sparse integer incidence routes are uploaded;
            // centered geometry/periodic maps remain in the shared device stencil.
            const auto symbolic = owner->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            geometry = std::make_shared<JacobianGeometryStorage>(
                stencil, owner->jacobianColumns->CopyColumns(), symbolic.HostView().CellFaceIncidences(),
                !owner->hasRotatedFrames, context);
            publicationGeometry = std::make_shared<FrozenJacobianOperator::Geometry>(owner->jacobianColumns, geometry, context);
            geometryRead.emplace(geometry->Read({device, nullptr}));
        }
        if (owner->equation.HasRANS())
        {
            DNDS_check_throw_info(owner->wallDistance, "device RANS Jacobian requires prepared wall distances");
            walls.emplace(owner->wallDistance->AcquireRead({device, nullptr}));
        }
        const auto status = [&](index rows, const char *name)
        {
            R::Array<NumericalStatus> array(R::RowLayout::Dense(rows, 1), device);
            array.SetName(name, DNDS_ALLOCATION_SITE());
            return array;
        };
        rowsStatus = status(owned, "Cauchy/Jacobian/rowStatus");
        if (!owner->equation.IsACM())
        {
            meansStatus = status(all, "Cauchy/Jacobian/meanStatus");
            assemblyStatus = status(owned, "Cauchy/Jacobian/assemblyStatus");
        }
        partial = status(CUDA::StateReductionScratchRows(all + 1), "Cauchy/Jacobian/statusPartials");
        result = status(1, "Cauchy/Jacobian/statusResult");
#else
        (void)context;
        DNDS_check_throw_info(false, "CUDA Jacobian preparation is not compiled");
#endif
    }
    std::shared_ptr<FiniteVolume::Workspace::DeviceJacobianData::Slot>
    FiniteVolume::Workspace::DeviceJacobianData::Acquire(bool roe)
    {
        for (const auto &slot : slots)
            if (slot->roe == roe && slot->publication.expired())
                return slot;
        auto slot = std::make_shared<Slot>();
        slot->roe = roe;
        const auto device = owner->metrics.Device();
        const index all = owner->metrics.NumCellProc();
        const auto table = [&](index rows, index width, const char *name)
        {
            R::Array<real> result(R::RowLayout::Dense(rows, width), device);
            result.SetName(std::string("Cauchy/Jacobian/slot") + std::to_string(slots.size()) + "/" + name, DNDS_ALLOCATION_SITE());
            return result;
        };
        slot->values = {table(all, components, "means"), table(all, 1, "pressure"), table(all, components, "pressureDerivative"),
                        table(owner->stencil.stencilFaces.RowCount(), 2, "radius"),
                        table(roe ? owner->stencil.IncidenceNumber() : 0, 3, "magnitudes")};
        if (inputOwned)
        {
            slot->values.means.Reset(R::RowLayout::TiledColMajor(all, components, 32), device);
            slot->values.means.SetName("Cauchy/Jacobian/slot" + std::to_string(slots.size()) + "/tiledMeans", DNDS_ALLOCATION_SITE());
        }
        slots.push_back(slot);
        return slot;
    }
    void FiniteVolume::Workspace::DeviceJacobianData::AppendStatus(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                                   const R::Array<NumericalStatus> &rows)
    {
#ifdef DNDS_USE_CUDA
        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto rows, auto partial, auto summary)
                        { CUDA::ReduceNumericalStatus(context, rows, summary.ConstView(), partial, summary); }, R::ReadAs<R::FixedRows<1>>(rows), R::WriteAs<R::FixedRows<1>>(partial), R::ReadWriteAs<R::FixedRows<1>>(result));
#endif
    }
    void FiniteVolume::Workspace::DeviceJacobianData::SubmitSource(Workspace &work, R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                                   const EquationDofs &state, R::Array<real> &diagonal, bool fullBlock, real scale)
    {
#ifdef DNDS_USE_CUDA
        if (!owner->equation.HasRANS())
            return;
        const bool polynomial = owner->settings.numerics.reconstruction == "variational";
        const int dim = owner->metrics.Dimension(), n = int(components);
        const auto &gradient = polynomial ? work.fields.front().gradient.value().Owned() : work.directGradients.front().Owned();
        // The enclosing synchronous preparation retains this shared object
        // through its final drain. Numerical scopes are also captured explicitly.
        const auto pin = work.deviceJacobian;
        R::SubmitArrays(engine, stream, [pin, n, dim, fullBlock, scale](R::ExecutionContext context, auto means, auto gradient, auto offsets, auto weights, auto lengths, auto cells, auto output, auto status)
                        {
            CUDA::SourceRequest request;
            if (pin->owner->equation.IsACM())
                request.acm = R::MetadataArrayView(pin->owner->deviceACMPhysics, context.device);
            else
                request.parameters = R::MetadataArrayView(pin->owner->devicePhysics, context.device);
            request.means = means;
            for (int d = 0; d < dim; ++d) request.gradients[d] = {gradient, dim, d};
            request.offsets = offsets; request.weights = weights; request.lengths = lengths;
            const auto walls = pin->walls->View();
            const auto mean = walls.cellMean;
            const auto points = walls.cells;
            request.wallMean = {mean.data, mean.offsets, mean.rows, mean.width, mean.stride, mean.sizes, mean.kind, mean.device};
            request.wallPoints = {points.data, points.offsets, points.rows, points.width, points.stride, points.sizes, points.kind, points.device};
            request.cells = pin->metric.View().cells; request.output = output; request.status = status;
            request.components = n; request.dimension = dim; request.fullBlock = fullBlock; request.scale = scale;
            request.evaluation = CUDA::SourceEvaluation::MeanDestruction;
            CUDA::AddRANSSources(context, request, cells); }, R::ReadAs<R::TableRows>(state.CoupledField().Owned()), R::ReadAs<R::TableRows>(gradient), R::ReadAs<R::FixedRows<1>>(work.sourceOffsets), R::ReadAs<R::FixedRows<1>>(work.sourceWeights), R::ReadAs<R::FixedRows<1>>(owner->ransCellLength), R::ReadAs<R::FixedRows<1>>(work.DeviceCellSelection(CFV::ReconstructionRegion::All)), R::ReadWriteAs<R::UniformTableRows>(diagonal), R::WriteAs<R::FixedRows<1>>(rowsStatus));
        AppendStatus(engine, stream, rowsStatus);
#else
        (void)work;
        (void)engine;
        (void)stream;
        (void)state;
        (void)diagonal;
        (void)fullBlock;
        (void)scale;
        DNDS_check_throw_info(false, "CUDA source Jacobian is not compiled");
#endif
    }
}
