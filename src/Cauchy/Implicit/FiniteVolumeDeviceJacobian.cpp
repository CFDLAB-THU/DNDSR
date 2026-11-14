#include "Cauchy/Implicit/FiniteVolumeDeviceJacobian.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeSourceKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    FiniteVolume::Workspace::DeviceJacobianData::DeviceJacobianData(std::shared_ptr<const Data> data, index width, R::ExecutionContext context)
        : owner(std::move(data)), metric(owner->metrics.AcquireRead({context.device, nullptr})), stencil(owner->stencil.Read({context.device, nullptr})),
          physics(owner->devicePhysics.Read({context.device, nullptr})), conditions(owner->deviceBoundaryConditions.Read({context.device, nullptr})), components(width)
    {
#ifdef DNDS_USE_CUDA
        const auto device = context.device;
        const index owned = owner->metrics.NumCell(), all = owner->metrics.NumCellProc();
        if (owner->settings.execution.kernels.stateInputLayout == "aosoa32")
        {
            inputOwned.emplace(R::RowLayout::Dense(owned, components), device, "Cauchy/Jacobian/tiledInput");
            inputGhost.emplace(R::RowLayout::Dense(all - owned, components), device, "Cauchy/Jacobian/tiledGhost");
        }
        // Symbolic setup only. Do not download full CFV metrics or basis caches.
        const auto samples = owner->stencil.jacobianSamples.To(R::Device::Host(), context);
        const auto offsets = owner->stencil.jacobianOffsets.To(R::Device::Host(), context);
        const auto areas = owner->stencil.auxiliaryFaceArea.To(R::Device::Host(), context);
        const auto s = samples.Read();
        const auto o = offsets.Read();
        const auto a = areas.Read();
        FiniteVolumeStencilView host;
        host.jacobianSamples = s.View<R::FixedRows<1>>();
        host.jacobianOffsets = o.View<R::FixedRows<1>>();
        host.auxiliaryFaceArea = a.View<R::FixedRows<1>>();
        const JacobianGeometryStorage prepared(host, owner->jacobianColumns->CopyColumns());
        geometry = std::make_shared<JacobianGeometryStorage>(prepared.Placed(device, context));
        publicationGeometry = std::make_shared<FrozenJacobianOperator::Geometry>(owner->jacobianColumns, geometry, context);
        geometryRead.emplace(geometry->Read({device, nullptr}));
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
        meansStatus = status(all, "Cauchy/Jacobian/meanStatus");
        rowsStatus = status(owned, "Cauchy/Jacobian/rowStatus");
        assemblyStatus = status(owned, "Cauchy/Jacobian/assemblyStatus");
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
                        table(roe ? owner->stencil.jacobianSamples.RowCount() : 0, 3, "magnitudes")};
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
        const bool quadrature = owner->settings.implicit.sourceJacobianMode == "rhsQuadrature";
        const bool polynomial = owner->settings.numerics.reconstruction == "variational";
        DNDS_check_throw_info(!quadrature || polynomial, "quadrature source Jacobian requires variational reconstruction");
        const int dim = owner->metrics.Dimension(), n = int(components);
        if (!quadrature && polynomial)
        {
            DNDS_check_throw_info(work.sourceJacobianSampling && work.sourceJacobianGradients.size() == 1,
                                  "device center-gradient source sampler is missing");
            for (int d = 0; d < dim; ++d)
                work.sourceJacobianSampling->Evaluate(engine, stream, state.CoupledField(), work.FluxCoefficients().CoupledField(),
                                                      work.sourceJacobianGradients[0][d], 1 + d);
        }
        const auto &empty = owner->deviceEmpty;
        const auto &g0 = quadrature ? work.sourceGradients[0][0] : polynomial ? work.sourceJacobianGradients[0][0]
                                                                              : work.directGradients[0].Owned();
        const auto &g1 = quadrature ? work.sourceGradients[0][1] : polynomial ? work.sourceJacobianGradients[0][1]
                                                                              : work.directGradients[0].Owned();
        const auto &g2 = dim == 3 ? (quadrature ? work.sourceGradients[0][2] : polynomial ? work.sourceJacobianGradients[0][2]
                                                                                          : work.directGradients[0].Owned())
                                  : empty;
        // The enclosing synchronous preparation retains this shared object
        // through its final drain. Numerical scopes are also captured explicitly.
        const auto pin = work.deviceJacobian;
        R::SubmitArrays(engine, stream, [pin, n, dim, quadrature, polynomial, fullBlock, scale](R::ExecutionContext context, auto means, auto samples, auto g0, auto g1, auto g2, auto offsets, auto weights, auto lengths, auto cells, auto output, auto status)
                        {
            CUDA::SourceRequest request;
            request.parameters = {pin->physics.View<PerfectGasSpatialParameters>().data, nullptr, 1, 1, 1, nullptr, R::RowLayoutKind::Table, context.device};
            request.means = means; request.samples = samples;
            const std::array<CUDA::SourceInput, 3> gradients{g0, g1, g2};
            for (int d = 0; d < dim; ++d) request.gradients[d] = {gradients[d], polynomial ? 1 : dim, polynomial ? 0 : d};
            request.offsets = offsets; request.weights = weights; request.lengths = lengths;
            const auto walls = pin->walls->View();
            const auto mean = walls.cellMean;
            const auto points = walls.cells;
            request.wallMean = {mean.data, mean.offsets, mean.rows, mean.width, mean.stride, mean.sizes, mean.kind, mean.device};
            request.wallPoints = {points.data, points.offsets, points.rows, points.width, points.stride, points.sizes, points.kind, points.device};
            request.cells = pin->metric.View().cells; request.output = output; request.status = status;
            request.components = n; request.dimension = dim; request.fullBlock = fullBlock; request.scale = scale;
            request.evaluation = quadrature ? CUDA::SourceEvaluation::QuadratureDestruction : CUDA::SourceEvaluation::MeanDestruction;
            CUDA::AddRANSSources(context, request, cells); }, R::ReadAs<R::TableRows>(state.CoupledField().Owned()), R::ReadAs<R::TableRows>(quadrature ? work.sourceValues[0] : empty), R::ReadAs<R::TableRows>(g0), R::ReadAs<R::TableRows>(g1), R::ReadAs<R::TableRows>(g2), R::ReadAs<R::FixedRows<1>>(work.sourceOffsets), R::ReadAs<R::FixedRows<1>>(work.sourceWeights), R::ReadAs<R::FixedRows<1>>(owner->ransCellLength), R::ReadAs<R::FixedRows<1>>(work.DeviceCellSelection(CFV::ReconstructionRegion::All)), R::ReadWriteAs<R::TableRows>(diagonal), R::WriteAs<R::FixedRows<1>>(rowsStatus));
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
