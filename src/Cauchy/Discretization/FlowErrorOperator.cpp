#include "Cauchy/Discretization/FlowErrorOperator.hpp"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/State/EquationArrayOperands.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/Core/OMP.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct FlowErrorOperator::Data
    {
        FiniteVolume plan;
        R::ExecutionContext context;
        std::vector<NumericalResult<CPU::FlowErrorStatistics>> workers;
        R::Array<index> offsets;
        R::Array<real> points;
#ifdef DNDS_USE_CUDA
        R::Array<CUDA::FlowErrorReduction> partial, result;
        R::Array<unsigned int> completed;
#endif
        Data(FiniteVolume spatial, R::ExecutionContext where) : plan(std::move(spatial)), context(where)
        {
            DNDS_check_throw_info(plan.Device() == context.device && !plan.Equation().HasMixture(), "flow error operator requires matching placement and perfect gas");
            if (context.device == R::Device::Host())
                return;
#ifdef DNDS_USE_CUDA
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "unsupported flow error backend");
            const auto geometry = plan.Metrics().AcquireRead(context);
            const auto meshRead = plan.AcquireMeshRead(Geom::MeshReadRequirements::CellFlux());
            const auto mesh = meshRead.HostView();
            const index rows = plan.Metrics().NumCell();
            constexpr int order = Geom::Elem::INT_ORDER_MAX;
            std::vector<index> starts(size_t(rows + 1), 0);
            std::vector<int> prepared;
            for (index iCell = 0; iCell < rows; ++iCell)
            {
                const auto element = mesh.GetCellElement(iCell);
                const Geom::Elem::Quadrature quadrature(element, order);
                starts[iCell + 1] = starts[iCell] + quadrature.GetNumPoints();
                if (std::find(prepared.begin(), prepared.end(), int(element.type)) == prepared.end())
                {
                    for (int iPoint = 0; iPoint < quadrature.GetNumPoints(); ++iPoint)
                        (void)Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iPoint);
                    prepared.push_back(int(element.type));
                }
            }
            R::Array<index> hostOffsets(R::RowLayout::Dense(rows + 1, 1));
            R::Array<real> hostPoints(R::RowLayout::Dense(starts.back(), 4));
            {
                auto offsetWrite = hostOffsets.Write();
                for (index iCell = 0; iCell <= rows; ++iCell)
                    offsetWrite.View()(iCell, 0) = starts[iCell];
                auto pointWrite = hostPoints.Write();
                const auto output = pointWrite.View<R::FixedRows<4>>();
                OMP::ForSetupRows(0, rows, [&](index iCell)
                                  {
                    Geom::tSmallCoords coords;
                    mesh.GetCoordsOnCell(iCell, coords);
                    const auto element = mesh.GetCellElement(iCell);
                    const Geom::Elem::Quadrature quadrature(element, order);
                    for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                    {
                        const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ);
                        const Geom::tPoint point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                        for (int d = 0; d < 3; ++d) output(starts[iCell] + iQ, d) = point[d];
                        output(starts[iCell] + iQ, 3) = Geom::Elem::CellJacobianDet(mesh.Dimension(), coords, shape) * quadrature.GetWeight(iQ);
                    } });
            }
            offsets = hostOffsets.To(context.device, context);
            points = hostPoints.To(context.device, context);
            partial = R::Array<CUDA::FlowErrorReduction>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(rows), 1), context.device);
            result = R::Array<CUDA::FlowErrorReduction>(R::RowLayout::Dense(1, 1), context.device);
            completed = R::Array<unsigned int>(R::RowLayout::Dense(1, 1), context.device);
            offsets.SetName("flowError/quadratureOffsets", DNDS_ALLOCATION_SITE());
            points.SetName("flowError/quadraturePoints", DNDS_ALLOCATION_SITE());
            partial.SetName("flowError/partials", DNDS_ALLOCATION_SITE());
            result.SetName("flowError/result", DNDS_ALLOCATION_SITE());
            completed.SetName("flowError/completedBlocks", DNDS_ALLOCATION_SITE());
#else
            DNDS_check_throw_info(false, "CUDA flow error is not compiled");
#endif
        }
    };
    FlowErrorOperator::FlowErrorOperator(FiniteVolume spatial, R::ExecutionContext context)
        : _data(std::make_shared<Data>(std::move(spatial), context)) {}
    std::array<FlowState, 3> FlowErrorOperator::Evaluate(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                         const EquationDofs &state, const FlowProfileSettings &profile, const FlowState &primitive, real time)
    {
        const auto retained = _data;
        std::array<FlowState, 3> errors{};
        real volume = 0;
        retained->plan.Metrics().Communication()->LocalStage([&]
                                                             {
        state.RequireSchema(retained->plan.Equation().Schema());
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == retained->context.device &&
            (retained->context.device == R::Device::Host() || stream.Context().stream == retained->context.stream) && state.Device() == retained->context.device &&
            state.Space().Same(retained->plan.Metrics().Space()) && std::isfinite(time), "flow error stream or state differs");
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(retained->plan.Metrics().AcquireRead(stream.Context()));
        const auto gas = retained->plan.Equation().Gas();
        const auto frame = retained->plan.Motion().At(time);
        if (retained->context.device == R::Device::Host())
        {
            const auto mesh = std::make_shared<Geom::MeshReadAccess>(retained->plan.AcquireMeshRead(Geom::MeshReadRequirements::CellFlux()));
            retained->workers.resize(OMP::DistributedThreads(), NumericalResult<CPU::FlowErrorStatistics>{CPU::FlowErrorStatistics{}});
            const auto result = std::make_shared<NumericalResult<CPU::FlowErrorStatistics>>(CPU::FlowErrorStatistics{});
            EquationArrayOperands::Inputs inputs;
            const auto mean = EquationArrayOperands::AddInput(state, inputs);
            index flowBlock = -1;
            for (index i = 0; i < state.BlockNumber(); ++i) if (state.Schema().blocks[i].name == "flow") flowBlock = i;
            DNDS_check_throw_info(flowBlock >= 0, "flow error needs a flow block");
            R::SubmitArrayBatch(engine, stream, [retained, geometry, mesh, mean, flowBlock, result, profile, primitive, gas, frame, time](auto, auto input, auto)
            {
                const EquationArrayOperands::Binding::ReadBlocks blocks{mean, input};
                *result = CPU::MeasureFlowError(mesh->HostView(), geometry->HostView(), blocks.Block(flowBlock).owned,
                    profile, primitive, gas, time, frame, {retained->workers.data(), index(retained->workers.size())});
            }, std::move(inputs), EquationArrayOperands::Outputs{});
            engine.Synchronize();
            const auto value = RequireNumericalSuccess(*result, "flow error oracle");
            errors = value.errors; volume = value.volume;
        }
        else
        {
#ifdef DNDS_USE_CUDA
            const auto parameters = PrepareFlowProfile(profile);
            R::SubmitArrays(engine, stream, [retained, geometry, parameters, primitive, gas, frame = frame.KernelParameters(), time](auto context, auto means, auto offsets, auto points, auto partial, auto result, auto completed)
            {
                CUDA::MeasureFlowError(context, {offsets, points, geometry->View().cells}, means,
                    parameters, primitive, gas, frame, time, partial, result, completed);
            }, R::ReadAs<R::TableRows>(state.CoupledField().Owned()), R::ReadAs<R::FixedRows<1>>(retained->offsets),
                R::ReadAs<R::FixedRows<4>>(retained->points), R::WriteAs<R::FixedRows<1>>(retained->partial),
                R::WriteAs<R::FixedRows<1>>(retained->result), R::WriteAs<R::FixedRows<1>>(retained->completed));
            engine.Synchronize();
            const auto host = retained->result.To(R::Device::Host(), stream.Context());
            const auto read = host.Read();
            const auto value = read.View()(0, 0);
            RequireNumericalSuccess(value.status, "flow error oracle");
            errors = value.errors; volume = value.volume;
#else
            DNDS_check_throw_info(false, "CUDA flow error is not compiled");
#endif
        } });
        const auto comm = retained->plan.Metrics().Communicator();
        volume = MPI::AllreduceValue<MPI::Reduction::Sum>(volume, comm);
        for (int norm = 0; norm < 3; ++norm)
            if (norm == 2)
                MPI::AllreduceInPlace<MPI::Reduction::Max>(errors[norm], comm);
            else
                MPI::AllreduceInPlace<MPI::Reduction::Sum>(errors[norm], comm);
        for (int j = 0; j < 5; ++j)
        {
            errors[0][j] /= volume;
            errors[1][j] = std::sqrt(errors[1][j] / volume);
        }
        return errors;
    }
}
