#include "Cauchy/State/StateDiagnostics.hpp"
#include "Cauchy/State/EquationArrayOperands.hpp"
#include "Cauchy/State/CPU/StatePropertiesKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Physics/PointWorkerContexts.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/State/CUDA/StateDiagnosticsKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationArrayOperands;
    struct StateDiagnostics::Data
    {
        CFV::FiniteVolumePlan metrics;
        EquationSystem equation;
        R::ExecutionContext context;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
#ifdef DNDS_USE_CUDA
        R::Array<CUDA::StateDiagnosticSummary> partial, result;
        R::Array<unsigned int> completed;
#endif
        Data(CFV::FiniteVolumePlan geometry, EquationSystem physics, R::ExecutionContext where,
             std::optional<Physics::Chemistry::ChemicalWorkerPool> pool)
            : metrics(std::move(geometry)), equation(std::move(physics)), context(where), workers(std::move(pool))
        {
            DNDS_check_throw_info(metrics.Device() == context.device && bool(workers) == equation.HasMixture(), "diagnostic placement or physics worker mismatch");
            if (context.device == R::Device::Host())
                return;
#ifdef DNDS_USE_CUDA
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && !equation.HasMixture(), "CUDA diagnostics require perfect gas or constant-density ACM");
            partial = R::Array<CUDA::StateDiagnosticSummary>(R::RowLayout::Dense(CUDA::StateDiagnosticScratchRows(metrics.NumCell()), 1), context.device);
            result = R::Array<CUDA::StateDiagnosticSummary>(R::RowLayout::Dense(1, 1), context.device);
            completed = R::Array<unsigned int>(R::RowLayout::Dense(1, 1), context.device);
            partial.SetName("diagnostics/partials", DNDS_ALLOCATION_SITE());
            result.SetName("diagnostics/result", DNDS_ALLOCATION_SITE());
            completed.SetName("diagnostics/completedBlocks", DNDS_ALLOCATION_SITE());
#else
            DNDS_check_throw_info(false, "CUDA diagnostics are not compiled");
#endif
        }
    };
    StateDiagnostics::StateDiagnostics(CFV::FiniteVolumePlan geometry, EquationSystem equation, R::ExecutionContext context,
                                       std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
        : _data(std::make_shared<Data>(std::move(geometry), std::move(equation), context, std::move(workers))) {}

    std::array<real, 14> StateDiagnostics::Evaluate(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                    const EquationDofs &mean, const EquationDofs &residual) const
    {
        mean.RequireCompatible(residual);
        const auto retained = _data;
        mean.RequireSchema(retained->equation.Schema());
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == retained->context.device &&
                                  (retained->context.device == R::Device::Host() || stream.Context().stream == retained->context.stream) && mean.Device() == retained->context.device &&
                                  mean.Space().Same(retained->metrics.Space()),
                              "diagnostic stream, space or residency differs");
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(retained->metrics.AcquireRead(stream.Context()));
        Inputs inputs;
        const auto means = AddInput(mean, inputs), rhs = AddInput(residual, inputs);
        index flowBlock = -1;
        for (index iBlock = 0; iBlock < mean.BlockNumber(); ++iBlock)
            if (mean.Schema().blocks[iBlock].name == "flow")
                flowBlock = iBlock;
        DNDS_check_throw_info(flowBlock >= 0, "diagnostics require the flow block");
        if (retained->context.device == R::Device::Host())
        {
            struct Result
            {
                std::array<real, 14> values{};
                NumericalStatus status;
            };
            const auto result = std::make_shared<Result>();
            result->values[2] = result->values[3] = result->values[13] = std::numeric_limits<real>::max();
            R::SubmitArrayBatch(engine, stream, [retained, geometry, result, means, rhs, flowBlock](auto, auto input, auto)
                                {
                auto workers = MakeCPUPointWorkers<CPU::StatePropertiesWorker>(retained->equation, retained->workers, {});
                const Binding::ReadBlocks blocks{means, input};
                std::optional<EquationBlockReadView> species;
                for (index iBlock = 0; iBlock < means.state.BlockNumber(); ++iBlock)
                    if (means.state.Schema().blocks[iBlock].name == "species") species = blocks.Block(iBlock);
                const auto components = rhs.Components(input);
                result->status = CPU::AccumulateStateDiagnostics(blocks.Block(flowBlock), species, components.View(), geometry->HostView(),
                    {workers.data(), index(workers.size())},
                    {result->values.data(), index(result->values.size())}); }, std::move(inputs), Outputs{});
            engine.Synchronize();
            RequireNumericalSuccess(result->status, "AccumulateStateDiagnostics");
            if (retained->equation.IsACM())
                result->values[13] = std::numeric_limits<real>::quiet_NaN();
            return result->values;
        }
#ifdef DNDS_USE_CUDA
        mean.CoupledField();
        residual.CoupledField();
        R::SubmitArrayBatch(engine, stream, [retained, geometry, means, rhs](auto context, auto input, auto, auto partial, auto result, auto completed)
                            { CUDA::AccumulateStateDiagnostics(context, Table(input[means.firstSlot]), Table(input[rhs.firstSlot]),
                                                               geometry->View().cells, retained->equation.IsACM() ? std::nullopt : std::optional{retained->equation.Gas()}, partial, result, completed); }, std::move(inputs), Outputs{}, std::make_tuple(R::WriteAs<R::FixedRows<1>>(retained->partial), R::WriteAs<R::FixedRows<1>>(retained->result), R::WriteAs<R::FixedRows<1>>(retained->completed)));
        engine.Synchronize();
        const auto host = retained->result.To(R::Device::Host(), stream.Context());
        const auto read = host.Read();
        const auto value = read.View()(0, 0);
        RequireNumericalSuccess(value.status, "AccumulateStateDiagnostics");
        std::array<real, 14> result{};
        result[0] = value.mass;
        result[1] = value.energy;
        result[4] = value.squaredResidual;
        result[2] = value.densityMinimum;
        result[3] = value.pressureMinimum;
        result[13] = retained->equation.IsACM() ? std::numeric_limits<real>::quiet_NaN() : value.temperatureMinimum;
        return result;
#else
        DNDS_check_throw_info(false, "CUDA diagnostics are not compiled");
        return {};
#endif
    }
}
