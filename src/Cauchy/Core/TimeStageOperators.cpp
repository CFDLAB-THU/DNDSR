#include "Cauchy/Core/TimeStageOperators.hpp"
#include "Cauchy/Core/CPU/TimeStageKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/State/EquationArrayOperands.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePositivityKernels.hpp"
#include <climits>
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Core/CUDA/TimeStageKernels.hpp"
#    include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationArrayOperands;
    struct TimeStageOperators::Data
    {
        CFV::FiniteVolumePlan geometry;
        R::ExecutionContext context;
#ifdef DNDS_USE_CUDA
        R::Array<CUDA::TimeStageSummary> partial, summary;
        R::Array<CUDA::StateReductionValue> stepIdentity, stepSummary;
#endif
        Data(CFV::FiniteVolumePlan metrics, R::ExecutionContext where) : geometry(std::move(metrics)), context(where)
        {
            DNDS_check_throw_info(geometry.Device() == context.device, "time-stage metric placement differs");
            if (context.device == R::Device::Host())
                return;
#ifdef DNDS_USE_CUDA
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "unsupported time-stage backend");
#else
            DNDS_check_throw_info(false, "CUDA time-stage operators are not compiled");
#endif
        }
        void Check(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state,
                   ExplicitStageCoefficients c) const
        {
            state.Validate();
            DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == context.device &&
                                      (context.device == R::Device::Host() || stream.Context().stream == context.stream) && state.Device() == context.device && state.Space().Same(geometry.Space()),
                                  "time-stage stream, state space or device differs");
            DNDS_check_throw_info(std::isfinite(c.dt) && c.dt > 0 && std::isfinite(c.historyWeight) && c.historyWeight >= 0 &&
                                      std::isfinite(c.stageWeight) && c.stageWeight >= 0,
                                  "invalid explicit-stage coefficients");
            if (context.device != R::Device::Host())
                DNDS_check_throw_info(state.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table &&
                                          state.ComponentNumber() >= 5 && state.ComponentNumber() <= 7 && state.Schema().blocks.front().name == "flow" &&
                                          (state.BlockNumber() == 1 || (state.BlockNumber() == 2 && state.Schema().blocks[1].name == "rans")),
                                      "explicit CUDA stage requires combined AoS flow5/SA/2EQ");
        }
        void CheckInputs(const EquationDofs &initial, const EquationDofs &rhs, const EquationDofs &stage) const
        {
            initial.RequireCompatible(stage);
            rhs.RequireCompatible(stage);
            EquationDofOperations::RequireDistinct(initial, stage);
            EquationDofOperations::RequireDistinct(rhs, stage);
        }
        template <class Host, class Device>
        ExplicitBlendStatistics Reduce(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                       Inputs inputs, Outputs outputs, Host host, Device device)
        {
            if (context.device == R::Device::Host())
            {
                struct Result
                {
                    ExplicitBlendStatistics blend;
                    NumericalStatus status;
                };
                const auto result = std::make_shared<Result>();
                R::SubmitArrayBatch(engine, stream, [result, host](auto, auto input, auto output)
                                    { host(input, output, result->blend, result->status); }, std::move(inputs), std::move(outputs));
                engine.Synchronize();
                RequireNumericalSuccess(result->status, "explicit-stage operator");
                return result->blend;
            }
#ifdef DNDS_USE_CUDA
            if (!summary.RowCount())
            {
                partial = R::Array<CUDA::TimeStageSummary>(R::RowLayout::Dense(CUDA::TimeStageScratchRows(geometry.NumCell()), 1), context.device);
                summary = R::Array<CUDA::TimeStageSummary>(R::RowLayout::Dense(1, 1), context.device);
                partial.SetName("time-stage/reduction/partials", DNDS_ALLOCATION_SITE());
                summary.SetName("time-stage/reduction/summary", DNDS_ALLOCATION_SITE());
            }
            R::SubmitArrayBatch(engine, stream, [device](auto context, auto input, auto output, auto scratch, auto value)
                                { device(context, input, output, scratch, value); }, std::move(inputs), std::move(outputs), std::make_tuple(R::WriteAs<R::FixedRows<1>>(partial), R::WriteAs<R::FixedRows<1>>(summary)));
            engine.Synchronize();
            const auto hostSummary = summary.To(R::Device::Host(), context);
            const auto read = hostSummary.Read();
            const auto value = read.View<R::FixedRows<1>>()(0, 0);
            RequireNumericalSuccess(value.status, "explicit-stage operator");
            return value.blend;
#else
            DNDS_check_throw_info(false, "CUDA time-stage operators are not compiled");
            return {};
#endif
        }
    };
    TimeStageOperators::TimeStageOperators(CFV::FiniteVolumePlan geometry, R::ExecutionContext context)
        : _data(std::make_shared<Data>(std::move(geometry), context)) {}
    real TimeStageOperators::MinimumStableStep(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                               const R::Array<real> &spectral, real cfl) const
    {
        const auto retained = _data;
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == retained->context.device &&
                                  (retained->context.device == R::Device::Host() || stream.Context().stream == retained->context.stream) && spectral.GetDevice() == retained->context.device &&
                                  spectral.RowCount() == retained->geometry.NumCell() && !spectral.Layout().IsVariable() && spectral.Layout().Width() == 1 &&
                                  std::isfinite(cfl) && cfl > 0,
                              "invalid CFL operator stream, spectral field or control");
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(retained->geometry.AcquireRead(stream.Context()));
        Inputs inputs{R::ReadAs<R::UniformTableRows>(spectral)};
        if (stream.Context().device == R::Device::Host())
        {
            const auto result = std::make_shared<NumericalResult<real>>(std::numeric_limits<real>::max());
            R::SubmitArrayBatch(engine, stream, [geometry, result, cfl](auto, auto input, auto)
                                { *result = CPU::MinimumStableStep(input[0], geometry->HostView(), cfl); }, std::move(inputs), Outputs{});
            engine.Synchronize();
            return RequireNumericalSuccess(*result, "MinimumStableStep");
        }
#ifdef DNDS_USE_CUDA
        if (!retained->stepSummary.RowCount())
        {
            R::Array<CUDA::StateReductionValue> hostIdentity(R::RowLayout::Dense(1, 1));
            hostIdentity.Write().At(0, 0) = CUDA::StableStepIdentity;
            retained->stepIdentity = hostIdentity.To(retained->context.device, retained->context);
            retained->stepSummary = R::Array<CUDA::StateReductionValue>(R::RowLayout::Dense(1, 1), retained->context.device);
            retained->stepIdentity.SetName("physical-step/CFL/identity", DNDS_ALLOCATION_SITE());
            retained->stepSummary.SetName("physical-step/CFL/summary", DNDS_ALLOCATION_SITE());
        }
        R::SubmitArrayBatch(engine, stream, [geometry, cfl](auto context, auto input, auto, auto identity, auto value)
                            { CUDA::MinimumStableStep(context, Table(input[0]), geometry->View().cells, cfl, identity, value); }, std::move(inputs), Outputs{}, std::make_tuple(R::ReadAs<R::FixedRows<1>>(retained->stepIdentity), R::WriteAs<R::FixedRows<1>>(retained->stepSummary)));
        engine.Synchronize();
        const auto host = retained->stepSummary.To(R::Device::Host(), stream.Context());
        const auto read = host.Read();
        const auto value = read.View<R::FixedRows<1>>()(0, 0);
        RequireNumericalSuccess(value.firstInvalid < 0 ? NumericalStatus{} : NumericalStatus{NumericalError::InvalidState, 1, value.firstInvalid}, "MinimumStableStep");
        return value.value;
#else
        DNDS_check_throw_info(false, "CUDA physical-step operators are not compiled");
        return 0;
#endif
    }
    PhysicalStepLimit TimeStageOperators::LimitPhysicalStep(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                            const EquationDofs &mean, const EquationDofs &residual, AdmissibilityPolicy policy,
                                                            real dt, real relax, real scale, int power) const
    {
        _data->Check(engine, stream, mean, {dt, 0, 1});
        _data->Check(engine, stream, residual, {dt, 0, 1});
        mean.RequireCompatible(residual);
        EquationDofOperations::RequireDistinct(mean, residual);
        const auto thermal = policy.View().thermal;
        DNDS_check_throw_info(std::isfinite(relax) && relax > 0 && relax <= 1 && std::isfinite(scale) && scale > 0 &&
                                  power > 0 && power <= INT_MAX / 2 && thermal.components == mean.ComponentNumber() &&
                                  std::isfinite(thermal.densityFloor) && thermal.densityFloor > 0,
                              "invalid physical-step thermal policy or controls");
        index first = 0, flowFirst = -1, speciesFirst = -1, independent = 0;
        for (const auto &block : mean.Schema().blocks)
        {
            if (block.name == "flow")
                flowFirst = first;
            if (block.name == "species")
            {
                speciesFirst = first;
                independent = index(block.components.size());
            }
            first += index(block.components.size());
        }
        DNDS_check_throw_info(thermal.flowFirst == flowFirst && thermal.speciesFirst == speciesFirst && thermal.independent == independent,
                              "physical-step policy component mapping differs");
        for (int c = 0; c < 2; ++c)
            DNDS_check_throw_info(std::isfinite(thermal.floors[c]) && thermal.floors[c] >= 0 && std::isfinite(thermal.densityCoefficients[c]) &&
                                      (thermal.speciesEnergies[c].empty() ? independent == 0 : thermal.speciesEnergies[c].size() == independent + 1),
                                  "physical-step policy coefficients are missing or invalid");
        Inputs inputs;
        const auto u = AddInput(mean, inputs), r = AddInput(residual, inputs);
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
        const auto host = [geometry, u, r, policy, dt, relax, scale, power](auto input, auto, auto &blend, auto &status)
        {
            const auto mean = u.Components<const real>(input), residual = r.Components<const real>(input);
            const auto result = CPU::LimitPhysicalStepRows(mean.View(), residual.View(), geometry->HostView(), policy.View().thermal, dt, relax, scale, power);
            status = result.status;
            if (status)
            {
                blend.limited = result.value.first;
                blend.alpha = result.value.second;
            }
        };
#ifdef DNDS_USE_CUDA
        const auto device = [geometry, u, r, policy, dt, relax, scale, power](auto context, auto input, auto, auto scratch, auto value)
        { CUDA::LimitPhysicalStep(context, geometry->View().cells, Table(input[u.firstSlot]), Table(input[r.firstSlot]), policy.View().thermal,
                                  dt, relax, scale, power, scratch, value); };
#else
        const auto device = [] {};
#endif
        const auto result = _data->Reduce(engine, stream, std::move(inputs), {}, host, device);
        return {dt, result.limited, result.alpha, false};
    }
    ExplicitBlendStatistics TimeStageOperators::MeasureBlend(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                             const EquationDofs &initial, const EquationDofs &stage, const EquationDofs &high, const EquationDofs &low,
                                                             ThermalBounds bounds, ExplicitStageCoefficients c) const
    {
        for (const auto *state : {&initial, &stage, &high, &low})
        {
            _data->Check(engine, stream, *state, c);
            stage.RequireCompatible(*state);
            DNDS_check_throw_info(state->ComponentNumber() == 5, "explicit conservative blend requires flow5");
            (void)state->CoupledField();
        }
        Inputs inputs;
        const auto a = AddInput(initial, inputs), u = AddInput(stage, inputs), h = AddInput(high, inputs), l = AddInput(low, inputs);
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
        const auto host = [geometry, a, u, h, l, bounds, c](auto input, auto, auto &blend, auto &status)
        {
            const auto result = CPU::MeasureExplicitBlend(input[a.firstSlot], input[u.firstSlot], input[h.firstSlot], input[l.firstSlot],
                                                          geometry->HostView(), bounds, c.dt, c.historyWeight, c.stageWeight);
            status = result.status;
            if (status)
                blend = result.value;
        };
#ifdef DNDS_USE_CUDA
        const auto device = [geometry, a, u, h, l, bounds, c](auto context, auto input, auto, auto scratch, auto value)
        { CUDA::MeasureExplicitBlend(context, geometry->View().cells, Table(input[a.firstSlot]), Table(input[u.firstSlot]),
                                     Table(input[h.firstSlot]), Table(input[l.firstSlot]), bounds, c, scratch, value); };
#else
        const auto device = [] {};
#endif
        return _data->Reduce(engine, stream, std::move(inputs), {}, host, device);
    }
    void TimeStageOperators::UpdateFlow(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                        const EquationDofs &initial, const EquationDofs &rhs, EquationDofs &stage,
                                        OptionalRef<const EquationDofs> low, real alpha, ExplicitStageCoefficients c) const
    {
        for (const auto *state : {&initial, &rhs, static_cast<const EquationDofs *>(&stage)})
        {
            _data->Check(engine, stream, *state, c);
            DNDS_check_throw_info(state->ComponentNumber() == 5, "explicit flow update requires flow5");
            (void)state->CoupledField();
        }
        _data->CheckInputs(initial, rhs, stage);
        DNDS_check_throw_info(alpha >= 0 && alpha <= 1 && (low || alpha == 1), "explicit flux blend lacks low residual or has invalid alpha");
        if (low)
        {
            _data->Check(engine, stream, *low, c);
            stage.RequireCompatible(*low);
            (void)low->CoupledField();
            EquationDofOperations::RequireDistinct(*low, stage);
        }
        Inputs inputs;
        Outputs outputs;
        const auto a = AddInput(initial, inputs), r = AddInput(rhs, inputs), y = AddOutput(stage, outputs, true);
        const auto l = low ? std::optional(AddInput(*low, inputs)) : std::nullopt;
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
        R::SubmitArrayBatch(engine, stream, [geometry, a, r, y, l, alpha, c](auto context, auto input, auto output)
                            {
            if (context.device == R::Device::Host())
                CPU::UpdateExplicitFlow(input[a.firstSlot], input[r.firstSlot], l ? input[l->firstSlot] : Read{}, output[y.firstSlot],
                    geometry->HostView(), c.dt, c.historyWeight, c.stageWeight, alpha);
            else
            {
#ifdef DNDS_USE_CUDA
                CUDA::UpdateExplicitFlow(context, geometry->View().cells, Table(input[a.firstSlot]), Table(input[r.firstSlot]),
                    l ? Table(input[l->firstSlot]) : CUDA::TimeStageInput{}, Table(output[y.firstSlot]), alpha, c);
#else
                DNDS_check_throw_info(false, "CUDA time-stage operators are not compiled");
#endif
            } }, std::move(inputs), std::move(outputs));
        engine.Synchronize(); // explicit stage acceptance is the next host decision
    }
    void TimeStageOperators::UpdateExtended(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                            const EquationDofs &initial, const EquationDofs &rhs, EquationDofs &stage, RANSStateView rans, ExplicitStageCoefficients c) const
    {
        for (const auto *state : {&initial, &rhs, static_cast<const EquationDofs *>(&stage)})
            _data->Check(engine, stream, *state, c);
        _data->CheckInputs(initial, rhs, stage);
        DNDS_check_throw_info(stage.Schema().blocks.front().name == "flow", "explicit update requires flow before scalar repair");
        index ransComponents = 0;
        for (const auto &block : stage.Schema().blocks)
            if (block.name == "rans")
                ransComponents = index(block.components.size());
        DNDS_check_throw_info(rans.components == ransComponents, "explicit RANS policy width differs from the state");
        Inputs inputs;
        Outputs outputs;
        const auto a = AddInput(initial, inputs), r = AddInput(rhs, inputs), y = AddOutput(stage, outputs, true);
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
        const auto host = [geometry, a, r, y, rans, c](auto input, auto output, auto &, NumericalStatus &status)
        {
            const Binding::ReadBlocks base{a, input}, rhs{r, input};
            const Binding::WriteBlocks target{y, output};
            const auto flowOutput = target.Block(0);
            const EquationBlockArrayView<const real> flow{flowOutput.parent.ConstView(), flowOutput.first, flowOutput.components};
            for (index iBlock = 0; iBlock < y.state.BlockNumber(); ++iBlock)
            {
                status = CPU::UpdateExplicitBlock(base.Block(iBlock).owned, rhs.Block(iBlock).owned, target.Block(iBlock), flow,
                                                  geometry->HostView(), rans, rans.components && y.state.Schema().blocks[iBlock].name == "rans",
                                                  c.dt, c.historyWeight, c.stageWeight);
                if (!status)
                    return;
            }
        };
#ifdef DNDS_USE_CUDA
        const auto device = [geometry, a, r, y, rans, c](auto context, auto input, auto output, auto scratch, auto value)
        { CUDA::UpdateExplicitExtended(context, geometry->View().cells, Table(input[a.firstSlot]), Table(input[r.firstSlot]),
                                       Table(output[y.firstSlot]), rans, c, scratch, value); };
#else
        const auto device = [] {};
#endif
        _data->Reduce(engine, stream, std::move(inputs), std::move(outputs), host, device);
    }
}
