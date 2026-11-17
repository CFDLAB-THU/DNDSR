#include "Cauchy/IO/OutputFields.hpp"
#include "Cauchy/IO/CPU/OutputFieldsKernels.hpp"
#include "Cauchy/Physics/PointWorkerContexts.hpp"
#include "Cauchy/State/EquationArrayOperands.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/IO/CUDA/OutputFieldsKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    OutputFields OutputFields::Submit(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                      const EquationDofs &state, const EquationSystem &equation,
                                      std::optional<Physics::Chemistry::ChemicalWorkerPool> pool)
    {
        state.RequireSchema(equation.Schema());
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == state.Device(), "output field stream and state placement differ");
        const int species = equation.HasMixture() ? equation.Mixture().SpeciesNumber() : 0;
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(state.Device() == R::Device::Host(), "CUDA output fields are not compiled");
#endif
        if (state.Device() != R::Device::Host())
            DNDS_check_throw_info(state.Device().space == R::MemorySpace::CUDA && !equation.HasMixture() &&
                                      state.PhysicalFieldNumber() == 1 && state.ComponentNumber() >= (equation.IsACM() ? 4 : 5) && state.ComponentNumber() <= (equation.IsACM() ? 6 : 7) &&
                                      state.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table,
                                  "CUDA output requires combined AoS perfect-gas or constant-density ACM flow with optional SA/2EQ");
        OutputFields result;
        const index rows = index(state.Space().OwnedRows());
        result._values = R::Array<real>(R::RowLayout::Dense(rows, state.ComponentNumber() + (equation.IsACM() ? 0 : 6 + species)), state.Device());
        result._status = R::Array<NumericalStatus>(R::RowLayout::Dense(rows, 1), state.Device());
        result._values.SetName("output/preparedFields", DNDS_ALLOCATION_SITE());
        result._status.SetName("output/fieldStatus", DNDS_ALLOCATION_SITE());
        EquationArrayOperands::Inputs inputs;
        const auto binding = EquationArrayOperands::AddInput(state, inputs);
        R::SubmitArrayBatch(engine, stream, [binding, equation, pool, species](auto context, auto input, auto, auto output, auto status)
                            {
                                if (context.device == R::Device::Host())
                                {
                                    auto workers = MakeCPUPointWorkers<CPU::OutputFieldsWorker>(equation, pool, {});
                                    const auto layout = binding.Layout(input);
                                    CPU::PrepareOutputFields(layout.Descriptor(), species, output, status, {workers.data(), index(workers.size())});
                                }
#ifdef DNDS_USE_CUDA
                                else if (equation.IsACM())
                                    CUDA::PrepareVelocityPressureOutput(context, EquationArrayOperands::Table(input[binding.firstSlot]), output, status);
                                else
                                    CUDA::PrepareOutputFields(context, equation.Gas(), EquationArrayOperands::Table(input[binding.firstSlot]), output, status);
#endif
                            },
                            std::move(inputs), EquationArrayOperands::Outputs{}, std::make_tuple(R::WriteAs<R::TableRows>(result._values), R::WriteAs<R::FixedRows<1>>(result._status)));
        return result;
    }
    R::Array<real> OutputFields::CopyToHost(R::ExecutionContext context) const
    {
        DNDS_check_throw_info(_values.Storage().IsIdle() && _status.Storage().IsIdle(), "output fields require completed production before host observation");
        const auto hostStatus = _status.To(R::Device::Host(), context);
        const auto read = hostStatus.Read();
        const auto status = read.View<R::FixedRows<1>>();
        for (index iRow = 0; iRow < status.RowCount(); ++iRow)
            RequireNumericalSuccess(status(iRow, 0), "output point properties");
        // CPU production already wrote a distinct immutable output allocation;
        // share that owner, never a live solution buffer or a redundant copy.
        if (_values.GetDevice() == R::Device::Host())
            return _values;
        const auto layout = _values.Layout();
        auto host = R::Array<real>::FromStorage(layout,
                                                R::Buffer::AllocatePinnedHost(CheckedSize::Multiply(layout.Elements(), sizeof(real))));
        host.SetName("output/pinnedFields", DNDS_ALLOCATION_SITE());
        host.CopyFrom(_values, context).Wait();
        return host;
    }
}
