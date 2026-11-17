#include "Cauchy/State/MeanStateCheck.hpp"
#include "Cauchy/State/EquationArrayOperands.hpp"
#include "Cauchy/State/CPU/StatePropertiesKernels.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationArrayOperands;
    struct MeanStateCheck::Data
    {
        Algebra::DofSpace space;
        AdmissibilityPolicy policy;
        R::ExecutionContext context;
        R::Array<uint8_t> activeRows;
#ifdef DNDS_USE_CUDA
        R::Array<CUDA::StateReductionValue> summary;
#endif
        Data(Algebra::DofSpace rows, AdmissibilityPolicy bounds, R::ExecutionContext where, std::optional<R::Array<uint8_t>> active)
            : space(std::move(rows)), policy(std::move(bounds)), context(where),
              activeRows(active ? *active : R::Array<uint8_t>(R::RowLayout::Dense(0, 1), where.device))
        {
            DNDS_check_throw_info(activeRows.GetDevice() == context.device && activeRows.Layout().Kind() == R::RowLayoutKind::Table &&
                                      activeRows.Layout().Width() == 1 && (!active || activeRows.RowCount() == index(space.OwnedRows())),
                                  "mean-state active row mask shape/placement differs");
            if (context.device == R::Device::Host())
                return;
#ifdef DNDS_USE_CUDA
            const auto thermal = policy.View().thermal;
            const int flowWidth = thermal.flowFirst < 0 ? 4 : 5;
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA &&
                                      thermal.independent == 0 && thermal.components == flowWidth + policy.RANS().components &&
                                      thermal.speciesEnergies[0].empty() && thermal.speciesEnergies[1].empty(),
                                  "CUDA mean-state check requires flow4 or flow5 with optional RANS");
            summary = R::Array<CUDA::StateReductionValue>(R::RowLayout::Dense(1, 1), context.device);
            summary.SetName("mean-state/check/summary", DNDS_ALLOCATION_SITE());
#else
            DNDS_check_throw_info(false, "CUDA mean-state check is not compiled");
#endif
        }
    };
    MeanStateCheck::MeanStateCheck(Algebra::DofSpace space, AdmissibilityPolicy policy, R::ExecutionContext context,
                                   std::optional<R::Array<uint8_t>> activeRows)
        : _data(std::make_shared<Data>(std::move(space), std::move(policy), context, std::move(activeRows))) {}
    NumericalStatus MeanStateCheck::Evaluate(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state) const
    {
        state.Validate();
        const auto retained = _data;
        const auto policy = retained->policy.View();
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == retained->context.device &&
                                  state.Device() == retained->context.device &&
                                  state.Space().Same(retained->space) && state.ComponentNumber() == policy.thermal.components,
                              "mean-state check stream, placement, space or width differs");
        Inputs inputs;
        const auto binding = AddInput(state, inputs);
        if (retained->context.device == R::Device::Host())
        {
            const auto status = std::make_shared<NumericalStatus>();
            R::SubmitArrayBatch(engine, stream, [retained, binding, status](auto, auto input, auto, auto active)
                                {
                const auto layout = binding.Layout(input);
                *status = CPU::ValidateMeanState(layout.Descriptor(), retained->policy.View(), active); }, std::move(inputs), Outputs{}, std::make_tuple(R::ReadAs<R::FixedRows<1>>(retained->activeRows)));
            engine.Synchronize();
            return *status;
        }
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(state.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table,
                              "CUDA mean-state check requires combined AoS");
        R::SubmitArrayBatch(engine, stream, [retained, binding](auto context, auto input, auto, auto summary, auto active)
                            { CUDA::AdmissibleMeans(context, Table(input[binding.firstSlot]), retained->policy.View(), summary, active); }, std::move(inputs), Outputs{}, std::make_tuple(R::WriteAs<R::FixedRows<1>>(retained->summary), R::ReadAs<R::FixedRows<1>>(retained->activeRows)));
        engine.Synchronize();
        const auto host = retained->summary.To(R::Device::Host(), stream.Context());
        const auto read = host.Read();
        const auto result = read.View<R::FixedRows<1>>()(0, 0);
        return result.value == 1 ? NumericalStatus{} : NumericalStatus{NumericalError::InvalidState, 1, result.firstInvalid};
#else
        DNDS_check_throw_info(false, "CUDA mean-state check is not compiled");
        return {NumericalError::UnsupportedRequest};
#endif
    }
}
