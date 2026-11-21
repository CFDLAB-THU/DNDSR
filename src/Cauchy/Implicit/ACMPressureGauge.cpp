#include "Cauchy/Implicit/ACMPressureGauge.hpp"
#include "Cauchy/State/EquationArrayOperands.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Core/OMP.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Implicit/CUDA/NonlinearKernels.hpp"
#endif

namespace DNDS::Cauchy::NonlinearOperatorDetail
{
    void SubmitPressureOffset(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                              EquationDofs &state, real offset)
    {
        namespace R = Runtime;
        state.Validate();
        DNDS_check_throw_info(state.Schema().AccumulationKind() == PhysicalAccumulationKind::ConstantDensityVelocityPressure &&
                                  std::isfinite(offset) && engine.Owns(stream) && stream.Context().device == state.Device(),
                              "pressure offset requires an ACM state, finite offset and matching execution lane");
        const auto flow = state.LocateBlock("flow");
        auto &rows = state.PhysicalField(flow.physicalField).Owned();
        if (state.Device() != R::Device::Host())
        {
#ifdef DNDS_USE_CUDA
            DNDS_check_throw_info(state.Device().space == R::MemorySpace::CUDA && state.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table,
                                  "CUDA pressure offset requires a combined AoS ACM state");
#else
            DNDS_check_throw_info(false, "device pressure offset requires CUDA");
#endif
        }
        if (!offset || !rows.RowCount())
            return;
        R::SubmitArrays(engine, stream, [column = flow.firstComponent + 3, offset](R::ExecutionContext ctx, auto field)
                        {
                            if (ctx.device == R::Device::Host())
                                OMP::ForRows(0, field.RowCount(), [&](index i)
                                             { field(i, column) += offset; });
#ifdef DNDS_USE_CUDA
                            else
                                CUDA::ShiftPressure(ctx, EquationArrayOperands::Table(field), offset);
#endif
                        },
                        R::ReadWriteAs<R::UniformTableRows>(rows));
    }
}
