#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct LinearProblemContext::ActionWorkspace
    {
        R::Array<NumericalStatus> rows, partial, result;
        ActionWorkspace(index count, R::Device device)
            : rows(R::RowLayout::Dense(count, 1))
        {
#ifdef DNDS_USE_CUDA
            // Once at problem construction, not per action or nonlinear update.
            {
                auto write = rows.Write();
                const auto view = write.View<R::FixedRows<1>>();
                for (index iRow = 0; iRow < count; ++iRow)
                    view(iRow, 0) = {};
            }
            rows = rows.To(device);
            partial = R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(count), 1), device);
            result = R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), device);
            rows.SetName("Cauchy/linear/actionStatus", DNDS_ALLOCATION_SITE());
            partial.SetName("Cauchy/linear/actionStatusPartials", DNDS_ALLOCATION_SITE());
            result.SetName("Cauchy/linear/actionStatusResult", DNDS_ALLOCATION_SITE());
#else
            (void)device;
            DNDS_check_throw_info(false, "CUDA action status is not compiled");
#endif
        }
    };
    void LinearProblemContext::PrepareActionWorkspace()
    {
        if (Device().space == R::MemorySpace::CUDA && !_offDiagonal)
            _actionWorkspace = std::make_shared<ActionWorkspace>(index(_space.OwnedRows()), Device());
    }
    R::Array<NumericalStatus> &LinearProblemContext::ActionStatus() const
    {
        DNDS_check_throw_info(_actionWorkspace, "device evaluated action workspace is missing");
        return _actionWorkspace->rows;
    }
    void LinearProblemContext::ResetQueuedActionStatus(R::EagerEngine &engine, const R::ExecutionStream &stream) const
    {
        if (!_actionWorkspace)
            return;
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == Device(), "linear action reset stream differs");
#ifdef DNDS_USE_CUDA
        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto status)
                        { CUDA::ClearNumericalStatus(context, status); }, R::WriteAs<R::FixedRows<1>>(_actionWorkspace->rows));
#endif
    }
    void LinearProblemContext::CheckQueuedActions(R::EagerEngine &engine, const R::ExecutionStream &stream) const
    {
        if (!_actionWorkspace)
            return;
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == Device(), "linear action observation stream differs");
#ifdef DNDS_USE_CUDA
        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto rows, auto partial, auto result)
                        { CUDA::ReduceNumericalStatus(context, rows, {}, partial, result); }, R::ReadAs<R::FixedRows<1>>(_actionWorkspace->rows), R::WriteAs<R::FixedRows<1>>(_actionWorkspace->partial), R::WriteAs<R::FixedRows<1>>(_actionWorkspace->result));
        stream.Record().WaitHost();
        const auto host = _actionWorkspace->result.To(R::Device::Host(), stream.Context());
        const auto read = host.Read();
        RequireNumericalSuccess(read.View<R::FixedRows<1>>()(0, 0), "evaluated linear action");
#endif
    }
}
