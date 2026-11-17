#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Cauchy/Implicit/CPU/LinearProblemKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Algebra/BatchedSVD.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Implicit/CUDA/LinearDiagonalKernels.hpp"
#    include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct LinearProblemContext::InverseWorkspace
    {
        R::ExecutionContext context;
        CUDADiagonalInverse kernel;
        R::Array<NumericalStatus> summary, rowStatus, statusScratch;
        std::optional<R::Array<index>> fallbackCount;
        R::Array<real> normalized, scales, singular, left, right, work;
        R::Array<int> info, truncated;
        std::shared_ptr<Algebra::BatchedSVD> provider;
        uint64_t sourceEpoch = 0, inverseEpoch = 0;
        explicit InverseWorkspace(R::ExecutionContext where, index rows, int width, bool fullBlock, CUDADiagonalInverse selected)
            : context(where), kernel(selected), summary(R::RowLayout::Dense(1, 1), where.device)
        {
            summary.SetName("linear/inverse/status");
            if (where.device == R::Device::Host())
            {
                fallbackCount.emplace(R::RowLayout::Dense(1, 1));
                return;
            }
#ifdef DNDS_USE_CUDA
            DNDS_check_throw_info(width >= 4 && width <= 7, "CUDA inverse requires widths 4 through 7");
            rowStatus = R::Array<NumericalStatus>(R::RowLayout::Dense(rows, 1), where.device);
            statusScratch = R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(rows), 1), where.device);
            rowStatus.SetName("linear/inverse/rowStatus");
            statusScratch.SetName("linear/inverse/statusScratch");
            if (!fullBlock)
                return;
            if (kernel == CUDADiagonalInverse::GuardedLU)
            {
                truncated = R::Array<int>(R::RowLayout::Dense(rows, 1), where.device);
                truncated.SetName("linear/inverse/fallbackMask");
                fallbackCount.emplace(R::RowLayout::Dense(1, 1), where.device);
                fallbackCount->SetName("linear/inverse/fallbackCount");
                return;
            }
            normalized = R::Array<real>(R::RowLayout::Dense(rows, width * width), where.device);
            scales = R::Array<real>(R::RowLayout::Dense(rows, width), where.device);
            singular = R::Array<real>(scales.Layout(), where.device);
            left = R::Array<real>(normalized.Layout(), where.device);
            right = R::Array<real>(normalized.Layout(), where.device);
            info = R::Array<int>(R::RowLayout::Dense(rows, 1), where.device);
            truncated = R::Array<int>(info.Layout(), where.device);
            normalized.SetName("linear/inverse/svdInput");
            scales.SetName("linear/inverse/scales");
            singular.SetName("linear/inverse/singularValues");
            left.SetName("linear/inverse/leftVectors");
            right.SetName("linear/inverse/rightVectors");
            info.SetName("linear/inverse/svdInfo");
            truncated.SetName("linear/inverse/truncatedModes");
            {
                auto a = normalized.Write(where), s = singular.Write(where), u = left.Write(where), v = right.Write(where);
                provider = std::make_shared<Algebra::BatchedSVD>(where, width, Algebra::BatchedSVDViews{a.View(), s.View(), u.View(), v.View()});
            }
            work = R::Array<real>(R::RowLayout::Dense(provider->WorkspaceElements(), 1), where.device);
            work.SetName("linear/inverse/svdWorkspace");
#else
            DNDS_check_throw_info(false, "CUDA inverse was not compiled");
#endif
        }
    };
    void LinearProblemContext::PrepareDiagonalInverse(R::ExecutionContext context, CUDADiagonalInverse kernel)
    {
        ValidateMatrix();
        DNDS_check_throw_info(!_inversePending && context.device == Device(), "inverse setup has pending work or a different device");
        DNDS_check_throw_info(kernel == CUDADiagonalInverse::BatchedSVD || kernel == CUDADiagonalInverse::GuardedLU, "invalid CUDA inverse kernel");
        if (_inverseWorkspace && _inverseWorkspace->context.device == context.device && _inverseWorkspace->context.stream == context.stream && _inverseWorkspace->kernel == kernel)
            return;
        _inverseWorkspace = std::make_shared<InverseWorkspace>(context, index(_space.OwnedRows()), int(_components), _diagonalStorage == DiagonalStorage::Block, kernel);
    }
    void LinearProblemContext::SubmitInvertDiagonal(R::EagerEngine &engine, const R::ExecutionStream &stream, int refinementSteps)
    {
        ValidateMatrix();
        const auto context = stream.Context();
        DNDS_check_throw_info(refinementSteps >= 0 && refinementSteps <= 3 &&
                                  (!refinementSteps || _diagonalStorage == DiagonalStorage::Block),
                              "inverse refinement requires block storage and 0..3 corrections");
        DNDS_check_throw_info(engine.Owns(stream) && context.device == Device() && !_inversePending && _inverseWorkspace &&
                                  _inverseWorkspace->context.device == context.device && _inverseWorkspace->context.stream == context.stream,
                              "inverse requires an idle prepared workspace on this stream");
        _inverseCurrent = false;
        _inverseFallbacks = _inverseSVDCells = 0;
        _inversePending = true;
        const auto retained = _inverseWorkspace;
        retained->sourceEpoch = _diagonal.MutationEpoch();
        const int width = int(_components);
        const bool fullBlock = _diagonalStorage == DiagonalStorage::Block;
        const bool rowMajor = RowMajorBlocks();
        if (context.device == R::Device::Host())
        {
            R::SubmitArrays(engine, stream, [retained, width, fullBlock, refinementSteps, rowMajor](R::ExecutionContext, auto diagonal, auto inverse, auto status, auto counts)
                            {
                const auto result = CPU::InvertLinearDiagonal({diagonal, inverse, width, fullBlock, refinementSteps, rowMajor});
                status(0, 0) = result.status;
                counts(0, 0) = result ? result.value : 0; }, R::ReadAs<R::UniformTableRows>(_diagonal), R::WriteAs<R::UniformTableRows>(_inverse), R::WriteAs<R::FixedRows<1>>(retained->summary), R::WriteAs<R::FixedRows<1>>(*retained->fallbackCount));
        }
        else
        {
#ifdef DNDS_USE_CUDA
            if (!fullBlock)
                R::SubmitArrays(engine, stream, [retained, width](R::ExecutionContext context, auto diagonal, auto inverse, auto status)
                                { CUDA::InvertDiagonalEntries(context, diagonal, inverse, status, width); }, R::ReadAs<R::UniformTableRows>(_diagonal), R::WriteAs<R::UniformTableRows>(_inverse), R::WriteAs<R::FixedRows<1>>(retained->rowStatus));
            else if (retained->kernel == CUDADiagonalInverse::GuardedLU)
                R::SubmitArrays(engine, stream, [retained, width, refinementSteps, rowMajor](R::ExecutionContext context, auto diagonal, auto inverse, auto status, auto fallback, auto count)
                                { CUDA::InvertSmallBlocks(context, diagonal, inverse, status, fallback, count, width, refinementSteps, rowMajor); }, R::ReadAs<R::UniformTableRows>(_diagonal), R::WriteAs<R::UniformTableRows>(_inverse), R::WriteAs<R::FixedRows<1>>(retained->rowStatus), R::WriteAs<R::FixedRows<1>>(retained->truncated), R::WriteAs<R::FixedRows<1>>(*retained->fallbackCount));
            else
            {
                R::SubmitArrays(engine, stream, [retained, width, rowMajor](R::ExecutionContext context, auto diagonal, auto normalized, auto scales, auto status)
                                { CUDA::PrepareDiagonalSVD(context, diagonal, normalized, scales, status, width, rowMajor); }, R::ReadAs<R::UniformTableRows>(_diagonal), R::WriteAs<R::UniformTableRows>(retained->normalized), R::WriteAs<R::UniformTableRows>(retained->scales), R::WriteAs<R::FixedRows<1>>(retained->rowStatus));
                R::SubmitArrays(engine, stream, [retained](R::ExecutionContext context, auto a, auto s, auto u, auto v, auto work, auto info)
                                { retained->provider->SubmitBorrowed(context, {a, s, u, v}, work, info); }, R::ReadWrite(retained->normalized), R::Write(retained->singular), R::Write(retained->left), R::Write(retained->right), R::Write(retained->work), R::Write(retained->info));
                R::SubmitArrays(engine, stream, [retained, width, refinementSteps, rowMajor](R::ExecutionContext context, auto diagonal, auto s, auto u, auto v, auto scales, auto info, auto inverse, auto status, auto truncated)
                                { CUDA::CompleteDiagonalSVD(context, diagonal, s, u, v, scales, info, inverse, status, truncated, width, refinementSteps, rowMajor); }, R::ReadAs<R::UniformTableRows>(_diagonal), R::ReadAs<R::UniformTableRows>(retained->singular), R::ReadAs<R::UniformTableRows>(retained->left), R::ReadAs<R::UniformTableRows>(retained->right), R::ReadAs<R::UniformTableRows>(retained->scales), R::ReadAs<R::FixedRows<1>>(retained->info), R::WriteAs<R::UniformTableRows>(_inverse), R::ReadWriteAs<R::FixedRows<1>>(retained->rowStatus), R::WriteAs<R::FixedRows<1>>(retained->truncated));
            }
            R::SubmitArrays(engine, stream, [retained](R::ExecutionContext context, auto rows, auto scratch, auto summary)
                            { CUDA::ReduceNumericalStatus(context, rows, {}, scratch, summary); }, R::ReadAs<R::FixedRows<1>>(retained->rowStatus), R::WriteAs<R::FixedRows<1>>(retained->statusScratch), R::WriteAs<R::FixedRows<1>>(retained->summary));
#else
            DNDS_check_throw_info(false, "CUDA inverse was not compiled");
#endif
        }
        retained->inverseEpoch = _inverse.MutationEpoch();
    }
    void LinearProblemContext::ObserveDiagonalInverse()
    {
        DNDS_check_throw_info(_inversePending && _inverseWorkspace, "no inverse preparation is pending");
        ValidateMatrix();
        const auto &retained = *_inverseWorkspace;
        // This is the explicit preparation observation boundary, not a hidden
        // wait inside a numerical kernel or a host copy of the matrix payload.
        const auto summary = Device() == R::Device::Host() ? retained.summary : retained.summary.To(R::Device::Host());
        const auto read = summary.Read();
        _inversePending = false;
        RequireNumericalSuccess(read.View<R::FixedRows<1>>()(0, 0), "queued diagonal inverse");
        DNDS_check_throw_info(_diagonal.MutationEpoch() == retained.sourceEpoch && _inverse.MutationEpoch() == retained.inverseEpoch,
                              "inverse operands changed after preparation submission");
        if (retained.fallbackCount)
        {
            const auto counts = Device() == R::Device::Host() ? *retained.fallbackCount : retained.fallbackCount->To(R::Device::Host());
            const auto count = counts.Read();
            _inverseFallbacks = count.View<R::FixedRows<1>>()(0, 0);
            _inverseSVDCells = _inverseFallbacks;
        }
        else
            _inverseSVDCells = _diagonalStorage == DiagonalStorage::Block ? index(_space.OwnedRows()) : 0;
        _diagonalVersion = _diagonal.Version();
        _inverseVersion = _inverse.Version();
        _inverseSourceEpoch = retained.sourceEpoch;
        _inverseEpoch = retained.inverseEpoch;
        _inverseCurrent = true;
    }
}
