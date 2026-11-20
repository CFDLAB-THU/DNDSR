#include "Cauchy/Implicit/LinearProblem.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Implicit/CPU/LinearBlockKernels.hpp"
#include <optional>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using ReadView = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
    using WriteView = R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>;

    struct LinearProblemContext::SGSIteration::State
    {
        std::shared_ptr<const Pattern> pattern;
        LinearPatternStorage::ReadAccess patternRead;
        LinearPatternView patternView;
        index components;
        DiagonalStorage mode;
        bool rowMajorBlocks;
        // Pin numeric allocation/layout owners, not communicator owners. A
        // numerical scope may retire on a worker; MPI resources stay outside.
        R::Array<real> rhs, forward, forwardGhost, output;
        std::optional<R::Array<real>> input, inputGhost;
        R::Array<real>::ReadLease diagonalRead, inverseRead, rhsRead;
        OffDiagonalReadAccess blocksRead;
        std::optional<R::Array<real>::ReadLease> inputRead, inputGhostRead, forwardRead, forwardGhostRead;
        std::optional<R::Array<real>::WriteLease> forwardWrite, outputWrite;
        uint64_t initialForwardGhost;
        bool middleExchange, started = false;
        int lower = 0, upper = 0; // 0=not started,1=first region complete,2=scan complete

        State(const LinearProblemContext &problem, const Field &b, const Field *old, Field &temporary, Field &result, bool middle)
            : pattern(problem._pattern), patternRead(pattern->rows.Read()), patternView(patternRead.View()), components(problem._components), mode(problem._diagonalStorage), rowMajorBlocks(problem._rowMajorBlocks),
              rhs(b.Owned()), forward(temporary.Owned()), forwardGhost(temporary.Ghost()), output(result.Owned()),
              diagonalRead(problem._diagonal.Read()), inverseRead(problem._inverse.Read()), rhsRead(rhs.Read()),
              blocksRead(problem.ReadOffDiagonal()), initialForwardGhost(forwardGhost.Version()), middleExchange(middle)
        {
            if (old)
            {
                input.emplace(old->Owned());
                inputGhost.emplace(old->Ghost());
                inputRead.emplace(input->Read());
            }
            DNDS_check_throw_info(problem._inverseCurrent && problem._diagonal.Version() == problem._diagonalVersion &&
                                      problem._inverse.Version() == problem._inverseVersion,
                                  "linear diagonal inverse is missing or stale");
            forwardWrite.emplace(forward.Write());
            try
            {
                outputWrite.emplace(output.Write());
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("Cauchy SGS failed");
                forwardWrite->CancelUnmodified();
                forwardWrite.reset();
                throw;
            }
        }
        ~State()
        {
            for (auto *write : {&forwardWrite, &outputWrite})
                if (*write)
                {
                    if (started)
                        (*write)->Fail();
                    else
                        (*write)->CancelUnmodified();
                }
        }
        template <bool Upper, class ForwardView>
        void Scan(ForwardView forwardValues, WriteView destination, ReadView ghostValues, Runtime::RowSelection rows)
        {
            const auto b = rhsRead.View<R::UniformTableRows>();
            const auto old = inputRead ? inputRead->View<R::UniformTableRows>() : b;
            const CPU::LinearBlockInput request{blocksRead.View(), patternView, rows,
                                                inverseRead.View<R::UniformTableRows>(), components, mode == DiagonalStorage::Block, pattern->sweepPartitions.HostView(), rowMajorBlocks};
            started = true;
            if constexpr (Upper)
                CPU::UpperSGSRows(request, b, old, forwardValues, ghostValues, destination, !input && !middleExchange);
            else
                CPU::LowerSGSRows(request, b, old, ghostValues, destination, !input);
        }
    };

    LinearProblemContext::SGSIteration::SGSIteration(std::unique_ptr<State> state) : _state(std::move(state)) {}
    LinearProblemContext::SGSIteration::SGSIteration(SGSIteration &&) noexcept = default;
    LinearProblemContext::SGSIteration &LinearProblemContext::SGSIteration::operator=(SGSIteration &&) noexcept = default;
    LinearProblemContext::SGSIteration::~SGSIteration() = default;
    LinearProblemContext::SGSIteration LinearProblemContext::BeginSGS(const Field &rhs, const Field &input, Field &forward, Field &output, bool middleExchange) const
    {
        ValidateMatrix();
        ValidateOutputs(input, forward);
        ValidateOutputs(input, output);
        ValidateOutputs(rhs, forward);
        ValidateOutputs(rhs, output);
        ValidateOutputs(forward, output);
        return SGSIteration(std::make_unique<SGSIteration::State>(*this, rhs, &input, forward, output, middleExchange));
    }
    LinearProblemContext::SGSIteration LinearProblemContext::BeginSGSZero(const Field &rhs, Field &forward, Field &output, bool middleExchange) const
    {
        ValidateMatrix();
        ValidateOutputs(rhs, forward);
        ValidateOutputs(rhs, output);
        ValidateOutputs(forward, output);
        return SGSIteration(std::make_unique<SGSIteration::State>(*this, rhs, nullptr, forward, output, middleExchange));
    }
    void LinearProblemContext::SGSIteration::Lower(LinearRegion region)
    {
        DNDS_check_throw_info(_state, "SGS iteration is finished or moved");
        auto &state = *_state;
        DNDS_check_throw_info((region == LinearRegion::All && state.lower == 0) || (region == LinearRegion::Core && state.lower == 0) ||
                                  (region == LinearRegion::HaloDependent && state.lower == 1),
                              "SGS lower regions must be All or Core then HaloDependent");
        if (state.inputGhost && region != LinearRegion::Core && state.pattern->hasGhostColumns)
            state.inputGhostRead.emplace(state.inputGhost->Read());
        const auto ghost = state.inputGhostRead ? state.inputGhostRead->View<R::UniformTableRows>() : ReadView{};
        const auto target = state.forwardWrite->View<R::UniformTableRows>();
        const auto rows = state.patternView.Rows(region);
        state.Scan<false>(target, target, ghost, rows);
        state.lower = region == LinearRegion::Core ? 1 : 2;
        if (state.lower == 2)
        {
            state.forwardWrite.reset();
            state.forwardRead.emplace(state.forward.Read());
        }
    }
    void LinearProblemContext::SGSIteration::Upper(LinearRegion region)
    {
        DNDS_check_throw_info(_state, "SGS iteration is finished or moved");
        auto &state = *_state;
        DNDS_check_throw_info(state.lower == 2 && ((region == LinearRegion::All && state.upper == 0) ||
                                                   (region == LinearRegion::HaloDependent && state.upper == 0) ||
                                                   (region == LinearRegion::Core && state.upper == 1)),
                              "SGS upper needs complete lower work and All or HaloDependent then Core");
        if (state.middleExchange && state.pattern->hasGhostColumns && !state.forwardGhostRead)
        {
            // Generation alone cannot prove the routing, but prevents silently
            // using a previous iteration's unchanged forward halo. Read() also
            // rejects unobserved/active MPI writes before the scan starts.
            auto read = state.forwardGhost.Read();
            DNDS_check_throw_info(state.forwardGhost.Version() != state.initialForwardGhost, "SGS middle exchange has not published a new forward halo");
            state.forwardGhostRead.emplace(std::move(read));
        }
        const auto ghost = state.middleExchange ? (state.forwardGhostRead ? state.forwardGhostRead->View<R::UniformTableRows>() : ReadView{})
                                                : (state.inputGhostRead ? state.inputGhostRead->View<R::UniformTableRows>() : ReadView{});
        const auto rows = state.patternView.Rows(region);
        state.Scan<true>(state.forwardRead->View<R::UniformTableRows>(), state.outputWrite->View<R::UniformTableRows>(), ghost, rows);
        state.upper = region == LinearRegion::HaloDependent ? 1 : 2;
    }
    void LinearProblemContext::SGSIteration::Finish()
    {
        DNDS_check_throw_info(_state && _state->lower == 2 && _state->upper == 2, "cannot publish an incomplete SGS iteration");
        _state->outputWrite.reset();
        _state.reset();
    }
}
