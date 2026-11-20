#include "Cauchy/Implicit/LinearProblem.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Implicit/CPU/LinearBlockKernels.hpp"
#include "Cauchy/State/EquationDofs.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct LinearProblemContext::DofSGSIteration::State
    {
        std::shared_ptr<const Pattern> pattern;
        LinearPatternStorage::ReadAccess patternRead;
        LinearPatternView patternView;
        index components;
        DiagonalStorage mode;
        bool rowMajorBlocks;
        EquationDofs rhs, forward, output;
        std::optional<EquationDofs> input;
        R::Array<real>::ReadLease diagonalRead, inverseRead;
        OffDiagonalReadAccess blocksRead;
        EquationDofs::ReadScope rhsRead;
        std::optional<EquationDofs::ReadScope> inputRead, inputAllRead, forwardRead, forwardAllRead;
        std::optional<EquationDofs::WriteScope> forwardWrite, outputWrite;
        std::vector<uint64_t> initialForwardGhost;
        bool middleExchange, started = false;
        int lower = 0, upper = 0;
        State(const LinearProblemContext &problem, const EquationDofs &b, const EquationDofs *old, EquationDofs &temporary, EquationDofs &result, bool middle)
            : pattern(problem._pattern), patternRead(pattern->rows.Read()), patternView(patternRead.View()), components(problem._components), mode(problem._diagonalStorage), rowMajorBlocks(problem._rowMajorBlocks),
              rhs(b.NumericAlias()), forward(temporary.NumericAlias()), output(result.NumericAlias()),
              diagonalRead(problem._diagonal.Read()), inverseRead(problem._inverse.Read()), blocksRead(problem.ReadOffDiagonal()),
              rhsRead(rhs.ReadOwned()), middleExchange(middle)
        {
            if (old)
            {
                input.emplace(old->NumericAlias());
                inputRead.emplace(input->ReadOwned());
            }
            DNDS_check_throw_info(problem._inverseCurrent && problem._diagonal.Version() == problem._diagonalVersion &&
                                      problem._inverse.Version() == problem._inverseVersion,
                                  "logical SGS inverse is absent or stale");
            for (index iField = 0; iField < forward.PhysicalFieldNumber(); ++iField)
                initialForwardGhost.push_back(forward.PhysicalField(iField).Ghost().Version());
            forwardWrite.emplace(forward.WriteOwned());
            try
            {
                outputWrite.emplace(output.WriteOwned());
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("Cauchy DOF SGS failed");
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
        template <bool Upper, class Forward>
        void Scan(Forward forwardValues, Algebra::DofComponentView<real> destination, Algebra::DofComponentView<const real> ghost,
                  Runtime::RowSelection rows)
        {
            const auto b = rhsRead.ComponentLayout();
            std::optional<Algebra::DofComponentLayout<const real>> old;
            if (inputRead)
                old.emplace(inputRead->ComponentLayout());
            const auto previous = old ? old->View() : b.View();
            const CPU::LinearBlockInput request{blocksRead.View(), patternView, rows,
                                                inverseRead.View<R::UniformTableRows>(), components, mode == DiagonalStorage::Block, pattern->sweepPartitions.HostView(), rowMajorBlocks};
            started = true;
            if constexpr (Upper)
                CPU::UpperSGSRows(request, b.View(), previous, forwardValues, ghost, destination, !input && !middleExchange);
            else
                CPU::LowerSGSRows(request, b.View(), previous, ghost, destination, !input);
        }
    };
    LinearProblemContext::DofSGSIteration::DofSGSIteration(std::unique_ptr<State> state) : _state(std::move(state)) {}
    LinearProblemContext::DofSGSIteration::DofSGSIteration(DofSGSIteration &&) noexcept = default;
    LinearProblemContext::DofSGSIteration &LinearProblemContext::DofSGSIteration::operator=(DofSGSIteration &&) noexcept = default;
    LinearProblemContext::DofSGSIteration::~DofSGSIteration() = default;

    LinearProblemContext::DofSGSIteration LinearProblemContext::BeginSGS(const EquationDofs &rhs, const EquationDofs &input,
                                                                         EquationDofs &forward, EquationDofs &output, bool middleExchange) const
    {
        ValidateMatrix();
        ValidateOutputs(input, forward);
        ValidateOutputs(input, output);
        ValidateOutputs(rhs, forward);
        ValidateOutputs(rhs, output);
        ValidateOutputs(forward, output);
        return DofSGSIteration(std::make_unique<DofSGSIteration::State>(*this, rhs, &input, forward, output, middleExchange));
    }
    LinearProblemContext::DofSGSIteration LinearProblemContext::BeginSGSZero(const EquationDofs &rhs, EquationDofs &forward,
                                                                             EquationDofs &output, bool middleExchange) const
    {
        ValidateMatrix();
        ValidateOutputs(rhs, forward);
        ValidateOutputs(rhs, output);
        ValidateOutputs(forward, output);
        return DofSGSIteration(std::make_unique<DofSGSIteration::State>(*this, rhs, nullptr, forward, output, middleExchange));
    }
    void LinearProblemContext::DofSGSIteration::Lower(LinearRegion region)
    {
        DNDS_check_throw_info(_state, "logical SGS iteration is finished or moved");
        auto &state = *_state;
        DNDS_check_throw_info((region == LinearRegion::All && state.lower == 0) || (region == LinearRegion::Core && state.lower == 0) ||
                                  (region == LinearRegion::HaloDependent && state.lower == 1),
                              "logical SGS lower regions must be All or Core then HaloDependent");
        if (state.input && region != LinearRegion::Core && state.pattern->hasGhostColumns)
            state.inputAllRead.emplace(state.input->ReadWithGhost());
        std::optional<Algebra::DofComponentLayout<const real>> ghost;
        if (state.inputAllRead)
            ghost.emplace(state.inputAllRead->ComponentLayout());
        const auto target = state.forwardWrite->ComponentLayout();
        const auto rows = state.patternView.ScanRows(region);
        state.Scan<false>(target.View(), target.View(), ghost ? ghost->View().GhostPart() : Algebra::DofComponentView<const real>{}, rows);
        state.lower = region == LinearRegion::Core ? 1 : 2;
        if (state.lower == 2)
        {
            state.forwardWrite.reset();
            state.forwardRead.emplace(state.forward.ReadOwned());
        }
    }
    void LinearProblemContext::DofSGSIteration::Upper(LinearRegion region)
    {
        DNDS_check_throw_info(_state, "logical SGS iteration is finished or moved");
        auto &state = *_state;
        DNDS_check_throw_info(state.lower == 2 && ((region == LinearRegion::All && state.upper == 0) ||
                                                   (region == LinearRegion::HaloDependent && state.upper == 0) || (region == LinearRegion::Core && state.upper == 1)),
                              "logical SGS upper needs complete lower work and All or HaloDependent then Core");
        if (state.middleExchange && state.pattern->hasGhostColumns && !state.forwardAllRead)
        {
            auto read = state.forward.ReadWithGhost();
            for (index iField = 0; iField < state.forward.PhysicalFieldNumber(); ++iField)
                DNDS_check_throw_info(state.forward.PhysicalField(iField).Ghost().Version() != state.initialForwardGhost[iField],
                                      "logical SGS middle exchange has not published every forward field");
            state.forwardAllRead.emplace(std::move(read));
        }
        std::optional<Algebra::DofComponentLayout<const real>> ghost;
        if (state.middleExchange && state.forwardAllRead)
            ghost.emplace(state.forwardAllRead->ComponentLayout());
        if (!state.middleExchange && state.inputAllRead)
            ghost.emplace(state.inputAllRead->ComponentLayout());
        const auto forward = state.forwardRead->ComponentLayout();
        const auto target = state.outputWrite->ComponentLayout();
        const auto rows = state.patternView.ScanRows(region);
        state.Scan<true>(forward.View(), target.View(), ghost ? ghost->View().GhostPart() : Algebra::DofComponentView<const real>{}, rows);
        state.upper = region == LinearRegion::HaloDependent ? 1 : 2;
    }
    void LinearProblemContext::DofSGSIteration::Finish()
    {
        DNDS_check_throw_info(_state && _state->lower == 2 && _state->upper == 2, "cannot publish an incomplete logical SGS iteration");
        _state->outputWrite.reset();
        _state.reset();
    }
    void LinearProblemContext::ApplySGS(const EquationDofs &rhs, const EquationDofs &input, EquationDofs &forward, EquationDofs &output) const
    {
        auto iteration = BeginSGS(rhs, input, forward, output);
        iteration.Lower();
        iteration.Upper();
        iteration.Finish();
    }
}
