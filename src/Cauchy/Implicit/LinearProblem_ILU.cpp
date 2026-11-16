#include "Cauchy/Implicit/LinearProblem.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Implicit/CPU/LinearBlockKernels.hpp"
#include "Solver/NativeBlockLU.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Serializer/JsonUtil.hpp"
#include "Geom/Transition/CorrectRCM.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cstdio>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct LinearProblemContext::ILUState
    {
        std::shared_ptr<Direct::SerialSymLUStructure> symbolic;
        std::unique_ptr<Direct::NativeBlockLU> factors;
        LinearPatternStorage factorMapping;
        // Per physical row, lower then upper factor slots map to immutable
        // operator-column slots. UnInitIndex denotes structural fill-in.
        uint64_t diagonalVersion = 0, offDiagonalVersion = 0;
        bool current = false;
        // Opt-in local attribution only: no additional collective or change
        // to numeric factorization. Gathering and decomposition are disjoint.
        bool profile = std::getenv("DNDS_CAUCHY_PROFILE_LINEAR") != nullptr;
        bool profileAllRanks = std::getenv("DNDS_CAUCHY_PROFILE_ALL_RANKS") != nullptr;
        std::array<real, 2> seconds{};
        index factorizations = 0;
        index inverseFallbacks = 0;
    };
    void LinearProblemContext::PrepareILUStructure(int fill, bool coreFirst, const std::vector<index> &localParts, const std::string &ordering)
    {
        ValidateMatrix();
        DNDS_check_throw_info(fill >= -1 && fill <= 5 && _space.Distributed(), "ILU structure needs a domain communicator and fill -1..5");
        DNDS_check_throw_info(ordering == "natural" || ordering == "rcm", "ILU ordering must be natural or rcm");
        auto prepared = std::make_shared<ILUState>();
        const auto patternRead = _pattern->rows.Read();
        const auto pattern = patternRead.View();
        const index rows = pattern.ownedRows;
        DNDS_check_throw_info(localParts.empty() || (localParts.size() >= 2 && localParts.front() == 0 && localParts.back() == rows &&
                                                     std::is_sorted(localParts.begin(), localParts.end())),
                              "invalid ILU local part boundaries");
        const auto parts = localParts.empty() ? std::vector<index>{0, rows} : localParts;
        std::vector<std::vector<index>> adjacency(rows);
        // Generic symbolic construction preserves supplied cross-part edges;
        // its independent-part numeric scans do not remove them. Build the
        // induced block-Jacobi graphs explicitly, before ordering or fill.
        for (index part = 0; part + 1 < index(parts.size()); ++part)
            for (index iCell = parts[part]; iCell < parts[part + 1]; ++iCell)
                for (auto column : pattern.columns[iCell])
                    if (column >= parts[part] && column < parts[part + 1])
                    {
                        adjacency[iCell].push_back(column);
                        adjacency[column].push_back(iCell);
                    }
        for (auto &row : adjacency)
        {
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }
        prepared->symbolic = std::make_shared<Direct::SerialSymLUStructure>(MPIInfo(_space.Communicator()), rows);
        if (ordering == "rcm")
        {
            auto &oldToNew = prepared->symbolic->localFillOrderingOld2New;
            auto &newToOld = prepared->symbolic->localFillOrderingNew2Old;
            oldToNew.resize(rows);
            newToOld.resize(rows);
            // Factor ordering owns only a rank-local induced graph. Use the
            // same header-only CorrectRCM algorithm as Euler, without mesh or
            // legacy storage. Independent local parts may never intermix.
            for (index part = 0; part + 1 < index(parts.size()); ++part)
            {
                const index first = parts[part], count = parts[part + 1] - first;
                if (!count)
                    continue;
                std::vector<std::vector<index>> local(count);
                for (index iCell = 0; iCell < count; ++iCell)
                    for (auto other : adjacency[first + iCell])
                        local[iCell].push_back(other - first);
                const auto row = [&](index iCell)
                { return R::RowSpan<const index>{local[iCell].data(), index(local[iCell].size())}; };
                auto graph = CorrectRCM::UndirectedGraphProxy(row, count);
                (void)graph.CheckAdj();
                std::vector<index> cm(count);
                CorrectRCM::CuthillMcKeeOrdering(graph, [&](index iCell) -> index &
                                                 { return cm[iCell]; }, index(0));
                std::vector<uint8_t> seen(count, 0);
                for (index iCell = 0; iCell < count; ++iCell)
                {
                    const index position = count - 1 - cm[iCell];
                    DNDS_check_throw_info(position >= 0 && position < count && !seen[position], "invalid CorrectRCM factor permutation");
                    seen[position] = 1;
                    oldToNew[first + iCell] = first + position;
                    newToOld[first + position] = first + iCell;
                }
            }
        }
        if (coreFirst)
            prepared->symbolic->SetCoreFirstOrdering(Core(), parts);
        prepared->symbolic->ObtainSymmetricSymbolicFactorization(adjacency, parts, fill, false);
        std::vector<std::vector<index>> sourceBlocks(rows);
        std::vector<index> forwardCore, backwardCore;
        for (index iCell = 0; iCell < rows; ++iCell)
        {
            auto &slots = sourceBlocks[iCell];
            const auto &source = pattern.columns[iCell];
            const auto &lower = prepared->symbolic->lowerTriStructure[iCell];
            const auto &upper = prepared->symbolic->upperTriStructure[iCell];
            slots.reserve(lower.size() + upper.size());
            for (const auto *columns : {&lower, &upper})
                for (const auto column : *columns)
                {
                    const auto found = std::find(source.begin(), source.end(), column);
                    slots.push_back(found == source.end() ? UnInitIndex : index(found - source.begin()));
                }
        }
        if (fill >= 0)
        {
            forwardCore = prepared->symbolic->ReadySolveRows(Core(), true);
            backwardCore = prepared->symbolic->ReadySolveRows(forwardCore, false);
        }
        prepared->factorMapping = LinearPatternStorage(sourceBlocks, {}, {}, forwardCore, backwardCore);
        prepared->factors = std::make_unique<Direct::NativeBlockLU>(prepared->symbolic, _components);
        _ilu = std::move(prepared);
    }
    void LinearProblemContext::FactorizeILU()
    {
        using Clock = std::chrono::steady_clock;
        const bool profile = _ilu && _ilu->profile;
        const auto start = profile ? Clock::now() : Clock::time_point{};
        ValidateMatrix();
        DNDS_check_throw_info(_ilu, "ILU structure is not prepared");
        _ilu->current = false;
        const auto diagonal = _diagonal.Read();
        const auto offDiagonal = ReadOffDiagonal();
        const auto d = diagonal.View<R::UniformTableRows>();
        Clock::time_point gathered{};
        // TODO(large-species factors): support component-partitioned storage
        // through the generic factor algebra, not by zeroing dense factors
        // after elimination. Coupled diagonal blocks can generate cross terms
        // even if every initial off-diagonal block is flow5+scalar diagonal.
        // Only the fully decoupled approximate operator closes exactly under
        // that representation; other projections need an explicit policy.
        const auto mappingRead = _ilu->factorMapping.Read();
        const auto mapping = mappingRead.View();
        _ilu->factors->FillAndFactorize([&](const auto &matrix)
                                        {
            CPU::FillLinearFactor(d, offDiagonal.View(), matrix, mapping.columns, _components,
                                  _diagonalStorage == DiagonalStorage::Block, _rowMajorBlocks);
            gathered = profile ? Clock::now() : Clock::time_point{}; });
        _ilu->diagonalVersion = _diagonal.Version();
        _ilu->offDiagonalVersion = OffDiagonalGeneration();
        _ilu->current = true;
        if (profile)
        {
            _ilu->inverseFallbacks += _ilu->factors->LastInverseFallbacks();
            _ilu->seconds[0] += std::chrono::duration<real>(gathered - start).count();
            _ilu->seconds[1] += std::chrono::duration<real>(Clock::now() - gathered).count();
            const int rank = MPIInfo(_space.Communicator()).rank;
            ++_ilu->factorizations;
            if ((_ilu->factorizations % 32 == 0 || (_ilu->profileAllRanks && _ilu->factorizations == 1)) && (!rank || _ilu->profileAllRanks))
            {
                const auto record = nlohmann::ordered_json{{"rank", rank}, {"rows", _pattern->partition->owned}, {"components", _components}, {"factorizations", _ilu->factorizations}, {"factor_blocks", _ilu->factors->Matrix().Offsets().back()}, {"inverse_fallbacks_last", _ilu->factors->LastInverseFallbacks()}, {"inverse_fallbacks_total", _ilu->inverseFallbacks}, {"phases", {"gather", "copy_and_decompose"}}, {"seconds", _ilu->seconds}}.dump();
                std::fprintf(stdout, "CAUCHY_ILU_PROFILE %s\n", record.c_str());
            }
        }
    }
    bool LinearProblemContext::ILUFactorsCurrent() const
    {
        try
        {
            ValidateMatrix();
            const auto diagonal = _diagonal.Read();
            const auto offDiagonal = ReadOffDiagonal();
            return _ilu && _ilu->current && _ilu->diagonalVersion == _diagonal.Version() &&
                   _ilu->offDiagonalVersion == OffDiagonalGeneration() && _ilu->factors->FactorsCurrent();
        }
        catch (const std::exception &)
        {
            MPI::AbortIfSolverFailure("Cauchy ILU operation failed");
            return false;
        }
    }
    std::vector<index> LinearProblemContext::ILUForwardCore() const
    {
        DNDS_check_throw_info(_ilu, "ILU structure is not prepared");
        return _ilu->factorMapping.CopyCore();
    }
    std::vector<index> LinearProblemContext::ILUBackwardCore() const
    {
        DNDS_check_throw_info(_ilu, "ILU structure is not prepared");
        return _ilu->factorMapping.CopyDependent();
    }
    void LinearProblemContext::ApplyILU(const Field &rhs, Field &forward, Field &output) const
    {
        ValidateMatrix();
        ValidateOutputs(rhs, forward);
        ValidateOutputs(rhs, output);
        ValidateOutputs(forward, output);
        const auto diagonal = _diagonal.Read();
        const auto offDiagonal = ReadOffDiagonal();
        DNDS_check_throw_info(_ilu && _ilu->current && _ilu->diagonalVersion == _diagonal.Version() &&
                                  _ilu->offDiagonalVersion == OffDiagonalGeneration(),
                              "ILU factors are missing or stale against the operator");
        _ilu->factors->Solve(rhs.Owned(), forward.Owned(), output.Owned());
    }
    void LinearProblemContext::ApplyILU(const EquationDofs &rhs, EquationDofs &forward, EquationDofs &output) const
    {
        ValidateMatrix();
        ValidateOutputs(rhs, forward);
        ValidateOutputs(rhs, output);
        ValidateOutputs(forward, output);
        const auto diagonal = _diagonal.Read();
        const auto offDiagonal = ReadOffDiagonal();
        DNDS_check_throw_info(_ilu && _ilu->current && _ilu->diagonalVersion == _diagonal.Version() &&
                                  _ilu->offDiagonalVersion == OffDiagonalGeneration(),
                              "logical ILU factors are absent or stale");
        const auto read = rhs.ReadOwned();
        auto middle = forward.WriteOwned();
        std::optional<EquationDofs::WriteScope> write;
        try
        {
            write.emplace(output.WriteOwned());
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy ILU operation failed");
            middle.CancelUnmodified();
            throw;
        }
        const auto b = read.ComponentLayout();
        const auto y = middle.ComponentLayout(), x = write->ComponentLayout();
        try
        {
            _ilu->factors->SolveRows(b.View(), y.View(), x.View());
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy ILU operation failed");
            middle.Fail();
            write->Fail();
            throw;
        }
    }
}
