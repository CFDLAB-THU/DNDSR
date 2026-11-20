#pragma once

#include "DNDS/Algebra/EigenView.hpp"
#include "Cauchy/Implicit/LinearPattern.hpp"
#include "DNDS/Core/OMP.hpp"
#include <array>

namespace DNDS::Cauchy::CPU
{
    /// Select the compile-time unit component stride once per operator, before
    /// any cell scan. Pitched AoS is still contiguous inside a logical row.
    /// Row-major, SoA and tiled rows keep their dynamic-stride descriptor.
    template <class Callback>
    void WithLinearMatrixRows(LinearDiagonalRows values, bool rowMajor, Callback &&callback)
    {
        if (values.kind == Runtime::RowLayoutKind::Table && !rowMajor)
        {
            const Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> contiguous{
                values.data, nullptr, values.rows, values.width, values.stride, nullptr, values.kind, values.device};
            callback(contiguous);
        }
        else
            callback(values);
    }
    /// Stateless Jacobi row update. Scratch is allocated once per call, never
    /// per row. Static zero-start removes all off-diagonal/input accesses.
    template <bool Zero, class RHS, class Owned, class Ghost, class Output, class Inverse, class Blocks>
    void JacobiRows(RHS rhs, Owned owned, Ghost ghost, Output output, Inverse inverse,
                    const Blocks &blocks, LinearPatternView pattern, Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> rows,
                    index components, bool fullDiagonal)
    {
        constexpr bool segmented = std::is_same_v<Output, Algebra::DofComponentView<real>>;
        const index parts = std::min<index>(OMP::DistributedThreads(), std::max<index>(1, rows.RowCount()));
        std::vector<std::array<Eigen::VectorXd, 4>> scratch(parts);
        for (auto &part : scratch)
            for (auto &vector : part)
                vector.resize(components);
        const auto copyRow = [&](auto view, index iRow, Eigen::VectorXd &target)
        {
            if constexpr (segmented)
                Algebra::PackEigenRow(view, iRow, target);
            else
                target = view[iRow];
        };
        // Each logical subrange has private dense scratch. No allocation or
        // shared reduction occurs inside the OpenMP arithmetic region.
        OMP::ForRows(0, parts, [&](index part)
                     {
          auto &value = scratch[part][0], &result = scratch[part][1];
          auto &neighbor = scratch[part][2], &product = scratch[part][3];
          for (index iRow = rows.RowCount() * part / parts; iRow < rows.RowCount() * (part + 1) / parts; ++iRow)
          {
            const index iCell = rows(iRow, 0);
            copyRow(rhs, iCell, value);
            if constexpr (!Zero)
                for (index j = 0; j < index(pattern.columns[iCell].size()); ++j)
                {
                    const index column = pattern.columns[iCell][j];
                    if (column < pattern.ownedRows)
                        copyRow(owned, column, neighbor);
                    else
                        copyRow(ghost, column - pattern.ownedRows, neighbor);
                    blocks.template Accumulate<Eigen::Dynamic, true>(iCell, j, neighbor, value, product);
                }
            if (fullDiagonal)
                result.noalias() = inverse[iCell] * value;
            else
                result.array() = inverse[iCell].col(0).array() * value.array();
            if constexpr (segmented)
                Algebra::UnpackEigenRow(output, iCell, result);
            else
                output[iCell] = result;
          } });
    }
    /// One ordered SGS scan. Input remote values are frozen by the enclosing
    /// phase scope. Lower local predecessors read forward; upper successors
    /// read output. No ownership/communication or per-row allocation here.
    template <bool Upper, int Size, bool ZeroInput = false, class RHS, class Old, class Forward, class Output, class Ghost, class Inverse, class Blocks>
    void ScanSGSRows(RHS rhs, Old old, Forward forward, Output output, Ghost ghost, Inverse inverse, const Blocks &blocks,
                     LinearPatternView pattern, Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> rows, index components, bool fullDiagonal,
                     SpanView<const WorkerRowRanges> partitions = {}, std::optional<WorkerRowRanges> worker = {})
    {
        constexpr bool scratch = std::is_same_v<Output, Algebra::DofComponentView<real>>;
        if constexpr (scratch && std::is_same_v<RHS, Algebra::DofComponentView<const real>> &&
                      std::is_same_v<Old, Algebra::DofComponentView<const real>> &&
                      std::is_same_v<Ghost, Algebra::DofComponentView<const real>>)
        {
            // Resolve the common coupled AoS case once, under the enclosing
            // logical-state leases. Split, permuted and column-major fields
            // keep the generic row-gather kernel below. The SGS scan order and
            // its frozen remote inputs are unchanged.
            const auto source = Algebra::TryOwnedTable(rhs), previous = Algebra::TryOwnedTable(old);
            const auto temporary = Algebra::TryOwnedTable(forward);
            const auto destination = Algebra::TryOwnedTable(output);
            const auto remote = ghost.RowCount() ? Algebra::TryGhostTable(ghost)
                                                 : std::optional<Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>>{
                                                       Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>{nullptr, nullptr, 0, components, components}};
            if (source && previous && temporary && destination && remote)
            {
                const auto scan = [&](auto width)
                {
                    constexpr int N = decltype(width)::value;
                    ScanSGSRows<Upper, N, ZeroInput>(Algebra::MakeEigenRowMatrices<N, 1>(*source, components, 1),
                                                     Algebra::MakeEigenRowMatrices<N, 1>(*previous, components, 1),
                                                     Algebra::MakeEigenRowMatrices<N, 1>(*temporary, components, 1),
                                                     Algebra::MakeEigenRowMatrices<N, 1>(*destination, components, 1),
                                                     Algebra::MakeEigenRowMatrices<N, 1>(*remote, components, 1),
                                                     inverse, blocks, pattern, rows, components, fullDiagonal, partitions, worker);
                };
                if (components == 5)
                    scan(std::integral_constant<int, 5>{});
                else if (components == 6)
                    scan(std::integral_constant<int, 6>{});
                else if (components == 7)
                    scan(std::integral_constant<int, 7>{});
                else
                    scan(std::integral_constant<int, Eigen::Dynamic>{});
                return;
            }
        }
        if (OMP::DistributedThreads() > 1 && partitions.size() > 1)
        {
            // Each block scans the selected logical rows in their original
            // order. Only its own output may be read before the team joins.
            OMP::ForRows(0, partitions.size(), [&](index part)
                         { ScanSGSRows<Upper, Size, ZeroInput>(rhs, old, forward, output, ghost, inverse, blocks,
                                                               pattern, rows, components, fullDiagonal, {}, partitions[part]); });
            return;
        }
        Eigen::VectorXd otherScratch(scratch ? components : 0), outputScratch(scratch ? components : 0);
        const auto readRow = [&](auto input, index iRow)
        {
            if constexpr (scratch)
                return Algebra::PackEigenRow(input, iRow, otherScratch);
            else
                return input[iRow]; // Explicit dense-view specialization only.
        };
        Eigen::Matrix<real, Size, 1> value(components);
        Eigen::Matrix<real, Blocks::needs_scratch ? Size : 0, 1> product(Blocks::needs_scratch ? components : 0);
        const index ownedRows = pattern.ownedRows;
        for (index scan = 0; scan < rows.RowCount(); ++scan)
        {
            const index iCell = rows(Upper ? rows.RowCount() - 1 - scan : scan, 0);
            if (worker && !worker->Contains(iCell))
                continue;
            value = readRow(rhs, iCell);
            const auto &columns = pattern.columns[iCell];
            for (index j = 0; j < index(columns.size()); ++j)
            {
                const auto column = columns[j];
                if constexpr (ZeroInput)
                {
                    // Dedicated kernel: old data is absent, not stored zeros.
                    // Upper remote values are analytic zeros in local SGS;
                    // a middle-exchange upper scan uses the ordinary kernel.
                    if (column >= ownedRows)
                        continue;
                    if constexpr (!Upper)
                        if (pattern.position(column, 0) >= pattern.position(iCell, 0))
                            continue;
                }
                const auto subtract = [&](const auto &input)
                {
                    blocks.template Accumulate<Size, true>(iCell, j, input, value, product);
                };
                if (column >= ownedRows)
                    subtract(readRow(ghost, column - ownedRows));
                else if (worker && !worker->Contains(column))
                {
                    // Lower input is the frozen old iterate (or analytic
                    // zero). Upper input is the fully completed lower result,
                    // never another block's concurrently changing output.
                    if constexpr (Upper)
                        subtract(readRow(forward, column));
                    else if constexpr (!ZeroInput)
                        subtract(readRow(old, column));
                }
                else if (pattern.position(column, 0) < pattern.position(iCell, 0))
                    subtract(readRow(forward, column));
                else if constexpr (Upper)
                    subtract(readRow(output, column));
                else if constexpr (!ZeroInput)
                    subtract(readRow(old, column));
            }
            auto result = [&]
            {
                if constexpr (scratch)
                    return Eigen::Ref<Eigen::VectorXd>(outputScratch);
                else
                    return output[iCell];
            }();
            if (fullDiagonal)
            {
                if constexpr (Size != Eigen::Dynamic)
                    result.noalias() = inverse[iCell].template topLeftCorner<Size, Size>() * value;
                else
                    result.noalias() = inverse[iCell] * value;
            }
            else
                result.array() = inverse[iCell].array() * value.array();
            if constexpr (scratch)
                Algebra::UnpackEigenRow(output, iCell, result);
        }
    }

    /// Matrix action shared by physical-field maps and logical component views.
    /// All ownership, matrix/field validation and row-dependency checks belong
    /// to the enclosing access scope. The field path keeps direct Eigen maps;
    /// only separate logical fields need one-row scratch, allocated before loops.
    template <int Size = Eigen::Dynamic, class Owned, class Ghost, class Output, class Diagonal, class Blocks>
    void ApplyBlockRows(Owned owned, Ghost ghost, Output output, Diagonal diagonal, const Blocks &blocks,
                        LinearPatternView pattern, Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> rows, index components, bool fullDiagonal)
    {
        constexpr bool scratch = std::is_same_v<Output, Algebra::DofComponentView<real>>;
        if constexpr (scratch && std::is_same_v<Owned, Algebra::DofComponentView<const real>> &&
                      std::is_same_v<Ghost, Algebra::DofComponentView<const real>>)
        {
            const auto source = Algebra::TryOwnedTable(owned);
            const auto destination = Algebra::TryOwnedTable(output);
            auto remote = ghost.RowCount() ? Algebra::TryGhostTable(ghost)
                                           : std::optional<Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>>{
                                                 Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>{nullptr, nullptr, 0, components, components}};
            if (source && destination && remote)
            {
                const auto apply = [&](auto width)
                {
                    constexpr int N = decltype(width)::value;
                    ApplyBlockRows<N>(Algebra::MakeEigenRowMatrices<N, 1>(*source, components, 1),
                                      Algebra::MakeEigenRowMatrices<N, 1>(*remote, components, 1),
                                      Algebra::MakeEigenRowMatrices<N, 1>(*destination, components, 1),
                                      diagonal, blocks, pattern, rows, components, fullDiagonal);
                };
                if (components == 5)
                    apply(std::integral_constant<int, 5>{});
                else if (components == 6)
                    apply(std::integral_constant<int, 6>{});
                else if (components == 7)
                    apply(std::integral_constant<int, 7>{});
                else
                    apply(std::integral_constant<int, Eigen::Dynamic>{});
                return;
            }
        }
        const index parts = std::min<index>(OMP::DistributedThreads(), std::max<index>(1, rows.RowCount()));
        struct Scratch
        {
            Eigen::VectorXd owned, other, output;
            Eigen::Matrix<real, Blocks::needs_scratch ? Size : 0, 1> product;
            explicit Scratch(index components)
                : owned(scratch ? components : 0), other(scratch ? components : 0), output(scratch ? components : 0),
                  product(Blocks::needs_scratch ? components : 0) {}
        };
        std::vector<Scratch> buffers;
        buffers.reserve(parts);
        for (index part = 0; part < parts; ++part)
            buffers.emplace_back(components);
        const auto readRow = [&](auto input, index iRow, auto &storage)
        {
            if constexpr (scratch)
                return Algebra::PackEigenRow(input, iRow, storage);
            else
                return input[iRow];
        };
        const index ownedRows = pattern.ownedRows;
        OMP::ForRows(0, parts, [&](index part)
                     {
          auto &ownScratch = buffers[part].owned, &otherScratch = buffers[part].other;
          auto &outputScratch = buffers[part].output;
          auto &product = buffers[part].product;
          for (index iRow = rows.RowCount() * part / parts; iRow < rows.RowCount() * (part + 1) / parts; ++iRow)
          {
            const index iCell = rows(iRow, 0);
            auto result = [&]
            {
                if constexpr (scratch)
                    return Eigen::Ref<Eigen::VectorXd>(outputScratch);
                else
                    return output[iCell];
            }();
            const auto value = readRow(owned, iCell, ownScratch);
            if (fullDiagonal)
            {
                if constexpr (Size != Eigen::Dynamic)
                    result.noalias() = diagonal[iCell].template topLeftCorner<Size, Size>() * value;
                else
                    result.noalias() = diagonal[iCell] * value;
            }
            else
            {
                if constexpr (Size != Eigen::Dynamic)
                    result.array() = diagonal[iCell].col(0).template head<Size>().array() * value.array();
                else
                    result.array() = diagonal[iCell].col(0).array() * value.array();
            }
            for (index j = 0; j < index(pattern.columns[iCell].size()); ++j)
            {
                const index column = pattern.columns[iCell][j];
                if (column < ownedRows)
                    blocks.template Accumulate<Size, false>(iCell, j, readRow(owned, column, otherScratch), result, product);
                else
                    blocks.template Accumulate<Size, false>(iCell, j, readRow(ghost, column - ownedRows, otherScratch), result, product);
            }
            if constexpr (scratch)
                Algebra::UnpackEigenRow(output, iCell, result);
          } });
    }

    /// Copy operator blocks to the prepared rank-local factor template.
    /// Missing entries are structural fill-in. All scopes are retained by the
    /// factorization facade; the generic LU implementation owns elimination.
    template <class Diagonal, class Blocks, class Factor>
    void FillILUBlocks(const Diagonal &diagonal, const Blocks &blocks, const Factor &factor,
                       LinearColumnsView sourceBlocks,
                       bool fullDiagonal)
    {
        OMP::ForRows(0, sourceBlocks.RowCount(), [&](index iCell)
                     {
            if (fullDiagonal)
                factor(iCell, 0) = diagonal[iCell];
            else
                factor(iCell, 0) = diagonal[iCell].col(0).asDiagonal();
            const auto slots = sourceBlocks[iCell];
            for (index j = 0; j < slots.size(); ++j)
            {
                auto block = factor(iCell, 1 + j);
                if (slots[j] == UnInitIndex)
                    block.setZero();
                else
                    block = blocks(iCell, slots[j]);
            } });
    }
}
