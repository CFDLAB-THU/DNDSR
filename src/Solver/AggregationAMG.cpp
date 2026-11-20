#include "Solver/AggregationAMG_Internal.hpp"
#include "Solver/ScaledBlockInverse.hpp"
#include <algorithm>

namespace DNDS::Solver
{
    using namespace AMGDetail;
    namespace
    {
        using Read = R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows>;
        using Write = R::ArrayView<real, R::AccessBoundsChecks, R::TableRows>;

        template <int B>
        void Invert(Read diagonal, Write inverse)
        {
            const auto status = OMP::ForRowsStatus(0, diagonal.RowCount(), [&](index iRow) noexcept
                                                   {
                const auto result = Direct::ScaledBlock::InvertBlock<B>(diagonal.Row(iRow), inverse.Row(iRow));
                return result == Direct::ScaledBlock::Success ? NumericalStatus{} :
                       NumericalStatus{NumericalError::InvalidState, int(result), iRow}; });
            DNDS_check_throw_info(status, "AMG diagonal inverse failed at block row " + std::to_string(status.row) +
                                              " (inverse status " + std::to_string(status.detail) + ")");
        }

        void ValidateOffDiagonal(Read blocks)
        {
            const auto status = OMP::ForRowsStatus(0, blocks.RowCount(), [&](index k) noexcept
                                                   {
                for (const real x : blocks[k])
                    if (!std::isfinite(x))
                        return NumericalStatus{NumericalError::InvalidInput, 0, k};
                return NumericalStatus{}; });
            DNDS_check_throw_info(status, "AMG off-diagonal block is nonfinite at entry " + std::to_string(status.row));
        }

        void InvertLevel(Level &level, int b)
        {
            auto diag = level.diagonal.Read();
            auto inv = level.inverse.Write();
            using InvertRows = void (*)(Read, Write);
            static constexpr InvertRows small[] = {nullptr, Invert<1>, Invert<2>, Invert<3>, Invert<4>,
                                                   Invert<5>, Invert<6>, Invert<7>, Invert<8>, Invert<9>};
            (b == 19 ? Invert<19> : small[b])(diag.View<R::TableRows>(), inv.View<R::TableRows>());
        }

        // Assemble A_c = P^T A P. Each worker owns a whole aggregate row,
        // including its diagonal. Scatter slots were frozen during setup.
        void RefreshCoarse(const Level &fine, Level &coarse, int b)
        {
            const auto starts = fine.memberOffsets.Read(), members = fine.members.Read(), offsets = fine.offsets.Read(), slots = fine.edgeSlot.Read();
            const auto source = fine.blocks.Read(), diagonal = fine.diagonal.Read();
            const auto coarseOffsets = coarse.offsets.Read();
            auto blocks = coarse.blocks.Write(), diag = coarse.diagonal.Write();
            auto v = blocks.View(), d = diag.View();
            const auto a = source.View(), ad = diagonal.View();
            const auto o = offsets.View(), dst = slots.View(), m = members.View(), s = starts.View(), co = coarseOffsets.View();
            Rows(d.RowCount(), [&](index row)
                 {
                for (int q = 0; q < b * b; ++q)
                    d(row, q) = 0;
                for (int k = co(row, 0); k < co(row + 1, 0); ++k)
                    for (int q = 0; q < b * b; ++q)
                        v(k, q) = 0;
                for (int p = s(row, 0); p < s(row + 1, 0); ++p)
                {
                    const auto i = m(p, 0);
                    for (int q = 0; q < b * b; ++q)
                        d(row, q) += ad(i, q);
                    for (int k = o(i, 0); k < o(i + 1, 0); ++k)
                    {
                        const int dest = dst(k, 0);
                        for (int q = 0; q < b * b; ++q)
                            (dest < 0 ? d(row, q) : v(dest, q)) += a(k, q);
                    }
                } });
        }
    }

    void AggregationAMG::Refresh(const R::Array<real> &values, const R::Array<real> &diagonal)
    {
        auto &data = *_data;
        auto &fine = data.levels.front();
        const int b = data.width;
        RequireTable(values, fine.columns.RowCount(), b * b);
        RequireTable(diagonal, fine.offsets.RowCount() - 1, b * b);
        const auto v = values.Read(), d = diagonal.Read();
        data.ready = false;
        fine.blocks = values;
        fine.diagonal = diagonal;
        for (index level = 0; level < LevelCount(); ++level)
        {
            if (level)
                RefreshCoarse(data.levels[level - 1], data.levels[level], b);
            const auto blocks = data.levels[level].blocks.Read();
            ValidateOffDiagonal(blocks.View<R::TableRows>());
            InvertLevel(data.levels[level], b);
        }
        data.valuesEpoch = values.MutationEpoch();
        data.diagonalEpoch = diagonal.MutationEpoch();
        data.ready = true;
    }

    void AggregationAMG::Data::Cycle(index depth, const R::Array<real> &input, R::Array<real> &output)
    {
        Level &level = levels[depth];
        const auto offsets = level.offsets.Read(), columns = level.columns.Read();
        const auto blocks = level.blocks.Read(), diagonal = level.diagonal.Read(), inv = level.inverse.Read(), rhs = input.Read();
        auto solution = output.Write(), defect = level.residual.Write();
        const auto o = offsets.View<R::TableRows>(), c = columns.View<R::TableRows>();
        const auto a = blocks.View<R::TableRows>(), d = diagonal.View<R::TableRows>();
        const auto inverse = inv.View<R::TableRows>(), f = rhs.View<R::TableRows>();
        const auto x = solution.View<R::TableRows>(), r = defect.View<R::TableRows>();
        const int b = width;
        const index n = x.RowCount();

        // Each gather reads the old iterate. The following row-parallel
        // update starts after every residual row is complete.
        const auto residual = [&]
        {
            Rows(n, [&](index i)
                 {
                auto *row = r.Row(i);
                std::copy_n(f.Row(i), b, row);
                const auto subtract = [&](const real *block, const real *value)
                {
                    for (int j = 0; j < b; ++j)
                        for (int q = 0; q < b; ++q)
                            row[j] -= block[j * b + q] * value[q];
                };
                subtract(d.Row(i), x.Row(i));
                for (int k = o(i, 0); k < o(i + 1, 0); ++k)
                    subtract(a.Row(k), x.Row(c(k, 0))); });
        };
        const auto smooth = [&](int passes, bool fromZero)
        {
            if (fromZero && !passes)
                Rows(n, [&](index i)
                     { std::fill_n(x.Row(i), b, real(0)); });
            for (int pass = 0; pass < passes; ++pass)
            {
                const bool first = fromZero && pass == 0;
                if (!first)
                    residual();
                Rows(n, [&](index i)
                     {
                    const auto *defectRow = first ? f.Row(i) : r.Row(i);
                    const auto *block = inverse.Row(i);
                    auto *value = x.Row(i);
                    for (int j = 0; j < b; ++j)
                    {
                        real dx = 0;
                        for (int q = 0; q < b; ++q)
                            dx += block[j * b + q] * defectRow[q];
                        value[j] = (first ? real(0) : value[j]) + settings.relaxation * dx;
                    } });
            }
        };
        if (depth + 1 == index(levels.size()))
        {
            smooth(settings.coarseSweeps, true);
            return;
        }
        smooth(settings.preSweeps, true);
        residual();
        auto &coarse = levels[depth + 1];
        {
            const auto starts = level.memberOffsets.Read(), members = level.members.Read();
            const auto s = starts.View(), m = members.View();
            auto restricted = coarse.rhs.Write();
            auto out = restricted.View();
            Rows(out.RowCount(), [&](index i)
                 {
                for (int j = 0; j < b; ++j)
                {
                    real sum = 0;
                    for (int p = s(i, 0); p < s(i + 1, 0); ++p)
                        sum += r(m(p, 0), j);
                    out(i, j) = sum;
                } });
        }
        Cycle(depth + 1, coarse.rhs, coarse.x);
        const auto group = level.group.Read();
        const auto correction = coarse.x.Read();
        const auto g = group.View();
        const auto e = correction.View();
        Rows(n, [&](index i)
             {
            for (int j = 0; j < b; ++j)
                x(i, j) += e(g(i, 0), j); });
        smooth(settings.postSweeps, false);
    }

    void AggregationAMG::Apply(const R::Array<real> &rhs, R::Array<real> &correction)
    {
        auto &data = *_data;
        auto &fine = data.levels.front();
        const index n = fine.offsets.RowCount() - 1;
        RequireTable(rhs, n, data.width);
        RequireTable(correction, n, data.width);
        DNDS_check_throw_info(data.ready && !rhs.Storage().Overlaps(correction.Storage()), "AMG apply requires a refreshed hierarchy and distinct vectors");
        const auto a = fine.blocks.Read(), d = fine.diagonal.Read();
        DNDS_check_throw_info(data.valuesEpoch == fine.blocks.MutationEpoch() && data.diagonalEpoch == fine.diagonal.MutationEpoch(),
                              "AMG matrix changed since refresh");
        data.Cycle(0, rhs, correction);
    }
}
