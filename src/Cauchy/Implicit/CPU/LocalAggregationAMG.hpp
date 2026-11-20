#pragma once

#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Solver/AggregationAMG.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <climits>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    class CPUAMG
    {
        index rows, width;
        std::vector<index> slots;
        R::Array<real> blocks, diagonal, input, output;
        std::unique_ptr<Solver::AggregationAMG> solver;
        uint64_t diagonalEpoch = 0, offEpoch = 0;
        bool current = false;

    public:
        CPUAMG(const LinearProblemContext &problem, int aggregateSize)
            : rows(problem.Space().OwnedRows()), width(problem.Components())
        {
            DNDS_check_throw_info(rows < INT_MAX && problem.Device() == R::Device::Host() &&
                                      problem.HasAssembledOffDiagonal() && problem.DiagonalMode() == DiagonalStorage::Block,
                                  "CPU AMG requires host assembled full blocks and int32 local rows");
            const auto pattern = problem.HostPattern().Read();
            const auto cols = pattern.View().columns;
            const auto strengths = problem.GeometricWeights().Read();
            const auto weightsView = strengths.View();
            std::vector<int> offsets{0}, columns;
            std::vector<real> weights;
            index globalSlot = 0;
            for (index i = 0; i < rows; ++i)
            {
                for (index j = 0; j < cols.RowSize(i); ++j, ++globalSlot)
                    if (cols(i, j) < rows)
                    {
                        DNDS_check_throw_info(columns.size() < INT_MAX, "CPU AMG local graph exceeds int32");
                        columns.push_back(int(cols(i, j)));
                        weights.push_back(weightsView(globalSlot, 0));
                        slots.push_back(globalSlot);
                    }
                offsets.push_back(int(columns.size()));
            }
            R::Array<int> o(R::RowLayout::Dense(rows + 1, 1)), c(R::RowLayout::Dense(columns.size(), 1));
            R::Array<real> w(R::RowLayout::Dense(columns.size(), 1));
            {
                auto ow = o.Write(), cw = c.Write();
                auto ww = w.Write();
                for (index i = 0; i <= rows; ++i)
                    ow.View()(i, 0) = offsets[i];
                for (index i = 0; i < index(columns.size()); ++i)
                {
                    cw.View()(i, 0) = columns[i];
                    ww.View()(i, 0) = weights[i];
                }
            }
            blocks = R::Array<real>(R::RowLayout::Dense(slots.size(), width * width));
            diagonal = R::Array<real>(R::RowLayout::Dense(rows, width * width));
            input = R::Array<real>(R::RowLayout::Dense(rows, width));
            output = R::Array<real>(R::RowLayout::Dense(rows, width));
            blocks.SetName("Cauchy/linear/cpuAMG/blocks", DNDS_ALLOCATION_SITE());
            diagonal.SetName("Cauchy/linear/cpuAMG/diagonal", DNDS_ALLOCATION_SITE());
            input.SetName("Cauchy/linear/cpuAMG/input", DNDS_ALLOCATION_SITE());
            output.SetName("Cauchy/linear/cpuAMG/output", DNDS_ALLOCATION_SITE());
            Solver::AggregationAMGSettings settings;
            settings.aggregateSize = aggregateSize;
            solver = std::make_unique<Solver::AggregationAMG>(o, c, w, int(width), settings);
        }
        void Update(R::EagerEngine &engine, const LinearProblemContext &problem)
        {
            const auto dEpoch = problem.Diagonal().MutationEpoch();
            const auto aEpoch = problem.OffDiagonal().Values().MutationEpoch();
            if (current && dEpoch == diagonalEpoch && aEpoch == offEpoch)
                return;
            engine.Synchronize();
            {
                const auto d = problem.Diagonal().Read(), a = problem.OffDiagonal().Values().Read();
                auto dw = diagonal.Write(), aw = blocks.Write();
                const auto dv = d.View<R::UniformTableRows>(), av = a.View<R::UniformTableRows>();
                const auto dstD = dw.View(), dstA = aw.View();
                for (index i = 0; i < rows; ++i)
                    for (index r = 0; r < width; ++r)
                        for (index c = 0; c < width; ++c)
                            dstD(i, r * width + c) = dv(i, problem.RowMajorBlocks() ? r * width + c : c * width + r);
                for (index i = 0; i < index(slots.size()); ++i)
                    for (index r = 0; r < width; ++r)
                        for (index c = 0; c < width; ++c)
                            dstA(i, r * width + c) = av(slots[i], problem.OffDiagonal().RowMajorBlocks() ? r * width + c : c * width + r);
            }
            solver->Refresh(blocks, diagonal);
            diagonalEpoch = dEpoch;
            offEpoch = aEpoch;
            current = true;
        }
        template <class Ops, class State>
        void Apply(R::EagerEngine &engine, const State &rhs, State &result)
        {
            engine.Synchronize();
            DNDS_check_throw_info(current, "CPU AMG matrix is stale");
            {
                auto dst = input.Write();
                const auto destination = dst.View();
                index first = 0;
                Ops::VisitFields(rhs, [&](const auto &part)
                                 {
                    const auto src = part.Owned().Read();
                    const auto view = src.template View<R::UniformTableRows>();
                    for (index i = 0; i < rows; ++i)
                        for (index j = 0; j < index(part.Width()); ++j)
                            destination(i, first + j) = view(i, j);
                    first += part.Width(); });
                DNDS_check_throw_info(first == width, "CPU AMG input width differs");
            }
            solver->Apply(input, output);
            {
                const auto src = output.Read();
                const auto source = src.View();
                index first = 0;
                Ops::VisitFields(result, [&](auto &part)
                                 {
                    auto dst = part.Owned().Write();
                    const auto view = dst.template View<R::UniformTableRows>();
                    for (index i = 0; i < rows; ++i)
                        for (index j = 0; j < index(part.Width()); ++j)
                            view(i, j) = source(i, first + j);
                    first += part.Width(); });
                DNDS_check_throw_info(first == width, "CPU AMG output width differs");
            }
        }
    };
}
