#include "Solver/AggregationAMG.hpp"
#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include "DNDS/Core/OMP.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// Manual, single-rank comparison on the same coupled 2-D block diffusion operator.
int main(int argc, char **argv)
{
    try
    {
        const int side = argc > 1 ? std::stoi(argv[1]) : 32;
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 10;
        const int ordinal = argc > 3 ? std::stoi(argv[3]) : 0;
        DNDS_check_throw_info(side >= 4 && side <= 512 && repeats > 0, "usage: benchmark [side 4..512] [repeats] [cuda ordinal]");
        constexpr int B = 3, BB = B * B;
        const int n = side * side, e = 4 * side * (side - 1);
        const int requestedThreads = DNDS::OMP::DistributedThreads();
        int observedThreads = 1;
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel num_threads(requestedThreads)
        {
#    pragma omp single
            observedThreads = omp_get_num_threads();
        }
#endif
        R::Array<int> offsets(R::RowLayout::Dense(n + 1, 1)), columns(R::RowLayout::Dense(e, 1));
        R::Array<double> weights(R::RowLayout::Dense(e, 1)), off(R::RowLayout::Dense(e, BB));
        R::Array<double> diag(R::RowLayout::Dense(n, BB)), rhs(R::RowLayout::Dense(n, B)), x(R::RowLayout::Dense(n, B));
        R::Array<double> flatValues(R::RowLayout::Dense((e + n) * BB, 1)), flatRhs(R::RowLayout::Dense(n * B, 1));
        {
            auto o = offsets.Write();
            auto c = columns.Write();
            auto w = weights.Write();
            auto v = off.Write();
            auto d = diag.Write();
            auto b = rhs.Write();
            auto fv = flatValues.Write();
            auto fb = flatRhs.Write();
            int k = 0;
            for (int row = 0; row < n; ++row)
            {
                o.View()(row, 0) = k;
                const int ix = row % side, iy = row / side;
                std::vector<int> neighbors;
                if (iy)
                    neighbors.push_back(row - side);
                if (ix)
                    neighbors.push_back(row - 1);
                if (ix + 1 < side)
                    neighbors.push_back(row + 1);
                if (iy + 1 < side)
                    neighbors.push_back(row + side);
                for (int col : neighbors)
                {
                    c.View()(k, 0) = col;
                    w.View()(k, 0) = 1;
                    for (int i = 0; i < B; ++i)
                        for (int j = 0; j < B; ++j)
                        {
                            const double a = i == j ? -1.0 : -0.04;
                            v.View()(k, i * B + j) = a;
                            fv.View()(k * BB + i * B + j, 0) = a;
                        }
                    ++k;
                }
                for (int i = 0; i < B; ++i)
                {
                    const double val = 1.0 + .1 * i + .2 * std::sin(.1 * ix) * std::cos(.1 * iy);
                    b.View()(row, i) = val;
                    fb.View()(row * B + i, 0) = val;
                    for (int j = 0; j < B; ++j)
                    {
                        const double a = i == j ? 4.2 + .1 * i : .12;
                        d.View()(row, i * B + j) = a;
                        fv.View()((e + row) * BB + i * B + j, 0) = a;
                    }
                }
            }
            o.View()(n, 0) = k;
            DNDS_check_throw_info(k == e, "incorrect graph size");
        }
        auto residual = [&](const R::Array<double> &correction, bool flat)
        {
            auto rr = rhs.Read();
            auto xx = correction.Read();
            auto oo = offsets.Read();
            auto cc = columns.Read();
            auto vv = off.Read();
            auto dd = diag.Read();
            double norm = 0, bn = 0;
            auto at = [&](int i, int j)
            { return flat ? xx.View()(i * B + j, 0) : xx.View()(i, j); };
            for (int i = 0; i < n; ++i)
                for (int p = 0; p < B; ++p)
                {
                    double ax = 0;
                    for (int q = 0; q < B; ++q)
                        ax += dd.View()(i, p * B + q) * at(i, q);
                    for (int k = oo.View()(i, 0); k < oo.View()(i + 1, 0); ++k)
                        for (int q = 0; q < B; ++q)
                            ax += vv.View()(k, p * B + q) * at(cc.View()(k, 0), q);
                    const double b = rr.View()(i, p);
                    norm += (b - ax) * (b - ax);
                    bn += b * b;
                }
            return std::sqrt(norm / bn);
        };
        const auto device = R::Device::CUDA(ordinal);
        const auto context = R::ExecutionContext::CUDA(ordinal);
        auto deviceValues = flatValues.To(device), deviceRhs = flatRhs.To(device), deviceWeights = weights.To(device);
        R::Array<double> deviceX(flatRhs.Layout(), device);
        std::cout << std::setprecision(10);
        for (int maxLevels : {1, 20})
        {
            DNDS::Solver::AggregationAMGSettings cpuSettings;
            cpuSettings.maxLevels = maxLevels;
            cpuSettings.minCoarseRows = 16;
            cpuSettings.aggregateSize = 2;
            auto start = Clock::now();
            DNDS::Solver::AggregationAMG cpu(offsets, columns, weights, B, cpuSettings);
            double cpuPattern = elapsed(start);
            start = Clock::now();
            cpu.Refresh(off, diag);
            double cpuSetup = elapsed(start);
            start = Clock::now();
            cpu.Apply(rhs, x);
            double cpuFirst = elapsed(start);
            const double cpuResidual = residual(x, false);
            start = Clock::now();
            for (int i = 0; i < repeats; ++i)
                cpu.Apply(rhs, x);
            double cpuApply = elapsed(start) / repeats;
            A::AmgXSettings gpuSettings;
            gpuSettings.maxLevels = maxLevels;
            gpuSettings.minCoarseRows = 16;
            gpuSettings.aggregateSize = 2;
            gpuSettings.geometricAggregation = true;
            start = Clock::now();
            A::AmgXPreconditioner gpu(device, gpuSettings);
            gpu.SetPattern(offsets, columns, B, true);
            gpu.SetGeometricWeights(deviceWeights);
            double gpuPattern = elapsed(start);
            start = Clock::now();
            gpu.AttachValues(context, deviceValues);
            double gpuSetup = elapsed(start);
            auto apply = [&]
            { auto r = deviceRhs.Read(context); auto z = deviceX.Write(context); gpu.ApplyBorrowed(context, r.View(), z.View()); };
            start = Clock::now();
            apply();
            double gpuFirst = elapsed(start);
            const auto hostX = deviceX.To(R::Device::Host());
            const double gpuResidual = residual(hostX, true);
            start = Clock::now();
            for (int i = 0; i < repeats; ++i)
                apply();
            double gpuApply = elapsed(start) / repeats;
            const auto levels = gpu.MemorySnapshot().levels.size();
            std::cout << "max_levels=" << maxLevels << " rows=" << n << " edges=" << e << " block=" << B << " repeats=" << repeats
                      << " omp_policy=" << requestedThreads << " omp_team=" << observedThreads
                      << " cpu_levels=" << cpu.LevelCount() << " cpu_pattern_s=" << cpuPattern << " cpu_refresh_s=" << cpuSetup
                      << " cpu_first_s=" << cpuFirst << " cpu_apply_s=" << cpuApply << " cpu_relative_residual=" << cpuResidual
                      << " gpu_levels=" << levels << " gpu_pattern_s=" << gpuPattern << " gpu_setup_s=" << gpuSetup
                      << " gpu_first_s=" << gpuFirst << " gpu_apply_s=" << gpuApply << " gpu_relative_residual=" << gpuResidual << std::endl;
            DNDS_check_throw_info(std::isfinite(cpuResidual) && std::isfinite(gpuResidual), "non-finite residual");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
