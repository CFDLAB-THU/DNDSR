#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include "DNDS/Algebra/AmgXDiagnostics.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <climits>
#include <chrono>
#include <iostream>
#include <string>

namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;
/// Fixed-work, single-device native block-CSR microbenchmark; no MPI or CFD claims.
int main(int argc, char **argv)
{
    try
    {
        const I rows = argc > 1 ? std::stoll(argv[1]) : 65536;
        const int width = argc > 2 ? std::stoi(argv[2]) : 5;
        const std::string storage = argc > 3 ? argv[3] : "upload";
        const int repeats = argc > 4 ? std::stoi(argv[4]) : 8;
        DNDS_check_throw_info(rows > 1 && rows <= INT_MAX / std::max(3, width) && repeats > 0 &&
                                  ((width >= 1 && width <= 10) || width == 19) && (storage == "upload" || storage == "native"),
                              "usage: dnds_benchmark_amgx [rows] [block_width] [upload|native] [fixed_applies]");
        const I entries = 2 * rows - 2, blockElements = I(width) * width;
        R::Array<int> offsets(R::RowLayout::Dense(rows + 1, 1)), columns(R::RowLayout::Dense(entries, 1));
        R::Array<double> host(R::RowLayout::Dense((entries + rows) * blockElements, 1));
        R::Array<double> hostRhs(R::RowLayout::Dense(rows * width, 1));
        {
            auto o = offsets.Write(), c = columns.Write();
            auto a = host.Write(), b = hostRhs.Write();
            I entry = 0;
            for (I iRow = 0; iRow < rows; ++iRow)
            {
                o.View()(iRow, 0) = int(entry);
                for (I column = std::max<I>(0, iRow - 1); column <= std::min<I>(rows - 1, iRow + 1); ++column)
                {
                    const bool diagonal = column == iRow;
                    const I block = diagonal ? entries + iRow : entry;
                    if (!diagonal)
                        c.View()(entry++, 0) = int(column);
                    for (int i = 0; i < width; ++i)
                        for (int j = 0; j < width; ++j)
                            a.View()(block * blockElements + i * width + j, 0) = diagonal ? (i == j ? 6 + .1 * i : -.01) : (i == j ? -1 : 0);
                }
                for (int j = 0; j < width; ++j)
                    b.View()(iRow * width + j, 0) = 1 + .01 * j;
            }
            o.View()(rows, 0) = int(entry);
        }
        const auto device = R::Device::CUDA(0);
        const auto context = R::ExecutionContext::CUDA(0);
        auto values = host.To(device), rhs = hostRhs.To(device);
        R::Array<double> correction(hostRhs.Layout(), device);
        const auto report = [&](const char *phase, double seconds, const A::AmgXMemorySnapshot &memory)
        {
            std::cout << nlohmann::ordered_json{{"phase", phase}, {"rows", rows}, {"block_width", width}, {"storage", storage}, {"seconds", seconds}, {"memory", nlohmann::ordered_json::parse(A::AmgXMemoryReport(memory))}}.dump() << std::endl;
        };
        using Clock = std::chrono::steady_clock;
        report("native_matrix", 0, A::AmgXPreconditioner::AllocatorMemorySnapshot(device));
        auto provider = std::make_unique<A::AmgXPreconditioner>(device);
        provider->SetPattern(offsets, columns, width, true);
        report("provider_created", 0, provider->MemorySnapshot());
        const auto update = [&]
        {
            if (storage == "native")
                provider->AttachValues(context, values);
            else
            {
                const auto read = values.Read(context);
                auto off = read.View(), diagonal = read.View();
                off.rows = entries * blockElements;
                diagonal.data += entries * blockElements;
                diagonal.rows = rows * blockElements;
                provider->UpdateValuesBorrowed(context, off, diagonal);
            }
        };
        const auto apply = [&]
        {
            const auto read = rhs.Read(context);
            auto write = correction.Write(context);
            provider->ApplyBorrowed(context, read.View(), write.View());
        };
        const auto timedReport = [&](const char *phase, Clock::time_point start, int divisor = 1)
        {
            // Freeze elapsed time before the diagnostic snapshot's explicit join.
            const double elapsed = std::chrono::duration<double>(Clock::now() - start).count() / divisor;
            const auto memory = provider->MemorySnapshot();
            report(phase, elapsed, memory);
        };
        auto start = Clock::now();
        update();
        timedReport("setup", start);
        start = Clock::now();
        apply();
        timedReport("first_apply", start);
        start = Clock::now();
        for (int iteration = 0; iteration < repeats; ++iteration)
            apply();
        timedReport("warm_apply_mean", start, repeats);
        // A real coefficient refresh, preserving matrix shape and ownership.
        {
            auto write = host.Write();
            for (I i = 0; i < (entries + rows) * blockElements; ++i)
                write.View()(i, 0) *= 1.01;
        }
        values.CopyFrom(host, context).Wait();
        start = Clock::now();
        update();
        timedReport("value_refresh", start);
        provider.reset();
        report("last_provider_retired", 0, A::AmgXPreconditioner::AllocatorMemorySnapshot(device));
        return 0;
    }
    catch (const std::exception &error)
    {
        // This benchmark has no collective work and performs no recovery.
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
