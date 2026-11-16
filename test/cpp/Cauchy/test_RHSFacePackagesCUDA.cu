#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace K = DNDS::Cauchy::CUDA;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    template <class T, class View>
    std::vector<T> Download(View view)
    {
        std::vector<T> values(view.RowCount());
        if (!values.empty())
            REQUIRE(cudaMemcpy(values.data(), view.data, values.size() * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
        return values;
    }
}

TEST_CASE("Complete-face packages preserve mixed rules tails and bounded point tiles")
{
    const auto device = R::Device::CUDA(0);
    const R::ExecutionContext context{device, nullptr};
    std::vector<I> counts;
    for (I iFace = 0; iFace < 43; ++iFace)
        counts.push_back(std::array<I, 5>{9, 4, 1, 81, 300}[iFace % 5]);
    R::Array<I> host(R::RowLayout::Dense(counts.size(), 1));
    {
        auto write = host.Write();
        for (I iFace = 0; iFace < I(counts.size()); ++iFace)
            write.View()(iFace, 0) = (iFace * 17) % I(counts.size());
    }
    const auto selected = host.To(device, context);
    for (const auto mapping : {K::RHSFaceMapping::WarpSegments, K::RHSFaceMapping::CTASegments})
        for (const int threads : {128, 256, 512})
        {
            K::RHSKernelControls controls;
            controls.mapping = mapping;
            controls.blockSize = threads;
            controls.modes = 19;
            const auto owner = K::RHSFacePackages::Prepare(selected, {counts.data(), I(counts.size())}, controls, context);
            const auto view = owner.View();
            CHECK(view.selection.data == selected.Read(context).View().data);
            REQUIRE(view.sharedBytes <= 48 * 1024);
            const auto packages = Download<K::RHSFacePackage>(view.packages);
            const auto segments = Download<K::RHSFaceSegment>(view.segments);
            I seen = 0;
            for (const auto &package : packages)
            {
                CHECK(package.firstSegment == seen);
                CHECK(package.faces > 0);
                int end = 0, tiles = 0;
                for (int slot = 0; slot < package.faces; ++slot)
                {
                    const auto segment = segments[package.firstSegment + slot];
                    CHECK(segment.auxiliaryFace == (seen * 17) % I(counts.size()));
                    CHECK(segment.points == counts[segment.auxiliaryFace]);
                    CHECK(segment.firstLane >= end);
                    CHECK(segment.lanes > 0);
                    CHECK(segment.firstLane + segment.lanes <= threads);
                    if (mapping == K::RHSFaceMapping::WarpSegments)
                        CHECK(segment.firstLane / 32 == (segment.firstLane + segment.lanes - 1) / 32);
                    end = segment.firstLane + segment.lanes;
                    tiles = std::max(tiles, int(segment.points / segment.lanes + (segment.points % segment.lanes != 0)));
                    ++seen;
                }
                CHECK(package.tiles == tiles);
            }
            CHECK(seen == I(counts.size()));
        }
}

TEST_CASE("Nine-point packages distinguish warp containment from tight CTA packing")
{
    const auto device = R::Device::CUDA(0);
    const R::ExecutionContext context{device, nullptr};
    std::vector<I> counts(40, 9);
    R::Array<I> host(R::RowLayout::Dense(counts.size(), 1));
    {
        auto write = host.Write();
        for (I iFace = 0; iFace < I(counts.size()); ++iFace)
            write.View()(iFace, 0) = iFace;
    }
    const auto selected = host.To(device, context);
    for (const auto mapping : {K::RHSFaceMapping::WarpSegments, K::RHSFaceMapping::CTASegments})
    {
        K::RHSKernelControls controls;
        controls.mapping = mapping;
        controls.modes = 19;
        const auto owner = K::RHSFacePackages::Prepare(selected, {counts.data(), I(counts.size())}, controls, context);
        const auto view = owner.View();
        const auto packages = Download<K::RHSFacePackage>(view.packages);
        const auto segments = Download<K::RHSFaceSegment>(view.segments);
        REQUIRE(!packages.empty());
        CHECK(packages[0].faces == (mapping == K::RHSFaceMapping::WarpSegments ? 12 : 14));
        bool crossesWarp = false;
        for (const auto &segment : segments)
            crossesWarp |= segment.firstLane / 32 != (segment.firstLane + segment.lanes - 1) / 32;
        CHECK(crossesWarp == (mapping == K::RHSFaceMapping::CTASegments));
        // A resource-limited profile must reject rather than silently launch
        // fewer threads, omit the low flux, or replace the requested mapping.
        controls.blockSize = 512;
        controls.produceLow = true;
        CHECK_THROWS(K::RHSFacePackages::Prepare(selected, {counts.data(), I(counts.size())}, controls, context));
    }
}
