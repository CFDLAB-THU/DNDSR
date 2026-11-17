#pragma once

#ifdef DNDS_USE_AMGX
#    include "CFV/Plans/ReconstructionAMGPlan.hpp"

namespace CFVTest
{
    /// Independent CPU fixed-point residual, including the prepared BC forcing.
    /// Relaxation is one in this fixture, so temporary-input is exactly g-A*x.
    inline double AMGResidualSquared(const C::VariationalReconstructionPlan &plan,
                                     const C::FVField &mean, const C::FVField &input,
                                     const C::ReconstructionBoundaryValues &boundary)
    {
        auto work = plan.BuildWorkspace(mean.Width());
        plan.DoReconstructionIterTo(mean, input, work, boundary);
        const auto original = input.Owned().Read(), updated = work.temporary.Owned().Read();
        double residual = 0;
        for (I iCell = 0; iCell < input.Owned().RowCount(); ++iCell)
            for (I j = 0; j < I(input.Width()); ++j)
            {
                const double value = updated.View()(iCell, j) - original.View()(iCell, j);
                residual += value * value;
            }
        return residual;
    }
}

TEST_CASE("Single-device VR AMG preserves the projected modal matrix and fixed component corrections")
{
    if (mpi.size != 1)
    {
        MESSAGE("VR AMG is a single-rank/device prototype; this distributed invocation provides no AMG numerical coverage");
        return;
    }
    // The provider test qualifies every native block width, including M=19.
    // Here a mixed three-cell mesh is deliberately small: its CPU sweep is an
    // independent oracle for the VR normalization, signs, BC forcing, and C=2
    // gather/scatter. Physical component count is independent of modal block M.
    auto mesh = CFVTest::MixedCells(mpi);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    constexpr I components = 2;
    const I rows = metrics.NumCell();
    const auto context = R::ExecutionContext::CUDA(0);
    std::vector<double> precomputedMatrix;
    for (const bool raw : {false, true})
    {
        CAPTURE(raw);
        C::ReconstructionSettings settings;
        settings.maxOrder = 1;
        settings.SORInstead = false;
        settings.jacobiRelax = 1;
        settings.storage.representation = raw ? "raw" : "precomputed";
        settings.storage.faceCoupling = raw;
        settings.storage.inverseTriangle = raw;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(
            metrics, settings, C::ReconstructionCapabilities::ForSweeps(false));
        const I modes = plan.NRecDOF(), width = components * modes;
        REQUIRE(modes == 2);
        auto mean = metrics.BuildField(components), zeroMean = metrics.BuildField(components);
        auto input = plan.BuildCoefficients(components);
        Fill(mean, 1.2);
        Fill(input, .03);
        {
            auto write = zeroMean.Owned().Write();
            for (I iCell = 0; iCell < rows; ++iCell)
                for (I j = 0; j < components; ++j)
                    write.View()(iCell, j) = 0;
        }
        auto boundary = plan.BuildBoundaryValues(components), zeroBoundary = plan.BuildBoundaryValues(components);
        {
            auto values = boundary.values.Write(), zero = zeroBoundary.values.Write();
            for (I q = 0; q < boundary.values.RowCount(); ++q)
                for (I j = 0; j < components; ++j)
                {
                    values.View()(q, j) = .9 + .013 * q + .11 * j;
                    zero.View()(q, j) = 0;
                }
        }
        auto sweep = plan.BuildWorkspace(components);
        plan.DoReconstructionIterTo(zeroMean, input, sweep, zeroBoundary);
        const double initial = CFVTest::AMGResidualSquared(plan, mean, input, boundary);
        REQUIRE(std::isfinite(initial));
        REQUIRE(initial > 1e-8);
        auto deviceMean = metrics.BuildField(R::RowLayout::Dense(0, components), context.device);
        auto deviceInput = metrics.BuildField(R::RowLayout::Dense(0, width), context.device);
        deviceMean.Owned().CopyFrom(mean.Owned(), context).Wait();
        deviceInput.Owned().CopyFrom(input.Owned(), context).Wait();
        std::optional<R::Array<double>> publicResult;
        for (const auto storage : {C::ReconstructionAMGStorage::PublicUpload, C::ReconstructionAMGStorage::NativeStorage})
        {
            const bool native = storage == C::ReconstructionAMGStorage::NativeStorage;
            CAPTURE(native);
            DNDS::Algebra::AmgXSettings amgSettings;
            amgSettings.cycles = 2;
            auto amg = C::ReconstructionAMGPlan::Prepare(plan, context, storage, amgSettings);
            const auto matrix = amg.MatrixValues().To(R::Device::Host());
            const auto values = matrix.Read();
            const auto offsets = amg.RowOffsets().Read(), columns = amg.Columns().Read();
            const auto original = input.Owned().Read(), updated = sweep.temporary.Owned().Read();
            std::vector<double> dense(size_t(rows * rows * modes * modes), 0);
            for (I iCell = 0; iCell < rows; ++iCell)
                for (I entry = offsets.View()(iCell, 0); entry < offsets.View()(iCell + 1, 0); ++entry)
                {
                    const I other = columns.View()(entry, 0);
                    REQUIRE(other >= 0);
                    REQUIRE(other < rows);
                    for (I i = 0; i < modes; ++i)
                        for (I j = 0; j < modes; ++j)
                            dense[size_t(((iCell * rows + other) * modes + i) * modes + j)] =
                                values.View()(entry * modes * modes + i * modes + j, 0);
                }
            for (I iCell = 0; iCell < rows; ++iCell)
                for (I component = 0; component < components; ++component)
                    for (I i = 0; i < modes; ++i)
                    {
                        double action = 0;
                        for (I other = 0; other < rows; ++other)
                            for (I j = 0; j < modes; ++j)
                                action += dense[size_t(((iCell * rows + other) * modes + i) * modes + j)] *
                                          original.View()(other, component * modes + j);
                        CHECK(action == doctest::Approx(original.View()(iCell, component * modes + i) -
                                                        updated.View()(iCell, component * modes + i))
                                            .epsilon(2e-11)
                                            .scale(1));
                    }
            if (!raw && !native)
                precomputedMatrix = dense;
            else
                for (I i = 0; i < I(dense.size()); ++i)
                    CHECK(dense[size_t(i)] == doctest::Approx(precomputedMatrix[size_t(i)]).epsilon(2e-11).scale(1));
            auto output = metrics.BuildField(R::RowLayout::Dense(0, width), context.device);
            auto work = amg.BuildWorkspace(components);
            const auto deviceBoundary = amg.PlaceBoundaryValues(boundary);
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            amg.SubmitDefectCorrection(engine, stream, deviceMean, deviceInput, deviceBoundary, output, work);
            engine.Close();
            const auto actual = output.Owned().To(R::Device::Host());
            auto result = plan.BuildCoefficients(components);
            result.Owned().CopyFrom(actual).Wait();
            const double final = CFVTest::AMGResidualSquared(plan, mean, result, boundary);
            CHECK(std::isfinite(final));
            CHECK(final < initial);
            if (!native)
                publicResult = actual;
            else
            {
                const auto expected = publicResult->Read(), observed = actual.Read();
                for (I iCell = 0; iCell < rows; ++iCell)
                    for (I j = 0; j < width; ++j)
                        CHECK(observed.View()(iCell, j) == doctest::Approx(expected.View()(iCell, j)).epsilon(2e-11).scale(1));
            }
            const auto memory = amg.MemorySnapshot();
            REQUIRE_FALSE(memory.levels.empty());
            CHECK(memory.nativeStorage == native);
            CHECK(memory.copies.applications == uint64_t(components));
            if (native)
            {
                CHECK(memory.copies.matrixDeviceToDevice == 0);
                CHECK(memory.levels.front().numericBorrowed == uint64_t(matrix.RowCount()) * sizeof(double));
                CHECK(memory.levels.front().numericOwned == 0);
            }
            else
            {
                CHECK(memory.copies.matrixDeviceToDevice > 0);
                CHECK(memory.levels.front().numericBorrowed == 0);
                CHECK(memory.levels.front().numericOwned > 0);
            }
        }
    }
}
#endif
