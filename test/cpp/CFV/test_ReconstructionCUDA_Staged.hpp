#pragma once

namespace CFVTest
{
    /// One quad adjoining two triangles gives three- and four-incidence rows.
    /// Independent global connectivity keeps the staging-bound/tail oracle
    /// separate from the reconstruction matrix implementation.
    inline std::shared_ptr<G::UnstructuredMesh> MixedCells(DNDS::MPIInfo mpi)
    {
        const auto cellRange = DNDS::EvenSplitRange(mpi.rank, mpi.size, I(3));
        const auto nodeRange = DNDS::EvenSplitRange(mpi.rank, mpi.size, I(6));
        const auto boundaryRange = DNDS::EvenSplitRange(mpi.rank, mpi.size, I(6));
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, nodeRange.second - nodeRange.first);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, cellRange.second - cellRange.first);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, boundaryRange.second - boundaryRange.first);
        const std::vector<std::vector<I>> allCells{{0, 1, 4, 3}, {1, 2, 5}, {1, 5, 4}};
        const std::vector<std::vector<I>> allBoundaries{{0, 1}, {1, 2}, {2, 5}, {5, 4}, {4, 3}, {3, 0}};
        const std::vector<std::vector<I>> localCells(allCells.begin() + cellRange.first, allCells.begin() + cellRange.second);
        const std::vector<std::vector<I>> localBoundaries(allBoundaries.begin() + boundaryRange.first, allBoundaries.begin() + boundaryRange.second);
        G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries,
                                 NativePeriodicFixtures::Rows(localCells), NativePeriodicFixtures::Rows(localBoundaries),
                                 R::Array<double>(R::RowLayout::Dense(nodes.NumOwned(), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(cells.NumOwned(), 1)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(boundaries.NumOwned(), 1))};
        {
            auto coordinates = input.coordinates.Write();
            for (I iNode = 0; iNode < nodes.NumOwned(); ++iNode)
            {
                const I global = nodeRange.first + iNode;
                coordinates.View()(iNode, 0) = global % 3;
                coordinates.View()(iNode, 1) = global / 3;
                coordinates.View()(iNode, 2) = 0;
            }
            auto cellInfo = input.cellInfo.Write(), boundaryInfo = input.boundaryInfo.Write();
            for (I iCell = 0; iCell < cells.NumOwned(); ++iCell)
                cellInfo.View()(iCell, 0) = {cellRange.first + iCell == 0 ? G::Elem::Quad4 : G::Elem::Tri3, G::INTERNAL_ZONE};
            for (I iBnd = 0; iBnd < boundaries.NumOwned(); ++iBnd)
                boundaryInfo.View()(iBnd, 0) = {G::Elem::Line2, 7};
        }
        auto mesh = G::UnstructuredMesh::FromNative(mpi, G::ConstructNativeMesh(std::move(input)));
        mesh->PrepareNativeTopology();
        return mesh;
    }
}

TEST_CASE("Stage-once VR pipelines preserve complete modal batches and queued dependencies")
{
    struct Profile
    {
        int fixture, degree;
        I components;
        bool raw, triangle;
    };
    // Cover the five compiled modal sizes (2/3/5/9/19) with complementary
    // geometry, physical BC, rotation and 5/6/7-component tails, not a Cartesian
    // product of geometry/degree/storage choices.
    for (const auto profile : {Profile{0, 1, 5, false, false}, Profile{2, 1, 6, true, false},
                               Profile{0, 3, 5, false, false}, Profile{1, 2, 6, true, false}, Profile{2, 3, 7, true, true},
                               Profile{2, 2, 6, false, false}, Profile{0, 2, 21, false, false}, Profile{2, 3, 37, true, false},
                               Profile{2, 3, 5, false, false}, Profile{0, 1, 8, true, true},
                               Profile{2, 1, 8, false, false}, Profile{2, 2, 8, true, false}})
    {
        CAPTURE(profile.fixture);
        auto mesh = profile.fixture == 0 ? CFVTest::MixedCells(mpi) : profile.fixture == 1 ? CFVTest::Sector(mpi)
                                                                                           : CFVTest::CartesianHexes(mpi, 3);
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
        C::ReconstructionSettings settings;
        settings.maxOrder = profile.degree;
        settings.cacheDiffBase = false;
        settings.jacobiRelax = .73;
        const auto capabilities = C::ReconstructionCapabilities::ForSweeps(false);
        auto referencePlan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, capabilities);
        auto mean = metrics.BuildField(profile.components), coefficients = referencePlan.BuildCoefficients(profile.components);
        Fill(mean, 1.3), Fill(coefficients, .03);
        metrics.BindHalo(mean).Begin().Wait();
        metrics.BindHalo(coefficients).Begin().Wait();
        auto referenceBoundary = referencePlan.BuildBoundaryValues(profile.components);
        {
            auto write = referenceBoundary.values.Write();
            for (I iPoint = 0; iPoint < referenceBoundary.values.RowCount(); ++iPoint)
                for (I iComponent = 0; iComponent < profile.components; ++iComponent)
                    write.View()(iPoint, iComponent) = .7 + .013 * iPoint + .09 * iComponent;
        }
        auto reference = referencePlan.BuildWorkspace(profile.components, std::vector<I>{1});
        referencePlan.DoReconstructionIterTo(mean, coefficients, reference, referenceBoundary);
        settings.storage.representation = profile.raw ? "raw" : "precomputed";
        settings.storage.faceCoupling = profile.raw;
        settings.storage.inverseTriangle = profile.triangle;
        for (int candidate = 0; candidate < 7; ++candidate)
        {
            CAPTURE(candidate);
            // Production FP64 plus general reference cover every profile.
            // Each optional comparison policy retains only its fixed fixture.
            const bool comparison = candidate != 0 && candidate != 6;
            if (comparison && (profile.fixture != 2 || profile.degree != 3 || profile.components != 5 || profile.raw))
                continue;
            const bool tensor = candidate == 2 || candidate == 4;
            const bool quantized = candidate == 3 || candidate == 4;
            settings.storage.representation = profile.raw && !quantized ? "raw" : "precomputed";
            settings.storage.faceCoupling = profile.raw && !quantized;
            settings.storage.inverseTriangle = profile.triangle && !quantized;
            settings.cudaSweep.threads = tensor && profile.components <= 7 ? 128 : 32;
            settings.cudaSweep.gridBlocks = 2;
            auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, capabilities);
            const auto context = R::ExecutionContext::CUDA(0);
            C::ReconstructionResidencyOptions placement;
            placement.matrixPrecision = quantized ? C::ReconstructionMatrixPrecision::FP32 : C::ReconstructionMatrixPrecision::FP64;
            auto resident = plan.Place(context.device, context, placement);
            const auto kernel = candidate == 4 ? C::ReconstructionDeviceKernel::TF32Pipelined : candidate == 3 ? C::ReconstructionDeviceKernel::FP32Pipelined
                                                                                            : candidate == 2   ? C::ReconstructionDeviceKernel::FP64TensorCoreAsync
                                                                                            : candidate == 1   ? C::ReconstructionDeviceKernel::FloatFloatPipelined
                                                                                            : candidate == 5   ? C::ReconstructionDeviceKernel::FP64MatrixShared
                                                                                            : candidate == 6   ? C::ReconstructionDeviceKernel::FP64
                                                                                                               : C::ReconstructionDeviceKernel::FP64Pipelined;
            if (!resident.SupportsKernel(kernel))
            {
                REQUIRE((tensor || candidate == 1)); // Float-float is an optional build capability.
                continue;
            }
            auto boundary = plan.BuildBoundaryValues(profile.components);
            boundary.values.CopyFrom(referenceBoundary.values).Wait();
            auto deviceBoundary = resident.PlaceBoundaryValues(boundary);
            auto deviceMean = metrics.BuildField(R::RowLayout::Dense(0, profile.components), context.device);
            const I coefficientWidth = profile.components * plan.NRecDOF();
            auto deviceCoefficients = resident.BuildCoefficients(profile.components,
                                                                 R::RowLayout::Dense(0, coefficientWidth));
            for (auto pair : {std::make_pair(&deviceMean, &mean), std::make_pair(&deviceCoefficients, &coefficients)})
            {
                pair.first->Owned().CopyFrom(pair.second->Owned(), context).Wait();
                pair.first->Ghost().CopyFrom(pair.second->Ghost(), context).Wait();
            }
            const auto producerInput = deviceCoefficients.Owned().Clone(context);
            Fill(deviceCoefficients, 0); // The producer restores the real input.
            auto work = resident.BuildWorkspace(profile.components, std::vector<I>{1});
            if (candidate == 0 && profile.fixture == 0 && profile.degree == 1 && profile.components == 5)
            {
                CHECK_FALSE(resident.SupportsKernel(C::ReconstructionDeviceKernel::FP64MatrixShared));
                R::EagerEngine rejected;
                const auto rejectedStream = rejected.CreateStream(std::make_shared<R::CUDALauncher>(0));
                CHECK_THROWS(resident.SubmitReconstructionIterTo(rejected, rejectedStream, deviceMean, deviceCoefficients,
                                                                 work, deviceBoundary, C::ReconstructionRegion::All,
                                                                 C::ReconstructionDeviceKernel::FP64MatrixShared));
            }
            R::EagerEngine engine;
            const auto producer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            const auto compute = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            R::SubmitArrays(engine, producer, [](R::ExecutionContext launchContext, auto source, auto target)
                            {
                                DNDS_check_throw_info(source.kind == R::RowLayoutKind::Table && target.kind == R::RowLayoutKind::Table &&
                                                          source.stride == target.stride && source.Width() == target.Width(),
                                                      "test VR producer requires identical physical pitches");
                                const auto bytes = size_t(source.RowCount() * source.stride) * sizeof(double);
                                if (bytes)
                                    DNDS_check_throw_info(cudaMemcpyAsync(target.data, source.data, bytes, cudaMemcpyDeviceToDevice,
                                                                          static_cast<cudaStream_t>(launchContext.stream)) == cudaSuccess,
                                                          "test VR producer copy failed"); }, R::Read(producerInput), R::Write(deviceCoefficients.Owned()));
            compute.WaitFor(producer.Record());
            // Frozen input, repeated writes: no host completion observation is
            // inserted between these real public queued operator submissions.
            for (int pass = 0; pass < 3; ++pass)
                resident.SubmitReconstructionIterTo(engine, compute, deviceMean, deviceCoefficients, work, deviceBoundary,
                                                    C::ReconstructionRegion::All, kernel);
            engine.Close();
            resident.CheckStatus(work);
            Equal(reference.temporary, work.temporary, candidate == 4 ? 3e-3 : quantized ? 2e-5
                                                                                         : 2e-10);
            if (candidate == 1 && profile.fixture == 2 && profile.degree == 3 && profile.components == 5)
            {
                // Narrow exponent overflow is an explicit failed publication,
                // not a silent fallback or partial overwrite of durable URec.
                Fill(deviceCoefficients, 1e100);
                const auto before = deviceCoefficients.Owned().To(R::Device::Host());
                R::EagerEngine invalid;
                const auto stream = invalid.CreateStream(std::make_shared<R::CUDALauncher>(0));
                resident.SubmitReconstructionIterTo(invalid, stream, deviceMean, deviceCoefficients, work, deviceBoundary,
                                                    C::ReconstructionRegion::All, kernel);
                resident.SubmitCommitReconstruction(invalid, stream, deviceCoefficients, work);
                invalid.Close();
                CHECK_THROWS(resident.CheckStatus(work));
                const auto after = deviceCoefficients.Owned().To(R::Device::Host());
                const auto left = before.Read(), right = after.Read();
                for (I iCell = 0; iCell < before.RowCount(); ++iCell)
                    for (I j = 0; j < deviceCoefficients.Width(); ++j)
                        CHECK(left.View()(iCell, j) == right.View()(iCell, j));
            }
        }
    }
}

TEST_CASE("CUDA VR launch policy rejects a non-warp thread count")
{
    C::ReconstructionSweepLaunch launch;
    launch.threads = 31;
    CHECK_THROWS(launch.Validate());
}

#ifdef DNDS_USE_CCCL_COMPUTE
TEST_CASE("Float-float VR preserves low limbs through a cancelling modal product")
{
    namespace D = C::Detail;
    const auto context = R::ExecutionContext::CUDA(0);
    const auto upload = [context](auto values, I width)
    {
        using T = typename decltype(values)::value_type;
        R::Array<T> host(R::RowLayout::Dense(I(values.size()) / width, width));
        {
            auto write = host.Write();
            for (I i = 0; i < I(values.size()); ++i)
                write.View()(i / width, i % width) = values[i];
        }
        return host.To(context.device, context);
    };
    // Deliberately independent of CFV setup matrices: [1,-1] dotted with
    // [1e8+.25,1e8] must recover .25. Rounding inputs/products to one float
    // instead produces zero, which a relative test on a large state can hide.
    auto offsets = upload(std::vector<I>{0, 1}, 1);
    auto neighbors = upload(std::vector<I>{0}, 1);
    auto boundaryOffsets = upload(std::vector<I>{0, 0}, 1);
    constexpr I modes = 19;
    std::vector<double> entries(modes * modes, 0);
    entries[0] = 1, entries[modes] = -1;
    auto matrix = upload(entries, modes * modes);
    auto vector = upload(std::vector<double>(modes, 0), modes);
    auto injection = upload(std::vector<double>{}, modes);
    auto mean = upload(std::vector<double>(5, 0), 5);
    std::vector<double> values(5 * modes, 0);
    for (I j = 0; j < 5; ++j)
        values[j * modes] = 1e8 + .25, values[j * modes + 1] = 1e8;
    auto coefficients = upload(values, 5 * modes);
    auto output = upload(std::vector<double>(5 * modes, -7), 5 * modes);
    auto status = upload(std::vector<int>{0}, 1);
    {
        auto offsetsRead = offsets.Read(context), neighborsRead = neighbors.Read(context), boundaryRead = boundaryOffsets.Read(context);
        auto matrixRead = matrix.Read(context), vectorRead = vector.Read(context), injectionRead = injection.Read(context);
        auto meanRead = mean.Read(context), coefficientsRead = coefficients.Read(context);
        auto statusWrite = status.Write(context);
        auto result = output.Write(context);
        D::ReconstructionDeviceView plan;
        plan.owned = 1, plan.n = modes, plan.relax = 1;
        plan.offsets = offsetsRead.View(), plan.neighbors = neighborsRead.View(), plan.boundaryOffsets = boundaryRead.View();
        plan.matrixAInvB = matrixRead.View(), plan.vectorAInvB = vectorRead.View(), plan.boundaryInjection = injectionRead.View();
        D::LaunchStagedJacobi(context, plan, {{}, 1, false}, {}, {meanRead.View(), {}}, {coefficientsRead.View(), {}}, {},
                              result.View(), statusWrite.View().ConstView(), {}, D::ReconstructionSweepArithmetic::FloatFloat);
        D::LaunchValidateJacobiResult(context, {{}, 1, false}, result.View().ConstView(), statusWrite.View());
        REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
    }
    const auto host = output.To(R::Device::Host());
    const auto read = host.Read();
    for (I j = 0; j < 5; ++j)
    {
        CHECK(read.View()(0, j * modes) == .25);
        for (I mode = 1; mode < modes; ++mode)
            CHECK(read.View()(0, j * modes + mode) == 0);
    }
    const auto flag = status.To(R::Device::Host());
    CHECK(flag.Read().View()(0, 0) == 0);
}
#endif
