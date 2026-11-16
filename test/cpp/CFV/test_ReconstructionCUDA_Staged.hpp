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
                               Profile{0, 3, 5, false, false}, Profile{1, 2, 6, true, false}, Profile{2, 3, 7, true, true}})
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
        for (int candidate = 0; candidate < 6; ++candidate)
        {
            CAPTURE(candidate);
            const bool tensor = candidate >= 4, cellRow = candidate % 2;
            settings.cudaSweep.mapping = candidate < 2 || tensor ? "warp" : "block";
            settings.cudaSweep.buffering = cellRow ? "cellRow" : "neighbor";
            // A complete row deliberately uses one warp/block: shared memory
            // scales with independent warp-cells, especially for 19x7 raw data.
            settings.cudaSweep.threads = cellRow ? (candidate < 2 || tensor ? 32 : 128) : (tensor ? 128 : candidate < 2 ? 64
                                                                                                                        : 96);
            settings.cudaSweep.gridBlocks = 2;
            settings.cudaSweep.asynchronous = true;
            auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, capabilities);
            const auto context = R::ExecutionContext::CUDA(0);
            auto resident = plan.Place(context.device, context);
            const auto kernel = tensor ? C::ReconstructionDeviceKernel::FP64TensorCoreAsync : C::ReconstructionDeviceKernel::FP64Pipelined;
            if (!resident.SupportsKernel(kernel))
            {
                REQUIRE(tensor); // SIMT must run on both supported devices.
                continue;
            }
            auto boundary = plan.BuildBoundaryValues(profile.components);
            boundary.values.CopyFrom(referenceBoundary.values).Wait();
            auto deviceBoundary = resident.PlaceBoundaryValues(boundary);
            auto deviceMean = metrics.BuildField(mean.Owned().Layout(), context.device);
            auto deviceCoefficients = resident.BuildCoefficients(profile.components);
            for (auto pair : {std::make_pair(&deviceMean, &mean), std::make_pair(&deviceCoefficients, &coefficients)})
            {
                pair.first->Owned().CopyFrom(pair.second->Owned(), context).Wait();
                pair.first->Ghost().CopyFrom(pair.second->Ghost(), context).Wait();
            }
            const auto producerInput = deviceCoefficients.Owned().Clone(context);
            Fill(deviceCoefficients, 0); // The producer restores the real input.
            auto work = resident.BuildWorkspace(profile.components, std::vector<I>{1});
            R::EagerEngine engine;
            const auto producer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            const auto compute = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            R::SubmitArrays(engine, producer, [](R::ExecutionContext launchContext, auto source, auto target)
                            {
                                const auto bytes = size_t(source.RowCount() * source.Width()) * sizeof(double);
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
            Equal(reference.temporary, work.temporary, 2e-9);
        }
    }
}

TEST_CASE("CUDA VR launch policy rejects a non-warp thread count")
{
    C::ReconstructionSweepLaunch launch;
    launch.threads = 31;
    CHECK_THROWS(launch.Validate());
}
