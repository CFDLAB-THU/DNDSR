#pragma once
#include "CFV/Math/ReconstructionMatrixAccess.hpp"

TEST_CASE("Independent VR matrix layouts and raw face storage preserve oriented distributed sweeps")
{
    struct Profile
    {
        const char *representation, *matrix, *inverse, *vector, *boundary;
        bool triangle, face;
    };
    const Profile profiles[]{
        {"precomputed", "soa", "aos", "aosoa32", "soa", false, false},
        {"precomputed", "aosoa32", "aos", "soa", "aosoa32", false, false},
        {"raw", "aos", "aos", "aos", "aos", false, false},
        {"raw", "aosoa32", "soa", "aos", "soa", true, false},
        {"raw", "soa", "aos", "aosoa32", "aos", false, true},
        {"raw", "aosoa32", "aosoa32", "soa", "aosoa32", true, true}};
    for (int fixture = 0; fixture < 3; ++fixture)
    {
        CAPTURE(fixture);
        std::shared_ptr<G::UnstructuredMesh> mesh;
        if (fixture == 0)
            mesh = CFVTest::Sector(mpi);
        else if (fixture == 1)
            mesh = CFVTest::CartesianHexes(mpi, 2);
        else
        {
            // One logical cell has two translated self faces. A face row must
            // not be deduplicated merely by the pair of persistent cell IDs.
            mesh = G::UnstructuredMesh::FromNative(mpi, G::ConstructNativeMesh(NativePeriodicFixtures::QuadraticTorus(mpi)));
            mesh->PrepareNativeTopology();
        }
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
        C::ReconstructionSettings settings;
        settings.maxOrder = fixture == 1 ? 3 : 2;
        const auto capabilities = C::ReconstructionCapabilities::ForSweeps(false);
        auto oracle = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, capabilities);
        const I components = fixture == 1 ? 7 : 5;
        auto mean = metrics.BuildField(components), frozen = oracle.BuildCoefficients(components);
        Fill(mean, 1), Fill(frozen, .02);
        metrics.BindHalo(mean).Begin().Wait();
        metrics.BindHalo(frozen).Begin().Wait();
        auto boundaryOracle = oracle.BuildBoundaryValues(components);
        {
            auto w = boundaryOracle.values.Write();
            for (I iPoint = 0; iPoint < boundaryOracle.values.RowCount(); ++iPoint)
                for (I j = 0; j < components; ++j)
                    w.View()(iPoint, j) = 1 + .1 * j + .001 * iPoint;
        }
        auto reference = oracle.BuildWorkspace(components, std::vector<I>{1});
        oracle.DoReconstructionIterTo(mean, frozen, reference, boundaryOracle);
        for (const auto &profile : profiles)
        {
            CAPTURE(profile.representation);
            CAPTURE(profile.matrix);
            CAPTURE(profile.triangle);
            CAPTURE(profile.face);
            auto &storage = settings.storage;
            storage.representation = profile.representation;
            storage.matrixLayout = profile.matrix, storage.inverseLayout = profile.inverse;
            storage.vectorLayout = profile.vector, storage.boundaryLayout = profile.boundary;
            storage.inverseTriangle = profile.triangle, storage.faceCoupling = profile.face;
            auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, capabilities);
            auto work = plan.BuildWorkspace(components, std::vector<I>{1});
            auto boundary = plan.BuildBoundaryValues(components);
            boundary.values.CopyFrom(boundaryOracle.values).Wait();
            plan.DoReconstructionIterTo(mean, frozen, work, boundary);
            Equal(reference.temporary, work.temporary, 2e-9);
            // Also run scheduled CPU operands; immediate reads are not allowed
            // to hide a missing raw inverse/face-side dependency declaration.
            R::EagerEngine cpuEngine;
            auto cpuStream = cpuEngine.CreateCPUStream();
            plan.SubmitReconstructionIterTo(cpuEngine, cpuStream, mean, frozen, work, boundary);
            cpuEngine.Synchronize();
            Equal(reference.temporary, work.temporary, 2e-9);
            if (storage.Raw())
                for (const auto &table : plan.StorageReport().tables)
                {
                    if (table.name == "matrixAInvB" || table.name == "vectorAInvB" || table.name == "boundaryInjection")
                        CHECK_FALSE(table.allocated);
                    if (table.name == "localInverse")
                        CHECK(table.payloadBytes == uint64_t(metrics.NumCell() * (profile.triangle ? plan.NRecDOF() * (plan.NRecDOF() + 1) / 2 : plan.NRecDOF() * plan.NRecDOF())) * sizeof(double));
                    if (table.name == "rawMatrixB" && profile.face)
                        CHECK(table.payloadBytes == uint64_t(mesh->NumFaceProc() * plan.NRecDOF() * plan.NRecDOF()) * sizeof(double));
                }
            auto resident = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
            if (storage.Raw())
                CHECK_FALSE(resident.SupportsKernel(C::ReconstructionDeviceKernel::FP64TensorCore));
            CompareSharedSweep(metrics, plan, resident, components);
            // A mixed LVR target tests inverse placement before relaxation and
            // alpha=1 exclusion, in the same raw/physical-layout combinations.
            auto target = plan.BuildCoefficients(components);
            Fill(target, .005);
            R::Array<double> alpha(R::RowLayout::Dense(metrics.NumCell(), 1));
            {
                auto w = alpha.Write();
                for (I iCell = 0; iCell < alpha.RowCount(); ++iCell)
                    w.View()(iCell, 0) = iCell % 3 == 0 ? 1 : iCell % 3 == 1 ? .3
                                                                             : 0;
            }
            plan.DoReconstructionIterLimitedTo(mean, frozen, work, boundary, target, alpha);
            const auto context = R::ExecutionContext::CUDA(0);
            auto dm = metrics.BuildField(R::RowLayout::Dense(0, components), resident.Device());
            auto du = resident.BuildCoefficients(components), dt = resident.BuildCoefficients(components);
            for (auto pair : {std::make_pair(&dm, &mean), std::make_pair(&du, &frozen), std::make_pair(&dt, &target)})
            {
                pair.first->Owned().CopyFrom(pair.second->Owned(), context).Wait();
                pair.first->Ghost().CopyFrom(pair.second->Ghost(), context).Wait();
            }
            auto db = resident.PlaceBoundaryValues(boundary);
            auto da = alpha.To(resident.Device());
            auto deviceWork = resident.BuildWorkspace(components, std::vector<I>{1});
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            resident.SubmitReconstructionIterLimitedTo(engine, stream, dm, du, deviceWork, db, dt, da);
            engine.Synchronize();
            resident.CheckStatus(deviceWork);
            Equal(work.temporary, deviceWork.temporary, 2e-9);
        }
    }
}

TEST_CASE("Face cross blocks transpose fully while only symmetric inverse entries use triangular storage")
{
    R::Array<double> matrix(R::RowLayout::TiledColMajor(1, 4, 32));
    R::Array<I> faces(R::RowLayout::Dense(2, 1));
    R::Array<uint8_t> sides(R::RowLayout::Dense(2, 1));
    {
        auto m = matrix.Write();
        auto f = faces.Write();
        auto s = sides.Write();
        m.View()(0, 0) = 2, m.View()(0, 1) = 7, m.View()(0, 2) = 3, m.View()(0, 3) = 11;
        f.View()(0, 0) = 0, f.View()(1, 0) = 0;
        s.View()(0, 0) = 0, s.View()(1, 0) = 1;
    }
    const auto m = matrix.Read();
    const auto f = faces.Read();
    const auto s = sides.Read();
    CHECK(C::Math::ReconstructionCouplingEntry(m.View(), f.View(), s.View(), 0, 0, 1, 2, true) == 3);
    CHECK(C::Math::ReconstructionCouplingEntry(m.View(), f.View(), s.View(), 1, 0, 1, 2, true) == 7);
    CHECK(C::Math::SymmetricTriangleEntry(0, 1) == C::Math::SymmetricTriangleEntry(1, 0));
}
