#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#define DNDS_MESH_FACADE_TEST_EMBEDDED
#include "test_MeshFacade.cpp"
#include "LegacyConstReferenceFixture.hpp"
#include "Geom/Unstructured/UnstructuredMeshSnapshot.hpp"
#include "Geom/Unstructured/NativeMeshBuilder.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <atomic>
#include <thread>

__global__ static void LegacyFacadeProbeKernel(const G::UnstructuredMeshLegacyView *descriptor, G::Test::MeshViewProbeResult *output)
{
    if (blockIdx.x || threadIdx.x)
        return;
    const auto &view = *descriptor;
    static_assert(decltype(view.FaceCells(0))::row_width == 2);
    static_assert(decltype(view.BoundaryCells(0))::row_width == 2);
    auto result = G::Test::MeshViewProbe(view);
    // Exercise the legacy member spellings on the device, including metadata
    // needed for ragged rows. The expected probe comes from the host query API.
    const G::MeshLegacyPairRead<I> pairs[]{view.cell2node, view.bnd2node, view.bnd2cell,
                                           view.cell2cell, view.node2cell, view.node2bnd,
                                           view.cell2face, view.face2node, view.face2cell,
                                           view.face2bnd, view.bnd2face, view.cell2edge,
                                           view.edge2node, view.edge2cell, view.cell2cellFace};
    result.relationChecksum = 0;
    for (int relation = 0; relation < int(G::MeshRelation::Count); ++relation)
        if (pairs[relation].isBuilt())
            for (I row = 0; row < pairs[relation].Size(); ++row)
            {
                const auto entries = pairs[relation][row];
                result.relationChecksum += (relation + 1) * (row + 1) * I(pairs[relation].RowSize(row) + 1);
                for (size_t slot = 0; slot < entries.size(); ++slot)
                    result.relationChecksum += (relation + 1) * (row + 1) * I(slot + 1) * G::Test::MeshViewProbeIndex(entries[slot]);
            }
    result.coordinateChecksum = 0;
    for (I node = 0; node < view.coords.Size(); ++node)
        for (int d = 0; d < 3; ++d)
            result.coordinateChecksum += (node + 1) * (d + 1) * view.coords[node][d];
    result.counts[0] = view.coords.father.Size();
    result.counts[1] = view.coords.son->Size();
    result.counts[2] = view.coords.Size();
    result.counts[3] = view.cell2node.father->Size();
    result.counts[4] = view.cell2node.son.Size();
    result.counts[5] = view.cell2node.Size();
    result.counts[6] = view.face2node.father.Size();
    result.counts[7] = view.face2node.son.Size();
    result.counts[8] = view.face2node.Size();
    *output = result;
}

__global__ static void LegacyFacadeParityKernel(const G::UnstructuredMeshLegacyView *descriptor,
                                                G::Test::LegacyConstMeshReference old, int *mismatches)
{
    if (blockIdx.x || threadIdx.x)
        return;
    const auto &view = *descriptor;
    int errors = 0;
    errors += view.dim != 2 || view.isPeriodic != old.isPeriodic;
    errors += view.coords.Size() != old.coords.Size();
    for (I node = 0; node < old.coords.Size(); ++node)
        for (int d = 0; d < 3; ++d)
            errors += view.coords[node][d] != old.coords[node][d];
    errors += view.cell2node.father.Size() != old.cellNodes.father.Size();
    errors += view.cell2node.son.Size() != old.cellNodes.son.Size();
    for (I cell = 0; cell < old.cellNodes.Size(); ++cell)
    {
        errors += view.cellElemInfo(cell, 0).type != old.cellInfo(cell, 0).type;
        errors += view.cellElemInfo(cell, 0).zone != old.cellInfo(cell, 0).zone;
        errors += view.cell2node.RowSize(cell) != old.cellNodes.RowSize(cell);
        for (DNDS::rowsize slot = 0; slot < old.cellNodes.RowSize(cell); ++slot)
        {
            errors += view.cell2node[cell][slot] != old.cellNodes(cell, slot);
            if (old.isPeriodic)
                errors += uint8_t(view.cell2nodePbi(cell, slot)) != uint8_t(old.cellBits(cell, slot));
        }
        errors += view.cell2face.RowSize(cell) != old.cellFaces.RowSize(cell);
        for (DNDS::rowsize slot = 0; slot < old.cellFaces.RowSize(cell); ++slot)
        {
            errors += view.cell2face.father[cell][slot] != old.cellFaces(cell, slot);
            if (old.isPeriodic)
                errors += uint8_t(view.cell2facePbi[cell][slot]) != uint8_t(old.slotBits(cell, slot));
        }
    }
    errors += view.face2cell.Size() != old.faceCells.Size();
    for (I face = 0; face < old.faceCells.Size(); ++face)
    {
        errors += view.face2cell.RowSize(face) != 2;
        errors += view.faceElemInfo(face, 0).type != old.faceInfo(face, 0).type;
        errors += view.faceElemInfo(face, 0).zone != old.faceInfo(face, 0).zone;
        for (DNDS::rowsize side = 0; side < 2; ++side)
        {
            errors += view.face2cell(face, side) != old.faceCells(face, side);
            errors += view.FaceCells(face)[side] != old.faceCells(face, side);
        }
        errors += view.face2node.RowSize(face) != old.faceNodes.RowSize(face);
        for (DNDS::rowsize slot = 0; slot < old.faceNodes.RowSize(face); ++slot)
        {
            errors += view.face2node(face, slot) != old.faceNodes(face, slot);
            if (old.isPeriodic)
                errors += uint8_t(view.face2nodePbi[face][slot]) != uint8_t(old.faceBits(face, slot));
        }
    }
    for (I boundary = 0; boundary < view.bnd2cell.Size(); ++boundary)
    {
        errors += view.BoundaryCells(boundary).size() != 2;
        errors += view.BoundaryCells(boundary)[0] != 0;
        errors += view.BoundaryCells(boundary)[1] != DNDS::UnInitIndex;
    }
    *mismatches += errors;
}

static int FacadeGPUOrdinal()
{
    const auto mpi = FacadeMPI();
    int devices = 0;
    REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
    REQUIRE(devices > 0);
    const char *visible = std::getenv("CUDA_VISIBLE_DEVICES");
    // The external one-rank-per-GPU wrapper verifies unique physical UUIDs.
    // Without visibility isolation, every rank must select its own GPU.
    const bool isolated = devices == 1 && visible && *visible && std::string(visible).find(',') == std::string::npos;
    REQUIRE((isolated || devices >= mpi.size));
    const int ordinal = isolated ? 0 : mpi.rank;
    REQUIRE(cudaSetDevice(ordinal) == cudaSuccess);
    return ordinal;
}

__global__ static void NormalizedCellPairsKernel(G::UnstructuredMeshView view, int *errors)
{
    if (blockIdx.x || threadIdx.x)
        return;
    static_assert(decltype(view.FaceCells(0))::row_width == 2);
    static_assert(decltype(view.BoundaryCells(0))::row_width == 2);
    int count = 0;
    count += view.FaceCells(0)[0] != 0 || view.FaceCells(0)[1] != 0;
    count += view.FaceCells(1)[1] != DNDS::UnInitIndex;
    count += view.FaceCells(3)[0] != -4 || view.FaceCells(3)[1] != -5;
    count += view.FaceCells(4)[0] != -6 || view.FaceCells(4)[1] != -7;
    count += view.FaceCells(4).data() - view.FaceCells(3).data() != 5;
    count += view.BoundaryCells(0)[0] != 0 || view.BoundaryCells(0)[1] != DNDS::UnInitIndex;
    count += view.BoundaryCells(1).data() - view.BoundaryCells(0).data() != 4;
    count += view.BoundaryCells(2)[0] != -4 || view.BoundaryCells(2)[1] != DNDS::UnInitIndex;
    count += view.BoundaryCells(3)[0] != -6 || view.BoundaryCells(3)[1] != DNDS::UnInitIndex;
    count += view.BoundaryCells(3).data() - view.BoundaryCells(2).data() != 6;
    *errors = count;
}

TEST_CASE("CUDA physical pair publication normalizes CSR and distinct padded owned ghost strides")
{
    const int ordinal = FacadeGPUOrdinal();
    const auto context = R::ExecutionContext::CUDA(ordinal);
    auto data = NativeFacadeFixture::Make(FacadeMPI(), true).data;
    auto padded = [](const std::vector<std::vector<I>> &rows, size_t capacity)
    {
        R::Array<I> values(R::RowLayout::Padded(std::vector<size_t>(rows.size(), 2), capacity));
        auto write = values.Write();
        for (size_t row = 0; row < rows.size(); ++row)
            for (size_t slot = 0; slot < 2; ++slot)
                write.View()(row, slot) = rows[row][slot];
        return values;
    };
    data.relations[size_t(G::MeshRelation::FaceCells)] = NativeFacadeFixture::Rows<I>({{0, 0}, {0, DNDS::UnInitIndex}, {0, DNDS::UnInitIndex}});
    data.relations[size_t(G::MeshRelation::FaceCells)]->ghost = padded({{-4, -5}, {-6, -7}}, 5);
    data.relations[size_t(G::MeshRelation::FaceNodes)]->ghost = NativeFacadeFixture::Rows<I>({{0, 1}, {1, 2}}).owned;
    data.relations[size_t(G::MeshRelation::FaceBoundaries)]->ghost = NativeFacadeFixture::Rows<I>({{DNDS::UnInitIndex}, {DNDS::UnInitIndex}}, 1).owned;
    data.faceInfo->ghost = NativeFacadeFixture::Rows<G::ElemInfo>({{{G::Elem::Line2, G::BC_ID_INTERNAL}}, {{G::Elem::Line2, G::BC_ID_INTERNAL}}}, 1).owned;
    data.faceNodesPbi->ghost = NativeFacadeFixture::Rows<G::NodePeriodicBits>({{{}, {}}, {{}, {}}}).owned;
    data.relations[size_t(G::MeshRelation::BoundaryCells)]->owned = padded({{0, DNDS::UnInitIndex}, {0, DNDS::UnInitIndex}}, 4);
    data.relations[size_t(G::MeshRelation::BoundaryCells)]->ghost = padded({{-4, DNDS::UnInitIndex}, {-6, DNDS::UnInitIndex}}, 6);
    data.relations[size_t(G::MeshRelation::BoundaryNodes)]->ghost = NativeFacadeFixture::Rows<I>({{0, 1}, {1, 2}}).owned;
    data.relations[size_t(G::MeshRelation::BoundaryFaces)]->ghost = NativeFacadeFixture::Rows<I>({{DNDS::UnInitIndex}, {DNDS::UnInitIndex}}, 1).owned;
    data.bndInfo->ghost = NativeFacadeFixture::Rows<G::ElemInfo>({{{G::Elem::Line2, 7}}, {{G::Elem::Line2, 7}}}, 1).owned;
    data.bndNodesPbi->ghost = NativeFacadeFixture::Rows<G::NodePeriodicBits>({{{}, {}}, {{}, {}}}).owned;
    auto host = G::NativeMeshSnapshot::FromNative(std::move(data), AllFacadeRequirements());
    auto placed = host.To(R::Device::CUDA(ordinal), context);
    int *deviceErrors = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&deviceErrors), sizeof(int)) == cudaSuccess);
    auto read = placed.AcquireRead(context);
    NormalizedCellPairsKernel<<<1, 1>>>(read.DeviceView(), deviceErrors);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    read.Complete(R::Completion::Record(context));
    read.Wait();
    int errors = -1;
    REQUIRE(cudaMemcpy(&errors, deviceErrors, sizeof(errors), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(errors == 0);
    REQUIRE(cudaFree(deviceErrors) == cudaSuccess);
}

struct FacadeStreamGate
{
    std::atomic<bool> release{false};
    cudaStream_t stream;
    explicit FacadeStreamGate(cudaStream_t value) : stream(value) {}
    static void CUDART_CB Wait(void *opaque)
    {
        auto &gate = *static_cast<FacadeStreamGate *>(opaque);
        while (!gate.release.load(std::memory_order_acquire))
            std::this_thread::yield();
    }
    ~FacadeStreamGate()
    {
        release.store(true, std::memory_order_release);
        // Exception-safe lifetime: the callback must stop touching this object
        // before its atomic storage is destroyed. No CUDA calls in the callback.
        cudaStreamSynchronize(stream);
    }
};

TEST_CASE("Mesh facade CUDA kernels traverse retained snapshots with host parity")
{
    const int ordinal = FacadeGPUOrdinal();
    const auto streamOwner = R::CreateCUDAStream(ordinal);
    const auto stream = R::CUDAStream(R::ExecutionContext::CUDA(ordinal, streamOwner));
    const auto secondStreamOwner = R::CreateCUDAStream(ordinal);
    const auto secondStream = R::CUDAStream(R::ExecutionContext::CUDA(ordinal, secondStreamOwner));
    const auto context = R::ExecutionContext::CUDA(ordinal, streamOwner);
    const auto secondContext = R::ExecutionContext::CUDA(ordinal, secondStreamOwner);
    cudaEvent_t gatePassed = nullptr, secondDone = nullptr;
    REQUIRE(cudaEventCreateWithFlags(&gatePassed, cudaEventDisableTiming) == cudaSuccess);
    REQUIRE(cudaEventCreateWithFlags(&secondDone, cudaEventDisableTiming) == cudaSuccess);
    G::Test::MeshViewProbeResult *output = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&output), sizeof(*output)) == cudaSuccess);
    G::Test::MeshViewProbeResult *secondOutput = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&secondOutput), sizeof(*secondOutput)) == cudaSuccess);
    G::Test::MeshViewProbeResult *actual = nullptr;
    REQUIRE(cudaMallocHost(reinterpret_cast<void **>(&actual), sizeof(*actual)) == cudaSuccess);
    G::Test::MeshViewProbeResult *secondActual = nullptr;
    REQUIRE(cudaMallocHost(reinterpret_cast<void **>(&secondActual), sizeof(*secondActual)) == cudaSuccess);
    G::UnstructuredMeshLegacyView *deviceDescriptors = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&deviceDescriptors), 2 * sizeof(*deviceDescriptors)) == cudaSuccess);
    for (int trial = 0; trial < 6; ++trial)
    {
        const bool twoReaders = trial >= 3;
        const int fixture = trial % 3;
        CAPTURE(twoReaders);
        CAPTURE(fixture);
        const bool periodic = fixture == 1, empty = fixture == 2;
        auto mesh = FacadeFixture(periodic, empty);
        G::Test::MeshViewProbeResult expected;
        {
            auto host = mesh->AcquireRead(AllFacadeRequirements());
            expected = G::Test::MeshViewProbe(host.HostView());
        }
        mesh->PlaceNative(R::Device::CUDA(ordinal), context);
        auto versions = mesh->versions;
        std::optional<G::MeshReadAccess> access, secondAccess;
        access.emplace(mesh->AcquireRead(AllFacadeRequirements(), context));
        if (twoReaders)
        {
            secondAccess.emplace(mesh->AcquireRead(AllFacadeRequirements(), secondContext));
            if (!empty)
                CHECK(access->DeviceView().Coord(0).data() == secondAccess->DeviceView().Coord(0).data());
        }
        const auto descriptor = access->LegacyDeviceView();
        REQUIRE(cudaMemcpyAsync(deviceDescriptors, &descriptor, sizeof(descriptor), cudaMemcpyHostToDevice, stream) == cudaSuccess);
        // Both readers borrow the same immutable placement; the descriptor copy
        // completes before the streams are deliberately gated below.
        REQUIRE(cudaMemcpyAsync(deviceDescriptors + 1, &descriptor, sizeof(descriptor), cudaMemcpyHostToDevice, stream) == cudaSuccess);
        FacadeStreamGate gate(stream);
        CHECK_THROWS(access->HostView());
        CHECK_FALSE(versions.CanMutate());
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Geometry));
        // CUDA lazy module loading can synchronize on the first kernel launch.
        // Load the probe before a host callback deliberately blocks this stream.
        REQUIRE(G::Test::LaunchMeshViewProbe(access->DeviceView(), output, stream) == cudaSuccess);
        LegacyFacadeProbeKernel<<<1, 1, 0, stream>>>(deviceDescriptors, output);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        REQUIRE(cudaLaunchHostFunc(stream, FacadeStreamGate::Wait, &gate) == cudaSuccess);
        REQUIRE(cudaEventRecord(gatePassed, stream) == cudaSuccess);
        LegacyFacadeProbeKernel<<<1, 1, 0, stream>>>(deviceDescriptors, output);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        REQUIRE(cudaMemcpyAsync(actual, output, sizeof(*actual), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
        if (twoReaders)
        {
            // Reacquisition uses the same immutable placement on another stream.
            // Gate both readers, then explicitly join the second reader onto the
            // acquisition stream before publishing its final completion.
            REQUIRE(cudaStreamWaitEvent(secondStream, gatePassed, 0) == cudaSuccess);
            LegacyFacadeProbeKernel<<<1, 1, 0, secondStream>>>(deviceDescriptors + 1, secondOutput);
            REQUIRE(cudaGetLastError() == cudaSuccess);
            REQUIRE(cudaMemcpyAsync(secondActual, secondOutput, sizeof(*secondActual), cudaMemcpyDeviceToHost, secondStream) == cudaSuccess);
            secondAccess->Complete(R::Completion::Record(secondContext));
            REQUIRE(cudaEventRecord(secondDone, secondStream) == cudaSuccess);
            REQUIRE(cudaStreamWaitEvent(stream, secondDone, 0) == cudaSuccess);
            CHECK(cudaStreamQuery(secondStream) == cudaErrorNotReady);
        }
        access->Complete(R::Completion::Record(context));
        CHECK_THROWS(access->DeviceView());
        CHECK_THROWS(access->LegacyDeviceView());
        // Drop the source and its native authority owners while both streams are
        // blocked. Read scopes retain CUDA values, shape and source read leases.
        CHECK(cudaStreamQuery(stream) == cudaErrorNotReady);
        mesh.reset();
        CHECK_FALSE(versions.CanMutate());
        gate.release.store(true, std::memory_order_release);
        access->Wait();
        CHECK_THROWS(access->Complete(R::Completion::Record(context)));
        CHECK_THROWS(access->DeviceView());
        CheckProbeEqual(*actual, expected);
        if (twoReaders)
        {
            secondAccess->Wait();
            CHECK_THROWS(secondAccess->DeviceView());
            CheckProbeEqual(*secondActual, expected);
        }
        CHECK_FALSE(versions.CanMutate());
        access.reset();
        if (twoReaders)
            CHECK_FALSE(versions.CanMutate());
        secondAccess.reset();
        CHECK(versions.CanMutate());
    }
    REQUIRE(cudaFree(deviceDescriptors) == cudaSuccess);
    REQUIRE(cudaFreeHost(secondActual) == cudaSuccess);
    REQUIRE(cudaFreeHost(actual) == cudaSuccess);
    REQUIRE(cudaFree(secondOutput) == cudaSuccess);
    REQUIRE(cudaFree(output) == cudaSuccess);
    REQUIRE(cudaEventDestroy(secondDone) == cudaSuccess);
    REQUIRE(cudaEventDestroy(gatePassed) == cudaSuccess);
}

TEST_CASE("Moved mesh CUDA access can outlive its destroyed acquisition stream")
{
    const int ordinal = FacadeGPUOrdinal();
    const auto streamOwner = R::CreateCUDAStream(ordinal);
    const auto stream = R::CUDAStream(R::ExecutionContext::CUDA(ordinal, streamOwner));
    auto mesh = FacadeFixture();
    mesh->PlaceNative(R::Device::CUDA(ordinal), R::ExecutionContext::CUDA(ordinal, streamOwner));
    std::optional<G::MeshReadAccess> movedFrom;
    movedFrom.emplace(mesh->AcquireRead(G::MeshReadRequirements::CellFlux(), R::ExecutionContext::CUDA(ordinal, streamOwner)));
    {
        G::MeshReadAccess destination(std::move(*movedFrom));
        CHECK_THROWS(movedFrom->DeviceView());
        destination.Complete(R::Completion::Record(R::ExecutionContext::CUDA(ordinal, streamOwner)));
        CHECK_THROWS(destination.DeviceView());
        destination.Wait();
        CHECK_THROWS(destination.DeviceView());
    }
    CHECK(mesh->versions.CanMutate());
    movedFrom.reset(); // Must not record an event on the destroyed stream.
    CHECK(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("Mesh facade CUDA completion validates the acquisition stream")
{
    const int ordinal = FacadeGPUOrdinal();
    const auto streamOwner = R::CreateCUDAStream(ordinal);
    const auto stream = R::CUDAStream(R::ExecutionContext::CUDA(ordinal, streamOwner));
    auto mesh = FacadeFixture(true);
    mesh->PlaceNative(R::Device::CUDA(ordinal), R::ExecutionContext::CUDA(ordinal, streamOwner));
    {
        auto access = mesh->AcquireRead(AllFacadeRequirements(), R::ExecutionContext::CUDA(ordinal, streamOwner));
        CHECK_THROWS(access.Complete(R::Completion::Record(R::ExecutionContext::CUDA(ordinal))));
        access.Complete(R::Completion::Record(R::ExecutionContext::CUDA(ordinal, streamOwner)));
        access.Wait();
    }
    CHECK(mesh->versions.CanMutate());
}

TEST_CASE("Native mesh CUDA queries match existing const legacy pair views")
{
    const int ordinal = FacadeGPUOrdinal();
    const auto streamOwner = R::CreateCUDAStream(ordinal);
    const auto stream = R::CUDAStream(R::ExecutionContext::CUDA(ordinal, streamOwner));
    int *deviceErrors = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&deviceErrors), sizeof(int)) == cudaSuccess);
    G::UnstructuredMeshLegacyView *deviceDescriptor = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&deviceDescriptor), sizeof(*deviceDescriptor)) == cudaSuccess);
    for (bool periodic : {false, true})
    {
        auto native = FacadeFixture(periodic);
        auto legacy = LegacyConstReferenceFixture(periodic);
        native->PlaceNative(R::Device::CUDA(ordinal), R::ExecutionContext::CUDA(ordinal, streamOwner));
        // Independent legacy allocations use their established device copies.
        // Reference acquisition is const; no legacy implementation is altered.
        legacy->coords.to_device(DNDS::DeviceBackend::CUDA);
        legacy->cell2node.to_device(DNDS::DeviceBackend::CUDA);
        legacy->face2node.to_device(DNDS::DeviceBackend::CUDA);
        legacy->cell2face.to_device(DNDS::DeviceBackend::CUDA);
        legacy->face2cell.to_device(DNDS::DeviceBackend::CUDA);
        legacy->cellElemInfo.to_device(DNDS::DeviceBackend::CUDA);
        legacy->faceElemInfo.to_device(DNDS::DeviceBackend::CUDA);
        legacy->cell2nodePbi.to_device(DNDS::DeviceBackend::CUDA);
        legacy->face2nodePbi.to_device(DNDS::DeviceBackend::CUDA);
        legacy->cell2facePbi.to_device(DNDS::DeviceBackend::CUDA);
        const auto &reference = std::as_const(*legacy);
        auto raggedReference = [](const auto &pair)
        {
            using Base = DNDS::ParArray<I, DNDS::NonUniformSize>;
            using View = DNDS::ArrayPair<Base>::t_deviceViewConst<DNDS::DeviceBackend::CUDA>;
            return View{static_cast<const Base &>(*pair.father).template deviceView<DNDS::DeviceBackend::CUDA>(),
                        static_cast<const Base &>(*pair.son).template deviceView<DNDS::DeviceBackend::CUDA>()};
        };
        using FixedBase = DNDS::ParArray<I, 2>;
        using FixedView = DNDS::ArrayPair<FixedBase>::t_deviceViewConst<DNDS::DeviceBackend::CUDA>;
        G::Test::LegacyConstMeshReference old{
            reference.coords.template deviceView<DNDS::DeviceBackend::CUDA>(),
            raggedReference(reference.cell2node),
            raggedReference(reference.face2node),
            raggedReference(reference.cell2face),
            FixedView{static_cast<const FixedBase &>(*reference.face2cell.father).template deviceView<DNDS::DeviceBackend::CUDA>(),
                      static_cast<const FixedBase &>(*reference.face2cell.son).template deviceView<DNDS::DeviceBackend::CUDA>()},
            reference.cellElemInfo.template deviceView<DNDS::DeviceBackend::CUDA>(),
            reference.faceElemInfo.template deviceView<DNDS::DeviceBackend::CUDA>(),
            reference.cell2nodePbi.template deviceView<DNDS::DeviceBackend::CUDA>(),
            reference.face2nodePbi.template deviceView<DNDS::DeviceBackend::CUDA>(),
            reference.cell2facePbi.template deviceView<DNDS::DeviceBackend::CUDA>(),
            reference.periodicInfo, periodic};
        auto access = native->AcquireLegacyRead(G::MeshReadRequirements::CellFlux(), R::ExecutionContext::CUDA(ordinal, streamOwner));
        CHECK_THROWS(access.HostView());
        CHECK_THROWS(access.operator->());
        const auto &descriptor = access.DeviceView();
        REQUIRE(cudaMemcpyAsync(deviceDescriptor, &descriptor, sizeof(descriptor), cudaMemcpyHostToDevice, stream) == cudaSuccess);
        REQUIRE(G::Test::LaunchLegacyMeshParity(access.DeviceView(), old, deviceErrors, stream) == cudaSuccess);
        LegacyFacadeParityKernel<<<1, 1, 0, stream>>>(deviceDescriptor, old, deviceErrors);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        access.Complete(R::Completion::Record(R::ExecutionContext::CUDA(ordinal, streamOwner)));
        CHECK_THROWS(access.DeviceView());
        access.Wait();
        int errors = -1;
        REQUIRE(cudaMemcpy(&errors, deviceErrors, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(errors == 0);
    }
    REQUIRE(cudaFree(deviceDescriptor) == cudaSuccess);
    REQUIRE(cudaFree(deviceErrors) == cudaSuccess);
}

TEST_CASE("Detached native CUDA mesh backing is placed once and reused by reads")
{
    const int ordinal = FacadeGPUOrdinal();
    const auto streamOwner = R::CreateCUDAStream(ordinal);
    const auto stream = R::CUDAStream(R::ExecutionContext::CUDA(ordinal, streamOwner));
    const auto context = R::ExecutionContext::CUDA(ordinal, streamOwner);
    G::Test::MeshViewProbeResult *output = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&output), sizeof(*output)) == cudaSuccess);
    auto mesh = FacadeFixture(true);
    auto host = G::NativeMeshSnapshot::Capture(*mesh, AllFacadeRequirements());
    const auto expected = G::Test::MeshViewProbe(host.AcquireRead().HostView());
    std::optional<G::NativeMeshSnapshot> placed;
    placed.emplace(host.To(R::Device::CUDA(ordinal), context));
    mesh.reset();
    const DNDS::real *firstCoordinateAllocation = nullptr;
    for (int repeat = 0; repeat < 3; ++repeat)
    {
        auto access = placed->AcquireRead(context);
        auto view = access.DeviceView();
        // Coordinates have a fixed table layout: this obtains only its device
        // pointer; no device values or CSR metadata are read by the host.
        const auto allocation = view.Coord(0).data();
        if (!repeat)
            firstCoordinateAllocation = allocation;
        CHECK(allocation == firstCoordinateAllocation);
        if (repeat == 2)
            placed.reset(); // Scope alone retains persistent backing.
        REQUIRE(G::Test::LaunchMeshViewProbe(view, output, stream) == cudaSuccess);
        access.Complete(R::Completion::Record(context));
        access.Wait();
        G::Test::MeshViewProbeResult actual;
        REQUIRE(cudaMemcpy(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess);
        CheckProbeEqual(actual, expected);
    }
    REQUIRE(cudaFree(output) == cudaSuccess);
}

namespace
{
    R::Array<I> NativeBuilderCUDARows(const std::vector<std::vector<I>> &rows)
    {
        R::RowBuilder builder(rows.size(), sizeof(I));
        for (size_t row = 0; row < rows.size(); ++row)
        {
            builder.ResizeRow(row, rows[row].size(), false);
            for (size_t slot = 0; slot < rows[row].size(); ++slot)
                builder.Row<I>(row)[slot] = rows[row][slot];
        }
        return R::Array<I>::FromSharedStorage(builder.Commit());
    }

    G::NativeMeshInput NativeBuilderCUDAInput(const DNDS::MPIInfo &mpi)
    {
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, 4);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, 2);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
        const I next = (mpi.rank + 1) % mpi.size, offset = 4 * next;
        // Two triangles share an edge, while their nodes belong to the next
        // rank. Complete node stars also bring the previous rank's cells in.
        G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, NativeBuilderCUDARows({{offset, offset + 1, offset + 2}, {offset + 1, offset + 3, offset + 2}}), NativeBuilderCUDARows({}), R::Array<DNDS::real>(R::RowLayout::Dense(4, 3)), R::Array<G::ElemInfo>(R::RowLayout::Dense(2, 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1)), {}};
        {
            auto coords = input.coordinates.Write();
            for (size_t row = 0; row < 4; ++row)
            {
                coords.View()(row, 0) = DNDS::real(4 * mpi.rank + I(row));
                coords.View()(row, 1) = DNDS::real(row * row);
                coords.View()(row, 2) = 0;
            }
            auto info = input.cellInfo.Write();
            for (size_t row = 0; row < 2; ++row)
                info.View()(row, 0) = G::ElemInfo{G::Elem::Tri3, G::INTERNAL_ZONE};
        }
        return input;
    }

    __device__ DNDS::real NativeBuilderCellValue(G::UnstructuredMeshView mesh, I cell)
    {
        DNDS::real value = 0;
        for (auto node : mesh.CellNodes(cell))
        {
            const auto coordinates = mesh.Coord(node);
            value += coordinates[0] + 2 * coordinates[1] + 3 * coordinates[2];
        }
        return value;
    }

    __global__ void NativeBuilderStencil(G::UnstructuredMeshView mesh, R::ArrayView<DNDS::real> output)
    {
        const I cell = I(blockIdx.x * blockDim.x + threadIdx.x);
        if (cell >= mesh.NumCellProc())
            return;
        auto value = NativeBuilderCellValue(mesh, cell);
        for (auto neighbor : mesh.CellCells(cell))
            value += 2 * NativeBuilderCellValue(mesh, neighbor);
        output(size_t(cell), 0) = value;
    }
}

TEST_CASE("Native builder CUDA stencil retains independent arrays through pending completion")
{
    const auto mpi = FacadeMPI();
    REQUIRE(mpi.size <= 2);
    const int ordinal = FacadeGPUOrdinal();
    const auto streamOwner = R::CreateCUDAStream(ordinal);
    const auto stream = R::CUDAStream(R::ExecutionContext::CUDA(ordinal, streamOwner));
    const auto device = R::Device::CUDA(ordinal);
    const auto context = R::ExecutionContext::CUDA(ordinal, streamOwner);
    std::optional<G::NativeMeshInput> input;
    input.emplace(NativeBuilderCUDAInput(mpi));
    G::NativeMeshBuildOptions options;
    options.completeNodeStar = true;
    auto state = G::ConstructNativeMesh(*input, options);
    auto placed = state->WithPlacement(device, context);
    // A placement adds device values and mirrored CSR/size metadata. Its host
    // shape metadata, native named registry, IDs and ancestry remain shared.
    size_t deviceBytes = size_t(state->NumOwned(G::EntityKind::Node) + state->NumGhost(G::EntityKind::Node)) * 3 * sizeof(DNDS::real) +
                         size_t(state->NumOwned(G::EntityKind::Cell) + state->NumGhost(G::EntityKind::Cell)) * sizeof(G::ElemInfo);
    const auto hostView = state->BorrowHostView();
    const auto cellsOnRank = state->Entities(G::EntityKind::Cell);
    const auto nodesOnRank = state->Entities(G::EntityKind::Node);
    const auto boundariesOnRank = state->Entities(G::EntityKind::Bnd);
    for (size_t slot = 0; slot < size_t(G::MeshRelation::Count); ++slot)
    {
        const auto relation = G::MeshRelation(slot);
        if (!hostView.HasRelation(relation))
            continue;
        const auto &space = relation == G::MeshRelation::NodeCells || relation == G::MeshRelation::NodeBoundaries ? nodesOnRank : relation == G::MeshRelation::BoundaryNodes || relation == G::MeshRelation::BoundaryCells ? boundariesOnRank
                                                                                                                                                                                                                           : cellsOnRank;
        size_t elements = 0;
        for (I row = 0; row < space.NumLocal(); ++row)
            elements += hostView.Relation(relation, row).size();
        // This primary builder fixture publishes CSR for all relations except
        // the fixed two-column physical boundary-cell slot (empty here).
        deviceBytes += elements * sizeof(I);
        if (relation != G::MeshRelation::BoundaryCells)
            deviceBytes += size_t(space.NumLocal() + 2) * sizeof(size_t);
    }
    CHECK(placed->ArrayBytes() == state->ArrayBytes() + deviceBytes);
    {
        auto samePlacement = placed->WithPlacement(device, context);
        CHECK(samePlacement->ArrayBytes() == placed->ArrayBytes());
    }
    auto mesh = G::UnstructuredMesh::FromNative(mpi, placed);
    I globalBytes = I(placed->ArrayBytes());
    DNDS::MPI::AllreduceOneIndex(globalBytes, MPI_SUM, mpi);
    CHECK(mesh->getArrayBytes() == globalBytes);
    const auto cells = mesh->Entities(G::EntityKind::Cell);
    CHECK(cells.NumGhost() == (mpi.size > 1 ? 2 : 0));
    CHECK(mesh->NumNodeGhost() == (mpi.size > 1 ? 4 : 0));
    std::vector<DNDS::real> expected(size_t(cells.NumLocal()));
    for (I row = 0; row < cells.NumLocal(); ++row)
    {
        const auto global = cells.LocalToGlobal(row);
        const auto nodeOwner = (global / 2 + 1) % mpi.size;
        // Independent exact oracle: triangle coordinate sums are 12*r+13
        // and 12*r+34; each stencil adds twice the other triangle's sum.
        expected[size_t(row)] = DNDS::real(36 * nodeOwner + (global % 2 ? 60 : 81));
    }
    R::Array<DNDS::real> output(R::RowLayout::Dense(expected.size(), 1), device);
    std::optional<G::MeshReadAccess> access;
    access.emplace(mesh->AcquireRead(G::MeshReadRequirements::Primary(), context));
    CHECK_THROWS(access->HostView());
    auto write = output.Write(context);
    // Resolve CUDA lazy module loading before deliberately holding the stream.
    NativeBuilderStencil<<<1, 32, 0, stream>>>(access->DeviceView(), write.View());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    {
        FacadeStreamGate gate(stream);
        REQUIRE(cudaLaunchHostFunc(stream, FacadeStreamGate::Wait, &gate) == cudaSuccess);
        NativeBuilderStencil<<<1, 32, 0, stream>>>(access->DeviceView(), write.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        const auto done = R::Completion::Record(context);
        access->Complete(done);
        write.Complete(done);
        CHECK_THROWS(access->DeviceView());
        CHECK(cudaStreamQuery(stream) == cudaErrorNotReady);
        input.reset();
        mesh.reset();
        placed.reset();
        state.reset();
        CHECK(cudaStreamQuery(stream) == cudaErrorNotReady);
        gate.release.store(true, std::memory_order_release);
        access->Wait();
        write.Wait();
    }
    access.reset();
    auto host = output.To(R::Device::Host());
    auto actual = host.Read();
    for (size_t row = 0; row < expected.size(); ++row)
        CHECK(actual.View()(row, 0) == expected[row]);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
