/**
 * @file test_DeviceTransferable.cu
 * @brief Unit tests for DeviceTransferable CRTP mixin on FiniteVolume.
 *
 * Tests CUDA and Host device transfers using the DeviceTransferable
 * mixin that FiniteVolume inherits. Verifies that data survives the
 * host->device->host round-trip by reading metric values through
 * FiniteVolumeDeviceView inside a CUDA kernel and comparing against
 * host-side values.
 *
 * This is a standalone MPI test (not doctest) following the existing
 * CUDA test pattern in app/DNDS/.
 *
 * NOTE: Do NOT "using namespace DNDS;" in .cu files -- DNDS::index
 * conflicts with POSIX index() from <strings.h> under nvcc.
 */

#include "CFV/FiniteVolume.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "DNDS/Device/CUDA_Utils.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace // anonymous -- avoid POSIX index() ambiguity
{
    using DNDS::DeviceBackend;
    using DNDS::MPIInfo;
    using DNDS::ssp;

    // ===================================================================
    // Mesh path helper (same convention as test_Reconstruction.cpp)
    // ===================================================================
    std::string meshPath(const std::string &name)
    {
        std::string f(__FILE__);
        for (int i = 0; i < 4; i++)
        {
            auto pos = f.rfind('/');
            if (pos == std::string::npos)
                pos = f.rfind('\\');
            if (pos != std::string::npos)
                f = f.substr(0, pos);
        }
        return f + "/data/mesh/" + name;
    }

    // ===================================================================
    // Build a simple mesh + FiniteVolume with constructed metrics
    // ===================================================================
    std::pair<ssp<DNDS::Geom::UnstructuredMesh>, ssp<DNDS::CFV::FiniteVolume>>
    buildMeshAndFV(MPIInfo &mpi)
    {
        constexpr int dim = 2;
        auto mesh = std::make_shared<DNDS::Geom::UnstructuredMesh>(mpi, dim);
        DNDS::Geom::UnstructuredMeshSerialRW reader(mesh, 0);

        reader.ReadFromCGNSSerial(meshPath("Uniform_3x3_wall.cgns"));
        reader.Deduplicate1to1Periodic();
        reader.BuildCell2Cell();

        DNDS::Geom::UnstructuredMeshSerialRW::PartitionOptions pOpt;
        pOpt.metisSeed = 42;
        reader.MeshPartitionCell2Cell(pOpt);
        reader.PartitionReorderToMeshCell2Cell();

        mesh->RecoverNode2CellAndNode2Bnd();
        mesh->RecoverCell2CellAndBnd2Cell();
        mesh->BuildGhostPrimary();
        mesh->AdjGlobal2LocalPrimary();
        mesh->InterpolateFace();
        mesh->AssertOnFaces();

        auto fv = std::make_shared<DNDS::CFV::FiniteVolume>(mpi, mesh);

        DNDS::CFV::FiniteVolumeSettings defaultSettings(dim);
        nlohmann::ordered_json j;
        defaultSettings.WriteIntoJson(j);
        j["maxOrder"] = 1;
        j["intOrder"] = 5;
        fv->parseSettings(j);

        fv->SetCellAtrBasic();
        fv->ConstructCellVolume();
        fv->ConstructCellBary();
        fv->ConstructCellCent();
        fv->ConstructCellIntJacobiDet();
        fv->ConstructCellIntPPhysics();
        fv->ConstructCellAlignedHBox();
        fv->ConstructCellMajorHBoxCoordInertia();
        fv->SetFaceAtrBasic();
        fv->ConstructFaceArea();
        fv->ConstructFaceCent();
        fv->ConstructFaceIntJacobiDet();
        fv->ConstructFaceIntPPhysics();
        fv->ConstructFaceUnitNorm();
        fv->ConstructFaceMeanNorm();
        fv->ConstructCellSmoothScale();

        return {mesh, fv};
    }

} // anonymous namespace

// ===================================================================
// CUDA kernel: read cell volumes and barycenters from device view,
// write into output buffers for host-side verification.
// ===================================================================
DNDS_GLOBAL void kernel_read_fv_metrics(
    DNDS::CFV::FiniteVolume::t_deviceView<DeviceBackend::CUDA> *fv_view,
    DNDS::real *out_volumes,
    DNDS::real *out_bary_x,
    DNDS::real *out_bary_y,
    DNDS::real *out_face_areas,
    DNDS::index numCells,
    DNDS::index numFaces)
{
    DNDS::index tid = (DNDS::index)blockIdx.x * (DNDS::index)blockDim.x + (DNDS::index)threadIdx.x;

    if (tid < numCells)
    {
        out_volumes[tid] = fv_view->GetCellVol(tid);
        auto bary = fv_view->GetCellBary(tid);
        out_bary_x[tid] = bary(0);
        out_bary_y[tid] = bary(1);
    }

    if (tid < numFaces)
    {
        out_face_areas[tid] = fv_view->GetFaceArea(tid);
    }
}

namespace // anonymous for test functions
{
    // ===================================================================
    // Test: Device state tracking via DeviceTransferable mixin
    // ===================================================================
    void test_device_state_tracking(DNDS::CFV::FiniteVolume &fv, MPIInfo &mpi)
    {
        if (mpi.rank == 0)
            std::cout << "  test_device_state_tracking... " << std::flush;

        auto B0 = fv.device();
        DNDS_assert_info(B0 == DeviceBackend::Unknown,
                         "Expected Unknown before any transfer");

        DNDS::index bytes0 = fv.getDeviceArrayBytes();
        DNDS_assert_info(bytes0 > 0, "Expected positive array bytes");

        fv.to_device(DeviceBackend::CUDA);
        auto B1 = fv.device();
        DNDS_assert_info(B1 == DeviceBackend::CUDA,
                         "Expected CUDA after to_device(CUDA)");

        DNDS::index bytes1 = fv.getDeviceArrayBytes();
        DNDS_assert_info(bytes1 > 0, "Expected positive array bytes on CUDA");

        fv.to_host();
        auto B2 = fv.device();
        DNDS_assert_info(B2 == DeviceBackend::Unknown,
                         "Expected Unknown after to_host()");

        if (mpi.rank == 0)
            std::cout << "PASSED (bytes=" << bytes0 << ")" << std::endl;
    }

    // ===================================================================
    // Test: Host device transfer preserves data access
    // ===================================================================
    void test_host_device_transfer(DNDS::CFV::FiniteVolume &fv, MPIInfo &mpi)
    {
        if (mpi.rank == 0)
            std::cout << "  test_host_device_transfer... " << std::flush;

        DNDS::index nCells = fv.mesh->NumCellProc();
        std::vector<DNDS::real> vol_before(nCells);
        for (DNDS::index i = 0; i < nCells; i++)
            vol_before[i] = fv.GetCellVol(i);

        fv.to_device(DeviceBackend::Host);
        auto B = fv.device();
        DNDS_assert_info(B == DeviceBackend::Host,
                         "Expected Host after to_device(Host)");

        for (DNDS::index i = 0; i < nCells; i++)
        {
            DNDS_assert_info(fv.GetCellVol(i) == vol_before[i],
                             "Cell volume changed after Host transfer");
        }

        fv.to_host();

        for (DNDS::index i = 0; i < nCells; i++)
        {
            DNDS_assert_info(fv.GetCellVol(i) == vol_before[i],
                             "Cell volume changed after to_host()");
        }

        if (mpi.rank == 0)
            std::cout << "PASSED" << std::endl;
    }

    // ===================================================================
    // Test: CUDA round-trip -- transfer FV to device, read in kernel,
    // transfer back, verify values match.
    // ===================================================================
    void test_cuda_round_trip(DNDS::CFV::FiniteVolume &fv, MPIInfo &mpi)
    {
        if (mpi.rank == 0)
            std::cout << "  test_cuda_round_trip... " << std::flush;

        DNDS::index nCells = fv.mesh->NumCellProc();
        DNDS::index nFaces = fv.mesh->NumFaceProc();

        // 1) Record host-side metric values
        std::vector<DNDS::real> host_volumes(nCells);
        std::vector<DNDS::real> host_bary_x(nCells);
        std::vector<DNDS::real> host_bary_y(nCells);
        std::vector<DNDS::real> host_face_areas(nFaces);

        for (DNDS::index i = 0; i < nCells; i++)
        {
            host_volumes[i] = fv.GetCellVol(i);
            auto bary = fv.GetCellBary(i);
            host_bary_x[i] = bary(0);
            host_bary_y[i] = bary(1);
        }
        for (DNDS::index i = 0; i < nFaces; i++)
            host_face_areas[i] = fv.GetFaceArea(i);

        // 2) Transfer mesh + FV to CUDA
        fv.mesh->to_device(DeviceBackend::CUDA);
        fv.to_device(DeviceBackend::CUDA);

        DNDS_assert_info(fv.device() == DeviceBackend::CUDA,
                         "Expected CUDA device after to_device");

        // 3) Build device view and copy to device memory
        auto fv_view_host = fv.deviceView<DeviceBackend::CUDA>();
        DNDS::CUDA::DeviceObject<DNDS::CFV::FiniteVolume::t_deviceView<DeviceBackend::CUDA>> fv_dev(fv_view_host);

        // 4) Allocate device output buffers
        DNDS::index maxN = std::max(nCells, nFaces);
        thrust::device_vector<DNDS::real> d_volumes(nCells);
        thrust::device_vector<DNDS::real> d_bary_x(nCells);
        thrust::device_vector<DNDS::real> d_bary_y(nCells);
        thrust::device_vector<DNDS::real> d_face_areas(nFaces);

        // 5) Launch kernel
        uint32_t threadsPerBlock = 256;
        auto [blocksPerGrid, unused] = DNDS::CUDA::calckernelSizeSimple(maxN, threadsPerBlock);
        kernel_read_fv_metrics<<<blocksPerGrid, threadsPerBlock>>>(
            fv_dev.get(),
            thrust::raw_pointer_cast(d_volumes.data()),
            thrust::raw_pointer_cast(d_bary_x.data()),
            thrust::raw_pointer_cast(d_bary_y.data()),
            thrust::raw_pointer_cast(d_face_areas.data()),
            nCells, nFaces);
        DNDS_CUDA_CHECKED(cudaDeviceSynchronize());

        // 6) Copy kernel results back to host
        thrust::host_vector<DNDS::real> h_volumes = d_volumes;
        thrust::host_vector<DNDS::real> h_bary_x = d_bary_x;
        thrust::host_vector<DNDS::real> h_bary_y = d_bary_y;
        thrust::host_vector<DNDS::real> h_face_areas = d_face_areas;

        // 7) Verify: values read on GPU must match host-side values
        for (DNDS::index i = 0; i < nCells; i++)
        {
            DNDS_assert_infof(
                std::abs(h_volumes[i] - host_volumes[i]) < 1e-14,
                "Cell %lld volume mismatch: GPU=%.16e host=%.16e",
                (long long)i, h_volumes[i], host_volumes[i]);
            DNDS_assert_infof(
                std::abs(h_bary_x[i] - host_bary_x[i]) < 1e-14,
                "Cell %lld bary_x mismatch: GPU=%.16e host=%.16e",
                (long long)i, h_bary_x[i], host_bary_x[i]);
            DNDS_assert_infof(
                std::abs(h_bary_y[i] - host_bary_y[i]) < 1e-14,
                "Cell %lld bary_y mismatch: GPU=%.16e host=%.16e",
                (long long)i, h_bary_y[i], host_bary_y[i]);
        }
        for (DNDS::index i = 0; i < nFaces; i++)
        {
            DNDS_assert_infof(
                std::abs(h_face_areas[i] - host_face_areas[i]) < 1e-14,
                "Face %lld area mismatch: GPU=%.16e host=%.16e",
                (long long)i, h_face_areas[i], host_face_areas[i]);
        }

        // 8) Transfer back to host and verify again
        fv.to_host();
        fv.mesh->to_host();

        DNDS_assert_info(fv.device() == DeviceBackend::Unknown,
                         "Expected Unknown after to_host");

        for (DNDS::index i = 0; i < nCells; i++)
        {
            DNDS_assert_infof(
                fv.GetCellVol(i) == host_volumes[i],
                "Cell %lld volume changed after round-trip: now=%.16e was=%.16e",
                (long long)i, fv.GetCellVol(i), host_volumes[i]);
        }

        if (mpi.rank == 0)
            std::cout << "PASSED (cells=" << nCells << ", faces=" << nFaces << ")" << std::endl;
    }

    // ===================================================================
    // Test: FiniteVolumeDeviceView is trivially copyable and host view
    // returns correct values without any device transfer.
    // ===================================================================
    void test_device_view_trivially_copyable(DNDS::CFV::FiniteVolume &fv, MPIInfo &mpi)
    {
        if (mpi.rank == 0)
            std::cout << "  test_device_view_trivially_copyable... " << std::flush;

        static_assert(
            std::is_trivially_copyable_v<DNDS::CFV::FiniteVolume::t_deviceView<DeviceBackend::CUDA>>,
            "FiniteVolumeDeviceView<CUDA> must be trivially copyable");
        static_assert(
            std::is_trivially_copyable_v<DNDS::CFV::FiniteVolume::t_deviceView<DeviceBackend::Host>>,
            "FiniteVolumeDeviceView<Host> must be trivially copyable");

        auto host_view = fv.deviceView<DeviceBackend::Host>();
        DNDS::index nCells = fv.mesh->NumCellProc();
        for (DNDS::index i = 0; i < nCells; i++)
        {
            DNDS_assert_info(
                host_view.GetCellVol(i) == fv.GetCellVol(i),
                "Host deviceView value mismatch");
        }

        if (mpi.rank == 0)
            std::cout << "PASSED" << std::endl;
    }

    // ===================================================================
    // Test: Multiple CUDA round-trips (stability check)
    // ===================================================================
    void test_cuda_multiple_round_trips(DNDS::CFV::FiniteVolume &fv, MPIInfo &mpi)
    {
        if (mpi.rank == 0)
            std::cout << "  test_cuda_multiple_round_trips... " << std::flush;

        DNDS::index nCells = fv.mesh->NumCellProc();
        std::vector<DNDS::real> expected_vols(nCells);
        for (DNDS::index i = 0; i < nCells; i++)
            expected_vols[i] = fv.GetCellVol(i);

        for (int trip = 0; trip < 3; trip++)
        {
            fv.mesh->to_device(DeviceBackend::CUDA);
            fv.to_device(DeviceBackend::CUDA);
            DNDS_assert_info(fv.device() == DeviceBackend::CUDA, "Expected CUDA");

            fv.to_host();
            fv.mesh->to_host();
            DNDS_assert_info(fv.device() == DeviceBackend::Unknown, "Expected Unknown");

            for (DNDS::index i = 0; i < nCells; i++)
            {
                DNDS_assert_infof(
                    fv.GetCellVol(i) == expected_vols[i],
                    "Trip %d cell %lld volume drift: %.16e vs %.16e",
                    trip, (long long)i, fv.GetCellVol(i), expected_vols[i]);
            }
        }

        if (mpi.rank == 0)
            std::cout << "PASSED (3 round-trips)" << std::endl;
    }

} // anonymous namespace

// ===================================================================
// main
// ===================================================================
int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();

    if (mpi.rank == 0)
        std::cout << "=== DeviceTransferable unit tests (CUDA + Host) ===" << std::endl;

    auto [mesh, fv] = buildMeshAndFV(mpi);

    // Compress all mesh CSR arrays once before any device transfer.
    // Compress() is a no-op for non-CSR arrays.
    mesh->op_on_device_arrays([](auto &v)
    {
        if (v.ref.father)
            v.ref.father->Compress();
        if (v.ref.son)
            v.ref.son->Compress();
    });

    if (mpi.rank == 0)
        std::cout << "Mesh built: " << mesh->NumCell() << " cells, "
                  << mesh->NumFace() << " faces" << std::endl;

    test_device_view_trivially_copyable(*fv, mpi);
    test_device_state_tracking(*fv, mpi);
    test_host_device_transfer(*fv, mpi);
    test_cuda_round_trip(*fv, mpi);
    test_cuda_multiple_round_trips(*fv, mpi);

    if (mpi.rank == 0)
        std::cout << "=== All DeviceTransferable tests PASSED ===" << std::endl;

    DNDS::MPI::Finalize();
    return 0;
}
