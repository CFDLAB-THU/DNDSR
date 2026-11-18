#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "DNDS/Runtime/Communication/NCCLTransfer.hpp"
#include "Geom/Unstructured/UnstructuredMeshSnapshot.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <cstdlib>
#include <optional>
#include "RotatedSectorFixture.hpp"
#include "ReconstructionFixture.hpp"

namespace C = DNDS::CFV;
namespace R = DNDS::Runtime;
namespace G = DNDS::Geom;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
static std::shared_ptr<R::NCCLTransferProvider> nccl;
static R::TransferBinding Halo(const C::VariationalReconstructionPlan &plan, C::FVField &field, R::TransferTransport transport = R::TransferTransport::Direct)
{
    R::TransferOptions options;
    options.provider = nccl;
    return plan.BindHalo(field, nccl ? R::TransferTransport::DevicePacked : transport, options);
}

static void Fill(C::FVField &field, double offset)
{
    auto host = field.Owned().To(R::Device::Host());
    {
        auto write = host.Write();
        for (I iRow = 0; iRow < host.RowCount(); ++iRow)
            for (I iComp = 0; iComp < I(field.Width()); ++iComp)
                write.View()(iRow, iComp) = offset + .01 * (iRow + 7 * mpi.rank + iComp);
    }
    field.Owned().CopyFrom(host, field.Owned().GetDevice() == R::Device::Host() ? R::ExecutionContext::Host() : R::ExecutionContext::CUDA(0)).Wait();
}
static void Equal(const C::FVField &a, const C::FVField &b, double tolerance = 5e-11)
{
    auto host = b.Owned().To(R::Device::Host());
    const auto left = a.Owned().Read(), right = host.Read();
    for (I iRow = 0; iRow < host.RowCount(); ++iRow)
        for (I iComp = 0; iComp < I(a.Width()); ++iComp)
            CHECK(right.View()(iRow, iComp) == doctest::Approx(left.View()(iRow, iComp)).epsilon(tolerance).scale(1));
}

static void CompareSharedSweep(const C::FiniteVolumePlan &metrics, const C::VariationalReconstructionPlan &plan,
                               const C::VariationalReconstructionPlan &resident, I components)
{
    auto mean = metrics.BuildField(components), coefficients = plan.BuildCoefficients(components);
    Fill(mean, 1.0);
    Fill(coefficients, .02);
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(coefficients).Begin().Wait();
    const auto context = R::ExecutionContext::CUDA(0);
    auto gpuMean = metrics.BuildField(mean.Owned().Layout(), resident.Device());
    auto gpuCoefficients = resident.BuildCoefficients(components);
    for (auto pair : {std::make_pair(&gpuMean, &mean), std::make_pair(&gpuCoefficients, &coefficients)})
    {
        pair.first->Owned().CopyFrom(pair.second->Owned(), context).Wait();
        pair.first->Ghost().CopyFrom(pair.second->Ghost(), context).Wait();
    }
    auto boundary = plan.BuildBoundaryValues(components);
    {
        auto write = boundary.values.Write();
        for (I iPoint = 0; iPoint < boundary.values.RowCount(); ++iPoint)
            for (I j = 0; j < components; ++j)
                write.View()(iPoint, j) = 1 + .1 * j + .001 * iPoint;
    }
    auto gpuBoundary = resident.PlaceBoundaryValues(boundary);
    const std::vector<I> vectors{1};
    auto cpu = plan.BuildWorkspace(components, vectors);
    auto gpu = resident.BuildWorkspace(components, vectors);
    const auto originalOwned = coefficients.Owned().Clone();
    const auto originalGhost = coefficients.Ghost().Clone();
    plan.DoReconstructionIter(mean, coefficients, cpu, boundary);
    std::vector<C::ReconstructionDeviceKernel> kernels{C::ReconstructionDeviceKernel::FP64,
                                                       C::ReconstructionDeviceKernel::FP64Pipelined};
    // Comparison policies have one deliberately bounded C=5/M=19 fixture.
    // Other shapes retain their full CPU/general/production sweep coverage.
    if (components == 5 && plan.NRecDOF() == 19 && !plan.Settings().storage.Raw())
        for (auto kernel : {C::ReconstructionDeviceKernel::FP64MatrixShared,
                            C::ReconstructionDeviceKernel::FloatFloatPipelined,
                            C::ReconstructionDeviceKernel::FP64TensorCoreAsync})
            if (resident.SupportsKernel(kernel))
                kernels.push_back(kernel);
    for (auto kernel : kernels)
        for (bool split : {false, true})
        {
            CAPTURE(int(kernel));
            CAPTURE(split);
            REQUIRE(resident.SupportsKernel(kernel));
            gpuCoefficients.Owned().CopyFrom(originalOwned, context).Wait();
            gpuCoefficients.Ghost().CopyFrom(originalGhost, context).Wait();
            if (!split)
                resident.DoReconstructionIter(gpuMean, gpuCoefficients, gpu, gpuBoundary, kernel);
            else
            {
                R::EagerEngine engine;
                const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
                resident.SubmitReconstructionIterTo(engine, stream, gpuMean, gpuCoefficients, gpu, gpuBoundary, C::ReconstructionRegion::Core, kernel);
                resident.SubmitReconstructionIterTo(engine, stream, gpuMean, gpuCoefficients, gpu, gpuBoundary, C::ReconstructionRegion::HaloDependent, kernel);
                resident.SubmitCommitReconstruction(engine, stream, gpuCoefficients, gpu);
                engine.Synchronize();
                resident.CheckStatus(gpu);
            }
            Equal(coefficients, gpuCoefficients);
        }
}

TEST_CASE("Queued FV gradients and linear-mode projection share CPU and CUDA incidence semantics")
{
    struct Profile
    {
        int dimension, components;
        C::GradientMethod method;
        bool sector;
    };
    for (const auto profile : {Profile{2, 5, C::GradientMethod::GreenGauss, false},
                               Profile{2, 6, C::GradientMethod::LeastSquares, true},
                               Profile{3, 7, C::GradientMethod::GGMP, false}})
    {
        INFO("gradient dimension=" << profile.dimension << ", components=" << profile.components << ", sector=" << profile.sector);
        auto mesh = profile.sector ? CFVTest::Sector(mpi) : profile.dimension == 3 ? CFVTest::CartesianHexes(mpi, 3)
                                                                                   : G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24).mesh;
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
        auto gradientPlan = C::GradientPlan::Prepare(*mesh, metrics, profile.method);
        C::ReconstructionSettings settings;
        settings.maxOrder = 3;
        auto reconstruction = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        const I components = profile.components, width = components * profile.dimension;
        auto mean = metrics.BuildField(components), expected = metrics.BuildField(width);
        // Distinct nonconstant means and boundary targets exercise every row
        // and periodic momentum component; legacy CPU analytic tests supply
        // the independent polynomial oracle for these preparation schemes.
        Fill(mean, 1.1);
        metrics.BindHalo(mean).Begin().Wait();
        auto boundary = gradientPlan.BuildBoundaryValues(components);
        {
            auto write = boundary.Write();
            for (I iPoint = 0; iPoint < boundary.RowCount(); ++iPoint)
                for (I j = 0; j < components; ++j)
                    write.View()(iPoint, j) = .3 + .017 * iPoint + .09 * j;
        }
        gradientPlan.Evaluate(mean, boundary, expected, {1});
        auto expectedCoefficients = reconstruction.BuildCoefficients(components);
        gradientPlan.ConvertToCoefficients(expected, reconstruction, expectedCoefficients);
        const auto device = R::Device::CUDA(0), host = R::Device::Host();
        const auto context = R::ExecutionContext::CUDA(0);
        auto deviceMetrics = metrics.Placed(device, context);
        auto resident = gradientPlan.Placed(deviceMetrics, context);
        auto deviceMean = deviceMetrics.BuildField(components), result = deviceMetrics.BuildField(width), recovered = deviceMetrics.BuildField(width);
        auto coefficients = deviceMetrics.BuildField(components * reconstruction.NRecDOF());
        deviceMean.Owned().CopyFrom(mean.Owned(), context).Wait();
        deviceMean.Ghost().CopyFrom(mean.Ghost(), context).Wait();
        auto deviceBoundary = boundary.To(device, context);
        auto workspace = resident.BuildWorkspace(components, {1}, true);
        R::EagerEngine engine;
        const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        {
            // No ghost read is acquired even if a caller has an active write.
            auto ghostWrite = deviceMean.Ghost().Write(context);
            resident.Evaluate(engine, stream, deviceMean, deviceBoundary, result, workspace, C::ReconstructionRegion::Core);
        }
        resident.Evaluate(engine, stream, deviceMean, deviceBoundary, result, workspace, C::ReconstructionRegion::HaloDependent);
        resident.ConvertToCoefficients(engine, stream, result, reconstruction, coefficients);
        resident.ExtractLinearGradients(engine, stream, coefficients, reconstruction, recovered);
        engine.Close();
        Equal(expected, result, 2e-10);
        Equal(expectedCoefficients, coefficients, 2e-10);
        Equal(expected, recovered, 2e-10);
        // Ghost-only extraction uses the cell-geometry offset, not row zero.
        metrics.BindHalo(expectedCoefficients).Begin().Wait();
        coefficients.Ghost().CopyFrom(expectedCoefficients.Ghost(), context).Wait();
        auto expectedGhost = metrics.BuildField(width);
        gradientPlan.ExtractLinearGradients(expectedCoefficients, reconstruction, expectedGhost, true);
        resident.ExtractLinearGradients(coefficients, reconstruction, recovered, true);
        const auto ghost = recovered.Ghost().To(host);
        const auto reference = expectedGhost.Ghost().Read(), actual = ghost.Read();
        for (I iRow = 0; iRow < ghost.RowCount(); ++iRow)
            for (I j = 0; j < width; ++j)
                CHECK(actual.View()(iRow, j) == doctest::Approx(reference.View()(iRow, j)).epsilon(2e-10).scale(1));
        metrics.BindHalo(expected).Begin().Wait();
        result.Ghost().CopyFrom(expected.Ghost(), context).Wait();
        auto expectedLimited = metrics.BuildField(width), limited = deviceMetrics.BuildField(width);
        R::Array<double> activation(R::RowLayout::Dense(metrics.NumCell(), 1));
        {
            auto write = activation.Write();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                write.View()(iCell, 0) = iCell % 2;
        }
        auto deviceActivation = activation.To(device, context);
        for (const I transformed : {I(0), I(5)})
        {
            auto transform = gradientPlan.BuildFaceTransform(0, transformed);
            if (transformed)
            {
                {
                    auto forward = transform.forward.Write(), inverse = transform.inverse.Write();
                    for (I iFace = 0; iFace < metrics.NumFaceProc(); ++iFace)
                    {
                        forward.View()(iFace, transformed) = .3; // (0,1), column-major
                        inverse.View()(iFace, transformed) = -.3;
                    }
                }
                gradientPlan.FinalizeFaceTransform(transform);
            }
            const auto deviceTransform = resident.PlaceFaceTransform(transform, context);
            gradientPlan.LimitWBAP3(expected, expectedLimited, {1}, std::cref(transform), C::ReconstructionRegion::All,
                                    std::cref(activation), .5);
            R::EagerEngine limiterEngine;
            const auto limiterStream = limiterEngine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            {
                auto ghostWrite = result.Ghost().Write(context);
                resident.LimitWBAP3(limiterEngine, limiterStream, result, limited, workspace, std::cref(deviceTransform),
                                    C::ReconstructionRegion::Core, std::cref(deviceActivation), .5);
            }
            resident.LimitWBAP3(limiterEngine, limiterStream, result, limited, workspace, std::cref(deviceTransform),
                                C::ReconstructionRegion::HaloDependent, std::cref(deviceActivation), .5);
            limiterEngine.Close();
            Equal(expectedLimited, limited, 2e-9);
        }
    }
}

TEST_CASE("Queued polynomial sampling covers cached and direct derivatives on owned and ghost rows")
{
    struct Profile
    {
        int dimension, degree, cacheOrder, components;
    };
    for (const auto profile : {Profile{2, 1, -1, 5}, Profile{2, 3, 1, 6}, Profile{3, 3, 3, 7}, Profile{3, 3, -1, 5}})
    {
        INFO("dimension=" << profile.dimension << ", degree=" << profile.degree << ", cache=" << profile.cacheOrder);
        auto mesh = profile.dimension == 2 ? G::OverlapTest::BuildNativeFixture(mpi, false).mesh : CFVTest::CartesianHexes(mpi, 3);
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
        C::ReconstructionSettings settings;
        settings.maxOrder = profile.degree;
        settings.cacheDiffBase = profile.cacheOrder >= 0;
        settings.cacheDiffBaseOrder = std::max(0, profile.cacheOrder);
        auto reconstruction = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        const I modes = reconstruction.NRecDOF(), rows = metrics.NumCellProc();
        R::Array<I> cells(R::RowLayout::Dense(rows, 1));
        R::Array<double> points(R::RowLayout::Dense(rows, 3));
        {
            const auto read = metrics.AcquireRead();
            const auto geometry = read.HostView();
            auto ids = cells.Write();
            auto position = points.Write();
            for (I iCell = 0; iCell < rows; ++iCell)
            {
                ids.View()(iCell, 0) = iCell;
                const auto center = geometry.GetCellCent(iCell);
                for (int d = 0; d < 3; ++d)
                    position.View()(iCell, d) = center[d] + (d < profile.dimension ? .13 * (d + 1) * geometry.cells(iCell, 7) : 0);
            }
        }
        auto mean = metrics.BuildField(profile.components), coefficients = metrics.BuildField(profile.components * modes);
        // Explicitly populate both spaces: this test probes sampler placement
        // and indexing, not the separately tested halo transport.
        for (auto *field : {&mean, &coefficients})
            for (auto *part : {&field->Owned(), &field->Ghost()})
            {
                auto write = part->Write();
                for (I iRow = 0; iRow < part->RowCount(); ++iRow)
                    for (I component = 0; component < I(field->Width()); ++component)
                        write.View()(iRow, component) = .017 * (component + 1) + .003 * (iRow + 1);
            }
        C::PolynomialSamplingPlan sampler(reconstruction, cells, points);
        const auto device = R::Device::CUDA(0);
        const auto context = R::ExecutionContext::CUDA(0);
        auto deviceMean = metrics.BuildField(R::RowLayout::Dense(0, profile.components), device);
        auto deviceCoefficients = metrics.BuildField(R::RowLayout::Dense(0, profile.components * modes), device);
        deviceMean.Owned().CopyFrom(mean.Owned(), context).Wait();
        deviceMean.Ghost().CopyFrom(mean.Ghost(), context).Wait();
        deviceCoefficients.Owned().CopyFrom(coefficients.Owned(), context).Wait();
        deviceCoefficients.Ghost().CopyFrom(coefficients.Ghost(), context).Wait();
        R::Array<double> expected(R::RowLayout::Dense(rows, profile.components));
        R::Array<double> workerResult(R::RowLayout::Dense(rows, profile.components));
        R::Array<double> deviceResult(R::RowLayout::Dense(rows, profile.components), device);
        for (int derivative = 0; derivative <= modes; ++derivative)
        {
            INFO("derivative=" << derivative);
            sampler.Evaluate(mean, coefficients, expected, derivative);
            R::EagerEngine worker(2), gpu;
            const auto workerStream = worker.CreateCPUStream();
            const auto gpuStream = gpu.CreateStream(std::make_shared<R::CUDALauncher>(0));
            std::optional<C::PolynomialSamplingPlan> resident{sampler.Placed(device, context)};
            sampler.Evaluate(worker, workerStream, mean, coefficients, workerResult, derivative);
            resident->Evaluate(gpu, gpuStream, deviceMean, deviceCoefficients, deviceResult, derivative);
            resident.reset(); // Queued views must not depend on the plan handle.
            worker.Close();
            gpu.Close();
            const auto hostResult = deviceResult.To(R::Device::Host());
            const auto a = expected.Read(), b = workerResult.Read(), c = hostResult.Read();
            for (I iRow = 0; iRow < rows; ++iRow)
                for (int component = 0; component < profile.components; ++component)
                {
                    CHECK(b.View()(iRow, component) == a.View()(iRow, component));
                    CHECK(c.View()(iRow, component) == doctest::Approx(a.View()(iRow, component)).epsilon(2e-11).scale(1));
                }
        }
    }
}
__global__ void MetricProbe(C::FiniteVolumeHostView metrics, C::ReconstructionBasisHostView basis,
                            G::UnstructuredMeshView mesh, I iPoint, int side, R::ArrayView<double> result)
{
    if (blockIdx.x || threadIdx.x)
        return;
    result(0, 0) = metrics.GetCellVol(0);
    result(0, 1) = metrics.GetCellCent(0)[0];
    result(0, 2) = basis.FaceMatrix(iPoint, side)[1][1];
    result(0, 3) = mesh.CellNodes(0).size();
    result(0, 4) = mesh.Coord(mesh.CellNodes(0)[0])[0];
}
__global__ void PlainMetricProbe(C::FiniteVolumeHostView metrics, R::ArrayView<double> result)
{
    for (I iCell = I(blockIdx.x) * blockDim.x + threadIdx.x; iCell < result.RowCount(); iCell += I(gridDim.x) * blockDim.x)
    {
        result(iCell, 0) = metrics.GetCellVol(iCell);
        const auto center = metrics.GetCellCent(iCell);
        for (int d = 0; d < 3; ++d)
            result(iCell, 1 + d) = center[d];
    }
}
struct BasisQuery
{
    I cell, point;
    int side;
    std::array<double, 3> position;
};
__global__ void PointBasisProbe(C::ReconstructionPointBasisView basis,
                                R::ArrayView<const BasisQuery, R::AccessBoundsChecks, R::FixedRows<1>> queries,
                                R::ArrayView<double, R::AccessBoundsChecks, R::TableRows> output)
{
    const int modes = basis.polynomial.size - 1;
    for (I iRow = I(blockIdx.x) * blockDim.x + threadIdx.x; iRow < queries.RowCount(); iRow += I(gridDim.x) * blockDim.x)
    {
        const auto query = queries(iRow, 0);
        for (int derivative = 0; derivative < basis.availableDerivatives; ++derivative)
            for (int mode = 0; mode < modes; ++mode)
                output(iRow, derivative * modes + mode) = basis.Evaluate(query.cell, query.point, query.side, query.position, derivative, mode);
    }
}
TEST_CASE("Shared quadrature CUDA placement reuses face allocation and retains queued scatter inputs")
{
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    auto mesh = CFVTest::Sector(mpi);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, 3);
    C::ReconstructionSettings settings;
    settings.maxOrder = 3;
    settings.intOrder = 3;
    settings.SORInstead = false;
    settings.cacheDiffBase = false;
    auto directPlan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    const I modes = directPlan.NRecDOF();
    auto mean = metrics.BuildField(5), coefficients = metrics.BuildField(5 * modes);
    Fill(mean, .7);
    Fill(coefficients, .03);
    std::vector<std::array<I, 2>> queries;
    std::vector<G::tPoint> positions;
    {
        auto scope = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
        const auto topology = scope.HostView();
        auto read = metrics.AcquireRead();
        const auto geometry = read.HostView();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto faces = topology.CellFaces(iCell);
            for (I slot = 0; slot < faces.size(); ++slot)
            {
                const I iFace = faces[slot], iPoint = geometry.FaceQuadratureBegin(iFace);
                const int side = topology.CellIsFaceBack(iCell, iFace, DNDS::rowsize(slot)) ? 0 : 1;
                queries.push_back({iCell, 2 * iPoint + side});
                positions.push_back(C::Detail::FacePointInCell(topology, iFace, side, geometry.GetFacePoint(iPoint)));
            }
        }
    }
    const I count = I(queries.size());
    R::Array<I> cells(R::RowLayout::Dense(count, 1)), rows(R::RowLayout::Dense(count, 1)), scatter(R::RowLayout::Dense(count, 1));
    R::Array<double> points(R::RowLayout::Dense(count, 3)), expected(R::RowLayout::Dense(count, 5));
    R::Array<double> initial(R::RowLayout::Dense(2 * count + 1, 5));
    {
        auto c = cells.Write(), r = rows.Write(), s = scatter.Write();
        auto p = points.Write(), output = initial.Write();
        for (I iPoint = 0; iPoint < count; ++iPoint)
        {
            c.View()(iPoint, 0) = queries[iPoint][0];
            r.View()(iPoint, 0) = queries[iPoint][1];
            s.View()(iPoint, 0) = 2 * (count - 1 - iPoint) + 1;
            for (int d = 0; d < 3; ++d)
                p.View()(iPoint, d) = positions[iPoint][d];
        }
        for (I iRow = 0; iRow < initial.RowCount(); ++iRow)
            for (int j = 0; j < 5; ++j)
                output.View()(iRow, j) = -123;
    }
    C::PolynomialSamplingPlan direct(directPlan, cells, points);
    auto deviceMean = metrics.BuildField(R::RowLayout::Dense(0, 5), device);
    auto deviceCoefficients = metrics.BuildField(R::RowLayout::Dense(0, 5 * modes), device);
    deviceMean.Owned().CopyFrom(mean.Owned(), context).Wait();
    deviceCoefficients.Owned().CopyFrom(coefficients.Owned(), context).Wait();
    for (bool cache : {false, true})
    {
        settings.cacheDiffBase = cache;
        settings.cacheDiffBaseOrder = 1;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        C::ReconstructionResidencyOptions options;
        options.basis = true;
        options.retainHostPayloads = false;
        auto residentPlan = plan.Place(device, context, options);
        for (int derivative : {0, 1, 3})
        {
            direct.Evaluate(mean, coefficients, expected, derivative);
            auto result = initial.To(device, context);
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            {
                C::SamplingPlacementMemo memo(device, context);
                if (cache)
                    memo.Seed(plan.SamplingFaceBasis(), residentPlan.SamplingFaceBasis());
                auto shared = C::PolynomialSamplingPlan::FromSharedBasis(metrics, plan, cells, points,
                                                                         plan.SamplingFaceBasis(), rows, cache ? 3 : 0, true);
                auto segment = shared.WithOutputRows(scatter, initial.RowCount()).Placed(memo);
                if (cache)
                    CHECK(segment.BasisStorage().Identity() == residentPlan.SamplingFaceBasis().Identity());
                else
                    CHECK(segment.BasisStorage().RowCount() == 0);
                segment.Evaluate(engine, stream, deviceMean, deviceCoefficients, result, derivative);
            } // Neither the placement memo nor local sampler owns queued lifetime.
            engine.Close();
            auto hostResult = result.To(R::Device::Host());
            auto output = hostResult.Read(), reference = expected.Read();
            for (I iRow = 0; iRow < initial.RowCount(); ++iRow)
                for (int j = 0; j < 5; ++j)
                {
                    const double wanted = iRow % 2 ? reference.View()(count - 1 - iRow / 2, j) : -123;
                    CHECK(output.View()(iRow, j) == doctest::Approx(wanted).epsilon(2e-11).scale(1));
                }
        }
    }
}

TEST_CASE("Resident face basis supports direct and partial caches without unrelated metric payloads")
{
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    for (int cacheOrder : {-1, 0, 1})
    {
        INFO("cacheOrder=" << cacheOrder);
        auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true);
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
        C::ReconstructionSettings settings;
        settings.maxOrder = 3;
        settings.cacheDiffBase = cacheOrder >= 0;
        settings.cacheDiffBaseOrder = std::max(0, cacheOrder);
        std::optional<C::VariationalReconstructionPlan> plan{C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings)};
        const I modes = plan->NRecDOF();
        std::vector<BasisQuery> queries;
        std::vector<std::vector<double>> expected;
        {
            const auto scope = fixture.mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
            const auto mesh = scope.HostView();
            const auto geometry = metrics.AcquireRead();
            const auto view = geometry.HostView();
            const auto basisRead = plan->AcquireBasisRead();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                for (const I iFace : mesh.CellFaces(iCell))
                {
                    const int side = mesh.FaceCells(iFace)[0] == iCell ? 0 : 1;
                    const I iPoint = view.FaceQuadratureBegin(iFace);
                    const auto point = C::Detail::FacePointInCell(mesh, iFace, side, view.GetFacePoint(iPoint));
                    queries.push_back({iCell, iPoint, side, {point[0], point[1], point[2]}});
                    C::Math::SmallMatrix scratch;
                    const auto matrix = basisRead.FaceBasis(iFace, side, iPoint, 1, scratch);
                    expected.emplace_back(3 * modes);
                    for (int derivative = 0; derivative < 3; ++derivative)
                        for (I mode = 0; mode < modes; ++mode)
                            expected.back()[derivative * modes + mode] = matrix(derivative, mode);
                }
        }
        R::Array<BasisQuery> hostQueries(R::RowLayout::Dense(I(queries.size()), 1));
        {
            auto write = hostQueries.Write();
            for (I iRow = 0; iRow < I(queries.size()); ++iRow)
                write.View()(iRow, 0) = queries[iRow];
        }
        auto residentQueries = hostQueries.To(device, context);
        C::ReconstructionResidencyOptions options;
        options.basis = true;
        options.retainHostPayloads = false;
        auto resident = plan->Place(device, context, options);
        plan.reset();
        auto read = resident.AcquireResidentRead();
        CHECK_THROWS(read.MetricView());
        bool moments = false, cellGeometry = false;
        for (const auto &buffer : resident.ResidentBuffers())
        {
            CHECK(buffer.name != "faceQuadrature");
            CHECK(buffer.name != "radialMoment");
            CHECK(buffer.name != "faceMean");
            moments |= buffer.name == "quadrature/cellMoments";
            cellGeometry |= buffer.name == "quadrature/cellGeometry";
        }
        CHECK(moments == (cacheOrder < 1));
        CHECK(cellGeometry == (cacheOrder < 1));
        R::Array<double> output(R::RowLayout::Dense(I(queries.size()), 3 * modes), device);
        {
            const auto queryRead = residentQueries.Read(context);
            auto write = output.Write(context);
            const auto basis = read.PointBasisView();
            const auto input = queryRead.View<R::FixedRows<1>>();
            const auto result = write.View<R::TableRows>();
            R::ValidateViewDevices(device, basis, input, result);
            R::CheckKernelArgumentBudget<decltype(basis), decltype(input), decltype(result)>();
            PointBasisProbe<<<2, 128>>>(basis, input, result);
            REQUIRE(cudaGetLastError() == cudaSuccess);
            write.Complete(R::Completion::Record(context));
            write.Wait();
        }
        const auto downloaded = output.To(R::Device::Host());
        const auto actual = downloaded.Read();
        for (I iRow = 0; iRow < I(queries.size()); ++iRow)
            for (I j = 0; j < 3 * modes; ++j)
                CHECK(actual.View()(iRow, j) == doctest::Approx(expected[iRow][j]).epsilon(2e-11).scale(1));
        {
            auto mutation = fixture.mesh->versions.BeginMutation(R::Change::Geometry);
            mutation.Commit();
        }
        // Invalidation survives retainHostPayloads=false and the destruction
        // of the host reconstruction owner; device data is not silently reused.
        CHECK_THROWS(resident.Validate());
    }
}
TEST_CASE("Geometry-only FV metrics can live on CUDA without retaining a host metric plan or allocating VR")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, false);
    std::optional<C::FiniteVolumePlan> host{C::FiniteVolumePlan::ConstructMetrics(fixture.mesh)};
    const I cells = host->NumCellProc();
    std::vector<std::array<double, 4>> expected(cells);
    {
        const auto read = host->AcquireRead();
        const auto view = read.View();
        for (I iCell = 0; iCell < cells; ++iCell)
        {
            const auto center = view.GetCellCent(iCell);
            expected[iCell] = {view.GetCellVol(iCell), center[0], center[1], center[2]};
        }
    }
    auto resident = host->Placed(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
    CHECK(resident.Space().Same(host->Space()));
    host.reset();
    auto read = resident.AcquireRead(R::ExecutionContext::CUDA(0));
    const auto view = read.View();
    view.ValidateDevice(R::Device::CUDA(0));
    CHECK_THROWS(read.HostView());
    R::Array<double> output(R::RowLayout::Dense(cells, 4), R::Device::CUDA(0));
    {
        auto writer = output.Write(R::ExecutionContext::CUDA(0));
        const auto result = writer.View();
        R::ValidateViewDevices(R::Device::CUDA(0), view, result);
        R::CheckKernelArgumentBudget<decltype(view), decltype(result)>();
        PlainMetricProbe<<<2, 128>>>(view, result);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        writer.Complete(R::Completion::Record(R::ExecutionContext::CUDA(0)));
        writer.Wait();
    }
    const auto downloaded = output.To(R::Device::Host());
    const auto result = downloaded.Read();
    for (I iCell = 0; iCell < cells; ++iCell)
        for (int j = 0; j < 4; ++j)
            CHECK(result.View()(iCell, j) == expected[iCell][j]);
}
TEST_CASE("Device publications retain actual mesh metrics and basis with explicit optional capabilities")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, false);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics);
    auto resident = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0), {true, true, true});
    auto read = resident.AcquireResidentRead();
    auto mesh = resident.MeshSnapshot().AcquireRead(R::ExecutionContext::CUDA(0));
    auto source = fixture.mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
    const auto view = source.HostView();
    const auto face = view.CellFaces(0)[0];
    const auto parents = view.FaceCells(face);
    const int side = parents[0] == 0 ? 0 : 1;
    auto metricRead = metrics.AcquireRead();
    const auto point = metricRead.HostView().FaceQuadratureBegin(face);
    R::Array<double> result(R::RowLayout::Dense(1, 5), R::Device::CUDA(0));
    {
        auto write = result.Write(R::ExecutionContext::CUDA(0));
        MetricProbe<<<1, 1>>>(read.MetricView(), read.BasisView(), mesh.DeviceView(), point, side, write.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        write.Complete(R::Completion::Record(R::ExecutionContext::CUDA(0)));
        write.Wait();
    }
    const auto host = result.To(R::Device::Host());
    const auto values = host.Read();
    CHECK(values.View()(0, 0) == metrics.GetCellVol(0));
    CHECK(values.View()(0, 1) == metrics.GetCellCent(0)[0]);
    CHECK(values.View()(0, 2) == plan.GetCachedFaceBasis(face, side, 0)(1, 1));
    CHECK(values.View()(0, 3) == view.CellNodes(0).size());
    CHECK(values.View()(0, 4) == view.Coord(view.CellNodes(0)[0])[0]);
    CHECK(resident.ResidentBuffers().size() >= 14);
    auto minimal = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
    CHECK_THROWS(minimal.AcquireResidentRead().MetricView());
    CHECK_THROWS(minimal.MeshSnapshot());
}
TEST_CASE("Device-only reconstruction placement releases host payloads and remains executable" * doctest::test_suite("halo"))
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, false);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    std::optional<C::VariationalReconstructionPlan> resident;
    std::optional<C::FVField> mean;
    std::optional<C::FaceCharacteristicTransform> transform;
    C::ReconstructionBoundaryValues boundary;
    {
        auto host = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics);
        auto hostBoundary = host.BuildBoundaryValues(5);
        {
            auto write = hostBoundary.values.Write();
            for (I iRow = 0; iRow < hostBoundary.values.RowCount(); ++iRow)
                for (I iComp = 0; iComp < 5; ++iComp)
                    write.View()(iRow, iComp) = 1 + .01 * iComp;
        }
        C::ReconstructionResidencyOptions options;
        options.retainHostPayloads = false;
        resident.emplace(host.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0), options));
        boundary = resident->PlaceBoundaryValues(hostBoundary);
        transform.emplace(resident->PlaceFaceTransform(host.BuildFaceTransform(0, 5)));
        mean.emplace(metrics.BuildField(R::RowLayout::ColMajor(0, 5, {32, 1}), resident->Device()));
        Fill(*mean, 1.0);
    }
    REQUIRE(resident);
    REQUIRE(mean);
    CHECK_FALSE(resident->RetainsHostPayloads());
    CHECK_THROWS(resident->Metrics());
    resident->Validate();
    CHECK(resident->Settings().maxOrder == 2);
    CHECK(resident->Capabilities().Supports(C::ReconstructionOperation::Sweep));
    CHECK(resident->StorageReport().retainedAllocationBytes > 0);
    auto coefficients = resident->BuildCoefficients(5, R::RowLayout::TiledColMajor(0, 5 * resident->NRecDOF(), 8));
    auto work = resident->BuildWorkspace(5, coefficients.Owned().Layout());
    Fill(coefficients, .02);
    Halo(*resident, *mean).Begin(R::ExecutionContext::CUDA(0)).Wait();
    auto coefficientHalo = Halo(*resident, coefficients, R::TransferTransport::DevicePacked);
    for (int step = 0; step < 2; ++step)
    {
        coefficientHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
        resident->DoReconstructionIter(*mean, coefficients, work, boundary, C::ReconstructionDeviceKernel::FP64);
        resident->CommitReconstruction(coefficients, work);
    }
    resident->DoLimiterWBAP_3Order(coefficients, work, 1, std::cref(*transform));
    resident->CommitReconstruction(coefficients, work);
    resident->CheckStatus(work);
    const auto hostArray = coefficients.Owned().To(R::Device::Host());
    const auto hostValues = hostArray.Read();
    for (I iRow = 0; iRow < hostValues.View().RowCount(); ++iRow)
        for (I iComp = 0; iComp < I(coefficients.Width()); ++iComp)
            CHECK(std::isfinite(hostValues.View()(iRow, iComp)));
}
TEST_CASE("CUDA sweep-only publication omits limiter matrices and scratch")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, false);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    auto host = C::VariationalReconstructionPlan::ConstructRecCoeff(
        metrics, {}, C::ReconstructionCapabilities::ForSweeps(false));
    auto resident = host.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
    CHECK_FALSE(resident.Capabilities().wbap);
    CHECK_FALSE(resident.Capabilities().linearSystem);
    CHECK_FALSE(resident.Capabilities().localPreconditioner);
    for (const auto &buffer : resident.ResidentBuffers())
    {
        CHECK(buffer.name != "secondary");
        CHECK(buffer.name != "neighborRotation");
        CHECK(buffer.name != "neighborRotationActive");
    }
    auto mean = metrics.BuildField(R::RowLayout::Dense(0, 5), resident.Device());
    auto coefficients = resident.BuildCoefficients(5);
    auto work = resident.BuildWorkspace(5);
    CHECK(work.scratch.RowCount() == 0);
    const auto boundary = resident.PlaceBoundaryValues(host.BuildBoundaryValues(5));
    // All zero means/coefs/boundary values need no halo traffic and must be a
    // fixed point, including ranks with owned/ghost mesh entities.
    resident.DoReconstructionIter(mean, coefficients, work, boundary);
    resident.CommitReconstruction(coefficients, work);
    const auto result = coefficients.Owned().To(R::Device::Host());
    const auto read = result.Read();
    for (I iRow = 0; iRow < result.RowCount(); ++iRow)
        for (I j = 0; j < I(coefficients.Width()); ++j)
            CHECK(read.View()(iRow, j) == 0);
    CHECK_THROWS(resident.DoLimiterWBAP_3Order(coefficients, work, 1));
}
TEST_CASE("Opt-in FP32 sweep maps retain FP64 coefficients and avoid duplicate device matrix storage")
{
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    // Quantized comparison policies are fixed to C=5/M=19. Rotated geometry
    // and extended components remain covered by the production FP64 tests.
    {
        const auto mesh = CFVTest::CartesianHexes(mpi, 3);
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
        C::ReconstructionSettings settings;
        settings.maxOrder = 3;
        settings.jacobiRelax = .7;
        auto cpu = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        C::ReconstructionResidencyOptions options;
        options.retainHostPayloads = false;
        auto full = cpu.Place(device, context, options);
        options.matrixPrecision = C::ReconstructionMatrixPrecision::FP32;
        auto reduced = cpu.Place(device, context, options);
        CHECK(reduced.SupportsKernel(C::ReconstructionDeviceKernel::FP32));
        CHECK(reduced.SupportsKernel(C::ReconstructionDeviceKernel::FP32FullAccumulation));
#ifdef DNDS_USE_CCCL_COMPUTE
        CHECK(reduced.SupportsKernel(C::ReconstructionDeviceKernel::FP32FloatFloatAccumulation));
#else
        CHECK_FALSE(reduced.SupportsKernel(C::ReconstructionDeviceKernel::FP32FloatFloatAccumulation));
#endif
        CHECK_FALSE(full.SupportsKernel(C::ReconstructionDeviceKernel::FP32FullAccumulation));
        CHECK_FALSE(reduced.SupportsKernel(C::ReconstructionDeviceKernel::FP64));
        const auto fullBuffers = full.ResidentBuffers(), smallBuffers = reduced.ResidentBuffers();
        for (const std::string name : {"matrixAInvB", "vectorAInvB", "boundaryInjection"})
        {
            const auto original = std::find_if(fullBuffers.begin(), fullBuffers.end(), [&](const auto &buffer)
                                               { return buffer.name == name; });
            const auto quantized = std::find_if(smallBuffers.begin(), smallBuffers.end(), [&](const auto &buffer)
                                                { return buffer.name == name + "32"; });
            REQUIRE(original != fullBuffers.end());
            REQUIRE(quantized != smallBuffers.end());
            CHECK(2 * quantized->bytes == original->bytes);
            CHECK(std::none_of(smallBuffers.begin(), smallBuffers.end(), [&](const auto &buffer)
                               { return buffer.name == name; }));
        }
        const I components = 5;
        auto mean = metrics.BuildField(components);
        auto input = cpu.BuildCoefficients(components);
        Fill(mean, 1.1);
        Fill(input, .01);
        metrics.BindHalo(mean).Begin().Wait();
        auto inputHalo = cpu.BindHalo(input);
        auto work = cpu.BuildWorkspace(components, std::vector<I>{1});
        auto boundary = cpu.BuildBoundaryValues(components);
        {
            auto write = boundary.values.Write();
            for (I iPoint = 0; iPoint < boundary.values.RowCount(); ++iPoint)
                for (I j = 0; j < components; ++j)
                    write.View()(iPoint, j) = 1.1 + .01 * iPoint + .02 * j;
        }
        auto deviceMean = metrics.BuildField(R::RowLayout::Dense(0, components), device);
        auto deviceInput = reduced.BuildCoefficients(components);
        auto deviceWork = reduced.BuildWorkspace(components, std::vector<I>{1});
        auto accumulationWork = reduced.BuildWorkspace(components, std::vector<I>{1});
        auto tensorWork = reduced.BuildWorkspace(components, std::vector<I>{1});
        auto deviceBoundary = reduced.PlaceBoundaryValues(boundary);
        deviceMean.Owned().CopyFrom(mean.Owned(), context).Wait();
        deviceMean.Ghost().CopyFrom(mean.Ghost(), context).Wait();
        R::EagerEngine engine;
        const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        CHECK_THROWS(full.SubmitReconstructionIterTo(engine, stream, deviceMean, deviceInput, deviceWork, deviceBoundary,
                                                     C::ReconstructionRegion::All, C::ReconstructionDeviceKernel::FP32FullAccumulation));
#ifndef DNDS_USE_CCCL_COMPUTE
        CHECK_THROWS(reduced.SubmitReconstructionIterTo(engine, stream, deviceMean, deviceInput, accumulationWork, deviceBoundary,
                                                        C::ReconstructionRegion::All, C::ReconstructionDeviceKernel::FP32FloatFloatAccumulation));
#endif
        // Compare each sweep with exactly the same input. This isolates local
        // arithmetic error instead of mixing it with iterative error growth.
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            inputHalo.Begin().Wait();
            deviceInput.Owned().CopyFrom(input.Owned(), context).Wait();
            deviceInput.Ghost().CopyFrom(input.Ghost(), context).Wait();
            const auto frozenOwned = input.Owned().To(R::Device::Host());
            cpu.DoReconstructionIter(mean, input, work, boundary);
            reduced.SubmitReconstructionIterTo(engine, stream, deviceMean, deviceInput, deviceWork, deviceBoundary,
                                               C::ReconstructionRegion::Core, C::ReconstructionDeviceKernel::FP32);
            reduced.SubmitReconstructionIterTo(engine, stream, deviceMean, deviceInput, deviceWork, deviceBoundary,
                                               C::ReconstructionRegion::HaloDependent, C::ReconstructionDeviceKernel::FP32);
            engine.Synchronize();
            reduced.CheckStatus(deviceWork);
            Equal(work.temporary, deviceWork.temporary, 2e-5);
            for (const auto kernel : {C::ReconstructionDeviceKernel::FP32FullAccumulation,
                                      C::ReconstructionDeviceKernel::FP32FloatFloatAccumulation, C::ReconstructionDeviceKernel::FP32Pipelined})
            {
                if (!reduced.SupportsKernel(kernel))
                    continue;
                CAPTURE(kernel);
                reduced.SubmitReconstructionIterTo(engine, stream, deviceMean, deviceInput, accumulationWork, deviceBoundary,
                                                   C::ReconstructionRegion::Core, kernel);
                reduced.SubmitReconstructionIterTo(engine, stream, deviceMean, deviceInput, accumulationWork, deviceBoundary,
                                                   C::ReconstructionRegion::HaloDependent, kernel);
                engine.Synchronize();
                reduced.CheckStatus(accumulationWork);
                // Same frozen CPU input and P3 oracle as the existing policy:
                // This fixed-shape Hex fixture exercises 3D BC sums.
                // Float-float still has FP32 map quantization error.
                Equal(work.temporary, accumulationWork.temporary, 2e-5);
                // The CPU convenience call commits into input. Compare the
                // device's immutable operand with the pre-call snapshot.
                const auto retained = deviceInput.Owned().To(R::Device::Host());
                const auto before = frozenOwned.Read(), after = retained.Read();
                for (I iCell = 0; iCell < retained.RowCount(); ++iCell)
                    for (I j = 0; j < I(input.Width()); ++j)
                        CHECK(after.View()(iCell, j) == before.View()(iCell, j));
            }
            for (const auto tensorKernel : {C::ReconstructionDeviceKernel::TF32TensorCore, C::ReconstructionDeviceKernel::TF32Pipelined})
                if (reduced.SupportsKernel(tensorKernel))
                {
                    // Deliberately a different arithmetic policy. Compare the
                    // same frozen inputs with the FP64 oracle as well as recording
                    // a stricter FP32 reference above; this is not a finite-only gate.
                    reduced.SubmitReconstructionIterTo(engine, stream, deviceMean, deviceInput, tensorWork, deviceBoundary,
                                                       C::ReconstructionRegion::All, tensorKernel);
                    engine.Synchronize();
                    reduced.CheckStatus(tensorWork);
                    Equal(work.temporary, tensorWork.temporary, 3e-3);
                }
            cpu.CommitReconstruction(input, work);
        }
        auto target = cpu.BuildCoefficients(components), deviceTarget = reduced.BuildCoefficients(components);
        Fill(target, .005);
        deviceTarget.Owned().CopyFrom(target.Owned(), context).Wait();
        R::Array<double> penalty(R::RowLayout::Dense(metrics.NumCell(), 1));
        {
            auto write = penalty.Write();
            for (I iCell = 0; iCell < penalty.RowCount(); ++iCell)
                write.View()(iCell, 0) = iCell % 3 == 0 ? 0 : iCell % 3 == 1 ? .3
                                                                             : 1;
        }
        auto devicePenalty = penalty.To(device);
        inputHalo.Begin().Wait();
        deviceInput.Owned().CopyFrom(input.Owned(), context).Wait();
        deviceInput.Ghost().CopyFrom(input.Ghost(), context).Wait();
        cpu.DoReconstructionIterLimitedTo(mean, input, work, boundary, target, penalty);
        auto limitedWork = full.BuildWorkspace(components, std::vector<I>{1});
        const auto fullBoundary = full.PlaceBoundaryValues(boundary);
        for (const auto kernel : {C::ReconstructionDeviceKernel::FP64, C::ReconstructionDeviceKernel::FP64Pipelined})
        {
            full.SubmitReconstructionIterLimitedTo(engine, stream, deviceMean, deviceInput, limitedWork, fullBoundary,
                                                   deviceTarget, devicePenalty, C::ReconstructionRegion::All, kernel);
            engine.Synchronize();
            full.CheckStatus(limitedWork);
            Equal(work.temporary, limitedWork.temporary, 5e-11);
        }
        {
            // Isolate mean/BC subtraction from dense products. A common 1e8
            // offset erases the small differences if the complete update uses
            // float; direct FP64-to-float-float conversion retains them. This
            // catches accidental float staging and a hidden FP64 outer sum.
            Fill(mean, 1e8);
            {
                auto write = input.Owned().Write();
                for (I iCell = 0; iCell < input.Owned().RowCount(); ++iCell)
                    for (I j = 0; j < I(input.Width()); ++j)
                        write.View()(iCell, j) = 0;
            }
            {
                auto write = boundary.values.Write();
                for (I q = 0; q < boundary.values.RowCount(); ++q)
                    for (I j = 0; j < components; ++j)
                        write.View()(q, j) = 1e8 + .00001 * q + .00002 * j;
            }
            metrics.BindHalo(mean).Begin().Wait();
            inputHalo.Begin().Wait();
            deviceMean.Owned().CopyFrom(mean.Owned(), context).Wait();
            deviceMean.Ghost().CopyFrom(mean.Ghost(), context).Wait();
            deviceInput.Owned().CopyFrom(input.Owned(), context).Wait();
            deviceInput.Ghost().CopyFrom(input.Ghost(), context).Wait();
            deviceBoundary = reduced.PlaceBoundaryValues(boundary);
            cpu.DoReconstructionIter(mean, input, work, boundary);
            {
                double maximum = 0;
                const auto read = work.temporary.Owned().Read();
                for (I iCell = 0; iCell < work.temporary.Owned().RowCount(); ++iCell)
                    for (I j = 0; j < I(input.Width()); ++j)
                        maximum = std::max(maximum, std::abs(read.View()(iCell, j)));
                CHECK(maximum > 1e-3);
            }
            for (const auto kernel : {C::ReconstructionDeviceKernel::FP32FullAccumulation,
                                      C::ReconstructionDeviceKernel::FP32FloatFloatAccumulation, C::ReconstructionDeviceKernel::FP32Pipelined})
            {
                if (!reduced.SupportsKernel(kernel))
                    continue;
                reduced.SubmitReconstructionIterTo(engine, stream, deviceMean, deviceInput, accumulationWork, deviceBoundary,
                                                   C::ReconstructionRegion::All, kernel);
                engine.Synchronize();
                reduced.CheckStatus(accumulationWork);
                if (kernel == C::ReconstructionDeviceKernel::FP32FloatFloatAccumulation)
                    Equal(work.temporary, accumulationWork.temporary, 2e-5);
                else
                {
                    const auto host = accumulationWork.temporary.Owned().To(R::Device::Host());
                    const auto read = host.Read();
                    for (I iCell = 0; iCell < host.RowCount(); ++iCell)
                        for (I j = 0; j < I(input.Width()); ++j)
                            CHECK(read.View()(iCell, j) == 0);
                }
            }
        }
        {
            // Narrow arithmetic overflow is deferred to retained status. A
            // queued commit must preserve the original durable FP64 state.
            Fill(input, 1e100);
            inputHalo.Begin().Wait();
            deviceInput.Owned().CopyFrom(input.Owned(), context).Wait();
            deviceInput.Ghost().CopyFrom(input.Ghost(), context).Wait();
            reduced.SubmitReconstructionIterTo(engine, stream, deviceMean, deviceInput, accumulationWork, deviceBoundary,
                                               C::ReconstructionRegion::All, C::ReconstructionDeviceKernel::FP32FullAccumulation);
            reduced.SubmitCommitReconstruction(engine, stream, deviceInput, accumulationWork);
            engine.Synchronize();
            CHECK_THROWS(reduced.CheckStatus(accumulationWork));
            Equal(input, deviceInput);
        }
    }
}

TEST_CASE("CUDA Jacobi and WBAP agree with CPU for physical DOF layouts and direct MPI split work" * doctest::test_suite("halo"))
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, false);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    if (mpi.size > 1 && !nccl)
        REQUIRE(DNDS::MPI::isCudaAware());
    for (int degree : {1, 2, 3})
    {
        C::ReconstructionSettings settings;
        settings.maxOrder = degree;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        auto resident = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
        const I components = degree == 3 ? 7 : 5, width = components * plan.NRecDOF();
        auto mean = metrics.BuildField(components);
        Fill(mean, 1.0);
        auto reference = plan.BuildWorkspace(components);
        auto referenceCoefficients = plan.BuildCoefficients(components);
        Fill(referenceCoefficients, .02);
        auto bc = plan.BuildBoundaryValues(components);
        {
            auto write = bc.values.Write();
            for (I row = 0; row < bc.values.RowCount(); ++row)
                for (I j = 0; j < components; ++j)
                    write.View()(row, j) = 1 + .1 * j;
        }
        auto deviceBC = resident.PlaceBoundaryValues(bc);
        const auto formats = std::vector<R::RowLayout>{R::RowLayout::Dense(0, width), R::RowLayout::ColMajor(0, width, {32, 1}), R::RowLayout::TiledColMajor(0, width, 8)};
        for (const auto &shape : formats)
        {
            CAPTURE(degree);
            CAPTURE(shape.Kind());
            Fill(referenceCoefficients, .02);
            auto deviceMean = metrics.BuildField(shape.Kind() == R::RowLayoutKind::Table
                                                     ? R::RowLayout::Dense(0, components)
                                                     : R::RowLayout::ColMajor(0, components, {32, 1}),
                                                 resident.Device());
            Fill(deviceMean, 1.0);
            auto work = resident.BuildWorkspace(components, shape);
            auto coefficients = resident.BuildCoefficients(components, shape);
            Fill(coefficients, .02);
            auto meanHalo = metrics.BindHalo(mean), referenceHalo = metrics.BindHalo(referenceCoefficients);
            auto deviceMeanHalo = Halo(resident, deviceMean), deviceHalo = Halo(resident, coefficients, shape.Kind() == R::RowLayoutKind::Table ? R::TransferTransport::Direct : R::TransferTransport::DevicePacked);
            meanHalo.Begin().Wait();
            deviceMeanHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            const auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            CHECK(resident.SupportsKernel(C::ReconstructionDeviceKernel::FP64));
            CHECK_FALSE(resident.SupportsKernel(static_cast<C::ReconstructionDeviceKernel>(-1)));
            CHECK_THROWS(resident.DoReconstructionIter(deviceMean, coefficients, work, deviceBC, static_cast<C::ReconstructionDeviceKernel>(-1)));
            for (int step = 0; step < 4; ++step)
            {
                referenceHalo.Begin().Wait();
                plan.DoReconstructionIter(mean, referenceCoefficients, reference, bc);
                R::TransferOperation pending;
                if (nccl)
                {
                    comm.WaitFor(stream.Record());
                    deviceHalo.Enqueue(engine, comm);
                }
                else
                    pending = deviceHalo.Begin(R::ExecutionContext::CUDA(0));
                const auto kernel = step == 3 && shape.Kind() == R::RowLayoutKind::Table
                                        ? C::ReconstructionDeviceKernel::FP64Pipelined
                                        : C::ReconstructionDeviceKernel::FP64;
                resident.SubmitReconstructionIterTo(engine, stream, deviceMean, coefficients, work, deviceBC, C::ReconstructionRegion::Core, kernel);
                if (nccl)
                    stream.WaitFor(comm.Record());
                else
                    pending.Wait();
                resident.SubmitReconstructionIterTo(engine, stream, deviceMean, coefficients, work, deviceBC, C::ReconstructionRegion::HaloDependent, kernel);
                resident.SubmitCommitReconstruction(engine, stream, coefficients, work);
                engine.Synchronize();
                resident.CheckStatus(work);
                Equal(referenceCoefficients, coefficients);
            }
            auto transform = plan.BuildFaceTransform(0, 5);
            {
                auto forward = transform.forward.Write(), inverse = transform.inverse.Write();
                for (I iFace = 0; iFace < metrics.NumFaceProc(); ++iFace)
                {
                    forward.View()(iFace, 5) = .2;
                    inverse.View()(iFace, 5) = -.2;
                }
            }
            plan.FinalizeFaceTransform(transform);
            auto deviceTransform = resident.PlaceFaceTransform(transform);
            for (int order = degree; order >= 1; --order)
            {
                referenceHalo.Begin().Wait();
                plan.DoLimiterWBAP_3Order(referenceCoefficients, reference, order, std::cref(transform));
                R::TransferOperation pending;
                if (nccl)
                {
                    comm.WaitFor(stream.Record());
                    deviceHalo.Enqueue(engine, comm);
                }
                else
                    pending = deviceHalo.Begin(R::ExecutionContext::CUDA(0));
                resident.SubmitLimiterWBAP_3OrderTo(engine, stream, coefficients, work, order, std::cref(deviceTransform), C::ReconstructionRegion::Core);
                if (nccl)
                    stream.WaitFor(comm.Record());
                else
                    pending.Wait();
                resident.SubmitLimiterWBAP_3OrderTo(engine, stream, coefficients, work, order, std::cref(deviceTransform), C::ReconstructionRegion::HaloDependent);
                resident.SubmitCommitReconstruction(engine, stream, coefficients, work);
                engine.Synchronize();
                resident.CheckStatus(work);
                Equal(referenceCoefficients, coefficients, 2e-9);
            }
        }
    }
}
TEST_CASE("CUDA preserves rotated periodic vectors and axisymmetric prepared operators" * doctest::test_suite("halo"))
{
    for (bool sector : {true, false})
    {
        auto mesh = sector ? CFVTest::Sector(mpi) : G::OverlapTest::BuildNativeFixture(mpi, false).mesh;
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, 5, !sector);
        C::ReconstructionSettings settings;
        settings.maxOrder = 2;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        auto resident = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
        CompareSharedSweep(metrics, plan, resident, 5);
        const I components = 21, width = components * plan.NRecDOF();
        const auto vectors = std::vector<I>{1, 7};
        auto mean = metrics.BuildField(components);
        Fill(mean, 1.0);
        auto cpu = plan.BuildWorkspace(components, vectors);
        auto cpuCoefficients = plan.BuildCoefficients(components);
        Fill(cpuCoefficients, .02);
        auto gpu = resident.BuildWorkspace(components, R::RowLayout::TiledColMajor(0, width, 16), vectors);
        auto gpuCoefficients = resident.BuildCoefficients(components, R::RowLayout::TiledColMajor(0, width, 16));
        Fill(gpuCoefficients, .02);
        auto gpuMean = metrics.BuildField(R::RowLayout::ColMajor(0, components, {32, 1}), resident.Device());
        Fill(gpuMean, 1.0);
        auto boundary = plan.BuildBoundaryValues(components);
        {
            auto write = boundary.values.Write();
            for (I iRow = 0; iRow < boundary.values.RowCount(); ++iRow)
                for (I j = 0; j < components; ++j)
                    write.View()(iRow, j) = 1 + .1 * j;
        }
        auto gpuBoundary = resident.PlaceBoundaryValues(boundary);
        auto meanHalo = metrics.BindHalo(mean), gpuMeanHalo = Halo(resident, gpuMean);
        auto cpuHalo = metrics.BindHalo(cpuCoefficients), gpuHalo = Halo(resident, gpuCoefficients);
        meanHalo.Begin().Wait();
        gpuMeanHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            cpuHalo.Begin().Wait();
            gpuHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
            plan.DoReconstructionIter(mean, cpuCoefficients, cpu, boundary);
            // Large coupled-component coverage belongs to the general kernel;
            // the staged suite separately checks dynamic components at P3.
            resident.DoReconstructionIter(gpuMean, gpuCoefficients, gpu, gpuBoundary, C::ReconstructionDeviceKernel::FP64);
            Equal(cpuCoefficients, gpuCoefficients, 2e-9);
        }
        for (int order = 2; order >= 1; --order)
        {
            cpuHalo.Begin().Wait();
            gpuHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
            plan.DoLimiterWBAP_3Order(cpuCoefficients, cpu, order);
            resident.DoLimiterWBAP_3Order(gpuCoefficients, gpu, order);
            Equal(cpuCoefficients, gpuCoefficients, 2e-8);
        }
    }
}
TEST_CASE("CUDA degree three 3D reconstruction supports 37 coupled components and empty peers" * doctest::test_suite("halo"))
{
    for (I gridWidth : {I(1), I(3)})
    {
        CAPTURE(gridWidth);
        auto mesh = CFVTest::CartesianHexes(mpi, gridWidth);
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
        C::ReconstructionSettings settings;
        settings.maxOrder = 3;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        REQUIRE(plan.NRecDOF() == 19);
        auto resident = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
        // The retained comparison catalog is tested only on its fixed shape.
        // Seven P3 fields separately exercise production's second output slot
        // and inactive lanes, including the fixture's empty participants.
        if (gridWidth == 1)
            CompareSharedSweep(metrics, plan, resident, 5);
        CompareSharedSweep(metrics, plan, resident, 7);
        const I components = 37, width = components * plan.NRecDOF();
        auto mean = metrics.BuildField(components);
        Fill(mean, 1.0);
        auto cpu = plan.BuildWorkspace(components);
        auto cpuCoefficients = plan.BuildCoefficients(components);
        Fill(cpuCoefficients, .02);
        auto gpu = resident.BuildWorkspace(components, R::RowLayout::ColMajor(0, width, {32, 1}));
        auto gpuCoefficients = resident.BuildCoefficients(components, R::RowLayout::ColMajor(0, width, {32, 1}));
        Fill(gpuCoefficients, .02);
        auto gpuMean = metrics.BuildField(R::RowLayout::TiledColMajor(0, components, 8), resident.Device());
        Fill(gpuMean, 1.0);
        auto boundary = plan.BuildBoundaryValues(components);
        {
            auto write = boundary.values.Write();
            for (I iRow = 0; iRow < boundary.values.RowCount(); ++iRow)
                for (I j = 0; j < components; ++j)
                    write.View()(iRow, j) = 1 + .1 * j;
        }
        auto gpuBoundary = resident.PlaceBoundaryValues(boundary);
        auto meanHalo = metrics.BindHalo(mean), gpuMeanHalo = Halo(resident, gpuMean);
        auto cpuHalo = metrics.BindHalo(cpuCoefficients), gpuHalo = Halo(resident, gpuCoefficients);
        meanHalo.Begin().Wait();
        gpuMeanHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
        for (int iteration = 0; iteration < 2; ++iteration)
        {
            cpuHalo.Begin().Wait();
            gpuHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
            plan.DoReconstructionIter(mean, cpuCoefficients, cpu, boundary);
            resident.DoReconstructionIter(gpuMean, gpuCoefficients, gpu, gpuBoundary, C::ReconstructionDeviceKernel::FP64);
            Equal(cpuCoefficients, gpuCoefficients, 2e-9);
        }
        for (int order = 3; order >= 1; --order)
        {
            cpuHalo.Begin().Wait();
            gpuHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
            plan.DoLimiterWBAP_3Order(cpuCoefficients, cpu, order);
            resident.DoLimiterWBAP_3Order(gpuCoefficients, gpu, order);
            Equal(cpuCoefficients, gpuCoefficients, 2e-8);
        }
    }
}

TEST_CASE("Device errors reject unsupported SOR stale publications and invalid indicators without silent output" * doctest::test_suite("halo"))
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, false);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    C::ReconstructionSettings settings;
    settings.SORInstead = true;
    auto sor = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    CHECK_THROWS(sor.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0)));
    settings.SORInstead = false;
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto resident = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
    auto work = resident.BuildWorkspace(5, R::RowLayout::TiledColMajor(0, 5 * plan.NRecDOF(), 8));
    auto coefficients = resident.BuildCoefficients(5, R::RowLayout::TiledColMajor(0, 5 * plan.NRecDOF(), 8));
    Fill(coefficients, 1.0);
    Fill(work.temporary, 9.0);
    // A real mixed active/inactive pass, followed by recovery/error checks.
    auto cpu = plan.BuildWorkspace(5);
    auto cpuCoefficients = plan.BuildCoefficients(5);
    Fill(cpuCoefficients, 1.0);
    R::Array<double> activation(R::RowLayout::Dense(metrics.NumCell(), 1));
    {
        auto write = activation.Write();
        for (I iRow = 0; iRow < metrics.NumCell(); ++iRow)
            write.View()(iRow, 0) = iRow % 2;
    }
    auto gpuActivation = activation.To(R::Device::CUDA(0));
    metrics.BindHalo(cpuCoefficients).Begin().Wait();
    Halo(resident, coefficients).Begin(R::ExecutionContext::CUDA(0)).Wait();
    plan.DoLimiterWBAP_3OrderTo(cpuCoefficients, cpu, 1, std::nullopt, C::ReconstructionRegion::All, std::cref(activation), .5);
    plan.CommitReconstruction(cpuCoefficients, cpu);
    R::EagerEngine activeEngine;
    auto activeStream = activeEngine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    resident.SubmitLimiterWBAP_3OrderTo(activeEngine, activeStream, coefficients, work, 1, std::nullopt, C::ReconstructionRegion::All, std::cref(gpuActivation), .5);
    resident.SubmitCommitReconstruction(activeEngine, activeStream, coefficients, work);
    activeEngine.Synchronize();
    resident.CheckStatus(work);
    Equal(cpuCoefficients, coefficients, 2e-9);
    auto bad = work;
    bad.temporary = coefficients;
    CHECK_THROWS(resident.SubmitCommitReconstruction(activeEngine, activeStream, coefficients, bad));
    bad = work;
    bad.components = std::numeric_limits<I>::max();
    CHECK_THROWS(resident.SubmitCommitReconstruction(activeEngine, activeStream, coefficients, bad));
    bad = work;
    bad.scratch = R::Array<double>(R::RowLayout::Padded(std::vector<size_t>(size_t(work.deviceBlocks * 6), 0), work.scratch.Layout().Width()), R::Device::CUDA(0));
    CHECK_THROWS(resident.SubmitCommitReconstruction(activeEngine, activeStream, coefficients, bad));
    Fill(coefficients, 1.0);
    Fill(work.temporary, 9.0);
    auto before = work.temporary.Owned().To(R::Device::Host());
    R::Array<double> indicator(R::RowLayout::Dense(metrics.NumCell(), 1));
    indicator.Write().At(0, 0) = std::numeric_limits<double>::quiet_NaN();
    auto deviceIndicator = indicator.To(R::Device::CUDA(0));
    R::EagerEngine engine;
    auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    resident.SubmitLimiterWBAP_3OrderTo(engine, stream, coefficients, work, 1, std::nullopt, C::ReconstructionRegion::All, std::cref(deviceIndicator));
    resident.SubmitCommitReconstruction(engine, stream, coefficients, work);
    engine.Synchronize();
    CHECK_THROWS(resident.CheckStatus(work));
    const auto actual = work.temporary.Owned().To(R::Device::Host());
    const auto a = actual.Read(), b = before.Read();
    for (I row = 0; row < actual.RowCount(); ++row)
        for (I j = 0; j < actual.RowSize(row); ++j)
            CHECK(a.View()(row, j) == b.View()(row, j));
    {
        auto mutation = fixture.mesh->versions.BeginMutation(R::Change::Geometry);
        mutation.Commit();
    }
    CHECK_THROWS(resident.Validate());
}
// Storage variants share the exact distributed fixtures and CPU oracle above.
#include "test_ReconstructionCUDA_Storage.hpp"
#include "test_ReconstructionCUDA_Staged.hpp"
#include "test_ReconstructionCUDA_AMG.hpp"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi = DNDS::MPIInfo(MPI_COMM_WORLD);
    if (const auto *value = std::getenv("DNDS_TEST_TRANSFER_PROVIDER"))
    {
        DNDS_check_throw_info(std::string(value) == "nccl", "unknown test transfer provider");
        nccl = R::NCCLTransferProvider::Create(mpi.comm, R::Device::CUDA(0));
    }
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    nccl.reset();
    MPI_Finalize();
    return result;
}
