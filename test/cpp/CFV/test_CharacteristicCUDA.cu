#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/GradientPlan.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include <cuda_runtime.h>

namespace C = DNDS::CFV;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

__global__ void ProduceMatrices(R::ArrayView<double> forward, R::ArrayView<double> inverse, double shear, bool invalid)
{
    for (I iFace = I(blockIdx.x) * blockDim.x + threadIdx.x; iFace < forward.RowCount(); iFace += I(gridDim.x) * blockDim.x)
        for (I j = 0; j < 25; ++j)
        {
            const double diagonal = j % 6 == 0;
            forward(iFace, j) = diagonal + (j == 5 ? shear : 0) + (invalid && j == 0 ? 1 : 0);
            inverse(iFace, j) = diagonal - (j == 5 ? shear : 0);
        }
}
template <class Transform>
static void Produce(R::EagerEngine &engine, const R::ExecutionStream &stream, Transform &transform, double shear, bool invalid = false)
{
    R::SubmitArrays(engine, stream, [shear, invalid](R::ExecutionContext c, auto left, auto right)
                    {
        R::ValidateViewDevices(c.device, left, right);
        R::CheckKernelArgumentBudget<decltype(left), decltype(right), double, bool>();
        if (!left.RowCount()) return;
        ProduceMatrices<<<4, 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(left, right, shear, invalid);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test characteristic producer launch failed"); }, R::Write(transform.forward), R::Write(transform.inverse));
}
template <class Transform, class Plan>
static void HostMatrices(const Plan &plan, Transform &transform, double shear)
{
    {
        auto left = transform.forward.Write(), right = transform.inverse.Write();
        for (I iFace = 0; iFace < transform.forward.RowCount(); ++iFace)
            for (I j = 0; j < 25; ++j)
            {
                left.View()(iFace, j) = (j % 6 == 0) + (j == 5 ? shear : 0);
                right.View()(iFace, j) = (j % 6 == 0) - (j == 5 ? shear : 0);
            }
    }
    plan.FinalizeFaceTransform(transform);
}
static void Fill(C::FVField &field)
{
    auto write = field.Owned().Write();
    for (I iCell = 0; iCell < field.Owned().RowCount(); ++iCell)
        for (I j = 0; j < I(field.Width()); ++j)
            write.View()(iCell, j) = .01 * (1 + j) * (iCell % 3 ? 1 : -.5);
}
static void Equal(const C::FVField &expected, const C::FVField &result)
{
    const auto downloaded = result.Owned().To(R::Device::Host());
    const auto left = expected.Owned().Read(), right = downloaded.Read();
    for (I iCell = 0; iCell < expected.Owned().RowCount(); ++iCell)
        for (I j = 0; j < I(expected.Width()); ++j)
            CHECK(right.View()(iCell, j) == doctest::Approx(left.View()(iCell, j)).epsilon(2e-10).scale(1));
}

TEST_CASE("Queued matrix production validates once before modal WBAP without a host drain")
{
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    auto mesh = DNDS::Geom::OverlapTest::BuildNativeFixture(mpi, false).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 3;
    const auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    const auto resident = plan.Place(device, context);
    auto input = plan.BuildCoefficients(7);
    Fill(input);
    plan.BindHalo(input).Begin().Wait();
    auto reference = plan.BuildWorkspace(7), work = resident.BuildWorkspace(7);
    auto coefficients = resident.BuildCoefficients(7);
    coefficients.Owned().CopyFrom(input.Owned(), context).Wait();
    coefficients.Ghost().CopyFrom(input.Ghost(), context).Wait();
    auto transform = plan.BuildFaceTransform(0, 5), queued = resident.BuildQueuedFaceTransform(0, 5);
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    std::vector<C::FVField> expectedPasses, actualPasses;
    for (int pass = 0; pass < 3; ++pass)
    {
        expectedPasses.push_back(plan.BuildCoefficients(7));
        actualPasses.push_back(resident.BuildCoefficients(7));
    }
    for (const double shear : {.4, -.7})
    {
        HostMatrices(plan, transform, shear);
        Produce(engine, stream, queued, shear);
        // Default production sequence has no inverse-check launch. Exercise
        // the opt-in diagnostic on the second refresh only.
        if (shear < 0)
            resident.SubmitValidateFaceTransform(engine, stream, queued, work);
        for (const int order : {3, 2, 1})
        {
            plan.DoLimiterWBAP_3OrderTo(input, reference, order, std::cref(transform));
            expectedPasses[3 - order].Owned().CopyFrom(reference.temporary.Owned()).Wait();
            resident.SubmitLimiterWBAP_3OrderTo(engine, stream, coefficients, work, order, std::cref(queued));
            resident.SubmitCommitReconstruction(engine, stream, actualPasses[3 - order], work);
        }
        engine.Synchronize();
        resident.CheckStatus(work);
        Equal(reference.temporary, work.temporary);
        for (int pass = 0; pass < 3; ++pass)
            Equal(expectedPasses[pass], actualPasses[pass]);
    }
    // One critical negative: bad device producer data is caught by numerical
    // publication and the guarded limiter preserves the old destination.
    auto before = work.temporary.Owned().To(R::Device::Host());
    Produce(engine, stream, queued, .2, true);
    resident.SubmitValidateFaceTransform(engine, stream, queued, work);
    resident.SubmitLimiterWBAP_3OrderTo(engine, stream, coefficients, work, 1, std::cref(queued));
    engine.Synchronize();
    if (transform.forward.RowCount())
        CHECK_THROWS(resident.CheckStatus(work));
    const auto after = work.temporary.Owned().To(R::Device::Host());
    const auto a = before.Read(), b = after.Read();
    for (I iCell = 0; iCell < before.RowCount(); ++iCell)
        for (I j = 0; j < before.RowSize(iCell); ++j)
            CHECK(a.View()(iCell, j) == b.View()(iCell, j));
}

TEST_CASE("Queued gradient WBAP shares the producer validation contract and leaves scalar components scalar")
{
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    auto mesh = DNDS::Geom::OverlapTest::BuildNativeFixture(mpi, false).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    const auto plan = C::GradientPlan::Prepare(*mesh, metrics);
    auto residentMetrics = metrics.Placed(device, context);
    const auto resident = plan.Placed(residentMetrics, context);
    auto input = metrics.BuildField(14), expected = metrics.BuildField(14);
    Fill(input);
    metrics.BindHalo(input).Begin().Wait();
    auto din = residentMetrics.BuildField(14), result = residentMetrics.BuildField(14);
    din.Owned().CopyFrom(input.Owned(), context).Wait();
    din.Ghost().CopyFrom(input.Ghost(), context).Wait();
    auto transform = plan.BuildFaceTransform(0, 5), queued = resident.BuildQueuedFaceTransform(0, 5);
    auto work = resident.BuildWorkspace(7, {}, true);
    HostMatrices(plan, transform, .4);
    plan.LimitWBAP3(input, expected, {}, std::cref(transform));
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    Produce(engine, stream, queued, .4);
    resident.LimitWBAP3(engine, stream, din, result, work, std::cref(queued));
    engine.Close();
    resident.CheckStatus(work);
    Equal(expected, result);
    const auto disabled = resident.BuildQueuedFaceTransform(0, 0);
    CHECK(disabled.count == 0);
    CHECK(disabled.forward.RowCount() == 0);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi = DNDS::MPIInfo(MPI_COMM_WORLD);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
