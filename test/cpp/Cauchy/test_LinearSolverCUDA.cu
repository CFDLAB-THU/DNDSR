#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include "Cauchy/Implicit/LocalAMGPreconditioner.hpp"
#include "DNDS/Algebra/LocalProducts.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <numeric>
#include <Eigen/Dense>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;
using DNDS::real;
static DNDS::MPIInfo mpi;

namespace
{
    int DeviceOrdinal()
    {
        int count = 0;
        DNDS_check_throw_info(cudaGetDeviceCount(&count) == cudaSuccess && count > 0, "CUDA solver test requires a GPU");
        const int ordinal = mpi.rank % count;
        DNDS_check_throw_info(cudaSetDevice(ordinal) == cudaSuccess, "CUDA test device selection failed");
        return ordinal;
    }
    real Exact(int rank, I cell, I j) { return .3 + .013 * rank + .02 * cell - .017 * j; }
    real Diagonal(I cell, I i, I j, I n) { return i == j ? 3 + .1 * cell : j == (i + 1) % n ? .02
                                                                                            : 0; }
    real Off(I cell, I other, I i, I j) { return i == j ? -.1 - .001 * cell : i == 1 && j == 4 ? .003 * (other + 1)
                                                                                               : 0; }

    // Independent whole-distribution recurrence: every owned and remote value
    // is frozen for one Jacobi update, with a CPU dense diagonal solve. This
    // does not call the production linear actions, halo or inverse kernels.
    std::vector<real> JacobiOracle(int components, int iterations, const std::vector<std::vector<I>> &columns)
    {
        const I count = I(mpi.size) * 4 * components;
        std::vector<real> current(count, 0), next(count);
        const auto offset = [components](int rank, I cell, I component)
        { return (I(rank) * 4 + cell) * components + component; };
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            for (int rank = 0; rank < mpi.size; ++rank)
                for (I iCell = 0; iCell < 4; ++iCell)
                {
                    Eigen::MatrixXd diagonal(components, components);
                    Eigen::VectorXd value(components);
                    for (I i = 0; i < components; ++i)
                    {
                        value[i] = 0;
                        for (I j = 0; j < components; ++j)
                        {
                            diagonal(i, j) = Diagonal(iCell, i, j, components);
                            value[i] += diagonal(i, j) * Exact(rank, iCell, j);
                        }
                        for (const auto other : columns[iCell])
                        {
                            const int owner = other < 4 ? rank : (rank + mpi.size - 1) % mpi.size;
                            const I local = other < 4 ? other : other - 2;
                            for (I j = 0; j < components; ++j)
                            {
                                value[i] += Off(iCell, other, i, j) * Exact(owner, local, j);
                                value[i] -= Off(iCell, other, i, j) * current[offset(owner, local, j)];
                            }
                        }
                    }
                    const Eigen::VectorXd solved = diagonal.fullPivLu().solve(value);
                    for (I j = 0; j < components; ++j)
                        next[offset(rank, iCell, j)] = solved[j];
                }
            current.swap(next);
        }
        return current;
    }
}

TEST_CASE("Reserved weighted products and finite checks use deterministic resident reductions")
{
    const auto device = R::Device::CUDA(DeviceOrdinal());
    for (I rows : {I(0), I(1), I(1009), I(10007)})
    {
        constexpr I n = 7;
        R::Array<real> x(R::RowLayout::Dense(rows, n, n + 2)), y(x.Layout());
        std::vector<real> scales(n);
        for (I j = 0; j < n; ++j)
            scales[j] = 1 / (1 + .2 * j);
        real expected = 0;
        {
            auto a = x.Write(), b = y.Write();
            for (I iRow = 0; iRow < rows; ++iRow)
                for (I j = 0; j < n; ++j)
                {
                    const real u = std::sin(.013 * iRow + .2 * j), v = .5 + std::cos(.021 * iRow - .3 * j);
                    a.View()(iRow, j) = u;
                    b.View()(iRow, j) = v;
                    expected += (u * scales[j]) * (v * scales[j]);
                }
        }
        for (int backend : {0, 1, 2})
        {
            CAPTURE(rows);
            CAPTURE(backend);
            const auto where = backend == 2 ? device : R::Device::Host();
            R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
            auto lane = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(device.ordinal)) : engine.CreateCPUStream();
            auto a = x.To(where), b = y.To(where);
            A::LocalProductWorkspace scratch(rows, n, 2, where);
            scratch.SetInverseScales(scales, lane.Context());
            scratch.SubmitProduct(engine, lane, a, b, 0);
            scratch.SubmitFinite(engine, lane, a, 1);
            engine.Synchronize();
            const auto values = scratch.ReadResults(2, lane.Context());
            CHECK(values[0] == doctest::Approx(expected).epsilon(3e-12).scale(1));
            CHECK(values[1] == 1);
            // Repeated queued calls reset the arrival counter and preserve the
            // fixed tree, regardless of which block happens to finish last.
            scratch.SubmitProduct(engine, lane, a, b, 0);
            scratch.SubmitProduct(engine, lane, a, b, 1);
            engine.Synchronize();
            const auto repeated = scratch.ReadResults(2, lane.Context());
            CHECK(repeated[0] == values[0]);
            CHECK(repeated[1] == values[0]);
            if (rows)
            {
                auto bad = x.Clone();
                {
                    auto write = bad.Write();
                    write.View()(rows - 1, n - 1) = std::numeric_limits<real>::infinity();
                }
                a.CopyFrom(bad, lane.Context()).Wait();
                scratch.SubmitFinite(engine, lane, a);
                engine.Synchronize();
                CHECK(scratch.ReadResults(1, lane.Context())[0] == 0);
            }
        }
    }
}

TEST_CASE("Distributed coupled Jacobi and left GMRES share queued orchestration")
{
    const auto cuda = R::Device::CUDA(DeviceOrdinal());
    const std::vector<std::vector<I>> columns{{1}, {0, 2}, {1, 3, 4}, {2, 5}};
    auto communication = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    A::DofSpace space(communication, 4, 2, "CUDA coupled linear oracle");
    std::vector<std::vector<I>> send(mpi.size), receive(mpi.size);
    send[(mpi.rank + 1) % mpi.size] = {2, 3};
    receive[(mpi.rank + mpi.size - 1) % mpi.size] = {0, 1};
    const auto route = R::RowTransferPlan::Known(communication, 4, 2, send, receive);
    for (const int n : {5, 6, 7})
    {
        C::DomainSettings domain;
        if (n > 5)
        {
            domain.equation.transport = "NewtonianFourier";
            domain.equation.viscosity = .01;
            domain.equation.rans.model = n == 6 ? "SA" : "KOmegaSST";
        }
        const auto equation = C::EquationSystem::Resolve(domain.equation);
        R::Array<real> hostDiagonal(R::RowLayout::Dense(4, n * n)), hostRhs(R::RowLayout::Dense(4, n, n + (n == 7 ? 2 : 0)));
        A::BlockMatrixStorage hostOff({1, 2, 3, 2}, n);
        {
            auto diagonal = hostDiagonal.Write(), rhs = hostRhs.Write();
            auto off = hostOff.Write();
            for (I iCell = 0; iCell < 4; ++iCell)
                for (I i = 0; i < n; ++i)
                {
                    real value = 0;
                    for (I j = 0; j < n; ++j)
                    {
                        diagonal.View()(iCell, i + n * j) = Diagonal(iCell, i, j, n);
                        value += Diagonal(iCell, i, j, n) * Exact(mpi.rank, iCell, j);
                    }
                    for (I k = 0; k < I(columns[iCell].size()); ++k)
                    {
                        const I other = columns[iCell][k];
                        const int owner = other < 4 ? mpi.rank : (mpi.rank + mpi.size - 1) % mpi.size;
                        const I local = other < 4 ? other : other - 2;
                        for (I j = 0; j < n; ++j)
                        {
                            off.ArrayView().Element(iCell, k, i, j) = Off(iCell, other, i, j);
                            value += Off(iCell, other, i, j) * Exact(owner, local, j);
                        }
                    }
                    rhs.View()(iCell, i) = value;
                }
        }
        for (int backend : {0, 1, 2})
            for (int mode : {0, 1})
            {
                CAPTURE(n);
                CAPTURE(backend);
                CAPTURE(mode);
                const auto device = backend == 2 ? cuda : R::Device::Host();
                R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
                const auto compute = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(device.ordinal)) : engine.CreateCPUStream();
                const auto comm = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(device.ordinal)) : engine.CreateCPUStream();
                auto problem = std::make_shared<C::LinearProblemContext>(space, n, columns, std::vector<I>{1, 0},
                                                                         C::DiagonalStorage::Block, C::OffDiagonalStorage::Assembled, device);
                problem->Diagonal().CopyFrom(hostDiagonal, compute.Context()).Wait();
                auto blocks = problem->OffDiagonal().Values();
                blocks.CopyFrom(hostOff.Values(), compute.Context()).Wait();
                A::DofField<real> field(space, hostRhs.To(device), R::Array<real>(R::RowLayout::Dense(2, n, hostRhs.Layout().Stride()), device));
                auto rhs = C::EquationDofs::Coupled(equation.Schema(), field);
                const auto bind = [&](auto &part)
                {
                    return R::PayloadPlan(route, part.Owned().Layout(), part.Ghost().Layout(), R::ElementSchema::Arithmetic<real>())
                        .Bind(part.Owned(), part.Ghost());
                };
                C::LinearSolveSettings settings;
                settings.method = mode == 0 ? "jacobi" : "gmres";
                settings.preconditioner = "jacobi";
                settings.maxIterations = 24;
                settings.subspace = 8;
                settings.relativeTolerance = 1e-11;
                settings.absoluteTolerance = 0;
                settings.jacobiLocalSweeps = 2;
                settings.cudaJacobiMapping = backend == 2 && n != 6 ? "componentGroup" : "thread";
                settings.residualCheckInterval = 2;
                C::LinearExecution execution;
                execution.overlap = n != 6;
                execution.collectiveChecks = false;
                execution.streams = C::LinearExecution::Streams{compute, comm};
                C::DofLinearSolver solver(problem, rhs, bind, settings, execution);
                std::vector<real> scales(n);
                for (I j = 0; j < n; ++j)
                    scales[j] = 1 + .3 * j;
                const auto report = solver.Solve(rhs, {}, scales);
                CHECK(report.usable);
                CHECK(report.converged);
                const auto answer = solver.Solution().CoupledField().Owned().To(R::Device::Host());
                const auto read = answer.Read();
                for (I iCell = 0; iCell < 4; ++iCell)
                    for (I j = 0; j < n; ++j)
                        CHECK(read.View()(iCell, j) == doctest::Approx(Exact(mpi.rank, iCell, j)).epsilon(3e-10).scale(1));
                if (mode == 0)
                    for (const int iterations : {3, 8})
                    {
                        CAPTURE(iterations);
                        settings.stopping = "fixedCount";
                        settings.maxIterations = iterations;
                        settings.jacobiLocalSweeps = 1;
                        C::DofLinearSolver fixed(problem, rhs, bind, settings, execution);
                        // A caller may keep this alias before Solve. Odd/even
                        // ping-pong must publish into its original allocation.
                        const auto retained = fixed.Solution();
                        const auto original = retained.CoupledField().Owned().SharedStorage();
                        const auto fixedReport = fixed.Solve(rhs);
                        CHECK(fixedReport.usable);
                        CHECK(fixedReport.iterations == iterations);
                        CHECK(fixedReport.jacobiSweeps == iterations);
                        CHECK(fixedReport.innerProductBatches == 0);
                        CHECK(fixed.Solution().CoupledField().Owned().SharedStorage() == original);
                        const auto expected = JacobiOracle(n, iterations, columns);
                        const auto host = retained.CoupledField().Owned().To(R::Device::Host());
                        const auto values = host.Read();
                        for (I iCell = 0; iCell < 4; ++iCell)
                            for (I j = 0; j < n; ++j)
                                CHECK(values.View()(iCell, j) == doctest::Approx(expected[(I(mpi.rank) * 4 + iCell) * n + j]).epsilon(3e-11).scale(1));
                    }
            }
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    mpi = DNDS::MPIInfo(MPI_COMM_WORLD);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    DNDS::MPI::Finalize();
    return result;
}

TEST_CASE("Copy-free local AmgX consumes native int32 unsorted row-major blocks")
{
    if (mpi.size != 1 || !A::AmgXPreconditioner::Available())
        return;
    constexpr I rows = 17;
    const auto device = R::Device::CUDA(DeviceOrdinal());
    auto communication = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    A::DofSpace space(communication, rows, 0, "native int32 local AMG");
    std::vector<std::vector<I>> columns(rows);
    for (I iCell = 0; iCell < rows; ++iCell)
        columns[iCell] = {(iCell + 1) % rows, (iCell + rows - 1) % rows};
    std::vector<I> core(rows);
    std::iota(core.begin(), core.end(), I(0));
    for (const int n : {5, 6, 7})
    {
        CAPTURE(n);
        C::LinearMatrixOptions options;
        options.indices = C::LinearIndexStorage::Int32;
        options.rowMajorBlocks = true;
        auto problemOwner = std::make_shared<C::LinearProblemContext>(space, n, columns, core, C::DiagonalStorage::Block,
                                                                      C::OffDiagonalStorage::Assembled, device, C::LinearSweepPartitions{}, options);
        auto &problem = *problemOwner;
        CHECK(problem.HostPattern().UsesInt32());
        R::Array<real> diagonal(R::RowLayout::Dense(rows, n * n)), rhs(R::RowLayout::Dense(rows, n));
        A::BlockMatrixStorage blocks(std::vector<I>(rows, 2), n, R::Device::Host(), {}, true);
        const auto exact = [](I iCell, I j)
        { return .4 + .03 * iCell + .07 * j; };
        {
            auto d = diagonal.Write(), b = rhs.Write();
            auto a = blocks.Write();
            for (I iCell = 0; iCell < rows; ++iCell)
                for (int i = 0; i < n; ++i)
                {
                    real value = 0;
                    for (int j = 0; j < n; ++j)
                    {
                        const real own = i == j ? 4 + .2 * i : .012 * (i + 1) - .019 * (j + 1);
                        d.View()(iCell, i * n + j) = own;
                        value += own * exact(iCell, j);
                        for (I k = 0; k < 2; ++k)
                        {
                            const real other = .003 * (i + 1) - .005 * (j + 1) + .001 * k;
                            a.ArrayView().Element(iCell, k, i, j) = other;
                            value += other * exact(columns[iCell][k], j);
                        }
                    }
                    b.View()(iCell, i) = value;
                }
        }
        R::EagerEngine engine;
        const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(device.ordinal));
        problem.Diagonal().CopyFrom(diagonal, stream.Context()).Wait();
        auto values = problem.OffDiagonal().Values();
        values.CopyFrom(blocks.Values(), stream.Context()).Wait();
        A::AmgXSettings settings;
        settings.maxLevels = 1;
        settings.coarseSweeps = 40;
        settings.relaxation = .9;
        C::LocalAMGPreconditioner preconditioner(problem, settings);
        auto input = rhs.To(device);
        R::Array<real> output(R::RowLayout::Dense(rows, n), device);
        preconditioner.Update(engine, stream, problem);
        preconditioner.Apply(engine, stream, input, output);
        const auto host = output.To(R::Device::Host());
        const auto read = host.Read();
        for (I iCell = 0; iCell < rows; ++iCell)
            for (int j = 0; j < n; ++j)
                CHECK(read.View()(iCell, j) == doctest::Approx(exact(iCell, j)).epsilon(3e-11));
        // The adapter must reject incompatible representation rather than
        // silently allocating a packed shadow matrix/vector.
        R::Array<real> padded(R::RowLayout::Dense(rows, n, n + 1), device);
        CHECK_THROWS(preconditioner.Apply(engine, stream, padded, output));
        C::DomainSettings domain;
        if (n > 5)
        {
            domain.equation.transport = "NewtonianFourier";
            domain.equation.viscosity = .01;
            domain.equation.rans.model = n == 6 ? "SA" : "KOmegaSST";
        }
        const auto equation = C::EquationSystem::Resolve(domain.equation);
        A::DofField<real> field(space, input, R::Array<real>(R::RowLayout::Dense(0, n), device));
        auto bundle = C::EquationDofs::Coupled(equation.Schema(), field);
        const auto route = R::RowTransferPlan::Known(communication, rows, 0, std::vector<std::vector<I>>(1), std::vector<std::vector<I>>(1));
        const auto bind = [&](auto &part)
        {
            return R::PayloadPlan(route, part.Owned().Layout(), part.Ghost().Layout(), R::ElementSchema::Arithmetic<real>())
                .Bind(part.Owned(), part.Ghost());
        };
        C::LinearSolveSettings controls;
        controls.method = "amgx";
        controls.stopping = "fixedCount";
        controls.maxIterations = 2;
        controls.relativeTolerance = 1e30;
        C::LinearExecution execution;
        execution.collectiveChecks = false;
        execution.overlap = false;
        execution.streams = C::LinearExecution::Streams{stream, engine.CreateStream(std::make_shared<R::CUDALauncher>(device.ordinal))};
        C::DofLinearSolver fixed(problemOwner, bundle, bind, controls, execution);
        const auto report = fixed.Solve(bundle);
        CHECK(report.usable);
        CHECK(report.stop == DNDS::Linear::LinearStop::FixedCount);
        CHECK(report.iterations == 2);
        CHECK(report.preconditionerApplications == 2);
        CHECK(report.operatorApplications == 1);
        CHECK(report.innerProductBatches == 0);
        CHECK_FALSE(report.converged);
        CHECK(std::isnan(report.trueResidual));
        engine.Close();
    }
}
