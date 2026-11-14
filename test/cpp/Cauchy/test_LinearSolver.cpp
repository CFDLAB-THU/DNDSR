#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include <cstdlib>
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include <Eigen/LU>
#include <optional>
#include <limits>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using Field = A::DofField<DNDS::real>;
using Matrix = Eigen::Matrix<double, 5, 5>;

namespace
{
    const std::vector<std::vector<DNDS::index>> columns{{1}, {0, 2}, {1, 3, 4}, {2, 5}};
    Matrix Diagonal(DNDS::index iCell)
    {
        Matrix value = Matrix::Identity() * (3 + .1 * iCell);
        for (int j = 0; j < 5; ++j)
            value(j, (j + 1) % 5) = .02;
        return value;
    }
    Matrix Off(DNDS::index iCell, DNDS::index other)
    {
        Matrix value = Matrix::Identity() * (-.1 - .001 * iCell);
        value(1, 4) = .003 * (other + 1);
        return value;
    }
    double RHS(int rank, DNDS::index iCell, int j) { return .4 + .1 * rank - .06 * iCell + .05 * j; }
    struct EvaluatedOff : C::OffDiagonalEvaluator
    {
        std::shared_ptr<const A::DofSpace::Partition> partition;
        std::vector<std::vector<DNDS::index>> declaredColumns = columns;
        DNDS::index components = 5;
        explicit EvaluatedOff(const A::DofSpace &space) : partition(space.Pin()) {}
        DNDS::index Components() const override { return components; }
        auto RowSpaceIdentity() const -> std::shared_ptr<const A::DofSpace::Partition> override { return partition; }
        std::vector<std::vector<DNDS::index>> Columns() const override { return declaredColumns; }
        DNDS::index RowNumber() const override { return DNDS::index(declaredColumns.size()); }
        DNDS::index BlockNumber(DNDS::index iRow) const override { return DNDS::index(declaredColumns[iRow].size()); }
        DNDS::SpanView<const DNDS::index> ColumnRow(DNDS::index iRow) const override
        {
            const auto &row = declaredColumns[iRow];
            return {row.data(), DNDS::index(row.size())};
        }
        void EvaluateBlock(DNDS::index iRow, DNDS::index iBlock, Eigen::Ref<Eigen::MatrixXd> output) const override
        {
            output = Off(iRow, columns[iRow][iBlock]);
        }
        void ApplyBlock(DNDS::index iRow, DNDS::index iBlock, InputVector input, Eigen::Ref<Eigen::VectorXd> output) const override
        {
            for (int j = 0; j < 5; ++j)
                output[j] = (-.1 - .001 * iRow) * input[j];
            output[1] += .003 * (columns[iRow][iBlock] + 1) * input[4];
        }
    };
    /// Reject dense extraction so a passing SGS/Jv test proves that the
    /// direct action contract is used, rather than only comparing values from
    /// two calls through the same dense producer.
    struct ActionOnlyOff final : EvaluatedOff
    {
        using EvaluatedOff::EvaluatedOff;
        void EvaluateBlock(DNDS::index, DNDS::index, Eigen::Ref<Eigen::MatrixXd>) const override
        {
            DNDS_check_throw_info(false, "action-only test producer cannot extract dense blocks");
        }
    };
    // Test-only instrumentation proves that zero products are actually
    // omitted, rather than merely comparing two numerically identical scans.
    struct CountedOff final : EvaluatedOff
    {
        using EvaluatedOff::EvaluatedOff;
        mutable DNDS::index products = 0;
        void ApplyBlock(DNDS::index iRow, DNDS::index iBlock, InputVector input, Eigen::Ref<Eigen::VectorXd> output) const override
        {
            ++products;
            EvaluatedOff::ApplyBlock(iRow, iBlock, input, output);
        }
    };
    struct Fixture
    {
        DNDS::MPIInfo mpi{MPI_COMM_WORLD};
        int rank = mpi.rank, size = mpi.size;
        std::shared_ptr<R::CommunicationContext> communication;
        A::DofSpace space;
        R::RowTransferPlan route;
        std::shared_ptr<C::LinearProblemContext> problem;
        Field rhs;
        static R::RowTransferPlan Route(const std::shared_ptr<R::CommunicationContext> &communication, int rank, int size)
        {
            std::vector<std::vector<DNDS::index>> send(size), receive(size);
            send[(rank + 1) % size] = {2, 3};
            receive[(rank + size - 1) % size] = {0, 1};
            return R::RowTransferPlan::Known(communication, 4, 2, send, receive);
        }
        Fixture(int layout = 0, bool evaluated = false)
            : communication(std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD)),
              space(communication, 4, 2, "coupled linear solver oracle"), route(Route(communication, rank, size)),
              problem(std::make_shared<C::LinearProblemContext>(space, 5, columns, std::vector<DNDS::index>{1, 0}, C::DiagonalStorage::Block,
                                                                evaluated ? C::OffDiagonalStorage::SemiMatrixFree : C::OffDiagonalStorage::Assembled)),
              rhs(space, layout == 0 ? R::RowLayout::Dense(0, 5, 8) : layout == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                  : R::RowLayout::TiledColMajor(0, 5, 32))
        {
            {
                auto diagonal = problem->Diagonal().Write();
                std::optional<A::BlockMatrixStorage::WriteAccess> off;
                if (!evaluated)
                    off.emplace(problem->OffDiagonal().Write());
                auto b = rhs.Owned().Write();
                for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                {
                    const auto block = Diagonal(iCell);
                    for (int j = 0; j < 25; ++j)
                        diagonal.View()(iCell, j) = block.data()[j];
                    for (DNDS::index k = 0; k < DNDS::index(columns[iCell].size()); ++k)
                        if (off)
                            off->View()(iCell, k) = Off(iCell, columns[iCell][k]);
                    for (int j = 0; j < 5; ++j)
                        b.View<R::UniformTableRows>()(iCell, j) = RHS(rank, iCell, j);
                }
            }
            if (evaluated)
                problem->PublishOffDiagonalEvaluator(std::make_shared<EvaluatedOff>(space));
        }
        R::TransferBinding Bind(Field &field) const
        {
            return R::PayloadPlan(route, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::Arithmetic<double>()).Bind(field.Owned(), field.Ghost());
        }
        Eigen::VectorXd Oracle() const
        {
            // Independent, tiny global dense solve. Ghost 4/5 denote the
            // preceding rank's owned rows 2/3, even for the self-ring case.
            Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(20 * size, 20 * size);
            Eigen::VectorXd b(20 * size);
            for (int owner = 0; owner < size; ++owner)
                for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                {
                    const DNDS::index row = 20 * owner + 5 * iCell;
                    matrix.block<5, 5>(row, row) += Diagonal(iCell);
                    for (auto other : columns[iCell])
                    {
                        const int source = other < 4 ? owner : (owner + size - 1) % size;
                        const DNDS::index local = other < 4 ? other : other - 2;
                        matrix.block<5, 5>(row, 20 * source + 5 * local) += Off(iCell, other);
                    }
                    for (int j = 0; j < 5; ++j)
                        b[row + j] = RHS(owner, iCell, j);
                }
            return matrix.fullPivLu().solve(b);
        }
    };
}
TEST_CASE("SGS and matrix action use direct evaluated products on strided rows")
{
    for (int layout : {0, 1, 2})
    {
        Fixture assembled(layout), evaluated(layout, true);
        evaluated.problem->PublishOffDiagonalEvaluator(std::make_shared<ActionOnlyOff>(evaluated.space));
        C::LinearSolveSettings settings;
        settings.method = "sgs";
        settings.stopping = "fixedCount";
        settings.maxIterations = 2;
        settings.exchangeBetweenSweeps = true;
        C::LinearSolver a(assembled.problem, assembled.rhs, [&](Field &field)
                          { return assembled.Bind(field); }, settings);
        C::LinearSolver b(evaluated.problem, evaluated.rhs, [&](Field &field)
                          { return evaluated.Bind(field); }, settings);
        REQUIRE(a.Solve(assembled.rhs).usable);
        REQUIRE(b.Solve(evaluated.rhs).usable);
        Field resultA(assembled.space, assembled.rhs.Owned().Layout());
        Field resultB(evaluated.space, evaluated.rhs.Owned().Layout());
        assembled.Bind(assembled.rhs).Begin().Wait();
        evaluated.Bind(evaluated.rhs).Begin().Wait();
        assembled.problem->Apply(assembled.rhs, resultA);
        evaluated.problem->Apply(evaluated.rhs, resultB);
        const auto xa = a.Solution().Owned().Read(), xb = b.Solution().Owned().Read();
        const auto ya = resultA.Owned().Read(), yb = resultB.Owned().Read();
        for (DNDS::index iCell = 0; iCell < 4; ++iCell)
            for (int j = 0; j < 5; ++j)
            {
                CHECK(xa.View<R::UniformTableRows>()(iCell, j) == doctest::Approx(xb.View<R::UniformTableRows>()(iCell, j)).epsilon(2e-14));
                CHECK(ya.View<R::UniformTableRows>()(iCell, j) == doctest::Approx(yb.View<R::UniformTableRows>()(iCell, j)).epsilon(2e-14));
            }
    }
}
TEST_CASE("Jacobi matches independent frozen-remote local-sweep oracle")
{
    for (bool evaluated : {false, true})
        for (int localSweeps : {1, 2})
            for (bool warm : {false, true})
                for (int iterations : {1, 2, 8})
                {
                    Fixture fixture(evaluated ? 1 : 2, evaluated);
                    C::LinearSolveSettings settings;
                    settings.method = "jacobi";
                    settings.stopping = "fixedCount";
                    settings.maxIterations = iterations;
                    settings.jacobiLocalSweeps = localSweeps;
                    C::LinearSolver solver(fixture.problem, fixture.rhs, [&](Field &field)
                                           { return fixture.Bind(field); }, settings);
                    Eigen::VectorXd expected = Eigen::VectorXd::Zero(20 * fixture.size);
                    if (warm)
                        for (int rank = 0; rank < fixture.size; ++rank)
                            for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                                for (int j = 0; j < 5; ++j)
                                    expected[20 * rank + 5 * iCell + j] = RHS(rank, iCell, j);
                    for (int iteration = 0; iteration < iterations; ++iteration)
                    {
                        const Eigen::VectorXd frozen = expected;
                        for (int sweep = 0; sweep < localSweeps; ++sweep)
                        {
                            Eigen::VectorXd next = expected;
                            for (int rank = 0; rank < fixture.size; ++rank)
                                for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                                {
                                    Eigen::Matrix<double, 5, 1> value;
                                    for (int j = 0; j < 5; ++j)
                                        value[j] = RHS(rank, iCell, j);
                                    for (auto column : columns[iCell])
                                        if (column < 4)
                                            value -= Off(iCell, column) * expected.segment<5>(20 * rank + 5 * column);
                                        else
                                            value -= Off(iCell, column) * frozen.segment<5>(20 * ((rank + fixture.size - 1) % fixture.size) + 5 * (column - 2));
                                    next.segment<5>(20 * rank + 5 * iCell) = Diagonal(iCell).fullPivLu().solve(value);
                                }
                            expected = std::move(next);
                        }
                    }
                    const auto report = solver.Solve(fixture.rhs, warm ? DNDS::OptionalRef<const C::LinearSolver::Field>(fixture.rhs) : DNDS::OptionalRef<const C::LinearSolver::Field>{});
                    REQUIRE(report.usable);
                    CHECK(report.jacobiSweeps == iterations * localSweeps);
                    CHECK(report.jacobiInputExchanges == iterations - !warm);
                    C::EquationSchema schema{"mock-coupled", "identity", "identity", {}};
                    A::DofBundle fields(fixture.space);
                    for (int block = 0; block < 2; ++block)
                    {
                        C::EquationBlockSchema part;
                        part.name = block == 0 ? "first" : "second";
                        const int width = block == 0 ? 3 : 2;
                        for (int j = 0; j < width; ++j)
                            part.components.push_back({part.name + std::to_string(j), "", part.name});
                        fields.Add(part.name, Field(fixture.space, block == 0 ? R::RowLayout::ColMajor(0, width) : R::RowLayout::TiledColMajor(0, width, 8)));
                        schema.blocks.push_back(std::move(part));
                    }
                    auto split = C::EquationDofs::Separate(std::move(schema), std::move(fields));
                    {
                        auto write = split.WriteOwned();
                        const auto layout = write.ComponentLayout();
                        for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                            for (int j = 0; j < 5; ++j)
                                layout.View()(iCell, j) = RHS(fixture.rank, iCell, j);
                    }
                    C::DofLinearSolver splitSolver(fixture.problem, split, [&](Field &field)
                                                   { return fixture.Bind(field); }, settings, {false, true});
                    REQUIRE(splitSolver.Solve(split, warm ? DNDS::OptionalRef<const C::EquationDofs>(split) : DNDS::OptionalRef<const C::EquationDofs>{}).usable);
                    const auto splitRead = splitSolver.Solution().ReadOwned();
                    const auto splitLayout = splitRead.ComponentLayout();
                    const auto result = solver.Solution().Owned().Read();
                    const auto input = fixture.rhs.Owned().Read();
                    for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                        for (int j = 0; j < 5; ++j)
                        {
                            CHECK(result.View<R::UniformTableRows>()(iCell, j) == doctest::Approx(expected[20 * fixture.rank + 5 * iCell + j]).epsilon(2e-13));
                            CHECK(input.View<R::UniformTableRows>()(iCell, j) == RHS(fixture.rank, iCell, j));
                            CHECK(splitLayout.View()(iCell, j) == doctest::Approx(expected[20 * fixture.rank + 5 * iCell + j]).epsilon(2e-13));
                        }
                }
}

TEST_CASE("Dedicated zero-start SGS needs no old storage and preserves explicit-zero results")
{
    for (int layout : {0, 1, 2})
        for (bool evaluated : {false, true})
            for (bool middle : {false, true})
            {
                Fixture fixture(layout, evaluated);
                std::shared_ptr<CountedOff> counter;
                if (evaluated)
                {
                    counter = std::make_shared<CountedOff>(fixture.space);
                    fixture.problem->PublishOffDiagonalEvaluator(counter);
                }
                fixture.problem->InvertDiagonal();
                const auto shape = fixture.rhs.Owned().Layout();
                Field zero(fixture.space, shape), forwardA(fixture.space, shape), forwardB(fixture.space, shape);
                Field resultA(fixture.space, shape), resultB(fixture.space, shape);
                for (auto *part : {&zero.Owned(), &zero.Ghost()})
                {
                    auto write = part->Write();
                    const auto view = write.View<R::UniformTableRows>();
                    for (DNDS::index iRow = 0; iRow < view.RowCount(); ++iRow)
                        for (int j = 0; j < 5; ++j)
                            view(iRow, j) = 0;
                }
                const auto run = [&](auto &rhs, auto &old, auto &forward, auto &result, Field &physicalForward, bool knownZero)
                {
                    if (counter)
                        counter->products = 0;
                    std::optional<R::Array<double>::WriteLease> unusedOwned, unusedGhost;
                    if (knownZero)
                    {
                        unusedOwned.emplace(zero.Owned().Write());
                        unusedGhost.emplace(zero.Ghost().Write());
                    }
                    auto sweep = knownZero ? fixture.problem->BeginSGSZero(rhs, forward, result, middle)
                                           : fixture.problem->BeginSGS(rhs, old, forward, result, middle);
                    sweep.Lower(C::LinearRegion::Core);
                    sweep.Lower(C::LinearRegion::HaloDependent);
                    if (counter)
                        CHECK(counter->products == (knownZero ? 3 : 8));
                    if (middle)
                        fixture.Bind(physicalForward).Begin().Wait();
                    sweep.Upper(C::LinearRegion::HaloDependent);
                    sweep.Upper(C::LinearRegion::Core);
                    sweep.Finish();
                    // Local upper retains all6 local products; with middle
                    // exchange it additionally consumes both forward ghosts.
                    if (counter)
                        CHECK(counter->products == (knownZero ? (middle ? 11 : 9) : 16));
                };
                for (bool logical : {false, true})
                {
                    if (logical)
                    {
                        const auto schema = C::EquationSystem::Resolve(C::EquationSettings{}).Schema();
                        auto rhs = C::EquationDofs::Coupled(schema, fixture.rhs), old = C::EquationDofs::Coupled(schema, zero);
                        auto a = C::EquationDofs::Coupled(schema, forwardA), b = C::EquationDofs::Coupled(schema, forwardB);
                        auto x = C::EquationDofs::Coupled(schema, resultA), y = C::EquationDofs::Coupled(schema, resultB);
                        run(rhs, old, a, x, forwardA, false);
                        run(rhs, old, b, y, forwardB, true);
                    }
                    else
                    {
                        run(fixture.rhs, zero, forwardA, resultA, forwardA, false);
                        run(fixture.rhs, zero, forwardB, resultB, forwardB, true);
                    }
                    const auto a = resultA.Owned().Read(), b = resultB.Owned().Read();
                    for (DNDS::index iRow = 0; iRow < 4; ++iRow)
                        for (int j = 0; j < 5; ++j)
                            CHECK(a.View<R::UniformTableRows>()(iRow, j) == b.View<R::UniformTableRows>()(iRow, j));
                }
            }
}
TEST_CASE("Residual observation cadence preserves stationary iterates and skips only diagnostic reductions")
{
    for (const auto method : {"sgs", "ilu"})
    {
        Fixture fixture;
        C::LinearSolveSettings settings;
        settings.method = method;
        settings.maxIterations = 5;
        settings.relativeTolerance = settings.absoluteTolerance = 0;
        C::LinearSolver frequent(fixture.problem, fixture.rhs, [&](Field &field)
                                 { return fixture.Bind(field); }, settings);
        settings.residualCheckInterval = 3;
        C::LinearSolver periodic(fixture.problem, fixture.rhs, [&](Field &field)
                                 { return fixture.Bind(field); }, settings);
        const auto a = frequent.Solve(fixture.rhs), b = periodic.Solve(fixture.rhs);
        CHECK(a.iterations == 5);
        CHECK(b.iterations == 5);
        CHECK(a.innerProductBatches == 7); // RHS, five residuals, final iterate.
        CHECK(b.innerProductBatches == 4); // RHS, iterations 3/5, final iterate.
        CHECK(a.trueResidual == doctest::Approx(b.trueResidual).epsilon(1e-13));
        const auto x = frequent.Solution().Owned().Read(), y = periodic.Solution().Owned().Read();
        for (DNDS::index iCell = 0; iCell < 4; ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(x.View<R::UniformTableRows>()(iCell, j) == y.View<R::UniformTableRows>()(iCell, j));
    }
}
TEST_CASE("Residual-stopped SGS publishes zero when a repeated solve needs no sweep")
{
    Fixture fixture;
    C::LinearSolveSettings settings;
    settings.method = "sgs";
    settings.maxIterations = 3;
    C::LinearSolver solver(fixture.problem, fixture.rhs, [&](Field &field)
                           { return fixture.Bind(field); }, settings);
    REQUIRE(solver.Solve(fixture.rhs).usable);
    A::Fill(fixture.rhs, 0.);
    const auto result = solver.Solve(fixture.rhs);
    CHECK(result.converged);
    CHECK(result.iterations == 0);
    const auto read = solver.Solution().Owned().Read();
    for (DNDS::index iRow = 0; iRow < 4; ++iRow)
        for (int j = 0; j < 5; ++j)
            CHECK(read.View<R::UniformTableRows>()(iRow, j) == 0);
}
TEST_CASE("Cauchy coupled linear solvers match a global dense oracle with real persistent halos")
{
    for (int selection = 0; selection < 16; ++selection)
    {
        const int scenario = selection % 8;
        Fixture fixture(scenario % 3, selection >= 8);
        C::LinearSolveSettings settings;
        settings.method = scenario < 3 ? "sgs" : scenario == 7 ? "ilu"
                                                               : "gmres";
        settings.preconditioner = scenario >= 5 ? "ilu" : "sgs";
        settings.exchangeBetweenSweeps = scenario == 2 || scenario == 4;
        settings.sgsSweeps = 2;
        settings.subspace = 8;
        settings.maxIterations = 80;
        settings.relativeTolerance = 1e-11;
        settings.absoluteTolerance = 1e-13;
        R::EagerEngine engine(scenario % 3 == 0 ? 2 : 0,
                              std::make_shared<R::AccessTrackingContext>());
        C::LinearExecution execution{scenario % 2 == 0, scenario % 3 != 1};
        execution.streams = C::LinearExecution::Streams{engine.CreateCPUStream(), engine.CreateCPUStream()};
        settings.residualCheckInterval = scenario >= 5 ? 3 : 1;
        C::LinearSolver solver(fixture.problem, fixture.rhs, [&](Field &field)
                               { return fixture.Bind(field); }, settings, execution);
        const auto version = fixture.rhs.Owned().Version();
        const auto report = solver.Solve(fixture.rhs);
        CHECK(report.converged);
        CHECK(report.usable);
        CHECK(report.iterations > 0);
        CHECK(report.trueResidual <= std::max(settings.absoluteTolerance, settings.relativeTolerance * report.initialTrueResidual));
        CHECK(fixture.rhs.Owned().Version() == version);
        const auto expected = fixture.Oracle();
        std::array<double, 20> first{};
        {
            const auto read = solver.Solution().Owned().Read();
            const auto values = read.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    first[5 * iCell + j] = values(iCell, j);
                    CHECK(values(iCell, j) == doctest::Approx(expected[fixture.rank * 20 + 5 * iCell + j]).epsilon(1e-10));
                }
        }
        // A new fixed preconditioner application starts from zero; no stale
        // previous solve may change either the correction or iteration count.
        const auto repeated = solver.Solve(fixture.rhs);
        CHECK(repeated.converged);
        CHECK(repeated.iterations == report.iterations);
        {
            const auto read = solver.Solution().Owned().Read();
            const auto values = read.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(values(iCell, j) == first[5 * iCell + j]);
        }
        const auto warmGuess = solver.Solution();
        const auto warm = solver.Solve(fixture.rhs, std::cref(warmGuess));
        CHECK(warm.converged);
        CHECK(warm.iterations == 0);
        const auto solutionVersion = solver.Solution().Owned().Version();
        CHECK_THROWS(solver.Solve(solver.Solution()));
        CHECK(solver.Solution().Owned().Version() == solutionVersion);
        CHECK(solver.Solve(fixture.rhs).converged); // Preflight rejection is reusable.
    }
}

TEST_CASE("Logical coupled SGS retains physical-field scan values with remote middle exchange")
{
    for (int components : {5, 6, 7})
        for (int layout = 0; layout < 3; ++layout)
        {
            Fixture fixture(layout);
            const auto prototype = layout == 0   ? R::RowLayout::Dense(0, components, components + 3)
                                   : layout == 1 ? R::RowLayout::ColMajor(0, components, {8, 1})
                                                 : R::RowLayout::TiledColMajor(0, components, 32);
            Field rhs(fixture.space, prototype);
            auto problem = std::make_shared<C::LinearProblemContext>(fixture.space, components, columns,
                                                                     std::vector<DNDS::index>{1, 0}, C::DiagonalStorage::Block);
            {
                auto diagonal = problem->Diagonal().Write();
                auto off = problem->OffDiagonal().Write();
                auto write = rhs.Owned().Write();
                const auto blocks = A::MakeEigenRowMatrices(diagonal.View<R::TableRows>(), components, components);
                for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                {
                    blocks[iCell].setIdentity();
                    blocks[iCell] *= 3 + .1 * iCell;
                    for (DNDS::index j = 0; j < components; ++j)
                    {
                        blocks[iCell](j, (j + 1) % components) = .02;
                        write.View<R::UniformTableRows>()(iCell, j) = RHS(fixture.rank, iCell, int(j));
                    }
                    for (DNDS::index k = 0; k < DNDS::index(columns[iCell].size()); ++k)
                        off.View()(iCell, k) = Eigen::MatrixXd::Identity(components, components) * (-.1 - .001 * iCell);
                }
            }
            auto schema = C::EquationSystem::Resolve(C::EquationSettings{}).Schema();
            for (int j = 5; j < components; ++j)
                schema.blocks[0].components.push_back({"extra" + std::to_string(j), "mock", "flow"});
            auto logicalRHS = C::EquationDofs::Coupled(schema, rhs);
            C::LinearSolveSettings settings;
            settings.method = "sgs";
            settings.stopping = "fixedCount";
            settings.maxIterations = 2;
            settings.exchangeBetweenSweeps = true;
            const auto bind = [&](Field &field)
            { return fixture.Bind(field); };
            C::LinearSolver fieldSolver(problem, rhs, bind, settings, {true, false});
            C::DofLinearSolver logicalSolver(problem, logicalRHS, bind, settings, {true, false});
            REQUIRE(fieldSolver.Solve(rhs).usable);
            REQUIRE(logicalSolver.Solve(logicalRHS).usable);
            const auto physical = fieldSolver.Solution().Owned().Read();
            const auto logical = logicalSolver.Solution().ReadOwned();
            const auto values = physical.View<R::UniformTableRows>();
            const auto logicalLayout = logical.ComponentLayout();
            for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                for (DNDS::index j = 0; j < components; ++j)
                    CHECK(logicalLayout.View()(iCell, j) == doctest::Approx(values(iCell, j)).epsilon(2e-14));
        }
}

TEST_CASE("GMRES fixed ILU corrections match one Arnoldi step of the distributed dense preconditioner")
{
    Fixture fixture(1);
    const int count = 20 * fixture.size;
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(count, count), local = matrix;
    Eigen::VectorXd b(count);
    for (int owner = 0; owner < fixture.size; ++owner)
        for (DNDS::index iCell = 0; iCell < 4; ++iCell)
        {
            const DNDS::index row = 20 * owner + 5 * iCell;
            matrix.block<5, 5>(row, row) = local.block<5, 5>(row, row) = Diagonal(iCell);
            for (auto other : columns[iCell])
            {
                const DNDS::index column = other < 4 ? 20 * owner + 5 * other
                                                     : 20 * ((owner + fixture.size - 1) % fixture.size) + 5 * (other - 2);
                matrix.block<5, 5>(row, column) += Off(iCell, other);
                if (other < 4)
                    local.block<5, 5>(row, column) += Off(iCell, other);
            }
            for (int j = 0; j < 5; ++j)
                b[row + j] = RHS(owner, iCell, j);
        }
    const Eigen::MatrixXd inverse = local.fullPivLu().inverse();
    const auto schema = C::EquationSystem::Resolve(C::EquationSettings{}).Schema();
    auto logicalRHS = C::EquationDofs::Coupled(schema, fixture.rhs);
    for (int corrections : {1, 2})
    {
        C::LinearSolveSettings settings;
        settings.method = "gmres";
        settings.preconditioner = "ilu";
        settings.iluPreconditionerCorrections = corrections;
        settings.maxIterations = settings.subspace = 1;
        settings.relativeTolerance = settings.absoluteTolerance = 0;
        settings.iluCoreFirst = false;
        const auto bind = [&](Field &field)
        { return fixture.Bind(field); };
        C::LinearSolver physical(fixture.problem, fixture.rhs, bind, settings, {true, false});
        C::DofLinearSolver logical(fixture.problem, logicalRHS, bind, settings, {true, false});
        // The local rank block is tridiagonal: its ILU(0) is the exact local
        // inverse. Build B_c=sum_{k=0}^{c-1}(I-M^-1 A)^k M^-1 independently.
        Eigen::MatrixXd preconditioner = inverse;
        for (int i = 1; i < corrections; ++i)
            preconditioner += inverse * (Eigen::MatrixXd::Identity(count, count) - matrix * preconditioner);
        const Eigen::VectorXd z = preconditioner * b;
        const Eigen::VectorXd action = preconditioner * matrix * z;
        const std::vector<double> scales = corrections == 1 ? std::vector<double>{} : std::vector<double>{2, .5, 3, 4, 7};
        Eigen::VectorXd weightedZ = z, weightedAction = action;
        if (!scales.empty())
            for (int i = 0; i < count; ++i)
                weightedZ[i] /= scales[i % 5], weightedAction[i] /= scales[i % 5];
        const Eigen::VectorXd expected = z * (weightedAction.dot(weightedZ) / weightedAction.squaredNorm());
        const auto first = physical.Solve(fixture.rhs, {}, scales), second = logical.Solve(logicalRHS, {}, scales);
        REQUIRE(first.usable);
        REQUIRE(second.usable);
        CHECK(first.iterations == 1);
        CHECK(second.iterations == 1);
        const auto read = physical.Solution().Owned().Read();
        const auto logicalRead = logical.Solution().ReadOwned();
        const auto layout = logicalRead.ComponentLayout();
        for (DNDS::index iCell = 0; iCell < 4; ++iCell)
            for (int j = 0; j < 5; ++j)
            {
                const auto value = expected[20 * fixture.rank + 5 * iCell + j];
                CHECK(read.View<R::UniformTableRows>()(iCell, j) == doctest::Approx(value).epsilon(2e-12));
                CHECK(layout.View()(iCell, j) == doctest::Approx(value).epsilon(2e-12));
            }
    }
}
TEST_CASE("Cauchy linear controls and rank-local access failures reject collectively before transfers")
{
    Fixture fixture;
    C::LinearSolveSettings settings;
    auto bind = [&](Field &field)
    { return fixture.Bind(field); };
    settings.method = "invalid";
    CHECK_THROWS(C::LinearSolver(fixture.problem, fixture.rhs, bind, settings));
    settings.method = "ilu";
    settings.iluOrdering = "unknown";
    CHECK_THROWS(C::LinearSolver(fixture.problem, fixture.rhs, bind, settings));
    settings.iluOrdering = "natural";
    settings.method = "gmres";
    settings.stopping = "fixedCount";
    CHECK_THROWS(C::LinearSolver(fixture.problem, fixture.rhs, bind, settings));
    settings.stopping = "residual";
    settings.preconditioner = "ilu";
    settings.exchangeBetweenSweeps = true;
    CHECK_THROWS(C::LinearSolver(fixture.problem, fixture.rhs, bind, settings));
    if (fixture.size > 1)
    {
        settings.exchangeBetweenSweeps = false;
        settings.subspace = 4 + fixture.rank;
        CHECK_THROWS(C::LinearSolver(fixture.problem, fixture.rhs, bind, settings));
    }
    settings.subspace = 8;
    settings.exchangeBetweenSweeps = false;
    settings.relativeTolerance = 1e-10;
    settings.maxIterations = 40;
    C::LinearSolver solver(fixture.problem, fixture.rhs, bind, settings);
    std::optional<R::Array<double>::WriteLease> writer;
    if (fixture.rank == 0)
        writer.emplace(fixture.rhs.Owned().Write());
    CHECK_THROWS(solver.Solve(fixture.rhs));
    writer.reset();
    if (fixture.rank == 0)
        writer.emplace(fixture.problem->Diagonal().Write());
    CHECK_THROWS(solver.Solve(fixture.rhs));
    writer.reset();
    CHECK(solver.Solve(fixture.rhs).converged);
}
TEST_CASE("Fixed-count stationary corrections preserve iterates without diagnostic actions or products")
{
    const auto schema = C::EquationSystem::Resolve(C::EquationSettings{}).Schema();
    for (const auto *method : {"sgs", "ilu"})
    {
        Fixture fixture(std::string(method) == "sgs" ? 1 : 2);
        const auto bind = [&](Field &field)
        { return fixture.Bind(field); };
        C::LinearSolveSettings settings;
        settings.method = method;
        settings.maxIterations = 2;
        settings.relativeTolerance = settings.absoluteTolerance = 0;
        settings.exchangeBetweenSweeps = settings.method == "sgs";
        const C::LinearExecution execution{settings.method == "ilu", false};
        C::LinearSolver reference(fixture.problem, fixture.rhs, bind, settings, execution);
        auto logicalRHS = C::EquationDofs::Coupled(schema, fixture.rhs);
        C::DofLinearSolver logicalReference(fixture.problem, logicalRHS, bind, settings, execution);
        settings.stopping = "fixedCount";
        settings.relativeTolerance = 1e20; // Explicit fixed count ignores tolerance.
        C::LinearSolver fixed(fixture.problem, fixture.rhs, bind, settings, execution);
        C::DofLinearSolver logical(fixture.problem, logicalRHS, bind, settings, execution);
        const auto rhsVersion = fixture.rhs.Owned().Version();
        for (bool warm : {false, true})
        {
            const auto checked = reference.Solve(fixture.rhs, warm ? DNDS::OptionalRef<const C::LinearSolver::Field>(fixture.rhs) : DNDS::OptionalRef<const C::LinearSolver::Field>{});
            const auto checkedLogical = logicalReference.Solve(logicalRHS, warm ? DNDS::OptionalRef<const C::EquationDofs>(logicalRHS) : DNDS::OptionalRef<const C::EquationDofs>{});
            REQUIRE(checked.iterations == 2);
            REQUIRE(checkedLogical.iterations == 2);
            const auto report = fixed.Solve(fixture.rhs, warm ? DNDS::OptionalRef<const C::LinearSolver::Field>(fixture.rhs) : DNDS::OptionalRef<const C::LinearSolver::Field>{});
            const auto equationReport = logical.Solve(logicalRHS, warm ? DNDS::OptionalRef<const C::EquationDofs>(logicalRHS) : DNDS::OptionalRef<const C::EquationDofs>{});
            for (const auto *value : {&report, &equationReport})
            {
                CHECK(value->stop == DNDS::Linear::LinearStop::FixedCount);
                CHECK(value->usable);
                CHECK_FALSE(value->converged);
                CHECK(value->iterations == 2);
                CHECK(std::isnan(value->initialTrueResidual));
                CHECK(std::isnan(value->trueResidual));
                CHECK(value->innerProductBatches == 0);
                CHECK(value->operatorApplications == (settings.method == "ilu" ? (warm ? 2 : 1) : 0));
            }
            const auto a = reference.Solution().Owned().Read(), b = fixed.Solution().Owned().Read();
            const auto c = logical.Solution().ReadOwned();
            const auto d = logicalReference.Solution().ReadOwned();
            const auto expected = a.View<R::UniformTableRows>(), actual = b.View<R::UniformTableRows>();
            const auto equation = c.Block("flow");
            const auto expectedEquation = d.Block("flow");
            for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                for (DNDS::index j = 0; j < 5; ++j)
                {
                    CHECK(actual(iCell, j) == expected(iCell, j));
                    // Fixed-count changes reporting, not the adapter's
                    // arithmetic. Static logical and dynamic physical Eigen
                    // kernels can differ in their last summation bits.
                    CHECK(equation(iCell, j) == expectedEquation(iCell, j));
                    CHECK(equation(iCell, j) == doctest::Approx(expected(iCell, j)).epsilon(2e-14));
                }
        }
        CHECK(fixture.rhs.Owned().Version() == rhsVersion);
    }
}
TEST_CASE("Fixed-count zero-start ILU overwrites dirty solution and handles zero corrections")
{
    Fixture fixture(2);
    const auto bind = [&](Field &field)
    { return fixture.Bind(field); };
    const auto schema = C::EquationSystem::Resolve(C::EquationSettings{}).Schema();
    auto rhs = C::EquationDofs::Coupled(schema, fixture.rhs);
    for (int count : {0, 1})
    {
        C::LinearSolveSettings settings;
        settings.method = "ilu";
        settings.stopping = "fixedCount";
        settings.maxIterations = count;
        C::DofLinearSolver solver(fixture.problem, rhs, bind, settings, {false, false});
        // Poison all owned output before each call; analytic zero must not
        // require clearing or reading it on the first triangular solve.
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            auto output = solver.Solution().NumericAlias();
            C::EquationDofOperations::Fill(output, std::numeric_limits<double>::quiet_NaN());
            const auto report = solver.Solve(rhs);
            CHECK(report.usable);
            CHECK(report.iterations == count);
            CHECK(report.operatorApplications == 0);
            const auto read = solver.Solution().ReadOwned();
            const auto values = read.Block("flow");
            for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    CHECK(std::isfinite(values(iCell, j)));
                    if (!count)
                        CHECK(values(iCell, j) == 0);
                }
        }
    }
}
TEST_CASE("Fixed-count final finite check agrees on a rank-local nonfinite update")
{
    Fixture fixture;
    C::LinearSolveSettings settings;
    settings.stopping = "fixedCount";
    settings.maxIterations = 0;
    Field guess(fixture.space, 5);
    {
        auto write = guess.Owned().Write();
        const auto values = write.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < 4; ++iCell)
            for (DNDS::index j = 0; j < 5; ++j)
                values(iCell, j) = fixture.rank == 0 ? std::numeric_limits<double>::quiet_NaN() : 0;
    }
    const bool fatalLocal = std::getenv("DNDS_TEST_LINEAR_FATAL_LOCAL") != nullptr;
    C::LinearSolver solver(fixture.problem, fixture.rhs, [&](Field &field)
                           { return fixture.Bind(field); }, settings, {false, !fatalLocal});
    const auto report = solver.Solve(fixture.rhs, std::cref(guess));
    if (fatalLocal)
    {
        // This subprocess tests termination on the invalid rank. Valid
        // peers may return their local result before that rank aborts.
        MPI_Barrier(fixture.space.Communicator());
        return;
    }
    CHECK_FALSE(report.usable);
    CHECK_FALSE(report.converged);
    CHECK(report.stop == DNDS::Linear::LinearStop::NonFinite);
    CHECK(report.iterations == 0);
    CHECK(report.innerProductBatches == 0);
}
TEST_CASE("Evaluated block publication validates row identity, column ordinals and active read scopes")
{
    Fixture fixture(1, true);
    const auto generation = fixture.problem->OffDiagonalGeneration();
    CHECK_THROWS(fixture.problem->PublishOffDiagonalEvaluator(nullptr));
    {
        A::DofSpace other(4, 2);
        CHECK_THROWS(fixture.problem->PublishOffDiagonalEvaluator(std::make_shared<EvaluatedOff>(other)));
    }
    for (int error = 0; error < 2; ++error)
    {
        auto invalid = std::make_shared<EvaluatedOff>(fixture.space);
        if (error == 0)
            invalid->components = 4;
        else
            std::swap(invalid->declaredColumns[2][0], invalid->declaredColumns[2][1]);
        CHECK_THROWS(fixture.problem->PublishOffDiagonalEvaluator(invalid));
    }
    {
        auto held = fixture.problem->ReadOffDiagonal();
        CHECK_THROWS(fixture.problem->PublishOffDiagonalEvaluator(std::make_shared<EvaluatedOff>(fixture.space)));
    }
    CHECK(fixture.problem->OffDiagonalGeneration() == generation);
    fixture.problem->PublishOffDiagonalEvaluator(std::make_shared<EvaluatedOff>(fixture.space));
    CHECK(fixture.problem->OffDiagonalGeneration() == generation + 1);
    {
        Fixture assembled;
        CHECK_THROWS(assembled.problem->PublishOffDiagonalEvaluator(std::make_shared<EvaluatedOff>(assembled.space)));
    }
    if (fixture.size > 1)
    {
        Fixture mixed(0, fixture.rank % 2);
        CHECK_THROWS(C::LinearSolver(mixed.problem, mixed.rhs, [&](Field &field)
                                     { return mixed.Bind(field); }));
    }
    std::optional<C::LinearProblemContext::OffDiagonalReadAccess> retained;
    {
        Fixture temporary(2, true);
        retained.emplace(temporary.problem->ReadOffDiagonal());
        temporary.problem.reset();
    }
    // Numeric access retains no communicator owners. Its frozen row descriptor
    // and producer survive the complete original context/fixture lifetime.
    retained->WithBlocks([&](const auto &view)
                         {
    const auto first = view(0, 0), second = view(1, 0);
    CHECK(first.isApprox(Off(0, 1), 1e-15));
    CHECK(second.isApprox(Off(1, 0), 1e-15));
    for (DNDS::index iCell = 0; iCell < 4; ++iCell)
        for (DNDS::index j = 0; j < DNDS::index(columns[iCell].size()); ++j)
            CHECK(view(iCell, j).isApprox(Off(iCell, columns[iCell][j]), 1e-15)); });
}
int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
