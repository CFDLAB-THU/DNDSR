#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Cauchy/IO/ProgressOutput.hpp"
#include "Cauchy/IO/OutputPlan.hpp"
#include "Cauchy/IO/OutputQueue.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <future>
#include <thread>
#include <atomic>

TEST_CASE("Output queue snapshots precede later mutations and writes stay ordered")
{
    using Queue = DNDS::Cauchy::OutputQueue;
    std::atomic<bool> failed{false}, wrongThread{false};
    const auto controller = std::this_thread::get_id();
    std::vector<int> source{1, 2, 3}, observed;
    std::promise<void> started, release;
    auto startedEvent = started.get_future();
    auto releaseEvent = release.get_future().share();
    Queue queue([&](std::exception_ptr)
                { failed = true; }, 2);
    queue.Enqueue([&]() -> Queue::Action
                  {
        CHECK(std::this_thread::get_id() == controller);
        auto snapshot = source;
        return [&, snapshot = std::move(snapshot)]
        {
            wrongThread.store(std::this_thread::get_id() == controller);
            started.set_value();
            releaseEvent.wait();
            observed.insert(observed.end(), snapshot.begin(), snapshot.end());
        }; });
    startedEvent.wait();
    source = {4, 5};
    queue.Enqueue([&]() -> Queue::Action
                  {
        auto snapshot = source;
        return [&, snapshot = std::move(snapshot)] { observed.insert(observed.end(), snapshot.begin(), snapshot.end()); }; });
    source = {99};
    release.set_value();
    queue.Drain();
    CHECK_FALSE(failed.load());
    CHECK_FALSE(wrongThread.load());
    CHECK(observed == std::vector<int>{1, 2, 3, 4, 5});
    queue.Close();
}

TEST_CASE("Output queue records a terminal writer failure without running later work")
{
    using Queue = DNDS::Cauchy::OutputQueue;
    std::atomic<int> failures{0};
    // Returning from this handler is specifically a UT allowance. Solver
    // collective writers must abort locally, never catch and resume.
    Queue queue([&](std::exception_ptr)
                { ++failures; });
    queue.Enqueue([]() -> Queue::Action
                  { return []
                    { throw std::runtime_error("test writer failure"); }; });
    CHECK_THROWS_AS(queue.Drain(), std::runtime_error);
    bool preparedAgain = false;
    CHECK_THROWS(queue.Enqueue([&]() -> Queue::Action
                               { preparedAgain = true; return [] {}; }));
    CHECK_FALSE(preparedAgain);
    CHECK(failures.load() == 1);
    queue.Close();
}

namespace C = DNDS::Cauchy;

namespace
{
    std::vector<std::string> Split(const std::string &line)
    {
        std::vector<std::string> fields;
        std::stringstream stream(line);
        for (std::string field; std::getline(stream, field, ',');)
            fields.push_back(std::move(field));
        if (!line.empty() && line.back() == ',')
            fields.emplace_back();
        return fields;
    }

    std::map<std::string, std::string> ReadRow(std::ifstream &stream, const std::vector<std::string> &header)
    {
        std::string line;
        REQUIRE(std::getline(stream, line));
        const auto fields = Split(line);
        REQUIRE(fields.size() == header.size());
        std::map<std::string, std::string> result;
        for (size_t i = 0; i < header.size(); ++i)
            result.emplace(header[i], fields[i]);
        return result;
    }
}

TEST_CASE("One progress controller constructs iteration stage and step CSV rows")
{
    C::EquationSchema schema;
    schema.family = "test";
    schema.blocks.push_back({"flow", {{"rho", "", "flow"}, {"rhoE", "", "flow"}}, {}});

    C::OutputSettings settings;
    settings.consoleInterval = 0;
    settings.implicit.consoleInterval = 0;
    settings.implicit.historyInterval = 2;
    settings.historySegmentRows = 2;
    settings.historyPrecision = 8;
    const auto serial = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() / ("dnds-cauchy-progress-" + std::to_string(serial));
    const auto bare = (directory / "case-D0-history").string();
    settings.prefix = (directory / "case").string();
    auto logger = std::make_shared<C::OperationalLogger>(settings, 0);
    DNDS::CsvLogPosition resume;
    {
        C::ProgressOutputController output(schema, settings, "fluid", bare, true, logger);
        output.BeginAttempt({3, -1, -1, -1, 1, .3, .01, 2});
        C::NonlinearProgress progress;
        progress.stage = 1;
        progress.coupledStage = 0;
        progress.residual = {2, 3};
        progress.initialResidual = {4, 5};
        progress.referenceResidual = {8, 10};
        progress.residualTargetRatio = .5;
        progress.pseudoCFL = 7;
        progress.iteration = 1; // filtered by the interval
        output.CaptureIteration(progress, 2.01);
        progress.iteration = 2;
        output.CaptureIteration(progress, 2.02);
        output.FlushIterations();

        C::NonlinearIterationReport stage;
        stage.residual = {1, 1.5};
        stage.initialResidual = {4, 5};
        stage.referenceResidual = {8, 10};
        stage.residualTargetRatio = .25;
        stage.pseudoCFL = 9;
        output.StageEnd({3, 1, 0, 99, 1, .3, .01, 2.1}, stage);

        C::AcceptedStepProgress step;
        step.diagnostics[0] = 11;
        step.diagnostics[4] = .125;
        output.StepEnd({3, 7, 6, 5, 1, .3, .01, 2.2}, step, true, false);
        output.Flush();
        resume = output.HistoryPosition();
    }
    CHECK(resume.block == 1);
    CHECK(resume.line == 1);
    {
        C::ProgressOutputController output(schema, settings, "fluid", bare, true, logger, resume);
        C::AcceptedStepProgress step;
        step.diagnostics[0] = 12;
        output.StepEnd({4, -1, -1, -1, 0, .4, .01, 3}, step, true, false);
        output.Flush();
        CHECK(output.HistoryPosition().block == 2);
        CHECK(output.HistoryPosition().line == 0);
    }

    std::ifstream first(bare + ".csv");
    REQUIRE(first);
    std::string line;
    REQUIRE(std::getline(first, line));
    const auto header = Split(line);
    CHECK(std::find(header.begin(), header.end(), "res0_rho") == header.end());
    CHECK(std::find(header.begin(), header.end(), "resref_rho") == header.end());
    const auto iteration = ReadRow(first, header);
    const auto stage = ReadRow(first, header);
    CHECK(iteration.at("step") == "3");
    CHECK(iteration.at("stage") == "1");
    CHECK(iteration.at("coupled_stage") == "0");
    CHECK(iteration.at("iteration") == "2");
    CHECK(std::stod(iteration.at("wall")) == doctest::Approx(2.02));
    CHECK(iteration.at("attempt") == "1");
    CHECK(iteration.at("status") == "0");
    CHECK(iteration.at("res_rho") == "2.00000000e+00");
    CHECK(stage.at("stage") == "1");
    CHECK(stage.at("iteration") == "-1");
    CHECK(stage.at("status") == "1");
    CHECK_FALSE(std::getline(first, line));

    std::ifstream second(bare + "_1.csv");
    REQUIRE(second);
    REQUIRE(std::getline(second, line));
    CHECK(Split(line) == header);
    const auto accepted = ReadRow(second, header);
    CHECK(accepted.at("stage") == "-1");
    CHECK(accepted.at("coupled_stage") == "-1");
    CHECK(accepted.at("iteration") == "-1");
    CHECK(accepted.at("status") == "2");
    CHECK(accepted.at("mass") == "1.10000000e+01");
    const auto resumed = ReadRow(second, header);
    CHECK(resumed.at("step") == "4");
    CHECK(resumed.at("mass") == "1.20000000e+01");
    CHECK_FALSE(std::getline(second, line));
    CHECK_FALSE(std::filesystem::exists(bare + "_2.csv"));
    std::filesystem::remove_all(directory);
}

TEST_CASE("Resolved output plan centralizes names cadence inheritance and exact endpoint snapping")
{
    C::OutputSettings settings;
    settings.prefix = "/tmp/cauchy-plan/case";
    settings.checkpointInterval = 7;
    settings.visualizationInterval = -1;
    settings.checkpointTimeInterval = .25;
    settings.visualizationTimeInterval = -1;
    settings.endpointSnapRelativeTolerance = 1e-10;
    const C::OutputPlan plan(settings, 2);
    CHECK(plan.VisualizationInterval() == 7);
    CHECK(plan.VisualizationTimeInterval() == .25);
    CHECK(plan.DomainPrefix(1) == "/tmp/cauchy-plan/case-D1");
    CHECK(plan.HistoryBare(0) == "/tmp/cauchy-plan/case-D0-history");
    CHECK(plan.CheckpointPath("-S12") == "/tmp/cauchy-plan/case-S12.h5");
    CHECK(plan.VisualizationPrefix(1, "-final") == "/tmp/cauchy-plan/case-D1-final");
    CHECK(plan.SnapshotManifestPath("-S12") == "/tmp/cauchy-plan/case-S12-manifest.json");
    CHECK(plan.InternalCheckpointPath(1, "-S12_0_3") ==
          "/tmp/cauchy-plan/case-D1-S12_0_3.h5");
    CHECK(C::OutputPlan::NextTimeOrdinal(.5, .25) == 3);
    CHECK(C::OutputPlan::TimeAt(3, .25) == .75);

    auto state = plan.InitialState(.5);
    state.history[0] = {2, 3};
    state.history[1] = {4, 5};
    const auto restored = plan.RestoreState(state.ToJson(), .5);
    CHECK(restored.nextCheckpointTime == 3);
    CHECK(restored.nextVisualizationTime == 3);
    CHECK(restored.history[1].block == 4);
    CHECK(restored.history[1].line == 5);
    auto changed = settings;
    changed.prefix += "-new";
    const C::OutputPlan changedPlan(changed, 2);
    const auto restartedBundle = changedPlan.RestoreState(state.ToJson(), .5);
    CHECK(restartedBundle.prefix == changed.prefix);
    CHECK(restartedBundle.history[0].block == 0);
    CHECK(restartedBundle.history[0].line == 0);

    const auto snapped = plan.SnapStep(.25, .25000000001, 2, 2, 2);
    REQUIRE(snapped.exactTime);
    CHECK(*snapped.exactTime == .5);
    CHECK(snapped.dt == .25);
    const auto ordinary = plan.SnapStep(.5, .1, 2, 3, 3);
    CHECK_FALSE(ordinary.exactTime);
    CHECK(ordinary.dt == .1);
    CHECK_THROWS(plan.SnapStep(1e20, 1e-10, 0, 0, 0));

    const auto token = C::OutputPlan::TimeToken(.1);
    REQUIRE(token.rfind("-T", 0) == 0);
    CHECK(std::stod(token.substr(2)) == .1);
}
