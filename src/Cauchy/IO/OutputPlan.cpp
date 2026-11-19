#include "Cauchy/IO/OutputPlan.hpp"

#include "DNDS/Core/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

namespace DNDS::Cauchy
{
    Json OutputControllerState::ToJson() const
    {
        Json positions = Json::array();
        for (const auto cursor : history)
            positions.push_back({{"block", cursor.block}, {"rows", cursor.line}});
        return {{"schema_version", schemaVersion},
                {"prefix", prefix},
                {"checkpoint_interval", checkpointInterval},
                {"visualization_interval", visualizationInterval},
                {"checkpoint_time_interval", checkpointTimeInterval},
                {"visualization_time_interval", visualizationTimeInterval},
                {"next_checkpoint_time", nextCheckpointTime},
                {"next_visualization_time", nextVisualizationTime},
                {"history", std::move(positions)}};
    }
    OutputControllerState OutputControllerState::FromJson(const Json &value)
    {
        OutputControllerState state;
        state.schemaVersion = value.at("schema_version").get<int>();
        state.prefix = value.at("prefix").get<std::string>();
        state.checkpointInterval = value.at("checkpoint_interval").get<index>();
        state.visualizationInterval = value.at("visualization_interval").get<index>();
        state.checkpointTimeInterval = value.at("checkpoint_time_interval").get<real>();
        state.visualizationTimeInterval = value.at("visualization_time_interval").get<real>();
        state.nextCheckpointTime = value.at("next_checkpoint_time").get<uint64_t>();
        state.nextVisualizationTime = value.at("next_visualization_time").get<uint64_t>();
        for (const auto &cursor : value.at("history"))
            state.history.push_back({cursor.at("block").get<int>(), cursor.at("rows").get<int64_t>()});
        DNDS_check_throw_info(state.schemaVersion == 1 && !state.prefix.empty() &&
                                  state.checkpointInterval >= 0 && state.visualizationInterval >= 0 &&
                                  std::isfinite(state.checkpointTimeInterval) && state.checkpointTimeInterval >= 0 &&
                                  std::isfinite(state.visualizationTimeInterval) && state.visualizationTimeInterval >= 0,
                              "invalid Cauchy output-controller state");
        return state;
    }

    OutputPlan::OutputPlan(OutputSettings settings, index domains)
        : _settings(std::move(settings)), _domains(domains),
          _visualizationInterval(_settings.visualizationInterval < 0
                                     ? _settings.checkpointInterval
                                     : _settings.visualizationInterval),
          _visualizationTimeInterval(_settings.visualizationTimeInterval < 0
                                         ? _settings.checkpointTimeInterval
                                         : _settings.visualizationTimeInterval)
    {
        DNDS_check_throw_info(_domains > 0 && !_settings.prefix.empty() &&
                                  _settings.checkpointInterval >= 0 && _visualizationInterval >= 0 &&
                                  std::isfinite(_settings.checkpointTimeInterval) && _settings.checkpointTimeInterval >= 0 &&
                                  std::isfinite(_visualizationTimeInterval) && _visualizationTimeInterval >= 0 &&
                                  std::isfinite(_settings.endpointSnapRelativeTolerance) &&
                                  _settings.endpointSnapRelativeTolerance >= 0,
                              "invalid resolved Cauchy output plan");
    }
    OutputPlan::SnapshotSelection OutputPlan::SelectedSnapshots(uint64_t step, index stage, index iteration) const
    {
        const auto matches = [step, stage, iteration](const OutputSnapshotLocation &location)
        {
            return uint64_t(location.step) == step && location.stage == stage && location.iteration == iteration;
        };
        return {std::any_of(_settings.checkpointSnapshots.begin(), _settings.checkpointSnapshots.end(), matches),
                std::any_of(_settings.visualizationSnapshots.begin(), _settings.visualizationSnapshots.end(), matches)};
    }
    std::string OutputPlan::DomainPrefix(index domain) const
    {
        DNDS_check_throw_info(domain >= 0 && domain < _domains, "output domain index is out of range");
        return _settings.prefix + "-D" + std::to_string(domain);
    }
    std::string OutputPlan::HistoryBare(index domain) const { return DomainPrefix(domain) + "-history"; }
    OutputControllerState OutputPlan::InitialState(real time) const
    {
        DNDS_check_throw_info(std::isfinite(time) && time >= 0, "invalid initial output-controller time");
        OutputControllerState state;
        state.prefix = _settings.prefix;
        state.checkpointInterval = _settings.checkpointInterval;
        state.visualizationInterval = _visualizationInterval;
        state.checkpointTimeInterval = _settings.checkpointTimeInterval;
        state.visualizationTimeInterval = _visualizationTimeInterval;
        state.nextCheckpointTime = NextTimeOrdinal(time, state.checkpointTimeInterval);
        state.nextVisualizationTime = NextTimeOrdinal(time, state.visualizationTimeInterval);
        state.history.resize(size_t(_domains));
        return state;
    }
    OutputControllerState OutputPlan::RestoreState(const Json &value, real time) const
    {
        auto state = OutputControllerState::FromJson(value);
        if (state.prefix != _settings.prefix)
            return InitialState(time);
        DNDS_check_throw_info(state.checkpointInterval == _settings.checkpointInterval &&
                                  state.visualizationInterval == _visualizationInterval &&
                                  state.checkpointTimeInterval == _settings.checkpointTimeInterval &&
                                  state.visualizationTimeInterval == _visualizationTimeInterval &&
                                  state.history.size() == size_t(_domains),
                              "checkpoint output cadence differs while continuing the same output prefix");
        DNDS_check_throw_info(state.nextCheckpointTime == NextTimeOrdinal(time, state.checkpointTimeInterval) &&
                                  state.nextVisualizationTime == NextTimeOrdinal(time, state.visualizationTimeInterval),
                              "checkpoint output-time cursor differs from its accepted clock");
        for (const auto cursor : state.history)
            DNDS_check_throw_info(cursor.block >= 0 && cursor.line >= 0 && cursor.line < _settings.historySegmentRows,
                                  "checkpoint CSV continuation cursor is invalid");
        return state;
    }
    std::string OutputPlan::ResolvedPath() const { return _settings.prefix + "-resolved.json"; }
    std::string OutputPlan::SummaryPath() const { return _settings.prefix + "-summary.json"; }
    std::string OutputPlan::RunManifestPath() const { return _settings.prefix + "-run-manifest.json"; }
    std::string OutputPlan::CheckpointPath(const std::string &token) const
    {
        DNDS_check_throw_info(!token.empty(), "checkpoint token is empty");
        return _settings.prefix + (_domains == 1 ? "-D0" : "") + token + ".h5";
    }
    std::string OutputPlan::InternalCheckpointPath(
        index domain, const std::string &token) const
    {
        DNDS_check_throw_info(!token.empty(),
                              "internal checkpoint token is empty");
        return DomainPrefix(domain) + token + ".h5";
    }
    std::string OutputPlan::VisualizationPrefix(index domain, const std::string &token) const
    {
        DNDS_check_throw_info(!token.empty(), "visualization token is empty");
        return DomainPrefix(domain) + token;
    }
    std::string OutputPlan::SnapshotManifestPath(const std::string &token) const
    {
        DNDS_check_throw_info(!token.empty(), "snapshot token is empty");
        return _settings.prefix + token + "-manifest.json";
    }
    std::string OutputPlan::InternalSnapshotManifestPath(index domain, const std::string &token) const
    {
        DNDS_check_throw_info(!token.empty(), "internal snapshot token is empty");
        return DomainPrefix(domain) + token + "-manifest.json";
    }
    std::string OutputPlan::StepToken(uint64_t step) { return "-S" + std::to_string(step); }
    std::string OutputPlan::InternalToken(uint64_t step, index stage, index iteration)
    {
        DNDS_check_throw_info(stage >= 0 && iteration >= 0, "internal output stage/iteration is invalid");
        return StepToken(step) + "_" + std::to_string(stage) + "_" + std::to_string(iteration);
    }
    std::string OutputPlan::TimeToken(real time)
    {
        DNDS_check_throw_info(std::isfinite(time), "output time is nonfinite");
        std::ostringstream text;
        text.imbue(std::locale::classic());
        text << "-T" << std::setprecision(std::numeric_limits<real>::max_digits10) << time;
        return text.str();
    }
    uint64_t OutputPlan::NextTimeOrdinal(real time, real interval)
    {
        if (!(interval > 0))
            return 0;
        DNDS_check_throw_info(std::isfinite(time) && time >= 0 && std::isfinite(interval),
                              "invalid output-time cursor");
        const auto completed = std::floor(time / interval);
        DNDS_check_throw_info(completed >= 0 && completed < real(std::numeric_limits<uint64_t>::max() - 1),
                              "Cauchy output-time ordinal overflow");
        return uint64_t(completed) + 1;
    }
    real OutputPlan::TimeAt(uint64_t ordinal, real interval)
    {
        DNDS_check_throw_info(std::isfinite(interval) && interval > 0, "invalid output-time interval");
        const real result = real(ordinal) * interval;
        DNDS_check_throw_info(std::isfinite(result), "Cauchy output time overflow");
        return result;
    }
    OutputPlan::StepTarget OutputPlan::SnapStep(real time, real proposedDt, real tEnd,
                                                uint64_t checkpointOrdinal, uint64_t visualizationOrdinal) const
    {
        DNDS_check_throw_info(std::isfinite(time) && std::isfinite(proposedDt) && proposedDt > 0 &&
                                  std::isfinite(tEnd) && tEnd >= 0,
                              "invalid physical time or proposed timestep");
        real target = std::numeric_limits<real>::infinity();
        if (tEnd > 0)
            target = tEnd;
        if (_settings.checkpointTimeInterval > 0)
            target = std::min(target, TimeAt(checkpointOrdinal, _settings.checkpointTimeInterval));
        if (_visualizationTimeInterval > 0)
            target = std::min(target, TimeAt(visualizationOrdinal, _visualizationTimeInterval));
        const real remaining = target - time;
        const real tolerance = _settings.endpointSnapRelativeTolerance * proposedDt;
        if (std::isfinite(target) && remaining > 0 && remaining <= proposedDt + tolerance)
            return {remaining, target};
        DNDS_check_throw_info(time + proposedDt > time,
                              "Cauchy physical clock cannot represent the proposed forward step");
        return {proposedDt, std::nullopt};
    }
}
