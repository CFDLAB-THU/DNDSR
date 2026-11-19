#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/IO/CsvLog.hpp"

#include <optional>

namespace DNDS::Cauchy
{
    /// Versioned frontend state stored with a checkpoint. It contains only
    /// output scheduling and append cursors; numerical payloads remain in the
    /// checkpoint's ordinary HDF arrays.
    struct OutputControllerState
    {
        int schemaVersion = 1;
        std::string prefix;
        index checkpointInterval = 0, visualizationInterval = 0;
        real checkpointTimeInterval = 0, visualizationTimeInterval = 0;
        uint64_t nextCheckpointTime = 0, nextVisualizationTime = 0;
        std::vector<CsvLogPosition> history;

        Json ToJson() const;
        static OutputControllerState FromJson(const Json &value);
    };

    /// Immutable frontend output policy. Numerical operators publish typed
    /// reports; this plan alone resolves cadence inheritance, names and exact
    /// physical-time endpoints.
    class OutputPlan
    {
        OutputSettings _settings;
        index _domains = 0;
        index _visualizationInterval = 0;
        real _visualizationTimeInterval = 0;

    public:
        struct StepTarget
        {
            real dt = 0;
            std::optional<real> exactTime;
        };
        struct SnapshotSelection
        {
            bool checkpoint = false, visualization = false;
        };

        OutputPlan(OutputSettings settings, index domains);
        const OutputSettings &Settings() const { return _settings; }
        index DomainNumber() const { return _domains; }
        index VisualizationInterval() const { return _visualizationInterval; }
        real VisualizationTimeInterval() const { return _visualizationTimeInterval; }
        /// Test the exact requested location. Omit stage and iteration to
        /// match the accepted step.
        SnapshotSelection SelectedSnapshots(uint64_t step, index stage = -1, index iteration = -1) const;
        OutputControllerState InitialState(real time) const;
        /// Restore scheduling/history continuation when the output prefix and
        /// resolved cadence are unchanged. A deliberately new prefix starts a
        /// new bundle at the same accepted time without borrowing old CSVs.
        OutputControllerState RestoreState(const Json &value, real time) const;
        std::string DomainPrefix(index domain) const;
        std::string HistoryBare(index domain) const;
        std::string ResolvedPath() const;
        std::string SummaryPath() const;
        std::string RunManifestPath() const;
        std::string CheckpointPath(const std::string &token) const;
        std::string InternalCheckpointPath(index domain,
                                           const std::string &token) const;
        std::string VisualizationPrefix(index domain, const std::string &token) const;
        std::string SnapshotManifestPath(const std::string &token) const;
        std::string InternalSnapshotManifestPath(index domain, const std::string &token) const;
        static std::string StepToken(uint64_t step);
        static std::string InternalToken(uint64_t step, index stage, index iteration);
        static std::string TimeToken(real time);
        static uint64_t NextTimeOrdinal(real time, real interval);
        static real TimeAt(uint64_t ordinal, real interval);
        StepTarget SnapStep(real time, real proposedDt, real tEnd,
                            uint64_t checkpointOrdinal, uint64_t visualizationOrdinal) const;
    };
}
