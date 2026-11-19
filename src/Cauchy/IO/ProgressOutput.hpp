#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "DNDS/IO/CsvLog.hpp"

#include <map>
#include <optional>
#include <tuple>

namespace spdlog
{
    class logger;
}

namespace DNDS::Cauchy
{
    /// Compact numeric CSV row. Column meaning lives in the file header and
    /// run manifest; absent values serialize as an empty cell, never false zero.
    class ProgressRow
    {
        friend class ProgressHistory;
        CsvRow _row;
        explicit ProgressRow(CsvRow row) : _row(std::move(row)) {}

    public:
        void Set(const std::string &name, const std::string &value);
        void Set(const std::string &name, real value);
        void Set(const std::string &name, index value);
        void Set(const std::string &name, uint64_t value);
    };

    /// One per-domain segmented detailed history, written only by that domain's
    /// root. The stable header is repeated once per segment for concatenation.
    class ProgressHistory
    {
        std::vector<std::string> _columns;
        CsvLog _log;
        int _precision = 10;

    public:
        ProgressHistory(CsvSchema schema, std::string bare, index segmentRows, int precision,
                        std::optional<CsvLogPosition> resume = std::nullopt);
        ProgressRow NewRow() const { return ProgressRow(_log.MakeRow()); }
        void Write(const ProgressRow &row);
        void Flush();
        CsvLogPosition Position() const { return _log.Position(); }
        const std::vector<std::string> &Columns() const { return _columns; }
    };

    /// Local operational logger. Scientific history/checkpoints never pass
    /// through spdlog. The default implementation is synchronous so teardown
    /// has no hidden worker lifetime. CLI and plain-text event files share
    /// messages; only the terminal sink adds color/emphasis.
    class OperationalLogger
    {
        std::shared_ptr<spdlog::logger> _logger, _console;
        int _globalRank = 0;

    public:
        struct TerminalStyle
        {
            std::string bold, red, green, yellow, orange, cyan, reset;
        };

    private:
        TerminalStyle _style;

    public:
        /// Domain progress writers may be nonzero world ranks. Grant their
        /// console sink explicitly; this does not change per-rank file naming
        /// or perform a collective. Progress cadence remains controller-owned.
        OperationalLogger(const OutputSettings &settings, int globalRank,
                          bool domainProgressWriter = false);
        ~OperationalLogger();
        void Info(const std::string &message);
        void Warn(const std::string &message);
        void Error(const std::string &message);
        /// Styled terminal row plus byte-clean event/diagnostic row.
        void Progress(const std::string &styled, const std::string &plain);
        const TerminalStyle &Style() const { return _style; }
        void Flush();
    };

    std::vector<std::string> EquationComponentNames(const EquationSchema &schema);
    CsvSchema DetailedHistorySchema(const EquationSchema &schema);
    std::vector<std::string> DetailedHistoryColumns(const EquationSchema &schema);

    enum class ProgressRecordStatus : index
    {
        NonlinearIteration = 0,
        ImplicitStageEnd = 1,
        AcceptedStepEnd = 2
    };
    struct ProgressLocation
    {
        uint64_t step = 0;
        index stage = -1, coupledStage = -1, iteration = -1, attempt = 0;
        real time = 0, dt = 0, wall = 0;
    };
    struct AcceptedStepProgress
    {
        std::array<real, 14> diagnostics{};
        std::vector<real> residual;
        real pseudoCFL = 0;
        index faceFallbacks = 0, retries = 0, implicitStages = 0, nonlinearIterations = 0, linearIterations = 0;
        index cappedStages = 0;
        index coupledIterations = 0, timePPLimitedCells = 0;
        std::array<real, 5> positivity{};
        real coupledResidualTargetRatio = 0, timePPMinimumAlpha = 1;
        /// Domain-root elapsed time from the first attempt through acceptance,
        /// including retries/iteration output, excluding accepted-step I/O and
        /// diagnostics. This is not a collective maximum or a device timer.
        real stepWallSeconds = 0;
        bool timeGrowthLimited = false;
    };

    /// The one construction point for detailed CSV and ordinary CLI progress.
    /// Numerical callbacks only CaptureIteration into memory. FlushIterations,
    /// StageEnd and StepEnd run at a safe frontend boundary after collective
    /// operators return, so sink failure cannot strand peers inside a solve.
    class ProgressOutputController
    {
        struct Captured
        {
            ProgressLocation location;
            NonlinearProgress progress;
        };
        EquationSchema _schema;
        OutputSettings _settings;
        std::string _domain;
        bool _writer = false;
        std::shared_ptr<OperationalLogger> _logger;
        std::unique_ptr<ProgressHistory> _history;
        ProgressLocation _attempt;
        std::vector<Captured> _captured;
        std::map<std::tuple<uint64_t, index, index>, std::pair<real, index>> _stageTiming;
        ProgressRow Base(const ProgressLocation &location, ProgressRecordStatus status) const;
        void FillResiduals(ProgressRow &row, const std::vector<real> &residual) const;
        void FillNonlinear(ProgressRow &row, const NonlinearProgress &progress) const;
        void ConsoleNonlinear(const ProgressLocation &location, const NonlinearProgress &progress,
                              real stageAverage) const;

    public:
        ProgressOutputController(EquationSchema schema, OutputSettings settings, std::string domain,
                                 std::string bareHistory, bool writer,
                                 std::shared_ptr<OperationalLogger> logger,
                                 std::optional<CsvLogPosition> resume = std::nullopt);
        void BeginAttempt(ProgressLocation location);
        /// Capture the frontend's elapsed wall time at observation, not the
        /// attempt-start time or the later sink-flush time.
        void CaptureIteration(const NonlinearProgress &progress, real wall);
        void FlushIterations();
        void StageEnd(ProgressLocation location, const NonlinearIterationReport &report);
        void StepEnd(ProgressLocation location, const AcceptedStepProgress &progress, bool forceHistory = false,
                     bool printConsole = true);
        void Flush();
        /// Valid only on the designated history writer after Flush.
        CsvLogPosition HistoryPosition() const;
    };
}
