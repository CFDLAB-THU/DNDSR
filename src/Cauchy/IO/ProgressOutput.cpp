#include "Cauchy/IO/ProgressOutput.hpp"

#include "DNDS/IO/OutputDir.hpp"
#include "DNDS/Errors.hpp"
#include <fmt/format.h>

#include <spdlog/logger.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace DNDS::Cauchy
{
    namespace
    {
        spdlog::level::level_enum LogLevel(const std::string &name)
        {
            if (name == "trace")
                return spdlog::level::trace;
            if (name == "debug")
                return spdlog::level::debug;
            if (name == "info")
                return spdlog::level::info;
            if (name == "warn")
                return spdlog::level::warn;
            if (name == "error")
                return spdlog::level::err;
            if (name == "critical")
                return spdlog::level::critical;
            if (name == "off")
                return spdlog::level::off;
            DNDS_check_throw_info(false, "unknown Cauchy diagnostic level");
            return spdlog::level::off;
        }
    }
    void ProgressRow::Set(const std::string &name, const std::string &value) { _row.Set(name, value); }
    void ProgressRow::Set(const std::string &name, real value) { _row.Set(name, value); }
    void ProgressRow::Set(const std::string &name, index value) { _row.Set(name, value); }
    void ProgressRow::Set(const std::string &name, uint64_t value) { _row.Set(name, value); }
    namespace
    {
        std::vector<std::string> ColumnNames(const CsvSchema &schema)
        {
            std::vector<std::string> result;
            result.reserve(schema.Columns().size());
            for (const auto &column : schema.Columns())
                result.push_back(column.name);
            return result;
        }
    }

    ProgressHistory::ProgressHistory(CsvSchema schema, std::string bare, index segmentRows, int precision,
                                     std::optional<CsvLogPosition> resume)
        : _columns(ColumnNames(schema)),
          _log(resume ? CsvLog(std::move(schema), bare, ".csv", segmentRows, *resume)
                      : CsvLog(std::move(schema), bare, ".csv", segmentRows)),
          _precision(precision)
    {
        DNDS_check_throw_info(!_columns.empty() && !bare.empty() && segmentRows > 0,
                              "invalid Cauchy detailed-history construction");
        auto sorted = _columns;
        std::sort(sorted.begin(), sorted.end());
        DNDS_check_throw_info(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
                              "duplicate Cauchy detailed-history column");
    }
    void ProgressHistory::Write(const ProgressRow &row)
    {
        _log.WriteLine(row._row, _precision);
    }
    void ProgressHistory::Flush() { _log.Flush(); }

    OperationalLogger::OperationalLogger(const OutputSettings &settings, int globalRank,
                                         bool domainProgressWriter) : _globalRank(globalRank)
    {
        std::vector<spdlog::sink_ptr> sinks;
        if ((globalRank == 0 || domainProgressWriter) &&
            (settings.consoleInterval > 0 || settings.implicit.consoleInterval > 0))
        {
            if (settings.consoleColor == "never")
                sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_mt>());
            else
            {
                auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
                console->set_color_mode(settings.consoleColor == "always" ? spdlog::color_mode::always
                                                                          : spdlog::color_mode::automatic);
                sinks.push_back(std::move(console));
            }
        }
        if (settings.diagnostics.perRank)
        {
            const auto path = settings.prefix + "-diagnostics-rank" + std::to_string(globalRank) + ".log";
            createOutputDir(path);
            sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                path, size_t(settings.diagnostics.maxBytes), size_t(settings.diagnostics.rotatedFiles)));
        }
        if (!sinks.empty())
        {
            _logger = std::make_shared<spdlog::logger>("cauchy-rank-" + std::to_string(globalRank), sinks.begin(), sinks.end());
            _logger->set_level(LogLevel(settings.diagnostics.level));
            _logger->set_pattern("%^[%l]%$ %v");
        }
    }
    OperationalLogger::~OperationalLogger()
    {
        if (_logger)
            _logger->flush();
    }
    void OperationalLogger::Info(const std::string &message)
    {
        if (_logger)
            _logger->info("[rank {}] {}", _globalRank, message);
    }
    void OperationalLogger::Warn(const std::string &message)
    {
        if (_logger)
            _logger->warn("[rank {}] {}", _globalRank, message);
    }
    void OperationalLogger::Error(const std::string &message)
    {
        if (_logger)
            _logger->error("[rank {}] {}", _globalRank, message);
    }
    void OperationalLogger::Flush()
    {
        if (_logger)
            _logger->flush();
    }
    std::vector<std::string> EquationComponentNames(const EquationSchema &schema)
    {
        std::vector<std::string> result;
        for (const auto &block : schema.blocks)
            for (const auto &component : block.components)
                result.push_back(component.name);
        return result;
    }
    CsvSchema DetailedHistorySchema(const EquationSchema &schema)
    {
        using T = CsvColumnType;
        std::vector<CsvColumn> result{{"step", T::Integer}, {"stage", T::Integer}, {"coupled_stage", T::Integer}, {"iteration", T::Integer}, {"attempt", T::Integer}, {"time", T::Real}, {"dt", T::Real}, {"wall", T::Real}, {"status", T::Integer}, {"pseudo_cfl", T::Real}, {"residual_target_ratio", T::Real}, {"linear_iterations", T::Integer}, {"vr_sweeps", T::Integer}, {"vr_relative_update", T::Real}, {"rhs_limited", T::Integer}, {"rhs_alpha_min", T::Real}, {"increment_limited", T::Integer}, {"increment_alpha_min", T::Real}, {"mass", T::Real}, {"energy", T::Real}, {"rho_min", T::Real}, {"p_min", T::Real}, {"residual_l2", T::Real}, {"force_x", T::Real}, {"force_y", T::Real}, {"force_z", T::Real}, {"rho_error_l1", T::Real}, {"rho_error_l2", T::Real}, {"rho_error_linf", T::Real}, {"temperature_min", T::Real}, {"face_fallbacks", T::Integer}, {"retries", T::Integer}, {"pp_rec_cells", T::Integer}, {"pp_beta_min", T::Real}, {"pp_stage_cells", T::Integer}, {"pp_alpha_min", T::Real}, {"characteristic_fallbacks", T::Integer}, {"implicit_stages", T::Integer}, {"nonlinear_iterations", T::Integer}, {"coupled_iterations", T::Integer}, {"coupled_residual_target_ratio", T::Real}, {"time_pp_cells", T::Integer}, {"time_pp_alpha", T::Real}, {"time_growth_limited", T::Integer}};
        for (const auto &name : EquationComponentNames(schema))
            result.push_back({"res_" + name, T::Real});
        return CsvSchema(std::move(result));
    }
    std::vector<std::string> DetailedHistoryColumns(const EquationSchema &schema)
    {
        return ColumnNames(DetailedHistorySchema(schema));
    }

    ProgressOutputController::ProgressOutputController(EquationSchema schema, OutputSettings settings, std::string domain,
                                                       std::string bareHistory, bool writer,
                                                       std::shared_ptr<OperationalLogger> logger,
                                                       std::optional<CsvLogPosition> resume)
        : _schema(std::move(schema)), _settings(std::move(settings)), _domain(std::move(domain)), _writer(writer),
          _logger(std::move(logger))
    {
        DNDS_check_throw_info(_logger && !_domain.empty(), "invalid Cauchy progress-output controller");
        if (_writer)
            _history = std::make_unique<ProgressHistory>(DetailedHistorySchema(_schema), std::move(bareHistory),
                                                         _settings.historySegmentRows, _settings.historyPrecision, resume);
    }
    ProgressRow ProgressOutputController::Base(const ProgressLocation &location, ProgressRecordStatus status) const
    {
        auto row = _history->NewRow();
        row.Set("step", location.step);
        row.Set("stage", location.stage);
        row.Set("coupled_stage", location.coupledStage);
        row.Set("iteration", location.iteration);
        row.Set("attempt", location.attempt);
        row.Set("time", location.time);
        row.Set("dt", location.dt);
        row.Set("wall", location.wall);
        row.Set("status", index(status));
        return row;
    }
    void ProgressOutputController::FillResiduals(ProgressRow &row, const std::vector<real> &residual) const
    {
        const auto names = EquationComponentNames(_schema);
        DNDS_check_throw_info(residual.size() == names.size(),
                              "Cauchy progress residual shape differs from equation schema");
        for (size_t i = 0; i < names.size(); ++i)
            row.Set("res_" + names[i], residual[i]);
    }
    void ProgressOutputController::FillNonlinear(ProgressRow &row, const NonlinearProgress &progress) const
    {
        row.Set("pseudo_cfl", progress.pseudoCFL);
        row.Set("residual_target_ratio", progress.residualTargetRatio);
        row.Set("linear_iterations", progress.linearIterations);
        row.Set("vr_sweeps", index(progress.reconstructionIterations));
        if (progress.reconstructionUpdateMeasured)
            row.Set("vr_relative_update", progress.reconstructionRelativeUpdate);
        row.Set("rhs_limited", progress.rhsLimited);
        row.Set("rhs_alpha_min", progress.minimumRHSAlpha);
        row.Set("increment_limited", progress.incrementLimited);
        row.Set("increment_alpha_min", progress.minimumIncrementAlpha);
        FillResiduals(row, progress.residual);
    }
    void ProgressOutputController::ConsoleNonlinear(const ProgressLocation &location,
                                                    const NonlinearProgress &progress, bool stageEnd) const
    {
        std::string residual;
        for (const real value : progress.residual)
            residual += fmt::format(" {:10.3e}", value);
        std::string message = fmt::format(
            "Cauchy {:>10} step {:7} stage {:2} coupled {:2} {} residual [{} ] target {:9.2e} CFL {:9.2e} wall {:10.3f}s VR {:3}:",
            _domain, location.step, location.stage, location.coupledStage,
            stageEnd ? std::string("end          ") : fmt::format("iteration {:4}", location.iteration),
            residual, progress.residualTargetRatio, progress.pseudoCFL, location.wall, progress.reconstructionIterations);
        if (progress.reconstructionUpdateMeasured)
            message += fmt::format("{:9.2e}", progress.reconstructionRelativeUpdate);
        else
            message += "unmeasured";
        message += fmt::format(" linear {:4}", progress.linearIterations);
        if (progress.terminal)
            message += progress.converged ? " converged" : " capped";
        _logger->Info(message);
    }
    void ProgressOutputController::CaptureIteration(const NonlinearProgress &progress, real wall)
    {
        if (!_writer)
            return;
        auto location = _attempt;
        location.wall = wall;
        location.stage = progress.stage;
        location.coupledStage = progress.coupledStage;
        location.iteration = progress.iteration;
        _captured.push_back({location, progress});
    }
    void ProgressOutputController::FlushIterations()
    {
        if (!_writer)
        {
            _captured.clear();
            return;
        }
        for (const auto &entry : _captured)
        {
            const bool history = _settings.implicit.historyInterval > 0 &&
                                 (entry.location.iteration % _settings.implicit.historyInterval == 0 || entry.progress.terminal);
            if (history)
            {
                auto row = Base(entry.location, ProgressRecordStatus::NonlinearIteration);
                FillNonlinear(row, entry.progress);
                _history->Write(row);
            }
            if (_settings.implicit.consoleInterval > 0 &&
                (entry.location.iteration % _settings.implicit.consoleInterval == 0 || entry.progress.terminal))
                ConsoleNonlinear(entry.location, entry.progress, false);
        }
        _captured.clear();
    }
    void ProgressOutputController::StageEnd(ProgressLocation location, const NonlinearIterationReport &report)
    {
        if (!_writer)
            return;
        location.iteration = -1;
        auto row = Base(location, ProgressRecordStatus::ImplicitStageEnd);
        NonlinearProgress progress;
        progress.stage = int(location.stage);
        progress.coupledStage = int(location.coupledStage);
        progress.residual = report.residual;
        progress.initialResidual = report.initialResidual;
        progress.referenceResidual = report.referenceResidual;
        progress.residualTargetRatio = report.residualTargetRatio;
        progress.pseudoCFL = report.pseudoCFL;
        progress.linearIterations = report.linearIterations;
        progress.rhsLimited = report.rhsLimited;
        progress.incrementLimited = report.incrementLimited;
        progress.minimumRHSAlpha = report.minimumRHSAlpha;
        progress.minimumIncrementAlpha = report.minimumIncrementAlpha;
        progress.reconstructionIterations = report.reconstructionIterations;
        progress.reconstructionRelativeUpdate = report.reconstructionRelativeUpdate;
        progress.reconstructionUpdateMeasured = report.reconstructionUpdateMeasured;
        FillNonlinear(row, progress);
        _history->Write(row);
        if (_settings.implicit.consoleInterval > 0)
            ConsoleNonlinear(location, progress, true);
    }
    void ProgressOutputController::StepEnd(ProgressLocation location, const AcceptedStepProgress &progress,
                                           bool forceHistory, bool printConsole)
    {
        if (!_writer)
            return;
        location.stage = location.coupledStage = location.iteration = -1;
        if (forceHistory || (_settings.historyInterval > 0 && location.step % uint64_t(_settings.historyInterval) == 0))
        {
            auto row = Base(location, ProgressRecordStatus::AcceptedStepEnd);
            const auto &v = progress.diagnostics;
            row.Set("mass", v[0]);
            row.Set("energy", v[1]);
            row.Set("rho_min", v[2]);
            row.Set("p_min", v[3]);
            row.Set("residual_l2", v[4]);
            row.Set("force_x", v[5]);
            row.Set("force_y", v[6]);
            row.Set("force_z", v[7]);
            row.Set("vr_sweeps", index(v[8]));
            row.Set("vr_relative_update", v[9]);
            row.Set("rho_error_l1", v[10]);
            row.Set("rho_error_l2", v[11]);
            row.Set("rho_error_linf", v[12]);
            row.Set("temperature_min", v[13]);
            row.Set("face_fallbacks", progress.faceFallbacks);
            row.Set("retries", progress.retries);
            row.Set("pp_rec_cells", index(progress.positivity[0]));
            row.Set("pp_beta_min", progress.positivity[1]);
            row.Set("pp_stage_cells", index(progress.positivity[2]));
            row.Set("pp_alpha_min", progress.positivity[3]);
            row.Set("characteristic_fallbacks", index(progress.positivity[4]));
            row.Set("implicit_stages", progress.implicitStages);
            row.Set("nonlinear_iterations", progress.nonlinearIterations);
            row.Set("linear_iterations", progress.linearIterations);
            row.Set("coupled_iterations", progress.coupledIterations);
            row.Set("coupled_residual_target_ratio", progress.coupledResidualTargetRatio);
            row.Set("time_pp_cells", progress.timePPLimitedCells);
            row.Set("time_pp_alpha", progress.timePPMinimumAlpha);
            row.Set("time_growth_limited", index(progress.timeGrowthLimited));
            _history->Write(row);
        }
        if (printConsole)
        {
            const auto &v = progress.diagnostics;
            _logger->Info(fmt::format(
                "Cauchy {:>10} step {:7} time {:11.4e} dt {:9.2e} residual {:10.3e} rho_min {:9.2e} p_min {:9.2e} VR {:3.0f}:{:9.2e} rho L1 {:9.2e} face fallbacks {:5} implicit {:5}/{:5} wall {:10.3f}s",
                _domain, location.step, location.time, location.dt, v[4], v[2], v[3], v[8], v[9], v[10],
                progress.faceFallbacks, progress.nonlinearIterations, progress.linearIterations, location.wall));
        }
    }
    void ProgressOutputController::Flush()
    {
        FlushIterations();
        if (_history)
            _history->Flush();
        _logger->Flush();
    }
    CsvLogPosition ProgressOutputController::HistoryPosition() const
    {
        DNDS_check_throw_info(_writer && _history, "history position is available only on its writer");
        return _history->Position();
    }
}
