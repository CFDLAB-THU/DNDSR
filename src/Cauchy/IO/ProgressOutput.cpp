#include "Cauchy/IO/ProgressOutput.hpp"

#include "DNDS/IO/OutputDir.hpp"
#include "DNDS/Errors.hpp"
#include <fmt/format.h>

#include <spdlog/logger.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/details/os.h>

#include <algorithm>
#include <iomanip>
#include <limits>
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

        std::string FormatResiduals(const EquationSchema &schema, const std::vector<real> &residual)
        {
            if (residual.empty())
                return "[]";
            std::string result;
            index offset = 0;
            for (const auto &block : schema.blocks)
            {
                const index width = index(block.components.size());
                DNDS_check_throw_info(offset + width <= index(residual.size()),
                                      "Cauchy progress residual shape differs from equation schema");
                if (!result.empty())
                    result += ' ';
                std::string label = block.name;
                if (block.name == "flow")
                    label = fmt::format("flow{}", width);
                else if (block.name == "rans")
                {
                    if (width == 1)
                        label = "SA";
                    else if (width == 2 && block.components.back().name == "rhoOmega")
                        label = "kOmega";
                    else if (width == 2 && block.components.back().name == "rhoEpsilon")
                        label = "kEpsilon";
                }
                result += label + " [";
                for (index j = 0; j < width; ++j)
                    result += fmt::format("{}{:9.3e}", j == 0 ? "" : " ", residual[size_t(offset + j)]);
                result += ']';
                offset += width;
            }
            DNDS_check_throw_info(offset == index(residual.size()),
                                  "Cauchy progress residual shape differs from equation schema");
            return result;
        }

        std::string FormatTimeProgress(const ProgressLocation &location, real pseudoCFL)
        {
            // A steady solve has no physical-time term. Display its infinite
            // physical step while retaining the configured value in CSV.
            const real displayDt = location.steady ? std::numeric_limits<real>::infinity() : location.dt;
            return fmt::format("t,dt,CFL [{:.3e},{:.3e},{:.3e}]", location.time, displayDt, pseudoCFL);
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
            const bool terminal = spdlog::details::os::in_terminal(stdout) &&
                                  spdlog::details::os::is_color_terminal();
            const auto enabled = [terminal](const std::string &policy)
            { return policy == "always" || (policy == "auto" && terminal); };
            const bool color = enabled(settings.consoleColor), bold = enabled(settings.consoleBold);
#ifndef _WIN32
            _style.bold = bold ? "\033[1m" : "";
            _style.red = color ? "\033[31m" : "";
            _style.green = color ? "\033[32m" : "";
            _style.yellow = color ? "\033[33m" : "";
            _style.orange = color ? "\033[38;5;208m" : "";
            _style.cyan = color ? "\033[36m" : "";
            _style.reset = color || bold ? "\033[0m" : "";
#endif
            auto console = std::make_shared<spdlog::sinks::stdout_sink_mt>();
            console->set_pattern("%v");
            _console = std::make_shared<spdlog::logger>("cauchy-console-rank-" + std::to_string(globalRank), console);
            _console->set_level(LogLevel(settings.diagnostics.level));
            _console->flush_on(spdlog::level::info);
        }
        if (settings.eventLog && (globalRank == 0 || domainProgressWriter))
        {
            const auto path = settings.prefix + "-events-rank" + std::to_string(globalRank) + ".log";
            createOutputDir(path);
            // Append across restarts. Rotating files receive the same unstyled
            // messages; ANSI codes are generated only by the console sink.
            auto events = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                path, size_t(settings.diagnostics.maxBytes), size_t(settings.diagnostics.rotatedFiles));
            events->set_pattern("%Y-%m-%dT%H:%M:%S.%e [%l] %v");
            sinks.push_back(std::move(events));
        }
        if (settings.diagnostics.perRank)
        {
            const auto path = settings.prefix + "-diagnostics-rank" + std::to_string(globalRank) + ".log";
            createOutputDir(path);
            auto diagnostics = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                path, size_t(settings.diagnostics.maxBytes), size_t(settings.diagnostics.rotatedFiles));
            diagnostics->set_pattern("%Y-%m-%dT%H:%M:%S.%e [%l] %v");
            sinks.push_back(std::move(diagnostics));
        }
        if (!sinks.empty())
        {
            _logger = std::make_shared<spdlog::logger>("cauchy-rank-" + std::to_string(globalRank), sinks.begin(), sinks.end());
            _logger->set_level(LogLevel(settings.diagnostics.level));
            _logger->flush_on(spdlog::level::info);
        }
    }
    OperationalLogger::~OperationalLogger()
    {
        if (_console)
            _console->flush();
        if (_logger)
            _logger->flush();
    }
    void OperationalLogger::Info(const std::string &message)
    {
        if (_console)
            _console->info("[info] [rank {}] {}", _globalRank, message);
        if (_logger)
            _logger->info("[rank {}] {}", _globalRank, message);
    }
    void OperationalLogger::Warn(const std::string &message)
    {
        if (_console)
            _console->warn("[warn] [rank {}] {}", _globalRank, message);
        if (_logger)
            _logger->warn("[rank {}] {}", _globalRank, message);
    }
    void OperationalLogger::Error(const std::string &message)
    {
        if (_console)
            _console->error("[error] [rank {}] {}", _globalRank, message);
        if (_logger)
            _logger->error("[rank {}] {}", _globalRank, message);
    }
    void OperationalLogger::Progress(const std::string &styled, const std::string &plain)
    {
        if (_console)
            _console->info(styled);
        if (_logger)
            _logger->info(plain);
    }
    void OperationalLogger::Flush()
    {
        if (_console)
            _console->flush();
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
    CsvSchema DetailedHistorySchema(const EquationSchema &schema, bool stateBounds)
    {
        using T = CsvColumnType;
        std::vector<CsvColumn> result{{"step", T::Integer}, {"stage", T::Integer}, {"coupled_stage", T::Integer}, {"iteration", T::Integer}, {"attempt", T::Integer}, {"time", T::Real}, {"dt", T::Real}, {"wall", T::Real}, {"iteration_wall", T::Real}, {"stage_iteration_wall_mean", T::Real}, {"step_wall", T::Real}, {"status", T::Integer}, {"pseudo_cfl", T::Real}, {"residual_target_ratio", T::Real}, {"linear_iterations", T::Integer}, {"vr_sweeps", T::Integer}, {"vr_relative_update", T::Real}, {"rhs_limited", T::Integer}, {"rhs_alpha_min", T::Real}, {"increment_limited", T::Integer}, {"increment_alpha_min", T::Real}, {"mass", T::Real}, {"energy", T::Real}, {"rho_min", T::Real}, {"p_min", T::Real}, {"residual_l2", T::Real}, {"force_x", T::Real}, {"force_y", T::Real}, {"force_z", T::Real}, {"rho_error_l1", T::Real}, {"rho_error_l2", T::Real}, {"rho_error_linf", T::Real}, {"temperature_min", T::Real}, {"face_fallbacks", T::Integer}, {"retries", T::Integer}, {"pp_rec_cells", T::Integer}, {"pp_beta_min", T::Real}, {"pp_stage_cells", T::Integer}, {"pp_alpha_min", T::Real}, {"characteristic_fallbacks", T::Integer}, {"implicit_stages", T::Integer}, {"nonlinear_iterations", T::Integer}, {"coupled_iterations", T::Integer}, {"coupled_residual_target_ratio", T::Real}, {"time_pp_cells", T::Integer}, {"time_pp_alpha", T::Real}, {"time_growth_limited", T::Integer}};
        if (schema.family == "incompressibleACM")
        {
            result.erase(std::remove_if(result.begin(), result.end(), [](const CsvColumn &column)
                                        { return column.name == "temperature_min" || column.name.compare(0, 10, "rho_error_") == 0; }),
                         result.end());
            for (auto &column : result)
                if (column.name == "energy")
                    column.name = "kinetic_energy";
            result.push_back({"continuity_rms", T::Real});
            result.push_back({"mean_pressure", T::Real});
        }
        for (const auto &name : EquationComponentNames(schema))
            result.push_back({"res_" + name, T::Real});
        if (stateBounds)
            for (const auto &name : EquationComponentNames(schema))
            {
                result.push_back({"min_" + name, T::Real});
                result.push_back({"max_" + name, T::Real});
            }
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
        // An empty display domain is the canonical single-domain spelling.
        // Multi-domain coordinators pass the configured domain name.
        DNDS_check_throw_info(_logger, "invalid Cauchy progress-output controller");
        if (_writer)
            _history = std::make_unique<ProgressHistory>(DetailedHistorySchema(_schema, _settings.historyStateBounds), bareHistory,
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
    void ProgressOutputController::FillStateBounds(ProgressRow &row, const std::vector<real> &bounds) const
    {
        if (!_settings.historyStateBounds || bounds.empty())
            return;
        const auto names = EquationComponentNames(_schema);
        DNDS_check_throw_info(bounds.size() == 2 * names.size(),
                              "Cauchy history state bounds shape differs from equation schema");
        for (size_t j = 0; j < names.size(); ++j)
        {
            row.Set("min_" + names[j], bounds[2 * j]);
            row.Set("max_" + names[j], bounds[2 * j + 1]);
        }
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
        row.Set("iteration_wall", progress.iterationWallSeconds);
        FillResiduals(row, progress.residual);
        FillStateBounds(row, progress.stateBounds);
    }
    void ProgressOutputController::BeginAttempt(ProgressLocation location)
    {
        _attempt = location;
        _stageTiming.clear();
    }
    void ProgressOutputController::ConsoleNonlinear(const ProgressLocation &location,
                                                    const NonlinearProgress &progress, real stageAverage) const
    {
        const auto residual = FormatResiduals(_schema, progress.residual);
        const auto identity = fmt::format("    {}Internal === Step [{:7},{:3},{:5}]",
                                          _domain.empty() ? "" : "D[" + _domain + "] ",
                                          location.step, location.stage, location.iteration);
        const auto physical = FormatTimeProgress(location, progress.pseudoCFL);
        const auto positivity = fmt::format("PP [rhs({},{:.3f}),inc({},{:.3f})]",
                                            progress.rhsLimited, progress.minimumRHSAlpha,
                                            progress.incrementLimited, progress.minimumIncrementAlpha);
        std::string vr = fmt::format("VR [{},", progress.reconstructionIterations);
        if (progress.reconstructionUpdateMeasured)
            vr += fmt::format("{:.3e}]", progress.reconstructionRelativeUpdate);
        else
            vr += "unmeasured]";
        std::string status;
        if (progress.terminal)
            status = progress.converged ? "converged" : "capped";
        const auto convergence = fmt::format("conv [{:.3e}, {:.3e}]",
                                             progress.residualTargetRatio, progress.relativeConverged);
        const auto plain = fmt::format("{}   res {}   {}   {}   {}   {}   linear [{}]   Time [{:.3f} s/iter avg]{}{}",
                                       identity, residual, convergence, physical, positivity, vr, progress.linearIterations,
                                       stageAverage, status.empty() ? "" : "   ", status);
        const auto &style = _logger->Style();
        const auto styled = fmt::format("{}{}{}   res {}{}{}   {}{}{}   {}{}{}   {}{}{}   {}   linear [{}]   {}Time [{:.3f} s/iter avg]{}{}{}{}",
                                        style.bold, identity, style.reset,
                                        style.red, residual, style.reset,
                                        style.orange, convergence, style.reset,
                                        style.green, physical, style.reset,
                                        (progress.rhsLimited || progress.incrementLimited) ? style.yellow : "", positivity,
                                        (progress.rhsLimited || progress.incrementLimited) ? style.reset : "",
                                        vr, progress.linearIterations,
                                        style.cyan, stageAverage, style.reset,
                                        status.empty() ? "" : "   ",
                                        status.empty() ? "" : (progress.converged ? style.green : style.yellow),
                                        status.empty() ? "" : status + style.reset);
        _logger->Progress(styled, plain);
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
    void ProgressOutputController::FlushIterations(bool snapshotDue)
    {
        if (!_writer)
        {
            _captured.clear();
            return;
        }
        for (const auto &entry : _captured)
        {
            auto &timing = _stageTiming[{entry.location.step, entry.location.stage, entry.location.coupledStage}];
            timing.first += entry.progress.iterationWallSeconds;
            ++timing.second;
            const real stageAverage = timing.first / real(timing.second);
            auto observation = entry.location;
            if (entry.progress.observedIteration >= 0)
                observation.iteration = entry.progress.observedIteration;
            const bool history = _settings.implicit.historyInterval > 0 && observation.iteration > 0 &&
                                 (observation.iteration % _settings.implicit.historyInterval == 0 || entry.progress.terminal);
            if (history)
            {
                auto row = Base(observation, ProgressRecordStatus::NonlinearIteration);
                FillNonlinear(row, entry.progress);
                row.Set("stage_iteration_wall_mean", stageAverage);
                _history->Write(row);
            }
            if (_settings.implicit.consoleInterval > 0 &&
                (entry.location.iteration % _settings.implicit.consoleInterval == 0 || entry.progress.terminal ||
                 (snapshotDue && &entry == &_captured.back())))
                ConsoleNonlinear(entry.location, entry.progress, stageAverage);
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
        progress.stateBounds = report.stateBounds;
        FillNonlinear(row, progress);
        _history->Write(row);
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
            row.Set(_schema.family == "incompressibleACM" ? "kinetic_energy" : "energy", v[1]);
            row.Set("rho_min", v[2]);
            row.Set("p_min", v[3]);
            row.Set("residual_l2", v[4]);
            // TODO(boundary_integrations): retain this aggregate force in the
            // combined canonical history. Add separately configured per-zone
            // segmented integration streams (force/moment/mass/energy/species
            // flux) using the same frozen RHS quadrature outputs and cursors.
            // Do not duplicate BC/EOS calculations in output writers.
            row.Set("force_x", v[5]);
            row.Set("force_y", v[6]);
            row.Set("force_z", v[7]);
            row.Set("vr_sweeps", index(v[8]));
            row.Set("vr_relative_update", v[9]);
            if (_schema.family == "incompressibleACM")
            {
                row.Set("continuity_rms", v[10]);
                row.Set("mean_pressure", v[11]);
            }
            else
            {
                row.Set("rho_error_l1", v[10]);
                row.Set("rho_error_l2", v[11]);
                row.Set("rho_error_linf", v[12]);
                row.Set("temperature_min", v[13]);
            }
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
            row.Set("step_wall", progress.stepWallSeconds);
            if (!progress.residual.empty())
                FillResiduals(row, progress.residual);
            FillStateBounds(row, progress.stateBounds);
            _history->Write(row);
        }
        if (printConsole)
        {
            const auto &v = progress.diagnostics;
            const auto residual = FormatResiduals(_schema, progress.residual);
            const auto identity = fmt::format("=== {}Step [{:7}]", _domain.empty() ? "" : "D[" + _domain + "] ", location.step);
            const auto physical = FormatTimeProgress(location, progress.pseudoCFL);
            const auto state = _schema.family == "incompressibleACM"
                                   ? fmt::format("state [pMin {:.3e},pMean {:.3e},continuityRMS {:.3e}]", v[3], v[11], v[10])
                                   : fmt::format("state [rhoMin {:.3e},pMin {:.3e},Tmin {:.3e}]", v[2], v[3], v[13]);
            const auto pp = fmt::format("PP [rec {},stage {},time {}]", index(progress.positivity[0]), index(progress.positivity[2]), progress.timePPLimitedCells);
            const auto work = fmt::format("work [stages {},nonlinear {},linear {}]", progress.implicitStages, progress.nonlinearIterations, progress.linearIterations);
            const auto plain = fmt::format("{}   res {}   {}   {}   {}   {}   Time [{:.3f} s]", identity, residual, physical, state, pp, work, progress.stepWallSeconds);
            const auto &style = _logger->Style();
            const auto styled = fmt::format("{}{}{}   res {}{}{}   {}{}{}   {}{}{}   {}{}{}   {}   {}Time [{:.3f} s]{}",
                                            style.bold, identity, style.reset, style.red, residual, style.reset,
                                            style.green, physical, style.reset, style.green, state, style.reset,
                                            (progress.positivity[0] || progress.positivity[2] || progress.timePPLimitedCells) ? style.yellow : "", pp,
                                            (progress.positivity[0] || progress.positivity[2] || progress.timePPLimitedCells) ? style.reset : "",
                                            work, style.cyan, progress.stepWallSeconds, style.reset);
            _logger->Progress(styled, plain);
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
