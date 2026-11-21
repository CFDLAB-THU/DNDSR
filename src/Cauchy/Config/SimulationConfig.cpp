#include "Cauchy/Config/SimulationConfig.hpp"

#include "DNDS/Config/JsonSchema.hpp"
#include "DNDS/MPI/Failure.hpp"
#include <set>
#include <numeric>
#include <optional>

namespace DNDS::Cauchy
{
    namespace
    {
        std::string Escape(std::string text)
        {
            std::string result;
            for (char c : text)
                result += c == '~' ? "~0" : c == '/' ? "~1"
                                                     : std::string(1, c);
            return result;
        }
        void Allowed(const Json &value, std::initializer_list<const char *> allowed, const std::string &path)
        {
            DNDS_check_throw_info(value.is_object(), "Cauchy configuration object required at " + path);
            for (const auto &[key, child] : value.items())
                DNDS_check_throw_info(std::find(allowed.begin(), allowed.end(), key) != allowed.end(), "unknown Cauchy key " + path + "/" + key);
        }
        using Origins = std::map<std::string, std::string>;
        void Origin(const Json &value, const std::string &path, const std::string &source, Origins &origins)
        {
            if ((value.is_object() || value.is_array()) && !value.empty())
                for (const auto &[key, child] : value.items())
                    Origin(child, path + "/" + Escape(key), source, origins);
            else
                origins[path] = source;
        }
        void Merge(Json &target, const Json &patch, const std::string &path, const std::string &source, Origins &origins)
        {
            if (patch.is_object() && target.is_object())
            {
                const std::string limitedVRPath = "/numerics/limitedVR";
                if (path.size() >= limitedVRPath.size() && path.compare(path.size() - limitedVRPath.size(), limitedVRPath.size(), limitedVRPath) == 0 && patch.contains("variant"))
                {
                    // Like an input preset, a variant selects explicit defaults
                    // for this layer; fields beside it then override them and a
                    // later layer wins. Save expanded controls and provenance.
                    target = CFV::LimitedVRSettings::ForVariant(patch.at("variant").get<std::string>());
                    Origin(target, path, source + " LVR variant defaults", origins);
                }
                if (!patch.empty())
                    origins.erase(path);
                for (const auto &[key, child] : patch.items())
                {
                    if (!target.contains(key))
                        target[key] = child.is_object() ? Json::object() : Json();
                    Merge(target[key], child, path + "/" + Escape(key), source, origins);
                }
            }
            else
            {
                for (auto it = origins.begin(); it != origins.end();)
                    if (it->first == path || it->first.compare(0, path.size() + 1, path + "/") == 0)
                        it = origins.erase(it);
                    else
                        ++it;
                target = patch;
                Origin(target, path, source, origins);
            }
        }
    }
    SimulationConfig SimulationConfig::Resolve(const Json &input, int size, const std::vector<ConfigOverride> &overrides)
    {
        DNDS_check_throw_info(size > 0, "invalid simulation size");
        Config::ValidatePartialJsonSchema(input, InputSchema());
        DNDS_check_throw_info(input.value("schemaVersion", 1) == 1, "unsupported Cauchy schema version");
        const auto supplied = [&](const char *section, const char *field)
        {
            const std::string pointer = std::string("/") + section + "/" + field;
            return (input.contains(section) && input.at(section).contains(field)) ||
                   std::any_of(overrides.begin(), overrides.end(), [&](const auto &override)
                               { return override.pointer == pointer; });
        };
        const auto activeDeprecatedAlias = [&](const char *section, const char *field)
        {
            const std::string pointer = std::string("/") + section + "/" + field;
            const auto active = [](const Json &value)
            { return value.is_number() && value.get<real>() >= 0; };
            if (input.contains(section) && input.at(section).contains(field) && active(input.at(section).at(field)))
                return true;
            return std::any_of(overrides.begin(), overrides.end(), [&](const auto &override)
                               { return override.pointer == pointer && active(override.value); });
        };
        DNDS_check_throw_info(!(supplied("time", "tEnd") && activeDeprecatedAlias("time", "finalTime")),
                              "specify only time.tEnd or deprecated time.finalTime");
        DNDS_check_throw_info(!(supplied("output", "checkpointInterval") && activeDeprecatedAlias("output", "interval")),
                              "specify only output.checkpointInterval or deprecated output.interval");
        const auto schema = DomainSettings::schema();
        const auto profiles = input.value("profiles", Json::object());
        DNDS_check_throw_info(profiles.is_object(), "Cauchy profiles must be named objects");
        for (const auto &[name, patch] : profiles.items())
            Config::ValidatePartialJsonSchema(patch, schema, "/profiles/" + Escape(name));
        const auto defaults = input.value("defaults", Json::object());
        Config::ValidatePartialJsonSchema(defaults, schema, "/defaults");
        std::vector<int> allRanks(size);
        std::iota(allRanks.begin(), allRanks.end(), 0);
        const auto domains = input.value("domains", Json::array({Json{{"name", "fluid"}, {"ranks", allRanks}}}));
        DNDS_check_throw_info(domains.is_array() && !domains.empty(), "Cauchy needs at least one domain");
        auto data = std::make_shared<Data>();
        data->root = Json{{"schemaVersion", 1}, {"domains", Json::array()}, {"time", TimeSettings{}}, {"output", OutputSettings{}}};
        Origin(data->root, "", "schema default", data->provenance);
        data->provenance.erase("/domains");
        for (index i = 0; i < index(domains.size()); ++i)
        {
            const auto &source = domains[i];
            const auto path = "/domains/" + std::to_string(i);
            Json settings = DomainSettings{};
            Origin(settings, path + "/settings", "schema default", data->provenance);
            const auto selected = source.value("profiles", std::vector<std::string>{});
            std::set<std::string> used;
            for (const auto &name : selected)
            {
                DNDS_check_throw_info(profiles.contains(name) && used.insert(name).second, "unknown/duplicate Cauchy profile " + name);
                Merge(settings, profiles[name], path + "/settings", "profile " + name, data->provenance);
            }
            Merge(settings, defaults, path + "/settings", "solver defaults", data->provenance);
            const auto patch = source.value("overrides", Json::object());
            Config::ValidatePartialJsonSchema(patch, schema, path + "/overrides");
            Merge(settings, patch, path + "/settings", "domain override", data->provenance);
            // Map entries are partial conditions too. Fill defaults without
            // overwriting the per-leaf origin of explicitly supplied values.
            for (auto &[zone, condition] : settings["boundaries"].items())
            {
                const Json original = condition;
                condition = BoundarySettings{};
                for (const auto &[key, value] : condition.items())
                    if (!original.contains(key))
                        Origin(value, path + "/settings/boundaries/" + Escape(zone) + "/" + key, "schema default", data->provenance);
                condition.update(original);
                // A map value is a partial object, including nested profile
                // sections. Resolve their defaults just like root sections.
                if (original.contains("profile"))
                {
                    Json profile = FlowProfileSettings{};
                    for (const auto &[key, value] : profile.items())
                        if (!original["profile"].contains(key))
                            Origin(value, path + "/settings/boundaries/" + Escape(zone) + "/profile/" + key, "schema default", data->provenance);
                    profile.update(original["profile"]);
                    condition["profile"] = std::move(profile);
                }
            }
            Json resolved{{"name", source.at("name")}, {"ranks", source.value("ranks", allRanks)}, {"settings", std::move(settings)}};
            Origin(resolved["name"], path + "/name", "domain descriptor", data->provenance);
            Origin(resolved["ranks"], path + "/ranks", "domain descriptor", data->provenance);
            data->root["domains"].push_back(std::move(resolved));
        }
        for (const auto *section : {"time", "output"})
            if (input.contains(section))
                Merge(data->root[section], input[section], std::string("/") + section, "solver override", data->provenance);
        if (input.contains("interfaces"))
        {
            DNDS_check_throw_info(input["interfaces"].is_array(), "interfaces must be an array of directed connections");
            data->root["interfaces"] = Json::array();
            for (const auto &entry : input["interfaces"])
            {
                Config::ValidatePartialJsonSchema(entry, InterfaceSettings::schema(), "/interfaces");
                Json resolved = InterfaceSettings{};
                resolved.update(entry);
                data->root["interfaces"].push_back(std::move(resolved));
            }
            Origin(data->root["interfaces"], "/interfaces", "solver override", data->provenance);
        }
        for (const auto &override : overrides)
        {
            const Json::json_pointer pointer(override.pointer);
            DNDS_check_throw_info(!override.pointer.empty() && data->root.contains(pointer), "CLI override must target an existing resolved key: " + override.pointer);
            Merge(data->root[pointer], override.value, override.pointer, "CLI " + override.pointer, data->provenance);
        }
        DNDS_check_throw_info(data->root["schemaVersion"] == 1, "unsupported resolved schema version");
        DNDS_check_throw_info(data->root["domains"].is_array() && !data->root["domains"].empty(), "resolved Cauchy configuration has no domains");
        std::map<std::string, index> domainIndices;
        for (index iDomain = 0; iDomain < index(data->root["domains"].size()); ++iDomain)
        {
            auto &resolved = data->root["domains"][iDomain];
            Allowed(resolved, {"name", "ranks", "settings"}, "/domains");
            Config::ValidatePartialJsonSchema(resolved.at("settings"), schema, "/domains/settings");
            // Coarse levels are dynamic map entries, so root defaults cannot
            // supply their omitted count/solver fields. Expand after CLI patches
            // as well; the solver patch itself stays partial for inheritance.
            for (auto &[name, level] : resolved["settings"]["implicit"]["multigrid"]["coarseLevelSettings"].items())
            {
                const Json original = level;
                level = MultigridLevelSettings{};
                for (const auto &[key, value] : level.items())
                    if (!original.contains(key))
                        Origin(value, "/domains/" + std::to_string(iDomain) + "/settings/implicit/multigrid/coarseLevelSettings/" + Escape(name) + "/" + key,
                               "schema default", data->provenance);
                level.update(original);
            }
            const bool acm = resolved["settings"]["equation"]["system"] == "incompressibleACM";
            const auto base = "/domains/" + std::to_string(iDomain) + "/settings/implicit/";
            auto &implicit = resolved["settings"]["implicit"];
            if (implicit["residualComponents"] == Json::array({"auto"}))
            {
                implicit["residualComponents"] = ImplicitSettings{}.ResolveResidualComponents(acm);
                Origin(implicit["residualComponents"], base + "residualComponents", "equation convergence defaults", data->provenance);
            }
            if (acm)
            {
                const auto gradientKey = "/domains/" + std::to_string(iDomain) + "/settings/numerics/viscousGradient";
                if (data->provenance.at(gradientKey) == "schema default")
                {
                    resolved["settings"]["numerics"]["viscousGradient"] = "polynomial";
                    data->provenance[gradientKey] = "ACM mechanical gradients";
                }
                if (data->provenance.at(base + "rejectUnconvergedStages") == "schema default")
                {
                    implicit["rejectUnconvergedStages"] = true;
                    data->provenance[base + "rejectUnconvergedStages"] = "ACM constrained stage";
                }
            }
            DomainConfig domain{{resolved.at("name").get<std::string>(), resolved.at("ranks").get<std::vector<int>>()}, resolved.at("settings").get<DomainSettings>()};
            const bool requestedCoreFirst = domain.settings.numerics.cfv.coreFirst;
            domain.settings = ResolveSweepOrdering(std::move(domain.settings));
            if (requestedCoreFirst != domain.settings.numerics.cfv.coreFirst)
            {
                resolved["settings"]["numerics"]["cfv"]["coreFirst"] = domain.settings.numerics.cfv.coreFirst;
                data->provenance["/domains/" + std::to_string(iDomain) + "/settings/numerics/cfv/coreFirst"] = "Cauchy SOR logical ordering";
            }
            DNDS_check_throw_info(domainIndices.emplace(domain.placement.name, iDomain).second, "duplicate Cauchy domain name");
            ValidateDomain(domain, size);
            ResolvedExecutionPlan::Resolve(domain);
            data->domains.push_back(std::move(domain));
        }
        Config::ValidatePartialJsonSchema(data->root["time"], TimeSettings::schema(), "/time");
        Config::ValidatePartialJsonSchema(data->root["output"], OutputSettings::schema(), "/output");
        data->time = data->root["time"].get<TimeSettings>();
        data->output = data->root["output"].get<OutputSettings>();
        if (data->time.finalTime >= 0)
        {
            DNDS_check_throw_info(data->time.tEnd == 0 || data->time.tEnd == data->time.finalTime,
                                  "time.tEnd and deprecated time.finalTime differ");
            data->time.tEnd = data->time.finalTime;
            data->time.finalTime = -1;
            data->root["time"]["tEnd"] = data->time.tEnd;
            data->root["time"]["finalTime"] = -1;
            data->provenance["/time/tEnd"] = "deprecated /time/finalTime alias";
        }
        if (data->output.interval >= 0)
        {
            DNDS_check_throw_info(data->output.checkpointInterval == OutputSettings{}.checkpointInterval ||
                                      data->output.checkpointInterval == data->output.interval,
                                  "output.checkpointInterval and deprecated output.interval differ");
            data->output.checkpointInterval = data->output.interval;
            data->output.interval = -1;
            data->root["output"]["checkpointInterval"] = data->output.checkpointInterval;
            data->root["output"]["interval"] = -1;
            data->provenance["/output/checkpointInterval"] = "deprecated /output/interval alias";
        }
        std::set<std::pair<std::string, int>> receivers;
        std::optional<std::string> interfaceCadence;
        for (const auto &entry : data->root.value("interfaces", Json::array()))
        {
            Config::ValidatePartialJsonSchema(entry, InterfaceSettings::schema(), "/interfaces");
            auto connection = entry.get<InterfaceSettings>();
            DNDS_check_throw_info(domainIndices.count(connection.donor) && domainIndices.count(connection.receiver) && connection.donor != connection.receiver,
                                  "interface donor/receiver names must identify distinct configured domains");
            DNDS_check_throw_info(connection.routing == "coarseBoxes" || connection.routing == "allGatherBoundary", "unknown interface routing mode");
            DNDS_check_throw_info(connection.refreshCadence == "stageFrozen" ||
                                      connection.refreshCadence == "nonlinearIteration",
                                  "unknown interface refresh cadence");
            if (!interfaceCadence)
                interfaceCadence = connection.refreshCadence;
            else
                DNDS_check_throw_info(
                    *interfaceCadence == connection.refreshCadence,
                    "all interface connections must use one coordinated refresh cadence");
            DNDS_check_throw_info(connection.coarseBoxesPerRank > 0 && std::isfinite(connection.maxGap) && connection.maxGap >= 0, "invalid interface search bounds");
            DNDS_check_throw_info(receivers.emplace(connection.receiver, connection.receiverZone).second, "interface receiver zone has multiple incoming connections");
            const index donor = domainIndices.at(connection.donor);
            const index receiver = domainIndices.at(connection.receiver);
            // Packets carry code-unit state. Check the full resolved material
            // and scaling contract before allocating either domain's mesh.
            const auto &domains = data->root.at("domains");
            const auto packetEquation = [&](index domain)
            {
                auto equation = domains.at(domain).at("settings").at("equation");
                // A pressure gauge controls a nonlinear solve. It does not change
                // the packet's units, fields, or pressure differences.
                equation.erase("acmPressureMean");
                return equation;
            };
            DNDS_check_throw_info(packetEquation(donor) == packetEquation(receiver),
                                  "interface " + connection.donor + " -> " + connection.receiver +
                                      " requires identical equation, material and scaling settings");
            const auto &boundaries = data->domains[receiver].settings.boundaries;
            const auto condition = boundaries.find(std::to_string(connection.receiverZone));
            DNDS_check_throw_info(condition != boundaries.end() && condition->second.kind == "interface", "receiver zone must have an interface boundary condition");
            data->interfaces.push_back(std::move(connection));
        }
        for (const auto &domain : data->domains)
            for (const auto &[zone, condition] : domain.settings.boundaries)
                DNDS_check_throw_info(condition.kind != "interface" || receivers.count({domain.placement.name, std::stoi(zone)}), "interface boundary has no incoming connection");
        const std::vector<std::string> timeMethods{"ssprk3", "bdf1", "bdf2", "trapz", "esdirk2", "esdirk3", "esdirk4", "ditr"};
        DNDS_check_throw_info(std::find(timeMethods.begin(), timeMethods.end(), data->time.method) != timeMethods.end() && std::isfinite(data->time.dt) && data->time.dt > 0 && data->time.steps >= 0 &&
                                  std::isfinite(data->time.tEnd) && data->time.tEnd >= 0 && data->time.finalTime == -1 && std::isfinite(data->time.cfl) && data->time.cfl >= 0 &&
                                  std::isfinite(data->time.wallSeconds) && data->time.wallSeconds >= 0 && data->time.maxRetries >= 0,
                              "Cauchy requires an available time method, positive dt and nonnegative steps");
        DNDS_check_throw_info(std::isfinite(data->time.steadyTolerance) && data->time.steadyTolerance >= 0 && data->time.steadyMinSteps >= 0, "invalid steady stopping controls");
        data->time.ditr.Validate();
        data->time.positivity.Validate();
        data->time.ValidateSourceIntegration();
        if (data->time.sourceSplitting == "strang")
            DNDS_check_throw_info(std::any_of(data->domains.begin(), data->domains.end(), [](const auto &domain)
                                              { return domain.settings.equation.source == "chemical"; }),
                                  "Strang source splitting requires a configured chemical source");
        if (data->time.steady)
        {
            if (data->time.positivity.enabled)
            {
                data->time.positivity.enabled = false;
                data->root["time"]["positivity"]["enabled"] = false;
                data->provenance["/time/positivity/enabled"] = "BDF1 steady limit";
            }
            data->time.method = "bdf1";
            data->time.steps = std::min(index(1), data->time.steps);
            data->time.tEnd = data->time.cfl = 0;
            data->time.maxRetries = 0;
            data->time.steadyMinSteps = std::min(index(1), data->time.steadyMinSteps);
            const Json effective = data->time;
            for (const auto *name : {"method", "steps", "tEnd", "cfl", "maxRetries", "steadyMinSteps"})
                if (data->root["time"][name] != effective[name])
                {
                    data->root["time"][name] = effective[name];
                    data->provenance[std::string("/time/") + name] = "BDF1 steady limit";
                }
        }
        for (const auto &domain : data->domains)
        {
            DNDS_check_throw_info(data->time.method != "ssprk3" || !domain.settings.implicit.multigrid.levels, "polynomial FAS requires an implicit time method");
            DNDS_check_throw_info(domain.settings.equation.system != "incompressibleACM" || data->time.method != "ssprk3",
                                  "SSPRK3 cannot advance the algebraic ACM continuity equation");
            DNDS_check_throw_info(domain.settings.equation.system != "incompressibleACM" || !data->time.positivity.enabled,
                                  "ACM has no thermodynamic physical timestep restriction");
            if (data->time.method == "ditr")
                data->time.ditr.ResolveControls(domain.settings.implicit);
        }
        const std::vector<std::string> diagnosticLevels{"trace", "debug", "info", "warn", "error", "critical", "off"};
        DNDS_check_throw_info(data->output.interval == -1 && data->output.checkpointInterval >= 0 && data->output.visualizationInterval >= -1 &&
                                  std::isfinite(data->output.checkpointTimeInterval) && data->output.checkpointTimeInterval >= 0 &&
                                  std::isfinite(data->output.visualizationTimeInterval) && data->output.visualizationTimeInterval >= -1 &&
                                  std::isfinite(data->output.endpointSnapRelativeTolerance) && data->output.endpointSnapRelativeTolerance >= 0 &&
                                  data->output.consoleInterval >= 0 && data->output.historyInterval > 0 && data->output.historySegmentRows > 0 &&
                                  data->output.historyPrecision >= 1 && data->output.historyPrecision <= std::numeric_limits<real>::max_digits10 &&
                                  data->output.implicit.consoleInterval >= 0 && data->output.implicit.historyInterval > 0 &&
                                  data->output.implicit.checkpointInterval >= 0 && data->output.implicit.visualizationInterval >= -1 &&
                                  (data->output.consoleColor == "auto" || data->output.consoleColor == "always" || data->output.consoleColor == "never") &&
                                  (data->output.consoleBold == "auto" || data->output.consoleBold == "always" || data->output.consoleBold == "never") &&
                                  std::find(diagnosticLevels.begin(), diagnosticLevels.end(), data->output.diagnostics.level) != diagnosticLevels.end() &&
                                  data->output.diagnostics.maxBytes > 0 && data->output.diagnostics.rotatedFiles > 0 &&
                                  data->output.diagnostics.allocationWarmupSteps >= 0 && !data->output.prefix.empty(),
                              "invalid Cauchy output controls");
        DNDS_check_throw_info(data->output.hdfDeflateLevel >= 0 && data->output.hdfDeflateLevel <= 9, "Cauchy output hdfDeflateLevel must be in [0,9]");
        const auto internalVisualization = data->output.implicit.visualizationInterval < 0
                                               ? data->output.implicit.checkpointInterval
                                               : data->output.implicit.visualizationInterval;
        const auto hasInternalSnapshots = [](const auto &locations, const char *field)
        {
            bool internal = false;
            for (const auto &location : locations)
            {
                const bool acceptedStep = location.stage == -1 && location.iteration == -1;
                const bool nonlinearUpdate = location.stage >= 0 && location.iteration > 0;
                DNDS_check_throw_info(location.step > 0 && (acceptedStep || nonlinearUpdate),
                                      std::string(field) + " requires a positive step and either stage=iteration=-1 or stage>=0 and iteration>0");
                internal = internal || nonlinearUpdate;
            }
            return internal;
        };
        const bool selectedInternalCheckpoint = hasInternalSnapshots(data->output.checkpointSnapshots, "output.checkpointSnapshots");
        const bool selectedInternalVisualization = hasInternalSnapshots(data->output.visualizationSnapshots, "output.visualizationSnapshots");
        const bool internalCheckpointEnabled = data->output.implicit.checkpointInterval > 0 || selectedInternalCheckpoint;
        const bool internalVisualizationEnabled = internalVisualization > 0 || selectedInternalVisualization;
        DNDS_check_throw_info(!internalCheckpointEnabled || data->domains.size() == 1,
                              "exact internal checkpoints currently require one domain");
        DNDS_check_throw_info((!internalCheckpointEnabled && !internalVisualizationEnabled) ||
                                  data->time.steady || data->time.method != "ssprk3",
                              "internal-iteration output requires an implicit method");
        return SimulationConfig(std::move(data));
    }
    std::vector<DomainRankGroup> SimulationConfig::RankGroups() const
    {
        std::vector<DomainRankGroup> result;
        for (const auto &domain : _data->domains)
            result.push_back(domain.placement);
        return result;
    }
    SimulationConfig SimulationConfig::ResolveCollective(const Json &input, MPI_Comm comm, const std::vector<ConfigOverride> &overrides)
    {
        DNDS_check_throw_info(comm != MPI_COMM_NULL, "Cauchy config needs a simulation communicator");
        int size = 0, rank = 0, valid = 1;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        std::optional<SimulationConfig> result;
        std::string error, serialized;
        try
        {
            result = Resolve(input, size, overrides);
            serialized = Json{{"root", result->Root()}, {"provenance", result->Provenance()}}.dump();
            DNDS_check_throw_info(serialized.size() <= 16 * 1024 * 1024, "Cauchy resolved configuration exceeds 16 MiB");
        }
        catch (const std::exception &exception)
        {
            MPI::AbortIfSolverFailure("Cauchy configuration rejected");
            error = exception.what();
            valid = 0;
        }
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, comm);
        DNDS_check_throw_info(valid, "Cauchy configuration rejected on a rank: " + error);
        int length = rank == 0 ? int(serialized.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, comm);
        std::string reference = rank == 0 ? serialized : std::string(size_t(length), '\0');
        MPI_Bcast(reference.data(), length, MPI_CHAR, 0, comm);
        valid = reference == serialized;
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, comm);
        MPI::CheckPreflight(valid, "Cauchy resolved configuration differs between simulation ranks");
        return std::move(*result);
    }
}
