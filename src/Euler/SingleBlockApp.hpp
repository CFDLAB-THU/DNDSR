#pragma once

#include <argparse.hpp>

#include "EulerSolver.hpp"

namespace DNDS::Euler
{
    constexpr static inline const char *getSingleBlockAppName(const EulerModel model)
    {
        if (model == NS)
            return "euler";
        else if (model == NS_SA)
            return "eulerSA";
        else if (model == NS_2D)
            return "euler2D";
        else if (model == NS_3D)
            return "euler3D";
        else if (model == NS_SA_3D)
            return "eulerSA3D";
        else if (model == NS_2EQ)
            return "euler2EQ";
        else if (model == NS_2EQ_3D)
            return "euler2EQ3D";
        else if (model == NS_EX)
            return "eulerEX";
        else if (model == NS_EX_3D)
            return "eulerEX3D";
        return "_error_app_name_";
    }

    template <EulerModel model>
    int RunSingleBlockConsoleApp(int argc, char *argv[])
    {
        using namespace std::literals;
        MPIInfo mpi;
        mpi.setWorld();

        std::string defaultConfJson = "../cases/"s + getSingleBlockAppName(model) + "_default_config.json"s;
        std::string confJson = "../cases/"s + getSingleBlockAppName(model) + "_config.json";
        std::vector<std::string> overwriteKeys, overwriteValues;

        argparse::ArgumentParser mainParser(getSingleBlockAppName(model), "version"s + " commit "s + DNDS_MACRO_TO_STRING(DNDS_CURRENT_COMMIT_HASH));
        std::string read_configPath;
        if (getnVarsFixed(model) == DynamicSize)
            mainParser.add_argument("field_n_variables").default_value<int>(5).scan<'i', int>();
        mainParser.add_argument("config").default_value("");
        mainParser.add_argument("-k", "--overwrite_key")
            .help("keys to the json entries to overwrite")
            .append()
            .default_value<std::vector<std::string>>({});
        mainParser.add_argument("-v", "--overwrite_value")
            .help("values to the json entries to overwrite")
            .append()
            .default_value<std::vector<std::string>>({});
        mainParser.add_argument("--debug").flag().default_value(false);
        mainParser.add_argument("--emit-schema")
            .help("Print JSON Schema for this solver's configuration and exit")
            .flag()
            .default_value(false);

        RegisterSignalHandler();
        GetSetVersionName(DNDS_MACRO_TO_STRING(DNDS_CURRENT_COMMIT_HASH));

        try
        {
            mainParser.parse_args(argc, argv);
            if (mainParser.get<bool>("--debug"))
            {
                Debug::isDebugging = true;
                Debug::MPIDebugHold(mpi);
            }
            read_configPath = mainParser.get("config");
            if (!read_configPath.empty())
            {
                confJson = read_configPath;
                std::filesystem::path p(read_configPath);
                if (p.is_relative())
                    p = "." / p; // so that not using a wrong path if using a file in cwd
                defaultConfJson = getStringForcePath(p.parent_path()) + "/"s + getSingleBlockAppName(model) + "_default_config.json"s;
            }
            overwriteKeys = mainParser.get<std::vector<std::string>>("--overwrite_key");
            overwriteValues = mainParser.get<std::vector<std::string>>("--overwrite_value");
            if (overwriteKeys.size() != overwriteValues.size())
                throw std::runtime_error("overwrite keys and values not matching");
        }
        catch (const std::exception &err)
        {
            std::cerr << err.what() << std::endl;
            std::cerr << mainParser;
            std::abort();
        }

        // ---- --emit-schema: print JSON Schema and exit (no MPI, no mesh) ----
        if (mainParser.get<bool>("--emit-schema"))
        {
            int nVars = getnVarsFixed(model);
            if (nVars == DynamicSize)
                nVars = mainParser.get<int>("field_n_variables");

            // Build a default-initialized solver to get the full default config JSON.
            // Then emit a basic JSON Schema inferred from the defaults, enriched with
            // metadata from any sections that have been migrated to DNDS_PARAM.
            Euler::EulerSolver<model> solver(mpi, nVars);

            // Write defaults to generate the full default config
            std::string schemaDefaultFile = "/tmp/dnds_schema_defaults_" + std::to_string(mpi.rank) + ".json";
            solver.ConfigureFromJson(schemaDefaultFile, false);

            // Build schema: for migrated sections, use ConfigRegistry; for the rest,
            // infer type from the default JSON values.
            if (mpi.rank == 0)
            {
                nlohmann::ordered_json schema;
                schema["$schema"] = "http://json-schema.org/draft-07/schema#";
                schema["title"] = std::string("DNDSR ") + getSingleBlockAppName(model) + " configuration";
                schema["type"] = "object";
                auto &props = schema["properties"] = nlohmann::ordered_json::object();

                // Migrated sections emit rich schema from their registries.
                using TConfig = typename Euler::EulerSolver<model>::Configuration;
                props["timeMarchControl"] = TConfig::TimeMarchControl::schema("Time marching settings");
                props["implicitReconstructionControl"] = TConfig::ImplicitReconstructionControl::schema("Implicit reconstruction settings");
                props["outputControl"] = TConfig::OutputControl::schema("Output and logging settings");
                props["implicitCFLControl"] = TConfig::ImplicitCFLControl::schema("Implicit CFL settings");
                props["convergenceControl"] = TConfig::ConvergenceControl::schema("Convergence criteria");
                props["dataIOControl"] = TConfig::DataIOControl::schema("Data I/O and mesh settings");
                props["limiterControl"] = TConfig::LimiterControl::schema("Slope limiter settings");
                props["linearSolverControl"] = TConfig::LinearSolverControl::schema("Linear solver settings");
                props["restartState"] = TConfig::RestartState::schema("Restart state");
                props["timeAverageControl"] = TConfig::TimeAverageControl::schema("Time-averaging settings");
                props["others"] = TConfig::Others::schema("Miscellaneous settings");
                props["eulerSettings"] = Euler::EulerEvaluatorSettings<model>::schema("Euler evaluator settings");
                props["vfvSettings"] = CFV::VRSettings::schema("Variational reconstruction settings");

                // For all other sections, read from the default JSON and infer types.
                auto fIn = std::ifstream(schemaDefaultFile);
                if (fIn)
                {
                    auto defaults = nlohmann::ordered_json::parse(fIn);
                    for (auto &[key, val] : defaults.items())
                    {
                        if (props.contains(key))
                            continue; // already handled by migrated section
                        nlohmann::ordered_json fieldSchema;
                        if (val.is_object())
                            fieldSchema["type"] = "object";
                        else if (val.is_array())
                            fieldSchema["type"] = "array";
                        else if (val.is_boolean())
                            fieldSchema["type"] = "boolean";
                        else if (val.is_number_integer())
                            fieldSchema["type"] = "integer";
                        else if (val.is_number_float())
                            fieldSchema["type"] = "number";
                        else if (val.is_string())
                            fieldSchema["type"] = "string";
                        fieldSchema["default"] = val;
                        props[key] = fieldSchema;
                    }
                }

                std::cout << schema.dump(4) << std::endl;
                std::filesystem::remove(schemaDefaultFile);
            }
            MPI::Barrier(mpi.comm);
            if (mpi.rank != 0)
                std::filesystem::remove(schemaDefaultFile);
            return 0;
        }

        try
        {
            int nVars = getnVarsFixed(model);
            if (nVars == DynamicSize)
                nVars = mainParser.get<int>("field_n_variables");
            if (mpi.rank == 0)
                log() << "Current MPI thread level: " << MPI::GetMPIThreadLevel() << std::endl;
            auto strategy = MPI::CommStrategy::Instance().GetArrayStrategy();
            Euler::EulerSolver<model> solver(mpi, nVars);
            if (mpi.rank == 0)
            {
                log() << "Reading configuration from " << confJson << std::endl;
                log() << "Using default configuration from " << defaultConfJson << std::endl;
            }
            solver.ConfigureFromJson(defaultConfJson, false);
            solver.ConfigureFromJson(defaultConfJson, true, confJson, overwriteKeys, overwriteValues);
            solver.ReadMeshAndInitialize();
            solver.RunImplicitEuler();
        }
        catch (const std::exception &e)
        {
            std::cerr << "DNDS top-level exception: " << e.what() << std::endl;
            std::abort();
        }
        catch (...)
        {
            std::cerr << "Unknown exception" << std::endl;
            std::abort();
        }
        return 0;
    }

}