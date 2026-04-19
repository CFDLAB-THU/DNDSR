/** @file SingleBlockApp.hpp
 *  @brief Main entry-point template for single-block Euler/Navier-Stokes solver executables.
 *
 *  Provides getSingleBlockAppName() for mapping EulerModel enum values to
 *  executable names, and the RunSingleBlockConsoleApp() template that wires up
 *  CLI argument parsing (argparse), JSON configuration loading with optional
 *  key/value overrides, JSON Schema emission, and the solve pipeline
 *  (mesh read → initialization → implicit time integration).
 */
#pragma once

#include <argparse.hpp>

#include "EulerSolver.hpp"

namespace DNDS::Euler
{
    /**
     * @brief Map an EulerModel enum value to its corresponding executable name string.
     *
     * Used to derive default configuration file paths and to set the program name
     * shown in --help output.
     *
     * @param model  Compile-time or runtime EulerModel variant.
     * @return Null-terminated C string with the executable name (e.g. "euler",
     *         "eulerSA", "euler3D"). Returns "_error_app_name_" for unrecognized models.
     */
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

    /**
     * @brief Main entry point for single-block solver console applications.
     *
     * This function template is instantiated for each EulerModel variant and
     * serves as the complete `main()` implementation for that solver executable.
     *
     * **Execution flow:**
     * 1. Initialize MPI and build default/user configuration file paths.
     * 2. Parse command-line arguments via argparse:
     *    - Positional `config` — path to the user JSON configuration file.
     *    - Positional `field_n_variables` (extended models with DynamicSize only).
     *    - `-k`/`--overwrite_key` and `-v`/`--overwrite_value` — repeated pairs
     *      for ad-hoc JSON config overrides.
     *    - `--debug` — attach-debugger mode (MPI hold).
     *    - `--emit-schema` — print the full JSON Schema for this solver's
     *      configuration and exit.
     * 3. In `--emit-schema` mode: instantiate a default EulerSolver, emit the
     *    schema produced by DNDS_DECLARE_CONFIG, and return.
     * 4. In normal mode: construct an EulerSolver, load the default config then
     *    the user config (with key/value overrides), call
     *    ReadMeshAndInitialize(), and RunImplicitEuler().
     *
     * @tparam model  Compile-time EulerModel selecting the equation set / dimension.
     * @param argc    Argument count forwarded from main().
     * @param argv    Argument vector forwarded from main().
     * @return 0 on success (abnormal paths call std::abort()).
     */
    template <EulerModel model>
    int RunSingleBlockConsoleApp(int argc, char *argv[])
    {
        using namespace std::literals;
        MPIInfo mpi;
        mpi.setWorld();

        std::string defaultConfJson = "../cases/"s + getSingleBlockAppName(model) + "_default_config.json"s;
        std::string confJson = "../cases/"s + getSingleBlockAppName(model) + "_config.json";
        std::vector<std::string> overwriteKeys, overwriteValues;

        argparse::ArgumentParser mainParser(getSingleBlockAppName(model), DNDS_VERSION_STRING);
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
        GetSetVersionName(DNDS_VERSION_STRING);

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

            // Build schema: Configuration is fully registered via DNDS_DECLARE_CONFIG,
            // so emitSchema() produces the complete recursive schema.
            if (mpi.rank == 0)
            {
                using TConfig = typename Euler::EulerSolver<model>::Configuration;
                nlohmann::ordered_json schema = TConfig::schema(
                    std::string("DNDSR ") + getSingleBlockAppName(model) + " configuration");
                schema["$schema"] = "http://json-schema.org/draft-07/schema#";
                schema["title"] = schema["description"];
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