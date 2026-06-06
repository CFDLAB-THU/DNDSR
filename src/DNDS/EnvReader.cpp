/// @file EnvReader.cpp
/// @brief Implementation of the centralized environment-variable reader with
/// typed access and a file-scope static cache.

#include "EnvReader.hpp"
#include "Defines.hpp" // for log()

#include <cctype>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

// MPI rank-0 guard for log output — keep MPI dependency self-contained.
// MPI is always available in DNDSR; the guards mirror MPI.hpp conventions.
#ifndef MPICH_SKIP_MPICXX
#    define MPICH_SKIP_MPICXX 1
#endif
#ifndef OMPI_SKIP_MPICXX
#    define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>

namespace DNDS
{

    namespace
    {
        /// Global cache: key = env-var name, value = raw string from getenv (or default).
        std::map<std::string, std::string> gEnvCache;
        std::mutex gEnvCacheMutex;

        /// Return true if MPI is initialised and we are world rank 0.
        bool isWorldRank0()
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized)
                return false;
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return rank == 0;
        }

        /// Lookup or read-and-cache a string env var.  The cache holds the raw
        /// getenv result (or defaultValue when unset).
        const std::string &lookupString(const char *name, const std::string &defaultValue)
        {
            std::lock_guard<std::mutex> lock(gEnvCacheMutex);
            auto it = gEnvCache.find(name);
            if (it != gEnvCache.end())
                return it->second;

            const char *env = std::getenv(name);
            if (!env || env[0] == '\0')
            {
                auto [ins, _] = gEnvCache.try_emplace(std::string(name), defaultValue);
                if (isWorldRank0())
                    log() << "EnvReader: " << name << " unset, using default \""
                          << defaultValue << "\"" << std::endl;
                return ins->second;
            }

            auto [ins, _] = gEnvCache.try_emplace(std::string(name), std::string(env));
            if (isWorldRank0())
                log() << "EnvReader: " << name << " = \"" << env << "\"" << std::endl;
            return ins->second;
        }
    } // anonymous namespace

    std::string GetEnvString(const char *name, const std::string &defaultValue)
    {
        return lookupString(name, defaultValue);
    }

    int GetEnvInt(const char *name, int defaultValue)
    {
        const std::string &s = lookupString(name, std::to_string(defaultValue));
        try
        {
            return std::stoi(s);
        }
        catch (...)
        {
            return defaultValue;
        }
    }

    double GetEnvDouble(const char *name, double defaultValue)
    {
        const std::string &s = lookupString(name, std::to_string(defaultValue));
        try
        {
            return std::stod(s);
        }
        catch (...)
        {
            return defaultValue;
        }
    }

    bool GetEnvBool(const char *name, bool defaultValue)
    {
        const std::string &s = lookupString(name, defaultValue ? "1" : "0");
        if (s.empty())
            return defaultValue;
        std::string lower;
        lower.reserve(s.size());
        for (auto c : s)
            lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        if (lower == "1" || lower == "true" || lower == "yes" || lower == "on")
            return true;
        if (lower == "0" || lower == "false" || lower == "no" || lower == "off")
            return false;
        // Unrecognized → return caller's default, not hardcoded false
        return defaultValue;
    }

} // namespace DNDS
