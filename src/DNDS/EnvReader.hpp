#pragma once
/// @file EnvReader.hpp
/// @brief Centralized environment-variable reader with typed defaults.
///
/// Provides a single entry point for all runtime environment-variable
/// queries, replacing scattered `std::getenv` calls and ad-hoc
/// `get_env_XXX()` helpers.  Every reader returns @p defaultValue when
/// the variable is unset, empty, or unparseable.

#include <cstdlib>
#include <string>

namespace DNDS
{

    /// @brief Read a string environment variable.
    /// @param name          Name of the environment variable.
    /// @param defaultValue  Value returned when the variable is unset or empty.
    inline std::string GetEnvString(const char *name, const std::string &defaultValue = "")
    {
        const char *env = std::getenv(name);
        if (!env || env[0] == '\0')
            return defaultValue;
        return std::string(env);
    }

    /// @brief Read an integer environment variable.
    /// @param name          Name of the environment variable.
    /// @param defaultValue  Value returned when the variable is unset, empty, or
    ///                      does not contain a valid integer.
    inline int GetEnvInt(const char *name, int defaultValue = 0)
    {
        const char *env = std::getenv(name);
        if (!env || env[0] == '\0')
            return defaultValue;
        try
        {
            return std::stoi(env);
        }
        catch (...)
        {
            return defaultValue;
        }
    }

    /// @brief Read a double-precision environment variable.
    /// @param name          Name of the environment variable.
    /// @param defaultValue  Value returned when the variable is unset, empty, or
    ///                      does not contain a valid floating-point number.
    inline double GetEnvDouble(const char *name, double defaultValue = 0.0)
    {
        const char *env = std::getenv(name);
        if (!env || env[0] == '\0')
            return defaultValue;
        try
        {
            return std::stod(env);
        }
        catch (...)
        {
            return defaultValue;
        }
    }

    /// @brief Read a boolean environment variable.
    /// @param name          Name of the environment variable.
    /// @param defaultValue  Value returned when the variable is unset or empty.
    ///
    /// Accepts case-insensitive forms of: "1", "true", "yes", "on".
    inline bool GetEnvBool(const char *name, bool defaultValue = false)
    {
        const char *env = std::getenv(name);
        if (!env || env[0] == '\0')
            return defaultValue;
        std::string s(env);
        for (auto &c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s == "1" || s == "true" || s == "yes" || s == "on")
            return true;
        return false;
    }

} // namespace DNDS
