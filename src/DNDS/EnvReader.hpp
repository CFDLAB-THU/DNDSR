#pragma once
/// @file EnvReader.hpp
/// @brief Centralized environment-variable reader with typed defaults.
///
/// Provides a single entry point for all runtime environment-variable
/// queries, replacing scattered `std::getenv` calls and ad-hoc
/// `get_env_XXX()` helpers.  Every reader returns @p defaultValue when
/// the variable is unset, empty, or unparseable.
///
/// Reads are cached on first access (global map).  On the first read of
/// each variable, the value (or default) is printed to DNDS::log() on
/// MPI world rank 0.

#include <string>

namespace DNDS
{

    /// @brief Read a string environment variable.
    /// @param name          Name of the environment variable.
    /// @param defaultValue  Value returned when the variable is unset or empty.
    std::string GetEnvString(const char *name, const std::string &defaultValue = "");

    /// @brief Read an integer environment variable.
    /// @param name          Name of the environment variable.
    /// @param defaultValue  Value returned when the variable is unset, empty, or
    ///                      does not contain a valid integer.
    int GetEnvInt(const char *name, int defaultValue = 0);

    /// @brief Read a double-precision environment variable.
    /// @param name          Name of the environment variable.
    /// @param defaultValue  Value returned when the variable is unset, empty, or
    ///                      does not contain a valid floating-point number.
    double GetEnvDouble(const char *name, double defaultValue = 0.0);

    /// @brief Read a boolean environment variable.
    /// @param name          Name of the environment variable.
    /// @param defaultValue  Value returned when the variable is unset or empty.
    ///
    /// Accepts case-insensitive forms of: "1", "true", "yes", "on".
    bool GetEnvBool(const char *name, bool defaultValue = false);

} // namespace DNDS
