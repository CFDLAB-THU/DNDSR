#pragma once
/// @file ConfigEnum.hpp
/// @brief Extended enum-to-JSON serialization macro that also exposes allowed
///        string values for JSON Schema generation.
///
/// @details
/// ## Problem
///
/// The standard `NLOHMANN_JSON_SERIALIZE_ENUM` macro generates `to_json`/`from_json`
/// overloads for an enum, but provides no way to programmatically query the set of
/// allowed string values at runtime.  The config schema generator needs this list
/// to produce `"enum": ["Roe", "HLLC", ...]` entries.
///
/// ## Solution
///
/// `DNDS_DEFINE_ENUM_JSON` wraps `NLOHMANN_JSON_SERIALIZE_ENUM` and additionally
/// generates a free function `_dnds_enum_allowed_values_<EnumType>()` that returns
/// `std::vector<std::string>` of the valid string representations.
///
/// The companion macro `DNDS_ENUM_ALLOWED_VALUES(EnumType)` calls this function.
///
/// ## Usage
///
/// @code
/// // In Gas.hpp (at namespace scope, after enum definition):
/// enum RiemannSolverType { UnknownRS, Roe, HLLC, HLLEP };
///
/// DNDS_DEFINE_ENUM_JSON(RiemannSolverType,
///     {UnknownRS, nullptr},
///     {Roe, "Roe"},
///     {HLLC, "HLLC"},
///     {HLLEP, "HLLEP"})
///
/// // Then in a config struct's DNDS_DECLARE_CONFIG body:
/// DNDS_FIELD(rsType, "Riemann solver type",
///            DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(RiemannSolverType)));
/// // The schema will include "enum": ["Roe", "HLLC", "HLLEP"]
/// @endcode
///
/// ## Backward Compatibility
///
/// Existing enums using `NLOHMANN_JSON_SERIALIZE_ENUM` continue to work for
/// serialization.  To also get schema enum values, replace with
/// `DNDS_DEFINE_ENUM_JSON`.  Until that replacement, pass allowed values
/// explicitly via `DNDS::Config::enum_values({"Roe","HLLC",...})` in the
/// `DNDS_FIELD` call.
///
/// ## Convention for the nullptr sentinel
///
/// Following the existing DNDSR convention, the first entry in the mapping
/// should map the "unknown/invalid" enum value to `nullptr`.  This entry is
/// skipped when building the allowed-values list for the schema (users should
/// never write `null` in a config file).

#include "DNDS/Serializer/JsonUtil.hpp"
#include <vector>
#include <string>

// ============================================================================
// Internal: Extract string values from the enum mapping pairs.
// ============================================================================
// The NLOHMANN_JSON_SERIALIZE_ENUM macro takes pairs like {EnumVal, "string"}
// or {EnumVal, nullptr}.  We need to extract just the non-null strings.
//
// We use a helper struct that can be constructed from either a string literal
// or nullptr, letting us filter nulls at runtime.

namespace DNDS
{
    namespace detail
    {
        /// @brief A pair of (enum-value, optional string) used to extract allowed values.
        ///
        /// Constructed implicitly from the `{EnumVal, "string"}` initializer pairs.
        /// When the string is nullptr (the unknown/sentinel value), `str` is empty
        /// and `isNull` is true.
        template <typename EnumType>
        struct EnumStringPair
        {
            EnumType value;
            std::string str;
            bool isNull;

            EnumStringPair(EnumType v, const char *s)
                : value(v), str(s ? s : ""), isNull(s == nullptr) {}
            EnumStringPair(EnumType v, std::nullptr_t)  // NOLINT(bugprone-macro-parentheses)
                : value(v), isNull(true) {}
        };

        /// @brief Extract non-null string values from a list of enum-string pairs.
        template <typename EnumType>
        std::vector<std::string> extractEnumStrings(
            std::initializer_list<EnumStringPair<EnumType>> pairs)
        {
            std::vector<std::string> result;
            result.reserve(pairs.size());
            for (const auto &p : pairs)
            {
                if (!p.isNull)
                    result.push_back(p.str);
            }
            return result;
        }
    } // namespace detail
} // namespace DNDS

// ============================================================================
// DNDS_DEFINE_ENUM_JSON(EnumType, ...)
// ============================================================================
/// @brief Define JSON serialization for an enum AND expose its allowed string values.
///
/// Drop-in replacement for `NLOHMANN_JSON_SERIALIZE_ENUM` that additionally
/// generates a function returning the allowed string values for schema emission.
///
/// The variadic arguments are `{EnumValue, "string"}` pairs, identical to those
/// accepted by `NLOHMANN_JSON_SERIALIZE_ENUM`.  Pairs with `nullptr` as the
/// string (sentinel/unknown values) are excluded from the allowed-values list.
///
/// @param EnumType_  The enum type.
/// @param ...        Comma-separated `{EnumValue, "string_or_nullptr"}` pairs.
///
/// Generates:
///   1. `NLOHMANN_JSON_SERIALIZE_ENUM(EnumType_, ...)` — standard serialization.
///   2. `inline std::vector<std::string> _dnds_enum_values_<mangled>()` — allowed values.
///      Accessed via `DNDS_ENUM_ALLOWED_VALUES(EnumType_)`.
#define DNDS_DEFINE_ENUM_JSON(EnumType_, ...)                                                 \
    /* (1) Standard nlohmann enum serialization */                                             \
    NLOHMANN_JSON_SERIALIZE_ENUM(EnumType_, __VA_ARGS__)                                      \
    /* (2) Allowed-values function for schema generation */                                    \
    inline std::vector<std::string> _dnds_enum_allowed_values_fn(EnumType_ *)                 \
    {                                                                                         \
        return ::DNDS::detail::extractEnumStrings<EnumType_>(__VA_ARGS__);                     \
    }

// ============================================================================
// DNDS_ENUM_ALLOWED_VALUES(EnumType)
// ============================================================================
/// @brief Get the list of allowed string values for an enum type.
///
/// Returns `std::vector<std::string>`.  Only works for enums defined with
/// `DNDS_DEFINE_ENUM_JSON`.  Uses ADL on a dummy pointer argument to find
/// the correct overload of `_dnds_enum_allowed_values_fn`.
///
/// @param EnumType_  The enum type (unquoted).
#define DNDS_ENUM_ALLOWED_VALUES(EnumType_) \
    _dnds_enum_allowed_values_fn(static_cast<EnumType_ *>(nullptr))
