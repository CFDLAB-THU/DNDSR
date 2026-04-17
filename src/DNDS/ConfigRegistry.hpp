#pragma once
/// @file ConfigRegistry.hpp
/// @brief Per-type configuration field registry with JSON serialization,
///        JSON Schema (draft-07) emission, and cross-field validation.
///
/// @details
/// ## Overview
///
/// `ConfigRegistry<T>` is a singleton that stores, for each config struct `T`:
///
///   - An ordered list of `FieldMeta` descriptors (one per JSON-visible member).
///   - Lists of cross-field checks (context-free and context-aware).
///
/// The registry is populated lazily on first use by the `DNDS_DECLARE_CONFIG`
/// machinery in ConfigParam.hpp.  It never affects the layout or trivial
/// copyability of `T` itself --- all metadata lives in static singletons,
/// so `T` remains a POD struct safe for CUDA device views.
///
/// ## Design Principles
///
/// 1. **POD structs are untouched.**  `DNDS_DECLARE_CONFIG` generates only
///    static methods and friend functions.  The struct has no base class, no
///    virtual methods, and no added instance data.  Structs embedded in CUDA
///    device views (e.g. `FiniteVolumeSettings`) remain trivially copyable.
///
/// 2. **Type-erased field accessors.**  Each `FieldMeta` stores `std::function`
///    lambdas for read, write, and schema emission.  These capture a
///    pointer-to-member and live only in host-side static storage.
///
/// 3. **Per-field + cross-field validation.**  Single-field constraints (range,
///    enum membership) are checked inside each field's `readField` closure at
///    parse time.  Multi-field constraints (mutual exclusion, conditional
///    requirements, derived-value consistency) are registered as standalone
///    check lambdas via `config.check()` / `config.check_ctx()`.
///
/// 4. **Context-aware validation.**  Some checks depend on runtime values not
///    stored in the config (e.g. `nVars`, `model`).  These use `ConfigContext`,
///    passed to `validateWithContext()`.
///
/// 5. **Incremental adoption.**  Sections using the old
///    `DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON` keep working.
///    The `--emit-schema` flag falls back to type-inference from default JSON
///    for unmigrated sections.  Migrated and unmigrated sections coexist.
///
/// ## Usage
///
/// Users interact with this file indirectly through ConfigParam.hpp:
///
/// @code
/// struct TimeMarchControl
/// {
///     real dtImplicit = 1e100;
///     int  nTimeStep  = 1000000;
///
///     DNDS_DECLARE_CONFIG(TimeMarchControl)
///     {
///         DNDS_FIELD(dtImplicit, "Max time step; 1e100 for steady",
///                    DNDS::Config::range(0));
///         DNDS_FIELD(nTimeStep,  "Max number of time steps",
///                    DNDS::Config::range(1));
///     }
/// };
/// @endcode
///
/// Direct registry access (for tooling, `--emit-schema`, etc.):
///
/// @code
/// auto schema = DNDS::ConfigRegistry<TimeMarchControl>::emitSchema("Time march");
/// auto errors = DNDS::ConfigRegistry<TimeMarchControl>::validate(myConfig);
/// @endcode

#include "Defines.hpp"
#include "JsonUtil.hpp"
#include "Errors.hpp"

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <optional>
#include <stdexcept>

#include <fmt/core.h>

namespace DNDS
{

    /// @brief Enumerates the JSON Schema type associated with a config field.
    ///
    /// Used by FieldMeta::schemaEntry to emit the correct `"type"` value in the
    /// generated JSON Schema.  The mapping is:
    ///
    /// | Tag            | JSON Schema type | C++ types                         |
    /// |----------------|------------------|-----------------------------------|
    /// | Bool           | "boolean"        | bool                              |
    /// | Int            | "integer"        | int, int32_t, int64_t, uint8_t    |
    /// | Real           | "number"         | real (double), float              |
    /// | String         | "string"         | std::string                       |
    /// | Enum           | "string" + enum  | any enum with DNDS_DEFINE_ENUM_JSON|
    /// | Array          | "array"          | std::vector<scalar>, Eigen::VectorXd |
    /// | Object         | "object"         | nested config section             |
    /// | ArrayOfObjects | "array" of objects | std::vector<Section>            |
    /// | MapOfObjects   | "object" (additionalProperties) | std::map<string,Section> |
    /// | Json           | {} (any)         | nlohmann::ordered_json            |
    enum class ConfigTypeTag
    {
        Bool,
        Int,
        Real,
        String,
        Enum,
        Array,
        Object,
        ArrayOfObjects,
        MapOfObjects,
        Json,
    };

    /// @brief Result of a single validation check.
    ///
    /// @param passed  true if the check passed, false otherwise.
    /// @param message Human-readable error description.  Empty when passed is true.
    ///                Should include the field name(s) involved and the violated
    ///                constraint for clear diagnostics.
    struct CheckResult
    {
        bool passed;
        std::string message;
    };

    /// @brief Runtime context supplied to context-aware validation checks.
    ///
    /// Some cross-field validations depend on values that are not stored in the
    /// config section itself --- for example, `nVars` depends on the EulerModel
    /// template parameter and is only known at runtime.
    ///
    /// Pass a populated ConfigContext to `validateWithContext()`.
    /// Checks registered via `config.check()` ignore this struct.
    /// Checks registered via `config.check_ctx()` receive it as a second argument.
    struct ConfigContext
    {
        int nVars = -1;       ///< Number of solution variables (model-dependent).
        int dim = -1;         ///< Spatial dimension (2 or 3).
        int gDim = -1;        ///< Geometric dimension (2 or 3).
        int modelCode = -1;   ///< Integer code identifying the EulerModel enum value.
    };

    /// @brief Descriptor for a single configuration field.
    ///
    /// Stores everything needed to serialize, deserialize, validate, and generate
    /// JSON Schema for one member of a config struct.  Created by
    /// `ConfigSectionBuilder::field()` (typically via the `DNDS_FIELD` macro)
    /// and stored in the per-type `ConfigRegistry<T>` singleton.
    ///
    /// All `std::function` members capture a pointer-to-member and description
    /// string.  They live in host-side static storage and never reference
    /// instance state.
    struct FieldMeta
    {
        std::string name;        ///< JSON key name (may differ from C++ member name for aliased fields).
        std::string description; ///< Human-readable description, used in JSON Schema and generated docs.
        ConfigTypeTag typeTag;   ///< JSON Schema type category.

        /// @brief Read this field from a JSON object into a struct instance.
        /// @param j   The JSON object to read from (must contain `name` as a key).
        /// @param obj Pointer to the struct instance (type-erased as void*).
        /// @throws nlohmann::json::out_of_range if the key is missing.
        /// @throws nlohmann::json::type_error if the JSON value type doesn't match.
        std::function<void(const nlohmann::ordered_json &j, void *obj)> readField;

        /// @brief Write this field from a struct instance into a JSON object.
        /// @param j   The JSON object to write into.
        /// @param obj Pointer to the struct instance (type-erased as const void*).
        std::function<void(nlohmann::ordered_json &j, const void *obj)> writeField;

        /// @brief Emit the JSON Schema fragment for this field.
        /// @return An ordered_json object like `{"type":"number","default":1e100,"description":"..."}`.
        ///
        /// For enum fields, also includes `"enum": [...]`.
        /// For object fields, delegates to the nested section's `ConfigRegistry::emitSchema()`.
        /// For array fields, includes `"items": {...}`.
        std::function<nlohmann::ordered_json()> schemaEntry;

        /// @brief Allowed string values for enum fields.  Empty for non-enum fields.
        std::vector<std::string> enumValues;

        /// @brief Optional minimum constraint for numeric fields (used in schema + validation).
        std::optional<double> minimum;

        /// @brief Optional maximum constraint for numeric fields (used in schema + validation).
        std::optional<double> maximum;

        /// @brief Auxiliary key-value info (emitted as `"x-<key>": "<value>"` in schema).
        /// Use for units, references, version notes, etc.
        std::map<std::string, std::string> auxInfo;
    };

    /// @brief A cross-field validation check that does not require runtime context.
    ///
    /// The function receives a const void* pointing to the config section struct.
    /// It should cast to `const T&` and inspect the fields, returning CheckResult.
    using CrossFieldCheck = std::function<CheckResult(const void *obj)>;

    /// @brief A cross-field validation check that requires runtime context.
    ///
    /// The function receives a const void* pointing to the config section struct
    /// and a ConfigContext carrying runtime values (nVars, dim, model, etc.).
    using ContextualCheck = std::function<CheckResult(const void *obj, const ConfigContext &ctx)>;

    /// @brief Per-type singleton registry of config field metadata and validation checks.
    ///
    /// @tparam T The config section struct type (e.g. LimiterControl, VRSettings).
    ///
    /// ## Thread Safety
    ///
    /// Registration happens lazily on first use of `to_json`, `from_json`, or
    /// `schema()` (via `_dnds_ensure_registered()`).  After the one-time init
    /// completes, the registry is read-only.  All const accessors and operations
    /// (`readFromJson`, `writeToJson`, `emitSchema`, `validate`) are safe to
    /// call concurrently from multiple threads.
    ///
    /// ## Registration Order
    ///
    /// Fields are stored in the order the `DNDS_FIELD` / `config.field()` calls
    /// execute inside the `_dnds_do_register()` body, which matches the source
    /// order.  This gives deterministic JSON key ordering and schema property
    /// ordering.
    template <typename T>
    class ConfigRegistry
    {
        /// @brief Mutable access to the field list (used only during static init).
        static std::vector<FieldMeta> &fieldsMut()
        {
            static std::vector<FieldMeta> fs;
            return fs;
        }

        /// @brief Mutable access to the context-free check list.
        static std::vector<CrossFieldCheck> &checksMut()
        {
            static std::vector<CrossFieldCheck> cs;
            return cs;
        }

        /// @brief Mutable access to the context-aware check list.
        static std::vector<ContextualCheck> &ctxChecksMut()
        {
            static std::vector<ContextualCheck> cs;
            return cs;
        }

    public:
        // ================================================================
        // Registration API (called by ConfigSectionBuilder during lazy init)
        // ================================================================

        /// @brief Register a single field's metadata.
        /// @param meta  The field descriptor to store.
        /// @return Always true (return value kept for legacy compatibility).
        static bool registerField(FieldMeta meta)
        {
            fieldsMut().push_back(std::move(meta));
            return true;
        }

        /// @brief Register a cross-field validation check (no runtime context needed).
        /// @param check  Lambda taking `const void*` and returning CheckResult.
        /// @return Always true.
        static bool registerCheck(CrossFieldCheck check)
        {
            checksMut().push_back(std::move(check));
            return true;
        }

        /// @brief Register a cross-field validation check that needs runtime context.
        /// @param check  Lambda taking `const void*` and `const ConfigContext&`, returning CheckResult.
        /// @return Always true.
        static bool registerContextualCheck(ContextualCheck check)
        {
            ctxChecksMut().push_back(std::move(check));
            return true;
        }

        // ================================================================
        // Read-only accessors (safe to call from any thread after main starts)
        // ================================================================

        /// @brief All registered field descriptors, in declaration order.
        static const std::vector<FieldMeta> &fields() { return fieldsMut(); }

        /// @brief All registered context-free checks.
        static const std::vector<CrossFieldCheck> &checks() { return checksMut(); }

        /// @brief All registered context-aware checks.
        static const std::vector<ContextualCheck> &contextualChecks() { return ctxChecksMut(); }

        // ================================================================
        // JSON Serialization
        // ================================================================

        /// @brief Deserialize all registered fields from a JSON object into a struct.
        ///
        /// For each field, reads `j[field.name]` and writes it into the
        /// corresponding member of `obj`.  If the field has a range constraint
        /// (from `DNDS::Config::range()`), the parsed numeric value is checked
        /// against min/max bounds before assignment.
        ///
        /// @param j   Source JSON object.
        /// @param obj Destination struct instance.
        /// @throws std::runtime_error on missing keys, type mismatch, or
        ///         range constraint violation.
        static void readFromJson(const nlohmann::ordered_json &j, T &obj)
        {
            for (const auto &f : fields())
            {
                try
                {
                    f.readField(j, &obj);
                }
                catch (const std::exception &e)
                {
                    throw std::runtime_error(
                        fmt::format("Error reading config field '{}': {}", f.name, e.what()));
                }
            }
        }

        /// @brief Serialize all registered fields from a struct into a JSON object.
        ///
        /// Fields are written in registration order, producing deterministic key ordering
        /// in the output JSON.
        ///
        /// @param j   Destination JSON object (existing keys are overwritten).
        /// @param obj Source struct instance.
        static void writeToJson(nlohmann::ordered_json &j, const T &obj)
        {
            for (const auto &f : fields())
                f.writeField(j, &obj);
        }

        // ================================================================
        // JSON Schema Generation
        // ================================================================

        /// @brief Emit a JSON Schema (draft-07) object describing all registered fields.
        ///
        /// The output looks like:
        /// @code
        /// {
        ///   "type": "object",
        ///   "description": "...",
        ///   "properties": {
        ///     "dtImplicit": { "type": "number", "default": 1e100, "description": "..." },
        ///     "nTimeStep": { "type": "integer", "default": 1000000, "description": "..." },
        ///     ...
        ///   }
        /// }
        /// @endcode
        ///
        /// @param sectionDescription  Optional description for the section itself.
        /// @return The schema JSON object.
        static nlohmann::ordered_json emitSchema(const std::string &sectionDescription = "")
        {
            nlohmann::ordered_json schema;
            schema["type"] = "object";
            if (!sectionDescription.empty())
                schema["description"] = sectionDescription;
            auto &props = schema["properties"] = nlohmann::ordered_json::object();
            for (const auto &f : fields())
                props[f.name] = f.schemaEntry();
            return schema;
        }

        // ================================================================
        // Validation
        // ================================================================

        /// @brief Run all context-free cross-field checks on a struct instance.
        ///
        /// @param obj  The struct to validate.
        /// @return A vector of failed CheckResults.  Empty if all checks pass.
        static std::vector<CheckResult> validate(const T &obj)
        {
            std::vector<CheckResult> failures;
            for (const auto &check : checks())
            {
                auto r = check(static_cast<const void *>(&obj));
                if (!r.passed)
                    failures.push_back(std::move(r));
            }
            return failures;
        }

        /// @brief Run all checks (both context-free and context-aware) on a struct instance.
        ///
        /// @param obj  The struct to validate.
        /// @param ctx  Runtime context (nVars, dim, model, etc.).
        /// @return A vector of failed CheckResults.  Empty if all checks pass.
        static std::vector<CheckResult> validateWithContext(const T &obj, const ConfigContext &ctx)
        {
            auto failures = validate(obj);
            for (const auto &check : contextualChecks())
            {
                auto r = check(static_cast<const void *>(&obj), ctx);
                if (!r.passed)
                    failures.push_back(std::move(r));
            }
            return failures;
        }

        // ================================================================
        // Key Validation (unknown-key detection)
        // ================================================================

        /// @brief Check that every key in a user-supplied JSON object corresponds to
        ///        a registered field.  Throws on the first unknown key found.
        ///
        /// This is the equivalent of EulerP's `valid_patch_keys()`, but generated
        /// automatically from the registry instead of requiring a hand-written
        /// default config to compare against.
        ///
        /// @param userJson  The user-supplied JSON object to validate.
        /// @throws std::runtime_error with the offending key path.
        static void validateKeys(const nlohmann::ordered_json &userJson)
        {
            if (!userJson.is_object())
                return;
            for (const auto &[key, val] : userJson.items())
            {
                bool found = false;
                for (const auto &f : fields())
                {
                    if (f.name == key)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    throw std::runtime_error(
                        fmt::format("Unknown configuration key '{}'. Check spelling or "
                                    "use --emit-schema to see valid keys.",
                                    key));
                }
            }
        }
    };

} // namespace DNDS
