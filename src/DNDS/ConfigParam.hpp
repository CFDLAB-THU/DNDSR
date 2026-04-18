#pragma once
/// @file ConfigParam.hpp
/// @brief pybind11-style configuration registration with macro-based field
///        declaration and namespace-scoped tag kwargs.
///
/// @details
/// ## Overview
///
/// Config sections are plain structs.  Metadata is registered in a static
/// method opened by the `DNDS_DECLARE_CONFIG(Type)` macro.  Inside that
/// body the user calls `DNDS_FIELD(member, "description", tags...)` which
/// is a macro that auto-stringifies the member name — no name duplication.
///
/// ## Example
///
/// @code
/// struct ImplicitCFLControl
/// {
///     real CFL = 10;
///     int  nForceLocalStartStep = INT_MAX;
///     bool useLocalDt = true;
///     real RANSRelax = 1;
///
///     DNDS_DECLARE_CONFIG(ImplicitCFLControl)
///     {
///         DNDS_FIELD(CFL,                   "CFL for implicit local dt");
///         DNDS_FIELD(nForceLocalStartStep,   "Step to force local dt",
///                    DNDS::Config::range(0));
///         DNDS_FIELD(useLocalDt,             "Use local (vs uniform) dTau");
///         DNDS_FIELD(RANSRelax,              "RANS under-relaxation factor",
///                    DNDS::Config::range(0.0, 1.0));
///
///         config.check([](const T &s) -> DNDS::CheckResult {
///             if (s.RANSRelax <= 0)
///                 return {false, "RANSRelax must be positive"};
///             return {true, ""};
///         });
///     }
/// };
/// @endcode
///
/// ## Adding a New Parameter
///
/// 1. Declare the member with default (plain C++).
/// 2. Add `DNDS_FIELD(member, "description")` in the DNDS_DECLARE_CONFIG body.
///
/// ## Tag kwargs (namespace-scoped)
///
/// Tags are passed as extra arguments to `DNDS_FIELD`:
///
/// | Tag | Purpose | Example |
/// |-----|---------|---------|
/// | `DNDS::Config::range(min)` | Min constraint (schema + runtime check) | `DNDS::Config::range(0)` |
/// | `DNDS::Config::range(min,max)` | Min+max constraint | `DNDS::Config::range(0.0, 1.0)` |
/// | `DNDS::Config::info(k,v)` | Aux info (`"x-<key>"` in schema) | `DNDS::Config::info("unit","Pa")` |
/// | `DNDS::Config::enum_values(v)` | Allowed string values for enum fields | `DNDS::Config::enum_values({"Roe","HLLC"})` |
///
/// ## Runtime Range Validation
///
/// When a field has a `range()` tag, `readFromJson()` checks the parsed value
/// against the min/max bounds and throws `std::runtime_error` with a clear
/// message on violation.  This catches bad config values at load time.
///
/// ## Section-Level Checks
///
/// @code
/// config.check([](const T &s) -> DNDS::CheckResult { ... });
/// config.check_ctx([](const T &s, const DNDS::ConfigContext &ctx) -> DNDS::CheckResult { ... });
/// @endcode
///
/// ## Nested Sections and Special Fields
///
/// Use explicit `config.*` calls (not the DNDS_FIELD macro) for these:
///
/// @code
/// config.field_section(&T::frameRotation, "frameConstRotation", "Rotating frame");
/// config.field_array_of<BoxInit>(&T::boxInits, "boxInitializers", "Box inits");
/// config.field_map_of<CoarseCtrl>(&T::coarseList, "coarseGridList", "Per-level");
/// config.field_json(&T::extra, "odeSettingsExtra", "Opaque ODE settings");
/// config.field_alias(&T::rsType, "riemannSolverType", "Riemann solver type");
/// @endcode
///
/// ## Trivial Copyability
///
/// The struct has no base class, no virtual methods, no instance-level data
/// introduced by the macro.  `DNDS_DECLARE_CONFIG` only generates static
/// methods and friend functions.

#include "ConfigRegistry.hpp"
#include <type_traits>
#include <utility>

namespace DNDS
{
    // ================================================================
    // Type-to-ConfigTypeTag mapping
    // ================================================================

    template <typename T, typename Enable = void>
    struct ConfigTypeTagOf
    {
        static constexpr ConfigTypeTag value = ConfigTypeTag::Object;
    };

    template <> struct ConfigTypeTagOf<bool>       { static constexpr ConfigTypeTag value = ConfigTypeTag::Bool; };
    template <> struct ConfigTypeTagOf<std::string> { static constexpr ConfigTypeTag value = ConfigTypeTag::String; };

    template <typename T>
    struct ConfigTypeTagOf<T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>>
    {
        static constexpr ConfigTypeTag value = ConfigTypeTag::Int;
    };

    template <typename T>
    struct ConfigTypeTagOf<T, std::enable_if_t<std::is_floating_point_v<T>>>
    {
        static constexpr ConfigTypeTag value = ConfigTypeTag::Real;
    };

    template <typename T>
    struct ConfigTypeTagOf<std::vector<T>>
    {
        static constexpr ConfigTypeTag value = ConfigTypeTag::Array;
    };

    template <typename T, std::size_t N>
    struct ConfigTypeTagOf<std::array<T, N>>
    {
        static constexpr ConfigTypeTag value = ConfigTypeTag::Array;
    };

    template <>
    struct ConfigTypeTagOf<nlohmann::ordered_json>
    {
        static constexpr ConfigTypeTag value = ConfigTypeTag::Json;
    };

    /// C++ enum types serialize as JSON strings via nlohmann, so map to Enum.
    template <typename T>
    struct ConfigTypeTagOf<T, std::enable_if_t<std::is_enum_v<T>>>
    {
        static constexpr ConfigTypeTag value = ConfigTypeTag::Enum;
    };

    /// Eigen matrix/vector types serialize as JSON arrays, so map them to Array.
    /// Detection uses the `Scalar` typedef and `RowsAtCompileTime` enum that
    /// all Eigen matrix expressions expose — no Eigen headers needed here.
    namespace detail
    {
        template <typename T, typename = void>
        struct is_eigen_type : std::false_type {};

        template <typename T>
        struct is_eigen_type<T, std::void_t<
            typename T::Scalar,
            decltype(static_cast<int>(T::RowsAtCompileTime)),
            decltype(static_cast<int>(T::ColsAtCompileTime))>> : std::true_type {};
    } // namespace detail

    template <typename T>
    struct ConfigTypeTagOf<T, std::enable_if_t<detail::is_eigen_type<T>::value>>
    {
        static constexpr ConfigTypeTag value = ConfigTypeTag::Array;
    };

    inline std::string schemaTypeString(ConfigTypeTag tag)
    {
        switch (tag)
        {
        case ConfigTypeTag::Bool:           return "boolean";
        case ConfigTypeTag::Int:            return "integer";
        case ConfigTypeTag::Real:           return "number";
        case ConfigTypeTag::String:         return "string";
        case ConfigTypeTag::Enum:           return "string";
        case ConfigTypeTag::Array:          return "array";
        case ConfigTypeTag::Object:         return "object";
        case ConfigTypeTag::ArrayOfObjects: return "array";
        case ConfigTypeTag::MapOfObjects:   return "object";
        case ConfigTypeTag::Json:           return {};
        }
        return {};
    }

    // ================================================================
    // Namespace-scoped tag factories: DNDS::Config::range(), etc.
    // ================================================================

    /// @brief Namespace for config field tag kwargs.
    ///
    /// Tags are lightweight value objects passed as extra arguments to
    /// `DNDS_FIELD(...)`.  They attach optional metadata (constraints,
    /// auxiliary info, enum values) to a field at registration time.
    namespace Config
    {
        /// @brief Numeric range constraint.
        /// Enforced at parse time in readFromJson and emitted in schema.
        struct RangeTag
        {
            std::optional<double> min;
            std::optional<double> max;
        };

        /// @brief Auxiliary info tag (emitted as `"x-<key>"` in schema).
        struct InfoTag
        {
            std::string key;
            std::string value;
        };

        /// @brief Explicit enum allowed-values tag.
        struct EnumValuesTag
        {
            std::vector<std::string> values;
        };

        /// @brief Create a minimum-only range constraint.
        inline RangeTag range(double min) { return {min, std::nullopt}; }

        /// @brief Create a min+max range constraint.
        inline RangeTag range(double min, double max) { return {{min}, {max}}; }

        /// @brief Create an auxiliary info tag.
        inline InfoTag info(std::string key, std::string value)
        {
            return {std::move(key), std::move(value)};
        }

        /// @brief Create an enum allowed-values tag.
        inline EnumValuesTag enum_values(std::vector<std::string> vals)
        {
            return {std::move(vals)};
        }
    } // namespace Config

    // ================================================================
    // Tag application helpers
    // ================================================================

    namespace detail
    {
        inline void applyTag(FieldMeta &meta, const Config::RangeTag &tag)
        {
            meta.minimum = tag.min;
            meta.maximum = tag.max;
        }

        inline void applyTag(FieldMeta &meta, const Config::InfoTag &tag)
        {
            meta.auxInfo[tag.key] = tag.value;
        }

        inline void applyTag(FieldMeta &meta, const Config::EnumValuesTag &tag)
        {
            meta.enumValues = tag.values;
            meta.typeTag = ConfigTypeTag::Enum;
        }

        inline void applyTags(FieldMeta & /*meta*/) {}

        template <typename Tag, typename... Rest>
        void applyTags(FieldMeta &meta, Tag &&tag, Rest &&...rest)
        {
            applyTag(meta, std::forward<Tag>(tag));
            applyTags(meta, std::forward<Rest>(rest)...);
        }

        /// @brief Build the schemaEntry closure from a fully-tagged FieldMeta.
        ///
        /// Captures the pointer-to-member, description, and all tag data
        /// (range, enum, aux info) to produce the JSON Schema fragment on demand.
        template <typename T, typename V>
        std::function<nlohmann::ordered_json()>
        makeSchemaEntry(V T::*member, const char *desc, const FieldMeta &meta)
        {
            // Capture a snapshot of the tag data.
            auto typeTag = meta.typeTag;
            auto minimum = meta.minimum;
            auto maximum = meta.maximum;
            auto enumVals = meta.enumValues;
            auto auxInfo = meta.auxInfo;

            return [member, desc, typeTag, minimum, maximum, enumVals, auxInfo]() -> nlohmann::ordered_json
            {
                nlohmann::ordered_json s;
                auto ts = schemaTypeString(typeTag);
                if (!ts.empty())
                    s["type"] = ts;
                T defaults{};
                s["default"] = nlohmann::ordered_json(defaults.*member);
                s["description"] = desc;
                if (minimum.has_value())
                    s["minimum"] = minimum.value();
                if (maximum.has_value())
                    s["maximum"] = maximum.value();
                if (!enumVals.empty())
                    s["enum"] = enumVals;
                for (const auto &kv : auxInfo)
                    s["x-" + kv.first] = kv.second;
                return s;
            };
        }

        /// @brief Build a runtime range-check closure.
        ///
        /// Returns a function that, given a JSON value for this field, checks
        /// it against min/max bounds and throws on violation.  Returns nullptr
        /// if no range constraint is set.
        template <typename T, typename V>
        std::function<void(const nlohmann::ordered_json &, const char *)>
        makeRangeChecker(const FieldMeta &meta)
        {
            if (!meta.minimum.has_value() && !meta.maximum.has_value())
                return nullptr;

            auto minimum = meta.minimum;
            auto maximum = meta.maximum;

            return [minimum, maximum](const nlohmann::ordered_json &val, const char *fieldName)
            {
                // Only check numeric types
                if (!val.is_number())
                    return;
                double v = val.get<double>();
                if (minimum.has_value() && v < minimum.value())
                {
                    throw std::runtime_error(
                        fmt::format("Config field '{}': value {} is below minimum {}",
                                    fieldName, v, minimum.value()));
                }
                if (maximum.has_value() && v > maximum.value())
                {
                    throw std::runtime_error(
                        fmt::format("Config field '{}': value {} is above maximum {}",
                                    fieldName, v, maximum.value()));
                }
            };
        }
    } // namespace detail

    // ================================================================
    // ConfigSectionBuilder<T> — the pybind11-style registrar
    // ================================================================

    /// @brief Builder object passed to the user's registration function.
    ///
    /// Named `config` in the `DNDS_DECLARE_CONFIG` expansion for readability.
    /// Provides `field()` (called via `DNDS_FIELD` macro), `check()`, `check_ctx()`,
    /// and specialized field registrars for sections, arrays, maps, and JSON blobs.
    template <typename T>
    class ConfigSectionBuilder
    {
    public:
        // ---- field(): scalar, enum, or any nlohmann-serializable member ----

        /// @brief Register a field with pointer-to-member.
        ///
        /// Typically called via the `DNDS_FIELD` macro (which auto-stringifies
        /// the member name).  Can also be called directly for aliased keys.
        ///
        /// @param member   Pointer-to-member (e.g. `&T::CFL`).
        /// @param jsonKey  The JSON key string.
        /// @param desc     Human-readable description.
        /// @param tags     Zero or more DNDS::Config tag objects.
        template <typename V, typename... Tags>
        void field(V T::*member, const char *jsonKey, const char *desc, Tags &&...tags)
        {
            FieldMeta meta;
            meta.name = jsonKey;
            meta.description = desc;
            meta.typeTag = ConfigTypeTagOf<V>::value;

            // Apply tags first so range/enum data is available for closures.
            detail::applyTags(meta, std::forward<Tags>(tags)...);

            // Build range checker (may be nullptr if no range tag).
            auto rangeChecker = detail::makeRangeChecker<T, V>(meta);

            meta.readField = [member, jsonKey, rangeChecker](const nlohmann::ordered_json &j, void *obj)
            {
                const auto &val = j.at(jsonKey);
                if (rangeChecker)
                    rangeChecker(val, jsonKey);
                static_cast<T *>(obj)->*member = val.template get<V>();
            };
            meta.writeField = [member, jsonKey](nlohmann::ordered_json &j, const void *obj)
            {
                j[jsonKey] = static_cast<const T *>(obj)->*member;
            };
            meta.schemaEntry = detail::makeSchemaEntry<T>(member, desc, meta);

            ConfigRegistry<T>::registerField(std::move(meta));
        }

        // ---- field_alias(): convenience alias for field() with different JSON key ----

        /// @brief Register a field whose JSON key differs from the C++ member name.
        template <typename V, typename... Tags>
        void field_alias(V T::*member, const char *jsonKey, const char *desc, Tags &&...tags)
        {
            field(member, jsonKey, desc, std::forward<Tags>(tags)...);
        }

        // ---- field_section(): nested config sub-section ----

        /// @brief Register a nested sub-section.  The sub-section type must
        ///        itself use DNDS_DECLARE_CONFIG.
        template <typename S, typename... Tags>
        void field_section(S T::*member, const char *jsonKey, const char *desc, Tags &&...tags)
        {
            FieldMeta meta;
            meta.name = jsonKey;
            meta.description = desc;
            meta.typeTag = ConfigTypeTag::Object;
            meta.readField = [member, jsonKey](const nlohmann::ordered_json &j, void *obj)
            {
                // In-place deserialization preserves non-serialized members
                // (e.g. EulerEvaluatorSettings::_nVars set by the constructor).
                from_json(j.at(jsonKey), static_cast<T *>(obj)->*member);
            };
            meta.writeField = [member, jsonKey](nlohmann::ordered_json &j, const void *obj)
            {
                j[jsonKey] = static_cast<const T *>(obj)->*member;
            };
            meta.schemaEntry = [desc]()
            {
                S::_dnds_ensure_registered();
                return ConfigRegistry<S>::emitSchema(desc);
            };
            detail::applyTags(meta, std::forward<Tags>(tags)...);
            ConfigRegistry<T>::registerField(std::move(meta));
        }

        // ---- field_array_of<S>(): std::vector<S> ----

        /// @brief Register a `std::vector<S>` field (array of sub-objects).
        template <typename S, typename... Tags>
        void field_array_of(std::vector<S> T::*member, const char *jsonKey, const char *desc, Tags &&...tags)
        {
            FieldMeta meta;
            meta.name = jsonKey;
            meta.description = desc;
            meta.typeTag = ConfigTypeTag::ArrayOfObjects;
            meta.readField = [member, jsonKey](const nlohmann::ordered_json &j, void *obj)
            {
                static_cast<T *>(obj)->*member = j.at(jsonKey).template get<std::vector<S>>();
            };
            meta.writeField = [member, jsonKey](nlohmann::ordered_json &j, const void *obj)
            {
                j[jsonKey] = static_cast<const T *>(obj)->*member;
            };
            meta.schemaEntry = [desc]() -> nlohmann::ordered_json
            {
                nlohmann::ordered_json s;
                s["type"] = "array";
                s["description"] = desc;
                S::_dnds_ensure_registered();
                s["items"] = ConfigRegistry<S>::emitSchema();
                return s;
            };
            detail::applyTags(meta, std::forward<Tags>(tags)...);
            ConfigRegistry<T>::registerField(std::move(meta));
        }

        // ---- field_map_of<S>(): std::map<std::string, S> ----

        /// @brief Register a `std::map<std::string, S>` field.
        template <typename S, typename... Tags>
        void field_map_of(std::map<std::string, S> T::*member, const char *jsonKey, const char *desc, Tags &&...tags)
        {
            FieldMeta meta;
            meta.name = jsonKey;
            meta.description = desc;
            meta.typeTag = ConfigTypeTag::MapOfObjects;
            meta.readField = [member, jsonKey](const nlohmann::ordered_json &j, void *obj)
            {
                static_cast<T *>(obj)->*member =
                    j.at(jsonKey).template get<std::map<std::string, S>>();
            };
            meta.writeField = [member, jsonKey](nlohmann::ordered_json &j, const void *obj)
            {
                j[jsonKey] = static_cast<const T *>(obj)->*member;
            };
            meta.schemaEntry = [desc]() -> nlohmann::ordered_json
            {
                nlohmann::ordered_json s;
                s["type"] = "object";
                s["description"] = desc;
                S::_dnds_ensure_registered();
                s["additionalProperties"] = ConfigRegistry<S>::emitSchema();
                return s;
            };
            detail::applyTags(meta, std::forward<Tags>(tags)...);
            ConfigRegistry<T>::registerField(std::move(meta));
        }

        // ---- field_json(): opaque JSON blob ----

        /// @brief Register an opaque `nlohmann::ordered_json` field.
        template <typename... Tags>
        void field_json(nlohmann::ordered_json T::*member, const char *jsonKey, const char *desc, Tags &&...tags)
        {
            FieldMeta meta;
            meta.name = jsonKey;
            meta.description = desc;
            meta.typeTag = ConfigTypeTag::Json;
            meta.readField = [member, jsonKey](const nlohmann::ordered_json &j, void *obj)
            {
                static_cast<T *>(obj)->*member = j.at(jsonKey);
            };
            meta.writeField = [member, jsonKey](nlohmann::ordered_json &j, const void *obj)
            {
                j[jsonKey] = static_cast<const T *>(obj)->*member;
            };
            meta.schemaEntry = [desc]() -> nlohmann::ordered_json
            {
                nlohmann::ordered_json s;
                s["description"] = desc;
                return s;
            };
            detail::applyTags(meta, std::forward<Tags>(tags)...);
            ConfigRegistry<T>::registerField(std::move(meta));
        }

        // ---- field_json_schema(): opaque JSON blob with explicit schema ----

        /// @brief Register an opaque `nlohmann::ordered_json` field with a
        ///        user-supplied schema generator.
        ///
        /// Use this for heterogeneous structures (e.g. arrays of discriminated
        /// union objects) where automatic schema inference is not possible.
        ///
        /// @param member       Pointer-to-member.
        /// @param jsonKey      JSON key name.
        /// @param desc         Human-readable description.
        /// @param schemaFn     Callable `() -> ordered_json` returning the
        ///                     full JSON Schema for this field.
        /// @param tags         Optional tag objects.
        template <typename FSchema, typename... Tags>
        void field_json_schema(nlohmann::ordered_json T::*member,
                               const char *jsonKey, const char *desc,
                               FSchema &&schemaFn, Tags &&...tags)
        {
            FieldMeta meta;
            meta.name = jsonKey;
            meta.description = desc;
            meta.typeTag = ConfigTypeTag::Json;
            meta.readField = [member, jsonKey](const nlohmann::ordered_json &j, void *obj)
            {
                static_cast<T *>(obj)->*member = j.at(jsonKey);
            };
            meta.writeField = [member, jsonKey](nlohmann::ordered_json &j, const void *obj)
            {
                j[jsonKey] = static_cast<const T *>(obj)->*member;
            };
            meta.schemaEntry = [fn = std::forward<FSchema>(schemaFn)]() -> nlohmann::ordered_json
            {
                return fn();
            };
            detail::applyTags(meta, std::forward<Tags>(tags)...);
            ConfigRegistry<T>::registerField(std::move(meta));
        }

        // ---- check(): cross-field validation ----

        /// @brief Register a context-free cross-field check.
        /// @param f  Lambda `(const T&) -> CheckResult`.
        template <typename F>
        auto check(F &&f) -> decltype(f(std::declval<const T &>()), void())
        {
            ConfigRegistry<T>::registerCheck(
                [fn = std::forward<F>(f)](const void *obj) -> CheckResult
                {
                    return fn(*static_cast<const T *>(obj));
                });
        }

        /// @brief Register a context-free cross-field check with a message string.
        /// @param msg  Error message shown when the check fails.
        /// @param pred Lambda `(const T&) -> bool`, returns true if the check passes.
        template <typename F>
        void check(const char *msg, F &&pred)
        {
            ConfigRegistry<T>::registerCheck(
                [message = std::string(msg), fn = std::forward<F>(pred)](const void *obj) -> CheckResult
                {
                    bool ok = fn(*static_cast<const T *>(obj));
                    return CheckResult{ok, ok ? "" : message};
                });
        }

        // ---- check_ctx(): context-aware cross-field validation ----

        /// @brief Register a context-aware cross-field check.
        /// @param f  Lambda `(const T&, const ConfigContext&) -> CheckResult`.
        template <typename F>
        auto check_ctx(F &&f) -> decltype(f(std::declval<const T &>(), std::declval<const ConfigContext &>()), void())
        {
            ConfigRegistry<T>::registerContextualCheck(
                [fn = std::forward<F>(f)](const void *obj, const ConfigContext &ctx) -> CheckResult
                {
                    return fn(*static_cast<const T *>(obj), ctx);
                });
        }

        // ---- post_read(): hook called after all fields are deserialized ----

        /// @brief Register a post-read hook for recomputing derived quantities.
        /// @param f  Lambda `(T&) -> void`, called after readFromJson completes.
        template <typename F>
        void post_read(F &&f)
        {
            ConfigRegistry<T>::registerPostReadHook(std::forward<F>(f));
        }
    };

} // namespace DNDS

// ============================================================================
// DNDS_FIELD(member, description, tags...)
// ============================================================================
/// @brief Register a field inside a DNDS_DECLARE_CONFIG body.
///
/// Auto-stringifies the member name so you never write it twice.
/// The JSON key equals the C++ member name.
///
/// @param name_  Member name (unquoted).
/// @param desc_  Description string literal.
/// @param ...    Zero or more DNDS::Config tag objects.
///
/// @code
/// DNDS_FIELD(CFL, "CFL for implicit dt", DNDS::Config::range(0));
/// @endcode
#define DNDS_FIELD(name_, desc_, ...) \
    config.field(&T::name_, #name_, desc_, ##__VA_ARGS__)

// ============================================================================
// DNDS_DECLARE_CONFIG(Type)
// ============================================================================
/// @brief Open a config section registration body.
///
/// Expands to:
///   - `using T = Type` (so DNDS_FIELD can reference `&T::member`).
///   - Lazy `_dnds_ensure_registered()` with one-time init guard.
///   - Friend `to_json` / `from_json` calling ensureRegistered first.
///   - `schema()`, `validate()`, `validateWithContext()`, `validateKeys()`.
///   - Opens `static void _dnds_do_register(ConfigSectionBuilder<T>& config)`
///     — the user provides the `{ ... }` body after the macro.
///
/// @code
/// struct MySection
/// {
///     real a = 1.0;
///     int  b = 42;
///
///     DNDS_DECLARE_CONFIG(MySection)
///     {
///         DNDS_FIELD(a, "The a parameter", DNDS::Config::range(0));
///         DNDS_FIELD(b, "The b parameter");
///     }
/// };
/// @endcode
#define DNDS_DECLARE_CONFIG(Type_)                                                            \
    using T = Type_;                                                                          \
    static void _dnds_ensure_registered()                                                     \
    {                                                                                         \
        static bool done = false;                                                             \
        if (done) return;                                                                     \
        done = true;                                                                          \
        ::DNDS::ConfigSectionBuilder<Type_> config;                                           \
        _dnds_do_register(config);                                                            \
    }                                                                                         \
    friend void to_json(nlohmann::ordered_json &j, const Type_ &t)                            \
    {                                                                                         \
        Type_::_dnds_ensure_registered();                                                     \
        ::DNDS::ConfigRegistry<Type_>::writeToJson(j, t);                                     \
    }                                                                                         \
    friend void from_json(const nlohmann::ordered_json &j, Type_ &t)                          \
    {                                                                                         \
        Type_::_dnds_ensure_registered();                                                     \
        ::DNDS::ConfigRegistry<Type_>::readFromJson(j, t);                                    \
    }                                                                                         \
    static nlohmann::ordered_json schema(const std::string &desc = "")                        \
    {                                                                                         \
        Type_::_dnds_ensure_registered();                                                     \
        return ::DNDS::ConfigRegistry<Type_>::emitSchema(desc);                               \
    }                                                                                         \
    std::vector<::DNDS::CheckResult> validate() const                                         \
    {                                                                                         \
        Type_::_dnds_ensure_registered();                                                     \
        return ::DNDS::ConfigRegistry<Type_>::validate(*this);                                \
    }                                                                                         \
    std::vector<::DNDS::CheckResult> validateWithContext(                                      \
        const ::DNDS::ConfigContext &ctx) const                                               \
    {                                                                                         \
        Type_::_dnds_ensure_registered();                                                     \
        return ::DNDS::ConfigRegistry<Type_>::validateWithContext(*this, ctx);                 \
    }                                                                                         \
    static void validateKeys(const nlohmann::ordered_json &j)                                 \
    {                                                                                         \
        Type_::_dnds_ensure_registered();                                                     \
        ::DNDS::ConfigRegistry<Type_>::validateKeys(j);                                       \
    }                                                                                         \
    static void _dnds_do_register(::DNDS::ConfigSectionBuilder<Type_> &config)
