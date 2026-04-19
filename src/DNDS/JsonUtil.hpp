#pragma once
/// @file JsonUtil.hpp
/// @brief JSON-to-Eigen conversion utilities and nlohmann_json helper macros.

#include "Defines.hpp"
#define JSON_ASSERT DNDS_assert
#include <nlohmann/json.hpp>
#include "EigenUtil.hpp"
#include "DeviceStorage.hpp"
#include "Vector.hpp"

namespace DNDS
{
    /// @brief Project-wide JSON type alias: nlohmann/json with ordered keys.
    /// @details Order preservation makes generated config files diffable across
    /// re-saves. Use this for all DNDSR-facing configuration objects; reach for
    /// the unordered `nlohmann::json` only where compatibility with third-party
    /// producers demands it.
    using t_jsonconfig = nlohmann::ordered_json;
}

namespace DNDS
{

    /// @brief Parse a JSON array into an `Eigen::VectorXd`. Throws a descriptive
    /// assertion on any JSON error.
    inline Eigen::VectorXd JsonGetEigenVector(const nlohmann::json &arr)
    {
        try
        {
            DNDS_assert(arr.is_array());
            Eigen::VectorXd ret;
            ret.resize(arr.size());
            for (int i = 0; i < ret.size(); i++)
                ret(i) = arr.at(i).get<double>();
            return ret;
        }
        catch (...)
        {
            DNDS_assert_info(false, "array parse bad: \n" + arr.dump());
            return Eigen::VectorXd{0};
        }
    }

    /// @brief Parse a JSON array into an `Eigen`::VectorFMTSafe (fixed-point-aware wrapper).
    /// @details Used by configuration paths that feed values into code paths
    /// compiled with the fixed-point Eigen shim.
    inline Eigen::VectorFMTSafe<real, -1> JsonGetEigenVectorFMTSafe(const nlohmann::json &arr)
    {
        try
        {
            DNDS_assert(arr.is_array());
            Eigen::VectorFMTSafe<real, -1> ret;
            ret.resize(arr.size());
            for (int i = 0; i < ret.size(); i++)
                ret(i) = arr.at(i).get<double>();
            return ret;
        }
        catch (...)
        {
            DNDS_assert_info(false, "array parse bad");
            return Eigen::VectorFMTSafe<real, -1>{0};
        }
    }

    /// @brief Dump an `Eigen::VectorXd` into a JSON array of doubles.
    inline auto EigenVectorGetJson(const Eigen::VectorXd &ve)
    {
        std::vector<real> v;
        v.resize(ve.size());
        for (size_t i = 0; i < ve.size(); i++)
            v[i] = ve[i];
        return nlohmann::json(v);
    }

    /// @brief Dump an `Eigen`::VectorFMTSafe into a JSON array of doubles.
    inline auto EigenVectorFMTSafeGetJson(const Eigen::VectorFMTSafe<real, -1> &ve)
    {
        std::vector<real> v;
        v.resize(ve.size());
        for (size_t i = 0; i < ve.size(); i++)
            v[i] = ve[i];
        return nlohmann::json(v);
    }

/**
 * @brief Helper macro: read (`read == true`) or write (`read == false`) a named
 * member into / out of a `jsonObj`.
 *
 * @details Used in the common pattern of mirroring a config struct between
 * JSON and C++:
 * ```cpp
 * #define __F(v) __DNDS__json_to_config(v)
 * __F(gamma); __F(CFL); __F(maxIter);
 * ```
 * Errors during reading are surfaced via @ref DNDS_assert_info with the member name.
 */
#define __DNDS__json_to_config(name)                                         \
    {                                                                        \
        if (read)                                                            \
            try                                                              \
            {                                                                \
                ((name) = jsonObj.at(#name).template get<decltype(name)>()); \
            }                                                                \
            catch (const std::exception &v)                                  \
            {                                                                \
                std::cerr << v.what() << std::endl;                          \
                DNDS_assert_info(false, #name);                              \
            }                                                                \
        else                                                                 \
            (jsonObj[#name] = (name));                                       \
    }
/**
 * @brief Like @ref NLOHMANN_DEFINE_TYPE_INTRUSIVE but targets `nlohmann::ordered_json`.
 * @details DNDSR prefers ordered JSON for configuration; this macro wires up
 * both `to_json` and `from_json` friend functions against the ordered type.
 */
#define DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(Type, ...)                                                                                                   \
    friend void to_json(nlohmann::ordered_json &nlohmann_json_j, const Type &nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) } \
    friend void from_json(const nlohmann::ordered_json &nlohmann_json_j, Type &nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__)) }

/// @brief Like @ref DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON but
/// additionally installs the unordered-JSON overloads for interop with code
/// that uses `nlohmann::json` directly.
#define DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_AND_UNORDERED_JSON(Type, ...)                                                                                         \
    friend void to_json(nlohmann::ordered_json &nlohmann_json_j, const Type &nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) }     \
    friend void from_json(const nlohmann::ordered_json &nlohmann_json_j, Type &nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__)) } \
    friend void to_json(nlohmann::json &nlohmann_json_j, const Type &nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) }             \
    friend void from_json(const nlohmann::json &nlohmann_json_j, Type &nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__)) }
}

namespace Eigen // why doesn't work?
{
    inline void to_json(nlohmann::json &j, const VectorXd &v)
    {
        j = DNDS::EigenVectorGetJson(v);
    }

    inline void from_json(const nlohmann::json &j, VectorXd &v)
    {
        v = DNDS::JsonGetEigenVector(j);
    }

    inline void to_json(nlohmann::ordered_json &j, const VectorXd &v)
    {
        j = DNDS::EigenVectorGetJson(v);
    }

    inline void from_json(const nlohmann::ordered_json &j, VectorXd &v)
    {
        v = DNDS::JsonGetEigenVector(j);
    }

    inline void to_json(nlohmann::ordered_json &j, const VectorFMTSafe<DNDS::real, -1> &v)
    {
        j = DNDS::EigenVectorFMTSafeGetJson(v);
    }

    inline void from_json(const nlohmann::ordered_json &j, VectorFMTSafe<DNDS::real, -1> &v)
    {
        v = DNDS::JsonGetEigenVectorFMTSafe(j);
    }

    inline void to_json(nlohmann::ordered_json &j, const Vector3d &v)
    {
        j = DNDS::EigenVectorGetJson(v);
    }

    inline void from_json(const nlohmann::ordered_json &j, Vector3d &v)
    {
        v = DNDS::JsonGetEigenVector(j);
    }
}

namespace DNDS
{
    inline void to_json(nlohmann::ordered_json &j, const host_device_vector<real> &v)
    {
        std::vector<real> v_vec = (std::vector<real>)(v);
        j = v_vec;
    }

    inline void from_json(const nlohmann::ordered_json &j, host_device_vector<real> &v)
    {
        std::vector<real> v_vec = j;
        v = v_vec;
    }
}
