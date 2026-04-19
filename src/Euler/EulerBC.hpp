/** @file EulerBC.hpp
 *  @brief Boundary condition types, handlers, integration recorders, and 1-D profile
 *         utilities for the compressible Euler/Navier-Stokes solver.
 *
 *  Provides:
 *  - EulerBCType enumeration of supported boundary condition kinds.
 *  - BoundaryHandler: per-zone BC storage with JSON (de)serialization.
 *  - IntegrationRecorder: MPI-reduced boundary integral accumulator.
 *  - AnchorPointRecorder: MPI_MINLOC anchor-point finder.
 *  - OutputPicker: registry mapping field names to cell-scalar lambdas.
 *  - OneDimProfile: 1-D profile with uniform/tanh nodes, cell-interval
 *    accumulation, MPI reduction, interpolation, and CSV output.
 */
#pragma once
#include "Euler.hpp"
#include "Geom/BoundaryCondition.hpp"
#include "Geom/Grid.hpp"

#include <unordered_map>

#include "DNDS/JsonUtil.hpp"
#include "DNDS/ConfigEnum.hpp"

namespace DNDS::Euler
{
    /**
     * @brief Boundary condition type identifiers for compressible flow solvers.
     *
     * Each enumerator selects the physics applied at a boundary zone:
     * - Far-field and inflow/outflow conditions use characteristic or prescribed states.
     * - Wall conditions enforce no-penetration (inviscid) or no-slip (viscous) constraints.
     * - Special conditions implement test-case-specific setups (Riemann problems, DMR,
     *   isentropic vortex, Rayleigh–Taylor).
     */
    enum EulerBCType
    {
        BCUnknown = 0,          ///< Uninitialized / invalid sentinel.
        BCFar,                  ///< Far-field (characteristic-based) boundary.
        BCWall,                 ///< No-slip viscous wall boundary.
        BCWallInvis,            ///< Inviscid slip wall boundary.
        BCWallIsothermal,       ///< Isothermal wall; requires wall temperature in the BC value vector.
        BCOut,                  ///< Supersonic outflow (extrapolation) boundary.
        BCOutP,                 ///< Back-pressure (subsonic) outflow boundary.
        BCIn,                   ///< Supersonic inflow (fully prescribed state) boundary.
        BCInPsTs,               ///< Subsonic inflow with prescribed total pressure and temperature.
        BCSym,                  ///< Symmetry plane boundary.
        BCSpecial,              ///< Test-case-specific boundary (Riemann, DMR, isentropic vortex, RT).
    };

    DNDS_DEFINE_ENUM_JSON(
        EulerBCType,
        {
            {BCUnknown, nullptr},
            {BCFar, "BCFar"},
            {BCWall, "BCWall"},
            {BCWallIsothermal, "BCWallIsothermal"},
            {BCWallInvis, "BCWallInvis"},
            {BCOut, "BCOut"},
            {BCOutP, "BCOutP"},
            {BCIn, "BCIn"},
            {BCInPsTs, "BCInPsTs"},
            {BCSym, "BCSym"},
            {BCSpecial, "BCSpecial"},
        })

    /**
     * @brief Convert an EulerBCType enumerator to its JSON string representation.
     * @param type  The boundary condition type to convert.
     * @return A human-readable string such as "BCFar", "BCWall", etc.
     */
    inline std::string to_string(EulerBCType type)
    {
        return nlohmann::json(type).get<std::string>();
    }

    /// @brief Build a JSON Schema (draft-07) for the bcSettings array.
    ///
    /// Each BC entry is a discriminated union keyed on `"type"`.  The schema
    /// uses `oneOf` with `"const"` on the discriminator so that IDEs can
    /// provide type-specific autocomplete.
    ///
    /// There are three groups:
    ///   - Flow BCs (BCFar, BCOut, BCOutP, BCIn, BCInPsTs):
    ///       required: type, name, value
    ///       optional: frameOption, anchorOption, integrationOption, valueExtra
    ///   - Wall BCs (BCWall, BCWallInvis, BCWallIsothermal, BCSpecial):
    ///       required: type, name
    ///       optional: value, frameOption, integrationOption, specialOption, valueExtra
    ///       (value is required for BCWallIsothermal)
    ///   - Symmetry (BCSym):
    ///       required: type, name
    ///       optional: rectifyOption, integrationOption, valueExtra
    ///
    /// @return An `ordered_json` object representing the JSON Schema array definition
    ///         suitable for validating a user-supplied boundary condition configuration.
    inline nlohmann::ordered_json bcSettingsSchema()
    {
        using json = nlohmann::ordered_json;

        auto numArray = []() -> json
        {
            json s;
            s["type"] = "array";
            s["items"] = json{{"type", "number"}};
            return s;
        };
        auto intProp = [](const char *desc) -> json
        {
            return json{{"type", "integer"}, {"description", desc}, {"default", 0}};
        };

        // -- Helper: build a single variant schema for one or more BC type strings.
        auto makeVariant = [&](std::vector<std::string> types,
                               const char *groupDesc,
                               json extraProps,
                               std::vector<std::string> required) -> json
        {
            json s;
            s["type"] = "object";
            s["description"] = groupDesc;

            json typeSchema;
            if (types.size() == 1)
                typeSchema = json{{"const", types[0]}};
            else
                typeSchema = json{{"enum", types}};
            typeSchema["description"] = "Boundary condition type";

            json props;
            props["type"] = typeSchema;
            props["name"] = json{{"type", "string"}, {"description", "Boundary zone name"}};
            for (auto &[k, v] : extraProps.items())
                props[k] = v;

            s["properties"] = props;
            s["required"] = required;
            return s;
        };

        // --- Flow BCs: BCFar, BCOut, BCOutP, BCIn, BCInPsTs ---
        auto flowTypes = std::vector<std::string>{"BCFar", "BCOut", "BCOutP", "BCIn", "BCInPsTs"};
        json flowProps;
        flowProps["value"] = numArray();
        flowProps["value"]["description"] = "BC state vector (size = nVars)";
        flowProps["frameOption"] = intProp("Reference frame option");
        flowProps["anchorOption"] = intProp("Anchor point option");
        flowProps["integrationOption"] = intProp("Integration option");
        flowProps["valueExtra"] = numArray();
        flowProps["valueExtra"]["description"] = "Extra BC values (e.g. anchor coordinates)";
        // Emit one variant per type for best IDE discrimination.
        json oneOf = json::array();
        for (auto &t : flowTypes)
        {
            oneOf.push_back(makeVariant({t}, ("Flow BC: " + t).c_str(),
                                        flowProps,
                                        {"type", "name", "value"}));
        }

        // --- Wall BCs: BCWall, BCWallInvis, BCWallIsothermal, BCSpecial ---
        json wallProps;
        wallProps["value"] = numArray();
        wallProps["value"]["description"] = "BC state vector (optional except for BCWallIsothermal)";
        wallProps["frameOption"] = intProp("Reference frame option");
        wallProps["integrationOption"] = intProp("Integration option");
        wallProps["specialOption"] = intProp("Special BC sub-type option");
        wallProps["valueExtra"] = numArray();
        wallProps["valueExtra"]["description"] = "Extra BC values";
        auto wallTypes = std::vector<std::string>{"BCWall", "BCWallInvis", "BCWallIsothermal", "BCSpecial"};
        for (auto &t : wallTypes)
        {
            auto req = std::vector<std::string>{"type", "name"};
            if (t == "BCWallIsothermal")
                req.push_back("value"); // value is required for isothermal
            oneOf.push_back(makeVariant({t}, ("Wall BC: " + t).c_str(),
                                        wallProps, req));
        }

        // --- Symmetry: BCSym ---
        json symProps;
        symProps["rectifyOption"] = intProp("Symmetry plane rectification option");
        symProps["integrationOption"] = intProp("Integration option");
        symProps["valueExtra"] = numArray();
        symProps["valueExtra"]["description"] = "Extra BC values";
        oneOf.push_back(makeVariant({"BCSym"}, "Symmetry BC",
                                    symProps, {"type", "name"}));

        // --- Assemble array schema ---
        json schema;
        schema["type"] = "array";
        schema["description"] = "Boundary condition settings (per-BC array)";
        schema["items"] = json{{"oneOf", oneOf}};
        return schema;
    }

    /**
     * @brief Per-zone boundary condition handler for Euler/Navier-Stokes solvers.
     *
     * Stores the BC type, state vector (value), option flags, and extra values
     * for every boundary zone in the mesh. Zones are identified by name; the
     * internal @c name2ID map translates mesh zone names to sequential BC indices.
     *
     * Default zones (far-field, wall, special test-case walls, etc.) are
     * pre-populated during construction from Geom::GetFaceName2IDDefault().
     *
     * Supports JSON round-trip serialization via ADL `from_json` / `to_json`.
     *
     * @tparam model  The EulerModel tag that determines nVarsFixed and the
     *                conservative variable layout.
     */
    template <EulerModel model>
    class BoundaryHandler
    {
        int nVars; ///< Number of conservative variables (runtime).

    public:
        static const int nVarsFixed = getnVarsFixed(model); ///< Compile-time variable count (or Eigen::Dynamic).
        using TU_R = Eigen::Vector<real, nVarsFixed>;       ///< Fixed-size (read-only) state vector type.
        using TU = Eigen::VectorFMTSafe<real, nVarsFixed>;  ///< fmt-printable state vector type.
        using TFlags = std::map<std::string, uint32_t>;     ///< Per-zone option flag map (key → integer flag).

    private:
        std::vector<TU> BCValues;                                        ///< State vectors for each BC zone.
        std::vector<EulerBCType> BCTypes;                                ///< BC type for each zone.
        std::vector<TFlags> BCFlags;                                     ///< Option flags for each zone.
        std::vector<Eigen::Vector<real, Eigen::Dynamic>> BCValuesExtra;  ///< Extra BC data (e.g. anchor coordinates).
        std::unordered_map<std::string, Geom::t_index> name2ID;         ///< Zone name → internal BC index.
        std::unordered_map<Geom::t_index, std::string> ID2name;         ///< Internal BC index → zone name (reverse map).

    public:
        /**
         * @brief Construct a BoundaryHandler and pre-populate default BC zones.
         *
         * Allocates storage for Geom::BC_ID_DEFAULT_MAX zones with uninitialized
         * values and populates the default zone names and types (far-field, wall,
         * special test-case boundaries) from Geom::GetFaceName2IDDefault().
         *
         * @param _nVars  Number of conservative variables (must match model).
         */
        BoundaryHandler(int _nVars) : nVars(_nVars)
        {
            BCValues.resize(Geom::BC_ID_DEFAULT_MAX);
            for (auto &v : BCValues)
                v.setConstant(UnInitReal);
            BCValuesExtra.resize(Geom::BC_ID_DEFAULT_MAX);
            for (auto &v : BCValuesExtra)
                v.setConstant(UnInitReal);
            BCFlags.resize(Geom::BC_ID_DEFAULT_MAX);
            name2ID = Geom::GetFaceName2IDDefault();
            BCTypes.resize(Geom::BC_ID_DEFAULT_MAX, BCUnknown);
            BCTypes[Geom::BC_ID_DEFAULT_FAR] = BCFar;
            BCTypes[Geom::BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR] = BCSpecial;
            BCTypes[Geom::BC_ID_DEFAULT_SPECIAL_DMR_FAR] = BCSpecial;
            BCTypes[Geom::BC_ID_DEFAULT_SPECIAL_IV_FAR] = BCSpecial;
            BCTypes[Geom::BC_ID_DEFAULT_SPECIAL_RT_FAR] = BCSpecial;
            BCTypes[Geom::BC_ID_DEFAULT_WALL] = BCWall;
            BCTypes[Geom::BC_ID_DEFAULT_WALL_INVIS] = BCWallInvis;
            BCTypes[Geom::BC_ID_NULL] = BCUnknown;
            RenewID2name();
        }

        /**
         * @brief Return the names of all registered BC zones, ordered by internal ID.
         * @return Vector of zone name strings.
         */
        std::vector<std::string> GetAllNames()
        {
            std::vector<std::string> ret;
            for (Geom::t_index i = 0; i < BCTypes.size(); i++)
                if (ID2name.count(i))
                    ret.push_back(ID2name.at(i));
            return ret;
        }

        /// @brief Return the number of registered BC zones (including default slots).
        Geom::t_index size()
        {
            return BCTypes.size();
        }

        /// @brief Rebuild the reverse map (ID2name) from the current name2ID map.
        void RenewID2name()
        {
            ID2name.clear();
            for (auto &p : name2ID)
                ID2name[p.second] = p.first;
        }

        using json = nlohmann::ordered_json;

        /// @brief Push a new BC zone from a JSON sub-object (not yet implemented).
        /// @param gS  JSON object describing the boundary condition.
        void PushBCWithJson(const json &gS)
        {
            // TODO
        }

        /**
         * @brief Deserialize an array of BC entries from JSON into @p bc.
         *
         * Each JSON element must contain at least `"type"` and `"name"`. Additional
         * keys depend on the BC group (flow, wall, symmetry). Validates that the
         * value vector dimension matches @c nVars and that zone names are unique.
         *
         * @param j   JSON array of BC objects.
         * @param bc  Target BoundaryHandler to populate.
         */
        friend void from_json(const json &j, BoundaryHandler<model> &bc)
        {
            DNDS_assert(j.is_array());
            for (auto &item : j)
            {
                EulerBCType bcType = item["type"].get<EulerBCType>();
                std::string bcName = item["name"];
                bc.BCFlags.emplace_back();
                switch (bcType)
                {
                case EulerBCType::BCFar:
                case EulerBCType::BCOut:
                case EulerBCType::BCOutP:
                case EulerBCType::BCIn:
                case EulerBCType::BCInPsTs:
                {
                    uint32_t frameOption = 0;
                    uint32_t anchorOption = 0;
                    uint32_t integrationOption = 0;
                    Eigen::VectorXd bcValueExtra;
                    if (item.count("frameOption"))
                        frameOption = item["frameOption"];
                    if (item.count("anchorOption"))
                        anchorOption = item["anchorOption"];
                    if (item.count("integrationOption"))
                        integrationOption = item["integrationOption"];
                    if (item.count("valueExtra"))
                        bcValueExtra = item["valueExtra"];
                    Eigen::VectorXd bcValue = item["value"];
                    DNDS_assert_info(bcValue.size() == bc.nVars, fmt::format("[{}] bc value dim not right", bcName));
                    bc.BCValues.push_back(bcValue);
                    bc.BCFlags.back()["frameOpt"] = frameOption;
                    bc.BCFlags.back()["anchorOpt"] = anchorOption;
                    bc.BCFlags.back()["integrationOpt"] = integrationOption;
                    bc.BCValuesExtra.push_back(bcValueExtra);
                }
                break;

                case EulerBCType::BCWall:
                case EulerBCType::BCWallIsothermal:
                case EulerBCType::BCWallInvis:
                case EulerBCType::BCSpecial:
                {
                    uint32_t frameOption = 0;
                    uint32_t integrationOption = 0;
                    uint32_t specialOption = 0;
                    Eigen::VectorXd bcValueExtra;
                    Eigen::VectorXd bcValue;
                    if (item.count("frameOption"))
                        frameOption = item["frameOption"];
                    if (item.count("integrationOption"))
                        integrationOption = item["integrationOption"];
                    if (item.count("valueExtra"))
                        bcValueExtra = item["valueExtra"];
                    if (item.count("value"))
                    {
                        bcValue = item["value"];
                        // std::cout << bcValue.transpose() << std::endl;
                        DNDS_assert_info(bcValue.size() == bc.nVars, fmt::format("[{}] bc value dim not right", bcName));
                    }
                    else
                    {
                        bcValue.setZero(bc.nVars);
                        if (bcType == EulerBCType::BCWallIsothermal)
                            DNDS_assert_info(false, "missing bc value for BCWallIsothermal");
                    }
                    if (item.count("specialOption"))
                        specialOption = item["specialOption"];
                    // DNDS_assert(false);
                    bc.BCValues.push_back(bcValue);
                    bc.BCFlags.back()["frameOpt"] = frameOption;
                    bc.BCFlags.back()["integrationOpt"] = integrationOption;
                    bc.BCFlags.back()["specialOpt"] = specialOption;
                    bc.BCValuesExtra.push_back(bcValueExtra);
                }
                break;

                case EulerBCType::BCSym:
                {
                    uint32_t rectifyOption = 0;
                    uint32_t integrationOption = 0;
                    Eigen::VectorXd bcValueExtra;
                    if (item.count("rectifyOption"))
                        rectifyOption = item["rectifyOption"];
                    if (item.count("integrationOption"))
                        integrationOption = item["integrationOption"];
                    if (item.count("valueExtra"))
                        bcValueExtra = item["valueExtra"];
                    bc.BCValues.push_back(TU::Zero(bc.nVars));
                    bc.BCFlags.back()["rectifyOpt"] = rectifyOption;
                    bc.BCFlags.back()["integrationOpt"] = integrationOption;
                    bc.BCValuesExtra.push_back(bcValueExtra);
                }
                break;

                default:
                    DNDS_assert(false);
                    break;
                }

                bc.BCTypes.push_back(bcType);
                DNDS_assert_info(bc.name2ID.count(bcName) == 0, "the bc names are duplicate");
                bc.name2ID[bcName] = bc.BCTypes.size() - 1;

                DNDS_assert(
                    bc.BCFlags.size() == bc.BCTypes.size() &&
                    bc.BCValues.size() == bc.BCTypes.size() &&
                    bc.BCValuesExtra.size() == bc.BCTypes.size());
            }
            bc.RenewID2name();
        }

        /**
         * @brief Serialize user-defined BC zones (those beyond the default slots) to JSON.
         *
         * Zones with index < Geom::BC_ID_DEFAULT_MAX are built-in defaults and are
         * not emitted.  The output is a JSON array suitable for round-trip with from_json().
         *
         * @param j   Output JSON array.
         * @param bc  Source BoundaryHandler to serialize.
         */
        friend void to_json(json &j, const BoundaryHandler<model> &bc)
        {
            j = json::array();
            for (Geom::t_index i = Geom::BC_ID_DEFAULT_MAX; i < bc.BCTypes.size(); i++)
            {
                json item;
                EulerBCType bcType = bc.BCTypes[i];
                item["type"] = bcType;
                item["name"] = bc.ID2name.at(i);
                item["__bcId"] = i; //! TODO: make bcId arbitrary not sequential?
                switch (bcType)
                {
                case EulerBCType::BCFar:
                case EulerBCType::BCOut:
                case EulerBCType::BCOutP:
                case EulerBCType::BCIn:
                case EulerBCType::BCInPsTs:
                {
                    item["value"] = static_cast<TU_R>(bc.BCValues.at(i)); // force begin() and end() to be exposed
                    item["frameOption"] = bc.BCFlags.at(i).at("frameOpt");
                    item["anchorOption"] = bc.BCFlags.at(i).at("anchorOpt");
                    item["valueExtra"] = bc.BCValuesExtra.at(i);
                }
                break;

                case EulerBCType::BCWall:
                case EulerBCType::BCWallIsothermal:
                case EulerBCType::BCWallInvis:
                case EulerBCType::BCSpecial:
                {
                    item["frameOption"] = bc.BCFlags.at(i).at("frameOpt");
                    item["integrationOption"] = bc.BCFlags.at(i).at("integrationOpt");
                    item["specialOption"] = bc.BCFlags.at(i).at("specialOpt");
                    item["valueExtra"] = bc.BCValuesExtra.at(i);
                    item["value"] = static_cast<TU_R>(bc.BCValues.at(i));
                }
                break;

                case EulerBCType::BCSym:
                {
                    item["rectifyOption"] = bc.BCFlags.at(i).at("rectifyOpt");
                    item["valueExtra"] = bc.BCValuesExtra.at(i);
                }
                break;

                default:
                    DNDS_assert(false);
                    break;
                }
                j.push_back(item);
            }
        }

        /**
         * @brief Look up the internal BC index for a given zone name.
         *
         * If periodic, the returned index is negative (by convention).
         *
         * @param name  Mesh zone name string.
         * @return The BC index, or Geom::BC_ID_NULL if the name is not registered.
         */
        Geom::t_index GetIDFromName(const std::string &name)
        {
            if (name2ID.count(name))
                return name2ID[name];
            else
                return Geom::BC_ID_NULL;
        }

        /**
         * @brief Look up the zone name for a given internal BC index.
         * @param id  Internal BC index.
         * @return The zone name, or "UnNamedBC" if the index has no associated name.
         */
        auto GetNameFormID(Geom::t_index id)
        {
            if (!ID2name.count(id))
                return std::string("UnNamedBC");
            return ID2name.at(id);
        }

        /**
         * @brief Retrieve the BC type for a given zone index.
         * @param id  Internal BC index.
         * @return The EulerBCType, or BCUnknown if @p id is not an external BC face.
         */
        EulerBCType GetTypeFromID(Geom::t_index id)
        {
            // std::cout << "id " << std::endl;
            if (!Geom::FaceIDIsExternalBC(id))
                return BCUnknown;
            return BCTypes.at(id);
        }

        /**
         * @brief Retrieve the BC state vector for a given zone index.
         * @param id  Internal BC index.
         * @return The state vector, or the default (index 0) vector if @p id is not an external BC.
         */
        TU GetValueFromID(Geom::t_index id)
        {
            if (!Geom::FaceIDIsExternalBC(id))
                return BCValues.at(0);
            return BCValues.at(id);
        }

        /**
         * @brief Retrieve the extra BC value vector for a given zone index.
         * @param id  Internal BC index.
         * @return The extra value vector (e.g. anchor coordinates), or default if not an external BC.
         */
        Eigen::Vector<real, Eigen::Dynamic> GetValueExtraFromID(Geom::t_index id)
        {
            if (!Geom::FaceIDIsExternalBC(id))
                return BCValuesExtra.at(0);
            return BCValuesExtra.at(id);
        }

        /**
         * @brief Retrieve an option flag for a given zone index (strict).
         *
         * Aborts if @p key is not present in the zone's flag map.
         *
         * @param id   Internal BC index.
         * @param key  Flag key string (e.g. "frameOpt", "anchorOpt").
         * @return The flag value, or 0 if @p id is not an external BC face.
         */
        uint32_t GetFlagFromID(Geom::t_index id, const std::string &key)
        {
            if (!Geom::FaceIDIsExternalBC(id))
                return 0;
            return BCFlags.at(id).at(key);
        }

        /**
         * @brief Retrieve an option flag for a given zone index (lenient).
         *
         * Returns 0 instead of aborting when @p key is absent.
         *
         * @param id   Internal BC index.
         * @param key  Flag key string.
         * @return The flag value, or 0 if the key is missing or @p id is not external.
         */
        uint32_t GetFlagFromIDSoft(Geom::t_index id, const std::string &key)
        {
            if (!Geom::FaceIDIsExternalBC(id) || !BCFlags.at(id).count(key))
                return 0;
            return BCFlags.at(id).at(key);
        }
    };

    /**
     * @brief Accumulator for MPI-reduced boundary-face integrals.
     *
     * Collects integrated quantities (e.g. force components, mass flux)
     * over boundary faces on each MPI rank, then performs an MPI_SUM
     * Allreduce to obtain global totals. The divisor @c div is reduced
     * in parallel and can be used for area-weighted averaging.
     */
    struct IntegrationRecorder
    {
        Eigen::Vector<real, Eigen::Dynamic> v; ///< Accumulated integral vector.
        real div;                               ///< Accumulated divisor (area weight).
        MPIInfo mpi;                            ///< MPI communicator info.

        /**
         * @brief Construct and zero-initialize the integration recorder.
         * @param _nmpi  MPI communicator wrapper.
         * @param siz    Length of the integral vector.
         */
        IntegrationRecorder(const MPIInfo &_nmpi, int siz) : mpi(_nmpi)
        {
            v.resize(siz);
            v.setZero();
            div = verySmallReal;
        }

        /// @brief Reset the integral vector and divisor to zero.
        void Reset()
        {
            v.setZero();
            div = verySmallReal;
        }

        /**
         * @brief Accumulate a contribution from a single boundary face.
         * @tparam TU     Vector type compatible with Eigen addition.
         * @param add     Integral contribution vector to add.
         * @param dAdd    Corresponding divisor (area) contribution.
         */
        template <class TU>
        void Add(TU &&add, real dAdd)
        {
            v += add;
            div += dAdd;
        }

        /// @brief MPI Allreduce (SUM) both the integral vector and the divisor.
        void Reduce()
        {
            // TODO: assure the consistency on different procs?
            Eigen::Vector<real, Eigen::Dynamic> v0 = v;
            MPI::Allreduce(v0.data(), v.data(), v.size(), DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI::AllreduceOneReal(div, MPI_SUM, mpi);
        }
    };

    /**
     * @brief Finds the cell closest to a specified anchor point across all MPI ranks.
     *
     * Each rank calls AddAnchor() with candidate cells; ObtainAnchorMPI() then
     * performs an MPI_MINLOC reduction to identify the globally closest cell and
     * broadcasts its state vector to all ranks. Used for inlet/outlet reference
     * state anchoring.
     *
     * @tparam nVarsFixed  Compile-time number of conservative variables (or Eigen::Dynamic).
     */
    template <int nVarsFixed>
    struct AnchorPointRecorder
    {
        using TU = Eigen::Vector<real, nVarsFixed>; ///< State vector type.
        MPIInfo mpi;                                ///< MPI communicator info.
        TU val;                                     ///< State vector of the closest cell found so far.
        real dist{veryLargeReal};                   ///< Distance to the closest cell found so far.

        /// @brief Construct with MPI communicator.
        /// @param _mpi  MPI communicator wrapper.
        AnchorPointRecorder(const MPIInfo &_mpi) : mpi(_mpi) {}

        /// @brief Reset the recorded distance to veryLargeReal (invalidate current anchor).
        void Reset() { dist = veryLargeReal; }

        /**
         * @brief Submit a candidate anchor cell on this rank.
         *
         * Keeps the candidate only if @p nDist is less than the current best.
         *
         * @param vin    State vector of the candidate cell.
         * @param nDist  Distance from the candidate cell center to the anchor point.
         */
        void AddAnchor(const TU &vin, real nDist)
        {
            if (nDist < dist)
                dist = nDist, val = vin;
        }

        /**
         * @brief Global MPI reduction to find the closest anchor across all ranks.
         *
         * Uses MPI_DOUBLE_INT + MPI_MINLOC to identify the rank holding the
         * minimum distance, then broadcasts that rank's state vector to all ranks.
         */
        void ObtainAnchorMPI()
        {
            struct DI
            {
                double d;
                MPI_int i;
            };
            union Doubleint
            {
                uint8_t pad[16];
                DI dint;
            };
            Doubleint minDist, minDistall;
            minDist.dint.d = dist;
            minDist.dint.i = mpi.rank;
            MPI::Allreduce(&minDist, &minDistall, 1, MPI_DOUBLE_INT, MPI_MINLOC, mpi.comm);
            // std::cout << minDistall.dint.d << std::endl;
            MPI::Bcast(val.data(), val.size(), DNDS_MPI_REAL, minDistall.dint.i, mpi.comm);
        }
    };

    /// @brief Function type returning a scalar value for a given cell index [0, NumCell).
    using tCellScalarFGet = std::function<real(
        index // iCell which is [0, mesh->NumCell)
        )>;
    /// @brief List of (field_name, getter_function) pairs for cell-scalar output.
    using tCellScalarList = std::vector<std::tuple<std::string, const tCellScalarFGet>>;

    /**
     * @brief Registry that maps named cell-scalar output fields to getter lambdas.
     *
     * Used to select which derived quantities (e.g. Mach number, pressure
     * coefficient) are written to output files. Call setMap() to register all
     * available fields, then getSubsetList() to extract a user-requested subset.
     */
    class OutputPicker
    {
    public:
        using tMap = std::map<std::string, tCellScalarFGet>; ///< Name-to-getter mapping type.

    private:
        tMap cellOutRegMap; ///< Registered cell-scalar output fields.

    public:
        /// @brief Register (or replace) the full set of available output fields.
        /// @param v  Map of field names to their getter functions.
        void setMap(const tMap &v)
        {
            cellOutRegMap = v;
        }

        /**
         * @brief Extract a subset of registered output fields by name.
         *
         * Asserts that every requested name exists in the registry.
         *
         * @param names  Ordered list of field names to extract.
         * @return A tCellScalarList with the requested (name, getter) pairs.
         */
        tCellScalarList getSubsetList(const std::vector<std::string> &names)
        {
            tCellScalarList ret;
            for (auto &name : names)
            {
                DNDS_assert(cellOutRegMap.count(name) == 1);
                ret.push_back(std::make_tuple(
                    name,
                    cellOutRegMap[name]));
            }
            return ret;
        }
    };

    /**
     * @brief One-dimensional profile for inlet/outlet boundary data.
     *
     * Discretizes a 1-D coordinate range into cells delimited by @c nodes.
     * Supports uniform and tanh (bilateral clustering) node distributions.
     * Cell values are accumulated via AddSimpleInterval(), reduced across MPI
     * ranks with Reduce(), then queried with Get() or interpolated with
     * GetPlain(). The profile can be written to CSV with OutProfileCSV().
     *
     * The number of cells equals `nodes.size() - 1`.
     *
     * @tparam nVarsFixed  Compile-time number of state variables (or Eigen::Dynamic).
     */
    template <int nVarsFixed>
    struct OneDimProfile
    {
        MPIInfo mpi;                                           ///< MPI communicator info.
        Eigen::Vector<real, Eigen::Dynamic> nodes;             ///< Node coordinates (size = nCells + 1).
        Eigen::Matrix<real, nVarsFixed, Eigen::Dynamic> v;     ///< Accumulated cell values (nVars × nCells).
        Eigen::RowVector<real, Eigen::Dynamic> div;            ///< Per-cell divisor (area weight).

        /// @brief Construct with MPI communicator (no allocation yet).
        /// @param _mpi  MPI communicator wrapper.
        OneDimProfile(const MPIInfo &_mpi) : mpi(_mpi) {}

        /// @brief Sort node coordinates in ascending order.
        void SortNodes()
        {
            std::sort(nodes.begin(), nodes.end());
        }

        /// @brief Return the number of cells (= nodes.size() - 1).
        index Size() const
        {
            return nodes.size() - 1;
        }

        /// @brief Return the length of cell @p i (= nodes[i+1] - nodes[i]).
        /// @param i  Cell index in [0, Size()).
        real Len(index i)
        {
            return nodes[i + 1] - nodes[i];
        }

        /**
         * @brief Allocate a uniformly spaced profile.
         * @param size   Number of cells.
         * @param nvars  Number of state variables.
         * @param minV   Left coordinate bound.
         * @param maxV   Right coordinate bound.
         */
        void GenerateUniform(index size, int nvars, real minV, real maxV)
        {
            nodes.setLinSpaced(size + 1, minV, maxV);
            v.resize(nvars, size);
            div.resize(size);
        }

        /**
         * @brief Allocate a tanh-clustered (bilateral) profile.
         * @param size   Number of cells.
         * @param nvars  Number of state variables.
         * @param minV   Left coordinate bound.
         * @param maxV   Right coordinate bound.
         * @param d0     First cell width controlling clustering intensity.
         */
        void GenerateTanh(index size, int nvars, real minV, real maxV, real d0)
        {
            Geom::GetTanhDistributionBilateral(minV, maxV, size, d0, nodes);
            // if (MPIWorldRank() == 0)
            //     std::cout << std::setprecision(12) << nodes.transpose() << std::endl;
            v.resize(nvars, size);
            div.resize(size);
        }

        /// @brief Zero both the value matrix and the divisor row vector.
        void SetZero()
        {
            v.setZero();
            div.setZero();
        }

        /**
         * @brief Accumulate a cell-mean value over the interval [lV, rV].
         *
         * Distributes @p vmean weighted by @p divV across every profile cell that
         * overlaps [lV, rV], proportional to the overlap fraction.
         *
         * @tparam TU    Vector type compatible with scalar multiplication and Eigen addition.
         * @param vmean  Mean state vector for the contributing interval.
         * @param divV   Divisor (area / weight) for the contributing interval.
         * @param lV     Left coordinate of the contributing interval.
         * @param rV     Right coordinate of the contributing interval.
         */
        template <class TU>
        void AddSimpleInterval(TU vmean, real divV, real lV, real rV)
        {
            index iCL = (std::lower_bound(nodes.begin(), nodes.end(), lV) - nodes.begin()) - index(1);
            iCL = std::min(Size() - 1, std::max(index(0), iCL));
            index iCR = std::upper_bound(nodes.begin(), nodes.end(), rV) - nodes.begin(); // max is Size() + 1
            iCR = std::min(Size(), iCR);
            // std::cout << iCL << " " << iCR << " " << lV << " " << rV << std::endl;
            for (index i = iCL; i < iCR; i++)
            {
                real cL = nodes[i];
                real cR = nodes[i + 1];
                real cinIntervalL = std::max(lV, cL);
                real cinInvervalR = std::min(rV, cR);
                real cinInvervalLenRel = std::max(cinInvervalR - cinIntervalL, 0.0) / (rV - lV);
                div[i] += divV * cinInvervalLenRel;
                v(EigenAll, i) += vmean * divV * cinInvervalLenRel;
            }
        }

        /// @brief MPI Allreduce (SUM) the value matrix and divisor across all ranks.
        void Reduce()
        {
            // TODO: assure the consistency on different procs?
            Eigen::RowVector<real, Eigen::Dynamic> div0 = div;
            Eigen::Matrix<real, nVarsFixed, Eigen::Dynamic> v0 = v;
            MPI::Allreduce(div0.data(), div.data(), div.size(), DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI::Allreduce(v0.data(), v.data(), v.size(), DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        }

        /**
         * @brief Retrieve the averaged value for cell @p i.
         *
         * Returns `v(:, i) / div(i)`, with a tiny epsilon to avoid division by zero.
         *
         * @param i  Cell index in [0, Size()).
         * @return The area-weighted average state vector for cell @p i.
         */
        Eigen::Vector<real, nVarsFixed> Get(index i) const
        {
            DNDS_assert(i < Size());
            return v(EigenAll, i) / (div(i) + verySmallReal);
        }

        /**
         * @brief Linearly interpolate the profile at coordinate @p v.
         *
         * Locates the cell containing @p v, then blends the averaged values of that
         * cell and its neighbors to produce a smooth interpolation.
         *
         * @param v  Coordinate at which to evaluate the profile.
         * @return Interpolated state vector.
         */
        Eigen::Vector<real, nVarsFixed> GetPlain(real v) const
        {
            index iCL = (std::lower_bound(nodes.begin(), nodes.end(), v) - nodes.begin()) - index(1);
            iCL = std::min(Size() - 1, std::max(index(0), iCL));
            real vL = nodes[iCL];
            real vR = nodes[iCL + 1];
            real vRel = (v - vL) / (vR - vL + verySmallReal);
            vRel = std::min(vRel, 1.);
            vRel = std::max(vRel, 0.);
            Eigen::Vector<real, nVarsFixed> valL = Get(std::max(iCL - 1, index(0)));
            Eigen::Vector<real, nVarsFixed> valR = Get(std::min(iCL + 1, Size() - 1));
            valL = 0.5 * (valL + Get(iCL));
            valR = 0.5 * (valR + Get(iCL));
            return vRel * valR + (1 - vRel) * valL;
        }

        /**
         * @brief Write the profile to a CSV stream after a valid MPI reduction.
         *
         * Should be called on a single MPI rank. Outputs one row per node
         * coordinate with interpolated values.
         *
         * @param o          Output stream to write CSV data to.
         * @param title      If true, write a CSV header row first.
         * @param precision  Floating-point precision for std::scientific output.
         */
        void OutProfileCSV(std::ostream &o, bool title = true, int precision = 16)
        {
            if (title)
            {
                o << "X";
                for (index i = 0; i < v.rows(); i++)
                    o << ", U" << std::to_string(i);
                o << "\n";
            }
            o << std::scientific << std::setprecision(precision);
            for (index iN = 0; iN < v.cols(); iN++)
            {
                o << nodes[iN];
                auto val = GetPlain(nodes[iN]);
                for (index i = 0; i < val.size(); i++)
                    o << ", " << val[i];
                o << "\n";
            }
        }
    };

}