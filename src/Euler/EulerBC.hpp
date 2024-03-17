#pragma once
#include "Euler.hpp"
#include "Geom/BoundaryCondition.hpp"

#include <unordered_map>

#include "json.hpp"

namespace DNDS::Euler
{
    enum EulerBCType
    {
        BCUnknown = 0,
        BCFar,
        BCWall,
        BCWallInvis,
        BCOut,
        BCOutPs,
        BCIn,
        BCInPsTs,
        BCSym,
        BCSpecial,
    };

    NLOHMANN_JSON_SERIALIZE_ENUM(
        EulerBCType,
        {
            {BCUnknown, nullptr},
            {BCFar, "BCFar"},
            {BCWall, "BCWall"},
            {BCWallInvis, "BCWallInvis"},
            {BCOut, "BCOut"},
            {BCOutPs, "BCOutPs"},
            {BCIn, "BCIn"},
            {BCInPsTs, "BCInPsTs"},
            {BCSym, "BCSym"},
            {BCSpecial, "BCSpecial"},
        });

    template <EulerModel model>
    class BoundaryHandler
    {
        int nVars;

    public:
        static const int nVarsFixed = getnVarsFixed(model);
        using TU_R = Eigen::Vector<real, nVarsFixed>;
        using TU = Eigen::VectorFMTSafe<real, nVarsFixed>;
        using TFlags = std::map<std::string, uint32_t>;

    private:
        std::vector<TU> BCValues;
        std::vector<EulerBCType> BCTypes;
        std::vector<TFlags> BCFlags;
        std::unordered_map<std::string, Geom::t_index> name2ID;
        std::unordered_map<Geom::t_index, std::string> ID2name;

    public:
        BoundaryHandler(int _nVars) : nVars(_nVars)
        {
            BCValues.resize(Geom::BC_ID_DEFAULT_MAX);
            for (auto &v : BCValues)
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

        void RenewID2name()
        {
            ID2name.clear();
            for (auto &p : name2ID)
                ID2name[p.second] = p.first;
        }

        using json = nlohmann::ordered_json;
        void PushBCWithJson(const json &gS)
        {
            // TODO
        }

        friend void from_json(const json &j, BoundaryHandler<model> &bc)
        {
            DNDS_assert(j.is_array());
            for (auto &item : j)
            {
                EulerBCType bcType = item["type"].get<EulerBCType>();
                std::string bcName = item["name"];
                switch (bcType)
                {
                case EulerBCType::BCFar:
                case EulerBCType::BCOut:
                case EulerBCType::BCOutPs:
                case EulerBCType::BCIn:
                case EulerBCType::BCInPsTs:
                {
                    uint32_t frameOption = 0;
                    if (item.count("frameOption"))
                        frameOption = item["frameOption"];
                    Eigen::VectorXd bcValue = item["value"];
                    DNDS_assert_info(bcValue.size() == bc.nVars, "bc value dim not right");
                    bc.BCValues.push_back(bcValue);
                    bc.BCFlags.emplace_back(TFlags{});
                    bc.BCFlags.back()["frameOpt"] = frameOption;
                }
                break;

                case EulerBCType::BCWall:
                case EulerBCType::BCWallInvis:
                {
                    uint32_t frameOption = 0;
                    if (item.count("frameOption"))
                        frameOption = item["frameOption"];
                    bc.BCValues.push_back(TU::Zero(bc.nVars));
                    bc.BCFlags.emplace_back(TFlags{});
                    bc.BCFlags.back()["frameOpt"] = frameOption;
                }
                break;

                case EulerBCType::BCSym:
                {
                    uint32_t rectifyOption = 0;
                    if (item.count("rectifyOption"))
                        rectifyOption = item["rectifyOption"];
                    bc.BCValues.push_back(TU::Zero(bc.nVars));
                    bc.BCFlags.emplace_back(TFlags{});
                    bc.BCFlags.back()["rectifyOpt"] = rectifyOption;
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
                    bc.BCValues.size() == bc.BCTypes.size());
            }
            bc.RenewID2name();
        }

        friend void to_json(json &j, const BoundaryHandler<model> &bc)
        {
            j = json::array();
            for (Geom::t_index i = Geom::BC_ID_DEFAULT_MAX; i < bc.BCTypes.size(); i++)
            {
                json item;
                EulerBCType bcType = bc.BCTypes[i];
                item["type"] = bcType;
                item["name"] = bc.ID2name.at(i);
                switch (bcType)
                {
                case EulerBCType::BCFar:
                case EulerBCType::BCOut:
                case EulerBCType::BCOutPs:
                case EulerBCType::BCIn:
                case EulerBCType::BCInPsTs:
                {
                    item["value"] = static_cast<TU_R>(bc.BCValues.at(i)); // force begin() and end() to be exposed
                    item["frameOption"] = bc.BCFlags.at(i).at("frameOpt");
                }
                break;

                case EulerBCType::BCWall:
                case EulerBCType::BCWallInvis:
                {
                    item["frameOption"] = bc.BCFlags.at(i).at("frameOpt");
                }
                break;

                case EulerBCType::BCSym:
                {
                    item["rectifyOption"] = bc.BCFlags.at(i).at("rectifyOpt");
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
         * if periodic, return with minus
         */
        Geom::t_index GetIDFromName(const std::string &name)
        {
            if (name2ID.count(name))
                return name2ID[name];
            else
                return Geom::BC_ID_NULL;
        }

        EulerBCType GetTypeFromID(Geom::t_index id)
        {
            // std::cout << "id " << std::endl;
            if (!Geom::FaceIDIsExternalBC(id))
                return BCUnknown;
            return BCTypes.at(id);
        }

        TU GetValueFromID(Geom::t_index id)
        {
            if (!Geom::FaceIDIsExternalBC(id))
                return BCValues.at(0);
            return BCValues.at(id);
        }

        uint32_t GetFlagFromID(Geom::t_index id, const std::string &key)
        {
            if (!Geom::FaceIDIsExternalBC(id))
                return 0;
            return BCFlags.at(id).at(key);
        }
    };

    using tCellScalarFGet = std::function<real(
        index // iCell which is [0, mesh->NumCell)
        )>;
    using tCellScalarList = std::vector<std::tuple<std::string, const tCellScalarFGet>>;

    class OutputPicker
    {
    public:
        using tMap = std::map<std::string, tCellScalarFGet>;

    private:
        tMap cellOutRegMap;

    public:
        void setMap(const tMap &v)
        {
            cellOutRegMap = v;
        }

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

}