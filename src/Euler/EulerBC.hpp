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
        BCIn,
        BCSpecial,
    };

    NLOHMANN_JSON_SERIALIZE_ENUM(
        EulerBCType,
        {
            {BCUnknown, nullptr},
            {BCFar, "BCFar"},
            {BCWall, "BCWall"},
            {BCWallInvis, "BCWallInvis"},
            {BCSpecial, "BCSpecial"},
            {BCOut, "BCOut"},
            {BCIn, "BCIn"},
        });

    template <EulerModel model>
    class BoundaryHandler
    {
    public:
        static const int nVarsFixed = getnVarsFixed(model);
        using TU_R = Eigen::Vector<real, nVarsFixed>;
        using TU = Eigen::VectorFMTSafe<real, nVarsFixed>;

    private:
        std::vector<TU> BCValues;
        std::vector<EulerBCType> BCTypes;
        std::unordered_map<std::string, Geom::t_index> name2ID;
        std::unordered_map<Geom::t_index, std::string> ID2name;

    public:
        BoundaryHandler()
        {
            BCValues.resize(Geom::BC_ID_DEFAULT_MAX);
            for (auto &v : BCValues)
                v.setConstant(UnInitReal);
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
                switch (bcType)
                {
                case EulerBCType::BCFar:
                case EulerBCType::BCWall:
                case EulerBCType::BCWallInvis:
                case EulerBCType::BCOut:
                case EulerBCType::BCIn:
                {
                    std::string bcName = item["name"];
                    Eigen::VectorXd bcValue = item["value"];

                    bc.BCTypes.push_back(bcType);
                    bc.BCValues.push_back(bcValue);
                    DNDS_assert_info(bc.name2ID.count(bcName) == 0, "the bc names are duplicate");
                    bc.name2ID[bcName] = bc.BCValues.size() - 1;
                }
                break;

                default:
                    DNDS_assert(false);
                    break;
                }
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
                switch (bcType)
                {
                case EulerBCType::BCFar:
                case EulerBCType::BCWall:
                case EulerBCType::BCWallInvis:
                case EulerBCType::BCOut:
                case EulerBCType::BCIn:
                {
                    item["type"] = bcType;
                    item["name"] = bc.ID2name.at(i);
                    item["value"] = static_cast<TU_R>(bc.BCValues.at(i)); // force begin() and end() to be exposed
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