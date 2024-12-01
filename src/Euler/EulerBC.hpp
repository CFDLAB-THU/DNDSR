#pragma once
#include "Euler.hpp"
#include "Geom/BoundaryCondition.hpp"
#include "Geom/Grid.hpp"

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
        BCOutP,
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
            {BCOutP, "BCOutP"},
            {BCIn, "BCIn"},
            {BCInPsTs, "BCInPsTs"},
            {BCSym, "BCSym"},
            {BCSpecial, "BCSpecial"},
        });

    inline std::string to_string(EulerBCType type)
    {
        return nlohmann::json(type).get<std::string>();
    }

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
        std::vector<Eigen::Vector<real, Eigen::Dynamic>> BCValuesExtra;
        std::unordered_map<std::string, Geom::t_index> name2ID;
        std::unordered_map<Geom::t_index, std::string> ID2name;

    public:
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

        Geom::t_index size()
        {
            return BCTypes.size();
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
                bc.BCFlags.emplace_back(TFlags{});
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
                    DNDS_assert_info(bcValue.size() == bc.nVars, "bc value dim not right");
                    bc.BCValues.push_back(bcValue);
                    bc.BCFlags.back()["frameOpt"] = frameOption;
                    bc.BCFlags.back()["anchorOpt"] = anchorOption;
                    bc.BCFlags.back()["integrationOpt"] = integrationOption;
                    bc.BCValuesExtra.push_back(bcValueExtra);
                }
                break;

                case EulerBCType::BCWall:
                case EulerBCType::BCWallInvis:
                case EulerBCType::BCSpecial:
                {
                    uint32_t frameOption = 0;
                    uint32_t integrationOption = 0;
                    uint32_t specialOption = 0;
                    Eigen::VectorXd bcValueExtra;
                    if (item.count("frameOption"))
                        frameOption = item["frameOption"];
                    if (item.count("integrationOption"))
                        integrationOption = item["integrationOption"];
                    if (item.count("valueExtra"))
                        bcValueExtra = item["valueExtra"];
                    if (item.count("specialOption"))
                        specialOption = item["specialOption"];
                    bc.BCValues.push_back(TU::Zero(bc.nVars));
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
                case EulerBCType::BCWallInvis:
                case EulerBCType::BCSpecial:
                {
                    item["frameOption"] = bc.BCFlags.at(i).at("frameOpt");
                    item["integrationOption"] = bc.BCFlags.at(i).at("integrationOpt");
                    item["specialOption"] = bc.BCFlags.at(i).at("specialOpt");
                    item["valueExtra"] = bc.BCValuesExtra.at(i);
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
         * if periodic, return with minus
         */
        Geom::t_index GetIDFromName(const std::string &name)
        {
            if (name2ID.count(name))
                return name2ID[name];
            else
                return Geom::BC_ID_NULL;
        }

        auto GetNameFormID(Geom::t_index id)
        {
            if (!ID2name.count(id))
                return std::string("UnNamedBC");
            return ID2name.at(id);
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

        Eigen::Vector<real, Eigen::Dynamic> GetValueExtraFromID(Geom::t_index id)
        {
            if (!Geom::FaceIDIsExternalBC(id))
                return BCValuesExtra.at(0);
            return BCValuesExtra.at(id);
        }

        uint32_t GetFlagFromID(Geom::t_index id, const std::string &key)
        {
            if (!Geom::FaceIDIsExternalBC(id))
                return 0;
            return BCFlags.at(id).at(key);
        }

        uint32_t GetFlagFromIDSoft(Geom::t_index id, const std::string &key)
        {
            if (!Geom::FaceIDIsExternalBC(id) || !BCFlags.at(id).count(key))
                return 0;
            return BCFlags.at(id).at(key);
        }
    };

    struct IntegrationRecorder
    {
        Eigen::Vector<real, Eigen::Dynamic> v;
        real div;
        MPIInfo mpi;

        IntegrationRecorder(const MPIInfo &_nmpi, int siz) : mpi(_nmpi)
        {
            v.resize(siz);
            v.setZero();
            div = verySmallReal;
        }

        void Reset()
        {
            v.setZero();
            div = verySmallReal;
        }

        template <class TU>
        void Add(TU &&add, real dAdd)
        {
            v += add;
            div += dAdd;
        }

        void Reduce()
        {
            // TODO: assure the consistency on different procs?
            Eigen::Vector<real, Eigen::Dynamic> v0 = v;
            MPI::Allreduce(v0.data(), v.data(), v.size(), DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI::AllreduceOneReal(div, MPI_SUM, mpi);
        }
    };

    template <int nVarsFixed>
    struct AnchorPointRecorder
    {
        using TU = Eigen::Vector<real, nVarsFixed>;
        MPIInfo mpi;
        TU val;
        real dist{veryLargeReal};
        AnchorPointRecorder(const MPIInfo &_mpi) : mpi(_mpi) {}

        void Reset() { dist = veryLargeReal; }

        void AddAnchor(const TU &vin, real nDist)
        {
            if (nDist < dist)
                dist = nDist, val = vin;
        }

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

    template <int nVarsFixed>
    struct OneDimProfile
    {
        MPIInfo mpi;
        Eigen::Vector<real, Eigen::Dynamic> nodes;
        Eigen::Matrix<real, nVarsFixed, Eigen::Dynamic> v;
        Eigen::RowVector<real, Eigen::Dynamic> div;

        OneDimProfile(const MPIInfo &_mpi) : mpi(_mpi) {}

        void SortNodes()
        {
            std::sort(nodes.begin(), nodes.end());
        }

        index Size() const
        {
            return nodes.size() - 1;
        }

        real Len(index i)
        {
            return nodes[i + 1] - nodes[i];
        }

        void GenerateUniform(index size, int nvars, real minV, real maxV)
        {
            nodes.setLinSpaced(size + 1, minV, maxV);
            v.resize(nvars, size);
            div.resize(size);
        }

        void GenerateTanh(index size, int nvars, real minV, real maxV, real d0)
        {
            Geom::GetTanhDistributionBilateral(minV, maxV, size, d0, nodes);
            // if (MPIWorldRank() == 0)
            //     std::cout << std::setprecision(12) << nodes.transpose() << std::endl;
            v.resize(nvars, size);
            div.resize(size);
        }

        void SetZero()
        {
            v.setZero();
            div.setZero();
        }

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
                v(Eigen::all, i) += vmean * divV * cinInvervalLenRel;
            }
        }

        void Reduce()
        {
            // TODO: assure the consistency on different procs?
            Eigen::RowVector<real, Eigen::Dynamic> div0 = div;
            Eigen::Matrix<real, nVarsFixed, Eigen::Dynamic> v0 = v;
            MPI::Allreduce(div0.data(), div.data(), div.size(), DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI::Allreduce(v0.data(), v.data(), v.size(), DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        }

        Eigen::Vector<real, nVarsFixed> Get(index i) const
        {
            DNDS_assert(i < Size());
            return v(Eigen::all, i) / (div(i) + verySmallReal);
        }

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
         * \brief after valid reduction
         * called on one proc
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