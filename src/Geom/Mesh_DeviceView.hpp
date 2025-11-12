#pragma once
#include "Elements.hpp"
#include "DNDS/Array.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"
#include "BoundaryCondition.hpp"
#include "DNDS/ArrayPair.hpp"
#include "PeriodicInfo.hpp"
#include "RadialBasisFunction.hpp"
#include "Solver/Direct.hpp"
#include "DNDS/ObjectUtils.hpp"
#include "DNDS/DeviceStorage.hpp"

namespace DNDS::Geom
{
    static const t_index INTERNAL_ZONE = -1;
    struct ElemInfo
    {
        t_index type = static_cast<t_index>(Elem::UnknownElem);
        /// @brief positive for BVnum, 0 for internal Elems, Negative for ?
        t_index zone = INTERNAL_ZONE;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ElemInfo, ElemInfo)

        DNDS_DEVICE_CALLABLE [[nodiscard]] Elem::ElemType getElemType() const
        {
            return static_cast<Elem::ElemType>(type);
        }

        DNDS_DEVICE_CALLABLE void setElemType(Elem::ElemType t)
        {
            type = static_cast<t_index>(t);
        }

        // bool ZoneIsInternal()
        // {
        //     return zone == INTERNAL_ZONE;
        // }
        // bool ZoneIsIndexed()
        // {
        //     return zone >= 0;
        // }

        static MPI_Datatype CommType()
        {
            static_assert(sizeof(ElemInfo) <= (4ULL * 2));
            return MPI_INT32_T;
        }
        static int CommMult() { return 2; }
        static std::string pybind11_name() { return "ElemInfo"; }
    };

}
namespace DNDS
{
    DNDS_DEVICE_STORAGE_BASE_DELETER_INST(Geom::ElemInfo, extern)
    DNDS_DEVICE_STORAGE_INST(Geom::ElemInfo, DeviceBackend::Host, extern)
#ifdef DNDS_USE_CUDA
    DNDS_DEVICE_STORAGE_INST(Geom::ElemInfo, DeviceBackend::CUDA, extern)
#endif
}
namespace DNDS::Geom
{

    using tAdjPair = DNDS::ArrayAdjacencyPair<DNDS::NonUniformSize>;
    using tAdj = decltype(tAdjPair::father);
    using tPbiPair = ArrayPair<ArrayNodePeriodicBits<DNDS::NonUniformSize>>;
    using tPbi = decltype(tPbiPair::father);
    using tAdj1Pair = DNDS::ArrayAdjacencyPair<1>;
    using tAdj1 = decltype(tAdj1Pair::father);
    using tAdj2Pair = DNDS::ArrayAdjacencyPair<2>;
    using tAdj2 = decltype(tAdj2Pair::father);
    using tAdj3Pair = DNDS::ArrayAdjacencyPair<3>;
    using tAdj3 = decltype(tAdj3Pair::father);
    using tAdj4Pair = DNDS::ArrayAdjacencyPair<4>;
    using tAdj4 = decltype(tAdj4Pair::father);
    using tAdj8Pair = DNDS::ArrayAdjacencyPair<8>;
    using tAdj8 = decltype(tAdj8Pair::father);
    using tCoordPair = DNDS::ArrayPair<DNDS::ArrayEigenVector<3>>;
    using tCoord = decltype(tCoordPair::father);
    using tElemInfoArrayPair = DNDS::ArrayPair<DNDS::ParArray<ElemInfo>>;
    using tElemInfoArray = DNDS::ssp<DNDS::ParArray<ElemInfo>>;
    using tIndPair = DNDS::ArrayPair<DNDS::ArrayIndex>;
    using tInd = decltype(tIndPair::father);

    using tFGetName = std::function<std::string(int)>;
    using tFGetData = std::function<DNDS::real(int, DNDS::index)>;
    using tFGetVecData = std::function<DNDS::real(int, DNDS::index, DNDS::rowsize)>;

    enum MeshAdjState
    {
        Adj_Unknown = 0,
        Adj_PointToLocal,
        Adj_PointToGlobal,
    };

    enum MeshElevationState
    {
        Elevation_Untouched = 0,
        Elevation_O1O2,
    };

#define DNDS_COPY_MEMBER_VIEW(obj, member) \
    member = (obj).member.template deviceView<B>();
#define DNDS_COPY_MEMBER(obj, member) \
    member = (obj).member;

    template <DeviceBackend B>
    struct UnstructuredMeshDeviceView
    {
        int dim = -1;
        bool isPeriodic{false};
        MeshAdjState adjPrimaryState{Adj_Unknown};
        // state of: face2cell, face2node
        MeshAdjState adjFacialState{Adj_Unknown};
        // state of: cell2face
        MeshAdjState adjC2FState{Adj_Unknown};
        // state of: node2cell, node2bnd
        MeshAdjState adjN2CBState{Adj_Unknown};
        // state of: cell2cellFace
        // MeshAdjState adjC2CFaceState{Adj_Unknown};

        Periodicity periodicInfo;

        /// reader
        tCoordPair::t_deviceView<B> coords;
        tAdjPair::t_deviceView<B> cell2node;
        tAdjPair::t_deviceView<B> bnd2node;
        tAdj2Pair::t_deviceView<B> bnd2cell;
        tAdjPair::t_deviceView<B> cell2cell;
        tElemInfoArrayPair::t_deviceView<B> cellElemInfo;
        tElemInfoArrayPair::t_deviceView<B> bndElemInfo;
        // tAdj1Pair::t_deviceView<B> cell2cellOrig; // no device
        // tAdj1Pair::t_deviceView<B> node2nodeOrig; // no device
        // tAdj1Pair::t_deviceView<B> bnd2bndOrig; // no device
        /// periodic only, after reader
        tPbiPair::t_deviceView<B> cell2nodePbi;
        tPbiPair::t_deviceView<B> bnd2nodePbi;

        auto device_array_list_primary()
        {
            return std::make_tuple(
                DNDS_MAKE_1_MEMBER_REF(coords),
                DNDS_MAKE_1_MEMBER_REF(cell2node),
                DNDS_MAKE_1_MEMBER_REF(bnd2node),
                DNDS_MAKE_1_MEMBER_REF(bnd2cell),
                DNDS_MAKE_1_MEMBER_REF(cell2cell),
                DNDS_MAKE_1_MEMBER_REF(cellElemInfo),
                DNDS_MAKE_1_MEMBER_REF(bndElemInfo),
                DNDS_MAKE_1_MEMBER_REF(cell2nodePbi),
                DNDS_MAKE_1_MEMBER_REF(bnd2nodePbi));
        }

        template <class TMain>
        void create_view_primary(TMain &&m_obj)
        {
            DNDS_COPY_MEMBER_VIEW(m_obj, coords);
            DNDS_COPY_MEMBER_VIEW(m_obj, cell2node);
            DNDS_COPY_MEMBER_VIEW(m_obj, bnd2node);
            DNDS_COPY_MEMBER_VIEW(m_obj, bnd2cell);
            DNDS_COPY_MEMBER_VIEW(m_obj, cell2cell);
            DNDS_COPY_MEMBER_VIEW(m_obj, cellElemInfo);
            DNDS_COPY_MEMBER_VIEW(m_obj, bndElemInfo);
            if (isPeriodic)
            {
                DNDS_COPY_MEMBER_VIEW(m_obj, cell2nodePbi);
                DNDS_COPY_MEMBER_VIEW(m_obj, bnd2nodePbi);
            }
        }

        tAdjPair::t_deviceView<B> node2cell;
        tAdjPair::t_deviceView<B> node2bnd;

        auto device_array_list_N2CB()
        {
            return std::make_tuple(
                DNDS_MAKE_1_MEMBER_REF(node2cell),
                DNDS_MAKE_1_MEMBER_REF(node2bnd));
        }

        template <class TMain>
        void create_view_N2CB(TMain &&m_obj)
        {
            DNDS_COPY_MEMBER_VIEW(m_obj, node2cell);
            DNDS_COPY_MEMBER_VIEW(m_obj, node2bnd);
        }

        /// interpolated
        tAdjPair::t_deviceView<B> cell2face;
        tAdjPair::t_deviceView<B> face2node;
        tAdj2Pair::t_deviceView<B> face2cell;
        tElemInfoArrayPair::t_deviceView<B> faceElemInfo;
        // std::vector<index> bnd2face; // no device
        // std::unordered_map<index, index> face2bnd; // no device
        /// periodic only, after interpolated
        tPbiPair::t_deviceView<B> face2nodePbi;

        auto device_array_list_facial()
        {
            return std::make_tuple(
                DNDS_MAKE_1_MEMBER_REF(face2cell),
                DNDS_MAKE_1_MEMBER_REF(face2node),
                DNDS_MAKE_1_MEMBER_REF(face2nodePbi),
                DNDS_MAKE_1_MEMBER_REF(faceElemInfo));
        }

        template <class TMain>
        void create_view_facial(TMain &&m_obj)
        {
            DNDS_COPY_MEMBER_VIEW(m_obj, face2cell);
            DNDS_COPY_MEMBER_VIEW(m_obj, face2node);
            if (isPeriodic)
                DNDS_COPY_MEMBER_VIEW(m_obj, face2nodePbi);
            DNDS_COPY_MEMBER_VIEW(m_obj, faceElemInfo);
        }

        auto device_array_list_C2F()
        {
            return std::make_tuple(
                DNDS_MAKE_1_MEMBER_REF(cell2face));
        }

        template <class TMain>
        void create_view_C2F(TMain &&m_obj)
        {
            DNDS_COPY_MEMBER_VIEW(m_obj, cell2face);
        }

        template <class TMain>
        DNDS_DEVICE_CALLABLE UnstructuredMeshDeviceView(TMain &mesh, index placeholder)
        {
            DNDS_assert(placeholder == UnInitIndex);

            DNDS_COPY_MEMBER(mesh, dim);
            DNDS_COPY_MEMBER(mesh, isPeriodic); //! this is needed after
            DNDS_COPY_MEMBER(mesh, adjPrimaryState);
            DNDS_COPY_MEMBER(mesh, adjFacialState);
            DNDS_COPY_MEMBER(mesh, adjC2FState);
            DNDS_COPY_MEMBER(mesh, adjN2CBState);
            // DNDS_COPY_MEMBER(mesh, adjC2CFaceState);

            if (adjPrimaryState)
                create_view_primary(mesh);
            if (adjFacialState)
                create_view_facial(mesh);
            if (adjC2FState)
                create_view_C2F(mesh);
        }

        DNDS_DEVICE_CALLABLE index NumNode() const { return coords.father.Size(); }
        DNDS_DEVICE_CALLABLE index NumCell() const { return cell2node.father.Size(); }
        DNDS_DEVICE_CALLABLE index NumFace() const { return face2node.father.Size(); }
        DNDS_DEVICE_CALLABLE index NumBnd() const { return bnd2node.father.Size(); }

        DNDS_DEVICE_CALLABLE index NumNodeGhost() const { return coords.son.Size(); }
        DNDS_DEVICE_CALLABLE index NumCellGhost() const { return cell2node.son.Size(); }
        DNDS_DEVICE_CALLABLE index NumFaceGhost() const { return face2node.son.Size(); }
        DNDS_DEVICE_CALLABLE index NumBndGhost() const { return bnd2node.son.Size(); }

        DNDS_DEVICE_CALLABLE index NumNodeProc() const { return coords.Size(); }
        DNDS_DEVICE_CALLABLE index NumCellProc() const { return cell2node.Size(); }
        DNDS_DEVICE_CALLABLE index NumFaceProc() const { return face2node.Size(); }
        DNDS_DEVICE_CALLABLE index NumBndProc() const { return bnd2node.Size(); }

        DNDS_DEVICE_CALLABLE Elem::Element GetCellElement(index iC) { return Elem::Element{cellElemInfo(iC, 0).getElemType()}; }
        DNDS_DEVICE_CALLABLE Elem::Element GetFaceElement(index iF) { return Elem::Element{faceElemInfo(iF, 0).getElemType()}; }
        DNDS_DEVICE_CALLABLE Elem::Element GetBndElement(index iB) { return Elem::Element{bndElemInfo(iB, 0).getElemType()}; }

        DNDS_DEVICE_CALLABLE t_index GetCellZone(index iC) { return cellElemInfo(iC, 0).zone; }
        DNDS_DEVICE_CALLABLE t_index GetFaceZone(index iF) { return faceElemInfo(iF, 0).zone; }
        DNDS_DEVICE_CALLABLE t_index GetBndZone(index iB) { return bndElemInfo(iB, 0).zone; }

        /**
         * @brief fA executes when if2c points to the donor side; fB the main side
         *
         * @tparam FA
         * @tparam FB
         * @tparam F0
         * @param iFace
         * @param if2c
         * @param fA
         * @param fB
         * @param f0
         * @return auto
         */
        template <class FA, class FB, class F0>
        DNDS_DEVICE_CALLABLE auto CellOtherCellPeriodicHandle(
            index iFace, rowsize if2c, FA &&fA, FB &&fB, F0 &&f0 = []() {})
        {
            if (!this->isPeriodic)
                return f0();
            auto faceID = this->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return f0();
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return fA();
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return fB();
        }

        /**
         * @brief directly load coords; gets faulty if isPeriodic!
         */
        template <class tC2n>
        DNDS_DEVICE_CALLABLE void __GetCoords(const tC2n &c2n, tSmallCoords &cs)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                DNDS_HD_assert(adjPrimaryState == Adj_PointToLocal);
                cs(EigenAll, i) = coords[iNode];
            }
        }

        /**
         * @brief directly load coords; gets faulty if isPeriodic!
         */
        template <class tC2n, class tCoordExt>
        DNDS_DEVICE_CALLABLE void __GetCoords(const tC2n &c2n, tSmallCoords &cs, tCoordExt &coo)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                DNDS_HD_assert(adjPrimaryState == Adj_PointToLocal);
                cs(EigenAll, i) = coo[iNode];
            }
        }

        /**
         * @brief specially for periodicity
         */
        template <class tC2n, class tC2nPbi>
        DNDS_DEVICE_CALLABLE void __GetCoordsOnElem(const tC2n &c2n, const tC2nPbi &c2nPbi, tSmallCoords &cs)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                DNDS_HD_assert(adjPrimaryState == Adj_PointToLocal);
                cs(EigenAll, i) = periodicInfo.GetCoordByBits(coords[iNode], c2nPbi[i]);
            }
        }

        /**
         * @brief specially for periodicity
         */
        template <class tC2n, class tC2nPbi, class tCoordExt>
        DNDS_DEVICE_CALLABLE void __GetCoordsOnElem(const tC2n &c2n, const tC2nPbi &c2nPbi, tSmallCoords &cs, tCoordExt &coo)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                DNDS_HD_assert(adjPrimaryState == Adj_PointToLocal);
                cs(EigenAll, i) = periodicInfo.GetCoordByBits(coo[iNode], c2nPbi[i]);
            }
        }

        DNDS_DEVICE_CALLABLE void GetCoordsOnCell(index iCell, tSmallCoords &cs)
        {
            if (!isPeriodic)
                __GetCoords(cell2node[iCell], cs);
            else
                __GetCoordsOnElem(cell2node[iCell], cell2nodePbi[iCell], cs);
        }

        DNDS_DEVICE_CALLABLE void GetCoordsOnCell(index iCell, tSmallCoords &cs, tCoordPair &coo)
        {
            if (!isPeriodic)
                __GetCoords(cell2node[iCell], cs, coo);
            else
                __GetCoordsOnElem(cell2node[iCell], cell2nodePbi[iCell], cs, coo);
        }

        DNDS_DEVICE_CALLABLE void GetCoordsOnFace(index iFace, tSmallCoords &cs)
        {
            if (!isPeriodic)
                __GetCoords(face2node[iFace], cs);
            else
                __GetCoordsOnElem(face2node[iFace], face2nodePbi[iFace], cs);
        }

        DNDS_DEVICE_CALLABLE tPoint GetCoordNodeOnCell(index iCell, rowsize ic2n)
        {
            if (!isPeriodic)
                return coords[cell2node(iCell, ic2n)];
            return periodicInfo.GetCoordByBits(coords[cell2node(iCell, ic2n)], cell2nodePbi(iCell, ic2n));
        }

        DNDS_DEVICE_CALLABLE tPoint GetCoordNodeOnFace(index iFace, rowsize if2n)
        {
            if (!isPeriodic)
                return coords[face2node(iFace, if2n)];
            return periodicInfo.GetCoordByBits(coords[face2node(iFace, if2n)], face2nodePbi(iFace, if2n));
        }

        // tPoint GetCoordWallDistOnCell(index iCell, rowsize ic2n)
        // {
        //     if (!isPeriodic)
        //         return nodeWallDist[cell2node(iCell, ic2n)];
        //     return periodicInfo.GetVectorByBits<3, 1>(nodeWallDist[cell2node(iCell, ic2n)], cell2nodePbi(iCell, ic2n));
        // }

        // tPoint GetCoordWallDistOnFace(index iFace, rowsize if2n)
        // {
        //     if (!isPeriodic)
        //         return nodeWallDist[face2node(iFace, if2n)];
        //     return periodicInfo.GetVectorByBits<3, 1>(nodeWallDist[face2node(iFace, if2n)], face2nodePbi(iFace, if2n));
        // }

        DNDS_DEVICE_CALLABLE bool CellIsFaceBack(index iCell, index iFace) const
        {
            DNDS_assert(face2cell(iFace, 0) == iCell || face2cell(iFace, 1) == iCell);
            return face2cell(iFace, 0) == iCell;
        }

        DNDS_DEVICE_CALLABLE index CellFaceOther(index iCell, index iFace) const
        {
            return CellIsFaceBack(iCell, iFace)
                       ? face2cell(iFace, 1)
                       : face2cell(iFace, 0);
        }
    };
}