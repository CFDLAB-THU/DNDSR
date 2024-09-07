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

namespace DNDS::Geom
{
    static const t_index INTERNAL_ZONE = -1;
    struct ElemInfo
    {
        t_index type = static_cast<t_index>(Elem::UnknownElem);
        /// @brief positive for BVnum, 0 for internal Elems, Negative for ?
        t_index zone = INTERNAL_ZONE;

        Elem::ElemType getElemType()
        {
            return static_cast<Elem::ElemType>(type);
        }

        void setElemType(Elem::ElemType t)
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

        static MPI_Datatype CommType() { return MPI_INT32_T; }
        static int CommMult() { return 2; }
    };

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

    struct UnstructuredMesh;

    struct UnstructuredMesh
    {
        MPI_int mRank{0};
        DNDS::MPIInfo mpi;
        int dim;
        bool isPeriodic{false};
        MeshAdjState adjPrimaryState{Adj_Unknown};
        MeshAdjState adjFacialState{Adj_Unknown};
        MeshAdjState adjC2FState{Adj_Unknown};
        Periodicity periodicInfo;
        index nNodeO1{-1};
        MeshElevationState elevState = Elevation_Untouched;
        /// reader
        tCoordPair coords;
        tAdjPair cell2node;
        tAdjPair bnd2node;
        tAdj2Pair bnd2cell;
        tAdjPair cell2cell;
        tElemInfoArrayPair cellElemInfo;
        tElemInfoArrayPair bndElemInfo;
        /// periodic only, after reader
        tPbiPair cell2nodePbi;

        /// inverse relations
        tAdjPair node2cell;
        tAdjPair node2bnd;

        /// interpolated
        // *! currently assume all these are Adj_PointToLocal
        tAdjPair cell2face;
        tAdjPair face2node;
        tAdj2Pair face2cell;
        tElemInfoArrayPair faceElemInfo;
        std::vector<index> bnd2face;
        std::unordered_map<index, index> face2bnd;
        /// periodic only, after interpolated
        tPbiPair face2nodePbi;

        /// parent built
        std::vector<index> node2parentNode;
        std::vector<index> node2bndNode;
        std::vector<index> cell2parentCell;

        /// for parallel out
        std::vector<index> vtkCell2nodeOffsets;
        std::vector<uint8_t> vtkCellType;
        std::vector<index> vtkCell2node;
        index vtkNodeOffset{-1};
        index vtkCellOffset{-1};
        index vtkCell2NodeGlobalSiz{-1};
        index vtkNCellGlobal{-1};
        index vtkNNodeGlobal{-1};
        tAdjPair cell2nodePeriodicRecreated;
        tCoordPair coordsPeriodicRecreated;
        std::vector<index> nodeRecreated2nodeLocal;

        struct HDF5OutSetting
        {
            size_t chunkSize = 128;
            int deflateLevel = 1;
        } hdf5OutSetting;

        /// only elevation
        tCoordPair coordsElevDisp;
        index nTotalMoved{-1};
        struct ElevationInfo
        {
            real RBFRadius = 1;
            real MaxIncludedAngle = 15;
            int nIter = 60;
            int nSearch = 30;
            RBF::RBFKernelType kernel = RBF::InversedDistanceA1;
            real refDWall = 1e-3;
            real RBFPower = 1;
        } elevationInfo;

        // tAdj1Pair bndFaces; // no comm needed for now

        /// for cell local factorization
        using tLocalMatStruct = std::vector<std::vector<index>>;
        tLocalMatStruct cell2cellFaceVLocal;

        UnstructuredMesh(const DNDS::MPIInfo &n_mpi, int n_dim)
            : mpi(n_mpi), dim(n_dim) {}

        /**
         * \brief
         * return normal negative:  mapping to un-found in father-son
         */
        index NodeIndexGlobal2Local(DNDS::index iNodeOther)
        {
            if (iNodeOther == UnInitIndex)
                return iNodeOther;
            DNDS::MPI_int rank;
            DNDS::index val;
            auto result = coords.trans.pLGhostMapping->search_indexAppend(iNodeOther, rank, val);
            if (result)
                return val;
            else
                return -1 - iNodeOther; // mapping to un-found in father-son
        }

        index NodeIndexLocal2Global(DNDS::index iNodeOther)
        {
            if (iNodeOther == UnInitIndex)
                return iNodeOther;
            if (iNodeOther < 0) // mapping to un-found in father-son
                return -1 - iNodeOther;
            else
                return coords.trans.pLGhostMapping->operator()(-1, iNodeOther);
        }

        index NodeIndexLocal2Global_NoSon(index iNode)
        {
            if (iNode < 0 || iNode >= coords.father->Size())
                return UnInitIndex;
            return coords.trans.pLGlobalMapping->operator()(mpi.rank, iNode);
        }

        /**
         * \brief
         * return normal negative:  mapping to un-found in father
         */
        index NodeIndexGlobal2Local_NoSon(index iNode)
        {
            auto [ret, rank, val] = coords.trans.pLGlobalMapping->search(iNode);
            DNDS_assert_info(ret, "search failed");
            if (rank == mpi.rank)
                return val;
            else
                return -1 - iNode;
        }

        /**
         * \brief
         * return normal negative:  mapping to un-found in father-son
         */
        index CellIndexGlobal2Local(DNDS::index iCellOther)
        {
            if (iCellOther == UnInitIndex)
                return iCellOther;
            DNDS::MPI_int rank;
            DNDS::index val;
            auto result = cellElemInfo.trans.pLGhostMapping->search_indexAppend(iCellOther, rank, val);
            if (result)
                return val;
            else
                return -1 - iCellOther; // mapping to un-found in father-son
        }

        index CellIndexLocal2Global(DNDS::index iCellOther)
        {
            if (iCellOther == UnInitIndex)
                return iCellOther;
            if (iCellOther < 0) // mapping to un-found in father-son
                return -1 - iCellOther;
            else
                return cellElemInfo.trans.pLGhostMapping->operator()(-1, iCellOther);
        }

        index CellIndexLocal2Global_NoSon(index iCell)
        {
            if (iCell < 0 || iCell >= cell2node.father->Size())
                return UnInitIndex;
            return cell2node.trans.pLGlobalMapping->operator()(mpi.rank, iCell);
        }

        /**
         * \brief
         * return normal negative:  mapping to un-found in father
         */
        index CellIndexGlobal2Local_NoSon(index iCell)
        {
            auto [ret, rank, val] = cell2node.trans.pLGlobalMapping->search(iCell);
            DNDS_assert_info(ret, "search failed");
            if (rank == mpi.rank)
                return val;
            else
                return -1 - iCell;
        }

        index BndIndexLocal2Global_NoSon(index iBnd)
        {
            if (iBnd < 0 || iBnd >= bnd2node.father->Size())
                return UnInitIndex;
            return bnd2node.trans.pLGlobalMapping->operator()(mpi.rank, iBnd);
        }

        /**
         * \brief
         * return normal negative:  mapping to un-found in father
         */
        index BndIndexGlobal2Local_NoSon(index iBnd)
        {
            auto [ret, rank, val] = bnd2node.trans.pLGlobalMapping->search(iBnd);
            DNDS_assert_info(ret, "search failed");
            if (rank == mpi.rank)
                return val;
            else
                return -1 - iBnd;
        }

        /**
         * \brief only requires father part of cell2node, bnd2node and coords
         * generates node2cell and node2bnd (father part)
         */
        void RecoverNode2CellAndNode2Bnd();

        /**
         * \brief needs to use RecoverNode2CellAndNode2Bnd before doing this.
         * Requires node2cell.father and builds a version of its son.
         */
        void RecoverCell2CellAndBnd2Cell();

        /**
         * @brief building ghost (son) from primary (currently only cell2cell)
         * @details
         * the face and bnd parts are currently only local (no comm available)
         * only builds comm data of cell and node
         * cells: current-father and cell2cell neighbor (face or node neighbor)
         * nodes: needed by all cells
         * faces/bnds: needed by all father cells
         *
         */
        void BuildGhostPrimary();
        void AdjGlobal2LocalPrimary();
        void AdjLocal2GlobalPrimary();
        void AdjGlobal2LocalPrimaryForBnd();
        void AdjLocal2GlobalPrimaryForBnd();

        void AdjGlobal2LocalFacial();
        void AdjLocal2GlobalFacial();
        void AdjGlobal2LocalC2F();
        void AdjLocal2GlobalC2F();

        void InterpolateFace();
        void AssertOnFaces();

        void ConstructBndMesh(UnstructuredMesh &bMesh);

        // void ReorderCellLocal();

        void ObtainLocalFactFillOrdering(Direct::SerialSymLUStructure &symLU, Direct::DirectPrecControl control);          // 1 uses metis, 2 uses MMD, //TODO 10 uses geometric based searching
        void ObtainSymmetricSymbolicFactorization(Direct::SerialSymLUStructure &symLU, Direct::DirectPrecControl control); // -1 use full LU, 0-3 use ilu(code),

        index NumNode() { return coords.father->Size(); }
        index NumCell() { return cell2node.father->Size(); }
        index NumFace() { return face2node.father->Size(); }
        index NumBnd() { return bnd2node.father->Size(); }

        index NumNodeGhost() { return coords.son->Size(); }
        index NumCellGhost() { return cell2node.son->Size(); }
        index NumFaceGhost() { return face2node.son->Size(); }

        index NumNodeProc() { return coords.Size(); }
        index NumCellProc() { return cell2node.Size(); }
        index NumFaceProc() { return face2node.Size(); }

        /// @warning must collectively call
        index NumCellGlobal() { return cell2node.father->globalSize(); }
        /// @warning must collectively call
        index NumNodeGlobal() { return coords.father->globalSize(); }
        /// @warning must collectively call
        index NumFaceGlobal() { return face2node.father->globalSize(); }
        /// @warning must collectively call
        index NumBndGlobal() { return bnd2node.father->globalSize(); }

        Elem::Element GetCellElement(index iC) { return Elem::Element{cellElemInfo(iC, 0).getElemType()}; }
        Elem::Element GetFaceElement(index iF) { return Elem::Element{faceElemInfo(iF, 0).getElemType()}; }
        Elem::Element GetBndElement(index iB) { return Elem::Element{bndElemInfo(iB, 0).getElemType()}; }

        t_index GetCellZone(index iC) { return cellElemInfo(iC, 0).zone; }
        t_index GetFaceZone(index iF) { return faceElemInfo(iF, 0).zone; }
        t_index GetBndZone(index iB) { return bndElemInfo(iB, 0).zone; }

        MPIInfo &getMPI() { return mpi; }

        void BuildO2FromO1Elevation(UnstructuredMesh &meshO1);
        void ElevatedNodesGetBoundarySmooth(const std::function<bool(t_index)> &FiFBndIdNeedSmooth);
        void ElevatedNodesSolveInternalSmooth();
        void ElevatedNodesSolveInternalSmoothV1Old();
        void ElevatedNodesSolveInternalSmoothV1();
        void ElevatedNodesSolveInternalSmoothV2();

        void BuildBisectO1FormO2(UnstructuredMesh &meshO2);

        bool IsO1();
        bool IsO2();

        /**
         * @brief directly load coords; gets faulty if isPeriodic!
         */
        template <class tC2n>
        void __GetCoords(const tC2n &c2n, tSmallCoords &cs)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                if (adjPrimaryState == Adj_PointToGlobal)
                    iNode = NodeIndexGlobal2Local(iNode), DNDS_assert_info(iNode >= 0, "iNode not found in main/ghost pair");
                cs(Eigen::all, i) = coords[iNode];
            }
        }

        /**
         * @brief directly load coords; gets faulty if isPeriodic!
         */
        template <class tC2n, class tCoordExt>
        void __GetCoords(const tC2n &c2n, tSmallCoords &cs, tCoordExt &coo)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                if (adjPrimaryState == Adj_PointToGlobal)
                    iNode = NodeIndexGlobal2Local(iNode), DNDS_assert_info(iNode >= 0, "iNode not found in main/ghost pair");
                cs(Eigen::all, i) = coo[iNode];
            }
        }

        /**
         * @brief specially for periodicity
         */
        template <class tC2n, class tC2nPbi>
        void __GetCoordsOnElem(const tC2n &c2n, const tC2nPbi &c2nPbi, tSmallCoords &cs)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                if (adjPrimaryState == Adj_PointToGlobal)
                    iNode = NodeIndexGlobal2Local(iNode), DNDS_assert_info(iNode >= 0, "iNode not found in main/ghost pair");
                cs(Eigen::all, i) = periodicInfo.GetCoordByBits(coords[iNode], c2nPbi[i]);
            }
        }

        /**
         * @brief specially for periodicity
         */
        template <class tC2n, class tC2nPbi, class tCoordExt>
        void __GetCoordsOnElem(const tC2n &c2n, const tC2nPbi &c2nPbi, tSmallCoords &cs, tCoordExt &coo)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                if (adjPrimaryState == Adj_PointToGlobal)
                    iNode = NodeIndexGlobal2Local(iNode), DNDS_assert_info(iNode >= 0, "iNode not found in main/ghost pair");
                cs(Eigen::all, i) = periodicInfo.GetCoordByBits(coo[iNode], c2nPbi[i]);
            }
        }

        void GetCoordsOnCell(index iCell, tSmallCoords &cs)
        {
            if (!isPeriodic)
                __GetCoords(cell2node[iCell], cs);
            else
                __GetCoordsOnElem(cell2node[iCell], cell2nodePbi[iCell], cs);
        }

        void GetCoordsOnCell(index iCell, tSmallCoords &cs, tCoordPair &coo)
        {
            if (!isPeriodic)
                __GetCoords(cell2node[iCell], cs, coo);
            else
                __GetCoordsOnElem(cell2node[iCell], cell2nodePbi[iCell], cs, coo);
        }

        void GetCoordsOnFace(index iFace, tSmallCoords &cs)
        {
            if (!isPeriodic)
                __GetCoords(face2node[iFace], cs);
            else
                __GetCoordsOnElem(face2node[iFace], face2nodePbi[iFace], cs);
        }

        tPoint GetCoordNodeOnCell(index iCell, rowsize ic2n)
        {
            if (!isPeriodic)
                return coords[cell2node(iCell, ic2n)];
            return periodicInfo.GetCoordByBits(coords[cell2node(iCell, ic2n)], cell2nodePbi(iCell, ic2n));
        }

        tPoint GetCoordNodeOnFace(index iFace, rowsize if2n)
        {
            if (!isPeriodic)
                return coords[face2node(iFace, if2n)];
            return periodicInfo.GetCoordByBits(coords[face2node(iFace, if2n)], face2nodePbi(iFace, if2n));
        }

        bool CellIsFaceBack(index iCell, index iFace) const
        {
            DNDS_assert(face2cell(iFace, 0) == iCell || face2cell(iFace, 1) == iCell);
            return face2cell(iFace, 0) == iCell;
        }

        index CellFaceOther(index iCell, index iFace) const
        {
            return CellIsFaceBack(iCell, iFace)
                       ? face2cell(iFace, 1)
                       : face2cell(iFace, 0);
        }

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
        template <class FA, class FB, class F0 = std::function<void(void)>>
        auto CellOtherCellPeriodicHandle(
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
         * \return
         * cell2cell for local mesh, which do not contain
         * the diagonal part; should be a diag-less symmetric adjacency matrix
         */
        auto GetCell2CellFaceVLocal()
        {
            DNDS_assert(this->adjPrimaryState == Adj_PointToLocal);
            std::vector<std::vector<index>> cell2cellFaceV;
            cell2cellFaceV.resize(this->NumCell());
            for (index iCell = 0; iCell < this->NumCell(); iCell++)
            {
                cell2cellFaceV[iCell].reserve(cell2face.RowSize(iCell)); // do not preserve the diagonal
                for (auto iFace : cell2face[iCell])
                {
                    index iCellOther = this->CellFaceOther(iCell, iFace);
                    if (iCellOther != UnInitIndex && iCellOther < this->NumCell()) //! must be local not ghost ptrs
                        cell2cellFaceV[iCell].push_back(iCellOther);
                }
            }
            return cell2cellFaceV;
        }

        void WriteSerialize(SerializerBase *serializer, const std::string &name);
        void ReadSerialize(SerializerBase *serializer, const std::string &name);

        template <class TFTrans>
        void TransformCoords(TFTrans &&FTrans)
        {
            for (index iNode = 0; iNode < coords.Size(); iNode++)
                coords[iNode] = FTrans(coords[iNode]);
        }

        void RecreatePeriodicNodes();

        void BuildVTKConnectivity();

        void PrintParallelVTKHDFDataArray(
            std::string fname, std::string seriesName,
            int arraySiz, int vecArraySiz, int arraySizPoint, int vecArraySizPoint,
            const tFGetName &names,
            const tFGetData &data,
            const tFGetName &vectorNames,
            const tFGetVecData &vectorData,
            const tFGetName &namesPoint,
            const tFGetData &dataPoint,
            const tFGetName &vectorNamesPoint,
            const tFGetVecData &vectorDataPoint,
            double t);

        void SetHDF5OutSetting(size_t chunkSiz, int deflateLevel)
        {
            hdf5OutSetting.chunkSize = chunkSiz;
            hdf5OutSetting.deflateLevel = deflateLevel;
        }
    };

}
namespace DNDS::Geom
{
    using tFDataFieldName = std::function<std::string(int)>;
    using tFDataFieldQuery = tFGetData;

    enum MeshReaderMode
    {
        UnknownMode,
        SerialReadAndDistribute,
        SerialOutput,
    };

    struct UnstructuredMeshSerialRW
    {
    private:
        int ascii_precision{16};
        std::string vtuFloatEncodeMode = "ascii";

    public:
        DNDS::ssp<UnstructuredMesh> mesh;

        MeshReaderMode mode{UnknownMode};

        bool dataIsSerialOut = false;
        bool dataIsSerialIn = false;

        tCoord coordSerial;                // created through reading
        tAdj cell2nodeSerial;              // created through reading
        tAdj bnd2nodeSerial;               // created through reading
        tElemInfoArray cellElemInfoSerial; // created through reading
        tElemInfoArray bndElemInfoSerial;  // created through reading
        tAdj2 bnd2cellSerial;              // created through reading
        tPbi cell2nodePbiSerial;           // created through reading-Deduplicate
        /***************************************************************/
        // Current Method: R/W don't manage actually used interpolation,
        // but manually get cell2cell or node2node
        // because: currently only support node based or cell based
        /***************************************************************/

        // tAdj face2nodeSerial;    // created through InterpolateTopology
        // tAdj2 face2cellSerial;   // created through InterpolateTopology
        // tAdj cell2faceSerial;    // created through InterpolateTopology
        // tElemInfoArray faceElemInfoSerial; // created through InterpolateTopology

        tAdj cell2cellSerial; // optionally created with GetCell2Cell()
        tAdj node2nodeSerial; // optionally created with GetNode2Node()

        tAdj cell2cellSerialFacial; // optionally created with GetCell2Cell()

        tAdj node2cellSerial; // not used for now
        tAdj node2faceSerial; // not used for now
        tAdj node2edgeSerial; // not used for now

        tAdj cell2faceSerial; // not used for now
        tAdj cell2edgeSerial; // not used for now

        tAdj face2nodeSerial; // not used for now
        tAdj face2faceSerial; // not used for now
        tAdj face2edgeSerial; // not used for now
        tAdj face2cellSerial; // not used for now

        tAdj edge2nodeSerial; // not used for now
        tAdj edge2cellSerial; // not used for now
        tAdj edge2edgeSerial; // not used for now
        tAdj edge2faceSerial; // not used for now

        DNDS::ArrayTransformerType<tCoord::element_type>::Type coordSerialOutTrans;                // used in serial out mode
        DNDS::ArrayTransformerType<tAdj::element_type>::Type cell2nodeSerialOutTrans;              // used in serial out mode
        DNDS::ArrayTransformerType<tPbi::element_type>::Type cell2nodePbiSerialOutTrans;           // used in serial out mode
        DNDS::ArrayTransformerType<tAdj::element_type>::Type bnd2nodeSerialOutTrans;               // used in serial out mode
        DNDS::ArrayTransformerType<tElemInfoArray::element_type>::Type cellElemInfoSerialOutTrans; // used in serial out mode
        DNDS::ArrayTransformerType<tElemInfoArray::element_type>::Type bndElemInfoSerialOutTrans;  // used in serial out mode

        std::vector<DNDS::MPI_int> cellPartition;
        std::vector<DNDS::MPI_int> nodePartition;
        std::vector<DNDS::MPI_int> bndPartition;

        DNDS::MPI_int mRank{0}, cnPart{0};

        UnstructuredMeshSerialRW(const decltype(mesh) &n_mesh, DNDS::MPI_int n_mRank)
            : mesh(n_mesh), mRank(n_mRank) {}

        /// @brief reads a cgns file as serial input
        /**
         * @details
         * the file MUST consist of one CGNS base node, and
         * multiple zones within.
         * All the zones are treated as a whole unstructured grid
         * Proofed on .cgns files generated from Pointwise
         * @warning //!Pointwise Options: with "Include Donor Information", "Treat Structured as Unstructured", "Unstructured Interfaces = Node-to-Node"
         */
        /// @todo //TODO Add some multi thread here!
        /// @param fName file name of .cgns file
        void ReadFromCGNSSerial(const std::string &fName, const t_FBCName_2_ID &FBCName_2_ID = FBC_Name_2_ID_Default);

        void ReadFromOpenFOAMAndConvertSerial(const std::string &fName, const std::map<std::string, std::string> &nameMapping, const t_FBCName_2_ID &FBCName_2_ID = FBC_Name_2_ID_Default);

        void Deduplicate1to1Periodic(real searchEps = 1e-8);

        // void InterpolateTopology();

        /**
         * \brief build cell2cell topology, with node-neighbors included
         * \todo add support for only face-neighbors
         */
        void BuildCell2Cell(); // For cell based purpose

        void BuildNode2Node(); // For node based purpose //!not yet implemented

        void MeshPartitionCell2Cell();

        void PartitionReorderToMeshCell2Cell();

        void ClearSerial()
        {
            coordSerial.reset();
            cell2nodeSerial.reset();
            cell2cellSerial.reset();
            cellElemInfoSerial.reset();
            bnd2nodeSerial.reset();
            bnd2cellSerial.reset();
            bndElemInfoSerial.reset();
            mode = UnknownMode;
            dataIsSerialIn = dataIsSerialOut = false;
        }

        /**
         * @brief should be called to build data for serial out
         */
        void BuildSerialOut();

        void GetCurrentOutputArrays(int flag,
                                    tCoordPair &coordOut,
                                    tAdjPair &cell2nodeOut,
                                    tPbiPair &cell2nodePbiOut,
                                    tElemInfoArrayPair &cellElemInfoOut,
                                    index &nCell, index &nNode);

        // void WriteToCGNSSerial(const std::string &fName);

        // void WriteSolutionToTecBinary(const std::string &fName,
        //                               int nField, const tFDataFieldName &names, const tFDataFieldQuery &data,
        //                               int nFieldBnd, const tFDataFieldName &namesBnd, const tFDataFieldQuery &dataBnd);

        /**
         * @brief names(idata) data(idata, ivolume)
         * https://tecplot.azureedge.net/products/360/current/360_data_format_guide.pdf
         * @todo //TODO add support for bnd export
         */
        void PrintSerialPartPltBinaryDataArray(
            std::string fname,
            int arraySiz, int arraySizPoint,
            const tFGetName &names,
            const tFGetData &data,
            const tFGetName &namesPoint,
            const tFGetData &dataPoint,
            double t, int flag);

        /**
         * @brief names(idata) data(idata, ivolume)
         * @todo //TODO add support for bnd export
         */
        void PrintSerialPartVTKDataArray(
            std::string fname, std::string seriesName,
            int arraySiz, int vecArraySiz, int arraySizPoint, int vecArraySizPoint,
            const tFGetName &names,
            const tFGetData &data,
            const tFGetName &vectorNames,
            const tFGetVecData &vectorData,
            const tFGetName &namesPoint,
            const tFGetData &dataPoint,
            const tFGetName &vectorNamesPoint,
            const tFGetVecData &vectorDataPoint,
            double t, int flag = 0);

        void SetASCIIPrecision(int n) { ascii_precision = n; }
        void SetVTKFloatEncodeMode(const std::string &v)
        {
            vtuFloatEncodeMode = v;
            DNDS_assert(v == "ascii" || v == "binary");
        }
    };
} // namespace geom
