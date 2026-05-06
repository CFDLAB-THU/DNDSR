#pragma once
#include "DNDS/Device/DeviceStorage.hpp"
#include "DNDS/Device/DeviceTransferable.hpp"
#include "Geom/Elements.hpp"
#include "DNDS/Array.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"
#include "Geom/BoundaryCondition.hpp"
#include "DNDS/ArrayPair.hpp"
#include "Geom/PeriodicInfo.hpp"
#include "Geom/RadialBasisFunction.hpp"
#include "DNDS/ObjectUtils.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "AdjIndexInfo.hpp"
#include "MeshConnectivity.hpp"
#include "ReorderPlan.hpp"

namespace DNDS::Direct
{
    struct SerialSymLUStructure;
    struct DirectPrecControl;
}
#include "Mesh_DeviceView.hpp"

namespace DNDS::Geom
{
    enum class MeshLoc : uint8_t
    {
        Unknown = 0,
        Node = 1,
        Face = 3,
        Cell = 4
    };

    struct PartitionOptions; // forward declaration; defined after UnstructuredMesh

    struct UnstructuredMesh : public DeviceTransferable<UnstructuredMesh>
    {
        MPI_int mRank{0};
        DNDS::MPIInfo mpi;
        int dim;
        bool isPeriodic{false};
        // state of: cell2node, cell2cell, bnd2node (only for non-bnd mesh object), bnd2cell (only for non-bnd mesh object)
        MeshAdjState adjPrimaryState{Adj_Unknown};
        // state of: face2cell, face2node, face2bnd
        MeshAdjState adjFacialState{Adj_Unknown};
        // state of: cell2face, bnd2face
        MeshAdjState adjC2FState{Adj_Unknown};
        // state of: node2cell, node2bnd
        MeshAdjState adjN2CBState{Adj_Unknown};
        // state of: cell2cellFace
        MeshAdjState adjC2CFaceState{Adj_Unknown};
        Periodicity periodicInfo;
        index nNodeO1{-1};
        MeshElevationState elevState = Elevation_Untouched;
        /// reader
        tCoordPair coords;
        AdjPairTracked<tAdjPair> cell2node; // → Node
        AdjPairTracked<tAdjPair> bnd2node;  // → Node
        AdjPairTracked<tAdj2Pair> bnd2cell; // → Cell
        AdjPairTracked<tAdjPair> cell2cell; // → Cell
        tElemInfoArrayPair cellElemInfo;
        tElemInfoArrayPair bndElemInfo;
        tAdj1Pair cell2cellOrig; // no device
        tAdj1Pair node2nodeOrig; // no device
        tAdj1Pair bnd2bndOrig;   // no device
        /// periodic only, after reader
        tPbiPair cell2nodePbi;
        tPbiPair bnd2nodePbi;

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

        /// inverse relations
        AdjPairTracked<tAdjPair> node2cell; // → Cell
        AdjPairTracked<tAdjPair> node2bnd;  // → Bnd

        auto device_array_list_N2CB()
        {
            return std::make_tuple(
                DNDS_MAKE_1_MEMBER_REF(node2cell),
                DNDS_MAKE_1_MEMBER_REF(node2bnd));
        }

        /// interpolated
        AdjPairTracked<tAdjPair> cell2face;  // → Face
        AdjPairTracked<tAdjPair> face2node;  // → Node
        AdjPairTracked<tAdj2Pair> face2cell; // → Cell
        tElemInfoArrayPair faceElemInfo;
        AdjPairTracked<tAdj1Pair> bnd2face; // → Face
        AdjPairTracked<tAdj1Pair> face2bnd; // → Bnd

        std::vector<index> bnd2faceV;               // no device
        std::unordered_map<index, index> face2bndM; // no device
        /// periodic only, after interpolated
        tPbiPair face2nodePbi;

        auto device_array_list_facial()
        {
            return std::make_tuple(
                DNDS_MAKE_1_MEMBER_REF(face2cell),
                DNDS_MAKE_1_MEMBER_REF(face2node),
                DNDS_MAKE_1_MEMBER_REF(face2nodePbi),
                DNDS_MAKE_1_MEMBER_REF(faceElemInfo),
                DNDS_MAKE_1_MEMBER_REF(face2bnd));
        }

        auto device_array_list_C2F()
        {
            return std::make_tuple(
                DNDS_MAKE_1_MEMBER_REF(cell2face),
                DNDS_MAKE_1_MEMBER_REF(bnd2face));
        }

        /// constructed on demand
        AdjPairTracked<tAdjPair> cell2cellFace; // → Cell

        /// parent built
        std::vector<index> node2parentNode; // from local-appended iNode to local-appended iNode in parent
        std::vector<index> node2bndNode;    // from local-appended iNode to local-appended iNode in bnd
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
            bool coll_on_data = false;
            bool coll_on_meta = true;
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
        tLocalMatStruct cell2cellFaceVLocalParts;

        std::vector<index> localPartitionStarts;

        /// wall dist:
        tCoordPair nodeWallDist;

        UnstructuredMesh(const DNDS::MPIInfo &n_mpi, int n_dim)
            : mpi(n_mpi), dim(n_dim) {}

        int getDim() const { return dim; }

        auto getCell2NodeIndexPbiRow(index iCell)
        {
            std::vector<Geom::NodeIndexPBI> ret;
            ret.reserve(cell2node[iCell].size());
            for (int ic2n = 0; ic2n < cell2node[iCell].size(); ic2n++)
                if (isPeriodic)
                    ret.push_back(Geom::NodeIndexPBI{cell2node[iCell][ic2n], cell2nodePbi[iCell][ic2n]});
                else
                    ret.push_back(Geom::NodeIndexPBI{cell2node[iCell][ic2n], Geom::NodePeriodicBits{}});
            return ret;
        }

        auto getBnd2NodeIndexPbiRow(index iBnd)
        {
            std::vector<Geom::NodeIndexPBI> ret;
            ret.reserve(bnd2node[iBnd].size());
            for (int ib2n = 0; ib2n < bnd2node[iBnd].size(); ib2n++)
                if (isPeriodic)
                    ret.push_back(Geom::NodeIndexPBI{bnd2node[iBnd][ib2n], bnd2nodePbi[iBnd][ib2n]});
                else
                    ret.push_back(Geom::NodeIndexPBI{bnd2node[iBnd][ib2n], Geom::NodePeriodicBits{}});
            return ret;
        }

        // =================================================================
        // Generic index conversion templates
        // =================================================================
        // The 4 templates below replace 12 structurally identical methods
        // (3 entity types x 4 variants).  The old named methods are kept
        // as thin inline wrappers so all existing call sites compile
        // unchanged.

        /**
         * \brief Ensure a pair has a ghost mapping on its transformer.
         *
         * If `trans.pLGhostMapping` is already set, does nothing.
         * Otherwise, creates a father-only ghost mapping (empty ghost set)
         * so that `IndexGlobal2Local` / `IndexLocal2Global` work even
         * before ghost layers are built.
         *
         * \pre `trans.pLGlobalMapping` must be set (call
         *      `createFatherGlobalMapping` first).
         * \warning Collective — calls MPI_Alltoall internally when creating
         *          the mapping.
         */
        template <class TPair>
        void EnsureGhostMapping(TPair &pair)
        {
            if (pair.trans.pLGhostMapping)
                return;
            DNDS_assert_info(pair.trans.pLGlobalMapping,
                             "EnsureGhostMapping: pLGlobalMapping must be set first");
            pair.trans.pLGhostMapping = AdjIndexInfo::makeFatherOnlyMapping(
                pair.trans.pLGlobalMapping,
                pair.father->Size(), mpi);
        }

        /**
         * \brief Global-to-local conversion using father+son ghost mapping.
         * \return local index, or (-1 - iGlobal) when not found in the pair.
         *         UnInitIndex passes through unchanged.
         */
        template <class TPair>
        index IndexGlobal2Local(TPair &pair, DNDS::index iGlobal)
        {
            DNDS_assert(pair.trans.pLGhostMapping);
            if (iGlobal == UnInitIndex)
                return iGlobal;
            DNDS::MPI_int rank = UnInitMPIInt;
            DNDS::index val = UnInitIndex;
            auto result = pair.trans.pLGhostMapping->search_indexAppend(iGlobal, rank, val);
            if (result)
                return val;
            else
                return -1 - iGlobal; // mapping to un-found in father-son
        }

        /**
         * \brief Local-to-global conversion using father+son ghost mapping.
         * \return global index, or the decoded global when the local is
         *         a negative "not-found" encoding.  UnInitIndex passes through.
         */
        template <class TPair>
        index IndexLocal2Global(TPair &pair, DNDS::index iLocal)
        {
            DNDS_assert(pair.trans.pLGhostMapping);
            if (iLocal == UnInitIndex)
                return iLocal;
            if (iLocal < 0) // mapping to un-found in father-son
                return -1 - iLocal;
            else
                return pair.trans.pLGhostMapping->operator()(-1, iLocal);
        }

        /**
         * \brief Local-to-global conversion using only the father's global
         *        mapping (no son / ghost layer).
         * \return global index.  Asserts if iLocal exceeds father size.
         *         Negative "not-found" encoding and UnInitIndex pass through.
         */
        template <class TPair>
        index IndexLocal2Global_NoSon(TPair &pair, index iLocal)
        {
            DNDS_assert(pair.father->pLGlobalMapping);
            if (iLocal == UnInitIndex)
                return UnInitIndex;
            if (iLocal < 0)
                return -1 - iLocal;
            if (iLocal >= pair.father->Size())
                DNDS_assert_info(false, "local idx not right: " + std::to_string(iLocal));
            return pair.father->pLGlobalMapping->operator()(mpi.rank, iLocal);
        }

        /**
         * \brief Global-to-local conversion using only the father's global
         *        mapping (no son / ghost layer).
         * \return local index if the global maps to this rank, otherwise
         *         (-1 - iGlobal).  UnInitIndex passes through.
         */
        template <class TPair>
        index IndexGlobal2Local_NoSon(TPair &pair, index iGlobal)
        {
            DNDS_assert(pair.father->pLGlobalMapping);
            if (iGlobal == UnInitIndex)
                return UnInitIndex;
            auto [ret, rank, val] = pair.father->pLGlobalMapping->search(iGlobal);
            DNDS_assert_info(ret, "search failed with input: " + std::to_string(iGlobal));
            if (rank == mpi.rank)
                return val;
            else
                return -1 - iGlobal;
        }

        // =================================================================
        // Named wrappers — Node
        // =================================================================
        index NodeIndexGlobal2Local(DNDS::index i) { return IndexGlobal2Local(coords, i); }
        index NodeIndexLocal2Global(DNDS::index i) { return IndexLocal2Global(coords, i); }
        index NodeIndexLocal2Global_NoSon(index i) { return IndexLocal2Global_NoSon(coords, i); }
        index NodeIndexGlobal2Local_NoSon(index i) { return IndexGlobal2Local_NoSon(coords, i); }

        // =================================================================
        // Named wrappers — Cell
        // =================================================================
        index CellIndexGlobal2Local(DNDS::index i) { return IndexGlobal2Local(cell2node, i); }
        index CellIndexLocal2Global(DNDS::index i) { return IndexLocal2Global(cell2node, i); }
        index CellIndexLocal2Global_NoSon(index i) { return IndexLocal2Global_NoSon(cell2node, i); }
        index CellIndexGlobal2Local_NoSon(index i) { return IndexGlobal2Local_NoSon(cell2node, i); }

        // =================================================================
        // Named wrappers — Bnd
        // =================================================================
        index BndIndexGlobal2Local(DNDS::index i) { return IndexGlobal2Local(bnd2node, i); }
        index BndIndexLocal2Global(DNDS::index i) { return IndexLocal2Global(bnd2node, i); }
        index BndIndexLocal2Global_NoSon(index i) { return IndexLocal2Global_NoSon(bnd2node, i); }
        index BndIndexGlobal2Local_NoSon(index i) { return IndexGlobal2Local_NoSon(bnd2node, i); }

        // =================================================================
        // Named wrappers — Face
        // =================================================================
        index FaceIndexGlobal2Local(DNDS::index i) { return IndexGlobal2Local(face2node, i); }
        index FaceIndexLocal2Global(DNDS::index i) { return IndexLocal2Global(face2node, i); }

        // =================================================================
        // Adjacency bulk-conversion helpers
        // =================================================================

        /**
         * \brief Apply a conversion function to every entry of an adjacency
         *        array's first \p nRows rows.
         *
         * Replaces the recurring nested-loop pattern:
         * \code
         *   for (index i = 0; i < nRows; i++)
         *       for (rowsize j = 0; j < adj.RowSize(i); j++)
         *           adj(i, j) = fn(adj(i, j));
         * \endcode
         *
         * \tparam TAdj  Any type with RowSize(index) and operator()(index, rowsize).
         * \tparam TFn   Callable  index(index).
         */
        template <class TAdj, class TFn>
        static void ConvertAdjEntries(TAdj &adj, index nRows, TFn &&fn)
        {
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                    adj(i, j) = fn(adj(i, j));
        }

        /**
         * \brief OpenMP-parallelized variant of ConvertAdjEntries.
         *
         * Same semantics as ConvertAdjEntries but with `#pragma omp parallel for`
         * over the outer row loop. Use when the adjacency array is large and
         * the conversion function is thread-safe.
         */
        template <class TAdj, class TFn>
        static void ConvertAdjEntriesOMP(TAdj &adj, index nRows, TFn &&fn)
        {
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                    adj(i, j) = fn(adj(i, j));
        }

        // =================================================================
        // Row-permutation helper
        // =================================================================

        /**
         * \brief Permute the father rows of an ArrayPair according to a
         *        mapping function.
         *
         * For each old row index \p i in [0, nRows), the row is moved to
         * position old2new(i).  Works with both CSR (variable-row-size)
         * and fixed-row-size array pairs: CSR pairs are Decompressed before
         * and Compressed after the permutation; fixed-size pairs are
         * copied directly.
         *
         * \tparam TPair  An ArrayPair-like type with father/son, IsCSR().
         * \tparam TFn    Callable  index(index) returning the new row index.
         */
        template <class TPair, class TFn>
        static void PermuteRows(TPair &pair, index nRows, TFn &&old2new)
        {
            using TArr = typename decltype(pair.father)::element_type;
            auto tmp = std::make_shared<TArr>(*pair.father); // deep copy via copy ctor
            if constexpr (TPair::IsCSR())
                pair.father->Decompress();
            for (index i = 0; i < nRows; i++)
            {
                index iNew = old2new(i);
                if constexpr (TPair::IsCSR())
                    pair.father->ResizeRow(iNew, tmp->RowSize(i));
                for (rowsize j = 0; j < tmp->RowSize(i); j++)
                    pair(iNew, j) = (*tmp)(i, j);
            }
            if constexpr (TPair::IsCSR())
                pair.father->Compress();
        }

        void SetPeriodicGeometry(
            const tPoint &translation1,
            const tPoint &rotationCenter1,
            const tPoint &eulerAngles1,
            const tPoint &translation2,
            const tPoint &rotationCenter2,
            const tPoint &eulerAngles2,
            const tPoint &translation3,
            const tPoint &rotationCenter3,
            const tPoint &eulerAngles3);

        /**
         * \brief only requires father part of cell2node, bnd2node and coords
         * generates node2cell and node2bnd (father part)
         */
        void RecoverNode2CellAndNode2Bnd();
        void RecoverNode2CellAndNode2BndLegacy();

        /**
         * \brief needs to use RecoverNode2CellAndNode2Bnd before doing this.
         * Requires node2cell.father and builds a version of its son.
         */
        void RecoverCell2CellAndBnd2Cell();
        void RecoverCell2CellAndBnd2CellLegacy();

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
        void BuildGhostPrimary(int nGhostLayers = 1);
        void BuildGhostPrimaryLegacy();
        void AdjGlobal2LocalPrimary();
        void AdjLocal2GlobalPrimary();
        // ForBnd: reduction of primary version, only on cell2node
        void AdjGlobal2LocalPrimaryForBnd();
        void AdjLocal2GlobalPrimaryForBnd();

        void AdjGlobal2LocalFacial();
        void AdjLocal2GlobalFacial();
        void AdjGlobal2LocalC2F();
        void AdjLocal2GlobalC2F();

        void BuildGhostN2CB();
        void AdjGlobal2LocalN2CB();
        void AdjLocal2GlobalN2CB();

        void AssertOnN2CB();

        void BuildCell2CellFace();
        void AdjGlobal2LocalC2CFace();
        void AdjLocal2GlobalC2CFace();

        void InterpolateFace();
        void BuildGhostFace();
        void MatchFaceBoundary();
        void InterpolateFaceLegacy();
        void AssertOnFaces();

        void ConstructBndMesh(UnstructuredMesh &bMesh);

        // =================================================================
        // Registry
        // =================================================================

        /**
         * \brief Populate a MeshConnectivity registry from this mesh's
         *        currently-built adjacencies.
         *
         * Registers all adjacency arrays whose father is non-null.
         * For each entity kind that appears as a source (.from) of any
         * registered adjacency, finds a pLGlobalMapping from any adj
         * array for that entity kind.
         *
         * \pre All entity kinds that have registered adjacencies must
         *      have at least one adj array with a valid pLGlobalMapping
         *      on its father.  Throws if this is not satisfied.
         *
         * \param dag   MeshConnectivity to populate (meshDim is set).
         */
        void fillRegistry(MeshConnectivity &dag) const;

        /// \overload Overload with an explicit skip set.
        /// AdjKinds in \p skip are excluded from registration.
        void fillRegistry(
            MeshConnectivity &dag,
            const std::unordered_set<AdjKind, AdjKindHash> &skip) const;

        // =================================================================
        // Reorder (distributed entity reordering framework)
        // =================================================================

        /**
         * \brief Build a ReorderRegistry containing all mesh members.
         *
         * Registers all built adj arrays (as type-erased callbacks) and all
         * companion arrays (coords, cellElemInfo, pbi, etc.).
         * Skips adjacencies involving destroyKinds.
         *
         * External code may extend the returned registry with its own arrays
         * before passing to ReorderPlan::build.
         */
        ReorderRegistry buildReorderRegistry(
            const std::unordered_set<EntityKind> &destroyKinds = {});

        /**
         * \brief Reorder entities using the general framework.
         *
         * Builds a ReorderRegistry, computes follow maps, builds a
         * ReorderPlan, applies it, rebuilds global mappings, and updates
         * idx states.
         *
         * \pre All adjacencies in Adj_PointToGlobal state.
         * \post All (non-destroyed) adjacencies in Adj_PointToGlobal.
         *       Ghost mappings stale (caller must rebuild ghosts).
         *       Global mappings fresh on reordered entities.
         *
         * \warning Collective.
         */
        void ReorderEntities(const ReorderInput &input);

        /**
         * \brief Build a ReorderPlan without applying it.
         *
         * Useful for external code to obtain the plan and apply it to
         * its own arrays after the mesh reorder.
         */
        ReorderPlan buildReorderPlan(const ReorderInput &input);

        /**
         * \brief Augment a ReorderInput with default follows and compute follow maps.
         *
         * When `input.follows` is empty and Cell is explicitly reordered,
         * default follows (Node/Bnd follow Cell) are added automatically.
         * Returns a finalised ReorderInput with all follows resolved to
         * explicit maps (ready for ReorderPlan::build).
         *
         * \param input    Original input (may have empty follows).
         * \param reg      Registry with global mappings and adj data.
         * \return ReorderInput with all entity kinds as explicit maps.
         */
        ReorderInput resolveFollows(
            const ReorderInput &input,
            const ReorderRegistry &reg);

        // void ReorderCellLocal();

        /**
         * \return
         * cell2cell for local mesh, which do not contain
         * the diagonal part; should be a diag-less symmetric adjacency matrix
         */
        tLocalMatStruct GetCell2CellFaceVLocal(bool onLocalPartition = false);

        void ObtainLocalFactFillOrdering(Direct::SerialSymLUStructure &symLU, Direct::DirectPrecControl control);                // 1 uses metis, 2 uses MMD, //TODO 10 uses geometric based searching
        void ObtainSymmetricSymbolicFactorization(Direct::SerialSymLUStructure &symLU, Direct::DirectPrecControl control) const; // -1 use full LU, 0-3 use ilu(code),
        /**
         * \warning RecreatePeriodicNodes and BuildVTKConnectivity results are invalid after this;
         * \warning bnd mesh's cell2parentCell is invalid after this
         */
        void ReorderLocalCells(int nParts = 1, int nPartsInner = 1);

        /// Legacy implementation preserved for reference/fallback.
        void ReorderLocalCellsLegacy(int nParts = 1, int nPartsInner = 1);

        int NLocalParts() const { return localPartitionStarts.size() ? localPartitionStarts.size() - 1 : 1; }
        index LocalPartStart(int iPart) const { return localPartitionStarts.size() ? localPartitionStarts.at(iPart) : 0; }
        index LocalPartEnd(int iPart) const { return localPartitionStarts.size() ? localPartitionStarts.at(iPart + 1) : this->NumCell(); }

        index NumNode() const
        {
            DNDS_check_throw_info(coords.father, "coords not initialized");
            return coords.father->Size();
        }
        index NumCell() const
        {
            DNDS_check_throw_info(cell2node.father, "cell2node not initialized");
            return cell2node.father->Size();
        }
        index NumFace() const
        {
            DNDS_check_throw_info(face2node.father, "face2node not initialized");
            return face2node.father->Size();
        }
        index NumBnd() const
        {
            DNDS_check_throw_info(bnd2node.father, "bnd2node not initialized");
            return bnd2node.father->Size();
        }

        index NumNodeGhost() const
        {
            DNDS_check_throw_info(coords.son, "coords not initialized");
            return coords.son->Size();
        }
        index NumCellGhost() const
        {
            DNDS_check_throw_info(cell2node.son, "cell2node not initialized");
            return cell2node.son->Size();
        }
        index NumFaceGhost() const
        {
            DNDS_check_throw_info(face2node.son, "face2node not initialized");
            return face2node.son->Size();
        }
        index NumBndGhost() const
        {
            DNDS_check_throw_info(bnd2node.son, "bnd2node not initialized");
            return bnd2node.son->Size();
        }

        index NumNodeProc() const
        {
            DNDS_check_throw_info(coords.father && coords.son, "coords not initialized");
            return coords.Size();
        }
        index NumCellProc() const
        {
            DNDS_check_throw_info(cell2node.father && cell2node.son, "cell2node not initialized");
            return cell2node.Size();
        }
        index NumFaceProc() const
        {
            DNDS_check_throw_info(face2node.father && face2node.son, "face2node not initialized");
            return face2node.Size();
        }
        index NumBndProc() const
        {
            DNDS_check_throw_info(bnd2node.father && bnd2node.son, "bnd2node not initialized");
            return bnd2node.Size();
        }

        /// @warning must collectively call
        index NumCellGlobal()
        {
            DNDS_check_throw_info(cell2node.father, "cell2node not initialized");
            return cell2node.father->globalSize();
        }
        /// @warning must collectively call
        index NumNodeGlobal()
        {
            DNDS_check_throw_info(coords.father, "coords not initialized");
            return coords.father->globalSize();
        }
        /// @warning must collectively call
        index NumFaceGlobal()
        {
            DNDS_check_throw_info(face2node.father, "face2node not initialized");
            return face2node.father->globalSize();
        }
        /// @warning must collectively call
        index NumBndGlobal()
        {
            DNDS_check_throw_info(bnd2node.father, "bnd2node not initialized");
            return bnd2node.father->globalSize();
        }

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
        void _detail_GetCoords(const tC2n &c2n, tSmallCoords &cs)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                if (adjPrimaryState == Adj_PointToGlobal)
                {
                    DNDS_assert(cell2node.isGlobal() && adjPrimaryState == Adj_PointToGlobal);
                    iNode = NodeIndexGlobal2Local(iNode), DNDS_assert_info(iNode >= 0, "iNode not found in main/ghost pair");
                }
                cs(EigenAll, i) = coords[iNode];
            }
        }

        /**
         * @brief directly load coords; gets faulty if isPeriodic!
         */
        template <class tC2n, class tCoordExt>
        void _detail_GetCoords(const tC2n &c2n, tSmallCoords &cs, tCoordExt &coo)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                if (adjPrimaryState == Adj_PointToGlobal)
                {
                    DNDS_assert(cell2node.isGlobal() && adjPrimaryState == Adj_PointToGlobal);
                    iNode = NodeIndexGlobal2Local(iNode), DNDS_assert_info(iNode >= 0, "iNode not found in main/ghost pair");
                }
                cs(EigenAll, i) = coo[iNode];
            }
        }

        /**
         * @brief specially for periodicity
         */
        template <class tC2n, class tC2nPbi>
        void _detail_GetCoordsOnElem(const tC2n &c2n, const tC2nPbi &c2nPbi, tSmallCoords &cs)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                if (adjPrimaryState == Adj_PointToGlobal)
                {
                    DNDS_assert(cell2node.isGlobal() && adjPrimaryState == Adj_PointToGlobal);
                    iNode = NodeIndexGlobal2Local(iNode), DNDS_assert_info(iNode >= 0, "iNode not found in main/ghost pair");
                }
                cs(EigenAll, i) = periodicInfo.GetCoordByBits(coords[iNode], c2nPbi[i]);
            }
        }

        /**
         * @brief specially for periodicity
         */
        template <class tC2n, class tC2nPbi, class tCoordExt>
        void _detail_GetCoordsOnElem(const tC2n &c2n, const tC2nPbi &c2nPbi, tSmallCoords &cs, tCoordExt &coo)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                if (adjPrimaryState == Adj_PointToGlobal)
                {
                    DNDS_assert(cell2node.isGlobal() && adjPrimaryState == Adj_PointToGlobal);
                    iNode = NodeIndexGlobal2Local(iNode), DNDS_assert_info(iNode >= 0, "iNode not found in main/ghost pair");
                }
                cs(EigenAll, i) = periodicInfo.GetCoordByBits(coo[iNode], c2nPbi[i]);
            }
        }

        void GetCoordsOnCell(index iCell, tSmallCoords &cs)
        {
            if (!isPeriodic)
                _detail_GetCoords(cell2node[iCell], cs);
            else
                _detail_GetCoordsOnElem(cell2node[iCell], cell2nodePbi[iCell], cs);
        }

        void GetCoordsOnCell(index iCell, tSmallCoords &cs, tCoordPair &coo)
        {
            if (!isPeriodic)
                _detail_GetCoords(cell2node[iCell], cs, coo);
            else
                _detail_GetCoordsOnElem(cell2node[iCell], cell2nodePbi[iCell], cs, coo);
        }

        void GetCoordsOnFace(index iFace, tSmallCoords &cs)
        {
            if (!isPeriodic)
                _detail_GetCoords(face2node[iFace], cs);
            else
                _detail_GetCoordsOnElem(face2node[iFace], face2nodePbi[iFace], cs);
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

        tPoint GetCoordWallDistOnCell(index iCell, rowsize ic2n)
        {
            if (!isPeriodic)
                return nodeWallDist[cell2node(iCell, ic2n)];
            return periodicInfo.GetVectorByBits<3, 1>(nodeWallDist[cell2node(iCell, ic2n)], cell2nodePbi(iCell, ic2n));
        }

        tPoint GetCoordWallDistOnFace(index iFace, rowsize if2n)
        {
            if (!isPeriodic)
                return nodeWallDist[face2node(iFace, if2n)];
            return periodicInfo.GetVectorByBits<3, 1>(nodeWallDist[face2node(iFace, if2n)], face2nodePbi(iFace, if2n));
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

        void WriteSerialize(Serializer::SerializerBaseSSP serializerP, const std::string &name);
        void ReadSerialize(Serializer::SerializerBaseSSP serializerP, const std::string &name);

        /**
         * @brief Reads mesh from an H5 serializer using even-split distribution, then
         * repartitions via ParMetis for locality.
         *
         * @details
         * This method enables reading a mesh written with any number of MPI ranks
         * into any number of MPI ranks, without requiring the original partition.
         *
         * Flow:
         * 1. Even-split read of all primary arrays (coords, cell2node, cellElemInfo,
         *    bnd2node, bndElemInfo, cell2nodePbi, bnd2nodePbi, origIndex arrays).
         * 2. Build distributed node2cell and cell2cell (node-neighbor) using existing
         *    RecoverNode2CellAndNode2Bnd() + RecoverCell2CellAndBnd2Cell().
         * 3. Filter cell2cell to facial neighbors only (shared O1 vertices >= dim).
         * 4. Call ParMETIS_V3_PartKway on the distributed facial graph.
         * 5. Redistribute cells, nodes, and boundaries to the new partition
         *    via ArrayTransformer push-based transfer.
         * 6. Set adjPrimaryState = Adj_PointToGlobal.
         *
         * After return, the caller should proceed with the standard rebuild sequence:
         *   RecoverNode2CellAndNode2Bnd -> RecoverCell2CellAndBnd2Cell ->
         *   BuildGhostPrimary -> AdjGlobal2LocalPrimary -> ...
         *
         * @param serializerP  H5 serializer (must be collective / non-per-rank)
         * @param name         Path name in the serializer hierarchy
         * @param partitionOptions  ParMetis partitioning options
         */
        void ReadSerializeAndDistribute(
            Serializer::SerializerBaseSSP serializerP,
            const std::string &name,
            const PartitionOptions &partitionOptions);

    private:
        // --- ReadSerializeAndDistribute sub-steps ---

        /// Read scalar metadata and all primary arrays with H5 even-split.
        /// Sets adjPrimaryState = Adj_PointToGlobal.
        void ReadDistributed_EvenSplitRead(
            Serializer::SerializerBaseSSP serializerP);

        /// Build facial cell2cell from the even-split data.
        ssp<tAdj::element_type> ReadDistributed_BuildFacialCell2Cell();

        /// Run ParMetis on the facial cell2cell graph.
        std::vector<MPI_int> ReadDistributed_PartitionParMetis(
            const ssp<tAdj::element_type> &cell2cellFacial,
            const PartitionOptions &partitionOptions);

        /// Derive node and bnd partitions from cell partition.
        struct EntityPartitions
        {
            std::vector<MPI_int> cellPartition;
            std::vector<MPI_int> nodePartition;
            std::vector<MPI_int> bndPartition;
        };
        EntityPartitions ReadDistributed_DeriveEntityPartitions(
            std::vector<MPI_int> cellPartition);

        /// Redistribute all primary arrays to the new partition.
        void ReadDistributed_Redistribute(
            const EntityPartitions &partitions);

        /// Legacy implementation preserved for reference/fallback.
        void ReadDistributed_RedistributeLegacy(
            const EntityPartitions &partitions);

    public:
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
            double t, MPI_Comm commDup);

        void SetHDF5OutSetting(size_t chunkSiz, int deflateLevel, bool coll_on_data, bool coll_on_meta)
        {
            hdf5OutSetting.chunkSize = chunkSiz;
            hdf5OutSetting.deflateLevel = deflateLevel;
            hdf5OutSetting.coll_on_data = coll_on_data;
            hdf5OutSetting.coll_on_meta = coll_on_meta;
        }

        void PrintMeshCGNS(std::string fname, const t_FBCID_2_Name &fbcid2name, const std::vector<std::string> &allNames);

        struct WallDistOptions
        {
            int subdivide_quad = 1;
            int method = 0;
            int wallDistExecution = 0;
            real minWallDist = 1e-10;
            int verbose = 0;
            WallDistOptions() {} //? why = default is not working

            DNDS_DECLARE_CONFIG(WallDistOptions)
            {
                // clang-format off
                DNDS_FIELD(subdivide_quad, "Subdivide quads for wall distance computation",
                           DNDS::Config::range(0));
                DNDS_FIELD(method, "Wall distance computation method (0: brute, 1: tree)",
                           DNDS::Config::range(0));
                DNDS_FIELD(wallDistExecution, "MPI concurrency (0: all parallel, 1: serial, >1: batched N ranks)",
                           DNDS::Config::range(0));
                DNDS_FIELD(minWallDist, "Minimum wall distance clamp",
                           DNDS::Config::range(0.0));
                DNDS_FIELD(verbose, "Verbosity level for wall distance computation",
                           DNDS::Config::range(0));
                // clang-format on
            }
        };
        void BuildNodeWallDist(const std::function<bool(Geom::t_index)> &fBndIsWall, WallDistOptions options = WallDistOptions{});

        template <class F>
        void op_on_device_arrays(F &&f)
        {
            for_each_member_list(this->device_array_list_primary(), f);
            if (adjFacialState && face2cell.isBuilt())
                for_each_member_list(this->device_array_list_facial(), f);
            if (adjC2FState && cell2face.isBuilt())
                for_each_member_list(this->device_array_list_C2F(), f);
            if (adjN2CBState && node2cell.isBuilt())
                for_each_member_list(this->device_array_list_N2CB(), f);
        }

        template <typename F>
        void for_each_device_member(F &&f)
        {
            op_on_device_arrays(std::forward<F>(f));
        }

        // to_device(), to_host(), device() are provided by
        // DeviceTransferable<UnstructuredMesh> via for_each_device_member().

        template <DeviceBackend B>
        using t_deviceView = UnstructuredMeshDeviceView<B>;

        template <DeviceBackend B>
        auto deviceView()
        {
            return t_deviceView<B>(*this, UnInitIndex);
        }

        index getArrayBytes()
        {
            index bytes = 0;
            auto acuumulate_bytes_arr = [&](auto &v)
            {
                if (v.ref.father)
                    bytes += v.ref.father->FullSizeBytes();
                if (v.ref.son)
                    bytes += v.ref.son->FullSizeBytes();
            };
            for_each_member_list(
                this->device_array_list_primary(),
                acuumulate_bytes_arr);
            for_each_member_list(
                this->device_array_list_facial(),
                acuumulate_bytes_arr);
            for_each_member_list(
                this->device_array_list_C2F(),
                acuumulate_bytes_arr);
            for_each_member_list(
                this->device_array_list_N2CB(),
                acuumulate_bytes_arr);
            MPI::AllreduceOneIndex(bytes, MPI_SUM, mpi);
            return bytes;
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

    struct PartitionOptions
    {
        std::string metisType = "KWAY";
        int metisUfactor = 20;
        int metisSeed = 0;
        int edgeWeightMethod = 0; // 0:no weight 1: faceSize weight
        int metisNcuts = 3;

        DNDS_DECLARE_CONFIG(PartitionOptions)
        {
            // clang-format off
            DNDS_FIELD(metisType, "METIS partitioning method",
                       DNDS::Config::enum_values({"KWAY", "RB"}));
            DNDS_FIELD(metisUfactor, "METIS imbalance factor (ufactor)",
                       DNDS::Config::range(1));
            DNDS_FIELD(metisSeed, "METIS random seed");
            DNDS_FIELD(edgeWeightMethod, "Edge weight method (0: none, 1: face size)",
                       DNDS::Config::range(0, 1));
            DNDS_FIELD(metisNcuts, "Number of cuts for METIS to try",
                       DNDS::Config::range(1));
            // clang-format on
        }
    };

    struct UnstructuredMeshSerialRW
    {
        using PartitionOptions = Geom::PartitionOptions; // backward compatibility alias

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
        tAdj1 cell2cellOrigSerial;         // created through reading
        tAdj1 node2nodeOrigSerial;         // created through reading
        tAdj1 bnd2bndOrigSerial;           // created through reading
        tPbi cell2nodePbiSerial;           // created through reading-Deduplicate
        tPbi bnd2nodePbiSerial;            // created through reading-Deduplicate
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

        /**
         * @brief directly load coords; gets faulty if isPeriodic!, analog of mesh's method
         */
        template <class tC2n, class tCoordExt>
        void _detail_GetCoordsSerial(const tC2n &c2n, tSmallCoords &cs, tCoordExt &coo)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                cs(EigenAll, i) = coo->operator[](iNode);
            }
        }

        /**
         * @brief specially for periodicity, analog of mesh's method
         */
        template <class tC2n, class tC2nPbi, class tCoordExt>
        void _detail_GetCoordsOnElemSerial(const tC2n &c2n, const tC2nPbi &c2nPbi, tSmallCoords &cs, tCoordExt &coo)
        {
            cs.resize(Eigen::NoChange, c2n.size());
            for (rowsize i = 0; i < c2n.size(); i++)
            {
                index iNode = c2n[i];
                cs(EigenAll, i) = mesh->periodicInfo.GetCoordByBits(coo->operator[](iNode), c2nPbi[i]);
            }
        }
        /**
         * @brief analog of mesh's method
         */
        void GetCoordsOnCellSerial(index iCell, tSmallCoords &cs, tCoord &coo)
        {
            if (!mesh->isPeriodic)
                _detail_GetCoordsSerial((*cell2nodeSerial)[iCell], cs, coo);
            else
                _detail_GetCoordsOnElemSerial((*cell2nodeSerial)[iCell], (*cell2nodePbiSerial)[iCell], cs, coo);
        }

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
        void ReadFromCGNSSerial(const std::string &fName, const t_FBCName_2_ID &FBCName_2_ID);

        auto ReadFromCGNSSerial(const std::string &fName)
        {
            AutoAppendName2ID appended_name2id;
            this->ReadFromCGNSSerial(fName, appended_name2id);
            return appended_name2id;
        }

        void ReadFromOpenFOAMAndConvertSerial(const std::string &fName, const std::map<std::string, std::string> &nameMapping, const t_FBCName_2_ID &FBCName_2_ID = FBC_Name_2_ID_Default);

        void Deduplicate1to1Periodic(real searchEps = 1e-8);

        // void InterpolateTopology();

        /**
         * \brief build cell2cell topology, with node-neighbors included
         * \todo add support for only face-neighbors
         */
        void BuildCell2Cell(); // For cell based purpose

        void BuildNode2Node(); // For node based purpose //!not yet implemented

        void MeshPartitionCell2Cell(const PartitionOptions &options);

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
            cell2cellOrigSerial.reset();
            node2nodeOrigSerial.reset();
            bnd2bndOrigSerial.reset();
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
