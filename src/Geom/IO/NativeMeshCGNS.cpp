#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/IO/NativeMeshPeriodicImport.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/IO/CGNS.hpp"
#include <pcgnslib.h>
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include "DNDS/Core/OMP.hpp"
#include <limits>
#include <set>
#include <atomic>

namespace DNDS::Geom
{
    namespace
    {
        std::atomic<uint64_t> nativeCopyBulkCalls{0}, nativeCopyBulkRows{0};
        std::atomic<uint64_t> nativeCopyRowCalls{0}, nativeCopyRowRows{0};

        void NativeRequire(bool condition)
        {
            DNDS_check_throw_info(condition, "invalid or unsupported native CGNS input");
        }
        index LocalCGNSNode(cgsize_t node, index count)
        {
            NativeRequire(node >= 1 && node <= count);
            return index(node) - 1;
        }
        void CheckCGNS(int result)
        {
            DNDS_check_throw_info(result == CG_OK, std::string("CGNS: ") + cg_get_error());
        }
#define NATIVE_CGNS_CHECK(call) CheckCGNS(call)
        struct CGNSFileGuard
        {
            int handle;
            ~CGNSFileGuard()
            {
                if (handle >= 0)
                    cg_close(handle);
            }
        };

        // Serial parser's exclusive construction form. The same mathematical
        // reader fills fixed tables directly and packs each variable CSR once.
        // No borrowed row survives ResizeRow, Compress or Commit.
        template <class T, size_t Width = 0, bool Coordinates = false>
        class Draft
        {
            std::optional<Runtime::RowBuilder> _draft;
            Runtime::Array<T> _array;
            std::optional<typename Runtime::Array<T>::WriteLease> _write;
            Runtime::ArrayView<T> _view;
            T *Pointer(index row) const
            {
                DNDS_check_throw_info(row >= 0 && size_t(row) < size_t(Size()), "native CGNS row out of bounds");
                return _draft ? _draft->template Row<T>(size_t(row)).data : _view.Row(size_t(row));
            }
            struct Row : Runtime::RowSpan<T>
            {
                using Runtime::RowSpan<T>::RowSpan;
                T *operator->() const { return this->data(); }
            };

        public:
            Draft() { Resize(0); }
            index Size() const { return index(_draft ? _draft->RowCount() : _view.RowCount()); }
            rowsize RowSize(index row) const
            {
                DNDS_check_throw_info(row >= 0 && row < Size(), "native CGNS row out of bounds");
                return rowsize(_draft ? _draft->RowSize(size_t(row)) : _view.RowSize(size_t(row)));
            }
            void Resize(index rows)
            {
                DNDS_check_throw_info(rows >= 0, "negative native CGNS row extent");
                _write.reset();
                _view = {};
                if constexpr (Width)
                {
                    _array = Runtime::Array<T>(Runtime::RowLayout::Dense(size_t(rows), Width));
                    _write.emplace(_array.Write());
                    _view = _write->View();
                    std::fill_n(_view.data, size_t(rows) * Width, T{});
                }
                else
                    _draft.emplace(size_t(rows), sizeof(T));
            }
            void ResizeRow(index row, size_t width)
            {
                DNDS_check_throw_info(_draft, "native CGNS draft shape already published");
                _draft->ResizeRow(size_t(row), width);
            }
            T &operator()(index row, size_t column) const
            {
                DNDS_check_throw_info(column < size_t(RowSize(row)), "native CGNS column out of bounds");
                return Pointer(row)[column];
            }
            auto operator[](index row) const
            {
                if constexpr (Coordinates)
                    return Eigen::Map<tPoint>(Pointer(row));
                else
                    return Row(Pointer(row), size_t(RowSize(row)));
            }
            void Compress()
            {
                if (!_draft)
                    return;
                _array = Runtime::Array<T>::FromSharedStorage(_draft->Commit());
                _draft.reset();
                _write.emplace(_array.Write());
                _view = _write->View();
            }
            Runtime::Array<T> Commit()
            {
                Compress();
                _write.reset();
                _view = {};
                return std::move(_array);
            }
        };
        using DraftCoords = std::shared_ptr<Draft<real, 3, true>>;
        using DraftAdj = std::shared_ptr<Draft<index>>;
        using DraftAdj1 = std::shared_ptr<Draft<index, 1>>;
        using DraftAdj2 = std::shared_ptr<Draft<index, 2>>;
        using DraftInfo = std::shared_ptr<Draft<ElemInfo, 1>>;
    }
    // =================================================================
    // File-local helpers for ReadFromCGNSSerial
    // =================================================================
    namespace
    {
        /**
         * \brief Deduplicate shared nodes across CGNS zones via DFS on the
         *        zone connectivity graph.
         *
         * \param[in]  ZoneCoords             Per-zone coordinate arrays.
         * \param[in]  ZoneConnect            Per-zone connectivity point lists (1-based).
         * \param[in]  ZoneConnectDonor       Per-zone donor point lists (1-based).
         * \param[in]  ZoneConnectTargetIZone Per-zone target zone indices.
         * \param[out] coordSerial            Assembled coordinate array (resized and filled).
         * \return     NodeOld2New mapping from concatenated zone-local indices to
         *             assembled (deduplicated) global indices, plus ZoneNodeStarts.
         */
#ifdef DNDS_USE_CGNS_ZONE_DFS_LEGACY
        std::pair<std::vector<DNDS::index>, std::vector<DNDS::index>>
        AssembleZoneNodes(
            const std::vector<DraftCoords> &ZoneCoords,
            const std::vector<std::vector<std::vector<cgsize_t>>> &ZoneConnect,
            const std::vector<std::vector<std::vector<cgsize_t>>> &ZoneConnectDonor,
            const std::vector<std::vector<int>> &ZoneConnectTargetIZone,
            DraftCoords &coordSerial,
            int dim)
        {
            std::vector<DNDS::index> ZoneNodeSizes(ZoneCoords.size());
            std::vector<DNDS::index> ZoneNodeStarts(ZoneCoords.size() + 1);
            ZoneNodeStarts[0] = 0;
            for (size_t i = 0; i < ZoneNodeSizes.size(); i++)
            {
                ZoneNodeSizes[i] = ZoneCoords[i]->Size();
                NativeRequire(ZoneNodeSizes[i] >= 0 && ZoneNodeSizes[i] <= std::numeric_limits<index>::max() - ZoneNodeStarts[i]);
                ZoneNodeStarts[i + 1] = ZoneNodeStarts[i] + ZoneNodeSizes[i];
            }

            std::vector<DNDS::index> NodeOld2New(ZoneNodeStarts.back(), -1);
            DNDS::index cTop = 0;

            // DFS over zone connectivity graph to deduplicate shared nodes
            std::set<int> zonesLeft;
            for (size_t iGZ = 0; iGZ < ZoneNodeSizes.size(); iGZ++)
                zonesLeft.insert(static_cast<int>(iGZ));
            std::vector<int> zonesFront;
            zonesFront.push_back(0);
            NativeRequire(!ZoneNodeSizes.empty());

            while (!zonesFront.empty())
            {
                int iGZ = zonesFront.back();
                zonesFront.pop_back();
                zonesLeft.erase(iGZ);
                for (size_t iOther = 0; iOther < ZoneConnect.at(iGZ).size(); iOther++)
                {
                    int iGZOther = ZoneConnectTargetIZone.at(iGZ).at(iOther);
                    if (zonesLeft.count(iGZOther))
                    {
                        zonesFront.push_back(iGZOther);
                    }
                    else
                    {
                        for (size_t iNode = 0; iNode < ZoneConnect.at(iGZ).at(iOther).size(); iNode++)
                        {
                            NodeOld2New.at(ZoneNodeStarts.at(iGZ) + ZoneConnect.at(iGZ).at(iOther).at(iNode) - 1) //! note ZoneConnect is 1-based
                                = NodeOld2New.at(ZoneNodeStarts.at(iGZOther) + ZoneConnectDonor.at(iGZ).at(iOther).at(iNode) - 1);
                            NativeRequire(NodeOld2New.at(ZoneNodeStarts.at(iGZ) + ZoneConnect.at(iGZ).at(iOther).at(iNode) - 1) >= 0);
                        }
                    }
                }
                for (DNDS::index iNode = ZoneNodeStarts.at(iGZ); iNode < ZoneNodeStarts.at(iGZ + 1); iNode++)
                {
                    if (NodeOld2New.at(iNode) < 0)
                    {
                        NodeOld2New.at(iNode) = cTop;
                        cTop++;
                    }
                }
            }
            DNDS::log() << "CGNS === Assembled Zones have NNodes " << cTop << std::endl;
            DNDS_check_throw_info(zonesLeft.empty(), "Did not reach all the zones, might missing zone-connectivities");

            coordSerial->Resize(cTop);
            for (DNDS::index i = 0; i < coordSerial->Size(); i++)
                coordSerial->operator[](i).setConstant(DNDS::UnInitReal);

            for (size_t iGZ = 0; iGZ < ZoneNodeSizes.size(); iGZ++)
            {
                for (DNDS::index iNode = ZoneNodeStarts.at(iGZ); iNode < ZoneNodeStarts.at(static_cast<int>(iGZ) + 1); iNode++)
                {
                    auto coordC = (*ZoneCoords[iGZ])[iNode - ZoneNodeStarts.at(iGZ)];
                    real dist = ((*coordSerial)[NodeOld2New.at(iNode)] - coordC).norm();
                    if (!DNDS::IsUnInitReal((*coordSerial)[NodeOld2New.at(iNode)][0]))
                        DNDS_check_throw_info(dist < 1e-10,
                                              "Not same points on the connection, distance is " + std::to_string(dist));
                    (*coordSerial)[NodeOld2New.at(iNode)] = coordC;
                }
            }

            return {std::move(NodeOld2New), std::move(ZoneNodeStarts)};
        }
#endif // DNDS_USE_CGNS_ZONE_DFS_LEGACY

#ifndef DNDS_USE_CGNS_ZONE_DFS_LEGACY
        /**
         * \brief Union-find helper: find with path compression.
         *
         * Operates on a flat index space (concatenated zone-local nodes).
         */
        static DNDS::index uf_find(std::vector<DNDS::index> &parent, DNDS::index x)
        {
            DNDS::index root = x;
            while (parent[root] != root)
                root = parent[root];
            while (parent[x] != x)
            {
                DNDS::index next = parent[x];
                parent[x] = root;
                x = next;
            }
            return root;
        }

        /// Union-find helper: union by the deterministic zone-major/node-major rule.
        /// The representative with the smaller flat index wins.
        static void uf_union(std::vector<DNDS::index> &parent, DNDS::index a, DNDS::index b)
        {
            DNDS::index ra = uf_find(parent, a);
            DNDS::index rb = uf_find(parent, b);
            if (ra != rb)
            {
                // Tie-break: smaller flat index is the representative.
                // This preserves the same representative order as the old DFS
                // path when equivalences are symmetric and zones are visited
                // in ascending zone index order.
                if (ra < rb)
                    parent[rb] = ra;
                else
                    parent[ra] = rb;
            }
        }

        /**
         * \brief Union-find variant of AssembleZoneNodes.
         *
         * Differs from the old DFS path in two ways:
         * 1. Collects all (zone,node) <-> (donorZone,donorNode) equivalences
         *    from ALL zone connectivity records up front, so one-sided
         *    Abutting1to1 links are tolerated.
         * 2. Uses deterministic union-find (smaller flat index wins) rather
         *    than DFS visit order for deduplication.
         *
         * Compact IDs are assigned by a zone-major / node-major scan over the
         * union-find representatives, so when the old DFS path already
         * deduplicated correctly, the new numbering matches the old numbering.
         *
         * \warning The legacy DFS path can be restored by defining
         *          DNDS_USE_CGNS_ZONE_DFS_LEGACY.  This union path may change
         *          valid node numbering in CGNS files that were previously
         *          processed correctly by the DFS path.  This can reorder cells
         *          and invalidate downstream data keyed by node numbering.
         */
        std::pair<std::vector<DNDS::index>, std::vector<DNDS::index>>
        AssembleZoneNodesUnion(
            const std::vector<DraftCoords> &ZoneCoords,
            const std::vector<std::vector<std::vector<cgsize_t>>> &ZoneConnect,
            const std::vector<std::vector<std::vector<cgsize_t>>> &ZoneConnectDonor,
            const std::vector<std::vector<int>> &ZoneConnectTargetIZone,
            DraftCoords &coordSerial,
            int dim)
        {
            std::vector<DNDS::index> ZoneNodeSizes(ZoneCoords.size());
            std::vector<DNDS::index> ZoneNodeStarts(ZoneCoords.size() + 1);
            ZoneNodeStarts[0] = 0;
            for (size_t i = 0; i < ZoneNodeSizes.size(); i++)
            {
                ZoneNodeSizes[i] = ZoneCoords[i]->Size();
                NativeRequire(ZoneNodeSizes[i] >= 0 && ZoneNodeSizes[i] <= std::numeric_limits<index>::max() - ZoneNodeStarts[i]);
                ZoneNodeStarts[i + 1] = ZoneNodeStarts[i] + ZoneNodeSizes[i];
            }

            DNDS::index totalRawNodes = ZoneNodeStarts.back();
            std::vector<DNDS::index> parent(totalRawNodes);
            for (DNDS::index i = 0; i < totalRawNodes; i++)
                parent[i] = i;

            // Collect all equivalences from zone connectivity.
            for (size_t iGZ = 0; iGZ < ZoneConnect.size(); iGZ++)
            {
                for (size_t iOther = 0; iOther < ZoneConnect[iGZ].size(); iOther++)
                {
                    int iGZOther = ZoneConnectTargetIZone[iGZ][iOther];
                    for (size_t iNode = 0; iNode < ZoneConnect[iGZ][iOther].size(); iNode++)
                    {
                        DNDS::index flatA = ZoneNodeStarts[iGZ] + ZoneConnect[iGZ][iOther][iNode] - 1; // 1-based → 0-based
                        DNDS::index flatB = ZoneNodeStarts[iGZOther] + ZoneConnectDonor[iGZ][iOther][iNode] - 1;
                        NativeRequire(flatA >= 0 && flatA < totalRawNodes);
                        NativeRequire(flatB >= 0 && flatB < totalRawNodes);
                        uf_union(parent, flatA, flatB);
                    }
                }
            }

            // Assign compact IDs by zone-major / node-major scan.
            // Each representative gets a compact ID on first encounter.
            std::vector<DNDS::index> NodeOld2New(totalRawNodes, -1);
            DNDS::index cTop = 0;
            for (size_t iGZ = 0; iGZ < ZoneNodeSizes.size(); iGZ++)
            {
                for (DNDS::index iNode = ZoneNodeStarts[iGZ]; iNode < ZoneNodeStarts[iGZ + 1]; iNode++)
                {
                    DNDS::index rep = uf_find(parent, iNode);
                    if (NodeOld2New[rep] < 0)
                    {
                        NodeOld2New[rep] = cTop;
                        cTop++;
                    }
                    NodeOld2New[iNode] = NodeOld2New[rep];
                }
            }

            DNDS::log() << "CGNS === (Union-Find) Assembled Zones have NNodes " << cTop << std::endl;

            coordSerial->Resize(cTop);
            for (DNDS::index i = 0; i < coordSerial->Size(); i++)
                coordSerial->operator[](i).setConstant(DNDS::UnInitReal);

            for (size_t iGZ = 0; iGZ < ZoneNodeSizes.size(); iGZ++)
            {
                for (DNDS::index iNode = ZoneNodeStarts[iGZ]; iNode < ZoneNodeStarts[static_cast<int>(iGZ) + 1]; iNode++)
                {
                    auto coordC = (*ZoneCoords[iGZ])[iNode - ZoneNodeStarts[iGZ]];
                    real dist = ((*coordSerial)[NodeOld2New[iNode]] - coordC).norm();
                    if (!DNDS::IsUnInitReal((*coordSerial)[NodeOld2New[iNode]][0]))
                        DNDS_check_throw_info(dist < 1e-10,
                                              "Not same points on the connection, distance is " + std::to_string(dist));
                    (*coordSerial)[NodeOld2New[iNode]] = coordC;
                }
            }

            return {std::move(NodeOld2New), std::move(ZoneNodeStarts)};
        }
#endif // !DNDS_USE_CGNS_ZONE_DFS_LEGACY

        /**
         * \brief Separate zone elements into volume cells and boundary faces.
         *
         * Converts element node indices from zone-local to assembled global
         * (via NodeOld2New), then fills cell2nodeSerial/cellElemInfoSerial and
         * bnd2nodeSerial/bndElemInfoSerial.
         */
        void SeparateVolumeAndBoundaryElements(
            const std::vector<DraftAdj> &ZoneElems,
            const std::vector<DraftInfo> &ZoneElemInfos,
            const std::vector<DNDS::index> &NodeOld2New,
            const std::vector<DNDS::index> &ZoneNodeStarts,
            DraftAdj &cell2nodeSerial,
            DraftInfo &cellElemInfoSerial,
            DraftAdj1 &cell2cellOrigSerial,
            DraftAdj &bnd2nodeSerial,
            DraftInfo &bndElemInfoSerial,
            DraftAdj1 &bnd2bndOrigSerial,
            DraftAdj1 &node2nodeOrigSerial,
            DraftCoords &coordSerial,
            int dim)
        {
            // Convert element node indices to assembled global and count
            DNDS::index nVolElem = 0;
            DNDS::index nBndElem = 0;
            for (size_t iGZ = 0; iGZ < ZoneElems.size(); iGZ++)
            {
                for (DNDS::index iElem = 0; iElem < ZoneElems[iGZ]->Size(); iElem++)
                {
                    for (DNDS::rowsize j = 0; j < ZoneElems[iGZ]->RowSize(iElem); j++)
                    {
                        const auto local = ZoneElems[iGZ]->operator()(iElem, j);
                        NativeRequire(local >= 0 && local < ZoneNodeStarts[iGZ + 1] - ZoneNodeStarts[iGZ]);
                        ZoneElems[iGZ]->operator()(iElem, j) = NodeOld2New.at(size_t(ZoneNodeStarts[iGZ] + local));
                    }
                    const auto type = ZoneElemInfos[iGZ]->operator()(iElem, 0).getElemType();
                    if (type == Elem::UnknownElem)
                    {
                        NativeRequire(ZoneElems[iGZ]->RowSize(iElem) == 0);
                        continue; // legal gaps between CGNS section number ranges
                    }
                    auto Elem = Elem::Element{type};
                    if (Elem.GetDim() == dim)
                        nVolElem++;
                    if (Elem.GetDim() == dim - 1 && !FaceIDIsTrueInternal(ZoneElemInfos[iGZ]->operator()(iElem, 0).zone))
                        nBndElem++;
                }
            }

            cell2nodeSerial->Resize(nVolElem);
            bnd2nodeSerial->Resize(nBndElem);
            cellElemInfoSerial->Resize(nVolElem);
            cell2cellOrigSerial->Resize(nVolElem);
            bndElemInfoSerial->Resize(nBndElem);
            node2nodeOrigSerial->Resize(coordSerial->Size());
            bnd2bndOrigSerial->Resize(nBndElem);
            nVolElem = 0;
            nBndElem = 0;
            for (size_t iGZ = 0; iGZ < ZoneElems.size(); iGZ++)
            {
                for (DNDS::index iElem = 0; iElem < ZoneElems[iGZ]->Size(); iElem++)
                {
                    const auto type = ZoneElemInfos[iGZ]->operator()(iElem, 0).getElemType();
                    if (type == Elem::UnknownElem)
                    {
                        NativeRequire(ZoneElems[iGZ]->RowSize(iElem) == 0);
                        continue; // legal gaps between CGNS section number ranges
                    }
                    auto Elem = Elem::Element{type};
                    if (Elem.GetDim() == dim)
                    {
                        cell2nodeSerial->ResizeRow(nVolElem, ZoneElems[iGZ]->RowSize(iElem));
                        for (DNDS::rowsize j = 0; j < ZoneElems[iGZ]->RowSize(iElem); j++)
                            cell2nodeSerial->operator()(nVolElem, j) = ZoneElems[iGZ]->operator()(iElem, j);
                        cellElemInfoSerial->operator()(nVolElem, 0) = ZoneElemInfos[iGZ]->operator()(iElem, 0);
                        nVolElem++;
                    }
                    if (Elem.GetDim() == dim - 1 && !FaceIDIsTrueInternal(ZoneElemInfos[iGZ]->operator()(iElem, 0).zone))
                    {
                        bnd2nodeSerial->ResizeRow(nBndElem, ZoneElems[iGZ]->RowSize(iElem));
                        for (DNDS::rowsize j = 0; j < ZoneElems[iGZ]->RowSize(iElem); j++)
                            bnd2nodeSerial->operator()(nBndElem, j) = ZoneElems[iGZ]->operator()(iElem, j);
                        bndElemInfoSerial->operator()(nBndElem, 0) = ZoneElemInfos[iGZ]->operator()(iElem, 0);
                        nBndElem++;
                    }
                }
            }
            log() << "CGNS === Vol Elem [  " << nVolElem << "  ]"
                  << ", "
                  << " Bnd Elem [  " << nBndElem << "  ]" << std::endl;

            bnd2nodeSerial->Compress();
            cell2nodeSerial->Compress();
        }

        /**
         * \brief Build bnd2cellSerial: for each boundary face, find its parent
         *        volume cell by vertex-set inclusion.
         *
         * Uses node-to-boundary index for linear complexity.
         */
        void BuildBnd2CellSerial(
            DraftAdj2 &bnd2cellSerial,
            const DraftAdj &bnd2nodeSerial,
            const DraftAdj &cell2nodeSerial,
            const DraftInfo &cellElemInfoSerial,
            const DraftInfo &bndElemInfoSerial,
            const DraftCoords &coordSerial)
        {
            bnd2cellSerial->Resize(bnd2nodeSerial->Size());
            for (DNDS::index iB = 0; iB < bnd2cellSerial->Size(); iB++)
                (*bnd2cellSerial)[iB][0] = (*bnd2cellSerial)[iB][1] = DNDS::UnInitIndex;

            // Build node-to-boundary index for linear complexity
            std::vector<std::vector<DNDS::index>> node2bnd(coordSerial->Size());
            std::vector<DNDS::rowsize> node2bndSiz(coordSerial->Size(), 0);
            std::vector<std::vector<DNDS::index>> bnd2nodeSorted(size_t(bnd2nodeSerial->Size()));
            for (DNDS::index iBFace = 0; iBFace < bnd2nodeSerial->Size(); iBFace++)
            {
                const auto vertices = Elem::Element{(*bndElemInfoSerial)(iBFace, 0).getElemType()}.GetNumVertices();
                auto &row = bnd2nodeSorted[size_t(iBFace)];
                row.assign((*bnd2nodeSerial)[iBFace].begin(), (*bnd2nodeSerial)[iBFace].begin() + vertices);
                std::sort(row.begin(), row.end());
            }
            for (DNDS::index iBFace = 0; iBFace < bnd2nodeSerial->Size(); iBFace++)
                for (DNDS::rowsize iN = 0; iN < Elem::Element{(*bndElemInfoSerial)(iBFace, 0).getElemType()}.GetNumVertices(); iN++)
                    node2bndSiz[(*bnd2nodeSerial)(iBFace, iN)]++;
            for (DNDS::index iNode = 0; iNode < static_cast<DNDS::index>(node2bnd.size()); iNode++)
                node2bnd[iNode].reserve(node2bndSiz[iNode]);
            for (DNDS::index iBFace = 0; iBFace < bnd2nodeSerial->Size(); iBFace++)
                for (DNDS::rowsize iN = 0; iN < Elem::Element{(*bndElemInfoSerial)(iBFace, 0).getElemType()}.GetNumVertices(); iN++)
                    node2bnd[(*bnd2nodeSerial)(iBFace, iN)].push_back(iBFace);

            // Search each cell for matching boundary faces
            std::vector<DNDS::index> cell2nodeRow;
            cell2nodeRow.reserve(8);
            for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            {
                auto CellElem = Elem::Element{(*cellElemInfoSerial)[iCell]->getElemType()};
                cell2nodeRow.assign((*cell2nodeSerial)[iCell].begin(), (*cell2nodeSerial)[iCell].begin() + CellElem.GetNumVertices());
                std::sort(cell2nodeRow.begin(), cell2nodeRow.end());
                for (auto iNode : cell2nodeRow)
                {
                    for (auto iB : node2bnd[iNode])
                    {
                        const auto &bnd2nodeRow = bnd2nodeSorted[size_t(iB)];
                        if (std::includes(cell2nodeRow.begin(), cell2nodeRow.end(), bnd2nodeRow.begin(), bnd2nodeRow.end()))
                        {
                            DNDS_check_throw_info(
                                (*bnd2cellSerial)[iB][0] == DNDS::UnInitIndex || (*bnd2cellSerial)[iB][0] == iCell, "bnd2cell not untouched!");
                            (*bnd2cellSerial)[iB][0] = iCell;
                        }
                    }
                }
            }
            for (DNDS::index iB = 0; iB < bnd2cellSerial->Size(); iB++)
                NativeRequire((*bnd2cellSerial)[iB][0] != DNDS::UnInitIndex);
        }
    } // anonymous namespace

    NativeSerialCopyCounters GetNativeSerialCopyCounters()
    {
        return {nativeCopyBulkCalls.load(std::memory_order_relaxed),
                nativeCopyBulkRows.load(std::memory_order_relaxed),
                nativeCopyRowCalls.load(std::memory_order_relaxed),
                nativeCopyRowRows.load(std::memory_order_relaxed)};
    }

    void ResetNativeSerialCopyCounters()
    {
        nativeCopyBulkCalls.store(0, std::memory_order_relaxed);
        nativeCopyBulkRows.store(0, std::memory_order_relaxed);
        nativeCopyRowCalls.store(0, std::memory_order_relaxed);
        nativeCopyRowRows.store(0, std::memory_order_relaxed);
    }

    NativeSerialMeshData ReadNativeCGNSSerial(const std::string &fName, int dimension, const t_FBCName_2_ID &FBCName_2_ID)
    {
        DNDS_check_throw_info(dimension == 2 || dimension == 3, "native CGNS dimension must be 2 or 3");

        // CGNS retains its parallel communicator after cgp_close; even a
        // subsequent cg_open file probe may broadcast on that communicator.
        // This API is local (only the chosen reader calls it), so establish a
        // local context explicitly. CGNS calls remain host-serialized.
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized)
            DNDS_check_throw_info(cgp_mpi_comm(MPI_COMM_SELF) == CG_OK, "cannot select local CGNS reader context");

        int cgErr = CG_OK;

        auto cell2nodeSerial = std::make_shared<Draft<index>>();
        auto bnd2nodeSerial = std::make_shared<Draft<index>>();
        auto coordSerial = std::make_shared<Draft<real, 3, true>>();
        auto cellElemInfoSerial = std::make_shared<Draft<ElemInfo, 1>>();
        auto bndElemInfoSerial = std::make_shared<Draft<ElemInfo, 1>>();
        auto bnd2cellSerial = std::make_shared<Draft<index, 2>>();
        auto cell2cellOrigSerial = std::make_shared<Draft<index, 1>>();
        auto node2nodeOrigSerial = std::make_shared<Draft<index, 1>>();
        auto bnd2bndOrigSerial = std::make_shared<Draft<index, 1>>();

        int cgns_file = -1;
        if (cg_open(fName.c_str(), CG_MODE_READ, &cgns_file))
        {
            std::cerr << fmt::format("cgns file cannot open: [{}]", fName) << std::endl;
            throw std::runtime_error(cg_get_error());
        }
        CGNSFileGuard fileGuard{cgns_file};
        int n_bases = -1;
        if (cg_nbases(cgns_file, &n_bases))
            throw std::runtime_error(cg_get_error());
        DNDS::log() << "CGNS === N bases: " << n_bases << std::endl;
        NativeRequire(n_bases > 0);
        NativeRequire(n_bases == 1);
        std::vector<std::pair<int, int>> Base_Zone;
        std::vector<std::string> BaseNames;
        std::vector<std::string> ZoneNames;
        // std::vector<std::string> ZoneFamilyNames;
        std::vector<std::array<cgsize_t, 9>> ZoneSizes;
        std::vector<DraftCoords> ZoneCoords; //! purely serial
        std::vector<DraftAdj> ZoneElems;
        std::vector<DraftInfo> ZoneElemInfos;
        std::vector<std::vector<std::vector<cgsize_t>>> ZoneConnect;
        std::vector<std::vector<std::vector<cgsize_t>>> ZoneConnectDonor;
        std::vector<std::vector<int>> ZoneConnectTargetIZone;
        /***************************************************************************/
        // TRAVERSE 1
        for (int iBase = 1; iBase <= n_bases; iBase++)
        {

            int cgns_base = -1;
            std::array<char, 48> basename{};
            int celldim = -1;
            int physdim = -1;
            int nzones = -1;

            if (cg_base_read(cgns_file, iBase, basename.data(), &celldim, &physdim))
                throw std::runtime_error(cg_get_error());
            DNDS_check_throw_info(celldim == dimension && (dimension == 2 || dimension == 3) && physdim >= dimension && physdim <= 3,
                                  "CGNS cell/physical dimensions are unsupported");
            if (cg_nzones(cgns_file, iBase, &nzones))
                throw std::runtime_error(cg_get_error());
            for (int iZone = 1; iZone <= nzones; iZone++)
            {
                std::array<char, 48> zonename{};
                std::array<cgsize_t, 9> size{0, 0, 0, 0, 0, 0, 0, 0, 0};
                if (cg_zone_read(cgns_file, iBase, iZone, zonename.data(), size.data()))
                    throw std::runtime_error(cg_get_error());
                ZoneType_t zoneType;
                if (cg_zone_type(cgns_file, iBase, iZone, &zoneType))
                    throw std::runtime_error(cg_get_error());
                NativeRequire(zoneType == Unstructured); //! only supports unstructured
                Base_Zone.emplace_back(iBase, iZone);
                BaseNames.emplace_back(basename.data());
                ZoneNames.emplace_back(zonename.data());

                // !  family name used for Volume condition not used now
                // if (cg_goto(cgns_file, iBase, "Zone_t", iZone, ""))
                //     throw std::runtime_error(cg_get_error());
                // char famname[48]{0};
                // if (cg_famname_read(famname))
                //     throw std::runtime_error(cg_get_error());
                // ZoneFamilyNames.push_back(famname); //* family name used for Volume condition
                //

                NativeRequire(size[0] > 0 && size[1] >= 0 && size[2] >= 0 && size[2] <= size[0]);
                DNDS::index nNodes = size[0];
                DNDS::index nVols = size[1];
                DNDS::index nBVertex = size[2];

                //*** read coords
                {
                    int ncoords;
                    NATIVE_CGNS_CHECK(cg_ncoords(cgns_file, iBase, iZone, &ncoords));
                    // NativeRequire(ncoords == dimension); //!Even if has Z coord, we only use X,Y in 2d

                    ZoneCoords.emplace_back(std::make_shared<decltype(ZoneCoords)::value_type::element_type>());
                    ZoneCoords.back()->Resize(nNodes);
                    std::vector<double> coordsRead(nNodes);
                    std::array<cgsize_t, 3> RMin{1, 0, 0};
                    std::array<cgsize_t, 3> RMax{nNodes, 0, 0};
                    //* READ X
                    NATIVE_CGNS_CHECK(cg_coord_read(cgns_file, iBase, iZone, "CoordinateX", RealDouble, RMin.data(), RMax.data(), coordsRead.data()));
                    OMP::ForIngressRows(0, nNodes, [&](index iNode)
                                        { ZoneCoords.back()->operator[](iNode)[0] = coordsRead.at(iNode); });
                    //* READ Y
                    NATIVE_CGNS_CHECK(cg_coord_read(cgns_file, iBase, iZone, "CoordinateY", RealDouble, RMin.data(), RMax.data(), coordsRead.data()));
                    OMP::ForIngressRows(0, nNodes, [&](index iNode)
                                        { ZoneCoords.back()->operator[](iNode)[1] = coordsRead.at(iNode); });
                    //* READ Z
                    if (dimension == 3)
                    {
                        NATIVE_CGNS_CHECK(cg_coord_read(cgns_file, iBase, iZone, "CoordinateZ", RealDouble, RMin.data(), RMax.data(), coordsRead.data()));
                    }
                    else
                        for (auto &i : coordsRead)
                            i = 0;
                    OMP::ForIngressRows(0, nNodes, [&](index iNode)
                                        { ZoneCoords.back()->operator[](iNode)[2] = coordsRead.at(iNode); });

                    DNDS::log() << "CGNS === Zone " << iZone << " Coords Reading Done" << std::endl;
                }
                // for (DNDS::index i = 0; i < ZoneCoords.back()->Size(); i++)
                //     std::cout << (*ZoneCoords.back())[i].transpose() << std::endl;

                //*** read Elems
                {
                    int nSections;
                    NATIVE_CGNS_CHECK(cg_nsections(cgns_file, iBase, iZone, &nSections));
                    NativeRequire(nSections >= 1);
                    // [[maybe_unused]]: These track the current section's start/end for potential
                    // validation of section contiguity. The original assertion (cend == start - 1) was
                    // removed because CGNS sections may have gaps. Preserved for documentation and
                    // in case stricter validation is needed for specific file formats.
                    [[maybe_unused]] cgsize_t cstart = 0;
                    [[maybe_unused]] cgsize_t cend = 0;
                    cgsize_t maxend = 0;
                    //*Total size
                    for (int iSection = 1; iSection <= nSections; iSection++)
                    {
                        std::array<char, 48> sectionName{};
                        ElementType_t etype;
                        cgsize_t start;
                        cgsize_t end;
                        int nBnd{0}, parentFlag{0};
                        NATIVE_CGNS_CHECK(cg_section_read(cgns_file, iBase, iZone, iSection, sectionName.data(), &etype, &start, &end, &nBnd, &parentFlag));
                        NativeRequire(start >= 1 && end >= start && end - start < INT_MAX);
                        // cgsize_t elemDataSize{0};
                        // NATIVE_CGNS_CHECK(cg_ElementDataSize(cgns_file, iBase, iZone, iSection, &elemDataSize));
                        // NativeRequire(cend == start - 1); //? testing//!not valid!
                        cstart = start;
                        cend = end;
                        maxend = std::max(end, maxend);
                    }
                    ZoneElems.emplace_back(std::make_shared<decltype(ZoneElems)::value_type::element_type>());
                    ZoneElems.back()->Resize(maxend);
                    ZoneElemInfos.emplace_back(std::make_shared<decltype(ZoneElemInfos)::value_type::element_type>());
                    ZoneElemInfos.back()->Resize(maxend);
                    std::vector<bool> occupied(size_t(maxend), false);
                    //*Resize row
                    for (int iSection = 1; iSection <= nSections; iSection++)
                    {
                        std::array<char, 48> sectionName{};
                        ElementType_t etype;
                        cgsize_t start;
                        cgsize_t end;
                        int nBnd{0}, parentFlag{0};
                        NATIVE_CGNS_CHECK(cg_section_read(cgns_file, iBase, iZone, iSection, sectionName.data(), &etype, &start, &end, &nBnd, &parentFlag));
                        cgsize_t elemDataSize{0};
                        NATIVE_CGNS_CHECK(cg_ElementDataSize(cgns_file, iBase, iZone, iSection, &elemDataSize));
                        NativeRequire(elemDataSize >= 0 && start >= 1 && end >= start && end <= maxend && end - start < INT_MAX);
                        for (cgsize_t row = start - 1; row < end; ++row)
                        {
                            NativeRequire(!occupied[size_t(row)]);
                            occupied[size_t(row)] = true;
                        }
                        std::vector<cgsize_t> elemsRead(size_t(elemDataSize), 0);

                        int nElemSec = int(end - start + 1);
                        if (etype == MIXED)
                        {
                            std::vector<cgsize_t> elemStarts(size_t(nElemSec) + 1); //* note size
                            NATIVE_CGNS_CHECK(cg_poly_elements_read(cgns_file, iBase, iZone, iSection, elemsRead.data(), elemStarts.data(), nullptr));
                            NativeRequire(elemStarts.front() == 0 && elemStarts.back() == elemDataSize);
                            for (cgsize_t i = 0; i < nElemSec; i++)
                            {
                                NativeRequire(elemStarts[size_t(i)] >= 0 && elemStarts[size_t(i + 1)] > elemStarts[size_t(i)] && elemStarts[size_t(i + 1)] <= elemDataSize);
                                auto c_etype = static_cast<ElementType_t>(elemsRead.at(elemStarts[i]));
                                if (_getElemTypeFromCGNSType(c_etype) == Elem::UnknownElem)
                                {
                                    DNDS::log() << "Error ETYPE " << std::to_string(c_etype) << std::endl;
                                    DNDS_check_throw_info(false, "Unsupported Element! ");
                                }
                                Elem::ElemType ct = _getElemTypeFromCGNSType(c_etype);
                                DNDS_check_throw_info(Elem::Element{ct}.GetNumNodes() + 1 == elemStarts[i + 1] - elemStarts[i],
                                                      "Element Node Number Mismatch!");
                                ZoneElems.back()->ResizeRow(start - 1 + i, Elem::Element{ct}.GetNumNodes());
                                for (t_index iNode = 0; iNode < Elem::Element{ct}.GetNumNodes(); iNode++)
                                    ZoneElems.back()->operator()(start - 1 + i, iNode) =
                                        LocalCGNSNode(elemsRead.at(elemStarts[i] + 1 + iNode), nNodes); //! convert to 0 based; pointing to zonal index
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).setElemType(ct);
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).zone = Geom::BC_ID_INTERNAL; //! initialized as inner,need new way of doing vol condition
                            }
                            /// @todo //TODO: TEST with actual data (MIXED TYPE) !!!!!!
                        }
                        else if (_getElemTypeFromCGNSType(etype) != Elem::UnknownElem)
                        {
                            Elem::ElemType ct = _getElemTypeFromCGNSType(etype);
                            NATIVE_CGNS_CHECK(cg_elements_read(cgns_file, iBase, iZone, iSection, elemsRead.data(), nullptr));
                            NativeRequire(elemDataSize == cgsize_t(Elem::Element{ct}.GetNumNodes()) * nElemSec);
                            for (cgsize_t i = 0; i < nElemSec; i++)
                            {

                                ZoneElems.back()->ResizeRow(start - 1 + i, Elem::Element{ct}.GetNumNodes());
                                for (t_index iNode = 0; iNode < Elem::Element{ct}.GetNumNodes(); iNode++)
                                    ZoneElems.back()->operator()(start - 1 + i, iNode) =
                                        LocalCGNSNode(elemsRead.at(Elem::Element{ct}.GetNumNodes() * i + iNode), nNodes); //! convert to 0 based; pointing to zonal index
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).setElemType(ct);
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).zone = Geom::BC_ID_INTERNAL; //! initialized as inner,need new way of doing vol condition
                            }
                        }
                        else
                        {
                            DNDS::log() << "Error ETYPE " << std::to_string(etype) << std::endl;
                            DNDS_check_throw_info(false, "Unsupported Element! ");
                        }
                    }
                    for (index row = 0; row < ZoneElems.back()->Size(); ++row)
                        for (rowsize slot = 0; slot < ZoneElems.back()->RowSize(row); ++slot)
                            NativeRequire((*ZoneElems.back())(row, size_t(slot)) >= 0 && (*ZoneElems.back())(row, size_t(slot)) < nNodes);
                    DNDS::log() << "CGNS === Zone " << iZone << " Elems Reading Done" << std::endl;
                }

                //* read BCs
                {
                    int nBC;
                    NATIVE_CGNS_CHECK(cg_nbocos(cgns_file, iBase, iZone, &nBC));
                    for (int iBC = 1; iBC <= nBC; iBC++)
                    {
                        std::array<char, 48> boconame{};
                        PointSetType_t pType;
                        cgsize_t nPts, normalListSize;
                        BCType_t bcType;
                        std::array<int, 3> NormalIndex{};
                        DataType_t normalDataType;
                        int nDataset;
                        GridLocation_t gloc;
                        NATIVE_CGNS_CHECK(cg_boco_info(cgns_file, iBase, iZone, iBC, boconame.data(), &bcType, &pType, &nPts, NormalIndex.data(), &normalListSize, &normalDataType, &nDataset));
                        NATIVE_CGNS_CHECK(cg_boco_gridlocation_read(cgns_file, iBase, iZone, iBC, &gloc));
                        DNDS_check_throw_info(pType == PointRange || pType == PointList, "Only PointRange / PointList supported in BC!");
                        if (dimension == 2)
                            NativeRequire(gloc == EdgeCenter);
                        if (dimension == 3)
                            NativeRequire(gloc == FaceCenter);
                        NativeRequire(nPts >= 0 && normalListSize >= 0 && (pType != PointRange || nPts == 2));
                        std::vector<cgsize_t> pts(size_t(nPts), 0);
                        // cg_boco_read returns normals in their stored datatype.
                        // No normals are consumed by mesh setup; nullptr skips them.
                        NATIVE_CGNS_CHECK(cg_boco_read(cgns_file, iBase, iZone, iBC, pts.data(), nullptr));

                        t_index BCCode = FBCName_2_ID(std::string(boconame.data()));
                        if (BCCode == BC_ID_NULL)
                        {
                            DNDS_check_throw_info(false, fmt::format("BC NAME [{}] NOT FOUND IN DATABASE", boconame.data()));
                        }
                        if (pType == PointRange)
                        {
                            DNDS_check_throw_info(pts.size() == 2 && pts[0] >= 1 &&
                                                      pts[0] <= pts[1] && pts[1] <= ZoneElems.back()->Size(),
                                                  "Invalid CGNS boundary PointRange");
                            for (DNDS::index i = pts[0] - 1; i < pts[1]; i++)
                            {
                                NativeRequire(ZoneElemInfos.back()->operator()(i, 0).getElemType() != Elem::UnknownElem &&
                                              Elem::Element{ZoneElemInfos.back()->operator()(i, 0).getElemType()}.GetDim() == dimension - 1);
                                ZoneElemInfos.back()->operator()(i, 0).zone = BCCode;
                            }
                        }
                        else if (pType == PointList)
                        {
                            for (auto i : pts)
                            {
                                DNDS_check_throw_info(i >= 1 && i <= ZoneElems.back()->Size(),
                                                      "Invalid CGNS boundary PointList index");
                                NativeRequire(ZoneElemInfos.back()->operator()(i - 1, 0).getElemType() != Elem::UnknownElem &&
                                              Elem::Element{ZoneElemInfos.back()->operator()(i - 1, 0).getElemType()}.GetDim() == dimension - 1);
                                ZoneElemInfos.back()->operator()(i - 1, 0).zone = BCCode; //* note that pts is 1-based
                            }
                        }
                        else
                            NativeRequire(false);
                    }

                    DNDS::log() << "CGNS === Zone " << iZone << " BCs Reading Done" << std::endl;
                }
            }
        }
        /***************************************************************************/

        /***************************************************************************/
        // TRAVERSE 2
        for (int iBase = 1; iBase <= n_bases; iBase++)
        {

            int cgns_base = -1;
            std::array<char, 48> basename{};
            int celldim = -1;
            int physdim = -1;
            int nzones = -1;

            if (cg_base_read(cgns_file, iBase, basename.data(), &celldim, &physdim))
                throw std::runtime_error(cg_get_error());
            NativeRequire(celldim == dimension);
            if (cg_nzones(cgns_file, iBase, &nzones))
                throw std::runtime_error(cg_get_error());
            for (int iZone = 1; iZone <= nzones; iZone++)
            {
                std::array<char, 48> zonename{};
                std::array<cgsize_t, 9> size{0, 0, 0, 0, 0, 0, 0, 0, 0};
                if (cg_zone_read(cgns_file, iBase, iZone, zonename.data(), size.data()))
                    throw std::runtime_error(cg_get_error());
                ZoneType_t zoneType;
                if (cg_zone_type(cgns_file, iBase, iZone, &zoneType))
                    throw std::runtime_error(cg_get_error());
                NativeRequire(zoneType == Unstructured); //! only supports unstructured

                DNDS::index nNodes = size[0];
                DNDS::index nVols = size[1];
                DNDS::index nBVertex = size[2];

                //*** read Connectivity
                {
                    ZoneConnect.emplace_back();
                    ZoneConnectDonor.emplace_back();
                    ZoneConnectTargetIZone.emplace_back();
                    int nConns;
                    NATIVE_CGNS_CHECK(cg_nconns(cgns_file, iBase, iZone, &nConns));
                    for (int iConn = 1; iConn <= nConns; iConn++)
                    {
                        std::array<char, 48> connName{}, donorName{};
                        GridLocation_t gLoc;
                        GridConnectivityType_t connType;
                        PointSetType_t ptType, ptType_donor;
                        cgsize_t npts, npts_donor;
                        ZoneType_t donorZT;
                        DataType_t donorDT;

                        NATIVE_CGNS_CHECK(cg_conn_info(cgns_file, iBase, iZone, iConn, connName.data(), &gLoc, &connType, &ptType, &npts,
                                                       donorName.data(), &donorZT, &ptType_donor, &donorDT, &npts_donor));
                        float rotationCenter[3], rotationAngle[3], translation[3];
                        const int periodicProperty = cg_conn_periodic_read(cgns_file, iBase, iZone, iConn, rotationCenter, rotationAngle, translation);
                        DNDS_check_throw_info(periodicProperty != CG_OK,
                                              "CGNS periodic connectivity properties are unsupported; use explicit BC main/donor labels and periodic import transforms");
                        DNDS_check_throw_info(periodicProperty == CG_NODE_NOT_FOUND, "cannot inspect CGNS connectivity periodic properties");

                        DNDS_check_throw_info(connType == Abutting1to1, "Only support Abutting1to1 in connection!");
                        DNDS_check_throw_info(ptType == PointList, "Only Supports PointList in connection!");
                        DNDS_check_throw_info(ptType_donor == PointListDonor, "Only Supports PointListDonor in connection!");
                        DNDS_check_throw_info(donorZT == Unstructured, "Only Supports Unstructured in connection!");
                        DNDS_check_throw_info(gLoc == Vertex, "Only Supports Vertex in connection!");
                        DNDS_check_throw_info(npts_donor == npts, "Only Supports npts_donor == npts in connection!");
                        // std::vector<cgsize_t> ptSet(npts);
                        // std::vector<cgsize_t> ptSet_donor(npts);
                        NativeRequire(npts >= 0 && (donorDT == Integer || donorDT == LongInteger));
                        ZoneConnectDonor.back().emplace_back(npts);
                        ZoneConnect.back().emplace_back(npts);
                        NATIVE_CGNS_CHECK(cg_conn_read(cgns_file, iBase, iZone, iConn, ZoneConnect.back().back().data(), sizeof(cgsize_t) == sizeof(int64_t) ? LongInteger : Integer, ZoneConnectDonor.back().back().data()));
                        int iGZFound = -1;
                        for (size_t iGZ = 0; iGZ < ZoneNames.size(); iGZ++) // find the donor
                        {
                            if (Base_Zone.at(iGZ).first == iBase)
                            {
                                if (ZoneNames.at(iGZ) == donorName.data())
                                {
                                    iGZFound = iGZ;
                                }
                            }
                        }
                        NativeRequire(iGZFound >= 0);
                        for (size_t point = 0; point < size_t(npts); ++point)
                        {
                            NativeRequire(ZoneConnect.back().back()[point] >= 1 && ZoneConnect.back().back()[point] <= nNodes);
                            NativeRequire(ZoneConnectDonor.back().back()[point] >= 1 && ZoneConnectDonor.back().back()[point] <= ZoneCoords[size_t(iGZFound)]->Size());
                        }
                        ZoneConnectTargetIZone.back().push_back(iGZFound);
                        DNDS::log() << "CGNS === Connection at Zone-Zone: " << iZone << " - " << iGZFound + 1 << std::endl;
                    }
                }
            }
        }

        /***************************************************************************/

        NativeRequire(!ZoneCoords.empty());
        for (const auto &coordinates : ZoneCoords)
            OMP::ForIngressRows(0, coordinates->Size(), [&](index iNode)
                                { NativeRequire((*coordinates)[iNode].allFinite()); });
            /***************************************************************************/
            // ASSEMBLE: deduplicate shared nodes across zones
#ifdef DNDS_USE_CGNS_ZONE_DFS_LEGACY
        auto [NodeOld2New, ZoneNodeStarts] = AssembleZoneNodes(
#else
        auto [NodeOld2New, ZoneNodeStarts] = AssembleZoneNodesUnion(
#endif
            ZoneCoords, ZoneConnect, ZoneConnectDonor, ZoneConnectTargetIZone,
            coordSerial, dimension);

        // Separate volume cells and boundary faces
        SeparateVolumeAndBoundaryElements(
            ZoneElems, ZoneElemInfos, NodeOld2New, ZoneNodeStarts,
            cell2nodeSerial, cellElemInfoSerial, cell2cellOrigSerial,
            bnd2nodeSerial, bndElemInfoSerial, bnd2bndOrigSerial,
            node2nodeOrigSerial, coordSerial, dimension);
        /***************************************************************************/

        /***************************************************************************/
        // Build bnd2cell inverse mapping
        BuildBnd2CellSerial(
            bnd2cellSerial, bnd2nodeSerial, cell2nodeSerial,
            cellElemInfoSerial, bndElemInfoSerial, coordSerial);

        // fill in original indices
        OMP::ForIngressRows(0, cell2cellOrigSerial->Size(), [&](index iCell)
                            { cell2cellOrigSerial->operator()(iCell, 0) = iCell; });
        OMP::ForIngressRows(0, node2nodeOrigSerial->Size(), [&](index iNode)
                            { node2nodeOrigSerial->operator()(iNode, 0) = iNode; });
        OMP::ForIngressRows(0, bnd2bndOrigSerial->Size(), [&](index iBnd)
                            { bnd2bndOrigSerial->operator()(iBnd, 0) = iBnd; });

        const int closeResult = cg_close(cgns_file);
        fileGuard.handle = -1;
        NATIVE_CGNS_CHECK(closeResult);

        log() << "CGNS === Serial Read Done" << std::endl;
        return {coordSerial->Commit(), cell2nodeSerial->Commit(), bnd2nodeSerial->Commit(), bnd2cellSerial->Commit(),
                cellElemInfoSerial->Commit(), bndElemInfoSerial->Commit(), cell2cellOrigSerial->Commit(),
                node2nodeOrigSerial->Commit(), bnd2bndOrigSerial->Commit()};
        // Memory with DM240-120 here: 18G ; after deconstruction done: 7.5G
    }

#undef NATIVE_CGNS_CHECK
    NativeOwnedMeshData ReadNativeCGNSOwned(
        const MPIInfo &mpi, const std::string &path, int dimension,
        const t_FBCName_2_ID &boundaryId, int readerRank, NativeMeshBuildOptions options,
        std::optional<NativePeriodicImportOptions> periodic, NativeSerialDistribution distribution)
    {
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native CGNS import requires a communicator");
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !options.communication || options.communication->ChecksCollectivePreflight());
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        {
            int rank, count;
            MPI_Comm_rank(mpi.comm, &rank);
            MPI_Comm_size(mpi.comm, &count);
            DNDS_check_throw_info(mpi.rank == rank && mpi.size == count && readerRank >= 0 && readerRank < count &&
                                      (dimension == 2 || dimension == 3) && options.cellLayers >= 1 &&
                                      (distribution == NativeSerialDistribution::Even || distribution == NativeSerialDistribution::ReaderOwned) &&
                                      path.size() <= size_t(INT_MAX), "invalid native CGNS import options"); });
        int settings[]{readerRank, dimension, options.cellLayers, int(options.completeNodeStar), int(bool(periodic)), int(distribution)};
        int minimum[6], maximum[6];
        MPI_Allreduce(settings, minimum, 6, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(settings, maximum, 6, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(std::equal(minimum, minimum + 6, maximum), "native CGNS import options differ across ranks");
        if (periodic)
        {
            std::array<real, 61> values{}, canonical{};
            FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                            {
                values[0] = periodic->tolerance;
                size_t offset = 1;
                for (size_t frame = 0; frame < 4; ++frame)
                {
                    for (int i = 0; i < 9; ++i) values[offset++] = periodic->frames.rotation[frame].map().data()[i];
                    for (int i = 0; i < 3; ++i) values[offset++] = periodic->frames.translation[frame].map()[i];
                    for (int i = 0; i < 3; ++i) values[offset++] = periodic->frames.rotationCenter[frame].map()[i];
                }
                for (auto value : values) DNDS_check_throw_info(std::isfinite(value), "nonfinite native periodic import option");
                DNDS_check_throw_info(periodic->tolerance > 0, "invalid native periodic import tolerance"); });
            canonical = values;
            MPI_Bcast(canonical.data(), int(canonical.size()), DNDS_MPI_REAL, readerRank, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                            { DNDS_check_throw_info(values == canonical, "native periodic import options differ across ranks"); });
        }
        int pathSize = int(path.size());
        MPI_Bcast(&pathSize, 1, MPI_INT, readerRank, mpi.comm);
        std::string expectedPath;
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        { expectedPath = mpi.rank == readerRank ? path : std::string(size_t(pathSize), '\0'); });
        MPI_Bcast(expectedPath.data(), pathSize, MPI_CHAR, readerRank, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        { DNDS_check_throw_info(path == expectedPath, "native CGNS import paths differ across ranks"); });

        NativeSerialMeshData serial;
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        { if (mpi.rank == readerRank) serial = ReadNativeCGNSSerial(path, dimension, boundaryId); });
        return DistributeNativeSerialMeshOwned(mpi, std::move(serial), dimension, readerRank, options, std::move(periodic), distribution);
    }

    NativeOwnedMeshData DistributeNativeSerialMeshOwned(
        const MPIInfo &mpi, NativeSerialMeshData serial, int dimension, int readerRank,
        NativeMeshBuildOptions options, std::optional<NativePeriodicImportOptions> periodic, NativeSerialDistribution distribution)
    {
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native serial distribution requires communicator");
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !options.communication || options.communication->ChecksCollectivePreflight());
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        {
            int rank = -1, size = 0;
            MPI_Comm_rank(mpi.comm, &rank); MPI_Comm_size(mpi.comm, &size);
            DNDS_check_throw_info(rank == mpi.rank && size == mpi.size && readerRank >= 0 && readerRank < size &&
                                  dimension >= 1 && dimension <= 3 && options.cellLayers >= 1 &&
                                  (distribution == NativeSerialDistribution::Even || distribution == NativeSerialDistribution::ReaderOwned),
                                  "native serial distribution options invalid"); });
        std::array<int, 6> parameters{dimension, readerRank, options.cellLayers, int(options.completeNodeStar), int(bool(periodic)), int(distribution)};
        auto expected = parameters;
        MPI_Bcast(expected.data(), int(expected.size()), MPI_INT, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        { DNDS_check_throw_info(parameters == expected, "native serial distribution options differ across ranks"); });
        if (periodic)
        {
            std::array<real, 61> frame{}, canonical{};
            frame[0] = periodic->tolerance;
            size_t offset = 1;
            for (size_t f = 0; f < 4; ++f)
            {
                for (int i = 0; i < 9; ++i)
                    frame[offset++] = periodic->frames.rotation[f].map().data()[i];
                for (int i = 0; i < 3; ++i)
                    frame[offset++] = periodic->frames.translation[f].map()[i];
                for (int i = 0; i < 3; ++i)
                    frame[offset++] = periodic->frames.rotationCenter[f].map()[i];
            }
            canonical = frame;
            MPI_Bcast(canonical.data(), int(canonical.size()), DNDS_MPI_REAL, readerRank, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                            { DNDS_check_throw_info(frame == canonical && std::isfinite(frame[0]) && frame[0] > 0, "native serial periodic frames differ or invalid tolerance"); });
        }
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        {
            DNDS_check_throw_info(!serial.periodic, "native serial ingress requires raw records before periodic canonicalization");
            const auto nc = serial.cellNodes.RowCount(), nn = serial.coordinates.RowCount(), nb = serial.boundaryNodes.RowCount();
            DNDS_check_throw_info(mpi.rank == readerRank || (nc == 0 && nn == 0 && nb == 0), "nonreader supplied native serial payload");
            if (mpi.rank != readerRank) return;
            DNDS_check_throw_info(nc <= size_t(INT64_MAX) && nn <= size_t(INT64_MAX) && nb <= size_t(INT64_MAX) &&
                serial.cellInfo.RowCount() == nc && serial.boundaryInfo.RowCount() == nb && serial.cellIds.RowCount() == nc &&
                serial.nodeIds.RowCount() == nn && serial.boundaryIds.RowCount() == nb, "native serial row counts differ");
            auto check = [&](const auto &array, size_t width)
            {
                auto read = array.Read();
                OMP::ForIngressRows(0, array.RowCount(), [&](index iRow)
                                  { DNDS_check_throw_info(read.View().RowSize(iRow) == width, "native serial companion width invalid"); });
            };
            check(serial.coordinates, 3); check(serial.cellInfo, 1); check(serial.boundaryInfo, 1);
            check(serial.cellIds, 1); check(serial.nodeIds, 1); check(serial.boundaryIds, 1);
            for (const auto *cone : {&serial.cellNodes, &serial.boundaryNodes})
            {
                auto read = cone->Read();
                OMP::ForIngressRows(0, cone->RowCount(), [&](index iRow)
                                  {
                    for (auto node : read.View()[iRow])
                        DNDS_check_throw_info(node >= 0 && uint64_t(node) < nn, "native serial cone node outside global range"); });
            }
            auto points = serial.coordinates.Read();
            OMP::ForIngressRows(0, index(nn), [&](index row)
            {
                for (size_t component = 0; component < 3; ++component)
                    DNDS_check_throw_info(std::isfinite(points.View()(row, component)), "native serial coordinate nonfinite");
            });
            if (periodic) serial = CanonicalizeNativePeriodic(std::move(serial), *periodic); });
        std::string identityWire;
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        {
            if (mpi.rank == readerRank)
            {
                nlohmann::json next = nlohmann::json::array();
                for (const auto &[kind, value] : serial.identity.nextIds) next.push_back({int(kind), value});
                identityWire = nlohmann::json{{"namespace", serial.identity.meshNamespace}, {"next", next}}.dump();
            }
            DNDS_check_throw_info(identityWire.size() <= size_t(INT_MAX), "native serial identity metadata too large"); });
        int identitySize = int(identityWire.size());
        MPI_Bcast(&identitySize, 1, MPI_INT, readerRank, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        { identityWire.resize(size_t(identitySize)); });
        MPI_Bcast(identityWire.data(), identitySize, MPI_CHAR, readerRank, mpi.comm);
        MeshIdentity identity;
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        {
            const auto json = nlohmann::json::parse(identityWire);
            identity.meshNamespace = json.at("namespace").get<std::string>();
            for (const auto &entry : json.at("next")) identity.nextIds.emplace(EntityKind(entry.at(0).get<int>()), entry.at(1).get<index>()); });
        int isPeriodic = serial.periodic;
        MPI_Bcast(&isPeriodic, 1, MPI_INT, readerRank, mpi.comm);
        index nodeHighWater = mpi.rank == readerRank && serial.identity.nextIds.count(EntityKind::Node) ? serial.identity.nextIds.at(EntityKind::Node) : 0;
        MPI_Bcast(&nodeHighWater, 1, DNDS_MPI_INDEX, readerRank, mpi.comm);
        index counts[]{index(serial.cellNodes.RowCount()), index(serial.coordinates.RowCount()), index(serial.boundaryNodes.RowCount())};
        MPI_Bcast(counts, 3, MPI_INT64_T, readerRank, mpi.comm);
        auto context = options.communication ? options.communication : std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        // Empty ranks must carry the reader's physical layout too. A default
        // empty Array has no table width; it is not a valid receiver prototype
        // for a fixed-width source (nor a declaration of compact CSR storage).
        auto storagePrototype = [&](auto source)
        {
            using Array = std::decay_t<decltype(source)>;
            const auto &shape = source.Layout();
            index storage[]{shape.IsRagged() ? 2 : shape.IsPadded() ? 1
                                                                    : 0,
                            shape.Width(), shape.Stride()};
            MPI_Bcast(storage, 3, DNDS_MPI_INDEX, readerRank, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                            {
                if (mpi.rank != readerRank)
                    source = Array(storage[0] == 2 ? Runtime::RowLayout::Ragged({}) :
                                   storage[0] == 1 ? Runtime::RowLayout::Padded({}, size_t(storage[2])) :
                                                    Runtime::RowLayout::Dense(0, size_t(storage[1]), size_t(storage[2]))); });
            return source;
        };
        if (distribution == NativeSerialDistribution::ReaderOwned)
        {
            auto localCopy = [&](auto source)
            {
                using Array = std::decay_t<decltype(source)>;
                source = storagePrototype(std::move(source));
                if (mpi.rank == readerRank && source.Layout().IsRagged())
                    return std::move(source);
                Array output(source.Layout());
                if (mpi.rank == readerRank)
                {
                    auto input = source.Read();
                    auto destination = output.Write();
                    const auto &layout = input.Layout();
                    if (!layout.IsVariable() && layout.Width() == layout.Stride())
                    {
                        const auto elements = layout.Elements();
                        DNDS_check_throw_info(destination.Layout().Elements() == elements,
                                              "native serial copy extent mismatch");
                        nativeCopyBulkCalls.fetch_add(1, std::memory_order_relaxed);
                        nativeCopyBulkRows.fetch_add(uint64_t(layout.RowCount()), std::memory_order_relaxed);
                        std::copy_n(input.View().data, elements, destination.View().data);
                    }
                    else
                    {
                        nativeCopyRowCalls.fetch_add(1, std::memory_order_relaxed);
                        nativeCopyRowRows.fetch_add(uint64_t(layout.RowCount()), std::memory_order_relaxed);
                        const auto sourceView = input.View();
                        const auto targetView = destination.View();
                        OMP::ForIngressRows(0, sourceView.RowCount(), [&](index iRow)
                                            { std::copy_n(sourceView.Row(iRow), sourceView.RowSize(iRow), targetView.Row(iRow)); });
                    }
                }
                return output;
            };
            const auto nodeSpace = EntitySpace::CreateOwned(EntityKind::Node, mpi,
                                                            mpi.rank == readerRank ? counts[1] : 0);
            NativeMeshInput input{mpi, dimension,
                                  nodeSpace.CreateSibling(EntityKind::Cell, mpi.rank == readerRank ? counts[0] : 0),
                                  nodeSpace,
                                  nodeSpace.CreateSibling(EntityKind::Bnd, mpi.rank == readerRank ? counts[2] : 0),
                                  localCopy(serial.cellNodes), localCopy(serial.boundaryNodes),
                                  localCopy(serial.coordinates), localCopy(serial.cellInfo),
                                  localCopy(serial.boundaryInfo)};
            input.identity = std::move(identity);
            input.ids.emplace(EntityKind::Cell, localCopy(serial.cellIds));
            input.ids.emplace(EntityKind::Node, localCopy(serial.nodeIds));
            input.ids.emplace(EntityKind::Bnd, localCopy(serial.boundaryIds));
            if (isPeriodic)
            {
                DNDS_check_throw_info(periodic, "native serial periodic frames missing");
                Runtime::Array<NodePeriodicBits> empty(Runtime::RowLayout::Ragged({}));
                input.periodic = true;
                input.periodicInfo = periodic->frames;
                input.identity.nextIds.emplace(EntityKind::Node, nodeHighWater);
                input.cellNodesPbi = localCopy(serial.cellNodesPbi ? *serial.cellNodesPbi : empty);
                input.boundaryNodesPbi = localCopy(serial.boundaryNodesPbi ? *serial.boundaryNodesPbi : empty);
            }
            return NativeMeshBuilder::Build(std::move(input), options);
        }
        auto route = [&](index total)
        {
            const index length = distribution == NativeSerialDistribution::ReaderOwned ? (mpi.rank == readerRank ? total : 0) : total / mpi.size + (mpi.rank < total % mpi.size);
            const index first = distribution == NativeSerialDistribution::ReaderOwned ? 0 : (total / mpi.size) * mpi.rank + std::min<index>(mpi.rank, total % mpi.size);
            std::vector<Runtime::RowRequest> requests;
            FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                            {
                requests.reserve(size_t(length));
                for (index row = 0; row < length; ++row)
                    requests.push_back({readerRank, first + row, size_t(row)}); });
            return Runtime::RowTransferPlan(context, mpi.rank == readerRank ? size_t(total) : 0, size_t(length), requests);
        };
        auto cells = route(counts[0]), nodes = route(counts[1]), boundaries = route(counts[2]);
        auto transfer = [&](auto source, const Runtime::RowTransferPlan &plan, const Runtime::ElementSchema &schema)
        {
            using Array = std::decay_t<decltype(source)>;
            source = storagePrototype(std::move(source));
            Runtime::PayloadPlan payload(plan, source.Layout(), schema);
            std::optional<Array> output;
            FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                            { output.emplace(payload.DestinationLayout()); });
            Runtime::TransferOptions transferOptions;
            transferOptions.context = context;
            transferOptions.mode = Runtime::TransferMode::Transient;
            payload.Bind(source, *output, Runtime::TransferTransport::Direct, transferOptions)
                .Begin()
                .Wait();
            return std::move(*output);
        };
        const auto indices = Runtime::ElementSchema::Arithmetic<index>();
        const auto elements = Runtime::ElementSchema::For<ElemInfo>();
        const auto nodeSpace = EntitySpace::CreateOwned(EntityKind::Node, mpi, index(nodes.DestinationRows()));
        NativeMeshInput input{mpi, dimension,
                              nodeSpace.CreateSibling(EntityKind::Cell, index(cells.DestinationRows())),
                              nodeSpace, nodeSpace.CreateSibling(EntityKind::Bnd, index(boundaries.DestinationRows())),
                              transfer(serial.cellNodes, cells, indices), transfer(serial.boundaryNodes, boundaries, indices),
                              transfer(serial.coordinates, nodes, Runtime::ElementSchema::Arithmetic<real>()),
                              transfer(serial.cellInfo, cells, elements), transfer(serial.boundaryInfo, boundaries, elements)};
        input.identity = std::move(identity);
        input.ids.emplace(EntityKind::Cell, transfer(serial.cellIds, cells, indices));
        input.ids.emplace(EntityKind::Node, transfer(serial.nodeIds, nodes, indices));
        input.ids.emplace(EntityKind::Bnd, transfer(serial.boundaryIds, boundaries, indices));
        if (isPeriodic)
        {
            Runtime::Array<NodePeriodicBits> empty(Runtime::RowLayout::Ragged({}));
            const auto bits = Runtime::ElementSchema::For<NodePeriodicBits>();
            input.periodic = true;
            input.periodicInfo = periodic->frames;
            input.identity.nextIds.emplace(EntityKind::Node, nodeHighWater);
            input.cellNodesPbi = transfer(serial.cellNodesPbi ? *serial.cellNodesPbi : empty, cells, bits);
            input.boundaryNodesPbi = transfer(serial.boundaryNodesPbi ? *serial.boundaryNodesPbi : empty, boundaries, bits);
        }
        return NativeMeshBuilder::Build(std::move(input), options);
    }

    std::shared_ptr<const UnstructuredMeshState> DistributeNativeSerialMesh(
        const MPIInfo &mpi, NativeSerialMeshData serial, int dimension, int readerRank,
        NativeMeshBuildOptions options, std::optional<NativePeriodicImportOptions> periodic, NativeSerialDistribution distribution)
    {
        auto owned = DistributeNativeSerialMeshOwned(mpi, std::move(serial), dimension, readerRank, options, std::move(periodic), distribution);
        GhostSpec recipe;
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        { recipe = DefaultNativeMeshGhostSpec(owned, options.cellLayers, options.completeNodeStar); });
        return NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    }

    std::shared_ptr<const UnstructuredMeshState> ReadNativeCGNS(
        const MPIInfo &mpi, const std::string &path, int dimension,
        const t_FBCName_2_ID &boundaryId, int readerRank, NativeMeshBuildOptions options,
        std::optional<NativePeriodicImportOptions> periodic, NativeSerialDistribution distribution)
    {
        auto owned = ReadNativeCGNSOwned(mpi, path, dimension, boundaryId, readerRank, options, std::move(periodic), distribution);
        GhostSpec recipe;
        FieldTransferDetail::LocalStage(mpi, options.communication, [&]
                                        { recipe = DefaultNativeMeshGhostSpec(owned, options.cellLayers, options.completeNodeStar); });
        return NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    }
}
