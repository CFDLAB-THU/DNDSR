#include "Mesh.hpp"
#include "Geom/CGNS.hpp"
#include "Geom/OpenFOAMMesh.hpp"

#include <fmt/core.h>

#include <cgnslib.h>

#include <set>

namespace DNDS::Geom
{
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
        std::pair<std::vector<DNDS::index>, std::vector<DNDS::index>>
        AssembleZoneNodes(
            const std::vector<tCoord> &ZoneCoords,
            const std::vector<std::vector<std::vector<cgsize_t>>> &ZoneConnect,
            const std::vector<std::vector<std::vector<cgsize_t>>> &ZoneConnectDonor,
            const std::vector<std::vector<int>> &ZoneConnectTargetIZone,
            tCoord &coordSerial,
            int dim)
        {
            std::vector<DNDS::index> ZoneNodeSizes(ZoneCoords.size());
            std::vector<DNDS::index> ZoneNodeStarts(ZoneCoords.size() + 1);
            ZoneNodeStarts[0] = 0;
            for (size_t i = 0; i < ZoneNodeSizes.size(); i++)
                ZoneNodeStarts[i + 1] = ZoneNodeStarts[i] + (ZoneNodeSizes[i] = ZoneCoords[i]->Size());

            std::vector<DNDS::index> NodeOld2New(ZoneNodeStarts.back(), -1);
            DNDS::index cTop = 0;

            // DFS over zone connectivity graph to deduplicate shared nodes
            std::set<int> zonesLeft;
            for (size_t iGZ = 0; iGZ < ZoneNodeSizes.size(); iGZ++)
                zonesLeft.insert(static_cast<int>(iGZ));
            std::vector<int> zonesFront;
            zonesFront.push_back(0);
            DNDS_assert(ZoneNodeSizes.size() >= 1);

            while (zonesFront.size())
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
                            DNDS_assert(NodeOld2New.at(ZoneNodeStarts.at(iGZ) + ZoneConnect.at(iGZ).at(iOther).at(iNode) - 1) >= 0);
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
            DNDS_assert_info(zonesLeft.empty(), "Did not reach all the zones, might missing zone-connectivities");

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
                        DNDS_assert_info(dist < 1e-10,
                                         "Not same points on the connection, distance is " + std::to_string(dist));
                    (*coordSerial)[NodeOld2New.at(iNode)] = coordC;
                }
            }

            return {std::move(NodeOld2New), std::move(ZoneNodeStarts)};
        }

        /**
         * \brief Separate zone elements into volume cells and boundary faces.
         *
         * Converts element node indices from zone-local to assembled global
         * (via NodeOld2New), then fills cell2nodeSerial/cellElemInfoSerial and
         * bnd2nodeSerial/bndElemInfoSerial.
         */
        void SeparateVolumeAndBoundaryElements(
            const std::vector<tAdj> &ZoneElems,
            const std::vector<tElemInfoArray> &ZoneElemInfos,
            const std::vector<DNDS::index> &NodeOld2New,
            const std::vector<DNDS::index> &ZoneNodeStarts,
            tAdj &cell2nodeSerial,
            tElemInfoArray &cellElemInfoSerial,
            tAdj1 &cell2cellOrigSerial,
            tAdj &bnd2nodeSerial,
            tElemInfoArray &bndElemInfoSerial,
            tAdj1 &bnd2bndOrigSerial,
            tAdj1 &node2nodeOrigSerial,
            tCoord &coordSerial,
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
                        ZoneElems[iGZ]->operator()(iElem, j) = NodeOld2New[ZoneNodeStarts[iGZ] + ZoneElems[iGZ]->operator()(iElem, j)];
                    auto Elem = Elem::Element{ZoneElemInfos[iGZ]->operator()(iElem, 0).getElemType()};
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
                    auto Elem = Elem::Element{ZoneElemInfos[iGZ]->operator()(iElem, 0).getElemType()};
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
            tAdj2 &bnd2cellSerial,
            const tAdj &bnd2nodeSerial,
            const tAdj &cell2nodeSerial,
            const tElemInfoArray &cellElemInfoSerial,
            const tElemInfoArray &bndElemInfoSerial,
            const tCoord &coordSerial)
        {
            bnd2cellSerial->Resize(bnd2nodeSerial->Size());
            for (DNDS::index iB = 0; iB < bnd2cellSerial->Size(); iB++)
                (*bnd2cellSerial)[iB][0] = (*bnd2cellSerial)[iB][1] = DNDS::UnInitIndex;

            // Build node-to-boundary index for linear complexity
            std::vector<std::vector<DNDS::index>> node2bnd(coordSerial->Size());
            std::vector<DNDS::rowsize> node2bndSiz(coordSerial->Size(), 0);
            for (DNDS::index iBFace = 0; iBFace < bnd2nodeSerial->Size(); iBFace++)
                for (DNDS::rowsize iN = 0; iN < Elem::Element{(*bndElemInfoSerial)(iBFace, 0).getElemType()}.GetNumVertices(); iN++)
                    node2bndSiz[(*bnd2nodeSerial)(iBFace, iN)]++;
            for (DNDS::index iNode = 0; iNode < static_cast<DNDS::index>(node2bnd.size()); iNode++)
                node2bnd[iNode].reserve(node2bndSiz[iNode]);
            for (DNDS::index iBFace = 0; iBFace < bnd2nodeSerial->Size(); iBFace++)
                for (DNDS::rowsize iN = 0; iN < Elem::Element{(*bndElemInfoSerial)(iBFace, 0).getElemType()}.GetNumVertices(); iN++)
                    node2bnd[(*bnd2nodeSerial)(iBFace, iN)].push_back(iBFace);

            // Search each cell for matching boundary faces
            for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            {
                auto CellElem = Elem::Element{(*cellElemInfoSerial)[iCell]->getElemType()};
                std::vector<DNDS::index> cell2nodeRow{(*cell2nodeSerial)[iCell].begin(), (*cell2nodeSerial)[iCell].begin() + CellElem.GetNumVertices()};
                std::sort(cell2nodeRow.begin(), cell2nodeRow.end());
                for (auto iNode : cell2nodeRow)
                {
                    for (auto iB : node2bnd[iNode])
                    {
                        auto BndElem = Elem::Element{(*bndElemInfoSerial)(iB, 0).getElemType()};
                        std::vector<DNDS::index> bnd2nodeRow{(*bnd2nodeSerial)[iB].begin(), (*bnd2nodeSerial)[iB].begin() + BndElem.GetNumVertices()};
                        std::sort(bnd2nodeRow.begin(), bnd2nodeRow.end());
                        if (std::includes(cell2nodeRow.begin(), cell2nodeRow.end(), bnd2nodeRow.begin(), bnd2nodeRow.end()))
                        {
                            DNDS_assert_info(
                                (*bnd2cellSerial)[iB][0] == DNDS::UnInitIndex || (*bnd2cellSerial)[iB][0] == iCell, "bnd2cell not untouched!");
                            (*bnd2cellSerial)[iB][0] = iCell;
                        }
                    }
                }
            }
            for (DNDS::index iB = 0; iB < bnd2cellSerial->Size(); iB++)
                DNDS_assert((*bnd2cellSerial)[iB][0] != DNDS::UnInitIndex);
        }
    } // anonymous namespace
    void UnstructuredMeshSerialRW::
        ReadFromCGNSSerial(const std::string &fName, const t_FBCName_2_ID &FBCName_2_ID)
    {
        mode = SerialReadAndDistribute;
        this->dataIsSerialIn = true;

        int cgErr = CG_OK;

        cell2nodeSerial = make_ssp<decltype(cell2nodeSerial)::element_type>(ObjName{"ReadFromCGNSSerial::cell2nodeSerial"}, mesh->getMPI());
        bnd2nodeSerial = make_ssp<decltype(bnd2nodeSerial)::element_type>(ObjName{"ReadFromCGNSSerial::bnd2nodeSerial"}, mesh->getMPI());
        coordSerial = make_ssp<decltype(coordSerial)::element_type>(ObjName{"ReadFromCGNSSerial::coordSerial"}, mesh->getMPI());
        cellElemInfoSerial = make_ssp<decltype(cellElemInfoSerial)::element_type>(ObjName{"ReadFromCGNSSerial::cellElemInfoSerial"}, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        bndElemInfoSerial = make_ssp<decltype(bndElemInfoSerial)::element_type>(ObjName{"ReadFromCGNSSerial::bndElemInfoSerial"}, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        bnd2cellSerial = make_ssp<decltype(bnd2cellSerial)::element_type>(ObjName{"ReadFromCGNSSerial::bnd2cellSerial"}, mesh->getMPI());
        cell2cellOrigSerial = make_ssp<decltype(cell2cellOrigSerial)::element_type>(ObjName{"ReadFromCGNSSerial::cell2cellOrigSerial"}, mesh->getMPI());
        node2nodeOrigSerial = make_ssp<decltype(node2nodeOrigSerial)::element_type>(ObjName{"ReadFromCGNSSerial::node2nodeOrigSerial"}, mesh->getMPI());
        bnd2bndOrigSerial = make_ssp<decltype(bnd2bndOrigSerial)::element_type>(ObjName{"ReadFromCGNSSerial::bnd2bndOrigSerial"}, mesh->getMPI());

        if (mRank != mesh->getMPI().rank) //! parallel done!!! now serial!!!
            return;

        int cgns_file = -1;
        if (cg_open(fName.c_str(), CG_MODE_READ, &cgns_file))
        {
            std::cerr << fmt::format("cgns file cannot open: [{}]", fName) << std::endl;
            cg_error_exit();
        }
        int n_bases = -1;
        if (cg_nbases(cgns_file, &n_bases))
            cg_error_exit();
        DNDS::log() << "CGNS === N bases: " << n_bases << std::endl;
        DNDS_assert(n_bases > 0);
        DNDS_assert(n_bases == 1);
        std::vector<std::pair<int, int>> Base_Zone;
        std::vector<std::string> BaseNames;
        std::vector<std::string> ZoneNames;
        // std::vector<std::string> ZoneFamilyNames;
        std::vector<std::array<cgsize_t, 9>> ZoneSizes;
        std::vector<tCoord> ZoneCoords; //! purely serial
        std::vector<tAdj> ZoneElems;
        std::vector<tElemInfoArray> ZoneElemInfos;
        std::vector<std::vector<std::vector<cgsize_t>>> ZoneConnect;
        std::vector<std::vector<std::vector<cgsize_t>>> ZoneConnectDonor;
        std::vector<std::vector<int>> ZoneConnectTargetIZone;
        /***************************************************************************/
        // TRAVERSE 1
        for (int iBase = 1; iBase <= n_bases; iBase++)
        {

            int cgns_base = -1;
            char basename[48]{0};
            int celldim = -1;
            int physdim = -1;
            int nzones = -1;

            if (cg_base_read(cgns_file, iBase, basename, &celldim, &physdim))
                cg_error_exit();
            DNDS_assert_info(celldim == mesh->dim, "CGNS file need to be with correct dim");
            if (cg_nzones(cgns_file, iBase, &nzones))
                cg_error_exit();
            for (int iZone = 1; iZone <= nzones; iZone++)
            {
                char zonename[48]{0};
                std::array<cgsize_t, 9> size{0, 0, 0, 0, 0, 0, 0, 0, 0};
                if (cg_zone_read(cgns_file, iBase, iZone, zonename, size.data()))
                    cg_error_exit();
                ZoneType_t zoneType;
                if (cg_zone_type(cgns_file, iBase, iZone, &zoneType))
                    cg_error_exit();
                DNDS_assert(zoneType == Unstructured); //! only supports unstructured
                Base_Zone.emplace_back(std::make_pair(iBase, iZone));
                BaseNames.emplace_back(basename);
                ZoneNames.emplace_back(zonename);

                // !  family name used for Volume condition not used now
                // if (cg_goto(cgns_file, iBase, "Zone_t", iZone, ""))
                //     cg_error_exit();
                // char famname[48]{0};
                // if (cg_famname_read(famname))
                //     cg_error_exit();
                // ZoneFamilyNames.push_back(famname); //* family name used for Volume condition
                //

                DNDS::index nNodes = size[0];
                DNDS::index nVols = size[1];
                DNDS::index nBVertex = size[2];

                //*** read coords
                {
                    int ncoords;
                    DNDS_CGNS_CALL_EXIT(cg_ncoords(cgns_file, iBase, iZone, &ncoords));
                    // DNDS_assert(ncoords == mesh->dim); //!Even if has Z coord, we only use X,Y in 2d

                    ZoneCoords.emplace_back(std::make_shared<decltype(ZoneCoords)::value_type::element_type>());
                    ZoneCoords.back()->Resize(nNodes);
                    std::vector<double> coordsRead(nNodes);
                    cgsize_t RMin[3]{1, 0, 0};
                    cgsize_t RMax[3]{nNodes, 0, 0};
                    //* READ X
                    DNDS_CGNS_CALL_EXIT(cg_coord_read(cgns_file, iBase, iZone, "CoordinateX", RealDouble, RMin, RMax, coordsRead.data()));
                    for (DNDS::index i = 0; i < nNodes; i++)
                        ZoneCoords.back()->operator[](i)[0] = coordsRead.at(i);
                    //* READ Y
                    DNDS_CGNS_CALL_EXIT(cg_coord_read(cgns_file, iBase, iZone, "CoordinateY", RealDouble, RMin, RMax, coordsRead.data()));
                    for (DNDS::index i = 0; i < nNodes; i++)
                        ZoneCoords.back()->operator[](i)[1] = coordsRead.at(i);
                    //* READ Z
                    if (mesh->dim == 3)
                    {
                        DNDS_CGNS_CALL_EXIT(cg_coord_read(cgns_file, iBase, iZone, "CoordinateZ", RealDouble, RMin, RMax, coordsRead.data()));
                    }
                    else
                        for (auto &i : coordsRead)
                            i = 0;
                    for (DNDS::index i = 0; i < nNodes; i++)
                        ZoneCoords.back()->operator[](i)[2] = coordsRead.at(i);

                    DNDS::log() << "CGNS === Zone " << iZone << " Coords Reading Done" << std::endl;
                }
                // for (DNDS::index i = 0; i < ZoneCoords.back()->Size(); i++)
                //     std::cout << (*ZoneCoords.back())[i].transpose() << std::endl;

                //*** read Elems
                {
                    int nSections;
                    DNDS_CGNS_CALL_EXIT(cg_nsections(cgns_file, iBase, iZone, &nSections));
                    DNDS_assert(nSections >= 1);
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
                        char sectionName[48];
                        ElementType_t etype;
                        cgsize_t start;
                        cgsize_t end;
                        int nBnd{0}, parentFlag{0};
                        DNDS_CGNS_CALL_EXIT(cg_section_read(cgns_file, iBase, iZone, iSection, sectionName, &etype, &start, &end, &nBnd, &parentFlag));
                        // cgsize_t elemDataSize{0};
                        // DNDS_CGNS_CALL_EXIT(cg_ElementDataSize(cgns_file, iBase, iZone, iSection, &elemDataSize));
                        // DNDS_assert(cend == start - 1); //? testing//!not valid!
                        cstart = start;
                        cend = end;
                        maxend = std::max(end, maxend);
                    }
                    ZoneElems.emplace_back(std::make_shared<decltype(ZoneElems)::value_type::element_type>());
                    ZoneElems.back()->Resize(maxend);
                    ZoneElemInfos.emplace_back(std::make_shared<decltype(ZoneElemInfos)::value_type::element_type>());
                    ZoneElemInfos.back()->Resize(maxend);
                    //*Resize row
                    for (int iSection = 1; iSection <= nSections; iSection++)
                    {
                        char sectionName[48];
                        ElementType_t etype;
                        cgsize_t start;
                        cgsize_t end;
                        int nBnd{0}, parentFlag{0};
                        DNDS_CGNS_CALL_EXIT(cg_section_read(cgns_file, iBase, iZone, iSection, sectionName, &etype, &start, &end, &nBnd, &parentFlag));
                        cgsize_t elemDataSize{0};
                        DNDS_CGNS_CALL_EXIT(cg_ElementDataSize(cgns_file, iBase, iZone, iSection, &elemDataSize));
                        std::vector<cgsize_t> elemsRead(elemDataSize);

                        int nElemSec = end - start + 1;
                        if (etype == MIXED)
                        {
                            std::vector<cgsize_t> elemStarts(nElemSec + 1); //* note size
                            DNDS_CGNS_CALL_EXIT(cg_poly_elements_read(cgns_file, iBase, iZone, iSection, elemsRead.data(), elemStarts.data(), NULL));
                            for (cgsize_t i = 0; i < nElemSec; i++)
                            {
                                auto c_etype = static_cast<ElementType_t>(elemsRead.at(elemStarts[i]));
                                if (__getElemTypeFromCGNSType(c_etype) == Elem::UnknownElem)
                                {
                                    DNDS::log() << "Error ETYPE " << std::to_string(c_etype) << std::endl;
                                    DNDS_assert_info(false, "Unsupported Element! ");
                                }
                                Elem::ElemType ct = __getElemTypeFromCGNSType(c_etype);
                                DNDS_assert_info(Elem::Element{ct}.GetNumNodes() + 1 == elemStarts[i + 1] - elemStarts[i],
                                                 "Element Node Number Mismatch!");
                                ZoneElems.back()->ResizeRow(start - 1 + i, Elem::Element{ct}.GetNumNodes());
                                for (t_index iNode = 0; iNode < Elem::Element{ct}.GetNumNodes(); iNode++)
                                    ZoneElems.back()->operator()(start - 1 + i, iNode) =
                                        elemsRead.at(elemStarts[i] + 1 + iNode) - 1; //! convert to 0 based; pointing to zonal index
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).setElemType(ct);
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).zone = Geom::BC_ID_INTERNAL; //! initialized as inner,need new way of doing vol condition
                            }
                            /// @todo //TODO: TEST with actual data (MIXED TYPE) !!!!!!
                        }
                        else if (__getElemTypeFromCGNSType(etype) != Elem::UnknownElem)
                        {
                            Elem::ElemType ct = __getElemTypeFromCGNSType(etype);
                            DNDS_CGNS_CALL_EXIT(cg_elements_read(cgns_file, iBase, iZone, iSection, elemsRead.data(), NULL));
                            DNDS_assert(elemDataSize / Elem::Element{ct}.GetNumNodes() == nElemSec);
                            for (cgsize_t i = 0; i < nElemSec; i++)
                            {

                                ZoneElems.back()->ResizeRow(start - 1 + i, Elem::Element{ct}.GetNumNodes());
                                for (t_index iNode = 0; iNode < Elem::Element{ct}.GetNumNodes(); iNode++)
                                    ZoneElems.back()->operator()(start - 1 + i, iNode) =
                                        elemsRead.at(Elem::Element{ct}.GetNumNodes() * i + iNode) - 1; //! convert to 0 based; pointing to zonal index
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).setElemType(ct);
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).zone = Geom::BC_ID_INTERNAL; //! initialized as inner,need new way of doing vol condition
                            }
                        }
                        else
                        {
                            DNDS::log() << "Error ETYPE " << std::to_string(etype) << std::endl;
                            DNDS_assert_info(false, "Unsupported Element! ");
                        }
                    }
                    DNDS::log() << "CGNS === Zone " << iZone << " Elems Reading Done" << std::endl;
                }

                //* read BCs
                {
                    int nBC;
                    DNDS_CGNS_CALL_EXIT(cg_nbocos(cgns_file, iBase, iZone, &nBC));
                    for (int iBC = 1; iBC <= nBC; iBC++)
                    {
                        char boconame[48];
                        PointSetType_t pType;
                        cgsize_t nPts, normalListSize;
                        BCType_t bcType;
                        int NormalIndex[3];
                        DataType_t normalDataType;
                        int nDataset;
                        GridLocation_t gloc;
                        DNDS_CGNS_CALL_EXIT(cg_boco_info(cgns_file, iBase, iZone, iBC, boconame, &bcType, &pType, &nPts, NormalIndex, &normalListSize, &normalDataType, &nDataset));
                        DNDS_CGNS_CALL_EXIT(cg_boco_gridlocation_read(cgns_file, iBase, iZone, iBC, &gloc));
                        DNDS_assert_info(pType == PointRange || pType == PointList, "Only PointRange / PointList supported in BC!");
                        if (mesh->dim == 2)
                            DNDS_assert(gloc == EdgeCenter);
                        if (mesh->dim == 3)
                            DNDS_assert(gloc == FaceCenter);
                        std::vector<cgsize_t> pts(nPts);
                        std::vector<double> normalBuf(normalListSize); // should have checked normalDataType, but it is not used here so not checked
                        DNDS_CGNS_CALL_EXIT(cg_boco_read(cgns_file, iBase, iZone, iBC, pts.data(), normalBuf.data()));
                        DNDS_assert(pts[0] >= 1 && pts[1] <= ZoneElems.back()->Size());

                        t_index BCCode = FBCName_2_ID(std::string(boconame));
                        if (BCCode == BC_ID_NULL)
                        {
                            DNDS_assert_info(false, fmt::format("BC NAME [{}] NOT FOUND IN DATABASE", boconame));
                        }
                        if (pType == PointRange)
                        {
                            for (DNDS::index i = pts[0] - 1; i < pts[1]; i++)
                                ZoneElemInfos.back()->operator()(i, 0).zone = BCCode; //! setting BC code
                        }
                        else if (pType == PointList)
                        {
                            for (auto i : pts)
                                ZoneElemInfos.back()->operator()(i - 1, 0).zone = BCCode; //* note that pts is 1-based
                        }
                        else
                            DNDS_assert(false);
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
            char basename[48]{0};
            int celldim = -1;
            int physdim = -1;
            int nzones = -1;

            if (cg_base_read(cgns_file, iBase, basename, &celldim, &physdim))
                cg_error_exit();
            DNDS_assert(celldim == mesh->dim);
            if (cg_nzones(cgns_file, iBase, &nzones))
                cg_error_exit();
            for (int iZone = 1; iZone <= nzones; iZone++)
            {
                char zonename[48]{0};
                std::array<cgsize_t, 9> size{0, 0, 0, 0, 0, 0, 0, 0, 0};
                if (cg_zone_read(cgns_file, iBase, iZone, zonename, size.data()))
                    cg_error_exit();
                ZoneType_t zoneType;
                if (cg_zone_type(cgns_file, iBase, iZone, &zoneType))
                    cg_error_exit();
                DNDS_assert(zoneType == Unstructured); //! only supports unstructured

                DNDS::index nNodes = size[0];
                DNDS::index nVols = size[1];
                DNDS::index nBVertex = size[2];

                //*** read Connectivity
                {
                    ZoneConnect.emplace_back();
                    ZoneConnectDonor.emplace_back();
                    ZoneConnectTargetIZone.emplace_back();
                    int nConns;
                    DNDS_CGNS_CALL_EXIT(cg_nconns(cgns_file, iBase, iZone, &nConns));
                    for (int iConn = 1; iConn <= nConns; iConn++)
                    {
                        char connName[48], donorName[48];
                        GridLocation_t gLoc;
                        GridConnectivityType_t connType;
                        PointSetType_t ptType, ptType_donor;
                        cgsize_t npts, npts_donor;
                        ZoneType_t donorZT;
                        DataType_t donorDT;

                        DNDS_CGNS_CALL_EXIT(cg_conn_info(cgns_file, iBase, iZone, iConn, connName, &gLoc, &connType, &ptType, &npts,
                                                         donorName, &donorZT, &ptType_donor, &donorDT, &npts_donor));

                        DNDS_assert_info(connType == Abutting1to1, "Only support Abutting1to1 in connection!");
                        DNDS_assert_info(ptType == PointList, "Only Supports PointList in connection!");
                        DNDS_assert_info(ptType_donor == PointListDonor, "Only Supports PointListDonor in connection!");
                        DNDS_assert_info(donorZT == Unstructured, "Only Supports Unstructured in connection!");
                        DNDS_assert_info(gLoc == Vertex, "Only Supports Vertex in connection!");
                        DNDS_assert_info(npts_donor == npts, "Only Supports npts_donor == npts in connection!");
                        // std::vector<cgsize_t> ptSet(npts);
                        // std::vector<cgsize_t> ptSet_donor(npts);
                        ZoneConnectDonor.back().emplace_back(npts);
                        ZoneConnect.back().emplace_back(npts);
                        DNDS_CGNS_CALL_EXIT(cg_conn_read(cgns_file, iBase, iZone, iConn, ZoneConnect.back().back().data(), donorDT, ZoneConnectDonor.back().back().data()));
                        int iGZFound = -1;
                        for (size_t iGZ = 0; iGZ < ZoneNames.size(); iGZ++) // find the donor
                        {
                            if (Base_Zone.at(iGZ).first == iBase)
                            {
                                if (ZoneNames.at(iGZ) == donorName)
                                {
                                    iGZFound = iGZ;
                                }
                            }
                        }
                        DNDS_assert(iGZFound >= 0);
                        ZoneConnectTargetIZone.back().push_back(iGZFound);
                        DNDS::log() << "CGNS === Connection at Zone-Zone: " << iZone << " - " << iGZFound + 1 << std::endl;
                    }
                }
            }
        }

        /***************************************************************************/

        /***************************************************************************/
        // ASSEMBLE: deduplicate shared nodes across zones
        auto [NodeOld2New, ZoneNodeStarts] = AssembleZoneNodes(
            ZoneCoords, ZoneConnect, ZoneConnectDonor, ZoneConnectTargetIZone,
            coordSerial, mesh->dim);

        // Separate volume cells and boundary faces
        SeparateVolumeAndBoundaryElements(
            ZoneElems, ZoneElemInfos, NodeOld2New, ZoneNodeStarts,
            cell2nodeSerial, cellElemInfoSerial, cell2cellOrigSerial,
            bnd2nodeSerial, bndElemInfoSerial, bnd2bndOrigSerial,
            node2nodeOrigSerial, coordSerial, mesh->dim);
        /***************************************************************************/

        /***************************************************************************/
        // Build bnd2cell inverse mapping
        BuildBnd2CellSerial(
            bnd2cellSerial, bnd2nodeSerial, cell2nodeSerial,
            cellElemInfoSerial, bndElemInfoSerial, coordSerial);

        // fill in original indices
        for (DNDS::index iCell = 0; iCell < cell2cellOrigSerial->Size(); iCell++)
            cell2cellOrigSerial->operator()(iCell, 0) = iCell;
        for (DNDS::index iNode = 0; iNode < node2nodeOrigSerial->Size(); iNode++)
            node2nodeOrigSerial->operator()(iNode, 0) = iNode;
        for (DNDS::index iBnd = 0; iBnd < bnd2bndOrigSerial->Size(); iBnd++)
            bnd2bndOrigSerial->operator()(iBnd, 0) = iBnd;

        cg_close(cgns_file);

        log() << "CGNS === Serial Read Done" << std::endl;
        // Memory with DM240-120 here: 18G ; after deconstruction done: 7.5G
    }

    void UnstructuredMeshSerialRW::
        ReadFromOpenFOAMAndConvertSerial(const std::string &fName, const std::map<std::string, std::string> &nameMapping, const t_FBCName_2_ID &FBCName_2_ID)
    {
        mode = SerialReadAndDistribute;
        this->dataIsSerialIn = true;

        cell2nodeSerial = make_ssp<decltype(cell2nodeSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::cell2nodeSerial"}, mesh->getMPI());
        bnd2nodeSerial = make_ssp<decltype(bnd2nodeSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::bnd2nodeSerial"}, mesh->getMPI());
        coordSerial = make_ssp<decltype(coordSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::coordSerial"}, mesh->getMPI());
        cellElemInfoSerial = make_ssp<decltype(cellElemInfoSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::cellElemInfoSerial"}, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        bndElemInfoSerial = make_ssp<decltype(bndElemInfoSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::bndElemInfoSerial"}, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        bnd2cellSerial = make_ssp<decltype(bnd2cellSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::bnd2cellSerial"}, mesh->getMPI());

        if (mRank != mesh->getMPI().rank) //! parallel done!!! now serial!!!
            return;

        std::filesystem::path ofFilePath(fName);
        std::ifstream pointsIFS(ofFilePath / "points");
        std::ifstream facesIFS(ofFilePath / "faces");
        std::ifstream ownerIFS(ofFilePath / "owner");
        std::ifstream neighbourIFS(ofFilePath / "neighbour");
        std::ifstream boundaryIFS(ofFilePath / "boundary");

        DNDS_assert_info(pointsIFS.is_open(), "points file not found!");
        DNDS_assert_info(facesIFS.is_open(), "faces file not found!");
        DNDS_assert_info(ownerIFS.is_open(), "owner file not found!");
        DNDS_assert_info(neighbourIFS.is_open(), "neighbour file not found!");
        DNDS_assert_info(boundaryIFS.is_open(), "boundary file not found!");

        OpenFOAM::OpenFOAMReader ofReader;
        ofReader.ReadPoints(pointsIFS);
        ofReader.ReadFaces(facesIFS);
        ofReader.ReadOwner(ownerIFS);
        ofReader.ReadNeighbour(neighbourIFS);
        ofReader.ReadBoundary(boundaryIFS);

        OpenFOAM::OpenFOAMConverter ofConverter;
        ofConverter.BuildFaceElemInfo(ofReader);
        ofConverter.BuildCell2Face(ofReader);
        ofConverter.BuildCell2Node(ofReader);

        coordSerial->Resize(ofReader.points.size());
        for (index iN = 0; iN < coordSerial->Size(); iN++)
            (*coordSerial)[iN] = ofReader.points.at(iN);
        log() << "OpenFOAM === got num node: " << coordSerial->Size() << std::endl;

        cell2nodeSerial->Resize(ofConverter.cell2node.size());
        cellElemInfoSerial->Resize(ofConverter.cell2node.size());
        for (index iC = 0; iC < cell2nodeSerial->Size(); iC++)
        {
            cell2nodeSerial->ResizeRow(iC, ofConverter.cell2node[iC].size());
            for (size_t iN = 0; iN < ofConverter.cell2node[iC].size(); iN++)
                (*cell2nodeSerial)[iC][iN] = ofConverter.cell2node[iC][iN];
            cellElemInfoSerial->operator()(iC, 0) = ofConverter.cellElemInfo.at(iC);
        }

        log() << "OpenFOAM === got num cell: " << cell2nodeSerial->Size() << std::endl;

        index nBnd = ofReader.owner.size() - ofReader.neighbour.size();
        bnd2nodeSerial->Resize(nBnd);
        bndElemInfoSerial->Resize(nBnd);
        bnd2cellSerial->Resize(nBnd);
        for (index iBnd = 0; iBnd < nBnd; iBnd++)
        {
            index iFaceOF = iBnd + ofReader.neighbour.size();
            bnd2nodeSerial->ResizeRow(iBnd, ofReader.faces[iFaceOF].size());
            for (size_t ib2c = 0; ib2c < ofReader.faces[iFaceOF].size(); ib2c++)
                (*bnd2nodeSerial)[iBnd][ib2c] = ofReader.faces[iFaceOF][ib2c];
            bnd2cellSerial->operator()(iBnd, 0) = ofReader.owner.at(iFaceOF);
            bnd2cellSerial->operator()(iBnd, 1) = UnInitIndex;
            bndElemInfoSerial->operator()(iBnd, 0) = ofConverter.faceElemInfo.at(iFaceOF);
        }

        log() << "nameMapping size: " << nameMapping.size() << std::endl;

        for (auto &bc : ofReader.boundaryConditions)
        {
            auto boconame = bc.first;
            if (nameMapping.count(boconame))
                boconame = nameMapping.at(boconame);
            t_index BCCode = FBCName_2_ID(std::string(boconame));
            if (BCCode == BC_ID_NULL)
            {
                DNDS_assert_info(false, fmt::format("BC NAME [{}] NOT FOUND IN DATABASE", boconame));
            }
            for (index iBndC = 0; iBndC < bc.second.nFaces; iBndC++)
                bndElemInfoSerial->operator()(iBndC + bc.second.startFace - ofReader.neighbour.size(), 0).zone = BCCode;
        }
        for (index iBnd = 0; iBnd < nBnd; iBnd++)
        {
            DNDS_assert(bndElemInfoSerial->operator()(iBnd, 0).zone != BC_ID_NULL);
        }

        log() << "OpenFOAM === got num bnd: " << nBnd << std::endl;

        log() << "OpenFOAM === Serial Read Done" << std::endl;
    }
}