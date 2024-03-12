#include "Mesh.hpp"

#include <set>

#include <nanoflann.hpp>

#include "PointCloud.hpp"
#include <unordered_map>

namespace DNDS::Geom
{

    void UnstructuredMeshSerialRW::
        Deduplicate1to1Periodic(real search_eps) // currently does not handle parallel input mode
    {
        DNDS_assert(this->dataIsSerialIn);
        // TODO: build periodic donor oct-trees
        // TODO: search in donors if periodic main and connect cell2cell

        PointCloudKDTree periodicDonorCenter1;
        PointCloudKDTree periodicDonorCenter2;
        PointCloudKDTree periodicDonorCenter3;
        std::vector<index> periodicDonorIBnd1;
        std::vector<index> periodicDonorIBnd2;
        std::vector<index> periodicDonorIBnd3;

        for (DNDS::index iBnd = 0; iBnd < bnd2cellSerial->Size(); iBnd++)
        {
            auto faceID = bndElemInfoSerial->operator()(iBnd, 0).zone;
            Eigen::Matrix<real, 3, Eigen::Dynamic> coordBnd;
            {
                coordBnd.resize(3, bnd2nodeSerial->RowSize(iBnd));
                for (rowsize ib2n = 0; ib2n < bnd2nodeSerial->RowSize(iBnd); ib2n++)
                    coordBnd(Eigen::all, ib2n) = coordSerial->operator[]((*bnd2nodeSerial)(iBnd, ib2n));
            }
            tPoint faceCent = coordBnd.rowwise().mean();

            if (FaceIDIsPeriodicMain(faceID))
            {
            }
            if (FaceIDIsPeriodicDonor(faceID))
            {
                if (faceID == BC_ID_PERIODIC_1_DONOR)
                    periodicDonorCenter1.pts.push_back(faceCent), periodicDonorIBnd1.push_back(iBnd);
                if (faceID == BC_ID_PERIODIC_2_DONOR)
                    periodicDonorCenter2.pts.push_back(faceCent), periodicDonorIBnd2.push_back(iBnd);
                if (faceID == BC_ID_PERIODIC_3_DONOR)
                    periodicDonorCenter3.pts.push_back(faceCent), periodicDonorIBnd3.push_back(iBnd);
            }
        }
        index nDonor = periodicDonorCenter1.pts.size() +
                       periodicDonorCenter2.pts.size() +
                       periodicDonorCenter3.pts.size();
        index nDonorAll{0};
        MPI::Allreduce(&nDonor, &nDonorAll, 1, DNDS_MPI_INDEX, MPI_SUM, mesh->getMPI().comm);
        if (nDonorAll)
            mesh->isPeriodic = true; //! below are all periodic code
        else
        {
            mesh->isPeriodic = false;
            return;
        }

        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTree>,
            PointCloudKDTree,
            3,
            index>;
        kdtree_t periodicDonorTree1(3, periodicDonorCenter1);
        kdtree_t periodicDonorTree2(3, periodicDonorCenter2);
        kdtree_t periodicDonorTree3(3, periodicDonorCenter3);

        std::unordered_map<index, index> periodicMainToDonor1;
        std::unordered_map<index, index> periodicMainToDonor2;
        std::unordered_map<index, index> periodicMainToDonor3;

        std::unordered_map<index, index> iNodeDonorToMain1;
        std::unordered_map<index, index> iNodeDonorToMain2;
        std::unordered_map<index, index> iNodeDonorToMain3;

        for (DNDS::index iBnd = 0; iBnd < bnd2cellSerial->Size(); iBnd++)
        {
            auto faceID = bndElemInfoSerial->operator()(iBnd, 0).zone;
            Eigen::Matrix<real, 3, Eigen::Dynamic> coordBnd;
            {
                coordBnd.resize(3, bnd2nodeSerial->RowSize(iBnd));
                for (rowsize ib2n = 0; ib2n < bnd2nodeSerial->RowSize(iBnd); ib2n++)
                    coordBnd(Eigen::all, ib2n) = coordSerial->operator[]((*bnd2nodeSerial)(iBnd, ib2n));
            }
            tPoint faceCent = coordBnd.rowwise().mean();

            if (FaceIDIsPeriodicMain(faceID))
            {
                tPoint faceCentTrans = mesh->periodicInfo.TransCoord(faceCent, faceID);
                std::vector<index> outIndices(2);
                Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;
                outDistancesSqr.resize(2);
                size_t nResults{0};
                if (faceID == BC_ID_PERIODIC_1)
                    nResults = periodicDonorTree1.knnSearch(faceCentTrans.data(), 2, outIndices.data(), outDistancesSqr.data());
                if (faceID == BC_ID_PERIODIC_2)
                    nResults = periodicDonorTree2.knnSearch(faceCentTrans.data(), 2, outIndices.data(), outDistancesSqr.data());
                if (faceID == BC_ID_PERIODIC_3)
                    nResults = periodicDonorTree3.knnSearch(faceCentTrans.data(), 2, outIndices.data(), outDistancesSqr.data());
                DNDS_assert_info(nResults >= 1, "Tree Search of periodic result number not enough");

                // std::cout << outDistancesSqr[0] << std::endl;
                DNDS_assert_info(outDistancesSqr[0] <= sqr(search_eps),
                                 fmt::format("nearest neighbour dist{} not matching under the threshold",
                                             outDistancesSqr[0]));
                if (nResults > 1)
                    DNDS_assert_info(outDistancesSqr[1] > sqr(search_eps),
                                     fmt::format("second nearest neighbour dist{} not matching over the threshold",
                                                 outDistancesSqr[1]));
                index donorIBnd{-1};
                if (faceID == BC_ID_PERIODIC_1)
                    donorIBnd = periodicDonorIBnd1.at(outIndices[0]), periodicMainToDonor1[iBnd] = donorIBnd;
                if (faceID == BC_ID_PERIODIC_2)
                    donorIBnd = periodicDonorIBnd2.at(outIndices[0]), periodicMainToDonor2[iBnd] = donorIBnd;
                if (faceID == BC_ID_PERIODIC_3)
                    donorIBnd = periodicDonorIBnd3.at(outIndices[0]), periodicMainToDonor3[iBnd] = donorIBnd;
                Eigen::Matrix<real, 3, Eigen::Dynamic> coordBndOther;
                {
                    coordBndOther.resize(3, bnd2nodeSerial->RowSize(donorIBnd));
                    for (rowsize ib2n = 0; ib2n < bnd2nodeSerial->RowSize(donorIBnd); ib2n++)
                        coordBndOther(Eigen::all, ib2n) = // put onto main's data
                            coordSerial->operator[]((*bnd2nodeSerial)(donorIBnd, ib2n));
                }
                DNDS_assert(coordBndOther.cols() == coordBnd.cols());
                for (rowsize ib2n = 0; ib2n < coordBnd.cols(); ib2n++)
                {
                    bool found = false;
                    for (rowsize jb2n = 0; jb2n < coordBnd.cols(); jb2n++)
                        if ((coordBndOther(Eigen::all, jb2n) -
                             mesh->periodicInfo.TransCoord(coordBnd(Eigen::all, ib2n), faceID))
                                .squaredNorm() < sqr(search_eps))
                        {
                            found = true;
                            if (faceID == BC_ID_PERIODIC_1)
                                iNodeDonorToMain1[(*bnd2nodeSerial)(donorIBnd, jb2n)] = (*bnd2nodeSerial)(iBnd, ib2n);
                            if (faceID == BC_ID_PERIODIC_2)
                                iNodeDonorToMain2[(*bnd2nodeSerial)(donorIBnd, jb2n)] = (*bnd2nodeSerial)(iBnd, ib2n);
                            if (faceID == BC_ID_PERIODIC_3)
                                iNodeDonorToMain3[(*bnd2nodeSerial)(donorIBnd, jb2n)] = (*bnd2nodeSerial)(iBnd, ib2n);
                        }
                    DNDS_assert(found);
                }
            }
            if (FaceIDIsPeriodicDonor(faceID))
            {
            }
        }
        /******************************************************************/
        // need to be done before cell2node points to de-duplicated
        /**********************************************************************************************************************/
        // getting node2cell // only primary vertices
        std::unordered_map<index, std::vector<index>> donorNode2Cell;
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            for (DNDS::rowsize iN = 0; iN < Elem::Element{(*cellElemInfoSerial)(iCell, 0).getElemType()}.GetNumVertices(); iN++)
            {
                auto iNode = (*cell2nodeSerial)(iCell, iN);
                if (iNodeDonorToMain1.count(iNode) || iNodeDonorToMain2.count(iNode) || iNodeDonorToMain3.count(iNode))
                {
                    donorNode2Cell[iNode].push_back(iCell);
                }
            }
        /**********************************************************************************************************************/

        DNDS_MAKE_SSP(cell2nodePbiSerial, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->getMPI());
        cell2nodePbiSerial->Resize(cell2nodeSerial->Size());
        for (index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            cell2nodePbiSerial->ResizeRow(iCell, cell2nodeSerial->RowSize(iCell));
        for (index iBnd = 0; iBnd < bnd2nodeSerial->Size(); iBnd++)
        {
            auto faceID = bndElemInfoSerial->operator()(iBnd, 0).zone;
            if (FaceIDIsPeriodicDonor(faceID))
            {
                for (auto iNodeFace : (*bnd2nodeSerial)[iBnd])
                {
                    DNDS_assert(donorNode2Cell.count(iNodeFace)); // node 2 cell means all the cells in bound are calculated
                    for (auto iCell : donorNode2Cell[iNodeFace])
                        for (auto iNode : (*bnd2nodeSerial)[iBnd])
                            for (rowsize ic2n = 0; ic2n < (*cell2nodeSerial).RowSize(iCell); ic2n++)
                            {
                                if ((*cell2nodeSerial)(iCell, ic2n) == iNode)
                                {
                                    if (faceID == BC_ID_PERIODIC_1_DONOR)
                                        (*cell2nodePbiSerial)(iCell, ic2n).setP1True();
                                    if (faceID == BC_ID_PERIODIC_2_DONOR)
                                        (*cell2nodePbiSerial)(iCell, ic2n).setP2True();
                                    if (faceID == BC_ID_PERIODIC_3_DONOR)
                                        (*cell2nodePbiSerial)(iCell, ic2n).setP3True();
                                }
                            }
                }
            }
        }
        /******************************************************************/

        std::vector<index> iNodeOld2New(coordSerial->Size(), 0);
        index nNodeNew;

        auto getNewNodeMap = [&](const decltype(iNodeDonorToMain1) &iNodeDonorToMainI) -> void
        {
            nNodeNew = 0;
            for (index iNode = 0; iNode < coordSerial->Size(); iNode++)
            {
                if (iNodeOld2New[iNode] < 0)
                    continue;
                bool isDonor = iNodeDonorToMainI.count(iNode);
                if (!isDonor)
                    iNodeOld2New[iNode] = nNodeNew++;
            }
            for (auto &p : iNodeDonorToMainI)
            {
                index iMain = p.second;
                iNodeOld2New.at(p.first) = -1 - iMain; // minus means duplication here; leaves a self-pointing pointer in the minus region
            }
        };
        getNewNodeMap(iNodeDonorToMain1);
        getNewNodeMap(iNodeDonorToMain2);
        getNewNodeMap(iNodeDonorToMain3);

        auto track = [&](index iNode) -> index
        {
            while (iNodeOld2New.at(iNode) < 0)
                iNode = -1 - iNodeOld2New[iNode];
            return iNodeOld2New.at(iNode); // recursive tracker
        };

        for (index i = 0; i < cell2nodeSerial->Size(); i++)
            for (auto &ii : (*cell2nodeSerial)[i])
                ii = track(ii);
        for (index i = 0; i < bnd2nodeSerial->Size(); i++)
            for (auto &ii : (*bnd2nodeSerial)[i])
                ii = track(ii);

        // some assertions:65
        // for (index iBnd = 0; iBnd < bnd2nodeSerial->Size(); iBnd++)
        // {
        //     auto faceID = bndElemInfoSerial->operator()(iBnd, 0).zone;
        //     if (FaceIDIsPeriodicDonor(faceID))
        //     {

        //     }
        // } //TODO

        cell2nodePbiSerial->Compress();

        decltype(coordSerial) coordSerialOld = coordSerial;
        DNDS_MAKE_SSP(coordSerial, mesh->getMPI());
        coordSerial->Resize(nNodeNew);
        for (index i = 0; i < coordSerialOld->Size(); i++)
            if (iNodeOld2New[i] >= 0)
                (*coordSerial)[iNodeOld2New[i]] = (*coordSerialOld)[i];
        coordSerialOld.reset();
    }

    void UnstructuredMeshSerialRW::
        BuildCell2Cell() // currently does not handle parallel input mode
    {
        DNDS_assert(this->dataIsSerialIn);
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  BuildCell2Cell" << std::endl;
        DNDS_MAKE_SSP(cell2cellSerial, mesh->getMPI());
        // if (mRank != mesh->getMPI().rank)
        //     return;
        /// TODO: abstract these: invert cone (like node 2 cell -> cell 2 node) (also support operating on pair)
        /**********************************************************************************************************************/
        // getting node2cell // only primary vertices
        std::vector<std::vector<DNDS::index>>
            node2cell(coordSerial->Size());
        // std::vector<DNDS::rowsize> node2cellSz(coordSerial->Size(), 0);
        // for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
        //     for (DNDS::rowsize iN = 0; iN < Elem::Element{(*cellElemInfoSerial)(iCell, 0).getElemType()}.GetNumVertices(); iN++)
        //         node2cellSz[(*cell2nodeSerial)(iCell, iN)]++;
        // for (DNDS::index iNode = 0; iNode < node2cell.size(); iNode++)
        //     node2cell[iNode].reserve(node2cellSz[iNode]);
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            for (DNDS::rowsize iN = 0; iN < Elem::Element{(*cellElemInfoSerial)(iCell, 0).getElemType()}.GetNumVertices(); iN++)
                node2cell[(*cell2nodeSerial)(iCell, iN)].push_back(iCell);
        // node2cellSz.clear();
        /**********************************************************************************************************************/
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  BuildCell2Cell Part 1" << std::endl;
        cell2cellSerial->Resize(cell2nodeSerial->Size());

        index nCells = cell2nodeSerial->Size();
        index nCellsDone = 0;
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
        {
            auto CellElem = Elem::Element{(*cellElemInfoSerial)[iCell]->getElemType()};
            std::vector<DNDS::index> cell2nodeRow{
                (*cell2nodeSerial)[iCell].begin(),
                (*cell2nodeSerial)[iCell].begin() + CellElem.GetNumVertices()};
            std::sort(cell2nodeRow.begin(), cell2nodeRow.end());
            // only primary vertices
            std::set<DNDS::index> c_neighbors; // could optimize?
                                               // std::vector<DNDS::index> c_neighbors;
                                               // c_neighbors.reserve(30);
                                               /****/
#ifdef DNDS_USE_OMP
            // #pragma omp single
#pragma omp critical
#endif
            {
                if (nCellsDone % (nCells / 1000 + 1) == 0)
                {
                    auto fmt = DNDS::log().flags();
                    DNDS::log() << "\r\033[K" << std::setw(5) << double(nCellsDone) / nCells * 100 << "%";
                    DNDS::log().flush();
                    DNDS::log().setf(fmt);
                }

                if (nCellsDone == nCells - 1)
                    DNDS::log()
                        // << "\r\033[K"
                        << std::endl;
            }
            /****/

            for (auto iNode : cell2nodeRow)
                for (auto iCellOther : node2cell[iNode])
                {
                    if (iCellOther == iCell || c_neighbors.count(iCellOther) == 1)
                        continue;
                    //! ** override: point-complete cell2cell info!
                    c_neighbors.insert(iCellOther);
                    continue;
                    //! ** override: point-complete cell2cell info!

                    auto CellElemO = Elem::Element{(*cellElemInfoSerial)[iCellOther]->getElemType()};
                    std::vector<DNDS::index> cell2nodeRowO{
                        (*cell2nodeSerial)[iCellOther].begin(),
                        (*cell2nodeSerial)[iCellOther].begin() + CellElemO.GetNumVertices()};
                    std::sort(cell2nodeRowO.begin(), cell2nodeRowO.end());
                    std::vector<DNDS::index> interSect;
                    interSect.reserve(32);
                    std::set_intersection(cell2nodeRowO.begin(), cell2nodeRowO.end(), cell2nodeRow.begin(), cell2nodeRow.end(),
                                          std::back_inserter(interSect));
                    if ((iCellOther != iCell) && interSect.size() > (mesh->dim - 1)) //! mesh->dim - 1 is a simple method

                        // (false ||
                        //  Elem::cellsAreFaceConnected(
                        //      cell2nodeSerial->operator[](iCell),
                        //      cell2nodeSerial->operator[](iCellOther),
                        //      CellElem,
                        //      Elem::Element{(*cellElemInfoSerial)[iCellOther]->getElemType()}))
                        c_neighbors.insert(iCellOther);
                    // c_neighbors.push_back(iCellOther);
                }
            // if (!c_neighbors.erase(iCell))
            //     DNDS_assert_info(false, "neighbors should include myself now");
            // std::sort(c_neighbors.begin(), c_neighbors.end());
            // auto last = std::unique(c_neighbors.begin(), c_neighbors.end());
            // c_neighbors.erase(last, c_neighbors.end());

            cell2cellSerial->ResizeRow(iCell, c_neighbors.size());
            DNDS::rowsize ic2c = 0;
            for (auto iCellOther : c_neighbors)
                (*cell2cellSerial)(iCell, ic2c++) = iCellOther;
#ifdef DNDS_USE_OMP
#pragma omp atomic
#endif
            nCellsDone++;
        }

        /*************************************************************************************************/
        DNDS_MAKE_SSP(cell2cellSerialFacial, mesh->getMPI());
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  BuildCell2Cell Part 2" << std::endl;
        cell2cellSerialFacial->Resize(cell2cellSerial->Size());
        nCellsDone = 0;
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (index iCell = 0; iCell < cell2cellSerial->Size(); iCell++)
        {
#ifdef DNDS_USE_OMP
#pragma omp critical
#endif
            {
                if (nCellsDone % (nCells / 1000 + 1) == 0)
                {
                    auto fmt = DNDS::log().flags();
                    DNDS::log() << "\r\033[K" << std::setw(5) << double(nCellsDone) / nCells * 100 << "%";
                    DNDS::log().flush();
                    DNDS::log().setf(fmt);
                }

                if (nCellsDone == nCells - 1)
                    DNDS::log()
                        // << "\r\033[K"
                        << std::endl;
            }
            std::vector<index> facialNeighbors;
            facialNeighbors.reserve(6);
            std::vector<index> c2ni((*cell2nodeSerial)[iCell].begin(),
                                    (*cell2nodeSerial)[iCell].begin() +
                                        Elem::Element{(*cellElemInfoSerial)[iCell]->getElemType()}.GetNumVertices());
            std::sort(c2ni.begin(), c2ni.end());
            for (auto iCellOther : (*cell2cellSerial)[iCell])
            {
                std::vector<index> c2nj((*cell2nodeSerial)[iCellOther].begin(),
                                        (*cell2nodeSerial)[iCellOther].begin() +
                                            Elem::Element{(*cellElemInfoSerial)[iCellOther]->getElemType()}.GetNumVertices());
                std::sort(c2nj.begin(), c2nj.end());
                std::vector<index> intersect;
                intersect.reserve(9);
                std::set_intersection(c2ni.begin(), c2ni.end(), c2nj.begin(), c2nj.end(), std::back_inserter(intersect));
                if (intersect.size() >= mesh->dim) // for 2d, exactly 2; for 3d, 3 or 4
                    facialNeighbors.push_back(iCellOther);
            }
            (*cell2cellSerialFacial).ResizeRow(iCell, facialNeighbors.size());
            (*cell2cellSerialFacial)[iCell] = facialNeighbors;
#ifdef DNDS_USE_OMP
#pragma omp atomic
#endif
            nCellsDone++;
        }
        /*************************************************************************************************/

        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Done  BuildCell2Cell" << std::endl;
    }
}