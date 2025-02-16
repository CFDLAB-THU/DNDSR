#include "Mesh.hpp"
namespace _METIS
{
#include <metis.h>
#include <parmetis.h>
}

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/minimum_degree_ordering.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

#include "CorrectRCM.hpp"

namespace _METIS
{
    static idx_t indexToIdx(DNDS::index v)
    {
        if constexpr (sizeof(DNDS::index) <= sizeof(idx_t))
            return v;
        else
            return DNDS::checkedIndexTo32(v);
    }
}

namespace DNDS::Geom
{
    void UnstructuredMeshSerialRW::
        MeshPartitionCell2Cell(const PartitionOptions &c_options)
    {
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  MeshPartitionCell2Cell" << std::endl;
        //! preset hyper config, should be optional in the future
        bool isSerial = true;
        _METIS::idx_t nPart = mesh->getMPI().size;
        cnPart = nPart;

        //! assuming all adj point to local numbers now
        // * Tend to local-global issues putting into
        // cell2cellSerial->Compress();
        // cell2cellSerial->AssertConsistent();
        // cell2cellSerial->createGlobalMapping();
        cell2cellSerialFacial->Compress();
        cell2cellSerialFacial->AssertConsistent();
        cell2cellSerialFacial->createGlobalMapping();

        std::vector<_METIS::idx_t> vtxdist(mesh->getMPI().size + 1);
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::MPI_int r = 0; r <= mesh->getMPI().size; r++)
            vtxdist[r] = _METIS::indexToIdx(cell2cellSerialFacial->pLGlobalMapping->ROffsets().at(r));
        std::vector<_METIS::idx_t> xadj(cell2cellSerialFacial->Size() + 1);
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iCell = 0; iCell < xadj.size(); iCell++)
            xadj[iCell] = _METIS::indexToIdx(cell2cellSerialFacial->rowPtr(iCell) - cell2cellSerialFacial->rowPtr(0));
        std::vector<_METIS::idx_t> adjncy(xadj.back());
        std::vector<_METIS::idx_t> adjncyWeights;
        DNDS_assert(cell2cellSerialFacial->DataSize() == xadj.back());
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iAdj = 0; iAdj < xadj.back(); iAdj++)
            adjncy[iAdj] = _METIS::indexToIdx(cell2cellSerialFacial->data()[iAdj]);
        if (c_options.edgeWeightMethod == 1)
        {
            adjncyWeights.reserve(xadj.back());
            std::vector<real> adjncyWeightsR;
            adjncyWeightsR.reserve(xadj.back());

            real maxDistMax{0};
            for (index iCell = 0; iCell < cell2cellSerialFacial->Size(); iCell++)
            {
                tSmallCoords coordsC;
                GetCoordsOnCellSerial(iCell, coordsC, coordSerial);
                std::vector<index> cell2nodeCV = cell2nodeSerial->operator[](iCell);
                std::sort(cell2nodeCV.begin(), cell2nodeCV.end());
                for (auto iCellOther : cell2cellSerialFacial->operator[](iCell))
                {
                    std::vector<index> cell2nodeCVOther = cell2nodeSerial->operator[](iCellOther);
                    std::sort(cell2nodeCVOther.begin(), cell2nodeCVOther.end());
                    std::vector<index> faceFound;
                    faceFound.reserve(9);
                    std::set_intersection(cell2nodeCV.begin(), cell2nodeCV.end(), cell2nodeCVOther.begin(), cell2nodeCVOther.end(), std::back_inserter(faceFound));
                    DNDS_assert(faceFound.size() >= 2);
                    std::vector<int> faceFoundC2F;
                    faceFoundC2F.reserve(faceFound.size());
                    for (auto iN : faceFound)
                        for (int i = 0; i < cell2nodeSerial->operator[](iCell).size(); i++)
                            if (cell2nodeSerial->operator[](iCell)[i] == iN)
                                faceFoundC2F.push_back(i);
                    DNDS_assert(faceFoundC2F.size() == faceFound.size());
                    tSmallCoords coordsF = coordsC(Eigen::all, faceFoundC2F);
                    real maxDist{0};
                    for (int i = 0; i < coordsF.cols(); i++)
                        for (int j = 0; j < coordsF.cols(); j++)
                            if (i != j)
                                maxDist = std::max(maxDist, (coordsF(Eigen::all, i) - coordsF(Eigen::all, j)).norm());
                    adjncyWeightsR.push_back(maxDist);
                    maxDistMax = std::max(maxDist, maxDistMax);
                }
            }
            auto weightMapping = [](real x) -> real
            { return std::pow(x, 1); };
            for (auto d : adjncyWeightsR)
                adjncyWeights.push_back(weightMapping(d / maxDistMax) * (INT_MAX - 1) + 1);
        }
        if (adjncy.empty())
            adjncy.resize(1, -1); //*coping with zero sized data

        _METIS::idx_t nCell = _METIS::indexToIdx(cell2cellSerialFacial->Size());
        _METIS::idx_t nCon{1}, options[METIS_NOPTIONS];
        _METIS::METIS_SetDefaultOptions(options);
        {
            options[_METIS::METIS_OPTION_OBJTYPE] = _METIS::METIS_OBJTYPE_CUT;
            options[_METIS::METIS_OPTION_CTYPE] = _METIS::METIS_CTYPE_SHEM; //? could try shem?
            options[_METIS::METIS_OPTION_IPTYPE] = _METIS::METIS_IPTYPE_GROW;
            options[_METIS::METIS_OPTION_RTYPE] = _METIS::METIS_RTYPE_FM;
            // options[METIS_OPTION_NO2HOP] = 0; // only available in metis 5.1.0
            options[_METIS::METIS_OPTION_NCUTS] = std::max(c_options.metisNcuts, 1);
            options[_METIS::METIS_OPTION_NITER] = 10;
            // options[_METIS::METIS_OPTION_UFACTOR] = 30; // load imbalance factor, fow k-way
            options[_METIS::METIS_OPTION_UFACTOR] = c_options.metisUfactor;
            options[_METIS::METIS_OPTION_MINCONN] = 1;
            options[_METIS::METIS_OPTION_CONTIG] = 1;                 // ! forcing contigious partition now ? necessary?
            options[_METIS::METIS_OPTION_SEED] = c_options.metisSeed; // ! seeding 0 for determined result
            options[_METIS::METIS_OPTION_NUMBERING] = 0;
            // options[_METIS::METIS_OPTION_DBGLVL] = _METIS::METIS_DBG_TIME | _METIS::METIS_DBG_IPART;
            options[_METIS::METIS_OPTION_DBGLVL] = _METIS::METIS_DBG_TIME;
        }
        std::vector<_METIS::idx_t> partOut(nCell);
        if (nCell == 0)
            partOut.resize(1, -1); //*coping with zero sized data
        if (nPart > 1)
        {
            if (mesh->getMPI().size == 1 || (isSerial && mesh->getMPI().rank == mRank))
            {
                _METIS::idx_t objval;
                DNDS_assert_info(c_options.metisType == std::string("KWAY") or c_options.metisType == std::string("RB"), "metisType must be KWAY or RB!");
                int ret = c_options.metisType == std::string("KWAY")
                              ? _METIS::METIS_PartGraphKway(
                                    &nCell, &nCon, xadj.data(), adjncy.data(), NULL, NULL, c_options.edgeWeightMethod ? adjncyWeights.data() : NULL,
                                    &nPart, NULL, NULL, options, &objval, partOut.data())
                              : _METIS::METIS_PartGraphRecursive(
                                    &nCell, &nCon, xadj.data(), adjncy.data(), NULL, NULL, c_options.edgeWeightMethod ? adjncyWeights.data() : NULL,
                                    &nPart, NULL, NULL, options, &objval, partOut.data());
                if (ret != _METIS::METIS_OK)
                {
                    DNDS::log() << "METIS returned not OK: [" << ret << "]" << std::endl;
                    DNDS_assert(false);
                }
            }
            else if (mesh->getMPI().size != 1 && (!isSerial))
            {
                ///@todo //TODO: parmetis needs testing!
                for (int i = 0; i < vtxdist.size() - 1; i++)
                    DNDS_assert_info(vtxdist[i + 1] - vtxdist[i] > 0, "need more than zero cells on each proc!");
                std::vector<_METIS::real_t> tpWeights(nPart * nCon, 1.0 / nPart); //! assuming homogenous
                _METIS::real_t ubVec[1]{1.05};
                DNDS_assert(nCon == 1);
                _METIS::idx_t optsC[3];
                _METIS::idx_t wgtflag{0}, numflag{0};
                optsC[0] = 1;
                optsC[1] = 1;
                optsC[2] = 0;
                _METIS::idx_t objval;
                int ret = _METIS::ParMETIS_V3_PartKway(
                    vtxdist.data(), xadj.data(), adjncy.data(), NULL, NULL, &wgtflag, &numflag,
                    &nCon, &nPart, tpWeights.data(), ubVec, optsC, &objval, partOut.data(),
                    &mesh->getMPI().comm);
                if (ret != _METIS::METIS_OK)
                {
                    DNDS::log() << "METIS returned not OK: [" << ret << "]" << std::endl;
                    DNDS_assert(false);
                }
            }
        }
        else
        {
            partOut.assign(partOut.size(), 0);
        }
        cellPartition.resize(cell2cellSerialFacial->Size());
        for (DNDS::index i = 0; i < cellPartition.size(); i++)
            cellPartition[i] = partOut[i];
        if (mesh->getMPI().rank == mRank)
        {
            std::vector<index> partCellCnt(nPart, 0);
            for (auto p : cellPartition)
                partCellCnt.at(p)++;
            auto [min, max] = std::minmax_element(partCellCnt.begin(), partCellCnt.end());
            log() << "UnstructuredMeshSerialRW === Done  MeshPartitionCell2Cell "
                  << fmt::format("ave [{}], min [{}], max [{}], ", real(cellPartition.size()) / nPart, *min, *max)
                  << fmt::format("ratio [{:.4f}] ", real(*min) / *max) << std::endl;
        }
    }

    // void UnstructuredMesh::ReorderCellLocal()
    // {
    //     //! currently not used

    //     /****************************/
    //     // got reordering, iCellNew = iPerm[iCell], iCell = perm[iCellNew], A(perm,perm) = Anew

    //     //! remember to add all cell related indices here for altering
    //     // DNDS_assert(this->adjPrimaryState == Adj_PointToLocal);
    //     // DNDS_assert(this->adjFacialState == Adj_PointToLocal);
    //     // DNDS_assert(this->adjC2FState == Adj_PointToLocal);
    // }

    void UnstructuredMesh::ObtainLocalFactFillOrdering(Direct::SerialSymLUStructure &symLU, Direct::DirectPrecControl control)
    {
        if (!control.useDirectPrec)
            return;
        cell2cellFaceVLocal = this->GetCell2CellFaceVLocal();
        auto &localFillOrderingNew2Old = symLU.localFillOrderingNew2Old;
        auto &localFillOrderingOld2New = symLU.localFillOrderingOld2New;
        if (!this->NumCell())
            return;
        if (control.getOrderingCode() == -1)
        {
            localFillOrderingNew2Old.reserve(this->NumCell());
            for (index i = 0; i < this->NumCell(); i++)
                if (i % 2 == 0)
                    localFillOrderingNew2Old.push_back(i);
            for (index i = 0; i < this->NumCell(); i++)
                if (i % 2 != 0)
                    localFillOrderingNew2Old.push_back(i);
            localFillOrderingOld2New.resize(this->NumCell());
            for (index i = 0; i < this->NumCell(); i++)
                localFillOrderingOld2New.at(localFillOrderingNew2Old.at(i)) = i;
        }
        else if (control.getOrderingCode() == 0)
        {
            // do nothing, natural order
        }
        else if (control.getOrderingCode() == 1) // Metis
        {
            _METIS::idx_t nCell = _METIS::indexToIdx(this->NumCell());
            _METIS::idx_t nCon{1}, options[METIS_NOPTIONS];
            _METIS::METIS_SetDefaultOptions(options);
            {
                options[_METIS::METIS_OPTION_CTYPE] = _METIS::METIS_CTYPE_SHEM;
                options[_METIS::METIS_OPTION_RTYPE] = _METIS::METIS_RTYPE_FM;
                options[_METIS::METIS_OPTION_IPTYPE] = _METIS::METIS_IPTYPE_EDGE;
                options[_METIS::METIS_OPTION_RTYPE] = _METIS::METIS_RTYPE_SEP1SIDED;
                options[_METIS::METIS_OPTION_NSEPS] = 1;
                options[_METIS::METIS_OPTION_NITER] = 10;
                options[_METIS::METIS_OPTION_UFACTOR] = 30;
                options[_METIS::METIS_OPTION_COMPRESS] = 0; // do not compress
                // options[_METIS::METIS_OPTION_CCORDER] = 0; //use default?
                options[_METIS::METIS_OPTION_SEED] = 0;    // ! seeding 0 for determined result
                options[_METIS::METIS_OPTION_PFACTOR] = 0; // not removing large vertices
                options[_METIS::METIS_OPTION_NUMBERING] = 0;
                // options[_METIS::METIS_OPTION_DBGLVL] = _METIS::METIS_DBG_TIME | _METIS::METIS_DBG_IPART;
            }
            std::vector<std::vector<index>> &cell2cellFaceV = cell2cellFaceVLocal;
            std::vector<_METIS::idx_t> adjncy, xadj, perm, iPerm;
            xadj.resize(nCell + 1);
            xadj[0] = 0;
            for (_METIS::idx_t iC = 0; iC < nCell; iC++)
                xadj[iC + 1] = xadj[iC] + cell2cellFaceV[iC].size(); //! check overflow!
            adjncy.resize(xadj.back());
            for (_METIS::idx_t iC = 0; iC < nCell; iC++)
                std::copy(cell2cellFaceV[iC].begin(), cell2cellFaceV[iC].end(), adjncy.begin() + xadj[iC]);
            perm.resize(nCell);
            iPerm.resize(nCell);

            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling metis" << std::endl;

            int ret = _METIS::METIS_NodeND(&nCell, xadj.data(), adjncy.data(), NULL, options, perm.data(), iPerm.data());
            DNDS_assert_info(ret == _METIS::METIS_OK, fmt::format("Metis return not ok, [{}]", ret));

            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): metis done" << std::endl;

            localFillOrderingNew2Old.resize(nCell);
            localFillOrderingOld2New.resize(nCell);
            for (index i = 0; i < this->NumCell(); i++)
            {
                localFillOrderingNew2Old[i] = perm[i];
                localFillOrderingOld2New[i] = iPerm[i];
            }
        }
        else if (control.getOrderingCode() == 2) // MMD
        {
            using namespace boost;
            using Graph = adjacency_list<vecS, vecS, directedS>;
            Graph cell2cellG(this->NumCell());
            std::vector<std::vector<index>> &cell2cellFaceV = cell2cellFaceVLocal;
            for (index iCell = 0; iCell < this->NumCell(); iCell++)
                for (auto iCOther : cell2cellFaceV[iCell])
                    add_edge(iCell, iCOther, cell2cellG);
            std::vector<index> supernodeSizes(this->NumCell(), 1), degree(this->NumCell(), 0);
            localFillOrderingNew2Old.resize(this->NumCell(), 0);
            localFillOrderingOld2New.resize(this->NumCell(), 0);
            boost::property_map<Graph, vertex_index_t>::type id = get(vertex_index, cell2cellG);
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling boost::minimum_degree_ordering" << std::endl;
            minimum_degree_ordering(
                cell2cellG,
                make_iterator_property_map(degree.data(), id, degree[0]),
                localFillOrderingOld2New.data(),
                localFillOrderingNew2Old.data(),
                make_iterator_property_map(supernodeSizes.data(), id, supernodeSizes[0]),
                0,
                id);
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): boost done" << std::endl;
        }
        else if (control.getOrderingCode() == 3) // RCM
        {
            using namespace boost;
            using Graph = adjacency_list<vecS, vecS, undirectedS, property<vertex_color_t, default_color_type, property<vertex_degree_t, int>>>;
            using Vertex = graph_traits<Graph>::vertex_descriptor;
            // typedef graph_traits<Graph>::vertices_size_type size_type;
            Graph cell2cellG(this->NumCell());
            std::vector<std::vector<index>> &cell2cellFaceV = cell2cellFaceVLocal;
            index bandWidthOld = 0;
            for (index iCell = 0; iCell < this->NumCell(); iCell++)
                for (auto iCOther : cell2cellFaceV[iCell])
                    add_edge(iCell, iCOther, cell2cellG), bandWidthOld = std::max(bandWidthOld, std::abs(iCell - iCOther));
            MPI::AllreduceOneIndex(bandWidthOld, MPI_MAX, this->mpi);
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling boost::cuthill_mckee_ordering, BW: " << bandWidthOld << std::endl;
            localFillOrderingNew2Old.resize(this->NumCell(), 0);
            localFillOrderingOld2New.resize(this->NumCell(), 0);
            Vertex startVert = vertex(0, cell2cellG);
            cuthill_mckee_ordering(cell2cellG, startVert, localFillOrderingNew2Old.rbegin(),
                                   get(vertex_color, cell2cellG), get(vertex_degree, cell2cellG));
            std::unordered_set<index> __checkOrder;
            for (auto v : localFillOrderingNew2Old)
                DNDS_assert(v < this->NumCell() && v >= 0), __checkOrder.insert(v);
            DNDS_assert_info(__checkOrder.size() == localFillOrderingNew2Old.size(), "The output of boost::cuthill_mckee_ordering is invalid!");

            for (index iCell = 0; iCell < this->NumCell(); iCell++)
                localFillOrderingOld2New[localFillOrderingNew2Old[iCell]] = iCell;
            for (auto v : localFillOrderingOld2New)
                DNDS_assert(v < this->NumCell() && v >= 0);
            index bandWidthNew = 0;
            for (index iCell = 0; iCell < this->NumCell(); iCell++)
                for (auto iCOther : cell2cellFaceV[iCell])
                    bandWidthNew = std::max(bandWidthNew, std::abs(localFillOrderingOld2New[iCell] - localFillOrderingOld2New[iCOther]));
            MPI::AllreduceOneIndex(bandWidthNew, MPI_MAX, this->mpi);

            if (mpi.rank == mRank)
                log()
                    << "UnstructuredMesh::ObtainLocalFactFillOrdering(): boost done, new BW: " << bandWidthNew << std::endl;
            // for (auto v : localFillOrderingOld2New)
            //     std::cout << v << ", ";
            // std::cout << std::endl;
        }
        else if (control.getOrderingCode() == 4) // CorrectRCM
        {
            index bandWidthOld = 0;
            // for (index iCell = 0; iCell < this->NumCell(); ++iCell)
            // {
            //     cell2cellFaceVLocal[iCell].resize(4);
            //     int x = iCell % 20, y = iCell / 20;
            //     cell2cellFaceVLocal[iCell][0] = mod(x - 1, 20) + y * 20;
            //     cell2cellFaceVLocal[iCell][1] = mod(x + 1, 20) + y * 20;
            //     cell2cellFaceVLocal[iCell][2] = x + mod(y - 1, 20) * 20;
            //     cell2cellFaceVLocal[iCell][3] = x + mod(y + 1, 20) * 20;
            // }

            std::vector<std::vector<index>> &cell2cellFaceV = cell2cellFaceVLocal;
            for (index iCell = 0; iCell < this->NumCell(); iCell++)
                for (auto iCOther : cell2cellFaceV[iCell])
                    bandWidthOld = std::max(bandWidthOld, std::abs(iCell - iCOther));
            MPI::AllreduceOneIndex(bandWidthOld, MPI_MAX, this->mpi);
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling CorrectRCM::CuthillMcKeeOrdering, BW: " << bandWidthOld << std::endl;
            localFillOrderingNew2Old.resize(this->NumCell(), 0);
            localFillOrderingOld2New.resize(this->NumCell(), 0);
            auto graphFunctor = [&](index i) -> t_IndexVec &
            { return cell2cellFaceV.at(i); }; // todo: need improvement in CorrectRCM: can pass a temporary functor and store
            auto graph = CorrectRCM::UndirectedGraphProxy(graphFunctor, this->NumCell());
            int ret = graph.CheckAdj();
            CorrectRCM::CuthillMcKeeOrdering(
                graph,
                [&](index i) -> index &
                {
                    return localFillOrderingOld2New.at(i);
                },
                0);
            for (auto &v : localFillOrderingOld2New)
                v = localFillOrderingOld2New.size() - 1 - v;

            std::unordered_set<index>
                __checkOrder;
            for (auto v : localFillOrderingOld2New)
                DNDS_assert(v < this->NumCell() && v >= 0), __checkOrder.insert(v);
            DNDS_assert_info(__checkOrder.size() == localFillOrderingOld2New.size(), "The output of CorrectRCM::CuthillMcKeeOrdering is invalid!");

            for (index iCell = 0; iCell < this->NumCell(); iCell++)
                localFillOrderingNew2Old[localFillOrderingOld2New[iCell]] = iCell;
            for (auto v : localFillOrderingNew2Old)
                DNDS_assert(v < this->NumCell() && v >= 0);
            index bandWidthNew = 0;
            for (index iCell = 0; iCell < this->NumCell(); iCell++)
                for (auto iCOther : cell2cellFaceV[iCell])
                    bandWidthNew = std::max(bandWidthNew, std::abs(localFillOrderingOld2New[iCell] - localFillOrderingOld2New[iCOther]));
            MPI::AllreduceOneIndex(bandWidthNew, MPI_MAX, this->mpi);

            if (mpi.rank == mRank)
                log()
                    << "UnstructuredMesh::ObtainLocalFactFillOrdering(): CorrectRCM::CuthillMcKeeOrdering done, new BW: " << bandWidthNew << std::endl;
            // for (auto v : localFillOrderingOld2New)
            //     std::cout << v << ", ";
            // std::cout << std::endl;
        }
        else
        {
            DNDS_assert_info(false, "No such ordering code");
        }
    }
}