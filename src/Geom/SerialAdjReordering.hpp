#pragma once

#include "Geometric.hpp"

#include "CorrectRCM.hpp"

#include "Metis.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/minimum_degree_ordering.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

namespace DNDS::Geom
{
    inline std::pair<std::vector<index>, std::vector<index>> ReorderSerialAdj_Metis(const tLocalMatStruct &mat)
    {
        _METIS::idx_t nCell = _METIS::indexToIdx(size_t_to_signed<index>(mat.size()));
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
        const std::vector<std::vector<index>> &cell2cellFaceV = mat;
        std::vector<_METIS::idx_t> adjncy, xadj, perm, iPerm;
        xadj.resize(nCell + 1);
        xadj[0] = 0;
        for (_METIS::idx_t iC = 0; iC < nCell; iC++)
            xadj[iC + 1] = signedIntSafeAdd<_METIS::idx_t>(xadj[iC], size_t_to_signed<_METIS::idx_t>(cell2cellFaceV[iC].size())); //! check overflow!
        adjncy.resize(xadj.back());
        for (_METIS::idx_t iC = 0; iC < nCell; iC++)
            std::copy(cell2cellFaceV[iC].begin(), cell2cellFaceV[iC].end(), adjncy.begin() + xadj[iC]);
        perm.resize(nCell);
        iPerm.resize(nCell);

        int ret = _METIS::METIS_NodeND(&nCell, xadj.data(), adjncy.data(), NULL, options, perm.data(), iPerm.data());
        DNDS_assert_info(ret == _METIS::METIS_OK, fmt::format("Metis return not ok, [{}]", ret));

        std::vector<index> localFillOrderingNew2Old, localFillOrderingOld2New;

        localFillOrderingNew2Old.resize(nCell);
        localFillOrderingOld2New.resize(nCell);
        for (index i = 0; i < nCell; i++)
        {
            localFillOrderingNew2Old[i] = perm[i];
            localFillOrderingOld2New[i] = iPerm[i];
        }

        return {localFillOrderingNew2Old, localFillOrderingOld2New};
    }

    inline std::pair<std::vector<index>, std::vector<index>> ReorderSerialAdj_BoostMMD(const tLocalMatStruct &mat)
    {
        std::vector<index> localFillOrderingNew2Old, localFillOrderingOld2New;
        using namespace boost;
        using Graph = adjacency_list<vecS, vecS, directedS>;
        Graph cell2cellG(mat.size());
        const std::vector<std::vector<index>> &cell2cellFaceV = mat;
        for (index iCell = 0; iCell < size_t_to_signed<index>(mat.size()); iCell++)
            for (auto iCOther : cell2cellFaceV[iCell])
                add_edge(iCell, iCOther, cell2cellG);
        std::vector<index> supernodeSizes(mat.size(), 1), degree(mat.size(), 0);
        localFillOrderingNew2Old.resize(mat.size(), 0);
        localFillOrderingOld2New.resize(mat.size(), 0);
        boost::property_map<Graph, vertex_index_t>::type id = get(vertex_index, cell2cellG);
        minimum_degree_ordering(
            cell2cellG,
            make_iterator_property_map(degree.data(), id, degree[0]),
            localFillOrderingOld2New.data(),
            localFillOrderingNew2Old.data(),
            make_iterator_property_map(supernodeSizes.data(), id, supernodeSizes[0]),
            0,
            id);
        return {localFillOrderingNew2Old, localFillOrderingOld2New};
    }

    inline std::pair<std::vector<index>, std::vector<index>> ReorderSerialAdj_BoostRCM(const tLocalMatStruct &mat, index &bandWidthOld, index &bandWidthNew)
    {
        std::vector<index> localFillOrderingNew2Old, localFillOrderingOld2New;
        using namespace boost;
        using Graph = adjacency_list<vecS, vecS, undirectedS, property<vertex_color_t, default_color_type, property<vertex_degree_t, int>>>;
        using Vertex = graph_traits<Graph>::vertex_descriptor;
        // typedef graph_traits<Graph>::vertices_size_type size_type;
        Graph cell2cellG(mat.size());
        const std::vector<std::vector<index>> &cell2cellFaceV = mat;
        bandWidthOld = 0;
        for (index iCell = 0; iCell < size_t_to_signed<index>(mat.size()); iCell++)
            for (auto iCOther : cell2cellFaceV[iCell])
                add_edge(iCell, iCOther, cell2cellG), bandWidthOld = std::max(bandWidthOld, std::abs(iCell - iCOther));
        localFillOrderingNew2Old.resize(mat.size(), 0);
        localFillOrderingOld2New.resize(mat.size(), 0);
        Vertex startVert = vertex(0, cell2cellG);
        cuthill_mckee_ordering(cell2cellG, startVert, localFillOrderingNew2Old.rbegin(),
                               get(vertex_color, cell2cellG), get(vertex_degree, cell2cellG));
        std::unordered_set<index> __checkOrder;
        for (auto v : localFillOrderingNew2Old)
            DNDS_assert(v < mat.size() && v >= 0), __checkOrder.insert(v);
        DNDS_assert_info(__checkOrder.size() == localFillOrderingNew2Old.size(), "The output of boost::cuthill_mckee_ordering is invalid!");

        for (index iCell = 0; iCell < size_t_to_signed<index>(mat.size()); iCell++)
            localFillOrderingOld2New[localFillOrderingNew2Old[iCell]] = iCell;
        for (auto v : localFillOrderingOld2New)
            DNDS_assert(v < size_t_to_signed<index>(mat.size()) && v >= 0);
        bandWidthNew = 0;
        for (index iCell = 0; iCell < size_t_to_signed<index>(mat.size()); iCell++)
            for (auto iCOther : cell2cellFaceV[iCell])
                bandWidthNew = std::max(bandWidthNew, std::abs(localFillOrderingOld2New[iCell] - localFillOrderingOld2New[iCOther]));

        return {localFillOrderingNew2Old, localFillOrderingOld2New};
    }

    inline std::pair<std::vector<index>, std::vector<index>> ReorderSerialAdj_CorrectRCM(const tLocalMatStruct &mat, index &bandWidthOld, index &bandWidthNew)
    {
        std::vector<index> localFillOrderingNew2Old, localFillOrderingOld2New;
        bandWidthOld = 0;
        // for (index iCell = 0; iCell < this->NumCell(); ++iCell)
        // {
        //     cell2cellFaceVLocal[iCell].resize(4);
        //     int x = iCell % 20, y = iCell / 20;
        //     cell2cellFaceVLocal[iCell][0] = mod(x - 1, 20) + y * 20;
        //     cell2cellFaceVLocal[iCell][1] = mod(x + 1, 20) + y * 20;
        //     cell2cellFaceVLocal[iCell][2] = x + mod(y - 1, 20) * 20;
        //     cell2cellFaceVLocal[iCell][3] = x + mod(y + 1, 20) * 20;
        // }

        const std::vector<std::vector<index>> &cell2cellFaceV = mat;
        for (index iCell = 0; iCell < size_t_to_signed<index>(mat.size()); iCell++)
            for (auto iCOther : cell2cellFaceV[iCell])
                bandWidthOld = std::max(bandWidthOld, std::abs(iCell - iCOther));

        localFillOrderingNew2Old.resize(mat.size(), 0);
        localFillOrderingOld2New.resize(mat.size(), 0);
        auto graphFunctor = [&](index i) -> const t_IndexVec &
        { return cell2cellFaceV.at(i); }; // todo: need improvement in CorrectRCM: can pass a temporary functor and store
        auto graph = CorrectRCM::UndirectedGraphProxy(graphFunctor, size_t_to_signed<int64_t>(mat.size()));
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
            DNDS_assert(v < size_t_to_signed<index>(mat.size()) && v >= 0), __checkOrder.insert(v);
        DNDS_assert_info(__checkOrder.size() == localFillOrderingOld2New.size(), "The output of CorrectRCM::CuthillMcKeeOrdering is invalid!");

        for (index iCell = 0; iCell < size_t_to_signed<index>(mat.size()); iCell++)
            localFillOrderingNew2Old[localFillOrderingOld2New[iCell]] = iCell;
        for (auto v : localFillOrderingNew2Old)
            DNDS_assert(v < size_t_to_signed<index>(mat.size()) && v >= 0);
        bandWidthNew = 0;
        for (index iCell = 0; iCell < size_t_to_signed<index>(mat.size()); iCell++)
            for (auto iCOther : cell2cellFaceV[iCell])
                bandWidthNew = std::max(bandWidthNew, std::abs(localFillOrderingOld2New[iCell] - localFillOrderingOld2New[iCOther]));

        return {localFillOrderingNew2Old, localFillOrderingOld2New};
    }
}