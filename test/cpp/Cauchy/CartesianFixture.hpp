#pragma once

#include "../Geom/NativePeriodicFixtures.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace CauchyTest
{
    /// Independently numbered Cartesian torus. Endpoints use distinct periodic
    /// image bits but the same canonical node IDs; construction is distributed
    /// from owned primary rows, including empty ranks, never a global bitmap.
    inline std::shared_ptr<DNDS::Geom::UnstructuredMesh> PeriodicQuads(
        const DNDS::MPIInfo &mpi, DNDS::index width, DNDS::real length = 1, DNDS::real firstInteriorXShift = 0)
    {
        namespace G = DNDS::Geom;
        namespace R = DNDS::Runtime;
        using I = DNDS::index;
        DNDS_check_throw_info(width >= 1 && width <= 512, "test/opt-in benchmark torus width outside bounds");
        const auto range = DNDS::EvenSplitRange(mpi.rank, mpi.size, width * width);
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, range.second - range.first);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, range.second - range.first);
        std::vector<std::vector<I>> cones, boundaries;
        std::vector<std::vector<G::NodePeriodicBits>> bits, boundaryBits;
        std::vector<int> zones;
        const auto id = [width](I x, I y)
        { return (y % width) * width + x % width; };
        const auto frame = [width](I x, I y)
        { return G::NodePeriodicBits{uint8_t((x == width ? 1 : 0) | (y == width ? 2 : 0))}; };
        for (I global = range.first; global < range.second; ++global)
        {
            const I x = global % width, y = global / width;
            cones.push_back({id(x, y), id(x + 1, y), id(x + 1, y + 1), id(x, y + 1)});
            bits.push_back({frame(x, y), frame(x + 1, y), frame(x + 1, y + 1), frame(x, y + 1)});
            auto face = [&](I x0, I y0, I x1, I y1, int zone)
            {
                boundaries.push_back({id(x0, y0), id(x1, y1)});
                boundaryBits.push_back({frame(x0, y0), frame(x1, y1)});
                zones.push_back(zone);
            };
            if (x == 0)
                face(x, y + 1, x, y, G::BC_ID_PERIODIC_1);
            if (x + 1 == width)
                face(x + 1, y, x + 1, y + 1, G::BC_ID_PERIODIC_1_DONOR);
            if (y == 0)
                face(x, y, x + 1, y, G::BC_ID_PERIODIC_2);
            if (y + 1 == width)
                face(x + 1, y + 1, x, y + 1, G::BC_ID_PERIODIC_2_DONOR);
        }
        auto bnd = nodes.CreateSibling(G::EntityKind::Bnd, I(boundaries.size()));
        G::NativeMeshInput input{mpi, 2, cells, nodes, bnd, NativePeriodicFixtures::Rows(cones), NativePeriodicFixtures::Rows(boundaries),
                                 R::Array<double>(R::RowLayout::Dense(nodes.NumOwned(), 3)), R::Array<G::ElemInfo>(R::RowLayout::Dense(cells.NumOwned(), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(bnd.NumOwned(), 1))};
        input.periodic = true;
        input.periodicInfo.translation[1].map() = G::tPoint{length, 0, 0};
        input.periodicInfo.translation[2].map() = G::tPoint{0, length, 0};
        input.cellNodesPbi = NativePeriodicFixtures::Rows(bits);
        input.boundaryNodesPbi = NativePeriodicFixtures::Rows(boundaryBits);
        {
            auto coords = input.coordinates.Write();
            auto info = input.cellInfo.Write(), boundaryInfo = input.boundaryInfo.Write();
            for (I iNode = 0; iNode < nodes.NumOwned(); ++iNode)
            {
                const I global = range.first + iNode;
                coords.View()(iNode, 0) = length * double(global % width) / width;
                if (global % width == 1)
                    coords.View()(iNode, 0) += firstInteriorXShift;
                coords.View()(iNode, 1) = length * double(global / width) / width;
            }
            for (I iCell = 0; iCell < cells.NumOwned(); ++iCell)
                info.View()(iCell, 0) = {G::Elem::Quad4, G::INTERNAL_ZONE};
            for (I iBnd = 0; iBnd < bnd.NumOwned(); ++iBnd)
                boundaryInfo.View()(iBnd, 0) = {G::Elem::Line2, zones[iBnd]};
        }
        auto mesh = G::UnstructuredMesh::FromNative(mpi, G::ConstructNativeMesh(std::move(input)));
        mesh->PrepareNativeTopology();
        return mesh;
    }
}
