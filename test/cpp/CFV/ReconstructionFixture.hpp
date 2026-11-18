#pragma once
#include "../Geom/NativePeriodicFixtures.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace CFVTest
{
    /// Real connected Hex8 mesh; independent Cartesian node IDs are the oracle.
    /// Only local rows and local boundary faces are constructed on each rank.
    inline std::shared_ptr<DNDS::Geom::UnstructuredMesh> CartesianHexes(DNDS::MPIInfo mpi, DNDS::index width, bool separateLowerWall = false)
    {
        namespace G = DNDS::Geom;
        namespace R = DNDS::Runtime;
        using I = DNDS::index;
        DNDS_check_throw_info(width > 0 && width <= 256, "fixture width must be in 1..256");
        const I plane = width + 1, total = width * width * width;
        const auto cellRange = DNDS::EvenSplitRange(mpi.rank, mpi.size, total);
        const auto nodeRange = DNDS::EvenSplitRange(mpi.rank, mpi.size, plane * plane * plane);
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, nodeRange.second - nodeRange.first);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, cellRange.second - cellRange.first);
        const auto node = [&](I x, I y, I z)
        { return (z * plane + y) * plane + x; };
        std::vector<std::vector<I>> cones, faces;
        G::Elem::Element hex{G::Elem::Hex8};
        for (I global = cellRange.first; global < cellRange.second; ++global)
        {
            const I x = global % width, y = (global / width) % width, z = global / (width * width);
            cones.push_back({node(x, y, z), node(x + 1, y, z), node(x + 1, y + 1, z), node(x, y + 1, z),
                             node(x, y, z + 1), node(x + 1, y, z + 1), node(x + 1, y + 1, z + 1), node(x, y + 1, z + 1)});
            for (int side = 0; side < 6; ++side)
            {
                std::vector<I> face(4);
                hex.ExtractFaceNodes(side, cones.back(), face);
                bool boundary = false;
                for (int d = 0; d < 3; ++d)
                    for (I limit : {I(0), width})
                        boundary |= std::all_of(face.begin(), face.end(), [&](I id)
                                                { return (d == 0 ? id % plane : d == 1 ? (id / plane) % plane
                                                                                       : id / (plane * plane)) == limit; });
                if (boundary)
                    faces.push_back(std::move(face));
            }
        }
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, I(faces.size()));
        G::NativeMeshInput input{mpi, 3, cells, nodes, boundaries, NativePeriodicFixtures::Rows(cones), NativePeriodicFixtures::Rows(faces),
                                 R::Array<double>(R::RowLayout::Dense(nodes.NumOwned(), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(cells.NumOwned(), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(boundaries.NumOwned(), 1))};
        {
            auto coords = input.coordinates.Write();
            for (I iNode = 0; iNode < nodes.NumOwned(); ++iNode)
            {
                const I global = nodeRange.first + iNode;
                coords.View()(iNode, 0) = double(global % plane) / width;
                coords.View()(iNode, 1) = double((global / plane) % plane) / width;
                coords.View()(iNode, 2) = double(global / (plane * plane)) / width;
            }
            auto cellInfo = input.cellInfo.Write(), boundaryInfo = input.boundaryInfo.Write();
            for (I iCell = 0; iCell < cells.NumOwned(); ++iCell)
                cellInfo.View()(iCell, 0) = {G::Elem::Hex8, G::INTERNAL_ZONE};
            for (I iBnd = 0; iBnd < boundaries.NumOwned(); ++iBnd)
            {
                const auto &face = faces[iBnd];
                const bool lowerWall = separateLowerWall && std::all_of(face.begin(), face.end(), [&](I id)
                                                                        { return (id / plane) % plane == 0; });
                boundaryInfo.View()(iBnd, 0) = {G::Elem::Quad4, lowerWall ? 8 : 7};
            }
        }
        auto mesh = G::UnstructuredMesh::FromNative(mpi, G::ConstructNativeMesh(std::move(input)));
        mesh->PrepareNativeTopology();
        return mesh;
    }
}
