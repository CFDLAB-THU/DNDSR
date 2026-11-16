#pragma once

#include "Cauchy/Boundary/BoundaryInputs.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include <algorithm>
#include <vector>

namespace CauchyTest
{
    /** @brief Minimal test routing over real canonical Geom/CFV publications.
     * @details Only owned-cell incident faces are active, preserving ghost-left
     * and self-periodic incidence semantics. No expanded sample/geometry table
     * is introduced to make a migrated numerical leaf test pass.
     */
    struct CompactManagedStencilFixture
    {
        std::shared_ptr<DNDS::Geom::UnstructuredMesh> mesh;
        DNDS::CFV::FiniteVolumePlan metrics;
        DNDS::Cauchy::FiniteVolumeStencilStorage stencil;
        DNDS::Runtime::Array<DNDS::Cauchy::BoundaryFaceRange> boundaryFaces;
        DNDS::index boundaryRows = 0;

        explicit CompactManagedStencilFixture(std::shared_ptr<DNDS::Geom::UnstructuredMesh> source, bool centers = false)
            : mesh(std::move(source)), metrics(DNDS::CFV::FiniteVolumePlan::ConstructMetrics(mesh, 5))
        {
            namespace R = DNDS::Runtime;
            using I = DNDS::index;
            const I faces = metrics.NumFaceProc();
            const auto geometry = metrics.AcquireTopologyRead();
            const auto topology = geometry.View();
            const auto metricRead = metrics.AcquireRead();
            const auto metric = metricRead.View();
            std::vector<I> active;
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                for (const I iFace : topology.CellFaces(iCell))
                    active.push_back(iFace);
            std::sort(active.begin(), active.end());
            active.erase(std::unique(active.begin(), active.end()), active.end());
            stencil.routing = R::Array<DNDS::Cauchy::FaceRouting>(R::RowLayout::Dense(faces, 1));
            stencil.columns = R::Array<I>(R::RowLayout::Dense(faces, 2));
            stencil.viscousDistance = R::Array<DNDS::real>(R::RowLayout::Dense(metrics.FaceQuadrature().RowCount(), 1));
            stencil.stencilFaces = R::Array<I>(R::RowLayout::Dense(active.size(), 1));
            stencil.jacobianOffsets = R::Array<I>(R::RowLayout::Dense(metrics.NumCell() + 1, 1));
            stencil.quadratureBoundaryValues = !centers;
            std::vector<DNDS::Cauchy::BoundaryFaceRange> ranges;
            {
                auto routing = stencil.routing.Write();
                auto indices = stencil.stencilFaces.Write(), columns = stencil.columns.Write(), offsets = stencil.jacobianOffsets.Write();
                for (I iFace = 0; iFace < faces; ++iFace)
                {
                    routing.View()(iFace, 0) = {};
                    columns.View()(iFace, 0) = columns.View()(iFace, 1) = DNDS::UnInitIndex;
                }
                for (I iAux = 0; iAux < I(active.size()); ++iAux)
                {
                    const I iFace = active[iAux];
                    auto &route = routing.View()(iFace, 0);
                    route.auxiliaryFace = iAux;
                    indices.View()(iAux, 0) = iFace;
                    if (topology.FaceCells(iFace)[1] == DNDS::UnInitIndex)
                    {
                        route.boundaryFace = I(ranges.size());
                        route.boundaryBegin = boundaryRows;
                        const I count = centers ? 0 : metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
                        ranges.push_back({iFace, boundaryRows, count, false});
                        boundaryRows += centers ? 1 : count;
                    }
                }
                offsets.View()(0, 0) = 0;
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                {
                    offsets.View()(iCell + 1, 0) = offsets.View()(iCell, 0) + topology.CellFaces(iCell).size();
                    std::vector<I> neighbors;
                    const auto cellFaces = topology.CellFaces(iCell);
                    for (DNDS::rowsize slot = 0; slot < cellFaces.size(); ++slot)
                    {
                        const I other = topology.CellFaceOther(iCell, cellFaces[slot], slot);
                        if (other != DNDS::UnInitIndex && other != iCell)
                            neighbors.push_back(other);
                    }
                    std::sort(neighbors.begin(), neighbors.end());
                    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
                    for (DNDS::rowsize slot = 0; slot < cellFaces.size(); ++slot)
                    {
                        const I iFace = cellFaces[slot], other = topology.CellFaceOther(iCell, iFace, slot);
                        if (other == DNDS::UnInitIndex || other == iCell)
                            continue;
                        const int side = topology.CellIsFaceBack(iCell, iFace, slot) ? 0 : 1;
                        columns.View()(iFace, side) = std::lower_bound(neighbors.begin(), neighbors.end(), other) - neighbors.begin();
                    }
                }
                stencil.incidenceCount = offsets.View()(metrics.NumCell(), 0);
            }
            boundaryFaces = R::Array<DNDS::Cauchy::BoundaryFaceRange>(R::RowLayout::Dense(ranges.size(), 1));
            {
                auto write = boundaryFaces.Write();
                for (I iRow = 0; iRow < I(ranges.size()); ++iRow)
                    write.View()(iRow, 0) = ranges[iRow];
            }
            stencil.Freeze(metrics);
        }
    };
}
