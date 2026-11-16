#include "CFV/Detail/ReconstructionData.hpp"

namespace DNDS::CFV::Detail
{
    void PrepareFaceBasis(ReconstructionData &data, const FiniteVolumeData &geometry)
    {
        if (!data.settings.cacheDiffBase)
            return;
        namespace R = Runtime;
        auto meshRead = geometry.mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto mesh = meshRead.HostView();
        auto cellRead = geometry.cell.Read();
        auto offsetRead = geometry.faceOffsets.Read();
        auto pointRead = geometry.faceQuadrature.Read();
        auto momentRead = data.moments.Read();
        const auto cells = cellRead.View<R::FixedRows<8>>();
        const auto offsets = offsetRead.View<R::FixedRows<1>>();
        const auto points = pointRead.View<R::FixedRows<7>>();
        const auto moments = momentRead.View<R::TableRows>();
        const Math::TaylorBasis polynomial(geometry.dim, data.settings.maxOrder);

        // Only owned-cell incidences are consumers. The allocated quadrature
        // numbering also contains ghost-only faces; retain their zero rows so
        // cache indexing stays identical across all sampling consumers.
        data.faceBasis = R::Array<real>(R::RowLayout::Dense(2 * points.RowCount(), data.cachedDerivatives * data.n));
        data.faceBasis.SetName("CFV/quadrature/face/basis", DNDS_ALLOCATION_SITE());
        auto cacheWrite = data.faceBasis.Write();
        const auto cache = cacheWrite.View<R::TableRows>();
        OMP::ForSetupRows(0, cache.RowCount(), [&](index iRow)
                          { std::fill(cache[iRow].begin(), cache[iRow].end(), 0.); });
        OMP::ForSetupRows(0, mesh.NumFaceProc(), [&](index iFace)
                          {
            const auto parents = mesh.FaceCells(iFace);
            // A face is active exactly when one canonical parent is owned.
            // This avoids concurrent scatter marks from neighboring cells.
            if (parents[0] >= mesh.NumCell() && (parents[1] == UnInitIndex || parents[1] >= mesh.NumCell()))
                return;
            DNDS_check_throw_info(parents[0] >= 0 && parents[0] < mesh.NumCellProc() &&
                                      (parents[1] == UnInitIndex || (parents[1] >= 0 && parents[1] < mesh.NumCellProc())),
                                  "face sampling basis lacks parent closure");
            Math::SmallMatrix values;
            for (index iPoint = offsets(iFace, 0); iPoint < offsets(iFace + 1, 0); ++iPoint)
            {
                const Geom::tPoint point{points(iPoint, 0), points(iPoint, 1), points(iPoint, 2)};
                for (int side = 0; side < 2; ++side)
                {
                    const index iCell = parents[side];
                    if (iCell == UnInitIndex)
                        continue;
                    // Store in each parent's native frame, before any VR-only
                    // derivative tensor pullback used by matrix assembly.
                    const auto parentPoint = side == 0 ? point : FacePointInCell(mesh, iFace, side, point);
                    polynomial.Reconstruction(values, parentPoint,
                                              {cells(iCell, 1), cells(iCell, 2), cells(iCell, 3)},
                                              cells(iCell, 7), moments[iCell], data.settings.cacheDiffBaseOrder);
                    for (int mode = 0; mode < data.n; ++mode)
                        for (int derivative = 0; derivative < data.cachedDerivatives; ++derivative)
                            cache(2 * iPoint + side, mode * data.cachedDerivatives + derivative) = values(derivative, mode);
                }
            } });
    }
}
