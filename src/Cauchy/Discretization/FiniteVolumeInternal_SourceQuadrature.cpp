#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Data::BuildSourceQuadrature()
    {
        namespace R = Runtime;
        if (!equation.HasRANS() &&
            (settings.equation.source != "chemical" || equation.Mixture().SpeciesNumber() == 1))
            return;
        auto result = std::make_shared<Detail::SourceQuadrature>();
        auto read = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto view = read.HostView();
        const auto geometry = metrics.AcquireRead();
        const auto metric = geometry.HostView();
        const int order = settings.numerics.RHSIntegrationOrder();
        std::vector<index> offsets(size_t(metrics.NumCell()) + 1, 0);
        std::vector<int> preparedTypes;
        for (index iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto element = view.GetCellElement(iCell);
            const Geom::Elem::Quadrature quadrature(element, order);
            offsets[iCell + 1] = offsets[iCell] + quadrature.GetNumPoints();
            // Populate lazy shape caches on the controller before parallel
            // filling. The type registry is small local setup metadata.
            if (std::find(preparedTypes.begin(), preparedTypes.end(), int(element.type)) == preparedTypes.end())
            {
                for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                    (void)Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ);
                preparedTypes.push_back(int(element.type));
            }
        }
        const index count = offsets.back();
        R::Array<index> cells(R::RowLayout::Dense(count, 1));
        R::Array<real> points(R::RowLayout::Dense(count, 3));
        result->weights = R::Array<real>(R::RowLayout::Dense(count, 1));
        {
            auto cellWrite = cells.Write();
            auto pointWrite = points.Write(), weightWrite = result->weights.Write();
            const auto c = cellWrite.View<R::FixedRows<1>>();
            const auto p = pointWrite.View<R::FixedRows<3>>();
            const auto w = weightWrite.View<R::FixedRows<1>>();
            OMP::ForSetupRows(0, metrics.NumCell(), [&](index iCell)
                              {
                Geom::tSmallCoords coords;
                view.GetCoordsOnCell(iCell, coords);
                const auto element = view.GetCellElement(iCell);
                const Geom::Elem::Quadrature quadrature(element, order);
                for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                {
                    const index iPoint = offsets[iCell] + iQ;
                    const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ);
                    const auto point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                    c(iPoint, 0) = iCell;
                    for (int d = 0; d < 3; ++d)
                        p(iPoint, d) = point[d];
                    w(iPoint, 0) = order == 1 ? metric.GetCellVol(iCell) :
                        Geom::Elem::CellJacobianDet(view.Dimension(), coords, shape) * quadrature.GetWeight(iQ);
                } });
        }
        if (settings.numerics.reconstruction == "variational")
            result->sampling = CFV::PolynomialSamplingPlan(*reconstruction, cells, points, equation.HasRANS()).WithDerivativeOrder(equation.HasRANS() ? 1 : 0).Named("Cauchy/quadrature/volume");
        if (equation.HasRANS() && reconstruction)
        {
            const index owned = metrics.NumCell();
            R::Array<index> centerCells(R::RowLayout::Dense(owned, 1));
            R::Array<real> centers(R::RowLayout::Dense(owned, 3));
            {
                auto cellWrite = centerCells.Write();
                auto pointWrite = centers.Write();
                const auto c = cellWrite.View<R::FixedRows<1>>();
                const auto p = pointWrite.View<R::FixedRows<3>>();
                OMP::ForSetupRows(0, owned, [&](index iCell)
                                  {
                    c(iCell, 0) = iCell;
                    const auto center = metric.GetCellCent(iCell);
                    for (int d = 0; d < 3; ++d)
                        p(iCell, d) = center[d]; });
            }
            result->jacobianSampling = CFV::PolynomialSamplingPlan(*reconstruction, centerCells, centers).WithDerivativeOrder(1).Named("Cauchy/quadrature/source-jacobian-center");
        }
        result->offsets = R::Array<index>(R::RowLayout::Dense(index(offsets.size()), 1));
        {
            auto write = result->offsets.Write();
            const auto rows = write.View<R::FixedRows<1>>();
            for (index iRow = 0; iRow < index(offsets.size()); ++iRow)
                rows(iRow, 0) = offsets[iRow];
        }
        result->offsets.SetName("Cauchy/quadrature/volume/offsets", DNDS_ALLOCATION_SITE());
        result->weights.SetName("Cauchy/quadrature/volume/weights", DNDS_ALLOCATION_SITE());
        quadrature.source = std::move(result);
    }
}
