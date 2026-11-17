#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Data::BuildPositivityQuadrature(bool limitedVR)
    {
        namespace R = Runtime;
        // Direct gradients and LVR's physical gradient target have their own
        // face-center check set. They never need modal face/volume queries.
        (void)limitedVR;
        if (!reconstruction || !settings.numerics.positivity)
            return;
        const auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto topology = meshRead.HostView();
        const auto metricRead = metrics.AcquireRead();
        const auto metric = metricRead.HostView();
        const int order = settings.numerics.RHSIntegrationOrder();
        const index owned = metrics.NumCell();
        Detail::ModalPositivityQuadrature prepared;
        std::vector<index> volumeOffsets(size_t(owned + 1), 0);
        std::vector<int> preparedTypes;
        for (index iCell = 0; iCell < owned; ++iCell)
        {
            const auto element = topology.GetCellElement(iCell);
            const Geom::Elem::Quadrature rule(element, order);
            volumeOffsets[iCell + 1] = volumeOffsets[iCell] + rule.GetNumPoints();
            index points = rule.GetNumPoints();
            for (const auto iFace : topology.CellFaces(iCell))
                points += metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
            prepared.rows += points;
            prepared.maximumCellPoints = std::max(prepared.maximumCellPoints, points);
            if (std::find(preparedTypes.begin(), preparedTypes.end(), int(element.type)) == preparedTypes.end())
            {
                for (int iQ = 0; iQ < rule.GetNumPoints(); ++iQ)
                    (void)Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ);
                preparedTypes.push_back(int(element.type));
            }
        }
        maximumPositivityPoints = prepared.maximumCellPoints;
        if (quadrature.source && quadrature.source->sampling)
        {
            // Source and PP enumerate exactly the same cell-major volume rule.
            // Keep one physical-point/basis owner, with no coordinate matching
            // and no new query IDs or scatter maps. Placement memo preserves
            // these aliases on the execution backend as well.
            const auto &source = *quadrature.source;
            DNDS_check_throw_info(source.sampling->QueryNumber() == volumeOffsets.back(),
                                  "source and modal PP volume rules differ");
            prepared.volumeOffsets = source.offsets;
            prepared.cachedDerivatives = source.sampling->CachedDerivativeNumber();
            if (prepared.cachedDerivatives)
                prepared.volumeBasis = source.sampling->BasisStorage();
            else
                prepared.volumePoints = source.sampling->PointStorage();
        }
        else
        {
            const index points = volumeOffsets.back();
            const int modes = reconstruction->NRecDOF();
            prepared.cachedDerivatives = settings.numerics.cfv.cacheDiffBase ? 1 : 0;
            prepared.volumeOffsets = R::Array<index>(R::RowLayout::Dense(owned + 1, 1));
            if (prepared.cachedDerivatives)
                prepared.volumeBasis = R::Array<real>(R::RowLayout::Dense(points, modes));
            else
                prepared.volumePoints = R::Array<real>(R::RowLayout::Dense(points, 3));
            const auto momentsRead = reconstruction->SamplingMoments().Read();
            const auto moments = momentsRead.View<R::TableRows>();
            const CFV::Math::TaylorBasis polynomial(metrics.Dimension(), settings.numerics.cfv.maxOrder);
            {
                auto offsetWrite = prepared.volumeOffsets.Write();
                const auto offsets = offsetWrite.View<R::FixedRows<1>>();
                for (index iRow = 0; iRow <= owned; ++iRow)
                    offsets(iRow, 0) = volumeOffsets[iRow];
                auto pointWrite = prepared.volumePoints.Write(), basisWrite = prepared.volumeBasis.Write();
                const auto positions = pointWrite.View<R::FixedRows<3>>();
                const auto basis = basisWrite.View<R::TableRows>();
                OMP::ForSetupRows(0, owned, [&](index iCell)
                                  {
                    Geom::tSmallCoords coords;
                    topology.GetCoordsOnCell(iCell, coords);
                    const auto element = topology.GetCellElement(iCell);
                    const Geom::Elem::Quadrature rule(element, order);
                    CFV::Math::SmallMatrix value;
                    for (int iQ = 0; iQ < rule.GetNumPoints(); ++iQ)
                    {
                        const index iPoint = volumeOffsets[iCell] + iQ;
                        const auto point = Geom::Elem::PPhysicsCoordD01Nj(coords,
                            Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ));
                        if (prepared.cachedDerivatives)
                        {
                            polynomial.Reconstruction(value, point, metric.GetCellCent(iCell),
                                                      metric.cells(iCell, 7), moments[iCell], 0);
                            for (int mode = 0; mode < modes; ++mode)
                                basis(iPoint, mode) = value(0, mode);
                        }
                        else
                            for (int d = 0; d < 3; ++d)
                                positions(iPoint, d) = point[d];
                    } });
            }
            prepared.volumeOffsets.SetName("Cauchy/quadrature/volume/offsets", DNDS_ALLOCATION_SITE());
            prepared.volumePoints.SetName("Cauchy/quadrature/volume/points", DNDS_ALLOCATION_SITE());
            prepared.volumeBasis.SetName("Cauchy/quadrature/volume/basis", DNDS_ALLOCATION_SITE());
        }
        quadrature.positivity = std::move(prepared);
    }
}
