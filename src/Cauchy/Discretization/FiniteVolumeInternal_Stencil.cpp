#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Data::BuildStencil()
    {
        namespace R = Runtime;
        const auto topologyRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto topology = topologyRead.HostView();
        const auto metricRead = metrics.AcquireRead();
        const auto metric = metricRead.HostView();
        const index owned = metrics.NumCell(), faces = metrics.NumFaceProc();
        std::vector<index> activeFaces, coreFaces, dependentFaces, rhsCore, rhsDependent,
            independentFaces, interfaceFaces, independentCells, interfaceCells;
        std::vector<index> offsets(size_t(owned) + 1), cellAuxiliary;
        std::vector<BoundaryFaceRange> boundaryRanges;
        std::set<int> localZones;

        metrics.Communication()->LocalStage([&]
                                            {
            for (const auto &[zone, condition] : settings.boundaries)
            {
                forceZones.push_back(std::stoi(zone));
                hasInterfaces = hasInterfaces || condition.kind == "interface";
            }
            std::sort(forceZones.begin(), forceZones.end());
            forceZones.erase(std::unique(forceZones.begin(), forceZones.end()), forceZones.end());
            for (index i = 0; i < index(forceZones.size()); ++i)
            {
                forceSlots.emplace(forceZones[i], i);
                boundaryConditions.push_back(boundary.At(forceZones[i]));
            }
            const index interiorCondition = index(boundaryConditions.size());
            boundaryConditions.emplace_back();
            for (index iCell = 0; iCell < owned; ++iCell)
            {
                const auto row = topology.CellFaces(iCell);
                maximumGradientPoints = std::max(maximumGradientPoints, index(row.size()));
                offsets[iCell + 1] = offsets[iCell] + index(row.size());
            }
            // Prefixes and boundary numbering stay in canonical cell order.
            // Every worker owns one preallocated incidence range; no concurrent
            // append, Array lease acquisition or MPI occurs in the fill.
            activeFaces.resize(size_t(offsets.back()));
            OMP::ForRows(0, owned, [&](index iCell)
            {
                const auto row = topology.CellFaces(iCell);
                std::copy(row.begin(), row.end(), activeFaces.begin() + offsets[iCell]);
            });
            std::sort(activeFaces.begin(), activeFaces.end());
            activeFaces.erase(std::unique(activeFaces.begin(), activeFaces.end()), activeFaces.end());
            stencil.routing = R::Array<FaceRouting>(R::RowLayout::Dense(faces, 1));
            stencil.columns = R::Array<index>(R::RowLayout::Dense(faces, 2));
            stencil.viscousDistance = R::Array<real>(R::RowLayout::Dense(metric.faceOffsets(faces, 0), 1));
            stencil.quadratureBoundaryValues = settings.numerics.reconstruction == "variational";
            stencil.routing.SetName("Cauchy/stencil/faceRouting", DNDS_ALLOCATION_SITE());
            stencil.columns.SetName("Cauchy/stencil/jacobianColumns", DNDS_ALLOCATION_SITE());
            stencil.viscousDistance.SetName("Cauchy/stencil/viscousDistance", DNDS_ALLOCATION_SITE());
            auto routingWrite = stencil.routing.Write();
            auto columnWrite = stencil.columns.Write();
            auto distanceWrite = stencil.viscousDistance.Write();
            const auto routing = routingWrite.View<R::FixedRows<1>>();
            const auto slots = columnWrite.View<R::FixedRows<2>>();
            const auto distance = distanceWrite.View<R::FixedRows<1>>();
            OMP::ForRows(0, faces, [&](index iFace)
            {
                routing(iFace, 0) = FaceRouting{};
                routing(iFace, 0).condition = interiorCondition;
                slots(iFace, 0) = slots(iFace, 1) = UnInitIndex;
            });
            OMP::ForRows(0, distance.RowCount(), [&](index iPoint)
            { distance(iPoint, 0) = 0; });
            rhsPeriodic = topology.IsPeriodic();
            for (index iAux = 0; iAux < index(activeFaces.size()); ++iAux)
            {
                const index iFace = activeFaces[iAux];
                const int points = int(metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace));
                if (iAux == 0)
                    rhsUniformFacePoints = points;
                else if (rhsUniformFacePoints != points)
                    rhsUniformFacePoints = 0;
                const auto parents = topology.FaceCells(iFace);
                DNDS_check_throw_info(parents[0] >= 0 && parents[0] < metrics.NumCellProc() &&
                                      (parents[1] == UnInitIndex || (parents[1] >= 0 && parents[1] < metrics.NumCellProc())),
                                      "canonical face parents require complete local input closure");
                auto &route = routing(iFace, 0);
                route.auxiliaryFace = iAux;
                const bool core = parents[0] < owned && (parents[1] == UnInitIndex || parents[1] < owned);
                (core ? coreFaces : dependentFaces).push_back(iAux);
                if (parents[1] == UnInitIndex)
                {
                    const int zone = topology.GetFaceZone(iFace);
                    const auto &condition = boundary.At(zone);
                    localZones.insert(zone);
                    route.condition = forceSlots.at(zone);
                    if (condition.kind == BoundaryKind::SlipWall || condition.kind == BoundaryKind::NoSlipWall)
                        route.forceSlot = forceSlots.at(zone);
                }
                hasRotatedFrames = hasRotatedFrames || metric.periodic.Rotates(iFace);
            }
            OMP::ForSetupRows(0, index(activeFaces.size()), [&](index iAux)
            {
                const index iFace = activeFaces[iAux];
                const auto parents = topology.FaceCells(iFace);
                real area = 0;
                Geom::tPoint center = Geom::tPoint::Zero();
                for (index q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
                {
                    area += metric.GetFaceWeight(q);
                    center += metric.GetFaceWeight(q) * metric.GetFacePoint(q);
                }
                DNDS_check_throw_info(std::isfinite(area) && area >= 0, "invalid viscous face area");
                center = area > 0 ? Geom::tPoint(center / area) : metric.GetFaceCent(iFace);
                const auto leftBary = metric.GetCellBary(parents[0]);
                const Geom::tPoint delta = parents[1] == UnInitIndex ? Geom::tPoint(2 * (center - leftBary)) :
                    Geom::tPoint(metric.periodic.OtherPoint(iFace, 0, metric.GetCellBary(parents[1])) - leftBary);
                // Preserve the quadrature-normal centroid projection and its
                // original two-cell volume/area clamp. Zero weighted-area axis
                // faces contribute no flux; their valid positive length avoids
                // an otherwise meaningless divide by zero.
                const real scale = area > 0 ? 2 * std::min(metric.GetCellVol(parents[0]),
                    metric.GetCellVol(parents[1] == UnInitIndex ? parents[0] : parents[1])) / area : metric.cells(parents[0], 7);
                for (index q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
                {
                    distance(q, 0) = std::max(.25 * scale, std::min(2 * scale, std::abs(delta.dot(metric.GetFaceNorm(q)))));
                    DNDS_check_throw_info(std::isfinite(distance(q, 0)) && distance(q, 0) > 0, "invalid viscous distance");
                }
            });
            std::vector<std::vector<index>> columns(static_cast<size_t>(owned));
            OMP::ForSetupRows(0, owned, [&](index iCell)
            {
                auto &row = columns[iCell];
                const auto cellFaces = topology.CellFaces(iCell);
                for (rowsize slot = 0; slot < rowsize(cellFaces.size()); ++slot)
                {
                    const auto other = topology.CellFaceOther(iCell, cellFaces[slot], slot);
                    if (other != UnInitIndex && other != iCell)
                        row.push_back(other);
                }
                std::sort(row.begin(), row.end());
                row.erase(std::unique(row.begin(), row.end()), row.end());
                for (rowsize slot = 0; slot < rowsize(cellFaces.size()); ++slot)
                {
                    const auto iFace = cellFaces[slot];
                    const int side = topology.CellIsFaceBack(iCell, iFace, slot) ? 0 : 1;
                    const auto other = topology.CellFaceOther(iCell, iFace, slot);
                    if (other != UnInitIndex && other != iCell)
                        slots(iFace, side) = index(std::lower_bound(row.begin(), row.end(), other) - row.begin());
                }
            });
            jacobianColumns = std::make_shared<LinearPatternStorage>(columns,
                std::vector<index>{}, std::vector<index>{}, std::vector<index>{}, std::vector<index>{});
            boundaryLayout = std::make_shared<Detail::BoundaryPointLayout>();
            boundaryLayout->mesh = mesh;
            boundaryLayout->stamp = mesh->versions.Stamp(R::Change::Topology | R::Change::Geometry | R::Change::Distribution | R::Change::Layout);
            boundaryLayout->integrationOrder = settings.numerics.RHSIntegrationOrder();
            boundaryLayout->meanIntegrationOrder = settings.numerics.cfv.intOrder;
            boundaryLayout->axisymmetric = metrics.Axisymmetric();
            cellAuxiliary.resize(size_t(offsets.back()));
            OMP::ForSetupRows(0, owned, [&](index iCell)
            {
                const auto cellFaces = topology.CellFaces(iCell);
                auto begin = cellAuxiliary.begin() + offsets[iCell];
                for (rowsize slot = 0; slot < rowsize(cellFaces.size()); ++slot)
                    begin[slot] = routing(cellFaces[slot], 0).auxiliaryFace;
                std::sort(begin, cellAuxiliary.begin() + offsets[iCell + 1]);
            });
            // Stable boundary IDs are a controller publication, independent
            // of worker completion order and identical at OMP=1.
            for (index iCell = 0; iCell < owned; ++iCell)
            {
                const auto cellFaces = topology.CellFaces(iCell);
                for (rowsize slot = 0; slot < rowsize(cellFaces.size()); ++slot)
                {
                    const auto iFace = cellFaces[slot];
                    auto &route = routing(iFace, 0);
                    if (topology.CellFaceOther(iCell, iFace, slot) != UnInitIndex)
                        continue;
                    const auto &condition = boundaryConditions[route.condition];
                    const bool interface = condition.kind == BoundaryKind::Interface;
                    const index count = stencil.quadratureBoundaryValues ? metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace) : 0;
                    route.boundaryFace = index(boundaryRanges.size());
                    route.boundaryBegin = stencil.quadratureBoundaryValues ? boundaryLayout->quadratureRows : UnInitIndex;
                    boundaryRanges.push_back({iFace, boundaryLayout->quadratureRows, count, interface});
                    const index begin = metric.FaceQuadratureBegin(iFace), end = metric.FaceQuadratureEnd(iFace);
                    const auto normal = metric.GetFaceMeanNorm(iFace);
                    real area = 0;
                    for (index iPoint = begin; iPoint < end; ++iPoint)
                        area += metric.GetFaceWeight(iPoint);
                    boundaryOutputRows.push_back({iFace, iCell, route.auxiliaryFace, count ? route.boundaryBegin : route.boundaryFace,
                                                  begin, end - begin, {normal[0], normal[1], normal[2]}, area,
                                                  topology.GetFaceZone(iFace), !stencil.quadratureBoundaryValues});
                    boundaryLayout->quadratureRows += count;
                    hasLocalInterfaceRows = hasLocalInterfaceRows || interface;
                }
            }
            DNDS_check_throw_info(index(boundaryRanges.size()) == gradient.NumBoundaryQuadrature(),
                                  "gradient boundary-center row count differs from canonical boundary faces");
            for (index iBoundary = 0; iBoundary < index(boundaryRanges.size()); ++iBoundary)
                DNDS_check_throw_info(gradient.GetBoundaryQuadrature(iBoundary).face == boundaryRanges[iBoundary].face,
                                      "gradient boundary centers have a different order");
            boundaryFaces = R::Array<BoundaryFaceRange>(R::RowLayout::Dense(index(boundaryRanges.size()), 1));
            {
                auto write = boundaryFaces.Write();
                const auto values = write.View<R::FixedRows<1>>();
                OMP::ForRows(0, index(boundaryRanges.size()), [&](index iBoundary)
                { values(iBoundary, 0) = boundaryRanges[iBoundary]; });
            }
            boundaryFaces.SetName("Cauchy/boundary/faceRanges", DNDS_ALLOCATION_SITE());
            boundaryLayout->faces = boundaryFaces;
            const auto isInterface = [&](index iAux)
            {
                const auto &route = routing(activeFaces[iAux], 0);
                return route.boundaryFace != UnInitIndex && boundaryRanges[route.boundaryFace].interface;
            };
            for (auto iAux : coreFaces)
                (isInterface(iAux) ? interfaceFaces : independentFaces).push_back(iAux);
            for (auto iCell : partition.Core())
            {
                bool interface = false;
                for (index entry = offsets[iCell]; entry < offsets[iCell + 1]; ++entry)
                {
                    const auto iAux = cellAuxiliary[entry];
                    const auto parents = topology.FaceCells(activeFaces[iAux]);
                    DNDS_check_throw_info(parents[0] < owned && (parents[1] == UnInitIndex || parents[1] < owned),
                                          "core auxiliary work depends on a ghost input");
                    interface = interface || isInterface(iAux);
                }
                (interface ? interfaceCells : independentCells).push_back(iCell);
            }
            const auto collectFaces = [&](const auto &cells, auto &output)
            {
                for (auto iCell : cells)
                    output.insert(output.end(), cellAuxiliary.begin() + offsets[iCell], cellAuxiliary.begin() + offsets[iCell + 1]);
                std::sort(output.begin(), output.end());
                output.erase(std::unique(output.begin(), output.end()), output.end());
            };
            collectFaces(partition.Core(), rhsCore);
            collectFaces(partition.HaloDependent(), rhsDependent);
            stencil.rhsRegions = {R::RowSelectionShape::FromIndices(partition.Core(), owned),
                                 R::RowSelectionShape::FromIndices(partition.HaloDependent(), owned),
                                 R::RowSelectionShape::FromIndices(rhsCore, index(activeFaces.size())),
                                 R::RowSelectionShape::FromIndices(rhsDependent, index(activeFaces.size()))};
            const auto setAuxiliaryRegion = [&](AuxiliaryRegion region, const auto &faces, const auto &cells)
            {
                stencil.auxiliaryRegions[size_t(region)] = {
                    R::RowSelectionShape::FromIndices(faces, index(activeFaces.size())),
                    R::RowSelectionShape::FromIndices(cells, owned)};
            };
            setAuxiliaryRegion(AuxiliaryRegion::Independent, independentFaces, independentCells);
            setAuxiliaryRegion(AuxiliaryRegion::Core, coreFaces, partition.Core());
            setAuxiliaryRegion(AuxiliaryRegion::InterfaceCore, interfaceFaces, interfaceCells);
            setAuxiliaryRegion(AuxiliaryRegion::Dependent, dependentFaces, partition.HaloDependent());
            const auto publish = [](const auto &values, auto &array, const char *name)
            {
                array = R::Array<index>(R::RowLayout::Dense(index(values.size()), 1));
                auto write = array.Write();
                const auto output = write.View();
                OMP::ForRows(0, index(values.size()), [&](index iRow)
                { output(iRow, 0) = values[iRow]; });
                array.SetName(std::string("Cauchy/stencil/") + name, DNDS_ALLOCATION_SITE());
            };
            publish(offsets, stencil.jacobianOffsets, "incidenceOffsets");
            stencil.incidenceCount = offsets.back();
            stencil.auxiliaryCellOffsets = stencil.jacobianOffsets;
            publish(activeFaces, stencil.stencilFaces, "activeFaces");
            publish(coreFaces, stencil.auxiliaryCoreFaces, "auxiliaryCoreFaces");
            publish(dependentFaces, stencil.auxiliaryDependentFaces, "auxiliaryDependentFaces");
            publish(rhsCore, stencil.rhsCoreFaces, "rhsCoreFaces");
            publish(rhsDependent, stencil.rhsDependentFaces, "rhsDependentFaces");
            publish(cellAuxiliary, stencil.auxiliaryCellFaces, "auxiliaryCellFaces");
            publish(partition.Core(), stencil.coreCells, "coreCells");
            publish(partition.HaloDependent(), stencil.dependentCells, "dependentCells");
            publish(independentFaces, stencil.auxiliaryInterfaceIndependentFaces, "auxiliaryInterfaceIndependentFaces");
            publish(interfaceFaces, stencil.auxiliaryInterfaceFaces, "auxiliaryInterfaceFaces");
            publish(independentCells, stencil.interfaceIndependentCells, "interfaceIndependentCells");
            publish(interfaceCells, stencil.interfaceCoreCells, "interfaceCoreCells");
            boundaryZones.assign(localZones.begin(), localZones.end()); });
        std::vector<int> zones(localZones.begin(), localZones.end()), counts(mesh->mpi.size), starts(counts.size() + 1, 0);
        const int localCount = int(zones.size());
        MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, metrics.Communicator());
        for (index i = 0; i < index(counts.size()); ++i)
            starts[i + 1] = starts[i] + counts[i];
        std::vector<int> global(size_t(starts.back()));
        MPI_Allgatherv(zones.data(), localCount, MPI_INT, global.data(), counts.data(), starts.data(), MPI_INT, metrics.Communicator());
        boundary.ValidateUsed(std::set<int>(global.begin(), global.end()));
        hasPressureBoundary = std::any_of(global.begin(), global.end(), [&](int zone)
                                          {
            const auto kind = boundary.At(zone).kind;
            return kind == BoundaryKind::PressureOutlet || kind == BoundaryKind::Farfield; });
    }
}
