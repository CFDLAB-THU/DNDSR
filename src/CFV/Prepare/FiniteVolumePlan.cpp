#include "DNDS/Algebra/DofHalo.hpp"
#include "CFV/Detail/ReconstructionData.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Math/HardEigen.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    using namespace Geom;
    using namespace Geom::Elem;
    const Algebra::DofSpace &FiniteVolumePlan::Space() const { return _data->dofs; }
    std::shared_ptr<R::CommunicationContext> FiniteVolumePlan::Communication() const { return _data->communication; }

    void ReconstructionSettings::Validate() const
    {
        functional.Validate();
        storage.Validate();
        cudaSweep.Validate();
        // Quadrature and polynomial degrees are independent numerical choices.
        // No degree+1 rule guarantees exact integration of the full functional.
        // Keep the supported-rule range; SVD handles the chosen discrete system.
        DNDS_check_throw_info(maxOrder >= 1 && maxOrder <= 3 && intOrder >= 1 && intOrder <= INT_ORDER_MAX &&
                                  cacheDiffBaseOrder >= 0 && cacheDiffBaseOrder <= 3 &&
                                  std::isfinite(jacobiRelax) && jacobiRelax > 0 && jacobiRelax < 2 &&
                                  std::isfinite(bcWeight) && bcWeight >= 0 &&
                                  std::isfinite(svdTolerance) && svdTolerance >= 0 && svdTolerance <= 1 &&
                                  std::isfinite(svdAbsoluteTolerance) && svdAbsoluteTolerance >= 0 &&
                                  std::isfinite(WBAP_nStd) && WBAP_nStd > 0,
                              "invalid finite-volume reconstruction settings");
        for (const auto weight : boundaryDerivativeWeights)
            DNDS_check_throw_info(std::isfinite(weight) && weight >= 0, "invalid boundary derivative weight");
        for (const auto &[zone, weights] : boundaryZoneWeights)
            for (const auto weight : weights)
                DNDS_check_throw_info(std::isfinite(weight) && weight >= 0, "invalid zone boundary derivative weight");
    }
    FiniteVolumePlan FiniteVolumePlan::ConstructMetrics(ssp<UnstructuredMesh> mesh, FiniteVolumeMetricOptions options)
    {
        DNDS_check_throw_info(mesh, "finite-volume metrics require a mesh");
        const auto mpi = mesh->getMPI();
        const int intOrder = options.integrationOrder;
        const int faceIntOrder = options.faceIntegrationOrder ? options.faceIntegrationOrder : intOrder;
        const bool axisymmetric = options.axisymmetric;
        const auto meshCommunication = mesh->PreparationCommunication();
        auto communication = meshCommunication
                                 ? meshCommunication->WithPreflight(options.collectivePreflight)
                                 : R::CommunicationContext::Duplicate(mpi.comm, options.collectivePreflight);
        std::shared_ptr<Detail::FiniteVolumeData> data;
        // All local allocation/geometry failures agree before the halo collective.
        communication->LocalStage([&]
                                  {
            DNDS_check_throw_info(intOrder >= 2 && intOrder <= INT_ORDER_MAX, "invalid FV integration order");
            DNDS_check_throw_info(faceIntOrder == 1 || faceIntOrder == intOrder, "FV face integration must be midpoint or the metric order");
            auto read = mesh->AcquireRead(MeshReadRequirements::CellFlux());
            const auto view = read.HostView();
            DNDS_check_throw_info(view.Dimension() == 2 || view.Dimension() == 3, "FV requires 2D/3D geometry");
            DNDS_check_throw_info(!axisymmetric || view.Dimension() == 2, "axisymmetric metrics require planar 2D geometry, radius=y");
            if (axisymmetric && view.IsPeriodic())
                for (index iFace = 0; iFace < view.NumFaceProc(); ++iFace)
                {
                    const int zone = view.GetFaceZone(iFace);
                    if (!FaceIDIsPeriodic(zone)) continue;
                    const int frame = FaceIDIsPeriodicMain(zone) ? -zone : -zone - 3;
                    DNDS_check_throw_info(view.Periodic().rotation[frame].map().row(1).isApprox(tPoint::UnitY().transpose(), 1e-12) &&
                                              std::abs(view.Periodic().translation[frame].map()[1]) < 1e-12,
                                          "axisymmetric periodic transforms must preserve radius=y");
                }
            data = std::make_shared<Detail::FiniteVolumeData>(mesh, intOrder, axisymmetric, communication);
            data->cell = R::Array<real>(R::RowLayout::Dense(view.NumCellProc(), 8));
            data->cell.SetName("CFV/metrics/cell", DNDS_ALLOCATION_SITE());
            data->radialMoment = R::Array<real>(R::RowLayout::Dense(view.NumCellProc(), 3));
            auto radialWrite = data->radialMoment.Write();
            const auto radial = radialWrite.View<R::FixedRows<3>>();
            data->faceOffsets = R::Array<index>(R::RowLayout::Dense(view.NumFaceProc() + 1, 1));
            auto offsets = data->faceOffsets.Write();
            auto offset = offsets.View<R::FixedRows<1>>();
            index count = 0;
            for (index iFace = 0; iFace < view.NumFaceProc(); ++iFace)
            {
                offset(iFace, 0) = count;
                count += Quadrature(view.GetFaceElement(iFace), faceIntOrder).GetNumPoints();
            }
            offset(view.NumFaceProc(), 0) = count;
            data->faceQuadrature = R::Array<real>(R::RowLayout::Dense(count, 7));
            auto cells = data->cell.Write();
            const auto cell = cells.View<R::FixedRows<8>>();
            // Immutable process-wide shape tables are fully constructed before
            // this setup. Each worker owns coordinates and integration scratch.
            OMP::ForSetupRows(0, view.NumCellProc(), [&](index iCell)
            {
                tSmallCoords coords;
                view.GetCoordsOnCell(iCell, coords);
                const auto element = view.GetCellElement(iCell);
                Quadrature quadrature(element, intOrder), exact(element, INT_ORDER_MAX);
                real volume = 0.;
                real plainVolume = 0;
                tPoint bary = tPoint::Zero(), plainMoment = tPoint::Zero();
                for (int iG = 0; iG < exact.GetNumPoints(); ++iG)
                {
                    const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, INT_ORDER_MAX, iG);
                    const real det = CellJacobianDet(view.Dimension(), coords, shape);
                    DNDS_check_throw_info(std::isfinite(det) && det > 0, "FV cell has nonpositive Jacobian");
                    const tPoint p = PPhysicsCoordD01Nj(coords, shape);
                    DNDS_check_throw_info(!axisymmetric || p[1] >= 0, "negative axisymmetric radius");
                    const real weight = det * exact.GetWeight(iG);
                    plainVolume += weight;
                    plainMoment += p * weight;
                    volume += weight * (axisymmetric ? p[1] : 1.);
                }
                for (int iG = 0; iG < quadrature.GetNumPoints(); ++iG)
                {
                    const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, intOrder, iG);
                    const tPoint p = PPhysicsCoordD01Nj(coords, shape);
                    bary += p * CellJacobianDet(view.Dimension(), coords, shape) * quadrature.GetWeight(iG) * (axisymmetric ? p[1] : 1.);
                }
                DNDS_check_throw_info(std::isfinite(volume) && volume > 0, "nonpositive FV cell volume");
                bary /= volume;
                for (int d = 0; d < 3; ++d)
                    radial(iCell, d) = axisymmetric ? (plainMoment[d] - plainVolume * bary[d]) / volume : 0;
                const auto cent = PPhysicsCoordD01Nj(coords, Geom::Elem::GetCachedShapeAtQuadrature(element.type, 1, 0));
                const tPoint box = (coords.colwise() - bary).array().abs().rowwise().maxCoeff();
                const real scale = box.head(view.Dimension()).maxCoeff();
                DNDS_check_throw_info(std::isfinite(volume) && scale > 0, "invalid FV cell geometry");
                cell(iCell, 0) = volume;
                for (int d = 0; d < 3; ++d) cell(iCell, 1 + d) = cent[d], cell(iCell, 4 + d) = bary[d];
                cell(iCell, 7) = scale;
            });
            auto points = data->faceQuadrature.Write();
            const auto point = points.View<R::FixedRows<7>>();
            data->faceMean = R::Array<real>(R::RowLayout::Dense(view.NumFaceProc(), 7));
            auto meanWrite = data->faceMean.Write();
            const auto mean = meanWrite.View<R::FixedRows<7>>();
            OMP::ForSetupRows(0, view.NumFaceProc(), [&](index iFace)
            {
                tSmallCoords coords;
                view.GetCoordsOnFace(iFace, coords);
                const auto element = view.GetFaceElement(iFace);
                real integratedArea = 0;
                tPoint integratedNormal = tPoint::Zero(), geometricNormal = tPoint::Zero();
                {
                    const Quadrature measure(element, intOrder);
                    for (int iG = 0; iG < measure.GetNumPoints(); ++iG)
                    {
                        const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, intOrder, iG);
                        const auto jacobian = ShapeJacobianCoordD01Nj(coords, shape);
                        const tPoint normal = view.Dimension() == 2 ? FacialJacobianToNormVec<2>(jacobian) : FacialJacobianToNormVec<3>(jacobian);
                        const auto position = PPhysicsCoordD01Nj(coords, shape);
                        const real det = normal.norm();
                        DNDS_check_throw_info(std::isfinite(det) && det > 0, "FV face has zero measure");
                        DNDS_check_throw_info(!axisymmetric || position[1] >= 0, "negative axisymmetric face radius");
                        const real weight = measure.GetWeight(iG) * (axisymmetric ? position[1] : 1.);
                        integratedArea += det * weight;
                        integratedNormal += normal * weight;
                        geometricNormal += normal * measure.GetWeight(iG);
                        // Reuse this geometry pass for high-order RHS records.
                        // The midpoint record is filled after integrating the
                        // scalar area and mean normal; it needs no second pass.
                        if (faceIntOrder != 1)
                        {
                            const index iPoint = offset(iFace, 0) + iG;
                            for (int d = 0; d < 3; ++d)
                                point(iPoint, d) = position[d], point(iPoint, d + 3) = normal[d] / det;
                            point(iPoint, 6) = det * weight;
                        }
                    }
                    DNDS_check_throw_info(std::isfinite(integratedArea) && integratedArea >= 0, "invalid integrated face measure");
                }
                const tPoint center = PPhysicsCoordD01Nj(coords, Geom::Elem::GetCachedShapeAtQuadrature(element.type, 1, 0));
                // A face on the symmetry axis has zero weighted area but still
                // needs a valid geometric direction for boundary evaluation.
                if (axisymmetric && integratedArea == 0)
                    integratedNormal = geometricNormal;
                const real normalLength = integratedNormal.norm();
                DNDS_check_throw_info(std::isfinite(normalLength) && normalLength > 0, "degenerate FV mean face normal");
                integratedNormal /= normalLength;
                for (int d = 0; d < 3; ++d)
                    mean(iFace, d) = center[d], mean(iFace, 3 + d) = integratedNormal[d];
                mean(iFace, 6) = integratedArea;
                if (faceIntOrder == 1)
                {
                    const index iPoint = offset(iFace, 0);
                    for (int d = 0; d < 3; ++d)
                        point(iPoint, d) = center[d], point(iPoint, d + 3) = integratedNormal[d];
                    point(iPoint, 6) = integratedArea;
                }
            }); });
        data->halo.emplace(data->cells.BuildRowTransferPlan(communication));
        return FiniteVolumePlan(std::move(data));
    }
    void FiniteVolumePlan::Validate() const
    {
        DNDS_check_throw_info(_data && _data->stamp.Valid(), "finite-volume plan is stale; reconstruct metrics");
        _data->cells.Validate();
    }
    FiniteVolumeRead::FiniteVolumeRead(std::shared_ptr<const Detail::FiniteVolumeData> data, R::ExecutionContext context)
        : _data(std::move(data)),
          _meshRead(std::make_shared<MeshReadAccess>(_data->mesh->AcquireRead(MeshReadRequirements::CellFlux()))),
          _cells(_data->cell.Read(context)), _quadrature(_data->faceQuadrature.Read(context)), _radialMoment(_data->radialMoment.Read(context)), _faceMean(_data->faceMean.Read(context)), _offsets(_data->faceOffsets.Read(context))
    {
        DNDS_check_throw_info(_data->stamp.Valid(), "finite-volume metrics changed before acquisition");
    }
    FiniteVolumeHostView FiniteVolumeRead::HostView() const
    {
        DNDS_check_throw_info(_data->cell.GetDevice() == R::Device::Host(), "HostView requested from device FV metrics");
        return View();
    }
    FiniteVolumeHostView FiniteVolumeRead::View() const
    {
        return {_cells.View<R::FixedRows<8>>(), _offsets.View<R::FixedRows<1>>(), _quadrature.View<R::FixedRows<7>>(), _radialMoment.View<R::FixedRows<3>>(), _faceMean.View<R::FixedRows<7>>()};
    }
    FiniteVolumeRead FiniteVolumePlan::AcquireRead(R::ExecutionContext context) const
    {
        Validate();
        DNDS_check_throw_info(context.device == Device(), "FV metric read context differs from placement");
        return FiniteVolumeRead(_data, context);
    }
    const R::Array<real> &FiniteVolumePlan::CellMetrics() const { return _data->cell; }
    const R::Array<real> &FiniteVolumePlan::FaceQuadrature() const { return _data->faceQuadrature; }
    const R::Array<real> &FiniteVolumePlan::FaceMetrics() const { return _data->faceMean; }
    const R::Array<real> &FiniteVolumePlan::RadialMoments() const { return _data->radialMoment; }
    const R::Array<index> &FiniteVolumePlan::FaceOffsets() const { return _data->faceOffsets; }
    MeshReadAccess FiniteVolumePlan::AcquireMeshRead(bool wallDistance) const
    {
        auto requirements = MeshReadRequirements::CellFlux();
        requirements.wallDistance = wallDistance;
        auto guard = _data->mesh->AcquireRead(requirements);
        Validate(); // The retained generation excludes mutation during validation.
        return guard;
    }
    R::Device FiniteVolumePlan::Device() const { return _data->cell.GetDevice(); }
    FiniteVolumePlan FiniteVolumePlan::Placed(R::Device device, R::ExecutionContext context) const
    {
        Validate();
        if (device == Device())
            return *this;
        auto data = std::make_shared<Detail::FiniteVolumeData>(*_data, device, context);
        DNDS_check_throw_info(data->stamp.Valid(), "FV geometry changed during metric placement");
        return FiniteVolumePlan(std::move(data));
    }
    index FiniteVolumePlan::NumCell() const { return _data->cells.NumOwned(); }
    MPI_Comm FiniteVolumePlan::Communicator() const { return _data->communication->Communicator(); }
    index FiniteVolumePlan::NumCellProc() const { return _data->cells.NumLocal(); }
    index FiniteVolumePlan::NumFaceProc() const { return _data->faceOffsets.RowCount() - 1; }
    int FiniteVolumePlan::Dimension() const { return _data->dim; }
    bool FiniteVolumePlan::Axisymmetric() const { return _data->axisymmetric; }
    const EntitySpace &FiniteVolumePlan::Cells() const { return _data->cells; }
    EntityWorkPartition FiniteVolumePlan::CellWorkPartition() const
    {
        Validate();
        std::call_once(_data->partitionOnce, [&]
                       { if (!_data->workPartition) _data->workPartition.emplace(EntityWorkPartition::CellNodeCell(*_data->mesh)); });
        _data->workPartition->Validate();
        return *_data->workPartition;
    }
    FVField FiniteVolumePlan::BuildField(index width) const
    {
        Validate();
        DNDS_check_throw_info(width > 0, "FV field width must be positive");
        FVField field(_data->dofs, R::RowLayout::Dense(NumCell(), width), Device());
        // Runtime allocation publishes zero-initialized device buffers. Keep
        // the existing explicit CPU initialization/version behavior unchanged.
        if (Device() == R::Device::Host())
            Detail::Zero(field);
        return field;
    }
    FVField FiniteVolumePlan::BuildField(R::RowLayout prototype) const
    {
        return BuildField(std::move(prototype), Device());
    }
    FVField FiniteVolumePlan::BuildField(R::RowLayout prototype, R::Device device) const
    {
        Validate();
        DNDS_check_throw_info(!prototype.IsVariable() && prototype.Width() > 0, "FV fields require a uniform physical table");
        return FVField(_data->dofs, std::move(prototype), device);
    }
    void FiniteVolumePlan::ValidateField(const FVField &field, index width) const
    {
        Validate();
        field.Validate();
        DNDS_check_throw_info(field.Space().Same(_data->dofs) && index(field.Width()) == width && field.Device() == Device(),
                              "FV field belongs to another plan/order or has the wrong width/placement");
    }
    R::TransferBinding FiniteVolumePlan::BindHalo(FVField &field, R::TransferOptions options) const
    {
        _data->communication->LocalStage([&]
                                         { ValidateField(field, field.Width()); });
        return R::PayloadPlan(*_data->halo, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::Arithmetic<real>())
            .Bind(field.Owned(), field.Ghost(), R::TransferTransport::Direct, std::move(options));
    }
    Algebra::DofHalo<> FiniteVolumePlan::BindTrackedHalo(FVField &field, R::TransferOptions options) const
    {
        _data->communication->LocalStage([&]
                                         { ValidateField(field, field.Width()); });
        return Algebra::DofHalo<>(field, *_data->halo, std::move(options));
    }
    real FiniteVolumePlan::GetCellVol(index iCell) const
    {
        Validate();
        auto read = _data->cell.Read();
        return read.View<R::FixedRows<8>, true>()(iCell, 0);
    }
    Runtime::Array<real> FiniteVolumePlan::BuildCellMaxLenScale() const
    {
        const auto geometry = AcquireRead();
        const auto metric = geometry.HostView();
        const auto access = _data->mesh->AcquireRead(MeshReadRequirements::Primary());
        const auto mesh = access.HostView();
        Runtime::Array<real> result(Runtime::RowLayout::Dense(NumCellProc(), 1));
        auto write = result.Write();
        const auto length = write.View<Runtime::FixedRows<1>>();
        tSmallCoords coordinates;
        for (index iCell = 0; iCell < NumCellProc(); ++iCell)
        {
            mesh.GetCoordsOnCell(iCell, coordinates);
            const auto element = mesh.GetCellElement(iCell);
            const Quadrature quadrature(element, INT_ORDER_MAX);
            const tPoint bary = metric.GetCellBary(iCell);
            tJacobi inertia = tJacobi::Zero();
            for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
            {
                const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, INT_ORDER_MAX, iQ);
                const tPoint centered = PPhysicsCoordD01Nj(coordinates, shape) - bary;
                // Legacy inertia uses the geometric Jacobian even in radial
                // integration, with its metric volume and barycentre.
                inertia += (centered * centered.transpose()) * CellJacobianDet(Dimension(), coordinates, shape) * quadrature.GetWeight(iQ);
            }
            inertia /= metric.GetCellVol(iCell);
            tJacobi frame = tJacobi::Identity();
            if (Dimension() == 3)
                frame = HardEigen::Eigen3x3RealSymPrincipalFrame(inertia);
            else
                frame({0, 1}, {0, 1}) = HardEigen::Eigen2x2RealSymPrincipalFrame(inertia({0, 1}, {0, 1}));
            const tSmallCoords centered = frame.transpose() * (coordinates.colwise() - bary);
            const tPoint halfBox = centered.array().abs().rowwise().maxCoeff();
            const real value = 2 * halfBox.head(Dimension()).maxCoeff();
            DNDS_check_throw_info(std::isfinite(value) && value > 0, "invalid principal-axis cell length");
            length(iCell, 0) = value;
        }
        return result;
    }
    tPoint FiniteVolumePlan::GetCellCent(index iCell) const
    {
        Validate();
        auto read = _data->cell.Read();
        const auto view = read.View<R::FixedRows<8>, true>();
        return {view(iCell, 1), view(iCell, 2), view(iCell, 3)};
    }
}
