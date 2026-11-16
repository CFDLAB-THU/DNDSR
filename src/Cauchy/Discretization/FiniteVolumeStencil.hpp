#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Core/Span.hpp"
#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Geom/Geometry/Geometric.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    struct SampleValue
    {
        index cell, other, point;
        int side;
        int zone;
        Geom::tPoint normal, ownPoint, otherPoint;
        Geom::tGPoint rotation;
        const BoundaryCondition &boundary;
        index auxiliaryFace = UnInitIndex;
        real viscousDistance = 0;
        index face = UnInitIndex;
        /// Dense column slot in this cell's canonical Jacobian row. Filled
        /// once after JacobianColumns() is constructed; UnInitIndex for
        /// diagonal/boundary samples and non-Jacobian quadrature samples.
        index jacobianColumn = UnInitIndex;
        /// Row in the incoming packet used by this sample: RHS flow for
        /// quadrature samples, faceFlow for centered Jacobian samples.
        index boundaryPoint = UnInitIndex;
        /// Dense RHSLocalDiagnostics force slot, resolved once during setup.
        /// Non-wall and interior samples keep UnInitIndex.
        index forceSlot = UnInitIndex;
        /// Exact setup-time frame test. Ordinary and translated-periodic faces
        /// keep components unchanged; even a small nonidentity rotation remains
        /// active. Default true keeps hand-built samples conservative.
        bool changesFrame = true;
    };

    /// Transient point arithmetic generated from canonical mesh/CFV views.
    /// Never allocate a field of these records: geometry, rotation and cell
    /// pairs remain shared publications. Boundary policies are held once.
    struct SampleRecord
    {
        index cell, other, point, condition;
        int side, zone;
        std::array<real, 3> normal, ownPoint, otherPoint;
        std::array<real, 9> rotation;
        index auxiliaryFace, face, jacobianColumn, boundaryPoint, forceSlot;
        real viscousDistance;
        bool changesFrame;
    };
    static_assert(std::is_trivially_copyable_v<SampleRecord>);

    /// CPU point expansion of plain field data; no owner, allocation or lease.
    inline SampleValue ExpandSample(const SampleRecord &record, SpanView<const BoundaryCondition> conditions)
    {
        const auto point = [](const auto &v)
        { return Geom::tPoint{v[0], v[1], v[2]}; };
        SampleValue sample{record.cell, record.other, record.point, record.side, record.zone,
                           point(record.normal), point(record.ownPoint), point(record.otherPoint),
                           Eigen::Map<const Geom::tGPoint>(record.rotation.data()), conditions[record.condition]};
        sample.auxiliaryFace = record.auxiliaryFace;
        sample.viscousDistance = record.viscousDistance;
        sample.face = record.face;
        sample.jacobianColumn = record.jacobianColumn;
        sample.boundaryPoint = record.boundaryPoint;
        sample.forceSlot = record.forceSlot;
        sample.changesFrame = record.changesFrame;
        return sample;
    }

}

namespace DNDS::Cauchy
{
    enum class FaceEvaluationLocation : uint8_t
    {
        Quadrature,
        Center
    };

    /// One routing record per local face; geometry and cell pairs remain in
    /// CFV/Geom. Boundary rows use the canonical cell-face traversal order.
    struct FaceRouting
    {
        index auxiliaryFace = UnInitIndex;
        index condition = 0;
        index boundaryFace = UnInitIndex;
        index boundaryBegin = UnInitIndex;
        index forceSlot = UnInitIndex;
    };
    static_assert(std::is_trivially_copyable_v<FaceRouting>);

    /** @brief Compact borrowed mesh/CFV stencil shared by RHS and Jacobians.
     * @details A SampleRecord is transient point arithmetic only. No expanded
     * cell-pair, geometry, rotation or query-to-face table is owned here.
     * Face-rooted work always uses canonical parent zero, including a ghost.
     * Cell-rooted work passes its actual incidence slot for self periodicity.
     */
    struct FiniteVolumeStencilView
    {
        using Indices = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
        Geom::CellFaceIncidenceView topology;
        CFV::FiniteVolumeHostView metric;
        Runtime::ArrayView<const FaceRouting, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> routing;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> columns;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> viscousDistance;
        Indices jacobianOffsets, stencilFaces, auxiliaryCoreFaces, auxiliaryDependentFaces,
            rhsCoreFaces, rhsDependentFaces, auxiliaryCellOffsets, auxiliaryCellFaces,
            coreCells, dependentCells,
            auxiliaryInterfaceIndependentFaces, auxiliaryInterfaceFaces,
            interfaceIndependentCells, interfaceCoreCells;
        bool quadratureBoundaryValues = true;

        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, topology, metric, routing, columns, viscousDistance,
                                         jacobianOffsets, stencilFaces, auxiliaryCoreFaces, auxiliaryDependentFaces, rhsCoreFaces,
                                         rhsDependentFaces, auxiliaryCellOffsets, auxiliaryCellFaces,
                                         coreCells, dependentCells, auxiliaryInterfaceIndependentFaces, auxiliaryInterfaceFaces,
                                         interfaceIndependentCells, interfaceCoreCells);
        }
        DNDS_DEVICE_CALLABLE index IncidenceNumber() const
        {
            return jacobianOffsets.RowCount() ? jacobianOffsets(jacobianOffsets.RowCount() - 1, 0) : 0;
        }
        DNDS_DEVICE_CALLABLE FiniteVolumeDetail::SampleRecord FaceSample(index iFace, int side, index iPoint,
                                                                         FaceEvaluationLocation location) const
        {
            const auto parents = topology.FaceCells(iFace);
            const auto route = routing(iFace, 0);
            const auto rotation = metric.periodic.OtherRotation(iFace, side);
            const auto physicalPoint = location == FaceEvaluationLocation::Center ? metric.GetFaceCent(iFace) : metric.GetFacePoint(iPoint);
            const auto point = metric.periodic.FacePoint(iFace, side, physicalPoint);
            const auto leftNormal = location == FaceEvaluationLocation::Center ? metric.GetFaceMeanNorm(iFace) : metric.GetFaceNorm(iPoint);
            const auto normal = side == 0 ? leftNormal : Geom::tPoint(-rotation * leftNormal);
            const auto other = metric.GetNeighborCent(iFace, side);
            FiniteVolumeDetail::SampleRecord result{};
            result.cell = parents[side];
            result.other = parents[1 - side];
            result.point = iPoint;
            result.condition = route.condition;
            result.side = side;
            result.zone = topology.GetFaceZone(iFace);
            for (int d = 0; d < 3; ++d)
            {
                result.normal[d] = normal[d];
                result.ownPoint[d] = point[d];
                result.otherPoint[d] = other[d];
            }
            for (int d = 0; d < 9; ++d)
                result.rotation[d] = rotation.data()[d];
            result.auxiliaryFace = route.auxiliaryFace;
            result.face = iFace;
            result.jacobianColumn = columns(iFace, side);
            result.boundaryPoint = result.other != UnInitIndex ? UnInitIndex : location == FaceEvaluationLocation::Center || !quadratureBoundaryValues ? route.boundaryFace
                                                                                                                                                       : route.boundaryBegin + iPoint - metric.FaceQuadratureBegin(iFace);
            result.forceSlot = route.forceSlot;
            result.viscousDistance = location == FaceEvaluationLocation::Quadrature ? viscousDistance(iPoint, 0) : 0;
            result.changesFrame = metric.periodic.Rotates(iFace);
            return result;
        }
        DNDS_DEVICE_CALLABLE FiniteVolumeDetail::SampleRecord CanonicalFaceSample(index iFace, index iPoint,
                                                                                  FaceEvaluationLocation location = FaceEvaluationLocation::Quadrature) const
        {
            return FaceSample(iFace, 0, iPoint, location);
        }
        /// Direct FV is a midpoint consumer, sharing the independently prepared
        /// center packet. Its scalar viscous distance remains the prepared
        /// quadrature-normal value; the high-order path is unchanged.
        DNDS_DEVICE_CALLABLE FiniteVolumeDetail::SampleRecord RHSFaceSample(index iFace, int side, index iPoint) const
        {
            auto sample = FaceSample(iFace, side, iPoint, quadratureBoundaryValues ? FaceEvaluationLocation::Quadrature : FaceEvaluationLocation::Center);
            sample.viscousDistance = viscousDistance(iPoint, 0);
            return sample;
        }
        DNDS_DEVICE_CALLABLE FiniteVolumeDetail::SampleRecord CellSample(index iCell, rowsize slot, index iPoint,
                                                                         FaceEvaluationLocation location = FaceEvaluationLocation::Quadrature) const
        {
            const auto iFace = topology.CellFaces(iCell)[slot];
            return FaceSample(iFace, topology.CellIsFaceBack(iCell, iFace, slot) ? 0 : 1, iPoint, location);
        }
        DNDS_DEVICE_CALLABLE FiniteVolumeDetail::SampleRecord GradientSample(index iCell, rowsize slot) const
        {
            const auto iFace = topology.CellFaces(iCell)[slot];
            return CellSample(iCell, slot, metric.FaceQuadratureBegin(iFace), FaceEvaluationLocation::Center);
        }
        DNDS_DEVICE_CALLABLE FiniteVolumeDetail::SampleRecord JacobianSample(index iCell, index entry) const
        {
            return GradientSample(iCell, rowsize(entry - jacobianOffsets(iCell, 0)));
        }
        /// Physical gradients represent an affine field about the cell mean's
        /// barycenter. Preserve that origin on curved cells and periodic images.
        DNDS_DEVICE_CALLABLE Geom::tPoint GradientDisplacement(index iCell, rowsize slot) const
        {
            const auto iFace = topology.CellFaces(iCell)[slot];
            const int side = topology.CellIsFaceBack(iCell, iFace, slot) ? 0 : 1;
            return metric.periodic.FacePoint(iFace, side, metric.GetFaceCent(iFace)) - metric.GetCellBary(iCell);
        }
    };

    /** @brief Immutable routing fields over retained canonical mesh/CFV data.
     * @details Freeze after setup. Its publication owns read leases, topology
     * and metric owners once; queued operators capture a ReadAccess handle.
     * Reading frozen metadata acquires no immediate lease and creates no event.
     * Changing topology/geometry/order requires a new publication.
     */
    struct FiniteVolumeStencilStorage
    {
        Runtime::Array<FaceRouting> routing{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> columns{Runtime::RowLayout::Dense(0, 2)};
        Runtime::Array<real> viscousDistance{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> jacobianOffsets{Runtime::RowLayout::Dense(0, 1)},
            stencilFaces{Runtime::RowLayout::Dense(0, 1)},
            auxiliaryCoreFaces{Runtime::RowLayout::Dense(0, 1)},
            auxiliaryDependentFaces{Runtime::RowLayout::Dense(0, 1)},
            rhsCoreFaces{Runtime::RowLayout::Dense(0, 1)},
            rhsDependentFaces{Runtime::RowLayout::Dense(0, 1)},
            auxiliaryCellOffsets{Runtime::RowLayout::Dense(0, 1)},
            auxiliaryCellFaces{Runtime::RowLayout::Dense(0, 1)},
            coreCells{Runtime::RowLayout::Dense(0, 1)},
            dependentCells{Runtime::RowLayout::Dense(0, 1)},
            auxiliaryInterfaceIndependentFaces{Runtime::RowLayout::Dense(0, 1)},
            auxiliaryInterfaceFaces{Runtime::RowLayout::Dense(0, 1)},
            interfaceIndependentCells{Runtime::RowLayout::Dense(0, 1)},
            interfaceCoreCells{Runtime::RowLayout::Dense(0, 1)};
        bool quadratureBoundaryValues = true;
        index incidenceCount = 0;
        index IncidenceNumber() const { return incidenceCount; }

    private:
        struct Publication
        {
            CFV::FiniteVolumeTopologyRead topology;
            CFV::FiniteVolumeRead metric;
            Runtime::Array<FaceRouting>::ReadLease routing;
            Runtime::Array<index>::ReadLease columns;
            Runtime::Array<real>::ReadLease distance;
            std::vector<Runtime::Array<index>::ReadLease> indices;
            FiniteVolumeStencilView view;
            Runtime::MetadataBuffer<FiniteVolumeStencilView> descriptor;
            Publication(const FiniteVolumeStencilStorage &s, const CFV::FiniteVolumePlan &geometry)
                : topology(geometry.AcquireTopologyRead()), metric(geometry.AcquireRead({geometry.Device(), nullptr})),
                  routing(s.routing.Read({geometry.Device(), nullptr})),
                  columns(s.columns.Read({geometry.Device(), nullptr})),
                  distance(s.viscousDistance.Read({geometry.Device(), nullptr}))
            {
                view.topology = topology.View();
                view.metric = metric.View();
                view.routing = routing.View<Runtime::FixedRows<1>>();
                view.columns = columns.View<Runtime::FixedRows<2>>();
                view.viscousDistance = distance.View<Runtime::FixedRows<1>>();
                view.quadratureBoundaryValues = s.quadratureBoundaryValues;
                const auto capture = [&](const Runtime::Array<index> &array)
                {
                    indices.push_back(array.Read({geometry.Device(), nullptr}));
                    return indices.back().View<Runtime::FixedRows<1>>();
                };
                view.jacobianOffsets = capture(s.jacobianOffsets);
                view.stencilFaces = capture(s.stencilFaces);
                view.auxiliaryCoreFaces = capture(s.auxiliaryCoreFaces);
                view.auxiliaryDependentFaces = capture(s.auxiliaryDependentFaces);
                view.rhsCoreFaces = capture(s.rhsCoreFaces);
                view.rhsDependentFaces = capture(s.rhsDependentFaces);
                view.auxiliaryCellOffsets = capture(s.auxiliaryCellOffsets);
                view.auxiliaryCellFaces = capture(s.auxiliaryCellFaces);
                view.coreCells = capture(s.coreCells);
                view.dependentCells = capture(s.dependentCells);
                view.auxiliaryInterfaceIndependentFaces = capture(s.auxiliaryInterfaceIndependentFaces);
                view.auxiliaryInterfaceFaces = capture(s.auxiliaryInterfaceFaces);
                view.interfaceIndependentCells = capture(s.interfaceIndependentCells);
                view.interfaceCoreCells = capture(s.interfaceCoreCells);
                view.ValidateDevice(geometry.Device());
                // The complete bundle is larger than many kernels need in
                // their parameter block. Publish its POD descriptor once;
                // device wrappers pass this whole-array view, never upload it
                // per invocation or add untracked pointer/count arguments.
                descriptor = Runtime::MetadataBuffer<FiniteVolumeStencilView>({view});
                if (geometry.Device() != Runtime::Device::Host())
                    descriptor = descriptor.CopyTo(geometry.Device(), {geometry.Device(), nullptr});
                descriptor.SetDiagnosticName("Cauchy/stencil/kernelDescriptor", DNDS_ALLOCATION_SITE());
            }
        };
        std::shared_ptr<const Publication> _publication;

    public:
        class ReadAccess
        {
            std::shared_ptr<const Publication> _owner;
            explicit ReadAccess(std::shared_ptr<const Publication> owner) : _owner(std::move(owner)) {}
            friend struct FiniteVolumeStencilStorage;

        public:
            FiniteVolumeStencilView View() const { return _owner->view; }
            Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> KernelDescriptor() const
            {
                const auto device = _owner->descriptor.GetDevice();
                const auto values = _owner->descriptor.View(device);
                return {values.data(), nullptr, index(values.size()), 1, 1, nullptr, Runtime::RowLayoutKind::Table, device};
            }
        };
        void Freeze(const CFV::FiniteVolumePlan &metrics)
        {
            DNDS_check_throw_info(!_publication, "stencil routing was already published");
            _publication = std::make_shared<Publication>(*this, metrics);
        }
        ReadAccess Read(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const
        {
            DNDS_check_throw_info(_publication && _publication->view.routing.GetDevice() == context.device,
                                  "stencil must be frozen on the requested device before use");
            return ReadAccess(_publication);
        }
        FiniteVolumeStencilStorage Placed(const CFV::FiniteVolumePlan &metrics, Runtime::ExecutionContext context) const
        {
            FiniteVolumeStencilStorage result;
            const auto device = metrics.Device();
            result.routing = routing.To(device, context);
            result.columns = columns.To(device, context);
            result.viscousDistance = viscousDistance.To(device, context);
            result.quadratureBoundaryValues = quadratureBoundaryValues;
            result.incidenceCount = incidenceCount;
            const auto place = [&](const auto &input, auto &output, const char *name)
            {
                output = input.To(device, context);
                output.SetName(std::string("Cauchy/stencil/") + name, DNDS_ALLOCATION_SITE());
            };
            place(jacobianOffsets, result.jacobianOffsets, "incidenceOffsets");
            place(stencilFaces, result.stencilFaces, "activeFaces");
            place(auxiliaryCoreFaces, result.auxiliaryCoreFaces, "auxiliaryCoreFaces");
            place(auxiliaryDependentFaces, result.auxiliaryDependentFaces, "auxiliaryDependentFaces");
            place(rhsCoreFaces, result.rhsCoreFaces, "rhsCoreFaces");
            place(rhsDependentFaces, result.rhsDependentFaces, "rhsDependentFaces");
            result.auxiliaryCellOffsets = result.jacobianOffsets;
            place(auxiliaryCellFaces, result.auxiliaryCellFaces, "auxiliaryCellFaces");
            place(coreCells, result.coreCells, "coreCells");
            place(dependentCells, result.dependentCells, "dependentCells");
            place(auxiliaryInterfaceIndependentFaces, result.auxiliaryInterfaceIndependentFaces, "auxiliaryInterfaceIndependentFaces");
            place(auxiliaryInterfaceFaces, result.auxiliaryInterfaceFaces, "auxiliaryInterfaceFaces");
            place(interfaceIndependentCells, result.interfaceIndependentCells, "interfaceIndependentCells");
            place(interfaceCoreCells, result.interfaceCoreCells, "interfaceCoreCells");
            result.routing.SetName("Cauchy/stencil/faceRouting", DNDS_ALLOCATION_SITE());
            result.columns.SetName("Cauchy/stencil/jacobianColumns", DNDS_ALLOCATION_SITE());
            result.viscousDistance.SetName("Cauchy/stencil/viscousDistance", DNDS_ALLOCATION_SITE());
            result.Freeze(metrics);
            return result;
        }
    };
}
