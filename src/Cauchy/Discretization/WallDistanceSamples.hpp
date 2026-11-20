#pragma once

#include "DNDS/Core/OptionalRef.hpp"
#include "Geom/Access/MeshAccess.hpp"

namespace DNDS::Runtime
{
    class CommunicationContext;
}

namespace DNDS::Cauchy
{
    namespace Detail
    {
        struct WallDistanceData;
    }

    /// Borrowed resident views, indexed by local cell/face and quadrature ordinal.
    /// Retain WallDistanceRead through every use. cellMean uses the arithmetic
    /// quadrature mean, or the chosen first-layer distance on selected cells.
    struct WallDistanceView
    {
        Runtime::ArrayView<const real, false, Runtime::CSRRows> cells, faces;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<1>> cellMean;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, cells, faces, cellMean);
        }
    };

    /// Pins the mesh generation, nodal source and immutable sampled buffers.
    /// All dependency checks occur at acquisition, outside numerical loops.
    class WallDistanceRead
    {
        std::shared_ptr<const Detail::WallDistanceData> _data;
        Geom::MeshReadAccess _mesh;
        Runtime::Array<real>::ReadLease _cells, _faces, _mean;
        explicit WallDistanceRead(std::shared_ptr<const Detail::WallDistanceData> data, Runtime::ExecutionContext context);
        friend class WallDistanceSamples;

    public:
        WallDistanceView HostView() const;
        /// Residency-tagged numerical views. Retain this access scope through
        /// asynchronous completion, or include its arrays in tracked operands.
        WallDistanceView View() const;
    };

    struct WallCellDistance
    {
        index cell;
        real distance;
    };
    using WallCellRows = Runtime::ArrayView<const WallCellDistance, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    enum class FirstLayerWallDistance
    {
        CellCenter,
        HalfMaxNodal,
    };

    /// Optional cache of a Geom-owned nodal wall-distance field. This does
    /// not search walls or create another wall-distance provider. Scalar nodal
    /// distances (norms of Geom's wall vectors) are interpolated by the mesh's
    /// P1/P2 shape functions, including mid-edge/face/interior nodes. Periodic
    /// vector rotations do not change their norms. Negative higher-order
    /// overshoot is clipped to the explicit nonnegative minimumDistance.
    ///
    /// Construct and AcquireRead are rank-local and make no MPI calls; all
    /// required nodal ghosts must already be published by Geom. Call inside a
    /// coordinated setup stage if subsequent work is collective. Copies share
    /// the immutable publication. Rebuild after wall, geometry, topology or
    /// distribution edits; acquisition rejects a stale publication.
    class WallDistanceSamples
    {
        std::shared_ptr<const Detail::WallDistanceData> _data;
        explicit WallDistanceSamples(std::shared_ptr<const Detail::WallDistanceData> data)
            : _data(std::move(data)) {}

    public:
        /// Mark cells incident to a selected wall zone and transfer the flags
        /// to their ghost copies. The predicate is evaluated during host setup.
        static Runtime::ArrayPair<uint8_t> BuildFirstLayerMask(
            Geom::UnstructuredMesh &mesh, std::shared_ptr<Runtime::CommunicationContext> communication,
            const std::function<bool(int)> &isWallZone);
        /// Center-based sampling broadcasts the floored center value over a cell
        /// and the average of adjacent floored centers over a face. A first-layer
        /// mean-distance override still serves mean consumers such as wall omega.
        static WallDistanceSamples Construct(ssp<Geom::UnstructuredMesh> mesh, int intOrder,
                                             real minimumDistance = 0,
                                             OptionalRef<const Runtime::ArrayPair<uint8_t>> firstLayer = {},
                                             FirstLayerWallDistance method = FirstLayerWallDistance::CellCenter,
                                             bool retainFirstLayer = false, bool replaceSamples = true,
                                             bool centerQuadratureSamples = false);
        const Runtime::Array<WallCellDistance> &FirstLayerCells() const;
        /// Minimum interpolated center sample over owned cells, before the floor.
        real LocalMinimumCellCenterDistance() const;
        /// Minimum applied distance of selected owned cells, or infinity if empty.
        real LocalMinimumFirstLayerDistance() const;
        WallDistanceRead AcquireRead(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const;
        Runtime::Device Device() const;
        /// Immutable sampled operands for queued readers. Retain the mesh wall
        /// generation and declare these arrays with Runtime::ReadAs; returning
        /// a const handle does not grant access or readiness.
        const Runtime::Array<real> &CellValues() const;
        const Runtime::Array<real> &FaceValues() const;
        const Runtime::Array<real> &CellMeans() const;
        /// Synchronous placement of immutable quadrature buffers only. The
        /// placed cache preserves mesh generation identity but retains no host
        /// source arrays; replacing the source handle releases its payloads.
        WallDistanceSamples Placed(Runtime::Device device, Runtime::ExecutionContext context) const;
        int IntegrationOrder() const;
    };
}
