#pragma once

#include "Geom/Access/MeshView.hpp"
#include "Geom/Unstructured/UnstructuredMeshFwd.hpp"
#include "DNDS/Runtime/Storage/Version.hpp"
#include <functional>
#include <memory>

namespace DNDS::Geom
{
    class CellFaceIncidenceStorage;

    /** @brief Read scope for canonical cell/face incidence on one backend.
     * @details The scope shares its immutable value/shape allocations and pins
     * the source mesh generation. A copied view is borrowed: retain this scope
     * through CPU worker or device completion, including every using stream.
     * Acquisition performs no placement, upload or device fence. Mutation of
     * the facade while a scope is live follows the ordinary mesh-read contract.
     */
    class CellFaceIncidenceRead
    {
        struct Data;
        std::shared_ptr<const Data> _data;
        Runtime::VersionTracker::ReadLease _generation;
        explicit CellFaceIncidenceRead(std::shared_ptr<const Data> data);
        friend class CellFaceIncidenceStorage;

    public:
        CellFaceIncidenceRead(const CellFaceIncidenceRead &) = delete;
        CellFaceIncidenceRead &operator=(const CellFaceIncidenceRead &) = delete;
        CellFaceIncidenceRead(CellFaceIncidenceRead &&) noexcept = default;
        CellFaceIncidenceRead &operator=(CellFaceIncidenceRead &&) = delete;
        CellFaceIncidenceView View() const;
    };

    /** @brief Minimal immutable placement of existing canonical mesh arrays.
     * @details Owns only CellFaces, FaceCells, face ElemInfo and, for periodic
     * meshes, CellFaces image bits. Host capture shares native storage; explicit
     * placement preserves its owned/ghost layouts and numbering. No coordinates,
     * node cones, primary metadata or inverse adjacency are copied to a device.
     * Copies share the same publication; rebuilding topology/order requires a
     * new capture. Geometry-only edits do not invalidate this topological cache.
     */
    class CellFaceIncidenceStorage
    {
        std::shared_ptr<const CellFaceIncidenceRead::Data> _data;
        explicit CellFaceIncidenceStorage(std::shared_ptr<const CellFaceIncidenceRead::Data> data) : _data(std::move(data)) {}

    public:
        static CellFaceIncidenceStorage Capture(std::shared_ptr<UnstructuredMesh> source);
        CellFaceIncidenceStorage Placed(Runtime::Device device, Runtime::ExecutionContext context) const;
        CellFaceIncidenceRead AcquireRead() const;
        Runtime::Device GetDevice() const;
        /// Allocation identities include shared row metadata. Deduplicate IDs
        /// when combining this view's attribution with another mesh owner.
        void VisitAllocations(const std::function<void(uint64_t, size_t, Runtime::Device)> &visitor) const;
    };
}
