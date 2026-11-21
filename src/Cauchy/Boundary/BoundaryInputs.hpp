#pragma once

#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Core/StateEpoch.hpp"
#include "Geom/Spatial/ElementGeometry.hpp"
#include <optional>
#include <atomic>

namespace DNDS::Cauchy
{
    class FiniteVolume;
    class InterfaceDonorPublication;
    class AdmissibilityPolicy;
    class MeanStateCheck;
    namespace Detail
    {
        struct BoundaryPointLayout;
    }
    /// Receiver reference-frame point row; quadrature=-1 denotes the maintained
    /// face center. This is prepared geometry, validated by its FiniteVolume.
    struct BoundaryPoint
    {
        index cell, face, quadrature;
        int zone;
        Vector3 point, normal;
        bool interface;
    };
    /// One immutable receiver-face range. Actual cell/zone/normal/position are
    /// queried from Geom/CFV; no persistent per-quadrature geometry is copied.
    /// quadratureCount may be zero for a center-only receiver. The center row
    /// is this record's ordinal in the boundary-face registry.
    struct BoundaryFaceRange
    {
        index face = UnInitIndex;
        index quadratureBegin = 0;
        index quadratureCount = 0;
        bool interface = false;
    };
    /// One completed inter-domain matching/sampling source retained by a
    /// boundary publication. Poses are exact values for this physical stage;
    /// routeGeneration changes whenever matching is rebuilt. routeOwner pins
    /// the distributed match, return route and donor geometry snapshot after
    /// the controller returns. Payload arrays remain owned by BoundaryInputs.
    struct BoundaryInterfaceSource
    {
        index donorDomain = -1, receiverDomain = -1;
        uint64_t routeGeneration = 0;
        Geom::RigidPose donorPose, receiverPose;
        std::shared_ptr<const void> routeOwner;
        /// Retains this rank's sampled donor fields and recorded generations.
        /// Empty on receiver-only ranks: remote allocations are owned on their
        /// donor ranks, never represented by a local pointer. The packet epoch
        /// and frozen route identify that remote sampling round. This retains
        /// storage, not a read lease or immutable copy of future donor values.
        std::shared_ptr<const InterfaceDonorPublication> localDonor;
    };
    /// Explicit incoming equation packet in receiver axes. QuadratureValues()
    /// holds RHS/reconstruction rows; FaceCenterValues() holds independently
    /// sampled approximate-Jacobian rows. Neither is an average of the other.
    /// Both carry the complete schema in canonical logical order and are
    /// published atomically; only interface rows are required. The coupling
    /// controller owns matching, interpolation and transfers.
    ///
    /// BeginBoundaryUpdate clears interface rows to NaN and withdraws the old
    /// publication; fill/transfer every required row, then PublishBoundaryInputs.
    /// This catches incomplete updates without a row-sized coverage bitmap.
    /// The packet adds no communication to a domain by itself. Copies share
    /// payload storage and the publication slot of one update round. A copy
    /// obtained during that round may declare queued array reads before the
    /// producer completes; it must import the producer event and publication
    /// dependency before consuming values. BeginBoundaryUpdate installs a new
    /// slot, so old copies never silently acquire a later round's identity.
    /// Mutation invalidates old copies through allocation/shape generations.
    struct BoundaryInputs
    {
    private:
        friend class FiniteVolume;
        EquationDofs _quadrature;
        EquationDofs _faceCenters;
        EquationPropertyKey _property;
        std::shared_ptr<const AdmissibilityPolicy> _admissibility;
        std::shared_ptr<const Detail::BoundaryPointLayout> _layout;
        /// Optional validated consumer of a superset packet, e.g. a pMG
        /// center-only reader of the live fine packet. Provider sections and
        /// publication identity remain unchanged; the alias never deletes data.
        std::shared_ptr<const Detail::BoundaryPointLayout> _consumerLayout;
        /// Optional resident validation workspaces; shared packet copies refer
        /// to the same serial publication round, never concurrent checks.
        std::shared_ptr<MeanStateCheck> _quadratureCheck, _faceCheck;
        struct Publication
        {
            real time;
            StateEpoch epoch;
            std::vector<uint64_t> versions, shapes, faceVersions, faceShapes;
            std::vector<BoundaryInterfaceSource> sources;
        };
        struct PublicationSlot
        {
            // Atomically installed once, after all writes and validation.
            // This is metadata publication, not an array readiness mechanism.
            std::shared_ptr<const Publication> value;
        };
        std::shared_ptr<PublicationSlot> _publication;
        std::shared_ptr<const Publication> Published() const
        {
            return _publication ? std::atomic_load(&_publication->value) : nullptr;
        }
        BoundaryInputs(EquationDofs quadrature, EquationDofs faceCenters,
                       EquationPropertyKey property,
                       std::shared_ptr<const AdmissibilityPolicy> admissibility)
            : _quadrature(std::move(quadrature)),
              _faceCenters(std::move(faceCenters)),
              _property(std::move(property)), _admissibility(std::move(admissibility)) {}

    public:
        /// O(1) identity of a completed publication, preserved by packet alias.
        /// This does not test live payload versions: ValidateBoundaryInputs
        /// separately rejects a write or shape edit after publication.
        bool SamePublication(const BoundaryInputs &other) const
        {
            const auto publication = Published();
            return publication && publication == other.Published();
        }
        /// Managed complete equation rows. Mutable access is intentional for a
        /// begun controller update; any later write invalidates publication by
        /// native array generations. No raw pointer escapes this packet API.
        EquationDofs &QuadratureValues() { return _quadrature; }
        const EquationDofs &QuadratureValues() const { return _quadrature; }
        EquationDofs &FaceCenterValues() { return _faceCenters; }
        const EquationDofs &FaceCenterValues() const { return _faceCenters; }
        /// Describes the frozen source snapshot supplied by the controller;
        /// these queries never refresh a field or assert current donor state.
        real Time() const
        {
            const auto publication = Published();
            DNDS_check_throw_info(publication, "boundary packet is unpublished");
            return publication->time;
        }
        StateEpoch Epoch() const
        {
            const auto publication = Published();
            DNDS_check_throw_info(publication, "boundary packet is unpublished");
            return publication->epoch;
        }
        /// Immutable source identities for an interface-controller packet.
        /// A manually prepared physical/test packet may have no sources.
        const std::vector<BoundaryInterfaceSource> &InterfaceSources() const
        {
            const auto publication = Published();
            DNDS_check_throw_info(publication,
                                  "boundary packet is unpublished");
            // The slot retains this immutable publication for its lifetime.
            return publication->sources;
        }
    };
}
