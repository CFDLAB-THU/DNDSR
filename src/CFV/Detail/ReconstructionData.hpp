#pragma once

#include "DNDS/Core/RowRanges.hpp"

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "CFV/Math/PolynomialReconstruction.hpp"
#include "CFV/Math/TaylorBasis.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "Geom/Access/EntityWorkPartition.hpp"
#include "Geom/Access/CellFaceIncidence.hpp"
#include "DNDS/Core/OMP.hpp"
#include <mutex>

namespace DNDS::CFV::Detail
{
    /// Stable logical plan identity independent of host/device payload lifetime.
    struct ReconstructionIdentity
    {
    };
    /// Implementation storage. Public plans expose no mutable aliases to it.
    struct FiniteVolumeData
    {
        ssp<Geom::UnstructuredMesh> mesh;
        Runtime::VersionStamp stamp;
        Geom::EntitySpace cells;
        std::shared_ptr<Runtime::CommunicationContext> communication;
        Algebra::DofSpace dofs;
        std::optional<Runtime::RowTransferPlan> halo;
        mutable std::once_flag partitionOnce;
        mutable std::optional<Geom::EntityWorkPartition> workPartition;
        int dim, intOrder;
        bool axisymmetric;
        // Cell table: volume, center[3], barycenter[3], isotropic half-box scale.
        Runtime::Array<real> cell;
        Runtime::Array<real> cellSmoothScaleFactors; // owned cells only
        // Integral (x-bary_r)*dA divided by radial volume, used by the weighted
        // divergence identity in axisymmetric Green-Gauss. Zero otherwise.
        Runtime::Array<real> radialMoment;
        // Face quadrature table: position[3], unit normal[3], weighted measure.
        Runtime::Array<real> faceQuadrature;
        // One geometric record per face: reference center, normalized area-
        // integrated normal, and scalar area. Independent of RHS quadrature.
        Runtime::Array<real> faceMean;
        Runtime::Array<index> faceOffsets;
        Runtime::Array<uint8_t> facePeriodicCodes;
        Runtime::Array<FacePeriodicParameters> periodicParameters;
        Runtime::Array<real> neighborCenters;
        /// Canonical incidence-only placement. It shares host native arrays
        /// and copies only the requested topology when placed on a device.
        /// Operator read scopes pin the source generation through queued use.
        std::optional<Geom::CellFaceIncidenceStorage> topology;
        FiniteVolumeData(ssp<Geom::UnstructuredMesh> m, int order, bool axis, std::shared_ptr<Runtime::CommunicationContext> context)
            : mesh(std::move(m)), stamp(mesh->versions.Stamp(Runtime::Change::Topology | Runtime::Change::Geometry |
                                                             Runtime::Change::Distribution | Runtime::Change::Layout)),
              cells(mesh->Entities(Geom::EntityKind::Cell)),
              communication(std::move(context)), dofs(communication, cells.NumOwned(), cells.NumGhost(), "FV cells"),
              dim(mesh->getDim()), intOrder(order), axisymmetric(axis)
        {
            topology.emplace(Geom::CellFaceIncidenceStorage::Capture(mesh));
        }
        FiniteVolumeData(const FiniteVolumeData &source, Runtime::Device device, Runtime::ExecutionContext context)
            : mesh(source.mesh), stamp(source.stamp), cells(source.cells), communication(source.communication),
              dofs(source.dofs), halo(source.halo), workPartition(source.workPartition),
              dim(source.dim), intOrder(source.intOrder), axisymmetric(source.axisymmetric),
              cell(source.cell.To(device, context)), cellSmoothScaleFactors(source.cellSmoothScaleFactors.To(device, context)),
              radialMoment(source.radialMoment.To(device, context)),
              faceQuadrature(source.faceQuadrature.To(device, context)), faceMean(source.faceMean.To(device, context)),
              faceOffsets(source.faceOffsets.To(device, context)),
              facePeriodicCodes(source.facePeriodicCodes.To(device, context)),
              periodicParameters(source.periodicParameters.To(device, context)),
              neighborCenters(source.neighborCenters.To(device, context)),
              topology(source.topology->Placed(device, context))
        {
            cell.SetName("CFV/metrics/cell", DNDS_ALLOCATION_SITE());
            cellSmoothScaleFactors.SetName("CFV/metrics/cellSmoothScaleFactors", DNDS_ALLOCATION_SITE());
            radialMoment.SetName("CFV/metrics/radialMoment", DNDS_ALLOCATION_SITE());
            faceQuadrature.SetName("CFV/metrics/faceQuadrature", DNDS_ALLOCATION_SITE());
            faceMean.SetName("CFV/metrics/faceMean", DNDS_ALLOCATION_SITE());
            faceOffsets.SetName("CFV/metrics/faceOffsets", DNDS_ALLOCATION_SITE());
            facePeriodicCodes.SetName("CFV/metrics/facePeriodicCodes", DNDS_ALLOCATION_SITE());
            periodicParameters.SetName("CFV/metrics/periodicParameters", DNDS_ALLOCATION_SITE());
            neighborCenters.SetName("CFV/metrics/neighborCenters", DNDS_ALLOCATION_SITE());
        }
    };
    struct ReconstructionData
    {
        std::shared_ptr<const ReconstructionIdentity> identity;
        FiniteVolumePlan metrics;
        ReconstructionSettings settings;
        ReconstructionCapabilities capabilities;
        uint64_t preparationRuntimeScratchBytes = 0;
        ReconstructionInverseDiagnostics inverseDiagnostics;
        std::optional<Runtime::Array<real>> inverseConditions;
        std::optional<Geom::EntityWorkPartition> workPartition;
        // Inverse logical order for split SOR. Independent of physical row order.
        std::vector<index> coreFirstPosition;
        /// Immutable host subpartition ranges copied from the prepared mesh.
        /// Block SOR reads changing output only within its own range; neighbors
        /// in other ranges always use the frozen input, independent of timing.
        Runtime::MetadataBuffer<WorkerRowRanges> hostWorkers;
        int hostThreads = 1;
        int n, cachedDerivatives;
        Runtime::Array<real> moments;
        Runtime::Array<real> faceBasis;
        // One row per owned cell-face incidence, preallocated from cell2face.
        Runtime::Array<index> offsets, neighbors, faces, boundaryOffsets;
        std::optional<Runtime::Array<real>> matrixAInvB, vectorAInvB, secondary;
        std::optional<Runtime::Array<real>> localMatrix, localInverse;
        // Unscaled blocks retain the original assembled operator even when
        // the truncated local inverse projects out small, nonzero modes.
        std::optional<Runtime::Array<real>> rawMatrixB, rawVectorB, rawBoundaryInjection;
        /// Raw face representation reuses faces as its row mapping. This byte
        /// records the canonical parent side; side one reads B transposed.
        std::optional<Runtime::Array<uint8_t>> rawFaceSide;
        const Runtime::Array<real> &SweepMatrices() const { return settings.storage.Raw() ? *rawMatrixB : *matrixAInvB; }
        const Runtime::Array<real> &SweepVectors() const { return settings.storage.Raw() ? *rawVectorB : *vectorAInvB; }
        const Runtime::Array<real> &SweepInjection() const { return settings.storage.Raw() ? *rawBoundaryInjection : *boundaryInjection; }
        Runtime::Array<real> neighborRotation;
        /// Setup-time incidence classification. A byte avoids hot-loop matrix
        /// comparisons and distinguishes ordinary/translated faces in a mesh
        /// that also contains genuinely rotated periodic incidences.
        Runtime::Array<uint8_t> neighborRotationActive;
        bool hasRotatedPeriodicity = false;
        // Dirichlet injection already multiplied by A^{-1} and quadrature weight.
        std::optional<Runtime::Array<real>> boundaryInjection;
        Runtime::Array<index> boundaryPoints;
        ReconstructionData(FiniteVolumePlan m, ReconstructionSettings s, ReconstructionCapabilities c)
            : identity(std::make_shared<const ReconstructionIdentity>()), metrics(std::move(m)), settings(s), capabilities(c), n(Math::PolynomialSize(metrics.Dimension(), s.maxOrder) - 1),
              cachedDerivatives(s.cacheDiffBase ? Math::TaylorBasis(metrics.Dimension(), s.maxOrder).DerivativeNumber(s.cacheDiffBaseOrder) : 0)
        {
            hostThreads = OMP::DistributedThreads();
        }
    };
    /// Prepare the immutable hot-path face cache independently of VR matrix
    /// assembly, after moments and geometry setup writers have been released.
    void PrepareFaceBasis(ReconstructionData &data, const FiniteVolumeData &geometry);

    inline void Zero(FVField &field)
    {
        for (auto *array : {&field.Owned(), &field.Ghost()})
        {
            auto write = array->Write();
            const auto view = write.View<Runtime::UniformTableRows>();
            OMP::ForRows(0, view.RowCount(), [&](index iRow)
                         { std::fill(view[iRow].begin(), view[iRow].end(), 0.); });
        }
    }
}
