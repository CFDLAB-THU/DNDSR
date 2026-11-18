#pragma once

#include "Cauchy/IO/BoundaryOutputGeometry.hpp"

#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/Boundary/ACMBoundary.hpp"
#include "Cauchy/Discretization/ACMKernelPoint.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/FiniteVolumeKernelPolicy.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSampleKernels.hpp"
// Private shared preparation data. Public callers use FiniteVolume and explicit
// Workspace/AuxiliaryValues owners; this header is not a mesh or solver API.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "CFV/Plans/ReconstructionAMGPlan.hpp"
#include "Cauchy/Discretization/FiniteVolumeQuadrature.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Boundary/PerfectGasBoundary.hpp"
#include "Cauchy/Discretization/PerfectGasKernelPoint.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeBoundaryKernels.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "DNDS/Runtime/Communication/CommunicationWorker.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    /// Immutable point-flux dispatch resolved with the FV plan. Flow5 and the
    /// extensible equation adapter consume the same policy; numerical kernels
    /// retain their specialized storage and arithmetic.
    inline FluxKernelPolicy ResolveFluxKernelPolicy(const DomainSettings &settings, const Riemann::RiemannFlux &flux)
    {
        return {flux.RequiredInputs(), settings.numerics.riemann.KernelSettings(),
                flux.Kind() == Riemann::Solver::Rusanov, flux.UsesTransverseSignal(), flux.UsesPressureStencil(),
                settings.numerics.noRsOnWall, settings.numerics.noGRPOnWall};
    }
    inline CFV::ReconstructionCapabilities ReconstructionSupport(const DomainSettings &settings)
    {
        const auto &numerics = settings.numerics;
        const bool limitedVR = numerics.limiter == "LVR" &&
                               (numerics.limitedVR.alphaMax > 0 || numerics.limitedVR.contactAlphaMax > 0);
        auto support = CFV::ReconstructionCapabilities::ForSweeps(numerics.limiter == "WBAP" || limitedVR);
        support.sweeps = numerics.reconstruction == "variational";
        return support;
    }
    inline Runtime::RowLayout FieldLayout(const Runtime::RowLayout &prototype, index rows, index width)
    {
        if (prototype.Kind() == Runtime::RowLayoutKind::ColMajorTable)
            return Runtime::RowLayout::ColMajor(rows, width, prototype.PaddingOptions());
        if (prototype.Kind() == Runtime::RowLayoutKind::TiledColMajorTable)
            return Runtime::RowLayout::TiledColMajor(rows, width, prototype.Stride());
        return Runtime::RowLayout::Dense(rows, width);
    }

}

namespace DNDS::Cauchy
{
    namespace Detail
    {
        struct BoundaryPointLayout
        {
            std::weak_ptr<Geom::UnstructuredMesh> mesh;
            Runtime::VersionStamp stamp;
            Runtime::Array<BoundaryFaceRange> faces{Runtime::RowLayout::Dense(0, 1)};
            index quadratureRows = 0;
            int integrationOrder = 1;
            int meanIntegrationOrder = 5;
            bool axisymmetric = false;
        };
    }
    struct BoundaryOutputGeometry;
    struct FiniteVolume::Data
    {
        ssp<Geom::UnstructuredMesh> mesh;
        DomainSettings settings;
        std::shared_ptr<Runtime::CommunicationWorker> communicationWorker;
        EquationSystem equation;
        BoundaryPointOperator boundaryPoint;
        RigidMotion motion;
        Riemann::RiemannFlux flux;
        FiniteVolumeDetail::FluxKernelPolicy fluxPolicy;
        BoundarySystem boundary;
        CFV::FiniteVolumePlan metrics;
        /// Present only for the variational family. Constant/P0 and explicit
        /// physical-gradient P1 own no Taylor basis, VR matrices or modal plan.
        std::optional<CFV::VariationalReconstructionPlan> reconstruction;
        /// Optional projected operator; shares the finest native matrix with
        /// its provider. Durable coefficients still belong to solver state.
        std::optional<CFV::ReconstructionAMGPlan> reconstructionAMG;
        /// Frozen operator dispatch, resolved at placement. The schedule has
        /// no device branch and CFV owns all numerical kernel selection.
        CFV::ReconstructionDeviceKernel reconstructionKernel = CFV::ReconstructionDeviceKernel::FP64;
        CFV::GradientPlan gradient;
        Geom::EntityWorkPartition partition;
        FiniteVolumeStencilStorage stencil;
        // Coarse exact dispatch avoids growing the frequently traversed Sample
        // structure. One rotated image retains the general Jacobian path for
        // the whole plan; ordinary/translated meshes skip identity products.
        bool hasRotatedFrames = false;
        /// One oriented mean-geometry face per cell incidence. Approximate
        /// Jacobians never traverse the RHS quadrature samples above.
        std::shared_ptr<const LinearPatternStorage> jacobianColumns;
        std::vector<int> boundaryZones;
        std::vector<BoundaryOutputRow> boundaryOutputRows;
        mutable std::shared_ptr<const BoundaryOutputGeometry> boundaryOutputGeometry;
        std::shared_ptr<const BoundaryOutputGeometry> PrepareBoundaryOutput() const;
        bool hasPressureBoundary = false;
        std::vector<int> forceZones;
        std::map<int, index> forceSlots;
        std::shared_ptr<Detail::BoundaryPointLayout> boundaryLayout;
        /// Mesh-sized inputs declared by the boundary operator; independent of
        /// the host setup vectors used by the still-pending RHS extraction.
        Runtime::Array<BoundaryFaceRange> boundaryFaces;
        /// Required RHS/VR packet section. Direct midpoint consumers share
        /// center values; no unused quadrature payload is allocated for them.
        index BoundaryValueRows() const
        {
            return stencil.quadratureBoundaryValues ? boundaryLayout->quadratureRows : boundaryFaces.RowCount();
        }
        std::vector<BoundaryCondition> boundaryConditions;
        /// Small immutable backend metadata. Numeric boundary kernels consume
        /// these resolved records, never the host condition owners/callbacks.
        /// Empty on the CPU plan; CUDA placement resolves and uploads it once.
        Runtime::MetadataBuffer<PerfectGasBoundaryParameters> deviceBoundaryConditions;
        Runtime::MetadataBuffer<ACM::BoundaryParameters> deviceACMBoundaryConditions;
        Runtime::MetadataBuffer<ACM::SpatialParameters> deviceACMPhysics;
        Runtime::MetadataBuffer<PerfectGasSpatialParameters> devicePhysics;
        Runtime::MetadataBuffer<BoundaryKind> deviceBoundaryKinds;
        Runtime::MetadataBuffer<CUDA::BoundaryKernelParameters> deviceBoundaryPolicy;
        /// Resident zero-row operand for absent optional fields. It is never
        /// a fallback host pointer or a hidden allocation during submission.
        Runtime::Array<real> deviceEmpty;
        /// Domain-wide coupling participation. Every rank uses this to retain
        /// the collective interface publication protocol, including ranks
        /// with no locally owned interface quadrature.
        bool hasInterfaces = false;
        /// Whether this rank's RHS actually consumes an incoming interface
        /// row. A published zero-row packet is valid and must not be mistaken
        /// for a missing numeric payload on interior-only partitions.
        bool hasLocalInterfaceRows = false;
        // Unique faces required by each cell-output region. A face shared
        // across the regions may occur in both; each region remains callable
        // independently, without a hidden prior-core publication contract.
        Detail::FiniteVolumeQuadrature quadrature;
        /// Setup extent for CPU row-local PP scratch; no device readback is
        /// needed when constructing a placed workspace.
        index maximumPositivityPoints = 0;
        index maximumGradientPoints = 0;
        std::shared_ptr<const LimitedVRGeometry> limitedVRGeometry;
        /// Small fixed-flow leaves retain their five-component affine bounds;
        /// orchestration and complete equation checks use defaultAdmissibility.
        std::optional<ThermalBounds> fixedFlowThermalBounds;
        std::optional<AdmissibilityPolicy> defaultAdmissibility;
        std::optional<WallDistanceSamples> wallDistance;
        Runtime::Array<real> ransCellLength;
        /// CPU-only immutable preparation; called before a spatial placement.
        void BuildSourceQuadrature();
        /// Build compact canonical face routing and incidence matrix slots.
        /// No expanded sample/rotation/BC geometry is retained or published.
        void BuildStencil();
        /// Allocate/fill PP query maps, borrowing face/source caches without
        /// changing the canonical per-cell point order or sampling rules.
        void BuildPositivityQuadrature(bool limitedVR);
        Runtime::TransferBinding BindHalo(CFV::FVField &field, const char *channel = "cauchy-field") const
        {
            return metrics.BindHalo(field, HaloTransferOptions(channel));
        }
        Algebra::DofHalo<> BindTrackedHalo(CFV::FVField &field, const char *channel = "cauchy-mean") const
        {
            return metrics.BindTrackedHalo(field, HaloTransferOptions(channel));
        }
        Runtime::TransferOptions HaloTransferOptions(const char *channel) const
        {
            Runtime::TransferOptions options;
            options.communicationWorker = communicationWorker;
            options.channel = channel;
            DNDS_check_throw_info(settings.execution.transferPreflight == "collective" || settings.execution.transferPreflight == "abortOnFailure", "invalid Cauchy transfer preflight policy");
            if (!settings.implicit.collectiveChecks || settings.execution.transferPreflight == "abortOnFailure")
                options.preflight = Runtime::TransferPreflight::AbortOnFailure;
            return options;
        }
        static CFV::FiniteVolumeMetricOptions MetricOptions(const DomainSettings &settings,
                                                            Diagnostics::SetupProgress progress)
        {
            CFV::FiniteVolumeMetricOptions options;
            options.integrationOrder = settings.numerics.cfv.intOrder;
            options.faceIntegrationOrder = settings.numerics.RHSIntegrationOrder();
            options.collectivePreflight = settings.implicit.collectiveChecks;
            options.progress = std::move(progress);
            return options;
        }
        static std::vector<int> GradientExtrapolatedZones(const DomainSettings &settings)
        {
            std::vector<int> zones;
            if (settings.numerics.gradient == "leastSquares")
                for (const auto &[zone, condition] : settings.boundaries)
                    if (condition.kind == "extrapolate")
                        zones.push_back(std::stoi(zone));
            return zones;
        }
        const ThermalBounds &FixedFlowThermalBounds() const
        {
            DNDS_check_throw_info(
                fixedFlowThermalBounds,
                "fixed-flow thermal policy is unavailable for this equation");
            return *fixedFlowThermalBounds;
        }
        Data(ssp<Geom::UnstructuredMesh> source, DomainSettings controls, EquationSystem physics,
             Diagnostics::SetupProgress progress)
            : mesh(std::move(source)), settings(std::move(controls)), equation(std::move(physics)), boundaryPoint(equation), motion(settings.motion, settings.mesh.dimension), flux(Riemann::RiemannFlux::Resolve(settings.numerics.flux)),
              fluxPolicy(FiniteVolumeDetail::ResolveFluxKernelPolicy(settings, flux)),
              boundary(settings.boundaries, equation), metrics(CFV::FiniteVolumePlan::ConstructMetrics(mesh, MetricOptions(settings, progress))),
              // Only the variational family owns Taylor bases, reconstruction
              // matrices and modal coefficient storage. Direct P0/P1 consume
              // means or physical gradients without a compatibility plan.
              reconstruction(settings.numerics.reconstruction == "variational"
                                 ? std::make_optional(CFV::VariationalReconstructionPlan::ConstructRecCoeff(
                                       metrics, settings.numerics.cfv,
                                       FiniteVolumeDetail::ReconstructionSupport(settings), progress))
                                 : std::nullopt),
              gradient(CFV::GradientPlan::Prepare(*mesh, metrics, settings.numerics.gradient == "greenGauss" ? CFV::GradientMethod::GreenGauss : settings.numerics.gradient == "ggmp" ? CFV::GradientMethod::GGMP
                                                                                                                                                                                      : CFV::GradientMethod::LeastSquares,
                                                  1e-13, progress, settings.numerics.leastSquares, GradientExtrapolatedZones(settings))),
              partition(metrics.CellWorkPartition())
        {
            if (settings.execution.communicationExecution == "worker")
                communicationWorker = std::make_shared<Runtime::CommunicationWorker>();
            if (!equation.HasMixture())
            {
                if (!equation.IsACM())
                    fixedFlowThermalBounds = equation.PrepareThermalBounds(
                        settings.numerics.densityFloor,
                        settings.numerics.temperatureFloor,
                        settings.numerics.pressureFloor);
                defaultAdmissibility.emplace(
                    equation, settings.numerics.densityFloor,
                    settings.numerics.temperatureFloor,
                    settings.numerics.sensibleEnergyFloor,
                    settings.numerics.pressureFloor,
                    equation.HasRANS()
                        ? RANSStateView{equation.RANS().Kind(),
                                        equation.RANS().ComponentNumber()}
                        : RANSStateView{});
            }
        }
    };
    /// Private owned-quadrature PP composition. Physical coefficient fields
    /// keep their layouts; numeric policy, samples and scratch are independent
    /// of nonlinear/time ownership. No communication is hidden in Apply.
    struct FiniteVolume::PositivityWorkspace
    {
        struct State;
        std::shared_ptr<State> state;
        PositivityWorkspace(std::shared_ptr<const Data> owner, const EquationDofs &prototype,
                            std::optional<Physics::Chemistry::ChemicalWorkerPool> workers);
        void Apply(const EquationDofs &input, const EquationDofs &unlimited, EquationDofs &flux, bool refresh);
        /// Common queued CPU/CUDA producer. Sampling, degree selection and
        /// bounded recovery stay cell-local. No controller drain or global
        /// retry decision occurs here; LocalStatistics observes the compact
        /// status/count/minimum only after the enclosing reporting drain.
        void Apply(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                   const EquationDofs &input, const EquationDofs &unlimited, EquationDofs &flux, bool refresh);
        std::pair<index, real> LocalStatistics() const;
    };
}
