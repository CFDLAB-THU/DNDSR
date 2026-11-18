#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/State/DerivedFields.hpp"
#include "DNDS/Core/OptionalRef.hpp"
#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Cauchy/Boundary/BoundaryInputs.hpp"
#include "Cauchy/IO/PreparedVisualization.hpp"
#include "Cauchy/Implicit/LinearProblem.hpp"
#include "CFV/Plans/GradientPlan.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Geom
{
    class BoundaryPatch;
}
#include "Geom/Access/EntityWorkPartition.hpp"
#include "Geom/Access/MeshAccess.hpp"
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace DNDS::Cauchy
{
    class DomainExecutionContext;
    struct QueuedEvaluationDependency;
    class InterfaceCoupling;
    class InterfaceDonorPublication;
    namespace EquationFVDetail
    {
        struct MeanLinearizationOutput;
    }
    struct ThermalView;
    class AdmissibilityPolicy;
    /// Persistent buffers/plans for explicit mean-gradient preparation. One
    /// field per physical equation allocation, columns component*dim+direction.
    /// Gradient values use the cell's mesh frame; periodic tensors are rotated
    /// at stencil consumption. Copies share buffers and must not execute at
    /// the same time. Build for the exact state binding used by the caller.
    struct MeanGradients
    {
        std::vector<CFV::FVField> fields;

    private:
        friend class FiniteVolume;
        struct Data;
        std::shared_ptr<Data> _data;
    };
    /// Optional prepared inputs to the stateless auxiliary property kernels.
    /// Two-equation RANS requires meanGradients. No implicit gradient solve or
    /// exchange is performed by BeginAuxValues/AuxiliaryPreparation.
    struct AuxiliaryInputs
    {
        OptionalRef<const MeanGradients> meanGradients;
        OptionalRef<const BoundaryInputs> boundary;
        /// Reserve outputs before a queued interface packet is ready. Only
        /// ApplyInterfaceIndependent may run until BindBoundary supplies it.
        bool deferBoundary = false;
    };
    /// Explicit result of physical-step selection. Alpha/count describe the PP
    /// bound before any independent growth cap or subsequent retry halving.
    struct PhysicalStepLimit
    {
        real dt = 0;
        index limitedCells = 0;
        real minimumAlpha = 1;
        bool growthLimited = false;
    };
    struct AuxiliaryRequest
    {
        bool meanThermodynamics = false;
        bool faceSpectralRadius = false;
        bool faceWaveJump = false;
        bool facePressureMinimum = false;
        bool cellSpectralSum = false;
        bool cellWaveJump = false;
        bool cellPressureMinimum = false;
        DerivedRequest derived;
    };
    /// Explicit, extensible output buffers. Every member is optional; absent
    /// outputs are neither allocated nor written. Face rows follow AuxiliaryFaces().
    /// Cell reductions write owned rows only; their halo exchange is a caller task.
    struct AuxiliaryValues
    {
        /// Local owned+ghost rows: [p,a,diffusivity], followed by [H,beta] for
        /// mixtures. Ghost properties are evaluated from the fresh mean halo;
        /// this rank-local cache has no independent transfer plan.
        std::optional<Runtime::Array<real>> meanThermodynamics;
        std::optional<Runtime::Array<real>> faceSpectralRadius; // convection, diffusion (speed)
        std::optional<Runtime::Array<real>> faceWaveJump;
        std::optional<Runtime::Array<real>> facePressureMinimum;
        std::optional<CFV::FVField> cellSpectralSum; // sum A*(lambda_c+lambda_v)
        std::optional<CFV::FVField> cellWaveJump;
        std::optional<CFV::FVField> cellPressureMinimum;
        std::optional<DerivedFields> derivedOwned, derivedGhost;

    private:
        friend class FiniteVolume;
        bool _derivedThermodynamics = false;
        struct EquationOrigin;
        std::shared_ptr<const EquationOrigin> _equationOrigin;
        // Only needed when a cell reduction is requested without its matching
        // public face output. Columns: spectral sum, wave jump, pressure min.
        std::optional<Runtime::Array<real>> _faceScratch;
        struct QueuedStorage;
        /// Persistent mandatory status/empty operands for queued preparation.
        /// Shared aliases are guarded by an exclusive preparation session.
        std::shared_ptr<QueuedStorage> _queuedStorage;
    };
    /// Named phase selection for one prepared finite-volume execution.
    enum class FiniteVolumeEvaluationMode
    {
        SpatialResidual,
        ReconstructionOnly,
        SamplingInputs
    };
    /// @brief Explicit finite-volume phase request used by solver orchestration.
    ///
    /// `SpatialResidual` replaces the supplied residual and may publish named
    /// auxiliary buffers. `ReconstructionOnly` performs the same mean/gradient/
    /// limiter preparation without flux/source evaluation. `SamplingInputs`
    /// refreshes only state/finalized-reconstruction halos for interface
    /// sampling. The request owns no arrays; its OptionalRef operands are valid
    /// only for this synchronous call. DomainExecutionContext retains backend
    /// resources and every queued operation retains its numerical owners.
    struct FiniteVolumeEvaluation
    {
        FiniteVolumeEvaluationMode mode = FiniteVolumeEvaluationMode::SpatialResidual;
        OptionalRef<EquationDofs> residual;
        OptionalRef<AuxiliaryValues> auxiliary;
        real time = 0;

        /// Optional owned-cell record paired with this spatial rate.
        OptionalRef<Runtime::Array<real>> sourceDerivative;

        static FiniteVolumeEvaluation Spatial(EquationDofs &output, real time = 0,
                                              OptionalRef<AuxiliaryValues> auxiliary = {}, OptionalRef<Runtime::Array<real>> derivative = {})
        {
            return {FiniteVolumeEvaluationMode::SpatialResidual,
                    OptionalRef<EquationDofs>(output), auxiliary, time, derivative};
        }
        static FiniteVolumeEvaluation Reconstruction(real time = 0)
        {
            return {FiniteVolumeEvaluationMode::ReconstructionOnly, {}, {}, time};
        }
        static FiniteVolumeEvaluation Sampling()
        {
            return {FiniteVolumeEvaluationMode::SamplingInputs, {}, {}, 0};
        }
    };
    /// Cell-centered discretization with explicit backend placement. Preparation retains immutable geometry,
    /// incidence, quadrature and polynomial data. Each simultaneous residual
    /// requires a distinct Workspace. No solver clock/history/IO policy here.
    class FiniteVolume
    {
        friend class InterfaceDonorPublication;
        struct Data;
        struct LimitedVRGeometry;
        struct LimitedVRWorkspace;
        struct MatrixFreeJacobian;
        struct JacobianData;
        struct PositivityWorkspace;
        std::shared_ptr<const Data> _data;
        explicit FiniteVolume(std::shared_ptr<const Data> data) : _data(std::move(data)) {}
        void ValidateBoundaryLayout(const BoundaryInputs &input) const;
        enum class BoundaryTarget
        {
            Quadrature,
            Gradient,
            PrimitiveGradient
        };
        struct BoundarySubmission
        {
            BoundaryTarget target = BoundaryTarget::Quadrature;
            bool exterior = false, repairRANS = true;
            real time = 0;
            /// Diagnostic surface sampling always permits the existing
            /// mean-state fallback, independently of production PP switches.
            bool forcePointFallback = false;
        };
        /// Queued boundary producer used by spatial/mean-gradient operators.
        /// Retains geometry and every operand through completion. Status is
        /// overwritten per output row; the enclosing operator must accumulate
        /// failures before reusing it and inspect them at its reporting drain.
        /// No chemistry, MPI, payload allocation, host observation or hidden fallback.
        void SubmitBoundaryValues(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                  const EquationDofs &, OptionalRef<const EquationDofs> coefficients,
                                  Runtime::Array<real> &output, Runtime::Array<NumericalStatus> &status,
                                  BoundarySubmission, OptionalRef<const Runtime::Array<real>> incoming = {}) const;

    public:
        enum class RHSKernelPolicy
        {
            Auto,
            Generic,
            FixedFlow5,
            Profiled
        };
        struct WorkspaceOptions
        {
            /// Resolve once during workspace construction. Generic is the
            /// layout/physics-general CPU oracle. Profiled exercises the generated CPU
            /// model/shape catalog, including Flow5. Auto retains the qualified
            /// fixed Flow5 leaf where the borrowed RHS requires identical arithmetic
            /// and uses the catalog elsewhere. FixedFlow5 requires one
            /// perfect-gas flow5 field and never changes the outer schedule.
            RHSKernelPolicy rhsKernel;
            bool captureBoundaryFlux;
            constexpr explicit WorkspaceOptions(RHSKernelPolicy policy = RHSKernelPolicy::Auto, bool capture = false)
                : rhsKernel(policy), captureBoundaryFlux(capture) {}
        };
        struct Workspace;
        struct MeanWorkspace;
        struct ReconstructionState;
        /// Local structural report for allocation/transfer acceptance gates.
        /// Counts describe modal reconstruction only; the independent mean
        /// halo belongs to the equation state and is intentionally excluded.
        struct ReconstructionStorageInfo
        {
            bool polynomialPlan = false;
            bool distinctFinalized = false;
            index coefficientFields = 0;
            index coefficientTransferBindings = 0;
            /// Actual retained value allocations. Structural layout metadata is
            /// shared and intentionally excluded. Ghost bytes equal one local
            /// receive payload across all coefficient halo bindings.
            size_t coefficientOwnedBytes = 0;
            size_t coefficientGhostBytes = 0;
        };
        /// Managed modal publication consumed by flux/source/Jacobian operators.
        /// Retains warm-source and selected polynomial handles and versions.
        /// Direct P0/P1 has no such publication: means and physical gradients
        /// are explicit, separate operands of those operators.
        class FinalizedReconstruction
        {
            friend class FiniteVolume;
            std::shared_ptr<const Data> _owner;
            EquationDofs _source, _values;
            std::vector<uint64_t> _sourceVersions, _valueVersions;
            bool _distinct = false;
            FinalizedReconstruction(std::shared_ptr<const Data> owner,
                                    const EquationDofs &source,
                                    const EquationDofs &values, bool distinct);

        public:
            /// Validate retained storage identity/version without MPI or access.
            void Validate() const;
            /// Selected finalized values. Validation is performed once here;
            /// the caller then acquires ordinary read leases for execution.
            const EquationDofs &Values() const;
            bool UsesDistinctStorage() const { return _distinct; }
        };
        /// Synchronous managed borrow of one durable reconstruction publication
        /// and the scratch that sampled it. Split flux/source/Jacobian calls
        /// must name both; Workspace never retains the coefficient allocation.
        struct JacobianContext
        {
            std::reference_wrapper<Workspace> workspace;
            OptionalRef<const ReconstructionState> reconstruction;
            /// Optional retained execution lane for this synchronous operator.
            /// CUDA requires it; the context owner manages its engine/resources.
            std::optional<Runtime::ExecutionStream> stream;
            /// Unscaled integrated source contribution returned alongside the correction rate.
            OptionalRef<const Runtime::Array<real>> sourceDerivative;
            explicit JacobianContext(Workspace &scratch) : workspace(scratch) {}
            JacobianContext(const ReconstructionState &state, Workspace &scratch)
                : workspace(scratch), reconstruction(state) {}
            JacobianContext(const ReconstructionState &state, Workspace &scratch, Runtime::ExecutionStream lane)
                : workspace(scratch), reconstruction(state), stream(std::move(lane)) {}
        };
        /// Reserved synchronous warm-guess publication. Construction validates
        /// every input and retains its reads, destination writes and workspace.
        /// Commit performs only prevalidated copies; abandonment cancels the
        /// untouched writers. Caller serializes workspace use until retirement.
        /// This permits several domains to reserve before any one publishes.
        class ReconstructionRestore
        {
            friend class FiniteVolume;
            std::function<void()> _commit;
            explicit ReconstructionRestore(std::function<void()> commit) : _commit(std::move(commit)) {}

        public:
            ReconstructionRestore(ReconstructionRestore &&other) noexcept { _commit.swap(other._commit); }
            ReconstructionRestore &operator=(ReconstructionRestore &&other) noexcept
            {
                if (this != &other)
                {
                    _commit = nullptr;
                    _commit.swap(other._commit);
                }
                return *this;
            }
            ReconstructionRestore(const ReconstructionRestore &) = delete;
            ReconstructionRestore &operator=(const ReconstructionRestore &) = delete;
            void Commit()
            {
                DNDS_check_throw_info(bool(_commit), "reconstruction restore was already committed or moved");
                auto commit = std::move(_commit);
                _commit = nullptr;
                commit();
            }
        };
        /// Reservation validates inputs (including read-only backend kernels)
        /// and acquires access without changing destinations. With
        /// collectiveChecks=false it performs no MPI; the caller must agree
        /// all reservations before any Commit.
        /// Simulation uses that mode for one cross-domain publication preflight.
        ReconstructionRestore ReserveReconstructionRestore(std::shared_ptr<ReconstructionState> state, const EquationDofs &guess,
                                                           OptionalRef<const Runtime::Array<real>> limitedVRHistory = {},
                                                           bool collectiveChecks = true,
                                                           std::optional<Runtime::ExecutionStream> stream = {}) const;
        /// Allocate durable variational coefficients and their communication
        /// bindings independently of reusable scratch. Direct P0/P1 allocate
        /// neither: their modal slots stay empty and physical gradients live
        /// separately in Workspace. Mean halo bindings remain available.
        std::shared_ptr<ReconstructionState> BuildReconstructionState(EquationDofs &state) const;
        /// Inspect one compatible reconstruction state without acquiring array
        /// access or entering MPI. P0/P1 must report all modal counts as zero.
        ReconstructionStorageInfo InspectReconstructionStorage(const ReconstructionState &state) const;
        /// Full logical-state CPU workspace on the same geometry/CFV plans.
        /// Each physical field keeps its layout and shared row routing; no
        /// whole-state packing is performed. Matching schema/physical grouping
        /// is collective; payload layouts may differ across ranks. A mixture
        /// requires a matching explicit worker pool, shared in capacity only.
        /// Initial equation-state profile: Rusanov, constant/gradient/VR,
        /// componentwise WBAP, Euler quadrature PP, polynomial Newtonian/Fourier
        /// diffusion and optional mixture chemistry. Extrapolate, farfield,
        /// constant prescribed and adiabatic/isothermal wall producers use the
        /// actual EOS. Operator-local fixed-flow leaves retain established
        /// flow5 arithmetic without a second workspace or execution loop.
        /// Allocate an owned, execution-resident derivative record for an RHS
        /// evaluation whose matching correction uses rhsQuadrature. Each live
        /// evaluation requiring independent storage must reserve its own record.
        std::optional<Runtime::Array<real>> BuildSourceDerivative(const EquationDofs &state) const;
        std::shared_ptr<Workspace> BuildWorkspace(EquationDofs &state,
                                                  std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                                                  WorkspaceOptions options = WorkspaceOptions()) const;
        /// Freeze one donor publication after predictor halos or complete
        /// reconstruction. `prepared=false` deliberately selects P0 for direct
        /// P1, breaking the initial interface/reconstruction dependency cycle;
        /// variational mode uses its warm guess as the historical predictor.
        InterfaceDonorPublication PublishInterfaceDonor(
            const EquationDofs &state,
            const ReconstructionState &reconstruction,
            const Workspace &workspace, bool prepared) const;
        /// Checkpoint-restore adapter for an independently staged variational
        /// warm guess. This preserves two-phase restart commit: interface
        /// sampling may validate the candidate without mutating the live
        /// ReconstructionState. Ordinary execution uses PublishInterfaceDonor.
        InterfaceDonorPublication PublishVariationalInterfaceDonor(
            const EquationDofs &state,
            const EquationDofs &coefficients) const;
        /// Separate typed scratch for Euler's zero-reconstruction physical-dt
        /// trial. Shares geometry and kernels, never the ordinary warm guess.
        /// Includes configured volume/rotation sources, ignores viscous faces.
        /// RANS sources still require their actual molecular properties.
        std::shared_ptr<MeanWorkspace> BuildMeanWorkspace(EquationDofs &state,
                                                          std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt) const;
        /// Local preflight of owner and bound state allocations. No numerical
        /// access, property refresh or communication occurs here.
        void ValidateWorkspace(const EquationDofs &state, const Workspace &workspace) const;
        /// Whether the prepared Jacobian operator set contains the established
        /// fixed-flow Roe leaf. This is an operator capability query, not a
        /// selection between solver orchestration paths.
        bool HasRoeJacobianKernel(const Workspace &workspace) const;
        /// Integrated residual for every logical equation block. CPU controller
        /// entry is synchronous; mean/coefficient transfers overlap owned-only
        /// core work when configured. Caller serializes workspace/state handles
        /// and coordinates collective call order. Failed scratch is discarded.
        /// This updates state ghosts and reconstruction scratch, never owned
        /// means. Each output physical allocation must be distinct from inputs.
        /// implicit.collectiveChecks=false omits optional stage/launch safety
        /// agreements and aborts on local rejection; matching call order stays
        /// the caller's responsibility. Required setup protocols and numerical
        /// reductions remain. Access checking is an independent engine choice.
        void Evaluate(EquationDofs &state, EquationDofs &residual, ReconstructionState &reconstruction, Workspace &workspace,
                      Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time = 0, OptionalRef<Runtime::Array<real>> sourceDerivative = {}) const;
        /// Prepare the requested auxiliary outputs during the mean exchange,
        /// before reconstruction/RHS. RHS reads this explicit publication;
        /// the caller may then consume its spectra for pseudo-time/Jacobian
        /// setup without recomputing and overwriting RHS-refined signals.
        /// Buffers must be distinct from state/residual and retained through
        /// this synchronous call. The no-aux overload retains its old scope.
        void Evaluate(EquationDofs &state, EquationDofs &residual, ReconstructionState &reconstruction, Workspace &workspace,
                      AuxiliaryValues &auxiliary, Runtime::EagerEngine &engine,
                      const Runtime::ExecutionStream &stream, real time = 0, OptionalRef<Runtime::Array<real>> sourceDerivative = {}) const;
        /// The same mean/gradient/reconstruction/limiter preparation as Evaluate,
        /// including completed halos, but no flux/source evaluation or dummy
        /// residual allocation. Use before independent EvaluateFlux/AddSource.
        void PrepareReconstruction(EquationDofs &state, ReconstructionState &reconstruction, Workspace &workspace, Runtime::EagerEngine &engine,
                                   const Runtime::ExecutionStream &stream, real time = 0) const;
        void RefreshSamplingInputs(EquationDofs &state, ReconstructionState &reconstruction, Workspace &workspace, Runtime::EagerEngine &engine,
                                   const Runtime::ExecutionStream &compute,
                                   const Runtime::ExecutionStream &communication) const;
        /// @brief Canonical solver-facing FV phase entry.
        /// @pre `context` owns the primary stream and outlives this call.
        /// @pre Request operands are compatible, nonaliased and retained until
        ///      synchronous completion.
        /// @post Spatial residual/reconstruction/sampling effects match the
        ///       corresponding compatibility entry above; no extra collective
        ///       or backend-dependent orchestration branch is introduced.
        void Execute(EquationDofs &state, ReconstructionState &reconstruction,
                     Workspace &workspace, DomainExecutionContext &context,
                     FiniteVolumeEvaluation evaluation) const;
        /// Same synchronous collective/access contract as Evaluate. Only this
        /// trial's private scratch and distinct residual are written; state
        /// owned values and ordinary reconstruction are unchanged.
        /// This zero-polynomial physical-dt probe retains parent quadrature,
        /// as in Euler; it is not a separately configured P0 FV operator.
        void EvaluateMean(EquationDofs &state, EquationDofs &residual, MeanWorkspace &workspace,
                          Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time = 0) const;
        /// Borrow modal warm coefficients; direct P0/P1 has none and rejects.
        /// Direct means and physical gradients are separate from modal storage.
        const EquationDofs &ReconstructionGuess(const ReconstructionState &state) const;
        /// Collective transactional owned warm-guess restore: finite-value and
        /// lease preflight precedes every write. No halo transfer/mean mutation.
        /// Caller joins workspace work and supplies a distinct snapshot.
        /// Requires a polynomial plan; direct P0/P1 rejects before any write.
        /// CUDA requires an explicit stream. With a stream, inline/worker CPU
        /// and CUDA share reserved submission and synchronous completion. No
        /// payload staging occurs; source buffers must already match placement.
        void RestoreReconstructionGuess(ReconstructionState &state, const EquationDofs &guess,
                                        OptionalRef<const Runtime::Array<real>> limitedVRHistory = {},
                                        std::optional<Runtime::ExecutionStream> stream = {}) const;
        /// Copy one complete durable polynomial publication, including the
        /// finalized limiter/PP coefficients and LVR history. Used when a new
        /// DITR midpoint starts from the accepted endpoint publication while
        /// both stages continue to share scratch.
        void CopyReconstructionState(ReconstructionState &destination,
                                     const ReconstructionState &source,
                                     std::optional<Runtime::ExecutionStream> stream = {}) const;
        /// Exact checkpoint restore of both warm and finalized coefficient
        /// publications. Unlike RestoreReconstructionGuess this never infers or
        /// recomputes limiter/PP output from the warm state.
        void RestoreReconstructionState(ReconstructionState &state,
                                        const EquationDofs &warm, const EquationDofs &finalized,
                                        OptionalRef<const Runtime::Array<real>> limitedVRHistory = {},
                                        std::optional<Runtime::ExecutionStream> stream = {}) const;
        std::optional<Runtime::Array<real>> LimitedVRHistory(const ReconstructionState &state) const;
        std::optional<Runtime::Array<real>> LimitedVRAlpha(const Workspace &workspace) const;
        /// Local reset; caller coordinates failures and joins workspace work.
        void ResetReconstruction(ReconstructionState &state, Workspace &workspace,
                                 std::optional<Runtime::ExecutionStream> stream = {}) const;
        /// Retain finalized modal coefficients. Direct P0/P1 rejects: its
        /// gradients are temporary fields, not modal reconstruction storage.
        FinalizedReconstruction AcquireFinalizedReconstruction(const ReconstructionState &state) const;
        /// Compatibility facade for existing synchronous checkpoint/output
        /// code. New split operators retain `FinalizedReconstruction` instead.
        const EquationDofs &FluxReconstruction(const ReconstructionState &state, const Workspace &workspace) const;
        /// Latest complete equation evaluation's global coefficient-update
        /// norm. No extra reduction occurs when reading these diagnostics.
        /// Fixed sweeps without a tolerance leave the update norm unmeasured
        /// (infinity); constant/gradient reconstruction reports zero.
        int LastReconstructionIterations(const Workspace &workspace) const;
        real LastReconstructionUpdate(const Workspace &workspace) const;
        /// Owned-only variational-polynomial projection. Direct-gradient PP is
        /// a Cauchy reconstruction stage performed by PrepareReconstruction or
        /// Evaluate after the physical gradient limiter. Standalone polynomial
        /// use starts from the unlimited guess; Evaluate preserves its preceding
        /// WBAP result. No halo transfer occurs here. Caller serializes use.
        void LimitReconstructionPP(const EquationDofs &state, ReconstructionState &reconstruction, Workspace &workspace) const;
        /// Collective count/min snapshot; no implicit refresh or mutation.
        std::pair<index, real> PositivityReconstructionStats(const Workspace &workspace) const;
        /// Whole-equation auxiliary transaction. Holds numeric input scopes
        /// and reserves optional output arrays before writing.
        /// Core reads owned inputs only; complete every input field's halo
        /// before HaloDependent. All runs those phases consecutively. No MPI
        /// or hidden state update occurs here. Finish publishes exact source,
        /// output and time provenance. Do not replace buffers during the scope.
        /// Keep FV/derived-bundle owners on the controller through worker joins;
        /// retained preparation plans may own MPI resources indirectly.
        class AuxiliaryPreparation
        {
            struct State;
            struct QueuedState;
            std::shared_ptr<State> _state;
            std::shared_ptr<QueuedState> _queued;
            explicit AuxiliaryPreparation(std::unique_ptr<State> state);
            explicit AuxiliaryPreparation(std::shared_ptr<QueuedState> state);
            static AuxiliaryPreparation BuildState(
                const FiniteVolume &plan, const EquationDofs &state,
                const AuxiliaryValues &output, real time,
                std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                AuxiliaryInputs inputs);
            static AuxiliaryPreparation BuildQueuedState(const FiniteVolume &, const EquationDofs &,
                                                         AuxiliaryValues &, real, AuxiliaryInputs);
            void BindQueuedBoundary(const BoundaryInputs &);
            void FinishQueued(AuxiliaryValues &);
            void ApplyQueued(Runtime::EagerEngine &, const Runtime::ExecutionStream &, CFV::ReconstructionRegion);
            void ApplyIndependentQueued(Runtime::EagerEngine &, const Runtime::ExecutionStream &);
            friend class FiniteVolume;

        public:
            AuxiliaryPreparation(const AuxiliaryPreparation &) = delete;
            AuxiliaryPreparation &operator=(const AuxiliaryPreparation &) = delete;
            AuxiliaryPreparation(AuxiliaryPreparation &&) noexcept;
            AuxiliaryPreparation &operator=(AuxiliaryPreparation &&) noexcept;
            ~AuxiliaryPreparation();
            /// Owned means and the prepared MPI-local, interface-independent
            /// auxiliary face/cell subsets. Requires deferBoundary at setup.
            void ApplyInterfaceIndependent();
            /// Bind a known packet before submitting boundary-consuming work.
            /// Synchronous execution acquires packet reads; queued execution
            /// retains handles and validates publication after caller ordering.
            /// CPU queued execution also accepts the stable PendingInputs of
            /// an interface round: its event and publication callback must
            /// precede consumption. It cannot bind unknown future allocations.
            /// No MPI or hidden wait is performed.
            void BindBoundary(const BoundaryInputs &input);
            void Apply(CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All);
            /// Enqueue the same core/dependent protocol. Both CPU and CUDA use
            /// retained Array operands; CUDA leaves status/results on device. No wait,
            /// transfer or MPI is hidden here. Finish is called only after the
            /// caller's segment drain, before externally publishing the bundle.
            void Apply(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                       CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All);
            void ApplyInterfaceIndependent(Runtime::EagerEngine &, const Runtime::ExecutionStream &);
            void Finish(AuxiliaryValues &output);
        };
        /// Property preparation with explicit CPU mixture workers. A resident
        /// CUDA plan returns a queued scope; use its engine/stream Apply overload.
        /// The caller owns collective order/error handling and halo completion;
        /// even an empty rank uses the same outer schedule. Boundary producers
        /// match Workspace. Two-equation closures require a current
        /// explicit mean-gradient publication; neither overload exchanges data.
        /// Begin retains handles/immutable geometry only. Mutable access is
        /// acquired by the first Apply: immediate for the synchronous overload,
        /// scheduled for the engine overload. Do not mix these execution modes.
        AuxiliaryPreparation BeginAuxValues(const EquationDofs &state, AuxiliaryValues &output, real time = 0,
                                            std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                                            AuxiliaryInputs inputs = {}) const;
        void PrepareAuxValues(const EquationDofs &state, AuxiliaryValues &output, real time = 0,
                              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                              AuxiliaryInputs inputs = {}) const;
        /// Collective allocation/binding only, not a gradient evaluation. The
        /// returned workspace is tied to these physical allocations and row
        /// spaces; rebuild after rebinding/redistributing the equation state.
        MeanGradients BuildMeanGradients(EquationDofs &state) const;
        /// Collective backend-neutral orchestration: mean halo + gradient core, dependent
        /// gradients, then gradient halo + auxiliary core, and auxiliary rest.
        /// Uses persistent scratch and explicit workers. No VR coefficients are
        /// read or changed. CPU inline/worker and CUDA use this same submission
        /// order; kernel dispatch stays in the boundary/CFV/auxiliary operators.
        /// Release conflicting views first. All ranks enter with the same
        /// physical grouping; no simultaneous use of shared workspace copies.
        /// Returns with completed outputs (resident on their selected backend).
        /// Issued mutation epochs validate in-segment gradient provenance;
        /// publication waits for the final join. Mutating state or gradients makes
        /// a retained auxiliary publication stale and validation rejects it.
        void PrepareAuxValues(EquationDofs &state, AuxiliaryValues &output, MeanGradients &gradients,
                              Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time = 0,
                              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                              AuxiliaryInputs inputs = {}) const;
        void PrepareAuxValues(EquationDofs &state, AuxiliaryValues &output, MeanGradients &gradients,
                              Runtime::EagerEngine &engine, const Runtime::ExecutionStream &compute,
                              const Runtime::ExecutionStream &communication, real time,
                              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                              AuxiliaryInputs inputs) const;
        void ValidateAuxValues(const EquationDofs &state, const AuxiliaryValues &values, real time = 0) const;
        /// Rank-local cache selection: auxiliary values must have used the
        /// currently bound interface publication, even when means/time match.
        /// No refresh, lease or collective is performed. Coordinated callers
        /// publish packets on all domain ranks before choosing a collective
        /// preparation path; full payload validation remains separate.
        bool AuxiliaryBoundaryCurrent(const AuxiliaryValues &values,
                                      const Workspace &workspace) const;
        /// Copyable immutable read scope for RHS geometry/basis data. Apply is
        /// stateless: no MPI, reconstruction, cache update or hidden counters.
        /// Raw input/output views require retained/declared Array leases and
        /// nonoverlapping output storage supplied by the caller. Core input
        /// views need only owned data. The checked Evaluate wrapper does this.
        class RHSOperator
        {
            struct ReadData;
            enum class FailurePolicy
            {
                Throw,
                Abort
            };
            std::shared_ptr<const ReadData> _read;
            explicit RHSOperator(std::shared_ptr<const ReadData> read) : _read(std::move(read)) {}
            template <FailurePolicy policy>
            RHSKernelDiagnostics ApplyFluxImpl(const RHSReadViews &input, const RHSWriteViews &output,
                                               CFV::ReconstructionRegion region, real time) const;
            template <FailurePolicy policy>
            void AddSourceImpl(const RHSReadViews &input, const RHSWriteViews &output,
                               CFV::ReconstructionRegion region, real time) const;
            friend class FiniteVolume;

        public:
            /// Checked synchronous borrowed-view entry point. The caller keeps
            /// every view lease and owner alive through return. Validation
            /// failures throw before or during this standalone kernel call.
            RHSKernelDiagnostics ApplyBorrowedChecked(const RHSReadViews &input, const RHSWriteViews &output,
                                                      CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
            /// Resolved production-kernel entry point. Its caller has already
            /// validated and retained the operation scope. A failed invariant
            /// terminates this rank directly; it never unwinds a collectively
            /// ordered worker path or calls MPI from that worker.
            RHSKernelDiagnostics ApplyBorrowedFatal(const RHSReadViews &input, const RHSWriteViews &output,
                                                    CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
            /// Production flux-only leaf used by the unified FV orchestration;
            /// volume sources remain a separate equation operator. Kernel
            /// invariant and disabled-fallback failures abort directly.
            RHSKernelDiagnostics ApplyFluxBorrowedFatal(const RHSReadViews &input, const RHSWriteViews &output,
                                                        CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
            /// Overwrite selected residual rows with integrated face flux only.
            /// Optional faceFlux follows the documented canonical face order.
            RHSKernelDiagnostics ApplyFlux(const RHSReadViews &input, const RHSWriteViews &output,
                                           CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
            /// Add owned-cell volume sources (flow5: rigid-frame source) to
            /// initialized output rows. No face, reconstruction or MPI work.
            void AddSource(const RHSReadViews &input, const RHSWriteViews &output,
                           CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
        };
        RHSOperator AcquireRHSOperator() const;
        /// Caller-owned local stencil-face table; no halo. Components follow
        /// logical equation order for whole-equation calls, flow5 for RHSOperator.
        Runtime::Array<real> BuildFaceFluxBuffer(index components) const;
        /// Reuse completed equation PrepareReconstruction/Evaluate publication.
        /// No halo exchange/reconstruction is hidden here. Input halos and all
        /// conflicting views must be ready/released. Flux overwrites selected
        /// owned rows; source adds to initialized owned rows. Output is distinct
        /// from state and workspace arrays. A face buffer is optional and follows
        /// AuxiliaryFaces() with the same orientation as RHSWriteViews::faceFlux.
        /// Caller serializes workspace use and coordinates distributed failures.
        void EvaluateFlux(const EquationDofs &state, EquationDofs &residual,
                          const ReconstructionState &reconstruction, Workspace &workspace,
                          CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0,
                          OptionalRef<Runtime::Array<real>> faceFlux = {},
                          OptionalRef<const AuxiliaryValues> auxiliary = {}) const;
        void AddSource(const EquationDofs &state, EquationDofs &residual,
                       const ReconstructionState &reconstruction, Workspace &workspace, real time = 0) const;
        /// Allocate a fixed cell-face-cell matrix stencil using the prototype's
        /// native row space. No payload is copied from the prototype. Evaluated
        /// mode omits off-diagonal storage; PrepareJacobian supplies its frozen
        /// physical snapshot. Both modes retain explicit diagonal storage.
        std::shared_ptr<LinearProblemContext> BuildLinearProblem(const EquationDofs &prototype, DiagonalStorage diagonal,
                                                                 OffDiagonalStorage offDiagonal = OffDiagonalStorage::Assembled,
                                                                 LinearMatrixOptions storage = {}) const;
        /// Prepare integrated approximate-Jacobian coefficients from
        /// frozen means and prepared face signals. No input mutation, MPI or
        /// inverse/factor preparation occurs here. inversePseudoTime is an
        /// owned-cell scalar table. Caller completes all halos before entry.
        /// Semi-matrix-free mode copies frozen physical data, reuses immutable
        /// geometry, and regenerates blocks during action/SGS/ILU filling. It
        /// retains no live mean/auxiliary leases after preparation returns.
        /// Whole-equation CPU approximation with the same integrated/scalar
        /// dissipation convention. Uses actual conservative EOS derivatives,
        /// ALE/rotation terms, mean-state chemical source derivatives and scalar
        /// molecular/species diffusion bounds. This is not a
        /// derivative of the complete reconstructed viscous/source residual.
        /// Roe characteristics are not connected for this profile and reject.
        /// Frozen evaluated blocks retain numeric snapshots/geometry only;
        /// they perform no EOS calls, MPI or live-state/cache access.
        /// The caller completes halos, serializes matrix/input bindings and
        /// coordinates failures. No inverses/factors are updated implicitly.
        /// Output writers are reserved together; partial failure poisons them.
        /// RANS/source publication requires a JacobianContext containing the
        /// solver-owned reconstruction state and its scratch workspace from a
        /// successful Evaluate at this state/time. Values/coefficients
        /// must remain unchanged. RANS derivatives use its current center
        /// gradients even on P0; chemical derivatives use the cell mean.
        void PrepareJacobian(const EquationDofs &state, const AuxiliaryValues &auxiliary,
                             const Runtime::Array<real> &inversePseudoTime, LinearProblemContext &output,
                             JacobianPreparation controls = {}, real time = 0,
                             std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                             std::optional<JacobianContext> context = std::nullopt) const;
        FiniteVolume(ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const EquationSystem &equation,
                     Diagnostics::SetupProgress progress = {});
        /// Explicit synchronous setup placement of immutable spatial inputs.
        /// The returned handle uses the same FV API/row spaces; numerical
        /// kernels are still selected by each operator. Does not move means,
        /// reconstruction state or scratch, and does not enable an unavailable
        /// solver capability. Drop the source handle to release CPU numerical
        /// caches. Reference mesh/configuration and host setup metadata remain.
        FiniteVolume Placed(Runtime::Device device, Runtime::ExecutionContext context) const;
        Runtime::Device Device() const { return Metrics().Device(); }
        bool SharesPlan(const FiniteVolume &other) const { return _data && _data == other._data; }
        const CFV::FiniteVolumePlan &Metrics() const;
        /// Bind through the metric space with this domain's explicit MPI
        /// launch policy. Setup remains collective in either policy.
        Runtime::TransferBinding BindHalo(CFV::FVField &field) const;
        const EquationSystem &Equation() const;
        /// Immutable resolved spatial controls, including source-free settings
        /// for a split Domain. Custom controllers copy this for their own
        /// nonlinear/hierarchy controls; changing a copy cannot mutate this plan.
        const DomainSettings &Settings() const;
        /// Whether an active boundary on any domain rank fixes ACM pressure.
        bool HasPressureBoundary() const;
        /// Variational-only polynomial publication. P0 and explicit P1 return
        /// an empty borrow and construct no hidden Taylor/VR plan.
        OptionalRef<const CFV::VariationalReconstructionPlan> PolynomialReconstruction() const;
        const Geom::UnstructuredMesh &Mesh() const;
        /// Managed host geometry access for setup/reporting operators. Retains
        /// the mesh generation and requested arrays; no unmanaged borrow escapes.
        Geom::MeshReadAccess AcquireMeshRead(Geom::MeshReadRequirements requirements) const;
        const RigidMotion &Motion() const;
        std::vector<BoundaryPoint> BoundaryPoints() const;
        /// Materialize a caller-owned matching/debug quadrature snapshot.
        /// Ordinary boundary kernels read the compact shared face registry.
        Runtime::Array<BoundaryPoint>::ReadLease ReadBoundaryPoints() const;
        /// Maintained face-center rows, in faceFlow order; quadrature=-1.
        std::vector<BoundaryPoint> BoundaryFaces() const;
        /// Materialize a caller-owned canonical face-center snapshot.
        Runtime::Array<BoundaryPoint>::ReadLease ReadBoundaryFaces() const;
        /// Direct owned-boundary selection for a physical coupling adapter.
        /// The result retains the reference mesh without exposing a mutable
        /// owner handle through the solver's read-only mesh facade.
        Geom::BoundaryPatch BuildBoundaryPatch(int zone) const;
        bool HasInterfaces() const;
        BoundaryInputs BuildBoundaryInputs(
            std::optional<Physics::Chemistry::ChemicalWorkerPool> workers =
                std::nullopt,
            std::optional<Runtime::ExecutionStream> stream = std::nullopt) const;
        /// Deep-copy one published packet into independent immutable storage.
        /// Use this when several physical/coupled stages must coexist while the
        /// interface coordinator reuses its mutable incoming allocation.
        BoundaryInputs FreezeBoundaryInputs(const BoundaryInputs &input,
                                            std::optional<Runtime::ExecutionStream> stream = std::nullopt) const;
        /// Synchronous local point operations, no MPI. Begin withdraws the
        /// publication; publish checks complete admissible interface rows only.
        /// With a supplied lane, clearing is queued there and callers order
        /// subsequent writers on that lane. Without it, clearing is synchronous.
        void BeginBoundaryUpdate(BoundaryInputs &input,
                                 std::optional<Runtime::ExecutionStream> stream = std::nullopt) const;
        void PublishBoundaryInputs(
            BoundaryInputs &input, real time, StateEpoch epoch = {},
            std::vector<BoundaryInterfaceSource> sources = {},
            std::optional<Runtime::ExecutionStream> stream = std::nullopt) const;
        void ValidateBoundaryInputs(const BoundaryInputs &input, real time) const;
        /// Metadata/provenance check only: grants no payload access/readiness.
        /// Queued operators call this after the packet event under declared
        /// quadrature and face-center Array reads. Synchronous callers should
        /// use ValidateBoundaryInputs, which acquires immediate read scopes.
        void ValidateBoundaryPublication(const BoundaryInputs &input, real time) const;
        /// Share a packet with another FV level only after proving identical
        /// boundary point order/geometry. No interpolation or payload copy.
        BoundaryInputs AliasBoundaryInputs(const BoundaryInputs &input) const;
        /// Equation-orchestration entry. The packet is a complete canonical
        /// logical row regardless of physical field grouping; the FV operator
        /// selects its equation/EOS/RANS leaf only after binding that row.
        void SetBoundaryInputs(Workspace &workspace, const BoundaryInputs &input) const;
        void SetBoundaryInputs(MeanWorkspace &workspace, const BoundaryInputs &input) const;
        /// Allocate the requested members once. Cell reductions without matching
        /// face outputs also need private face scratch. Callers may instead
        /// supply compatible buffers themselves, including strided tables;
        /// BeginAuxValues lazily allocates their missing private scratch once.
        AuxiliaryValues BuildAuxiliaryValues(AuxiliaryRequest request) const;
        /// Add every auxiliary channel required by the configured RHS flux to
        /// a caller's explicit request. BuildAuxiliaryValues remains an exact
        /// allocator; RHS callers use this union so an independently requested
        /// spectral/dTau bundle cannot omit H-correction or pressure-stencil
        /// inputs consumed by the same numerical operator.
        AuxiliaryRequest CompleteRHSAuxiliaryRequest(AuxiliaryRequest request = {}) const;
        /// Setup/diagnostic copy of the face ordinals. Execution uses the
        /// retained stencil Array view, without copying this list.
        std::vector<index> AuxiliaryFaces() const;
        real StableStep(const EquationDofs &state, const AuxiliaryValues &values, real cfl, real time = 0) const;
        /// Interpretation of stored polynomial rows; independent of physical
        /// flow schema. Restart may restore means and cold-start incompatible
        /// coefficients rather than matching a C++ static row type.
        Json ReconstructionMetadata() const;
        /// Collective Euler compress=1 bound from dt*scale*meanResidual/volume.
        /// Inputs are owned-only CPU flow5 fields; state ghost writes may be
        /// active. No input mutation, time advance, growth cap or communication
        /// of payloads. A zero returned dt means no positive admissible bound.
        /// All ranks enter with matching finite positive dt/scale, relax in
        /// (0,1], and positive compressionPower. Mean residual must be ready
        /// and nonaliased with state. Uses the equation's thermal energy bounds.
        /// Whole-equation compress=1 bound with logical, strided owned inputs.
        /// Caller supplies the matching model's frozen thermal policy and
        /// retains its coefficient buffers. Species move with flow along the
        /// thermal line; RANS positivity adds no separate dt ratio. Inputs must
        /// be ready, distinct and admissible. No EOS, mutation or payload MPI.
        PhysicalStepLimit LimitPhysicalStep(const EquationDofs &state, const EquationDofs &meanResidual,
                                            const ThermalView &thermal, real dt, real relax, real scale, int compressionPower = 1) const;
        /// Initialize the logical flow block from a cell-averaged analytic
        /// profile. The current producer is available for pure perfect-gas
        /// flow5 states; extended states use InitializeConstant/InitializeField.
        void Initialize(EquationDofs &state, const FlowProfileSettings &profile, const FlowState &primitive) const;
        /// Fill exact code-unit conservative values in equation-schema order.
        /// Applies the initial rigid-frame rotation to flow momentum only.
        /// Caller resolves/validates physical input with ResolvePhysicalState.
        void InitializeConservative(EquationDofs &state, const std::vector<real> &values) const;
        /// Collective error-coordinated constant initialization in ground axes.
        /// Uses actual EOS/full thermochemical energy, resolves named species
        /// once, and writes logical blocks through their native layout. Positive
        /// temperature overrides primitive p. No ghost exchange or reconstruction
        /// initialization; caller releases conflicting views/async work first.
        /// Mixtures require an explicit matching worker pool. RANS requires
        /// every named conservative scalar; all inputs are validated before
        /// writing any block. Does not infer turbulence values from flow.
        void InitializeConstant(EquationDofs &state, const FlowState &primitive, const std::map<std::string, real> &massFractions = {},
                                real temperature = 0, std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                                const std::map<std::string, real> &ransConservative = {}) const;
        /// Setup-only, collective error guard. Apply configured spatial
        /// overrides to the initialized constant background and conservatively
        /// average cell quadrature states. Writes owned values; no halo refresh.
        /// One serial expression/EOS context is retained for the whole pass.
        /// A point failure can leave partial setup output; discard it on error.
        void InitializeField(EquationDofs &state, const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers = {}) const;
        /// Volume-normalized L1/L2/Linf error in conservative cell averages.
        /// One-shot analytic error report. A persistent FlowErrorOperator reuses
        /// reference quadrature across reports; CUDA callers supply their lane.
        std::array<FlowState, 3> Error(const EquationDofs &state, const FlowProfileSettings &profile, const FlowState &primitive, real time,
                                       std::optional<Runtime::ExecutionStream> stream = {}) const;
        /// Prepare the complete equation mean-state predicate. Mixtures require
        /// their retained worker pool exactly once here; subsequent row checks
        /// perform no EOS inversion or worker acquisition.
        AdmissibilityPolicy PrepareAdmissibility(
            std::optional<Physics::Chemistry::ChemicalWorkerPool> workers =
                std::nullopt) const;
        /// Collective owned-only admissibility using the plan's default
        /// non-mixture policy. Never repairs a state silently. Mixture callers
        /// prepare and retain an explicit policy instead.
        bool Admissible(const EquationDofs &state) const;
        /// Collective, layout-agnostic complete-equation mean-state check.
        /// The policy owner must outlive this synchronous call.
        bool Admissible(const EquationDofs &state,
                        const AdmissibilityPolicy &policy) const;
        index LastFaceFallbacks(const Workspace &workspace) const;
        /// Prepare the correction's mean-state signals. For centered RANS,
        /// restore the gradient when another coupled stage used the workspace.
        /// Quadrature derivatives stay paired with their original rates.
        /// A supplied interface dependency is imported before boundary reads;
        /// flow/mixture auxiliary core work independent of the interface can
        /// run first. Two-equation gradient/auxiliary work shares the enclosing
        /// schedule and mean-halo ordering. Return is quiescent for all providers.
        void PrepareCorrectionAux(EquationDofs &state, ReconstructionState &reconstruction, Workspace &workspace, AuxiliaryValues &auxiliary,
                                  Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time) const;
        void PrepareCorrectionAux(EquationDofs &state, ReconstructionState &reconstruction, Workspace &workspace, AuxiliaryValues &auxiliary,
                                  Runtime::EagerEngine &engine, const Runtime::ExecutionStream &compute,
                                  const Runtime::ExecutionStream &communication, real time,
                                  OptionalRef<const QueuedEvaluationDependency> dependency = {}) const;
        /// A stored rate can outlive the scratch gradient of a coupled stage.
        /// Refresh its centered RANS gradient if another evaluation overwrote
        /// the workspace. Requires the matching stage's published mean halo and interface.
        void EnsureCenterSourceGradient(EquationDofs &state, ReconstructionState &reconstruction, Workspace &workspace,
                                        Runtime::EagerEngine &engine, const Runtime::ExecutionStream &compute,
                                        const Runtime::ExecutionStream &communication, real time) const;

        /// Five-component affine bound retained for specialized perfect-gas
        /// flow kernels and their compatibility tests. Complete-equation
        /// orchestration uses PrepareAdmissibility/Admissible instead.
        const ThermalBounds &ThermalConstraints() const;
        /// Collective checked projection of a fresh copy of the unlimited
        /// guess into flux coefficients, retaining the unlimited guess. Reads only
        /// owned means and cached owned-cell points; a mean ghost write may be
        /// active. Coefficient transfers must be complete before publication.
        /// Explicit outputs are limited coefficients and per-call beta stats.
        /// Does not reconstruct, exchange, advance time or change cell means.
        index LastCharacteristicFallbacks(const Workspace &workspace) const;
        /// Specialized flow5 conservative-blend output. This is an
        /// operator-local capability; generic equations do not yet construct a
        /// separate low-order residual.
        const EquationDofs &LowOrderResidual(const Workspace &workspace) const;
        /// Collective sum of the last completed equation RHS's wall momentum
        /// flux, in mesh axes, with the same sign as pressure*n - traction.
        /// Per-quadrature outputs include the actual numerical pressure, RANS
        /// stress and viscous traction; no separate EOS/flux recomputation.
        /// Workspace access is controller-serialized; do not read during Evaluate.
        /// Production callers supply their compute lane. Standalone synchronous
        /// CPU callers may omit it; resident CUDA input requires an explicit lane.
        /// @todo Add Euler-style per-BC integration selection/reporting for
        /// force, moment and conservative mass/energy/species fluxes. The
        /// current frontend reports only the selected-zone aggregate force;
        /// no generic per-BC integration history files are implemented.
        Vector3 LastBoundaryForce(const Workspace &workspace, const std::vector<int> &zones,
                                  std::optional<Runtime::ExecutionStream> stream = {}) const;
        /// All conservative blocks plus mean thermodynamic/species fields and
        /// reconstructed boundary traces. Reads an observation workspace only;
        /// does not modify the production reconstruction or any state values.
        void WriteVisualization(const EquationDofs &state, const ReconstructionState &reconstruction, const Workspace &workspace,
                                const std::string &prefix, real time, int hdfDeflateLevel,
                                const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers) const;
        /// Complete numerical observation before returning immutable host file
        /// input. The returned object retains geometry, not solver fields.
        PreparedVisualization PrepareVisualization(const EquationDofs &, const ReconstructionState &, const Workspace &,
                                                   real time, int hdfDeflateLevel,
                                                   const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers) const;

    private:
        /// Output adapter: resident boundary tracing/properties, followed by an
        /// explicit host I/O snapshot. Never downloads coefficients for CPU
        /// sampling or evaluates point physics inside the file writer.
        /// Resolve the retained interface publication after the canonical
        /// equation-level provenance check. Every equation packet uses this one
        /// leaf input.
        std::optional<BoundaryInputs> PreparedBoundaryInputs(const AuxiliaryValues &values) const;
        std::optional<BoundaryInputs> PreparedBoundaryInputs(const Workspace &workspace) const;
        /// Reconstruction-independent trace producer. Retains immutable
        /// geometry and physics while reserving mutable payloads at submission;
        /// numerical CPU/CUDA dispatch does not alter the enclosing schedule.
        void SubmitMeanGradientBoundary(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                        const EquationDofs &, MeanGradients &, real time,
                                        std::optional<Physics::Chemistry::ChemicalWorkerPool>,
                                        OptionalRef<const Runtime::Array<real>> incoming = {}) const;
        /// Queue the gradient-dependent auxiliary segment. An enclosing FV
        /// schedule may already have ordered the mean halo; exchangeMeans=false
        /// reuses that ordering rather than launching a second transfer. Deferred
        /// boundary inputs expose stable buffers and must have an imported event.
        AuxiliaryPreparation QueueMeanGradientAuxValues(EquationDofs &, AuxiliaryValues &, MeanGradients &,
                                                        Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                                        const Runtime::ExecutionStream &, real time,
                                                        std::optional<Physics::Chemistry::ChemicalWorkerPool>,
                                                        AuxiliaryInputs, bool exchangeMeans) const;
        /// Publish the queued gradient/auxiliary generations after completion.
        void FinishMeanGradientAuxValues(const EquationDofs &, AuxiliaryValues &, MeanGradients &,
                                         AuxiliaryPreparation &) const;
        /// Established fixed perfect-gas Roe leaf selected by the equation
        /// Jacobian resolver. It is not a second public state API.
        void PrepareFlow5Jacobian(const CFV::FVField &state, const AuxiliaryValues &auxiliary,
                                  const Runtime::Array<real> &inversePseudoTime, LinearProblemContext &output,
                                  JacobianPreparation controls, real time) const;
        void PrepareDeviceJacobian(const EquationDofs &, const AuxiliaryValues &, const Runtime::Array<real> &inversePseudoTime,
                                   LinearProblemContext &, JacobianPreparation, real time, const JacobianContext &) const;
        ReconstructionRestore ReserveReconstructionRestoreImpl(ReconstructionState &state, const EquationDofs &guess,
                                                               OptionalRef<const Runtime::Array<real>> limitedVRHistory,
                                                               std::shared_ptr<ReconstructionState> retained, bool collectiveChecks = true,
                                                               OptionalRef<const EquationDofs> finalized = {},
                                                               std::optional<Runtime::ExecutionStream> stream = {}) const;
        std::shared_ptr<Workspace> BuildWorkspaceImpl(EquationDofs &state,
                                                      std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                                      bool meanOnly, WorkspaceOptions options = WorkspaceOptions()) const;
        enum class EvaluationPurpose
        {
            Spatial,
            RestoreModalGradient,
            RestoreDirectGradient
        };
        void EvaluateImpl(EquationDofs &state, OptionalRef<EquationDofs> residual,
                          ReconstructionState &reconstruction, Workspace &workspace, Runtime::EagerEngine &engine,
                          const Runtime::ExecutionStream &compute,
                          const Runtime::ExecutionStream &communication, real time,
                          OptionalRef<AuxiliaryValues> auxiliary = {},
                          OptionalRef<const QueuedEvaluationDependency> dependency = {}, OptionalRef<Runtime::Array<real>> sourceDerivative = {},
                          EvaluationPurpose purpose = EvaluationPurpose::Spatial) const;
        void ValidateEquationSourceJacobian(const EquationDofs &state,
                                            const JacobianContext &context, real time) const;
        bool AddEquationSourceJacobian(const EquationDofs &state,
                                       const JacobianContext &context,
                                       const Runtime::Array<real>::WriteLease &diagonal, DiagonalStorage storage,
                                       real scale, PointEvaluator &evaluator,
                                       OptionalRef<const EquationFVDetail::MeanLinearizationOutput> linearization = {}, bool rowMajorBlocks = false) const;
        index PrepareFaceCharacteristicMatrices(const EquationDofs &state,
                                                Runtime::Array<real> &forward,
                                                Runtime::Array<real> &inverse,
                                                OptionalRef<const AuxiliaryValues> auxiliary, real time) const;
        index PrepareFaceCharacteristics(const EquationDofs &state, CFV::FaceCharacteristicTransform &transform,
                                         OptionalRef<const AuxiliaryValues> auxiliary, real time) const;
        index PrepareFaceCharacteristics(const EquationDofs &state,
                                         CFV::GradientFaceCharacteristicTransform &transform,
                                         OptionalRef<const AuxiliaryValues> auxiliary, real time) const;
    };
    /// Immutable publication record for one interface sampling round.
    /// It retains shared equation arrays (not a deep value snapshot) and their
    /// finite-volume owner; the
    /// coupling adapter validates recorded versions before reading. Direct
    /// P0, physical-gradient P1 and variational coefficients are distinct
    /// prepared modes, hidden from Simulation orchestration.
    class InterfaceDonorPublication
    {
        friend class FiniteVolume;
        friend class InterfaceCoupling;
        enum class Kind
        {
            Constant,
            Gradient,
            Variational
        };
        std::shared_ptr<const FiniteVolume::Data> _owner;
        EquationDofs _means;
        std::optional<EquationDofs> _coefficients;
        std::vector<CFV::FVField> _gradients;
        std::vector<uint64_t> _meanVersions, _reconstructionVersions;
        Kind _kind = Kind::Constant;
        InterfaceDonorPublication(std::shared_ptr<const FiniteVolume::Data> owner,
                                  EquationDofs means)
            : _owner(std::move(owner)), _means(std::move(means)) {}

    public:
        /// Recorded local allocation generations, not a distributed clock.
        /// Historical packet provenance remains valid after these arrays advance.
        const std::vector<uint64_t> &MeanVersions() const { return _meanVersions; }
        const std::vector<uint64_t> &ReconstructionVersions() const { return _reconstructionVersions; }
        /// Pre-sampling local validation; no leases, MPI, sampling or refresh.
        /// Do not revalidate against live arrays when consuming a frozen packet.
        void Validate() const;
        const EquationSchema &Schema() const { return _means.Schema(); }
        index PhysicalFieldNumber() const
        {
            return _means.PhysicalFieldNumber();
        }
    };

}
