#pragma once

#include "Cauchy/Discretization/FiniteVolumeReconstructionStateInternal.hpp"
#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include "Cauchy/Discretization/BoundaryForceReduction.hpp"

#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace FiniteVolumeDetail
    {
        /// Bind boundary rows (modes=1) or component-major polynomial rows to
        /// the prototype equation grouping. Payload arrays remain shared; only
        /// schema names and logical block offsets are adapted during setup.
        inline EquationDofs BindFieldSchema(const EquationDofs &prototype, Algebra::DofBundle fields,
                                            index modes)
        {
            auto schema = prototype.Schema();
            std::vector<EquationDofBinding> bindings;
            for (index iBlock = 0; iBlock < prototype.BlockNumber(); ++iBlock)
            {
                const auto location = prototype.LocateBlock(iBlock);
                auto &block = schema.blocks[iBlock];
                bindings.push_back({block.name, prototype.Fields().Name(location.physicalField),
                                    location.firstComponent * modes});
                if (modes != 1)
                {
                    auto components = std::move(block.components);
                    block.components.clear();
                    block.vectorStarts.clear();
                    for (const auto &component : components)
                        for (index mode = 0; mode < modes; ++mode)
                            block.components.push_back({component.name + ":mode" + std::to_string(mode),
                                                        component.units, "polynomial-coefficient"});
                }
            }
            if (modes != 1)
                schema.family = "polynomial:" + schema.family;
            return {std::move(schema), std::move(fields), std::move(bindings)};
        }
    }
    struct FiniteVolume::Workspace
    {
        R::SessionAccess session;
        struct Field
        {
            std::optional<CFV::ReconstructionWorkspace> reconstruction;
            std::optional<CFV::ReconstructionAMGWorkspace> reconstructionAMG;
            std::optional<CFV::ReconstructionBoundaryValues> reconstructionBoundary;
            Runtime::Array<real> boundaryValues;
            std::optional<CFV::FVField> gradient;
            std::optional<CFV::GradientWorkspace> gradientExecution;
            std::optional<R::Array<real>> gradientBoundary;
            std::vector<index> vectorStarts;
        };
        std::shared_ptr<const Data> owner;
        enum ProfilePhase
        {
            BoundaryPhase,
            GradientPhase,
            VRPhase,
            PPPhase,
            RHSPhase,
            SamplingPhase,
            SourcePhase,
            TotalPhase,
            ProfilePhases
        };
        bool profiling = std::getenv("DNDS_CAUCHY_PROFILE_EQUATION_FV") != nullptr;
        bool profileAllRanks = std::getenv("DNDS_CAUCHY_PROFILE_ALL_RANKS") != nullptr;
        std::array<real, ProfilePhases> profileSeconds{};
        std::array<index, ProfilePhases> profileCalls{};
        /// Optional CPU-lane interval, retained independently of the workspace.
        /// Includes work and scheduling gaps between the two queued markers;
        /// it is not a CUDA kernel duration or a sum of exclusive CPU cycles.
        struct QueuedPhaseSample
        {
            ProfilePhase phase;
            std::chrono::steady_clock::time_point start;
            real seconds = 0;
            std::atomic<bool> complete{false};
            explicit QueuedPhaseSample(ProfilePhase value) : phase(value) {}
        };
        std::vector<std::shared_ptr<QueuedPhaseSample>> queuedPhaseSamples;
        std::shared_ptr<QueuedPhaseSample> BeginQueuedPhase(Runtime::EagerEngine &, const Runtime::ExecutionStream &, ProfilePhase);
        static void EndQueuedPhase(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                   const std::shared_ptr<QueuedPhaseSample> &);
        /// Merge completed optional samples on the controller after the join.
        /// No atomics or shared counters occur inside numerical loops.
        void PublishQueuedPhases();
        struct PhaseTime
        {
            using Clock = std::chrono::steady_clock;
            Workspace *work;
            ProfilePhase phase;
            Clock::time_point start{};
            PhaseTime(Workspace &input, ProfilePhase section) : work(input.profiling ? &input : nullptr), phase(section)
            {
                if (work)
                    start = Clock::now();
            }
            ~PhaseTime()
            {
                if (!work)
                    return;
                work->profileSeconds[phase] += std::chrono::duration<real>(Clock::now() - start).count();
                const int rank = work->owner->mesh->getMPI().rank;
                if (++work->profileCalls[phase] % 32 == 0 && phase == TotalPhase && (!rank || work->profileAllRanks))
                {
                    const auto record = Json{{"rank", rank}, {"owned_cells", work->owner->metrics.NumCell()}, {"meanOnly", work->meanOnly}, {"reconstruction", work->owner->settings.numerics.reconstruction}, {"firstOrderConvection", work->owner->settings.numerics.firstOrderConvection}, {"phase_timing", work->owner->metrics.Device() == Runtime::Device::Host() ? "cpu_lane_intervals" : "host_total_only"}, {"phases", {"boundary", "gradient", "vr", "pp", "rhs", "sampling", "source", "evaluate"}}, {"seconds", work->profileSeconds}, {"calls", work->profileCalls}}.dump();
                    std::fprintf(stdout, "CAUCHY_EQUATION_FV_PROFILE %s\n", record.c_str());
                }
            }
        };
        const bool meanOnly;
        const bool captureBoundaryFlux;
        mutable std::optional<BoundaryForceReduction> boundaryForceReduction;
        index componentNumber = 0, physicalFieldNumber = 0;
        std::vector<Field> fields;
        EquationDofs boundary;
        /// Operation-scoped retained binding. It is installed by the
        /// orchestration call and cleared on every exit; the workspace never
        /// becomes the durable coefficient owner.
        std::optional<EquationDofs> activeCoefficients, activeFluxCoefficients;
        // Accepted direct-FV physical gradients, separate from source sampling.
        // WBAP/PP consume their explicit polynomial projection; the direct RHS
        // reads only these physical gradients and never evaluates a Taylor basis.
        std::vector<CFV::FVField> directGradients, directViscousGradients;
        /// Unfiltered explicit P1 keeps the GradientPlan publication directly;
        /// no gradient-to-modal-to-gradient round trip or coefficient halo.
        bool nativeDirectGradient = false;
        std::vector<R::TransferBinding> directGradientHalo, directGradientInputHalo,
            directViscousGradientHalo;
        /// Legacy-compatible primitive-gradient viscosity is prepared inside
        /// this workspace but consumed by the same fixed-flow RHS leaf.
        std::optional<CFV::FVField> primitive, primitiveGradient;
        std::optional<R::Array<real>> primitiveBoundary;
        std::optional<CFV::GradientWorkspace> primitiveGradientExecution;
        /// Temporary device status, separate from durable reconstruction state.
        /// Reset once per FV evaluation; every boundary/gradient producer folds
        /// into result before reusing its row buffer. Inspect only after drain.
        struct PreparationStatusStorage
        {
            R::Array<NumericalStatus> quadrature, primitive, meanOwned, meanGhost, partial, result;
            bool pending = false;
        };
        std::optional<PreparationStatusStorage> preparationStatus;
        struct DeviceFluxData;
        std::shared_ptr<DeviceFluxData> deviceFlux;
        struct DeviceJacobianData;
        std::shared_ptr<DeviceJacobianData> deviceJacobian;
        struct CharacteristicDeviceStorage;
        std::shared_ptr<CharacteristicDeviceStorage> characteristicDevice;
        struct ReconstructionNormStorage
        {
            R::Array<real> partial, result;
        };
        std::optional<ReconstructionNormStorage> reconstructionNorm;
        struct CharacteristicSubmission
        {
            index fallbacks = 0;
            std::atomic<bool> completed{false};
            bool published = false;
            std::shared_ptr<CharacteristicDeviceStorage> device;
            uint64_t resultEpoch = 0;
        };
        std::optional<R::TransferBinding> primitiveGradientHalo;
        struct DirectGradientPublication
        {
            struct BoundaryField
            {
                std::weak_ptr<const R::ArrayStorage> storage;
                uint64_t shape = 0, version = 0;
            };
            // Weak identities reject rebinding/ABA without retaining old fields.
            // The workspace/operation owns the arrays and packet; this stamp is
            // validation metadata, never an asynchronous lifetime mechanism.
            std::weak_ptr<const R::ArrayStorage> mean, input, output;
            uint64_t meanShape = 0, meanVersion = 0, inputShape = 0,
                     inputVersion = 0, outputShape = 0, outputVersion = 0;
            R::VersionStamp geometry;
            std::vector<BoundaryField> boundary;
            real time = std::numeric_limits<real>::quiet_NaN();
            std::string reconstruction, gradient, limiter;

            void Publish(const Data &plan, const R::Array<real> &meanValues,
                         const R::Array<real> &inputValues,
                         const R::Array<real> &outputValues,
                         const std::optional<BoundaryInputs> &incoming,
                         real stageTime)
            {
                const auto meanRead = meanValues.Read({meanValues.GetDevice(), nullptr});
                const auto inputRead = inputValues.Read({inputValues.GetDevice(), nullptr});
                const auto outputRead = outputValues.Read({outputValues.GetDevice(), nullptr});
                mean = meanValues.SharedStorage();
                input = inputValues.SharedStorage();
                output = outputValues.SharedStorage();
                meanShape = meanValues.ShapeVersion();
                meanVersion = meanValues.Version();
                inputShape = inputValues.ShapeVersion();
                inputVersion = inputValues.Version();
                outputShape = outputValues.ShapeVersion();
                outputVersion = outputValues.Version();
                geometry = plan.boundaryLayout->stamp;
                boundary.clear();
                if (plan.hasInterfaces)
                {
                    DNDS_check_throw_info(
                        incoming && incoming->Time() == stageTime,
                        "direct gradient publication needs the current interface packet");
                    const auto &values = incoming->QuadratureValues();
                    boundary.reserve(size_t(values.PhysicalFieldNumber()));
                    for (index iField = 0;
                         iField < values.PhysicalFieldNumber(); ++iField)
                    {
                        const auto &array = values.PhysicalField(iField).Owned();
                        const auto read = array.Read({array.GetDevice(), nullptr});
                        boundary.push_back({array.SharedStorage(),
                                            array.ShapeVersion(),
                                            array.Version()});
                    }
                }
                time = stageTime;
                reconstruction = plan.settings.numerics.reconstruction;
                gradient = plan.settings.numerics.gradient;
                limiter = plan.settings.numerics.limiter;
            }

            bool Current(const Data &plan, const R::Array<real> &meanValues,
                         const R::Array<real> &inputValues,
                         const R::Array<real> &outputValues,
                         const std::optional<BoundaryInputs> &incoming,
                         real stageTime) const
            {
                // Acquire before reading generations: checking first would
                // permit a writer to publish a new generation in the gap.
                const auto meanRead = meanValues.Read({meanValues.GetDevice(), nullptr});
                const auto inputRead = inputValues.Read({inputValues.GetDevice(), nullptr});
                const auto outputRead = outputValues.Read({outputValues.GetDevice(), nullptr});
                if (time != stageTime || !geometry.Valid() ||
                    reconstruction != plan.settings.numerics.reconstruction ||
                    gradient != plan.settings.numerics.gradient ||
                    limiter != plan.settings.numerics.limiter ||
                    mean.lock() != meanValues.SharedStorage() ||
                    input.lock() != inputValues.SharedStorage() ||
                    output.lock() != outputValues.SharedStorage() ||
                    meanShape != meanValues.ShapeVersion() ||
                    meanVersion != meanValues.Version() ||
                    inputShape != inputValues.ShapeVersion() ||
                    inputVersion != inputValues.Version() ||
                    outputShape != outputValues.ShapeVersion() ||
                    outputVersion != outputValues.Version())
                    return false;
                if (!plan.hasInterfaces)
                    return boundary.empty();
                if (!incoming || incoming->Time() != stageTime ||
                    boundary.size() !=
                        size_t(incoming->QuadratureValues()
                                   .PhysicalFieldNumber()))
                    return false;
                const auto &values = incoming->QuadratureValues();
                for (index iField = 0;
                     iField < values.PhysicalFieldNumber(); ++iField)
                {
                    const auto &array = values.PhysicalField(iField).Owned();
                    const auto read = array.Read({array.GetDevice(), nullptr});
                    const auto &saved = boundary[size_t(iField)];
                    if (saved.storage.lock() != array.SharedStorage() ||
                        saved.shape != array.ShapeVersion() ||
                        saved.version != array.Version())
                        return false;
                }
                return true;
            }
        };
        std::shared_ptr<PositivityWorkspace> positivity;
        std::unique_ptr<LimitedVRWorkspace> limitedVR;
        ThermalPolicy thermal;
        /// Setup-resolved fixed-flow leaf. It owns immutable kernel reads; the
        /// surrounding Workspace still owns the only reconstruction/RHS
        /// schedule and all operand leases.
        bool genericFluxOracle = false;
        std::optional<RHSOperator> fixedFlowRHS;
        /// One synchronous fixed-flow invocation's leases and plain views.
        /// Capacity is resolved by the prepared operator contract; no heap
        /// allocation occurs while packaging a core/dependent RHS call.
        struct RHSAccessScope
        {
            struct FieldRead
            {
                std::optional<R::Array<real>::ReadLease> owned, ghost;
            };
            static constexpr size_t FieldCapacity = 8;
            bool ghosts = false;
            size_t fieldCount = 0;
            std::array<FieldRead, FieldCapacity> fields;
            std::optional<R::Array<real>::ReadLease> thermodynamics;
            std::optional<DerivedFields::ReadScope> derivedOwned, derivedGhost;
            std::optional<R::Array<real>::WriteLease> result, lowOrder, face, boundaryMomentum;
            RHSReadViews input;
            RHSWriteViews output;

            explicit RHSAccessScope(bool includeGhosts) : ghosts(includeGhosts) {}
            RHSFieldView ReadField(const CFV::FVField &field)
            {
                DNDS_check_throw_info(fieldCount < fields.size(),
                                      "prepared RHS access capacity exceeded");
                auto &slot = fields[fieldCount++];
                slot.owned.emplace(field.Owned().Read());
                if (ghosts)
                    slot.ghost.emplace(field.Ghost().Read());
                return {slot.owned->View<R::UniformTableRows>(),
                        slot.ghost ? slot.ghost->View<R::UniformTableRows>() : RHSScalarView{}};
            }
            RHSFlowFieldView ReadFlowField(const CFV::FVField &field)
            {
                DNDS_check_throw_info(fieldCount < fields.size(),
                                      "prepared RHS access capacity exceeded");
                auto &slot = fields[fieldCount++];
                slot.owned.emplace(field.Owned().Read());
                if (ghosts)
                    slot.ghost.emplace(field.Ghost().Read());
                return {slot.owned->View<RHSFlowRows>(),
                        slot.ghost ? slot.ghost->View<RHSFlowRows>() : decltype(slot.owned->View<RHSFlowRows>()){}};
            }
        };
        RHSLocalDiagnostics diagnostics;
        std::array<index, 2> faceFallbacks{};
        std::optional<CFV::FaceCharacteristicTransform> flowCharacteristic;
        std::optional<CFV::GradientFaceCharacteristicTransform> gradientFlowCharacteristic;
        index flowCharacteristicField = -1;
        index characteristicFallbacks = 0;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        /// Published receiver-frame interface traces. The workspace retains
        /// their allocation/publication identity; each evaluation pins the
        /// numeric rows through its final dependent consumer.
        std::optional<BoundaryInputs> incoming;
        /// Operator-owned packed auxiliary channels needed by flux profiles
        /// whose RHS cannot be evaluated from one face pair alone. Ordinary
        /// fluxes avoid this allocation and their existing fused path.
        std::optional<AuxiliaryValues> rhsAuxiliary;
        R::MetadataBuffer<index> logicalFirst, vectorStarts;
        R::MetadataBuffer<index> logicalField, logicalComponent;
        index flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        R::Array<index> sourceOffsets{R::RowLayout::Dense(0, 1)};
        R::Array<real> sourceWeights;
        std::optional<CFV::PolynomialSamplingPlan> sourceSampling;
        /// Per-area viscous traction on a no-slip wall from the last full RHS.
        /// Its sign uses n outward from the fluid. Other boundary rows are zero.
        R::Array<real> boundaryMomentumFlux;
        /// Integrated residual contribution in the canonical owned-parent
        /// frame, one row per local stencil face. Refilled on every RHS call.
        R::Array<real> faceFlux;
        /// Optional first-order conservative flux path used by explicit flow5
        /// positivity. It is an output of the same equation RHS operator, not
        /// a second finite-volume frontend or a second reconstruction pass.
        std::optional<EquationDofs> lowOrderResidual;
        std::optional<R::Array<real>> lowOrderFaceFlux;
        bool boundaryMomentumCurrent = false, boundaryFacialFluxCurrent = false;
        std::optional<MeanGradients> meanGradients;
        bool chemical;
        int iterations = 0;
        real relativeUpdate = 0;
        index directPPLimited = 0;
        real directPPMinimum = 1;
        /// Reused direct-P1 CPU scratch. Workspace session access serializes
        /// these buffers; no allocation occurs in the cell/point operator.
        R::Array<real> directLimiterScratch, directPPSamples;
        index cpuGradientWorkers = 1;
        struct WeakPublication
        {
            std::vector<std::weak_ptr<const R::ArrayStorage>> storage;
            std::vector<uint64_t> shapes, versions;
            explicit WeakPublication(const EquationDofs &values)
            {
                storage.reserve(size_t(values.PhysicalFieldNumber()));
                shapes.reserve(storage.capacity());
                versions.reserve(storage.capacity());
                for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                {
                    const auto &array = values.PhysicalField(iField).Owned();
                    storage.push_back(array.SharedStorage());
                    shapes.push_back(array.ShapeVersion());
                    versions.push_back(array.Version());
                }
            }
            bool Current(const EquationDofs &values) const
            {
                if (storage.size() != size_t(values.PhysicalFieldNumber()))
                    return false;
                for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                {
                    const auto &array = values.PhysicalField(iField).Owned();
                    if (storage[size_t(iField)].lock() != array.SharedStorage() ||
                        shapes[size_t(iField)] != array.ShapeVersion() ||
                        versions[size_t(iField)] != array.Version())
                        return false;
                }
                return true;
            }
        };
        struct CenterGradientInput
        {
            WeakPublication mean;
            // Remote means may change independently of this rank's owned
            // generation. Every completed halo replacement advances its version.
            std::vector<uint64_t> ghostVersions;
            std::optional<WeakPublication> boundaryCenters, boundaryQuadrature;
            real time;
            CenterGradientInput(const EquationDofs &state, const std::optional<BoundaryInputs> &packet, real t) : mean(state), time(t)
            {
                ghostVersions.reserve(state.PhysicalFieldNumber());
                for (index field = 0; field < state.PhysicalFieldNumber(); ++field)
                    ghostVersions.push_back(state.PhysicalField(field).Ghost().Version());
                if (packet)
                {
                    boundaryCenters.emplace(packet->FaceCenterValues());
                    boundaryQuadrature.emplace(packet->QuadratureValues());
                }
            }
            bool Current(const EquationDofs &state, const std::optional<BoundaryInputs> &packet, real t) const
            {
                if (time != t || !mean.Current(state) || bool(packet) != bool(boundaryCenters))
                    return false;
                for (index field = 0; field < state.PhysicalFieldNumber(); ++field)
                    if (ghostVersions[field] != state.PhysicalField(field).Ghost().Version())
                        return false;
                return !packet || (boundaryCenters->Current(packet->FaceCenterValues()) && boundaryQuadrature->Current(packet->QuadratureValues()));
            }
        };
        std::optional<CenterGradientInput> centerGradientInput;
        std::optional<EquationDofOperations::Binding> rhsState;
        std::optional<WeakPublication> rhsCoefficients;
        real rhsTime = 0;
        // Keep auxiliary cache metadata after the existing numerical fields.
        std::vector<std::array<DirectGradientPublication, 2>> directGradientPublications;

        class ActiveReconstruction
        {
            Workspace *_work = nullptr;

        public:
            ActiveReconstruction(Workspace &work, const ReconstructionState &state) : _work(&work)
            {
                DNDS_check_throw_info(!work.activeCoefficients && !work.activeFluxCoefficients,
                                      "workspace already has an active reconstruction binding");
                if (state.coefficients)
                {
                    work.activeCoefficients.emplace(state.WarmValues());
                    work.activeFluxCoefficients.emplace(state.FinalizedValues());
                }
            }
            ActiveReconstruction(const ActiveReconstruction &) = delete;
            ActiveReconstruction &operator=(const ActiveReconstruction &) = delete;
            ActiveReconstruction(ActiveReconstruction &&other) noexcept : _work(other._work) { other._work = nullptr; }
            ~ActiveReconstruction()
            {
                if (_work)
                {
                    _work->activeCoefficients.reset();
                    _work->activeFluxCoefficients.reset();
                }
            }
        };
        ActiveReconstruction BindReconstruction(const ReconstructionState &state)
        {
            return ActiveReconstruction(*this, state);
        }
        const EquationDofs &Coefficients() const
        {
            DNDS_check_throw_info(activeCoefficients, "workspace has no active reconstruction binding");
            return *activeCoefficients;
        }
        const EquationDofs &FluxCoefficients() const
        {
            DNDS_check_throw_info(activeFluxCoefficients, "workspace has no active finalized reconstruction binding");
            return *activeFluxCoefficients;
        }
        EquationDofs &FluxCoefficients()
        {
            DNDS_check_throw_info(activeFluxCoefficients, "workspace has no active finalized reconstruction binding");
            return *activeFluxCoefficients;
        }

        static std::vector<Field> BuildFields(const Data &data, const EquationDofs &state, bool meanOnly);
        void BuildSourceSampling();
        static EquationDofs BindBoundary(const EquationDofs &state, const std::vector<Field> &fields, index rows);
        Workspace(std::shared_ptr<const Data> data, EquationDofs &state,
                  std::optional<Physics::Chemistry::ChemicalWorkerPool> pool, bool meanTrial, bool capture);
        void BindHalos(EquationDofs &state);
        void RefreshDirectGradients(const EquationDofs &state, bool ghost, real time);
        /// Queue physical-gradient copies/projections without acquiring
        /// mutable payload leases on the controller. Native direct gradients
        /// and aliased P0 source gradients already contain their final values.
        /// Dependencies are caller ordered; completed-version caching cannot
        /// skip work while a new mean/coefficient producer is pending.
        void SubmitDirectGradients(R::EagerEngine &, const R::ExecutionStream &,
                                   const EquationDofs &state, bool ghost);
        /// Record completed direct-gradient provenance after the enclosing
        /// schedule drains. This never submits work or synchronizes a device.
        void PublishDirectGradients(const EquationDofs &state, bool ghost, real time);
        void PrepareDirectGradientLimiter(const EquationDofs &state, bool barth);
        void ApplyDirectGradientPositivity(const EquationDofs &state);
        void PreparePrimitiveGradient(const EquationDofs &state, CFV::ReconstructionRegion region);
        /// Common queued boundary producer. Mutable fields are reserved at
        /// submission, including split CPU equation layouts. The CPU evaluator
        /// is retained until completion; if absent, the CPU launcher acquires
        /// a private evaluator from the workspace pool during execution.
        /// CUDA uses its placed immutable policy.
        /// The caller must not share a mutable chemistry evaluator across lanes.
        void SubmitBoundary(R::EagerEngine &, const R::ExecutionStream &,
                            const EquationDofs &, bool polynomial, real time,
                            std::shared_ptr<PointEvaluator> evaluator = {},
                            OptionalRef<const R::Array<real>> incoming = {},
                            OptionalRef<const R::Array<real>> incomingCenters = {});
        void BuildPreparationStatus();
        /// One backend-neutral numerical status segment per spatial call.
        /// CPU leaves report during execution; CUDA summaries are inspected
        /// only after the caller's final drain. Neither method waits implicitly.
        void BeginNumericalStatus(R::EagerEngine &, const R::ExecutionStream &, bool withRHS);
        void CheckNumericalStatus(bool withRHS);
        void ResetPreparationStatus(R::EagerEngine &, const R::ExecutionStream &);
        void CheckPreparationStatus();
        void AppendPreparationStatus(R::EagerEngine &, const R::ExecutionStream &, const R::Array<NumericalStatus> &);
        /// Queued CFV gradient construction uses the same entry for CPU/CUDA.
        /// Primitive conversion is an operator-local backend dispatch; neither
        /// this entry nor CFV hides halo communication or host observation.
        void SubmitGradients(R::EagerEngine &, const R::ExecutionStream &, const EquationDofs &, CFV::ReconstructionRegion);
        void BuildCharacteristicStorage();
        /// Mean-based flow-five producer; scalar extension equations remain
        /// untouched. CPU consumes inline gas or declared cached mixture
        /// properties, CUDA evaluates perfect-gas thermodynamics inline.
        /// Matrices and all payload reads are reserved at submission. Follow
        /// by WBAP on the same lane or import its event on another lane.
        std::shared_ptr<CharacteristicSubmission> SubmitCharacteristics(
            R::EagerEngine &, const R::ExecutionStream &, const EquationDofs &,
            OptionalRef<const AuxiliaryValues> auxiliary = {});
        /// Inspect mandatory status and merge fallback counts after drain.
        void PublishCharacteristics(const std::shared_ptr<CharacteristicSubmission> &);
        /// Queue owned reconstruction update/reference norms, with a bounded
        /// reusable device reduction. Observe only at a requested stopping
        /// interval; the global reduction remains outside the async segment.
        void SubmitReconstructionNorm(R::EagerEngine &, const R::ExecutionStream &, const EquationDofs &current);
        std::array<real, 2> ReadReconstructionNorm() const;
        void SubmitDeviceGradientLimiter(R::EagerEngine &, const R::ExecutionStream &, const EquationDofs &, bool barth, CFV::ReconstructionRegion);
        void SubmitDeviceGradientPositivity(R::EagerEngine &, const R::ExecutionStream &, const EquationDofs &);
        /// Unsplit common direct-gradient limiter. Input/output and scratch
        /// accesses are reserved at submission; equation dispatch stays local.
        void SubmitGradientLimiter(R::EagerEngine &, const R::ExecutionStream &, const EquationDofs &, bool barth);
        struct GradientPositivitySubmission
        {
            index limited = 0;
            real minimum = 1;
            std::optional<R::Array<index>> deviceCount;
            std::optional<R::Array<real>> deviceMinimum;
            uint64_t countEpoch = 0, minimumEpoch = 0;
            std::atomic<bool> completed{false};
            bool published = false;
        };
        struct LimiterStatisticsStorage
        {
            R::Array<NumericalStatus> partialStatus, status;
            R::Array<index> partialCount, count;
            R::Array<real> partialMinimum, minimum;
        };
        std::optional<LimiterStatisticsStorage> gradientStatistics;
        std::shared_ptr<GradientPositivitySubmission> pendingGradientStatistics;
        /// Record CPU results or device scalar reductions. Publish after the
        /// enclosing drain and numerical-status check, before reusing scratch.
        std::shared_ptr<GradientPositivitySubmission> SubmitGradientPositivity(
            R::EagerEngine &, const R::ExecutionStream &, const EquationDofs &);
        void PublishGradientPositivity(const std::shared_ptr<GradientPositivitySubmission> &);
        /// Device branch of the boundary operator; called by the common
        /// spatial schedule, never from a numerical leaf. No host wait.
        void SubmitDeviceBoundary(R::EagerEngine &, const R::ExecutionStream &,
                                  const EquationDofs &, bool polynomial, real time,
                                  OptionalRef<const R::Array<real>> incoming = {},
                                  OptionalRef<const R::Array<real>> incomingCenters = {});
        /// Device flux preparation owns only immutable descriptor scopes and
        /// reusable scratch. Durable means/coefficients are explicit operands.
        void BuildDeviceFlux();
        const R::Array<index> &DeviceCellSelection(CFV::ReconstructionRegion) const;
        void ResetDeviceFluxStatus(R::EagerEngine &, const R::ExecutionStream &);
        void CheckDeviceFluxStatus();
        void SubmitDeviceRHS(R::EagerEngine &, const R::ExecutionStream &,
                             const EquationDofs &, EquationDofs &, CFV::ReconstructionRegion,
                             real time, OptionalRef<const AuxiliaryValues> auxiliary = {},
                             OptionalRef<const R::Array<real>> incoming = {});
        struct RHSSubmission
        {
            std::optional<RHSKernelDiagnostics> fixedDiagnostics;
            index faceFallbacks = 0;
            CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All;
            std::atomic<bool> completed{false};
            bool published = false; ///< Controller-only publication guard.
        };
        /// Backend-local flux dispatch with submission-time Array operands.
        /// CPU returns retained diagnostics to publish after the enclosing
        /// segment drains. CUDA uses the workspace's device status segment and
        /// returns an empty record. The same scheduling caller uses both.
        /// A shared point evaluator is exclusive to this stream until drain.
        std::shared_ptr<RHSSubmission> SubmitRHS(
            R::EagerEngine &, const R::ExecutionStream &, const EquationDofs &, EquationDofs &,
            CFV::ReconstructionRegion, real time, std::shared_ptr<PointEvaluator> evaluator = {},
            OptionalRef<const AuxiliaryValues> auxiliary = {},
            OptionalRef<R::Array<real>> faceOutput = {},
            OptionalRef<const R::Array<real>> incoming = {});
        /// Publish one CPU region's diagnostics after successful execution.
        /// Empty CUDA records are harmless; CheckDeviceFluxStatus is the
        /// device segment's numerical-failure observation boundary.
        void PublishRHS(const std::shared_ptr<RHSSubmission> &);
        void ApplyFixedFlowRHS(const EquationDofs &state, EquationDofs &result,
                               CFV::ReconstructionRegion region, real time,
                               OptionalRef<const AuxiliaryValues> auxiliary,
                               OptionalRef<R::Array<real>> faceOutput,
                               RHSScalarView incoming);
        void ApplyRHS(const EquationDofs &state, EquationDofs &result,
                      CFV::ReconstructionRegion region, PointEvaluator &evaluator,
                      real time, OptionalRef<const AuxiliaryValues> auxiliary = {},
                      OptionalRef<R::Array<real>> faceOutput = {},
                      RHSScalarView incoming = {});
        void ApplyEquationRHS(const EquationDofs &state, CFV::ReconstructionRegion region,
                              PointEvaluator &evaluator, real time,
                              OptionalRef<const AuxiliaryValues> auxiliary,
                              const EquationDofs::ReadScope &mean,
                              EquationLayoutView<const real> owned, EquationLayoutView<const real> remote,
                              EquationLayoutView<real> destination, EquationLayoutView<real> lowDestination, bool produceLow,
                              OptionalRef<R::Array<real>> faceOutput, RHSScalarView incoming);
        /// Integrate sources into owned CUDA residual rows. Numerical failures
        /// are recorded in the active preparation segment.
        void SubmitDeviceVolumeSource(R::EagerEngine &, const R::ExecutionStream &,
                                      const EquationDofs &, EquationDofs &result, real time,
                                      OptionalRef<EquationDofs> lowOrder = {}, OptionalRef<R::Array<real>> sourceDerivative = {});
        /// Common CPU/CUDA source addition. Direct RANS gradients must already
        /// be prepared. Modal values use the shared volume basis inside the source kernel. Only rigid-
        /// frame flow sources are also added to the explicit lowOrder output,
        /// preserving the synchronous source contract. An optional evaluator
        /// is retained and exclusive to this lane; otherwise a worker from
        /// the retained workspace pool is acquired only during execution.
        /// Optional derivative storage belongs to the caller pairing it with this rate.
        void SubmitVolumeSource(R::EagerEngine &, const R::ExecutionStream &,
                                const EquationDofs &, EquationDofs &result, real time,
                                std::shared_ptr<PointEvaluator> evaluator = {},
                                OptionalRef<EquationDofs> lowOrder = {}, OptionalRef<R::Array<real>> sourceDerivative = {});
        bool AddMeanSourceJacobian(
            const EquationDofs &state, const R::Array<real>::WriteLease &diagonal,
            DiagonalStorage storage, real scale, PointEvaluator &evaluator,
            OptionalRef<const EquationFVDetail::MeanLinearizationOutput> linearization = {}, bool rowMajorBlocks = false) const;
        void PrepareVolumeSource(const EquationDofs &state,
                                 OptionalRef<EquationDofs> result,
                                 PointEvaluator &evaluator, real time = 0,
                                 OptionalRef<EquationDofs> lowOrder = {});
    };

    struct FiniteVolume::MeanWorkspace
    {
        /// Zero-polynomial probe storage is temporary by definition; it is not
        /// the solver's warm reconstruction publication.
        std::shared_ptr<ReconstructionState> reconstruction;
        std::shared_ptr<Workspace> work;
    };

}
