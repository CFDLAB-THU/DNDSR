#pragma once

#include "DNDS/Core/RowRanges.hpp"

#include "DNDS/Algebra/BlockMatrixStorage.hpp"
#include "Cauchy/Implicit/LinearPattern.hpp"
#include "Cauchy/Implicit/OffDiagonalView.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Algebra/ComponentSelection.hpp"
#include "DNDS/Core/OptionalRef.hpp"
#include "DNDS/Runtime/Storage/Metadata.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace DNDS::Cauchy
{
    class EquationDofs;
    enum class LinearInputLayout
    {
        Native,
        Tiled32
    };
    /// Explicit CUDA small-block kernel selection. CPU always retains the
    /// canonical guarded Eigen LU/SVD implementation.
    enum class CUDADiagonalInverse
    {
        BatchedSVD,
        GuardedLU
    };
    /// Launch mapping only; the component and sparse accumulation orders are
    /// unchanged. CUDA component groups are available for assembled Jacobi.
    struct LinearKernelOptions
    {
        bool componentGroups = false;
    };
    enum class DiagonalStorage
    {
        Entries,
        Block
    };
    enum class OffDiagonalStorage
    {
        Assembled,
        DirectEvaluated,
        SemiMatrixFree = DirectEvaluated // Compatibility spelling.
    };
    /// Optional CPU SGS physical-row subpartitions, copied at construction.
    /// Starts includes both endpoints 0 and ownedRows. Empty means one block.
    /// OMP=1 always retains the original global logical sweep.
    struct LinearSweepPartitions
    {
        std::vector<index> starts;
        std::vector<WorkerRowRanges> workers;
    };
    /// Independent numeric matrix layouts and the primary symbolic index ABI.
    /// Int32 is opt-in for providers which borrow that exact representation.
    struct LinearMatrixOptions
    {
        Algebra::MatrixTableOptions diagonal, inverse, offDiagonal;
        LinearIndexStorage indices = LinearIndexStorage::Int64;
        /// Dense component order, independent of physical table layout. The
        /// default remains column-major; AmgX chooses primary row-major blocks
        /// for diagonal, inverse and off-diagonal together, without shadows.
        bool rowMajorBlocks = false;
        bool cpuAMG = false;
        /// Allocate [off-diagonal][diagonal] in one native buffer, with disjoint
        /// Array aliases. Only the compatible single-device AmgX route uses it.
        bool amgxNativeStorage = false;
    };
    /// Integrated residual convention: A=V Gamma/dTau+physicalMass V M-residualScale J_R.
    /// The equation defines the pseudo mass Gamma and physical accumulation
    /// derivative M independently. J_R acts on the integrated spatial residual.
    /// Roe dissipation requires a full diagonal block. Scalar dissipation may
    /// additionally compose physical boundary derivatives into that diagonal.
    struct JacobianPreparation
    {
        real physicalMass = 0, residualScale = 1;
        bool roeDissipation = false;
        /// Empty means the complete logical state (flow5 for the field adapter).
        /// A subset embeds its principal matrix in that complete storage:
        /// unselected rows/columns are zero, diagonal is one.
        /// The matching RHS must be zero in those unselected equations.
        std::vector<index> updateComponents;
        /// False reproduces Euler's non-Roe boundary approximation: retain
        /// only 0.5*residualScale*faceArea*lambda on the diagonal, with no
        /// exterior-state chain rule. Interior/source blocks are unchanged.
        /// ACM applies it to the boundary part of its assembled Jacobian.
        /// Compressible Roe linearization and RHS BCs are independent of it.
        bool scalarBoundaryDerivative = true;
    };

    /// Frozen rank-owned block rows, including off-rank operator columns. The
    /// row space and stencil are immutable; numerical values are filled through
    /// native Array leases. No EOS, mesh mutation, MPI, nonlinear or time policy
    /// is hidden here. The caller serializes numeric preparation and execution.
    /// TODO(large-species Jacobian): optionally store off-diagonal flow5 dense
    /// blocks plus scalar extended entries, retaining full diagonal blocks.
    /// With flow/extension coupling omitted AND extended diagonal blocks also
    /// diagonal, this is an exact direct-sum representation/factorization of
    /// that approximate operator. Otherwise ILU fill can generate cross terms;
    /// projected/compressed factors change the preconditioner and need explicit
    /// controls and validation. See CauchyImplicit.md, large-species TODO.
    class LinearProblemContext
    {
        using Field = Algebra::DofField<real>;
        void JacobiImpl(const Field &, const Field *, Field &, LinearRegion) const;
        void JacobiImpl(const EquationDofs &, const EquationDofs *, EquationDofs &, LinearRegion) const;
        template <class State>
        void SubmitRows(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                        OptionalRef<const State> rhs, OptionalRef<const State> input,
                        State &output, LinearRegion region, LinearKernelOptions options = {}) const;
        struct Pattern
        {
            std::shared_ptr<const Algebra::DofSpace::Partition> partition;
            LinearPatternStorage rows;
            Runtime::MetadataBuffer<WorkerRowRanges> sweepPartitions;
            /// Host validation metadata only; resident numerical operators
            /// consume rows. This avoids copying device indices per publication.
            std::shared_ptr<const LinearPatternStorage> hostRows;
            bool hasGhostColumns = false;
            explicit Pattern(std::shared_ptr<const Algebra::DofSpace::Partition> rowsIdentity)
                : partition(std::move(rowsIdentity)) {}
        };
        Algebra::DofSpace _space;
        std::shared_ptr<const Pattern> _pattern;
        std::shared_ptr<const LinearPatternStorage> _validatedColumns;
        index _components;
        DiagonalStorage _diagonalStorage;
        bool _rowMajorBlocks;
        Runtime::Array<real> _diagonal, _inverse;
        std::optional<Runtime::Array<real>> _nativeAMGXValues;
        std::optional<Runtime::Array<real>> _geometricWeights;
        OffDiagonalStorage _offDiagonalStorage;
        std::optional<Algebra::BlockMatrixStorage> _offDiagonal;
        std::shared_ptr<const OffDiagonalEvaluator> _offDiagonalEvaluator;
        uint64_t _offDiagonalGeneration = 0;
        uint64_t _diagonalVersion = 0, _inverseVersion = 0;
        uint64_t _inverseSourceEpoch = 0, _inverseEpoch = 0;
        Runtime::ArrayStorage::Shared _diagonalStoragePin, _offDiagonalStoragePin;
        uint64_t _diagonalShape = 0;
        bool _inverseCurrent = false;
        index _inverseFallbacks = 0;
        index _inverseSVDCells = 0;
        struct InverseWorkspace;
        std::shared_ptr<InverseWorkspace> _inverseWorkspace;
        bool _inversePending = false;
        struct ActionWorkspace;
        std::shared_ptr<ActionWorkspace> _actionWorkspace;
        struct InputWorkspace;
        std::shared_ptr<InputWorkspace> _inputWorkspace;
        void PrepareActionWorkspace();
        Runtime::Array<NumericalStatus> &ActionStatus() const;
        struct ILUState;
        std::shared_ptr<ILUState> _ilu;
        static std::shared_ptr<const Pattern> BuildPattern(const Algebra::DofSpace &space, std::vector<std::vector<index>> columns, std::vector<index> core, LinearSweepPartitions partitions, LinearIndexStorage indices);
        const Pattern &Layout() const
        {
            DNDS_check_throw_info(_pattern, "linear context was moved");
            return *_pattern;
        }
        void ValidateMatrix() const;
        void ValidateField(const Field &field) const;
        void ValidateOutputs(const Field &input, const Field &output) const;
        void ValidateField(const EquationDofs &field) const;
        void ValidateOutputs(const EquationDofs &input, const EquationDofs &output) const;

    public:
        /// Off-diagonal row columns use the owned+ghost local indexing of the
        /// supplied DOF space. Rows must have no duplicates or self column.
        /// Core rows may be conservative, but must not reference a ghost input.
        /// Logical scans always use [core][noncore], preserving each input list
        /// order; physical arrays and MPI mappings are not reordered.
        LinearProblemContext(Algebra::DofSpace space, index components,
                             std::vector<std::vector<index>> columns,
                             std::vector<index> core, DiagonalStorage diagonal,
                             OffDiagonalStorage offDiagonal = OffDiagonalStorage::Assembled,
                             Runtime::Device device = Runtime::Device::Host(), LinearSweepPartitions partitions = {},
                             LinearMatrixOptions matrixOptions = {});
        LinearProblemContext(const LinearProblemContext &) = delete;
        LinearProblemContext &operator=(const LinearProblemContext &) = delete;
        LinearProblemContext(LinearProblemContext &&) noexcept = default;
        LinearProblemContext &operator=(LinearProblemContext &&) noexcept = default;
        const Algebra::DofSpace &Space() const
        {
            Layout();
            return _space;
        }
        index Components() const { return _components; }
        bool RowMajorBlocks() const { return _rowMajorBlocks; }
        bool HasNativeAMGXValues() const { return bool(_nativeAMGXValues); }
        /// Publish one immutable strength per off-diagonal CSR slot at setup.
        /// Synthetic operators without an FV mesh must supply their graph weights explicitly.
        void SetGeometricWeights(Runtime::Array<real> weights);
        const Runtime::Array<real> &GeometricWeights() const
        {
            DNDS_check_throw_info(_geometricWeights, "flow AMG requires geometric strengths matching its native CSR pattern");
            return *_geometricWeights;
        }
        const Runtime::Array<real> &NativeAMGXValues() const
        {
            DNDS_check_throw_info(_nativeAMGXValues, "linear matrix has no native AmgX packed allocation");
            return *_nativeAMGXValues;
        }
        Runtime::Device Device() const { return _diagonal.GetDevice(); }
        /// Prepare optional layout shadows for vector reads only. Primary state,
        /// matrix blocks and output remain unchanged. Call before queued actions;
        /// repeated selection of the current layout does not allocate or drain.
        void ConfigureInputLayout(LinearInputLayout layout);
        /// Explicit host numerical decision boundary for queued evaluated
        /// actions. Reduces device status, drains, then reports the first error.
        /// CPU/assembled providers have no additional status workspace.
        void CheckQueuedActions(Runtime::EagerEngine &, const Runtime::ExecutionStream &) const;
        /// Explicitly begin a fresh action segment after previous observation.
        /// Ordinary successful actions need no reset; errors remain sticky.
        void ResetQueuedActionStatus(Runtime::EagerEngine &, const Runtime::ExecutionStream &) const;
        DiagonalStorage DiagonalMode() const { return _diagonalStorage; }
        std::vector<std::vector<index>> Columns() const { return Layout().rows.CopyColumns(); }
        /// Native host symbolic publication, shared with a compatible provider.
        /// No index conversion, sorting or rank-local filtering is performed.
        const LinearPatternStorage &HostPattern() const { return Layout().hostRows ? *Layout().hostRows : Layout().rows; }
        const LinearPatternStorage &ResidentPattern() const { return Layout().rows; }
        /// Validate an immutable producer stencil once, then compare identity.
        /// Caller serializes preparation with context use. Retaining the token
        /// prevents address reuse; a new mesh/order supplies a new token.
        void ValidateColumns(const std::shared_ptr<const LinearPatternStorage> &columns)
        {
            if (_validatedColumns == columns && columns)
                return;
            DNDS_check_throw_info(columns && columns->CopyColumns() == Columns(), "linear producer stencil differs");
            _validatedColumns = columns;
        }
        std::vector<index> Order() const { return Layout().rows.CopyOrder(); }
        std::vector<index> Core() const { return Layout().rows.CopyCore(); }
        std::vector<index> HaloDependent() const { return Layout().rows.CopyDependent(); }
        void Validate() const { ValidateMatrix(); }
        Runtime::Array<real> &Diagonal() { return _diagonal; }
        const Runtime::Array<real> &Diagonal() const { return _diagonal; }
        OffDiagonalStorage OffDiagonalMode() const { return _offDiagonalStorage; }
        bool HasAssembledOffDiagonal() const { return bool(_offDiagonal); }
        /// Explicit storage access is unavailable in semi-matrix-free mode.
        Algebra::BlockMatrixStorage &OffDiagonal();
        const Algebra::BlockMatrixStorage &OffDiagonal() const;
        /// Numeric preparation only; serialize with all context access. The
        /// returned immutable producer may supply reusable geometry to the next
        /// preparation. Numerical consumers use ReadOffDiagonal instead.
        const std::shared_ptr<const OffDiagonalEvaluator> &PreparedOffDiagonalEvaluator() const { return _offDiagonalEvaluator; }
        /// Publish a fully frozen producer. A short exclusive diagonal lease
        /// guards this publication against active operators/SGS/factor reads.
        /// No diagonal value or generation is changed by this publication.
        void PublishOffDiagonalEvaluator(std::shared_ptr<const OffDiagonalEvaluator> evaluator);
        uint64_t OffDiagonalGeneration() const;
        /// Retains numerical storage or the frozen evaluator. Evaluated reads
        /// also pin the diagonal as publication guard. Retain this scope through
        /// all block use; no source means/communicators are retained by it.
        class OffDiagonalReadAccess
        {
            std::optional<Algebra::BlockMatrixStorage::ReadAccess> _assembled;
            std::optional<Runtime::Array<real>::ReadLease> _publication;
            std::shared_ptr<const OffDiagonalEvaluator> _evaluator;
            index _components;
            friend class LinearProblemContext;
            OffDiagonalReadAccess(Algebra::BlockMatrixStorage::ReadAccess assembled, index components)
                : _assembled(std::move(assembled)), _components(components) {}
            OffDiagonalReadAccess(Runtime::Array<real>::ReadLease publication, std::shared_ptr<const OffDiagonalEvaluator> evaluator, index components)
                : _publication(std::move(publication)), _evaluator(std::move(evaluator)), _components(components) {}

        public:
            /// Borrow numerical inputs; this scope must outlive every use.
            OffDiagonalView View() const
            {
                DNDS_check_throw_info(_assembled || _evaluator, "off-diagonal read scope is moved");
                return {_assembled ? std::optional(_assembled->View()) : std::nullopt,
                        _evaluator.get(), _components};
            }
            template <class Callback>
            void WithActions(Callback &&callback) const { View().WithActions(std::forward<Callback>(callback)); }
            template <class Callback>
            void WithBlocks(Callback &&callback) const { View().WithBlocks(std::forward<Callback>(callback)); }
        };
        /// Retain numeric matrix/publication storage in the supplied placement.
        /// View()/WithActions remain CPU-only; queued device operators consume
        /// their own retained whole-array views instead of Eigen block access.
        OffDiagonalReadAccess ReadOffDiagonal(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const;
        /// Explicit preparation of diagonal inverses. Block mode follows the
        /// active Euler guarded LU/SVD inversion policy. Later diagonal writes
        /// invalidate the inverses; active writers are rejected by read leases.
        /// refinementSteps=0 preserves that policy exactly. Values 1..3 opt
        /// into guarded residual corrections, preserving discarded SVD modes.
        void InvertDiagonal(int refinementSteps = 0);
        index LastInverseFallbacks() const { return _inverseFallbacks; }
        /// Number of blocks actually processed by the selected SVD path.
        index LastInverseSVDCells() const { return _inverseSVDCells; }
        /// Setup-only workspace/provider creation. Repeated calls on the same
        /// context reuse allocations. Do this before entering the solve loop.
        void PrepareDiagonalInverse(Runtime::ExecutionContext context,
                                    CUDADiagonalInverse kernel = CUDADiagonalInverse::BatchedSVD);
        /// Queue CPU inversion or CUDA scale/SVD/recovery/status reduction.
        /// The current inverse is invalid until ObserveDiagonalInverse succeeds.
        /// Every operand and provider is retained beyond this object's lifetime.
        void SubmitInvertDiagonal(Runtime::EagerEngine &, const Runtime::ExecutionStream &, int refinementSteps = 0);
        /// Explicit numerical observation after the controller drains the
        /// preparation segment. Only compact status is copied from CUDA.
        /// Solver failures take the normal rank-local abort path.
        void ObserveDiagonalInverse();
        /// Stateless matrix action. Core acquires no input ghost lease. Every
        /// output allocation must be distinct from inputs and numeric storage.
        void Apply(const Field &input, Field &output, LinearRegion region = LinearRegion::All) const;
        /// Coupled matrix action on logical blocks without whole-state packing.
        /// Input/output may group components differently; semantics, row space
        /// and device must agree. Core reads no ghost part. Caller retains the
        /// matrix and serializes preparation; return is synchronous on CPU.
        void Apply(const EquationDofs &input, EquationDofs &output, LinearRegion region = LinearRegion::All) const;
        /// Two-buffer block Jacobi: output=D^-1(rhs-offDiagonal*input).
        /// No ordered new-value reads or communication. Core consumes owned
        /// input only; caller refreshes ghosts before dependent/All work.
        /// Zero-start omits the input entirely, including its halo and reads.
        void Jacobi(const Field &rhs, const Field &input, Field &output, LinearRegion region = LinearRegion::All) const;
        void JacobiZero(const Field &rhs, Field &output, LinearRegion region = LinearRegion::All) const;
        void Jacobi(const EquationDofs &rhs, const EquationDofs &input, EquationDofs &output, LinearRegion region = LinearRegion::All) const;
        void JacobiZero(const EquationDofs &rhs, EquationDofs &output, LinearRegion region = LinearRegion::All) const;
        /** @brief Managed queued row actions using the same CPU/device schedule.
         * @details Reserve all numerical inputs/outputs at submission; retain
         * symbolic arrays and frozen evaluators through operation completion.
         * Core reads owned input only. Output is distinct and its halo remains
         * stale. CPU supports segmented layouts; CUDA requires combined AoS.
         * No transfer, stream join or host reduction is performed here.
         */
        void SubmitApply(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const Field &, Field &, LinearRegion = LinearRegion::All) const;
        void SubmitApply(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const EquationDofs &, EquationDofs &, LinearRegion = LinearRegion::All) const;
        void SubmitJacobi(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const Field &, const Field &, Field &, LinearRegion = LinearRegion::All, LinearKernelOptions = {}) const;
        void SubmitJacobi(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const EquationDofs &, const EquationDofs &, EquationDofs &, LinearRegion = LinearRegion::All, LinearKernelOptions = {}) const;
        /// Zero-start reserves neither old input nor off-diagonal values.
        void SubmitJacobiZero(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const Field &, Field &, LinearRegion = LinearRegion::All, LinearKernelOptions = {}) const;
        void SubmitJacobiZero(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const EquationDofs &, EquationDofs &, LinearRegion = LinearRegion::All, LinearKernelOptions = {}) const;
        /// Move-only access scope for one staged symmetric-GS application.
        /// Lower runs Core then HaloDependent; Upper runs HaloDependent then
        /// Core. All is available for either complete scan. Finish publishes
        /// the result only after both scans complete. Abandoning a started
        /// iteration invalidates unfinished outputs instead of publishing them.
        /// The scope retains pattern, allocations and numerical input leases;
        /// it remains valid even if the original context/field handles expire.
        class SGSIteration
        {
            struct State;
            std::unique_ptr<State> _state;
            explicit SGSIteration(std::unique_ptr<State> state);
            friend class LinearProblemContext;

        public:
            SGSIteration(const SGSIteration &) = delete;
            SGSIteration &operator=(const SGSIteration &) = delete;
            SGSIteration(SGSIteration &&) noexcept;
            SGSIteration &operator=(SGSIteration &&) noexcept;
            ~SGSIteration();
            void Lower(LinearRegion region = LinearRegion::All);
            void Upper(LinearRegion region = LinearRegion::All);
            void Finish();
        };
        /// Core lower work may start while input ghosts are being written.
        /// Complete the input halo before lower dependent work. By default
        /// remote old values remain frozen across both scans. If the explicit
        /// middleExchange option is true, caller exchanges completed forward
        /// values before Upper; that join is exposed. No MPI is posted here.
        /// Callers serialize phase calls and do not replace field handles or
        /// routing during the iteration. Synchronous CPU entry points only.
        SGSIteration BeginSGS(const Field &rhs, const Field &input, Field &forward, Field &output, bool middleExchange = false) const;
        /// Dedicated zero-start application: no input array, read lease, zero
        /// fill or zero test. A statically selected scan omits old increments.
        /// Local forward/upper terms remain; remote upper terms are zero
        /// unless middleExchange requests the freshly exchanged forward halo.
        SGSIteration BeginSGSZero(const Field &rhs, Field &forward, Field &output, bool middleExchange = false) const;
        /// Same scan/order/communication contract for logical equation fields.
        /// Owns numeric aliases and leases, not an MPI context. The controller
        /// retains communication owners and completes each declared halo phase.
        class DofSGSIteration
        {
            struct State;
            std::unique_ptr<State> _state;
            explicit DofSGSIteration(std::unique_ptr<State> state);
            friend class LinearProblemContext;

        public:
            DofSGSIteration(const DofSGSIteration &) = delete;
            DofSGSIteration &operator=(const DofSGSIteration &) = delete;
            DofSGSIteration(DofSGSIteration &&) noexcept;
            DofSGSIteration &operator=(DofSGSIteration &&) noexcept;
            ~DofSGSIteration();
            void Lower(LinearRegion region = LinearRegion::All);
            void Upper(LinearRegion region = LinearRegion::All);
            void Finish();
        };
        DofSGSIteration BeginSGS(const EquationDofs &rhs, const EquationDofs &input, EquationDofs &forward, EquationDofs &output, bool middleExchange = false) const;
        DofSGSIteration BeginSGSZero(const EquationDofs &rhs, EquationDofs &forward, EquationDofs &output, bool middleExchange = false) const;
        /// Two-buffer local symmetric GS application with distinct forward
        /// scratch. Frozen remote inputs are used throughout both scans; there
        /// is no MPI between lower and upper. This unsplit convenience owns all
        /// leases for the call; staged/communication orchestration is separate.
        void ApplySGS(const Field &rhs, const Field &input, Field &forward, Field &output) const;
        void ApplySGS(const EquationDofs &rhs, const EquationDofs &input, EquationDofs &forward, EquationDofs &output) const;
        /// Rank-local incomplete/full factor structure. Off-rank columns stay
        /// in Apply but are excluded from local factors. Core-first ordering
        /// is optional and precedes symbolic fill; local parts are independent.
        /// Select rank-local factor ordering before symbolic fill. RCM is
        /// computed within each independent local part, then optionally stably
        /// grouped core first. Physical mesh/DOF row order is unchanged.
        void PrepareILUStructure(int fill = 0, bool coreFirst = true, const std::vector<index> &localParts = {},
                                 const std::string &ordering = "natural");
        /// Copy current native coefficients into the prepared factor template,
        /// then factor numerically. No inverse/SGS state or MPI halo is changed.
        void FactorizeILU();
        bool ILUFactorsCurrent() const;
        /// Dependency-safe ready rows from the conservative input core through
        /// both symbolic scans. Complete LU provides no bounded-core promise.
        std::vector<index> ILUForwardCore() const;
        std::vector<index> ILUBackwardCore() const;
        /// One full local factor solve, no communication between scans. Parent
        /// matrix generations are checked while reads are pinned. RHS, forward
        /// scratch and output are separate native fields on the same row space.
        void ApplyILU(const Field &rhs, Field &forward, Field &output) const;
        /// Same local factors/order on logical blocks. Only one-row dense
        /// scratch is gathered at a time; no distributed state repacking.
        void ApplyILU(const EquationDofs &rhs, EquationDofs &forward, EquationDofs &output) const;
    };
}
