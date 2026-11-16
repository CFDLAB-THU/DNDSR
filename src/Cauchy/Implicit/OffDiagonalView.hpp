#pragma once
#include "DNDS/Algebra/BlockMatrixStorage.hpp"
#include "DNDS/Algebra/DofField.hpp"
#include "DNDS/Core/NumericalStatus.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace DNDS::Runtime
{
    class EagerEngine;
    class ExecutionStream;
}

namespace DNDS::Cauchy
{
    /** @brief Managed operands of one queued evaluated row action.
     * @details Array copies share allocation/layout ownership; no payload is
     * copied. rhs selects Jacobi and diagonal then holds the inverse. A missing
     * ghost input denotes a proved owned-only selection. status is caller
     * scratch, retained through completion and observed at its normal numerical
     * decision boundary. It is not hidden mutable state in the evaluator.
     */
    struct EvaluatedLinearAction
    {
        Runtime::Array<index> rows;
        Runtime::Array<real> diagonal, owned, output;
        std::optional<Runtime::Array<real>> rhs, ghost;
        Runtime::Array<NumericalStatus> status;
        bool fullDiagonal = true;
        bool rowMajorBlocks = false;
    };
    /// Immutable physical block producer. Columns and block ordinals belong to
    /// LinearProblemContext. Implementations own frozen numerical inputs; they
    /// must not borrow live means, perform communication or mutate hidden caches.
    /// Blocks use column-major component order, including periodic transforms.
    class OffDiagonalEvaluator
    {
    public:
        using InputVector = Eigen::Ref<const Eigen::VectorXd, 0, Eigen::InnerStride<Eigen::Dynamic>>;
        virtual ~OffDiagonalEvaluator() = default;
        /// Device selection is explicit. CPU producers retain their block-level
        /// arithmetic; a resident provider must implement the whole-row queued
        /// entry. There is no implicit host copy or Eigen/device fallback.
        virtual Runtime::Device Device() const { return Runtime::Device::Host(); }
        virtual void SubmitRows(Runtime::EagerEngine &, const Runtime::ExecutionStream &, EvaluatedLinearAction) const
        {
            DNDS_check_throw_info(false, "evaluated Jacobian has no queued device row provider");
        }
        virtual index Components() const = 0;
        virtual std::shared_ptr<const Algebra::DofSpace::Partition> RowSpaceIdentity() const = 0;
        /// Setup/diagnostic copy; never called by block arithmetic.
        virtual std::vector<std::vector<index>> Columns() const = 0;
        virtual index RowNumber() const = 0;
        virtual index BlockNumber(index iRow) const = 0;
        /// Borrow one immutable CSR column row without reconstructing a host
        /// vector. The retained evaluator owns this range through publication.
        virtual SpanView<const index> ColumnRow(index iRow) const = 0;
        virtual void EvaluateBlock(index iRow, index iBlock, Eigen::Ref<Eigen::MatrixXd> output) const = 0;
        /// Apply one frozen off-diagonal block directly, without constructing
        /// a components-by-components temporary. This contract has no dense
        /// fallback: every producer supplies its directional action. Input/output are
        /// distinct vectors, owned by the enclosing kernel scope. A strided
        /// input borrows SoA/AoSoA rows without hidden Eigen packing; output
        /// is contiguous caller scratch, allocated once before the scan.
        virtual void ApplyBlock(index iRow, index iBlock, InputVector input,
                                Eigen::Ref<Eigen::VectorXd> output) const = 0;
    };

    /** @brief Borrowed off-diagonal producer used only inside numerical kernels.
     * @details ReadOffDiagonal() retains the assembled arrays or immutable
     * evaluator for the entire call. This descriptor acquires no lease and
     * performs no communication. The evaluator pointer is a low-level borrow;
     * it must not escape the enclosing operator's retained access scope.
     */
    struct OffDiagonalView
    {
        std::optional<Algebra::BlockMatrixView<const real>> assembled;
        const OffDiagonalEvaluator *evaluator = nullptr;
        index components = 0;

    private:
        template <int Size>
        struct EvaluatedBlocks
        {
            const OffDiagonalEvaluator *evaluator;
            index components;
            Eigen::Matrix<real, Size, Size> operator()(index iRow, index iBlock) const
            {
                if constexpr (Runtime::AccessBoundsChecks)
                {
                    DNDS_assert(iRow >= 0 && iRow < evaluator->RowNumber() && iBlock >= 0 && iBlock < evaluator->BlockNumber(iRow));
                }
                Eigen::Matrix<real, Size, Size> result(components, components);
                evaluator->EvaluateBlock(iRow, iBlock, result);
                return result;
            }
        };
        struct EvaluatedActions
        {
            static constexpr bool needs_scratch = true;
            const OffDiagonalEvaluator *evaluator;
            template <int Size, bool Subtract, class Input, class Output, class Scratch>
            void Accumulate(index iRow, index iBlock, const Input &input, Output &output, Scratch &scratch) const
            {
                evaluator->ApplyBlock(iRow, iBlock, input, scratch);
                if constexpr (Subtract)
                    output -= scratch;
                else
                    output += scratch;
            }
        };
        template <class Blocks>
        struct AssembledActions
        {
            static constexpr bool needs_scratch = false;
            Blocks blocks;
            template <int Size, bool Subtract, class Input, class Output, class Scratch>
            void Accumulate(index iRow, index iBlock, const Input &input, Output &output, Scratch &) const
            {
                const auto add = [&](const auto &block)
                {
                    if constexpr (Subtract)
                        output.noalias() -= block * input;
                    else
                        output.noalias() += block * input;
                };
                if constexpr (Size != Eigen::Dynamic)
                    add(blocks(iRow, iBlock).template topLeftCorner<Size, Size>());
                else
                    add(blocks(iRow, iBlock));
            }
        };

    public:
        /// Resolve storage once outside scans. SGS/Jv request an action,
        /// while ILU continues to request explicit blocks with WithBlocks.
        template <class Callback>
        void WithActions(Callback &&callback) const
        {
            DNDS_check_throw_info(assembled || evaluator, "off-diagonal read scope is moved");
            if (assembled)
            {
                assembled->WithLayout([&](const auto &blocks)
                                      { callback(AssembledActions<std::decay_t<decltype(blocks)>>{blocks}); });
            }
            else
                callback(EvaluatedActions{evaluator});
        }
        /// Dispatch once outside loops. The callback receives a callable
        /// (row,block): assembled mode returns the ordinary stable map;
        /// evaluated mode returns an owning block value, with flow5 on the
        /// stack. Querying another block never overwrites a previous result.
        /// Other component counts use the generic dynamic dense matrix.
        template <class Callback>
        void WithBlocks(Callback &&callback) const
        {
            DNDS_check_throw_info(assembled || evaluator, "off-diagonal read scope is moved");
            if (assembled)
                assembled->WithLayout(std::forward<Callback>(callback));
            else if (components == 5)
                callback(EvaluatedBlocks<5>{evaluator, components});
            else
                callback(EvaluatedBlocks<Eigen::Dynamic>{evaluator, components});
        }
    };
}
