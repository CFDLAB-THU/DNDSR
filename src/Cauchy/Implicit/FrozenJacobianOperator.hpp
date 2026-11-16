#pragma once

#include "Cauchy/Implicit/OffDiagonalView.hpp"
#include "Cauchy/Implicit/LinearPattern.hpp"
#include "Cauchy/Implicit/CUDA/FrozenJacobianKernels.hpp"

namespace DNDS::Cauchy
{
    /** @brief Immutable resident evaluated Jacobian over retained FV geometry.
     * @details Preparation finishes and validates before construction. The
     * retained read scopes seal numerical arrays and geometry against writes.
     * Only canonical column metadata stays on host for publication checks;
     * numerical payloads have one resident allocation, with no host mirror.
     * Read seals use the default device context so they cannot retain a raw
     * preparation-stream address past its owner's destruction. Actual queued
     * uses retain the publication until their own stream completion.
     * All asynchronous uses retain this operation object plus their explicit
     * I/O operands. Shared immutable stencil/CFV publications stay alive with
     * geometry; no MPI operation, mutable EOS worker or live equation field
     * participates in an evaluated action.
     * The initial provider supports CUDA combined-AoS perfect-gas 5/6/7 only.
     * Its host block methods reject use rather than performing device copies.
     */
    class FrozenJacobianOperator final : public OffDiagonalEvaluator,
                                         public std::enable_shared_from_this<FrozenJacobianOperator>
    {
    public:
        /// Validated once during symbolic setup and shared across numerical
        /// publications. Comparison may copy device indices to host here; it
        /// never copies a numerical matrix or repeats in a linear action.
        class Geometry
        {
            std::shared_ptr<const LinearPatternStorage> _columns;
            std::shared_ptr<const JacobianGeometryStorage> _resident;
            LinearPatternStorage::ReadAccess _columnRead;
            JacobianGeometryStorage::ReadAccess _residentRead;
            friend class FrozenJacobianOperator;

        public:
            Geometry(std::shared_ptr<const LinearPatternStorage> hostColumns,
                     std::shared_ptr<const JacobianGeometryStorage> residentGeometry,
                     Runtime::ExecutionContext context);
        };
        struct Values
        {
            Runtime::Array<real> means, pressure, pressureDerivative, radius, magnitudes;
        };

    private:
        std::shared_ptr<const Algebra::DofSpace::Partition> _partition;
        std::shared_ptr<const Geometry> _geometry;
        Values _values;
        Runtime::Array<real>::ReadLease _means, _pressure, _derivative, _radius, _magnitudes;
        Runtime::Array<real> _empty;
        CUDA::FrozenJacobianParameters _parameters;
        Runtime::Device _device;

    public:
        FrozenJacobianOperator(std::shared_ptr<const Algebra::DofSpace::Partition> partition,
                               std::shared_ptr<const Geometry> geometry,
                               Values values, CUDA::FrozenJacobianParameters parameters,
                               Runtime::ExecutionContext context);
        Runtime::Device Device() const override { return _device; }
        index Components() const override { return _parameters.components; }
        std::shared_ptr<const Algebra::DofSpace::Partition> RowSpaceIdentity() const override { return _partition; }
        std::vector<std::vector<index>> Columns() const override { return _geometry->_columns->CopyColumns(); }
        index RowNumber() const override { return _geometry->_columnRead.View().ownedRows; }
        index BlockNumber(index iRow) const override { return _geometry->_columnRead.View().columns.RowSize(iRow); }
        SpanView<const index> ColumnRow(index iRow) const override;
        void EvaluateBlock(index, index, Eigen::Ref<Eigen::MatrixXd>) const override;
        void ApplyBlock(index, index, InputVector, Eigen::Ref<Eigen::VectorXd>) const override;
        void SubmitRows(Runtime::EagerEngine &, const Runtime::ExecutionStream &, EvaluatedLinearAction) const override;
    };
}
