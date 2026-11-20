#include "Cauchy/Implicit/FrozenJacobianOperator.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        template <class T>
        const T &RequireOwner(const std::shared_ptr<const T> &owner)
        {
            DNDS_check_throw_info(owner, "frozen Jacobian requires live symbolic owners");
            return *owner;
        }
    }
    FrozenJacobianOperator::Geometry::Geometry(std::shared_ptr<const LinearPatternStorage> hostColumns,
                                               std::shared_ptr<const JacobianGeometryStorage> residentGeometry, R::ExecutionContext context)
        : _columns(std::move(hostColumns)), _resident(std::move(residentGeometry)),
          _columnRead(RequireOwner(_columns).Read()), _residentRead(RequireOwner(_resident).Read({context.device, nullptr}))
    {
        DNDS_check_throw_info(_columns->CopyColumns() == _resident->CopyColumns(),
                              "frozen Jacobian resident geometry differs from its publication column order");
    }
    FrozenJacobianOperator::FrozenJacobianOperator(std::shared_ptr<const Algebra::DofSpace::Partition> partition,
                                                   std::shared_ptr<const Geometry> geometry,
                                                   Values values, CUDA::FrozenJacobianParameters parameters, R::ExecutionContext context)
        : _partition(std::move(partition)), _geometry(std::move(geometry)),
          _values(std::move(values)), _means(_values.means.Read({context.device, nullptr})), _pressure(_values.pressure.Read({context.device, nullptr})),
          _derivative(_values.pressureDerivative.Read({context.device, nullptr})), _radius(_values.radius.Read({context.device, nullptr})),
          _magnitudes(_values.magnitudes.Read({context.device, nullptr})), _empty(R::RowLayout::Dense(0, 0), context.device),
          _parameters(parameters), _device(context.device)
    {
        DNDS_check_throw_info(_partition && _geometry && _device.space == R::MemorySpace::CUDA && parameters.components >= 5 && parameters.components <= 7,
                              "frozen Jacobian requires a CUDA 5/6/7 profile");
        const index rows = index(_partition->owned), all = rows + index(_partition->ghost);
        _geometry->_residentRead.View().ValidateDevice(_device);
        DNDS_check_throw_info(RowNumber() == rows && _geometry->_residentRead.View().columns.RowCount() == rows &&
                                  _values.means.RowCount() == all && _values.pressure.RowCount() == all && _values.pressureDerivative.RowCount() == all &&
                                  _values.means.Layout().Width() == size_t(parameters.components) && _values.pressureDerivative.Layout().Width() == size_t(parameters.components) &&
                                  _values.pressure.Layout().Width() == 1 && _values.radius.Layout().Width() == 2 &&
                                  (!parameters.roe || parameters.components == 5),
                              "frozen Jacobian publication shape differs");
        DNDS_check_throw_info(_values.means.Layout().Kind() == R::RowLayoutKind::Table ||
                                  (_values.means.Layout().Kind() == R::RowLayoutKind::TiledColMajorTable && _values.means.Layout().Stride() == 32),
                              "frozen Jacobian means require AoS or tiled32");
        for (const auto *array : {&_values.pressure, &_values.pressureDerivative, &_values.radius, &_values.magnitudes})
            DNDS_check_throw_info(array->Layout().Kind() == R::RowLayoutKind::Table, "frozen Jacobian publication requires AoS tables");
    }
    SpanView<const index> FrozenJacobianOperator::ColumnRow(index iRow) const
    {
        const auto row = _geometry->_columnRead.View().columns[iRow];
        DNDS_check_throw_info(!row.narrow, "evaluated Jacobian column span requires its native wide geometry");
        return {row.wide, row.size()};
    }
    void FrozenJacobianOperator::EvaluateBlock(index, index, Eigen::Ref<Eigen::MatrixXd>) const
    {
        DNDS_check_throw_info(false, "resident evaluated Jacobian requires queued whole-row execution");
    }
    void FrozenJacobianOperator::ApplyBlock(index, index, InputVector, Eigen::Ref<Eigen::VectorXd>) const
    {
        DNDS_check_throw_info(false, "resident evaluated Jacobian requires queued whole-row execution");
    }
    void FrozenJacobianOperator::SubmitRows(R::EagerEngine &engine, const R::ExecutionStream &stream, EvaluatedLinearAction action) const
    {
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == _device, "frozen Jacobian stream/device differs");
        const bool jacobi = bool(action.rhs), full = action.fullDiagonal, rowMajorBlocks = action.rowMajorBlocks;
        const auto check = [&](const R::Array<real> &input)
        {
            DNDS_check_throw_info(input.GetDevice() == _device && !action.output.Storage().Overlaps(input.Storage()),
                                  "frozen Jacobian action residency or output alias differs");
        };
        check(action.diagonal);
        check(action.owned);
        if (action.ghost)
            check(*action.ghost);
        if (action.rhs)
            check(*action.rhs);
        for (const auto *input : {&_values.means, &_values.pressure, &_values.pressureDerivative, &_values.radius, &_values.magnitudes})
            check(*input);
        const auto retained = shared_from_this();
        R::SubmitArrays(engine, stream, [retained, jacobi, full, rowMajorBlocks, selection = action.selection](R::ExecutionContext context, auto means, auto pressure, auto derivative, auto radius, auto magnitudes, auto diagonal, auto rhs, auto owned, auto ghost, auto rows, auto output, auto status)
                        {
#ifdef DNDS_USE_CUDA
                            const CUDA::FrozenJacobianInput snapshot{retained->_geometry->_residentRead.View(), means, pressure, derivative, radius, magnitudes, retained->_parameters};
                            CUDA::ApplyFrozenJacobian(context, snapshot, {selection, rows}, diagonal, full, rhs, owned, ghost, output, status, jacobi, rowMajorBlocks);
#else
                            DNDS_check_throw_info(false, "frozen CUDA Jacobian provider is not compiled");
#endif
                        },
                        R::ReadAs<R::UniformTableRows>(_values.means), R::ReadAs<R::TableRows>(_values.pressure), R::ReadAs<R::TableRows>(_values.pressureDerivative), R::ReadAs<R::TableRows>(_values.radius), R::ReadAs<R::TableRows>(_values.magnitudes), R::ReadAs<R::UniformTableRows>(action.diagonal), R::ReadAs<R::UniformTableRows>(action.rhs ? *action.rhs : _empty), R::ReadAs<R::UniformTableRows>(action.owned), R::ReadAs<R::UniformTableRows>(action.ghost ? *action.ghost : _empty), R::ReadAs<R::FixedRows<1>>(action.rows), R::WriteAs<R::TableRows>(action.output), R::ReadWriteAs<R::FixedRows<1>>(action.status));
    }
}
