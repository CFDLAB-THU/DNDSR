#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "CFV/Detail/ReconstructionData.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    PolynomialSamplingPlan::PolynomialSamplingPlan(VariationalReconstructionPlan reconstruction,
                                                   const R::Array<index> &cells, const R::Array<real> &points, bool cacheDerivatives,
                                                   R::ExecutionContext sourceContext)
        : PolynomialSamplingPlan(reconstruction.Metrics(), reconstruction, cells, points, cacheDerivatives, sourceContext) {}
    PolynomialSamplingPlan::PolynomialSamplingPlan(FiniteVolumePlan metrics, VariationalReconstructionPlan reconstruction,
                                                   const R::Array<index> &cells, const R::Array<real> &points, bool cacheDerivatives,
                                                   R::ExecutionContext sourceContext)
        : PolynomialSamplingPlan(metrics, reconstruction, cells, points, cacheDerivatives, sourceContext, std::nullopt) {}
    PolynomialSamplingPlan::PolynomialSamplingPlan(FiniteVolumePlan metrics, VariationalReconstructionPlan reconstruction,
                                                   const R::Array<index> &cells, const R::Array<real> &points, bool cacheDerivatives,
                                                   R::ExecutionContext sourceContext, std::optional<SharedBasisInput> sharedBasis)
        : _mesh(metrics._data->mesh), _stamp(metrics._data->stamp),
          _cellSpace(metrics.Cells()), _space(metrics.Space()),
          _basisRows(R::RowLayout::Dense(0, 1)), _outputRows(R::RowLayout::Dense(0, 1)),
          _points(R::RowLayout::Dense(0, 3)), _basis(R::RowLayout::Dense(0, 0)),
          _moments(R::RowLayout::Dense(0, 0)), _cellGeometry(R::RowLayout::Dense(0, 8)), _empty(R::RowLayout::Dense(0, 0)),
          _dimension(metrics.Dimension()), _degree(reconstruction.Settings().maxOrder),
          _modes(reconstruction.NRecDOF()), _maximumDerivative(cacheDerivatives ? _modes : 0)
    {
        DNDS_check_throw_info(reconstruction.Space().Same(metrics.Space()) &&
                                  (reconstruction.Device() == R::Device::Host() || sourceContext.device == reconstruction.Device()) &&
                                  (metrics.Device() == R::Device::Host() || sourceContext.device == metrics.Device()),
                              "sampling setup needs matching metric space and source context");
        auto geometry = metrics.AcquireRead(metrics.Device() == R::Device::Host() ? R::ExecutionContext::Host() : sourceContext);
        const auto &sourceMoments = reconstruction.SamplingMoments();
        const auto hostMoments = sourceMoments.GetDevice() == R::Device::Host() ? sourceMoments : sourceMoments.To(R::Device::Host(), sourceContext);
        const auto hostCells = metrics.Device() == R::Device::Host() ? metrics._data->cell : metrics._data->cell.To(R::Device::Host(), sourceContext);
        Validate();
        DNDS_check_throw_info(cells.GetDevice() == R::Device::Host() && points.GetDevice() == R::Device::Host() &&
                                  cells.Layout().Kind() == R::RowLayoutKind::Table && cells.Layout().Width() == 1 &&
                                  points.Layout().Kind() == R::RowLayoutKind::Table && points.Layout().Width() == 3 &&
                                  cells.RowCount() == points.RowCount(),
                              "polynomial sampling input shape/placement mismatch");
        const Math::TaylorBasis polynomial(_dimension, _degree);
        const int cacheOrder = cacheDerivatives ? reconstruction.Settings().cacheDiffBaseOrder : 0;
        _cachedDerivatives = sharedBasis ? sharedBasis->derivatives : reconstruction.Settings().cacheDiffBase ? polynomial.DerivativeNumber(cacheOrder)
                                                                                                              : 0;
        _ownsBasis = !sharedBasis;
        DNDS_check_throw_info(_cachedDerivatives >= 0 && _cachedDerivatives <= _modes + 1,
                              "sampling cache derivative count outside polynomial space");
        const bool direct = _cachedDerivatives <= _maximumDerivative;
        _cells = R::Array<index>(R::RowLayout::Dense(cells.RowCount(), 1));
        if (sharedBasis)
        {
            const auto &source = sharedBasis->basis;
            const auto &rows = sharedBasis->rows;
            DNDS_check_throw_info(source.GetDevice() == R::Device::Host() && source.Layout().Kind() == R::RowLayoutKind::Table &&
                                      rows.GetDevice() == R::Device::Host() && rows.Layout().Kind() == R::RowLayoutKind::Table &&
                                      rows.Layout().Width() == 1 && (rows.RowCount() == 0 || rows.RowCount() == cells.RowCount()) &&
                                      (!_cachedDerivatives || source.Layout().Width() == _modes * _cachedDerivatives),
                                  "shared sampling basis/map shape or placement mismatch");
            if (_cachedDerivatives)
            {
                _basis = source;
                const auto read = rows.Read();
                const auto map = read.View<R::FixedRows<1>>();
                OMP::ForSetupRows(0, cells.RowCount(), [&](index iPoint)
                                  {
                    const index row = rows.RowCount() ? map(iPoint, 0) : iPoint;
                    DNDS_check_throw_info(row >= 0 && row < source.RowCount(), "shared sampling basis row outside cache"); });
                if (rows.RowCount())
                {
                    _basisRows = R::Array<index>(R::RowLayout::Dense(rows.RowCount(), 1));
                    auto write = _basisRows.Write();
                    const auto snapshot = write.View<R::FixedRows<1>>();
                    OMP::ForRows(0, rows.RowCount(), [&](index iPoint)
                                 { snapshot(iPoint, 0) = map(iPoint, 0); });
                }
            }
        }
        else if (_cachedDerivatives)
            _basis = R::Array<real>(R::RowLayout::Dense(cells.RowCount(), _modes * _cachedDerivatives));
        if (direct)
        {
            _points = R::Array<real>(R::RowLayout::Dense(cells.RowCount(), 3));
            _moments = hostMoments;
            _cellGeometry = hostCells;
        }
        const auto cellRead = cells.Read();
        const auto pointRead = points.Read();
        const auto ids = cellRead.View<R::FixedRows<1>>();
        const auto inputPoints = pointRead.View<R::FixedRows<3>>();
        // The boolean publication is controller work; all validation and
        // polynomial fill runs in independent rows below. No shared OR race.
        for (index iPoint = 0; iPoint < ids.RowCount() && !_needsGhost; ++iPoint)
            _needsGhost = ids(iPoint, 0) >= _cellSpace.NumOwned();
        {
            auto cellWrite = _cells.Write();
            auto pointWrite = _points.Write();
            std::optional<R::Array<real>::WriteLease> basisWrite;
            if (_ownsBasis)
                basisWrite.emplace(_basis.Write());
            const auto snapshot = cellWrite.View<R::FixedRows<1>>();
            const auto positions = pointWrite.View<R::FixedRows<3>>();
            const auto cached = basisWrite ? basisWrite->View<R::TableRows>() : R::ArrayView<real, R::AccessBoundsChecks, R::TableRows>{};
            const auto momentRead = hostMoments.Read();
            const auto moments = momentRead.View<R::TableRows>();
            const auto cellReadGeometry = hostCells.Read();
            FiniteVolumeHostView geom;
            geom.cells = cellReadGeometry.View<R::FixedRows<8>, false>();
            OMP::ForSetupRows(0, ids.RowCount(), [&](index iPoint)
                              {
            const index iCell = ids(iPoint, 0);
            DNDS_check_throw_info(iCell >= 0 && iCell < _cellSpace.NumLocal() &&
                                      std::isfinite(inputPoints(iPoint, 0)) && std::isfinite(inputPoints(iPoint, 1)) && std::isfinite(inputPoints(iPoint, 2)),
                                  "invalid sampling cell or point");
            snapshot(iPoint, 0) = iCell;
            const Geom::tPoint point{inputPoints(iPoint, 0), inputPoints(iPoint, 1), inputPoints(iPoint, 2)};
            if (direct)
                for (int d = 0; d < 3; ++d)
                    positions(iPoint, d) = point[d];
            if (_cachedDerivatives && _ownsBasis)
            {
                Math::SmallMatrix full;
                polynomial.Reconstruction(full, point, geom.GetCellCent(iCell), geom.cells(iCell, 7), moments[iCell], cacheOrder);
                Eigen::Map<Eigen::MatrixXd>(cached[iPoint].data(), _cachedDerivatives, _modes) = full;
            } });
        }
        NameOwnedBuffers();
        FreezeBasis();
    }
    void PolynomialSamplingPlan::Validate() const
    {
        DNDS_check_throw_info(_mesh && _stamp.Valid(), "polynomial sampling geometry or numbering changed");
        _cellSpace.Validate();
        if (_basisFrozen)
            DNDS_check_throw_info(_basisState.Matches(_basis) && _momentsState.Matches(_moments) &&
                                      _cellGeometryState.Matches(_cellGeometry),
                                  "immutable sampling basis or normalization was mutated through an alias; rebuild the publication");
    }
    void PolynomialSamplingPlan::FreezeBasis()
    {
        _basisState.Capture(_basis);
        _momentsState.Capture(_moments);
        _cellGeometryState.Capture(_cellGeometry);
        _basisFrozen = true;
    }
    void PolynomialSamplingPlan::ValidateField(const FVField &field, index width) const
    {
        Validate();
        field.Validate();
        DNDS_check_throw_info(field.Space().Same(_space) && index(field.Width()) == width && field.Device() == Device(),
                              "sampling field shape, row space or placement mismatch");
    }
    PolynomialSamplingPlan PolynomialSamplingPlan::Placed(R::Device device, R::ExecutionContext context) const
    {
        SamplingPlacementMemo memo(device, context);
        return Placed(memo);
    }
    PolynomialSamplingPlan PolynomialSamplingPlan::Placed(SamplingPlacementMemo &memo) const
    {
        Validate();
        auto result = *this;
        result._cells = memo.Place(_cells);
        result._basisRows = memo.Place(_basisRows);
        result._outputRows = memo.Place(_outputRows);
        result._points = memo.Place(_points);
        result._basis = memo.Place(_basis);
        result._moments = memo.Place(_moments);
        result._cellGeometry = memo.Place(_cellGeometry);
        result._empty = memo.Place(_empty);
        result.NameOwnedBuffers();
        result.FreezeBasis();
        result.Validate();
        return result;
    }
}
