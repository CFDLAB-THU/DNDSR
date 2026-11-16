#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "DNDS/Core/OMP.hpp"
#include <algorithm>

namespace DNDS::CFV
{
    namespace R = Runtime;

    PolynomialSamplingPlan PolynomialSamplingPlan::FromSharedBasis(
        FiniteVolumePlan metrics, VariationalReconstructionPlan reconstruction,
        const R::Array<index> &cells, const R::Array<real> &points,
        const R::Array<real> &basis, const R::Array<index> &rows,
        int derivatives, bool cacheDerivatives, R::ExecutionContext sourceContext)
    {
        return PolynomialSamplingPlan(metrics, reconstruction, cells, points, cacheDerivatives,
                                      sourceContext, SharedBasisInput{basis, rows, derivatives});
    }

    PolynomialSamplingPlan PolynomialSamplingPlan::WithOutputRows(const R::Array<index> &rows, index totalRows) const
    {
        Validate();
        DNDS_check_throw_info(Device() == R::Device::Host() && rows.GetDevice() == R::Device::Host() &&
                                  rows.Layout().Kind() == R::RowLayoutKind::Table && rows.Layout().Width() == 1 &&
                                  rows.RowCount() == QueryNumber() && totalRows >= QueryNumber(),
                              "sampling output scatter map needs host query rows and sufficient output extent");
        auto result = *this;
        result._outputSize = totalRows;
        result._outputRows = R::Array<index>(R::RowLayout::Dense(QueryNumber(), 1));
        const auto read = rows.Read();
        const auto input = read.View<R::FixedRows<1>>();
        auto write = result._outputRows.Write();
        const auto output = write.View<R::FixedRows<1>>();
        std::vector<index> sorted(static_cast<size_t>(QueryNumber()));
        OMP::ForSetupRows(0, QueryNumber(), [&](index iPoint)
                          {
            const index row = input(iPoint, 0);
            DNDS_check_throw_info(row >= 0 && row < totalRows, "sampling output scatter row outside output extent");
            output(iPoint, 0) = row;
            sorted[iPoint] = row; });
        // Canonical quadrature segments already follow increasing output rows.
        // Avoid sorting millions of ordinals for that normal setup path.
        if (!std::is_sorted(sorted.begin(), sorted.end()))
            std::sort(sorted.begin(), sorted.end());
        DNDS_check_throw_info(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
                              "sampling output scatter destinations must be unique within a segment");
        result._outputRows.SetName(_name + "/outputRows", DNDS_ALLOCATION_SITE());
        return result;
    }

    PolynomialSamplingPlan PolynomialSamplingPlan::WithDerivativeOrder(int order) const
    {
        Validate();
        DNDS_check_throw_info(order >= 0 && order <= _degree, "sampling derivative order outside polynomial degree");
        const int derivatives = Math::TaylorBasis(_dimension, _degree).DerivativeNumber(order);
        DNDS_check_throw_info(derivatives - 1 <= _maximumDerivative, "sampling derivative restriction cannot widen the original contract");
        auto result = *this;
        result._maximumDerivative = derivatives - 1;
        if (_cachedDerivatives >= derivatives && (_points.RowCount() || _moments.RowCount() || _cellGeometry.RowCount()))
        {
            // These are metadata-only empty arrays on the same backend. Basis
            // and maps retain their original allocations; no payload is copied.
            result._points = R::Array<real>(R::RowLayout::Dense(0, 3), Device());
            result._moments = R::Array<real>(R::RowLayout::Dense(0, 0), Device());
            result._cellGeometry = R::Array<real>(R::RowLayout::Dense(0, 8), Device());
        }
        result.NameOwnedBuffers();
        result.FreezeBasis();
        return result;
    }

    void PolynomialSamplingPlan::NameOwnedBuffers() const
    {
        _cells.SetName(_name + "/cells", DNDS_ALLOCATION_SITE());
        _points.SetName(_name + "/points", DNDS_ALLOCATION_SITE());
        _basisRows.SetName(_name + "/basisRows", DNDS_ALLOCATION_SITE());
        _outputRows.SetName(_name + "/outputRows", DNDS_ALLOCATION_SITE());
        if (_ownsBasis)
            _basis.SetName(_name + "/basis", DNDS_ALLOCATION_SITE());
        _empty.SetName(_name + "/empty", DNDS_ALLOCATION_SITE());
        // Moments and cell geometry retain their shared owner names. Borrowed
        // face/source basis keeps its publication name, regardless of consumers.
    }

    PolynomialSamplingPlan PolynomialSamplingPlan::Named(std::string name) const
    {
        DNDS_check_throw_info(!name.empty(), "sampling diagnostic name must not be empty");
        auto result = *this;
        result._name = std::move(name);
        result.NameOwnedBuffers();
        return result;
    }
}
