#include "DNDS/Algebra/LocalProducts.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Core/OMP.hpp"
#include <cmath>

namespace DNDS::Algebra
{
    namespace R = Runtime;
    namespace
    {
        void ReduceHost(Detail::ProductInput x, Detail::ProductInput y, Detail::ProductInput scales,
                        Detail::ProductOutput partial, Detail::ProductOutput result, index slot, bool finite)
        {
            const index partitions = std::min<index>(partial.RowCount(), std::min<index>(OMP::DistributedThreads(), std::max<index>(1, x.RowCount())));
            OMP::ForRows(0, partitions, [&](index part)
                         {
            real value = finite ? 1 : 0;
            const index begin = part * (x.RowCount() / partitions) + std::min(part, x.RowCount() % partitions);
            const index end = begin + x.RowCount() / partitions + (part < x.RowCount() % partitions);
            for (index iRow = begin; iRow < end; ++iRow)
                for (index j = 0; j < x.Width(); ++j)
                    if (finite)
                        value = std::isfinite(x(iRow, j)) ? value : 0;
                    else
                    {
                        const real scale = scales.RowCount() ? scales(j, 0) : 1;
                        value += (x(iRow, j) * scale) * (y(iRow, j) * scale);
                    }
            partial(part, 0) = value; });
            real value = partial(0, 0);
            for (index part = 1; part < partitions; ++part)
                value = finite ? std::min(value, partial(part, 0)) : value + partial(part, 0);
            result(slot, 0) = value;
        }
    }
    LocalProductWorkspace::LocalProductWorkspace(index rows, index components, index capacity, R::Device device)
        : _rows(rows), _components(components), _device(device)
    {
        DNDS_check_throw_info(rows >= 0 && components > 0 && capacity > 0 && rows <= INT64_MAX / components,
                              "invalid local reduction shape");
        _scales = R::Array<real>(R::RowLayout::Dense(0, 1), device);
        const index count = rows * components;
        _partial = R::Array<real>(R::RowLayout::Dense(device == R::Device::Host() ? OMP::DistributedThreads() : std::max<index>(1, std::min<index>(256, count / 256 + (count % 256 != 0))), 1), device);
        _result = R::Array<real>(R::RowLayout::Dense(capacity, 1), device);
        _completed = R::Array<unsigned int>(R::RowLayout::Dense(device == R::Device::Host() ? 0 : 1, 1), device);
        _scales.SetName("DNDS/localProducts/inverseScales", DNDS_ALLOCATION_SITE());
        _partial.SetName("DNDS/localProducts/treePartials", DNDS_ALLOCATION_SITE());
        _result.SetName("DNDS/localProducts/results", DNDS_ALLOCATION_SITE());
        _completed.SetName("DNDS/localProducts/completedBlocks", DNDS_ALLOCATION_SITE());
    }
    void LocalProductWorkspace::SetInverseScales(const std::vector<real> &values, R::ExecutionContext context)
    {
        DNDS_check_throw_info(context.device == _device && (values.empty() || index(values.size()) == _components),
                              "local product scales placement or width differs");
        for (const auto value : values)
            DNDS_check_throw_info(std::isfinite(value) && value > 0, "invalid inverse product scale");
        if (_inverseScales == values)
            return;
        DNDS_check_throw_info(_scales.Storage().IsIdle() && _partial.Storage().IsIdle() && _result.Storage().IsIdle() && _completed.Storage().IsIdle(),
                              "cannot change reduction scales during queued use");
        R::Array<real> host(R::RowLayout::Dense(values.size(), 1));
        {
            auto write = host.Write();
            for (index j = 0; j < index(values.size()); ++j)
                write.View()(j, 0) = values[j];
        }
        _scales = host.To(_device, context);
        _scales.SetName("DNDS/localProducts/inverseScales", DNDS_ALLOCATION_SITE());
        _inverseScales = values;
    }
    void LocalProductWorkspace::Submit(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                       const R::Array<real> &x, const R::Array<real> &y, index slot, bool finite)
    {
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == _device && slot >= 0 && slot < _result.RowCount(),
                              "local reduction stream or result slot differs");
        for (const auto *input : {&x, &y})
            DNDS_check_throw_info(input->GetDevice() == _device && input->RowCount() == _rows && !input->Layout().IsVariable() &&
                                      input->Layout().Width() == _components && !input->Storage().Overlaps(_result.Storage()) &&
                                      !input->Storage().Overlaps(_partial.Storage()),
                                  "local reduction input shape, placement or alias differs");
        R::SubmitArrays(engine, stream, [slot, finite](R::ExecutionContext context, auto x, auto y, auto scales, auto partial, auto result, auto completed)
                        {
            if (context.device == R::Device::Host())
                ReduceHost(x.DynamicView(), y.DynamicView(), scales.DynamicView(), partial.DynamicView(), result.DynamicView(), slot, finite);
            else
                Detail::ReduceLocalProductCUDA(context, x.DynamicView(), y.DynamicView(), scales.DynamicView(),
                                               partial.DynamicView(), result.DynamicView(), completed, slot, finite); }, R::ReadAs<R::UniformTableRows>(x), R::ReadAs<R::UniformTableRows>(y), R::ReadAs<R::FixedRows<1>>(_scales), R::WriteAs<R::FixedRows<1>>(_partial), R::WriteAs<R::FixedRows<1>>(_result), R::WriteAs<R::FixedRows<1>>(_completed));
    }
    void LocalProductWorkspace::SubmitProduct(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                              const R::Array<real> &x, const R::Array<real> &y, index slot)
    {
        Submit(engine, stream, x, y, slot, false);
    }
    void LocalProductWorkspace::SubmitFinite(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                             const R::Array<real> &x, index slot)
    {
        Submit(engine, stream, x, x, slot, true);
    }
    std::vector<real> LocalProductWorkspace::ReadResults(index count, R::ExecutionContext context) const
    {
        DNDS_check_throw_info(context.device == _device && count >= 0 && count <= _result.RowCount(), "local reduction result extent differs");
        const auto host = _result.To(R::Device::Host(), context);
        const auto read = host.Read();
        std::vector<real> values(count);
        for (index i = 0; i < count; ++i)
            values[i] = read.View()(i, 0);
        return values;
    }
#ifndef DNDS_USE_CUDA
    void Detail::ReduceLocalProductCUDA(R::ExecutionContext, ProductInput, ProductInput, ProductInput, ProductOutput, ProductOutput, ProductCompletion, index, bool)
    {
        DNDS_check_throw_info(false, "CUDA local reductions are not compiled");
    }
#endif
}
