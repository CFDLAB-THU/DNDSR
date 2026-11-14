#pragma once

#include "DNDS/Algebra/DofAlgebra.hpp"

namespace DNDS::Algebra
{
    /** @brief Reusable managed physical-layout copy for an operator input.
     * @details Refresh is queued on the consumer stream. The key uses submission
     * mutation epochs, not completed versions, so an enqueued writer invalidates
     * a previous conversion. Refresh binds to its first live execution lane;
     * another lane rejects explicitly. The caller serializes this workspace
     * and orders cross-stream source producers and shadow consumers explicitly.
     * No persistent read lease is held.
     * The returned Array handle shares ownership; kernel submission must declare
     * its read access normally. Source and shadow remain distinct allocations.
     */
    class ArrayInputCache
    {
        Runtime::Array<real> _values;
        std::weak_ptr<const Runtime::ArrayStorage> _sourceStorage;
        Runtime::ExecutionStream _lane;
        std::uint64_t _source = 0, _epoch = 0, _shape = 0, _shadowEpoch = 0;
        bool _current = false;
        bool _bound = false;

    public:
        ArrayInputCache() = default;
        explicit ArrayInputCache(const Runtime::Array<real> &prototype, const std::string &name)
            : ArrayInputCache(prototype.Layout(), prototype.GetDevice(), name) {}
        ArrayInputCache(const Runtime::RowLayout &shape, Runtime::Device device, const std::string &name)
        {
            Runtime::UniformTableRows::Validate(shape);
            _values.Reset(Runtime::RowLayout::TiledColMajor(shape.RowCount(), shape.Width(), 32), device);
            _values.SetName(name);
        }
        /// Returns true when a conversion was submitted. Allocation belongs in
        /// operator preparation; Refresh never resizes or silently moves devices.
        bool Refresh(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                     const Runtime::Array<real> &source)
        {
            DNDS_check_throw_info(engine.Owns(stream) && source.GetDevice() == stream.Context().device &&
                                      _values.GetDevice() == source.GetDevice() &&
                                      source.RowCount() == _values.RowCount() && source.Layout().Width() == _values.Layout().Width(),
                                  "operator input cache shape, device or stream differs");
            DNDS_check_throw_info(!_bound || _lane.SameLane(stream),
                                  "operator input cache is bound to another execution lane");
            if (!_bound)
            {
                _lane = stream;
                _bound = true;
            }
            const auto identity = source.Storage().Identity();
            const auto epoch = source.MutationEpoch(), shape = source.ShapeVersion();
            if (_current && _sourceStorage.lock() == source.SharedStorage() &&
                _source == identity && _epoch == epoch && _shape == shape &&
                _shadowEpoch == _values.MutationEpoch())
                return false;
            Runtime::SubmitArrays(engine, stream, [](Runtime::ExecutionContext context, auto input, auto output)
                                  { ApplyRowViews(context, PointwiseOperation::Copy, output.DynamicView(), input.DynamicView()); }, Runtime::ReadAs<Runtime::UniformTableRows>(source), Runtime::WriteAs<Runtime::UniformTableRows>(_values));
            _source = identity;
            _sourceStorage = source.SharedStorage();
            _epoch = epoch;
            _shape = shape;
            _shadowEpoch = _values.MutationEpoch();
            _current = true;
            return true;
        }
        const Runtime::Array<real> &Values() const { return _values; }
    };
}
