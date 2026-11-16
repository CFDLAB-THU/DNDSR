#pragma once

#include "DNDS/Algebra/DofField.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Core/OMP.hpp"
#include <cmath>
#include <array>
#include <functional>
#include <limits>
#include <optional>

namespace DNDS::Algebra
{
    namespace Detail
    {
        enum class Pointwise
        {
            Fill,
            Copy,
            Scale,
            Axpy,
            Combination
        };
    }
    using PointwiseOperation = Detail::Pointwise;

    /// Arithmetic core for leased host row views, including logical segments
    /// of a wider physical field. Callers validate matching shapes/CPU placement
    /// and retain access scopes. This function acquires no access, reads no
    /// ghosts implicitly, and performs no allocation/communication. Binary
    /// inputs may alias output only when their logical indexing is identical.
    template <class Output, class X, class Y>
    void ApplyHostRows(PointwiseOperation operation, Output output, X x, Y y, double a = 0, double b = 0)
    {
        using T = std::remove_reference_t<decltype(output(0, 0))>;
        OMP::ForRows(0, output.RowCount(), [&](index iRow)
                     {
            for (index iCol = 0; iCol < output.RowSize(iRow); ++iCol)
            {
                switch (operation)
                {
                case PointwiseOperation::Fill:
                    output(iRow, iCol) = T(a);
                    break;
                case PointwiseOperation::Copy:
                    output(iRow, iCol) = x(iRow, iCol);
                    break;
                case PointwiseOperation::Scale:
                    output(iRow, iCol) *= T(a);
                    break;
                case PointwiseOperation::Axpy:
                    output(iRow, iCol) += T(a) * x(iRow, iCol);
                    break;
                case PointwiseOperation::Combination:
                {
                    const auto left = x(iRow, iCol), right = y(iRow, iCol);
                    output(iRow, iCol) = T(a) * left + T(b) * right;
                    break;
                }
                }
            } });
    }
    namespace Detail
    {
        enum class Reduction
        {
            Dot,
            Sum,
            Min,
            Max,
            Squares
        };
        void ApplyCUDA(Runtime::ExecutionContext, Pointwise, Runtime::ArrayView<float>, Runtime::ArrayView<const float>, Runtime::ArrayView<const float>, double, double);
        void ApplyCUDA(Runtime::ExecutionContext, Pointwise, Runtime::ArrayView<double>, Runtime::ArrayView<const double>, Runtime::ArrayView<const double>, double, double);
        void ReduceCUDA(Runtime::ExecutionContext, Reduction, Runtime::ArrayView<const float>, Runtime::ArrayView<const float>, Runtime::ArrayView<double>);
        void ReduceCUDA(Runtime::ExecutionContext, Reduction, Runtime::ArrayView<const double>, Runtime::ArrayView<const double>, Runtime::ArrayView<double>);

        template <class T>
        void Apply(Runtime::ExecutionContext context, Pointwise operation, Runtime::ArrayView<T> output,
                   Runtime::ArrayView<const T> x = {}, Runtime::ArrayView<const T> y = {}, double a = 0, double b = 0)
        {
            Runtime::ValidateViewDevices(context.device, output, x, y);
            if (context.device.space == Runtime::MemorySpace::CUDA)
            {
                ApplyCUDA(context, operation, output, x, y, a, b);
                return;
            }
            ApplyHostRows(operation, output, x, y, a, b);
        }
        template <class T>
        void Reduce(Runtime::ExecutionContext context, Reduction operation, Runtime::ArrayView<const T> x,
                    Runtime::ArrayView<const T> y, Runtime::ArrayView<double> result)
        {
            Runtime::ValidateViewDevices(context.device, x, y, result);
            if (context.device.space == Runtime::MemorySpace::CUDA)
            {
                ReduceCUDA(context, operation, x, y, result);
                return;
            }
            const double identity = operation == Reduction::Min ? std::numeric_limits<double>::infinity() : operation == Reduction::Max ? -std::numeric_limits<double>::infinity()
                                                                                                                                        : 0;
            const auto accumulate = [&](index begin, index end, auto sums)
            {
                for (index iCol = 0; iCol < result.width; ++iCol)
                    sums[iCol] = identity;
                for (index iRow = begin; iRow < end; ++iRow)
                    for (index iCol = 0; iCol < x.width; ++iCol)
                    {
                        const double value = x(iRow, iCol);
                        if (operation == Reduction::Dot)
                            sums[0] += value * double(y(iRow, iCol));
                        else if (operation == Reduction::Sum)
                            sums[iCol] += value;
                        else if (operation == Reduction::Squares)
                            sums[iCol] += value * value;
                        else if (operation == Reduction::Min)
                            sums[iCol] = std::fmin(sums[iCol], value);
                        else
                            sums[iCol] = std::fmax(sums[iCol], value);
                    }
            };
            const index partitions = std::min<index>(OMP::DistributedThreads(), std::max<index>(1, x.RowCount()));
            if (partitions == 1)
            {
                accumulate(0, x.RowCount(), result[0]);
                return;
            }
            // Result/scratch are local to this submitted operation. Contiguous
            // partitions and their ordered final merge avoid floating atomics;
            // OMP=1 follows the original row/component addition order exactly.
            std::vector<double> partials(size_t(partitions * result.width));
            OMP::ForRows(0, partitions, [&](index part)
                         {
                const index begin = part * (x.RowCount() / partitions) + std::min(part, x.RowCount() % partitions);
                const index end = begin + x.RowCount() / partitions + (part < x.RowCount() % partitions);
                accumulate(begin, end, SpanView<double>{partials.data() + part * result.width, result.width}); });
            for (index iCol = 0; iCol < result.width; ++iCol)
            {
                double sum = partials[size_t(iCol)];
                for (index part = 1; part < partitions; ++part)
                    if (operation == Reduction::Min)
                        sum = std::fmin(sum, partials[size_t(part * result.width + iCol)]);
                    else if (operation == Reduction::Max)
                        sum = std::fmax(sum, partials[size_t(part * result.width + iCol)]);
                    else
                        sum += partials[size_t(part * result.width + iCol)];
                result(0, iCol) = sum;
            }
        }
        template <class Field>
        void Check(const Field &field, const Runtime::ExecutionStream &stream)
        {
            field.Validate();
            DNDS_check_throw_info(field.Device() == stream.Context().device, "algebra stream and field placement differ");
        }
        template <class A, class B>
        void Pair(const A &a, const B &b, bool writing = false)
        {
            a.Validate();
            b.Validate();
            DNDS_check_throw_info(a.Space().Same(b.Space()) && a.Width() == b.Width() && a.Device() == b.Device(),
                                  "algebra operand distribution, component width or placement differs");
            if (writing)
                DNDS_check_throw_info(!Overlaps(a.Owned().Storage(), b.Ghost().Storage()), "algebra output overlaps an operand's ghost storage");
            if (writing && Overlaps(a.Owned().Storage(), b.Owned().Storage()))
                DNDS_check_throw_info(a.Owned().Storage().SameAddress(b.Owned().Storage()) &&
                                          a.Owned().Layout() == b.Owned().Layout(),
                                      "algebra partial overlap is unsupported");
        }
        template <class F>
        void Immediate(Runtime::Device device, F &&run)
        {
            Runtime::EagerEngine engine;
            // Synchronous convenience calls still dispatch by storage. They
            // require idle fields and drain before returning; hot queued calls
            // keep using the caller's engine/stream instead of creating one.
            const auto stream = [&]
            {
                if (device == Runtime::Device::Host())
                    return engine.CreateCPUStream();
#ifdef DNDS_USE_CUDA
                if (device.space == Runtime::MemorySpace::CUDA)
                    return engine.CreateStream(std::make_shared<Runtime::CUDALauncher>(device.ordinal));
#endif
                DNDS_check_throw_info(false, "immediate DOF algebra backend is unavailable");
                return Runtime::ExecutionStream{};
            }();
            run(engine, stream);
            engine.Synchronize();
        }
        template <class F>
        void Immediate(F &&run) { Immediate(Runtime::Device::Host(), std::forward<F>(run)); }
        template <class Field>
        void RequireIdle(const Field &field)
        {
            field.Validate();
            DNDS_check_throw_info(field.Owned().Storage().IsIdle(), "immediate algebra requires idle owned storage; finish pending accesses first");
        }
        struct GlobalSchema
        {
            size_t width, fieldWidth, scalarBytes;
            int operation, protocol;
        };
        std::vector<double> Aggregate(const DofSpace &, const std::function<GlobalSchema()> &schema,
                                      const std::function<std::vector<double>()> &local);
    }

    /// Explicit result owner. Get waits this submitted prefix before exposing
    /// host values; it is separate from ordinary mutation command submission.
    class ComponentResult
    {
        Runtime::Array<double, Runtime::TableRows> _values;
        Runtime::StreamEvent _ready;

    public:
        ComponentResult(Runtime::Array<double, Runtime::TableRows> values, Runtime::StreamEvent ready)
            : _values(std::move(values)), _ready(std::move(ready)) {}
        std::vector<double> Get() const
        {
            _ready.WaitHost();
            const auto host = _values.GetDevice() == Runtime::Device::Host() ? _values : _values.To(Runtime::Device::Host());
            const auto read = host.Read();
            return {read.View().Row(0), read.View().Row(0) + host.Layout().Width()};
        }
    };
    class ScalarResult
    {
        ComponentResult _values;
        bool _sqrt;

    public:
        ScalarResult(ComponentResult values, bool squareRoot = false) : _values(std::move(values)), _sqrt(squareRoot) {}
        double Get() const
        {
            const auto value = _values.Get()[0];
            return _sqrt ? std::sqrt(value) : value;
        }
        ScalarResult SquareRoot() const
        {
            auto result = *this;
            result._sqrt = true;
            return result;
        }
    };

    /// Backend-dispatched arithmetic on retained whole-field views. No leases,
    /// allocation, halo or synchronization are hidden here. The caller declares
    /// operands and validates matching logical shapes/aliasing before invoking
    /// this host launcher; residency is checked once at this boundary.
    template <class T>
    void ApplyRowViews(Runtime::ExecutionContext context, PointwiseOperation operation,
                       Runtime::ArrayView<T> output, Runtime::ArrayView<const T> x = {},
                       Runtime::ArrayView<const T> y = {}, double a = 0, double b = 0)
    {
        Detail::Apply(context, operation, output, x, y, a, b);
    }

    template <class T, size_t N>
    void Fill(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, DofField<T, N> &output, T value)
    {
        Detail::Check(output, stream);
        Runtime::SubmitArrays(engine, stream, [partition = output.Space().Pin(), value](Runtime::ExecutionContext context, auto out)
                              { Detail::Apply(context, Detail::Pointwise::Fill, out.DynamicView(), {}, {}, value); }, Runtime::Write(output.Owned()));
    }
    template <class T, size_t N>
    void Scale(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, DofField<T, N> &output, T value)
    {
        Detail::Check(output, stream);
        Runtime::SubmitArrays(engine, stream, [partition = output.Space().Pin(), value](Runtime::ExecutionContext context, auto out)
                              { Detail::Apply(context, Detail::Pointwise::Scale, out.DynamicView(), {}, {}, value); }, Runtime::ReadWrite(output.Owned()));
    }
    template <class T, size_t A, size_t B>
    void Copy(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, DofField<T, A> &output, const DofField<T, B> &input)
    {
        Detail::Check(output, stream);
        Detail::Pair(output, input, true);
        Runtime::SubmitArrays(engine, stream, [partition = output.Space().Pin()](Runtime::ExecutionContext context, auto x, auto out)
                              { Detail::Apply(context, Detail::Pointwise::Copy, out.DynamicView(), x.DynamicView()); }, Runtime::Read(input.Owned()), Runtime::Write(output.Owned()));
    }
    template <class T, size_t A, size_t B>
    void Axpy(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, DofField<T, A> &output, T alpha, const DofField<T, B> &input)
    {
        Detail::Check(output, stream);
        Detail::Pair(output, input, true);
        Runtime::SubmitArrays(engine, stream, [partition = output.Space().Pin(), alpha](Runtime::ExecutionContext context, auto x, auto out)
                              { Detail::Apply(context, Detail::Pointwise::Axpy, out.DynamicView(), x.DynamicView(), {}, alpha); }, Runtime::Read(input.Owned()), Runtime::ReadWrite(output.Owned()));
    }
    template <class T, size_t A, size_t B, size_t C>
    void LinearCombination(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, DofField<T, A> &output,
                           T alpha, const DofField<T, B> &x, T beta, const DofField<T, C> &y)
    {
        Detail::Check(output, stream);
        Detail::Pair(output, x, true);
        Detail::Pair(output, y, true);
        Runtime::SubmitArrays(engine, stream, [partition = output.Space().Pin(), alpha, beta](Runtime::ExecutionContext context, auto a, auto b, auto out)
                              { Detail::Apply(context, Detail::Pointwise::Combination, out.DynamicView(), a.DynamicView(), b.DynamicView(), alpha, beta); }, Runtime::Read(x.Owned()), Runtime::Read(y.Owned()), Runtime::Write(output.Owned()));
    }

    template <class T, size_t N>
    void Fill(DofField<T, N> &out, T value)
    {
        Detail::RequireIdle(out);
        Detail::Immediate(out.Device(), [&](auto &e, const auto &s)
                          { Fill(e, s, out, value); });
    }
    template <class T, size_t N>
    void Scale(DofField<T, N> &out, T value)
    {
        Detail::RequireIdle(out);
        Detail::Immediate(out.Device(), [&](auto &e, const auto &s)
                          { Scale(e, s, out, value); });
    }
    template <class T, size_t A, size_t B>
    void Copy(DofField<T, A> &out, const DofField<T, B> &in)
    {
        Detail::RequireIdle(out);
        Detail::RequireIdle(in);
        Detail::Immediate(out.Device(), [&](auto &e, const auto &s)
                          { Copy(e, s, out, in); });
    }
    template <class T, size_t A, size_t B>
    void Axpy(DofField<T, A> &out, T alpha, const DofField<T, B> &in)
    {
        Detail::RequireIdle(out);
        Detail::RequireIdle(in);
        Detail::Immediate(out.Device(), [&](auto &e, const auto &s)
                          { Axpy(e, s, out, alpha, in); });
    }
    template <class T, size_t A, size_t B, size_t C>
    void LinearCombination(DofField<T, A> &out, T a, const DofField<T, B> &x, T b, const DofField<T, C> &y)
    {
        Detail::RequireIdle(out);
        Detail::RequireIdle(x);
        Detail::RequireIdle(y);
        Detail::Immediate(out.Device(), [&](auto &e, const auto &s)
                          { LinearCombination(e, s, out, a, x, b, y); });
    }

    template <class T, size_t A, size_t B>
    ScalarResult DotLocal(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, const DofField<T, A> &x, const DofField<T, B> &y)
    {
        Detail::Check(x, stream);
        Detail::Pair(x, y);
        Runtime::Array<double, Runtime::TableRows> result(Runtime::RowLayout::Dense(1, 1), x.Device());
        Runtime::SubmitArrays(engine, stream, [partition = x.Space().Pin()](Runtime::ExecutionContext context, auto a, auto b, auto out)
                              { Detail::Reduce(context, Detail::Reduction::Dot, a.DynamicView(), b.DynamicView(), out.DynamicView()); }, Runtime::Read(x.Owned()), Runtime::Read(y.Owned()), Runtime::Write(result));
        return ScalarResult(ComponentResult(std::move(result), stream.Record()));
    }
    template <class T, size_t N>
    ScalarResult NormLocal(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, const DofField<T, N> &x)
    {
        auto dot = DotLocal(engine, stream, x, x);
        // Retain the dot result through the scalar transformation's Get call.
        return dot.SquareRoot();
    }
    template <class T, size_t N>
    ComponentResult ComponentsLocal(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                                    const DofField<T, N> &x, Detail::Reduction operation)
    {
        Detail::Check(x, stream);
        Runtime::Array<double, Runtime::TableRows> result(Runtime::RowLayout::Dense(1, x.Width()), x.Device());
        Runtime::SubmitArrays(engine, stream, [partition = x.Space().Pin(), operation](Runtime::ExecutionContext context, auto a, auto out)
                              { Detail::Reduce(context, operation, a.DynamicView(), {}, out.DynamicView()); }, Runtime::Read(x.Owned()), Runtime::Write(result));
        return ComponentResult(std::move(result), stream.Record());
    }
    template <class T, size_t N>
    ComponentResult ComponentSumLocal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x) { return ComponentsLocal(e, s, x, Detail::Reduction::Sum); }
    template <class T, size_t N>
    ComponentResult ComponentMinLocal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x) { return ComponentsLocal(e, s, x, Detail::Reduction::Min); }
    template <class T, size_t N>
    ComponentResult ComponentMaxLocal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x) { return ComponentsLocal(e, s, x, Detail::Reduction::Max); }
    template <class T, size_t A, size_t B>
    double DotLocal(const DofField<T, A> &x, const DofField<T, B> &y)
    {
        Detail::RequireIdle(x);
        Detail::RequireIdle(y);
        double value = 0;
        Detail::Immediate(x.Device(), [&](auto &e, const auto &s)
                          { value = DotLocal(e, s, x, y).Get(); });
        return value;
    }
    template <class T, size_t N>
    double NormLocal(const DofField<T, N> &x) { return std::sqrt(DotLocal(x, x)); }
    template <class T, size_t N>
    std::vector<double> ComponentSumLocal(const DofField<T, N> &x)
    {
        Detail::RequireIdle(x);
        std::vector<double> value;
        Detail::Immediate(x.Device(), [&](auto &e, const auto &s)
                          { value = ComponentSumLocal(e, s, x).Get(); });
        return value;
    }
    template <class T, size_t N>
    std::vector<double> ComponentMinLocal(const DofField<T, N> &x)
    {
        Detail::RequireIdle(x);
        std::vector<double> value;
        Detail::Immediate(x.Device(), [&](auto &e, const auto &s)
                          { value = ComponentMinLocal(e, s, x).Get(); });
        return value;
    }
    template <class T, size_t N>
    std::vector<double> ComponentMaxLocal(const DofField<T, N> &x)
    {
        Detail::RequireIdle(x);
        std::vector<double> value;
        Detail::Immediate(x.Device(), [&](auto &e, const auto &s)
                          { value = ComponentMaxLocal(e, s, x).Get(); });
        return value;
    }

    template <class T, size_t A, size_t B>
    double DotGlobal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, A> &x, const DofField<T, B> &y, bool immediate = false)
    {
        return Detail::Aggregate(x.Space(), [&]
                                 { Detail::Check(x, s); Detail::Pair(x, y); if (immediate) { Detail::RequireIdle(x); Detail::RequireIdle(y); } return Detail::GlobalSchema{1, x.Width(), sizeof(T), 0, 0}; }, [&]
                                 { return std::vector<double>{DotLocal(e, s, x, y).Get()}; })[0];
    }
    template <class T, size_t N>
    double NormGlobal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x, bool immediate = false)
    {
        return std::sqrt(Detail::Aggregate(x.Space(), [&]
                                           { Detail::Check(x, s); if (immediate) Detail::RequireIdle(x); return Detail::GlobalSchema{1, x.Width(), sizeof(T), 0, 1}; }, [&]
                                           { return std::vector<double>{DotLocal(e, s, x, x).Get()}; })[0]);
    }
    template <class T, size_t N>
    std::vector<double> ComponentSumGlobal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x, bool immediate = false)
    {
        return Detail::Aggregate(x.Space(), [&]
                                 { Detail::Check(x, s); if (immediate) Detail::RequireIdle(x); return Detail::GlobalSchema{x.Width(), x.Width(), sizeof(T), 0, 2}; }, [&]
                                 { return ComponentSumLocal(e, s, x).Get(); });
    }
    template <class T, size_t N>
    std::vector<double> ComponentMinGlobal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x, bool immediate = false)
    {
        return Detail::Aggregate(x.Space(), [&]
                                 { Detail::Check(x, s); if (immediate) Detail::RequireIdle(x); return Detail::GlobalSchema{x.Width(), x.Width(), sizeof(T), 1, 3}; }, [&]
                                 { return ComponentMinLocal(e, s, x).Get(); });
    }
    template <class T, size_t N>
    std::vector<double> ComponentMaxGlobal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x, bool immediate = false)
    {
        return Detail::Aggregate(x.Space(), [&]
                                 { Detail::Check(x, s); if (immediate) Detail::RequireIdle(x); return Detail::GlobalSchema{x.Width(), x.Width(), sizeof(T), 2, 4}; }, [&]
                                 { return ComponentMaxLocal(e, s, x).Get(); });
    }
    template <class T, size_t A, size_t B>
    double DotGlobal(const DofField<T, A> &x, const DofField<T, B> &y)
    {
        double value = 0;
        Detail::Immediate(x.Device(), [&](auto &e, const auto &s)
                          { value = DotGlobal(e, s, x, y, true); });
        return value;
    }
    template <class T, size_t N>
    double NormGlobal(const DofField<T, N> &x)
    {
        double value = 0;
        Detail::Immediate(x.Device(), [&](auto &e, const auto &s)
                          { value = NormGlobal(e, s, x, true); });
        return value;
    }
    template <class T, size_t N>
    std::vector<double> ComponentSumGlobal(const DofField<T, N> &x)
    {
        std::vector<double> value;
        Detail::Immediate(x.Device(), [&](auto &e, const auto &s)
                          { value = ComponentSumGlobal(e, s, x, true); });
        return value;
    }
    template <class T, size_t N>
    std::vector<double> ComponentMinGlobal(const DofField<T, N> &x)
    {
        std::vector<double> value;
        Detail::Immediate(x.Device(), [&](auto &e, const auto &s)
                          { value = ComponentMinGlobal(e, s, x, true); });
        return value;
    }
    template <class T, size_t N>
    std::vector<double> ComponentMaxGlobal(const DofField<T, N> &x)
    {
        std::vector<double> value;
        Detail::Immediate(x.Device(), [&](auto &e, const auto &s)
                          { value = ComponentMaxGlobal(e, s, x, true); });
        return value;
    }
}
