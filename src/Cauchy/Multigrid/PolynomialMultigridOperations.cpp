#include "Cauchy/Multigrid/PolynomialMultigridOperations.hpp"
#include "Cauchy/Multigrid/CPU/PolynomialMultigridKernels.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Algebra/ComponentSelection.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Multigrid/CUDA/PolynomialMultigridKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        void SameStorageLayout(const EquationDofs &a, const EquationDofs &b)
        {
            a.Validate();
            b.Validate();
            a.RequireSchema(b.Schema());
            DNDS_check_throw_info(a.Device() == b.Device() && a.Space().OwnedRows() == b.Space().OwnedRows() &&
                                      a.PhysicalFieldNumber() == b.PhysicalFieldNumber(),
                                  "pMG identity transfer row/device/grouping mismatch");
            for (index iBlock = 0; iBlock < a.BlockNumber(); ++iBlock)
            {
                const auto x = a.LocateBlock(iBlock), y = b.LocateBlock(iBlock);
                DNDS_check_throw_info(x.physicalField == y.physicalField && x.firstComponent == y.firstComponent &&
                                          x.componentNumber == y.componentNumber,
                                      "pMG identity transfer logical binding mismatch");
            }
        }
        R::MetadataBuffer<uint8_t> ColumnMask(const EquationDofs &state, index iField, const std::vector<index> &components)
        {
            const Algebra::ComponentSelection selection(state.ComponentNumber(), components);
            std::vector<uint8_t> values(state.PhysicalField(iField).Width(), 2);
            index offset = 0;
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            {
                const auto binding = state.LocateBlock(iBlock);
                if (binding.physicalField == iField)
                    for (index j = 0; j < binding.componentNumber; ++j)
                        values[binding.firstComponent + j] = uint8_t(selection.Includes(offset + j));
                offset += binding.componentNumber;
            }
            return R::MetadataBuffer<uint8_t>(std::move(values));
        }
#ifdef DNDS_USE_CUDA
        CUDA::FASSelection DeviceMask(const R::MetadataBuffer<uint8_t> &mask)
        {
            DNDS_check_throw_info(mask.size() >= 4 && mask.size() <= 7, "unsupported device FAS mask width");
            CUDA::FASSelection result;
            for (index j = 0; j < mask.size(); ++j)
                result.active[j] = mask.HostView()[j];
            return result;
        }
#endif
    }
    struct PolynomialMultigridOperations::Data
    {
        R::ExecutionStream stream;
        R::Array<index> partial, result;
        explicit Data(R::ExecutionStream lane) : stream(std::move(lane)) {}
        void Check(const EquationDofs &state) const
        {
            state.Validate();
            DNDS_check_throw_info(state.Device() == stream.Context().device, "pMG algebra placement differs from its compute lane");
            if (state.Device() != R::Device::Host())
                DNDS_check_throw_info(state.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table &&
                                          state.ComponentNumber() >= 4 && state.ComponentNumber() <= 7,
                                      "CUDA pMG requires a combined AoS equation with 4 to 7 components");
        }
    };
    PolynomialMultigridOperations::PolynomialMultigridOperations(R::ExecutionStream lane)
        : QueuedEquationOperations(lane), _data(std::make_shared<Data>(std::move(lane))) {}
    void PolynomialMultigridOperations::CopyIdentityRows(const EquationDofs &input, EquationDofs &output) const
    {
        DNDS_check_throw_info(_data, "pMG transfer requires an explicit compute lane");
        _data->Check(input);
        _data->Check(output);
        SameStorageLayout(input, output);
        RequireDistinct(input, output);
        auto engine = _data->stream.LockEngine();
        for (index iField = 0; iField < input.PhysicalFieldNumber(); ++iField)
        {
            const auto &source = input.PhysicalField(iField).Owned();
            auto &target = output.PhysicalField(iField).Owned();
            DNDS_check_throw_info(source.Layout().Width() == target.Layout().Width(), "pMG identity transfer physical width differs");
            R::SubmitArrays(engine, _data->stream, [](R::ExecutionContext context, auto x, auto y)
                            { Algebra::ApplyRowViews(context, Algebra::PointwiseOperation::Copy, y.DynamicView(), x.DynamicView(), {}, 0, 0); }, R::ReadAs<R::UniformTableRows>(source), R::WriteAs<R::UniformTableRows>(target));
        }
    }
    void PolynomialMultigridOperations::ProjectCorrection(EquationDofs &state, const std::vector<index> &components) const
    {
        DNDS_check_throw_info(_data, "pMG projection requires an explicit compute lane");
        _data->Check(state);
        auto engine = _data->stream.LockEngine();
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            const auto mask = ColumnMask(state, iField, components);
            R::SubmitArrays(engine, _data->stream, [mask](R::ExecutionContext context, auto values)
                            {
                if (context.device == R::Device::Host()) CPU::UpdateSelectedRows<false>(values, values.ConstView(), mask.HostView());
                else
                {
#ifdef DNDS_USE_CUDA
                    CUDA::UpdateSelectedRows(context, values, {}, DeviceMask(mask), false);
#else
                    DNDS_check_throw_info(false, "CUDA pMG is not compiled");
#endif
                } }, R::ReadWriteAs<R::UniformTableRows>(state.PhysicalField(iField).Owned()));
        }
    }
    void PolynomialMultigridOperations::AddSelected(EquationDofs &state, const EquationDofs &correction, const std::vector<index> &components) const
    {
        if (components.empty())
        {
            Axpy(state, 1, correction);
            return;
        }
        DNDS_check_throw_info(_data, "pMG update requires an explicit compute lane");
        _data->Check(state);
        _data->Check(correction);
        state.RequireCompatible(correction);
        SameStorageLayout(state, correction);
        RequireDistinct(state, correction);
        auto engine = _data->stream.LockEngine();
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            const auto mask = ColumnMask(state, iField, components);
            R::SubmitArrays(engine, _data->stream, [mask](R::ExecutionContext context, auto delta, auto values)
                            {
                if (context.device == R::Device::Host()) CPU::UpdateSelectedRows<true>(values, delta, mask.HostView());
                else
                {
#ifdef DNDS_USE_CUDA
                    CUDA::UpdateSelectedRows(context, values, delta, DeviceMask(mask), true);
#else
                    DNDS_check_throw_info(false, "CUDA pMG is not compiled");
#endif
                } }, R::ReadAs<R::UniformTableRows>(correction.PhysicalField(iField).Owned()), R::ReadWriteAs<R::UniformTableRows>(state.PhysicalField(iField).Owned()));
        }
    }
    void PolynomialMultigridOperations::CheckFrozen(const EquationDofs &state, const EquationDofs &saved, const std::vector<index> &components) const
    {
        DNDS_check_throw_info(_data, "pMG frozen check requires an explicit compute lane");
        _data->Check(state);
        _data->Check(saved);
        state.RequireCompatible(saved);
        SameStorageLayout(state, saved);
        auto engine = _data->stream.LockEngine();
        if (state.Device() == R::Device::Host())
        {
            const auto badRow = std::make_shared<index>(-1);
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                const auto mask = ColumnMask(state, iField, components);
                R::SubmitArrays(engine, _data->stream, [mask, badRow](R::ExecutionContext, auto a, auto b)
                                {
                    const index found = CPU::FirstChangedFrozenRow(a, b, mask.HostView());
                    if (found >= 0 && (*badRow < 0 || found < *badRow)) *badRow = found; }, R::ReadAs<R::UniformTableRows>(state.PhysicalField(iField).Owned()), R::ReadAs<R::UniformTableRows>(saved.PhysicalField(iField).Owned()));
            }
            engine.Synchronize();
            DNDS_check_throw_info(*badRow < 0, "FAS callback changed a frozen component");
        }
        else
        {
#ifdef DNDS_USE_CUDA
            const auto context = _data->stream.Context();
            const index scratchRows = CUDA::FrozenCheckScratchRows(index(state.Space().OwnedRows()));
            if (_data->partial.RowCount() != scratchRows || _data->result.RowCount() != 1)
            {
                engine.Synchronize();
                _data->partial = R::Array<index>(R::RowLayout::Dense(scratchRows, 1), context.device);
                _data->result = R::Array<index>(R::RowLayout::Dense(1, 1), context.device);
                _data->partial.SetName("pMG/frozen/partials", DNDS_ALLOCATION_SITE());
                _data->result.SetName("pMG/frozen/firstChanged", DNDS_ALLOCATION_SITE());
            }
            const auto mask = ColumnMask(state, 0, components);
            R::SubmitArrays(engine, _data->stream, [mask](R::ExecutionContext context, auto a, auto b, auto partial, auto result)
                            { CUDA::CheckFrozenRows(context, a, b, DeviceMask(mask), partial, result); }, R::ReadAs<R::UniformTableRows>(state.CoupledField().Owned()), R::ReadAs<R::UniformTableRows>(saved.CoupledField().Owned()), R::WriteAs<R::FixedRows<1>>(_data->partial), R::WriteAs<R::FixedRows<1>>(_data->result));
            engine.Synchronize();
            const auto host = _data->result.To(R::Device::Host(), context);
            const auto read = host.Read();
            DNDS_check_throw_info(read.View()(0, 0) < 0, "FAS callback changed a frozen component");
#else
            DNDS_check_throw_info(false, "CUDA pMG is not compiled");
#endif
        }
    }
}
