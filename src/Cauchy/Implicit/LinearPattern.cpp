#include "Cauchy/Implicit/LinearPattern.hpp"
#include <climits>

namespace DNDS::Cauchy
{
    namespace
    {
        Runtime::Array<index> Publish(const std::vector<index> &values, const char *name)
        {
            Runtime::Array<index> result(Runtime::RowLayout::Dense(values.size(), 1));
            {
                auto write = result.Write();
                const auto view = write.View<Runtime::FixedRows<1>>();
                for (index iRow = 0; iRow < index(values.size()); ++iRow)
                    view(iRow, 0) = values[iRow];
            }
            result.SetName(name, DNDS_ALLOCATION_SITE());
            return result;
        }
        std::vector<index> Copy(const Runtime::Array<index> &values)
        {
            const auto host = values.GetDevice() == Runtime::Device::Host() ? values : values.To(Runtime::Device::Host());
            const auto read = host.Read();
            const auto view = read.View<Runtime::FixedRows<1>>();
            std::vector<index> result(view.RowCount());
            for (index iRow = 0; iRow < view.RowCount(); ++iRow)
                result[iRow] = view(iRow, 0);
            return result;
        }
    }
    LinearPatternStorage::LinearPatternStorage(
        const std::vector<std::vector<index>> &columns,
        const std::vector<index> &order, const std::vector<index> &position,
        const std::vector<index> &core, const std::vector<index> &dependent, LinearIndexStorage indices)
        : _selections{Runtime::RowSelectionShape::FromIndices(order, columns.size()),
                      Runtime::RowSelectionShape::FromIndices(core, columns.size()),
                      Runtime::RowSelectionShape::FromIndices(dependent, columns.size())},
          _order(Publish(order, "Cauchy/linear/pattern/order")), _position(Publish(position, "Cauchy/linear/pattern/position")),
          _core(Publish(core, "Cauchy/linear/pattern/core")), _dependent(Publish(dependent, "Cauchy/linear/pattern/dependent"))
    {
        DNDS_check_throw_info(indices == LinearIndexStorage::Int64 || indices == LinearIndexStorage::Int32,
                              "invalid linear native index storage");
        if (indices == LinearIndexStorage::Int32)
        {
            DNDS_check_throw_info(columns.size() <= size_t(INT_MAX), "int32 linear row count overflow");
            index entries = 0;
            for (const auto &row : columns)
            {
                DNDS_check_throw_info(row.size() <= size_t(INT_MAX - entries), "int32 linear entry count overflow");
                entries += index(row.size());
                for (const auto column : row)
                    DNDS_check_throw_info(column >= 0 && column <= INT_MAX, "int32 linear column overflow");
            }
            _offsets32.emplace(Runtime::RowLayout::Dense(columns.size() + 1, 1));
            _columns32.emplace(Runtime::RowLayout::Dense(entries, 1));
            _offsets32->SetName("Cauchy/linear/pattern/rowOffsets32", DNDS_ALLOCATION_SITE());
            _columns32->SetName("Cauchy/linear/pattern/columns32", DNDS_ALLOCATION_SITE());
            auto offsets = _offsets32->Write(), values = _columns32->Write();
            const auto o = offsets.View<Runtime::FixedRows<1>>(), c = values.View<Runtime::FixedRows<1>>();
            index next = 0;
            for (index iRow = 0; iRow < index(columns.size()); ++iRow)
            {
                o(iRow, 0) = int(next);
                for (const auto column : columns[iRow])
                    c(next++, 0) = int(column);
            }
            o(index(columns.size()), 0) = int(next);
            return;
        }
        std::vector<size_t> lengths;
        lengths.reserve(columns.size());
        for (const auto &row : columns)
            lengths.push_back(row.size());
        _columns = Runtime::Array<index>(Runtime::RowLayout::Ragged(lengths));
        _columns.SetName("Cauchy/linear/pattern/columns64", DNDS_ALLOCATION_SITE());
        auto write = _columns.Write();
        const auto view = write.View<Runtime::CSRRows>();
        for (index iRow = 0; iRow < index(columns.size()); ++iRow)
            for (index j = 0; j < index(columns[iRow].size()); ++j)
                view(iRow, j) = columns[iRow][j];
    }
    LinearPatternStorage LinearPatternStorage::Placed(Runtime::Device device, Runtime::ExecutionContext context) const
    {
        LinearPatternStorage result;
        result._selections = _selections;
        result._columns = _columns.To(device, context);
        if (_offsets32)
        {
            result._offsets32 = _offsets32->To(device, context);
            result._columns32 = _columns32->To(device, context);
        }
        result._order = _order.To(device, context);
        result._position = _position.To(device, context);
        result._core = _core.To(device, context);
        result._dependent = _dependent.To(device, context);
        return result;
    }
    std::vector<std::vector<index>> LinearPatternStorage::CopyColumns() const
    {
        if (_offsets32)
        {
            const auto offsets = _offsets32->GetDevice() == Runtime::Device::Host() ? *_offsets32 : _offsets32->To(Runtime::Device::Host());
            const auto columns = _columns32->GetDevice() == Runtime::Device::Host() ? *_columns32 : _columns32->To(Runtime::Device::Host());
            const auto offsetRead = offsets.Read(), columnRead = columns.Read();
            const LinearColumnsView view{offsetRead.View<Runtime::FixedRows<1>>(), columnRead.View<Runtime::FixedRows<1>>()};
            std::vector<std::vector<index>> result(view.RowCount());
            for (index iRow = 0; iRow < view.RowCount(); ++iRow)
                for (const auto column : view[iRow])
                    result[iRow].push_back(column);
            return result;
        }
        const auto host = _columns.GetDevice() == Runtime::Device::Host() ? _columns : _columns.To(Runtime::Device::Host());
        const auto read = host.Read();
        const auto view = read.View<Runtime::CSRRows>();
        std::vector<std::vector<index>> result(view.RowCount());
        for (index iRow = 0; iRow < view.RowCount(); ++iRow)
            for (const auto column : view[iRow])
                result[iRow].push_back(column);
        return result;
    }
    std::vector<index> LinearPatternStorage::CopyOrder() const { return Copy(_order); }
    std::vector<index> LinearPatternStorage::CopyCore() const { return Copy(_core); }
    std::vector<index> LinearPatternStorage::CopyDependent() const { return Copy(_dependent); }
}
