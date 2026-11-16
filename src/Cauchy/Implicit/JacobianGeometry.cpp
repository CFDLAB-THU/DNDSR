#include "Cauchy/Implicit/JacobianGeometry.hpp"

namespace DNDS::Cauchy
{
    JacobianGeometryStorage JacobianGeometryStorage::Placed(Runtime::Device device, Runtime::ExecutionContext context) const
    {
        JacobianGeometryStorage result;
        result._columns = _columns.To(device, context);
        result._blockOffsets = _blockOffsets.To(device, context);
        result._points = _points.To(device, context);
        result.sameComponentFrames = sameComponentFrames;
        return result;
    }
    JacobianGeometryStorage::JacobianGeometryStorage(FiniteVolumeStencilView stencil,
                                                     CFV::FiniteVolumeHostView metric, const std::vector<std::vector<index>> &columns)
    {
        Build(stencil, metric, columns, false);
    }
    JacobianGeometryStorage::JacobianGeometryStorage(FiniteVolumeStencilView stencil, const std::vector<std::vector<index>> &columns)
    {
        Build(stencil, {}, columns, true);
    }
    void JacobianGeometryStorage::Build(FiniteVolumeStencilView stencil,
                                        CFV::FiniteVolumeHostView metric, const std::vector<std::vector<index>> &columns, bool auxiliaryAreas)
    {
        Runtime::ValidateViewDevices(Runtime::Device::Host(), stencil.jacobianSamples, stencil.jacobianOffsets);
        if (auxiliaryAreas)
            stencil.auxiliaryFaceArea.ValidateDevice(Runtime::Device::Host());
        else
            metric.faceMean.ValidateDevice(Runtime::Device::Host());
        std::vector<size_t> columnLengths;
        index blocks = 0;
        for (const auto &row : columns)
        {
            columnLengths.push_back(row.size());
            blocks += index(row.size());
        }
        _columns = Runtime::Array<index>(Runtime::RowLayout::Ragged(columnLengths));
        _blockOffsets = Runtime::Array<index>(Runtime::RowLayout::Dense(columns.size() + 1, 1));
        _columns.SetName("Cauchy/linear/jacobianGeometry/columns", DNDS_ALLOCATION_SITE());
        _blockOffsets.SetName("Cauchy/linear/jacobianGeometry/blockOffsets", DNDS_ALLOCATION_SITE());
        std::vector<size_t> pointLengths(blocks, 0);
        {
            auto columnWrite = _columns.Write(), offsetWrite = _blockOffsets.Write();
            const auto c = columnWrite.View<Runtime::CSRRows>();
            const auto offsets = offsetWrite.View<Runtime::FixedRows<1>>();
            index first = 0;
            for (index iCell = 0; iCell < index(columns.size()); ++iCell)
            {
                offsets(iCell, 0) = first;
                for (index j = 0; j < index(columns[iCell].size()); ++j)
                    c(iCell, j) = columns[iCell][j];
                for (index entry = stencil.jacobianOffsets(iCell, 0); entry < stencil.jacobianOffsets(iCell + 1, 0); ++entry)
                {
                    const auto &sample = stencil.jacobianSamples(entry, 0);
                    if (sample.other == UnInitIndex || sample.other == iCell)
                        continue;
                    DNDS_check_throw_info(sample.jacobianColumn >= 0 && sample.jacobianColumn < c.RowSize(iCell) &&
                                              c(iCell, sample.jacobianColumn) == sample.other,
                                          "evaluated Jacobian stencil has an invalid prepared column");
                    ++pointLengths[first + sample.jacobianColumn];
                    sameComponentFrames = sameComponentFrames && !sample.changesFrame;
                }
                first += c.RowSize(iCell);
            }
            offsets(index(columns.size()), 0) = first;
        }
        _points = Runtime::Array<JacobianPointRecord>(Runtime::RowLayout::Ragged(pointLengths));
        _points.SetName("Cauchy/linear/jacobianGeometry/points", DNDS_ALLOCATION_SITE());
        const auto offsetsRead = _blockOffsets.Read();
        const auto offsets = offsetsRead.View<Runtime::FixedRows<1>>();
        auto write = _points.Write();
        const auto points = write.View<Runtime::CSRRows>();
        std::vector<index> next(blocks, 0);
        for (index iCell = 0; iCell < index(columns.size()); ++iCell)
            for (index entry = stencil.jacobianOffsets(iCell, 0); entry < stencil.jacobianOffsets(iCell + 1, 0); ++entry)
            {
                const auto &sample = stencil.jacobianSamples(entry, 0);
                if (sample.other == UnInitIndex || sample.other == iCell)
                    continue;
                const index iBlock = offsets(iCell, 0) + sample.jacobianColumn;
                DNDS_check_throw_info(auxiliaryAreas ? sample.auxiliaryFace >= 0 && sample.auxiliaryFace < stencil.auxiliaryFaceArea.RowCount()
                                                     : sample.face >= 0 && sample.face < metric.faceMean.RowCount(),
                                      "Jacobian geometry face area row is missing");
                points(iBlock, next[iBlock]++) = {sample.other, sample.auxiliaryFace, entry,
                                                  sample.normal, sample.ownPoint, sample.rotation,
                                                  auxiliaryAreas ? stencil.auxiliaryFaceArea(sample.auxiliaryFace, 0) : metric.GetFaceArea(sample.face), sample.changesFrame};
            }
    }
    std::vector<std::vector<index>> JacobianGeometryStorage::CopyColumns() const
    {
        const auto host = _columns.GetDevice() == Runtime::Device::Host() ? _columns : _columns.To(Runtime::Device::Host());
        const auto read = host.Read();
        const auto view = read.View<Runtime::CSRRows>();
        std::vector<std::vector<index>> result(view.RowCount());
        for (index iRow = 0; iRow < view.RowCount(); ++iRow)
            for (const auto column : view[iRow])
                result[iRow].push_back(column);
        return result;
    }
}
