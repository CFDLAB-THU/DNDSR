#include "Cauchy/Implicit/JacobianGeometry.hpp"

namespace DNDS::Cauchy
{
    JacobianGeometryStorage::JacobianGeometryStorage(FiniteVolumeStencilStorage::ReadAccess stencil,
                                                     const std::vector<std::vector<index>> &columns)
        : _stencil(std::move(stencil))
    {
        const auto view = _stencil.View();
        view.ValidateDevice(Runtime::Device::Host());
        sameComponentFrames = true;
        for (index iCell = 0; iCell < index(columns.size()); ++iCell)
            for (const auto iFace : view.topology.CellFaces(iCell))
                sameComponentFrames = sameComponentFrames && !view.metric.periodic.Rotates(iFace);
        Build(columns, view.topology, Runtime::ExecutionContext::Host());
    }
    JacobianGeometryStorage::JacobianGeometryStorage(FiniteVolumeStencilStorage::ReadAccess stencil,
                                                     const std::vector<std::vector<index>> &columns,
                                                     Geom::CellFaceIncidenceView symbolicTopology,
                                                     bool sameFrames, Runtime::ExecutionContext context)
        : _stencil(std::move(stencil)), sameComponentFrames(sameFrames)
    {
        Build(columns, symbolicTopology, context);
    }
    void JacobianGeometryStorage::Build(const std::vector<std::vector<index>> &columns,
                                        Geom::CellFaceIncidenceView topology, Runtime::ExecutionContext context)
    {
        topology.ValidateDevice(Runtime::Device::Host());
        DNDS_check_throw_info(_stencil.View().routing.GetDevice() == context.device &&
                                  _stencil.View().jacobianOffsets.RowCount() == index(columns.size()) + 1 &&
                                  topology.cellFaces.RowCount() >= index(columns.size()),
                              "Jacobian symbolic routing and retained stencil row spaces/devices differ");
        std::vector<size_t> columnLengths;
        index blocks = 0;
        for (const auto &row : columns)
        {
            DNDS_check_throw_info(std::is_sorted(row.begin(), row.end()) &&
                                      std::adjacent_find(row.begin(), row.end()) == row.end(),
                                  "Jacobian symbolic columns must be sorted and unique");
            columnLengths.push_back(row.size());
            blocks += index(row.size());
        }
        _columns = Runtime::Array<index>(Runtime::RowLayout::Ragged(columnLengths));
        _blockOffsets = Runtime::Array<index>(Runtime::RowLayout::Dense(index(columns.size()) + 1, 1));
        std::vector<size_t> lengths(blocks, 0);
        const auto columnSlot = [&](index iCell, index other)
        {
            const auto &row = columns[iCell];
            const auto found = std::lower_bound(row.begin(), row.end(), other);
            DNDS_check_throw_info(found != row.end() && *found == other,
                                  "Jacobian symbolic incidence has no sparse column");
            return index(found - row.begin());
        };
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
                const auto faces = topology.CellFaces(iCell);
                for (rowsize slot = 0; slot < faces.size(); ++slot)
                {
                    const index other = topology.CellFaceOther(iCell, faces[slot], slot);
                    if (other != UnInitIndex && other != iCell)
                        ++lengths[first + columnSlot(iCell, other)];
                }
                first += c.RowSize(iCell);
            }
            offsets(index(columns.size()), 0) = first;
        }
        _incidences = Runtime::Array<index>(Runtime::RowLayout::Ragged(lengths));
        {
            const auto offsetsRead = _blockOffsets.Read();
            const auto offsets = offsetsRead.View<Runtime::FixedRows<1>>();
            auto write = _incidences.Write();
            const auto incidences = write.View<Runtime::CSRRows>();
            std::vector<index> next(blocks, 0);
            index firstIncidence = 0;
            for (index iCell = 0; iCell < index(columns.size()); ++iCell)
            {
                const auto faces = topology.CellFaces(iCell);
                for (rowsize slot = 0; slot < faces.size(); ++slot)
                {
                    const index other = topology.CellFaceOther(iCell, faces[slot], slot);
                    if (other == UnInitIndex || other == iCell)
                        continue;
                    const index iBlock = offsets(iCell, 0) + columnSlot(iCell, other);
                    incidences(iBlock, next[iBlock]++) = firstIncidence + slot;
                }
                firstIncidence += faces.size();
            }
        }
        if (context.device != Runtime::Device::Host())
        {
            _columns = _columns.To(context.device, context);
            _blockOffsets = _blockOffsets.To(context.device, context);
            _incidences = _incidences.To(context.device, context);
        }
        _columns.SetName("Cauchy/linear/jacobianGeometry/columns", DNDS_ALLOCATION_SITE());
        _blockOffsets.SetName("Cauchy/linear/jacobianGeometry/blockOffsets", DNDS_ALLOCATION_SITE());
        _incidences.SetName("Cauchy/linear/jacobianGeometry/incidences", DNDS_ALLOCATION_SITE());
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
