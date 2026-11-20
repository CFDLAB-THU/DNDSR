#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Access/MeshAccess.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include <nlohmann/json.hpp>

namespace DNDS::Geom
{
    MeshReadAccess::MeshReadAccess(MeshReadAccess &&other) noexcept
        : _execution(std::move(other._execution)), _meshRead(std::move(other._meshRead)), _sourceRead(std::move(other._sourceRead)),
          _nativeOwner(std::move(other._nativeOwner)), _stamp(std::move(other._stamp)),
          _context(other._context), _storage(other._storage), _holds(std::move(other._holds)),
          _completion(std::move(other._completion)), _completed(other._completed), _active(other._active)
    {
        other._active = false;
    }
    void MeshReadAccess::Validate(const MeshReadRequirements &requirements, MeshAuditOptions audit) const
    {
        auto total = [](const auto &pair)
        { return pair.owned.RowCount() + pair.ghost.RowCount(); };
        auto sameRows = [](const auto &a, const auto &b)
        {
            return a.owned.RowCount() == b.owned.RowCount() && a.ghost.RowCount() == b.ghost.RowCount();
        };
        auto width = [](const auto &pair, size_t expected)
        {
            // Layouts live on host even for device arrays. Shape validation
            // below uses captured ArrayLayout through the construction helpers.
            DNDS_check_throw_info(pair.owned.width == expected && (!pair.ghost.rows || pair.ghost.width == expected),
                                  "mesh uniform component count mismatch");
        };
        width(_storage.coords, 3);
        width(_storage.cellElemInfo, 1);
        width(_storage.bndElemInfo, 1);
        auto cellPairWidth = [&](MeshRelation relation)
        {
            const auto &pair = _storage.relations[size_t(relation)];
            width(pair, 2);
            for (const auto *segment : {&pair.owned, &pair.ghost})
                DNDS_check_throw_info(!segment->offsets && !segment->sizes && segment->width == 2 && segment->stride >= 2,
                                      "physical cell incidence must use a two-slot table");
        };
        cellPairWidth(MeshRelation::BoundaryCells);
        auto rows = [&](MeshRelation kind)
        { return total(_storage.relations[static_cast<size_t>(kind)]); };
        DNDS_check_throw_info(sameRows(_storage.cellElemInfo, _storage.relations[static_cast<size_t>(MeshRelation::CellNodes)]) &&
                                  sameRows(_storage.bndElemInfo, _storage.relations[static_cast<size_t>(MeshRelation::BoundaryNodes)]),
                              "mesh element information does not match entity rows");
        if (requirements.faces)
        {
            width(_storage.faceElemInfo, 1);
            cellPairWidth(MeshRelation::FaceCells);
            DNDS_check_throw_info(sameRows(_storage.faceElemInfo, _storage.relations[static_cast<size_t>(MeshRelation::FaceNodes)]), "face information row mismatch");
        }
        if (requirements.edges)
        {
            width(_storage.edgeElemInfo, 1);
            DNDS_check_throw_info(sameRows(_storage.edgeElemInfo, _storage.relations[static_cast<size_t>(MeshRelation::EdgeNodes)]), "edge information row mismatch");
        }
        if (requirements.wallDistance)
        {
            width(_storage.nodeWallDist, 3);
            DNDS_check_throw_info(sameRows(_storage.nodeWallDist, _storage.coords), "wall distance row mismatch");
        }
        const auto nodeCount = total(_storage.coords);
        const auto cellCount = rows(MeshRelation::CellNodes);
        const auto bndCount = rows(MeshRelation::BoundaryNodes);
        const auto faceCount = rows(MeshRelation::FaceNodes);
        const auto edgeCount = rows(MeshRelation::EdgeNodes);
        const auto &cellRows = _storage.relations[static_cast<size_t>(MeshRelation::CellNodes)];
        const auto &bndRows = _storage.relations[static_cast<size_t>(MeshRelation::BoundaryNodes)];
        const auto &faceRows = _storage.relations[static_cast<size_t>(MeshRelation::FaceNodes)];
        const auto &edgeRows = _storage.relations[static_cast<size_t>(MeshRelation::EdgeNodes)];
        if (_context.device == Runtime::Device::Host())
        {
            auto elementRows = [&](const auto &info, const auto &nodes, int dimension)
            {
                for (size_t row = 0; row < total(nodes); ++row)
                {
                    const auto type = info(row, 0).type;
                    // Current supported element IDs are the contiguous range
                    // [Line2, ElemType_NUM); reject before dispatch can assert.
                    DNDS_check_throw_info(type > Elem::UnknownElem && type < Elem::ElemType_NUM,
                                          "mesh element information has an unsupported type");
                    const Elem::Element element{static_cast<Elem::ElemType>(type)};
                    DNDS_check_throw_info(element.GetDim() == dimension,
                                          "mesh element dimension differs from its entity role");
                    DNDS_check_throw_info(nodes.RowSize(row) == size_t(element.GetNumNodes()) &&
                                              nodes.RowSize(row) <= size_t(Elem::CellNumNodeMax),
                                          "mesh node incidence width differs from element type");
                }
            };
            elementRows(_storage.cellElemInfo, cellRows, _storage.dim);
            elementRows(_storage.bndElemInfo, bndRows, _storage.dim - 1);
            if (requirements.faces)
            {
                elementRows(_storage.faceElemInfo, faceRows, _storage.dim - 1);
                const auto &cellFaces = _storage.relations[static_cast<size_t>(MeshRelation::CellFaces)];
                DNDS_check_throw_info(sameRows(cellFaces, cellRows), "cell face rows differ from cell distribution");
                for (size_t row = 0; row < total(cellRows); ++row)
                    DNDS_check_throw_info(cellFaces.RowSize(row) == size_t(Elem::Element{_storage.cellElemInfo(row, 0).getElemType()}.GetNumFaces()),
                                          "mesh face incidence width differs from cell element type");
            }
            if (requirements.edges)
                elementRows(_storage.edgeElemInfo, edgeRows, 1);
        }
        const index sourceOwned[]{cellRows.owned.RowCount(), bndRows.owned.RowCount(), bndRows.owned.RowCount(), cellRows.owned.RowCount(),
                                  _storage.coords.owned.RowCount(), _storage.coords.owned.RowCount(), cellRows.owned.RowCount(),
                                  faceRows.owned.RowCount(), faceRows.owned.RowCount(), faceRows.owned.RowCount(), bndRows.owned.RowCount(),
                                  cellRows.owned.RowCount(), edgeRows.owned.RowCount(), edgeRows.owned.RowCount(), cellRows.owned.RowCount()};
        const index sourceGhost[]{cellRows.ghost.RowCount(), bndRows.ghost.RowCount(), bndRows.ghost.RowCount(), cellRows.ghost.RowCount(),
                                  _storage.coords.ghost.RowCount(), _storage.coords.ghost.RowCount(), cellRows.ghost.RowCount(),
                                  faceRows.ghost.RowCount(), faceRows.ghost.RowCount(), faceRows.ghost.RowCount(), bndRows.ghost.RowCount(),
                                  cellRows.ghost.RowCount(), edgeRows.ghost.RowCount(), edgeRows.ghost.RowCount(), cellRows.ghost.RowCount()};
        const index targetCounts[]{nodeCount, nodeCount, cellCount, cellCount, cellCount, bndCount,
                                   faceCount, nodeCount, cellCount, bndCount, faceCount,
                                   edgeCount, nodeCount, cellCount, cellCount};
        std::vector<bool> requiredFaces(faceCount, false);
        if (requirements.faces && _context.device == Runtime::Device::Host())
        {
            const auto &faceCells = _storage.relations[static_cast<size_t>(MeshRelation::FaceCells)];
            for (size_t face = 0; face < faceCount; ++face)
                DNDS_check_throw_info((_storage.isPeriodic && FaceIDIsPeriodic(_storage.faceElemInfo(face, 0).zone)) || faceCells(face, 0) < 0 ||
                                          faceCells(face, 0) != faceCells(face, 1),
                                      "only a periodic face can reference the same cell on both sides");
            for (size_t i = 0; i < (requirements.allFaceCellsLocal ? faceCount : faceRows.owned.RowCount()); ++i)
                requiredFaces[i] = true;
            const auto &cellFaces = _storage.relations[static_cast<size_t>(MeshRelation::CellFaces)];
            for (size_t i = 0; i < cellFaces.owned.RowCount(); ++i)
                for (size_t j = 0; j < cellFaces.RowSize(i); ++j)
                {
                    const auto face = cellFaces(i, j);
                    DNDS_check_throw_info(face >= 0 && size_t(face) < faceCount,
                                          "owned cell face is outside requested local closure");
                    if (audit.closure)
                        DNDS_check_throw_info(faceCells(size_t(face), 0) == index(i) || faceCells(size_t(face), 1) == index(i),
                                              "owned cell face does not reference its source cell");
                    requiredFaces[size_t(face)] = true;
                }
        }
        for (size_t r = 0; r < static_cast<size_t>(MeshRelation::Count); ++r)
        {
            if (!_storage.relationPresent[r])
                continue;
            const auto &relation = _storage.relations[r];
            DNDS_check_throw_info(relation.owned.RowCount() == sourceOwned[r] && relation.ghost.RowCount() == sourceGhost[r],
                                  "mesh relation owned/ghost rows differ from source entity space");
            if (_context.device != Runtime::Device::Host())
                continue;
            const auto kind = static_cast<MeshRelation>(r);
            const bool requireLocal = kind == MeshRelation::CellNodes || kind == MeshRelation::BoundaryNodes ||
                                      kind == MeshRelation::FaceNodes || kind == MeshRelation::EdgeNodes;
            for (size_t i = 0; i < total(relation); ++i)
                for (size_t j = 0; j < relation.RowSize(i); ++j)
                {
                    const auto target = relation(i, j);
                    const bool validLocal = target >= 0 && size_t(target) < targetCounts[r];
                    // Optional outer rows keep the legacy encoded-unavailable
                    // convention. Inverse support lists can be incomplete even
                    // for owned nodes unless a wider cell closure was requested.
                    bool unavailableAllowed = !requireLocal &&
                                              (i >= relation.owned.RowCount() || kind == MeshRelation::NodeCells ||
                                               kind == MeshRelation::NodeBoundaries);
                    bool absentAllowed = kind == MeshRelation::FaceBoundaries ||
                                         kind == MeshRelation::BoundaryFaces ||
                                         kind == MeshRelation::CellFaceCells ||
                                         (kind == MeshRelation::BoundaryCells && j == 1);
                    if (kind == MeshRelation::FaceCells)
                    {
                        const bool external = FaceIDIsExternalBC(_storage.faceElemInfo(i, 0).zone);
                        absentAllowed = j == 1 && external;
                        unavailableAllowed = !requiredFaces[i] && target != UnInitIndex;
                        DNDS_check_throw_info(j != 1 || external == (target == UnInitIndex),
                                              "mesh face second-cell sentinel disagrees with its boundary zone");
                    }
                    if (!(validLocal || (target == UnInitIndex && absentAllowed) ||
                          (target < 0 && unavailableAllowed)))
                        DNDS_check_throw_info(false, fmt::format(
                                                         "mesh relation {} row {}/{} slot {} target {} is not a valid local entity "
                                                         "(local count {}) or permitted sentinel",
                                                         static_cast<int>(kind), i, total(relation), j, target, targetCounts[r]));
                }
        }
        auto periodicShape = [&](const auto &bits, MeshRelation kind)
        {
            const auto &relation = _storage.relations[static_cast<size_t>(kind)];
            DNDS_check_throw_info(bits.owned.RowCount() == relation.owned.RowCount() && bits.ghost.RowCount() == relation.ghost.RowCount(),
                                  "mesh periodic row distribution mismatch");
            if (_context.device == Runtime::Device::Host())
                for (size_t i = 0; i < total(relation); ++i)
                    DNDS_check_throw_info(bits.RowSize(i) == relation.RowSize(i), "mesh periodic incidence width mismatch");
        };
        if (_storage.isPeriodic)
        {
            periodicShape(_storage.cell2nodePbi, MeshRelation::CellNodes);
            periodicShape(_storage.bnd2nodePbi, MeshRelation::BoundaryNodes);
            if (requirements.faces)
            {
                periodicShape(_storage.cell2facePbi, MeshRelation::CellFaces);
                periodicShape(_storage.face2nodePbi, MeshRelation::FaceNodes);
                if (_context.device == Runtime::Device::Host())
                {
                    const auto &cellFaces = _storage.relations[static_cast<size_t>(MeshRelation::CellFaces)];
                    const auto &faceCells = _storage.relations[static_cast<size_t>(MeshRelation::FaceCells)];
                    for (size_t cell = 0; cell < cellFaces.owned.RowCount(); ++cell)
                        for (size_t slot = 0; slot < cellFaces.RowSize(cell); ++slot)
                        {
                            const auto face = cellFaces(cell, slot);
                            if (faceCells(size_t(face), 0) != index(cell) || faceCells(size_t(face), 1) != index(cell))
                                continue;
                            size_t incidences = 0, backSides = 0;
                            for (size_t other = 0; other < cellFaces.RowSize(cell); ++other)
                                if (cellFaces(cell, other) == face)
                                {
                                    ++incidences;
                                    backSides += !bool(_storage.cell2facePbi(cell, other));
                                }
                            DNDS_check_throw_info(incidences == 2 && backSides == 1,
                                                  "self-periodic face requires exactly two cell-face incidences with opposite side bits");
                        }
                }
            }
            if (requirements.edges)
            {
                periodicShape(_storage.cell2edgePbi, MeshRelation::CellEdges);
                periodicShape(_storage.edge2nodePbi, MeshRelation::EdgeNodes);
            }
        }
    }

    MeshReadAccess::~MeshReadAccess()
    {
        // Keep the mesh/version lease until all device uses have finished, not
        // merely until launch returns. Worker CPU users retain this owner.
        Wait();
    }
    UnstructuredMeshView MeshReadAccess::HostView() const
    {
        DNDS_check_throw_info(_active && !_completed, "mesh read access was moved or sealed");
        DNDS_check_throw_info(_context.device == Runtime::Device::Host(), "HostView requires host acquisition");
        _stamp.Validate();
        return UnstructuredMeshView(_storage);
    }
    UnstructuredMeshView MeshReadAccess::DeviceView() const
    {
        DNDS_check_throw_info(_active && !_completed, "mesh read access was moved or sealed");
        DNDS_check_throw_info(_context.device.IsValid() && _context.device != Runtime::Device::Host(), "DeviceView requires a device acquisition");
        _stamp.Validate();
        return UnstructuredMeshView(_storage);
    }
    void MeshReadAccess::Complete(Runtime::Completion completion)
    {
        DNDS_check_throw_info(_active && !_completed && completion.Matches(_context), "mesh completion context mismatch or already attached");
        _completion = std::move(completion);
        _completed = true;
        for (auto &hold : _holds)
            hold->Complete(_completion);
    }
    void MeshReadAccess::Wait()
    {
        if (!_active)
            return;
        if (!_completed)
            Complete(Runtime::Completion::Record(_context));
        _completion.Wait();
    }

    void MeshReadAccess::Restrict(const MeshReadRequirements &requirements)
    {
        auto hide = [&](std::initializer_list<MeshRelation> relations)
        {
            for (auto relation : relations)
            {
                const auto slot = size_t(relation);
                _storage.relationPresent[slot] = false;
                _storage.relationStates[slot] = Adj_Unknown;
                _storage.relations[slot] = {};
            }
        };
        if (!requirements.faces)
        {
            hide({MeshRelation::CellFaces, MeshRelation::FaceNodes, MeshRelation::FaceCells,
                  MeshRelation::FaceBoundaries, MeshRelation::BoundaryFaces});
            _storage.faceElemInfo = {};
            _storage.cell2facePbi = {};
            _storage.face2nodePbi = {};
        }
        if (!requirements.inverse)
            hide({MeshRelation::NodeCells, MeshRelation::NodeBoundaries});
        if (!requirements.edges)
        {
            hide({MeshRelation::CellEdges, MeshRelation::EdgeNodes, MeshRelation::EdgeCells});
            _storage.edgeElemInfo = {};
            _storage.cell2edgePbi = {};
            _storage.edge2nodePbi = {};
        }
        if (!requirements.cellFaceNeighbors)
            hide({MeshRelation::CellFaceCells});
        if (!requirements.wallDistance)
        {
            _storage.nodeWallDist = {};
            _storage.hasNodeWallDist = false;
        }
    }

}
