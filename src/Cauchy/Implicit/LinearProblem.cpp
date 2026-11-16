#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Implicit/CPU/LinearBlockKernels.hpp"
#include "Cauchy/Implicit/CPU/LinearProblemKernels.hpp"
#include "DNDS/MPI/Failure.hpp"
#include <limits>
#include <optional>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        index DiagonalWidth(index components, DiagonalStorage mode)
        {
            DNDS_check_throw_info(components > 0 && components <= std::numeric_limits<rowsize>::max() / components,
                                  "invalid linear block dimension");
            DNDS_check_throw_info(mode == DiagonalStorage::Entries || mode == DiagonalStorage::Block, "invalid linear diagonal mode");
            return mode == DiagonalStorage::Block ? components * components : components;
        }
        std::vector<index> Counts(const std::vector<std::vector<index>> &columns)
        {
            std::vector<index> result;
            result.reserve(columns.size());
            for (const auto &row : columns)
                result.push_back(index(row.size()));
            return result;
        }

    }
    std::shared_ptr<const LinearProblemContext::Pattern> LinearProblemContext::BuildPattern(
        const Algebra::DofSpace &space, std::vector<std::vector<index>> columns, std::vector<index> core, LinearSweepPartitions partitions, LinearIndexStorage indices)
    {
        DNDS_check_throw_info(space.OwnedRows() <= size_t(INT64_MAX) && space.GhostRows() <= size_t(INT64_MAX) - space.OwnedRows(),
                              "linear row space exceeds signed index range");
        auto data = std::make_shared<Pattern>(space.Pin());
        std::vector<index> dependent, order, position;
        const index rows = index(data->partition->owned), allRows = rows + index(data->partition->ghost);
        if (partitions.starts.empty())
            partitions.starts = {0, rows};
        DNDS_check_throw_info(partitions.starts.size() >= 2 && partitions.starts.front() == 0 &&
                                  partitions.starts.back() == rows && std::is_sorted(partitions.starts.begin(), partitions.starts.end()),
                              "invalid linear SGS subpartition starts");
        data->sweepPartitions = R::MetadataBuffer<index>(std::move(partitions.starts));
        DNDS_check_throw_info(index(columns.size()) == rows, "linear stencil row count differs from its DOF space");
        std::vector<unsigned char> isCore(rows, 0);
        for (auto iCell : core)
        {
            DNDS_check_throw_info(iCell >= 0 && iCell < rows && !isCore[iCell], "invalid or duplicate linear core row");
            isCore[iCell] = 1;
        }
        for (index iCell = 0; iCell < rows; ++iCell)
        {
            auto sorted = columns[iCell];
            std::sort(sorted.begin(), sorted.end());
            DNDS_check_throw_info(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "duplicate linear stencil column");
            for (auto column : sorted)
            {
                DNDS_check_throw_info(column >= 0 && column < allRows && column != iCell && (!isCore[iCell] || column < rows),
                                      "invalid linear stencil column or core ghost dependency");
                data->hasGhostColumns |= column >= rows;
            }
            if (!isCore[iCell])
                dependent.push_back(iCell);
        }
        order = core;
        order.insert(order.end(), dependent.begin(), dependent.end());
        position.resize(rows);
        for (index i = 0; i < rows; ++i)
            position[order[i]] = i;
        data->rows = LinearPatternStorage(columns, order, position, core, dependent, indices);
        return data;
    }
    LinearProblemContext::LinearProblemContext(Algebra::DofSpace space, index components,
                                               std::vector<std::vector<index>> columns,
                                               std::vector<index> core, DiagonalStorage diagonal, OffDiagonalStorage offDiagonal, R::Device device, LinearSweepPartitions partitions, LinearMatrixOptions matrixOptions)
        : _space(std::move(space)), _pattern(BuildPattern(_space, std::move(columns), std::move(core), std::move(partitions), matrixOptions.indices)), _components(components), _diagonalStorage(diagonal), _rowMajorBlocks(matrixOptions.rowMajorBlocks),
          _diagonal(matrixOptions.diagonal.MakeLayout(_pattern->partition->owned, DiagonalWidth(components, diagonal)), device, matrixOptions.diagonal.Allocation()),
          _inverse(matrixOptions.inverse.MakeLayout(_pattern->partition->owned, DiagonalWidth(components, diagonal)), device, matrixOptions.inverse.Allocation()), _offDiagonalStorage(offDiagonal)
    {
        _diagonal.SetName("Cauchy/linear/diagonal", DNDS_ALLOCATION_SITE());
        _inverse.SetName("Cauchy/linear/inverse", DNDS_ALLOCATION_SITE());
        DNDS_check_throw_info(offDiagonal == OffDiagonalStorage::Assembled || offDiagonal == OffDiagonalStorage::DirectEvaluated,
                              "invalid off-diagonal storage mode");
        if (offDiagonal == OffDiagonalStorage::Assembled)
        {
            _offDiagonal.emplace(Counts(Columns()), components, device, matrixOptions.offDiagonal, matrixOptions.rowMajorBlocks);
            _offDiagonal->Values().SetName("Cauchy/linear/offDiagonal/values", DNDS_ALLOCATION_SITE());
            _offDiagonal->OffsetArray().SetName("Cauchy/linear/offDiagonal/blockOffsets", DNDS_ALLOCATION_SITE());
        }
        if (device != R::Device::Host())
        {
            auto placed = std::make_shared<Pattern>(_pattern->partition);
            placed->hostRows = std::make_shared<const LinearPatternStorage>(_pattern->rows);
            placed->rows = _pattern->rows.Placed(device, {device, nullptr});
            placed->hasGhostColumns = _pattern->hasGhostColumns;
            placed->sweepPartitions = _pattern->sweepPartitions;
            _pattern = std::move(placed);
        }
        _diagonalStoragePin = _diagonal.SharedStorage();
        _diagonalShape = _diagonal.ShapeVersion();
        if (_offDiagonal)
            _offDiagonalStoragePin = _offDiagonal->Values().SharedStorage();
        PrepareActionWorkspace();
    }
    void LinearProblemContext::ValidateMatrix() const
    {
        Layout();
        DNDS_check_throw_info(_diagonal.SharedStorage() == _diagonalStoragePin && _diagonal.ShapeVersion() == _diagonalShape &&
                                  (!_offDiagonal || _offDiagonal->Values().SharedStorage() == _offDiagonalStoragePin),
                              "linear matrix allocation or shape changed");
    }
    void LinearProblemContext::ValidateField(const Field &field) const
    {
        Layout();
        field.Validate();
        DNDS_check_throw_info(field.Space().Same(_space) && index(field.Width()) == _components && field.Device() == R::Device::Host(),
                              "linear field space, component count or backend differs");
    }
    void LinearProblemContext::ValidateOutputs(const Field &input, const Field &output) const
    {
        ValidateField(input);
        ValidateField(output);
        for (const auto *source : {&input.Owned(), &input.Ghost(), &_diagonal, &_inverse})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(output.Owned().Storage(), source->Storage()), "linear output aliases input or matrix storage");
        if (_offDiagonal)
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(output.Owned().Storage(), _offDiagonal->Values().Storage()), "linear output aliases matrix storage");
    }
    Algebra::BlockMatrixStorage &LinearProblemContext::OffDiagonal()
    {
        DNDS_check_throw_info(_offDiagonal, "semi-matrix-free context has no assembled off-diagonal storage");
        return *_offDiagonal;
    }
    const Algebra::BlockMatrixStorage &LinearProblemContext::OffDiagonal() const
    {
        DNDS_check_throw_info(_offDiagonal, "semi-matrix-free context has no assembled off-diagonal storage");
        return *_offDiagonal;
    }
    void LinearProblemContext::PublishOffDiagonalEvaluator(std::shared_ptr<const OffDiagonalEvaluator> evaluator)
    {
        ValidateMatrix();
        DNDS_check_throw_info(!_offDiagonal && evaluator && evaluator->Device() == Device() && evaluator->Components() == _components &&
                                  evaluator->RowSpaceIdentity() == _pattern->partition &&
                                  _offDiagonalGeneration < UINT64_MAX,
                              "invalid evaluated off-diagonal publication");
        // Compare retained CSR rows directly. Columns() is a diagnostic copy;
        // rebuilding nested host vectors here would allocate every update.
        const auto patternRead = HostPattern().Read();
        const auto columns = patternRead.View().columns;
        DNDS_check_throw_info(evaluator->RowNumber() == columns.RowCount(), "evaluated row count differs");
        for (index iRow = 0; iRow < columns.RowCount(); ++iRow)
        {
            const auto expected = columns[iRow];
            const auto actual = evaluator->ColumnRow(iRow);
            DNDS_check_throw_info(actual.size() == expected.size() &&
                                      std::equal(actual.begin(), actual.end(), expected.begin()),
                                  "evaluated column order differs");
        }
        auto publication = _diagonal.Write({Device(), nullptr});
        _offDiagonalEvaluator = std::move(evaluator);
        ++_offDiagonalGeneration;
        publication.CancelUnmodified();
    }
    uint64_t LinearProblemContext::OffDiagonalGeneration() const
    {
        return _offDiagonal ? _offDiagonal->Values().Version() : _offDiagonalGeneration;
    }
    LinearProblemContext::OffDiagonalReadAccess LinearProblemContext::ReadOffDiagonal(R::ExecutionContext context) const
    {
        ValidateMatrix();
        if (_offDiagonal)
            return {_offDiagonal->Read(context), _components};
        auto publication = _diagonal.Read(context);
        DNDS_check_throw_info(_offDiagonalEvaluator, "semi-matrix-free off-diagonal data have not been prepared");
        return {std::move(publication), _offDiagonalEvaluator, _components};
    }
    void LinearProblemContext::InvertDiagonal(int refinementSteps)
    {
        DNDS_check_throw_info(!_inversePending, "diagonal inverse preparation is already pending");
        DNDS_check_throw_info(refinementSteps >= 0 && refinementSteps <= 3 &&
                                  (!refinementSteps || _diagonalStorage == DiagonalStorage::Block),
                              "inverse refinement requires block storage and 0..3 corrections");
        _inverseCurrent = false;
        _inverseFallbacks = 0;
        ValidateMatrix();
        DNDS_check_throw_info(_diagonal.GetDevice() == R::Device::Host() && _diagonal.RowCount() == _inverse.RowCount() && _diagonal.Layout().Width() == _inverse.Layout().Width(), "linear diagonal shape/backend changed");
        const auto read = _diagonal.Read();
        std::optional<R::Array<real>::WriteLease> write(_inverse.Write());
        const auto input = read.View<R::UniformTableRows>();
        const auto output = write->View<R::UniformTableRows>();
        const index width = DiagonalWidth(_components, _diagonalStorage);
        DNDS_check_throw_info(input.RowCount() == index(_pattern->partition->owned) && input.Width() == width, "linear diagonal shape changed");
        _inverseFallbacks = RequireNumericalSuccess(CPU::InvertLinearDiagonal(
                                                        {input, output, _components,
                                                         _diagonalStorage == DiagonalStorage::Block, refinementSteps, RowMajorBlocks()}),
                                                    "InvertLinearDiagonal");
        _inverseSVDCells = _inverseFallbacks;
        _diagonalVersion = _diagonal.Version();
        // Publish before recording the inverse generation.
        write.reset();
        _inverseVersion = _inverse.Version();
        _inverseSourceEpoch = _diagonal.MutationEpoch();
        _inverseEpoch = _inverse.MutationEpoch();
        _inverseCurrent = true;
    }
    void LinearProblemContext::Apply(const Field &input, Field &output, LinearRegion region) const
    {
        ValidateOutputs(input, output);
        ValidateMatrix();
        DNDS_check_throw_info(region == LinearRegion::All || region == LinearRegion::Core || region == LinearRegion::HaloDependent, "invalid linear operator region");
        const auto owned = input.Owned().Read();
        std::optional<R::Array<real>::ReadLease> ghost;
        if (region != LinearRegion::Core && _pattern->hasGhostColumns)
            ghost.emplace(input.Ghost().Read());
        const auto diagonal = _diagonal.Read();
        const auto patternRead = _pattern->rows.Read();
        const auto pattern = patternRead.View();
        const auto offDiagonal = ReadOffDiagonal();
        auto destination = output.Owned().Write();
        const auto remote = ghost ? ghost->View<R::UniformTableRows>() : CPU::LinearReadRows{};
        const auto rows = region == LinearRegion::All ? pattern.order : region == LinearRegion::Core ? pattern.core
                                                                                                     : pattern.dependent;
        CPU::ApplyLinearRows({offDiagonal.View(), pattern, rows, diagonal.View<R::UniformTableRows>(), _components, _diagonalStorage == DiagonalStorage::Block, {}, _rowMajorBlocks},
                             owned.View<R::UniformTableRows>(),
                             remote, destination.View<R::UniformTableRows>());
    }
    void LinearProblemContext::JacobiImpl(const Field &rhs, const Field *input, Field &output, LinearRegion region) const
    {
        ValidateMatrix();
        ValidateOutputs(rhs, output);
        if (input)
            ValidateOutputs(*input, output);
        DNDS_check_throw_info(region == LinearRegion::All || region == LinearRegion::Core || region == LinearRegion::HaloDependent, "invalid Jacobi region");
        const auto diagonal = _diagonal.Read();
        const auto inverse = _inverse.Read();
        DNDS_check_throw_info(_inverseCurrent && _diagonal.Version() == _diagonalVersion && _inverse.Version() == _inverseVersion,
                              "Jacobi diagonal inverse is missing or stale");
        const auto right = rhs.Owned().Read();
        std::optional<R::Array<real>::ReadLease> owned, ghost;
        if (input)
        {
            owned.emplace(input->Owned().Read());
            if (region != LinearRegion::Core && _pattern->hasGhostColumns)
                ghost.emplace(input->Ghost().Read());
        }
        const auto patternRead = _pattern->rows.Read();
        const auto pattern = patternRead.View();
        const auto blocks = ReadOffDiagonal();
        auto destination = output.Owned().Write();
        const auto b = right.View<R::UniformTableRows>();
        const auto x = owned ? owned->View<R::UniformTableRows>() : b;
        const auto remote = ghost ? ghost->View<R::UniformTableRows>() : CPU::LinearReadRows{};
        const auto rows = region == LinearRegion::All ? pattern.order : region == LinearRegion::Core ? pattern.core
                                                                                                     : pattern.dependent;
        try
        {
            CPU::JacobiLinearRows({blocks.View(), pattern, rows, inverse.View<R::UniformTableRows>(), _components, _diagonalStorage == DiagonalStorage::Block, {}, _rowMajorBlocks},
                                  b, x, remote,
                                  destination.View<R::UniformTableRows>(), !input);
        }
        catch (...)
        {
            destination.Fail();
            MPI::AbortIfSolverFailure("Cauchy Jacobi kernel failed");
            throw;
        }
    }
    void LinearProblemContext::Jacobi(const Field &rhs, const Field &input, Field &output, LinearRegion region) const { JacobiImpl(rhs, &input, output, region); }
    void LinearProblemContext::JacobiZero(const Field &rhs, Field &output, LinearRegion region) const { JacobiImpl(rhs, nullptr, output, region); }

    void LinearProblemContext::ApplySGS(const Field &rhs, const Field &input, Field &forward, Field &output) const
    {
        auto iteration = BeginSGS(rhs, input, forward, output);
        iteration.Lower();
        iteration.Upper();
        iteration.Finish();
    }
}
