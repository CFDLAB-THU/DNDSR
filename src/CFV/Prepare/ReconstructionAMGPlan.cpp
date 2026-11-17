#include "CFV/Plans/ReconstructionAMGPlan.hpp"
#include "CFV/Detail/ReconstructionData.hpp"
#include "CFV/Detail/ReconstructionAMGKernels.hpp"
#include "CFV/CPU/ReconstructionAMGKernels.hpp"
#include "CFV/Math/ReconstructionMatrixAccess.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <climits>

namespace DNDS::CFV
{
    namespace R = Runtime;
    struct ReconstructionAMGPlan::Data
    {
        FiniteVolumePlan metrics;
        std::shared_ptr<const Detail::ReconstructionIdentity> identity;
        R::Device device;
        index rows = 0, entries = 0, boundaryPoints = 0;
        int modes = 0;
        R::Array<int> offsets, columns, deviceOffsets, deviceColumns;
        R::Array<index> incidenceOffsets, neighbors, boundaryOffsets;
        R::Array<real> values, vectors, injection;
        std::vector<R::Array<int>::ReadLease> intReads;
        std::vector<R::Array<index>::ReadLease> indexReads;
        std::vector<R::Array<real>::ReadLease> realReads;
        Detail::ReconstructionAMGGeometry geometry;
        // Destroy the provider before any borrowed fine storage/read scope.
        std::unique_ptr<Algebra::AmgXPreconditioner> provider;
        Data(FiniteVolumePlan geometry, std::shared_ptr<const Detail::ReconstructionIdentity> owner, R::Device placement)
            : metrics(std::move(geometry)), identity(std::move(owner)), device(placement) {}
    };

    ReconstructionAMGPlan ReconstructionAMGPlan::Prepare(const VariationalReconstructionPlan &source,
                                                         R::ExecutionContext context, ReconstructionAMGStorage storage,
                                                         Algebra::AmgXSettings settings)
    {
        source.Validate();
        DNDS_check_throw_info(storage == ReconstructionAMGStorage::PublicUpload || storage == ReconstructionAMGStorage::NativeStorage,
                              "unknown VR AMG storage selection");
        DNDS_check_throw_info(source.Device() == R::Device::Host() && context.device.space == R::MemorySpace::CUDA &&
                                  source._data && !source._data->hasRotatedPeriodicity && source._data->capabilities.sweeps,
                              "VR AMG prototype needs a host sweep plan, CUDA, and no rotated periodicity");
        const auto &host = *source._data;
        const auto &space = host.metrics.Space();
        DNDS_check_throw_info(!space.Distributed() || MPIInfo(space.Communicator()).size == 1,
                              "VR AMG currently supports one rank/device; no hidden principal extraction");
        const index rows = host.metrics.NumCell(), n = host.n;
        DNDS_check_throw_info(n > 0 && n <= 19 && rows <= INT_MAX / n, "VR AMG rows/modes exceed provider extent");
        const auto offsetRead = host.offsets.Read(), neighborRead = host.neighbors.Read(), faceRead = host.faces.Read();
        const auto boundaryRead = host.boundaryOffsets.Read();
        const auto offsets = offsetRead.View(), neighbors = neighborRead.View(), faces = faceRead.View(), boundary = boundaryRead.View();
        const auto matrixRead = host.SweepMatrices().Read(), vectorRead = host.SweepVectors().Read(), injectionRead = host.SweepInjection().Read();
        const auto matrices = matrixRead.View(), vector = vectorRead.View(), injection = injectionRead.View();
        std::optional<R::Array<real>::ReadLease> inverseRead;
        std::optional<R::Array<uint8_t>::ReadLease> sideRead;
        if (host.settings.storage.Raw())
            inverseRead.emplace(host.localInverse->Read());
        if (host.settings.storage.faceCoupling)
            sideRead.emplace(host.rawFaceSide->Read());
        const auto inverse = inverseRead ? inverseRead->View() : R::ArrayView<const real>{};
        const auto sides = sideRead ? sideRead->View() : R::ArrayView<const uint8_t>{};
        std::vector<std::vector<int>> columns(static_cast<size_t>(rows));
        std::vector<int> rowOffsets(size_t(rows + 1), 0);
        for (index iCell = 0; iCell < rows; ++iCell)
        {
            auto &row = columns[size_t(iCell)];
            row.reserve(size_t(offsets(iCell + 1, 0) - offsets(iCell, 0) + 1));
            row.push_back(int(iCell));
            for (index entry = offsets(iCell, 0); entry < offsets(iCell + 1, 0); ++entry)
                if (const index other = neighbors(entry, 0); other != UnInitIndex)
                {
                    DNDS_check_throw_info(other >= 0 && other < rows, "VR AMG requires truly local neighbor columns");
                    row.push_back(int(other));
                }
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            DNDS_check_throw_info(row.size() <= size_t(INT_MAX - rowOffsets[size_t(iCell)]), "VR AMG nonzeros exceed int32");
            rowOffsets[size_t(iCell + 1)] = rowOffsets[size_t(iCell)] + int(row.size());
        }
        const index entries = rowOffsets.back();
        DNDS_check_throw_info(entries + 1 <= INT_MAX / (n * n), "VR AMG numeric extent exceeds provider int32 arithmetic");
        R::Array<int> primaryOffsets(R::RowLayout::Dense(rows + 1, 1)), primaryColumns(R::RowLayout::Dense(entries, 1));
        R::Array<real> numeric(R::RowLayout::Dense((entries + 1) * n * n, 1));
        R::Array<real> normalizedVectors(R::RowLayout::Dense(vector.rows, n)), normalizedInjection(R::RowLayout::Dense(injection.rows, n));
        primaryOffsets.SetName("CFV/AMG/hostRowOffsets", DNDS_ALLOCATION_SITE());
        primaryColumns.SetName("CFV/AMG/hostColumns", DNDS_ALLOCATION_SITE());
        numeric.SetName("CFV/AMG/setupOperator", DNDS_ALLOCATION_SITE());
        normalizedVectors.SetName("CFV/AMG/setupMeanInjection", DNDS_ALLOCATION_SITE());
        normalizedInjection.SetName("CFV/AMG/setupBoundaryInjection", DNDS_ALLOCATION_SITE());
        {
            auto offsetWrite = primaryOffsets.Write(), columnWrite = primaryColumns.Write();
            auto valueWrite = numeric.Write(), vectorWrite = normalizedVectors.Write(), injectionWrite = normalizedInjection.Write();
            const auto ptr = offsetWrite.View(), col = columnWrite.View();
            const auto a = valueWrite.View(), v = vectorWrite.View(), b = injectionWrite.View();
            for (index i = 0; i <= rows; ++i)
                ptr(i, 0) = rowOffsets[size_t(i)];
            for (index iCell = 0; iCell < rows; ++iCell)
                for (index entry = ptr(iCell, 0); entry < ptr(iCell + 1, 0); ++entry)
                    col(entry, 0) = columns[size_t(iCell)][size_t(entry - ptr(iCell, 0))];
            CPU::AssembleReconstructionAMG({ptr.ConstView(), col.ConstView(), offsets, neighbors, faces, boundary,
                                            matrices, vector, injection, inverse, sides, a, v, b, int(n),
                                            host.settings.storage.Raw(), host.settings.storage.inverseTriangle, host.settings.storage.faceCoupling});
            // AmgX's inline-diagonal representation reserves one sentinel block.
            for (index i = entries * n * n; i < (entries + 1) * n * n; ++i)
                a(i, 0) = 0;
        }
        ReconstructionAMGPlan result;
        result._data = std::make_shared<Data>(host.metrics, host.identity, context.device);
        auto &data = *result._data;
        data.rows = rows, data.entries = entries, data.modes = int(n);
        data.boundaryPoints = host.boundaryPoints.RowCount();
        data.offsets = std::move(primaryOffsets), data.columns = std::move(primaryColumns);
        data.deviceOffsets = data.offsets.To(context.device, context), data.deviceColumns = data.columns.To(context.device, context);
        data.incidenceOffsets = host.offsets.To(context.device, context), data.neighbors = host.neighbors.To(context.device, context);
        data.boundaryOffsets = host.boundaryOffsets.To(context.device, context);
        data.values = numeric.To(context.device, context), data.vectors = normalizedVectors.To(context.device, context);
        data.injection = normalizedInjection.To(context.device, context);
        data.values.SetName("CFV/AMG/nativeOperator", DNDS_ALLOCATION_SITE());
        data.vectors.SetName("CFV/AMG/meanInjection", DNDS_ALLOCATION_SITE());
        data.injection.SetName("CFV/AMG/boundaryInjection", DNDS_ALLOCATION_SITE());
        data.deviceOffsets.SetName("CFV/AMG/rowOffsets", DNDS_ALLOCATION_SITE());
        data.deviceColumns.SetName("CFV/AMG/columns", DNDS_ALLOCATION_SITE());
        data.incidenceOffsets.SetName("CFV/AMG/incidenceOffsets", DNDS_ALLOCATION_SITE());
        data.neighbors.SetName("CFV/AMG/neighbors", DNDS_ALLOCATION_SITE());
        data.boundaryOffsets.SetName("CFV/AMG/boundaryOffsets", DNDS_ALLOCATION_SITE());
        settings.blockFormat = Algebra::AmgXBlockFormat::RowMajor;
        data.provider = std::make_unique<Algebra::AmgXPreconditioner>(context.device, settings);
        data.provider->SetPattern(data.offsets, data.columns, int(n), false);
        if (storage == ReconstructionAMGStorage::NativeStorage)
            data.provider->AttachValues(context, data.values);
        else
        {
            const auto read = data.values.Read(context);
            auto view = read.View();
            view.rows = entries * n * n; // The public API owns its own sentinel.
            data.provider->UpdateValuesBorrowed(context, view);
        }
        // These immutable read scopes pin storage and its abstract stream
        // resource. Queued consumers retain Data through completion; after a
        // healthy engine.Close(), late scope fences observe retired completion
        // without another native call. Do not mark a scope complete before its
        // view is formed: Complete intentionally revokes further view access.
        for (const auto *array : {&data.deviceOffsets, &data.deviceColumns})
            data.intReads.push_back(array->Read(context));
        for (const auto *array : {&data.incidenceOffsets, &data.neighbors, &data.boundaryOffsets})
            data.indexReads.push_back(array->Read(context));
        for (const auto *array : {&data.values, &data.vectors, &data.injection})
            data.realReads.push_back(array->Read(context));
        data.geometry = {data.indexReads[0].View(), data.indexReads[1].View(), data.indexReads[2].View(),
                         data.realReads[1].View(), data.realReads[2].View(), data.intReads[0].View(), data.intReads[1].View(), data.realReads[0].View(), int(n)};
        return result;
    }

    ReconstructionAMGWorkspace ReconstructionAMGPlan::BuildWorkspace(index components) const
    {
        DNDS_check_throw_info(_data && components > 0 && components <= INT_MAX / _data->modes, "invalid VR AMG component count");
        ReconstructionAMGWorkspace work{R::Array<real>(R::RowLayout::Dense(_data->rows, components * _data->modes), _data->device),
                                        R::Array<real>(R::RowLayout::Dense(_data->rows * _data->modes, 1), _data->device),
                                        R::Array<real>(R::RowLayout::Dense(_data->rows * _data->modes, 1), _data->device), components};
        work.defect.SetName("CFV/AMG/defect", DNDS_ALLOCATION_SITE());
        work.packedDefect.SetName("CFV/AMG/componentRHS", DNDS_ALLOCATION_SITE());
        work.packedCorrection.SetName("CFV/AMG/componentCorrection", DNDS_ALLOCATION_SITE());
        return work;
    }
    ReconstructionBoundaryValues ReconstructionAMGPlan::PlaceBoundaryValues(const ReconstructionBoundaryValues &source) const
    {
        DNDS_check_throw_info(_data && source.plan == _data->identity, "VR AMG boundary belongs to another reconstruction plan");
        auto result = source;
        result.values = source.values.To(_data->device, {_data->device, nullptr});
        return result;
    }
    void ReconstructionAMGPlan::SubmitDefectCorrection(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                       const FVField &mean, const FVField &input,
                                                       const ReconstructionBoundaryValues &boundary, FVField &output,
                                                       ReconstructionAMGWorkspace &work) const
    {
        DNDS_check_throw_info(_data && engine.Owns(stream) && stream.Context().device == _data->device, "VR AMG stream differs");
        auto data = _data;
        data->metrics.Validate();
        const index components = work.components, width = components * data->modes;
        DNDS_check_throw_info(components > 0 && components <= INT_MAX / data->modes &&
                                  work.defect.RowCount() == data->rows && work.defect.Layout().Width() == width &&
                                  work.packedDefect.RowCount() == data->rows * data->modes && work.packedDefect.Layout().Width() == 1 &&
                                  work.packedCorrection.RowCount() == data->rows * data->modes && work.packedCorrection.Layout().Width() == 1,
                              "VR AMG workspace shape differs");
        for (const auto *field : {&mean, &input, static_cast<const FVField *>(&output)})
            DNDS_check_throw_info(field->Space().Same(data->metrics.Space()) && field->Device() == data->device, "VR AMG field space/device differs");
        DNDS_check_throw_info(mean.Width() == components && input.Width() == width && output.Width() == width &&
                                  boundary.plan == data->identity && boundary.values.GetDevice() == data->device &&
                                  boundary.values.RowCount() == data->boundaryPoints &&
                                  (!data->boundaryPoints || boundary.values.Layout().Width() == components) &&
                                  !input.Owned().Storage().Overlaps(output.Owned().Storage()) &&
                                  !mean.Owned().Storage().Overlaps(output.Owned().Storage()),
                              "VR AMG field width/alias differs");
        const std::array<const R::Array<real> *, 7> operands{
            &mean.Owned(), &input.Owned(), &output.Owned(), &work.defect, &work.packedDefect, &work.packedCorrection, &boundary.values};
        for (const auto *array : operands)
            DNDS_check_throw_info(array->GetDevice() == data->device && array->Layout().Kind() == R::RowLayoutKind::Table,
                                  "VR AMG requires resident AoS fields/scratch");
        const std::array<const R::Array<real> *, 3> scratch{&work.defect, &work.packedDefect, &work.packedCorrection};
        for (index i = 0; i < index(scratch.size()); ++i)
        {
            for (index j = 0; j < i; ++j)
                DNDS_check_throw_info(!scratch[size_t(i)]->Storage().Overlaps(scratch[size_t(j)]->Storage()), "VR AMG scratch aliases");
            const std::array<const R::Array<real> *, 5> inputs{
                &mean.Owned(), &input.Owned(), &output.Owned(), &boundary.values, &data->values};
            for (const auto *array : inputs)
                DNDS_check_throw_info(!scratch[size_t(i)]->Storage().Overlaps(array->Storage()), "VR AMG scratch aliases an operand");
        }
        R::SubmitArrays(engine, stream, [data](R::ExecutionContext context, auto means, auto values, auto bc, auto defect)
                        { Detail::BuildReconstructionAMGDefect(context, data->geometry, means, values, bc, defect); }, R::Read(mean.Owned()), R::Read(input.Owned()), R::Read(boundary.values), R::Write(work.defect));
        // TODO(LVR_AMGX, deferred): the true gated residual is
        // (I-D)*(b-A*x)+D*(target-x); reuse this base hierarchy as a preconditioner
        // rather than blending a solved base result. See docs/guides/amgx.md.
        // TODO(AMGX_STREAM_ORDERED, deferred): queue component adaptation and a
        // preallocated provider V-cycle on the managed caller stream.
        // The current provider is a deliberate host boundary. Drain MPI progress
        // before any provider-wide CUDA join, identically to the flow adapter.
        for (index component = 0; component < components; ++component)
        {
            R::SubmitArrays(engine, stream, [data, component](R::ExecutionContext context, auto source, auto packed)
                            { Detail::GatherReconstructionAMGComponent(context, source, packed, component, data->modes); }, R::Read(work.defect), R::Write(work.packedDefect));
            engine.Synchronize();
            {
                const auto read = work.packedDefect.Read(stream.Context());
                auto write = work.packedCorrection.Write(stream.Context());
                data->provider->ApplyBorrowed(stream.Context(), read.View(), write.View());
            }
            R::SubmitArrays(engine, stream, [data, component](R::ExecutionContext context, auto source, auto packed, auto destination)
                            { Detail::ScatterReconstructionAMGComponent(context, source, packed, destination, component, data->modes); }, R::Read(input.Owned()), R::Read(work.packedCorrection), R::Write(output.Owned()));
        }
    }
    Algebra::AmgXMemorySnapshot ReconstructionAMGPlan::MemorySnapshot() const
    {
        DNDS_check_throw_info(_data, "empty VR AMG plan");
        return _data->provider->MemorySnapshot();
    }
    const R::Array<real> &ReconstructionAMGPlan::MatrixValues() const
    {
        DNDS_check_throw_info(_data, "empty VR AMG plan");
        return _data->values;
    }
    const R::Array<int> &ReconstructionAMGPlan::RowOffsets() const
    {
        DNDS_check_throw_info(_data, "empty VR AMG plan");
        return _data->offsets;
    }
    const R::Array<int> &ReconstructionAMGPlan::Columns() const
    {
        DNDS_check_throw_info(_data, "empty VR AMG plan");
        return _data->columns;
    }
#ifndef DNDS_USE_CUDA
    namespace Detail
    {
        void BuildReconstructionAMGDefect(R::ExecutionContext, ReconstructionAMGGeometry, R::ArrayView<const real>, R::ArrayView<const real>, R::ArrayView<const real>, R::ArrayView<real>)
        {
            DNDS_check_throw_info(false, "CUDA VR AMG is unavailable");
        }
        void GatherReconstructionAMGComponent(R::ExecutionContext, R::ArrayView<const real>, R::ArrayView<real>, index, int)
        {
            DNDS_check_throw_info(false, "CUDA VR AMG is unavailable");
        }
        void ScatterReconstructionAMGComponent(R::ExecutionContext, R::ArrayView<const real>, R::ArrayView<const real>, R::ArrayView<real>, index, int)
        {
            DNDS_check_throw_info(false, "CUDA VR AMG is unavailable");
        }
    }
#endif
}
