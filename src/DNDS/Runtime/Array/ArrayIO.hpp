#pragma once

#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Storage/ElementSchema.hpp"
#include "DNDS/Serializer/SerializerBase.hpp"
#include <functional>
#include <optional>

namespace DNDS::Runtime
{
    template <class T>
    std::string ArraySchema() { return ElementSchema::For<T>().Identity(); }

    namespace Detail
    {
        /// Required checkpoint acceptance, independent of transfer preflight.
        /// A rank's file/codec/shape failure must not let peers enter the next
        /// collective dataset operation. Per-rank serializers check locally;
        /// mesh/checkpoint owners coordinate their wider transaction separately.
        inline void IOCheck(bool valid, const MPIInfo &mpi, bool perRank, const char *message)
        {
            int all = valid;
            if (!perRank)
                MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_INT, MPI_MIN, mpi.comm);
            DNDS_check_throw_info(all, message);
        }
        /// Pin the published shape before entering collective I/O. A moved
        /// handle or exclusive shape edit on one rank must reject every rank
        /// before any checkpoint group or payload is written. Shared by the
        /// typed C++ and dynamic Python frontends.
        template <class ArrayHandle>
        ArrayStorage::Shared IOPublishedStorage(const ArrayHandle &array, const MPIInfo &mpi, bool perRank)
        {
            ArrayStorage::Shared storage;
            try
            {
                storage = array.SharedStorage();
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("runtime array I/O preparation failed");
            }
            IOCheck(bool(storage), mpi, perRank,
                    "Array checkpoint requires a live published source on every rank");
            return storage;
        }
        inline void IOAgree(const std::string &value, const MPIInfo &mpi, bool perRank)
        {
            if (perRank)
                return;
            IOCheck(value.size() <= INT32_MAX, mpi, false, "checkpoint metadata too long");
            int size = mpi.rank == 0 ? int(value.size()) : 0;
            MPI_Bcast(&size, 1, MPI_INT, 0, mpi.comm);
            std::string root(size, '\0');
            if (mpi.rank == 0)
                root = value;
            MPI_Bcast(root.data(), size, MPI_CHAR, 0, mpi.comm);
            IOCheck(value == root, mpi, false, "checkpoint name/schema differs across ranks");
        }
        struct SerializerPath
        {
            Serializer::SerializerBaseSSP serializer;
            std::string path;
            explicit SerializerPath(Serializer::SerializerBaseSSP value) : serializer(std::move(value)), path(serializer->GetCurrentPath()) {}
            ~SerializerPath() { serializer->GoToPath(path); }
        };
        inline int NativeByteOrder()
        {
            const uint16_t one = 1;
            return *reinterpret_cast<const uint8_t *>(&one) == 1 ? 0 : 1;
        }
        inline int FileByteOrder(const Serializer::SerializerBaseSSP &serializer, bool writing)
        {
            SerializerPath guard(serializer);
            serializer->GoToPath("/");
            int endian = NativeByteOrder();
            if (serializer->ListCurrentPath().count("runtime_array_byte_order"))
                serializer->ReadInt("runtime_array_byte_order", endian);
            else
            {
                DNDS_check_throw_info(writing, "native Array archive lacks file byte order");
                serializer->WriteInt("runtime_array_byte_order", endian);
            }
            DNDS_check_throw_info(endian == 0 || endian == 1, "invalid native Array file byte order");
            DNDS_check_throw_info(!writing || endian == NativeByteOrder(), "cannot append native bytes with a different file byte order");
            return endian;
        }
        inline std::vector<Serializer::ByteMemorySegment> IOSegments(const RowLayout &layout, size_t bytes)
        {
            std::vector<Serializer::ByteMemorySegment> segments;
            for (index row = 0; row < layout.RowCount(); ++row)
            {
                const auto count = CheckedSize::Multiply(size_t(layout.RowSize(row)), bytes);
                if (!count)
                    continue;
                const auto start = CheckedSize::Multiply(size_t(layout.Offset(row)), bytes);
                if (!segments.empty() && segments.back().offset + segments.back().size == DNDS::index(start))
                    segments.back().size += DNDS::index(count);
                else
                    segments.push_back({DNDS::index(start), DNDS::index(count)});
            }
            return segments;
        }
        /// One component in increasing logical-row order. This order is also
        /// monotonic in physical memory for every layout. For a variable-row
        /// destination the caller has verified that all rows have the saved width.
        inline std::vector<Serializer::ByteMemorySegment> IOColumnSegments(const RowLayout &layout, size_t bytes, index iComp)
        {
            DNDS_check_throw_info(iComp >= 0 && (layout.IsVariable() || iComp < layout.Width()), "Array component selection exceeds row width");
            std::vector<Serializer::ByteMemorySegment> segments;
            if (!layout.RowCount())
                return segments;
            const index chunk = layout.Kind() == RowLayoutKind::ColMajorTable ? layout.RowCount() : layout.Kind() == RowLayoutKind::TiledColMajorTable ? layout.Stride()
                                                                                                                                                       : 1;
            for (index iRow = 0; iRow < layout.RowCount();)
            {
                const auto count = std::min(chunk, layout.RowCount() - iRow);
                segments.push_back({index(CheckedSize::Multiply(size_t(layout.ElementOffset(iRow, iComp)), bytes)),
                                    index(CheckedSize::Multiply(size_t(count), bytes))});
                iRow += count;
            }
            return segments;
        }
        /// Read one packed-row dataset. HDF5 ranks enter the same number of
        /// calls even when they choose different physical layouts or row widths.
        inline void ReadRowPayload(Serializer::SerializerBase &serializer, const std::string &name,
                                   uint8_t *pointer, index memoryBytes, const RowLayout &layout, size_t bytes,
                                   Serializer::ArrayGlobalOffset window, const MPIInfo &mpi, bool perRank)
        {
            const index columns = layout.IsContiguousRows() || !layout.RowCount() ? 0 : layout.Width();
            const index passes = perRank ? columns : MPI::AllreduceValue<MPI::Reduction::Max>(columns, mpi.comm);
            if (!passes)
            {
                auto region = perRank ? Serializer::ArrayGlobalOffset_Unknown : window;
                serializer.ReadByteSegments(name, pointer, memoryBytes, IOSegments(layout, bytes), region);
                return;
            }
            IOCheck(layout.IsContiguousRows() ||
                        window.size() == index(CheckedSize::Multiply(CheckedSize::Multiply(size_t(layout.RowCount()), size_t(layout.Width())), bytes)),
                    mpi, perRank, "strided native read requires uniform source rows");
            for (index j = 0; j < passes; ++j)
            {
                Serializer::ByteFilePattern pattern;
                std::vector<Serializer::ByteMemorySegment> segments;
                if (j < columns)
                {
                    pattern = {window.offset() + j * index(bytes), index(bytes), layout.Width() * index(bytes), layout.RowCount()};
                    segments = IOColumnSegments(layout, bytes, j);
                }
                else if (!j && layout.IsContiguousRows() && window.size())
                {
                    pattern = {window.offset(), window.size(), window.size(), 1};
                    segments = IOSegments(layout, bytes);
                }
                serializer.ReadBytePattern(name, pointer, memoryBytes, segments, pattern);
            }
        }
        inline std::vector<DNDS::index> IOCounts(DNDS::index local, const MPIInfo &mpi, bool perRank)
        {
            std::vector<DNDS::index> result(perRank ? 1 : mpi.size);
            if (perRank)
                result[0] = local;
            else
                MPI_Allgather(&local, 1, DNDS_MPI_INDEX, result.data(), 1, DNDS_MPI_INDEX, mpi.comm);
            return result;
        }
        inline DNDS::index IOSum(const std::vector<DNDS::index> &values, size_t count)
        {
            DNDS::index result = 0;
            for (size_t i = 0; i < count; ++i)
            {
                DNDS_check_throw_info(values[i] >= 0 && values[i] <= INT64_MAX - result, "native Array global count overflow");
                result += values[i];
            }
            return result;
        }
        inline Serializer::ArrayGlobalOffset IORows(Serializer::ArrayGlobalOffset requested,
                                                    const std::vector<DNDS::index> &counts, const MPIInfo &mpi, bool perRank)
        {
            const auto total = IOSum(counts, counts.size());
            if (requested == Serializer::ArrayGlobalOffset_Unknown)
            {
                DNDS_check_throw_info(perRank || counts.size() == size_t(mpi.size), "matching-rank Array read requires writer rank count");
                return {counts[perRank ? 0 : mpi.rank], perRank ? 0 : IOSum(counts, mpi.rank)};
            }
            DNDS_check_throw_info(!perRank, "per-rank JSON Array reads require the saved local partition");
            if (requested == Serializer::ArrayGlobalOffset_EvenSplit)
            {
                const auto begin = (total / mpi.size) * mpi.rank + (total % mpi.size) * mpi.rank / mpi.size;
                const auto end = (total / mpi.size) * (mpi.rank + 1) + (total % mpi.size) * (mpi.rank + 1) / mpi.size;
                return {end - begin, begin};
            }
            DNDS_check_throw_info(requested.isDist() && requested.size() >= 0 && requested.offset() <= total && requested.size() <= total - requested.offset(),
                                  "native Array read row partition is outside the dataset");
            return requested;
        }
        inline Serializer::ArrayGlobalOffset IOCheckedRows(Serializer::ArrayGlobalOffset requested,
                                                           const std::vector<DNDS::index> &counts, const MPIInfo &mpi, bool perRank)
        {
            std::optional<Serializer::ArrayGlobalOffset> region;
            try
            {
                region.emplace(IORows(requested, counts, mpi, perRank));
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("runtime array I/O preparation failed");
            }
            IOCheck(bool(region), mpi, perRank, "native Array row partition is invalid or differs from saved rank metadata");
            return *region;
        }
        template <class T>
        ElementSchema IOSchema(const std::string &requested, const MPIInfo &mpi, bool perRank)
        {
            std::optional<ElementSchema> schema;
            bool valid = true;
            try
            {
                schema.emplace(ElementSchema::For<T>());
                valid = requested.empty() || requested == schema->Identity();
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("runtime array I/O preparation failed");
                valid = false;
            }
            IOCheck(valid, mpi, perRank, "Array element schema is unregistered or requested identity differs from its registration");
            return *schema;
        }
    }

    /// Native host I/O directly from retained storage; no legacy Array or
    /// whole-payload packing allocation. Metadata scratch is O(rows+ranks).
    /// JSON keeps its required document representation. Finish producers and
    /// match collective order; device storage requires a direct I/O backend.
    inline void WriteArrayStorage(const Buffer &storage, const RowLayout &layout, const ElementSchema &schema,
                                  Serializer::SerializerBaseSSP serializer, const std::string &name, const MPIInfo &mpi)
    {
        DNDS_check_throw_info(serializer, "null Array serializer");
        const bool perRank = serializer->IsPerRank();
        const auto bytes = schema.Extent();
        Detail::IOAgree(name + ":" + schema.Identity() + ":" + schema.Definition() + ":" + std::to_string(Detail::NativeByteOrder()), mpi, perRank);
        bool valid = bytes && layout.RowCount() <= size_t(INT64_MAX) && storage.Bytes() <= size_t(INT64_MAX) && storage.GetDevice() == Device::Host();
        valid = valid && (layout.IsVariable() || layout.Width() <= size_t(INT64_MAX) / bytes);
        valid = valid && storage.Bytes() / bytes == layout.Elements() && storage.Bytes() % bytes == 0;
        size_t elements = 0;
        for (size_t row = 0; row < layout.RowCount(); ++row)
        {
            valid = valid && layout.RowSize(row) <= size_t(INT64_MAX) / bytes && layout.RowSize(row) <= SIZE_MAX - elements;
            if (valid)
                elements += layout.RowSize(row);
        }
        valid = valid && elements <= size_t(INT64_MAX) / bytes;
        Detail::IOCheck(valid, mpi, perRank, "native Array I/O requires valid host storage and representable counts; device staging is disabled");
        std::optional<ReadAccess> read;
        try
        {
            read.emplace(storage.Read());
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("runtime array I/O preparation failed");
            valid = false;
        }
        Detail::IOCheck(valid, mpi, perRank, "Array checkpoint cannot acquire its source read lease");
        int columns = !layout.IsContiguousRows();
        if (!perRank)
            MPI_Allreduce(MPI_IN_PLACE, &columns, 1, MPI_INT, MPI_MAX, mpi.comm);
        int dense = !layout.IsVariable();
        if (!perRank)
            MPI_Allreduce(MPI_IN_PLACE, &dense, 1, MPI_INT, MPI_MIN, mpi.comm);
        Detail::IOCheck(!columns || dense, mpi, perRank, "column-encoded checkpoints require uniform rows on every rank");
        if (dense)
            Detail::IOAgree(std::to_string(layout.Width()), mpi, perRank);
        const auto rows = Detail::IOCounts(DNDS::index(layout.RowCount()), mpi, perRank);
        const auto counts = Detail::IOCounts(DNDS::index(elements), mpi, perRank);
        const auto globalRows = Detail::IOSum(rows, rows.size()), globalElements = Detail::IOSum(counts, counts.size());
        Detail::IOCheck(globalElements <= INT64_MAX / DNDS::index(bytes) && globalRows < INT64_MAX, mpi, perRank, "native Array file size overflow");
        const auto rowStart = perRank ? 0 : Detail::IOSum(rows, mpi.rank);
        const auto elementStart = perRank ? 0 : Detail::IOSum(counts, mpi.rank);
        Detail::FileByteOrder(serializer, true);
        Detail::SerializerPath guard(serializer);
        serializer->CreatePath(name);
        serializer->GoToPath(name);
        serializer->WriteInt("runtime_array_format", columns ? 4 : 3);
        serializer->WriteIndex("element_bytes", bytes);
        serializer->WriteInt("writer_ranks", mpi.size);
        serializer->WriteString("element_schema", schema.Identity());
        serializer->WriteString("element_definition", schema.Definition());
        serializer->WriteString("cpp_type_name", schema.CppTypeName());
        serializer->WriteInt("row_encoding", dense ? 0 : 1);
        serializer->WriteIndex("row_width", dense ? DNDS::index(layout.Width()) : -1);
        serializer->WriteIndex("rows_global", globalRows);
        serializer->WriteIndex("elements_global", globalElements);
        serializer->WriteIndexVector("row_counts", {DNDS::index(layout.RowCount())}, Serializer::ArrayGlobalOffset_Parts);
        if (columns)
        {
            const auto padding = layout.PaddingOptions();
            serializer->WriteIndexVector("physical_layout", {index(layout.Kind()), layout.Stride(), padding.rowPaddingMultiple, padding.extraRows},
                                         perRank ? Serializer::ArrayGlobalOffset_Unknown : Serializer::ArrayGlobalOffset_Parts);
        }
        if (!dense)
        {
            std::vector<DNDS::index> offsets;
            offsets.reserve(layout.RowCount() + 1);
            DNDS::index cursor = elementStart;
            for (size_t row = 0; row < layout.RowCount(); ++row)
            {
                offsets.push_back(cursor);
                cursor += DNDS::index(layout.RowSize(row));
            }
            if (perRank || mpi.rank == mpi.size - 1)
                offsets.push_back(cursor);
            serializer->WriteIndexVector("row_offsets", offsets, perRank ? Serializer::ArrayGlobalOffset_Unknown : Serializer::ArrayGlobalOffset{globalRows + 1, rowStart});
        }
        if (columns)
            for (index iComp = 0; iComp < layout.Width(); ++iComp)
                serializer->WriteByteSegments("values_" + std::to_string(iComp), read->View<uint8_t>().data, index(storage.Bytes()),
                                              Detail::IOColumnSegments(layout, bytes, iComp),
                                              perRank ? Serializer::ArrayGlobalOffset_Unknown : Serializer::ArrayGlobalOffset_Parts);
        else
            serializer->WriteByteSegments("values", read->View<uint8_t>().data, DNDS::index(storage.Bytes()), Detail::IOSegments(layout, bytes),
                                          perRank ? Serializer::ArrayGlobalOffset_Unknown : Serializer::ArrayGlobalOffset_Parts);
    }
    template <class T>
    void WriteArray(const Array<T> &array, Serializer::SerializerBaseSSP serializer,
                    const std::string &name, const MPIInfo &mpi, const std::string &requestedSchema = {})
    {
        DNDS_check_throw_info(serializer, "null Array serializer");
        const auto schema = Detail::IOSchema<T>(requestedSchema, mpi, serializer->IsPerRank());
        auto storage = Detail::IOPublishedStorage(array, mpi, serializer->IsPerRank());
        WriteArrayStorage(storage->Values(), storage->Layout(), schema, serializer, name, mpi);
    }
    struct ArrayArchive
    {
        RowLayout layout;
        Buffer storage;
    };

    /// Read directly into the final allocation; validate dynamic logical shape
    /// and then swap declared words in place before publishing the result.
    inline ArrayArchive ReadArrayStorage(Serializer::SerializerBaseSSP serializer, const std::string &name,
                                         const MPIInfo &mpi, Serializer::ArrayGlobalOffset requested, Device device,
                                         const ElementSchema &schema, const std::function<RowLayout(const RowLayout &)> &selectLayout = {},
                                         const std::string &legacyIdentity = {})
    {
        DNDS_check_throw_info(serializer, "null Array serializer");
        const bool perRank = serializer->IsPerRank();
        const size_t bytes = schema.Extent();
        Detail::IOAgree(name + ":" + schema.Identity() + ":" + schema.Definition() + ":" + legacyIdentity, mpi, perRank);
        Detail::IOCheck(device == Device::Host(), mpi, perRank, "native Array read requires a direct device-capable backend for device storage; staging is disabled");
        Detail::SerializerPath guard(serializer);
        serializer->GoToPath(name);
        int format = 0, ranks = 0;
        DNDS::index storedBytes = 0;
        std::string storedSchema;
        serializer->ReadInt("runtime_array_format", format);
        serializer->ReadIndex("element_bytes", storedBytes);
        serializer->ReadInt("writer_ranks", ranks);
        serializer->ReadString("element_schema", storedSchema);
        const auto &expectedIdentity = format == 2 && !legacyIdentity.empty() ? legacyIdentity : schema.Identity();
        bool valid = (format == 2 || format == 3 || format == 4) && bytes && storedBytes == DNDS::index(bytes) && storedSchema == expectedIdentity && ranks > 0;
        valid = valid && (format < 3 || legacyIdentity.empty() || legacyIdentity == schema.Identity());
        valid = valid && (!(requested == Serializer::ArrayGlobalOffset_Unknown) || ranks == mpi.size);
        valid = valid && (!perRank || requested == Serializer::ArrayGlobalOffset_Unknown);
        Detail::IOCheck(valid, mpi, perRank, "Array checkpoint schema or read partition mismatch");
        RowLayout logical;
        Serializer::ArrayGlobalOffset data{0, 0};
        int endian = Detail::NativeByteOrder();
        std::string dataPath = "values";
        Serializer::ArrayGlobalOffset rowRegion{0, 0};
        std::optional<RowLayout> savedPhysical;
        if (format >= 3)
        {
            std::string definition;
            serializer->ReadString("element_definition", definition);
            endian = Detail::FileByteOrder(serializer, false);
            Detail::IOCheck(definition == schema.Definition() && (endian == Detail::NativeByteOrder() || schema.CanSwapWords()), mpi, perRank,
                            "Array field definition differs or opaque data cannot change endian");
            int encoding = -1;
            DNDS::index totalRows = 0, totalElements = 0, width = 0;
            serializer->ReadInt("row_encoding", encoding);
            serializer->ReadIndex("rows_global", totalRows);
            serializer->ReadIndex("elements_global", totalElements);
            serializer->ReadIndex("row_width", width);
            Detail::IOCheck((encoding == 0 || (encoding == 1 && format == 3)) && totalRows >= 0 && totalRows < INT64_MAX && totalElements >= 0 && totalElements <= INT64_MAX / DNDS::index(bytes) &&
                                (encoding != 0 || (width >= 0 && width <= INT64_MAX / DNDS::index(bytes))),
                            mpi, perRank, "invalid native Array shape metadata");
            if (encoding == 0)
                Detail::IOCheck(width == 0 ? totalElements == 0 : totalRows <= INT64_MAX / width && totalRows * width == totalElements,
                                mpi, perRank, "native dense Array element count mismatch");
            DNDS::index payloadBytes = 0;
            auto payloadQuery = perRank ? Serializer::ArrayGlobalOffset_Unknown : Serializer::ArrayGlobalOffset_One;
            if (format == 4)
            {
                for (index iComp = 0; iComp < width; ++iComp)
                {
                    serializer->ReadUint8Array("values_" + std::to_string(iComp), nullptr, payloadBytes, payloadQuery);
                    Detail::IOCheck(payloadBytes == totalRows * index(bytes), mpi, perRank, "native Array component payload extent differs");
                }
            }
            else
            {
                serializer->ReadUint8Array("values", nullptr, payloadBytes, payloadQuery);
                Detail::IOCheck(payloadBytes == totalElements * DNDS::index(bytes), mpi, perRank, "native Array payload size differs from logical shape");
            }
            std::vector<DNDS::index> counts;
            Detail::IOCheck(serializer->IndexVectorSize("row_counts") == (perRank ? 1 : ranks), mpi, perRank,
                            "native Array writer-rank metadata extent differs from its header");
            auto countRegion = perRank ? Serializer::ArrayGlobalOffset_Unknown : Serializer::ArrayGlobalOffset_One;
            serializer->ReadIndexVector("row_counts", counts, countRegion);
            valid = counts.size() == size_t(perRank ? 1 : ranks);
            DNDS::index countedRows = 0;
            for (const auto count : counts)
            {
                valid = valid && count >= 0 && count <= INT64_MAX - countedRows;
                if (valid)
                    countedRows += count;
            }
            Detail::IOCheck(valid && countedRows == totalRows, mpi, perRank, "native Array row partition metadata mismatch");
            auto region = Detail::IOCheckedRows(requested, counts, mpi, perRank);
            rowRegion = region;
            if (encoding == 0)
            {
                logical = RowLayout::Dense(size_t(region.size()), size_t(width));
                data = {region.size() * width * DNDS::index(bytes), region.offset() * width * DNDS::index(bytes)};
            }
            else
            {
                std::vector<DNDS::index> offsets;
                Detail::IOCheck(serializer->IndexVectorSize("row_offsets") == totalRows + 1, mpi, perRank,
                                "native CSR Array row-offset extent differs from its shape");
                auto window = perRank ? Serializer::ArrayGlobalOffset_Unknown : Serializer::ArrayGlobalOffset{region.size() + 1, region.offset()};
                serializer->ReadIndexVector("row_offsets", offsets, window);
                valid = offsets.size() == size_t(region.size()) + 1;
                std::vector<size_t> lengths(size_t(region.size()));
                for (size_t row = 0; valid && row < lengths.size(); ++row)
                {
                    valid = offsets[row] >= 0 && offsets[row + 1] >= offsets[row] && offsets[row + 1] <= totalElements;
                    if (valid)
                        lengths[row] = size_t(offsets[row + 1] - offsets[row]);
                }
                valid = valid && !offsets.empty() && offsets.front() >= 0 && offsets.back() <= totalElements;
                valid = valid && (region.offset() != 0 || offsets.front() == 0) &&
                        (region.offset() + region.size() != totalRows || offsets.back() == totalElements);
                Detail::IOCheck(valid, mpi, perRank, "native CSR Array row offsets are invalid");
                logical = RowLayout::Ragged(lengths);
                data = {(offsets.back() - offsets.front()) * DNDS::index(bytes), offsets.front() * DNDS::index(bytes)};
            }
            if (format == 4)
            {
                Detail::IOCheck(serializer->IndexVectorSize("physical_layout") == (perRank ? 4 : index(ranks) * 4),
                                mpi, perRank, "native physical-layout metadata extent differs");
                // Matching-rank reads restore physical placement. Repartitioned
                // reads default to dense logical rows, or use selectLayout.
                const bool restoreHere = requested == Serializer::ArrayGlobalOffset_Unknown;
                const bool restoreAny = perRank ? restoreHere : MPI::AllreduceValue<MPI::Reduction::Max>(int(restoreHere), mpi.comm) != 0;
                if (restoreAny)
                {
                    std::vector<index> descriptor;
                    auto window = perRank ? Serializer::ArrayGlobalOffset_Unknown : Serializer::ArrayGlobalOffset{restoreHere ? 4 : 0, index(mpi.rank) * 4};
                    serializer->ReadIndexVector("physical_layout", descriptor, window);
                    valid = descriptor.size() == (restoreHere ? 4u : 0u);
                    try
                    {
                        if (restoreHere && valid)
                        {
                            if (descriptor[0] == index(RowLayoutKind::Table))
                                savedPhysical = RowLayout::Dense(size_t(region.size()), size_t(width), size_t(descriptor[1]));
                            else if (descriptor[0] == index(RowLayoutKind::ColMajorTable))
                                savedPhysical = RowLayout::ColMajorWithPitch(size_t(region.size()), size_t(width), size_t(descriptor[1]), {descriptor[2], descriptor[3]});
                            else if (descriptor[0] == index(RowLayoutKind::TiledColMajorTable))
                                savedPhysical = RowLayout::TiledColMajor(size_t(region.size()), size_t(width), size_t(descriptor[1]));
                            else
                                valid = false;
                        }
                    }
                    catch (...)
                    {
                        MPI::AbortIfSolverFailure("runtime array I/O preparation failed");
                        valid = false;
                    }
                    Detail::IOCheck(valid, mpi, perRank, "invalid native physical-layout descriptor");
                }
            }
        }
        else
        {
            // Read the old byte-CSR wire format without constructing a legacy Array.
            serializer->GoToPath("rows");
            std::vector<DNDS::index> counts;
            if (perRank)
            {
                DNDS::index count = 0;
                serializer->ReadIndex("size", count);
                counts.push_back(count);
            }
            else
            {
                auto partition = Serializer::ArrayGlobalOffset{ranks, 0};
                serializer->ReadIndexVector("size", counts, partition);
            }
            auto region = Detail::IOCheckedRows(requested, counts, mpi, perRank);
            ssp<host_device_vector<DNDS::index>> offsets;
            data = serializer->ReadSharedRowStartVector("pRowStart", offsets, region);
            std::vector<size_t> lengths(size_t(region.size()));
            valid = offsets && offsets->size() == lengths.size() + 1;
            for (size_t row = 0; valid && row < lengths.size(); ++row)
            {
                const auto count = (*offsets)[row + 1] - (*offsets)[row];
                valid = count >= 0 && count % DNDS::index(bytes) == 0;
                if (valid)
                    lengths[row] = size_t(count) / bytes;
            }
            Detail::IOCheck(valid, mpi, perRank, "legacy byte-CSR Array row shape mismatch");
            logical = RowLayout::Ragged(lengths);
            dataPath = "data";
        }
        RowLayout layout;
        std::optional<Buffer> result;
        try
        {
            layout = selectLayout ? selectLayout(logical) : savedPhysical.value_or(logical);
            valid = layout.RowCount() == logical.RowCount() &&
                    (layout.IsVariable() || logical.IsVariable() || layout.Width() == logical.Width());
            for (size_t row = 0; valid && row < layout.RowCount(); ++row)
                valid = layout.RowSize(row) == logical.RowSize(row);
            valid = valid && CheckedSize::Multiply(layout.Elements(), bytes) <= size_t(INT64_MAX);
            if (valid)
                result.emplace(Buffer::Allocate(layout.Elements() * bytes, Device::Host(), AllocationOptions{schema.Alignment()}));
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("runtime array I/O preparation failed");
            valid = false;
        }
        Detail::IOCheck(valid, mpi, perRank, "native Array output policy disagrees with logical shape or allocation failed");
        {
            auto write = result->Write();
            auto pointer = write.View<uint8_t>().data;
            if (format == 4)
                // Collective dataset order comes from the archive. A selected
                // CSR or padded destination can lack a width even with rows.
                for (index iComp = 0; iComp < logical.Width(); ++iComp)
                {
                    auto region = perRank ? Serializer::ArrayGlobalOffset_Unknown : Serializer::ArrayGlobalOffset{index(layout.RowCount()) * index(bytes), rowRegion.offset() * index(bytes)};
                    serializer->ReadByteSegments("values_" + std::to_string(iComp), pointer, index(result->Bytes()),
                                                 Detail::IOColumnSegments(layout, bytes, iComp), region);
                }
            else
                Detail::ReadRowPayload(*serializer, dataPath, pointer, index(result->Bytes()), layout, bytes, data, mpi, perRank);
            if (endian != Detail::NativeByteOrder())
                for (index iRow = 0; iRow < layout.RowCount(); ++iRow)
                {
                    if (layout.IsContiguousRows())
                        schema.SwapWords(pointer + layout.Offset(iRow) * bytes, layout.RowSize(iRow));
                    else
                        for (index iComp = 0; iComp < layout.Width(); ++iComp)
                            schema.SwapWords(pointer + layout.ElementOffset(iRow, iComp) * bytes, 1);
                }
        }
        return {std::move(layout), std::move(*result)};
    }
    template <class T, class L = RowLayout>
    Array<T, L> ReadArray(Serializer::SerializerBaseSSP serializer, const std::string &name, const MPIInfo &mpi,
                          Serializer::ArrayGlobalOffset partition = Serializer::ArrayGlobalOffset_Unknown,
                          Device device = Device::Host(), const std::string &requestedSchema = {})
    {
        DNDS_check_throw_info(serializer, "null Array serializer");
        const auto schema = Detail::IOSchema<T>({}, mpi, serializer->IsPerRank());
        auto select = [](const RowLayout &logical)
        {
            if constexpr (std::is_same_v<L, RowLayout>)
                return logical;
            else
            {
                std::vector<size_t> lengths(logical.RowCount());
                for (size_t row = 0; row < lengths.size(); ++row)
                    lengths[row] = logical.RowSize(row);
                RowLayout result;
                if constexpr (L::layout_kind == RowLayoutKind::Table)
                {
                    const auto width = L::row_width != DynamicExtent ? L::row_width : !logical.IsVariable() ? logical.Width()
                                                                                  : lengths.empty()         ? 0
                                                                                                            : lengths.front();
                    DNDS_check_throw_info(logical.IsVariable() || logical.Width() == width, "native Array saved row width differs");
                    for (auto length : lengths)
                        DNDS_check_throw_info(length == width, "native Array rows are not the requested uniform width");
                    result = RowLayout::Dense(logical.RowCount(), width, L::row_stride == DynamicExtent ? width : L::row_stride);
                }
                else if constexpr (L::layout_kind == RowLayoutKind::ColMajorTable || L::layout_kind == RowLayoutKind::TiledColMajorTable)
                {
                    const auto width = L::row_width != DynamicExtent ? index(L::row_width) : !logical.IsVariable() ? logical.Width()
                                                                                         : lengths.empty()         ? 0
                                                                                                                   : index(lengths.front());
                    DNDS_check_throw_info(logical.IsVariable() || logical.Width() == width, "native Array saved row width differs");
                    for (auto length : lengths)
                        DNDS_check_throw_info(length == size_t(width), "native Array rows are not uniform");
                    if constexpr (L::layout_kind == RowLayoutKind::ColMajorTable)
                        result = RowLayout::ColMajor(size_t(logical.RowCount()), size_t(width));
                    else
                        result = RowLayout::TiledColMajor(size_t(logical.RowCount()), size_t(width), L::component_stride);
                }
                else if constexpr (L::layout_kind == RowLayoutKind::Padded)
                {
                    const auto maximum = lengths.empty() ? 0 : *std::max_element(lengths.begin(), lengths.end());
                    result = RowLayout::Padded(lengths, L::row_capacity == DynamicExtent ? maximum : L::row_capacity);
                }
                else
                    result = RowLayout::Ragged(lengths);
                L::Validate(result);
                return result;
            }
        };
        auto archive = ReadArrayStorage(serializer, name, mpi, partition, device, schema, std::is_same_v<L, RowLayout> ? std::function<RowLayout(const RowLayout &)>{} : select, requestedSchema);
        return Array<T, L>::FromStorage(std::move(archive.layout), std::move(archive.storage));
    }
}
