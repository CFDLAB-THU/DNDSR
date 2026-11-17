#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/KernelViews.hpp"
#include "DNDS/Algebra/EigenView.hpp"
#include "DNDS/Runtime/Array/ArrayCopy.hpp"
#include "DNDS/Core/OMP.hpp"
#include <future>
#include <set>
#ifdef DNDS_UNIX_LIKE
#    include <sys/wait.h>
#    include <sys/resource.h>
#    include <unistd.h>
#endif
#include <map>
#include <stdexcept>

namespace R = DNDS::Runtime;

TEST_CASE("Native row teams use the same explicit policy on controller and CPU worker")
{
    const int requested = DNDS::OMP::DistributedThreads();
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
    const int dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
#endif
    auto probe = [requested]
    {
        std::vector<int> seen(size_t(std::max(32, requested * 4)), -1);
        DNDS::OMP::ForRows(0, DNDS::index(seen.size()), [&](DNDS::index iRow)
                           {
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
            seen[size_t(iRow)] = omp_get_thread_num();
#else
            seen[size_t(iRow)] = 0;
#endif
            // Nested helpers stay within this worker; no nested team.
            DNDS::OMP::ForRows(0, 2, [&](DNDS::index)
            { DNDS_assert(DNDS::OMP::DistributedThreads() == 1); }); });
        return std::set<int>(seen.begin(), seen.end());
    };
    const auto controller = probe();
    auto worker = std::async(std::launch::async, [probe]
                             {
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
        omp_set_dynamic(0);
#endif
        return probe(); });
    CHECK(controller.size() == size_t(requested));
    CHECK(worker.get() == controller);
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
    omp_set_dynamic(dynamic);
#endif
}

TEST_CASE("Native setup failures are emitted after all row workers join")
{
    std::vector<int> visited(32);
    try
    {
        DNDS::OMP::ForSetupRows(0, 32, [&](DNDS::index iRow)
                                {
            visited[size_t(iRow)] = 1;
            if (iRow == 3 || iRow == 19)
                throw std::runtime_error(std::to_string(iRow)); });
        FAIL("setup failure was lost");
    }
    catch (const std::runtime_error &error)
    {
        CHECK(std::string(error.what()) == "3");
    }
    if (DNDS::OMP::DistributedThreads() > 1)
        CHECK(std::all_of(visited.begin(), visited.end(), [](int count)
                          { return count == 1; }));
    else
        CHECK(visited[4] == 0); // Canonical serial exception short-circuits.
}

TEST_CASE("Array mutation epochs cover aliases and canceled reservations without claiming completion")
{
    R::Array<double> array(R::RowLayout::Dense(7, 5));
    auto alias = array;
    const auto initial = array.MutationEpoch();
    const auto version = array.Version();
    CHECK_NOTHROW(array.Storage().ValidateReservation(R::AccessMode::Write, R::Device::Host()));
    CHECK(array.MutationEpoch() == initial);
    CHECK(array.Version() == version);
    auto pending = array.Storage().Reserve(R::AccessMode::Write, R::Device::Host(), false);
    CHECK_NOTHROW(array.Storage().ValidateReservation(R::AccessMode::Write, R::Device::Host()));
    CHECK_THROWS(array.Storage().ValidateReservation(R::AccessMode::Write, R::Device::Host(), true));
    CHECK(alias.MutationEpoch() == initial + 1);
    CHECK(alias.Version() == version);
    CHECK_THROWS(alias.Read());
    pending.Cancel();
    CHECK(array.MutationEpoch() == initial + 1);
    CHECK(array.Version() == version);
    {
        auto write = alias.Write();
        CHECK(array.MutationEpoch() == initial + 2);
        write.CancelUnmodified();
    }
    CHECK(array.Version() == version);
    {
        auto write = array.Write();
        write.View()(0, 0) = 3.;
    }
    CHECK(alias.MutationEpoch() == initial + 3);
    CHECK(alias.Version() == version + 1);
    {
        auto read = alias.Read();
        CHECK(read.View()(0, 0) == 3.);
        CHECK_THROWS(array.Storage().ValidateReservation(R::AccessMode::Write, R::Device::Host()));
        CHECK_NOTHROW(array.Storage().ValidateReservation(R::AccessMode::Read, R::Device::Host()));
    }
    CHECK(array.MutationEpoch() == initial + 3);
}

TEST_CASE("Allocation diagnostics count shared capacity once and do not retain arrays")
{
    struct DiagnosticsScope
    {
        DiagnosticsScope()
        {
            R::AllocationDiagnostics::Enable();
            R::AllocationDiagnostics::CaptureStacks();
        }
        ~DiagnosticsScope() { R::AllocationDiagnostics::Enable(false); }
    } scope;
    const auto initial = R::AllocationDiagnostics::Snapshot().liveBytes;
    std::optional<R::Array<double>::ReadLease> pin;
    uint64_t identity = 0;
    {
        R::Array<double> values(R::RowLayout::Dense(4, 5));
        values.SetName("D0/U/owned", DNDS_ALLOCATION_SITE());
        auto alias = values;
        CHECK(alias.Name() == "D0/U/owned");
        identity = values.Identity();
        pin.emplace(values.Read());
        const auto snapshot = R::AllocationDiagnostics::Snapshot();
        const auto record = std::find_if(snapshot.allocations.begin(), snapshot.allocations.end(),
                                         [identity](const auto &entry)
                                         { return entry.identity == identity; });
        REQUIRE(record != snapshot.allocations.end());
        CHECK(record->bytes == 4 * 5 * sizeof(double));
        CHECK(record->names == std::vector<std::string>{"D0/U/owned"});
        CHECK(record->activity.readers == 1);
        CHECK_FALSE(record->externalProvider);
        CHECK_FALSE(record->function.empty());
        CHECK_FALSE(record->creationStack.empty());
        CHECK(snapshot.liveBytes == initial + record->bytes);
        {
            auto copied = values.Clone();
            CHECK(copied.Name() == values.Name());
            CHECK(copied.Identity() != values.Identity());
            CHECK(R::AllocationDiagnostics::Snapshot().liveBytes == initial + 2 * record->bytes);
        }
    }
    // Only the explicit access retains the payload after both handles die.
    CHECK(R::AllocationDiagnostics::Snapshot().liveBytes == initial + 4 * 5 * sizeof(double));
    pin.reset();
    const auto finished = R::AllocationDiagnostics::Snapshot();
    CHECK(finished.liveBytes == initial);
    CHECK(std::none_of(finished.allocations.begin(), finished.allocations.end(),
                       [identity](const auto &entry)
                       { return entry.identity == identity; }));
}

TEST_CASE("Whole-array residency survives projections without changing row access")
{
    static_assert(std::is_trivially_copyable_v<R::ArrayView<double>>);
    static_assert(std::is_standard_layout_v<R::ArrayView<double>>);
    R::Array<double> values(R::RowLayout::Dense(3, 2));
    auto access = values.Write();
    const auto view = access.View<R::FixedRows<2>, true>();
    CHECK(view.GetDevice() == R::Device::Host());
    CHECK(view.ConstView().GetDevice() == R::Device::Host());
    CHECK(view.DynamicView().GetDevice() == R::Device::Host());
    const R::ArrayView<double, false, R::FixedRows<2>> unchecked = view;
    CHECK(unchecked.GetDevice() == R::Device::Host());
    const auto csr = access.View<R::CSRRows>();
    const auto padded = access.View<R::PaddedRows>();
    CHECK(csr.GetDevice() == R::Device::Host());
    CHECK(padded.GetDevice() == R::Device::Host());
    CHECK_NOTHROW(R::ValidateViewDevices(R::Device::Host(), view, unchecked, csr, padded));
    view[1][0] = 11.;
    CHECK(csr(1, 0) == 11.);
    CHECK(padded[1][0] == 11.);

    CHECK_THROWS(view.ValidateDevice(R::Device::CUDA(0)));
    auto unbound = view;
    unbound.device = R::Device::Unspecified();
    CHECK_THROWS(unbound.ValidateDevice(R::Device::Host()));
    // Descriptor-only check: no device allocation or dereference is performed.
    auto foreign = view;
    foreign.device = R::Device::CUDA(1);
    CHECK_THROWS(foreign.ValidateDevice(R::Device::CUDA(0)));
    const R::ArrayView<const double> absent;
    CHECK_NOTHROW(absent.ValidateDevice(R::Device::CUDA(0)));
    const R::PairReadView<double, true, R::FixedRows<2>> pair{view.ConstView(), {}};
    CHECK_NOTHROW(pair.ValidateDevice(R::Device::Host()));
}

TEST_CASE("Scheduled operands publish placement in borrowed whole-array views")
{
    R::Array<double> values(R::RowLayout::Dense(3, 2));
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    R::SubmitArrays(engine, stream, [](R::ExecutionContext context, R::ArrayView<double> output)
                    {
                        output.ValidateDevice(context.device);
                        for (DNDS::index iRow = 0; iRow < output.RowCount(); ++iRow)
                            for (DNDS::index j = 0; j < output.Width(); ++j)
                                output(iRow, j) = 10 * iRow + j; }, R::Write(values));
    engine.Synchronize();
    CHECK(values.Read().View()(2, 1) == 21.);
}

TEST_CASE("Kernel spans preserve signed slices and metadata publications own their storage")
{
    static_assert(std::is_trivially_copyable_v<DNDS::SpanView<const int>>);
    int source[]{4, 7, 11, 19};
    DNDS::SpanView<int> values{source, 4};
    const DNDS::SpanView<const int> readOnly = values;
    auto tail = values.Subspan(2, 2);
    tail[1] = 23;
    CHECK(readOnly.size() == 4);
    CHECK(readOnly[3] == 23);
    CHECK(values.Subspan(4, 0).empty());
    const R::MetadataBuffer<int> published(std::vector<int>{4, 7, 11, 23});
    auto independent = published.CopyTo(R::Device::Host(), R::ExecutionContext::Host());
    CHECK(independent.Identity() != published.Identity());
    CHECK(independent == published);
    const auto span = independent.HostView();
    const auto owner = independent;
    independent = R::MetadataBuffer<int>{};
    CHECK(span[2] == 11);
    CHECK(owner.HostView()[3] == 23);
}

TEST_CASE("Unused Array access cancellation preserves values validity and generation")
{
    R::Array<int> values(R::RowLayout::Dense(3, 2));
    {
        auto write = values.Write();
        write.View()(1, 1) = 17;
    }
    const auto version = values.Version();
    try
    {
        auto unused = values.Write();
        try
        {
            throw std::runtime_error("collective preflight rejected on a peer");
        }
        catch (...)
        {
            unused.CancelUnmodified();
            CHECK_FALSE(unused.Active());
            throw;
        }
    }
    catch (const std::runtime_error &)
    {
    }
    CHECK(values.IsValid());
    CHECK(values.Version() == version);
    CHECK(values.Read().View()(1, 1) == 17);
    auto unusedRead = values.Read();
    unusedRead.CancelUnmodified();
    CHECK_FALSE(unusedRead.Active());
    CHECK_NOTHROW(values.Write());
}

TEST_CASE("Runtime row layouts reject unsigned shapes that cannot be indexed by signed views")
{
    const auto tooLarge = size_t(INT64_MAX) + 1;
    CHECK_THROWS(R::RowLayout::Dense(tooLarge, 0));
    CHECK_THROWS(R::RowLayout::Dense(0, tooLarge));
    CHECK_THROWS(R::RowLayout::Dense(2, 0, size_t(INT64_MAX)));
    CHECK_THROWS(R::RowLayout::Padded({}, tooLarge));
    CHECK_THROWS(R::RowLayout::Padded({0, 0}, size_t(INT64_MAX)));
    CHECK_THROWS(R::RowLayout::Ragged({size_t(INT64_MAX), 1}));
    const auto emptyRows = R::RowLayout::Dense(size_t(INT64_MAX), 0);
    CHECK(emptyRows.RowCount() == INT64_MAX);
    CHECK(emptyRows.Offset(INT64_MAX - 1) == 0);
    CHECK_THROWS(emptyRows.RowSize(-1));
    CHECK_THROWS(emptyRows.Offset(-1));
}

TEST_CASE("Generated immutable metadata preserves ordinal values and abandons failed construction")
{
    std::vector<DNDS::index> visited;
    auto metadata = R::MetadataBuffer<uint64_t>::Generate(4, [&](DNDS::index i)
                                                          {
        visited.push_back(i);
        return uint64_t(100 + 7 * i); });
    CHECK(visited == std::vector<DNDS::index>{0, 1, 2, 3});
    CHECK(std::vector<uint64_t>(metadata.begin(), metadata.end()) == std::vector<uint64_t>{100, 107, 114, 121});
    const auto alias = metadata;
    CHECK(alias.Identity() == metadata.Identity());
    CHECK(alias.data() == metadata.data());
    auto retained = metadata.Read();
    CHECK(retained.View<uint64_t>().data[3] == 121);
    const auto empty = R::MetadataBuffer<uint64_t>::Generate(0, [](DNDS::index) -> uint64_t
                                                             { throw std::runtime_error("empty generator must not run"); });
    CHECK(empty.empty());
    CHECK_THROWS(R::MetadataBuffer<uint64_t>::Generate(size_t(INT64_MAX) + 1, [](DNDS::index)
                                                       { return uint64_t(0); }));
    visited.clear();
    CHECK_THROWS(R::MetadataBuffer<uint64_t>::Generate(4, [&](DNDS::index i) -> uint64_t
                                                       {
        visited.push_back(i);
        if (i == 2) throw std::runtime_error("generator failure");
        return uint64_t(i); }));
    CHECK(visited == std::vector<DNDS::index>{0, 1, 2});
    CHECK(alias[3] == 121);
    CHECK(R::MetadataBuffer<uint64_t>::Generate(1, [](DNDS::index)
                                                { return uint64_t(9); })[0] == 9);
    const auto shape = R::RowLayout::Ragged({0, 3, 0, 2, 1});
    CHECK(std::vector<size_t>(shape.Offsets().begin(), shape.Offsets().end()) == std::vector<size_t>{0, 0, 3, 3, 5, 6});
    CHECK(shape.RowSize(2) == 0);
    CHECK(shape.Elements() == 6);
}

template <class Policy>
static void CheckEagerTypedCPU(const R::RowLayout &shape)
{
    R::EagerEngine engine(2);
    auto stream = engine.CreateCPUStream();
    R::Array<int> input(shape), output(shape);
    auto retained = input;
    R::SubmitArrays(engine, stream, [](R::ExecutionContext, auto values)
                    {
        static_assert(std::is_same_v<decltype(values), R::ArrayView<int, R::AccessBoundsChecks, Policy>>);
        for (DNDS::index row = 0; row < values.RowCount(); ++row)
            for (DNDS::index column = 0; column < values[row].size(); ++column)
                values(row, column) = int(10 * row + column); }, R::WriteAs<Policy>(input));
    R::SubmitArrays(engine, stream, [](R::ExecutionContext, auto source, auto target)
                    {
        static_assert(std::is_same_v<decltype(source), R::ArrayView<const int, R::AccessBoundsChecks, Policy>>);
        static_assert(std::is_same_v<decltype(target), R::ArrayView<int, R::AccessBoundsChecks, Policy>>);
        for (DNDS::index row = 0; row < source.RowCount(); ++row)
            for (DNDS::index column = 0; column < source[row].size(); ++column)
                target(row, column) = 3 * source(row, column); }, R::ReadAs<Policy>(input), R::WriteAs<Policy>(output));
    // Existing concrete dynamic-view launchers remain valid conversion targets.
    R::SubmitArrays(engine, stream, [](R::ExecutionContext, R::ArrayView<int> values)
                    {
        for (DNDS::index row = 0; row < values.RowCount(); ++row)
            for (auto &value : values[row]) ++value; }, R::ReadWrite(output));
    input.Reset(shape);
    stream.Record().WaitHost();
    auto source = retained.Read(), actual = output.Read();
    for (DNDS::index row = 0; row < shape.RowCount(); ++row)
        for (DNDS::index column = 0; column < shape.RowSize(row); ++column)
        {
            CHECK(source.View()(row, column) == int(10 * row + column));
            CHECK(actual.View()(row, column) == int(3 * (10 * row + column) + 1));
        }
}

TEST_CASE("Worker eager submissions retain all five typed policies and generic launcher compatibility")
{
    CheckEagerTypedCPU<R::TableRows>(R::RowLayout::Dense(3, 3, 5));
    CheckEagerTypedCPU<R::FixedRows<3, 5>>(R::RowLayout::Dense(3, 3, 5));
    CheckEagerTypedCPU<R::PaddedRows>(R::RowLayout::Padded({0, 2, 5}, 5));
    CheckEagerTypedCPU<R::StaticMaxRows<5>>(R::RowLayout::Padded({0, 2, 5}, 5));
    CheckEagerTypedCPU<R::CSRRows>(R::RowLayout::Ragged({0, 2, 5}));
}

template <class Policy>
static void CheckLayoutSpecificAccess(const R::RowLayout &shape)
{
    R::Array<int> values(shape);
    {
        auto write = values.Write();
        const auto view = write.template View<Policy, true>();
        static_assert(!decltype(view)::is_dynamic);
        static_assert(decltype(view)::layout_kind == Policy::layout_kind);
        for (DNDS::index row = 0; row < shape.RowCount(); ++row)
        {
            CHECK(view.Row(row) == view.data + shape.Offset(row));
            CHECK(view[row].size() == shape.RowSize(row));
            for (DNDS::index column = 0; column < shape.RowSize(row); ++column)
                view[row][column] = int(20 * row + column);
        }
    }
    auto read = values.Read();
    auto typed = read.template View<Policy>();
    for (DNDS::index row = 0; row < shape.RowCount(); ++row)
    {
        CHECK(typed.Row(row) == read.View().Row(row));
        CHECK(typed.RowSize(row) == shape.RowSize(row));
        for (DNDS::index column = 0; column < shape.RowSize(row); ++column)
            CHECK(read.View()(row, column) == int(20 * row + column));
    }
}

TEST_CASE("Five layout policies produce distinct typed views and share dynamic storage")
{
    static_assert(std::is_same_v<R::Array<int>, R::Array<int, R::FixedRows<3>>>);
    static_assert(std::is_same_v<R::ArrayPair<int>, R::ArrayPair<int, R::FixedRows<3>>>);
    static_assert(std::is_same_v<R::GenericRowView<int>, R::ArrayView<int>>);
    static_assert(std::is_same_v<decltype(std::declval<R::RowLayout>().RowCount()), DNDS::index>);
    static_assert(std::is_same_v<decltype(std::declval<R::RowSpan<int>>().size()), DNDS::index>);
    static_assert(std::is_same_v<decltype(std::declval<R::ArrayView<int>>().RowCount()), DNDS::index>);
    using Table = R::ArrayView<int, false, R::TableRows>;
    using Fixed = R::ArrayView<int, false, R::FixedRows<3>>;
    using Padded = R::ArrayView<int, false, R::PaddedRows>;
    using Max = R::ArrayView<int, false, R::StaticMaxRows<5>>;
    using CSR = R::ArrayView<int, false, R::CSRRows>;
    static_assert(!std::is_same_v<Table, Fixed> && !std::is_same_v<Table, Padded> && !std::is_same_v<Table, CSR>);
    static_assert(!std::is_same_v<Padded, Max> && !std::is_same_v<Padded, CSR> && !std::is_same_v<Max, CSR>);
    static_assert(!Table::is_dynamic && !Fixed::is_dynamic && !Padded::is_dynamic && !Max::is_dynamic && !CSR::is_dynamic);
    static_assert(std::is_trivially_copyable_v<Table> && std::is_trivially_copyable_v<Padded> && std::is_trivially_copyable_v<CSR>);
    CheckLayoutSpecificAccess<R::TableRows>(R::RowLayout::Dense(3, 3, 5));
    CheckLayoutSpecificAccess<R::FixedRows<3>>(R::RowLayout::Dense(3, 3, 5));
    CheckLayoutSpecificAccess<R::PaddedRows>(R::RowLayout::Padded({0, 2, 5}, 5));
    CheckLayoutSpecificAccess<R::StaticMaxRows<5>>(R::RowLayout::Padded({0, 2, 5}, 5));
    CheckLayoutSpecificAccess<R::CSRRows>(R::RowLayout::Ragged({0, 2, 5}));
    CheckLayoutSpecificAccess<R::TableRows>(R::RowLayout::Dense(0, 3, 5));
    CheckLayoutSpecificAccess<R::PaddedRows>(R::RowLayout::Padded({}, 5));
    CheckLayoutSpecificAccess<R::CSRRows>(R::RowLayout::Ragged({}));
    R::Array<int> dense(R::RowLayout::Dense(2, 2, 3));
    CHECK(dense.Read().template View<R::PaddedRows>().RowSize(1) == 2);
    CHECK(dense.Read().template View<R::CSRRows>().RowSize(1) == 2);
    {
        auto write = dense.Write();
        const auto table = write.View<R::TableRows, true>();
        const auto csr = write.View<R::CSRRows, true>();
        std::fill_n(table.data, 6, -91);
        for (DNDS::index iRow = 0; iRow < 2; ++iRow)
        {
            CHECK(csr[iRow].size() == 2);
            CHECK(csr[iRow].data() == table.data + iRow * 3);
            csr[iRow][0] = int(10 + iRow);
            csr(iRow, 1) = int(20 + iRow);
        }
        CHECK(table(0, 0) == 10);
        CHECK(table(0, 1) == 20);
        CHECK(table(1, 0) == 11);
        CHECK(table(1, 1) == 21);
        CHECK(table.data[2] == -91);
        CHECK(table.data[5] == -91);
    }
    R::Array<int> irregularPadded(R::RowLayout::Padded({1, 2}, 2));
    R::Array<int> irregularCSR(R::RowLayout::Ragged({1, 2}));
    CHECK_THROWS(irregularPadded.Read().template View<R::TableRows>());
    CHECK_THROWS(irregularCSR.Read().template View<R::TableRows>());
}

TEST_CASE("Uniform variable layouts project to typed tables in direct and eager access")
{
    for (const auto &shape : {R::RowLayout::Ragged({3, 3, 3}), R::RowLayout::Padded({3, 3, 3}, 3)})
    {
        CheckLayoutSpecificAccess<R::FixedRows<3, 3>>(shape);
        CheckEagerTypedCPU<R::TableRows>(shape);
        CheckEagerTypedCPU<R::FixedRows<3, 3>>(shape);
    }
    R::Array<int> padded(R::RowLayout::Padded({2, 2}, 3));
    CHECK_THROWS(padded.Read().template View<R::TableRows>());
    R::Array<int> empty(R::RowLayout::Ragged({0, 0}));
    CHECK(empty.Read().template View<R::FixedRows<0, 0>>()[1].empty());
}

TEST_CASE("Static row policy reaches typed access and preserves dynamic interoperability")
{
    using Fixed = R::Array<double>;
    using WriteView = decltype(std::declval<Fixed::WriteLease>().template View<R::FixedRows<3>>());
    using ReadView = decltype(std::declval<Fixed::ReadLease>().template View<R::FixedRows<3>>());
    using ReadRow = decltype(std::declval<ReadView>()[0]);
    static_assert(WriteView::row_width == 3 && ReadView::row_width == 3);
    static_assert(ReadRow::row_width == 3);
    static_assert(ReadView::row_stride == R::DynamicExtent);
    static_assert(decltype(std::declval<Fixed::WriteLease>().template View<R::FixedRows<3, 5>>())::row_stride == 5);
    static_assert(decltype(std::declval<Fixed::WriteLease>().template View<R::FixedRows<3, 5>>()[0])::row_capacity == 5);
    static_assert(std::is_same_v<decltype(std::declval<ReadView>()(0, 0)), const double &>);
    static_assert(std::is_same_v<decltype(std::declval<WriteView>()[0][0]), double &>);
    static_assert(std::is_trivially_copyable_v<WriteView> && std::is_trivially_copyable_v<ReadRow>);
    static_assert(!std::is_convertible_v<R::ArrayView<double>, WriteView>);

    Fixed values(R::RowLayout::Dense(2, 3, 5));
    {
        auto write = values.Write();
        const auto view = write.View<R::FixedRows<3>, true>();
        for (DNDS::index row = 0; row < view.RowCount(); ++row)
            for (DNDS::index column = 0; column < decltype(view)::row_width; ++column)
                view[row][column] = 100 * row + column;
        R::ArrayView<double, true> dynamic = view;
        CHECK(dynamic.RowSize(1) == 3);
        CHECK(dynamic.Row(1) - dynamic.Row(0) == 5);
        R::RowSpan<double, true> row = view[1];
        CHECK(row.size() == 3);
        CHECK(row[2] == 102);
        CHECK(write.DynamicView().Row(1) == view.Row(1));
    }
    auto staticAlias = R::Array<double>::FromSharedStorage(values.SharedStorage());
    CHECK(staticAlias.Identity() == values.Identity());
    {
        R::Array<double>::ReadLease generic = staticAlias.Read();
        CHECK(generic.View()(1, 2) == 102);
        CHECK(generic.Layout().Stride() == 5);
    }
    auto read = staticAlias.Read();
    const auto view = read.View<R::FixedRows<3, 5>>();
    staticAlias.Reset(R::RowLayout::Dense(0, 3, 5));
    values.Reset(R::RowLayout::Dense(0, 3));
    CHECK(view[1].size() == 3);
    CHECK(view[1][2] == 102);
    CHECK(view.Row(1) - view.Row(0) == 5);
    CHECK_THROWS(read.At(0, 3));
    CHECK_THROWS(read.At(-1, 0));
    CHECK_THROWS(read.At(0, -1));
    const auto raw = read.DynamicView();
    CHECK(raw.width == 3);
    CHECK(raw.stride == 5);
}

TEST_CASE("Static width and capacity policies validate empty layouts and physical stride")
{
    using Fixed = R::Array<int>;
    using Max = R::Array<int>;
    Fixed fixed(R::RowLayout::Dense(0, 2, 4));
    Max padded(R::RowLayout::Padded({}, 4));
    CHECK(fixed.RowCount() == 0);
    CHECK(fixed.Layout().Width() == 2);
    CHECK(fixed.Layout().Stride() == 4);
    CHECK(padded.RowCount() == 0);
    CHECK(padded.Layout().IsPadded());
    CHECK(padded.Layout().Stride() == 4);
    CHECK(fixed.Read().View<R::FixedRows<2, 4>>().RowCount() == 0);
    CHECK(padded.Read().View<R::StaticMaxRows<4>>().RowCount() == 0);
    CHECK_THROWS(Fixed(R::RowLayout::Dense(0, 3, 4)).Read().View<R::FixedRows<2, 4>>());
    CHECK_THROWS(Fixed(R::RowLayout::Dense(0, 2, 3)).Read().View<R::FixedRows<2, 4>>());
    CHECK_THROWS(Fixed(R::RowLayout::Ragged({})).Read().View<R::FixedRows<2, 4>>());
    CHECK_THROWS(Max(R::RowLayout::Padded({}, 5)).Read().View<R::StaticMaxRows<4>>());
    CHECK_THROWS(Max(R::RowLayout::Dense(0, 4, 5)).Read().View<R::StaticMaxRows<4>>());
    R::Array<int> wrong(R::RowLayout::Padded({1, 3}, 5));
    CHECK_THROWS(wrong.Read().View<R::StaticMaxRows<4>>());
    R::Array<int> zeroWidth(R::RowLayout::Dense(2, 0, 4));
    auto zero = zeroWidth.Read();
    CHECK(zero.View<R::FixedRows<0, 4>>()[1].empty());
    CHECK(zero.View<R::FixedRows<0, 4>>()[1].begin() == zero.View<R::FixedRows<0, 4>>()[1].end());
    CHECK(zero.View<R::FixedRows<0, 4>>().Row(1) - zero.View<R::FixedRows<0, 4>>().Row(0) == 4);
}

TEST_CASE("Static maximum capacity retains variable logical widths and untouched padding")
{
    using Max = R::Array<int>;
    using View = decltype(std::declval<Max::WriteLease>().template View<R::StaticMaxRows<5>>());
    using Row = decltype(std::declval<View>()[0]);
    static_assert(View::row_width == R::DynamicExtent && View::row_capacity == 5 && View::row_stride == 5);
    static_assert(Row::row_width == R::DynamicExtent && Row::row_capacity == 5);
    static_assert(std::is_trivially_copyable_v<View> && std::is_trivially_copyable_v<Row>);
    Max values(R::RowLayout::Padded({0, 2, 5}, 5));
    {
        auto raw = values.Storage().Write();
        const auto flat = raw.View<int>();
        for (size_t i = 0; i < flat.size; ++i)
            flat[i] = -23;
    }
    {
        auto write = values.Write();
        auto view = write.View<R::StaticMaxRows<5>, true>();
        CHECK(view[0].empty());
        for (DNDS::index row = 0; row < view.RowCount(); ++row)
            for (DNDS::index column = 0; column < view[row].size(); ++column)
                view(row, column) = int(row * 10 + column);
        CHECK_THROWS(write.At(1, 2));
    }
    {
        auto read = values.Read();
        CHECK(read.View().RowSize(0) == 0);
        CHECK(read.View().RowSize(1) == 2);
        CHECK(read.View()[2][4] == 24);
        auto raw = values.Storage().Read();
        const auto flat = raw.View<int>();
        for (DNDS::index row = 0; row < values.RowCount(); ++row)
            for (DNDS::index column = values.RowSize(row); column < 5; ++column)
                CHECK(flat[row * 5 + column] == -23);
    }
    {
        auto edit = values.EditRows();
        edit.Draft().ResizeRow(1, 4);
        CHECK_THROWS(edit.Draft().ResizeRow(1, 6));
        edit.Commit();
    }
    CHECK(values.Read().View()[1].size() == 4);
    CHECK(values.Read().View()(1, 1) == 11);
    R::ArrayPair<int> pair{std::move(values), Max(R::RowLayout::Padded({1}, 5))};
    pair.ghost.Write().View()(0, 0) = 73;
    auto combined = pair.ReadCombined();
    static_assert(decltype(combined.View<R::StaticMaxRows<5>>())::row_capacity == 5);
    CHECK(combined.View<R::StaticMaxRows<5>>()[3].size() == 1);
    CHECK(combined.View<R::StaticMaxRows<5>>()(3, 0) == 73);
    R::PairReadView<int> generic = combined.View();
    CHECK(generic.RowSize(1) == 4);
    CHECK(generic(3, 0) == 73);
}

TEST_CASE("Array allocation accounting retains capacity and deduplicates aliased metadata")
{
    auto backing = R::Buffer::Allocate(128);
    auto first = R::Array<double>::FromStorage(R::RowLayout::Dense(2, 2), backing.Subspan(16, 4 * sizeof(double)));
    auto second = R::Array<double>::FromStorage(R::RowLayout::Dense(1, 1), backing.Subspan(32, sizeof(double)));
    R::Array<double> csr(R::RowLayout::Ragged({2, 0, 3}));
    auto otherValues = R::Array<double>::FromSharedStorage(csr.SharedStorage()->WithValues(R::Buffer::Allocate(5 * sizeof(double)), sizeof(double)));
    R::Array<int> padded(R::RowLayout::Padded({1, 3}, 5));
    std::map<uint64_t, size_t> allocations;
    auto visit = [&](uint64_t identity, size_t capacity, R::Device device)
    {
        CHECK(device == R::Device::Host());
        auto [entry, inserted] = allocations.emplace(identity, capacity);
        CHECK(entry->second == capacity);
    };
    first.SharedStorage()->VisitAllocations(visit);
    second.SharedStorage()->VisitAllocations(visit);
    auto retained = csr.Read();
    csr.SharedStorage()->VisitAllocations(visit);
    otherValues.SharedStorage()->VisitAllocations(visit);
    padded.SharedStorage()->VisitAllocations(visit);
    size_t total = 0;
    for (const auto &[identity, bytes] : allocations)
        total += bytes;
    CHECK(allocations.size() == 6);
    CHECK(total == 128 + 2 * 5 * sizeof(double) + 4 * sizeof(size_t) + 2 * 5 * sizeof(int) + 2 * sizeof(size_t));
}

TEST_CASE("Consumed Array handles reject queries while retained access survives move and release")
{
    const auto host = R::ExecutionContext::Host();
    for (const auto &shape : {R::RowLayout::Dense(3, 2, 4), R::RowLayout::Padded({0, 2, 1}, 4),
                              R::RowLayout::Ragged({0, 2, 1})})
        for (bool release : {false, true})
        {
            R::Array<int> array(shape), other(shape);
            {
                auto write = array.Write();
                write.View()(1, 0) = 173;
            }
            auto access = array.Read();
            auto retained = std::move(access);
            CHECK_FALSE(access.Active());
            CHECK_THROWS(access.Layout());
            CHECK_THROWS(access.View());
            std::optional<R::Array<int>> moved;
            if (release)
                array.ReleaseOn(host);
            else
                moved.emplace(std::move(array));
            CHECK_FALSE(array.IsValid());
            CHECK_FALSE(array.SharedStorage());
            CHECK_THROWS(array.ShapeVersion());
            CHECK_THROWS(array.Layout());
            CHECK_THROWS(array.RowCount());
            CHECK_THROWS(array.RowSize(0));
            CHECK_THROWS(array.GetDevice());
            CHECK_THROWS(array.Identity());
            CHECK_THROWS(array.Version());
            CHECK_THROWS(array.Storage());
            CHECK_THROWS(array.Read());
            CHECK_THROWS(array.Write());
            CHECK_THROWS(array.EditRows());
            CHECK_THROWS(array.Clone());
            CHECK_THROWS(array.To(R::Device::Host()));
            CHECK_THROWS(array.CopyFrom(other));
            CHECK_THROWS(other.CopyFrom(array));
            CHECK(other.IsValid());
            CHECK_NOTHROW(other.Write());
            auto copiedInvalid = array;
            CHECK_FALSE(copiedInvalid.IsValid());
            CHECK_THROWS(copiedInvalid.Read());
            CHECK(array.CanReleaseOrdered(host));
            CHECK_NOTHROW(array.ReleaseOn(host));
            CHECK_NOTHROW(array.ReleaseOn(host));
            CHECK(retained.At(1, 0) == 173);
            array.Reset(shape);
            CHECK(array.IsValid());
            CHECK(retained.At(1, 0) == 173);
            {
                auto write = array.Write();
                auto active = std::move(write);
                CHECK_FALSE(write.Active());
                CHECK_THROWS(write.Layout());
                CHECK_THROWS(write.View());
                active.View()(1, 0) = 24;
            }
            CHECK(array.Read().At(1, 0) == 24);
        }
}

TEST_CASE("Native failed write scope invalidates its aliases")
{
    R::Array<double> array(R::RowLayout::Dense(1, 1));
    auto alias = array;
    try
    {
        auto write = array.Write();
        write.At(0, 0) = 17;
        throw std::runtime_error("partial operator");
    }
    catch (const std::runtime_error &)
    {
    }
    CHECK_FALSE(alias.IsValid());
    CHECK_THROWS(alias.Read());
    array.Reset(R::RowLayout::Dense(1, 1));
    CHECK(array.IsValid());
    CHECK_FALSE(alias.IsValid());
}

TEST_CASE("native Array retains row layout and allocation across reset")
{
    R::Array<double> a(R::RowLayout::Ragged({2, 0, 3}));
    {
        auto write = a.Write();
        write.At(2, 1) = 42;
        CHECK_THROWS(write.At(1, 0));
        CHECK_THROWS(a.Read());
    }
    CHECK(a.Version() == 1);
    auto alias = a;
    auto read = a.Read();
    a.Reset(R::RowLayout::Dense(8, 5, 8));
    CHECK(a.Identity() != alias.Identity());
    CHECK(read.At(2, 1) == 42);
    CHECK(alias.Read().At(2, 1) == 42);
    auto clone = alias.Clone();
    CHECK(clone.Identity() != alias.Identity());
    CHECK(clone.Read().At(2, 1) == 42);
    CHECK(a.Layout().Elements() == 64);
}

TEST_CASE("native ArrayPair core only needs owned access")
{
    R::ArrayPair<double> pair;
    pair.owned.Reset(R::RowLayout::Dense(4, 2));
    pair.ghost.Reset(R::RowLayout::Dense(2, 2));
    {
        auto incoming = pair.ghost.Write();
        auto core = pair.ReadOwned();
        auto send = pair.owned.Read();
        CHECK(core.View().RowCount() == 4);
        CHECK_THROWS(pair.ReadCombined());
        CHECK_THROWS(pair.Read());
        incoming.At(0, 1) = 7;
    }
    auto ring = pair.ReadCombined();
    CHECK(ring.View().RowCount() == 6);
    CHECK(ring.View()[4].size() == 2);
    CHECK(ring.View().Row(4)[1] == 7);
    R::Array<double> fixed(R::RowLayout::Dense(0, 5));
    CHECK(fixed.Layout().Width() == 5);
    fixed.Reset(R::RowLayout::Dense(2, 3));
    CHECK_THROWS(fixed.Read().template View<R::FixedRows<5>>());
}

TEST_CASE("Fully fixed row spans use one pointer and general rows preserve borrowed access")
{
    constexpr auto dynamic = R::DynamicExtent;
    using Fixed = R::RowSpan<int, true, 3, 3, 3>;
    using DynamicSize = R::RowSpan<int, true, dynamic, dynamic, 3>;
    using DynamicStride = R::RowSpan<int, true, 3>;
    using Dynamic = R::RowSpan<int, true>;
    static_assert(sizeof(Fixed) == sizeof(int *));
    static_assert(sizeof(DynamicSize) == sizeof(int *) + 2 * sizeof(DNDS::index));
    static_assert(sizeof(DynamicStride) == sizeof(int *) + 2 * sizeof(DNDS::index));
    static_assert(sizeof(Dynamic) == sizeof(int *) + 2 * sizeof(DNDS::index));
    static_assert(std::is_standard_layout_v<Fixed> && std::is_trivially_copyable_v<Fixed>);
    static_assert(std::is_standard_layout_v<DynamicSize> && std::is_trivially_copyable_v<DynamicSize>);
    static_assert(std::is_standard_layout_v<DynamicStride> && std::is_trivially_copyable_v<DynamicStride>);
    static_assert(std::is_standard_layout_v<Dynamic> && std::is_trivially_copyable_v<Dynamic>);
    int data[9] = {};
    auto check = [&](auto row)
    {
        CHECK(row.data() == data);
        CHECK(row.size() == 3);
        CHECK(row.ElementStride() == 3);
        for (DNDS::index j = 0; j < row.size(); ++j)
            row[j] = int(10 + j);
        R::RowSpan<const int> generic = row;
        CHECK(generic.ElementStride() == 3);
        CHECK(generic.size() == 3);
        CHECK(generic.end() - generic.begin() == 3);
        CHECK(generic[2] == 12);
        CHECK(*(row.cbegin() + 1) == 11);
    };
    check(Fixed{data, 3, 3});
    check(DynamicSize{data, 3, 3});
    check(DynamicStride{data, 3, 3});
    check(Dynamic{data, 3, 3});
    CHECK(data[1] == 0);
    CHECK(data[4] == 0);
    Fixed fixedDefault;
    CHECK(fixedDefault.data() == nullptr);
    CHECK(fixedDefault.size() == 3); // Static shape is part of the type, even without a payload.
    CHECK(fixedDefault.ElementStride() == 3);
    CHECK(DynamicSize{}.empty());
    CHECK(DynamicStride{}.ElementStride() == 1);
    CHECK(Dynamic{}.empty());
}

TEST_CASE("Native sized row views preserve logical extents and retained metadata")
{
    static_assert(std::is_trivially_copyable_v<R::RowSpan<int>>);
    static_assert(std::is_same_v<decltype(std::declval<R::ArrayView<const int>>()[0][0]), const int &>);
    static_assert(std::is_same_v<decltype(std::declval<R::ArrayView<int>>()[0][0]), int &>);
    R::RowSpan<int> empty;
    CHECK(empty.empty());
    CHECK(empty.begin() == empty.end());
    CHECK(empty.cbegin() == empty.cend());

    for (const auto &shape : {R::RowLayout::Dense(3, 2, 4), R::RowLayout::Padded({2, 0, 3}, 4), R::RowLayout::Ragged({2, 0, 3})})
    {
        R::Array<int> values(shape);
        {
            auto write = values.Write();
            auto view = write.View<true>();
            for (DNDS::index row = 0; row < view.RowCount(); ++row)
            {
                auto span = view[row];
                CHECK(span.size() == shape.RowSize(row));
                CHECK(span.data() == view.Row(row));
                size_t column = 0;
                for (auto &value : span)
                    value = int(10 * row + column++);
            }
        }
        auto read = values.Read();
        auto view = read.View<true>();
        values.Reset(R::RowLayout::Dense(0, 0));
        for (DNDS::index row = 0; row < view.RowCount(); ++row)
        {
            const auto span = view[row];
            CHECK(span.size() == shape.RowSize(row));
            for (DNDS::index column = 0; column < span.size(); ++column)
                CHECK(span[column] == int(10 * row + column));
            CHECK(span.empty() == (span.size() == 0));
        }
    }
}

TEST_CASE("Native fixed POD rows share read and write span access")
{
    struct Entry
    {
        int64_t id;
        double weight;
    };
    R::Array<Entry> values(R::RowLayout::Dense(2, 2, 3));
    {
        auto write = values.Write();
        write.View<R::FixedRows<2>>()[1][0] = {71, 1.5};
        write.View<R::FixedRows<2>>()[1][1] = {72, 2.5};
    }
    auto read = values.Read();
    CHECK(read.View<R::FixedRows<2>>()[1].size() == 2);
    CHECK(read.View<R::FixedRows<2>>()[1][0].id == 71);
    CHECK(read.View<R::FixedRows<2>>()[1][1].weight == 2.5);
    CHECK(read.View<R::FixedRows<2>>()[1].data() - read.View<R::FixedRows<2>>()[0].data() == 3);
}

TEST_CASE("native layout zero rows padding and overflow validation")
{
    R::Array<int> zero(R::RowLayout::Ragged({0, 0}));
    CHECK(zero.Read().View().RowSize(1) == 0);
    CHECK_THROWS(R::RowLayout::Dense(1, 4, 2));
    CHECK_THROWS(R::RowLayout::Dense(SIZE_MAX, 2));
    CHECK_THROWS(R::RowLayout::Ragged({SIZE_MAX, 1}));
    R::Array<int> padded(R::RowLayout::Dense(3, 2, 4));
    auto w = padded.Write();
    CHECK(w.View().Row(1) - w.View().Row(0) == 4);
    CHECK_THROWS(w.At(0, 2));
}

TEST_CASE("native typed handles share one allocation and logical metadata record")
{
    R::Array<double> fixed(R::RowLayout::Dense(3, 5, 8));
    auto dynamic = R::Array<double>::FromSharedStorage(fixed.SharedStorage());
    CHECK(dynamic.SharedStorage() == fixed.SharedStorage());
    CHECK(dynamic.SharedStorage()->SharedLayout() == fixed.SharedStorage()->SharedLayout());
    dynamic.Write().At(1, 3) = 19;
    CHECK(fixed.Read().At(1, 3) == 19);
    CHECK(fixed.Version() == dynamic.Version());
    {
        auto read = fixed.Read();
        CHECK_THROWS(dynamic.Write());
    }
    CHECK_THROWS(dynamic.Read().template View<R::FixedRows<7>>());
    CHECK_THROWS(R::Array<int32_t>::FromSharedStorage(dynamic.SharedStorage()));
    fixed.Reset(R::RowLayout::Dense(2, 5));
    CHECK(fixed.SharedStorage() != dynamic.SharedStorage());
    CHECK(dynamic.Read().At(1, 3) == 19);

    R::Array<int64_t> ragged(R::RowLayout::Ragged({0, 2, 3}));
    auto alias = R::Array<int64_t>::FromSharedStorage(ragged.SharedStorage());
    CHECK(alias.Layout().Offsets().data() == ragged.Layout().Offsets().data());
    auto access = alias.Write();
    ragged.Reset(R::RowLayout::Dense(0, 0));
    alias.Reset(R::RowLayout::Dense(0, 0));
    access.At(2, 2) = 42; // Lease retains values and metadata after both handles detach.
    CHECK(access.At(2, 2) == 42);
}

TEST_CASE("Imported six-equation state retains external owner and one access domain")
{
    int releases = 0;
    constexpr size_t rows = 37, stride = 8;
    auto owner = std::shared_ptr<void>(new double[rows * stride](), [&releases](void *p)
                                       { ++releases; delete[] static_cast<double *>(p); });
    auto allocation = R::Allocation::Import(owner, rows * stride * sizeof(double));
    auto buffer = R::Buffer::FromAllocation(allocation, 0, allocation.Bytes());
    auto array = R::Array<double, R::FixedRows<6>>::FromStorage(R::RowLayout::Dense(rows, 6, stride), buffer);
    auto ring = buffer.Subspan(30 * stride * sizeof(double), 7 * stride * sizeof(double));
    auto overlap = R::Buffer::FromAllocation(allocation, 31 * stride * sizeof(double), stride * sizeof(double));
    CHECK(ring.Identity() == array.Identity());
    CHECK(overlap.Identity() == array.Identity());
    {
        auto edit = array.Write();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < 6; ++j)
                edit.At(i, j) = 100 * i + j;
        CHECK_THROWS(ring.Read());
    }
    CHECK(overlap.Read().View<double>()[5] == 3105);
    CHECK(ring.Version() == array.Version());
    CHECK_THROWS(buffer.Subspan(buffer.Bytes(), 1));
    CHECK_THROWS(R::Buffer::FromAllocation(allocation, SIZE_MAX, 1));
    auto misaligned = buffer.Subspan(1, sizeof(double));
    CHECK_THROWS(R::Array<double>::FromStorage(R::RowLayout::Dense(1, 1), misaligned));
    auto lease = array.Read();
    owner.reset();
    allocation = {};
    buffer = ring = overlap = misaligned = {};
    array.Reset(R::RowLayout::Dense(0, 6));
    CHECK(releases == 0);
    CHECK(lease.At(36, 5) == 3605);
    lease.Wait();
    CHECK(releases == 1);
}

TEST_CASE("Buffer address queries detect foreign aliases without acquiring access")
{
    auto owner = std::make_shared<std::vector<uint8_t>>(64);
    const std::shared_ptr<void> address(owner, owner->data());
    const auto allocation = R::Allocation::Import(address, 64);
    auto whole = R::Buffer::FromAllocation(allocation, 0, 64);
    const auto first = whole.Subspan(0, 32), next = whole.Subspan(32, 32);
    const auto crossing = whole.Subspan(16, 32), empty = whole.Subspan(16, 0);
    // Separate imports deliberately have different controllers. Physical alias
    // detection must still work; this does not make such imports lease-safe.
    const auto foreign = R::Buffer::FromAllocation(R::Allocation::Import(address, 64), 0, 64);
    auto writer = whole.Write();
    CHECK_THROWS(whole.Read());
    CHECK(whole.SameRange(foreign));
    CHECK(whole.SameAddress(first));
    CHECK_FALSE(whole.SameRange(first));
    CHECK(first.Overlaps(foreign));
    CHECK(first.Overlaps(crossing));
    CHECK(crossing.Overlaps(first));
    CHECK_FALSE(first.Overlaps(next));
    CHECK_FALSE(whole.Overlaps(empty));
    CHECK_FALSE(R::Buffer{}.Overlaps(whole));
    CHECK(whole.IsAligned(1));
    CHECK_FALSE(whole.IsAligned(0));
    CHECK(R::Buffer{}.IsAligned(64));
}

TEST_CASE("Imported read-only storage rejects writes without invalidating readers")
{
    auto owner = std::make_shared<std::vector<int64_t>>(19, 41);
    auto token = std::shared_ptr<void>(owner, owner->data());
    auto allocation = R::Allocation::Import(token, owner->size() * sizeof(int64_t), R::Device::Host(), false);
    auto buffer = R::Buffer::FromAllocation(allocation, 0, allocation.Bytes());
    CHECK_THROWS(buffer.Write());
    CHECK(buffer.IsValid());
    CHECK(buffer.Version() == 0);
    CHECK(buffer.Read().View<int64_t>()[18] == 41);
    CHECK_THROWS(R::Allocation::Import({}, 0));
    auto emptyOwner = std::shared_ptr<void>(owner, nullptr);
    CHECK_THROWS(R::Allocation::Import(emptyOwner, 1));
    auto empty = R::Allocation::Import(emptyOwner, 0);
    CHECK(R::Buffer::FromAllocation(empty, 0, 0).Read().Bytes() == 0);
}

TEST_CASE("Independent reconstruction fields share buffered CSR metadata and retain old shape")
{
    std::vector<size_t> lengths;
    for (DNDS::index row = 0; row < 97; ++row)
        lengths.push_back(row % 11 == 0 ? 0 : (row % 3 + 1) * 5);
    auto shape = R::RowLayout::Ragged(lengths);
    auto copiedShape = shape;
    CHECK(copiedShape.Offsets().Identity() == shape.Offsets().Identity());
    R::Array<double> coefficients(shape);
    R::Array<int64_t> indices(shape);
    CHECK(indices.Layout().Offsets().data() == coefficients.Layout().Offsets().data());
    auto secondStorage = coefficients.SharedStorage()->WithValues(R::Buffer::Allocate(shape.Elements() * sizeof(double)), sizeof(double));
    auto second = R::Array<double>::FromSharedStorage(secondStorage);
    CHECK(second.SharedStorage()->SharedLayout() == coefficients.SharedStorage()->SharedLayout());
    CHECK(second.Identity() != coefficients.Identity());
    auto read = coefficients.Read();
    second.Write().At(96, 4) = 17;
    CHECK(read.At(96, 4) == 0);
    coefficients.Reset(R::RowLayout::Dense(1, 5));
    indices.Reset(R::RowLayout::Dense(0, 0));
    copiedShape = shape = {};
    CHECK(read.Layout().RowSize(96) == lengths[96]);
    CHECK(second.Read().At(96, 4) == 17);
    auto metadata = read.Layout().Offsets().Read();
    CHECK(metadata.Bytes() == (lengths.size() + 1) * sizeof(size_t));
}

TEST_CASE("Known-row construction fills independently and reports worker failures")
{
    constexpr size_t rows = 8193;
    auto values = R::BuildArrayRows<DNDS::index>(rows, [](size_t)
                                                 { return size_t(1); }, [](size_t row, auto output)
                                                 { output[0] = DNDS::index(3 * row); }, size_t(1), true);
    auto read = values.Read();
    REQUIRE(read.View().RowCount() == rows);
    for (DNDS::index row : {DNDS::index(0), DNDS::index(1), DNDS::index(rows / 2), DNDS::index(rows - 1)})
    {
        CHECK(read.View().RowSize(row) == 1);
        CHECK(read.View()(row, 0) == DNDS::index(3 * row));
    }

    CHECK_THROWS_AS((R::BuildArrayRows<int>(rows, [](size_t)
                                            { return size_t(1); }, [](size_t row, auto)
                                            {
                                               if (row == 4097)
                                                   throw std::runtime_error("worker failure"); }, size_t(1), true)),
                    std::runtime_error);
}

TEST_CASE("Physical tables preserve logical rows across padding and tile boundaries")
{
    for (DNDS::index nRow : std::vector<DNDS::index>{0, 1, 8, 9, 33})
        for (DNDS::index width : std::vector<DNDS::index>{1, 3, 5, 21, 37})
            for (const auto &shape : {R::RowLayout::Dense(nRow, width, width + 2),
                                      R::RowLayout::ColMajor(nRow, width, {8, 1}),
                                      R::RowLayout::TiledColMajor(nRow, width, 8)})
            {
                R::Array<double> values(shape);
                {
                    auto write = values.Write();
                    const auto view = write.View<true>();
                    auto flat = write.Layout().Elements();
                    std::fill_n(view.data, flat, -901.0);
                    for (DNDS::index iRow = 0; iRow < nRow; ++iRow)
                        for (DNDS::index iComp = 0; iComp < width; ++iComp)
                            view[iRow][iComp] = 1000 * iRow + iComp;
                }
                const auto read = values.Read();
                const auto generic = read.View();
                const auto uniform = read.View<R::UniformTableRows>();
                std::vector<bool> touched(shape.Elements(), false);
                for (DNDS::index iRow = 0; iRow < nRow; ++iRow)
                {
                    CHECK(generic[iRow].ElementStride() == shape.ComponentStride());
                    const std::vector<double> copied(generic[iRow].begin(), generic[iRow].end());
                    CHECK(copied.size() == size_t(width));
                    for (DNDS::index iComp = 0; iComp < width; ++iComp)
                    {
                        CHECK(generic(iRow, iComp) == 1000 * iRow + iComp);
                        CHECK(uniform[iRow][iComp] == copied[size_t(iComp)]);
                        touched[size_t(shape.ElementOffset(iRow, iComp))] = true;
                    }
                }
                for (DNDS::index iElement = 0; iElement < DNDS::index(shape.Elements()); ++iElement)
                    if (!touched[size_t(iElement)])
                        CHECK(generic.data[iElement] == -901.0);
            }
    CHECK(R::RowLayout::ColMajor(32, 5, {32, 0}).Stride() == 32);
    CHECK(R::RowLayout::ColMajor(32, 5, {32, 1}).Stride() == 64);
    CHECK(R::RowLayout::ColMajor(33, 5, {8, 0}).WithRowCount(2).Stride() == 8);
    CHECK(R::RowLayout::TiledColMajor(9, 5, 8).Elements() == 80);
    CHECK_THROWS(R::RowLayout::ColMajor(1, 5, {0, 0}));
    CHECK_THROWS(R::RowLayout::ColMajor(1, 5, {8, -1}));
    CHECK_THROWS(R::RowLayout::ColMajor(size_t(INT64_MAX), 5, {32, 1}));
    CHECK_THROWS(R::RowLayout::TiledColMajor(1, 5, 0));
    CHECK_THROWS(R::RowLayout::TiledColMajor(size_t(INT64_MAX), 5, 32));
    CHECK_THROWS(R::RowLayout::ColMajorWithPitch(9, 5, 8));
}

TEST_CASE("Physical projections are checked once and retain specialized component strides")
{
    R::Array<int> soa(R::RowLayout::ColMajor(9, 5, {8, 0}));
    R::Array<int> tiled(R::RowLayout::TiledColMajor(9, 5, 8));
    auto write = tiled.Write();
    const auto view = write.View<R::FixedWidthRows<R::TiledColMajorTable<8>, 5>, true>();
    static_assert(decltype(view)::row_width == 5);
    static_assert(decltype(view)::component_stride == 8);
    static_assert(decltype(view[0])::component_stride == 8);
    static_assert(std::is_trivially_copyable_v<decltype(view)>);
    view(8, 4) = 71;
    CHECK(write.At(8, 4) == 71);
    CHECK_THROWS(write.View<R::TiledColMajorTable<4>>());
    CHECK_THROWS(write.View<R::TableRows>());
    CHECK_THROWS(write.View<R::FixedWidthRows<R::TiledColMajorTable<8>, 4>>());
    CHECK(soa.Read().View<R::ColMajorTable>().ElementStride() == 16);
    CHECK_THROWS(soa.Read().View<R::FixedRows<5>>());
}

TEST_CASE("Kernel projections preserve fixed row extents and independent padded owned ghost pitches")
{
    R::ArrayPair<int> pair{R::Array<int>(R::RowLayout::Dense(3, 5, 8)),
                           R::Array<int>(R::RowLayout::Dense(2, 5, 6))};
    for (auto *part : {&pair.owned, &pair.ghost})
    {
        auto write = part->Write();
        const auto view = R::ProjectKernelView<R::FixedRows<5>>(write.View<R::UniformTableRows>());
        static_assert(decltype(view)::row_width == 5);
        static_assert(decltype(view[0])::row_width == 5);
        static_assert(std::is_trivially_copyable_v<decltype(view)>);
        for (DNDS::index iRow = 0; iRow < view.RowCount(); ++iRow)
            for (DNDS::index j = 0; j < 5; ++j)
                view[iRow][j] = int(100 * iRow + j);
    }
    const auto read = pair.Read();
    const auto view = R::ProjectKernelPair<R::FixedRows<5>>(read.View());
    static_assert(decltype(view)::row_width == 5);
    CHECK(view.owned.stride == 8);
    CHECK(view.ghost.stride == 6);
    CHECK(view[2][4] == 204);
    CHECK(view[4][4] == 104);
    CHECK(view.owned.data == read.View().owned.data);
    CHECK(view.ghost.data == read.View().ghost.data);
    CHECK_THROWS(R::ProjectKernelView<R::FixedRows<6>>(read.View().owned));
    R::Array<int> csr(R::RowLayout::Ragged({5, 5}));
    CHECK_THROWS(R::ProjectKernelView<R::FixedRows<5>>(csr.Read().View()));

    R::Array<int> tiled(R::RowLayout::TiledColMajor(9, 5, 8));
    auto write = tiled.Write();
    const auto fixed = R::ProjectKernelView<R::FixedWidthRows<R::TiledColMajorTable<8>, 5>>(write.View());
    static_assert(decltype(fixed[0])::row_width == 5);
    static_assert(decltype(fixed[0])::component_stride == 8);
    fixed[8][4] = 73;
    CHECK(write.At(8, 4) == 73);
}

TEST_CASE("Pair views keep independent column pitches and eager workers retain physical layout")
{
    R::ArrayPair<int> pair{R::Array<int>(R::RowLayout::ColMajor(33, 5, {32, 0})), R::Array<int>(R::RowLayout::ColMajor(3, 5, {8, 0}))};
    {
        auto write = pair.owned.Write();
        write.At(32, 4) = 812;
    }
    {
        auto write = pair.ghost.Write();
        write.At(2, 4) = 713;
    }
    const auto read = pair.Read();
    CHECK(read.View()(32, 4) == 812);
    CHECK(read.View()[35][4] == 713);
    CheckEagerTypedCPU<R::ColMajorTable>(R::RowLayout::ColMajor(9, 5, {8, 1}));
    CheckEagerTypedCPU<R::TiledColMajorTable<8>>(R::RowLayout::TiledColMajor(9, 5, 8));
}

TEST_CASE("Allocation alignment is provider-enforced and over-aligned POD survives placement snapshots")
{
    struct alignas(256) Record
    {
        double value;
    };
    R::Array<Record> values(R::RowLayout::Dense(3, 2));
    CHECK(reinterpret_cast<uintptr_t>(values.Storage().LegacyPointer()) % 256 == 0);
    {
        auto write = values.Write();
        write.At(1, 1).value = 41;
    }
    auto snapshot = values.To(R::Device::Host());
    CHECK(snapshot.Read().At(1, 1).value == 41);
    CHECK(reinterpret_cast<uintptr_t>(snapshot.Storage().LegacyPointer()) % 256 == 0);
    R::Array<double> aligned(R::RowLayout::ColMajor(33, 5, {32, 0}), R::Device::Host(), R::AllocationOptions{4096});
    CHECK(reinterpret_cast<uintptr_t>(aligned.Storage().LegacyPointer()) % 4096 == 0);
    CHECK_THROWS(R::Buffer::Allocate(10, R::Device::Host(), R::AllocationOptions{3}));
    CHECK_THROWS(R::Array<double>(R::RowLayout::Dense(1, 1), R::Device::Host(), R::AllocationOptions{3}));
}

TEST_CASE("Local Eigen matrices honor physical component stride without packing")
{
    for (const auto &shape : {R::RowLayout::Dense(9, 6, 8), R::RowLayout::ColMajor(9, 6, {8, 0}), R::RowLayout::TiledColMajor(9, 6, 8)})
    {
        R::Array<double> values(shape);
        {
            auto write = values.Write();
            const auto matrices = DNDS::Algebra::MakeEigenRowMatrices(write.View<R::UniformTableRows>(), 2, 3);
            for (DNDS::index iRow = 0; iRow < 9; ++iRow)
                for (DNDS::index i = 0; i < 2; ++i)
                    for (DNDS::index j = 0; j < 3; ++j)
                        matrices[iRow](i, j) = 100 * iRow + i + 2 * j;
        }
        const auto read = values.Read();
        const auto matrices = DNDS::Algebra::MakeEigenRowMatrices(read.View<R::UniformTableRows>(), 2, 3);
        const auto fixed = DNDS::Algebra::MakeEigenRowMatrices<2, 3>(read.View<R::UniformTableRows>(), 2, 3);
        static_assert(decltype(fixed[0])::RowsAtCompileTime == 2 && decltype(fixed[0])::ColsAtCompileTime == 3);
        CHECK(fixed[1].isApprox(matrices[1]));
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(matrices[0].data())>>);
        for (DNDS::index iRow = 0; iRow < 9; ++iRow)
            CHECK(matrices[iRow].sum() == 600 * iRow + 15);
        CHECK_THROWS(DNDS::Algebra::MakeEigenRowMatrices(read.View(), 3, 3));
        CHECK_THROWS(DNDS::Algebra::MakeEigenRowMatrices(read.View(), 2, 3, R::Device::CUDA(0)));
    }
}

TEST_CASE("Explicit logical conversion preserves values and retained source views")
{
    R::Array<double> source(R::RowLayout::Dense(9, 5, 7));
    {
        auto write = source.Write();
        for (DNDS::index iRow = 0; iRow < 9; ++iRow)
            for (DNDS::index iComp = 0; iComp < 5; ++iComp)
                write.View()(iRow, iComp) = 100 * iRow + iComp;
    }
    const auto retained = source.Read();
    for (const auto &shape : {R::RowLayout::ColMajor(9, 5, {8, 1}), R::RowLayout::TiledColMajor(9, 5, 8)})
    {
        const auto converted = R::ConvertLayout(source, shape);
        const auto read = converted.Read();
        CHECK(converted.Identity() != source.Identity());
        for (DNDS::index iRow = 0; iRow < 9; ++iRow)
            for (DNDS::index iComp = 0; iComp < 5; ++iComp)
                CHECK(read.View()(iRow, iComp) == retained.View()(iRow, iComp));
    }
    CHECK_THROWS(R::ConvertLayout(source, R::RowLayout::ColMajor(8, 5)));
    CHECK_THROWS(R::ConvertLayout(source, R::RowLayout::ColMajor(9, 4)));
}

#if defined(DNDS_UNIX_LIKE) && !defined(DNDS_NDEBUG)
TEST_CASE("Checked element projections reject invalid row and component before memory access")
{
    for (int failure = 0; failure < 3; ++failure)
    {
        const auto child = fork();
        REQUIRE(child >= 0);
        if (child == 0)
        {
            struct rlimit limit
            {
                0, 0
            };
            setrlimit(RLIMIT_CORE, &limit);
            std::signal(SIGABRT, SIG_DFL);
            (void)std::freopen("/dev/null", "w", stderr);
            double data[40]{};
            R::ArrayView<double, true, R::TiledColMajorTable<8>> view{data, nullptr, 3, 5, 8, nullptr, R::RowLayoutKind::TiledColMajorTable};
            if (failure == 0)
                view(-1, 0) = 1;
            if (failure == 1)
                view(0, 5) = 1;
            if (failure == 2)
                view[0][-1] = 1;
            _exit(0);
        }
        int status = 0;
        REQUIRE(waitpid(child, &status, 0) == child);
        CHECK(WIFSIGNALED(status));
        if (WIFSIGNALED(status))
            CHECK(WTERMSIG(status) == SIGABRT);
    }
}
#endif
