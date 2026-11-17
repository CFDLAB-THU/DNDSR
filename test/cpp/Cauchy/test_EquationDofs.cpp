#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationViews.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "Solver/FieldTimeStepper.hpp"
#include "Cauchy/State/EquationTimeStepper.hpp"
#include "Cauchy/State/EquationGMRES.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include <numeric>
#include <optional>

namespace C = DNDS::Cauchy;
namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
namespace
{
    C::EquationSchema Schema(int flow)
    {
        C::EquationSchema schema{"mock-coupled", "mock-energy", "identity", {}};
        C::EquationBlockSchema block;
        block.name = "flow";
        for (const auto *name : flow == 4 ? std::vector<const char *>{"u", "v", "w", "p"}
                                          : std::vector<const char *>{"rho", "rhou", "rhov", "rhow", "rhop"})
            block.components.push_back({name, "mock-units", "flow"});
        block.vectorStarts = {flow == 4 ? 0 : 1};
        schema.blocks.push_back(std::move(block));
        for (const auto &[name, count] : std::vector<std::pair<std::string, int>>{{"rans", flow == 4 ? 1 : 2}, {"species", flow == 4 ? 12 : 14}})
        {
            C::EquationBlockSchema extra;
            extra.name = name;
            for (int j = 0; j < count; ++j)
                extra.components.push_back({name + std::to_string(j), "mock-units", name});
            schema.blocks.push_back(std::move(extra));
        }
        return schema;
    }
    I Width(const C::EquationSchema &schema)
    {
        I count = 0;
        for (const auto &block : schema.blocks)
            count += I(block.components.size());
        return count;
    }
    R::RowLayout Layout(I width, int layout)
    {
        if (layout == 1)
            return R::RowLayout::ColMajor(0, width, {8, 1});
        if (layout == 2)
            return R::RowLayout::TiledColMajor(0, width, 32);
        return R::RowLayout::Dense(0, width, width + 3);
    }
    struct Fixture
    {
        std::shared_ptr<R::CommunicationContext> communication;
        int active;
        I rows, ghosts;
        A::DofSpace space;
        R::RowTransferPlan route;
        static R::RowTransferPlan Route(const std::shared_ptr<R::CommunicationContext> &context, int active, I rows, I ghosts)
        {
            std::vector<std::vector<I>> send(mpi.size), receive(mpi.size);
            if (mpi.rank < active)
            {
                send[(mpi.rank + 1) % active] = {rows - 2, rows - 1};
                receive[(mpi.rank + active - 1) % active] = {0, 1};
            }
            return R::RowTransferPlan::Known(context, rows, ghosts, send, receive);
        }
        Fixture()
            : communication(std::make_shared<R::CommunicationContext>(mpi.comm)), active(mpi.size > 2 ? mpi.size - 1 : mpi.size),
              rows(mpi.rank < active ? 35 : 0), ghosts(mpi.rank < active ? 2 : 0),
              space(communication, size_t(rows), size_t(ghosts), "semantic DOF ring"), route(Route(communication, active, rows, ghosts)) {}
        C::EquationDofs Build(const C::EquationSchema &schema, int layout, bool split) const
        {
            if (!split)
                return C::EquationDofs::Coupled(schema, A::DofField<double>(space, Layout(Width(schema), layout)), "packed");
            A::DofBundle fields(space);
            for (const auto &block : schema.blocks)
                fields.Add(block.name, A::DofField<double>(space, Layout(I(block.components.size()), layout)));
            return C::EquationDofs::Separate(schema, std::move(fields));
        }
        R::TransferBinding Bind(A::DofField<double> &field) const
        {
            return R::PayloadPlan(route, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::For<double>())
                .Bind(field.Owned(), field.Ghost());
        }
    };
    double Value(int owner, I row, I component) { return 1 + 100 * owner + 2 * row + .125 * component; }
    void Fill(C::EquationDofs &state)
    {
        auto write = state.WriteOwned();
        I first = 0;
        for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
        {
            const auto out = write.Block(iBlock);
            for (I iRow = 0; iRow < out.RowCount(); ++iRow)
                for (I j = 0; j < out.ComponentNumber(); ++j)
                    out(iRow, j) = Value(mpi.rank, iRow, first + j);
            first += out.ComponentNumber();
        }
    }
    void CheckValues(const C::EquationDofs &state, double scale = 1, double shift = 0)
    {
        const auto read = state.ReadOwned();
        I first = 0;
        for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
        {
            const auto in = read.Block(state.Schema().blocks[iBlock].name);
            for (I iRow = 0; iRow < in.owned.RowCount(); ++iRow)
                for (I j = 0; j < in.ComponentNumber(); ++j)
                    CHECK(in(iRow, j) == scale * Value(mpi.rank, iRow, first + j) + shift);
            first += in.ComponentNumber();
        }
    }
    void Pointwise(const C::EquationDofs &input, C::EquationDofs &output)
    {
        input.RequireCompatible(output);
        const auto read = input.ReadOwned();
        auto write = output.WriteOwned();
        for (I iBlock = 0; iBlock < input.BlockNumber(); ++iBlock)
        {
            const auto in = read.Block(iBlock).owned;
            const auto out = write.Block(iBlock);
            for (I iRow = 0; iRow < out.RowCount(); ++iRow)
                for (I j = 0; j < out.ComponentNumber(); ++j)
                    out[iRow][j] = 3 * in[iRow][j] - 2;
        }
    }
}
static_assert(std::is_trivially_copyable_v<C::EquationBlockArrayView<const double>>);
static_assert(std::is_trivially_copyable_v<C::EquationBlockReadView>);
static_assert(C::EquationBlockArrayView<double, R::AccessBoundsChecks, 5, R::TableRows>::RowType::row_width == 5);
static_assert(C::EquationBlockArrayView<double, R::AccessBoundsChecks, 5, R::TableRows>::RowType::component_stride == 1);
static_assert(C::EquationBlockArrayView<double, R::AccessBoundsChecks, 5, R::TiledColMajorTable<32>>::RowType::component_stride == 32);
static_assert(C::EquationBlockedView<double, C::EquationViewKind::CombinedAoS, 5>::component_stride == 1);
static_assert(C::EquationBlockedView<double, C::EquationViewKind::CombinedSoA, 5>::component_stride == R::DynamicExtent);
static_assert(C::EquationBlockedView<double, C::EquationViewKind::CombinedAoSoA, 5, 32>::component_stride == 32);
TEST_CASE("Equation schema mapping uses semantic names instead of component count or ordinal")
{
    const auto source = Schema(5);
    auto target = source;
    std::swap(target.blocks[1], target.blocks[2]);
    const auto mapping = source.MapComponentsTo(target);
    REQUIRE(mapping.size() == size_t(Width(source)));
    for (I j = 0; j < 5; ++j)
        CHECK(mapping[size_t(j)] == j);
    for (I j = 0; j < 2; ++j)
        CHECK(mapping[size_t(5 + j)] == 19 + j);
    for (I j = 0; j < 14; ++j)
        CHECK(mapping[size_t(7 + j)] == 5 + j);

    auto renamedSpecies = source;
    renamedSpecies.blocks[2].components[3].name = "another_species";
    CHECK_THROWS(source.MapComponentsTo(renamedSpecies));

    auto changedVector = source;
    changedVector.blocks[0].vectorStarts.clear();
    CHECK_THROWS(source.MapComponentsTo(changedVector));

    auto changedEnergy = source;
    changedEnergy.energyConvention = "another-energy";
    CHECK_THROWS(source.MapComponentsTo(changedEnergy));
}
TEST_CASE("Whole-equation dispatch preserves named blocks and scalar order across bounded view families")
{
    Fixture fixture;
    for (int width : {5, 6, 7, 21, 17})
        for (int layout = 0; layout < 3; ++layout)
            for (bool split : {false, true})
            {
                auto schema = Schema(width == 17 ? 4 : 5);
                if (width <= 7)
                {
                    schema.blocks.resize(width == 5 ? 1 : 2);
                    if (width > 5)
                        schema.blocks[1].components.resize(width - 5);
                }
                auto input = fixture.Build(schema, layout, split);
                Fill(input);
                auto output = input.Clone();
                {
                    const auto read = input.ReadOwned();
                    auto write = output.WriteOwned();
                    const C::EquationViewLayout<const double> source(input, read);
                    const C::EquationViewLayout<double> target(output, write);
                    const auto kind = source.Combined()
                                          ? (source.Table() && source.Flow5()
                                                 ? C::EquationViewKind::CombinedAoS
                                             : source.ColMajor() && source.Flow5()
                                                 ? C::EquationViewKind::CombinedSoA
                                             : source.TiledColMajor(32) && source.Flow5()
                                                 ? C::EquationViewKind::CombinedAoSoA
                                                 : C::EquationViewKind::Combined)
                                      : source.Flow5() ? (source.FlowTable() ? C::EquationViewKind::SeparateFlow5AoS : C::EquationViewKind::SeparateFlow5)
                                                       : C::EquationViewKind::Generic;
                    const auto automaticKind = source.Combined() && source.Flow5()
                                                   ? (layout == 0
                                                          ? C::EquationViewKind::CombinedAoS
                                                      : layout == 1
                                                          ? C::EquationViewKind::CombinedSoA
                                                          : C::EquationViewKind::CombinedAoSoA)
                                                   : C::EquationViewKind::Generic;
                    C::DispatchCPUEquationLayouts(
                        [&](auto u, auto v)
                        {
                            CHECK(decltype(u)::kind == automaticKind);
                            CHECK(decltype(v)::kind == automaticKind);
                        },
                        source, target);
                    const auto kernel = [&](auto u, auto v)
                    {
                    using V = decltype(u);
                    using Row = decltype(u[I(0)]);
                    using BlockRow = decltype(u[I(0)].RANS());
                    static_assert(std::is_trivially_copyable_v<V> && std::is_standard_layout_v<V>);
                    static_assert(std::is_trivially_copyable_v<Row> && std::is_standard_layout_v<Row>);
                    static_assert(Row::row_width == V::row_width && Row::component_stride == V::component_stride);
                    static_assert(BlockRow::component_stride == V::component_stride);
                    static_assert(Row::has_single_parent == V::combined);
                    CHECK(V::combined == source.Combined());
                    CHECK((V::flow_width == 5) == (width != 17));
                    CHECK((V::row_width != R::DynamicExtent) == (source.Combined() && width <= 7));
                    const auto generic = source.View();
                    for (I iCell = 0; iCell < fixture.rows; ++iCell)
                    {
                        // A retained row survives a temporary view wrapper.
                        const auto row = decltype(u)(u)[iCell];
                        static_assert(std::is_same_v<decltype(row[0]), const double &>);
                        static_assert(std::is_same_v<decltype(v[iCell][0]), double &>);
                        CHECK(row.Flow().size() == (width == 17 ? 4 : 5));
                        CHECK(row.RANS().size() == (width <= 7 ? width - 5 : width == 17 ? 1 : 2));
                        CHECK(row.Species().size() == (width <= 7 ? 0 : width == 17 ? 12 : 14));
                        if constexpr (Row::has_single_parent)
                        {
                            if (!row.RANS().empty())
                                CHECK(row.RANS().ElementStride() == row.ElementStride());
                            if (!row.Species().empty())
                                CHECK(row.Species().ElementStride() == row.ElementStride());
                        }
                        for (I j = 0; j < width; ++j)
                        {
                            CHECK(row[j] == Value(mpi.rank, iCell, j));
                            CHECK(row[j] == generic(iCell, j));
                            CHECK(&row[j] == &generic(iCell, j));
                            if constexpr (Row::has_single_parent)
                                CHECK(&row[j] == row.data() + j * row.ElementStride());
                            if constexpr (Row::component_stride != R::DynamicExtent)
                                CHECK(row.ElementStride() == I(Row::component_stride));
                            v[iCell][j] = 2 * row[j] - 3;
                        }
                    } };
                    if (kind == C::EquationViewKind::CombinedAoS)
                    {
                        if (width == 5)
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedAoS, 5>(kernel, source, target);
                        else if (width == 6)
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedAoS, 6>(kernel, source, target);
                        else if (width == 7)
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedAoS, 7>(kernel, source, target);
                        else
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedAoS>(kernel, source, target);
                    }
                    else if (kind == C::EquationViewKind::CombinedSoA)
                    {
                        if (width == 5)
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedSoA, 5>(kernel, source, target);
                        else if (width == 6)
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedSoA, 6>(kernel, source, target);
                        else if (width == 7)
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedSoA, 7>(kernel, source, target);
                        else
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedSoA>(kernel, source, target);
                    }
                    else if (kind == C::EquationViewKind::CombinedAoSoA)
                    {
                        if (width == 5)
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedAoSoA, 5, 32>(kernel, source, target);
                        else if (width == 6)
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedAoSoA, 6, 32>(kernel, source, target);
                        else if (width == 7)
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedAoSoA, 7, 32>(kernel, source, target);
                        else
                            C::DispatchCPUEquationViews<C::EquationViewKind::CombinedAoSoA, R::DynamicExtent, 32>(kernel, source, target);
                    }
                    else if (kind == C::EquationViewKind::Combined)
                        C::DispatchCPUEquationViews<C::EquationViewKind::Combined>(kernel, source, target);
                    else if (kind == C::EquationViewKind::SeparateFlow5AoS)
                        C::DispatchCPUEquationViews<C::EquationViewKind::SeparateFlow5AoS>(kernel, source, target);
                    else if (kind == C::EquationViewKind::SeparateFlow5)
                        C::DispatchCPUEquationViews<C::EquationViewKind::SeparateFlow5>(kernel, source, target);
                    else
                        C::DispatchCPUEquationViews<>(kernel, source, target);
                }
                CheckValues(output, 2, -3);
            }
}
TEST_CASE("Fixed equation block views preserve parent pitch and component ranges without changing storage")
{
    Fixture fixture;
    for (int layout = 0; layout < 3; ++layout)
        for (bool split : {false, true})
        {
            auto state = fixture.Build(Schema(5), layout, split);
            Fill(state);
            const auto read = state.ReadOwned();
            const auto flow = read.Block<5>("flow");
            const auto rans = read.Block("rans").owned;
            const auto species = read.Block("species").owned;
            static_assert(decltype(flow)::row_width == 5);
            static_assert(decltype(rans)::row_width == R::DynamicExtent);
            static_assert(decltype(species)::row_width == R::DynamicExtent);
            CHECK(flow.owned.parent.Width() == (split ? 5 : 21));
            CHECK(rans.parent.Width() == (split ? 2 : 21));
            CHECK(flow.ghost.RowCount() == 0); // Owned-only access never exposes ghost rows.
            for (I iRow = 0; iRow < fixture.rows; ++iRow)
            {
                CHECK(flow[iRow].size() == 5);
                CHECK(flow[iRow][4] == Value(mpi.rank, iRow, 4));
                CHECK(rans[iRow][1] == Value(mpi.rank, iRow, 6));
                CHECK(species[iRow][13] == Value(mpi.rank, iRow, 20));
            }
            CHECK_THROWS(read.Block<6>("flow"));
            if (layout == 0)
            {
                const auto table = flow.AsTable();
                static_assert(decltype(table)::component_stride == 1);
                CHECK(table.owned.parent.data == flow.owned.parent.data);
                CHECK(table.owned.parent.stride == flow.owned.parent.stride);
                const auto extra = rans.AsTable();
                CHECK(extra.first == (split ? 0 : 5));
                for (I iRow = 0; iRow < fixture.rows; ++iRow)
                {
                    CHECK(table[iRow][4] == Value(mpi.rank, iRow, 4));
                    CHECK(extra[iRow][1] == Value(mpi.rank, iRow, 6));
                }
            }
            else
                CHECK_THROWS(flow.owned.AsTable());
        }
}
TEST_CASE("Logical flow4/5 with RANS and many species maps exactly across physical layouts and storage groups")
{
    Fixture fixture;
    for (int flow : {4, 5})
        for (int layout = 0; layout < 3; ++layout)
            for (bool split : {false, true})
            {
                const auto schema = Schema(flow);
                auto state = fixture.Build(schema, layout, split);
                Fill(state);
                CheckValues(state);
                CHECK(state.ComponentNumber() == (flow == 4 ? 17 : 21));
                CHECK(state.BlockNumber() == 3);
                CHECK(state.PhysicalFieldNumber() == (split ? 3 : 1));
                I componentOffset = 0;
                for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto location = state.LocateBlock(schema.blocks[iBlock].name);
                    CHECK(location.physicalField == (split ? iBlock : 0));
                    CHECK(location.firstComponent == (split ? 0 : componentOffset));
                    CHECK(location.componentNumber == I(schema.blocks[iBlock].components.size()));
                    componentOffset += location.componentNumber;
                }
                // Independent native full-field oracle catches a segment view
                // that rewrites a tiled parent's physical component count.
                I logicalFirst = 0;
                for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                {
                    const auto &field = state.PhysicalField(iField);
                    const auto read = field.Owned().Read();
                    const auto view = read.View<R::UniformTableRows>();
                    for (I iRow = 0; iRow < fixture.rows; ++iRow)
                        for (I j = 0; j < view.Width(); ++j)
                            CHECK(view(iRow, j) == Value(mpi.rank, iRow, logicalFirst + j));
                    logicalFirst += view.Width();
                }
                auto output = state.RebindFields(state.Fields().Clone());
                Pointwise(state, output);
                CheckValues(output, 3, -2);
                CheckValues(state);
                if (split)
                    CHECK_THROWS(state.CoupledField());
                else
                    CHECK(state.CoupledField().Width() == size_t(state.ComponentNumber()));
                {
                    auto writer = output.WriteOwned();
                    const auto selected = writer.Block("rans");
                    for (I iRow = 0; iRow < selected.RowCount(); ++iRow)
                        for (I j = 0; j < selected.ComponentNumber(); ++j)
                            selected(iRow, j) = -7;
                    // Each physical writer is acquired once even when all
                    // three logical ranges share one allocation.
                    const auto flowView = writer.Block("flow");
                    if (flow == 5)
                    {
                        const auto fixed = flowView.Fixed<5>();
                        static_assert(decltype(fixed[0])::row_width == 5);
                    }
                    else
                        CHECK_THROWS(flowView.Fixed<5>());
                }
                const auto read = output.ReadOwned();
                I first = 0;
                for (I iBlock = 0; iBlock < output.BlockNumber(); ++iBlock)
                {
                    const auto block = read.Block(iBlock).owned;
                    for (I iRow = 0; iRow < fixture.rows; ++iRow)
                        for (I j = 0; j < block.ComponentNumber(); ++j)
                            CHECK(block(iRow, j) == (iBlock == 1 ? -7 : 3 * Value(mpi.rank, iRow, first + j) - 2));
                    first += block.ComponentNumber();
                }
            }
}
TEST_CASE("Hybrid equation blocks expose physical grouping and rebind across parent layouts")
{
    Fixture fixture;
    A::DofBundle fields(fixture.space);
    fields.Add("chemistry", A::DofField<double>(fixture.space, Layout(14, 1)));
    fields.Add("fluid", A::DofField<double>(fixture.space, Layout(7, 2)));
    C::EquationDofs state(Schema(5), std::move(fields), {{"flow", "fluid", 0}, {"rans", "fluid", 5}, {"species", "chemistry", 0}});
    Fill(state);
    const auto flow = state.LocateBlock("flow"), rans = state.LocateBlock("rans"), species = state.LocateBlock("species");
    CHECK(flow.physicalField == 1);
    CHECK(rans.physicalField == flow.physicalField);
    CHECK(rans.firstComponent == 5);
    CHECK(species.physicalField == 0);
    CHECK(state.PhysicalField(flow.physicalField).Width() == 7);
    CHECK(state.PhysicalField(species.physicalField).Width() == 14);
    A::DofBundle stageFields(fixture.space);
    stageFields.Add("chemistry", A::DofField<double>(fixture.space, Layout(14, 2)));
    stageFields.Add("fluid", A::DofField<double>(fixture.space, Layout(7, 0)));
    auto stage = state.RebindFields(std::move(stageFields));
    Pointwise(state, stage);
    CheckValues(stage, 3, -2);
    CheckValues(state);
    CHECK(stage.LocateBlock("rans").firstComponent == 5);
}
TEST_CASE("Owned-only pointwise algebra and reductions overlap direct halo writes without regional splitting")
{
    Fixture fixture;
    for (int layout = 0; layout < 3; ++layout)
        for (bool split : {false, true})
        {
            auto state = fixture.Build(Schema(5), layout, split);
            Fill(state);
            auto output = state.Clone();
            std::vector<R::TransferBinding> bindings;
            bindings.reserve(size_t(state.PhysicalFieldNumber()));
            for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                bindings.push_back(fixture.Bind(state.PhysicalField(iField)));
            std::vector<R::TransferOperation> transfers;
            transfers.reserve(bindings.size());
            for (auto &binding : bindings)
                transfers.push_back(binding.Begin());
            Pointwise(state, output);
            CheckValues(output, 3, -2);
            const auto products = C::EquationDofOperations::LocalProducts(state, {&state, &output});
            std::array<double, 2> productReference{};
            for (I iRow = 0; iRow < fixture.rows; ++iRow)
                for (I component = 0; component < state.ComponentNumber(); ++component)
                {
                    const auto value = Value(mpi.rank, iRow, component);
                    productReference[0] += value * value;
                    productReference[1] += value * (3 * value - 2);
                }
            CHECK(products[0] == doctest::Approx(productReference[0]).epsilon(1e-13));
            CHECK(products[1] == doctest::Approx(productReference[1]).epsilon(1e-13));
            if (fixture.ghosts)
            {
                CHECK_THROWS(state.ReadWithGhost());
                CHECK_THROWS(state.WriteOwned()); // Direct sends still read owned storage.
            }
            double sum = 0;
            {
                const auto read = state.ReadOwned();
                for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto v = read.Block(iBlock).owned;
                    for (I iRow = 0; iRow < v.RowCount(); ++iRow)
                        for (I j = 0; j < v.ComponentNumber(); ++j)
                            sum += v(iRow, j) * v(iRow, j);
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, fixture.communication->Communicator());
            double expected = 0;
            for (int owner = 0; owner < fixture.active; ++owner)
                for (I iRow = 0; iRow < 35; ++iRow)
                    for (I j = 0; j < state.ComponentNumber(); ++j)
                        expected += Value(owner, iRow, j) * Value(owner, iRow, j);
            CHECK(sum == doctest::Approx(expected).epsilon(1e-13));
            for (auto &transfer : transfers)
                transfer.Wait();
            const auto read = state.ReadWithGhost();
            I first = 0;
            for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            {
                const auto in = read.Block(iBlock);
                for (I iGhost = 0; iGhost < fixture.ghosts; ++iGhost)
                    for (I j = 0; j < in.ComponentNumber(); ++j)
                        CHECK(in(fixture.rows + iGhost, j) == Value((mpi.rank + fixture.active - 1) % fixture.active, 33 + iGhost, first + j));
                first += in.ComponentNumber();
            }
        }
}
TEST_CASE("Constant-density physical accumulation omits only the algebraic pressure channel")
{
    Fixture fixture;
    auto schema = Schema(4);
    schema.family = "incompressible-constant-density";
    schema.energyConvention = "none";
    schema.physicalAccumulation = "constant-density-velocity-pressure";
    schema.blocks.resize(2); // flow4 + one differential RANS scalar
    for (int layout = 0; layout < 3; ++layout)
        for (bool split : {false, true})
        {
            CAPTURE(layout);
            CAPTURE(split);
            auto state = fixture.Build(schema, layout, split);
            Fill(state);
            auto result = C::EquationDofOperations::NewVector(state);
            const auto check = [&](double differentialScale, double pressureScale)
            {
                const auto read = result.ReadOwned();
                I first = 0;
                for (I iBlock = 0; iBlock < result.BlockNumber(); ++iBlock)
                {
                    const auto values = read.Block(iBlock).owned;
                    for (I iRow = 0; iRow < values.RowCount(); ++iRow)
                        for (I j = 0; j < values.ComponentNumber(); ++j)
                        {
                            const I component = first + j;
                            const double scale = component == 3 ? pressureScale : differentialScale;
                            CHECK(values(iRow, j) == doctest::Approx(scale * Value(mpi.rank, iRow, component)));
                        }
                    first += values.ComponentNumber();
                }
            };

            C::EquationDofOperations::PhysicalCopyScale(result, 2, state);
            check(2, 0);
            C::EquationDofOperations::PhysicalLinearCombination(result, 2, state, -1, state);
            check(1, 0);
            C::EquationDofOperations::PhysicalStateAndRate(result, 2, state, -1, state);
            check(1, -1);
            C::EquationDofOperations::Copy(result, state);
            C::EquationDofOperations::PhysicalAxpy(result, 2, state);
            check(3, 1);
            C::EquationDofOperations::AssembleTimeResidual(result, state, state, state, 2, 3);
            check(2, 4);
        }
}
TEST_CASE("Semantic validation and writer reservation reject bad bindings without partial publication")
{
    Fixture fixture;
    const auto schema = Schema(5);
    const auto build = [&](C::EquationSchema requested, std::vector<C::EquationDofBinding> bindings, I width = 21)
    {
        A::DofBundle fields(fixture.space);
        fields.Add("packed", A::DofField<double>(fixture.space, Layout(width, 2)));
        return C::EquationDofs(std::move(requested), std::move(fields), std::move(bindings));
    };
    const std::vector<C::EquationDofBinding> ordinary{{"flow", "packed", 0}, {"rans", "packed", 5}, {"species", "packed", 7}};
    auto malformed = ordinary;
    malformed[1].first = 4;
    CHECK_THROWS(build(schema, malformed));
    malformed = ordinary;
    malformed[0].first = -1;
    CHECK_THROWS(build(schema, malformed));
    malformed = ordinary;
    malformed[0].first = 20;
    CHECK_THROWS(build(schema, malformed));
    malformed = ordinary;
    malformed[2].field = "missing";
    CHECK_THROWS(build(schema, malformed));
    malformed = ordinary;
    malformed[2].block = "flow";
    CHECK_THROWS(build(schema, malformed));
    CHECK_THROWS(build(schema, ordinary, 22));
    auto badSchema = schema;
    badSchema.blocks[0].vectorStarts = {4};
    CHECK_THROWS(build(badSchema, ordinary));
    badSchema = schema;
    badSchema.blocks[1].name = "flow";
    CHECK_THROWS(build(badSchema, ordinary));
    auto permuted = build(schema, {{"species", "packed", 0}, {"flow", "packed", 14}, {"rans", "packed", 19}});
    Fill(permuted);
    CheckValues(permuted);
    CHECK_THROWS(permuted.CoupledField());
    {
        const auto read = permuted.PhysicalField(0).Owned().Read();
        const auto values = read.View<R::UniformTableRows>();
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            for (I j = 0; j < 14; ++j)
                CHECK(values(iRow, j) == Value(mpi.rank, iRow, 7 + j));
            for (I j = 0; j < 5; ++j)
                CHECK(values(iRow, 14 + j) == Value(mpi.rank, iRow, j));
            for (I j = 0; j < 2; ++j)
                CHECK(values(iRow, 19 + j) == Value(mpi.rank, iRow, 5 + j));
        }
    }
    auto state = fixture.Build(schema, 1, true);
    Fill(state);
    std::vector<uint64_t> versions;
    for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        versions.push_back(state.PhysicalField(iField).Owned().Version());
    {
        const auto held = state.PhysicalField(2).Owned().Read();
        CHECK_THROWS(state.WriteOwned()); // Earlier field reservations must cancel unchanged.
    }
    for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        CHECK(state.PhysicalField(iField).Owned().Version() == versions[iField]);
    CheckValues(state);
    CHECK_THROWS(state.ReadOwned().Block("missing"));
    {
        auto write = state.WriteOwned();
        CHECK_THROWS(write.Block(3));
        write.CancelUnmodified();
    }
    const auto model = C::EquationSystem::Resolve(C::EquationSettings{});
    auto pressure = model.Schema();
    pressure.blocks[0].components.back().name = "rhop";
    auto pressureState = C::EquationDofs::Coupled(pressure, A::DofField<double>(fixture.space, 5));
    CHECK(pressureState.ComponentNumber() == 5);
    CHECK_THROWS(pressureState.RequireSchema(model.Schema())); // Same width, different equation meaning.
    CHECK_THROWS(pressureState.RequireSchema(model.Schema()));
    auto ordinaryState = C::EquationDofs::Coupled(model.Schema(), A::DofField<double>(fixture.space, 5));
    CHECK_NOTHROW(ordinaryState.RequireSchema(model.Schema()));
    CHECK(model.ResolveKernelProfile(R::Device::Host()) == C::EquationKernelProfile::CompressiblePerfectGasCPU);
    CHECK_THROWS(model.ResolveKernelProfile(R::Device::CUDA(0))); // No CPU fallback for a missing device kernel.
}
TEST_CASE("Retained scopes preserve allocation lifetime and storage replacement requires explicit rebinding")
{
    Fixture fixture;
    auto retained = [&]
    {
        auto temporary = fixture.Build(Schema(5), 2, false);
        Fill(temporary);
        return temporary.ReadOwned();
    }();
    // Churn equally sized allocations after the state owner has disappeared.
    std::vector<A::DofField<double>> churn;
    for (int i = 0; i < 8; ++i)
    {
        churn.emplace_back(fixture.space, Layout(21, 2));
        A::Fill(churn.back(), -999.);
    }
    const auto flow = retained.Block("flow").owned.Fixed<5>();
    for (I iRow = 0; iRow < fixture.rows; ++iRow)
        for (I j = 0; j < 5; ++j)
            CHECK(flow[iRow][j] == Value(mpi.rank, iRow, j));
    auto state = fixture.Build(Schema(5), 0, false);
    Fill(state);
    auto &physical = state.CoupledField();
    const auto before = state.ReadOwned();
    physical.Owned() = R::Array<double>(physical.Owned().Layout());
    CHECK_THROWS(state.Validate());
    const auto old = before.Block("species").owned;
    for (I iRow = 0; iRow < fixture.rows; ++iRow)
        for (I j = 0; j < 14; ++j)
            CHECK(old(iRow, j) == Value(mpi.rank, iRow, 7 + j));
}
TEST_CASE("Numeric equation aliases and read scopes retain rows and values without retaining MPI context")
{
    std::weak_ptr<R::CommunicationContext> context;
    std::shared_ptr<const A::DofSpace::Partition> identity;
    std::shared_ptr<C::EquationDofs> numeric;
    std::shared_ptr<C::EquationDofs::ReadScope> read;
    {
        auto communication = std::make_shared<R::CommunicationContext>(mpi.comm);
        context = communication;
        const A::DofSpace space(communication, 2, 0, "numeric worker lifetime");
        identity = space.Pin();
        auto state = C::EquationDofs::Coupled(Schema(5), A::DofField<double>(space, R::RowLayout::TiledColMajor(0, 21, 32)));
        Fill(state);
        numeric = std::make_shared<C::EquationDofs>(state.NumericAlias());
        read = std::make_shared<C::EquationDofs::ReadScope>(state.ReadOwned());
        CHECK(numeric->Space().Same(space));
        CHECK_FALSE(numeric->Space().Distributed());
    }
    CHECK(context.expired()); // MPI resources retire here on the controller.
    CHECK(numeric->Space().Pin() == identity);
    double result = 0;
    R::EagerEngine engine(2);
    const auto stream = engine.CreateCPUStream();
    engine.Submit(stream, [read = std::move(read), numeric = std::move(numeric), &result](auto) mutable
                  {
                      result = read->Block("species")(1, 13);
                      read.reset();
                      numeric.reset(); // Last numeric owners may retire on a worker.
                  });
    stream.Record().WaitHost();
    CHECK(result == Value(mpi.rank, 1, 20));
}

TEST_CASE("Coupled matrix action ILU SGS and left GMRES preserve separate and permuted logical equation fields")
{
    Fixture fixture;
    const auto schema = Schema(5);
    constexpr I nv = 21;
    const auto diagonal = [nv](I row)
    {
        Eigen::MatrixXd value(nv, nv);
        for (I j = 0; j < nv; ++j)
            for (I i = 0; i < nv; ++i)
                value(i, j) = (i == j ? 4 + .001 * row + .002 * i : 0) + .001 * (i + 1) * (j + 1) / double(nv * nv);
        return value;
    };
    const auto off = [nv](I row, I column)
    {
        Eigen::MatrixXd value(nv, nv);
        for (I j = 0; j < nv; ++j)
            for (I i = 0; i < nv; ++i)
                value(i, j) = (i == j ? -.03 - .00001 * row : 0) + .0001 * (i + 1) * (column + j + 1) / double(nv * nv);
        return value;
    };
    const auto vector = [nv](int owner, I row)
    {
        Eigen::VectorXd value(nv);
        for (I j = 0; j < nv; ++j)
            value[j] = Value(owner, row, j);
        return value;
    };
    std::vector<std::vector<I>> columns(size_t(fixture.rows));
    std::vector<I> core;
    for (I iRow = 0; iRow < fixture.rows; ++iRow)
    {
        if (iRow > 0)
            columns[iRow].push_back(iRow - 1);
        if (iRow + 1 < fixture.rows)
            columns[iRow].push_back(iRow + 1);
        if (iRow + 2 >= fixture.rows)
            columns[iRow].push_back(fixture.rows + iRow - (fixture.rows - 2));
        else
            core.push_back(iRow);
    }
    auto problem = std::make_shared<C::LinearProblemContext>(fixture.space, nv, columns, core, C::DiagonalStorage::Block);
    {
        auto d = problem->Diagonal().Write();
        auto b = problem->OffDiagonal().Write();
        const auto matrices = A::MakeEigenRowMatrices(d.View<R::TableRows>(), nv, nv);
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            matrices[iRow] = diagonal(iRow);
            for (I j = 0; j < I(columns[iRow].size()); ++j)
                b.View()(iRow, j) = off(iRow, columns[iRow][j]);
        }
    }
    problem->PrepareILUStructure(-1, false);
    problem->FactorizeILU();
    problem->InvertDiagonal();
    for (int layout = 0; layout < 3; ++layout)
    {
        auto state = fixture.Build(schema, layout, layout != 0);
        if (layout == 2)
        {
            A::DofBundle fields(fixture.space);
            fields.Add("packed", A::DofField<double>(fixture.space, Layout(nv, 2)));
            state = C::EquationDofs(schema, std::move(fields), {{"flow", "packed", 14}, {"rans", "packed", 19}, {"species", "packed", 0}});
        }
        Fill(state);
        auto result = fixture.Build(schema, (layout + 1) % 3, true);
        auto rhs = C::EquationDofOperations::NewVector(result), localRhs = C::EquationDofOperations::NewVector(result);
        auto forward = C::EquationDofOperations::NewVector(result), solved = C::EquationDofOperations::NewVector(result);
        std::vector<R::TransferBinding> halo;
        for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            halo.push_back(fixture.Bind(state.PhysicalField(iField)));
        std::vector<R::TransferOperation> operations;
        for (auto &binding : halo)
            operations.push_back(binding.Begin());
        problem->Apply(state, result, C::LinearRegion::Core);
        for (auto &operation : operations)
            operation.Wait();
        problem->Apply(state, result, C::LinearRegion::HaloDependent);
        {
            const auto read = result.ReadOwned();
            const auto layoutRead = read.ComponentLayout();
            const auto values = layoutRead.View();
            auto write = rhs.WriteOwned(), localWrite = localRhs.WriteOwned();
            const auto rhsLayout = write.ComponentLayout(), localLayout = localWrite.ComponentLayout();
            const auto b = rhsLayout.View(), local = localLayout.View();
            for (I iRow = 0; iRow < fixture.rows; ++iRow)
            {
                Eigen::VectorXd expected = diagonal(iRow) * vector(mpi.rank, iRow);
                Eigen::VectorXd localExpected = expected;
                for (const auto column : columns[iRow])
                {
                    if (column < fixture.rows)
                    {
                        const Eigen::VectorXd addition = off(iRow, column) * vector(mpi.rank, column);
                        expected += addition;
                        localExpected += addition;
                    }
                    else
                    {
                        const int previous = (mpi.rank + fixture.active - 1) % fixture.active;
                        expected.noalias() += off(iRow, column) * vector(previous, fixture.rows - 2 + column - fixture.rows);
                    }
                }
                for (I j = 0; j < nv; ++j)
                {
                    CHECK(values(iRow, j) == doctest::Approx(expected[j]).epsilon(2e-13));
                    b(iRow, j) = expected[j];
                    local(iRow, j) = localExpected[j];
                }
            }
        }
        problem->ApplyILU(localRhs, forward, solved);
        {
            const auto read = solved.ReadOwned();
            const auto view = read.ComponentLayout();
            for (I iRow = 0; iRow < fixture.rows; ++iRow)
                for (I j = 0; j < nv; ++j)
                    CHECK(view.View()(iRow, j) == doctest::Approx(Value(mpi.rank, iRow, j)).epsilon(2e-12));
        }
        // A non-solution old iterate makes ordering/remote-state choices
        // observable. Independent dense row solves do not use the SGS kernel.
        auto old = state.Clone();
        C::EquationDofOperations::Scale(old, .2);
        std::vector<R::TransferBinding> oldHalo, forwardHalo;
        for (I iField = 0; iField < old.PhysicalFieldNumber(); ++iField)
            oldHalo.push_back(fixture.Bind(old.PhysicalField(iField)));
        for (I iField = 0; iField < forward.PhysicalFieldNumber(); ++iField)
            forwardHalo.push_back(fixture.Bind(forward.PhysicalField(iField)));
        const auto rightAt = [&](int owner, I row)
        {
            Eigen::VectorXd b = diagonal(row) * vector(owner, row);
            for (const auto column : columns[row])
            {
                const int peer = column < fixture.rows ? owner : (owner + fixture.active - 1) % fixture.active;
                b.noalias() += off(row, column) * vector(peer, column < fixture.rows ? column : fixture.rows - 2 + column - fixture.rows);
            }
            return b;
        };
        const auto lowerAt = [&](int owner)
        {
            Eigen::MatrixXd y(fixture.rows, nv);
            for (I row = 0; row < fixture.rows; ++row)
            {
                Eigen::VectorXd value = rightAt(owner, row);
                for (const auto column : columns[row])
                {
                    Eigen::VectorXd input(nv);
                    if (column < row)
                        input = y.row(column).transpose();
                    else if (column < fixture.rows)
                        input = .2 * vector(owner, column);
                    else
                        input = .2 * vector((owner + fixture.active - 1) % fixture.active, fixture.rows - 2 + column - fixture.rows);
                    value.noalias() -= off(row, column) * input;
                }
                y.row(row) = diagonal(row).partialPivLu().solve(value).transpose();
            }
            return y;
        };
        for (bool middle : {false, true})
        {
            std::vector<R::TransferOperation> active;
            for (auto &binding : oldHalo)
                active.push_back(binding.Begin());
            auto sweep = problem->BeginSGS(rhs, old, forward, solved, middle);
            sweep.Lower(C::LinearRegion::Core);
            for (auto &operation : active)
                operation.Wait();
            sweep.Lower(C::LinearRegion::HaloDependent);
            if (middle)
            {
                active.clear();
                for (auto &binding : forwardHalo)
                    active.push_back(binding.Begin());
                for (auto &operation : active)
                    operation.Wait();
            }
            sweep.Upper(C::LinearRegion::HaloDependent);
            sweep.Upper(C::LinearRegion::Core);
            sweep.Finish();
            const auto lower = lowerAt(mpi.rank);
            const auto remote = lowerAt((mpi.rank + fixture.active - 1) % fixture.active);
            Eigen::MatrixXd expected(fixture.rows, nv);
            for (I row = fixture.rows; row > 0;)
            {
                --row;
                Eigen::VectorXd value = rightAt(mpi.rank, row);
                for (const auto column : columns[row])
                {
                    Eigen::VectorXd input(nv);
                    if (column < row)
                        input = lower.row(column).transpose();
                    else if (column < fixture.rows)
                        input = expected.row(column).transpose();
                    else if (middle)
                        input = remote.row(fixture.rows - 2 + column - fixture.rows).transpose();
                    else
                        input = .2 * vector((mpi.rank + fixture.active - 1) % fixture.active, fixture.rows - 2 + column - fixture.rows);
                    value.noalias() -= off(row, column) * input;
                }
                expected.row(row) = diagonal(row).partialPivLu().solve(value).transpose();
            }
            const auto read = solved.ReadOwned();
            const auto mapping = read.ComponentLayout();
            const auto output = mapping.View();
            for (I iRow = 0; iRow < fixture.rows; ++iRow)
                for (I j = 0; j < nv; ++j)
                    CHECK(output(iRow, j) == doctest::Approx(expected(iRow, j)).epsilon(2e-12));
        }
        // Exercise the production controller, including every physical field's
        // canonical halo. The independent dense row oracles above and the known
        // exact solution remain separate from its matrix/SGS implementation.
        for (int scenario = 0; scenario < 6; ++scenario)
        {
            C::LinearSolveSettings controls;
            controls.method = scenario < 2 ? "sgs" : scenario == 5 ? "ilu"
                                                                   : "gmres";
            controls.preconditioner = scenario < 3 ? "sgs" : "ilu";
            controls.sgsSweeps = 2;
            controls.exchangeBetweenSweeps = scenario == 1;
            controls.iluFill = scenario == 4 ? -1 : 0;
            controls.subspace = 8;
            controls.maxIterations = 40;
            controls.relativeTolerance = 1e-12;
            controls.absoluteTolerance = 1e-10;
            C::DofLinearSolver solver(problem, state, [&](auto &field)
                                      { return fixture.Bind(field); },
                                      controls, {scenario != 0 && scenario != 3, scenario != 2});
            const auto report = solver.Solve(rhs);
            CHECK(report.converged);
            CHECK(report.usable);
            CHECK(report.iterations > 0);
            {
                const auto read = solver.Solution().ReadOwned();
                const auto mapping = read.ComponentLayout();
                const auto values = mapping.View();
                for (I iRow = 0; iRow < fixture.rows; ++iRow)
                    for (I j = 0; j < nv; ++j)
                        CHECK(values(iRow, j) == doctest::Approx(Value(mpi.rank, iRow, j)).epsilon(2e-10));
            }
            if (scenario == 2)
            {
                const auto again = solver.Solve(rhs);
                CHECK(again.converged);
                CHECK(again.iterations == report.iterations);
                const auto warmGuess = solver.Solution();
                const auto warm = solver.Solve(rhs, std::cref(warmGuess));
                CHECK(warm.converged);
                CHECK(warm.iterations == 0);
            }
        }
        if (mpi.rank == 0)
        {
            const auto generation = forward.PhysicalField(0).Owned().Version();
            const auto outputGeneration = solved.PhysicalField(0).Owned().Version();
            auto pin = solved.PhysicalField(solved.PhysicalFieldNumber() - 1).Owned().Read();
            CHECK_THROWS(problem->ApplyILU(localRhs, forward, solved));
            CHECK(forward.PhysicalField(0).Owned().Version() == generation);
            CHECK(solved.PhysicalField(0).Owned().Version() == outputGeneration);
        }
    }
}
TEST_CASE("Logical linear driver rejects inconsistent physical binding before any halo setup")
{
    Fixture fixture;
    const auto schema = Schema(5);
    constexpr I nv = 21;
    std::vector<std::vector<I>> columns(size_t(fixture.rows));
    std::vector<I> core(size_t(fixture.rows));
    std::iota(core.begin(), core.end(), I(0));
    auto problem = std::make_shared<C::LinearProblemContext>(fixture.space, nv, columns, core, C::DiagonalStorage::Entries);
    {
        auto write = problem->Diagonal().Write();
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
            for (I j = 0; j < nv; ++j)
                write.View()(iRow, j) = 1;
    }
    int bindings = 0;
    const auto bind = [&](auto &field)
    { ++bindings; return fixture.Bind(field); };
    if (mpi.size > 1)
    {
        auto inconsistent = fixture.Build(schema, mpi.rank % 3, mpi.rank == 0);
        CHECK_THROWS(C::DofLinearSolver(problem, inconsistent, bind));
        CHECK(bindings == 0);
    }
    auto state = fixture.Build(schema, mpi.rank % 3, true);
    Fill(state);
    C::DofLinearSolver solver(problem, state, bind);
    // One critical runtime conflict: the last field is busy on one rank.
    // Entry rejection must leave every output reusable and unchanged.
    std::optional<R::Array<double>::WriteLease> writer;
    if (mpi.rank == 0)
        writer.emplace(state.PhysicalField(state.PhysicalFieldNumber() - 1).Owned().Write());
    const auto version = solver.Solution().PhysicalField(0).Owned().Version();
    CHECK_THROWS(solver.Solve(state));
    CHECK(solver.Solution().PhysicalField(0).Owned().Version() == version);
    writer.reset();
    CHECK(solver.Solve(state).converged);
}
TEST_CASE("Logical scalar-diagonal SGS handles all blocks and fails an abandoned partial output")
{
    const A::DofSpace space(3, 0, "scalar logical SGS");
    const auto schema = Schema(4);
    A::DofBundle fields(space);
    for (const auto &block : schema.blocks)
        fields.Add(block.name, A::DofField<double>(space, Layout(I(block.components.size()), 1)));
    auto rhs = C::EquationDofs::Separate(schema, std::move(fields));
    Fill(rhs);
    auto old = rhs.Clone(), forward = rhs.Clone(), output = rhs.Clone();
    C::LinearProblemContext problem(space, rhs.ComponentNumber(), std::vector<std::vector<I>>(3), {0, 1, 2}, C::DiagonalStorage::Entries);
    {
        auto write = problem.Diagonal().Write();
        const auto d = write.View<R::TableRows>();
        for (I iRow = 0; iRow < 3; ++iRow)
            for (I j = 0; j < rhs.ComponentNumber(); ++j)
                d(iRow, j) = 2 + .1 * iRow + .01 * j;
    }
    problem.InvertDiagonal();
    problem.ApplySGS(rhs, old, forward, output);
    {
        const auto read = output.ReadOwned();
        const auto map = read.ComponentLayout();
        const auto values = map.View();
        for (I iRow = 0; iRow < 3; ++iRow)
            for (I j = 0; j < rhs.ComponentNumber(); ++j)
                CHECK(values(iRow, j) == doctest::Approx(Value(mpi.rank, iRow, j) / (2 + .1 * iRow + .01 * j)).epsilon(1e-14));
    }
    {
        auto unfinished = problem.BeginSGS(rhs, old, forward, output);
        unfinished.Lower(C::LinearRegion::Core);
    }
    CHECK_THROWS(output.PhysicalField(0).Owned().Read());
}

TEST_CASE("The same BDF and coupled block solve interface advances all flow RANS and species components")
{
    Fixture fixture;
    for (int flow : {4, 5})
        for (int layout = 0; layout < 3; ++layout)
        {
            auto state = fixture.Build(Schema(flow), layout, false);
            Fill(state);
            auto trial = state.Clone();
            const I nv = state.ComponentNumber();
            std::vector<double> a(size_t(nv), 0.);
            double square = 0;
            for (I j = 0; j < nv; ++j)
            {
                a[j] = double(j + 1) / double(nv);
                square += a[j] * a[j];
            }
            DNDS::ODE::FieldTimeStepper stepper(state.CoupledField(), "bdf1");
            I solves = 0;
            const auto solve = [&](auto &candidate, const auto &source, double mass, double scale, double, int)
            {
                ++solves;
                auto problem = std::make_shared<C::LinearProblemContext>(fixture.space, nv, std::vector<std::vector<I>>(size_t(fixture.rows)),
                                                                         std::vector<I>{}, C::DiagonalStorage::Block);
                {
                    auto diagonal = problem->Diagonal().Write();
                    const auto values = diagonal.View<R::UniformTableRows>();
                    for (I iRow = 0; iRow < fixture.rows; ++iRow)
                        for (I j = 0; j < nv; ++j)
                            for (I i = 0; i < nv; ++i)
                                values(iRow, i + nv * j) = (i == j ? mass + scale : 0) + scale * a[i] * a[j];
                }
                problem->InvertDiagonal();
                C::LinearSolveSettings control;
                control.maxIterations = 1;
                control.relativeTolerance = 1e-12;
                control.absoluteTolerance = 1e-10;
                C::LinearSolver solver(problem, source, [&](auto &field)
                                       { return fixture.Bind(field); }, control);
                const auto report = solver.Solve(source);
                CHECK(report.converged);
                CHECK(report.iterations == 1);
                A::Copy(candidate, solver.Solution());
                return report.converged;
            };
            // F(U)=-U-a(a.U), coupling every logical segment. BDF1 does not
            // need a separate explicit rate call; its stage solver sees all nv.
            const auto evaluate = [](auto &, auto &, double, int)
            { FAIL("BDF1 must not request an explicit rate"); };
            constexpr double dt = .025;
            for (int step = 0; step < 2; ++step)
            {
                stepper.BeginStep(state.CoupledField(), trial.CoupledField(),
                                  step * dt, dt, evaluate);
                const auto group = stepper.PrepareGroup(trial.CoupledField());
                const auto equation = stepper.PrepareEquation(group, trial.CoupledField(), 0);
                const auto &c = equation.coefficients;
                REQUIRE(solve(equation.state, equation.resOther, c.inverseDt,
                              c.rhsScale, c.time, group.stage));
                stepper.CompleteGroup(group, trial.CoupledField(), evaluate);
                {
                    const auto initial = state.ReadOwned(), result = trial.ReadOwned();
                    for (I iRow = 0; iRow < fixture.rows; ++iRow)
                    {
                        double dot = 0;
                        I first = 0;
                        for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                        {
                            const auto values = initial.Block(iBlock).owned;
                            for (I j = 0; j < values.ComponentNumber(); ++j)
                                dot += a[first + j] * values(iRow, j);
                            first += values.ComponentNumber();
                        }
                        first = 0;
                        for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                        {
                            const auto x = initial.Block(iBlock).owned, y = result.Block(iBlock).owned;
                            for (I j = 0; j < x.ComponentNumber(); ++j)
                            {
                                // Independent Sherman-Morrison oracle, not the
                                // native block inverse or the stage-solver result.
                                const double expected = x(iRow, j) / (1 + dt) - dt * a[first + j] * dot / ((1 + dt) * (1 + dt + dt * square));
                                CHECK(y(iRow, j) == doctest::Approx(expected).epsilon(2e-12));
                            }
                            first += x.ComponentNumber();
                        }
                    }
                }
                // Immediate algebra requires idle inputs; the engine-aware
                // variants support compatible active readers.
                A::Copy(state.CoupledField(), trial.CoupledField());
                stepper.Commit();
            }
            CHECK(solves == 2);
        }
}
TEST_CASE("One time driver advances coupled and separate equation storage with identical coupled stage mathematics")
{
    Fixture fixture;
    constexpr I nv = 21;
    std::array<double, nv> a{};
    double square = 0;
    for (I j = 0; j < nv; ++j)
    {
        a[j] = double(j + 1) / nv;
        square += a[j] * a[j];
    }
    const auto evaluate = [&](C::EquationDofs &state, C::EquationDofs &rate, double, int)
    {
        const auto input = state.ReadOwned();
        auto output = rate.WriteOwned();
        std::array<C::EquationBlockArrayView<const double>, 3> u;
        std::array<C::EquationBlockArrayView<double>, 3> f;
        const I offsets[]{0, 5, 7};
        for (I iBlock = 0; iBlock < 3; ++iBlock)
        {
            u[iBlock] = input.Block(iBlock).owned;
            f[iBlock] = output.Block(iBlock);
        }
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            double projection = 0;
            for (I iBlock = 0; iBlock < 3; ++iBlock)
                for (I j = 0; j < u[iBlock].ComponentNumber(); ++j)
                    projection += a[offsets[iBlock] + j] * u[iBlock](iRow, j);
            for (I iBlock = 0; iBlock < 3; ++iBlock)
                for (I j = 0; j < u[iBlock].ComponentNumber(); ++j)
                    f[iBlock](iRow, j) = -u[iBlock](iRow, j) - a[offsets[iBlock] + j] * projection;
        }
    };
    const auto solve = [&](C::EquationDofs &state, const C::EquationDofs &source, double mass, double scale, double, int)
    {
        const auto input = source.ReadOwned();
        auto output = state.WriteOwned();
        std::array<C::EquationBlockArrayView<const double>, 3> b;
        std::array<C::EquationBlockArrayView<double>, 3> u;
        const I offsets[]{0, 5, 7};
        for (I iBlock = 0; iBlock < 3; ++iBlock)
        {
            b[iBlock] = input.Block(iBlock).owned;
            u[iBlock] = output.Block(iBlock);
        }
        const double diagonal = mass + scale;
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            double projection = 0;
            for (I iBlock = 0; iBlock < 3; ++iBlock)
                for (I j = 0; j < b[iBlock].ComponentNumber(); ++j)
                    projection += a[offsets[iBlock] + j] * b[iBlock](iRow, j);
            for (I iBlock = 0; iBlock < 3; ++iBlock)
                for (I j = 0; j < b[iBlock].ComponentNumber(); ++j)
                    u[iBlock](iRow, j) = b[iBlock](iRow, j) / diagonal - scale * a[offsets[iBlock] + j] * projection / (diagonal * (diagonal + scale * square));
        }
        return true;
    };
    using Field = A::DofField<double>;
    const auto nativeRate = [&](Field &state, Field &rate, double, int)
    {
        const auto read = state.Owned().Read();
        auto write = rate.Owned().Write();
        const auto u = read.View<R::UniformTableRows>();
        const auto f = write.View<R::UniformTableRows>();
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            double projection = 0;
            for (I j = 0; j < nv; ++j)
                projection += a[j] * u(iRow, j);
            for (I j = 0; j < nv; ++j)
                f(iRow, j) = -u(iRow, j) - a[j] * projection;
        }
    };
    const auto nativeSolve = [&](Field &state, const Field &source, double mass, double scale, double, int)
    {
        const auto read = source.Owned().Read();
        auto write = state.Owned().Write();
        const auto b = read.View<R::UniformTableRows>();
        const auto u = write.View<R::UniformTableRows>();
        const double diagonal = mass + scale;
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            double projection = 0;
            for (I j = 0; j < nv; ++j)
                projection += a[j] * b(iRow, j);
            for (I j = 0; j < nv; ++j)
                u(iRow, j) = b(iRow, j) / diagonal - scale * a[j] * projection / (diagonal * (diagonal + scale * square));
        }
        return true;
    };
    const auto converged = [&](const C::EquationDofs &residual)
    {
        const auto read = residual.ReadOwned();
        double error = 0;
        for (I iBlock = 0; iBlock < 3; ++iBlock)
        {
            const auto values = read.Block(iBlock).owned;
            for (I iRow = 0; iRow < fixture.rows; ++iRow)
                for (I j = 0; j < values.ComponentNumber(); ++j)
                    error = std::max(error, std::abs(values(iRow, j)));
        }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        return error < 1e-8;
    };
    const auto nativeConverged = [&](const Field &residual)
    {
        const auto read = residual.Owned().Read();
        double error = 0;
        const auto values = read.View<R::UniformTableRows>();
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
            for (I j = 0; j < nv; ++j)
                error = std::max(error, std::abs(values(iRow, j)));
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        return error < 1e-8;
    };
    const auto coupled = [&](auto &definition, const auto &group, auto &endpoint, const auto &rate, const auto &update,
                             const auto &operations, const auto &stopping)
    {
        auto residual = operations.NewVector(endpoint);
        // One shared iteration visits both unknowns once, then observes the
        // method-provided I+H defect. No nested full stage nonlinear solves.
        for (int iteration = 0; iteration < 200; ++iteration)
        {
            for (int coupledStage = 0; coupledStage < 2; ++coupledStage)
            {
                const auto equation = definition.PrepareEquation(group, endpoint, coupledStage);
                const auto &c = equation.coefficients;
                if (!update(equation.state, equation.resOther, c.inverseDt, c.rhsScale, c.time, coupledStage))
                    return false;
                definition.CompleteEquation(group, endpoint, rate);
            }
            definition.AssembleStoppingResidual(group, endpoint, residual);
            if (stopping(residual))
                return true;
        }
        return false;
    };
    const auto prepare = [&](auto &driver, const auto &initial, auto &trial,
                             double time, double dt, const auto &rate,
                             const auto &update, const auto &operations,
                             const auto &stopping)
    {
        driver.BeginStep(initial, trial, time, dt, rate);
        for (int stage = 0; stage < driver.GroupNumber(); ++stage)
        {
            const auto group = driver.PrepareGroup(trial);
            if (group.coupledStages == 2)
            {
                if (!coupled(driver, group, trial, rate, update, operations, stopping))
                    return false;
            }
            else if (group.implicit)
            {
                const auto equation = driver.PrepareEquation(group, trial, 0);
                const auto &c = equation.coefficients;
                if (!update(equation.state, equation.resOther, c.inverseDt,
                            c.rhsScale, c.time, group.stage))
                    return false;
            }
            driver.CompleteGroup(group, trial, rate);
        }
        return true;
    };
    for (const std::string method : {"bdf1", "bdf2", "trapz", "esdirk2", "esdirk3", "esdirk4", "ditr"})
    {
        INFO(method);
        A::DofBundle fields(fixture.space);
        fields.Add("flow", Field(fixture.space, Layout(5, 0)));
        fields.Add("rans", Field(fixture.space, Layout(2, 2)));
        fields.Add("species", Field(fixture.space, Layout(14, 1)));
        auto state = C::EquationDofs::Separate(Schema(5), std::move(fields));
        Fill(state);
        auto candidate = C::EquationDofOperations::NewVector(state);
        auto reference = fixture.Build(Schema(5), 0, false);
        Fill(reference);
        auto nativeCandidate = DNDS::ODE::FieldTimeOperations::NewVector(reference.CoupledField());
        C::EquationTimeStepper driver(state, method);
        DNDS::ODE::FieldTimeStepper oracle(reference.CoupledField(), method);
        const auto compare = [&]
        {
            const auto read = candidate.ReadOwned();
            const auto flat = nativeCandidate.Owned().Read();
            const auto values = flat.View<R::UniformTableRows>();
            I first = 0;
            for (I iBlock = 0; iBlock < 3; ++iBlock)
            {
                const auto block = read.Block(iBlock).owned;
                for (I iRow = 0; iRow < fixture.rows; ++iRow)
                    for (I j = 0; j < block.ComponentNumber(); ++j)
                        CHECK(block(iRow, j) == values(iRow, first + j));
                first += block.ComponentNumber();
            }
        };
        double time = 0;
        for (double dt : {.02, .03})
        {
            REQUIRE(prepare(driver, state, candidate, time, dt, evaluate, solve,
                            C::EquationDofOperations{}, converged));
            REQUIRE(prepare(oracle, reference.CoupledField(), nativeCandidate,
                            time, dt, nativeRate, nativeSolve,
                            DNDS::ODE::FieldTimeOperations{}, nativeConverged));
            compare();
            C::EquationDofOperations::Copy(state, candidate);
            driver.Commit();
            A::Copy(reference.CoupledField(), nativeCandidate);
            oracle.Commit();
            // Only BDF owns previous-step history; DIRK/DITR are one-step
            // methods and deliberately expose no previous-step dt cache.
            CHECK(driver.PreviousDt() == (method == "bdf1" || method == "bdf2" ? dt : 0));
            time += dt;
        }
        if (method == "bdf2")
        {
            auto restored = driver.Fresh();
            restored->RestoreHistory(driver.PreviousState(), driver.PreviousDt());
            CHECK(restored->HasHistory());
            const auto before = state.ReadOwned();
            REQUIRE(prepare(*restored, state, candidate, time, .025, evaluate,
                            solve, C::EquationDofOperations{}, converged));
            REQUIRE(prepare(oracle, reference.CoupledField(), nativeCandidate,
                            time, .025, nativeRate, nativeSolve,
                            DNDS::ODE::FieldTimeOperations{}, nativeConverged));
            compare();
            if (fixture.rows)
                CHECK(before.Block("species")(0, 0) == state.ReadOwned().Block("species")(0, 0));
        }
    }
}
TEST_CASE("Queued equation algebra retains temporaries and preserves logical blocks across layouts")
{
    Fixture fixture;
    for (int workers : {0, 2})
        for (int layout : {0, 1, 2})
            for (bool split : {false, true})
            {
                CAPTURE(workers);
                CAPTURE(layout);
                CAPTURE(split);
                auto state = fixture.Build(Schema(5), layout, split);
                auto output = fixture.Build(Schema(5), (layout + 1) % 3, !split);
                auto result = C::EquationDofOperations::NewVector(output);
                Fill(state);
                R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
                const auto produce = engine.CreateCPUStream(), consume = engine.CreateCPUStream();
                {
                    auto temporary = C::EquationDofOperations::NewVector(state);
                    C::EquationDofOperations::Copy(engine, produce, temporary, state);
                    C::EquationDofOperations::Scale(engine, produce, temporary, 3);
                    consume.WaitFor(produce.Record());
                    C::EquationDofOperations::Fill(engine, consume, output, -2);
                    C::EquationDofOperations::Axpy(engine, consume, output, 1, temporary);
                    C::EquationDofOperations::LinearCombination(engine, consume, result, .5, output, .5, output);
                    // No intermediate drain and no surviving caller handle to
                    // temporary: the submitted operands retain its allocations.
                }
                engine.Synchronize();
                CheckValues(state);
                CheckValues(output, 3, -2);
                CheckValues(result, 3, -2);

                // A late-block conflict must reject before any earlier output
                // field is modified. One representative access failure is enough.
                if (workers == 0 && layout == 0 && !split)
                {
                    const auto held = output.PhysicalField(2).Owned().Read();
                    CHECK_THROWS(C::EquationDofOperations::Copy(engine, consume, output, state));
                }
                CheckValues(output, 3, -2);
            }
}
TEST_CASE("Equation algebra reserves every output before writing and time sources cannot be silently mutated")
{
    Fixture fixture;
    auto state = fixture.Build(Schema(5), 2, true);
    Fill(state);
    auto candidate = C::EquationDofOperations::NewVector(state);
    C::EquationDofOperations::Fill(candidate, -7);
    std::vector<uint64_t> versions;
    for (I iField = 0; iField < candidate.PhysicalFieldNumber(); ++iField)
        versions.push_back(candidate.PhysicalField(iField).Owned().Version());
    {
        const auto held = candidate.PhysicalField(2).Owned().Read();
        CHECK_THROWS(C::EquationDofOperations::Copy(candidate, state));
    }
    for (I iField = 0; iField < candidate.PhysicalFieldNumber(); ++iField)
    {
        CHECK(candidate.PhysicalField(iField).Owned().Version() == versions[iField]);
        const auto read = candidate.PhysicalField(iField).Owned().Read();
        const auto values = read.View<R::UniformTableRows>();
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
            for (I j = 0; j < values.Width(); ++j)
                CHECK(values(iRow, j) == -7);
    }
    C::EquationTimeStepper driver(state, "bdf1");
    driver.BeginStep(state, candidate, 0, .01);
    const auto group = driver.PrepareGroup(candidate);
    const auto equation = driver.PrepareEquation(group, candidate, 0);
    {
        auto write = const_cast<C::EquationDofs &>(equation.resOther).WriteOwned();
        if (fixture.rows)
            write.Block("species")(0, 0) += 1;
    }
    CHECK_THROWS(driver.CompleteGroup(group, candidate));
    CHECK_FALSE(driver.HasPendingTrial());
    CHECK_FALSE(driver.HasHistory());
    CheckValues(state);
}
int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    doctest::Context context(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
