#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Connectivity/ConnectivitySetupRows.hpp"
#include "NativeMeshOverlapFixture.hpp"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include <set>
#include <filesystem>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    DNDS::MPIInfo MPI()
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        return mpi;
    }
    template <class T>
    R::Array<T> Rows(const std::vector<std::vector<T>> &rows)
    {
        R::RowBuilder builder(rows.size(), sizeof(T));
        for (size_t row = 0; row < rows.size(); ++row)
        {
            builder.ResizeRow(row, rows[row].size());
            for (size_t slot = 0; slot < rows[row].size(); ++slot)
                builder.Row<T>(row)[slot] = rows[row][slot];
        }
        return R::Array<T>::FromSharedStorage(builder.Commit());
    }
    struct RelationFixture
    {
        DNDS::MPIInfo mpi;
        G::MeshAdjacency cellNodes;
        G::EntitySpace Entities(G::EntityKind kind) const
        {
            return kind == G::EntityKind::Cell ? cellNodes.Source() : cellNodes.Target();
        }
    };
    std::unique_ptr<RelationFixture> Fixture(bool periodic = false, bool emptyRanks = false)
    {
        const auto mpi = MPI();
        const I count = !emptyRanks || mpi.rank == 0 ? 1 : 0;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, count * (periodic ? 2 : 4));
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, count);
        std::vector<std::vector<I>> rows;
        if (count)
        {
            const I offset = nodes.LocalToGlobal(0);
            rows.push_back({offset, offset + (periodic ? 0 : 1), offset + (periodic ? 1 : 2), offset + (periodic ? 1 : 3)});
        }
        return std::make_unique<RelationFixture>(RelationFixture{mpi, G::MeshAdjacency(G::Adj::Cell2Node, cells, nodes, Rows(rows))});
    }
    G::SubEntityQueryPbi QuadFaces()
    {
        G::SubEntityQueryPbi query;
        query.numSubEntities = [](I)
        { return 4; };
        query.describe = [](I, int)
        { return G::SubEntityDesc{2, 2, G::t_index(G::Elem::Line2)}; };
        query.extractNodes = [](I, int sub, const std::function<I(int)> &nodes, I *out)
        { out[0] = nodes(sub); out[1] = nodes((sub + 1) % 4); };
        query.extractPbi = [](I, int sub, const std::function<G::NodePeriodicBits(int)> &bits, G::NodePeriodicBits *out)
        { out[0] = bits(sub); out[1] = bits((sub + 1) % 4); };
        return query;
    }
    G::OwnershipResolverMulti Owner()
    {
        return [](const std::vector<I> &, const std::vector<DNDS::MPI_int> &ranks, I)
        {
            auto mpi = MPI();
            return G::OwnershipDecision{*std::min_element(ranks.begin(), ranks.end()) == mpi.rank, {}};
        };
    }
}

TEST_CASE("Native interpolation retains ordered cones and periodic self incidences with empty ranks")
{
    for (bool periodic : {false, true})
        for (bool emptyRanks : {false, true})
        {
            auto mesh = Fixture(periodic, emptyRanks);
            auto cells = mesh->Entities(G::EntityKind::Cell), nodes = mesh->Entities(G::EntityKind::Node);
            auto original = mesh->cellNodes.Read();
            std::vector<std::vector<I>> values;
            for (I row = 0; row < original.NumOwned(); ++row)
                values.push_back(original.GlobalRow(row));
            G::MeshAdjacency native(G::Adj::Cell2Node, cells, nodes, Rows(values));
            auto read = native.Read();
            std::optional<G::RelationPeriodicBits> bits;
            if (periodic)
            {
                std::vector<std::vector<G::NodePeriodicBits>> pbi(size_t(cells.NumOwned()),
                                                                  {G::NodePeriodicBits{0}, G::NodePeriodicBits{1}, G::NodePeriodicBits{1}, G::NodePeriodicBits{0}});
                bits.emplace(read, Rows(pbi));
            }
            auto result = G::AdjacencyRegistry::Interpolate(read, G::EntityKind::Face, QuadFaces(), Owner(), bits ? &*bits : nullptr);
            auto oracle = G::AdjacencyRegistry::Interpolate(original, G::EntityKind::Face, QuadFaces(), Owner(), bits ? &*bits : nullptr);
            CHECK(result.entities.NumOwned() == cells.NumOwned() * (periodic ? 3 : 4));
            CHECK(result.entities.NumGlobal() == (emptyRanks ? 1 : mesh->mpi.size) * (periodic ? 3 : 4));
            auto p2e = result.parent2entity.Read();
            auto e2n = result.entity2node.Read();
            auto e2p = result.entity2parent.Read();
            for (I row = 0; row < cells.NumOwned(); ++row)
            {
                CHECK(p2e.GlobalRow(row) == oracle.parent2entity.Read().GlobalRow(row));
                CHECK(p2e.RowSize(row) == 4);
                for (int slot = 0; slot < 4; ++slot)
                {
                    const auto entity = *result.entities.GlobalToLocal(p2e.Raw(row, size_t(slot)));
                    const int origin = periodic && slot == 3 ? 1 : slot;
                    CHECK(e2n.GlobalRow(entity) == std::vector<I>{values[size_t(row)][size_t(origin)], values[size_t(row)][size_t((origin + 1) % 4)]});
                }
                if (periodic)
                {
                    CHECK(p2e.Raw(row, 1) == p2e.Raw(row, 3));
                    const auto self = *result.entities.GlobalToLocal(p2e.Raw(row, 1));
                    CHECK(e2p.GlobalRow(self) == std::vector<I>{cells.LocalToGlobal(row), cells.LocalToGlobal(row)});
                    auto pbi = result.parent2entityPbi->Read();
                    CHECK(uint8_t(pbi.View()(size_t(row), 1)) == 0);
                    CHECK(uint8_t(pbi.View()(size_t(row), 3)) == 1);
                }
            }
            CHECK(result.entities.Valid());
            // Output numbering owns its controller rather than depending on
            // the local result object's temporary construction scope.
            auto retained = result.entities;
            CHECK(retained.Valid());
        }
}

TEST_CASE("Native GhostSpec discovers multilayer remote rows and publishes shared routes")
{
    auto mesh = Fixture();
    auto mpi = mesh->mpi;
    auto cells = mesh->Entities(G::EntityKind::Cell), nodes = mesh->Entities(G::EntityKind::Node);
    const I next = (mpi.rank + 1) % mpi.size;
    std::vector<std::vector<I>> neighbors{{next, DNDS::UnInitIndex}};
    G::MeshAdjacency c2c(G::Adj::Cell2Cell, cells, cells, Rows(neighbors));
    auto c2n = mesh->cellNodes.Read();
    G::AdjacencyRegistry setup(mpi);
    setup.Register(c2c.Read());
    setup.Register(std::move(c2n));
    G::GhostSpec spec{{{G::EntityKind::Cell,
                        {G::Adj::Cell2Cell, G::Adj::Cell2Cell, G::Adj::Cell2Node},
                        G::EntityKind::Node},
                       {G::EntityKind::Cell, {G::Adj::Cell2Cell, G::Adj::Cell2Cell}, G::EntityKind::Cell}}};
    auto discovered = setup.Evaluate(G::CompiledGhostTree::compile(spec));
    const I second = (mpi.rank + 2) % mpi.size;
    std::vector<I> expectedCells;
    // The Cell2Cell chain ends after its second hop. The first hop is branch
    // state for the longer node chain, not an implicit endpoint collection.
    if (second != mpi.rank)
        expectedCells.push_back(second);
    auto indices = [&](G::EntityKind kind)
    {
        auto found = discovered.ghostIndices.find(kind);
        return found == discovered.ghostIndices.end() ? std::vector<I>{} : found->second;
    };
    CHECK(indices(G::EntityKind::Cell) == expectedCells);
    std::vector<I> expectedNodes;
    if (second != mpi.rank)
        for (int j = 0; j < 4; ++j)
            expectedNodes.push_back(second * 4 + j);
    CHECK(indices(G::EntityKind::Node) == expectedNodes);
    // Repeated traversal reuses immutable global payloads. A second registry
    // checks local numbering, including encoded unavailable remote targets.
    auto repeated = setup.Evaluate(G::CompiledGhostTree::compile(spec));
    CHECK(repeated.ghostIndices == discovered.ghostIndices);
    auto localCells = G::MeshAdjacency::Convert(c2c.Read(), G::Adj_PointToLocal);
    auto localNodes = G::MeshAdjacency::Convert(mesh->cellNodes.Read(), G::Adj_PointToLocal);
    G::AdjacencyRegistry localSetup(mpi);
    localSetup.Register(localCells.Read());
    localSetup.Register(localNodes.Read());
    for (int repeat = 0; repeat < 2; ++repeat)
        CHECK(localSetup.Evaluate(G::CompiledGhostTree::compile(spec)).ghostIndices == discovered.ghostIndices);
    auto publication = setup.BuildGhosted(discovered);
    // Traversal supports need not be published. The same registry/tree and
    // canonical mappings still determine the selected adjacency's ghosts.
    auto selected = setup.BuildGhosted(discovered, {G::Adj::Cell2Node});
    REQUIRE(selected.relations.size() == 1);
    CHECK(selected.spaces.at(G::EntityKind::Cell).NumGhost() == I(expectedCells.size()));
    auto selectedRead = selected.relations.at(G::Adj::Cell2Node).Read();
    auto mapsOnly = setup.BuildGhosted(discovered, {});
    CHECK(mapsOnly.relations.empty());
    CHECK(mapsOnly.spaces.at(G::EntityKind::Cell).NumGhost() == I(expectedCells.size()));
    CHECK_THROWS(setup.BuildGhosted(discovered, {G::Adj::Cell2Node, G::Adj::Cell2Node}));
    CHECK_THROWS(setup.BuildGhosted(discovered, {G::Adj::Node2Cell}));
    if (mpi.size > 1)
        CHECK_THROWS(setup.BuildGhosted(discovered,
                                        mpi.rank == 0 ? std::vector<G::AdjacencyDescriptor>{G::Adj::Cell2Node}
                                                      : std::vector<G::AdjacencyDescriptor>{}));
    auto ghosted = publication.relations.at(G::Adj::Cell2Node).Read();
    CHECK(ghosted.RowCount() == 1 + I(expectedCells.size()));
    for (size_t row = 0; row < expectedCells.size(); ++row)
    {
        std::vector<I> expected;
        for (int j = 0; j < 4; ++j)
            expected.push_back(expectedCells[row] * 4 + j);
        CHECK(ghosted.GlobalRow(I(row) + 1) == expected);
        CHECK(selectedRead.GlobalRow(I(row) + 1) == expected);
    }
    // Independently reconstructed exact expectations above replace old live
    // mesh parity. A fresh equal-shape domain must still be rejected.
    auto foreign = Fixture();
    CHECK_THROWS(setup.Register(G::MeshAdjacencyRead(G::Adj::Node2Cell, foreign->Entities(G::EntityKind::Node),
                                                     foreign->Entities(G::EntityKind::Cell), G::Adj_PointToGlobal, Rows(std::vector<std::vector<I>>(4)))));
}

TEST_CASE("Branched ghost tree keeps frontiers separate and unions endpoint kinds")
{
    auto mesh = Fixture();
    const auto mpi = mesh->mpi;
    const auto cells = mesh->Entities(G::EntityKind::Cell), nodes = mesh->Entities(G::EntityKind::Node);
    const I next = (mpi.size == 1) ? 0 : (mpi.rank + 1) % mpi.size;
    std::vector<std::vector<I>> neighbors(size_t(cells.NumOwned()));
    if (mpi.size > 1)
        for (auto &row : neighbors)
            row = {next, DNDS::UnInitIndex};
    G::MeshAdjacency c2c(G::Adj::Cell2Cell, cells, cells, Rows(neighbors));
    auto c2n = mesh->cellNodes.Read();
    auto n2c = G::AdjacencyRegistry::Inverse(c2n, mpi);
    G::AdjacencyRegistry registry(mpi);
    registry.Register(std::move(c2n));
    registry.Register(n2c.Read());
    registry.Register(c2c.Read());

    const G::GhostChain shortNode{G::EntityKind::Cell,
                                  {G::Adj::Cell2Cell, G::Adj::Cell2Node},
                                  G::EntityKind::Node};
    const G::GhostChain longNode{G::EntityKind::Cell,
                                 {G::Adj::Cell2Cell, G::Adj::Cell2Node, n2c.Kind(), G::Adj::Cell2Node},
                                 G::EntityKind::Node};
    const G::GhostChain cellBranch{G::EntityKind::Cell,
                                   {G::Adj::Cell2Cell, G::Adj::Cell2Node, n2c.Kind(), G::Adj::Cell2Cell},
                                   G::EntityKind::Cell};
    auto evaluate = [&](const std::vector<G::GhostChain> &chains)
    { return registry.Evaluate(G::CompiledGhostTree::compile({chains})); };
    const auto shortOnly = evaluate({shortNode});
    const auto longOnly = evaluate({longNode});
    const auto cellOnly = evaluate({cellBranch});
    const auto combined = evaluate({shortNode, longNode, cellBranch});

    std::vector<I> expectedNodes;
    for (const auto *part : {&shortOnly, &longOnly})
    {
        const auto found = part->ghostIndices.find(G::EntityKind::Node);
        if (found != part->ghostIndices.end())
            expectedNodes.insert(expectedNodes.end(), found->second.begin(), found->second.end());
    }
    std::sort(expectedNodes.begin(), expectedNodes.end());
    expectedNodes.erase(std::unique(expectedNodes.begin(), expectedNodes.end()), expectedNodes.end());
    const auto combinedNodes = combined.ghostIndices.find(G::EntityKind::Node);
    CHECK((combinedNodes == combined.ghostIndices.end() ? std::vector<I>{} : combinedNodes->second) == expectedNodes);

    const auto cellFound = cellOnly.ghostIndices.find(G::EntityKind::Cell);
    const auto combinedCells = combined.ghostIndices.find(G::EntityKind::Cell);
    CHECK((combinedCells == combined.ghostIndices.end() ? std::vector<I>{} : combinedCells->second) ==
          (cellFound == cellOnly.ghostIndices.end() ? std::vector<I>{} : cellFound->second));

    // The shared Cell2Node endpoint is explicitly collected by shortNode even
    // though longer chains continue through it. Its state remains the parent
    // for both child branches.
    auto combinedTree = G::CompiledGhostTree::compile(G::GhostSpec{{shortNode, longNode, cellBranch}});
    REQUIRE(combinedTree.Roots().size() == 1);
    REQUIRE(combinedTree.Roots()[0].children.size() == 1);
    const auto &cellHop = combinedTree.Roots()[0].children[0];
    REQUIRE(cellHop.children.size() == 1);
    const auto &nodeEndpoint = cellHop.children[0];
    CHECK(nodeEndpoint.hop == G::Adj::Cell2Node);
    CHECK(nodeEndpoint.collect);
    REQUIRE(nodeEndpoint.children.size() == 1);
    const auto &nodeToCell = nodeEndpoint.children[0];
    REQUIRE(nodeToCell.children.size() == 2);
    for (const auto &terminal : nodeToCell.children)
        CHECK(terminal.collect);

    if (mpi.size > 1)
    {
        const auto shortNodes = shortOnly.ghostIndices.find(G::EntityKind::Node);
        const auto longNodes = longOnly.ghostIndices.find(G::EntityKind::Node);
        CHECK(shortNodes != shortOnly.ghostIndices.end());
        CHECK(longNodes != longOnly.ghostIndices.end());
    }
}

TEST_CASE("Collected intermediate owned nodes remain available to downstream ghost branches")
{
    const auto mpi = MPI();
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, 1);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, 1);
    const I next = (mpi.rank + 1) % mpi.size;
    G::MeshAdjacency c2n(G::Adj::Cell2Node, cells, nodes, Rows<I>({{mpi.rank}}));
    G::MeshAdjacency n2c(G::Adj::Node2Cell, nodes, cells, Rows<I>({{next}}));
    G::AdjacencyRegistry registry(mpi);
    registry.Register(c2n.Read());
    registry.Register(n2c.Read());
    G::GhostSpec spec{{{G::EntityKind::Cell, {G::Adj::Cell2Node}, G::EntityKind::Node},
                       {G::EntityKind::Cell, {G::Adj::Cell2Node, G::Adj::Node2Cell}, G::EntityKind::Cell},
                       {G::EntityKind::Cell, {G::Adj::Cell2Node, G::Adj::Node2Cell, G::Adj::Cell2Node}, G::EntityKind::Node}}};
    const auto result = registry.Evaluate(G::CompiledGhostTree::compile(spec));
    for (const auto kind : {G::EntityKind::Cell, G::EntityKind::Node})
    {
        const auto found = result.ghostIndices.find(kind);
        const auto actual = found == result.ghostIndices.end() ? std::vector<I>{} : found->second;
        CHECK(actual == (mpi.size > 1 ? std::vector<I>{next} : std::vector<I>{}));
    }
}

TEST_CASE("Native ghost publication agrees complete same-level relation schedules")
{
    auto mesh = Fixture();
    auto mpi = mesh->mpi;
    auto cells = mesh->Entities(G::EntityKind::Cell);
    const G::AdjacencyDescriptor viaNode{G::EntityKind::Cell, G::EntityKind::Cell, G::EntityKind::Node};
    const G::AdjacencyDescriptor viaFace{G::EntityKind::Cell, G::EntityKind::Cell, G::EntityKind::Face};
    const I next = (mpi.rank + 1) % mpi.size;
    G::MeshAdjacency nodeRelation(viaNode, cells, cells, Rows(std::vector<std::vector<I>>{{next}}));
    G::MeshAdjacency faceRelation(viaFace, cells, cells, Rows(std::vector<std::vector<I>>{{mpi.rank, next}}));
    G::AdjacencyRegistry setup(mpi);
    // Different insertion order must compile the same sorted collective order.
    if (mpi.rank % 2)
    {
        setup.Register(faceRelation.Read());
        setup.Register(nodeRelation.Read());
    }
    else
    {
        setup.Register(nodeRelation.Read());
        setup.Register(faceRelation.Read());
    }
    G::GhostResult discovery;
    if (next != mpi.rank)
        discovery.ghostIndices[G::EntityKind::Cell] = {next};
    auto result = setup.BuildGhosted(discovery);
    auto nodeRows = result.relations.at(viaNode).Read();
    auto faceRows = result.relations.at(viaFace).Read();
    CHECK(nodeRows.GlobalRow(0) == std::vector<I>{next});
    CHECK(faceRows.GlobalRow(0) == std::vector<I>{mpi.rank, next});
    if (next != mpi.rank)
    {
        CHECK(nodeRows.GlobalRow(1) == std::vector<I>{(next + 1) % mpi.size});
        CHECK(faceRows.GlobalRow(1) == std::vector<I>{next, (next + 1) % mpi.size});
    }
    if (mpi.size > 1)
    {
        G::AdjacencyRegistry mismatch(mpi);
        if (mpi.rank == 0)
            mismatch.Register(faceRelation.Read());
        else
            mismatch.Register(nodeRelation.Read());
        // Equal relation counts/from/to and row extents must not hide the
        // differing intermediary or enter incompatible payload collectives.
        CHECK_THROWS(mismatch.BuildGhosted(discovery));
    }
}

TEST_CASE("Exact payload aliases preserve named descriptors with rank-local alias differences")
{
    const auto mpi = MPI();
    auto fixture = Fixture();
    const auto &base = fixture->cellNodes;
    auto key = [&](const char *name)
    {
        auto result = G::Adj::Cell2Node;
        result.path = result.Path();
        result.selection = G::RelationSelection::Incidence;
        result.order = G::RelationOrder::Topological;
        result.periodic = G::RelationPeriodicPolicy::EntityIdentity;
        result.semanticName = name;
        result.semanticVersion = 1;
        return result;
    };
    const auto a = key("alias.a"), b = key("alias.b"), c = key("alias.c"),
               d = key("alias.d"), e = key("alias.e");
    auto independent = mpi.rank % 2 ? base.Owned().Clone() : base.Owned();
    auto reversed = base.Owned().Clone();
    {
        auto write = reversed.Write();
        auto row = write.View()[0];
        std::reverse(row.begin(), row.end());
    }
    G::MeshAdjacency first(a, base.Source(), base.Target(), base.Owned());
    G::MeshAdjacency second(b, base.Source(), base.Target(), independent);
    G::MeshAdjacency third(c, base.Source(), base.Target(), base.Owned());
    G::MeshAdjacency fourth(d, base.Source(), base.Target(), reversed);
    // Every rank has a cache hit, but on different prior transfers. Reusing
    // each local hit would silently give the ghost its donor's wrong row.
    G::MeshAdjacency fifth(e, base.Source(), base.Target(), mpi.rank % 2 ? reversed : base.Owned());
    G::AdjacencyRegistry setup(mpi);
    setup.Register(first.Read());
    setup.Register(second.Read());
    setup.Register(third.Read());
    setup.Register(fourth.Read());
    setup.Register(fifth.Read());
    G::GhostResult discovery;
    if (mpi.size > 1)
        discovery.ghostIndices[G::EntityKind::Cell].push_back((mpi.rank + 1) % mpi.size);
    const auto result = setup.BuildGhosted(discovery);
    const auto reference = base.Read();
    for (const auto &descriptor : {a, b, c, d, e})
    {
        const auto &relation = result.relations.at(descriptor);
        CHECK(relation.Kind() == descriptor);
        const auto read = relation.Read();
        auto expected = [&](I donor)
        {
            std::vector<I> row{4 * donor, 4 * donor + 1, 4 * donor + 2, 4 * donor + 3};
            if (descriptor == d || (descriptor == e && donor % 2))
                std::reverse(row.begin(), row.end());
            return row;
        };
        CHECK(read.GlobalRow(0) == expected(mpi.rank));
        if (mpi.size > 1)
        {
            const auto donor = I((mpi.rank + 1) % mpi.size);
            CHECK(read.GlobalRow(1) == expected(donor));
        }
    }
}

TEST_CASE("Aliased adjacency reads preserve constraints order and independent lease lifetimes")
{
    const auto mpi = MPI();
    auto cells = G::EntitySpace::CreateOwned(G::EntityKind::Cell, mpi, 3);
    const I first = 3 * mpi.rank;
    std::vector<I> ghostIds;
    if (mpi.size > 1)
        ghostIds.push_back(3 * ((mpi.rank + 1) % mpi.size));
    cells = cells.WithGhosts(ghostIds);
    auto descriptor = [](const char *name, bool sorted, bool removeSelf)
    {
        auto key = G::Adj::Cell2Cell;
        key.path = {G::EntityKind::Cell, G::EntityKind::Cell};
        key.selection = G::RelationSelection::Custom;
        key.order = sorted ? G::RelationOrder::SortedUnique : G::RelationOrder::Topological;
        key.periodic = G::RelationPeriodicPolicy::EntityIdentity;
        key.removeSelf = removeSelf;
        key.semanticName = name;
        key.semanticVersion = 1;
        return key;
    };
    const auto ordered = descriptor("read.ordered", false, false),
               sorted = descriptor("read.sorted", true, false),
               self = descriptor("read.self", false, true),
               strict = descriptor("read.strict", true, true);
    auto expected = [&](I global)
    {
        std::vector<I> row{(global + 1) % cells.NumGlobal(), (global + 2) % cells.NumGlobal()};
        std::sort(row.begin(), row.end());
        return row;
    };
    for (auto state : {G::Adj_PointToGlobal, G::Adj_PointToLocal})
    {
        auto encode = [&](I global)
        {
            auto row = expected(global);
            if (state == G::Adj_PointToLocal)
                for (auto &value : row)
                {
                    const auto local = cells.GlobalToLocal(value);
                    value = local ? *local : -1 - value;
                }
            return row;
        };
        std::vector<std::vector<I>> ownedRows, ghostRows;
        for (I iRow = 0; iRow < 3; ++iRow)
            ownedRows.push_back(encode(first + iRow));
        for (auto global : ghostIds)
            ghostRows.push_back(encode(global));
        auto owned = Rows(ownedRows), ghost = Rows(ghostRows);
        // Every descriptor order must retain the caller's order, regardless
        // of which descriptor is chosen as the content-proof basis.
        std::vector<int> permutation{0, 1, 2, 3};
        const std::vector<G::AdjacencyDescriptor> descriptors{ordered, sorted, self, strict};
        do
        {
            std::vector<G::AdjacencyDescriptor> keys;
            for (auto item : permutation)
                keys.push_back(descriptors[item]);
            auto reads = G::MeshAdjacencyRead::AcquireAliases(keys, cells, cells, state, owned, &ghost);
            REQUIRE(reads.size() == keys.size());
            for (I iRead = 0; iRead < I(reads.size()); ++iRead)
            {
                CHECK(reads[iRead].Kind() == keys[iRead]);
                for (I iRow = 0; iRow < cells.NumLocal(); ++iRow)
                    CHECK(reads[iRead].GlobalRow(iRow) == expected(cells.LocalToGlobal(iRow)));
            }
            {
                auto retained = std::move(reads.back());
                reads.clear();
                CHECK_THROWS(owned.Write());
                if (!ghostIds.empty())
                    CHECK_THROWS(ghost.Write());
                CHECK(retained.GlobalRow(0) == expected(first));
            }
            CHECK_NOTHROW(owned.Write());
            CHECK_NOTHROW(ghost.Write());
        } while (std::next_permutation(permutation.begin(), permutation.end()));
        // No descriptor in this set proves both constraints: the factory must
        // scan any constraint not implied by the already validated basis.
        auto independent = G::MeshAdjacencyRead::AcquireAliases({ordered, sorted, self}, cells, cells, state, owned, &ghost);
        CHECK(independent.size() == 3);
    }
    auto owned = Rows<I>({expected(first), expected(first + 1), expected(first + 2)});
    auto ghost = Rows<I>(ghostIds.empty() ? std::vector<std::vector<I>>{} : std::vector<std::vector<I>>{expected(ghostIds.front())});
    auto malformed = owned.Clone();
    {
        auto write = malformed.Write();
        auto row = write.View()[0];
        std::reverse(row.begin(), row.end());
    }
    // Self-only passes; sorted-only must still scan and reject after weaker
    // scopes have been acquired. Failed acquisition must release every lease.
    CHECK_THROWS(G::MeshAdjacencyRead::AcquireAliases({ordered, sorted, self}, cells, cells,
                                                      G::Adj_PointToGlobal, malformed, &ghost));
    CHECK_NOTHROW(malformed.Write());
    {
        auto write = malformed.Write();
        write.View()(0, 0) = first;
        write.View()(0, 1) = first;
    }
    CHECK_THROWS(G::MeshAdjacencyRead::AcquireAliases({ordered, self}, cells, cells,
                                                      G::Adj_PointToGlobal, malformed, &ghost));
    {
        auto write = malformed.Write();
        write.View()(0, 0) = cells.NumGlobal();
    }
    CHECK_THROWS(G::MeshAdjacencyRead::AcquireAliases({ordered}, cells, cells,
                                                      G::Adj_PointToGlobal, malformed, &ghost));
    CHECK_NOTHROW(malformed.Write());
    auto wrongFamily = ordered;
    wrongFamily.to = G::EntityKind::Node;
    wrongFamily.path.back() = G::EntityKind::Node;
    CHECK_THROWS(G::MeshAdjacencyRead::AcquireAliases({strict, wrongFamily}, cells, cells,
                                                      G::Adj_PointToGlobal, owned, &ghost));
    CHECK_NOTHROW(owned.Write());
    if (!ghostIds.empty())
    {
        auto badGhost = Rows<I>({{0, 0}});
        CHECK_THROWS(G::MeshAdjacencyRead::AcquireAliases({ordered, sorted}, cells, cells,
                                                          G::Adj_PointToGlobal, owned, &badGhost));
        CHECK_NOTHROW(badGhost.Write());
    }
    CHECK(G::MeshAdjacencyRead::AcquireAliases({}, cells, cells, G::Adj_PointToGlobal, owned).empty());
}

TEST_CASE("Global adjacency validation agrees with an independent scalar oracle")
{
    const auto mpi = MPI();
    const auto cells = G::EntitySpace::CreateOwned(G::EntityKind::Cell, mpi, 1);
    const auto self = cells.LocalToGlobal(0);
    const std::array<I, 5> pool{DNDS::UnInitIndex, -1, self, (self + 1) % cells.NumGlobal(), cells.NumGlobal()};
    for (bool sorted : {false, true})
        for (bool removeSelf : {false, true})
        {
            auto key = G::Adj::Cell2Cell;
            key.path = {G::EntityKind::Cell, G::EntityKind::Cell};
            key.selection = G::RelationSelection::Custom;
            key.order = sorted ? G::RelationOrder::SortedUnique : G::RelationOrder::Topological;
            key.periodic = G::RelationPeriodicPolicy::EntityIdentity;
            key.removeSelf = removeSelf;
            key.semanticName = "validation.scalar-oracle";
            key.semanticVersion = 1;
            REQUIRE_NOTHROW(key.Validate());
            for (I length = 0, combinations = 1; length <= 3; ++length, combinations *= I(pool.size()))
                for (I code = 0; code < combinations; ++code)
                {
                    std::vector<I> values(size_t(length), 0);
                    auto remaining = code;
                    for (I iSlot = 0; iSlot < length; ++iSlot)
                    {
                        values[size_t(iSlot)] = pool[size_t(remaining % I(pool.size()))];
                        remaining /= I(pool.size());
                    }
                    bool expected = std::all_of(values.begin(), values.end(), [&](I value)
                                                { return value == DNDS::UnInitIndex || (value >= 0 && value < cells.NumGlobal()); });
                    if (sorted)
                        expected = expected && std::find(values.begin(), values.end(), DNDS::UnInitIndex) == values.end() &&
                                   std::adjacent_find(values.begin(), values.end(), std::greater_equal<I>{}) == values.end();
                    if (removeSelf)
                        expected = expected && std::find(values.begin(), values.end(), self) == values.end();
                    auto data = Rows(std::vector<std::vector<I>>{values});
                    bool accepted = true;
                    try
                    {
                        G::MeshAdjacencyRead read(key, cells, cells, G::Adj_PointToGlobal, data);
                    }
                    catch (const std::exception &)
                    {
                        accepted = false;
                    }
                    CHECK(accepted == expected);
                    CHECK(data.IsValid());
                    const auto read = data.Read();
                    const auto row = read.View()[0];
                    CHECK(std::vector<I>(row.begin(), row.end()) == values);
                }
        }
}

TEST_CASE("Adjacency self checks retain ghost-only source numbering on empty owners")
{
    const auto mpi = MPI();
    const auto ownedCells = G::EntitySpace::CreateOwned(G::EntityKind::Cell, mpi, mpi.rank == 0 ? 2 : 0);
    const auto cells = ownedCells.WithGhosts(mpi.rank == 0 ? std::vector<I>{} : std::vector<I>{0, 1});
    auto owned = Rows(mpi.rank == 0 ? std::vector<std::vector<I>>{{1}, {0}} : std::vector<std::vector<I>>{});
    auto ghost = Rows(mpi.rank == 0 ? std::vector<std::vector<I>>{} : std::vector<std::vector<I>>{{1}, {0}});
    auto key = G::Adj::Cell2Cell;
    key.path = {G::EntityKind::Cell, G::EntityKind::Cell};
    key.selection = G::RelationSelection::Custom;
    key.order = G::RelationOrder::SortedUnique;
    key.periodic = G::RelationPeriodicPolicy::EntityIdentity;
    key.removeSelf = true;
    key.semanticName = "validation.ghost-only";
    key.semanticVersion = 1;
    REQUIRE_NOTHROW(key.Validate());
    CHECK_NOTHROW(G::MeshAdjacencyRead(key, cells, cells, G::Adj_PointToGlobal, owned, &ghost));
    auto &changed = mpi.rank == 0 ? owned : ghost;
    changed.Write().At(0, 0) = 0;
    CHECK_THROWS(G::MeshAdjacencyRead(key, cells, cells, G::Adj_PointToGlobal, owned, &ghost));
    changed.Write().At(0, 0) = 1;
    CHECK_NOTHROW(G::MeshAdjacencyRead(key, cells, cells, G::Adj_PointToGlobal, owned, &ghost));
}

TEST_CASE("Native 3D edge interpolation retains four-parent incidence and empty owners")
{
    auto mpi = MPI();
    const I count = 4 / mpi.size + (mpi.rank < 4 % mpi.size ? 1 : 0);
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == 0 ? 18 : 0);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, count);
    std::vector<I> ghostNodes;
    if (mpi.rank != 0)
        for (I node = 0; node < 18; ++node)
            ghostNodes.push_back(node);
    nodes = nodes.WithGhosts(std::move(ghostNodes));
    const I first = count ? cells.LocalToGlobal(0) : 0;
    auto cellNodes = [](I cell)
    {
        const I x = cell % 2, y = cell / 2;
        const auto node = [](I a, I b, I z)
        { return a + 3 * b + 9 * z; };
        return std::vector<I>{node(x, y, 0), node(x + 1, y, 0), node(x + 1, y + 1, 0), node(x, y + 1, 0),
                              node(x, y, 1), node(x + 1, y, 1), node(x + 1, y + 1, 1), node(x, y + 1, 1)};
    };
    std::vector<std::vector<I>> owned;
    for (I row = 0; row < count; ++row)
        owned.push_back(cellNodes(first + row));
    std::vector<I> ghostCells;
    for (I cell = 0; cell < 4; ++cell)
        if (cell < first || cell >= first + count)
            ghostCells.push_back(cell);
    cells = cells.WithGhosts(std::move(ghostCells));
    G::MeshAdjacency initial(G::Adj::Cell2Node, cells, nodes, Rows(owned));
    auto native = initial.WithGhostRows(G::EntityGhostMapping(cells));
    auto cone = native.Read();
    G::SubEntityQueryPbi query;
    query.numSubEntities = [](I)
    { return G::Elem::Element{G::Elem::Hex8}.GetNumEdges(); };
    query.describe = [](I, int sub)
    {
        auto edge = G::Elem::Element{G::Elem::Hex8}.ObtainEdge(sub);
        return G::SubEntityDesc{edge.GetNumVertices(), edge.GetNumNodes(), G::t_index(edge.type)};
    };
    query.extractNodes = [](I, int sub, const std::function<I(int)> &read, I *out)
    {
        std::vector<I> parent(8), edge(2);
        for (int j = 0; j < 8; ++j)
            parent[size_t(j)] = read(j);
        G::Elem::Element{G::Elem::Hex8}.ExtractEdgeNodes(sub, parent, edge);
        std::copy(edge.begin(), edge.end(), out);
    };
    auto result = G::AdjacencyRegistry::Interpolate(cone, G::EntityKind::Edge, query, Owner());
    // Four unit hexes in a 2x2x1 block have 33 axis-aligned edges. The
    // central vertical edge (4,13) has all four parents, rather than width 2.
    constexpr int pairs[12][2]{{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    std::map<std::pair<I, I>, std::vector<I>> expected;
    for (I cell = 0; cell < 4; ++cell)
    {
        auto row = cellNodes(cell);
        for (auto &pair : pairs)
            expected[std::minmax(row[size_t(pair[0])], row[size_t(pair[1])])].push_back(cell);
    }
    REQUIRE(expected.size() == 33);
    CHECK(result.entities.NumGlobal() == 33);
    auto e2n = result.entity2node.Read();
    auto e2p = result.entity2parent.Read();
    std::vector<int> occurrences(expected.size(), 0);
    for (I edge = 0; edge < e2n.NumOwned(); ++edge)
    {
        auto vertices = e2n.GlobalRow(edge), parents = e2p.GlobalRow(edge);
        REQUIRE(vertices.size() == 2);
        auto found = expected.find(std::minmax(vertices[0], vertices[1]));
        REQUIRE(found != expected.end());
        std::sort(parents.begin(), parents.end());
        CHECK(parents == found->second);
        if (found->first == std::make_pair(I(4), I(13)))
            CHECK(parents == std::vector<I>{0, 1, 2, 3});
        ++occurrences[size_t(std::distance(expected.begin(), found))];
    }
    MPI_Allreduce(MPI_IN_PLACE, occurrences.data(), int(occurrences.size()), MPI_INT, MPI_SUM, mpi.comm);
    for (auto count : occurrences)
        CHECK(count == 1);
    G::AdjacencyRegistry generated(mpi);
    generated.Register(result.parent2entity.Read());
    generated.Register(result.entity2node.Read());
    G::GhostSpec spec{{{G::EntityKind::Cell, {G::Adj::Cell2Edge}, G::EntityKind::Edge}}};
    auto detached = generated.BuildGhosted(generated.Evaluate(G::CompiledGhostTree::compile(spec)));
    auto completeEdges = detached.relations.at(G::Adj::Edge2Node).Read();
    auto p2e = result.parent2entity.Read();
    for (I parent = 0; parent < p2e.RowCount(); ++parent)
    {
        CHECK(p2e.RowSize(parent) == 12);
        for (size_t slot = 0; slot < p2e.RowSize(parent); ++slot)
            CHECK(p2e.Raw(parent, slot) != DNDS::UnInitIndex);
    }
    for (I parent = 0; parent < p2e.NumOwned(); ++parent)
        for (size_t slot = 0; slot < p2e.RowSize(parent); ++slot)
        {
            const auto edge = detached.spaces.at(G::EntityKind::Edge).GlobalToLocal(p2e.Raw(parent, slot));
            REQUIRE(edge);
            auto vertices = completeEdges.GlobalRow(*edge);
            std::vector<I> wanted(2);
            query.extractNodes(parent, int(slot), [&](int j)
                               { return cone.ReadEntry(parent, size_t(j)).global; }, wanted.data());
            std::sort(vertices.begin(), vertices.end());
            std::sort(wanted.begin(), wanted.end());
            CHECK(vertices == wanted);
        }
}

TEST_CASE("Current authority feeds native setup without aliasing detached native setup copies")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(MPI());
    auto &mesh = *fixture.mesh;
    const auto tree = G::CompiledGhostTree::compile(G::GhostSpec{{{G::EntityKind::Cell, {G::Adj::Cell2Cell, G::Adj::Cell2Node}, G::EntityKind::Node}}});
    auto evaluate = [&]
    {
        G::AdjacencyRegistry setup(mesh.mpi);
        setup.RegisterMesh(mesh, G::MeshReadRequirements::Primary());
        return setup.Evaluate(tree);
    };
    auto baseline = evaluate();
    auto interpolate = [&]
    {
        auto owner = mesh.AcquireRead(G::MeshReadRequirements::Primary());
        auto view = owner.HostView();
        auto cone = G::MeshAdjacencyRead::Current(mesh, G::Adj::Cell2Node, G::MeshReadRequirements::Primary());
        G::SubEntityQueryPbi query;
        query.numSubEntities = [view](I parent)
        { return view.GetCellElement(parent).GetNumFaces(); };
        query.describe = [view](I parent, int sub)
        {
            auto face = view.GetCellElement(parent).ObtainFace(sub);
            return G::SubEntityDesc{face.GetNumVertices(), face.GetNumNodes(), G::t_index(face.type)};
        };
        query.extractNodes = [view](I parent, int sub, const std::function<I(int)> &nodes, I *out)
        {
            auto element = view.GetCellElement(parent);
            std::vector<I> parentNodes(size_t(element.GetNumNodes()));
            for (size_t j = 0; j < parentNodes.size(); ++j)
                parentNodes[j] = nodes(int(j));
            std::vector<I> faceNodes(size_t(element.ObtainFace(sub).GetNumNodes()));
            element.ExtractFaceNodes(sub, parentNodes, faceNodes);
            std::copy(faceNodes.begin(), faceNodes.end(), out);
        };
        auto result = G::AdjacencyRegistry::Interpolate(cone, G::EntityKind::Face, query, Owner());
        auto e2n = result.entity2node.Read();
        auto p2e = result.parent2entity.Read();
        CHECK(p2e.RowCount() == view.NumCellProc());
        std::vector<std::vector<I>> output;
        for (I row = 0; row < e2n.NumOwned(); ++row)
            output.push_back(e2n.GlobalRow(row));
        G::AdjacencyRegistry generated(mesh.mpi);
        generated.Register(result.parent2entity.Read());
        generated.Register(result.entity2node.Read());
        G::GhostSpec closure{{{G::EntityKind::Cell, {{G::EntityKind::Cell, G::EntityKind::Face}}, G::EntityKind::Face},
                              {G::EntityKind::Cell, {{G::EntityKind::Cell, G::EntityKind::Face}, {G::EntityKind::Face, G::EntityKind::Node}}, G::EntityKind::Node}}};
        auto detached = generated.BuildGhosted(generated.Evaluate(G::CompiledGhostTree::compile(closure)));
        auto edges = detached.relations.at(G::Adj::Face2Node).Read();
        for (I parent = 0; parent < view.NumCell(); ++parent)
            for (size_t slot = 0; slot < p2e.RowSize(parent); ++slot)
            {
                const auto entity = detached.spaces.at(G::EntityKind::Face).GlobalToLocal(p2e.Raw(parent, slot));
                REQUIRE(entity);
                auto actual = edges.GlobalRow(*entity);
                std::vector<I> expected(size_t(view.GetCellElement(parent).ObtainFace(int(slot)).GetNumNodes()));
                query.extractNodes(parent, int(slot), [&](int j)
                                   { return cone.ReadEntry(parent, size_t(j)).global; }, expected.data());
                std::sort(actual.begin(), actual.end());
                std::sort(expected.begin(), expected.end());
                CHECK(actual == expected);
            }
        return output;
    };
    auto expected = interpolate();
    auto detached = G::NativeMeshAssembler::CaptureOwned(*mesh.CurrentNativeState(), mesh.mpi);
    auto copy = detached.relations[size_t(G::MeshRelation::CellNodes)]->To(R::Device::Host());
    {
        auto write = copy.Write();
        for (size_t cell = 0; cell < copy.RowCount(); ++cell)
            for (size_t slot = 0; slot < write.View().RowSize(cell); ++slot)
                write.View()(cell, slot) = DNDS::UnInitIndex;
    }
    CHECK(evaluate().ghostIndices == baseline.ghostIndices);
    CHECK(interpolate() == expected);
}

TEST_CASE("Native periodic distributed interpolation preserves owner frames and ordered companions")
{
    auto mpi = MPI();
    G::NativePeriodicImportOptions periodic;
    periodic.tolerance = 1e-8;
    periodic.frames.translation[1].map() = G::tPoint{10, 0, 0};
    periodic.frames.translation[2].map() = G::tPoint{0, 10, 0};
    const auto path = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() / "data/mesh/IV10_10.cgns";
    auto state = G::ReadNativeCGNS(mpi, path.string(), 2, G::FBC_Name_2_ID_Default, 0, {1, true}, periodic);
    state = G::PrepareNativeMeshTopology(*state, mpi);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, state);
    auto run = [&]
    {
        auto owner = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
        auto view = owner.HostView();
        auto cone = G::MeshAdjacencyRead::Current(*mesh, G::Adj::Cell2Node, G::MeshReadRequirements::Primary());
        G::RelationPeriodicBits periodic(cone);
        auto result = G::AdjacencyRegistry::Interpolate(cone, G::EntityKind::Face, QuadFaces(), Owner(), &periodic);
        auto parents = result.parent2entity.Read();
        G::AdjacencyRegistry generated(mpi);
        generated.Register(result.parent2entity.Read());
        generated.Register(result.entity2node.Read());
        G::GhostSpec spec{{{G::EntityKind::Cell, {G::Adj::Cell2Face}, G::EntityKind::Face}}};
        auto detached = generated.BuildGhosted(generated.Evaluate(G::CompiledGhostTree::compile(spec)));
        auto faces = detached.relations.at(G::Adj::Face2Node).Read();
        R::ElementSchema schema(G::NodePeriodicBits::CommType(), sizeof(G::NodePeriodicBits), "Geom.NodePeriodicBits",
                                R::ElementSchema::TypeIdentity<G::NodePeriodicBits>());
        // Generated owned order matches the publication; its ghost map is new.
        auto ghosts = detached.routes.at(G::EntityKind::Face).TransferRows(detached.spaces.at(G::EntityKind::Face), *result.entity2nodePbi, schema);
        auto ownedBits = result.entity2nodePbi->Read();
        auto ghostBits = ghosts.Read();
        auto frameBits = result.parent2entityPbi->Read();
        I localFaces = result.entities.NumOwned(), totalFaces = 0;
        MPI_Allreduce(&localFaces, &totalFaces, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        CHECK(totalFaces == mesh->Entities(G::EntityKind::Face).NumGlobal());
        std::vector<std::vector<I>> identity;
        for (I parent = 0; parent < view.NumCell(); ++parent)
        {
            identity.push_back(parents.GlobalRow(parent));
            for (size_t slot = 0; slot < parents.RowSize(parent); ++slot)
            {
                const auto face = detached.spaces.at(G::EntityKind::Face).GlobalToLocal(parents.Raw(parent, slot));
                REQUIRE(face);
                auto nodes = faces.GlobalRow(*face);
                const auto bitsView = *face < localFaces ? ownedBits.View() : ghostBits.View();
                const auto bitsRow = size_t(*face < localFaces ? *face : *face - localFaces);
                std::vector<G::tPoint> actual, expected;
                for (size_t j = 0; j < nodes.size(); ++j)
                {
                    const auto node = mesh->Entities(G::EntityKind::Node).GlobalToLocal(nodes[j]);
                    REQUIRE(node);
                    const auto combined = G::NodePeriodicBits{uint8_t(uint8_t(bitsView(bitsRow, j)) ^ uint8_t(frameBits.View()(size_t(parent), slot)))};
                    actual.push_back(view.Periodic().GetCoordByBits(view.Coord(*node), combined));
                }
                expected.push_back(view.GetCoordNodeOnCell(parent, DNDS::rowsize(slot)));
                expected.push_back(view.GetCoordNodeOnCell(parent, DNDS::rowsize((slot + 1) % 4)));
                const bool same = actual[0].isApprox(expected[0]) && actual[1].isApprox(expected[1]);
                const bool reversed = actual[0].isApprox(expected[1]) && actual[1].isApprox(expected[0]);
                CHECK((same || reversed));
            }
        }
        return identity;
    };
    auto expected = run();
    auto detached = G::NativeMeshAssembler::CaptureOwned(*state, mpi);
    auto copy = detached.relations[size_t(G::MeshRelation::CellNodes)]->To(R::Device::Host());
    {
        auto write = copy.Write();
        for (size_t row = 0; row < copy.RowCount(); ++row)
            for (size_t slot = 0; slot < write.View().RowSize(row); ++slot)
                write.View()(row, slot) = DNDS::UnInitIndex;
    }
    CHECK(run() == expected);
}

TEST_CASE("Traversal scratch preserves independent publications and exact rows after tree growth")
{
    const auto mpi = MPI();
    const I peer = (mpi.rank + 1) % mpi.size;
    auto cells = G::EntitySpace::CreateOwned(G::EntityKind::Cell, mpi, 1);
    auto nodes = cells.CreateSibling(G::EntityKind::Node, 4);
    G::MeshAdjacency neighbors(G::Adj::Cell2Cell, cells, cells, Rows(std::vector<std::vector<I>>{{peer}}));
    G::MeshAdjacency cones(G::Adj::Cell2Node, cells, nodes,
                           Rows(std::vector<std::vector<I>>{{mpi.rank * 4, mpi.rank * 4 + 1, mpi.rank * 4 + 2, mpi.rank * 4 + 3}}));
    G::AdjacencyRegistry registry(mpi);
    registry.Register(neighbors.Read());
    registry.Register(cones.Read());
    G::GhostSpec spec{{{G::EntityKind::Cell, {G::Adj::Cell2Cell}, G::EntityKind::Cell},
                       {G::EntityKind::Cell, {G::Adj::Cell2Cell, G::Adj::Cell2Node}, G::EntityKind::Node}}};
    auto discovery = registry.Evaluate(G::CompiledGhostTree::compile(spec));
    auto first = registry.BuildGhosted(discovery, {G::Adj::Cell2Node});
    auto second = registry.BuildGhosted(discovery, {G::Adj::Cell2Node});
    const auto verify = [&](const auto &publication)
    {
        auto rows = publication.relations.at(G::Adj::Cell2Node).Read();
        for (I iRow = rows.NumOwned(); iRow < rows.RowCount(); ++iRow)
        {
            const I global = rows.Source().LocalToGlobal(iRow);
            CHECK(rows.GlobalRow(iRow) == std::vector<I>{global * 4, global * 4 + 1, global * 4 + 2, global * 4 + 3});
        }
    };
    verify(first);
    verify(second);
    if (mpi.size > 1)
    {
        CHECK(first.relations.at(G::Adj::Cell2Node).Ghost()->Identity() !=
              second.relations.at(G::Adj::Cell2Node).Ghost()->Identity());
    }
    // Published output may be written after its read scopes end. An active or
    // completed write must exclude reuse without contaminating publication.
    std::optional<R::Array<I>::WriteLease> writer;
    if (mpi.size > 1 && mpi.rank == 0)
    {
        auto alias = *first.relations.at(G::Adj::Cell2Node).Ghost();
        writer.emplace(alias.Write());
        writer->At(0, 0) = peer * 4 + 2;
    }
    auto whileWriting = registry.BuildGhosted(discovery, {G::Adj::Cell2Node});
    verify(second);
    verify(whileWriting);
    writer.reset();
    auto repaired = registry.BuildGhosted(discovery, {G::Adj::Cell2Node});
    verify(repaired);
    auto differentFrontier = spec;
    differentFrontier.chains.push_back({G::EntityKind::Cell,
                                        {G::Adj::Cell2Cell, G::Adj::Cell2Cell, G::Adj::Cell2Node},
                                        G::EntityKind::Node});
    auto differentDiscovery = registry.Evaluate(G::CompiledGhostTree::compile(differentFrontier));
    auto sameCountDifferentIds = registry.BuildGhosted(differentDiscovery, {G::Adj::Cell2Node});
    CHECK(sameCountDifferentIds.spaces.at(G::EntityKind::Cell).NumGhost() == (mpi.size > 1 ? 1 : 0));
    verify(sameCountDifferentIds);
    // The larger recipe still uses the same registered relations. Its union
    // can be larger than the last individual fetched frontier, so row counts
    // or element layouts alone cannot justify scratch reuse.
    spec.chains.push_back({G::EntityKind::Cell, {G::Adj::Cell2Cell, G::Adj::Cell2Cell}, G::EntityKind::Cell});
    spec.chains.push_back({G::EntityKind::Cell, {G::Adj::Cell2Cell, G::Adj::Cell2Cell, G::Adj::Cell2Node}, G::EntityKind::Node});
    auto expanded = registry.Evaluate(G::CompiledGhostTree::compile(spec));
    auto grown = registry.BuildGhosted(expanded, {G::Adj::Cell2Node});
    CHECK(grown.spaces.at(G::EntityKind::Cell).NumGhost() == (mpi.size > 2 ? 2 : mpi.size > 1 ? 1
                                                                                              : 0));
    verify(grown);
}

TEST_CASE("Entity ghost mapping rejects rank-local unpublished shapes and remains reusable")
{
    const auto mpi = MPI();
    auto cells = G::EntitySpace::CreateOwned(G::EntityKind::Cell, mpi, 3);
    const I peer = (mpi.rank + 1) % mpi.size;
    std::vector<I> globals;
    if (mpi.size > 1)
        globals = {peer * 3, peer * 3 + 2};
    cells = cells.WithGhosts(globals);
    G::EntityGhostMapping mapping(cells);
    const auto schema = R::ElementSchema::Arithmetic<I>();
    R::Array<I> owned(R::RowLayout::Dense(3, 5, 7));
    const auto fill = [&]
    {
        auto write = owned.Write();
        for (I iRow = 0; iRow < 3; ++iRow)
            for (I iCol = 0; iCol < 5; ++iCol)
                write.At(iRow, iCol) = 1000 + 16 * (mpi.rank * 3 + iRow) + iCol;
    };
    const auto transfer = [&](bool asynchronous)
    {
        if (!asynchronous)
            return mapping.TransferRows(cells, owned, schema);
        auto pending = mapping.BeginRows(cells, owned, schema);
        pending.operation.Wait();
        return std::move(pending.ghost);
    };
    const auto verify = [&]
    {
        for (bool asynchronous : {false, true})
        {
            auto ghost = transfer(asynchronous);
            auto read = ghost.Read();
            CHECK(read.View().RowCount() == I(globals.size()));
            for (I iRow = 0; iRow < I(globals.size()); ++iRow)
                for (I iCol = 0; iCol < 5; ++iCol)
                    CHECK(read.At(iRow, iCol) == 1000 + 16 * globals[iRow] + iCol);
        }
    };
    fill();
    verify();
    std::optional<R::ExclusiveRows> editor;
    if (mpi.rank == 0)
        editor.emplace(owned.EditRows());
    CHECK_THROWS(transfer(false));
    CHECK_THROWS(transfer(true));
    if (editor)
        editor->Commit();
    editor.reset();
    verify();
    if (mpi.rank == 0)
    {
        auto abandoned = owned.EditRows();
    }
    CHECK_THROWS(transfer(false));
    CHECK_THROWS(transfer(true));
    owned.Reset(R::RowLayout::Dense(3, 5, 7));
    fill();
    verify();
    std::optional<R::Array<I>> retained;
    if (mpi.rank == 0)
        retained.emplace(std::move(owned));
    CHECK_THROWS(transfer(false));
    CHECK_THROWS(transfer(true));
    owned.Reset(R::RowLayout::Dense(3, 5, 7));
    fill();
    verify();
}

TEST_CASE("Named native relations distinguish incidence any all exact and face versus node neighbors")
{
    const auto mpi = MPI();
    const I next = (mpi.rank + 1) % mpi.size, previous = (mpi.rank + mpi.size - 1) % mpi.size;
    auto cells = G::EntitySpace::CreateOwned(G::EntityKind::Cell, mpi, 3);
    auto nodes = cells.CreateSibling(G::EntityKind::Node, 5);
    auto faces = cells.CreateSibling(G::EntityKind::Face, 2);
    if (mpi.size > 1)
    {
        std::vector<I> remoteNodes;
        for (I j = 0; j < 5; ++j)
            remoteNodes.push_back(next * 5 + j);
        nodes = nodes.WithGhosts(remoteNodes);
        faces = faces.WithGhosts({previous * 2});
    }
    // c1 touches one face vertex; c2 contains every vertex but is not an
    // element-face incidence. Repeated n0 models a periodic self incidence.
    const I n = mpi.rank * 5;
    G::MeshAdjacency cellNodes(G::Adj::Cell2Node, cells, nodes,
                               Rows(std::vector<std::vector<I>>{{n + 2, n, n + 1, n}, {n + 3, n}, {n + 4, n + 1, n}}));
    G::MeshAdjacency faceNodes(G::Adj::Face2Node, faces, nodes,
                               Rows(std::vector<std::vector<I>>{{next * 5 + 1, next * 5, next * 5}, {}}));
    G::MeshAdjacency cellFaces(G::Adj::Cell2Face, cells, faces,
                               Rows(std::vector<std::vector<I>>{{previous * 2}, {}, {}}));
    auto c2n = cellNodes.Read(), f2n = faceNodes.Read(), c2f = cellFaces.Read();
    auto inverse = G::AdjacencyRegistry::Inverse(c2n, mpi);
    G::EntityGhostMapping nodeRoute(nodes);
    auto nodeSupport = inverse.WithGhostRows(nodeRoute);
    auto n2c = nodeSupport.Read();
    auto direct = G::AdjacencyRegistry::Inverse(c2f, mpi);
    auto any = G::AdjacencyRegistry::Compose(f2n, n2c);
    auto all = G::AdjacencyRegistry::ComposeAll(f2n, n2c);
    CHECK_THROWS(G::AdjacencyDescriptor::Composition(f2n.Kind(), n2c.Kind(), G::RelationSelection::Custom));
    const auto exactKind = G::AdjacencyDescriptor::Composition(f2n.Kind(), n2c.Kind(), G::RelationSelection::Exact,
                                                               false, "fixture.oriented-element-face", 1);
    auto exact = G::AdjacencyRegistry::ComposeFiltered(f2n, n2c, exactKind, [](I, I, int count)
                                                       { return count == 2; }, [](I, I target, const std::vector<I> &)
                                                       { return target % 3 == 0; });
    const std::vector<G::MeshAdjacency *> variants{&direct, &any, &all, &exact};
    const std::vector<std::vector<I>> expected{{next * 3}, {next * 3, next * 3 + 1, next * 3 + 2}, {next * 3, next * 3 + 2}, {next * 3}};
    std::unordered_set<G::AdjacencyDescriptor, G::AdjacencyDescriptorHash> kinds;
    G::AdjacencyRegistry registry(mpi);
    for (size_t variant = 0; variant < variants.size(); ++variant)
    {
        auto read = variants[variant]->Read();
        CHECK(read.GlobalRow(0) == expected[variant]);
        CHECK(read.GlobalRow(1).empty());
        kinds.insert(read.Kind());
        registry.Register(variants[variant]->Read());
        // The actual native registry retains every full key independently.
    }
    CHECK(kinds.size() == 4);
    // Traverse each stored definition through the actual distributed evaluator.
    for (size_t variant = 0; variant < variants.size(); ++variant)
    {
        const auto kind = variants[variant]->Read().Kind();
        G::GhostSpec spec{{{G::EntityKind::Face, {kind}, G::EntityKind::Cell}}};
        auto tree = G::CompiledGhostTree::compile(spec);
        CHECK(tree.dump().find(G::adjKindName(kind)) != std::string::npos);
        auto discovery = registry.Evaluate(tree);
        const auto found = discovery.ghostIndices.find(G::EntityKind::Cell);
        if (mpi.size > 1)
        {
            REQUIRE(found != discovery.ghostIndices.end());
            CHECK(found->second == expected[variant]);
        }
        else
            CHECK(found == discovery.ghostIndices.end());
    }
    auto published = registry.BuildGhosted({});
    CHECK(published.relations.size() == 4);
    // Inversion of the face cone provides a genuine node-centered distributed
    // traversal: each owned n0/n1 sees the previous rank's face.
    auto n2f = G::AdjacencyRegistry::Inverse(f2n, mpi);
    registry.Register(n2f.Read());
    CHECK(n2f.Read().Kind() != G::Adj::Node2Face);
    G::GhostSpec nodeSpec{{{G::EntityKind::Node, {n2f.Read().Kind()}, G::EntityKind::Face}}};
    auto nodeDiscovery = registry.Evaluate(G::CompiledGhostTree::compile(nodeSpec));
    if (mpi.size > 1)
        CHECK(nodeDiscovery.ghostIndices.at(G::EntityKind::Face) == std::vector<I>{previous * 2});
    else
        CHECK(nodeDiscovery.ghostIndices.empty());
    for (auto *variant : variants)
        CHECK(published.relations.at(variant->Read().Kind()).Read().GlobalRow(0) == variant->Read().GlobalRow(0));

    auto nodeNeighbors = G::AdjacencyRegistry::Compose(c2n, n2c, true);
    G::EntityGhostMapping faceRoute(faces);
    auto faceSupport = direct.WithGhostRows(faceRoute);
    auto faceNeighbors = G::AdjacencyRegistry::Compose(c2f, faceSupport.Read(), true);
    CHECK(nodeNeighbors.Read().Kind() != faceNeighbors.Read().Kind());
    CHECK(nodeNeighbors.Read().GlobalRow(0) == std::vector<I>{mpi.rank * 3 + 1, mpi.rank * 3 + 2});
    CHECK(nodeNeighbors.Read().GlobalRow(1) == std::vector<I>{mpi.rank * 3, mpi.rank * 3 + 2});
    CHECK(faceNeighbors.Read().GlobalRow(0).empty());
    CHECK(faceNeighbors.Read().GlobalRow(1).empty());
    auto includeSelf = G::AdjacencyRegistry::Compose(c2n, n2c);
    CHECK(includeSelf.Read().Kind() != nodeNeighbors.Read().Kind());
    CHECK(includeSelf.Read().GlobalRow(0) == std::vector<I>{mpi.rank * 3, mpi.rank * 3 + 1, mpi.rank * 3 + 2});
    auto nodeFaces = G::AdjacencyRegistry::Compose(n2c, c2f);
    auto completeNodeFaces = nodeFaces.WithGhostRows(nodeRoute);
    auto longerPath = G::AdjacencyRegistry::Compose(f2n, completeNodeFaces.Read());
    CHECK(longerPath.Read().Kind().Path() == std::vector<G::EntityKind>{G::EntityKind::Face, G::EntityKind::Node,
                                                                        G::EntityKind::Cell, G::EntityKind::Face});
    CHECK(longerPath.Read().GlobalRow(0) == std::vector<I>{mpi.rank * 2});
    CHECK(longerPath.Read().GlobalRow(1).empty());
    CHECK(longerPath.Read().Kind() != G::Adj::Face2Face);
    auto frameKind = exactKind;
    frameKind.periodic = G::RelationPeriodicPolicy::FrameMatched;
    CHECK(frameKind != exactKind);
    auto topologicalKind = exactKind;
    topologicalKind.order = G::RelationOrder::Topological;
    CHECK(topologicalKind != exactKind);
    auto allInverse = G::AdjacencyRegistry::Inverse(all.Read(), mpi);
    CHECK(allInverse.Read().Kind().selection == G::RelationSelection::All);
    CHECK(allInverse.Read().Kind().Path() == std::vector<G::EntityKind>{G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Face});
    CHECK(allInverse.Read().GlobalRow(0) == std::vector<I>{previous * 2});
    CHECK(allInverse.Read().GlobalRow(1).empty());
    CHECK(allInverse.Read().GlobalRow(2) == std::vector<I>{previous * 2});
    // Equal rows must not erase semantic version identity or hide a collective
    // mismatch before publication/traversal enters its payload collectives.
    auto version2 = exactKind;
    version2.semanticVersion = 2;
    G::MeshAdjacency other(version2, faces, cells, Rows(std::vector<std::vector<I>>{expected[3], {}}));
    CHECK(other.Read().Kind() != exact.Read().Kind());
    if (mpi.size > 1)
    {
        G::AdjacencyRegistry mismatch(mpi);
        mismatch.Register(mpi.rank == 0 ? exact.Read() : other.Read());
        CHECK_THROWS(mismatch.BuildGhosted({}));
        auto kind = mpi.rank == 0 ? exactKind : version2;
        G::GhostSpec spec{{{G::EntityKind::Face, {kind}, G::EntityKind::Cell}}};
        CHECK_THROWS(mismatch.Evaluate(G::CompiledGhostTree::compile(spec)));
    }
}

TEST_CASE("Native interpolation matches repeated canonical IDs after XOR permutes frame order")
{
    const auto mpi = MPI();
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == 0 ? 1 : 0);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? 2 : 0);
    auto cones = Rows<I>(mpi.rank == 0 ? std::vector<std::vector<I>>{{0, 0, 0}, {0, 0, 0}} : std::vector<std::vector<I>>{});
    auto bits = Rows<G::NodePeriodicBits>(mpi.rank == 0 ? std::vector<std::vector<G::NodePeriodicBits>>{{{0}, {1}, {2}}, {{1}, {0}, {3}}} : std::vector<std::vector<G::NodePeriodicBits>>{});
    G::MeshAdjacency relation(G::Adj::Cell2Node, cells, nodes, std::move(cones));
    auto read = relation.Read();
    G::RelationPeriodicBits periodic(read, std::move(bits));
    G::SubEntityQueryPbi query;
    query.numSubEntities = [](I)
    { return 1; };
    query.describe = [](I, int)
    { return G::SubEntityDesc{3, 3, G::Elem::Tri3}; };
    query.extractNodes = [](I, int, const std::function<I(int)> &get, I *output)
    { for (int slot = 0; slot < 3; ++slot) output[slot] = get(slot); };
    query.extractPbi = [](I, int, const std::function<G::NodePeriodicBits(int)> &get, G::NodePeriodicBits *output)
    { for (int slot = 0; slot < 3; ++slot) output[slot] = get(slot); };
    auto result = G::AdjacencyRegistry::Interpolate(read, G::EntityKind::Face, query, [mpi](const std::vector<I> &, const std::vector<DNDS::MPI_int> &owners, I)
                                                    { return G::OwnershipDecision{*std::min_element(owners.begin(), owners.end()) == mpi.rank, {}}; }, &periodic);
    CHECK(result.entities.NumGlobal() == 1);
    if (mpi.rank == 0)
    {
        CHECK(result.parent2entity.Read().GlobalRow(0) == result.parent2entity.Read().GlobalRow(1));
        auto labels = result.parent2entityPbi->Read();
        CHECK(uint8_t(labels.View()(0, 0)) == 0);
        CHECK(uint8_t(labels.View()(1, 0)) == 1);
        CHECK(result.entity2parent.Read().GlobalRow(0) == std::vector<I>{0, 1});
    }
}

TEST_CASE("Independent setup expansions preserve row order and join failures")
{
    std::vector<I> output{71}, expected = output;
    for (I iRow = 0; iRow < 64; ++iRow)
        for (I slot = 0; slot < iRow % 4; ++slot)
            expected.push_back(10 * iRow + slot);
    G::ConnectivityRows::AppendSetupRows(64, output, [](I iRow, const auto &emit)
                                         {
        for (I slot = 0; slot < iRow % 4; ++slot)
            emit(10 * iRow + slot); });
    CHECK(output == expected);
    CHECK_THROWS_AS(G::ConnectivityRows::AppendSetupRows(64, output, [](I iRow, const auto &emit)
                                                         {
        if (iRow == 17)
            throw std::runtime_error("expected local setup failure");
        emit(iRow); }),
                    std::runtime_error);
}

int main(int argc, char **argv)
{
    // Native row kernels may create OMP/CPU workers even with controller MPI.
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    const auto result = doctest::Context(argc, argv).run();
    DNDS::MPI::Finalize();
    return result;
}
