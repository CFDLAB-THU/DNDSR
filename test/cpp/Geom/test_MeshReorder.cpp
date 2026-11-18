// Pure permutation/field utility coverage. ArrayPair containers here are
// test fields and maps, never legacy mesh authority or geometry storage.
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Transition/ReorderPlan.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "NativeReorderChecks.hpp"
#include <numeric>
#include <set>
#include <optional>
using namespace DNDS;
using namespace DNDS::Geom;
using idx = DNDS::index;
static MPIInfo worldMPI()
{
    MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

TEST_CASE("classifyAdj basic classification")
{
    CHECK(classifyAdj(Adj::Cell2Cell, {}) == AdjAction::SKIP);
    CHECK(classifyAdj(Adj::Cell2Cell, {EntityKind::Cell}) == AdjAction::SELF);
    CHECK(classifyAdj(Adj::Cell2Node, {}) == AdjAction::SKIP);
    CHECK(classifyAdj(Adj::Cell2Node, {EntityKind::Cell}) == AdjAction::RELOCATE);
    CHECK(classifyAdj(Adj::Cell2Node, {EntityKind::Node}) == AdjAction::REMAP);
    CHECK(classifyAdj(Adj::Cell2Node, {EntityKind::Cell, EntityKind::Node}) == AdjAction::RELOCATE_REMAP);
}

TEST_CASE("ReorderRegistry register and query")
{
    auto mpi = worldMPI();
    ReorderRegistry reg;

    // Register a global mapping
    auto gm = make_ssp<GlobalOffsetsMapping>();
    gm->setMPIAlignBcast(mpi, 10);
    reg.registerGlobalMapping(EntityKind::Cell, gm);

    CHECK(reg.getGlobalMapping(EntityKind::Cell) == gm);
    CHECK(reg.getGlobalMapping(EntityKind::Node) == nullptr);

    // Register an adj
    bool remapCalled = false;
    bool relocateCalled = false;
    reg.registerAdj(
        Adj::Cell2Node,
        [&](const PermutationTransfer::LookupResult &)
        { remapCalled = true; },
        [&](const PermutationTransfer &, const MPIInfo &)
        { relocateCalled = true; },
        "cell2node");

    CHECK(reg.adjs.size() == 1);
    CHECK(reg.adjs[0].kind == Adj::Cell2Node);
    CHECK(reg.adjs[0].name == "cell2node");

    // Register a companion
    bool compCalled = false;
    reg.registerCompanion(
        EntityKind::Cell,
        [&](const PermutationTransfer &, const MPIInfo &)
        { compCalled = true; },
        "cellElemInfo");

    CHECK(reg.companions.size() == 1);
    CHECK(reg.companions[0].kind == EntityKind::Cell);
}

TEST_CASE("ReorderPlan::apply cell-only local permutation")
{
    auto mpi = worldMPI();
    const DNDS::index nCell = 8;
    const DNDS::index nNode = 4;

    // Create synthetic cell2node: each cell references 2 nodes (global)
    ArrayAdjacencyPair<2> cell2node;
    cell2node.InitPair("cell2node", mpi);
    cell2node.father->Resize(nCell);
    cell2node.father->createGlobalMapping();

    DNDS::index cellOffset = (*cell2node.father->pLGlobalMapping)(mpi.rank, 0);

    // Create synthetic node array
    ArrayAdjacencyPair<1> nodeArr;
    nodeArr.InitPair("nodeArr", mpi);
    nodeArr.father->Resize(nNode);
    nodeArr.father->createGlobalMapping();

    DNDS::index nodeOffset = (*nodeArr.father->pLGlobalMapping)(mpi.rank, 0);

    // Fill cell2node: cell i references nodes (i%nNode) and ((i+1)%nNode)
    for (DNDS::index i = 0; i < nCell; i++)
    {
        cell2node(i, 0) = nodeOffset + (i % nNode);
        cell2node(i, 1) = nodeOffset + ((i + 1) % nNode);
    }

    // Create a companion array (cellElemInfo analog)
    ArrayAdjacencyPair<1> cellInfo;
    cellInfo.InitPair("cellInfo", mpi);
    cellInfo.father->Resize(nCell);
    for (DNDS::index i = 0; i < nCell; i++)
        cellInfo(i, 0) = 1000 + cellOffset + i; // tag = 1000 + global

    // Build registry
    ReorderRegistry reg;
    reg.registerGlobalMapping(EntityKind::Cell, cell2node.father->pLGlobalMapping);
    reg.registerGlobalMapping(EntityKind::Node, nodeArr.father->pLGlobalMapping);

    reg.registerAdj(
        Adj::Cell2Node,
        nullptr, // no remap needed (only Cell reordered, not Node)
        [&](const PermutationTransfer &t, const MPIInfo &m)
        { t.transferRows(cell2node, m); },
        "cell2node");

    reg.registerCompanion(
        EntityKind::Cell,
        [&](const PermutationTransfer &t, const MPIInfo &m)
        { t.transferRows(cellInfo, m); },
        "cellInfo");

    // Build plan: reverse cell permutation (all local)
    std::vector<MPI_int> cellPartition(nCell, mpi.rank);
    ReorderInput input;
    input.explicitMaps.push_back(EntityReorderMap{EntityKind::Cell, cellPartition});

    auto plan = ReorderPlan::build(input, reg, mpi);
    CHECK(plan.isLocalOnly);
    CHECK(plan.reorderedKinds.count(EntityKind::Cell));
    CHECK_FALSE(plan.reorderedKinds.count(EntityKind::Node));

    // Apply
    plan.apply(reg, mpi);

    // Since partition = all-self and ordering is preserved within rank,
    // the data should be unchanged (identity permutation via fromPartition).
    for (DNDS::index i = 0; i < nCell; i++)
    {
        CHECK(cell2node(i, 0) == nodeOffset + (i % nNode));
        CHECK(cell2node(i, 1) == nodeOffset + ((i + 1) % nNode));
        CHECK(cellInfo(i, 0) == 1000 + cellOffset + i);
    }
}

TEST_CASE("ReorderPlan::apply node-only remap")
{
    auto mpi = worldMPI();
    const DNDS::index nCell = 4;
    const DNDS::index nNode = 6;

    // cell2node: Cell->Node (cell rows fixed, node entries need remapping)
    ArrayAdjacencyPair<2> cell2node;
    cell2node.InitPair("cell2node", mpi);
    cell2node.father->Resize(nCell);
    cell2node.father->createGlobalMapping();

    ArrayAdjacencyPair<1> nodeArr;
    nodeArr.InitPair("nodeArr", mpi);
    nodeArr.father->Resize(nNode);
    nodeArr.father->createGlobalMapping();

    DNDS::index nodeOffset = (*nodeArr.father->pLGlobalMapping)(mpi.rank, 0);

    // Fill: cell i refs nodes i and i+1
    for (DNDS::index i = 0; i < nCell; i++)
    {
        cell2node(i, 0) = nodeOffset + i;
        cell2node(i, 1) = nodeOffset + i + 1;
    }

    // Node companion: coords analog
    ArrayAdjacencyPair<1> coords;
    coords.InitPair("coords", mpi);
    coords.father->Resize(nNode);
    for (DNDS::index i = 0; i < nNode; i++)
        coords(i, 0) = 500 + nodeOffset + i; // value = 500 + globalNode

    // Build registry
    ReorderRegistry reg;
    reg.registerGlobalMapping(EntityKind::Cell, cell2node.father->pLGlobalMapping);
    reg.registerGlobalMapping(EntityKind::Node, nodeArr.father->pLGlobalMapping);

    reg.registerAdj(
        Adj::Cell2Node,
        [&](const PermutationTransfer::LookupResult &lookup)
        {
            for (DNDS::index i = 0; i < nCell; i++)
                for (rowsize j = 0; j < 2; j++)
                {
                    DNDS::index &v = cell2node(i, j);
                    if (v != UnInitIndex)
                        v = lookup.resolve(v);
                }
        },
        nullptr, // no relocate (Cell not reordered)
        "cell2node");

    reg.registerCompanion(
        EntityKind::Node,
        [&](const PermutationTransfer &t, const MPIInfo &m)
        { t.transferRows(coords, m); },
        "coords");

    // Reorder nodes: all stay local (identity partition)
    std::vector<MPI_int> nodePartition(nNode, mpi.rank);
    ReorderInput input;
    input.explicitMaps.push_back(EntityReorderMap{EntityKind::Node, nodePartition});

    auto plan = ReorderPlan::build(input, reg, mpi);
    CHECK(plan.isLocalOnly);
    CHECK(plan.reorderedKinds.count(EntityKind::Node));
    CHECK_FALSE(plan.reorderedKinds.count(EntityKind::Cell));

    // Apply
    plan.apply(reg, mpi);

    // With identity partition (all stay on same rank), new globals are
    // contiguous starting at newGlobalOffsets[mpi.rank].
    // For identity: newGlobalIndices[i] = nodeOffset + i (unchanged).
    // So remap should be identity, coords should be unchanged.
    for (DNDS::index i = 0; i < nCell; i++)
    {
        // Since it's identity partition, old globals map to same new globals
        auto &transfer = plan.transfers.at(EntityKind::Node);
        DNDS::index expectedNode0 = transfer.newGlobalIndices[i];
        DNDS::index expectedNode1 = transfer.newGlobalIndices[i + 1];
        CHECK(cell2node(i, 0) == expectedNode0);
        CHECK(cell2node(i, 1) == expectedNode1);
    }

    for (DNDS::index i = 0; i < nNode; i++)
        CHECK(coords(i, 0) == 500 + nodeOffset + i);
}

TEST_CASE("PermutationTransfer + buildLookup: reverse permutation value tracking")
{
    auto mpi = worldMPI();
    const DNDS::index nLocal = 6;

    // Build a simple cell2node-like adj with known values
    ArrayAdjacencyPair<2> cell2node;
    cell2node.InitPair("cell2node", mpi);
    cell2node.father->Resize(nLocal);
    cell2node.father->createGlobalMapping();

    DNDS::index cellOffset = (*cell2node.father->pLGlobalMapping)(mpi.rank, 0);

    // Fill with tracker values: cell i stores (100 + cellOffset + i, 200 + cellOffset + i)
    for (DNDS::index i = 0; i < nLocal; i++)
    {
        cell2node(i, 0) = 100 + cellOffset + i;
        cell2node(i, 1) = 200 + cellOffset + i;
    }

    // Snapshot original values for verification
    std::vector<DNDS::index> origCol0(nLocal), origCol1(nLocal);
    for (DNDS::index i = 0; i < nLocal; i++)
    {
        origCol0[i] = cell2node(i, 0);
        origCol1[i] = cell2node(i, 1);
    }

    // Build a reverse permutation: old i -> new (nLocal-1-i)
    std::vector<DNDS::index> old2new(nLocal);
    for (DNDS::index i = 0; i < nLocal; i++)
        old2new[i] = nLocal - 1 - i;

    auto pt = PermutationTransfer::fromLocalPermutation(
        old2new, cell2node.father->pLGlobalMapping, mpi);
    CHECK(pt.isLocalOnly);

    // Verify newGlobalIndices[i] = myOffset + old2new[i] = cellOffset + (nLocal-1-i)
    for (DNDS::index i = 0; i < nLocal; i++)
        CHECK(pt.newGlobalIndices[i] == cellOffset + (nLocal - 1 - i));

    // Transfer rows: after permutation, row (nLocal-1-i) should contain old row i's data
    pt.transferRows(cell2node, mpi);

    for (DNDS::index i = 0; i < nLocal; i++)
    {
        DNDS::index newSlot = old2new[i];
        CHECK(cell2node(newSlot, 0) == origCol0[i]);
        CHECK(cell2node(newSlot, 1) == origCol1[i]);
    }

    // Build lookup and verify resolve() produces the correct new globals
    auto lookup = pt.buildLookup({}, mpi);

    for (DNDS::index i = 0; i < nLocal; i++)
    {
        DNDS::index oldGlobal = cellOffset + i;
        DNDS::index expectedNewGlobal = cellOffset + (nLocal - 1 - i);
        CHECK(lookup.resolve(oldGlobal) == expectedNewGlobal);
    }
}

TEST_CASE("PermutationTransfer distributed value tracking cross-rank")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return;

    const DNDS::index nLocal = 4;

    ArrayAdjacencyPair<1> arr;
    arr.InitPair("arr", mpi);
    arr.father->Resize(nLocal);
    arr.father->createGlobalMapping();

    DNDS::index myOffset = (*arr.father->pLGlobalMapping)(mpi.rank, 0);
    DNDS::index globalSize = arr.father->pLGlobalMapping->globalSize();

    // Tag each entry with a unique global-based sentinel
    const DNDS::index TAG_BASE = 100000;
    for (DNDS::index i = 0; i < nLocal; i++)
        arr(i, 0) = TAG_BASE + myOffset + i;

    // Round-robin partition: entry i goes to rank (i % mpi.size)
    std::vector<MPI_int> partition(nLocal);
    for (DNDS::index i = 0; i < nLocal; i++)
        partition[i] = static_cast<MPI_int>(i % mpi.size);

    auto pt = PermutationTransfer::fromPartition(
        partition, arr.father->pLGlobalMapping, mpi);
    CHECK_FALSE(pt.isLocalOnly);

    pt.transferRows(arr, mpi);

    // After transfer: every entry on this rank carries a valid tag
    DNDS::index nAfter = arr.father->Size();
    std::set<DNDS::index> receivedTags;
    for (DNDS::index i = 0; i < nAfter; i++)
    {
        DNDS::index v = arr(i, 0);
        CHECK(v >= TAG_BASE);
        CHECK(v < TAG_BASE + globalSize);
        receivedTags.insert(v);
    }
    // All received tags are unique (no duplicates)
    CHECK(receivedTags.size() == static_cast<size_t>(nAfter));

    // Sum of nAfter across all ranks must equal original global size
    DNDS::index totalAfter = 0;
    MPI_Allreduce(&nAfter, &totalAfter, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
    CHECK(totalAfter == globalSize);
}

TEST_CASE("PermutationTransfer::buildLookup cross-rank resolve with pullSet")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return;

    const DNDS::index nLocal = 4;

    auto gm = make_ssp<GlobalOffsetsMapping>();
    gm->setMPIAlignBcast(mpi, nLocal);

    DNDS::index myOffset = (*gm)(mpi.rank, 0);

    // Round-robin partition: entry i stays if i%size==myRank, else moves
    std::vector<MPI_int> partition(nLocal);
    for (DNDS::index i = 0; i < nLocal; i++)
        partition[i] = static_cast<MPI_int>((mpi.rank + i) % mpi.size);

    auto pt = PermutationTransfer::fromPartition(partition, gm, mpi);

    // Build a pull set: globals on the next rank
    int nextRank = (mpi.rank + 1) % mpi.size;
    DNDS::index nextOffset = (*gm)(nextRank, 0);
    std::vector<DNDS::index> pullSet;
    for (DNDS::index i = 0; i < nLocal; i++)
        pullSet.push_back(nextOffset + i);
    std::sort(pullSet.begin(), pullSet.end());

    auto lookup = pt.buildLookup(pullSet, mpi);

    // For each old global in pullSet, resolve should give some valid new global
    // within [0, globalSize).
    DNDS::index globalSize = gm->globalSize();
    for (auto oldG : pullSet)
    {
        DNDS::index newG = lookup.resolve(oldG);
        CHECK(newG >= 0);
        CHECK(newG < globalSize);
    }

    // Verify my own locals resolve too
    for (DNDS::index i = 0; i < nLocal; i++)
    {
        DNDS::index oldG = myOffset + i;
        DNDS::index newG = lookup.resolve(oldG);
        CHECK(newG >= 0);
        CHECK(newG < globalSize);
    }

    // UnInitIndex passthrough
    CHECK(lookup.resolve(UnInitIndex) == UnInitIndex);
}

TEST_CASE("Native field uses exact local permutation and preserves old aliases")
{
    auto mpi = worldMPI();
    ArrayAdjacencyPair<1> entity;
    entity.InitPair("localField.entities", mpi);
    entity.father->Resize(3);
    entity.father->createGlobalMapping();
    auto permutation = PermutationTransfer::fromLocalPermutation({2, 0, 1}, entity.father->pLGlobalMapping, mpi);
    auto idsSlot = std::make_shared<Runtime::Field<int64_t>>(Runtime::Field<int64_t>({3, 1, 1}, {9, 3, 3, 0}, "cell"));
    auto &ids = *idsSlot;
    {
        auto edit = ids.Write();
        for (size_t i = 0; i < 3; ++i)
            edit.View<int64_t>()[i] = 100 * mpi.rank + int64_t(i);
    }
    auto old = ids;
    RegisterField(idsSlot, EntityKind::Cell, "ids").fn(permutation, mpi);
    CHECK(ids.GetDistribution().epoch == 4);
    CHECK(old.GetDistribution().epoch == 3);
    CHECK(ids.Read().View<int64_t>()[0] == 100 * mpi.rank + 1);
    CHECK(ids.Read().View<int64_t>()[1] == 100 * mpi.rank + 2);
    CHECK(ids.Read().View<int64_t>()[2] == 100 * mpi.rank);
    CHECK(old.Read().View<int64_t>()[0] == 100 * mpi.rank);
}

TEST_CASE("Explicit transfer layout preserves static stride and rejects one-rank capacity errors before publication")
{
    namespace R = Runtime;
    const auto mpi = worldMPI();
    auto mapping = std::make_shared<GlobalOffsetsMapping>();
    mapping->setMPIAlignBcast(mpi, 3);
    for (bool distributed : {false, true})
    {
        auto transfer = distributed ? PermutationTransfer::fromPartition(std::vector<MPI_int>(3, (mpi.rank + 1) % mpi.size), mapping, mpi)
                                    : PermutationTransfer::fromLocalPermutation({2, 0, 1}, mapping, mpi);
        auto fieldSlot = std::make_shared<R::Array<double>>(R::Array<double>(R::RowLayout::Dense(3, 3, 5)));
        auto &field = *fieldSlot;
        {
            auto write = field.Write();
            for (DNDS::index iRow = 0; iRow < 3; ++iRow)
                for (DNDS::index j = 0; j < 3; ++j)
                    write.View<R::FixedRows<3, 5>>()(iRow, j) = 1000 * mpi.rank + 10 * iRow + j;
        }
        const auto identity = field.Identity();
        auto bad = RegisterArray(fieldSlot, EntityKind::Cell, "bad-capacity", R::ElementSchema::Arithmetic<double>(),
                                 R::ExecutionContext::Host(), CompanionDisposition::TransferIfReordered,
                                 R::TransferLayout::Padded(mpi.rank == 0 ? 2 : 5));
        CHECK_THROWS(bad.fn(transfer, mpi));
        CHECK(field.Identity() == identity);
        CHECK(field.Read().View()(1, 2) == 1000 * mpi.rank + 12);
        auto good = RegisterArray(fieldSlot, EntityKind::Cell, "strided-state", R::ElementSchema::Arithmetic<double>(),
                                  R::ExecutionContext::Host(), CompanionDisposition::TransferIfReordered,
                                  R::TransferLayout::Table(3, 5));
        good.fn(transfer, mpi);
        auto read = field.Read();
        const auto view = read.View<R::FixedRows<3, 5>, true>();
        for (DNDS::index iRow = 0; iRow < 3; ++iRow)
            for (DNDS::index j = 0; j < 3; ++j)
            {
                const auto owner = distributed ? (mpi.rank + mpi.size - 1) % mpi.size : mpi.rank;
                const auto oldRow = distributed ? iRow : (iRow + 1) % 3;
                CHECK(view(iRow, j) == 1000 * owner + 10 * oldRow + j);
            }
    }
}

TEST_CASE("Staged field bundle preserves every source when a later preparation fails")
{
    auto mpi = worldMPI();
    ArrayAdjacencyPair<1> entity;
    entity.InitPair("staged.entities", mpi);
    entity.father->Resize(3);
    entity.father->createGlobalMapping();
    auto transfer = PermutationTransfer::fromLocalPermutation({2, 0, 1}, entity.father->pLGlobalMapping, mpi);
    auto historySlot = std::make_shared<Runtime::Field<int64_t>>(Runtime::Field<int64_t>({3, 1, 1}, {17, 5, 3, 0}, "cell"));
    auto &history = *historySlot;
    auto stagesSlot = std::make_shared<Runtime::Array<int64_t>>(Runtime::Array<int64_t>(Runtime::RowLayout::Dense(3, 1)));
    auto &stages = *stagesSlot;
    for (size_t i = 0; i < 3; ++i)
    {
        history.Write().View<int64_t>()[i] = 100 * mpi.rank + int64_t(i);
        stages.Write().At(i, 0) = 1000 + 100 * mpi.rank + int64_t(i);
    }
    const auto historyIdentity = history.Values().Identity();
    const auto stageIdentity = stages.Identity();
    auto first = RegisterField(historySlot, EntityKind::Cell, "history");
    auto second = RegisterArray(stagesSlot, EntityKind::Cell, "stages");
    int preparedFirst = 0;
    auto prepareFirst = first.prepare;
    first.prepare = [&](const auto &permutation, const auto &m)
    {
        ++preparedFirst;
        return prepareFirst(permutation, m);
    };
    second.prepare = [&](const auto &, const auto &m) -> std::unique_ptr<PreparedCompanion>
    {
        // The callback owns its bounded local preflight. The bundle must not
        // wrap an arbitrary MPI-containing callback in another catch/agreement.
        DNDS::MPI::LocalStage(m, [&]
                              {
            if (m.rank == 0)
                throw std::runtime_error("injected second-field prepare failure"); });
        return std::make_unique<FieldTransferDetail::Prepared<Runtime::Array<int64_t>>>(stagesSlot, stages);
    };
    ReorderRegistry registry;
    registry.companions = {first, second};
    ReorderPlan plan;
    plan.reorderedKinds.insert(EntityKind::Cell);
    plan.transfers.emplace(EntityKind::Cell, transfer);
    CHECK_THROWS(plan.apply(registry, mpi));
    CHECK(preparedFirst == 1);
    CHECK(history.Values().Identity() == historyIdentity);
    CHECK(stages.Identity() == stageIdentity);
    CHECK(history.GetDistribution().epoch == 5);
    for (size_t i = 0; i < 3; ++i)
    {
        CHECK(history.Read().View<int64_t>()[i] == 100 * mpi.rank + int64_t(i));
        CHECK(stages.Read().At(i, 0) == 1000 + 100 * mpi.rank + int64_t(i));
    }
}

TEST_CASE("Staged Array registration rejects duplicate destinations and permits distinct old aliases")
{
    auto mpi = worldMPI();
    ArrayAdjacencyPair<1> entity;
    entity.InitPair("staged.alias.entities", mpi);
    entity.father->Resize(2);
    entity.father->createGlobalMapping();
    ReorderPlan plan;
    plan.reorderedKinds.insert(EntityKind::Cell);
    plan.transfers.emplace(EntityKind::Cell,
                           PermutationTransfer::fromLocalPermutation({1, 0}, entity.father->pLGlobalMapping, mpi));
    auto valuesSlot = std::make_shared<Runtime::Array<int64_t>>(Runtime::Array<int64_t>(Runtime::RowLayout::Dense(2, 1)));
    auto &values = *valuesSlot;
    values.Write().At(0, 0) = 11;
    values.Write().At(1, 0) = 22;
    auto sameObjectSlot = valuesSlot;
    auto &sameObject = *sameObjectSlot;
    ReorderRegistry duplicate;
    duplicate.companions = {RegisterArray(valuesSlot, EntityKind::Cell, "first"),
                            RegisterArray(sameObjectSlot, EntityKind::Cell, "second")};
    const auto oldIdentity = values.Identity();
    CHECK_THROWS(plan.apply(duplicate, mpi));
    CHECK(values.Identity() == oldIdentity);
    CHECK(values.Read().At(0, 0) == 11);

    auto distinctHandleSlot = std::make_shared<std::decay_t<decltype(values)>>(values);

    auto &distinctHandle = *distinctHandleSlot;
    ReorderRegistry aliases;
    aliases.companions = {RegisterArray(valuesSlot, EntityKind::Cell, "first"),
                          RegisterArray(distinctHandleSlot, EntityKind::Cell, "second")};
    plan.apply(aliases, mpi);
    CHECK(values.Identity() != oldIdentity);
    CHECK(distinctHandle.Identity() != oldIdentity);
    CHECK(values.Read().At(0, 0) == 22);
    CHECK(distinctHandle.Read().At(0, 0) == 22);
}

TEST_CASE("ReorderPlan::apply RELOCATE_REMAP (both source and target reordered)")
{
    auto mpi = worldMPI();
    const DNDS::index nCell = 4;
    const DNDS::index nNode = 6;

    // cell2node: Cell -> Node. Both will be reordered.
    ArrayAdjacencyPair<2> cell2node;
    cell2node.InitPair("cell2node", mpi);
    cell2node.father->Resize(nCell);
    cell2node.father->createGlobalMapping();

    ArrayAdjacencyPair<1> nodeArr;
    nodeArr.InitPair("nodeArr", mpi);
    nodeArr.father->Resize(nNode);
    nodeArr.father->createGlobalMapping();

    DNDS::index cellOffset = (*cell2node.father->pLGlobalMapping)(mpi.rank, 0);
    DNDS::index nodeOffset = (*nodeArr.father->pLGlobalMapping)(mpi.rank, 0);

    // Fill: cell i references nodes i and i+1 (in local indexing converted to global)
    for (DNDS::index i = 0; i < nCell; i++)
    {
        cell2node(i, 0) = nodeOffset + i;
        cell2node(i, 1) = nodeOffset + i + 1;
    }

    // Snapshot pre-reorder cell2node for each cell by its old global
    std::map<DNDS::index, std::array<DNDS::index, 2>> oldCell2NodeByGlobal;
    for (DNDS::index i = 0; i < nCell; i++)
        oldCell2NodeByGlobal[cellOffset + i] = {cell2node(i, 0), cell2node(i, 1)};

    // Build registry
    ReorderRegistry reg;
    reg.registerGlobalMapping(EntityKind::Cell, cell2node.father->pLGlobalMapping);
    reg.registerGlobalMapping(EntityKind::Node, nodeArr.father->pLGlobalMapping);

    reg.registerAdj(
        Adj::Cell2Node,
        [&](const PermutationTransfer::LookupResult &lookup)
        {
            for (DNDS::index i = 0; i < cell2node.father->Size(); i++)
                for (rowsize j = 0; j < 2; j++)
                {
                    DNDS::index &v = cell2node(i, j);
                    if (v != UnInitIndex)
                        v = lookup.resolve(v);
                }
        },
        [&](const PermutationTransfer &t, const MPIInfo &m)
        { t.transferRows(cell2node, m); },
        "cell2node");

    // Identity partitions (local-only) but with REMAP logic exercised
    std::vector<MPI_int> cellPartition(nCell, mpi.rank);
    std::vector<MPI_int> nodePartition(nNode, mpi.rank);

    ReorderInput input;
    input.explicitMaps.push_back(EntityReorderMap{EntityKind::Cell, cellPartition});
    input.explicitMaps.push_back(EntityReorderMap{EntityKind::Node, nodePartition});

    auto plan = ReorderPlan::build(input, reg, mpi);
    CHECK(plan.reorderedKinds.count(EntityKind::Cell));
    CHECK(plan.reorderedKinds.count(EntityKind::Node));
    CHECK(classifyAdj(Adj::Cell2Node, plan.reorderedKinds) == AdjAction::RELOCATE_REMAP);

    plan.apply(reg, mpi);

    // With identity partitions, new globals == old globals, so values should be unchanged.
    for (DNDS::index i = 0; i < nCell; i++)
    {
        CHECK(cell2node(i, 0) == nodeOffset + i);
        CHECK(cell2node(i, 1) == nodeOffset + i + 1);
    }
}

TEST_CASE("NativeRowTransfer exact native routes with empty ranks and skew")
{
    const auto mpi = worldMPI();
    const auto count = [&](int rank) -> DNDS::index
    { return mpi.size > 1 && rank == mpi.size - 1 ? 0 : 3 + rank; };
    const DNDS::index n = count(mpi.rank);
    auto mapping = make_ssp<GlobalOffsetsMapping>();
    mapping->setMPIAlignBcast(mpi, n);
    SUBCASE("local reversed permutation")
    {
        std::vector<DNDS::index> permutation(n);
        for (DNDS::index i = 0; i < n; ++i)
            permutation[i] = n - i - 1;
        auto transfer = PermutationTransfer::fromLocalPermutation(permutation, mapping, mpi);
        auto route = NativeRowTransfer(transfer, mpi);
        Runtime::Array<int64_t> source(Runtime::RowLayout::Dense(n, 1));
        Runtime::Array<int64_t> destination(Runtime::RowLayout::Dense(n, 1));
        {
            auto write = source.Write();
            for (DNDS::index i = 0; i < n; ++i)
                write.At(i, 0) = 1000 * mpi.rank + i;
        }
        Runtime::PayloadPlan(route, source.Layout(), destination.Layout(), Runtime::ElementSchema::Arithmetic<int64_t>())
            .Bind(source, destination)
            .Begin()
            .Wait();
        auto read = destination.Read();
        for (DNDS::index i = 0; i < n; ++i)
            CHECK(read.At(i, 0) == 1000 * mpi.rank + n - i - 1);
        if (mpi.rank == 0 && n > 1)
            transfer.localOld2New[1] = transfer.localOld2New[0];
        CHECK_THROWS(NativeRowTransfer(transfer, mpi));
    }
    SUBCASE("distributed sender order and local slot order")
    {
        const auto target = [&](int rank, DNDS::index i)
        { return int(i % 3 == 0 ? (rank + 1) % mpi.size : 0); };
        std::vector<MPI_int> partition(n);
        for (DNDS::index i = 0; i < n; ++i)
            partition[i] = target(mpi.rank, i);
        auto transfer = PermutationTransfer::fromPartition(partition, mapping, mpi);
        auto route = NativeRowTransfer(transfer, mpi);
        std::vector<int64_t> expected;
        for (int rank = 0; rank < mpi.size; ++rank)
            for (DNDS::index i = 0; i < count(rank); ++i)
                if (target(rank, i) == mpi.rank)
                    expected.push_back(1000 * rank + i);
        CHECK(route.DestinationRows() == expected.size());
        Runtime::Array<int64_t> source(Runtime::RowLayout::Dense(n, 1));
        Runtime::Array<int64_t> destination(Runtime::RowLayout::Dense(expected.size(), 1));
        {
            auto write = source.Write();
            for (DNDS::index i = 0; i < n; ++i)
                write.At(i, 0) = 1000 * mpi.rank + i;
        }
        Runtime::PayloadPlan(route, source.Layout(), destination.Layout(), Runtime::ElementSchema::Arithmetic<int64_t>())
            .Bind(source, destination)
            .Begin()
            .Wait();
        auto read = destination.Read();
        for (size_t i = 0; i < expected.size(); ++i)
            CHECK(read.At(i, 0) == expected[i]);
        if (mpi.size > 1)
        {
            if (mpi.rank == 0)
                transfer.pushStart.back() += 1;
            CHECK_THROWS(NativeRowTransfer(transfer, mpi));
        }
    }
    SUBCASE("collective mode rejection")
    {
        PermutationTransfer transfer;
        transfer.isLocalOnly = mpi.rank == 0;
        if (mpi.size > 1)
            CHECK_THROWS(NativeRowTransfer(transfer, mpi));
    }
}

namespace T = NativeReorderChecks;

static void checkSparseTargets(const T::State &source, const MPIInfo &mpi, const std::vector<AdjacencyDescriptor> &keys)
{
    auto mesh = UnstructuredMesh::FromNative(mpi, source);
    for (const auto &key : keys)
    {
        auto read = MeshAdjacencyRead::Current(*mesh, key, source->Requirements());
        std::set<DNDS::index> required;
        for (DNDS::index row = 0; row < read.NumOwned(); ++row)
            for (size_t slot = 0; slot < read.RowSize(row); ++slot)
            {
                const auto entry = read.ReadEntry(row, slot);
                if (entry.absent)
                    continue;
                CHECK(entry.global >= 0);
                CHECK(entry.global < read.Target().NumGlobal());
                if (read.Target().Owner(entry.global) != mpi.rank)
                    required.insert(entry.global);
            }
        DNDS::index previous = -1;
        for (const auto global : required)
        {
            CHECK(global > previous);
            CHECK(read.Target().Owner(global) != mpi.rank);
            REQUIRE(read.Target().GlobalToLocal(global).has_value());
            previous = global;
        }
    }
}

TEST_CASE("Audit regression: local cell reorder retains periodic face rows")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true, false, true);
    auto expected = T::RecordAll(source);
    auto retained = source->Snapshot(Runtime::Device::Host()).AcquireRead();
    auto result = NativeMeshTransition::ReorderLocalCells(*source, mpi, 2, 1);
    T::Check(source, result, {}, mpi, expected);
    CHECK(retained.HostView().IsPeriodic());
}

TEST_CASE("ReorderEntities cell-only local on UniformSquare_10")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi);
    auto input = T::Place(source, EntityKind::Cell, mpi);
    auto result = T::Move(source, input, mpi);
    CHECK(result.state->NumOwned(EntityKind::Cell) == source->NumOwned(EntityKind::Cell));
    T::RoundTrip(result.state, mpi, {Adj::Cell2Node, Adj::Cell2Cell, Adj::Bnd2Cell});
}

TEST_CASE("ReorderEntities cell-only with face destruction on UniformSquare_10")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true);
    auto input = T::Place(source, EntityKind::Cell, mpi);
    input.destroyKinds = {EntityKind::Face};
    auto result = T::Move(source, input, mpi);
    CHECK(result.state->HasEntities(EntityKind::Face));
    CHECK(result.state->NextEntityId(EntityKind::Face) > source->NextEntityId(EntityKind::Face));
    T::RoundTrip(result.state, mpi, {Adj::Cell2Face, Adj::Face2Node, Adj::Face2Cell});
}

TEST_CASE("ReorderEntities cell distributed round-robin with follow")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return;
    auto source = T::Fixture(mpi);
    T::Follow(source, mpi, EntityKind::Cell, {{EntityKind::Node, Adj::Node2Cell}, {EntityKind::Bnd, Adj::Bnd2Cell}});
}

TEST_CASE("ReorderEntities node-only local on UniformSquare_10")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi);
    auto result = T::Move(source, T::Place(source, EntityKind::Node, mpi), mpi);
    CHECK(result.state->NumOwned(EntityKind::Node) == source->NumOwned(EntityKind::Node));
    CHECK(result.state->NumOwned(EntityKind::Cell) == source->NumOwned(EntityKind::Cell));
}

TEST_CASE("ReorderEntities with external companion array (solver-like)")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi);
    auto mesh = UnstructuredMesh::FromNative(mpi, source);
    const auto count = size_t(mesh->NumCell());
    auto solverDOFSlot = std::make_shared<Runtime::Array<int64_t>>(Runtime::Array<int64_t>(Runtime::RowLayout::Dense(count, 3, 5)));
    auto &solverDOF = *solverDOFSlot;
    auto stableIdsSlot = std::make_shared<Runtime::Field<int64_t>>(Runtime::Field<int64_t>({count, 1, 1}, {7, 0, count, 0}, "cell"));
    auto &stableIds = *stableIdsSlot;
    auto historySlot = std::make_shared<Runtime::Field<double>>(Runtime::Field<double>({count, 2, 2}, {7, 0, count, 0}, "cell"));
    auto &history = *historySlot;
    std::vector<size_t> widths(count);
    for (size_t row = 0; row < count; ++row)
        widths[row] = size_t(source->OriginalEntityId(EntityKind::Cell, DNDS::index(row)) % 3);
    auto raggedSlot = std::make_shared<Runtime::Array<int64_t>>(Runtime::Array<int64_t>(Runtime::RowLayout::Ragged(widths)));
    auto &ragged = *raggedSlot;
    {
        auto dof = solverDOF.Write();
        auto ids = stableIds.Write();
        auto records = history.Write();
        auto extra = ragged.Write();
        for (size_t row = 0; row < count; ++row)
        {
            const auto id = source->OriginalEntityId(EntityKind::Cell, DNDS::index(row));
            ids.View<int64_t>()[row] = id;
            records.View<double>()[2 * row] = double(id) + .25;
            records.View<double>()[2 * row + 1] = double(id) + .5;
            for (size_t slot = 0; slot < 3; ++slot)
                dof.View()(row, slot) = 10 * id + DNDS::index(slot);
            for (size_t slot = 0; slot < widths[row]; ++slot)
                extra.View()(row, slot) = 10 * id + DNDS::index(slot);
        }
    }
    const auto oldDof = solverDOF, oldRagged = ragged;
    const auto oldDofIdentity = solverDOF.Identity(), oldRaggedIdentity = ragged.Identity();
    const auto oldHistory = history;
    std::vector<CompanionEntry> fields{RegisterArray(solverDOFSlot, EntityKind::Cell, "solverDOF"),
                                       RegisterField(stableIdsSlot, EntityKind::Cell, "stableIds"), RegisterField(historySlot, EntityKind::Cell, "history"),
                                       RegisterArray(raggedSlot, EntityKind::Cell, "nativeRagged")};
    MeshTransitionHandle handle(mesh);
    auto input = T::Place(source, EntityKind::Cell, mpi, 3);
    if (mpi.size > 1)
    {
        auto mismatch = fields;
        if (mpi.rank == 0)
            mismatch.back().name = "different-registration";
        CHECK_THROWS(PreparedMeshTransition::Prepare(handle, input, mismatch));
        CHECK(handle.Get() == mesh);
        CHECK(solverDOF.Identity() == oldDofIdentity);
        CHECK(ragged.Identity() == oldRaggedIdentity);
    }
    auto prepared = PreparedMeshTransition::Prepare(handle, input, fields);
    CHECK(handle.Get() == mesh);
    CHECK(solverDOF.Identity() == oldDofIdentity);
    prepared.Publish();
    auto after = handle.Get()->CurrentNativeState();
    auto ids = stableIds.Read();
    auto records = history.Read();
    auto values = solverDOF.Read();
    auto extra = ragged.Read();
    CHECK(history.GetDistribution().epoch == oldHistory.GetDistribution().epoch + 1);
    for (size_t row = 0; row < size_t(after->NumOwned(EntityKind::Cell)); ++row)
    {
        const auto id = after->OriginalEntityId(EntityKind::Cell, DNDS::index(row));
        CHECK(ids.View<int64_t>()[row] == id);
        CHECK(records.View<double>()[2 * row] == double(id) + .25);
        CHECK(records.View<double>()[2 * row + 1] == double(id) + .5);
        for (size_t slot = 0; slot < 3; ++slot)
            CHECK(values.View()(row, slot) == 10 * id + DNDS::index(slot));
        REQUIRE(extra.View().RowSize(row) == size_t(id % 3));
        for (size_t slot = 0; slot < extra.View().RowSize(row); ++slot)
            CHECK(extra.View()(row, slot) == 10 * id + DNDS::index(slot));
    }
    CHECK(oldDof.Identity() == oldDofIdentity);
    CHECK(oldRagged.Identity() == oldRaggedIdentity);
    CHECK(oldDof.RowCount() == count);
    CHECK(oldHistory.GetDistribution().epoch == 0);
}

TEST_CASE("Staged field attachment rejection precedes mesh mutation")
{
    auto mpi = worldMPI();
    auto mesh = UnstructuredMesh::FromNative(mpi, T::Fixture(mpi));
    const auto count = size_t(mesh->NumCell());
    auto firstSlot = std::make_shared<Runtime::Field<int64_t>>(Runtime::Field<int64_t>({count, 1, 1}, {7, 2, count, 0}, "cell"));
    auto &first = *firstSlot;
    auto secondSlot = std::make_shared<Runtime::Field<int64_t>>(Runtime::Field<int64_t>({count, 1, 1}, {7, 2, count, 0}, "cell"));
    auto &second = *secondSlot;
    auto a = RegisterField(firstSlot, EntityKind::Cell, "first");
    auto b = RegisterField(secondSlot, EntityKind::Cell, "second");
    SUBCASE("rank-local association mismatch")
    {
        if (mpi.rank == 0)
            b = RegisterField(secondSlot, EntityKind::Node, "second");
    }
    SUBCASE("rank-local disposition mismatch")
    {
        if (mpi.rank == 0)
            b.disposition = CompanionDisposition::ProhibitReorder;
    }
    SUBCASE("same destination handle with different names")
    {
        auto aliasSlot = firstSlot;
        auto &alias = *aliasSlot;
        b = RegisterField(aliasSlot, EntityKind::Cell, "second");
    }
    SUBCASE("same entity domain mismatch")
    {
        second = Runtime::Field<int64_t>({count, 1, 1}, {8, 2, count, 0}, "cell");
    }
    SUBCASE("same entity epoch mismatch")
    {
        second = Runtime::Field<int64_t>({count, 1, 1}, {7, 3, count, 0}, "cell");
    }
    SUBCASE("rank-local second preparation failure")
    {
        b.prepare = [&](const auto &, const auto &m) -> std::unique_ptr<PreparedCompanion>
        {
            // Recover only a bounded local preflight owned by the callback.
            // The outer transition cannot catch arbitrary collective work.
            DNDS::MPI::LocalStage(m, [&]
                                  {
                if (m.rank == 0)
                    throw std::runtime_error("second preparation rejected"); });
            return std::make_unique<FieldTransferDetail::Prepared<Runtime::Field<int64_t>>>(secondSlot, second);
        };
    }
    const auto identity = first.Values().Identity();
    const auto secondIdentity = second.Values().Identity();
    const auto stamp = mesh->versions.Stamp();
    ReorderInput input;
    input.explicitMaps.push_back({EntityKind::Cell, std::vector<MPI_int>(count, mpi.rank)});
    MeshTransitionHandle handle(mesh);
    CHECK_THROWS(PreparedMeshTransition::Prepare(handle, input, {a, b}));
    CHECK(handle.Get() == mesh);
    CHECK(mesh->versions.CanMutate());
    CHECK(stamp.Valid());
    CHECK(first.Values().Identity() == identity);
    CHECK(second.Values().Identity() == secondIdentity);
    CHECK(first.GetDistribution().epoch == 2);
}

TEST_CASE("buildReorderRegistry populates pullSets for off-rank references")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return;
    auto source = T::Fixture(mpi);
    checkSparseTargets(source, mpi, {Adj::Cell2Node, Adj::Cell2Cell, Adj::Bnd2Node, Adj::Bnd2Cell, Adj::Node2Cell});
    auto result = T::Move(source, T::Place(source, EntityKind::Cell, mpi, 2), mpi);
    T::RoundTrip(result.state, mpi, {Adj::Cell2Node, Adj::Cell2Cell, Adj::Bnd2Node, Adj::Bnd2Cell});
}

TEST_CASE("ReorderEntities: edge destroy reconstructs all edge arrays and state")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true, true);
    auto input = T::Place(source, EntityKind::Cell, mpi, 1);
    input.destroyKinds = {EntityKind::Edge};
    auto result = T::Move(source, input, mpi);
    CHECK(result.state->HasEntities(EntityKind::Face));
    CHECK(result.state->HasEntities(EntityKind::Edge));
    T::RoundTrip(result.state, mpi, {Adj::Cell2Face, Adj::Face2Node, Adj::Cell2Edge, Adj::Edge2Node, Adj::Edge2Cell});
}

TEST_CASE("ReorderEntities: combined face and edge destroy reconstructs both groups")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true, true);
    auto input = T::Place(source, EntityKind::Cell, mpi, 1);
    input.destroyKinds = {EntityKind::Face, EntityKind::Edge};
    auto result = T::Move(source, input, mpi);
    CHECK(result.state->HasEntities(EntityKind::Face));
    CHECK(result.state->HasEntities(EntityKind::Edge));
    T::RoundTrip(result.state, mpi, {Adj::Cell2Face, Adj::Face2Node, Adj::Cell2Edge, Adj::Edge2Node, Adj::Edge2Cell});
}

TEST_CASE("ReorderEntities: face destroy reconstructs cell2facePbi on periodic mesh")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true, false, true);
    auto input = T::Place(source, EntityKind::Cell, mpi, 2);
    input.destroyKinds = {EntityKind::Face};
    auto result = T::Move(source, input, mpi);
    CHECK(result.state->BorrowHostView().IsPeriodic());
    CHECK(result.state->NextEntityId(EntityKind::Face) > source->NextEntityId(EntityKind::Face));
    T::RoundTrip(result.state, mpi, {Adj::Cell2Face, Adj::Face2Node});
}

TEST_CASE("ReorderEntities: face-preserving cell reorder reattaches face arrays")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true);
    auto result = T::Move(source, T::Place(source, EntityKind::Cell, mpi, 2), mpi);
    CHECK(result.state->NextEntityId(EntityKind::Face) == source->NextEntityId(EntityKind::Face));
}

TEST_CASE("buildReorderRegistry pull-sets include edge sources after interpolation")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return;
    auto source = T::Fixture(mpi, true, true);
    checkSparseTargets(source, mpi, {Adj::Cell2Edge, Adj::Edge2Node, Adj::Edge2Cell});
    auto result = T::Move(source, T::Place(source, EntityKind::Node, mpi, 2), mpi);
    T::RoundTrip(result.state, mpi, {Adj::Cell2Edge, Adj::Edge2Node, Adj::Edge2Cell});
}

TEST_CASE("buildReorderRegistry rejects non-global face adj after face interpolation")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true);
    auto mesh = UnstructuredMesh::FromNative(mpi, source);
    auto local = MeshAdjacencyRead::Current(*mesh, Adj::Face2Node, source->Requirements());
    auto detached = MeshAdjacency::Convert(local, Adj_PointToLocal);
    CHECK_THROWS(detached.WithGhostRows(EntityGhostMapping(detached.Source())));
    auto global = MeshAdjacency::Convert(local, Adj_PointToGlobal);
    CHECK_NOTHROW(global.WithGhostRows(EntityGhostMapping(global.Source())));
    T::RoundTrip(source, mpi, {Adj::Face2Node});
}

TEST_CASE("buildReorderRegistry registers cell2facePbi as cell companion for periodic mesh")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true, false, true);
    auto input = T::Place(source, EntityKind::Cell, mpi, 2);
    auto expected = T::RecordAll(source);
    auto result = NativeMeshTransition::Build(*source, mpi, input);
    T::Check(source, result, input, mpi, expected);
    CHECK(result.state->BorrowHostView().IsPeriodic());
}

TEST_CASE("ReorderEntities: node companion arrays reattached after reorder")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi);
    auto mesh = UnstructuredMesh::FromNative(mpi, source);
    auto nodeSpace = mesh->Entities(EntityKind::Node);
    Runtime::Array<DNDS::index> values(Runtime::RowLayout::Dense(size_t(nodeSpace.NumOwned()), 3));
    {
        auto write = values.Write();
        for (DNDS::index row = 0; row < nodeSpace.NumOwned(); ++row)
            for (int d = 0; d < 3; ++d)
                write.View()(size_t(row), size_t(d)) = 100 * source->OriginalEntityId(EntityKind::Node, row) + d;
    }
    EntityField<DNDS::index> field(nodeSpace, values);
    auto old = values;
    MeshTransitionHandle handle(mesh);
    auto prepared = PreparedMeshTransition::Prepare(handle, T::Place(source, EntityKind::Node, mpi, 1), {field.Register("nodeCompanion")});
    prepared.Publish();
    auto after = handle.Get()->CurrentNativeState();
    auto read = field.Read(handle.Get()->Entities(EntityKind::Node));
    for (DNDS::index row = 0; row < after->NumOwned(EntityKind::Node); ++row)
        for (int d = 0; d < 3; ++d)
            CHECK(read.View()(size_t(row), size_t(d)) == 100 * after->OriginalEntityId(EntityKind::Node, row) + d);
    CHECK_NOTHROW(old.Read());
    CHECK_NOTHROW(field.BuildHaloRoute(handle.Get()->Entities(EntityKind::Node)));
}

TEST_CASE("ReorderEntities: bnd2faceV and face2bndM reconstructed on Face destroy")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true);
    auto input = T::Place(source, EntityKind::Cell, mpi, 1);
    input.destroyKinds = {EntityKind::Face};
    auto result = T::Move(source, input, mpi);
    auto mesh = UnstructuredMesh::FromNative(mpi, result.state);
    auto boundaries = MeshAdjacencyRead::Current(*mesh, Adj::Bnd2Face, result.state->Requirements());
    auto faces = MeshAdjacencyRead::Current(*mesh, Adj::Face2Bnd, result.state->Requirements());
    CHECK(boundaries.NumOwned() == result.state->NumOwned(EntityKind::Bnd));
    for (DNDS::index row = 0; row < boundaries.NumOwned(); ++row)
    {
        REQUIRE(boundaries.RowSize(row) == 1);
        const auto face = boundaries.Target().GlobalToLocal(boundaries.ReadEntry(row, 0).global);
        REQUIRE(face.has_value());
        REQUIRE(faces.RowSize(*face) == 1);
        CHECK(faces.ReadEntry(*face, 0).global == boundaries.Source().LocalToGlobal(row));
    }
}

TEST_CASE("InterpolateEdge: edge arrays built and adjEdgeState is Adj_PointToLocal")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true, true);
    auto access = source->Snapshot(Runtime::Device::Host()).AcquireRead();
    const auto view = access.HostView();
    CHECK(view.RelationState(MeshRelation::EdgeNodes) == Adj_PointToLocal);
    CHECK(view.RelationState(MeshRelation::CellEdges) == Adj_PointToLocal);
    for (DNDS::index cell = 0; cell < view.NumCell(); ++cell)
    {
        CHECK(view.CellEdges(cell).size() == size_t(view.GetCellElement(cell).GetNumEdges()));
        for (const auto edge : view.CellEdges(cell))
        {
            CHECK(edge >= 0);
            CHECK(edge < view.NumEdgeProc());
            CHECK(view.EdgeNodes(edge).size() == 2);
        }
    }
    for (DNDS::index edge = 0; edge < view.NumEdge(); ++edge)
    {
        for (const auto node : view.EdgeNodes(edge))
        {
            CHECK(node >= 0);
            CHECK(node < view.NumNodeProc());
        }
        for (const auto cell : view.EdgeCells(edge))
        {
            CHECK(cell >= 0);
            CHECK(cell < view.NumCellProc());
        }
    }
}

TEST_CASE("InterpolateEdge: AdjLocal2GlobalEdge / AdjGlobal2LocalEdge round-trip")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true, true);
    T::RoundTrip(source, mpi, {Adj::Cell2Edge, Adj::Edge2Node, Adj::Edge2Cell});
}

TEST_CASE("ReorderLocalCells: existing edges retain exact IDs and rows")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true, true);
    auto expected = T::RecordAll(source);
    auto result = NativeMeshTransition::ReorderLocalCells(*source, mpi, 2, 1);
    T::Check(source, result, {}, mpi, expected);
    CHECK(result.state->HasEntities(EntityKind::Edge));
    CHECK(result.state->NextEntityId(EntityKind::Edge) == source->NextEntityId(EntityKind::Edge));
}

TEST_CASE("ReorderEntities: edge-preserving cell reorder reattaches edge arrays")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true, true);
    auto result = T::Move(source, T::Place(source, EntityKind::Cell, mpi, 1), mpi);
    CHECK(result.state->NextEntityId(EntityKind::Edge) == source->NextEntityId(EntityKind::Edge));
}

TEST_CASE("ReorderEntities: Node-explicit distributed round-robin")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return;
    auto source = T::Fixture(mpi);
    auto result = T::Move(source, T::Place(source, EntityKind::Node, mpi, 1), mpi);
    CHECK(result.state->NumGlobal(EntityKind::Node) == source->NumGlobal(EntityKind::Node));
}

TEST_CASE("ReorderEntities: Bnd-explicit distributed round-robin")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return;
    auto source = T::Fixture(mpi);
    auto result = T::Move(source, T::Place(source, EntityKind::Bnd, mpi, 1), mpi);
    CHECK(result.state->NumGlobal(EntityKind::Bnd) == source->NumGlobal(EntityKind::Bnd));
}

TEST_CASE("ReorderEntities: FollowSpec Bnd follows Node via bnd2node")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi);
    T::Follow(source, mpi, EntityKind::Node, {{EntityKind::Bnd, Adj::Bnd2Node}});
}

TEST_CASE("ReorderEntities: face-neighbor relation has independent target lookup coverage")
{
    auto mpi = worldMPI();
    auto source = T::Fixture(mpi, true);
    checkSparseTargets(source, mpi, {Adj::Cell2CellFace, Adj::Face2Cell, Adj::Cell2Cell});
    auto input = T::Place(source, EntityKind::Cell, mpi, 2);
    auto result = T::Move(source, input, mpi);
    T::RoundTrip(result.state, mpi, {Adj::Cell2CellFace, Adj::Cell2Cell});
}

TEST_CASE("ReorderLocalCells preserves multi-layer ghost identities and connectivity")
{
    auto mpi = worldMPI();
    for (const int layers : {2, 3})
    {
        CAPTURE(layers);
        auto source = T::Fixture(mpi, true, false, true, layers, "Uniform32_Periodic.cgns", 1);
        auto expected = T::RecordAll(source);
        auto result = NativeMeshTransition::ReorderLocalCells(*source, mpi, 2, 1, layers);
        T::Check(source, result, {}, mpi, expected);
        const auto &transfer = result.transfers.at(EntityKind::Cell);
        CHECK(transfer.isLocalOnly);
        // Capture source ghost identities before local reorder and compare after
        // the exact local permutation; retained ghost rows must not shrink to1ring.
        std::set<DNDS::index> beforeGhosts, afterGhosts;
        for (DNDS::index row = source->NumOwned(EntityKind::Cell); row < source->Entities(EntityKind::Cell).NumLocal(); ++row)
            beforeGhosts.insert(source->OriginalEntityId(EntityKind::Cell, row));
        for (DNDS::index row = result.state->NumOwned(EntityKind::Cell); row < result.state->Entities(EntityKind::Cell).NumLocal(); ++row)
            afterGhosts.insert(result.state->OriginalEntityId(EntityKind::Cell, row));
        CHECK(afterGhosts == beforeGhosts);
    }
}

TEST_CASE("ReorderEntities: Face-explicit identity and arriving cell rows")
{
    auto mpi = worldMPI();
    bool periodic = false;
    SUBCASE("ordinary mesh") {}
    SUBCASE("periodic mesh") { periodic = true; }
    auto source = T::Fixture(mpi, true, false, periodic);
    auto input = T::Place(source, EntityKind::Face, mpi);
    auto cellMap = T::Place(source, EntityKind::Cell, mpi, 2);
    input.explicitMaps.push_back(cellMap.explicitMaps[0]);
    input.follows.push_back({EntityKind::Node, EntityKind::Cell, Adj::Node2Cell});
    input.follows.push_back({EntityKind::Bnd, EntityKind::Cell, Adj::Bnd2Cell});
    auto result = T::Move(source, input, mpi);
    CHECK(result.transfers.at(EntityKind::Face).isLocalOnly);
    for (DNDS::index face = 0; face < source->NumOwned(EntityKind::Face); ++face)
        CHECK(result.state->OriginalEntityId(EntityKind::Face, face) == source->OriginalEntityId(EntityKind::Face, face));
}

TEST_CASE("ReorderEntities: Cell explicit, all topology follows Cell on 3D mesh")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return;
    auto source = T::Fixture(mpi, true, true);
    T::Follow(source, mpi, EntityKind::Cell, {{EntityKind::Node, Adj::Node2Cell}, {EntityKind::Bnd, Adj::Bnd2Cell}, {EntityKind::Face, Adj::Face2Cell}, {EntityKind::Edge, Adj::Edge2Cell}});
}

TEST_CASE("ReorderEntities: Node explicit, all topology follows Node on 3D mesh")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return;
    auto source = T::Fixture(mpi, true, true);
    T::Follow(source, mpi, EntityKind::Node, {{EntityKind::Cell, Adj::Cell2Node}, {EntityKind::Bnd, Adj::Bnd2Node}, {EntityKind::Face, Adj::Face2Node}, {EntityKind::Edge, Adj::Edge2Node}});
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    T::fixtures.clear();
    MPI_Finalize();
    return result;
}
