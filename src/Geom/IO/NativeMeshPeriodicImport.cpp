#include "Geom/IO/NativeMeshPeriodicImport.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include "Geom/Spatial/PointCloud.hpp"
#include "DNDS/Core/OMP.hpp"
#include <nanoflann.hpp>
#include <queue>

namespace DNDS::Geom
{
    NativeSerialMeshData CanonicalizeNativePeriodic(NativeSerialMeshData source, const NativePeriodicImportOptions &options)
    {
        auto setupStage = options.progress.Stage("mesh/periodic");
        DNDS_check_throw_info(!source.periodic && !source.cellNodesPbi && !source.boundaryNodesPbi,
                              "periodic serial import must not already be canonicalized");
        const real tolerance = options.tolerance;
        DNDS_check_throw_info(std::isfinite(tolerance) && tolerance > 0 && std::isfinite(tolerance * tolerance), "invalid periodic import tolerance");
        const auto &frames = options.frames;
        for (size_t group = 0; group < 4; ++group)
        {
            const auto rotation = frames.rotation[group].map();
            DNDS_check_throw_info(rotation.allFinite() && frames.translation[group].map().allFinite() && frames.rotationCenter[group].map().allFinite() &&
                                      (rotation.transpose() * rotation - tGPoint::Identity()).norm() < 1e-10 && std::abs(rotation.determinant() - 1) < 1e-10,
                                  "periodic import requires finite rigid transforms");
        }
        DNDS_check_throw_info(source.coordinates.GetDevice() == Runtime::Device::Host() && source.nodeIds.RowCount() == source.coordinates.RowCount() &&
                                  source.cellInfo.RowCount() == source.cellNodes.RowCount() && source.boundaryInfo.RowCount() == source.boundaryNodes.RowCount(),
                              "invalid serial periodic input row inventory");
        auto coords = source.coordinates.Read();
        auto nodeIds = source.nodeIds.Read();
        auto boundaryNodes = source.boundaryNodes.Read();
        auto boundaryInfo = source.boundaryInfo.Read();
        bool hasPeriodicLabels = false;
        for (size_t row = 0; row < source.boundaryInfo.RowCount(); ++row)
        {
            DNDS_check_throw_info(boundaryInfo.View().RowSize(row) == 1, "invalid serial boundary info shape");
            hasPeriodicLabels |= FaceIDIsPeriodic(boundaryInfo.View()(row, 0).zone);
        }
        if (!hasPeriodicLabels)
            return source;
        std::vector<tPoint> points(source.coordinates.RowCount());
        std::vector<index> persistedIds(points.size());
        OMP::ForSetupRows(0, index(points.size()), [&](index node)
                          {
            DNDS_check_throw_info(coords.View().RowSize(size_t(node)) == 3 && nodeIds.View().RowSize(size_t(node)) == 1, "invalid serial periodic node data shape");
            points[size_t(node)] = tPoint{coords.View()(size_t(node), 0), coords.View()(size_t(node), 1), coords.View()(size_t(node), 2)};
            const auto id = nodeIds.View()(size_t(node), 0);
            DNDS_check_throw_info(points[size_t(node)].allFinite() && id >= 0 && id < INT64_MAX, "invalid serial periodic point or persistent ID");
            persistedIds[size_t(node)] = id; });
        std::sort(persistedIds.begin(), persistedIds.end());
        DNDS_check_throw_info(std::adjacent_find(persistedIds.begin(), persistedIds.end()) == persistedIds.end(),
                              "serial periodic persistent IDs are duplicated");
        std::array<PointCloudKDTree, 3> donors;
        std::array<std::vector<index>, 3> donorRows, mainRows;
        auto cone = [&](size_t row)
        {
            DNDS_check_throw_info(boundaryInfo.View().RowSize(row) == 1, "invalid serial periodic boundary info shape");
            const auto type = boundaryInfo.View()(row, 0).type;
            DNDS_check_throw_info(type > Elem::UnknownElem && type < Elem::ElemType_NUM, "invalid periodic boundary element type");
            const Elem::Element element{static_cast<Elem::ElemType>(type)};
            DNDS_check_throw_info(boundaryNodes.View().RowSize(row) == size_t(element.GetNumNodes()), "periodic boundary cone differs from its element type");
            std::vector<index> result;
            for (auto node : boundaryNodes.View()[row])
            {
                DNDS_check_throw_info(node >= 0 && size_t(node) < points.size(), "periodic boundary node is out of range");
                result.push_back(node);
            }
            return result;
        };
        auto center = [&](size_t row)
        {
            const auto nodes = cone(row);
            tPoint result = tPoint::Zero();
            for (auto node : nodes)
                result += points[size_t(node)];
            return tPoint(result / real(nodes.size()));
        };
        for (size_t row = 0; row < source.boundaryNodes.RowCount(); ++row)
        {
            const auto zone = boundaryInfo.View()(row, 0).zone;
            if (!FaceIDIsPeriodic(zone))
                continue;
            const size_t group = size_t((-zone - 1) % 3);
            if (FaceIDIsPeriodicMain(zone))
                mainRows[group].push_back(index(row));
            else
            {
                donorRows[group].push_back(index(row));
                donors[group].pts.push_back(center(row));
            }
        }
        bool periodic = false;
        std::vector<std::vector<std::pair<index, uint8_t>>> graph(points.size());
        std::array<std::vector<index>, 3> donorToMain, mainToDonor;
        for (auto &mapping : donorToMain)
            mapping.assign(points.size(), UnInitIndex);
        for (auto &mapping : mainToDonor)
            mapping.assign(points.size(), UnInitIndex);
        using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<real, PointCloudKDTree>, PointCloudKDTree, 3, index>;
        const real squaredTolerance = tolerance * tolerance;
        for (size_t group = 0; group < 3; ++group)
        {
            DNDS_check_throw_info(mainRows[group].size() == donorRows[group].size(), "periodic main/donor boundary counts differ");
            if (mainRows[group].empty())
                continue;
            periodic = true;
            Tree tree(3, donors[group]);
            std::vector<uint8_t> used(donorRows[group].size(), 0);
            for (auto main : mainRows[group])
            {
                const auto zone = t_index(-1 - int(group));
                const tPoint target = frames.TransCoord(center(size_t(main)), zone);
                index indices[2]{};
                real distances[2]{};
                const auto count = tree.knnSearch(target.data(), 2, indices, distances);
                DNDS_check_throw_info(count && distances[0] <= squaredTolerance && (count < 2 || distances[1] > squaredTolerance),
                                      "periodic boundary center pairing is missing or ambiguous");
                const auto donor = donorRows[group].at(size_t(indices[0]));
                DNDS_check_throw_info(!used[size_t(indices[0])] && boundaryInfo.View()(size_t(main), 0).type == boundaryInfo.View()(size_t(donor), 0).type,
                                      "periodic face pairing is not a one-to-one equal-type mapping");
                used[size_t(indices[0])] = 1;
                const auto mainNodes = cone(size_t(main)), donorNodes = cone(size_t(donor));
                std::vector<index> usedNodes;
                usedNodes.reserve(donorNodes.size());
                for (auto node : mainNodes)
                {
                    const tPoint transformed = frames.TransCoord(points[size_t(node)], zone);
                    index match = UnInitIndex;
                    for (auto candidate : donorNodes)
                        if ((points[size_t(candidate)] - transformed).squaredNorm() <= squaredTolerance)
                        {
                            DNDS_check_throw_info(match == UnInitIndex, "periodic face node pairing is ambiguous");
                            match = candidate;
                        }
                    DNDS_check_throw_info(match != UnInitIndex && std::find(usedNodes.begin(), usedNodes.end(), match) == usedNodes.end(),
                                          "periodic face node pairing is missing or nonbijective");
                    usedNodes.push_back(match);
                    DNDS_check_throw_info(match != node, "periodic main/donor share an original node; fixed-point image labels require an explicit occurrence policy");
                    auto &donorMain = donorToMain[group][size_t(match)];
                    auto &mainDonor = mainToDonor[group][size_t(node)];
                    DNDS_check_throw_info((donorMain == UnInitIndex || donorMain == node) &&
                                              (mainDonor == UnInitIndex || mainDonor == match),
                                          "periodic boundary node maps are inconsistent");
                    const bool firstMapping = donorMain == UnInitIndex;
                    DNDS_check_throw_info(firstMapping == (mainDonor == UnInitIndex),
                                          "periodic boundary node maps have mismatched insertion state");
                    donorMain = node;
                    mainDonor = match;
                    if (firstMapping)
                    {
                        const auto mask = uint8_t(1U << group);
                        graph[size_t(node)].emplace_back(match, mask);
                        graph[size_t(match)].emplace_back(node, mask);
                    }
                }
            }
        }
        if (!periodic)
            return source;
        std::vector<index> roots(points.size(), UnInitIndex), compact(points.size(), UnInitIndex);
        std::vector<uint8_t> labels(points.size(), 0);
        // Most volume nodes do not participate in periodic identification.
        // Publish their singleton components in parallel, leaving only the
        // small active boundary graph for deterministic component traversal.
        OMP::ForSetupRows(0, index(points.size()), [&](index node)
                          {
            if (graph[size_t(node)].empty())
                roots[size_t(node)] = node; });
        for (size_t start = 0; start < points.size(); ++start)
        {
            if (roots[start] != UnInitIndex)
                continue;
            std::vector<index> component{index(start)};
            roots[start] = index(start);
            for (size_t cursor = 0; cursor < component.size(); ++cursor)
                for (const auto &[other, mask] : graph[size_t(component[cursor])])
                {
                    const auto expected = uint8_t(labels[size_t(component[cursor])] ^ mask);
                    if (roots[size_t(other)] == UnInitIndex)
                    {
                        roots[size_t(other)] = index(start);
                        labels[size_t(other)] = expected;
                        component.push_back(other);
                    }
                    else
                        DNDS_check_throw_info(labels[size_t(other)] == expected, "inconsistent periodic image cycle");
                }
            std::sort(component.begin(), component.end());
            uint8_t images = 0;
            for (auto node : component)
            {
                const auto bit = uint8_t(1U << labels[size_t(node)]);
                DNDS_check_throw_info(!(images & bit), "periodic component has multiple original nodes in the same image");
                images |= bit;
            }
            DNDS_check_throw_info(component.size() <= 8, "periodic component exceeds the three-bit image model");
            index root = UnInitIndex;
            for (auto candidate : component)
            {
                bool donor = false;
                for (const auto &mapping : donorToMain)
                    donor |= mapping[size_t(candidate)] != UnInitIndex;
                if (donor)
                    continue;
                bool valid = true;
                for (auto node : component)
                    valid = valid && (frames.GetCoordByBits(points[size_t(candidate)], NodePeriodicBits{uint8_t(labels[size_t(node)] ^ labels[size_t(candidate)])}) - points[size_t(node)]).norm() <= tolerance;
                if (valid)
                {
                    root = candidate;
                    break;
                }
            }
            DNDS_check_throw_info(root != UnInitIndex, "periodic image chains have no main root reconstructing original coordinates");
            const auto rootMask = labels[size_t(root)];
            for (auto node : component)
            {
                roots[size_t(node)] = root;
                labels[size_t(node)] ^= rootMask;
            }
        }
        std::vector<uint8_t> keep(roots.size());
        OMP::ForSetupRows(0, index(roots.size()), [&](index node)
                          { keep[size_t(node)] = roots[size_t(node)] == node; });
        const index keptNumber = OMP::ExclusivePrefixRows(
            index(roots.size()), [&](index node) noexcept -> index
            { return keep[size_t(node)]; },
            [&](index node, index offset) noexcept
            {
                if (node < index(compact.size()) && keep[size_t(node)])
                    compact[size_t(node)] = offset;
            });
        std::vector<index> kept(static_cast<size_t>(keptNumber));
        OMP::ForSetupRows(0, index(roots.size()), [&](index node)
                          {
            if (keep[size_t(node)])
                kept[size_t(compact[size_t(node)])] = node; });
        OMP::ForSetupRows(0, index(roots.size()), [&](index node)
                          { compact[size_t(node)] = compact[size_t(roots[size_t(node)])]; });
        NativeSerialMeshData result = source;
        result.coordinates = Runtime::Array<real>(Runtime::RowLayout::Dense(kept.size(), 3));
        result.nodeIds = Runtime::Array<index>(Runtime::RowLayout::Dense(kept.size(), 1));
        {
            auto output = result.coordinates.Write();
            auto ids = result.nodeIds.Write();
            OMP::ForSetupRows(0, index(kept.size()), [&](index row)
                              {
                for (size_t component = 0; component < 3; ++component)
                    output.View()(size_t(row), component) = points[size_t(kept[size_t(row)])][component];
                // Periodic identification precedes mesh publication. Match
                // the legacy serial convention: retained canonical nodes have
                // compact IDs in retained input-row order, not sparse labels
                // copied from the main image's pre-identification input row.
                ids.View()(size_t(row), 0) = row; });
        }
        auto remap = [&](const Runtime::Array<index> &input, Runtime::Array<index> &output, std::optional<Runtime::Array<NodePeriodicBits>> &bits)
        {
            auto read = input.Read();
            output = Runtime::BuildArrayRows<index>(input.RowCount(), [&](size_t row)
                                                    { return read.View().RowSize(row); }, [&](size_t row, auto target)
                                                    {
                size_t slot = 0;
                for (auto node : read.View()[row])
                {
                    DNDS_check_throw_info(node >= 0 && size_t(node) < points.size(), "periodic cone node is out of range");
                    target[slot++] = compact[size_t(node)];
                } }, {}, true);
            bits = Runtime::BuildArrayRows<NodePeriodicBits>(input.RowCount(), [&](size_t row)
                                                             { return read.View().RowSize(row); }, [&](size_t row, auto target)
                                                             {
                size_t slot = 0;
                for (auto node : read.View()[row])
                    target[slot++] = NodePeriodicBits{labels[size_t(node)]}; }, {}, true);
        };
        remap(source.cellNodes, result.cellNodes, result.cellNodesPbi);
        remap(source.boundaryNodes, result.boundaryNodes, result.boundaryNodesPbi);
        result.periodic = true;
        result.periodicInfo = frames;
        auto &highWater = result.identity.nextIds[EntityKind::Node];
        // Fresh ingress creates compact canonical IDs, not aliases of the
        // pre-identification labels. New-node allocation must start after the
        // canonical range, as in legacy elevation/refinement. Preserve only an
        // explicitly supplied reservation, rather than reserving discarded IDs.
        highWater = std::max(highWater, index(kept.size()));
        return result;
    }
}
