#include "Solver/AggregationAMG_Internal.hpp"
#include <algorithm>
#include <climits>

namespace DNDS::Solver
{
    using namespace AMGDetail;
    namespace
    {
        struct Graph
        {
            std::vector<int> offsets, columns;
            std::vector<real> weights;
            int Rows() const { return int(offsets.size() - 1); }
        };

        template <class T>
        R::Array<T> Store(const std::string &label, const std::vector<T> &v)
        {
            auto a = Table<T>(label, v.size(), 1);
            auto lease = a.Write();
            const auto row = lease.View();
            for (index i = 0; i < index(v.size()); ++i)
                row(i, 0) = v[i];
            return a;
        }

        Graph ReadGraph(const R::Array<int> &offsets, const R::Array<int> &columns, const R::Array<real> &weights)
        {
            const index n = offsets.RowCount() - 1, edges = columns.RowCount();
            DNDS_check_throw_info(n >= 0 && n < INT_MAX && edges <= INT_MAX, "aggregation AMG requires int32 local CSR");
            RequireTable(offsets, n + 1, 1);
            RequireTable(columns, edges, 1);
            RequireTable(weights, edges, 1);
            auto o = offsets.Read(), c = columns.Read();
            auto w = weights.Read();
            Graph g;
            g.offsets.reserve(n + 1);
            g.columns.reserve(edges);
            g.weights.reserve(edges);
            const auto offsetsView = o.View(), columnsView = c.View();
            const auto weightsView = w.View();
            for (index i = 0; i <= n; ++i)
                g.offsets.push_back(offsetsView(i, 0));
            DNDS_check_throw_info(g.offsets.front() == 0 && g.offsets.back() == edges &&
                                      std::is_sorted(g.offsets.begin(), g.offsets.end()),
                                  "invalid AMG offsets");
            std::vector<int> last(n, -1);
            for (int i = 0; i < n; ++i)
                for (int k = g.offsets[i]; k < g.offsets[i + 1]; ++k)
                {
                    const int j = columnsView(k, 0);
                    const real s = weightsView(k, 0);
                    DNDS_check_throw_info(j >= 0 && j < n && j != i && last[j] != i &&
                                              std::isfinite(s) && s >= 0,
                                          "invalid AMG edge");
                    last[j] = i;
                    g.columns.push_back(j);
                    g.weights.push_back(s);
                }
            return g;
        }

        Graph Coarsen(const Graph &g, Level &level, int target, const std::string &name)
        {
            std::vector<int> group(g.Rows(), -1), members, starts{0};
            for (int seed = 0; seed < g.Rows(); ++seed)
            {
                if (group[seed] >= 0)
                    continue;
                const auto id = int(starts.size() - 1);
                members.push_back(seed);
                group[seed] = id;
                while (int(members.size()) - starts.back() < target)
                {
                    real best = 0;
                    int candidate = -1;
                    for (int p = starts.back(); p < int(members.size()); ++p)
                        for (int k = g.offsets[members[p]]; k < g.offsets[members[p] + 1]; ++k)
                        {
                            const int j = g.columns[k];
                            const real s = g.weights[k];
                            if (group[j] < 0 && (s > best || (s == best && j < candidate)))
                            {
                                best = s;
                                candidate = j;
                            }
                        }
                    if (candidate < 0)
                        break;
                    group[candidate] = id;
                    members.push_back(candidate);
                }
                starts.push_back(int(members.size()));
            }
            Graph coarse;
            coarse.offsets.push_back(0);
            if (int(starts.size()) == g.Rows() + 1)
                return coarse;
            std::vector<int> slot(g.columns.size(), -1), lastSlot(starts.size() - 1, -1);
            for (int a = 0; a < int(starts.size() - 1); ++a)
            {
                for (int p = starts[a]; p < starts[a + 1]; ++p)
                    for (int k = g.offsets[members[p]]; k < g.offsets[members[p] + 1]; ++k)
                    {
                        const int b = group[g.columns[k]];
                        if (a == b)
                            continue; // This contribution belongs to the coarse diagonal.
                        int &dest = lastSlot[b];
                        if (dest < coarse.offsets.back())
                        {
                            dest = int(coarse.columns.size());
                            coarse.columns.push_back(b);
                            coarse.weights.push_back(0);
                        }
                        coarse.weights[dest] += g.weights[k];
                        DNDS_check_throw_info(std::isfinite(coarse.weights[dest]), "coarse geometric weight overflow");
                        slot[k] = dest;
                    }
                coarse.offsets.push_back(int(coarse.columns.size()));
            }
            level.group = Store(name + "aggregate", group);
            level.members = Store(name + "members", members);
            level.memberOffsets = Store(name + "memberOffsets", starts);
            level.edgeSlot = Store(name + "coarseBlockSlot", slot);
            return coarse;
        }
    }

    AggregationAMG::AggregationAMG(const R::Array<int> &offsets, const R::Array<int> &columns,
                                   const R::Array<real> &weights, int b, AggregationAMGSettings settings)
        : _data(std::make_unique<Data>())
    {
        DNDS_check_throw_info(b >= 1 && (b <= 9 || b == 19), "unsupported aggregation AMG block width");
        DNDS_check_throw_info(settings.aggregateSize >= 2 && settings.minCoarseRows >= 1 && settings.maxLevels >= 1 &&
                                  settings.preSweeps >= 0 && settings.postSweeps >= 0 && settings.coarseSweeps > 0 &&
                                  std::isfinite(settings.relaxation) && settings.relaxation > 0,
                              "invalid aggregation AMG settings");
        _data->width = b;
        _data->settings = settings;
        Graph graph = ReadGraph(offsets, columns, weights);
        for (;;)
        {
            const auto depth = _data->levels.size();
            const std::string name = "Solver/aggregationAMG/level" + std::to_string(depth) + "/";
            auto &level = _data->levels.emplace_back();
            const int n = graph.Rows();
            level.offsets = Store(name + "offsets", graph.offsets);
            level.columns = Store(name + "columns", graph.columns);
            if (depth)
            {
                level.blocks = Table<real>(name + "blocks", graph.columns.size(), b * b);
                level.diagonal = Table<real>(name + "diagonal", n, b * b);
                level.rhs = Table<real>(name + "rhs", n, b);
                level.x = Table<real>(name + "correction", n, b);
            }
            level.inverse = Table<real>(name + "diagonalInverse", n, b * b);
            level.residual = Table<real>(name + "residual", n, b);
            if (n <= settings.minCoarseRows || int(depth + 1) >= settings.maxLevels)
                break;
            auto coarse = Coarsen(graph, level, settings.aggregateSize, name);
            if (!coarse.Rows())
                break;
            graph = std::move(coarse);
        }
    }
    AggregationAMG::~AggregationAMG() = default;
    index AggregationAMG::LevelCount() const { return _data->levels.size(); }
}
