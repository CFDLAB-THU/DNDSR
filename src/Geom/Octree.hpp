#include "DNDS/Defines.hpp"
#include "Geometric.hpp"

namespace Geom
{
    class Octree
    {
    public:
        static const int LLL = 0;
        static const int LLR = 1;
        static const int LRL = 2;
        static const int LRR = 3;
        static const int RLL = 4;
        static const int RLR = 5;
        static const int RRL = 6;
        static const int RRR = 7;
        using tRange = Eigen::Vector2d;

    private:
        struct Node
        {
            std::array<Node *, 8> children;
            tRange xRange, yRange, zRange;
            DNDS::index i;
            tPoint p;
            Node(DNDS::index ni, const tPoint &np, const tRange &nX, const tRange &nY, const tRange &nZ)
                : xRange(nX), yRange(nY), zRange(nZ), i(ni), p(np)
            {
                for (auto &i : children)
                    i = nullptr;
            }

            bool InsertIfUnique(DNDS::index ni, const tPoint &np, double tol)
            {
                // TODO
            }
        } root;

    public:
    };
}