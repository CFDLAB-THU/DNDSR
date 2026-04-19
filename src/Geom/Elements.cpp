#include "Elements.hpp"

namespace DNDS::Geom::Elem
{

    Eigen::Matrix<t_real, 3, Eigen::Dynamic> GetStandardCoord(ElemType t)
    {
        auto ret = Eigen::Matrix<t_real, 3, Eigen::Dynamic>{};

        DispatchElementType(t, [&](auto tr)
        {
            using T = decltype(tr);
            ret.resize(3, T::numNodes);
            for (int j = 0; j < T::numNodes; j++)
                for (int d = 0; d < 3; d++)
                    ret(d, j) = T::standardCoords[j * 3 + d];
        });

        return ret;
    }
}
