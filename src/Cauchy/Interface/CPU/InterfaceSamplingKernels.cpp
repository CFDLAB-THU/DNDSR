#include "Cauchy/Interface/CPU/InterfaceSamplingKernels.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy::CPU
{
    void BuildInterfaceQueries(const InterfaceQueryRequest &request)
    {
        DNDS_assert(request.referencePoints.RowCount() == request.worldPoints.RowCount() &&
                    request.referencePoints.RowCount() == request.ids.RowCount());
        OMP::ForRows(0, request.referencePoints.RowCount(), [&](index iPoint)
                     {
            for (int d = 0; d < 3; ++d)
            {
                real value = request.referencePoints(iPoint, d);
                if (request.moving)
                {
                    value = 0;
                    for (int e = 0; e < 3; ++e)
                        value += request.rotation[size_t(3 * d + e)] *
                                 request.referencePoints(iPoint, e);
                    value += request.translation[size_t(d)];
                }
                request.worldPoints(iPoint, d) = value;
            }
            request.ids(iPoint, 0) = iPoint; });
    }
}
