#include "Geom/Quadrature.hpp"

const void *QuadratureCachePeer()
{
    return &DNDS::Geom::Elem::detail::NBufferAtQuadrature;
}
