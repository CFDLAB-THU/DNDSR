#include "doctest.h"
#include "Geom/Quadrature.hpp"
#include "Geom/BoundaryCondition.hpp"

const void *QuadratureCachePeer();

TEST_CASE("Local performance: quadrature cache has one owner and exact values")
{
    using namespace DNDS::Geom::Elem;
    CHECK(QuadratureCachePeer() == &detail::NBufferAtQuadrature);
    size_t matrices = 0;
    for (int i = 1; i < ElemType_NUM; ++i)
    {
        Element elem{ElemType(i)};
        for (int order = 0; order <= INT_ORDER_MAX; ++order)
        {
            const auto &cached = detail::NBufferAtQuadrature.buf.at(i).at(order);
            const auto scheme = GetQuadratureScheme(elem.GetParamSpace(), order);
            REQUIRE(cached.size() == size_t(scheme));
            for (int g = 0; g < scheme; ++g)
            {
                DNDS::Geom::tPoint point{0, 0, 0};
                DNDS::real weight;
                GetQuadraturePoint(elem.GetParamSpace(), scheme, g, point, weight);
                tD01Nj direct(4, elem.GetNumNodes());
                elem.GetD01Nj(point, direct);
                CHECK((direct.array() == cached[g].array()).all());
                ++matrices;
            }
        }
    }
    CHECK(matrices > 0);
}

TEST_CASE("Local performance: immutable boundary lookup preserves mutable defaults")
{
    using namespace DNDS::Geom;
    auto first = GetFaceName2IDDefault();
    auto second = GetFaceName2IDDefault();
    for (const auto &[name, id] : first)
        CHECK(FBC_Name_2_ID_Default(name) == id);
    CHECK(FBC_Name_2_ID_Default("unknown-local-boundary") == BC_ID_NULL);
    first["WALL"] = 777;
    CHECK(second.at("WALL") == BC_ID_DEFAULT_WALL);
    CHECK(FBC_Name_2_ID_Default("WALL") == BC_ID_DEFAULT_WALL);
}
