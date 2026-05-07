#pragma once

// =============================================================================
// Quadrature.hpp - High-level quadrature interface
// =============================================================================
// This file provides the Quadrature class and related utilities for numerical
// integration over finite elements. The actual quadrature data and low-level
// access functions are in the Quadratures/ subdirectory.
//
// Structure:
//   - Quadratures/QuadratureBase.hpp: Constants and type aliases
//   - Quadratures/*.hpp: Individual parametric space quadrature data
//   - QuadratureHub.hpp: Unified interface for scheme selection and point access
//   - Quadrature.hpp: This file - Quadrature class and integration utilities
// =============================================================================

#include "QuadratureHub.hpp"

namespace DNDS::Geom::Elem
{
    // =========================================================================
    // SummationNoOp - Helper class for no-op accumulation
    // =========================================================================
    class SummationNoOp
    {
    public:
        void operator+=(const SummationNoOp &R)
        {
        }

        SummationNoOp operator*(const SummationNoOp &R) const
        {
            return {};
        }

        friend SummationNoOp operator*(t_real L, const SummationNoOp &R)
        {
            return {};
        }

        friend SummationNoOp operator*(const SummationNoOp &L, t_real R)
        {
            return {};
        }
    };

    // =========================================================================
    namespace detail
    {

        // Precomputed Shape Function Buffers at Quadrature Points
        // =========================================================================
        static struct TNBufferAtQuadrature
        {
            std::array<std::array<std::vector<tD01Nj>, INT_ORDER_MAX + 1>, ElemType_NUM> buf;

            TNBufferAtQuadrature()
            {
                for (t_index i = 1; i < ElemType_NUM; i++)
                {
                    Element c_elem{ElemType(i)};
                    for (int order = 0; order <= INT_ORDER_MAX; order++)
                    {
                        auto int_scheme = GetQuadratureScheme(c_elem.GetParamSpace(), order);
                        buf.at(i).at(order).resize(int_scheme);
                        for (auto &m : buf.at(i).at(order))
                            m.resize(4, c_elem.GetNumNodes());
                        for (int iG = 0; iG < int_scheme; iG++)
                        {
                            tPoint pParam{0, 0, 0};
                            t_real w;
                            GetQuadraturePoint(c_elem.GetParamSpace(), int_scheme, iG, pParam, w);
                            c_elem.GetD01Nj(pParam, buf.at(i).at(order).at(iG));
                        }
                    }
                }
            }

        } NBufferAtQuadrature{};

    } // namespace detail

    // =========================================================================
    // Quadrature Class
    // =========================================================================
    // Main interface for numerical integration over elements.
    //
    // Usage:
    //   Element elem{Quad4};
    //   Quadrature quad(elem, 3);  // Order 3 integration
    //
    //   real sum = 0;
    //   quad.Integration(sum, [](real& acc, int iG, tPoint p, tD01Nj D01Nj) {
    //       acc = function_value(p);
    //   });
    // =========================================================================
    struct Quadrature
    {
        Element elem;
        int int_order;
        ParamSpace ps = UnknownPSpace;
        t_index int_scheme = 0;

        DNDS_DEVICE_CALLABLE Quadrature(Element n_elem = Element{UnknownElem}, int n_int_order = 0)
            : elem(n_elem), int_order(n_int_order), ps(elem.GetParamSpace()),
              int_scheme(GetQuadratureScheme(ps, int_order))
        {
        }

        /**
         * @brief General integration method with shape function access
         * @param f  Callback: f(TAcc& acc, int iG, tPoint pParam, tD01Nj D01Nj)
         */
        template <class TAcc, class TFunc>
        DNDS_DEVICE_CALLABLE void Integration(TAcc &buf, TFunc &&f)
        {
            for (t_index iG = 0; iG < int_scheme; iG++)
            {
                tPoint pParam{0, 0, 0};
                t_real w;
                GetQuadraturePoint(ps, int_scheme, iG, pParam, w);
                TAcc acc;
                f(acc, iG, pParam, detail::NBufferAtQuadrature.buf.at(elem.type).at(int_order).at(iG));
                buf += acc * w;
            }
        }

        /**
         * @brief Simple integration without shape function access
         * @param f  Callback: f(TAcc& acc, int iG)
         */
        template <class TAcc, class TFunc>
        DNDS_DEVICE_CALLABLE std::enable_if_t<std::is_invocable_v<TFunc, TAcc &, int>> IntegrationSimple(TAcc &buf, TFunc &&f)
        {
            for (t_index iG = 0; iG < int_scheme; iG++)
            {
                tPoint pParam{0, 0, 0};
                t_real w;
                GetQuadraturePoint(ps, int_scheme, iG, pParam, w);
                TAcc acc;
                f(acc, iG);
                buf += acc * w;
            }
        }

        /**
         * @brief Simple integration with weight access
         * @param f  Callback: f(TAcc& acc, int iG, real w)
         */
        template <class TAcc, class TFunc>
        DNDS_DEVICE_CALLABLE std::enable_if_t<std::is_invocable_v<TFunc, TAcc &, int, real>> IntegrationSimple(TAcc &buf, TFunc &&f)
        {
            for (t_index iG = 0; iG < int_scheme; iG++)
            {
                tPoint pParam{0, 0, 0};
                t_real w;
                GetQuadraturePoint(ps, int_scheme, iG, pParam, w);
                TAcc acc;
                f(acc, iG, w);
                buf += acc * w;
            }
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] auto GetQuadraturePointInfo(int iG) const
        {
            tPoint pParam{0, 0, 0};
            t_real w;
            GetQuadraturePoint(ps, int_scheme, iG, pParam, w);
            return std::make_tuple(pParam, w);
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] real GetWeight(int iG) const
        {
            return std::get<1>(GetQuadraturePointInfo(iG));
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] t_index GetNumPoints() const { return int_scheme; }
    };

    // =========================================================================
    // Quadrature Patch Generation (for visualization/post-processing)
    // =========================================================================
    // Generates triangle patches for visualizing quadrature point distributions.
    // Used for debugging and visualization purposes.
    // =========================================================================
    inline auto GetQuadPatches(Quadrature &q)
    {
        std::vector<std::array<t_index, 3>> ret;
        using tArr = std::array<t_index, 3>;
        ret.reserve(40);
        if (q.elem.GetParamSpace() == ParamSpace::LineSpace)
        {
            if (q.elem.GetOrder() == 1 || q.elem.GetOrder() == 2) // omitting the middle point
            {
                ret.emplace_back(tArr{1, -1, 0});
                for (int i = 1; i < q.GetNumPoints(); i++)
                    ret.emplace_back(tArr{-i, -1 - i, 0});
                ret.emplace_back(tArr{-q.GetNumPoints(), 2, 0});
            }
        }
        else if (q.elem.GetParamSpace() == ParamSpace::QuadSpace)
        {
            if (q.elem.GetOrder() == 1 || q.elem.GetOrder() == 2) // omitting the middle point
            {
                auto nGL = static_cast<t_index>(std::round(std::sqrt(q.GetNumPoints())));
                auto ijG = [=](int i, int j)
                { return nGL * j + i; };
                t_index iMax = nGL - 1;

                if (q.elem.GetOrder() == 1)
                {
                    ret.emplace_back(tArr{1, 2, -1 - ijG(0, 0)});
                    for (int i = 0; i < iMax; i++)
                        ret.emplace_back(tArr{-1 - ijG(i + 1, 0), -1 - ijG(i, 0), 2});
                    ret.emplace_back(tArr{3, 4, -1 - ijG(iMax, iMax)});
                    for (int i = 0; i < iMax; i++)
                        ret.emplace_back(tArr{-1 - ijG(i, iMax), -1 - ijG(i + 1, iMax), 4});
                    ret.emplace_back(tArr{2, 3, -1 - ijG(iMax, 0)});
                    for (int i = 0; i < iMax; i++)
                        ret.emplace_back(tArr{-1 - ijG(iMax, i + 1), -1 - ijG(iMax, i), 3});
                    ret.emplace_back(tArr{4, 1, -1 - ijG(0, iMax)});
                    for (int i = 0; i < iMax; i++)
                        ret.emplace_back(tArr{-1 - ijG(0, i), -1 - ijG(0, i + 1), 1});
                }
                else if (q.elem.GetOrder() == 2)
                {
                    ret.emplace_back(tArr{1, 5, -1 - ijG(0, 0)});
                    ret.emplace_back(tArr{5, 2, -1 - ijG(iMax, 0)});
                    for (int i = 0; i < iMax; i++)
                        ret.emplace_back(tArr{-1 - ijG(i + 1, 0), -1 - ijG(i, 0), 5});
                    ret.emplace_back(tArr{3, 7, -1 - ijG(iMax, iMax)});
                    ret.emplace_back(tArr{7, 4, -1 - ijG(0, iMax)});
                    for (int i = 0; i < iMax; i++)
                        ret.emplace_back(tArr{-1 - ijG(i, iMax), -1 - ijG(i + 1, iMax), 7});
                    ret.emplace_back(tArr{2, 6, -1 - ijG(iMax, 0)});
                    ret.emplace_back(tArr{6, 3, -1 - ijG(iMax, iMax)});
                    for (int i = 0; i < iMax; i++)
                        ret.emplace_back(tArr{-1 - ijG(iMax, i + 1), -1 - ijG(iMax, i), 6});
                    ret.emplace_back(tArr{4, 8, -1 - ijG(0, iMax)});
                    ret.emplace_back(tArr{8, 1, -1 - ijG(0, 0)});
                    for (int i = 0; i < iMax; i++)
                        ret.emplace_back(tArr{-1 - ijG(0, i), -1 - ijG(0, i + 1), 8});
                }

                for (int i = 0; i < iMax; i++)
                    for (int j = 0; j < iMax; j++)
                    {
                        ret.emplace_back(tArr{-1 - ijG(i, j), -1 - ijG(i + 1, j), -1 - ijG(i, j + 1)});
                        ret.emplace_back(tArr{-1 - ijG(i + 1, j + 1), -1 - ijG(i, j + 1), -1 - ijG(i + 1, j)});
                    }
            }
        }
        else if (q.elem.GetParamSpace() == ParamSpace::TriSpace)
        {
            if (q.GetNumPoints() == 1)
            {
                if (q.elem.GetOrder() == 1)
                {
                    ret.emplace_back(tArr{1, 2, -1});
                    ret.emplace_back(tArr{2, 3, -1});
                    ret.emplace_back(tArr{3, 1, -1});
                }
                else if (q.elem.GetOrder() == 2)
                {
                    ret.emplace_back(tArr{1, 4, -1});
                    ret.emplace_back(tArr{4, 2, -1});
                    ret.emplace_back(tArr{2, 5, -1});
                    ret.emplace_back(tArr{5, 3, -1});
                    ret.emplace_back(tArr{3, 6, -1});
                    ret.emplace_back(tArr{6, 1, -1});
                }
            }
            else if (q.GetNumPoints() == 3)
            {
                if (q.elem.GetOrder() == 1)
                {
                    ret.emplace_back(tArr{1, 2, -3});
                    ret.emplace_back(tArr{2, -1, -3});
                    ret.emplace_back(tArr{2, 3, -1});
                    ret.emplace_back(tArr{3, -2, -1});
                    ret.emplace_back(tArr{3, 1, -2});
                    ret.emplace_back(tArr{1, -3, -2});

                    ret.emplace_back(tArr{-1, -2, -3});
                }
                else if (q.elem.GetOrder() == 2)
                {
                    ret.emplace_back(tArr{1, 4, -3});
                    ret.emplace_back(tArr{4, 2, -1});
                    ret.emplace_back(tArr{2, 5, -1});
                    ret.emplace_back(tArr{5, 3, -2});
                    ret.emplace_back(tArr{3, 6, -2});
                    ret.emplace_back(tArr{6, 1, -3});

                    ret.emplace_back(tArr{-1, -3, 4});
                    ret.emplace_back(tArr{-2, -1, 5});
                    ret.emplace_back(tArr{-3, -2, 6});

                    ret.emplace_back(tArr{-1, -2, -3});
                }
            }
            else if (q.GetNumPoints() == 6)
            {
                if (q.elem.GetOrder() == 1)
                {
                    ret.emplace_back(tArr{1, 2, -6});
                    ret.emplace_back(tArr{2, 3, -4});
                    ret.emplace_back(tArr{3, 1, -5});

                    ret.emplace_back(tArr{-6, -1, 1});
                    ret.emplace_back(tArr{-2, -6, 2});
                    ret.emplace_back(tArr{-4, -2, 2});
                    ret.emplace_back(tArr{-3, -4, 3});
                    ret.emplace_back(tArr{-5, -3, 3});
                    ret.emplace_back(tArr{-1, -5, 1});

                    ret.emplace_back(tArr{-4, -5, -6});
                    ret.emplace_back(tArr{-1, -6, -5});
                    ret.emplace_back(tArr{-6, -2, -4});
                    ret.emplace_back(tArr{-5, -4, -3});
                }
                else if (q.elem.GetOrder() == 2)
                {
                    ret.emplace_back(tArr{1, 4, -6});
                    ret.emplace_back(tArr{4, 2, -6});
                    ret.emplace_back(tArr{2, 5, -4});
                    ret.emplace_back(tArr{5, 3, -4});
                    ret.emplace_back(tArr{3, 6, -5});
                    ret.emplace_back(tArr{6, 1, -5});

                    ret.emplace_back(tArr{-6, -1, 1});
                    ret.emplace_back(tArr{-2, -6, 2});
                    ret.emplace_back(tArr{-4, -2, 2});
                    ret.emplace_back(tArr{-3, -4, 3});
                    ret.emplace_back(tArr{-5, -3, 3});
                    ret.emplace_back(tArr{-1, -5, 1});

                    ret.emplace_back(tArr{-4, -5, -6});
                    ret.emplace_back(tArr{-1, -6, -5});
                    ret.emplace_back(tArr{-6, -2, -4});
                    ret.emplace_back(tArr{-5, -4, -3});
                }
            }
            else if (q.GetNumPoints() == 7)
            {
                static const std::array<int, 7> idxTri6ToTri7{
                    -1, 5, 6, 7, 2, 3, 4};
                if (q.elem.GetOrder() == 1)
                {
                    ret.emplace_back(tArr{1, 2, -idxTri6ToTri7[6]});
                    ret.emplace_back(tArr{2, 3, -idxTri6ToTri7[4]});
                    ret.emplace_back(tArr{3, 1, -idxTri6ToTri7[5]});

                    ret.emplace_back(tArr{-idxTri6ToTri7[6], -idxTri6ToTri7[1], 1});
                    ret.emplace_back(tArr{-idxTri6ToTri7[2], -idxTri6ToTri7[6], 2});
                    ret.emplace_back(tArr{-idxTri6ToTri7[4], -idxTri6ToTri7[2], 2});
                    ret.emplace_back(tArr{-idxTri6ToTri7[3], -idxTri6ToTri7[4], 3});
                    ret.emplace_back(tArr{-idxTri6ToTri7[5], -idxTri6ToTri7[3], 3});
                    ret.emplace_back(tArr{-idxTri6ToTri7[1], -idxTri6ToTri7[5], 1});

                    ret.emplace_back(tArr{-idxTri6ToTri7[4], -idxTri6ToTri7[5], -1});
                    ret.emplace_back(tArr{-idxTri6ToTri7[5], -idxTri6ToTri7[6], -1});
                    ret.emplace_back(tArr{-idxTri6ToTri7[6], -idxTri6ToTri7[4], -1});

                    ret.emplace_back(tArr{-idxTri6ToTri7[1], -idxTri6ToTri7[6], -idxTri6ToTri7[5]});
                    ret.emplace_back(tArr{-idxTri6ToTri7[6], -idxTri6ToTri7[2], -idxTri6ToTri7[4]});
                    ret.emplace_back(tArr{-idxTri6ToTri7[5], -idxTri6ToTri7[4], -idxTri6ToTri7[3]});
                }
                else if (q.elem.GetOrder() == 2)
                {
                    ret.emplace_back(tArr{1, 4, -idxTri6ToTri7[6]});
                    ret.emplace_back(tArr{4, 2, -idxTri6ToTri7[6]});
                    ret.emplace_back(tArr{2, 5, -idxTri6ToTri7[4]});
                    ret.emplace_back(tArr{5, 3, -idxTri6ToTri7[4]});
                    ret.emplace_back(tArr{3, 6, -idxTri6ToTri7[5]});
                    ret.emplace_back(tArr{6, 1, -idxTri6ToTri7[5]});

                    ret.emplace_back(tArr{-idxTri6ToTri7[6], -idxTri6ToTri7[1], 1});
                    ret.emplace_back(tArr{-idxTri6ToTri7[2], -idxTri6ToTri7[6], 2});
                    ret.emplace_back(tArr{-idxTri6ToTri7[4], -idxTri6ToTri7[2], 2});
                    ret.emplace_back(tArr{-idxTri6ToTri7[3], -idxTri6ToTri7[4], 3});
                    ret.emplace_back(tArr{-idxTri6ToTri7[5], -idxTri6ToTri7[3], 3});
                    ret.emplace_back(tArr{-idxTri6ToTri7[1], -idxTri6ToTri7[5], 1});

                    ret.emplace_back(tArr{-idxTri6ToTri7[4], -idxTri6ToTri7[5], -1});
                    ret.emplace_back(tArr{-idxTri6ToTri7[5], -idxTri6ToTri7[6], -1});
                    ret.emplace_back(tArr{-idxTri6ToTri7[6], -idxTri6ToTri7[4], -1});

                    ret.emplace_back(tArr{-idxTri6ToTri7[1], -idxTri6ToTri7[6], -idxTri6ToTri7[5]});
                    ret.emplace_back(tArr{-idxTri6ToTri7[6], -idxTri6ToTri7[2], -idxTri6ToTri7[4]});
                    ret.emplace_back(tArr{-idxTri6ToTri7[5], -idxTri6ToTri7[4], -idxTri6ToTri7[3]});
                }
            }
            else if (q.GetNumPoints() == 12)
            {
                if (q.elem.GetOrder() == 1)
                {
                    ret.emplace_back(tArr{1, 2, -4});
                    ret.emplace_back(tArr{2, 3, -5});
                    ret.emplace_back(tArr{3, 1, -6});

                    ret.emplace_back(tArr{-7, -4, 2});
                    ret.emplace_back(tArr{-9, -7, 2});
                    ret.emplace_back(tArr{-5, -9, 2});

                    ret.emplace_back(tArr{-11, -5, 3});
                    ret.emplace_back(tArr{-12, -11, 3});
                    ret.emplace_back(tArr{-6, -12, 3});

                    ret.emplace_back(tArr{-10, -6, 1});
                    ret.emplace_back(tArr{-8, -10, 1});
                    ret.emplace_back(tArr{-4, -8, 1});

                    ret.emplace_back(tArr{-4, -7, -8});
                    ret.emplace_back(tArr{-9, -5, -11});
                    ret.emplace_back(tArr{-10, -12, -6});
                    ret.emplace_back(tArr{-8, -7, -1});
                    ret.emplace_back(tArr{-2, -9, -11});
                    ret.emplace_back(tArr{-3, -12, -10});

                    ret.emplace_back(tArr{-7, -9, -2});
                    ret.emplace_back(tArr{-7, -2, -1});
                    ret.emplace_back(tArr{-11, -12, -3});
                    ret.emplace_back(tArr{-11, -3, -2});
                    ret.emplace_back(tArr{-10, -8, -1});
                    ret.emplace_back(tArr{-10, -1, -3});

                    ret.emplace_back(tArr{-1, -2, -3});
                }
                else if (q.elem.GetOrder() == 2)
                {
                    ret.emplace_back(tArr{1, 4, -4});
                    ret.emplace_back(tArr{4, 2, -5});
                    ret.emplace_back(tArr{2, 5, -5});
                    ret.emplace_back(tArr{5, 3, -6});
                    ret.emplace_back(tArr{3, 6, -6});
                    ret.emplace_back(tArr{6, 1, -4});

                    ret.emplace_back(tArr{-7, -4, 4});
                    ret.emplace_back(tArr{-9, -7, 4});
                    ret.emplace_back(tArr{-5, -9, 4});

                    ret.emplace_back(tArr{-11, -5, 5});
                    ret.emplace_back(tArr{-12, -11, 5});
                    ret.emplace_back(tArr{-6, -12, 5});

                    ret.emplace_back(tArr{-10, -6, 6});
                    ret.emplace_back(tArr{-8, -10, 6});
                    ret.emplace_back(tArr{-4, -8, 6});

                    ret.emplace_back(tArr{-4, -7, -8});
                    ret.emplace_back(tArr{-9, -5, -11});
                    ret.emplace_back(tArr{-10, -12, -6});
                    ret.emplace_back(tArr{-8, -7, -1});
                    ret.emplace_back(tArr{-2, -9, -11});
                    ret.emplace_back(tArr{-3, -12, -10});

                    ret.emplace_back(tArr{-7, -9, -2});
                    ret.emplace_back(tArr{-7, -2, -1});
                    ret.emplace_back(tArr{-11, -12, -3});
                    ret.emplace_back(tArr{-11, -3, -2});
                    ret.emplace_back(tArr{-10, -8, -1});
                    ret.emplace_back(tArr{-10, -1, -3});

                    ret.emplace_back(tArr{-1, -2, -3});
                }
            }
            else
                DNDS_assert(false);
        }
        else
            DNDS_assert(false);
        return ret;
    }

    // =========================================================================
    // Jacobian Determinant Helpers
    // =========================================================================
    template <int dim>
    real CellJacobianDet(const Geom::tSmallCoords &coordsCell, const Geom::Elem::tD01Nj &DiNj)
    {
        using namespace Geom;
        real JDet{0};
        tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsCell, DiNj);
        if constexpr (dim == 2)
            JDet = J(EigenAll, 0).cross(J(EigenAll, 1)).stableNorm();
        else
            JDet = J.fullPivLu().determinant();
        return JDet;
    }

    inline real CellJacobianDet(int dim, const Geom::tSmallCoords &coordsCell, const Geom::Elem::tD01Nj &DiNj)
    {
        if (dim == 2)
            return CellJacobianDet<2>(coordsCell, DiNj);
        else
            return CellJacobianDet<3>(coordsCell, DiNj);
    }

    template <int dim>
    real FaceJacobianDet(const Geom::tSmallCoords &coords, const Geom::Elem::tD01Nj &DiNj)
    {
        using namespace Geom;
        real JDet{0};
        tJacobi J = Elem::ShapeJacobianCoordD01Nj(coords, DiNj);
        if constexpr (dim == 2)
            JDet = J(EigenAll, 0).stableNorm();
        else
            JDet = J(EigenAll, 0).cross(J(EigenAll, 1)).stableNorm();
        return JDet;
    }

    inline real FaceJacobianDet(int dim, const Geom::tSmallCoords &coords, const Geom::Elem::tD01Nj &DiNj)
    {
        if (dim == 2)
            return FaceJacobianDet<2>(coords, DiNj);
        else
            return FaceJacobianDet<3>(coords, DiNj);
    }

} // namespace DNDS::Geom::Elem