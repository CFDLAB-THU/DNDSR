#pragma once

#include "DNDS/Defines.hpp"
#include <assert.h>
#include <iostream>

namespace DNDS::ETensor
{
    using namespace Eigen;
    template <typename T, Index d0, Index d1, Index d2>
    class ETensorR3
    {
        static const Index siz = d0 * d1 * d2;
        static const Index stride0 = d1 * d2;
        static const Index stride1 = d2;
        static const Index stride2 = 1;
        T data[siz];

    public:
        ETensorR3(const T &fill)
        {
            for (Index i = 0; i < siz; i++)
                data[i] = fill;
        }

        ETensorR3() = default;

        T &operator()(Index i0, Index i1, Index i2)
        {
            return data[i2 + d2 * (i1 + (d1 * i0))];
        }

        using M01 = Matrix<T, d0, d1, RowMajor>;
        using M12 = Matrix<T, d1, d2, RowMajor>;
        using M02 = Matrix<T, d0, d2, RowMajor>;

        using Map01 = Map<M01, Unaligned, Stride<stride0, stride1>>;
        using Map12 = Map<M12, Unaligned, Stride<stride1, stride2>>;
        using Map02 = Map<M02, Unaligned, Stride<stride0, stride2>>;

        Map01 GetMap01(Index i2)
        {
            DNDS_assert(i2 < d2);
            return Map01(data + i2 * stride2);
        }

        Map12 GetMap12(Index i0)
        {
            DNDS_assert(i0 < d0);
            return Map12(data + i0 * stride0);
        }

        Map02 GetMap02(Index i1)
        {
            DNDS_assert(i1 < d1);
            return Map02(data + i1 * stride1);
        }

        template <class Tmat>
        void MatTransform0(const Tmat &Rmat)
        {
            DNDS_assert(Rmat.cols() == Rmat.rows() && Rmat.rows() == d0);
            for (Index i2 = 0; i2 < d2; i2++)
            {
                Map01 m = GetMap01(i2);
                m = Rmat.transpose() * m;
            }
        }

        template <class Tmat>
        void MatTransform1(const Tmat &Rmat)
        {
            DNDS_assert(Rmat.cols() == Rmat.rows() && Rmat.rows() == d1);
            for (Index i2 = 0; i2 < d2; i2++)
            {
                Map01 m = GetMap01(i2);
                m = m * Rmat;
            }
        }

        template <class Tmat>
        void MatTransform2(const Tmat &Rmat)
        {
            DNDS_assert(Rmat.cols() == Rmat.rows() && Rmat.rows() == d2);
            for (Index i0 = 0; i0 < d0; i0++)
            {
                Map12 m = GetMap12(i0);
                m = m * Rmat;
            }
        }

        template <Index dout, class Tmat>
        ETensorR3<T, dout, d1, d2> MatTransform0d(const Tmat &Rmat)
        {
            DNDS_assert(Rmat.rows() == d0 && Rmat.cols() == dout);
            ETensorR3<T, dout, d1, d2> res;
            for (Index i2 = 0; i2 < d2; i2++)
            {
                auto m = GetMap01(i2);
                auto mout = res.GetMap01(i2);
                mout = Rmat.transpose() * m;
            }
            return res;
        }

        template <Index dout, class Tmat>
        ETensorR3<T, d0, dout, d2> MatTransform1d(const Tmat &Rmat)
        {
            DNDS_assert(Rmat.rows() == d1 && Rmat.cols() == dout);
            ETensorR3<T, d0, dout, d2> res;
            // const TDerived &rmat = Rmat;
            for (Index i2 = 0; i2 < d2; i2++)
            {
                Map01 m = GetMap01(i2);
                auto mout = res.GetMap01(i2);
                mout = m * Rmat;
            }
            return res;
        }

        template <Index dout, class Tmat>
        ETensorR3<T, d0, d1, dout> MatTransform2d(const Tmat &Rmat)
        {
            DNDS_assert(Rmat.rows() == d2 && Rmat.cols() == dout);
            ETensorR3<T, d0, d1, dout> res;
            for (Index i0 = 0; i0 < d0; i0++)
            {
                auto m = GetMap12(i0);
                auto mout = res.GetMap12(i0);
                mout = m * Rmat;
            }
            return res;
        }
    };
    template <typename T, Index d0, Index d1, Index d2>
    std::ostream &operator<<(std::ostream &out, ETensorR3<T, d0, d1, d2> &R)
    {
        out << "[";
        for (Index i0 = 0; i0 < d0; i0++)
            out << R.GetMap12(i0) << "\n\n";
        out << "]\n";
        return out;
    }

    template <typename T, Index d0, Index d1, Index d2>
    std::ostream &operator<<(std::ostream &out, ETensorR3<T, d0, d1, d2> &&R)
    {
        out << "[";
        for (Index i0 = 0; i0 < d0; i0++)
            out << R.GetMap12(i0) << "\n\n";
        out << "]\n";
        return out;
    }
}
