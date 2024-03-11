#pragma once
#include "DNDS/Defines.hpp"

namespace DNDS::Geom
{
    using t_index = int32_t;
    const t_index invalid_index = INT32_MAX;
    using t_real = double;
    using tPoint = Eigen::Vector3d;
    using tJacobi = Eigen::Matrix3d;
    using tGPoint = Eigen::Matrix3d;
    using tSmallCoords = Eigen::Matrix<real, 3, Eigen::Dynamic>;
    struct SmallCoordsAsVector : public tSmallCoords
    {
        auto operator[](Eigen::Index i)
        {
            return tSmallCoords::operator()(Eigen::all, i);
        }

        auto operator[](Eigen::Index i) const
        {
            return tSmallCoords::operator()(Eigen::all, i);
        }
    };

    static_assert(std::is_signed_v<t_index>);

    inline std::vector<real> JacobiToSTDVector(const tJacobi &j)
    {
        std::vector<real> ret(9);
        for (int i = 0; i < 9; i++)
            ret[i] = j(i);
        return ret; // TODO this is bullshit code
    }

    inline tJacobi STDVectorToJacobi(const std::vector<real> &v)
    {
        tJacobi ret;
        DNDS_assert(v.size() == 9);
        for (int i = 0; i < 9; i++)
            ret(i) = v[i];
        return ret; //  TODO this is bullshit code
    }

    inline std::vector<real> VectorToSTDVector(const Eigen::VectorXd &v)
    {
        return std::vector<real>(v.begin(), v.end()); //  TODO this is bullshit code
    }

    inline Eigen::VectorXd STDVectorToVector(const std::vector<real> &v)
    {
        auto ret = Eigen::VectorXd(v.size());
        for (size_t i = 0; i < v.size(); i++)
            ret[i] = v[i]; //  TODO this is bullshit code
        return ret;
    }
}

namespace DNDS::Geom
{
    /**
     * @brief get normal vector from facial jacobian
     * note that
     * $$
     * J_{ij} = \partial_{\xi_j} x_i
     * $$
     */
    template <int d>
    inline tPoint FacialJacobianToNormVec(const tJacobi &J)
    {
        static_assert(d == 2 || d == 3);
        if constexpr (d == 2)
        {
            DNDS_assert_info(J(2, 0) == 0, "Must be a line in x-y plane");
            DNDS_assert_info(J(Eigen::all, 1).norm() == 0, "Must be a line in x-y plane");
            DNDS_assert_info(J(Eigen::all, 2).norm() == 0, "Must be a line in x-y plane");
            return tPoint{J(1, 0), -J(0, 0), 0};
        }
        if constexpr (d == 3)
        {
            DNDS_assert_info(J(Eigen::all, 2).norm() == 0, "Must be a face");
            return J(Eigen::all, 0).cross(J(Eigen::all, 1));
        }
    }

    /**
     * @brief input uNorm should already be normalized
     *
     */
    template <int dim>
    inline Eigen::Matrix<real, dim, dim> NormBuildLocalBaseV(const Eigen::Ref<const Eigen::Vector<real, dim>> &uNorm)
    {
        Eigen::Matrix<real, dim, dim> base;
        static_assert(dim == 2 || dim == 3, "dim is bad");
        if constexpr (dim == 3)
        {
            base({0, 1, 2}, 0) = uNorm;
            if (std::abs(uNorm(2)) < 0.9)
                base({0, 1, 2}, 1) = -uNorm.cross(tPoint{0, 0, 1}).normalized();
            else
                base({0, 1, 2}, 1) = -uNorm.cross(tPoint{0, 1, 0}).normalized();

            base({0, 1, 2}, 2) = base({0, 1, 2}, 0).cross(base({0, 1, 2}, 1)).normalized();
        }
        else
        {
            base({0, 1}, 0) = uNorm;
            base(0, 1) = -uNorm(1); // x2=-y1
            base(1, 1) = uNorm(0);  // y2=x1
        }
        return base;
    }

    inline tGPoint RotZ(real theta)
    {
        theta *= pi / 180.0;
        return tGPoint{{cos(theta), -sin(theta), 0},
                       {sin(theta), cos(theta), 0},
                       {0, 0, 1}};
    }

    inline tGPoint RotX(real theta)
    {
        theta *= pi / 180.0;
        return tGPoint{{1, 0, 0},
                       {0, cos(theta), -sin(theta)},
                       {0, sin(theta), cos(theta)}};
    }

    inline tGPoint RotY(real theta)
    {
        theta *= pi / 180.0;
        return tGPoint{
            {cos(theta), 0, sin(theta)},
            {0, 1, 0},
            {-sin(theta), 0, cos(theta)}};
    }
}
