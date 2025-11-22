#pragma once
#include "DNDS/ArrayDOF.hpp"
#include "Eigen/Core"

namespace DNDS::EulerP
{
    static constexpr int nVarsFlow = 5;
    static constexpr int I4 = 4;
    using TU = Eigen::Vector<real, nVarsFlow>;
    using TDiffU = Eigen::Matrix<real, 3, nVarsFlow>;

    using TUFull = Eigen::Vector<real, Eigen::Dynamic>;
    using TDiffUFull = Eigen::Matrix<real, 3, Eigen::Dynamic>;

    using TUFullMap = Eigen::Map<TUFull>;
    using TDiffUFullMap = Eigen::Map<TDiffUFull>;

    using TUDof = ArrayDof<nVarsFlow, 1>;
    using TUGrad = ArrayDof<3, nVarsFlow>;

    using TUScalar = ArrayDof<1, 1>;
    using TUScalarGrad = ArrayDof<3, 1>;

    using TUScalar2 = ArrayDof<2, 1>;

    using TUVec = ArrayDof<3, 1>;
    using TUVecGrad = ArrayDof<3, 3>;

    static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<nVarsFlow - 1>);
    static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<nVarsFlow - 2>);
    static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<nVarsFlow - 3>);
}