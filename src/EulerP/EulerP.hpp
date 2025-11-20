#pragma once
#include "DNDS/ArrayDOF.hpp"

namespace DNDS::EulerP
{
    static constexpr int nVarsFlow = 5;
    using TU = Eigen::Vector<real, nVarsFlow>;
    using TDiffU = Eigen::Matrix<real, 3, nVarsFlow>;

    using TUDof = ArrayDof<nVarsFlow, 1>;
    using TUGrad = ArrayDof<3, nVarsFlow>;

    using TUScalar = ArrayDof<1, 1>;
    using TUScalarGrad = ArrayDof<3, 1>;

    using TUVec = ArrayDof<3, 1>;
    using TUVecGrad = ArrayDof<3, 3>;
}