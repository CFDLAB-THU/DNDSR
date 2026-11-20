#pragma once
#include "Cauchy/Discretization/EquationPhysicsProfile.hpp"

#include "Cauchy/Implicit/ACMJacobianRow.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Cauchy/Physics/ACMFlux.hpp"
#include "Physics/RANS/Point.hpp"
#include "DNDS/Algebra/BlockMatrixStorage.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"
#include <array>
#include <algorithm>
#include <cmath>

namespace DNDS::Cauchy::CPU
{
    struct ACMJacobianPreparation
    {
        ACM::Parameters physics;
        FiniteVolumeStencilView stencil;
        SpanView<const BoundaryCondition> conditions;
        Algebra::DofComponentView<const real> input;
        RHSScalarView tau, radius;
        decltype(WallDistanceView::cellMean) wallMean;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> diagonal;
        Algebra::BlockMatrixArrayView<real> offDiagonal;
        real physicalInverseDt, residualScale;
        std::array<bool, 6> active;
        EquationPhysicsProfile profile;
        bool physicalBoundary = true;
        RigidFrameParameters frame;
        RHSScalarView incoming;
    };
    NumericalStatus PrepareACMFluxJacobian(const ACMJacobianPreparation &) noexcept;
}
