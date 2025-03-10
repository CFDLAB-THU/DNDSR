#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Geom/Quadrature.hpp"
#include "Geom/Mesh.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatirx.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"

namespace DNDS::CFV
{
    using tCoeffPair = DNDS::ArrayPair<DNDS::ParArray<real, NonUniformSize>>;
    using tCoeff = decltype(tCoeffPair::father);
    using t3VecsPair = DNDS::ArrayPair<DNDS::ArrayEigenUniMatrixBatch<3, 1>>;
    using t3Vecs = decltype(t3VecsPair::father);
    using t3VecPair = Geom::tCoordPair;
    using t3Vec = Geom::tCoord;
    using t3MatPair = DNDS::ArrayPair<DNDS::ArrayEigenMatrix<3, 3>>;
    using t3Mat = decltype(t3MatPair::father);

    using tVVecPair = ::DNDS::ArrayPair<DNDS::ArrayEigenVector<DynamicSize>>;
    using tVVec = decltype(tVVecPair::father);
    using tMatsPair = DNDS::ArrayPair<DNDS::ArrayEigenUniMatrixBatch<DynamicSize, DynamicSize>>;
    using tMats = decltype(tMatsPair::father);
    using tVecsPair = DNDS::ArrayPair<DNDS::ArrayEigenUniMatrixBatch<DynamicSize, 1>>;
    using tVecs = decltype(tVecsPair::father);
    using tVMatPair = DNDS::ArrayPair<DNDS::ArrayEigenMatrix<DynamicSize, DynamicSize>>;
    using tVMat = decltype(tVMatPair::father);
    
    // Corresponds to mean/rec dofs
    template <int nVarsFixed>
    using tURec = DNDS::ArrayPair<DNDS::ArrayEigenMatrix<DynamicSize, nVarsFixed>>;
    template <int nVarsFixed>
    using tUDof = DNDS::ArrayPair<DNDS::ArrayEigenMatrix<nVarsFixed, 1>>;
    template <int nVarsFixed, int gDim>
    using tUGrad = DNDS::ArrayPair<DNDS::ArrayEigenMatrix<gDim, nVarsFixed>>;

    using tScalarPair = DNDS::ArrayPair<DNDS::ParArray<real, 1>>;
    using tScalar = decltype(tScalarPair::father);

    struct RecAtr
    {
        real relax = UnInitReal;
        uint8_t Order = -1;
        uint8_t NDOF = -1;
        uint8_t NDIFF = -1;
        uint8_t intOrder = 1;
    };
}