#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Geom/Quadrature.hpp"
#include "Geom/Mesh.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatrix.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"
#include "DNDS/ArrayDOF.hpp"

namespace DNDS::CFV
{
    using tCoeffPair = DNDS::ArrayPair<DNDS::ArrayEigenVector<NonUniformSize>>;
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

    struct RecAtr
    {
        real relax = UnInitReal;
        uint8_t Order = static_cast<uint8_t>(-1);
        uint8_t NDOF = static_cast<uint8_t>(-1);
        uint8_t NDIFF = static_cast<uint8_t>(-1);
        uint8_t intOrder = 1;

        static MPI_Datatype CommType()
        {
            //! TODO: make this not endian-sensitive: use static registering
            static_assert(sizeof(RecAtr) <= 1ULL * 16);
            return MPI_INT8_T;
        }
        static int CommMult() { return 16; }
        static std::string pybind11_name() { return "RecAtr"; }
    };

    using tRecAtrPair = DNDS::ArrayPair<DNDS::ParArray<RecAtr, 1>>;
    using tRecAtr = decltype(tRecAtrPair::father);
}

namespace DNDS::CFV
{
    // Corresponds to mean/rec dofs
    template <int nVarsFixed>
    using tURec = ArrayDof<DynamicSize, nVarsFixed>;

    template <int nVarsFixed>
    using tUDof = ArrayDof<nVarsFixed, 1>;

    template <int nVarsFixed, int gDim>
    using tUGrad = ArrayDof<gDim, nVarsFixed>;

    using tScalarPair = DNDS::ArrayPair<DNDS::ArrayEigenVector<1>>;
    using tScalar = decltype(tScalarPair::father);
}

namespace DNDS
{
//     DNDS_DEVICE_STORAGE_BASE_DELETER_INST(CFV::RecAtr, extern)
//     DNDS_DEVICE_STORAGE_INST(CFV::RecAtr, DeviceBackend::Host, extern)
// #ifdef DNDS_USE_CUDA
//     DNDS_DEVICE_STORAGE_INST(CFV::RecAtr, DeviceBackend::CUDA, extern)
// #endif
}