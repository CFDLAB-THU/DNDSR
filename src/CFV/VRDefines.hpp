#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Geom/Quadrature.hpp"
#include "Geom/Mesh.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatrix.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"
#include "DNDS/ArrayDOF.hpp"
#include "Geom/BaseFunction.hpp"

namespace DNDS::CFV
{
    /**
     * @brief Returns the reconstruction DOF index range [start, end] (inclusive)
     * for a given polynomial order within the given spatial dimension.
     *
     * Reconstruction DOFs exclude the cell mean (order-0 DOF), so the indices
     * are 0-based into the rec coefficient matrix rows. For example, in 2D:
     *   order 1 -> [0, 1]   (2 DOFs: dx, dy)
     *   order 2 -> [2, 4]   (3 DOFs: dxx, dxy, dyy)
     *   order 3 -> [5, 8]   (4 DOFs: dxxx, dxxy, dxyy, dyyy)
     *
     * @param pOrder  Polynomial order (must be 1, 2, or 3)
     * @return std::pair<int, int> {LimStart, LimEnd} (both inclusive)
     */
    template <int dim>
    inline std::pair<int, int> GetRecDOFRange(int pOrder)
    {
        DNDS_assert(pOrder >= 1 && pOrder <= 3);
        int limStart = Geom::Base::GetNDof<dim>(pOrder - 1) - 1;
        int limEnd = Geom::Base::GetNDof<dim>(pOrder) - 2;
        return {limStart, limEnd};
    }
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