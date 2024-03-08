#pragma once

#include "DNDS/Defines.hpp" // for correct  DNDS_SWITCH_INTELLISENSE
#include "Euler.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "Solver/Direct.hpp"

namespace DNDS::Euler
{
    // static const int nVarsFixed = 5;

    // DNDS_SWITCH_INTELLISENSE(
    //     template <int nVarsFixed = 5>,
    //     template <int nVarsFixed_masked>
    // )
    template <int nVarsFixed>
    struct JacobianLocalLU
        : public Direct::LocalLUBase<
              JacobianLocalLU<nVarsFixed>,
              Eigen::Matrix<real, nVarsFixed, nVarsFixed>,
              ArrayDOFV<nVarsFixed>>
    {
        using tLocalMat = ArrayEigenUniMatrixBatch<nVarsFixed, nVarsFixed>;
        using tComponent = Eigen::Matrix<real, nVarsFixed, nVarsFixed>;
        using tVec = ArrayDOFV<nVarsFixed>;
        using tBase = Direct::LocalLUBase<
            JacobianLocalLU<nVarsFixed>,
            Eigen::Matrix<real, nVarsFixed, nVarsFixed>,
            ArrayDOFV<nVarsFixed>>;
        // using tIndices = Geom::UnstructuredMesh::tLocalMatStruct;
        /**
         * \brief
         * each Row:
         * 0: diag,
         * [1,1+symLU->lowerTriStructure[iCell].size()), lower
         * [1+symLU->lowerTriStructure[iCell].size(), end), upper
         */
        tLocalMat LDU; //

        JacobianLocalLU(const ssp<Direct::SerialSymLUStructure> &nMesh, int nVarsC) : tBase{nMesh}
        {
            DNDS_assert(tBase::symLU->lowerTriStructure.size() == tBase::symLU->Num());
            LDU.Resize(tBase::symLU->Num(), nVarsC, nVarsC);
            for (index iCell = 0; iCell < tBase::symLU->Num(); iCell++)
            {
                LDU.ResizeRow(iCell,
                              1 + tBase::symLU->lowerTriStructure[iCell].size() +
                                  tBase::symLU->upperTriStructure[iCell].size());
                for (auto &v : LDU[iCell])
                    v.setZero();
            }
            LDU.Compress();
        }

        void setZero()
        {
            tBase::isDecomposed = false;
            for (index iCell = 0; iCell < tBase::symLU->Num(); iCell++)
                for (auto &v : LDU[iCell])
                    v.setZero();
        }

        auto GetDiag(index i) // compliant to LocalLUBase
        {
            return LDU(i, 0);
        }
        auto GetLower(index i, int iInLow) // compliant to LocalLUBase
        {
            return LDU(i, 1 + iInLow);
        }
        auto GetUpper(index i, int iInUpp) // compliant to LocalLUBase
        {
            return LDU(i, 1 + tBase::symLU->lowerTriStructure[i].size() + iInUpp);
        }

        void PrintLog()
        {
            log() << "nz Entries with Diag part inverse-ed" << std::endl;
            for (index iCell = 0; iCell < tBase::symLU->Num(); iCell++)
            {
                log() << "=== Row " << iCell << std::endl
                      << std::setprecision(10);
                for (auto &v : LDU[iCell])
                    log() << v << std::endl
                          << std::endl;
            }
        }

        tComponent InvertDiag(const tComponent& v)
        {
            tComponent AI;
            {
                auto luDiag = v.fullPivLu();
                DNDS_assert(luDiag.isInvertible());
                AI = luDiag.inverse();
            }
            {
                // Eigen::MatrixXd A = v;
                // Eigen::MatrixXd AII;
                // HardEigen::EigenLeastSquareInverse(A, AII, 0.0);
                // AI = AII;
            }
            return AI;
        }
    };

    template <int nVarsFixed>
    struct JacobianLocalLDLT
        : public Direct::LocalLDLTBase<
              JacobianLocalLDLT<nVarsFixed>,
              Eigen::Matrix<real, nVarsFixed, nVarsFixed>,
              ArrayDOFV<nVarsFixed>>
    {
        using tLocalMat = ArrayEigenUniMatrixBatch<nVarsFixed, nVarsFixed>;
        using tComponent = Eigen::Matrix<real, nVarsFixed, nVarsFixed>;
        using tVec = ArrayDOFV<nVarsFixed>;
        using tBase = Direct::LocalLDLTBase<
            JacobianLocalLDLT<nVarsFixed>,
            Eigen::Matrix<real, nVarsFixed, nVarsFixed>,
            ArrayDOFV<nVarsFixed>>;
        // using tIndices = Geom::UnstructuredMesh::tLocalMatStruct;
        /**
         * \brief
         * each Row:
         * 0: diag,
         * [1,1+symLU->lowerTriStructure[iCell].size()), lower
         * [1+symLU->lowerTriStructure[iCell].size(), end), upper
         */
        tLocalMat LDU; //

        JacobianLocalLDLT(const ssp<Direct::SerialSymLUStructure> &nMesh, int nVarsC) : tBase{nMesh}
        {
            DNDS_assert(tBase::symLU->lowerTriStructure.size() == tBase::symLU->Num());
            LDU.Resize(tBase::symLU->Num(), nVarsC, nVarsC);
            for (index iCell = 0; iCell < tBase::symLU->Num(); iCell++)
            {
                LDU.ResizeRow(iCell,
                              1 + tBase::symLU->lowerTriStructure[iCell].size());
                for (auto &v : LDU[iCell])
                    v.setZero();
            }
            LDU.Compress();
        }

        void setZero()
        {
            tBase::isDecomposed = false;
            for (index iCell = 0; iCell < tBase::symLU->Num(); iCell++)
                for (auto &v : LDU[iCell])
                    v.setZero();
        }

        auto GetDiag(index i) // compliant to LocalLDLTBase
        {
            return LDU(i, 0);
        }
        auto GetLower(index i, int iInLow) // compliant to LocalLDLTBase
        {
            return LDU(i, 1 + iInLow);
        }

        void PrintLog()
        {
            log() << "nz Entries with Diag part inverse-ed" << std::endl;
            for (index iCell = 0; iCell < tBase::symLU->Num(); iCell++)
            {
                log() << "=== Row " << iCell << std::endl
                      << std::setprecision(10);
                for (auto &v : LDU[iCell])
                    log() << v << std::endl
                          << std::endl;
            }
        }

        tComponent InvertDiag(const tComponent &v)
        {
            tComponent AI;
            {
                auto luDiag = v.fullPivLu();
                DNDS_assert(luDiag.isInvertible());
                AI = luDiag.inverse();
            }
            {
                // Eigen::MatrixXd A = v;
                // Eigen::MatrixXd AII;
                // HardEigen::EigenLeastSquareInverse(A, AII, 0.0);
                // AI = AII;
            }
            return AI;
        }
    };
}