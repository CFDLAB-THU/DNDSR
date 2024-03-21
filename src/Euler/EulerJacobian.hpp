#pragma once

#include "DNDS/Defines.hpp" // for correct  DNDS_SWITCH_INTELLISENSE
#include "Euler.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "Solver/Direct.hpp"

namespace DNDS::Euler
{
    // static const int nVarsFixed = 5;

    template <int nVarsFixed>
    class JacobianDiagBlock
    {
    public:
        using TU = Eigen::Vector<real, nVarsFixed>;
        using tComponent = Eigen::Matrix<real, nVarsFixed, nVarsFixed>;
        using tComponentDiag = Eigen::Vector<real, nVarsFixed>;

    private:
        ArrayDOFV<nVarsFixed> _dataDiag, _dataDiagInvert;
        DNDS::ArrayPair<DNDS::ArrayEigenMatrix<nVarsFixed, nVarsFixed>> _data, _dataInvert;
        bool hasInvert{false};
        int _mode{0};

    public:
        JacobianDiagBlock() {}

        void SetModeAndInit(int mode, int nVarsC, ArrayDOFV<nVarsFixed> &mock)
        {
            _mode = mode;
            if (isBlock())
            {
                DNDS_MAKE_SSP(_data.father, mock.father->getMPI());
                DNDS_MAKE_SSP(_data.son, mock.father->getMPI());
                _data.father->Resize(mock.father->Size(), nVarsC, nVarsC);
                _data.son->Resize(mock.son->Size() * 0, nVarsC, nVarsC);
                DNDS_MAKE_SSP(_dataInvert.father, mock.father->getMPI());
                DNDS_MAKE_SSP(_dataInvert.son, mock.father->getMPI());
                _dataInvert.father->Resize(mock.father->Size(), nVarsC, nVarsC);
                _dataInvert.son->Resize(mock.son->Size() * 0, nVarsC, nVarsC); // ! warning, sons are set to zero sizes
            }
            else
            {
                DNDS_MAKE_SSP(_dataDiag.father, mock.father->getMPI());
                DNDS_MAKE_SSP(_dataDiag.son, mock.father->getMPI());
                _dataDiag.father->Resize(mock.father->Size(), nVarsC, 1);
                _dataDiag.son->Resize(mock.son->Size() * 0, nVarsC, 1);
                DNDS_MAKE_SSP(_dataDiagInvert.father, mock.father->getMPI());
                DNDS_MAKE_SSP(_dataDiagInvert.son, mock.father->getMPI());
                _dataDiagInvert.father->Resize(mock.father->Size(), nVarsC, 1);
                _dataDiagInvert.son->Resize(mock.son->Size() * 0, nVarsC, 1);
            }
        }

        bool isBlock() const { return _mode; }

        auto getBlock(index iCell)
        {
            DNDS_assert(isBlock());
            return _data[iCell];
        }

        auto getDiag(index iCell)
        {
            DNDS_assert(!isBlock());
            return _dataDiag[iCell];
        }

        tComponent getValue(index iCell) const
        {
            if (isBlock())
                return _data[iCell];
            else
                return _dataDiag[iCell].asDiagonal();
        }

        index Size()
        {
            if (isBlock())
                return _data.Size();
            else
                return _dataDiag.Size();
        }

        void GetInvert()
        {
            if (!hasInvert)
            {
                for (index iCell = 0; iCell < Size(); iCell++)
                    if (isBlock())
                    {
                        auto luDiag = _data[iCell].fullPivLu();
                        DNDS_assert(luDiag.isInvertible());
                        _dataInvert[iCell] = luDiag.inverse();
                    }
                    else
                    {
                        DNDS_assert(_dataDiag[iCell].array().abs().minCoeff() != 0);
                        _dataDiagInvert[iCell] = _dataDiag[iCell].array().inverse();
                    }
                hasInvert = true;
            }
        }

        template <class TV>
        TU MatVecLeft(index iCell, TV v)
        {
            if (isBlock())
                return _data[iCell] * v;
            else
                return _dataDiag[iCell].asDiagonal() * v;
        }

        template <class TV>
        TU MatVecLeftInvert(index iCell, TV v)
        {
            if (isBlock())
                return _dataInvert[iCell] * v;
            else
                return _dataDiagInvert[iCell].asDiagonal() * v;
        }

        void clearValues()
        {
            if (isBlock())
            {
                for (index i = 0; i < _data.Size(); i++)
                    _data[i].setZero();
            }
            else
            {
                for (index i = 0; i < _dataDiag.Size(); i++)
                    _dataDiag[i].setZero();
            }
            hasInvert = false;
        }
    };

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