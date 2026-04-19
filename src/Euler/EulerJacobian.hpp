/** @file EulerJacobian.hpp
 *  @brief Jacobian storage and factorization structures for implicit time stepping.
 *
 *  Provides block-diagonal and sparse-local Jacobian data structures used in
 *  the implicit solver (e.g. backward-Euler, LUSGS, GMRES with ILU/LDLT
 *  preconditioning):
 *
 *  - JacobianDiagBlock: per-cell diagonal or block-diagonal Jacobian with
 *    forward multiply and inverse operations.
 *  - JacobianLocalLU: full LU factorization of the cell-local Jacobian
 *    using the mesh adjacency structure (SerialSymLUStructure).
 *  - JacobianLocalLDLT: symmetric LDLT factorization variant (lower + diagonal only).
 *
 *  All classes are templated on nVarsFixed for compile-time Eigen sizing.
 */
#pragma once

#include "DNDS/Defines.hpp" // for correct  DNDS_SWITCH_INTELLISENSE
#include "Euler.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "Solver/Direct.hpp"

namespace DNDS::Euler
{
    /**
     * @brief Per-cell diagonal or block-diagonal Jacobian storage for implicit time stepping.
     *
     * Operates in one of two modes selected by SetModeAndInit():
     * - **Scalar-diagonal** (mode 0): stores an nVars-length diagonal vector per cell.
     *   Matrix-vector products are element-wise multiplications.
     * - **Matrix-block** (mode 1): stores a full nVars×nVars matrix per cell.
     *   Matrix-vector products are dense matrix multiplies.
     *
     * Both modes support lazy inversion via GetInvert(). The scalar mode inverts
     * element-wise; the matrix mode uses diagonal-preconditioned full-pivot LU
     * (InvertDiag pattern) for robustness on ill-conditioned blocks.
     *
     * @tparam nVarsFixed  Compile-time number of conservative variables
     *                     (or Eigen::Dynamic).
     */
    template <int nVarsFixed>
    class JacobianDiagBlock
    {
    public:
        using TU = Eigen::Vector<real, nVarsFixed>;                  ///< State vector type.
        using tComponent = Eigen::Matrix<real, nVarsFixed, nVarsFixed>; ///< Full block matrix type.
        using tComponentDiag = Eigen::Vector<real, nVarsFixed>;      ///< Diagonal vector type.

    private:
        ArrayDOFV<nVarsFixed> _dataDiag, _dataDiagInvert;           ///< Diagonal data and its inverse (scalar mode).
        DNDS::ArrayPair<DNDS::ArrayEigenMatrix<nVarsFixed, nVarsFixed>> _data, _dataInvert; ///< Block data and its inverse (matrix mode).
        bool hasInvert{false};                                      ///< True after GetInvert() has computed the inverse.
        int _mode{0};                                               ///< Storage mode: 0 = scalar-diagonal, nonzero = matrix-block.

    public:
        /// @brief Default constructor; mode and storage are uninitialized until SetModeAndInit().
        JacobianDiagBlock() {}

        /**
         * @brief Select the storage mode and allocate arrays matching @p mock's layout.
         *
         * In block mode (mode != 0), allocates nVarsC×nVarsC matrices per cell.
         * In scalar-diagonal mode (mode == 0), allocates nVarsC-length vectors.
         * Ghost (son) arrays are allocated with zero size.
         *
         * @param mode    Storage mode: 0 = scalar-diagonal, nonzero = matrix-block.
         * @param nVarsC  Runtime number of variables (must equal nVarsFixed when fixed).
         * @param mock    A reference ArrayDOFV whose MPI layout and sizes are mimicked.
         */
        void SetModeAndInit(int mode, int nVarsC, ArrayDOFV<nVarsFixed> &mock)
        {
            _mode = mode;
            if (isBlock())
            {
                _data.InitPair("JacobianLocalLU::SetModeAndInit::_data", mock.father->getMPI());
                _data.father->Resize(mock.father->Size(), nVarsC, nVarsC);
                _data.son->Resize(mock.son->Size() * 0, nVarsC, nVarsC);
                _dataInvert.InitPair("JacobianLocalLU::SetModeAndInit::_dataInvert", mock.father->getMPI());
                _dataInvert.father->Resize(mock.father->Size(), nVarsC, nVarsC);
                _dataInvert.son->Resize(mock.son->Size() * 0, nVarsC, nVarsC); // ! warning, sons are set to zero sizes
            }
            else
            {
                _dataDiag.InitPair("JacobianLocalLU::SetModeAndInit::_dataDiag", mock.father->getMPI());
                _dataDiag.father->Resize(mock.father->Size(), nVarsC, 1);
                _dataDiag.son->Resize(mock.son->Size() * 0, nVarsC, 1);
                _dataDiagInvert.InitPair("JacobianLocalLU::SetModeAndInit::_dataDiagInvert", mock.father->getMPI());
                _dataDiagInvert.father->Resize(mock.father->Size(), nVarsC, 1);
                _dataDiagInvert.son->Resize(mock.son->Size() * 0, nVarsC, 1);
            }
        }

        /// @brief Return true if using matrix-block mode, false for scalar-diagonal.
        bool isBlock() const { return _mode; }

        /**
         * @brief Access the full nVars×nVars block matrix for cell @p iCell (block mode only).
         * @param iCell  Local cell index.
         * @return Eigen map/ref to the stored block matrix.
         */
        auto getBlock(index iCell)
        {
            DNDS_assert(isBlock());
            return _data[iCell];
        }

        /**
         * @brief Access the diagonal vector for cell @p iCell (scalar-diagonal mode only).
         * @param iCell  Local cell index.
         * @return Eigen map/ref to the stored diagonal vector.
         */
        auto getDiag(index iCell)
        {
            DNDS_assert(!isBlock());
            return _dataDiag[iCell];
        }

        /**
         * @brief Return the Jacobian as a full matrix for cell @p iCell (either mode).
         *
         * In scalar-diagonal mode the returned matrix is constructed as asDiagonal().
         *
         * @param iCell  Local cell index.
         * @return nVars×nVars matrix (by value).
         */
        tComponent getValue(index iCell) const
        {
            if (isBlock())
                return _data[iCell];
            else
                return _dataDiag[iCell].asDiagonal();
        }

        /// @brief Return the number of cells in the local partition.
        index Size()
        {
            if (isBlock())
                return _data.Size();
            else
                return _dataDiag.Size();
        }

        /**
         * @brief Compute and cache the inverse of every cell's Jacobian block.
         *
         * In matrix-block mode, uses diagonal-preconditioned full-pivot LU
         * (the InvertDiag pattern) for robustness on ill-conditioned blocks.
         * In scalar-diagonal mode, inverts element-wise.
         *
         * This is a lazy operation: subsequent calls are no-ops until clearValues()
         * invalidates the cache.
         */
        void GetInvert()
        {
            if (!hasInvert)
            {
#if defined(DNDS_DIST_MT_USE_OMP)
#pragma omp parallel for schedule(runtime)
#endif
                for (index iCell = 0; iCell < Size(); iCell++)
                    if (isBlock())
                    {
                        DNDS_assert(_data[iCell].diagonal().array().abs().minCoeff() != 0);
                        tComponent preCon = _data[iCell].diagonal().array().inverse().matrix().asDiagonal() * _data[iCell];
                        auto luDiag = preCon.fullPivLu();
                        DNDS_assert(luDiag.isInvertible());
                        _dataInvert[iCell] = luDiag.inverse() * _data[iCell].diagonal().array().inverse().matrix().asDiagonal();
                        if (!_dataInvert[iCell].allFinite() || _dataInvert[iCell].hasNaN())
                        {
                            std::cout << "xxxx"
                                      << "\n";
                            std::cout << _data[iCell] << "\n";
                            std::cout << preCon << "\n";
                            std::cout << luDiag.inverse() << "\n";
                            std::cout << std::endl;
                            DNDS_assert(false);
                        }
                    }
                    else
                    {
                        DNDS_assert(_dataDiag[iCell].array().abs().minCoeff() != 0);
                        _dataDiagInvert[iCell] = _dataDiag[iCell].array().inverse();
                    }
                hasInvert = true;
            }
        }

        /**
         * @brief Left-multiply a vector by the Jacobian block for cell @p iCell.
         * @tparam TV    Vector type compatible with Eigen matrix-vector product.
         * @param iCell  Local cell index.
         * @param v      Input vector of length nVars.
         * @return Result of J * v.
         */
        template <class TV>
        TU MatVecLeft(index iCell, TV v)
        {
            if (isBlock())
                return _data[iCell] * v;
            else
                return _dataDiag[iCell].asDiagonal() * v;
        }

        /**
         * @brief Left-multiply a vector by the *inverse* Jacobian block for cell @p iCell.
         *
         * GetInvert() must have been called first; otherwise behavior is undefined.
         *
         * @tparam TV    Vector type compatible with Eigen matrix-vector product.
         * @param iCell  Local cell index.
         * @param v      Input vector of length nVars.
         * @return Result of J^{-1} * v.
         */
        template <class TV>
        TU MatVecLeftInvert(index iCell, TV v)
        {
            if (isBlock())
                return _dataInvert[iCell] * v;
            else
                return _dataDiagInvert[iCell].asDiagonal() * v;
        }

        /// @brief Zero all stored values and invalidate the cached inverse.
        void clearValues()
        {
            if (isBlock())
            {
#if defined(DNDS_DIST_MT_USE_OMP)
#pragma omp parallel for schedule(static)
#endif
                for (index i = 0; i < _data.Size(); i++)
                    _data[i].setZero();
            }
            else
            {
#if defined(DNDS_DIST_MT_USE_OMP)
#pragma omp parallel for schedule(static)
#endif
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
    /**
     * @brief Complete LU factorization of the cell-local Jacobian for GMRES preconditioning.
     *
     * Stores diagonal (D), lower-triangular (L), and upper-triangular (U) blocks
     * indexed by the mesh adjacency graph (SerialSymLUStructure). Inherits the
     * decompose/solve interface from Direct::LocalLUBase via CRTP.
     *
     * **Row layout in LDU for each cell iCell:**
     * - Index 0: diagonal block D.
     * - Indices [1, 1 + nLower): lower-triangular off-diagonal blocks L.
     * - Indices [1 + nLower, end): upper-triangular off-diagonal blocks U.
     *
     * where nLower = symLU->lowerTriStructure[iCell].size().
     *
     * @tparam nVarsFixed  Compile-time number of conservative variables
     *                     (or Eigen::Dynamic).
     */
    template <int nVarsFixed>
    struct JacobianLocalLU
        : public Direct::LocalLUBase<
              JacobianLocalLU<nVarsFixed>,
              Eigen::Matrix<real, nVarsFixed, nVarsFixed>,
              ArrayDOFV<nVarsFixed>>
    {
        using tLocalMat = ArrayEigenUniMatrixBatch<nVarsFixed, nVarsFixed>; ///< Batch storage for nVars×nVars blocks.
        using tComponent = Eigen::Matrix<real, nVarsFixed, nVarsFixed>;    ///< Single block matrix type.
        using tVec = ArrayDOFV<nVarsFixed>;                               ///< Distributed vector type.
        using tBase = Direct::LocalLUBase<
            JacobianLocalLU<nVarsFixed>,
            Eigen::Matrix<real, nVarsFixed, nVarsFixed>,
            ArrayDOFV<nVarsFixed>>;                                       ///< CRTP base providing decompose/solve.
        // using tIndices = Geom::UnstructuredMesh::tLocalMatStruct;
        /**
         * @brief Sparse row storage for D, L, U blocks.
         *
         * Each row i stores blocks in the order:
         * [0] = diag,
         * [1, 1 + symLU->lowerTriStructure[i].size()) = lower-triangular,
         * [1 + symLU->lowerTriStructure[i].size(), end) = upper-triangular.
         */
        tLocalMat LDU;

        /**
         * @brief Construct and allocate the LU storage for the given adjacency structure.
         * @param nMesh   Shared pointer to the symmetric LU sparsity structure.
         * @param nVarsC  Runtime number of conservative variables.
         */
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

        /// @brief Zero all D, L, U blocks and mark the factorization as not yet decomposed.
        void setZero()
        {
            tBase::isDecomposed = false;
#if defined(DNDS_DIST_MT_USE_OMP)
#pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < tBase::symLU->Num(); iCell++)
                for (auto &v : LDU[iCell])
                    v.setZero();
        }

        /// @brief Return the diagonal block for row @p i (compliant with LocalLUBase CRTP).
        /// @param i  Cell (row) index.
        auto GetDiag(index i) // compliant to LocalLUBase
        {
            return LDU(i, 0);
        }
        /// @brief Return lower-triangular block @p iInLow for row @p i (compliant with LocalLUBase CRTP).
        /// @param i       Cell (row) index.
        /// @param iInLow  Index within the lower-triangular neighbor list.
        auto GetLower(index i, int iInLow) // compliant to LocalLUBase
        {
            return LDU(i, 1 + iInLow);
        }
        /// @brief Return upper-triangular block @p iInUpp for row @p i (compliant with LocalLUBase CRTP).
        /// @param i       Cell (row) index.
        /// @param iInUpp  Index within the upper-triangular neighbor list.
        auto GetUpper(index i, int iInUpp) // compliant to LocalLUBase
        {
            return LDU(i, 1 + tBase::symLU->lowerTriStructure[i].size() + iInUpp);
        }

        /// @brief Print all non-zero entries (with diagonal inverted) to the log stream.
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

        /**
         * @brief Invert a diagonal block using diagonal-preconditioned full-pivot LU.
         *
         * Preconditions the matrix by scaling rows with the inverse of the diagonal,
         * then applies Eigen's fullPivLu(). This two-step approach improves numerical
         * stability for ill-conditioned Jacobian blocks.
         *
         * @param v  The nVars×nVars block matrix to invert.
         * @return The inverse matrix v^{-1}.
         */
        tComponent InvertDiag(const tComponent &v)
        {
            tComponent AI;
            {
                DNDS_assert(v.diagonal().array().abs().minCoeff() != 0);
                tComponent preCon = v.diagonal().array().inverse().matrix().asDiagonal() * v;
                auto luDiag = preCon.fullPivLu();
                DNDS_assert_info(luDiag.isInvertible(), [&]()
                                 {
                    std::cerr << v << "\n\n" << preCon << "\n\n";
                    return "=== error info ==="; }());
                AI = luDiag.inverse() * v.diagonal().array().inverse().matrix().asDiagonal();
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

    /**
     * @brief Symmetric LDLT factorization of the cell-local Jacobian.
     *
     * Similar to JacobianLocalLU but exploits symmetry: only the diagonal and
     * lower-triangular blocks are stored (no separate upper-triangular storage).
     * Inherits the decompose/solve interface from Direct::LocalLDLTBase via CRTP.
     *
     * **Row layout in LDU for each cell iCell:**
     * - Index 0: diagonal block D.
     * - Indices [1, 1 + nLower): lower-triangular off-diagonal blocks L.
     *
     * The upper triangle is implicitly L^T.
     *
     * @tparam nVarsFixed  Compile-time number of conservative variables
     *                     (or Eigen::Dynamic).
     */
    template <int nVarsFixed>
    struct JacobianLocalLDLT
        : public Direct::LocalLDLTBase<
              JacobianLocalLDLT<nVarsFixed>,
              Eigen::Matrix<real, nVarsFixed, nVarsFixed>,
              ArrayDOFV<nVarsFixed>>
    {
        using tLocalMat = ArrayEigenUniMatrixBatch<nVarsFixed, nVarsFixed>; ///< Batch storage for nVars×nVars blocks.
        using tComponent = Eigen::Matrix<real, nVarsFixed, nVarsFixed>;    ///< Single block matrix type.
        using tVec = ArrayDOFV<nVarsFixed>;                               ///< Distributed vector type.
        using tBase = Direct::LocalLDLTBase<
            JacobianLocalLDLT<nVarsFixed>,
            Eigen::Matrix<real, nVarsFixed, nVarsFixed>,
            ArrayDOFV<nVarsFixed>>;                                       ///< CRTP base providing decompose/solve.
        // using tIndices = Geom::UnstructuredMesh::tLocalMatStruct;
        /**
         * @brief Sparse row storage for D and L blocks.
         *
         * Each row i stores blocks in the order:
         * [0] = diag,
         * [1, 1 + symLU->lowerTriStructure[i].size()) = lower-triangular.
         */
        tLocalMat LDU;

        /**
         * @brief Construct and allocate the LDLT storage for the given adjacency structure.
         *
         * Unlike JacobianLocalLU, only diagonal + lower blocks are allocated
         * (no upper-triangular storage).
         *
         * @param nMesh   Shared pointer to the symmetric LU sparsity structure.
         * @param nVarsC  Runtime number of conservative variables.
         */
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

        /// @brief Zero all D and L blocks and mark the factorization as not yet decomposed.
        void setZero()
        {
            tBase::isDecomposed = false;
#if defined(DNDS_DIST_MT_USE_OMP)
#pragma omp parallel for schedule(static)
#endif
            for (index iCell = 0; iCell < tBase::symLU->Num(); iCell++)
                for (auto &v : LDU[iCell])
                    v.setZero();
        }

        /// @brief Return the diagonal block for row @p i (compliant with LocalLDLTBase CRTP).
        /// @param i  Cell (row) index.
        auto GetDiag(index i) // compliant to LocalLDLTBase
        {
            return LDU(i, 0);
        }
        /// @brief Return lower-triangular block @p iInLow for row @p i (compliant with LocalLDLTBase CRTP).
        /// @param i       Cell (row) index.
        /// @param iInLow  Index within the lower-triangular neighbor list.
        auto GetLower(index i, int iInLow) // compliant to LocalLDLTBase
        {
            return LDU(i, 1 + iInLow);
        }

        /// @brief Print all non-zero entries (with diagonal inverted) to the log stream.
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

        /**
         * @brief Invert a diagonal block using diagonal-preconditioned full-pivot LU.
         *
         * Same algorithm as JacobianLocalLU::InvertDiag: preconditions with the
         * diagonal inverse, then applies Eigen's fullPivLu().
         *
         * @param v  The nVars×nVars block matrix to invert.
         * @return The inverse matrix v^{-1}.
         */
        tComponent InvertDiag(const tComponent &v)
        {
            tComponent AI;
            {
                DNDS_assert(v.diagonal().array().abs().minCoeff() != 0);
                tComponent preCon = v.diagonal().array().inverse().matrix().asDiagonal() * v;
                auto luDiag = preCon.fullPivLu();
                DNDS_assert(luDiag.isInvertible());
                AI = luDiag.inverse() * v.diagonal().array().inverse().matrix().asDiagonal();
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