#pragma once
/// @file HardEigen.hpp
/// @brief Robust linear-algebra primitives for small / moderately-sized
/// Eigen matrices that are more numerically careful than Eigen's default paths.
///
/// These are used on the "hot" side of reconstruction / limiter code where
/// matrices may be nearly rank-deficient and a plain `A.inverse()` or
/// `A.jacobiSvd().solve(b)` would introduce large errors.

#include "Defines.hpp"

namespace DNDS::HardEigen
{

    /// @brief Moore-Penrose pseudoinverse via SVD, dropping singular values below
    /// `svdTol` (relative to the largest). Returns the largest / smallest ratio
    /// (condition number).
    real EigenLeastSquareInverse(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI, real svdTol = 0);

    /**
     * @brief Pseudoinverse with a choice of singular-value filter.
     *
     * @param A       Input matrix.
     * @param AI      Output pseudoinverse.
     * @param svdTol  Filter tolerance (relative to the largest singular value).
     * @param mode    Filter selection:
     *                - `0`: standard "lsqminnorm" -- drop *smallest* singular values
     *                  below the tolerance (stabilises rank-deficient systems);
     *                - `1`: drop *largest* singular values above `1/svdTol` relative
     *                  to the smallest (inverse filtering, rarely needed).
     * @return Condition number of `A` (ratio of largest to smallest
     *         post-filter singular value).
     */
    real EigenLeastSquareInverse_Filtered(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI, real svdTol = 0, int mode = 0);

    /// @brief Least-squares solve `A * AIB ~= B` via a rank-revealing QR-style
    /// decomposition; returns the computed rank of `A`.
    Eigen::Index EigenLeastSquareSolve(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::MatrixXd &AIB);

    /// @brief Analytic eigen-decomposition of a 3x3 real symmetric matrix.
    /// Returns the eigenvector matrix (columns = eigenvectors, scaled so that
    /// diagonal == eigenvalues).
    Eigen::Matrix3d Eigen3x3RealSymEigenDecomposition(const Eigen::Matrix3d &A);
    /// @brief Analytic 2x2 analogue of #Eigen3x3RealSymEigenDecomposition.
    Eigen::Matrix2d Eigen2x2RealSymEigenDecomposition(const Eigen::Matrix2d &A);

    /// @brief Condition number of a 3x3 SPD matrix from its eigenvalues.
    real Eigen3x3RealSymEigenDecompositionGetCond(const Eigen::Matrix3d &A);
    /// @brief Like #Eigen3x3RealSymEigenDecompositionGetCond but returns
    /// `lambda0 / lambda1` only (ignores the smallest eigenvalue).
    real Eigen3x3RealSymEigenDecompositionGetCond01(const Eigen::Matrix3d &A);
    /// @brief 2x2 analogue of #Eigen3x3RealSymEigenDecompositionGetCond.
    real Eigen2x2RealSymEigenDecompositionGetCond(const Eigen::Matrix2d &A);

    /// @brief Eigen-decomposition with eigenvector columns normalised to unit length.
    Eigen::Matrix3d Eigen3x3RealSymEigenDecompositionNormalized(const Eigen::Matrix3d &A);
    /// @brief 2x2 analogue of #Eigen3x3RealSymEigenDecompositionNormalized.
    Eigen::Matrix2d Eigen2x2RealSymEigenDecompositionNormalized(const Eigen::Matrix2d &A);
}
