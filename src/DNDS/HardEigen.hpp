#pragma once
#include "Defines.hpp"

namespace DNDS::HardEigen
{

    real EigenLeastSquareInverse(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI, real svdTol = 0);

    /**
     * @brief
     *
     * @param A input
     * @param AI output, inverse of A
     * @param svdTol tolerance
     * @param mode
     *  if mode == 0, use regular SVD's "lsqminnorm", filters smallest singular values;
     *  if mode == 1, use inverse filtering, which filters out largest ones
     * @return real, condition number of A
     */
    real EigenLeastSquareInverse_Filtered(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI, real svdTol = 0, int mode = 0);

    Eigen::Index EigenLeastSquareSolve(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::MatrixXd &AIB);

    Eigen::Matrix3d Eigen3x3RealSymEigenDecomposition(const Eigen::Matrix3d &A);
    Eigen::Matrix2d Eigen2x2RealSymEigenDecomposition(const Eigen::Matrix2d &A);

    real Eigen3x3RealSymEigenDecompositionGetCond(const Eigen::Matrix3d &A);
    real Eigen3x3RealSymEigenDecompositionGetCond01(const Eigen::Matrix3d &A);
    real Eigen2x2RealSymEigenDecompositionGetCond(const Eigen::Matrix2d &A);

    Eigen::Matrix3d Eigen3x3RealSymEigenDecompositionNormalized(const Eigen::Matrix3d &A);
    Eigen::Matrix2d Eigen2x2RealSymEigenDecompositionNormalized(const Eigen::Matrix2d &A);
}
