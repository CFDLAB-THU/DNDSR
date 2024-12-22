#pragma once

#include "DNDS/Defines.hpp"
#include "Geometric.hpp"
#include "json.hpp"

namespace DNDS::Geom::RBF
{
    enum RBFKernelType
    {
        UnknownRBFKernel = -1,
        Distance = 0,
        DistanceA1,
        InversedDistanceA1,
        InversedDistanceA1Compact,
        Gaussian,
        CPC2,
        CPC0,
    };

    NLOHMANN_JSON_SERIALIZE_ENUM(
        RBFKernelType,
        {{RBFKernelType::UnknownRBFKernel, nullptr},
         {RBFKernelType::Distance, "Distance"},
         {RBFKernelType::DistanceA1, "DistanceA1"},
         {RBFKernelType::InversedDistanceA1, "InversedDistanceA1"},
         {RBFKernelType::InversedDistanceA1Compact, "InversedDistanceA1Compact"},
         {RBFKernelType::Gaussian, "Gaussian"},
         {RBFKernelType::CPC2, "CPC2"},
         {RBFKernelType::CPC0, "CPC0"}})
    inline bool KernelIsCompact(RBFKernelType t)
    {
        return t == CPC0 || t == CPC2 || t == InversedDistanceA1Compact;
    }
}

namespace DNDS::Geom::RBF
{

    inline real
    GetMaxRij(const tSmallCoords &cent, const tSmallCoords &xs)
    {
        MatrixXR RiXj;
        RiXj.resize(cent.cols(), xs.cols());
        for (int iC = 0; iC < cent.cols(); iC++)
        {
            RiXj(iC, Eigen::all) = (xs.colwise() - cent(Eigen::all, iC)).colwise().norm();
        }
        return RiXj.array().maxCoeff();
    }

    template <class TIn>
    inline MatrixXR
    FRBFBasis(TIn RiXj, RBFKernelType kernel)
    {
        using real = DNDS::real;
        MatrixXR NiXj;
        switch (kernel)
        {
        /*****/ // using (modified) distance Kernel
        case Distance:
            NiXj = RiXj.array();
            break;
        case DistanceA1:
            NiXj = 1 + RiXj.array();
            break;
        case InversedDistanceA1:
            NiXj = (1 + RiXj.array()).inverse();
            break;
        case InversedDistanceA1Compact:
            NiXj = (0.2 + RiXj.array()).inverse() - 1 / 1.2;
            NiXj.array() *= (RiXj.array() < 1.0).template cast<real>(); // this kernel is bounded
            break;
        case CPC2:
            /*****/ // using CPC2 Kernel
            NiXj = (1 - RiXj.array()).square().square() * (RiXj.array() * 4 + 1);
            NiXj.array() *= (RiXj.array() < 1.0).template cast<real>(); // this kernel is bounded
            break;
        case CPC0:
            /*****/ // using CPC0 Kernel
            NiXj = (1 - RiXj.array()).square();
            NiXj.array() *= (RiXj.array() < 1.0).template cast<real>(); // this kernel is bounded
            break;
        case Gaussian:
            /*****/ // using Gaussian kernel
            NiXj = (-(RiXj.array().square())).exp();
            break;
        default:
            DNDS_assert(false);
        }
        return NiXj;
    }

    template <class Tcent, class Tx>
    inline MatrixXR // redurn Ni at Xj
    RBFCPC2(Tcent cent, Tx xs, real R, RBFKernelType kernel = Gaussian)
    {
        MatrixXR RiXj;
        RiXj.resize(cent.cols(), xs.cols());
        for (int iC = 0; iC < cent.cols(); iC++)
        {
            RiXj(iC, Eigen::all) = (xs.colwise() - cent(Eigen::all, iC)).colwise().norm() * (1. / R);
        }
        // MatrixXR NiXj = (1 + RiXj.array().square()).inverse();
        return FRBFBasis(RiXj, kernel);
    }

    template <class TF>
    inline MatrixXR
    RBFInterpolateSolveCoefs(const tSmallCoords &xs, const TF fs, real R, RBFKernelType kernel = Gaussian)
    {
        MatrixXR PT;
        PT.resize(4, xs.cols());
        PT(0, Eigen::all).setConstant(1);
        PT({1, 2, 3}, Eigen::all) = xs;
        MatrixXR M = RBFCPC2(xs, xs, R, kernel);
        MatrixXR A;
        A.setZero(xs.cols() + 4, xs.cols() + 4);
        A.topLeftCorner(xs.cols(), xs.cols()) = M;
        A.bottomLeftCorner(4, xs.cols()) = PT;
        A.topRightCorner(xs.cols(), 4) = PT.transpose();
        MatrixXR RHS;
        RHS.setZero(xs.cols() + 4, fs.cols());
        DNDS_assert(fs.rows() == xs.cols());
        RHS.topRows(xs.cols()) = fs;
        auto LDLT = A.ldlt();
        MatrixXR ret = LDLT.solve(RHS);
        return ret;
    }

    template <class TF>
    inline MatrixXR
    RBFInterpolateSolveCoefsNoPoly(const tSmallCoords &xs, const TF fs, real R, RBFKernelType kernel = Gaussian)
    {
        MatrixXR M = RBFCPC2(xs, xs, R, kernel);
        MatrixXR ret;
        ret.setZero(xs.cols() + 4, fs.cols());
        auto LDLTm = M.ldlt();
        ret.topRows(xs.cols()) = LDLTm.solve(fs);

        return ret;
    }
}
