#pragma once

#include "BaseFunction.hpp"

namespace DNDS::Geom::Base
{
    template <int dim = 3, int rank = 0, int powV = 1, class VLe, class VRi>
    real NormSymDiffOrderTensorV(VLe &&Le, VRi &&Ri)
    {
        real ret = 0;
        if constexpr (dim == 3)
        {
            if constexpr (powV == 1)
            {
                if constexpr (rank == 0)
                {
                    ret += Le(0, 0) * Ri(0, 0) * diffNCombs[0];
                }
                else if constexpr (rank == 1)
                {
                    ret += Le(0, 0) * Ri(0, 0) * diffNCombs[0 + 1];
                    ret += Le(1, 0) * Ri(1, 0) * diffNCombs[1 + 1];
                    ret += Le(2, 0) * Ri(2, 0) * diffNCombs[2 + 1];
                }
                else if constexpr (rank == 2)
                {
                    ret += Le(0, 0) * Ri(0, 0) * diffNCombs[0 + 4];
                    ret += Le(1, 0) * Ri(1, 0) * diffNCombs[1 + 4];
                    ret += Le(2, 0) * Ri(2, 0) * diffNCombs[2 + 4];
                    ret += Le(3, 0) * Ri(3, 0) * diffNCombs[3 + 4];
                    ret += Le(4, 0) * Ri(4, 0) * diffNCombs[4 + 4];
                    ret += Le(5, 0) * Ri(5, 0) * diffNCombs[5 + 4];
                }
                else if constexpr (rank == 3)
                {
                    ret += Le(0, 0) * Ri(0, 0) * diffNCombs[0 + 10];
                    ret += Le(1, 0) * Ri(1, 0) * diffNCombs[1 + 10];
                    ret += Le(2, 0) * Ri(2, 0) * diffNCombs[2 + 10];
                    ret += Le(3, 0) * Ri(3, 0) * diffNCombs[3 + 10];
                    ret += Le(4, 0) * Ri(4, 0) * diffNCombs[4 + 10];
                    ret += Le(5, 0) * Ri(5, 0) * diffNCombs[5 + 10];
                    ret += Le(6, 0) * Ri(6, 0) * diffNCombs[6 + 10];
                    ret += Le(7, 0) * Ri(7, 0) * diffNCombs[7 + 10];
                    ret += Le(8, 0) * Ri(8, 0) * diffNCombs[8 + 10];
                    ret += Le(9, 0) * Ri(9, 0) * diffNCombs[9 + 10];
                }
                else
                {
                    DNDS_assert(false);
                }
            }
            else
            {
                if constexpr (rank == 0)
                {
                    ret += Le(0, 0) * Ri(0, 0) * std::pow(diffNCombs[0], powV);
                }
                else if constexpr (rank == 1)
                {
                    ret += Le(0, 0) * Ri(0, 0) * std::pow(diffNCombs[0 + 1], powV);
                    ret += Le(1, 0) * Ri(1, 0) * std::pow(diffNCombs[1 + 1], powV);
                    ret += Le(2, 0) * Ri(2, 0) * std::pow(diffNCombs[2 + 1], powV);
                }
                else if constexpr (rank == 2)
                {
                    ret += Le(0, 0) * Ri(0, 0) * std::pow(diffNCombs[0 + 4], powV);
                    ret += Le(1, 0) * Ri(1, 0) * std::pow(diffNCombs[1 + 4], powV);
                    ret += Le(2, 0) * Ri(2, 0) * std::pow(diffNCombs[2 + 4], powV);
                    ret += Le(3, 0) * Ri(3, 0) * std::pow(diffNCombs[3 + 4], powV);
                    ret += Le(4, 0) * Ri(4, 0) * std::pow(diffNCombs[4 + 4], powV);
                    ret += Le(5, 0) * Ri(5, 0) * std::pow(diffNCombs[5 + 4], powV);
                }
                else if constexpr (rank == 3)
                {
                    ret += Le(0, 0) * Ri(0, 0) * std::pow(diffNCombs[0 + 10], powV);
                    ret += Le(1, 0) * Ri(1, 0) * std::pow(diffNCombs[1 + 10], powV);
                    ret += Le(2, 0) * Ri(2, 0) * std::pow(diffNCombs[2 + 10], powV);
                    ret += Le(3, 0) * Ri(3, 0) * std::pow(diffNCombs[3 + 10], powV);
                    ret += Le(4, 0) * Ri(4, 0) * std::pow(diffNCombs[4 + 10], powV);
                    ret += Le(5, 0) * Ri(5, 0) * std::pow(diffNCombs[5 + 10], powV);
                    ret += Le(6, 0) * Ri(6, 0) * std::pow(diffNCombs[6 + 10], powV);
                    ret += Le(7, 0) * Ri(7, 0) * std::pow(diffNCombs[7 + 10], powV);
                    ret += Le(8, 0) * Ri(8, 0) * std::pow(diffNCombs[8 + 10], powV);
                    ret += Le(9, 0) * Ri(9, 0) * std::pow(diffNCombs[9 + 10], powV);
                }
                else
                {
                    DNDS_assert(false);
                }
            }
        }
        else
        {
            if constexpr (powV == 1)
            {
                if constexpr (rank == 0)
                {
                    ret += Le(0) * Ri(0) * diffNCombs2D[0];
                }
                else if constexpr (rank == 1)
                {
                    ret += Le(0) * Ri(0) * diffNCombs2D[0 + 1];
                    ret += Le(1) * Ri(1) * diffNCombs2D[1 + 1];
                }
                else if constexpr (rank == 2)
                {
                    ret += Le(0) * Ri(0) * diffNCombs2D[0 + 3];
                    ret += Le(1) * Ri(1) * diffNCombs2D[1 + 3];
                    ret += Le(2) * Ri(2) * diffNCombs2D[2 + 3];
                }
                else if constexpr (rank == 3)
                {
                    ret += Le(0) * Ri(0) * diffNCombs2D[0 + 6];
                    ret += Le(1) * Ri(1) * diffNCombs2D[1 + 6];
                    ret += Le(2) * Ri(2) * diffNCombs2D[2 + 6];
                    ret += Le(3) * Ri(3) * diffNCombs2D[3 + 6];
                }
                else
                {
                    DNDS_assert(false);
                }
            }
            else
            {
                if constexpr (rank == 0)
                {
                    ret += Le(0) * Ri(0) * std::pow(diffNCombs2D[0], powV);
                }
                else if constexpr (rank == 1)
                {
                    ret += Le(0) * Ri(0) * std::pow(diffNCombs2D[0 + 1], powV);
                    ret += Le(1) * Ri(1) * std::pow(diffNCombs2D[1 + 1], powV);
                }
                else if constexpr (rank == 2)
                {
                    ret += Le(0) * Ri(0) * std::pow(diffNCombs2D[0 + 3], powV);
                    ret += Le(1) * Ri(1) * std::pow(diffNCombs2D[1 + 3], powV);
                    ret += Le(2) * Ri(2) * std::pow(diffNCombs2D[2 + 3], powV);
                }
                else if constexpr (rank == 3)
                {
                    ret += Le(0) * Ri(0) * std::pow(diffNCombs2D[0 + 6], powV);
                    ret += Le(1) * Ri(1) * std::pow(diffNCombs2D[1 + 6], powV);
                    ret += Le(2) * Ri(2) * std::pow(diffNCombs2D[2 + 6], powV);
                    ret += Le(3) * Ri(3) * std::pow(diffNCombs2D[3 + 6], powV);
                }
                else
                {
                    DNDS_assert(false);
                }
            }
        }
        return ret;
    }

    template <int dim, int rank, class VLe, class Trans>
    void TransSymDiffOrderTensorV(VLe &&Le, Trans &&trans)
    {
        if constexpr (dim == 3)
        {
            if constexpr (rank == 0)
            {
            }
            else if constexpr (rank == 1)
            {
                Le = trans * Le;
            }
            else if constexpr (rank == 2)
            {
                /*
                00
                11
                22
                01
                12
                02*/
                Eigen::Matrix3d symTensorR2{
                    {Le(0), Le(3), Le(5)},
                    {Le(3), Le(1), Le(4)},
                    {Le(5), Le(4), Le(2)}};
                symTensorR2 = trans * symTensorR2 * trans.transpose();
                Le(0) = symTensorR2(0, 0), Le(1) = symTensorR2(1, 1), Le(2) = symTensorR2(2, 2);
                Le(3) = symTensorR2(0, 1), Le(4) = symTensorR2(1, 2), Le(5) = symTensorR2(0, 2);
            }
            else if constexpr (rank == 3)
            {

                /*
                000
                111
                222
                001
                011
                112
                122
                022
                002
                012*/
                DNDS::ETensor::ETensorR3<real, 3, 3, 3> symTensorR3;
                symTensorR3(0, 0, 0) = Le(0);
                symTensorR3(1, 1, 1) = Le(1);
                symTensorR3(2, 2, 2) = Le(2);

                symTensorR3(0, 0, 1) = symTensorR3(0, 1, 0) = symTensorR3(1, 0, 0) = Le(3);
                symTensorR3(0, 1, 1) = symTensorR3(1, 1, 0) = symTensorR3(1, 0, 1) = Le(4);
                symTensorR3(1, 1, 2) = symTensorR3(1, 2, 1) = symTensorR3(2, 1, 1) = Le(5);
                symTensorR3(1, 2, 2) = symTensorR3(2, 2, 1) = symTensorR3(2, 1, 2) = Le(6);
                symTensorR3(0, 2, 2) = symTensorR3(2, 2, 0) = symTensorR3(2, 0, 2) = Le(7);
                symTensorR3(0, 0, 2) = symTensorR3(0, 2, 0) = symTensorR3(2, 0, 0) = Le(8);

                symTensorR3(0, 1, 2) = symTensorR3(1, 2, 0) = symTensorR3(2, 0, 1) =
                    symTensorR3(0, 2, 1) = symTensorR3(2, 1, 0) = symTensorR3(1, 0, 2) = Le(9);

                symTensorR3.MatTransform0(trans.transpose());
                symTensorR3.MatTransform1(trans.transpose());
                symTensorR3.MatTransform2(trans.transpose());
                Le(0) = symTensorR3(0, 0, 0);
                Le(1) = symTensorR3(1, 1, 1);
                Le(2) = symTensorR3(2, 2, 2);
                Le(3) = symTensorR3(0, 0, 1);
                Le(4) = symTensorR3(0, 1, 1);
                Le(5) = symTensorR3(1, 1, 2);
                Le(6) = symTensorR3(1, 2, 2);
                Le(7) = symTensorR3(0, 2, 2);
                Le(8) = symTensorR3(0, 0, 2);
                Le(9) = symTensorR3(0, 1, 2);
            }
            else
            {
                DNDS_assert(false);
            }
        }
        else // 2-d tensor
        {
            if constexpr (rank == 0)
            {
            }
            else if constexpr (rank == 1)
            {
                Le = trans * Le;
            }
            else if constexpr (rank == 2)
            {
                Eigen::Matrix2d symTensorR2{{Le(0), Le(1)},
                                            {Le(1), Le(2)}};
                symTensorR2 = trans * symTensorR2 * trans.transpose();
                Le(0) = symTensorR2(0, 0), Le(1) = symTensorR2(0, 1), Le(2) = symTensorR2(1, 1);
            }
            else if constexpr (rank == 3)
            {

                DNDS::ETensor::ETensorR3<real, 2, 2, 2> symTensorR3;
                symTensorR3(0, 0, 0) = Le(0);
                symTensorR3(0, 0, 1) = symTensorR3(0, 1, 0) = symTensorR3(1, 0, 0) = Le(1);
                symTensorR3(0, 1, 1) = symTensorR3(1, 0, 1) = symTensorR3(1, 1, 0) = Le(2);
                symTensorR3(1, 1, 1) = Le(3);
                symTensorR3.MatTransform0(trans.transpose());
                symTensorR3.MatTransform1(trans.transpose());
                symTensorR3.MatTransform2(trans.transpose());
                Le(0) = symTensorR3(0, 0, 0);
                Le(1) = symTensorR3(0, 0, 1);
                Le(2) = symTensorR3(0, 1, 1);
                Le(3) = symTensorR3(1, 1, 1);
            }
            else
            {
                DNDS_assert(false);
            }
        }
    }

    template <int dim, class TMat>
    inline void ConvertDiffsLinMap(TMat &&mat, const Geom::tGPoint &dXijdXi)
    {
        int rows = mat.rows();
        if constexpr (dim == 2)
            switch (rows)
            {
            case 10:
                for (int iB = 0; iB < mat.cols(); iB++)
                {
                    TransSymDiffOrderTensorV<2, 3>(
                        mat(Eigen::seq(Eigen::fix<6>, Eigen::fix<9>), iB),
                        dXijdXi({0, 1}, {0, 1}));
                }
            case 6:
                for (int iB = 0; iB < mat.cols(); iB++)
                    TransSymDiffOrderTensorV<2, 2>(
                        mat(Eigen::seq(Eigen::fix<3>, Eigen::fix<5>), iB),
                        dXijdXi({0, 1}, {0, 1}));

            case 3:
                mat({1, 2}, Eigen::all) = dXijdXi({0, 1}, {0, 1}) * mat({1, 2}, Eigen::all);
            case 1:
                break;

            default:
                std::cerr << mat.rows() << std::endl;
                DNDS_assert(false);
                break;
            }
        else // dim ==3
            switch (rows)
            {
            case 20:
                for (int iB = 0; iB < mat.cols(); iB++)
                {
                    TransSymDiffOrderTensorV<3, 3>(
                        mat(Eigen::seq(Eigen::fix<10>, Eigen::fix<19>), iB),
                        dXijdXi);
                }
            case 10:
                for (int iB = 0; iB < mat.cols(); iB++)
                    TransSymDiffOrderTensorV<3, 2>(
                        mat(Eigen::seq(Eigen::fix<4>, Eigen::fix<9>), iB),
                        dXijdXi);

            case 4:
                mat({1, 2, 3}, Eigen::all) = dXijdXi * mat({1, 2, 3}, Eigen::all);
            case 1:
                break;

            default:
                std::cerr << mat.rows() << std::endl;
                DNDS_assert(false);
                break;
            }
    }

    template <int dim, class TMat, class TDiBjB>
    inline void ConvertDiffsFullMap(TMat &&mat, TDiBjB &&DxiDx)
    {
        int rows = mat.rows();
        int nBase = mat.cols();

        int cmaxDiffOrder = ndiff2order<dim>(rows);
        if (cmaxDiffOrder < 0)
        {
            std::cerr << mat.rows() << std::endl;
            DNDS_assert(false);
        }

        if (cmaxDiffOrder == 0)
            return;
        static const auto &diffOperatorIJK2IcDim = dim == 2 ? diffOperatorIJK2I2D : diffOperatorIJK2I;
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> out;
        out.resize(rows, nBase);

        static const int nDiffSizC1 = dim == 2 ? ndiffSizC2D[1] : ndiffSizC[1];
        static const auto seq0t1 = Eigen::seq(Eigen::fix<1>, Eigen::fix<nDiffSizC1 - 1>);
        for (int iB = 0; iB < nBase; iB++)
            for (int ii = 1; ii < nDiffSizC1; ii++)
            {
                int i = dim == 2 ? diffOperatorDimList2D[ii][0] : diffOperatorDimList[ii][0];
                real &ccv = out(ii, iB);
                ccv = 0;
                for (int m = 0; m < dim; m++)
                    ccv += DxiDx(std::get<1>(diffOperatorIJK2IcDim)[i], m) *
                           mat(std::get<1>(diffOperatorIJK2IcDim)[m], iB);
            }
        mat(seq0t1, Eigen::all) = out(seq0t1, Eigen::all);
        if (cmaxDiffOrder == 1)
            return;

        static const int nDiffSizC2 = dim == 2 ? ndiffSizC2D[2] : ndiffSizC[2];
        static const auto seq1t2 = Eigen::seq(Eigen::fix<nDiffSizC1>, Eigen::fix<nDiffSizC2 - 1>);
        for (int iB = 0; iB < nBase; iB++)
            for (int ii = nDiffSizC1; ii < nDiffSizC2; ii++)
            {
                int i = dim == 2 ? diffOperatorDimList2D[ii][0] : diffOperatorDimList[ii][0];
                int j = dim == 2 ? diffOperatorDimList2D[ii][1] : diffOperatorDimList[ii][1];
                real &ccv = out(ii, iB);
                ccv = 0;
                for (int m = 0; m < dim; m++)
                    ccv += DxiDx(std::get<2>(diffOperatorIJK2IcDim)[i][j], m) *
                           mat(std::get<1>(diffOperatorIJK2IcDim)[m], iB);
                for (int m = 0; m < dim; m++)
                    for (int n = 0; n < dim; n++)
                        ccv += DxiDx(std::get<1>(diffOperatorIJK2IcDim)[i], m) *
                               DxiDx(std::get<1>(diffOperatorIJK2IcDim)[j], n) *
                               mat(std::get<2>(diffOperatorIJK2IcDim)[m][n], iB);
            }
        mat(seq1t2, Eigen::all) = out(seq1t2, Eigen::all);
        if (cmaxDiffOrder == 2)
            return;

        static const int nDiffSizC3 = dim == 2 ? ndiffSizC2D[3] : ndiffSizC[3];
        static const auto seq2t3 = Eigen::seq(Eigen::fix<nDiffSizC2>, Eigen::fix<nDiffSizC3 - 1>);
        for (int iB = 0; iB < nBase; iB++)
            for (int ii = nDiffSizC2; ii < nDiffSizC3; ii++)
            {
                int i = dim == 2 ? diffOperatorDimList2D[ii][0] : diffOperatorDimList[ii][0];
                int j = dim == 2 ? diffOperatorDimList2D[ii][1] : diffOperatorDimList[ii][1];
                int k = dim == 2 ? diffOperatorDimList2D[ii][2] : diffOperatorDimList[ii][2];
                real &ccv = out(ii, iB);
                ccv = 0;
                for (int m = 0; m < dim; m++)
                    ccv += DxiDx(std::get<3>(diffOperatorIJK2IcDim)[i][j][k], m) *
                           mat(std::get<1>(diffOperatorIJK2IcDim)[m], iB);
                for (int m = 0; m < dim; m++)
                    for (int n = 0; n < dim; n++)
                        ccv +=
                            DxiDx(std::get<2>(diffOperatorIJK2IcDim)[i][j], m) *
                                DxiDx(std::get<1>(diffOperatorIJK2IcDim)[k], n) *
                                mat(std::get<2>(diffOperatorIJK2IcDim)[m][n], iB) +
                            DxiDx(std::get<2>(diffOperatorIJK2IcDim)[i][k], m) *
                                DxiDx(std::get<1>(diffOperatorIJK2IcDim)[j], n) *
                                mat(std::get<2>(diffOperatorIJK2IcDim)[m][n], iB) +
                            DxiDx(std::get<2>(diffOperatorIJK2IcDim)[j][k], m) *
                                DxiDx(std::get<1>(diffOperatorIJK2IcDim)[i], n) *
                                mat(std::get<2>(diffOperatorIJK2IcDim)[m][n], iB);
                for (int m = 0; m < dim; m++)
                    for (int n = 0; n < dim; n++)
                        for (int p = 0; p < dim; p++)
                            ccv +=
                                DxiDx(std::get<1>(diffOperatorIJK2IcDim)[i], m) *
                                DxiDx(std::get<1>(diffOperatorIJK2IcDim)[j], n) *
                                DxiDx(std::get<1>(diffOperatorIJK2IcDim)[k], p) *
                                mat(std::get<3>(diffOperatorIJK2IcDim)[m][n][p], iB);
            }
        mat(seq2t3, Eigen::all) = out(seq2t3, Eigen::all);
        if (cmaxDiffOrder == 3)
            return;
    }

    class CFVPeriodicity : public Geom::Periodicity
    {
    public:
        using tBase = Geom::Periodicity;
        using tBase::tBase;

        CFVPeriodicity(const tBase &vBase) : tBase(vBase) {} // copy from tBase

        template <int dim, class TU>
        void TransDiValueInplace(TU &u, Geom::t_index id)
        {
            using namespace Geom;
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            ConvertDiffsLinMap<dim>(u, rotation[i]);
        }

        template <int dim, class TU>
        void TransDiValueBackInplace(TU &u, Geom::t_index id)
        {
            using namespace Geom;
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            ConvertDiffsLinMap<dim>(u, rotation[i].transpose());
        }
    };

    /**
     * \brief
     * invert compact form of all diffs of shape functions
     * \arg DxDxi NDiff x 3 shape where NDiff is number of derivatives determined by `maxDiff` which is max derivative order
     * \arg DxiDx NDiff x 3 shape
     */
    template <int dim, int maxDiff, class TDiBjA, class TDiBjB>
    void DxDxi2DxiDx(TDiBjA &&DxDxi, TDiBjB &&DxiDx)
    {
        static const int nDiffSizCC = dim == 2 ? ndiffSizC2D[maxDiff] : ndiffSizC[maxDiff];
        if (maxDiff <= 0) // diff 0 not altered, DxDxi(0, :) are x values and DxiDx(0, :) are xi values (supposedly)
            return;
        static const int nDiffSizC1 = dim == 2 ? ndiffSizC2D[1] : ndiffSizC[1];
        Eigen::Matrix<real, 3, 3> Dx_j_Dxi_i;
        if constexpr (dim == 2)
        {
            Dx_j_Dxi_i.setZero();
            Dx_j_Dxi_i(2, 2) = 1;
            Dx_j_Dxi_i({0, 1}, {0, 1}) = DxDxi({1, 2}, {0, 1});
        }
        else
            Dx_j_Dxi_i = DxDxi({1, 2, 3}, {0, 1, 2});
        auto lu_cur = Dx_j_Dxi_i.fullPivLu();
        Eigen::Matrix<real, 3, 3> Dxi_j_Dx_i = lu_cur.inverse(); // Dxi_j_Dx_i Dx_k_Dxi_j = delta_ik

        if constexpr (dim == 2)
            DxiDx({1, 2}, {0, 1, 2}) = Dxi_j_Dx_i({0, 1}, {0, 1, 2});
        else
            DxiDx({1, 2, 3}, {0, 1, 2}) = Dxi_j_Dx_i;

        if (maxDiff == 1)
            return;
        static const auto &diffOperatorIJK2IcDim = dim == 2 ? diffOperatorIJK2I2D : diffOperatorIJK2I;

        static const int nDiffSizC2 = dim == 2 ? ndiffSizC2D[2] : ndiffSizC[2];
        static const auto seq1t2 = Eigen::seq(Eigen::fix<nDiffSizC1>, Eigen::fix<nDiffSizC2 - 1>); // 2D: 3 4 5
        Eigen::Matrix<real, nDiffSizCC, 3> M_RHS;
        M_RHS(seq1t2, Eigen::all).setZero();
        for (int j = 0; j < dim; j++)
            // for (int i = 0; i < dim; i++)
            //     for (int k = 0; k < dim; k++)
            for (int ii = nDiffSizC1; ii < nDiffSizC2; ii++)
            {
                int i = dim == 2 ? diffOperatorDimList2D[ii][0] : diffOperatorDimList[ii][0];
                int k = dim == 2 ? diffOperatorDimList2D[ii][1] : diffOperatorDimList[ii][1];
                real &ccv = M_RHS(ii, j);
                ccv = 0;
                for (int m = 0; m < dim; m++)
                    for (int n = 0; n < dim; n++)
                        ccv -= DxDxi(std::get<2>(diffOperatorIJK2IcDim)[m][n], j) *
                               DxiDx(std::get<1>(diffOperatorIJK2IcDim)[i], m) *
                               DxiDx(std::get<1>(diffOperatorIJK2IcDim)[k], n);
            }
        DxiDx(seq1t2, Eigen::all) = M_RHS(seq1t2, Eigen::all) /* ikj */ * Dxi_j_Dx_i;

        if (maxDiff == 2)
            return;
        static const int nDiffSizC3 = dim == 2 ? ndiffSizC2D[3] : ndiffSizC[3];
        static const auto seq2t3 = Eigen::seq(Eigen::fix<nDiffSizC2>, Eigen::fix<nDiffSizC3 - 1>); // 2D: 6 7 8 9
        M_RHS(seq2t3, Eigen::all).setZero();
        for (int j = 0; j < dim; j++)
            // for (int i = 0; i < dim; i++)
            //     for (int k = 0; k < dim; k++)
            //         for (int l = 0; l < dim; l++)
            for (int ii = nDiffSizC2; ii < nDiffSizC3; ii++)
            {
                int i = dim == 2 ? diffOperatorDimList2D[ii][0] : diffOperatorDimList[ii][0];
                int k = dim == 2 ? diffOperatorDimList2D[ii][1] : diffOperatorDimList[ii][1];
                int l = dim == 2 ? diffOperatorDimList2D[ii][2] : diffOperatorDimList[ii][2];
                real &ccv = M_RHS(ii, j);
                ccv = 0;
                for (int m = 0; m < dim; m++)
                    for (int n = 0; n < dim; n++)
                        ccv -= DxDxi(std::get<2>(diffOperatorIJK2IcDim)[m][n], j) *
                                   DxiDx(std::get<2>(diffOperatorIJK2IcDim)[i][k], m) *
                                   DxiDx(std::get<1>(diffOperatorIJK2IcDim)[l], n) +
                               DxDxi(std::get<2>(diffOperatorIJK2IcDim)[m][n], j) *
                                   DxiDx(std::get<2>(diffOperatorIJK2IcDim)[k][l], m) *
                                   DxiDx(std::get<1>(diffOperatorIJK2IcDim)[i], n) +
                               DxDxi(std::get<2>(diffOperatorIJK2IcDim)[m][n], j) *
                                   DxiDx(std::get<2>(diffOperatorIJK2IcDim)[i][l], m) *
                                   DxiDx(std::get<1>(diffOperatorIJK2IcDim)[k], n);
                for (int m = 0; m < dim; m++)
                    for (int n = 0; n < dim; n++)
                        for (int p = 0; p < dim; p++)
                            ccv -= DxDxi(std::get<3>(diffOperatorIJK2IcDim)[m][n][p], j) *
                                   DxiDx(std::get<1>(diffOperatorIJK2IcDim)[i], m) *
                                   DxiDx(std::get<1>(diffOperatorIJK2IcDim)[k], n) *
                                   DxiDx(std::get<1>(diffOperatorIJK2IcDim)[l], p);
            }
        DxiDx(seq2t3, Eigen::all) = M_RHS(seq2t3, Eigen::all) /* ikj */ * Dxi_j_Dx_i;
        if (maxDiff == 3)
            return;
        DNDS_assert_info(false, "maxDiff is too large!");
    }
}