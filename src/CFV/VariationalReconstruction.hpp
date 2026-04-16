#pragma once

#include <unordered_map>
#include <set>
#include <utility>

#include "VRDefines.hpp"
#include "VRSettings.hpp"
#include "Geom/BaseFunction.hpp"
#include "Geom/DiffTensors.hpp"
#include "FiniteVolume.hpp"

namespace DNDS::CFV
{
    /**
     * @brief Accumulates the weighted diff-order tensor inner product into Conj
     * across all polynomial orders present in the given diff vectors.
     *
     * Replaces the 4 copies of the switch(cnDiffs) fallthrough pattern
     * in FFaceFunctional. Iterates over all (i,j) pairs and accumulates
     * NormSymDiffOrderTensorV contributions for each polynomial order.
     *
     * @tparam dim       Spatial dimension (2 or 3)
     * @tparam powV      Combinatorial power (1 unless USE_ECCENTRIC_COMB_POW_2)
     * @param DiffI      Diff base values for left side (cnDiffs x nColsI)
     * @param DiffJ      Diff base values for right side (cnDiffs x nColsJ)
     * @param Conj       Output conjugation matrix (nColsI x nColsJ), accumulated into
     * @param wgd        Squared face weight vector (one entry per order)
     * @param cnDiffs    Number of diff rows (= total NDOF for the max order present)
     * @param faceL      Length scale; pass 0 for anisotropic path (all faceLPow = 1)
     */
    template <int dim, int powV = 1, class TDiffI, class TDiffJ>
    inline void AccumulateDiffOrderContributions(
        const Eigen::MatrixBase<TDiffI> &DiffI,
        const Eigen::MatrixBase<TDiffJ> &DiffJ,
        MatrixXR &Conj,
        const Eigen::Vector<real, Eigen::Dynamic> &wgd,
        int cnDiffs,
        real faceL)
    {
        using namespace Geom::Base;

        // Each order's start index and count are compile-time constants,
        // so segment<N>() produces fixed-size Eigen expressions (no dynamic alloc).
        auto accumOrder = [&](auto order_tag, auto start_tag, auto count_tag, int wIdx)
        {
            constexpr int order = decltype(order_tag)::value;
            constexpr int start = decltype(start_tag)::value;
            constexpr int count = decltype(count_tag)::value;
            // faceL^(order*2) via integer multiplication; faceL==0 means anisotropic (scale=1)
            real faceLPow;
            if constexpr (order == 0)
                faceLPow = 1.0;
            else if constexpr (order == 1)
                faceLPow = faceL * faceL;
            else if constexpr (order == 2)
            {
                real fL2 = faceL * faceL;
                faceLPow = fL2 * fL2;
            }
            else // order == 3
            {
                real fL2 = faceL * faceL;
                faceLPow = fL2 * fL2 * fL2;
            }
            real scale = wgd(wIdx) * (faceL == 0 ? 1.0 : faceLPow);
            for (int i = 0; i < DiffI.cols(); i++)
                for (int j = 0; j < DiffJ.cols(); j++)
                    Conj(i, j) += NormSymDiffOrderTensorV<dim, order, powV>(
                                      DiffI.col(i).template segment<count>(start),
                                      DiffJ.col(j).template segment<count>(start)) *
                                  scale;
        };

        if constexpr (dim == 2)
        {
            // dim=2 NDOF per order: 1, 2, 3, 4  cumulative: 1, 3, 6, 10
            DNDS_assert(cnDiffs == 10 || cnDiffs == 6 || cnDiffs == 3 || cnDiffs == 1);
            if (cnDiffs >= 10) accumOrder(std::integral_constant<int, 3>{}, std::integral_constant<int, 6>{}, std::integral_constant<int, 4>{}, 3);
            if (cnDiffs >= 6)  accumOrder(std::integral_constant<int, 2>{}, std::integral_constant<int, 3>{}, std::integral_constant<int, 3>{}, 2);
            if (cnDiffs >= 3)  accumOrder(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{}, 1);
            accumOrder(std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{}, std::integral_constant<int, 1>{}, 0);
        }
        else
        {
            // dim=3 NDOF per order: 1, 3, 6, 10  cumulative: 1, 4, 10, 20
            DNDS_assert(cnDiffs == 20 || cnDiffs == 10 || cnDiffs == 4 || cnDiffs == 1);
            if (cnDiffs >= 20) accumOrder(std::integral_constant<int, 3>{}, std::integral_constant<int, 10>{}, std::integral_constant<int, 10>{}, 3);
            if (cnDiffs >= 10) accumOrder(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{},  std::integral_constant<int, 6>{},  2);
            if (cnDiffs >= 4)  accumOrder(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{},  std::integral_constant<int, 3>{},  1);
            accumOrder(std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{}, std::integral_constant<int, 1>{}, 0);
        }
    }

    /**
     * @brief
     * The VR class that provides any information needed in high-order CFV
     *
     * @details
     * VR holds a primitive mesh and any needed derived information about geometry and
     * general reconstruction coefficients
     *
     * The (differentiable) input of VR is merely: geometry (mesh),
     * and the weight inputs if using distributed weight;
     * output are all values derived using construct* method
     *
     */
    template <int dim = 2>
    class VariationalReconstruction : public FiniteVolume
    {
    public:
        int getDim() const { return dim; }
        using t_base = FiniteVolume;

    private:
        VRSettings settings = VRSettings{dim};

    public:
        [[nodiscard]] const VRSettings &getSettings() const
        {
            return settings;
        }

        void parseSettings(VRSettings::json &j)
        {
            settings.ParseFromJson(j);
            // Slice the FiniteVolumeSettings base portion into the base class copy.
            // FiniteVolume methods read maxOrder, intOrder, ignoreMeshGeometryDeficiency,
            // and nIterCellSmoothScale from its own `settings` member. This keeps
            // them in sync. All VR-specific fields remain only in `this->settings`.
            t_base::settings = settings;
        }

    public:
        using TFTrans = std::function<void(Eigen::Ref<MatrixXR>, Geom::t_index)>;

    private:
        /**
         * @brief Holds the base function moments, face weights, diff-base caches,
         * and boundary VR point caches produced by ConstructBaseAndWeight().
         *
         * These members are built once and then read by FDiffBaseValue(),
         * GetIntPointDiffBaseValue(), FFaceFunctional(), and indirectly by
         * reconstruction and limiter methods.
         */
        struct VRBaseWeight
        {
            t3VecPair faceAlignedScales;     /// @brief face-aligned length scales
            t3MatPair faceMajorCoordScale;   /// @brief face major coordinate transform
            tVVecPair cellBaseMoment;        /// @brief cell base function moments
            tVVecPair faceWeight;            /// @brief face quadrature weights
            tMatsPair cellDiffBaseCache;     /// @brief cached diff-base values at cell quad points
            tMatsPair faceDiffBaseCache;     /// @brief cached diff-base values at face quad points
            tVMatPair cellDiffBaseCacheCent; /// @brief cached diff-base values at cell centroids
            tVMatPair faceDiffBaseCacheCent; /// @brief cached diff-base values at face centroids

            struct BndVRPointCache
            {
                Geom::tPoint norm;
                Geom::tPoint PPhy;
                real JDet;
                RowVectorXR D0Bj;
            };
            std::unordered_map<index, std::vector<BndVRPointCache>> bndVRCaches; /// @brief boundary face VR caches
        };

        VRBaseWeight baseWeight_;
        using BndVRPointCache = typename VRBaseWeight::BndVRPointCache;

        /**
         * @brief Holds the reconstruction coefficient matrices produced by ConstructRecCoeff().
         *
         * Grouping these members clarifies the data-ownership boundary: VRBaseWeight
         * members (above) are built by ConstructBaseAndWeight(), while these
         * coefficient members are built by ConstructRecCoeff() from the base/weight data.
         */
        struct VRCoefficients
        {
            tMatsPair matrixAB;        /// @brief A and B blocks per face, used during ConstructRecCoeff
            tVecsPair vectorB;         /// @brief B vector per cell
            bool needOriginalMatrix = false;
            tMatsPair matrixAAInvB;    /// @brief A^{-1}B blocks per cell (main reconstruction matrix)
            tMatsPair vectorAInvB;     /// @brief A^{-1}b vectors per cell
            tVMatPair matrixSecondary; /// @brief secondary-rec matrices on each face
            tVMatPair matrixAHalf_GG;  /// @brief Green-Gauss half matrices

            tVMatPair matrixA; /// @brief full A matrix per cell (when needOriginalMatrix)

            std::vector<Eigen::MatrixXd> volIntCholeskyL;
            bool needVolIntCholeskyL = false;
            std::vector<Eigen::MatrixXd> matrixACholeskyL;
            bool needMatrixACholeskyL = false;

            auto GetCellRecMatAInv(index iCell)
            {
                return matrixAAInvB(iCell, 0);
            }

            auto GetCellRecMatAInvB(index iCell, int ic2f)
            {
                return matrixAAInvB(iCell, 1 + ic2f);
            }

            auto &get_matrixAAInvB() { return matrixAAInvB; }
            auto &get_vectorAInvB() { return vectorAInvB; }
        };

        VRCoefficients coefficients_;

        // for periodic transformation inside scalars, eg. velocity components
        TFTrans FTransPeriodic, FTransPeriodicBack;

    public:
        VariationalReconstruction(MPIInfo nMpi, ssp<Geom::UnstructuredMesh> nMesh)
            : FiniteVolume(nMpi, nMesh)
        {
            DNDS_assert(dim == mesh->dim);
        }

        void SetPeriodicTransformations(const TFTrans &nFTransPeriodic, const TFTrans &nFTransPeriodicBack)
        {
            FTransPeriodic = nFTransPeriodic;
            FTransPeriodicBack = nFTransPeriodicBack;
        }

        // directly SetPeriodicTransformations with mesh knowing that there is only
        // a 2D/3D vector in it
        template <size_t pDim>
        void SetPeriodicTransformations(std::array<int, pDim> Seq123)
        {
            SetPeriodicTransformations(
                [mesh = mesh, Seq123](auto u, Geom::t_index id)
                {
                    u(EigenAll, Seq123) = mesh->periodicInfo.TransVector<pDim, Eigen::Dynamic>(u(EigenAll, Seq123).transpose(), id).transpose();
                },
                [mesh = mesh, Seq123](auto u, Geom::t_index id)
                {
                    u(EigenAll, Seq123) = mesh->periodicInfo.TransVectorBack<pDim, Eigen::Dynamic>(u(EigenAll, Seq123).transpose(), id).transpose();
                });
        }

        void SetPeriodicTransformations()
        {
            SetPeriodicTransformations(
                [](auto u, Geom::t_index id) {},
                [](auto u, Geom::t_index id) {});
        }

        /**
         * @brief Applies the appropriate periodic transformation to one or more data matrices.
         *
         * Determines from `if2c` and `faceID` whether the current cell is the donor
         * or main side of the periodic pair, and calls FTransPeriodic or
         * FTransPeriodicBack accordingly. No-op when the mesh is not periodic.
         *
         * @param if2c   Face-to-cell local index (0 = back, 1 = front)
         * @param faceID Boundary zone ID of the face
         * @param data   One or more Eigen matrix references to transform
         */
        template <typename... Ts>
        void ApplyPeriodicTransform(int if2c, Geom::t_index faceID, Ts &...data) const
        {
            if (!mesh->isPeriodic)
                return;
            DNDS_assert(FTransPeriodic && FTransPeriodicBack);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                (FTransPeriodic(data, faceID), ...);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                (FTransPeriodicBack(data, faceID), ...);
        }

        void ConstructMetrics();

        using tFGetBoundaryWeight = std::function<real(Geom::t_index, int)>;

        void ConstructBaseAndWeight(const tFGetBoundaryWeight &id2faceDircWeight = [](Geom::t_index id, int iOrder)
                                    { return 1.0; });

        void ConstructRecCoeff();

        auto GetCellRecMatAInv(index iCell)
        {
            return coefficients_.GetCellRecMatAInv(iCell);
        }

        auto GetCellRecMatAInvB(index iCell, int ic2f)
        {
            return coefficients_.GetCellRecMatAInvB(iCell, ic2f);
        }

        auto &get_matrixAAInvB()
        {
            return coefficients_.get_matrixAAInvB();
        }

        auto &get_vectorAInvB()
        {
            return coefficients_.get_vectorAInvB();
        }

        /**
         * @brief flag = 0 means use moment data, or else use no moment (as 0)
         * pPhy must be relative to cell
         * if iFace < 0, means anywhere
         * if iFace > 0, iG == -1, means center; iG < -1, then anywhere
         */
        template <class TOut>
        void FDiffBaseValue(TOut &DiBj,
                            const Geom::tPoint &pPhy, // conventional input above
                            index iCell, index iFace, int iG, int flag = 0)
        {
            using namespace Geom;
            using namespace Geom::Base;

            auto pCen = cellCent[iCell];
            tPoint pPhysicsC = pPhy - pCen;

            if (!settings.baseSettings.localOrientation)
            {
                tPoint simpleScale = cellAlignedHBox[iCell];
                if (!settings.baseSettings.anisotropicLengths)
                {
                    if constexpr (dim == 2)
                        simpleScale({0, 1}).setConstant(simpleScale({0, 1}).array().maxCoeff());
                    else
                        simpleScale.setConstant(simpleScale.array().maxCoeff());
                }
                tPoint pPhysicsCScaled = pPhysicsC.array() / simpleScale.array();
                if constexpr (dim == 2)
                    FPolynomialFill2D(DiBj, pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2), simpleScale(0), simpleScale(1), simpleScale(2), (int)DiBj.rows(), (int)DiBj.cols());
                else
                    FPolynomialFill3D(DiBj, pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2), simpleScale(0), simpleScale(1), simpleScale(2), (int)DiBj.rows(), (int)DiBj.cols());
            }
            else
            {
                tPoint simpleScale = cellMajorHBox[iCell];
                if (!settings.baseSettings.anisotropicLengths)
                {
                    if constexpr (dim == 2)
                        simpleScale({0, 1}).setConstant(simpleScale({0, 1}).array().maxCoeff());
                    else
                        simpleScale.setConstant(simpleScale.array().maxCoeff());
                }
                tPoint pPhysicsCMajor = cellMajorCoord[iCell].transpose() * pPhysicsC;
                tPoint pPhysicsCScaled = pPhysicsCMajor.array() / simpleScale.array();
                if constexpr (dim == 2)
                    FPolynomialFill2D(DiBj, pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2), simpleScale(0), simpleScale(1), simpleScale(2), DiBj.rows(), DiBj.cols());
                else
                    FPolynomialFill3D(DiBj, pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2), simpleScale(0), simpleScale(1), simpleScale(2), DiBj.rows(), DiBj.cols());
                tGPoint dXijdxi = cellMajorCoord[iCell];
                ConvertDiffsLinMap<dim>(DiBj, dXijdxi);
            }

            if (flag == 0)
            {
                auto baseMoment = baseWeight_.cellBaseMoment[iCell];
                DiBj(0, EigenAll) -= baseMoment.transpose();
            }
        }

        /**
         * @brief if if2c < 0, then calculated, if maxDiff == 255, then seen as all diffs
         * if iFace < 0, then seen as cell int points; if iG < 1, then seen as center
         * @todo : divide GetIntPointDiffBaseValue into different calls
         * @warning maxDiff is max(diffList) + 1 not len(difflist)
         * @todo:  //TODO add support for rotational periodic boundary!
         */
        template <class TList>
        MatrixXR
        GetIntPointDiffBaseValue(
            index iCell, index iFace, rowsize if2c, int iG,
            TList &&diffList = EigenAll,
            uint8_t maxDiff = UINT8_MAX)
        {
            if (iFace >= 0)
            {
                maxDiff = std::min(maxDiff, GetFaceAtr(iFace).NDIFF);
                if (if2c < 0)
                    if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                if (settings.cacheDiffBase && maxDiff <= settings.cacheDiffBaseSize)
                {
                    // auto gFace = this->GetFaceQuad(iFace);

                    if (iG >= 0)
                    {
                        return baseWeight_.faceDiffBaseCache(iFace, iG + (baseWeight_.faceDiffBaseCache.RowSize(iFace) / 2) * if2c)(
                            std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, EigenLast));
                    }
                    else
                    {
                        int maxNDOF = int(baseWeight_.faceDiffBaseCacheCent[iFace].cols()) / 2;
                        return baseWeight_.faceDiffBaseCacheCent[iFace](
                            std::forward<TList>(diffList),
                            Eigen::seq(if2c * maxNDOF + 1,
                                       if2c * maxNDOF + maxNDOF - 1));
                    }
                }
                else
                {
                    // Actual computing:
                    ///@todo //!!!!TODO: take care of periodic case
                    MatrixXR dbv;
                    dbv.resize(maxDiff, GetCellAtr(iCell).NDOF);
                    FDiffBaseValue(dbv, GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, iG), iCell, iFace, iG, 0);
                    return dbv(std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, EigenLast));
                }
            }
            else
            {
                maxDiff = std::min(maxDiff, GetCellAtr(iCell).NDIFF);
                if (settings.cacheDiffBase && maxDiff <= settings.cacheDiffBaseSize)
                {
                    if (iG >= 0)
                    {
                        return baseWeight_.cellDiffBaseCache(iCell, iG)(
                            std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, EigenLast));
                    }
                    else
                    {
                        return baseWeight_.cellDiffBaseCacheCent[iCell](
                            std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, EigenLast));
                    }
                }
                else
                {
                    MatrixXR dbv;
                    dbv.resize(maxDiff, GetCellAtr(iCell).NDOF);
                    FDiffBaseValue(dbv, GetCellQuadraturePPhys(iCell, iG), iCell, -1, iG, 0);
                    return dbv(std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, EigenLast));
                }
            }
        }

        /**
         * @brief if if2c < 0, then calculated with iCell hen seen as all diffs
         */
        auto
        GetMatrixSecondary(index iCell, index iFace, index if2c)
        {
            if (if2c < 0)
                if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            int maxNDOFM1 = coefficients_.matrixSecondary[iFace].cols() / 2;
            return coefficients_.matrixSecondary[iFace](
                EigenAll, Eigen::seq(
                              if2c * maxNDOFM1 + 0,
                              if2c * maxNDOFM1 + maxNDOFM1 - 1));
        }

        template <class TDiffIDerived, class TDiffJDerived>
        auto FFaceFunctional(
            const Eigen::MatrixBase<TDiffIDerived> &DiffI, const Eigen::MatrixBase<TDiffJDerived> &DiffJ,
            index iFace, index iCellL, index iCellR)
        {
            using namespace Geom;
            using namespace Geom::Base;
            Eigen::Vector<real, Eigen::Dynamic> wgd = baseWeight_.faceWeight[iFace].array().square();
            DNDS_assert(DiffI.rows() == DiffJ.rows());
            int cnDiffs = DiffI.rows();

            MatrixXR Conj;
            Conj.resize(DiffI.cols(), DiffJ.cols());
            Conj.setZero();

            //* PJH - rotation invariant scheme
            tPoint faceLV;
            switch (settings.functionalSettings.scaleType)
            {
            case VRSettings::FunctionalSettings::ScaleType::BaryDiff:
                faceLV = baseWeight_.faceAlignedScales[iFace];
                break;
            case VRSettings::FunctionalSettings::ScaleType::MeanAACBB:
            case VRSettings::FunctionalSettings::ScaleType::CellMax:
                faceLV = baseWeight_.faceAlignedScales[iFace];
                break;
            default:
                DNDS_assert(false);
            }

            // real faceL = (faceLV.array().maxCoeff());
            // * warning component_3 of the scale vector in 2D is forced to 1! not 0!
            real faceL = 0;
            if (settings.functionalSettings.scaleType == VRSettings::FunctionalSettings::ScaleType::MeanAACBB)
                faceL = std::sqrt(faceLV(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>)).array().square().mean());
            if (settings.functionalSettings.scaleType == VRSettings::FunctionalSettings::ScaleType::BaryDiff ||
                settings.functionalSettings.scaleType == VRSettings::FunctionalSettings::ScaleType::CellMax)
                faceL = faceLV(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>)).norm();
            real faceLOrig = faceL;
            if (settings.functionalSettings.scaleType == VRSettings::FunctionalSettings::ScaleType::CellMax)
            {
                faceL = this->GetCellMaxLenScale(mesh->face2cell(iFace, 0)) * 0.5;
                if (mesh->face2cell(iFace, 1) != UnInitIndex)
                    faceL = std::max(faceL, this->GetCellMaxLenScale(mesh->face2cell(iFace, 1)));
            }
            // CellMax option is the same as BaryDiff in terms of faceLOrig (being the Bary Dist)
            // CellMax only uses another faceL

            faceL *= settings.functionalSettings.scaleMultiplier;

#ifdef USE_ECCENTRIC_COMB_POW_2
            static constexpr int powV = 2;
#else
            static constexpr int powV = 1;
#endif

            if (!settings.functionalSettings.useAnisotropicFunctional)
            {
                AccumulateDiffOrderContributions<dim, powV>(DiffI, DiffJ, Conj, wgd, cnDiffs, faceL);
            }
            else
            {
                using TMatCopy = Eigen::Matrix<
                    real,
                    Eigen::MatrixBase<TDiffIDerived>::RowsAtCompileTime,
                    Eigen::MatrixBase<TDiffIDerived>::ColsAtCompileTime>;
                TMatCopy DiffI_Norm = DiffI;
                TMatCopy DiffJ_Norm = DiffJ;
                tGPoint coordTrans = baseWeight_.faceMajorCoordScale[iFace].transpose() *
                                     settings.functionalSettings.scaleMultiplier;
                if constexpr (dim == 2)
                    coordTrans(2, 2) = 1;
                {
                    // tPoint norm = this->GetFaceNorm(iFace, -1);
                    // real normScale = (coordTrans * norm).norm();
                    // coordTrans(0, EigenAll) = norm.transpose() * faceL;
                    // coordTrans({1, 2}, EigenAll).setZero();
                }
                {
                    // coordTrans = Geom::NormBuildLocalBaseV<3>(norm).transpose() * faceL;
                }
                {
                    // coordTrans *=  (2 * std::sqrt(3));
                    // tPoint lengths = coordTrans.rowwise().norm();
                    // // tPoint lengthsNew = lengths.array() * faceL / (faceL + lengths.array());
                    // tPoint lengthsNew = lengths.array().min(faceL);
                    // coordTrans.array().colwise() *= lengthsNew.array() / (lengths.array() + verySmallReal);

                    // // if (lengths(0) > 1e2 * lengths(1) || lengths(1) > 1e2 * lengths(0))
                    // // {
                    // //     std::cout << "FACE " << iFace << std::endl;
                    // //     std::cout << lengths << std::endl;
                    // //     std::cout << faceL << std::endl;
                    // // }
                }
                {
                    // tPoint norm = this->GetFaceNorm(iFace, -1);
                    // auto getCellFaceMajorPNorm = [&](index iCell) -> tPoint
                    // {
                    //     tGPoint cellMajorCoordTrans = this->cellMajorCoord[iCell].transpose().rowwise().normalized();
                    //     tPoint normCos = (cellMajorCoordTrans * norm).array().abs();
                    //     tPoint pNorm;
                    //     if (normCos(0) < normCos(1))
                    //         pNorm = cellMajorCoordTrans(1, EigenAll).transpose();
                    //     else
                    //         pNorm = cellMajorCoordTrans(0, EigenAll).transpose();
                    //     return pNorm;
                    // };
                    // tPoint pNormL = getCellFaceMajorPNorm(iCellL);
                    // tPoint pNormR = getCellFaceMajorPNorm(iCellR);
                    // auto getProjection = [](tPoint n) -> tGPoint
                    // {
                    //     tGPoint U = Geom::NormBuildLocalBaseV<3>(n).transpose();
                    //     return U.transpose() * Eigen::Vector3d{1, 0, 0}.asDiagonal() * U;
                    // };
                    // ConvertDiffsLinMap<dim>(DiffI_Norm, getProjection(pNormL));
                    // ConvertDiffsLinMap<dim>(DiffJ_Norm, getProjection(pNormR));

                    // coordTrans(0, EigenAll) = norm.transpose() * faceL;
                    // coordTrans({1, 2}, EigenAll).setZero();
                }
                if (settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::Norm ||
                    settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::CentDiff ||
                    settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::WallDist)
                {
                    tPoint norm = this->GetFaceNorm(iFace, -1);
                    real areaL = this->GetFaceArea(iFace);
                    if constexpr (dim == 3)
                        areaL = std::sqrt(areaL);
                    real tw = 1. / std::min({1.0, faceLOrig / (areaL + 0.001 * faceLOrig)});
                    real nw = 1;
                    // real tw = 1. / std::max({1.0, areaL / (faceLOrig + 0.001 * areaL)});
                    // real tw = 1.0;
                    if (settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::CentDiff)
                    {
                        if (mesh->face2cell(iFace, 1) != UnInitIndex)
                        {
                            norm = this->GetOtherCellBaryFromCell(mesh->face2cell(iFace, 0),
                                                                  mesh->face2cell(iFace, 1), iFace) -
                                   this->GetCellBary(mesh->face2cell(iFace, 0));
                            norm.normalize();
                        }
                    }
                    else if (
                        settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::WallDist)
                    {
                        DNDS_assert_info(mesh->nodeWallDist.father && mesh->nodeWallDist.father->Size() == mesh->NumNode(), "must build mesh's nodeWallDist before this");
                        real meanAR = GetCellAR(mesh->face2cell(iFace, 0));
                        real wd = 0.0;
                        real h_min_LR = GetCellNodeMinLenScale(mesh->face2cell(iFace, 0));
                        if (mesh->face2cell(iFace, 1) != UnInitIndex)
                        {
                            h_min_LR = std::min(h_min_LR, GetCellNodeMinLenScale(mesh->face2cell(iFace, 1)));
                            meanAR = 0.5 * (meanAR + GetCellAR(mesh->face2cell(iFace, 1)));
                            norm.setZero();
                            real wdDiv = 0.0;
                            for (int if2n = 0; if2n < mesh->face2node[iFace].size(); if2n++)
                            {
                                auto d = mesh->GetCoordWallDistOnFace(iFace, 0);
                                norm += d;
                                wd += d.norm();
                                wdDiv += 1.0;
                            }
                            wd /= wdDiv;
                            norm.stableNormalize();
                        }
                        // else: norm stays face norm, wd is 0.0
                        // original WallDist ani
                        tw = 1.0;
                        nw = 1. / meanAR;
                        // WallDist V1:
                        auto f_dhInterp = [](real d)
                        {
                            real d0 = 1e-6;
                            real d1 = 1;
                            real h0 = 1e-6;
                            real h1 = 1;
                            real a = std::log(h1 / h0) / (d1 - d0);
                            d = std::max(d, d0);
                            real ret = h0 * std::exp(a * (d - d0));
                            // ret = std::min(ret, h1);
                            return ret;
                        };
                        real h_reference = f_dhInterp(wd);
                        h_reference = std::max(h_min_LR, h_reference);
                        // std::cout << wd << ", " << h_reference << std::endl;
                        tw = 1.0;
                        nw = std::min(1.0, h_reference / faceL);
                    }

                    coordTrans = Geom::NormBuildLocalBaseV<3>(norm).transpose() * faceL;
                    coordTrans(0, EigenAll) *= nw;
                    coordTrans({1, 2}, EigenAll) *= tw * settings.functionalSettings.tanWeightScale;
                }
                else if (settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::InertiaCoordBBNorm)
                {
                    // same as norm but using InertiaCoordBB for tan
                    tPoint norm = this->GetFaceNorm(iFace, -1);
                    real areaL = this->GetFaceArea(iFace);
                    if constexpr (dim == 3)
                        areaL = std::sqrt(areaL);
                    // real tw = 1. / std::min({1.0, faceLOrig / (areaL + 0.001 * faceLOrig)});
                    real nw = 1;

                    tGPoint coordTransN = coordTrans.rowwise().normalized();
                    tPoint near_norm = (coordTransN * norm).array().abs();
                    int iMax = -1;
                    real maxCos = near_norm.maxCoeff(&iMax);
                    tGPoint coordTrans_new = coordTrans;

                    coordTrans_new(iMax, EigenAll) = norm.transpose() * (faceL * nw);
                    tGPoint coordTrans_newN = coordTrans_new.rowwise().normalized();

                    tPoint axis_ev = coordTrans_newN.eigenvalues().array().abs();
                    real axis_cond = axis_ev.maxCoeff() / axis_ev.minCoeff(); // valid for 2-d as 0,0,1 is also normalized
                    if (axis_cond < 10.0)
                        coordTrans = coordTrans_new;
                }
                else if (settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::InertiaCoordBBSym)
                {
                    tGPoint coordTransN = coordTrans.rowwise().normalized(); // supposed to be a unitary matrix
                    coordTrans = coordTransN.transpose() * coordTrans;
                }
                else if (settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::InertiaCoordBB ||
                         settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::InertiaCoord)
                {
                }

                ConvertDiffsLinMap<dim>(DiffI_Norm, coordTrans);
                ConvertDiffsLinMap<dim>(DiffJ_Norm, coordTrans);

                AccumulateDiffOrderContributions<dim, powV>(DiffI_Norm, DiffJ_Norm, Conj, wgd, cnDiffs, 0);
            }
            return Conj;
        }

        real GetGreenGauss1WeightOnCell(index iCell)
        {
            if (settings.functionalSettings.greenGaussSpacial == 0)
            {
                real AR = GetCellAR(iCell);
                real v = std::max(0.0, std::log(AR));
                return settings.functionalSettings.greenGauss1Weight *
                       std::pow(std::tanh(v / 4), 3);
            }
            else
            {
                return settings.functionalSettings.greenGauss1Weight;
            }
        }

        real GetCellAR(index iCell)
        {
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
            auto lens = this->cellMajorHBox[iCell](Seq012);
            return (lens.maxCoeff() + verySmallReal) / (lens.minCoeff() + verySmallReal);
        }

        template <int nVarsFixed = 1>
        void MatrixAMult(tURec<nVarsFixed> &uRec, tURec<nVarsFixed> &uRec1)
        {
            DNDS_assert(coefficients_.matrixA.Size() == mesh->NumCellProc());
            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
                uRec1[iCell] = coefficients_.matrixA[iCell] * uRec[iCell];
        }

        template <int nVarsFixed = 1>
        auto DownCastURecOrder(
            int curOrder,
            index iCell,
            tURec<nVarsFixed> &uRec,
            int downCastMethod)
        {
            int degree2Start = dim == 3 ? 3 : 2;
            int degree3Start = dim == 3 ? 9 : 5;

            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> ret = uRec[iCell];
            if (downCastMethod == 1)
                DNDS_assert(coefficients_.volIntCholeskyL.size() == mesh->cell2node.father->Size());
            if (downCastMethod == 2)
                DNDS_assert(coefficients_.matrixACholeskyL.size() == mesh->cell2node.father->Size());

            auto toOrtho = [&]()
            {
                switch (downCastMethod)
                {
                case 0:
                    break;
                case 1:
                    coefficients_.volIntCholeskyL.at(iCell).template triangularView<Eigen::Lower>().applyThisOnTheLeft(ret);
                    break;
                case 2:
                    coefficients_.matrixACholeskyL.at(iCell).template triangularView<Eigen::Lower>().applyThisOnTheLeft(ret);
                    break;
                default:
                    DNDS_assert(false);
                    break;
                }
            };

            auto toOrigin = [&]()
            {
                switch (downCastMethod)
                {
                case 0:
                    break;
                case 1:
                    ret = coefficients_.volIntCholeskyL.at(iCell).template triangularView<Eigen::Lower>().solve(ret);
                    break;
                case 2:
                    ret = coefficients_.matrixACholeskyL.at(iCell).template triangularView<Eigen::Lower>().solve(ret);
                    break;
                default:
                    DNDS_assert(false);
                    break;
                }
            };

            switch (curOrder)
            {
            case 0:
                break;
            case 1:
                ret *= 0.0;
                break;
            case 2:
            {
                toOrtho();
                ret(Eigen::seq(degree2Start, EigenLast), EigenAll) *= 0.0;
                toOrigin();
            }
            break;
            case 3:
            {
                toOrtho();
                ret(Eigen::seq(degree3Start, EigenLast), EigenAll) *= 0.0;
                toOrigin();
            }
            break;
            default:
                std::cout << "bad input order : " << curOrder << std::endl;
                DNDS_assert(false);
                break;
            }

            return ret;
        }

        template <int nVarsFixed>
        void BuildURec(tURec<nVarsFixed> &u, int nVars, bool buildSon = true, bool buildTrans = true)
        {
            using namespace Geom::Base;
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            u.InitPair("VR::BuildURec::u", mpi);
            u.father->Resize(mesh->NumCell(), maxNDOF - 1, nVars);
            if (buildSon)
                u.son->Resize(mesh->NumCellGhost(), maxNDOF - 1, nVars);
            if (buildTrans)
            {
                DNDS_assert(buildSon);
                u.TransAttach();
                u.trans.BorrowGGIndexing(mesh->cell2node.trans);
                u.trans.createMPITypes();
                u.trans.initPersistentPull();
                u.trans.initPersistentPush();
            }

            for (index iCell = 0; iCell < u.Size(); iCell++)
                u[iCell].setZero();
        }

        template <int nVarsFixed>
        void BuildUGrad(tUGrad<nVarsFixed, dim> &u, int nVars, bool buildSon = true, bool buildTrans = true)
        {
            using namespace Geom::Base;
            u.InitPair("VR::BuildUGrad::u", mpi);
            u.father->Resize(mesh->NumCell(), dim, nVars);
            if (buildSon)
                u.son->Resize(mesh->NumCellGhost(), dim, nVars);
            if (buildTrans)
            {
                DNDS_assert(buildSon);
                u.TransAttach();
                u.trans.BorrowGGIndexing(mesh->cell2node.trans);
                u.trans.createMPITypes();
                u.trans.initPersistentPull();
                u.trans.initPersistentPush();
            }

            for (index iCell = 0; iCell < u.Size(); iCell++)
                u[iCell].setZero();
        }

        void BuildScalar(tScalarPair &u, bool buildSon = true, bool buildTrans = true)
        {
            u.InitPair("VR::BuildScalar::u", mpi);
            u.father->Resize(mesh->NumCell());
            if (buildSon)
                u.son->Resize(mesh->NumCellGhost());
            if (buildTrans)
            {
                DNDS_assert(buildSon);
                u.TransAttach();
                u.trans.BorrowGGIndexing(mesh->cell2node.trans);
                u.trans.createMPITypes();
                u.trans.initPersistentPull();
                u.trans.initPersistentPush();
            }

            for (index iCell = 0; iCell < u.Size(); iCell++)
                u(iCell, 0) = 0;
        }

        template <int nVarsFixed = 1>
        void BuildUDofNode(tUDof<nVarsFixed> &u, int nVars, bool buildSon = true, bool buildTrans = true)
        {
            u.InitPair("VR::BuildUDofNode::u", mpi);
            u.father->Resize(mesh->NumNode(), nVars, 1);
            if (buildSon)
                u.son->Resize(mesh->NumNodeGhost(), nVars, 1);
            if (buildTrans)
            {
                DNDS_assert(buildSon);
                u.TransAttach();
                u.trans.BorrowGGIndexing(mesh->coords.trans);
                u.trans.createMPITypes();
                u.trans.initPersistentPull();
                u.trans.initPersistentPush();
            }

            for (index iCell = 0; iCell < u.Size(); iCell++)
                u[iCell].setZero();
        }

        template <int nVarsFixed = 5>
        using TFBoundary = std::function<Eigen::Vector<real, nVarsFixed>(
            const Eigen::Vector<real, nVarsFixed> &, // UBL
            const Eigen::Vector<real, nVarsFixed> &, // UMEAN
            index, index, int,                       // iCell, iFace, ig
            const Geom::tPoint &,                    // Norm
            const Geom::tPoint &,                    // pPhy
            Geom::t_index fType                      // fCode
            )>;

        template <int nVarsFixed = 5>
        using TFBoundaryDiff = std::function<Eigen::Vector<real, nVarsFixed>(
            const Eigen::Vector<real, nVarsFixed> &, // UBL
            const Eigen::Vector<real, nVarsFixed> &, // dUMEAN
            const Eigen::Vector<real, nVarsFixed> &, // UMEAN
            index, index, int,                       // iCell, iFace, ig
            const Geom::tPoint &,                    // Norm
            const Geom::tPoint &,                    // pPhy
            Geom::t_index fType                      // fCode
            )>;

    private: // only used in reconstruction implementation hxx
        template <int nVarsFixed = 5>
        auto GetBoundaryRHS(tURec<nVarsFixed> &uRec,
                            tUDof<nVarsFixed> &u,
                            index iCell, index iFace,
                            const TFBoundary<nVarsFixed> &FBoundary);

    private: // only used in reconstruction implementation hxx
        template <int nVarsFixed = 5>
        auto GetBoundaryRHSDiff(tURec<nVarsFixed> &uRec,
                                tURec<nVarsFixed> &uRecDiff,
                                tUDof<nVarsFixed> &u,
                                index iCell, index iFace,
                                const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff);

    public:
        /**
         * \brief fallback reconstruction method,
         * explicit 2nd order FV reconstruction
         * \param uRec output, reconstructed gradients
         * \param u input, mean values
         * \param FBoundary see TFBoundary
         * \param method currently 1==2nd-order-GaussGreen
         */
        template <int nVarsFixed = 5>
        void DoReconstruction2ndGrad(
            tUGrad<nVarsFixed, dim> &uRec,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            int method);

        /**
         * \brief fallback reconstruction method,
         * explicit 2nd order FV reconstruction
         * \param uRec output, reconstructed gradients
         * \param u input, mean values
         * \param FBoundary see TFBoundary
         * \param method currently 1==2nd-order-GaussGreen
         */
        template <int nVarsFixed = 5>
        void DoReconstruction2nd(
            tURec<nVarsFixed> &uRec,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            int method,
            const std::vector<int> &mask);

        /**
         * \brief iterative variational reconstruction method
         * \details
         * DoReconstructionIter updates uRec (could locate into uRecNew)
         * $$
         * ur_i = A_{i}^{-1}B_{ij}ur_j +  A_{i}^{-1}b_{i}
         * $$
         * which is a fixed-point solver (using SOR/Jacobi sweeping)
         *
         * if recordInc, value in the output array is actually defined as :
         * $$
         * -(A_{i}^{-1}B_{ij}ur_j +  A_{i}^{-1}b_{i}) + ur_i
         * $$
         * which is the RHS of Block-Jacobi preconditioned system
         *
         * \param uRec input/ouput, reconstructed coefficients, is output if putIntoNew==false, otherwise is used as medium value
         * \param uRecNew input/ouput, reconstructed coefficients, is output if putIntoNew==true, otherwise is used as medium value
         * \param u input, mean values
         * \param FBoundary see TFBoundary
         * \param putIntoNew put valid output into uRecNew or not
         * \param recordInc if true, uRecNew holds the incremental value to this iteration
         * @warning mind that uRec could be overwritten
         */
        template <int nVarsFixed = 5>
        void DoReconstructionIter(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            bool putIntoNew = false,
            bool recordInc = false,
            bool uRecIsZero = false);
        /***********************************************************/

        /**
         * @brief puts into uRecNew with Mat * uRecDiff; uses the Block-jacobi preconditioned reconstruction system as Mat.
         * \details
         * the matrix operation can be viewed as
         * $$
         * vr_i = ur_j - A_{i}^{-1}B_{ij}ur_j
         * $$
         * which is the Jacobian of the DoReconstructionIter's RHS output.
         * Note that we account for nonlinear effects from BC
         * \param uRec input, the base value
         * \param uRecDiff input, the diff value, $x$ in $y=Jx$, or the disturbance value
         * \param uRecNew output, the result, $y$, or the response value
         * \param u input, mean value
         * \param FBoundary Vec F(const Vec &uL, const Vec& dUL, const tPoint &unitNorm, const tPoint &p, t_index faceID),
         * with Vec == Eigen::Vector<real, nVarsFixed>
         * uRecDiff should be untouched
         */
        template <int nVarsFixed = 5>
        void DoReconstructionIterDiff(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecDiff,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff);
        /***********************************************************/

        /** //TODO
         * @brief do a SOR iteration from uRecNew, with uRecInc as the RHSterm of Block-Jacobi preconditioned system
         */
        template <int nVarsFixed = 5>
        void DoReconstructionIterSOR(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecInc,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff,
            bool reverse = false);

        /***********************************************************/

        /***********************************************************/

        template <int nVarsFixed, int nVarsSee = 2>
        void DoCalculateSmoothIndicator(
            tScalarPair &si, tURec<nVarsFixed> &uRec, tUDof<nVarsFixed> &u,
            const std::array<int, nVarsSee> &varsSee);

        template <int nVarsFixed>
        using TFPost = std::function<void(Eigen::Matrix<real, 1, nVarsFixed> &)>; // post process reconstructed values (as row vector)

        template <int nVarsFixed>
        void DoCalculateSmoothIndicatorV1(
            tScalarPair &si, tURec<nVarsFixed> &uRec, tUDof<nVarsFixed> &u,
            const Eigen::Vector<real, nVarsFixed> &varsSee,
            const TFPost<nVarsFixed> &FPost);

        static const int maxRecDOFBatch = dim == 2 ? 4 : 10;
        static const int maxRecDOF = dim == 2 ? 9 : 19;
        static const int maxNDiff = dim == 2 ? 10 : 20;
        static const int maxNeighbour = 7;

        template <int nVarsFixed>
        using tLimitBatch = Eigen::Matrix<real, -1, nVarsFixed, 0, CFV::VariationalReconstruction<dim>::maxRecDOFBatch>;

        template <int nVarsFixed>
        using tFMEig = std::function<tLimitBatch<nVarsFixed>(
            const Eigen::Vector<real, nVarsFixed> &uL,
            const Eigen::Vector<real, nVarsFixed> &uR,
            const Geom::tPoint &uNorm,
            const Eigen::Ref<tLimitBatch<nVarsFixed>> &data)>;

        /**
         * @brief FM(uLeft,uRight,norm) gives vsize * vsize mat of Left Eigen Vectors
         *
         */
        template <int nVarsFixed>
        void DoLimiterWBAP_C(
            tUDof<nVarsFixed> &u,
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecNew,
            tURec<nVarsFixed> &uRecBuf,
            tScalarPair &si,
            bool ifAll,
            const tFMEig<nVarsFixed> &FM, const tFMEig<nVarsFixed> &FMI,
            bool putIntoNew = false);

        /**
         * @brief FM(uLeft,uRight,norm) gives vsize * vsize mat of Left Eigen Vectors
         *
         */
        template <int nVarsFixed>
        void DoLimiterWBAP_3(
            tUDof<nVarsFixed> &u,
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecNew,
            tURec<nVarsFixed> &uRecBuf,
            tScalarPair &si,
            bool ifAll,
            const tFMEig<nVarsFixed> &FM, const tFMEig<nVarsFixed> &FMI,
            bool putIntoNew = false);

        void WriteSerializeRecMatrix(const Serializer::SerializerBaseSSP &serializerP)
        {
            using namespace Geom;
            std::string name = "VR_Matrix";
            auto cwd = serializerP->GetCurrentPath();
            serializerP->CreatePath(name);
            serializerP->GoToPath(name);

            mesh->BuildCell2CellFace();
            mesh->cell2cellFace.WriteSerialize(serializerP, "cell2cellFace", true);
            mesh->AdjGlobal2LocalC2CFace();

            DNDS_assert(coefficients_.matrixAAInvB.father->Size() == mesh->NumCell());
            DNDS_assert(coefficients_.matrixAAInvB.son->Size() == mesh->NumCellGhost());
            coefficients_.matrixAAInvB.WriteSerialize(serializerP, "matrixAAInvB", false);

            serializerP->GoToPath(cwd);
        }
    };
}
// NOLINTBEGIN(bugprone-macro-parentheses)
#define DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(dim, nVarsFixed, ext)          \
    namespace DNDS::CFV                                                                         \
    {                                                                                           \
        ext template void VariationalReconstruction<dim>::DoReconstruction2ndGrad<nVarsFixed>(  \
            tUGrad<nVarsFixed, dim> & uRec,                                                     \
            tUDof<nVarsFixed> &u,                                                               \
            const TFBoundary<nVarsFixed> &FBoundary,                                            \
            int method);                                                                        \
                                                                                                \
        ext template void VariationalReconstruction<dim>::DoReconstruction2nd<nVarsFixed>(      \
            tURec<nVarsFixed> & uRec,                                                           \
            tUDof<nVarsFixed> &u,                                                               \
            const TFBoundary<nVarsFixed> &FBoundary,                                            \
            int method,                                                                         \
            const std::vector<int> &mask);                                                      \
                                                                                                \
        ext template void VariationalReconstruction<dim>::DoReconstructionIter<nVarsFixed>(     \
            tURec<nVarsFixed> & uRec,                                                           \
            tURec<nVarsFixed> &uRecNew,                                                         \
            tUDof<nVarsFixed> &u,                                                               \
            const TFBoundary<nVarsFixed> &FBoundary,                                            \
            bool putIntoNew,                                                                    \
            bool recordInc,                                                                     \
            bool uRecIsZero);                                                                   \
                                                                                                \
        ext template void VariationalReconstruction<dim>::DoReconstructionIterDiff<nVarsFixed>( \
            tURec<nVarsFixed> & uRec,                                                           \
            tURec<nVarsFixed> &uRecDiff,                                                        \
            tURec<nVarsFixed> &uRecNew,                                                         \
            tUDof<nVarsFixed> &u,                                                               \
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff);                                   \
                                                                                                \
        ext template void VariationalReconstruction<dim>::DoReconstructionIterSOR<nVarsFixed>(  \
            tURec<nVarsFixed> & uRec,                                                           \
            tURec<nVarsFixed> &uRecInc,                                                         \
            tURec<nVarsFixed> &uRecNew,                                                         \
            tUDof<nVarsFixed> &u,                                                               \
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff,                                    \
            bool reverse);                                                                      \
    }

DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(2, 4, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(2, 5, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(2, 6, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(2, 7, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(2, Eigen::Dynamic, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(3, 5, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(3, 6, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(3, 7, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(3, Eigen::Dynamic, extern)

#define DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(dim, nVarsFixed, ext)             \
    namespace DNDS::CFV                                                                              \
    {                                                                                                \
        ext template void VariationalReconstruction<dim>::DoCalculateSmoothIndicator<nVarsFixed, 2>( \
            tScalarPair & si, tURec<nVarsFixed> &uRec, tUDof<nVarsFixed> &u,                         \
            const std::array<int, 2> &varsSee);                                                      \
                                                                                                     \
        ext template void VariationalReconstruction<dim>::DoCalculateSmoothIndicatorV1<nVarsFixed>(  \
            tScalarPair & si, tURec<nVarsFixed> &uRec, tUDof<nVarsFixed> &u,                         \
            const Eigen::Vector<real, nVarsFixed> &varsSee,                                          \
            const TFPost<nVarsFixed> &FPost);                                                        \
                                                                                                     \
        ext template void VariationalReconstruction<dim>::DoLimiterWBAP_C(                           \
            tUDof<nVarsFixed> &u,                                                                    \
            tURec<nVarsFixed> &uRec,                                                                 \
            tURec<nVarsFixed> &uRecNew,                                                              \
            tURec<nVarsFixed> &uRecBuf,                                                              \
            tScalarPair &si,                                                                         \
            bool ifAll,                                                                              \
            const tFMEig<nVarsFixed> &FM, const tFMEig<nVarsFixed> &FMI,                             \
            bool putIntoNew);                                                                        \
                                                                                                     \
        ext template void VariationalReconstruction<dim>::DoLimiterWBAP_3(                           \
            tUDof<nVarsFixed> &u,                                                                    \
            tURec<nVarsFixed> &uRec,                                                                 \
            tURec<nVarsFixed> &uRecNew,                                                              \
            tURec<nVarsFixed> &uRecBuf,                                                              \
            tScalarPair &si,                                                                         \
            bool ifAll,                                                                              \
            const tFMEig<nVarsFixed> &FM, const tFMEig<nVarsFixed> &FMI,                             \
            bool putIntoNew);                                                                        \
    }
// NOLINTEND(bugprone-macro-parentheses)
DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(2, 1, extern)
DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(2, 4, extern)
DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(2, 5, extern)
DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(2, 6, extern)
DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(2, 7, extern)
DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(2, Eigen::Dynamic, extern)
DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(3, 1, extern)
DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(3, 5, extern)
DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(3, 6, extern)
DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(3, 7, extern)
DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(3, Eigen::Dynamic, extern)