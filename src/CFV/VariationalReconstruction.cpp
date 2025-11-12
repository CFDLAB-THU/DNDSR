
#include "VariationalReconstruction.hpp"
#include "DNDS/HardEigen.hpp"

namespace DNDS::CFV
{
    template <int dim>
    void
    VariationalReconstruction<dim>::
        ConstructMetrics()
    {
        using namespace Geom;
        using namespace Geom::Elem;

        /***************************************/
        // get volumes
        this->SetCellAtrBasic();
        this->ConstructCellVolume();
        this->ConstructCellBary();
        this->ConstructCellCent();
        this->ConstructCellIntJacobiDet();
        this->ConstructCellIntPPhysics();
        this->ConstructCellAlignedHBox();
        this->ConstructCellMajorHBoxCoordInertia();

        if (mpi.rank == mRank)
            log()
                << fmt::format(
                       "VariationalReconstruction<dim>::ConstructMetrics() === \n ===Sum/Min/Max Volume [{:.10g};  {:.5g}, {:.5g}]",
                       sumVolume, minVolume, maxVolume)
                << std::endl;

        /***************************************/
        // get areas
        this->SetFaceAtrBasic();
        this->ConstructFaceCent();
        this->ConstructFaceArea();
        this->ConstructFaceIntJacobiDet();
        this->ConstructFaceIntPPhysics();
        this->ConstructFaceUnitNorm();
        this->ConstructFaceMeanNorm();

        this->ConstructCellSmoothScale();
    }

    template void
    VariationalReconstruction<2>::
        ConstructMetrics();
    template void
    VariationalReconstruction<3>::
        ConstructMetrics();

    template <int dim>
    void
    VariationalReconstruction<dim>::
        ConstructBaseAndWeight(const tFGetBoundaryWeight &id2faceDircWeight)
    {
        using namespace Geom;
        using namespace Geom::Elem;
        using namespace Geom::Base;
        int maxNDOF = GetNDof<dim>(settings.maxOrder);
        // for polynomial: ndiff = ndof
        int maxNDIFF = maxNDOF;
        /******************************/
        // *cell's moment and cache
        this->MakePairDefaultOnCell(cellBaseMoment, maxNDOF);
        if (settings.cacheDiffBase)
        {
            this->MakePairDefaultOnCell(
                cellDiffBaseCache,
                std::min(maxNDIFF, static_cast<int>(settings.cacheDiffBaseSize)), maxNDOF);
            this->MakePairDefaultOnCell(
                cellDiffBaseCacheCent,
                std::min(maxNDIFF, static_cast<int>(settings.cacheDiffBaseSize)), maxNDOF);
        }
        if (needVolIntCholeskyL)
            volIntCholeskyL.resize(mesh->NumCellProc());
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            GetCellAtr(iCell).relax = settings.jacobiRelax;
            auto qCell = this->GetCellQuad(iCell);
            auto qCellO1 = this->GetCellQuadO1(iCell);
            auto qCellMax = Quadrature{mesh->GetCellElement(iCell), INT_ORDER_MAX};
            if (settings.cacheDiffBase)
            {
                cellDiffBaseCache.ResizeRow(iCell, qCell.GetNumPoints());
            }
            // std::cout << "hare" << std::endl;
            tSmallCoords coordsCell;
            mesh->GetCoordsOnCell(iCell, coordsCell);

            Eigen::RowVector<real, Eigen::Dynamic> m;
            m.setZero(GetCellAtr(iCell).NDOF);
            qCellMax.Integration(
                m,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsCell, DiNj);
                    real JDet = CellJacobianDet<dim>(coordsCell, DiNj) * (axisSymmetric ? pPhy(1) : 1.);
                    Eigen::RowVector<real, Eigen::Dynamic> vv;
                    vv.resizeLike(m);
                    this->FDiffBaseValue(vv, pPhy, iCell, -1, -2, 1); // un-dispatched call
                    DNDS_assert(vv(0) == 1);                          // must have 0th base
                    vInc = vv * JDet;
                });
            // std::cout << m << std::endl;
            cellBaseMoment[iCell] = m.transpose() / m(0); // 0th must be good
            SummationNoOp noOp;
            qCell.Integration(
                noOp,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    if (settings.cacheDiffBase)
                    {
                        MatrixXR dbv;
                        dbv.resize(
                            std::min(GetCellAtr(iCell).NDIFF, settings.cacheDiffBaseSize),
                            GetCellAtr(iCell).NDOF);
                        this->FDiffBaseValue(dbv, this->GetCellQuadraturePPhys(iCell, iG), iCell, -1, iG, 0);
                        cellDiffBaseCache(iCell, iG) = dbv;
                    }
                });
            qCellO1.Integration(
                noOp,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    if (settings.cacheDiffBase)
                    {
                        MatrixXR dbv;
                        dbv.resize(
                            std::min(GetCellAtr(iCell).NDIFF, settings.cacheDiffBaseSize),
                            GetCellAtr(iCell).NDOF);
                        this->FDiffBaseValue(dbv, this->GetCellQuadraturePPhys(iCell, -1), iCell, -1, -1, 0);
                        cellDiffBaseCacheCent[iCell] = dbv;
                    }
                });

            //****** Get Orthogonization Coefs
            MatrixXR MBiBj;
            MBiBj.setZero(GetCellAtr(iCell).NDOF - 1, GetCellAtr(iCell).NDOF - 1);
            qCell.Integration(
                MBiBj,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    MatrixXR D0Bj =
                        this->GetIntPointDiffBaseValue(iCell, -1, -1, iG, std::array<int, 1>{0}, 1);
                    vInc = (D0Bj.transpose() * D0Bj);
                    vInc *= this->GetCellJacobiDet(iCell, iG);
                });
            if (needVolIntCholeskyL)
                volIntCholeskyL.at(iCell).resizeLike(MBiBj),
                    volIntCholeskyL.at(iCell) = MBiBj.llt().matrixL();
        }

        /******************************/
        // *face's weight and cache

        this->MakePairDefaultOnFace(faceWeight, settings.maxOrder + 1);
        this->MakePairDefaultOnFace(faceAlignedScales);
        this->MakePairDefaultOnFace(faceMajorCoordScale, 3, 3);
        if (settings.cacheDiffBase)
        {
            this->MakePairDefaultOnFace(
                faceDiffBaseCache,
                std::min(maxNDIFF, static_cast<int>(settings.cacheDiffBaseSize)), maxNDOF);
            this->MakePairDefaultOnFace(
                faceDiffBaseCacheCent,
                std::min(maxNDIFF, static_cast<int>(settings.cacheDiffBaseSize)), maxNDOF * 2);
        }
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto qFace = this->GetFaceQuad(iFace);
            auto qFaceO1 = this->GetFaceQuadO1(iFace);
            if (settings.cacheDiffBase)
                faceDiffBaseCache.ResizeRow(iFace, 2 * qFace.GetNumPoints());

            tSmallCoords coords, coordsC;
            mesh->GetCoordsOnFace(iFace, coords);
            coordsC = coords;
            coordsC.colwise() -= faceCent[iFace];

            // * get bdv cache
            SummationNoOp noOp;
            for (int if2c = 0; if2c < 2; if2c++)
            {
                index iCell = mesh->face2cell(iFace, if2c);
                if (iCell == UnInitIndex)
                    continue;
                if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)))
                    DNDS_assert(if2c == 0);
                else if (FaceIDIsPeriodic(mesh->GetFaceZone(iFace)))
                {
                    // TODO: handle the case with periodic
                    // DNDS_assert(false); //! do nothing?
                }
                qFace.Integration(
                    noOp,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        if (settings.cacheDiffBase)
                        {
                            MatrixXR dbv;
                            dbv.resize(
                                std::min(GetFaceAtr(iFace).NDIFF, settings.cacheDiffBaseSize),
                                GetCellAtr(iCell).NDOF);
                            this->FDiffBaseValue(dbv, this->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, iG), iCell, iFace, iG, 0);
                            faceDiffBaseCache(iFace, if2c * qFace.GetNumPoints() + iG) = dbv;
                        }
                    });
                qFaceO1.Integration(
                    noOp,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        if (settings.cacheDiffBase)
                        {
                            MatrixXR dbv;
                            dbv.resize(
                                std::min(GetFaceAtr(iFace).NDIFF, settings.cacheDiffBaseSize),
                                GetCellAtr(iCell).NDOF);
                            this->FDiffBaseValue(dbv, this->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, -1), iCell, iFace, -1, 0);
                            int maxNDOF = GetNDof<dim>(settings.maxOrder);
                            faceDiffBaseCacheCent[iFace](
                                EigenAll,
                                Eigen::seq(if2c * maxNDOF,
                                           if2c * maxNDOF + maxNDOF - 1)) = dbv;
                        }
                    });
            }

            // *get face (derivatives) scale: cell average AlignedHBox mode
            //! note: if use cell-2-cell distance, note periodic, use wrapped call here
            tPoint faceScale{0, 0, 0};
            int nF2C{0};
            for (int if2c = 0; if2c < 2; if2c++)
            {
                index iCell = mesh->face2cell(iFace, if2c);
                if (iCell == UnInitIndex)
                    continue;
                if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)))
                    DNDS_assert(if2c == 0);
                else if (FaceIDIsPeriodic(mesh->GetFaceZone(iFace)))
                {
                    // TODO: handle the case with periodic
                    // DNDS_assert(false); // !do nothing for now
                }
                faceScale += cellAlignedHBox[iCell];
                nF2C++;
            }
            DNDS_assert(nF2C > 0);
            faceScale /= nF2C;
            faceAlignedScales[iFace] = faceScale;
            Geom::tPoint faceBaryDiffV;
            if (mesh->face2cell(iFace, 1) != UnInitIndex)
            {
                faceBaryDiffV =
                    this->GetOtherCellBaryFromCell(mesh->face2cell(iFace, 0),
                                                   mesh->face2cell(iFace, 1), iFace) -
                    this->GetCellBary(mesh->face2cell(iFace, 0));
            }
            else
            {
                Geom::tPoint vCB = this->GetFaceQuadraturePPhysFromCell(
                                       iFace,
                                       mesh->face2cell(iFace, 0), 0, -1) -
                                   this->GetCellBary(mesh->face2cell(iFace, 0));
                auto uNorm = this->GetFaceNormFromCell(
                    iFace,
                    mesh->face2cell(iFace, 0), 0, -1);
                faceBaryDiffV =
                    2 * vCB.dot(uNorm) *
                    this->GetFaceNorm(iFace, -1);
            }
            Geom::tPoint faceBaryDiffL, faceBaryDiffR;
            faceBaryDiffL =
                this->GetCellBary(mesh->face2cell(iFace, 0)) -
                this->GetFaceQuadraturePPhysFromCell(iFace, mesh->face2cell(iFace, 0), 0, -1);
            faceBaryDiffR = faceBaryDiffV + faceBaryDiffL;
            // std::cout << faceBaryDiffV.transpose() << std::endl;
            // std::cout << faceBaryDiffV.norm() << std::endl;
            real volL, volR;
            volL = volR = std::pow(GetCellVol(mesh->face2cell(iFace, 0)) + verySmallReal, settings.functionalSettings.inertiaWeightPower);
            Geom::tGPoint cellInertiaL = cellInertia[mesh->face2cell(iFace, 0)] * volL;
            Geom::tGPoint cellInertiaR = cellInertiaL;
            if (mesh->face2cell(iFace, 1) != UnInitIndex)
            {
                volR = std::pow(GetCellVol(mesh->face2cell(iFace, 1)) + verySmallReal, settings.functionalSettings.inertiaWeightPower);
                cellInertiaR = this->GetOtherCellInertiaFromCell(
                                   mesh->face2cell(iFace, 0),
                                   mesh->face2cell(iFace, 1), iFace) *
                               volR;
            }
            Geom::tGPoint faceInertiaC =
                (cellInertiaL + cellInertiaR) / (volL + volR);

            Geom::tGPoint faceCoord;
            faceCoord.setIdentity();
            if constexpr (dim == 3)
                faceCoord = HardEigen::Eigen3x3RealSymEigenDecomposition(faceInertiaC);
            else
                faceCoord({0, 1}, {0, 1}) = HardEigen::Eigen2x2RealSymEigenDecomposition(faceInertiaC({0, 1}, {0, 1}));
            faceMajorCoordScale[iFace] = faceCoord;
            if (settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::InertiaCoordBB ||
                settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::InertiaCoordBBNorm ||
                settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::AnisotropicType::InertiaCoordBBSym)
            {
                Geom::tGPoint faceCoordNorm = faceCoord.colwise().normalized();
                tSmallCoords coordsL, coordsR;
                coordsL = coordsC.colwise() - faceBaryDiffL;
                coordsR = coordsC.colwise() - faceBaryDiffR;
                coordsL = faceCoordNorm.transpose() * coordsL;
                coordsR = faceCoordNorm.transpose() * coordsR;
                Geom::tPoint faceCoordBB =
                    coordsL.array().abs().rowwise().maxCoeff().max(
                        coordsR.array().abs().rowwise().maxCoeff());

                faceMajorCoordScale[iFace] = faceCoordNorm * faceCoordBB.asDiagonal();
            }
            real AR2 = faceMajorCoordScale[iFace].col(0).norm() / faceMajorCoordScale[iFace].col(1).norm();
            // if (std::abs(std::log10(AR2)) > 2.0)
            // {
            //     std::cout
            //         << "Face scale: ";
            //     std::cout << this->GetFaceNorm(iFace, -1).transpose() << "\n\n";
            //     std::cout << faceCent[iFace].transpose() << "\n\n";
            //     std::cout << faceInertiaC << std::endl;

            //     faceCoord = faceMajorCoordScale[iFace];
            //     faceCoord.colwise().normalize();
            //     std::cout << faceCoord << "\n\n";
            //     std::cout
            //         << faceMajorCoordScale[iFace] << "\n"
            //         << std::endl;
            // }
            if (settings.functionalSettings.scaleType == VRSettings::FunctionalSettings::ScaleType::BaryDiff)
            {
                faceAlignedScales[iFace] = faceBaryDiffV;
                if constexpr (dim == 2)
                    faceAlignedScales[iFace](2) = 1;
            }

            // *get geom weight ic2f
            real wg = 1;
            if (settings.functionalSettings.geomWeightScheme == VRSettings::FunctionalSettings::GeomWeightScheme::HQM_SD)
                wg = settings.functionalSettings.geomWeightBias +
                     std::pow(std::pow(this->GetFaceArea(iFace), 1. / real(dim - 1)) / faceBaryDiffV.norm(), settings.functionalSettings.geomWeightPower * 0.5);
            if (settings.functionalSettings.geomWeightScheme == VRSettings::FunctionalSettings::GeomWeightScheme::SD_Power)
                wg = settings.functionalSettings.geomWeightBias +
                     std::pow(this->GetFaceArea(iFace), settings.functionalSettings.geomWeightPower1 * 0.5) *
                         std::pow(faceBaryDiffV.norm(), settings.functionalSettings.geomWeightPower2 * 0.5);

            // *get dir weight
            Eigen::Vector<real, Eigen::Dynamic> wd;
            wd.resize(settings.maxOrder + 1);
            switch (settings.functionalSettings.dirWeightScheme)
            {
            case VRSettings::FunctionalSettings::DirWeightScheme::Factorial:
                for (int p = 0; p < wd.size(); p++)
                    wd[p] = 1. / factorials[p];
                break;
            case VRSettings::FunctionalSettings::DirWeightScheme::HQM_OPT:
                switch (settings.maxOrder)
                {
                case 1:
                    wd[0] = wd[1] = 1;
                    break;
                case 2:
                    wd[0] = 1, wd[1] = 0.4643, wd[2] = 0.1559;
                    break;
                case 3:
                    wd[0] = 1, wd[1] = .5295, wd[2] = wd[3] = .2117;
                    break;
                default:
                    DNDS_assert(false);
                    break;
                }
                break;
            case VRSettings::FunctionalSettings::DirWeightScheme::ManualDirWeight:
                switch (settings.maxOrder)
                {
                case 3:
                    wd[3] = settings.functionalSettings.manualDirWeights(3);
                case 2:
                    wd[2] = settings.functionalSettings.manualDirWeights(2);
                case 1:
                    wd[1] = settings.functionalSettings.manualDirWeights(1);
                    wd[0] = settings.functionalSettings.manualDirWeights(0);
                    break;
                default:
                    DNDS_assert(false);
                    break;
                }
                break;
            case VRSettings::FunctionalSettings::DirWeightScheme::TEST_OPT:
            {
                if (settings.maxOrder != 3) // use manual value
                {
                    switch (settings.maxOrder)
                    {
                    case 3:
                        wd[3] = settings.functionalSettings.manualDirWeights(3);
                    case 2:
                        wd[2] = settings.functionalSettings.manualDirWeights(2);
                    case 1:
                        wd[1] = settings.functionalSettings.manualDirWeights(1);
                        wd[0] = settings.functionalSettings.manualDirWeights(0);
                        break;
                    default:
                        DNDS_assert(false);
                        break;
                    }
                }
                else if (dim == 3) // current scheme of a/d b/d geom weighting
                {
                    auto faceSpan = Geom::Elem::GetElemNodeMajorSpan(coords);
                    real a = faceSpan(0);
                    real b = faceSpan(1);
                    real d = faceBaryDiffV.norm();
                    real r1 = a / (d + verySmallReal);
                    real r2 = b / (d + verySmallReal);
                    real rr = std::sqrt(sqr(r1) + sqr(r2));
                    real lrr = std::log10(rr);

                    wd[0] = std::sqrt(std::pow(1.0031 + 0.86 * std::pow(r1 * r2, 0.5327), 0.98132));
                    wd[1] = wd[0] * std::sqrt(0.3604 - 0.0774 * std::exp(-0.5 * sqr((lrr - 1.8e-10) / 0.1325)));
                    wd[2] = wd[0] * std::sqrt(0.0676 - 0.0547 * std::exp(-0.5 * sqr((lrr + 2.74e-10) / 0.1376)));
                    wd[3] = wd[0] * std::sqrt(5.299e-2 - 4.63e-2 * std::exp(-0.5 * sqr((lrr - 9.028e-12) / 0.175)));
                    wg = 1; // force no wg
                }
                else if (dim == 2)
                {
                    real r = this->GetFaceArea(iFace) / faceBaryDiffV.norm();
                    real lr = std::log10(r);
                    wd[0] = std::sqrt(std::pow(1.00745 + r, 1.004));
                    wd[1] = wd[0] * std::sqrt(0.359 - 0.0674 * std::exp(-sqr(lr - 5.18e-11) / (2 * sqr(0.124))));
                    wd[2] = wd[0] * std::sqrt(0.0676 - 0.0507 * std::exp(-sqr(lr - 3.15e-10) / (2 * sqr(0.137))));
                    wd[3] = wd[0] * std::sqrt(0.0529 - 0.0487 * std::exp(-sqr(lr - 1.33e-10) / (2 * sqr(0.167))));
                    wg = 1; // force no wg
                }
            }
            break;
            default:
                DNDS_assert(false);
                break;
            }

            if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)) || FaceIDIsPeriodic(mesh->GetFaceZone(iFace)))
            // if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)))
            {
                for (int iOrder = 0; iOrder <= settings.maxOrder; iOrder++)
                    wd(iOrder) *= id2faceDircWeight(mesh->GetFaceZone(iFace), iOrder) * settings.bcWeight;
            }
            faceWeight[iFace] = wd * wg;
        }

        if (settings.cacheDiffBase)
        {
            faceDiffBaseCache.CompressBoth();
            cellDiffBaseCache.CompressBoth();

            // faceDiffBaseCache.trans.pullOnce(); //!err: need adding comm preparation first
            // faceDiffBaseCacheCent.trans.pullOnce(); //!err: need adding comm preparation first
        }

        // faceWeight.trans.pullOnce(); //!err: need adding comm preparation first
        // faceAlignedScales.trans.pullOnce(); //!err: need adding comm preparation first
        // faceMajorCoordScale.trans.pullOnce(); //!err: need adding comm preparation first

        if (!settings.intOrderVRIsSame())
        {
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                auto c2f = mesh->cell2face[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = CellFaceOther(iCell, iFace);
                    auto faceID = mesh->GetFaceZone(iFace);
                    int if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                    if (iCellOther == UnInitIndex)
                    {
                        DNDS_assert(FaceIDIsExternalBC(faceID));
                        DNDS_assert(bndVRCaches.count(iFace) == 0);
                        auto qFace = Quadrature(mesh->GetFaceElement(iFace), settings.intOrderVRValue());
                        bndVRCaches[iFace].reserve(qFace.GetNumPoints());
                        tSmallCoords coords;
                        mesh->GetCoordsOnFace(iFace, coords);
                        SummationNoOp noOp;
                        qFace.Integration(
                            noOp,
                            [&](auto &vInc, int __xxx_iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                            {
                                tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                                tJacobi J = Elem::ShapeJacobianCoordD01Nj(coords, DiNj);
                                real JDet = JacobiDetFace<dim>(J) * (axisSymmetric ? pPhy(1) : 1.);
                                tPoint np = FacialJacobianToNormVec<dim>(J);
                                RowVectorXR dbv, dbvD;
                                dbvD.resize(1, GetCellAtr(iCell).NDOF);
                                this->FDiffBaseValue(dbvD, this->GetFacePointFromCell(iFace, iCell, -1, pPhy), iCell, iFace, -2, 0);
                                dbv = dbvD(0, Eigen::seq(Eigen::fix<1>, EigenLast));
                                BndVRPointCache cacheEntry;
                                cacheEntry.D0Bj = dbv;
                                cacheEntry.JDet = JDet;
                                cacheEntry.norm = np;
                                cacheEntry.PPhy = pPhy;
                                bndVRCaches.at(iFace).emplace_back(std::move(cacheEntry));
                            });
                    }
                }
            }
        }

        if (mpi.rank == mRank)
            log()
                << "VariationalReconstruction<dim>::ConstructBaseAndWeight() done" << std::endl;
    }

    template void
    VariationalReconstruction<2>::
        ConstructBaseAndWeight(const tFGetBoundaryWeight &id2faceDircWeight);
    template void
    VariationalReconstruction<3>::
        ConstructBaseAndWeight(const tFGetBoundaryWeight &id2faceDircWeight);

    template <int dim>
    void
    VariationalReconstruction<dim>::ConstructRecCoeff()
    {
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        using namespace Geom;
        using namespace Geom::Elem;
        using namespace Geom::Base;
        int maxNDOF = GetNDof<dim>(settings.maxOrder);
        if (needOriginalMatrix)
            this->MakePairDefaultOnCell(matrixAB, maxNDOF - 1, maxNDOF - 1);
        this->MakePairDefaultOnCell(matrixAAInvB, maxNDOF - 1, maxNDOF - 1);
        if (needOriginalMatrix)
            this->MakePairDefaultOnCell(vectorB, maxNDOF - 1, 1);
        this->MakePairDefaultOnCell(vectorAInvB, maxNDOF - 1, 1);
        if (settings.functionalSettings.greenGauss1Weight != 0)
            this->MakePairDefaultOnCell(matrixAHalf_GG, dim, (maxNDOF - 1));
        this->MakePairDefaultOnCell(matrixA, (maxNDOF - 1), (maxNDOF - 1));
        if (needMatrixACholeskyL)
            matrixACholeskyL.resize(mesh->NumCellProc());
        real maxCond = 0.0;
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++) // only non-ghost
        {
            if (needOriginalMatrix)
                matrixAB.ResizeRow(iCell, mesh->cell2face.RowSize(iCell) + 1);
            matrixAAInvB.ResizeRow(iCell, mesh->cell2face.RowSize(iCell) + 1);
            if (needOriginalMatrix)
                vectorB.ResizeRow(iCell, mesh->cell2face.RowSize(iCell));
            vectorAInvB.ResizeRow(iCell, mesh->cell2face.RowSize(iCell));
            std::vector<MatrixXR> local_Bs;
            std::vector<Eigen::Vector<real, Eigen::Dynamic>> local_bs;
            local_Bs.reserve(mesh->cell2face.RowSize(iCell));
            local_bs.reserve(mesh->cell2face.RowSize(iCell));
            //*get A
            MatrixXR A;
            A.setZero(GetCellAtr(iCell).NDOF - 1, GetCellAtr(iCell).NDOF - 1);
            for (int ic2f = 0; ic2f < mesh->cell2face.RowSize(iCell); ic2f++)
            {
                index iFace = mesh->cell2face(iCell, ic2f);
                auto qFace = this->GetFaceQuad(iFace);
                auto qFaceVR = Quadrature(mesh->GetFaceElement(iFace), settings.intOrderVRValue());
                tSmallCoords coords;
                if (!settings.intOrderVRIsSame())
                    mesh->GetCoordsOnFace(iFace, coords);
                qFaceVR.Integration(
                    A,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        decltype(A) DiffI;
                        real JDet{0};
                        if (settings.intOrderVRIsSame())
                        {
                            DiffI = this->GetIntPointDiffBaseValue(iCell, iFace, -1, iG, EigenAll);
                            JDet = this->GetFaceJacobiDet(iFace, iG);
                        }
                        else
                        {
                            tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                            JDet = FaceJacobianDet(dim, coords, DiNj) * (axisSymmetric ? pPhy(1) : 1.);
                            MatrixXR dbv;
                            dbv.resize(GetFaceAtr(iFace).NDIFF, GetCellAtr(iCell).NDOF);
                            this->FDiffBaseValue(dbv, this->GetFacePointFromCell(iFace, iCell, -1, pPhy), iCell, iFace, -2, 0);
                            DiffI = dbv(EigenAll, Eigen::seq(Eigen::fix<1>, EigenLast));
                        }
                        vInc = this->FFaceFunctional(DiffI, DiffI, iFace, iCell, iCell);
                        vInc *= JDet;
                    });
            }
            Eigen::Matrix<real, dim, Eigen::Dynamic> AHalf_GG;
            if (settings.functionalSettings.greenGauss1Weight != 0)
            {
                //* reduced functional on compact stencil
                AHalf_GG.resize(Eigen::NoChange, A.cols());
                AHalf_GG.setZero();
                // AHalf's central part:
                auto qCell = this->GetCellQuad(iCell);
                qCell.IntegrationSimple(
                    AHalf_GG,
                    [&](decltype(AHalf_GG) &inc, int iG)
                    {
                        inc = this->GetIntPointDiffBaseValue(
                            iCell, -1, -1, iG,
                            Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), dim + 1);
                        // using no scaling here
                        inc *= this->GetCellJacobiDet(iCell, iG);
                    });
                for (int ic2f = 0; ic2f < mesh->cell2face.RowSize(iCell); ic2f++)
                {
                    index iFace = mesh->cell2face(iCell, ic2f);
                    index iCellOther = this->CellFaceOther(iCell, iFace);
                    auto qFace = this->GetFaceQuad(iFace);
                    int if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                    qFace.IntegrationSimple(
                        AHalf_GG,
                        [&](decltype(AHalf_GG) &inc, int iG)
                        {
                            tPoint normOut = this->GetFaceNormFromCell(iFace, iCell, if2c, iG) *
                                             (if2c ? -1 : 1);
                            inc = normOut(Seq012) *
                                  this->GetIntPointDiffBaseValue(
                                      iCell, iFace, -1, iG,
                                      0, 1);
                            inc *= (1 - settings.functionalSettings.greenGauss1Bias);
                            if (iCellOther != UnInitIndex)
                            {
                                real dLR = (GetOtherCellBaryFromCell(iCell, iCellOther, iFace) - GetCellBary(iCell)).norm();
                                inc -= normOut(Seq012) *
                                       (normOut(Seq012).transpose() *
                                        this->GetIntPointDiffBaseValue(iCell, iFace, -1, iG, Seq123, dim + 1)) *
                                       dLR *
                                       settings.functionalSettings.greenGauss1Penalty;
                            }

                            inc *= (-1) * this->GetFaceJacobiDet(iFace, iG);
                        });
                }
                // std::cout << "-------------\n";
                // std::cout << AHalf_GG << std::endl;
                // std::cout << "A\n"
                //           << A << std::endl;
                // std::cout << "IncA\n"
                //           << (AHalf_GG.transpose() * AHalf_GG) << std::endl;
                A += (AHalf_GG.transpose() * AHalf_GG) * this->GetGreenGauss1WeightOnCell(iCell);
                matrixAHalf_GG[iCell] = AHalf_GG;
            }

            // if (iCell == 71)
            // {
            // if (std::abs(A(0, 0) - 0.2083333333) > 1e-5)
            // {
            //     std::cout << "=================" << std::endl;
            //     std::cout << A << std::endl;
            //     std::cout << cellCent[iCell] << std::endl;
            // }
            // std::abort();
            // }
            //*get B
            for (int ic2f = 0; ic2f < mesh->cell2face.RowSize(iCell); ic2f++)
            {
                index iFace = mesh->cell2face(iCell, ic2f);
                auto qFace = this->GetFaceQuad(iFace);
                auto qFaceVR = Quadrature(mesh->GetFaceElement(iFace), settings.intOrderVRValue());
                tSmallCoords coords;
                if (!settings.intOrderVRIsSame())
                    mesh->GetCoordsOnFace(iFace, coords);
                index iCellOther = CellFaceOther(iCell, iFace);
                int if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                auto faceID = mesh->GetFaceZone(iFace);
                if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)))
                {
                    DNDS_assert(iCellOther == UnInitIndex);
                    local_Bs.emplace_back();
                    continue;
                    // if is periodic, then use internal //TODO!
                }
                MatrixXR B;
                B.setZero(GetCellAtr(iCell).NDOF - 1, GetCellAtr(iCellOther).NDOF - 1);
                qFaceVR.Integration(
                    B,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        decltype(B) DiffI;
                        decltype(B) DiffJ;
                        real JDet{0};
                        if (settings.intOrderVRIsSame())
                        {
                            JDet = this->GetFaceJacobiDet(iFace, iG);
                            DiffI = this->GetIntPointDiffBaseValue(iCell, iFace, -1, iG, EigenAll);
                            DiffJ = this->GetIntPointDiffBaseValue(iCellOther, iFace, -1, iG, EigenAll);
                        }
                        else
                        {
                            tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                            JDet = FaceJacobianDet(dim, coords, DiNj) * (axisSymmetric ? pPhy(1) : 1.);
                            MatrixXR dbvI, dbvJ;
                            dbvI.resize(GetFaceAtr(iFace).NDIFF, GetCellAtr(iCell).NDOF);
                            dbvJ.resize(GetFaceAtr(iFace).NDIFF, GetCellAtr(iCellOther).NDOF);
                            this->FDiffBaseValue(dbvI, this->GetFacePointFromCell(iFace, iCell, -1, pPhy), iCell, iFace, -2, 0);
                            this->FDiffBaseValue(dbvJ, this->GetFacePointFromCell(iFace, iCellOther, -1, pPhy), iCellOther, iFace, -2, 0);

                            DiffI = dbvI(EigenAll, Eigen::seq(Eigen::fix<1>, EigenLast));
                            DiffJ = dbvJ(EigenAll, Eigen::seq(Eigen::fix<1>, EigenLast));
                        }
                        if (mesh->isPeriodic)
                        {
                            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                            {
                                periodicity.TransDiValueInplace<dim>(DiffJ, faceID);
                            }
                            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                            {
                                periodicity.TransDiValueBackInplace<dim>(DiffJ, faceID);
                            }
                        }

                        vInc = this->FFaceFunctional(DiffI, DiffJ, iFace, iCell, iCellOther);
                        vInc *= JDet;
                    });
                if (settings.functionalSettings.greenGauss1Weight != 0)
                {
                    decltype(AHalf_GG) BHalf_GG;
                    BHalf_GG.resize(Eigen::NoChange, B.cols());
                    BHalf_GG.setZero();
                    qFace.IntegrationSimple(
                        BHalf_GG,
                        [&](decltype(BHalf_GG) &inc, int iG)
                        {
                            tPoint normOut = this->GetFaceNormFromCell(iFace, iCell, if2c, iG) *
                                             (if2c ? -1 : 1);
                            inc = normOut(Seq012) *
                                  this->GetIntPointDiffBaseValue(iCellOther, iFace, 1 - if2c, iG, 0, 1); // need 1-if2c!!if2c is for iCell!
                            inc *= settings.functionalSettings.greenGauss1Bias;
                            real dLR = (GetOtherCellBaryFromCell(iCell, iCellOther, iFace) - GetCellBary(iCell)).norm();
                            inc += normOut(Seq012) *
                                   (normOut(Seq012).transpose() *
                                    this->GetIntPointDiffBaseValue(iCellOther, iFace, 1 - if2c, iG, Seq123, dim + 1)) *
                                   dLR *
                                   settings.functionalSettings.greenGauss1Penalty;
                            inc *= this->GetFaceJacobiDet(iFace, iG);
                        });
                    // std::cout << " BH " << ic2f << " " << iFace << "\n"
                    //           << BHalf_GG << std::endl;
                    // std::cout << "B\n"
                    //           << B << std::endl;
                    // std::cout << "IncB\n"
                    //           << AHalf_GG.transpose() * BHalf_GG << std::endl;
                    B += AHalf_GG.transpose() * BHalf_GG * this->GetGreenGauss1WeightOnCell(iCell);
                }
                if (iCellOther == iCell) //* coincide periodic
                    A -= B, B *= 0;
                if (needOriginalMatrix)
                    matrixAB(iCell, 1 + ic2f) = B;
                local_Bs.emplace_back(std::move(B));
            }

            //*get b
            for (int ic2f = 0; ic2f < mesh->cell2face.RowSize(iCell); ic2f++)
            {
                index iFace = mesh->cell2face(iCell, ic2f);
                int if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                auto qFace = this->GetFaceQuad(iFace);
                auto qFaceVR = Quadrature(mesh->GetFaceElement(iFace), settings.intOrderVRValue());
                tSmallCoords coords;
                if (!settings.intOrderVRIsSame())
                    mesh->GetCoordsOnFace(iFace, coords);
                VectorXR b;
                b.setZero(GetCellAtr(iCell).NDOF - 1, 1);
                qFaceVR.Integration(
                    b,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        Eigen::RowVector<real, Eigen::Dynamic> DiffI;
                        real JDet{0};
                        if (settings.intOrderVRIsSame())
                        {
                            JDet = this->GetFaceJacobiDet(iFace, iG);
                            DiffI = this->GetIntPointDiffBaseValue(iCell, iFace, -1, iG, std::array<int, 1>{0}, 1);
                        }
                        else
                        {
                            tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                            JDet = FaceJacobianDet(dim, coords, DiNj) * (axisSymmetric ? pPhy(1) : 1.);
                            MatrixXR dbv;
                            dbv.resize(1, GetCellAtr(iCell).NDOF);
                            this->FDiffBaseValue(dbv, this->GetFacePointFromCell(iFace, iCell, -1, pPhy), iCell, iFace, -2, 0);
                            DiffI = dbv(EigenAll, Eigen::seq(Eigen::fix<1>, EigenLast));
                        }
                        vInc = this->FFaceFunctional(DiffI, Eigen::MatrixXd::Ones(1, 1), iFace, iCell, iCell);
                        vInc *= JDet;
                    });
                if (settings.functionalSettings.greenGauss1Weight != 0)
                {
                    Eigen::Matrix<real, dim, 1> bHalf_GG;
                    bHalf_GG.setZero();
                    qFace.IntegrationSimple(
                        bHalf_GG,
                        [&](auto &inc, int iG)
                        {
                            inc = this->GetFaceNormFromCell(iFace, iCell, -1, iG)(Seq012) *
                                  (if2c ? -1 : 1);
                            inc *= settings.functionalSettings.greenGauss1Bias * this->GetFaceJacobiDet(iFace, iG);
                        });
                    // std::cout << " bH " << ic2f << " " << iFace << "\n"
                    //           << bHalf_GG << std::endl;
                    // std::cout << "b\n"
                    //           << b << std::endl;
                    // std::cout << "Incb\n"
                    //           << AHalf_GG.transpose() * bHalf_GG << std::endl;
                    b += AHalf_GG.transpose() * bHalf_GG * this->GetGreenGauss1WeightOnCell(iCell);
                }
                if (needOriginalMatrix)
                    vectorB(iCell, ic2f) = b;
                local_bs.emplace_back(std::move(b));
            }

            // * get AInv and fill A, AInv

            decltype(A) AInv;
            real aCond{0};
            if (settings.svdTolerance)
                aCond = HardEigen::EigenLeastSquareInverse_Filtered(A, AInv, settings.svdTolerance, 1);
            else
                aCond = HardEigen::EigenLeastSquareInverse(A, AInv, settings.svdTolerance);
            if (needOriginalMatrix)
                matrixAB(iCell, 0) = A;
            matrixAAInvB(iCell, 0) = AInv;
            matrixA[iCell] = A;

            maxCond = std::max(aCond, maxCond);
            if (needMatrixACholeskyL)
                matrixACholeskyL.at(iCell) = A.llt().matrixL();
            DNDS_assert(AInv.allFinite());

            // * get AInvB and AInvb
            for (int ic2f = 0; ic2f < mesh->cell2face.RowSize(iCell); ic2f++)
            {
                index iFace = mesh->cell2face(iCell, ic2f);
                auto qFace = this->GetFaceQuad(iFace);
                index iCellOther = CellFaceOther(iCell, iFace);
                int if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                auto faceID = mesh->GetFaceZone(iFace);
                if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)))
                {
                    DNDS_assert(iCellOther == UnInitIndex);
                    continue;
                }
                matrixAAInvB(iCell, 1 + ic2f) = AInv * local_Bs.at(ic2f);
                vectorAInvB(iCell, ic2f) = AInv * local_bs.at(ic2f);
            }
            // std::cout << "=============" << std::endl;
            // std::cout << AInv << std::endl;
            // std::abort();
        }
        if (needOriginalMatrix)
            matrixAB.CompressBoth();
        matrixAAInvB.CompressBoth();
        vectorAInvB.CompressBoth();
        if (needOriginalMatrix)
            vectorB.CompressBoth();
        matrixAHalf_GG.CompressBoth();

        // standard comm for matrixA
        matrixA.TransAttach();
        matrixA.trans.BorrowGGIndexing(mesh->cell2cell.trans);
        matrixA.trans.createMPITypes();
        matrixA.trans.pullOnce();

        // Get Secondary matrices
        this->MakePairDefaultOnFace(matrixSecondary, maxNDOF - 1, (maxNDOF - 1) * 2);
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            index iCellL = mesh->face2cell(iFace, 0);
            index iCellR = mesh->face2cell(iFace, 1);
            if (iCellR == UnInitIndex) // only for faces with two
                continue;
            // get dbv from 1st derivative to last
            MatrixXR DiffI = this->GetIntPointDiffBaseValue(iCellL, iFace, 0, -1, Eigen::seq(1, EigenLast));
            MatrixXR DiffJ = this->GetIntPointDiffBaseValue(iCellR, iFace, 1, -1, Eigen::seq(1, EigenLast));
            MatrixXR M2_L2R, M2_R2L;
            HardEigen::EigenLeastSquareSolve(DiffI, DiffJ, M2_R2L);
            HardEigen::EigenLeastSquareSolve(DiffJ, DiffI, M2_L2R);
            // TODO: cleanse M2s' lower triangle
            int maxNDOFM1 = matrixSecondary[iFace].cols() / 2;
            matrixSecondary[iFace](
                EigenAll, Eigen::seq(
                              0 * maxNDOFM1 + 0,
                              0 * maxNDOFM1 + maxNDOFM1 - 1)) = M2_R2L;
            matrixSecondary[iFace](
                EigenAll, Eigen::seq(
                              1 * maxNDOFM1 + 0,
                              1 * maxNDOFM1 + maxNDOFM1 - 1)) = M2_L2R;
            // std::cout << "DiffI\n"
            //           << DiffI << std::endl;
            // std::cout << "DiffJ\n"
            //           << DiffJ << std::endl;

            // std::cout << "M2_R2L\n";
            // std::cout << this->GetMatrixSecondary(-1, iFace, 0) << std::endl;
            // std::cout << "M2_L2R\n";
            // std::cout << this->GetMatrixSecondary(-1, iFace, 1) << std::endl;
            // std::abort();
        }
        matrixSecondary.CompressBoth();

        real maxCondAll = 0;
        MPI::Allreduce(&maxCond, &maxCondAll, 1, DNDS_MPI_REAL, MPI_MAX, mesh->getMPI().comm);
        if (mesh->getMPI().rank == 0)
            log() << std::scientific << std::setprecision(3)
                  << "VariationalReconstruction<dim>::ConstructRecCoeff() === A cond Max: [ " << maxCondAll << "] " << std::endl;
    }

    template void
    VariationalReconstruction<2>::ConstructRecCoeff();

    template void
    VariationalReconstruction<3>::ConstructRecCoeff();
}