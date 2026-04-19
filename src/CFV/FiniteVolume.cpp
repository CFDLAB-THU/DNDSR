#include "FiniteVolume.hpp"

#include "Geom/Quadrature.hpp"

namespace DNDS::CFV
{
    void FiniteVolume::SetCellAtrBasic()
    {
        this->MakePairDefaultOnCell(cellAtr, "FV::cellAtr");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            cellAtr(iCell, 0).intOrder = settings.intOrder;
            cellAtr(iCell, 0).Order = settings.maxOrder;
            cellAtr(iCell, 0).NDIFF = this->maxNDOF();
            cellAtr(iCell, 0).NDOF = this->maxNDOF();
        }
    }

    void FiniteVolume::ConstructCellVolume()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        this->MakePairDefaultOnCell(volumeLocal, "FV::volumeLocal");

        sumVolume = 0.0;
        minVolume = veryLargeReal;
        maxVolume = -veryLargeReal;

#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            auto eCell = mesh->GetCellElement(iCell);
            auto qCellMax = Quadrature{eCell, INT_ORDER_MAX};
            tSmallCoords coordsCell;
            mesh->GetCoordsOnCell(iCell, coordsCell);
            //****** Get Int Point Det and Vol
            real v{0};
            { // using more accurate sum volume
                real v1{0};
                qCellMax.Integration(
                    v1,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsCell, DiNj);
                        real JDet = CellJacobianDet(getDim(), coordsCell, DiNj) * (axisSymmetric ? pPhy(1) : 1.);
                        vInc = 1 * JDet;
                    });
                v = v1;
            }
            if (!settings.ignoreMeshGeometryDeficiency)
            {
                DNDS_assert_info(v >= 0, fmt::format("cell has ill area result, v = {}, cellType {}", v, int(eCell.type)));
            }
            else
            {
                if (v > 0)
                    ; // good
                else
                { // v = std::max(0.0, v);
                    v = std::abs(v);
                }
            }
            v += verySmallReal;
            volumeLocal[iCell][0] = v;

            if (iCell < mesh->NumCell()) // non-ghost
#ifdef DNDS_USE_OMP
#    pragma omp critical
#endif
            {
                sumVolume += v;
                minVolume = std::min(minVolume, v);
                maxVolume = std::max(maxVolume, v);
            }
        }
        MPI::AllreduceOneReal(sumVolume, MPI_SUM, mpi);
        MPI::AllreduceOneReal(minVolume, MPI_MIN, mpi);
        MPI::AllreduceOneReal(maxVolume, MPI_MAX, mpi);
        volGlobal = sumVolume;
    }

    void FiniteVolume::ConstructCellIntJacobiDet()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        DNDS_assert_info(volumeLocal.father->Size() == mesh->NumCell(), "need to do ConstructCellVolume() first");
        DNDS_assert_info(cellAtr.father->Size() == mesh->NumCell(), "need to do SetCellAtrBasic() first");

        this->MakePairDefaultOnCell(cellIntJacobiDet, "FV::cellIntJacobiDet");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            auto eCell = mesh->GetCellElement(iCell);
            auto qCell = Quadrature{eCell, GetCellAtr(iCell).intOrder};
            cellIntJacobiDet.ResizeRow(iCell, qCell.GetNumPoints());

            tSmallCoords coordsCell;
            mesh->GetCoordsOnCell(iCell, coordsCell);
            //****** Get Int Point Det and Vol
            real v{0};
            qCell.Integration(
                v,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsCell, DiNj);
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsCell, DiNj);
                    real JDet = CellJacobianDet(getDim(), coordsCell, DiNj);
                    // JDet = std::abs(JDet); // use this to pass check even with bad mesh
                    if (axisSymmetric)
                        JDet *= std::max(verySmallReal, pPhy(1));
                    vInc = 1 * JDet;
                    cellIntJacobiDet(iCell, iG) = JDet;
                });
            v = GetCellVol(iCell);
            if (!settings.ignoreMeshGeometryDeficiency)
            {
                for (int iG = 0; iG < qCell.GetNumPoints(); iG++)
                    DNDS_assert_info(
                        cellIntJacobiDet(iCell, iG) / (v + verySmallReal) >= 0,
                        fmt::format("cell has ill jacobi det, det/V {}, cellType {}", cellIntJacobiDet(iCell, iG) / v, int(eCell.type)));
            }
            else
            {
                if ((cellIntJacobiDet[iCell].array() > 0.0).all())
                    ; // good
                else
                { // v = std::max(0.0, v);
                    cellIntJacobiDet[iCell].setConstant(GetCellVol(iCell) / GetCellParamVol(iCell));
                }
            }
            for (int iG = 0; iG < qCell.GetNumPoints(); iG++)
                cellIntJacobiDet(iCell, iG) += verySmallReal;
        }
        cellIntJacobiDet.CompressBoth();
    }

    void FiniteVolume::ConstructCellIntPPhysics()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        DNDS_assert_info(volumeLocal.father->Size() == mesh->NumCell(), "need to do ConstructCellVolume() first");
        DNDS_assert_info(cellAtr.father->Size() == mesh->NumCell(), "need to do SetCellAtrBasic() first");

        this->MakePairDefaultOnCell(cellIntPPhysics, "FV::cellIntPPhysics");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            auto eCell = mesh->GetCellElement(iCell);
            auto qCell = Quadrature{eCell, GetCellAtr(iCell).intOrder};
            cellIntPPhysics.ResizeRow(iCell, qCell.GetNumPoints());

            tSmallCoords coordsCell;
            mesh->GetCoordsOnCell(iCell, coordsCell);
            //****** Get Int Point Det and PPhy
            real v{0};
            qCell.Integration(
                v,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsCell, DiNj);
                    cellIntPPhysics(iCell, iG) = pPhy;
                });
        }
        cellIntPPhysics.CompressBoth();
    }

    void FiniteVolume::ConstructCellBary()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        DNDS_assert_info(volumeLocal.father->Size() == mesh->NumCell(), "need to do ConstructCellVolume() first");
        DNDS_assert_info(cellAtr.father->Size() == mesh->NumCell(), "need to do SetCellAtrBasic() first");

        this->MakePairDefaultOnCell(cellBary, "FV::cellBary");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            auto eCell = mesh->GetCellElement(iCell);
            auto qCell = Quadrature{eCell, GetCellAtr(iCell).intOrder};
            auto qCellMax = Quadrature{eCell, INT_ORDER_MAX};

            tSmallCoords coordsCell;
            mesh->GetCoordsOnCell(iCell, coordsCell);
            //****** Get Bary
            real v{0};
            tPoint b{0, 0, 0};
            qCell.Integration(
                b,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsCell, DiNj);
                    real JDet = CellJacobianDet(getDim(), coordsCell, DiNj);
                    // JDet = std::abs(JDet); // use this to pass check even with bad mesh
                    if (axisSymmetric)
                        JDet *= std::max(verySmallReal, pPhy(1));
                    vInc = pPhy * JDet;
                    v += JDet * qCell.GetWeight(iG);
                });
            v = GetCellVol(iCell); //! FIX: is this good choice?
            cellBary[iCell] = b / v;
        }
    }

    void FiniteVolume::ConstructCellCent()
    {
        using namespace Geom;
        using namespace Geom::Elem;

        this->MakePairDefaultOnCell(cellCent, "FV::cellCent");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            auto eCell = mesh->GetCellElement(iCell);
            auto qCellO1 = Quadrature{eCell, 1};

            tSmallCoords coordsCell;
            mesh->GetCoordsOnCell(iCell, coordsCell);
            //****** Get Int Point Det and PPhy
            SummationNoOp noOp;
            qCellO1.Integration(
                noOp,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsCell, DiNj);
                    cellCent[iCell] = pPhy;
                });
        }
    }

    void FiniteVolume::ConstructCellAlignedHBox()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        DNDS_assert_info(cellBary.father->Size() == mesh->NumCell(), "need to do ConstructCellBary() first");

        this->MakePairDefaultOnCell(cellAlignedHBox, "FV::cellAlignedHBox");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            auto eCell = mesh->GetCellElement(iCell);
            auto qCell = Quadrature{eCell, GetCellAtr(iCell).intOrder};
            auto qCellMax = Quadrature{eCell, INT_ORDER_MAX};

            tSmallCoords coordsCell;
            mesh->GetCoordsOnCell(iCell, coordsCell);
            //****** Get HBox aligned
            tSmallCoords coordsCellC = coordsCell.colwise() - this->GetCellBary(iCell);
            DNDS_assert(coordsCellC.cols() == coordsCell.cols());
            tPoint hBox = coordsCellC.array().abs().rowwise().maxCoeff();
            if (getDim() == 2)
                hBox(2) = 1;
            cellAlignedHBox[iCell] = hBox;
        }
    }

    void FiniteVolume::ConstructCellMajorHBoxCoordInertia()
    {
        using namespace Geom;
        using namespace Geom::Elem;

        this->MakePairDefaultOnCell(cellMajorHBox, "FV::cellMajorHBox");
        this->MakePairDefaultOnCell(cellMajorCoord, "FV::cellMajorCoord", 3, 3);
        this->MakePairDefaultOnCell(cellInertia, "FV::cellInertia", 3, 3);
        DNDS_assert_info(cellBary.father->Size() == mesh->NumCell(), "need to do ConstructCellBary() first");

#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            auto eCell = mesh->GetCellElement(iCell);
            auto qCellMax = Quadrature{eCell, INT_ORDER_MAX};

            tSmallCoords coordsCell;
            mesh->GetCoordsOnCell(iCell, coordsCell);
            //****** Get Int Point Det and Vol
            real v = GetCellVol(iCell);
            tSmallCoords coordsCellC = coordsCell.colwise() - this->GetCellBary(iCell);
            DNDS_assert(coordsCellC.cols() == coordsCell.cols());

            //****** Get Major Axis
            tJacobi inertia;
            inertia.setZero();
            qCellMax.Integration(
                inertia,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsCell, DiNj);
                    real JDet = CellJacobianDet(getDim(), coordsCell, DiNj) * (axisSymmetric ? 1 /* using geometrical */ : 1.);
                    tPoint pPhyC = (pPhy - GetCellBary(iCell));
                    vInc = (pPhyC * pPhyC.transpose()) * JDet;
                });
            inertia /= this->GetCellVol(iCell);
            real inerNorm = inertia.norm();
            real inerCond = 1;
            if (getDim() == 2)
                inerCond = HardEigen::Eigen2x2RealSymEigenDecompositionGetCond(inertia({0, 1}, {0, 1}));
            else
                inerCond = HardEigen::Eigen3x3RealSymEigenDecompositionGetCond(inertia);

            if (inerCond < 1 + smallReal)
            {
                inertia(0, 0) += inerNorm * smallReal * 10;
                inertia(1, 1) += inerNorm * smallReal;
            }

            cellInertia[iCell] = inertia;
            tJacobi decRet;
            decRet.setIdentity();
            if (getDim() == 3)
                decRet = HardEigen::Eigen3x3RealSymEigenDecompositionNormalized(inertia);
            else
                decRet({0, 1}, {0, 1}) = HardEigen::Eigen2x2RealSymEigenDecompositionNormalized(inertia({0, 1}, {0, 1}));
            cellMajorCoord[iCell] = decRet;
            tSmallCoords coordsCellM = cellMajorCoord[iCell].transpose() * coordsCellC;
            tPoint hBoxM = coordsCellM.array().abs().rowwise().maxCoeff();
            if (getDim() == 2)
                hBoxM(2) = 1;
            cellMajorHBox[iCell] = hBoxM;
        }
    }

    void FiniteVolume::SetFaceAtrBasic()
    {
        this->MakePairDefaultOnFace(faceAtr, "FV::faceAtr");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            faceAtr(iFace, 0).intOrder = settings.intOrder;
            // faceAtr(iCell, 0).Order = settings.maxOrder;
            faceAtr(iFace, 0).NDIFF = this->maxNDOF();
            faceAtr(iFace, 0).NDOF = 0;
        }
    }

    void FiniteVolume::ConstructFaceArea()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        DNDS_assert_info(faceAtr.father->Size() == mesh->NumFace(), "need to do SetFaceAtrBasic() first");

        this->MakePairDefaultOnFace(faceArea, "FV::faceArea");
        axisFaces.clear();
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto eFace = mesh->GetFaceElement(iFace);
            auto qFace = Quadrature{eFace, GetFaceAtr(iFace).intOrder};
            auto qFaceO1 = Quadrature{eFace, 1};
            DNDS_assert(qFaceO1.GetNumPoints() == 1);

            tSmallCoords coords;
            mesh->GetCoordsOnFace(iFace, coords);

            //****** Get Int Point Det and Vol
            real v{0};
            qFace.Integration(
                v,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coords, DiNj);
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                    real JDet = FaceJacobianDet(getDim(), coords, DiNj);
                    if (axisSymmetric)
                        JDet *= std::max(verySmallReal, pPhy(1));
                    vInc = 1 * JDet;
                });
            v += verySmallReal;
            if (axisSymmetric) // could this be helpful out of axisSymmetric context?
            {
                if (v < std::pow(this->GetCellVol(mesh->face2cell[iFace][0]), (real(getDim()) - 1) / real(getDim())) * smallReal)
                    axisFaces.insert(iFace);
            }
            faceArea[iFace][0] = v;
            DNDS_assert_info(v > 0, "face has ill area result");
        }
    }

    void FiniteVolume::ConstructFaceCent()
    {
        using namespace Geom;
        using namespace Geom::Elem;

        this->MakePairDefaultOnFace(faceCent, "FV::faceCent");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto eFace = mesh->GetFaceElement(iFace);
            auto qFaceO1 = Quadrature{eFace, 1};
            DNDS_assert(qFaceO1.GetNumPoints() == 1);

            tSmallCoords coords;
            mesh->GetCoordsOnFace(iFace, coords);

            //****** Get Center
            SummationNoOp noOp;
            qFaceO1.Integration(
                noOp,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                    faceCent[iFace] = pPhy;
                });
        }
    }

    void FiniteVolume::ConstructFaceIntJacobiDet()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        DNDS_assert_info(faceAtr.father->Size() == mesh->NumFace(), "need to do SetFaceAtrBasic() first");
        // DNDS_assert_info(faceArea.father->Size() == mesh->NumFace(), "need to do ConstructFaceArea() first");

        this->MakePairDefaultOnFace(faceIntJacobiDet, "FV::faceIntJacobiDet");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto eFace = mesh->GetFaceElement(iFace);
            auto qFace = Quadrature{eFace, GetFaceAtr(iFace).intOrder};
            auto qFaceO1 = Quadrature{eFace, 1};
            DNDS_assert(qFaceO1.GetNumPoints() == 1);
            faceIntJacobiDet.ResizeRow(iFace, qFace.GetNumPoints());
            tSmallCoords coords;
            mesh->GetCoordsOnFace(iFace, coords);
            //****** Get Int Point Det and Vol
            real v{0};
            qFace.Integration(
                v,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coords, DiNj);
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                    real JDet = FaceJacobianDet(getDim(), coords, DiNj);
                    if (axisSymmetric)
                        JDet *= std::max(verySmallReal, pPhy(1));
                    vInc = 1 * JDet;
                    faceIntJacobiDet(iFace, iG) = JDet;
                });
            // v = GetFaceArea(iFace);
            for (int iG = 0; iG < qFace.GetNumPoints(); iG++)
                faceIntJacobiDet(iFace, iG) += verySmallReal;
            for (int iG = 0; iG < qFace.GetNumPoints(); iG++)
                DNDS_assert_info(faceIntJacobiDet(iFace, iG) / v > 1e-10, "face has ill jacobi det");
        }
        faceIntJacobiDet.CompressBoth();
    }

    void FiniteVolume::ConstructFaceIntPPhysics()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        DNDS_assert_info(faceAtr.father->Size() == mesh->NumFace(), "need to do SetFaceAtrBasic() first");

        this->MakePairDefaultOnFace(faceIntPPhysics, "FV::faceIntPPhysics");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto eFace = mesh->GetFaceElement(iFace);
            auto qFace = Quadrature{eFace, GetFaceAtr(iFace).intOrder};
            faceIntPPhysics.ResizeRow(iFace, qFace.GetNumPoints());
            tSmallCoords coords;
            mesh->GetCoordsOnFace(iFace, coords);
            //****** Get Int Point Det and Vol
            real v{0};
            qFace.Integration(
                v,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                    faceIntPPhysics(iFace, iG) = pPhy;
                });
        }
        faceIntPPhysics.CompressBoth();
    }

    void FiniteVolume::ConstructFaceUnitNorm()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        DNDS_assert_info(faceAtr.father->Size() == mesh->NumFace(), "need to do SetFaceAtrBasic() first");
        // DNDS_assert_info(faceIntJacobiDet.father->Size() == mesh->NumFace(), "need to do ConstructFaceIntJacobiDet() first");

        this->MakePairDefaultOnFace(faceUnitNorm, "FV::faceUnitNorm");
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto eFace = mesh->GetFaceElement(iFace);
            auto qFace = Quadrature{eFace, GetFaceAtr(iFace).intOrder};
            faceUnitNorm.ResizeRow(iFace, qFace.GetNumPoints());
            tSmallCoords coords;
            mesh->GetCoordsOnFace(iFace, coords);
            //****** Get Int Point Norm/pPhy and Mean Norm
            tPoint n{0, 0, 0};
            qFace.Integration(
                n,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coords, DiNj);
                    tPoint np;
                    if (getDim() == 2)
                        np = FacialJacobianToNormVec<2>(J);
                    else
                        np = FacialJacobianToNormVec<3>(J);
                    np.stableNormalize();
                    faceUnitNorm(iFace, iG) = np;
                    // vInc = np * GetFaceJacobiDet(iFace, iG);
                });
        }
        faceUnitNorm.CompressBoth();
    }

    void FiniteVolume::ConstructFaceMeanNorm()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        DNDS_assert_info(faceAtr.father->Size() == mesh->NumFace(), "need to do SetFaceAtrBasic() first");
        DNDS_assert_info(faceCent.father->Size() == mesh->NumFace(), "need to do ConstructFaceCent() first");
        DNDS_assert_info(cellCent.father->Size() == mesh->NumCell(), "need to do ConstructCellCent() first");

        this->MakePairDefaultOnFace(faceMeanNorm, "FV::faceMeanNorm");
        axisFaces.clear();
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto eFace = mesh->GetFaceElement(iFace);
            auto qFace = Quadrature{eFace, GetFaceAtr(iFace).intOrder};
            tSmallCoords coords;
            mesh->GetCoordsOnFace(iFace, coords);

            //****** Get Int Point Norm/pPhy and Mean Norm
            tPoint n{0, 0, 0};
            qFace.Integration(
                n,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coords, DiNj);
                    tPoint np;
                    if (getDim() == 2)
                        np = FacialJacobianToNormVec<2>(J);
                    else
                        np = FacialJacobianToNormVec<3>(J);
                    np.stableNormalize();
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                    real JDet = FaceJacobianDet(getDim(), coords, DiNj);
                    if (axisSymmetric)
                        JDet *= std::max(verySmallReal, pPhy(1));
                    vInc = np * JDet;
                });
            // faceMeanNorm[iFace] = n / v;
            faceMeanNorm[iFace] = n;
            faceMeanNorm[iFace].stableNormalize(); // might not be unit if is on axis face for axisSymmetric

            /// ! if the faces and f2c are created right, and not distorting too much
            if (!settings.ignoreMeshGeometryDeficiency)
                DNDS_assert_info(
                    (this->GetFaceQuadraturePPhysFromCell(iFace, mesh->face2cell(iFace, 0), -1, -1) -
                     this->GetCellQuadraturePPhys(mesh->face2cell(iFace, 0), -1))
                            .dot(faceMeanNorm[iFace]) >= 0,
                    "face mean norm is not the same side as faceCenter - cellCenter");
        }
    }

    void FiniteVolume::ConstructCellSmoothScale()
    {
        DNDS_assert_info(volumeLocal.father->Size() == mesh->NumCell(), "need to do ConstructCellVolume() first");
        DNDS_assert_info(faceArea.father->Size() == mesh->NumFace(), "need to do ConstructFaceArea() first");
        
        this->MakePairDefaultOnCell(cellSmoothScale, "FV::cellSmoothScale");
        cellSmoothScale.TransAttach();
        cellSmoothScale.trans.BorrowGGIndexing(mesh->cell2cell.trans);
        cellSmoothScale.trans.createMPITypes();
        cellSmoothScale.trans.initPersistentPull();
        std::vector<real> cellScale(mesh->NumCellProc());
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            real faceSum{0};
            for (auto iFace : mesh->cell2face[iCell])
                faceSum += this->GetFaceArea(iFace);
            cellScale[iCell] = this->GetCellVol(iCell) / (faceSum + verySmallReal);
            cellSmoothScale(iCell, 0) = cellScale[iCell];
        }
        std::vector<real> cellScaleNew = cellScale;
        for (int iter = 1; iter <= settings.nIterCellSmoothScale; iter++)
        {
            cellSmoothScale.trans.startPersistentPull();
            cellSmoothScale.trans.waitPersistentPull();
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                real nAdj{0}, sum{0};
                for (index iFace : mesh->cell2face[iCell])
                {
                    index iCellOther = this->CellFaceOther(iCell, iFace);
                    if (iCellOther != UnInitIndex)
                    {
                        nAdj += 1.;
                        sum += cellSmoothScale(iCellOther, 0);
                    }
                }
                real smootherCentWeight = 1;
                sum += nAdj * smootherCentWeight * cellSmoothScale(iCell, 0);
                sum /= nAdj * (1 + smootherCentWeight);
                cellScaleNew[iCell] = std::min(cellSmoothScale(iCell, 0), sum);
            }
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                cellSmoothScale(iCell, 0) = cellScaleNew[iCell];
        }
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            cellSmoothScale(iCell, 0) /= cellScale[iCell];
        cellSmoothScale.trans.startPersistentPull();
        cellSmoothScale.trans.waitPersistentPull();

        real minCellSmoothScale{veryLargeReal};
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            minCellSmoothScale = std::min(cellSmoothScale(iCell, 0), minCellSmoothScale);
        MPI::AllreduceOneReal(minCellSmoothScale, MPI_MIN, mpi);

        if (mpi.rank == mRank)
            log() << fmt::format("=== cellSmoothScale min [{:.5g}]", minCellSmoothScale)
                  << std::endl;
    }
}