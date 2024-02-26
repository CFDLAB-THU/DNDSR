#pragma once
#include "EulerEvaluator.hpp"
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

namespace DNDS::Euler
{

    template <EulerModel model>
    void EulerEvaluator<model>::GetWallDist()
    {
        using TriArray = ArrayEigenMatrix<3, 3>;
        ssp<TriArray> TrianglesLocal, TrianglesFull;
        DNDS_MAKE_SSP(TrianglesLocal, mesh->getMPI());
        DNDS_MAKE_SSP(TrianglesFull, mesh->getMPI());
        std::vector<Eigen::Matrix<real, 3, 3>> Triangles;
        for (index iBnd = 0; iBnd < mesh->NumBnd(); iBnd++)
        {
            if (pBCHandler->GetTypeFromID(mesh->GetBndZone(iBnd)) == EulerBCType::BCWall)
            {
                index iFace = mesh->bnd2face[iBnd];
                auto elem = mesh->GetFaceElement(iFace);
                if (elem.type == Geom::Elem::ElemType::Line2 || elem.type == Geom::Elem::ElemType::Line3) //!
                {
                    Geom::tSmallCoords coords;
                    mesh->GetCoordsOnFace(iFace, coords);
                    Eigen::Matrix<real, 3, 3> tri;
                    mesh->GetCoordsOnFace(iFace, coords);
                    tri(Eigen::all, 0) = coords(Eigen::all, 0);
                    tri(Eigen::all, 1) = coords(Eigen::all, 1);
                    tri(Eigen::all, 2) = coords(Eigen::all, 1) + Geom::tPoint{0., 0., vfv->GetFaceArea(iFace)};
                    Triangles.push_back(tri);
                }
                else if (elem.type == Geom::Elem::ElemType::Tri3 || elem.type == Geom::Elem::ElemType::Tri6) //! TODO
                {
                    Geom::tSmallCoords coords;
                    mesh->GetCoordsOnFace(iFace, coords);
                    Eigen::Matrix<real, 3, 3> tri;
                    tri(Eigen::all, 0) = coords(Eigen::all, 0);
                    tri(Eigen::all, 1) = coords(Eigen::all, 1);
                    tri(Eigen::all, 2) = coords(Eigen::all, 2);
                    Triangles.push_back(tri);
                }
                else if (elem.type == Geom::Elem::ElemType::Quad4 || elem.type == Geom::Elem::ElemType::Quad9)
                {
                    Geom::tSmallCoords coords;
                    mesh->GetCoordsOnFace(iFace, coords);
                    Eigen::Matrix<real, 3, 3> tri;
                    tri(Eigen::all, 0) = coords(Eigen::all, 0);
                    tri(Eigen::all, 1) = coords(Eigen::all, 1);
                    tri(Eigen::all, 2) = coords(Eigen::all, 2);
                    Triangles.push_back(tri);
                    tri(Eigen::all, 0) = coords(Eigen::all, 0);
                    tri(Eigen::all, 1) = coords(Eigen::all, 2);
                    tri(Eigen::all, 2) = coords(Eigen::all, 3);
                    Triangles.push_back(tri);
                }
                else
                {
                    DNDS_assert_info(false, "This elem not implemented yet for GetWallDist()");
                }
            }
        }
        TrianglesLocal->Resize(Triangles.size(), 3, 3);
        for (index i = 0; i < TrianglesLocal->Size(); i++)
            (*TrianglesLocal)[i] = Triangles[i];
        Triangles.clear();
        ArrayTransformerType<TriArray>::Type TrianglesTransformer;
        TrianglesTransformer.setFatherSon(TrianglesLocal, TrianglesFull);
        TrianglesTransformer.createFatherGlobalMapping();

        std::vector<index> pullingSet;
        pullingSet.resize(TrianglesTransformer.pLGlobalMapping->globalSize());
        for (index i = 0; i < pullingSet.size(); i++)
            pullingSet[i] = i;
        TrianglesTransformer.createGhostMapping(pullingSet);
        TrianglesTransformer.createMPITypes();
        TrianglesTransformer.pullOnce();
        if (mesh->coords.father->getMPI().rank == 0)
            log() << fmt::format("=== EulerEvaluator<model>::GetWallDist() with minWallDist = {:.4e}, ", settings.minWallDist)
                  << " To search in " << TrianglesFull->Size() << std::endl;

        typedef CGAL::Simple_cartesian<double> K;
        typedef K::FT FT;
        // typedef K::Ray_3 Ray;
        // typedef K::Line_3 Line;
        typedef K::Point_3 Point;
        typedef K::Triangle_3 Triangle;
        typedef std::vector<Triangle>::iterator Iterator;
        typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
        typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
        typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

        std::vector<Triangle> triangles;
        triangles.reserve(TrianglesFull->Size());

        for (index i = 0; i < TrianglesFull->Size(); i++)
        {
            Point p0((*TrianglesFull)[i](0, 0), (*TrianglesFull)[i](1, 0), (*TrianglesFull)[i](2, 0));
            Point p1((*TrianglesFull)[i](0, 1), (*TrianglesFull)[i](1, 1), (*TrianglesFull)[i](2, 1));
            Point p2((*TrianglesFull)[i](0, 2), (*TrianglesFull)[i](1, 2), (*TrianglesFull)[i](2, 2));
            triangles.push_back(Triangle(p0, p1, p2));
        }
        TrianglesLocal->Resize(0, 3, 3);
        TrianglesFull->Resize(0, 3, 3);
        double minDist = veryLargeReal;
        this->dWall.resize(mesh->NumCellProc());

        if (!triangles.empty())
        {
            // std::cout << "tree building" << std::endl;
            Tree tree(triangles.begin(), triangles.end());

            // std::cout << "tree built" << std::endl;
            // search

            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
            {
                // std::cout << "iCell " << iCell << std::endl;
                auto quadCell = vfv->GetCellQuad(iCell);
                dWall[iCell].resize(quadCell.GetNumPoints());
                for (int ig = 0; ig < quadCell.GetNumPoints(); ig++)
                {
                    // std::cout << "iG " << ig << std::endl;
                    auto p = vfv->GetCellQuadraturePPhys(iCell, ig);
                    Point pQ(p[0], p[1], p[2]);
                    // std::cout << "pQ " << pQ << std::endl;
                    // Point closest_point = tree.closest_point(pQ);
                    FT sqd = tree.squared_distance(pQ);
                    // std::cout << "sqd" << sqd << std::endl;
                    dWall[iCell][ig] = std::sqrt(sqd);
                    // dWall[iCell][ig] = p(0) < 0 ? p({0, 1}).norm() : p(1); // test for plate BL
                    if (dWall[iCell][ig] < minDist)
                        minDist = dWall[iCell][ig];
                    dWall[iCell][ig] = std::max(settings.minWallDist, dWall[iCell][ig]);
                }
            }
        }
        else
        {
            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
            {
                // std::cout << "iCell " << iCell << std::endl;
                auto quadCell = vfv->GetCellQuad(iCell);
                dWall[iCell].setConstant(quadCell.GetNumPoints(), std::pow(veryLargeReal, 1. / 4.));
            }
        }
        std::cout << "MinDist: " << minDist << std::endl;
    }

    // Eigen::Vector<real, -1> EulerEvaluator::CompressRecPart(
    //     const Eigen::Vector<real, -1> &umean,
    //     const Eigen::Vector<real, -1> &uRecInc)1

    //! evaluates dt and facial spectral radius
    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateDt(
        ArrayDOFV<1> &dt,
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        real CFL, real &dtMinall, real MaxDt,
        bool UseLocaldt)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "EvaluateDt 1");
        for (auto &i : lambdaCell)
            i = 0.0;

        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto f2c = mesh->face2cell[iFace];
            TVec unitNorm = vfv->GetFaceNorm(iFace, -1)(Seq012);

            index iCellL = f2c[0];
            TU UL = u[iCellL];
            this->UFromCell2Face(UL, iFace, f2c[0], 0);
            TU uMean = UL;
            real pL, asqrL, HL, pR, asqrR, HR;
            TVec vL = UL(Seq123) / UL(0);
            TVec vR = vL;
            Gas::IdealGasThermal(UL(I4), UL(0), vL.squaredNorm(),
                                 settings.idealGasProperty.gamma,
                                 pL, asqrL, HL);
            pR = pL, HR = HL, asqrR = asqrL;
            if (f2c[1] != UnInitIndex)
            {
                TU UR = u[f2c[1]];
                this->UFromCell2Face(UR, iFace, f2c[1], 1);
                uMean = (uMean + UR) * 0.5;
                vR = UR(Seq123) / UR(0);
                Gas::IdealGasThermal(UR(I4), UR(0), vR.squaredNorm(),
                                     settings.idealGasProperty.gamma,
                                     pR, asqrR, HR);
            }
            TDiffU GradULxy, GradURxy;
            GradULxy.resize(Eigen::NoChange, nVars);
            GradURxy.resize(Eigen::NoChange, nVars);
            GradULxy.setZero(), GradURxy.setZero();
            if constexpr (gDim == 2)
                GradULxy({0, 1}, Eigen::all) =
                    vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, -1, std::array<int, 2>{1, 2}, 3) *
                    uRec[f2c[0]]; // 2d here
            else
                GradULxy({0, 1, 2}, Eigen::all) =
                    vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, -1, std::array<int, 3>{1, 2, 3}, 4) *
                    uRec[f2c[0]]; // 3d here
            this->DiffUFromCell2Face(GradULxy, iFace, f2c[0], 0);
            GradURxy = GradULxy;
            if (f2c[1] != UnInitIndex)
            {
                if constexpr (gDim == 2)
                    GradURxy({0, 1}, Eigen::all) =
                        vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, -1, std::array<int, 2>{1, 2}, 3) *
                        uRec[f2c[1]]; // 2d here
                else
                    GradURxy({0, 1, 2}, Eigen::all) =
                        vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, -1, std::array<int, 3>{1, 2, 3}, 4) *
                        uRec[f2c[1]]; // 3d here
                this->DiffUFromCell2Face(GradURxy, iFace, f2c[1], 1);
            }
            TDiffU GradUMeanXY = (GradURxy + GradULxy) / 2;

            DNDS_assert(uMean(0) > 0);
            TVec veloMean = (uMean(Seq123).array() / uMean(0)).matrix();
            // real veloNMean = veloMean.dot(unitNorm); // original
            real veloNMean = 0.5 * (vL + vR).dot(unitNorm); // paper
            real veloNL = vL.dot(unitNorm);
            real veloNR = vR.dot(unitNorm);

            // real ekFixRatio = 0.001;
            // Eigen::Vector3d velo = uMean({1, 2, 3}) / uMean(0);
            // real vsqr = velo.squaredNorm();
            // real Ek = vsqr * 0.5 * uMean(0);
            // real Efix = Ek * ekFixRatio;
            // real e = uMean(4) - Ek;
            // if (e < 0)
            //     e = 0.5 * Efix;
            // else if (e < Efix)
            //     e = (e * e + Efix * Efix) / (2 * Efix);
            // uMean(4) = Ek + e;

            real pMean, asqrMean, HMean;
            Gas::IdealGasThermal(uMean(I4), uMean(0), veloMean.squaredNorm(),
                                 settings.idealGasProperty.gamma,
                                 pMean, asqrMean, HMean);

            pMean = (pL + pR) * 0.5;
            real aMean = sqrt(settings.idealGasProperty.gamma * pMean / uMean(0)); // paper

            // DNDS_assert(asqrMean >= 0);
            // real aMean = std::sqrt(asqrMean); // original
            real lambdaConvection = std::abs(veloNMean) + aMean;
            lambdaConvection = std::max(std::sqrt(asqrL) + std::abs(veloNL), std::sqrt(asqrR) + std::abs(veloNR));
            DNDS_assert_info(
                asqrL >= 0 && asqrR >= 0,
                fmt::format(" mean value violates PP! asqr: [{} {}]", asqrL, asqrR));

            // ! refvalue:
            real muRef = settings.idealGasProperty.muGas;

            real gamma = settings.idealGasProperty.gamma;
            real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * uMean(0));
            real muf = muEff(uMean, T);
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                real cnu1 = 7.1;
                real Chi = uMean(I4 + 1) * muRef / muf;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (Chi < 10)
                    Chi = 0.05 * std::log(1 + std::exp(20 * Chi));
#endif
                real Chi3 = std::pow(Chi, 3);
                real fnu1 = Chi3 / (Chi3 + std::pow(cnu1, 3));
                muf *= std::max((1 + Chi * fnu1), 1.0);
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                real mut = RANS::GetMut_RealizableKe<dim>(uMean, GradUMeanXY, muf);
                muf = muf + mut;
            }
            real lamVis = muf / uMean(0) *
                          std::max(4. / 3., gamma / settings.idealGasProperty.prGas);

            real lamFace = lambdaConvection * vfv->GetFaceArea(iFace);

            real area = vfv->GetFaceArea(iFace);
            real areaSqr = area * area;
            real volR = vfv->GetCellVol(iCellL);
            // lambdaCell[iCellL] += lamFace + 2 * lamVis * areaSqr / fv->GetCellVol(iCellL);
            if (f2c[1] != UnInitIndex) // can't be non local
                                       // lambdaCell[f2c[1]] += lamFace + 2 * lamVis * areaSqr / fv->volumeLocal[f2c[1]],
                volR = vfv->GetCellVol(f2c[1]);

            lambdaFace[iFace] = lambdaConvection + lamVis * area * (1. / vfv->GetCellVol(iCellL) + 1. / volR);
            lambdaFaceC[iFace] = std::abs(veloNMean) + lamVis * area * (1. / vfv->GetCellVol(iCellL) + 1. / volR); // passive part
            lambdaFaceVis[iFace] = lamVis * area * (1. / vfv->GetCellVol(iCellL) + 1. / volR);

            // if (f2c[0] == 10756)
            // {
            //     std::cout << "----Lambdas" << std::setprecision(16) << iFace << std::endl;
            //     std::cout << lambdaConvection << std::endl;
            //     std::cout << lambdaFaceVis[iFace] << std::endl;
            //     std::cout << veloNMean << " " << aMean << std::endl;
            //     std::cout << gamma << " " << pMean << " " << uMean(0) << std::endl;
            // }

            lambdaCell[iCellL] += lambdaFace[iFace] * vfv->GetFaceArea(iFace);
            if (f2c[1] != UnInitIndex) // can't be non local
                lambdaCell[f2c[1]] += lambdaFace[iFace] * vfv->GetFaceArea(iFace);

            deltaLambdaFace[iFace] = std::abs((vR - vL).dot(unitNorm)) + std::sqrt(std::abs(asqrR - asqrL)) * 0.7071;
        }
        real dtMin = veryLargeReal;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            // std::cout << fv->GetCellVol(iCell) << " " << (lambdaCell[iCell]) << " " << CFL << std::endl;
            // exit(0);
            dt[iCell](0) = std::min(CFL * vfv->GetCellVol(iCell) / (lambdaCell[iCell] + 1e-100), MaxDt);
            dtMin = std::min(dtMin, dt[iCell](0));
            // if (iCell == 10756)
            // {
            //     std::cout << std::endl;
            // }
        }

        MPI::Allreduce(&dtMin, &dtMinall, 1, DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);

        // if (uRec.father->getMPI().rank == 0)
        //     std::cout << "dt min is " << dtMinall << std::endl;
        if (!UseLocaldt)
        {
            dt.setConstant(dtMinall);
        }
        // if (uRec.father->getMPI().rank == 0)
        // log() << "dt: " << dtMin << std::endl;
    }

    template <EulerModel model>
    typename EulerEvaluator<model>::TU EulerEvaluator<model>::fluxFace(
        const TU &ULxy,
        const TU &URxy,
        const TU &ULMeanXy,
        const TU &URMeanXy,
        const TDiffU &DiffUxy,
        const TVec &unitNorm,
        const TMat &normBase,
        TU &FLfix,
        TU &FRfix,
        Geom::t_index btype,
        typename Gas::RiemannSolverType rsType,
        index iFace, int ig)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS

        TU UR = URxy;
        TU UL = ULxy;
        UR(Seq123) = normBase(Seq012, Seq012).transpose() * UR(Seq123);
        UL(Seq123) = normBase(Seq012, Seq012).transpose() * UL(Seq123);
        TU ULMean = ULMeanXy;
        TU URMean = URMeanXy;
        ULMean(Seq123) = normBase(Seq012, Seq012).transpose() * ULMean(Seq123);
        URMean(Seq123) = normBase(Seq012, Seq012).transpose() * URMean(Seq123);
        // if (btype == BoundaryType::Wall_NoSlip)
        //     UR(Seq123) = -UL(Seq123);
        // if (btype == BoundaryType::Wall_Euler)
        //     UR(1) = -UL(1);

        TU UMeanXy = 0.5 * (ULxy + URxy);

        real pMean, asqrMean, Hmean;
        real gamma = settings.idealGasProperty.gamma;
        Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                             gamma, pMean, asqrMean, Hmean);

        // ! refvalue:
        // PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterB);
        real muRef = settings.idealGasProperty.muGas;
        real TRef = settings.idealGasProperty.TRef;

        real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
        real CSut = settings.idealGasProperty.CSutherland;
        real mufPhy, muf;
        // mufPhy = muf = settings.idealGasProperty.muGas *
        //                std::pow(T / settings.idealGasProperty.TRef, 1.5) *
        //                (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
        //                (T + settings.idealGasProperty.CSutherland);
        // muf = muRef * (T / TRef) * std::sqrt(T / TRef) * (TRef + CSut) / (T + CSut); // this is much faster on bssct?? at the start for vorstreet case //! ??
        // muf = muRef * std::pow(T / TRef, 1.5) * (TRef + CSut) / (T + CSut);
        // muf = muRef * pow(T / TRef, 1.5) * (TRef + CSut) / (T + CSut);
        muf = muEff(UMeanXy, T);
        mufPhy = muf;
        // PerformanceTimer::Instance().StopTimer(PerformanceTimer::LimiterB);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
        real fnu1 = 0.;
        if constexpr (model == NS_SA || model == NS_SA_3D)
        {
            real cnu1 = 7.1;
            real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
#ifdef USE_NS_SA_NEGATIVE_MODEL
            if (Chi < 10) //*negative fix
                Chi = 0.05 * std::log(1 + std::exp(20 * Chi));
#endif
            real Chi3 = std::pow(Chi, 3);
            fnu1 = Chi3 / (Chi3 + std::pow(cnu1, 3));
            muf *= std::max((1 + Chi * fnu1), 1.0);
        }
        if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
        {
            real mut = RANS::GetMut_RealizableKe<dim>(UMeanXy, DiffUxy, muf);
            muf = muf + mut;
        }

        real k = settings.idealGasProperty.CpGas * (muf - mufPhy) / 0.9 +
                 settings.idealGasProperty.CpGas * mufPhy / settings.idealGasProperty.prGas;
        TDiffU VisFlux;
        VisFlux.resizeLike(DiffUxy);
        VisFlux.setZero();
        Gas::ViscousFlux_IdealGas<dim>(
            UMeanXy, DiffUxy, unitNorm, pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWall,
            settings.idealGasProperty.gamma,
            muf,
            k,
            settings.idealGasProperty.CpGas,
            VisFlux);

        // if (mesh->face2cellLocal[iFace][0] == 10756)
        // {
        //     std::cout << "Face " << iFace << " " << mesh->face2cellLocal[iFace][1] << std::endl;
        //     std::cout << DiffUxy << std::endl;
        //     std::cout << VisFlux << std::endl;
        //     std::cout << unitNorm << std::endl;
        //     std::cout << unitNorm.transpose() * VisFlux << std::endl;
        //     std::cout << muf << " " << k << std::endl;

        // }
        // if (iFace == 16404)
        // {
        //     std::cout << std::setprecision(10);
        //     std::cout << "Face " << iFace << " " << mesh->face2cellLocal[iFace][0] << " " << mesh->face2cellLocal[iFace][1] << std::endl;
        //     std::cout << DiffUxy << std::endl;
        //     std::cout << VisFlux << std::endl;
        //     std::cout << unitNorm << std::endl;
        //     std::cout << unitNorm.transpose() * VisFlux << std::endl;
        //     std::cout << muf << " " << k << std::endl;
        //     std::cout << lambdaFace[iFace] << std::endl;
        //     exit(-1);

        // }
        if constexpr (model == NS_SA || model == NS_SA_3D)
        {
            real sigma = 2. / 3.;
            Eigen::Matrix<real, dim, 1> diffRhoNu = DiffUxy(Seq012, {I4 + 1}) * muRef;
            Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
            Eigen::Matrix<real, dim, 1> diffNu = (diffRhoNu - UMeanXy(I4 + 1) * muRef / UMeanXy(0) * diffRho) / UMeanXy(0);

            real cn1 = 16;
            real fn = 1;
#ifdef USE_NS_SA_NEGATIVE_MODEL
            if (UMeanXy(I4 + 1) < 0)
            {
                real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
                fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
            }
#endif
            VisFlux(Seq012, {I4 + 1}) = diffNu * (mufPhy + UMeanXy(I4 + 1) * muRef * fn) / sigma / muRef;
        }
        if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
        {
            RANS::GetVisFlux_RealizableKe<dim>(UMeanXy, DiffUxy, muf - mufPhy, mufPhy, VisFlux);
        }
#endif

        {
            TU wLMean, wRMean;
            Gas::IdealGasThermalConservative2Primitive<dim>(ULMean, wLMean, gamma);
            Gas::IdealGasThermalConservative2Primitive<dim>(URMean, wRMean, gamma);
            Gas::GasInviscidFlux<dim>(ULMean, wLMean(Seq123), wLMean(I4), FLfix);
            Gas::GasInviscidFlux<dim>(URMean, wRMean(Seq123), wRMean(I4), FRfix);
            FLfix(Seq123) = normBase * FLfix(Seq123);
            FRfix(Seq123) = normBase * FRfix(Seq123);
            if (model == NS_SA || model == NS_SA_3D)
            {
                FLfix(I4 + 1) = wLMean(1) * ULMean(I4 + 1);
                FRfix(I4 + 1) = wRMean(1) * URMean(I4 + 1); // F_5 = rhoNut * un
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                FLfix({I4 + 1, I4 + 2}) = wLMean(1) * ULMean({I4 + 1, I4 + 2});
                FRfix({I4 + 1, I4 + 2}) = wRMean(1) * URMean({I4 + 1, I4 + 2}); // F_5 = rhoNut * un
            }
            // FLfix *= 0;
            // FRfix *= 0;
        }

        auto exitFun = [&]()
        {
            std::cout << "face at" << vfv->GetFaceQuadraturePPhys(iFace, -1) << '\n';
            std::cout << "UL" << UL.transpose() << '\n';
            std::cout << "UR" << UR.transpose() << std::endl;
        };

        real lam0{0}, lam123{0}, lam4{0};
        lam123 = std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5;

        if (pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWall)
        {
#ifdef USE_NO_RIEMANN_ON_WALL
            TU UL_Prim, UR_Prim;
            UL_Prim.resizeLike(UL);
            UL_Prim.resizeLike(UR);
            Gas::IdealGasThermalConservative2Primitive<dim>(UL, UL_Prim, gamma);
            Gas::IdealGasThermalConservative2Primitive<dim>(UR, UR_Prim, gamma);
            UL_Prim(Seq123).setZero();
            UR_Prim(Seq123).setZero();
            Gas::IdealGasThermalPrimitive2Conservative<dim>(UL_Prim, UL, gamma);
            // Gas::IdealGasThermalPrimitive2Conservative<dim>(UR_Prim, UR, gamma);
            UR = UL;
#else

#endif
        }
        if (pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWallInvis)
        {
        }

        auto RSWrapper = [&](Gas::RiemannSolverType rsType, auto &UL, auto &UR, auto &ULm, auto &URm, real gamma, auto &finc, real dLambda)
        {
            if (rsType == Gas::RiemannSolverType::HLLEP)
                Gas::HLLEPFlux_IdealGas<dim, 0>(
                    UL, UR, ULm, URm, gamma, finc, dLambda,
                    exitFun);
            else if (rsType == Gas::RiemannSolverType::HLLEP_V1)
                Gas::HLLEPFlux_IdealGas<dim, 1>(
                    UL, UR, ULm, URm, gamma, finc, dLambda,
                    exitFun);
            else if (rsType == Gas::RiemannSolverType::HLLC)
                Gas::HLLCFlux_IdealGas_HartenYee<dim>(
                    UL, UR, gamma, finc, dLambda,
                    exitFun);
            else if (rsType == Gas::RiemannSolverType::Roe)
                Gas::RoeFlux_IdealGas_HartenYee<dim>(
                    UL, UR, ULm, URm, gamma, finc, dLambda,
                    exitFun, lam0, lam123, lam4);
            else if (rsType == Gas::RiemannSolverType::Roe_M1)
                Gas::RoeFlux_IdealGas_HartenYee<dim, 1>(
                    UL, UR, ULm, URm, gamma, finc, dLambda,
                    exitFun, lam0, lam123, lam4);
            else if (rsType == Gas::RiemannSolverType::Roe_M2)
                Gas::RoeFlux_IdealGas_HartenYee<dim, 2>(
                    UL, UR, ULm, URm, gamma, finc, dLambda,
                    exitFun, lam0, lam123, lam4);
            else if (rsType == Gas::RiemannSolverType::Roe_M3)
                Gas::RoeFlux_IdealGas_HartenYee<dim, 3>(
                    UL, UR, ULm, URm, gamma, finc, dLambda,
                    exitFun, lam0, lam123, lam4);
            else if (rsType == Gas::RiemannSolverType::Roe_M4)
                Gas::RoeFlux_IdealGas_HartenYee<dim, 4>(
                    UL, UR, ULm, URm, gamma, finc, dLambda,
                    exitFun, lam0, lam123, lam4);
            else if (rsType == Gas::RiemannSolverType::Roe_M5)
                Gas::RoeFlux_IdealGas_HartenYee<dim, 5>(
                    UL, UR, ULm, URm, gamma, finc, dLambda,
                    exitFun, lam0, lam123, lam4);
            else
                DNDS_assert(false);
            // std::cout << "HERE2" << std::endl;
            // if (btype == BoundaryType::Wall_NoSlip || btype == BoundaryType::Wall_Euler)
            //     finc(0) = 0; //! enforce mass leak = 0
        };

        TU finc;
        finc.resizeLike(ULxy);
        // std::cout << "HERE" << std::endl;
        if (settings.rsRotateScheme == 0)
        {
            TU &ULm = settings.rsMeanValueEig == 1 ? ULMean : UL;
            TU &URm = settings.rsMeanValueEig == 1 ? URMean : UR;
            RSWrapper(rsType, UL, UR, ULm, URm, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace]);
        }
        else if (settings.rsRotateScheme)
        {
            TVec veloL = settings.rsRotateScheme == 1 ? UL(Seq123) / UL(0) : ULMean(Seq123) / ULMean(0);
            TVec veloR = settings.rsRotateScheme == 1 ? UR(Seq123) / UR(0) : URMean(Seq123) / URMean(0);
            TVec diffVelo = veloR - veloL;
            real diffVeloN = diffVelo.norm();
            real veloLN = veloL.norm();
            real veloRN = veloR.norm();
            if (diffVeloN < (smallReal * 10) * (veloLN + veloRN) || diffVeloN < std::sqrt(verySmallReal))
            {
                TU &ULm = settings.rsMeanValueEig == 1 ? ULMean : UL;
                TU &URm = settings.rsMeanValueEig == 1 ? URMean : UR;
                RSWrapper(rsType, UL, UR, ULm, URm, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace]);
            }
            else // use rotate
            {
                TVec N1 = diffVelo / diffVeloN;
                DNDS_assert_info(std::abs(N1.norm() - 1) < 1e-5,
                                 fmt::format("{}", diffVeloN));

                real N1Proj = N1(0);
                TVec N2 = -N1 * N1Proj;
                N2(0) += 1;
                real N2Proj = N2.norm();
                if (N2Proj < 10 * smallReal) // N is fully N1
                {
                    Gas::RiemannSolverType rsTypeAux = settings.rsTypeAux;
                    TU &ULm = settings.rsMeanValueEig == 1 ? ULMean : UL;
                    TU &URm = settings.rsMeanValueEig == 1 ? URMean : UR;
                    RSWrapper(rsTypeAux ? rsTypeAux : Gas::RiemannSolverType::Roe_M2,
                              UL, UR, ULm, URm, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace]);
                }
                else
                {
                    N2 /= N2Proj;
                    DNDS_assert_info(std::abs(N1.norm() - 1) < 1e-5 && std::abs(N2.norm() - 1) < 1e-5,
                                     fmt::format("{},{}", N1Proj, N2Proj));
                    auto fullN = N1 * N1Proj + N2 * N2Proj;
                    DNDS_assert_info(std::abs(fullN(0) - 1) < 1e-5 && std::abs(fullN.norm() - 1) < 1e-5 && std::abs(N1.dot(N2)) < 1e-5,
                                     fmt::format("{},{}", N1Proj, N2Proj));
                    if (N2Proj < 0)
                        N2 *= -1, N2Proj *= -1;
                    if (N1Proj < 0)
                        N1 *= -1, N1Proj *= -1; //! riemann solver should distinguish L & R, if n is inverted, then L-R is inconsistent

                    // {std::cout << N1.transpose() << ", ";
                    // std::cout << N2.transpose() << ", ";
                    // std::cout << N1Proj << " " << N2Proj <<std::endl;}

                    TMat normBaseN1 = Geom::NormBuildLocalBaseV<dim>(N1);
                    TMat normBaseN2 = Geom::NormBuildLocalBaseV<dim>(N2);

                    TU ULMeanN1 = ULMean;
                    TU URMeanN1 = URMean;
                    ULMeanN1(Seq123) = normBaseN1.transpose() * ULMeanN1(Seq123);
                    URMeanN1(Seq123) = normBaseN1.transpose() * URMeanN1(Seq123);
                    TU ULN1 = UL;
                    TU URN1 = UR;
                    ULN1(Seq123) = normBaseN1.transpose() * ULN1(Seq123);
                    URN1(Seq123) = normBaseN1.transpose() * URN1(Seq123);

                    TU ULMeanN2 = ULMean;
                    TU URMeanN2 = URMean;
                    ULMeanN2(Seq123) = normBaseN2.transpose() * ULMeanN2(Seq123);
                    URMeanN2(Seq123) = normBaseN2.transpose() * URMeanN2(Seq123);
                    TU ULN2 = UL;
                    TU URN2 = UR;
                    ULN2(Seq123) = normBaseN2.transpose() * ULN2(Seq123);
                    URN2(Seq123) = normBaseN2.transpose() * URN2(Seq123);

                    TU fincN1;
                    fincN1.resizeLike(ULxy);
                    TU fincN2;
                    fincN2.resizeLike(ULxy);

                    TU &ULmN1 = settings.rsMeanValueEig == 1 ? ULMeanN1 : ULN1;
                    TU &URmN1 = settings.rsMeanValueEig == 1 ? URMeanN1 : URN1;
                    TU &ULmN2 = settings.rsMeanValueEig == 1 ? ULMeanN2 : ULN2;
                    TU &URmN2 = settings.rsMeanValueEig == 1 ? URMeanN2 : URN2;

                    RSWrapper(rsType,
                              ULN2, URN2, ULmN2, URmN2, settings.idealGasProperty.gamma, fincN2, deltaLambdaFace[iFace]);
                    Gas::RiemannSolverType rsTypeAux = settings.rsTypeAux;
                    RSWrapper(rsTypeAux ? rsTypeAux : Gas::RiemannSolverType::Roe_M2,
                              ULN1, URN1, ULmN1, URmN1, settings.idealGasProperty.gamma, fincN1, deltaLambdaFace[iFace]);

                    fincN1(Seq123) = normBaseN1 * fincN1(Seq123);
                    fincN2(Seq123) = normBaseN2 * fincN2(Seq123);

                    { // display rotation diff
                      // TU &ULm = settings.rsMeanValueEig == 1 ? ULMean : UL;
                      // TU &URm = settings.rsMeanValueEig == 1 ? URMean : UR;
                      // RSWrapper(rsType, UL, UR, UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace]);
                      // {
                      //     std::cout << N1.transpose() << "\n" << N2.transpose() << "\n";
                      //     std::cout << N1Proj <<" " <<N2Proj << std::endl;
                      //     std::cout << ULN1.transpose() <<" ---- " << URN1.transpose() << std::endl;
                      //     std::cout << fincN1.transpose() << std::endl;
                      //     std::cout << ULN2.transpose() <<" ---- " << URN2.transpose() << std::endl;
                      //     std::cout << fincN2.transpose() << std::endl;
                      //     std::cout << "---\n" << normBaseN1 << "\n---\n";
                      //     std::cout << "---\n" << normBaseN2 << "\n---\n";

                        //     std::cout << (N1Proj * fincN1 + N2Proj * fincN2).transpose() << std::endl;
                        //     std::cout << finc.transpose() << std::endl;
                        //     std::abort();
                        // }
                    }
                    finc = N1Proj * fincN1 + N2Proj * fincN2;
                }
            }
        }
        else
            DNDS_assert(false);

#ifndef USE_ENTROPY_FIXED_LAMBDA_IN_SA
        lam123 = (std::abs(UL(1) / UL(0)) + std::abs(UR(1) / UR(0))) * 0.5; //! high fix
                                                                            // lam123 = std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5; //! low fix
#endif

        if constexpr (model == NS_SA || model == NS_SA_3D)
        {
            // real lambdaFaceCC = sqrt(std::abs(asqrMean)) + std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5;
            real lambdaFaceCC = lam123; //! using velo instead of velo + a
            finc(I4 + 1) =
                ((UL(1) / UL(0) * UL(I4 + 1) + UR(1) / UR(0) * UR(I4 + 1)) -
                 (UR(I4 + 1) - UL(I4 + 1)) * lambdaFaceCC) *
                0.5;
        }
        if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
        {
            real lambdaFaceCC = lam123; //! using velo instead of velo + a
            finc({I4 + 1, I4 + 2}) =
                ((UL(1) / UL(0) * UL({I4 + 1, I4 + 2}) + UR(1) / UR(0) * UR({I4 + 1, I4 + 2})) -
                 (UR({I4 + 1, I4 + 2}) - UL({I4 + 1, I4 + 2})) * lambdaFaceCC) *
                0.5;
        }
        finc(Seq123) = normBase * finc(Seq123);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
        finc -= VisFlux.transpose() * unitNorm * 1;
#endif

        if (finc.hasNaN() || (!finc.allFinite()))
        {
            std::cout << finc.transpose() << std::endl;
            std::cout << ULxy.transpose() << std::endl;
            std::cout << URxy.transpose() << std::endl;
            std::cout << DiffUxy << std::endl;
            std::cout << unitNorm << std::endl;
            std::cout << normBase << std::endl;
            std::cout << T << std::endl;
            std::cout << muf << std::endl;
            std::cout << pMean << std::endl;
            DNDS_assert(false);
        }

        return -finc;
    }

}