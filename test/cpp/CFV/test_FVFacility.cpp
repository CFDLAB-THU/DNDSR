#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Legacy/VariationalReconstruction.hpp"
#include "CFV/Legacy/VariationalReconstruction_LimiterProcedure.hxx"
#include "CFV/Plans/InterfaceSamplingPlan.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include "Geom/Interface/BoundaryInterface.hpp"
#include "Geom/Access/EntityWorkPartition.hpp"
#include "../Geom/InterfaceFixture.hpp"

using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;
namespace C = DNDS::CFV;
static MPIInfo mpi;
static std::string MeshPath(const char *name)
{
    std::string path = __FILE__;
    for (int i = 0; i < 4; ++i)
        path = path.substr(0, path.find_last_of('/'));
    return path + "/data/mesh/" + name;
}
static ssp<UnstructuredMesh> Mesh()
{
    PartitionOptions partition;
    partition.metisSeed = 42;
    AutoAppendName2ID names;
    auto mesh = ReadNativeMeshFromCGNS(mpi, 2, MeshPath("Uniform_3x3_wall.cgns"), partition,
                                       [&](const std::string &name)
                                       { return names(name); });
    mesh->PrepareNativeTopology();
    return mesh;
}
static DNDS::real Polynomial(const tPoint &p, int degree)
{
    DNDS::real v = 1 + 2 * p[0] - .5 * p[1];
    if (degree >= 2)
        v += .3 * p[0] * p[0] + .2 * p[0] * p[1] - .1 * p[1] * p[1];
    if (degree >= 3)
        v += .02 * p[0] * p[0] * p[1] - .03 * p[1] * p[1] * p[1];
    return v;
}
TEST_CASE("FV metrics share managed mesh communication with independent preflight policy")
{
    auto communication = R::CommunicationContext::Duplicate(mpi.comm);
    AutoAppendName2ID names;
    auto owned = ReadAndDeriveNativeMeshFromCGNSOwned(mpi, 2, MeshPath("Uniform_3x3_wall.cgns"), {}, [&](const std::string &name)
                                                      { return names(name); }, {}, 0, 0, 0, {}, communication);
    PrepareMeshOptions prepare;
    prepare.buildSerialOut = false;
    auto mesh = UnstructuredMesh::FromNative(mpi, PrepareNativeMeshForSolverOwned(std::move(owned), mpi, prepare));
    REQUIRE(mesh->PreparationCommunication());
    C::FiniteVolumeMetricOptions options;
    options.collectivePreflight = false;
    const auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, options);
    CHECK(metrics.Communication()->SharesResource(*communication));
    CHECK_FALSE(metrics.Communication()->ChecksCollectivePreflight());
    CHECK(communication->ChecksCollectivePreflight());
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(metrics.Communicator(), communication->Communicator(), &comparison);
    CHECK(comparison == MPI_IDENT);
}

TEST_CASE("Midpoint face publication preserves integrated measures and cell metrics")
{
    for (const bool curved : {false, true})
    {
        CAPTURE(curved);
        auto mesh = Mesh();
        if (curved)
        {
            auto quadratic = std::make_shared<UnstructuredMesh>(mpi, 2);
            quadratic->BuildO2FromO1Elevation(*mesh);
            quadratic->PrepareNativeTopology();
            quadratic->TransformCoords([](tPoint p) -> tPoint
                                       { p[1] += .05 * p[0] * p[0]; return p; });
            mesh = std::move(quadratic);
        }
        const auto full = C::FiniteVolumePlan::ConstructMetrics(mesh, 5);
        C::FiniteVolumeMetricOptions metricOptions;
        metricOptions.faceIntegrationOrder = 1;
        const auto midpoint = C::FiniteVolumePlan::ConstructMetrics(mesh, metricOptions);
        const auto aRead = full.AcquireRead(), bRead = midpoint.AcquireRead();
        const auto a = aRead.HostView(), b = bRead.HostView();
        for (DNDS::index iCell = 0; iCell < full.NumCellProc(); ++iCell)
            for (DNDS::index j = 0; j < 8; ++j)
                CHECK(a.cells(iCell, j) == b.cells(iCell, j));
        int nonplanar = 0;
        for (DNDS::index iFace = 0; iFace < full.NumFaceProc(); ++iFace)
        {
            CHECK(b.FaceQuadratureEnd(iFace) - b.FaceQuadratureBegin(iFace) == 1);
            DNDS::real area = 0;
            tPoint normal = tPoint::Zero();
            for (DNDS::index iPoint = a.FaceQuadratureBegin(iFace); iPoint < a.FaceQuadratureEnd(iFace); ++iPoint)
            {
                area += a.GetFaceWeight(iPoint);
                normal += a.GetFaceWeight(iPoint) * a.GetFaceNorm(iPoint);
            }
            CHECK(b.GetFaceWeight(b.FaceQuadratureBegin(iFace)) == doctest::Approx(area).epsilon(1e-14));
            CHECK(a.GetFaceArea(iFace) == doctest::Approx(area).epsilon(1e-14));
            CHECK((a.GetFaceMeanNorm(iFace) - normal.normalized()).norm() < 1e-13);
            CHECK((a.GetFaceCent(iFace) - b.GetFacePoint(b.FaceQuadratureBegin(iFace))).norm() < 1e-13);
            CHECK((a.GetFaceMeanNorm(iFace) - b.GetFaceNorm(b.FaceQuadratureBegin(iFace))).norm() < 1e-13);
            // A curved face's scalar measure must not be replaced by the norm
            // of its integrated normal (the endpoint chord length in 2D).
            nonplanar += area > normal.norm() * (1 + 1e-8);
        }
        MPI_Allreduce(MPI_IN_PLACE, &nonplanar, 1, MPI_INT, MPI_SUM, mpi.comm);
        CHECK((nonplanar > 0) == curved);
    }
}
TEST_CASE("Runtime FV matches legacy VR sweeps and reproduces cell-average polynomials")
{
    auto mesh = Mesh();
    for (int degree = 1; degree <= 3; ++degree)
        for (bool hqm : {false, true})
        {
            CAPTURE(degree);
            CAPTURE(hqm);
            C::ReconstructionSettings settings;
            settings.maxOrder = degree, settings.intOrder = 6, settings.HQMWeights = hqm;
            settings.boundaryDerivativeWeights = {1, 0, 0, 0}; // Physical polynomial BC values, without homogeneous derivative penalties.
            auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, 6);
            auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
            auto u = metrics.BuildField(1);
            auto workspace = plan.BuildWorkspace(1);
            auto coefficients = plan.BuildCoefficients(1);
            auto boundary = plan.BuildBoundaryValues(1);
            C::VariationalReconstruction<2> legacy(mpi, mesh);
            C::VRSettings old(2);
            nlohmann::ordered_json json;
            old.WriteIntoJson(json);
            json["maxOrder"] = degree, json["intOrder"] = 6, json["SORInstead"] = false, json["cacheDiffBase"] = true;
            if (hqm)
            {
                json["functionalSettings"]["dirWeightScheme"] = "HQM_OPT";
                json["functionalSettings"]["geomWeightScheme"] = "HQM_SD";
                json["functionalSettings"]["geomWeightBias"] = 1.;
            }
            legacy.parseSettings(json);
            legacy.ConstructMetrics();
            if (degree == 1 && !hqm)
            {
                const auto lengths = metrics.BuildCellMaxLenScale();
                const auto read = lengths.Read();
                const auto values = read.View<R::FixedRows<1>>();
                for (DNDS::index iCell = 0; iCell < metrics.NumCellProc(); ++iCell)
                    CHECK(values(iCell, 0) == doctest::Approx(legacy.GetCellMaxLenScale(iCell)).epsilon(2e-11));
            }
            legacy.ConstructBaseAndWeight([](t_index, int order)
                                          { return order == 0 ? 1. : 0.; });
            legacy.ConstructRecCoeff();
            C::tUDof<1> oldMean;
            C::tURec<1> oldRec, oldNext;
            legacy.BuildUDof(oldMean, 1), legacy.BuildURec(oldRec, 1), legacy.BuildURec(oldNext, 1);
            oldRec.setConstant(0), oldNext.setConstant(0);
            {
                auto write = u.Owned().Write();
                auto values = write.View<R::FixedRows<1>>();
                auto read = mesh->AcquireRead(MeshReadRequirements::Primary());
                const auto view = read.HostView();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                {
                    Elem::Quadrature q(view.GetCellElement(iCell), 6);
                    tSmallCoords coords;
                    view.GetCoordsOnCell(iCell, coords);
                    DNDS::real average = 0;
                    q.Integration(average, [&](auto &v, int, const tPoint &, const auto &shape)
                                  { v = Polynomial(Elem::PPhysicsCoordD01Nj(coords, shape), degree) * Elem::CellJacobianDet(2, coords, shape); });
                    values(iCell, 0) = average / metrics.GetCellVol(iCell);
                    oldMean[iCell](0) = values(iCell, 0);
                }
            }
            {
                auto write = boundary.values.Write();
                for (DNDS::index i = 0; i < plan.NumBoundaryQuadrature(); ++i)
                    write.View<R::FixedRows<1>>()(i, 0) = Polynomial(plan.GetBoundaryQuadrature(i).point, degree);
            }
            auto meanHalo = metrics.BindHalo(u), recHalo = metrics.BindHalo(coefficients);
            meanHalo.Begin().Wait();
            oldMean.trans.startPersistentPull(), oldMean.trans.waitPersistentPull();
            C::VariationalReconstruction<2>::TFBoundary<1> bc = [degree](const auto &, const auto &, DNDS::index, DNDS::index, int, const tPoint &, const tPoint &p, t_index)
            { return Eigen::Vector<DNDS::real, 1>{Polynomial(p, degree)}; };
            for (int iteration = 0; iteration < 12; ++iteration)
            {
                recHalo.Begin().Wait();
                oldRec.trans.startPersistentPull(), oldRec.trans.waitPersistentPull();
                plan.DoReconstructionIter(u, coefficients, workspace, boundary);
                legacy.DoReconstructionIter<1>(oldRec, oldNext, oldMean, bc, true);
                std::swap(oldRec, oldNext);
                auto read = coefficients.Owned().Read();
                const auto values = read.View<R::TableRows>();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                    for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                        CHECK(values(iCell, mode) == doctest::Approx(oldRec[iCell](mode, 0)).epsilon(2e-9).scale(1.));
            }
            for (int iteration = 0; iteration < 800; ++iteration)
            {
                recHalo.Begin().Wait();
                plan.DoReconstructionIter(u, coefficients, workspace, boundary);
            }
            R::Array<DNDS::index> ids(R::RowLayout::Dense(mesh->NumCell(), 1));
            R::Array<DNDS::real> points(R::RowLayout::Dense(mesh->NumCell(), 3)), result(R::RowLayout::Dense(mesh->NumCell(), 1));
            {
                auto idWrite = ids.Write();
                auto pointWrite = points.Write();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                {
                    idWrite.View()(iCell, 0) = iCell;
                    const tPoint p = metrics.GetCellCent(iCell) + tPoint{.01, -.02, 0};
                    for (int d = 0; d < 3; ++d)
                        pointWrite.View()(iCell, d) = p[d];
                }
            }
            plan.Evaluate(u, coefficients, ids, points, result);
            {
                C::PolynomialSamplingPlan valueOnly(plan, ids, points, false);
                R::Array<DNDS::real> cached(R::RowLayout::Dense(mesh->NumCell(), 1));
                valueOnly.Evaluate(u, coefficients, cached);
                {
                    auto a = result.Read(), b = cached.Read();
                    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                        CHECK(a.View()(iCell, 0) == doctest::Approx(b.View()(iCell, 0)).epsilon(2e-14));
                }
                CHECK_THROWS(valueOnly.Evaluate(u, coefficients, cached, 1));
            }
            auto pointRead = points.Read(), resultRead = result.Read();
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            {
                const auto p = pointRead.View()[iCell];
                CHECK(resultRead.View()(iCell, 0) == doctest::Approx(Polynomial({p[0], p[1], p[2]}, degree)).epsilon(2e-7));
            }
            if (mpi.size == 1)
            {
                std::vector<DNDS::index> selected(size_t(mesh->NumBnd()));
                std::iota(selected.begin(), selected.end(), DNDS::index(0));
                auto patch = BoundaryPatch::Create(mesh, selected);
                R::Array<DNDS::index> queryIds(R::RowLayout::Dense(patch.Size(), 1));
                R::Array<double> queryPoints(R::RowLayout::Dense(patch.Size(), 3)), sampled(R::RowLayout::Dense(patch.Size(), 1));
                {
                    auto read = mesh->AcquireRead(MeshReadRequirements::CellFlux());
                    const auto view = read.HostView();
                    auto write = queryPoints.Write();
                    auto id = queryIds.Write();
                    for (DNDS::index i = 0; i < patch.Size(); ++i)
                    {
                        const auto face = patch.Element(i).face;
                        const auto nodes = view.FaceNodes(face);
                        const tPoint p = .4 * view.Coord(nodes[0]) + .6 * view.Coord(nodes[1]);
                        for (int d = 0; d < 3; ++d)
                            write.View()(i, d) = p[d];
                        id.View()(i, 0) = 2 * patch.Size() - i;
                    }
                }
                auto correspondence = InterfacePointMap::Match(patch, queryPoints, queryIds);
                C::InterfaceSamplingPlan sampling(plan, correspondence);
                sampling.Evaluate(u, coefficients, sampled);
                auto read = sampled.Read();
                for (DNDS::index i = 0; i < patch.Size(); ++i)
                    CHECK(read.View()(i, 0) == doctest::Approx(Polynomial(correspondence.Get(i).donorPoint, degree)).epsilon(2e-7));
            }
        }
}
TEST_CASE("Independent functional weights and all-order boundary penalties match configured legacy sweeps")
{
    auto mesh = Mesh();
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, 6);
    using F = C::ReconstructionFunctionalSettings;
    for (int scenario = 0; scenario < 6; ++scenario)
    {
        CAPTURE(scenario);
        C::ReconstructionSettings settings;
        settings.maxOrder = 3, settings.intOrder = 6, settings.bcWeight = .7;
        settings.boundaryDerivativeWeights = {1, 1, 1, 1};
        if (scenario == 0)
        {
            int localZone = INT_MAX, zone = INT_MAX;
            auto read = mesh->AcquireRead(MeshReadRequirements::CellFlux());
            if (mesh->NumBnd())
                localZone = read.HostView().GetBndInfo(0).zone;
            MPI_Allreduce(&localZone, &zone, 1, MPI_INT, MPI_MIN, mpi.comm);
            settings.boundaryZoneWeights[zone] = {.9, .3, 0, 0};
        }
        settings.cacheDiffBase = scenario % 2 == 0;
        auto &f = settings.functional;
        if (scenario == 1)
            f.dirWeightScheme = F::DirectionalWeight::HQM_OPT;
        if (scenario == 2)
            f.dirWeightScheme = F::DirectionalWeight::ManualDirWeight, f.manualDirWeights = {1.1, .8, .3, .1};
        if (scenario == 3)
            f.dirWeightScheme = F::DirectionalWeight::TEST_OPT;
        if (scenario == 4)
            f.geomWeightScheme = F::GeometricWeight::SD_Power, f.geomWeightPower1 = .3, f.geomWeightPower2 = -.2, f.geomWeightBias = .1;
        if (scenario == 5)
            f.scaleType = F::LengthScale::MeanAACBB, f.scaleMultiplier = .8;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        auto work = plan.BuildWorkspace(1);
        auto coefficients = plan.BuildCoefficients(1);
        auto mean = metrics.BuildField(1);
        auto boundary = plan.BuildBoundaryValues(1);
        C::VariationalReconstruction<2> legacy(mpi, mesh);
        C::VRSettings old(2);
        nlohmann::ordered_json json, functional;
        old.WriteIntoJson(json);
        to_json(functional, f);
        for (const auto &[key, value] : functional.items())
            json["functionalSettings"][key] = value;
        // Legacy's unused degree-four entry is outside this degree-three API.
        json["functionalSettings"]["manualDirWeights"].push_back(0.);
        json["maxOrder"] = 3, json["intOrder"] = 6, json["SORInstead"] = false, json["bcWeight"] = settings.bcWeight;
        legacy.parseSettings(json);
        legacy.ConstructMetrics();
        legacy.ConstructBaseAndWeight([&](t_index zone, int order)
                                      {
            const auto found=settings.boundaryZoneWeights.find(zone);
            return found==settings.boundaryZoneWeights.end() ? settings.boundaryDerivativeWeights[order] : found->second[order]; });
        legacy.ConstructRecCoeff();
        C::tUDof<1> oldMean;
        C::tURec<1> oldRec, oldNext;
        legacy.BuildUDof(oldMean, 1);
        legacy.BuildURec(oldRec, 1);
        legacy.BuildURec(oldNext, 1);
        oldRec.setConstant(0), oldNext.setConstant(0);
        {
            auto u = mean.Owned().Write();
            auto b = boundary.values.Write();
            for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
                u.View()(iCell, 0) = oldMean[iCell](0) = std::sin(.7 * mesh->OriginalEntityId(EntityKind::Cell, iCell));
            for (DNDS::index iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
            {
                const auto p = plan.GetBoundaryQuadrature(iPoint).point;
                b.View()(iPoint, 0) = 1 + p[0] * p[1];
            }
        }
        auto halo = metrics.BindHalo(coefficients);
        metrics.BindHalo(mean).Begin().Wait();
        oldMean.trans.startPersistentPull();
        oldMean.trans.waitPersistentPull();
        C::VariationalReconstruction<2>::TFBoundary<1> bc = [](const auto &, const auto &, DNDS::index, DNDS::index, int, const tPoint &, const tPoint &p, t_index)
        { return Eigen::Vector<DNDS::real, 1>{1 + p[0] * p[1]}; };
        for (int iteration = 0; iteration < 4; ++iteration)
        {
            halo.Begin().Wait();
            oldRec.trans.startPersistentPull();
            oldRec.trans.waitPersistentPull();
            plan.DoReconstructionIter(mean, coefficients, work, boundary);
            legacy.DoReconstructionIter<1>(oldRec, oldNext, oldMean, bc, true);
            std::swap(oldRec, oldNext);
            auto read = coefficients.Owned().Read();
            for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
                for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                    CHECK(read.View()(iCell, mode) == doctest::Approx(oldRec[iCell](mode, 0)).epsilon(2e-9).scale(1));
        }
        if (metrics.NumCell())
        {
            auto read = mesh->AcquireRead(MeshReadRequirements::CellFlux());
            const auto iFace = read.HostView().CellFaces(0)[0];
            const auto side = read.HostView().CellIsFaceBack(0, iFace, 0) ? 0 : 1;
            if (settings.cacheDiffBase)
            {
                const auto basis = plan.GetCachedFaceBasis(iFace, side, 0);
                CHECK(basis.rows() == plan.Metrics().Dimension() + 1);
                CHECK(basis.cols() == plan.NRecDOF());
                CHECK(basis.allFinite());
            }
            else
                CHECK_THROWS(plan.GetCachedFaceBasis(iFace, side, 0));
        }
    }
}

TEST_CASE("WBAP explicit five-component face transforms match legacy and preserve a passive sixth field")
{
    auto mesh = Mesh();
    C::ReconstructionSettings settings;
    settings.maxOrder = 2;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, settings.intOrder);
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto work = plan.BuildWorkspace(6);
    auto coefficients = plan.BuildCoefficients(6);
    auto disabled = plan.BuildFaceTransform(3, 0);
    CHECK(disabled.count == 0);
    CHECK(disabled.forward.RowCount() == 0);
    CHECK(disabled.inverse.RowCount() == 0);
    CHECK_NOTHROW(plan.FinalizeFaceTransform(disabled));
    auto disabledWork = plan.BuildWorkspace(6);
    auto disabledCoefficients = plan.BuildCoefficients(6);
    DNDS::Algebra::Fill(disabledCoefficients, 0.);
    metrics.BindHalo(disabledCoefficients).Begin().Wait();
    CHECK_NOTHROW(plan.DoLimiterWBAP_3Order(disabledCoefficients, disabledWork, 2, std::cref(disabled)));
    auto characteristic = plan.BuildFaceTransform(0, 5);
    Eigen::Matrix<DNDS::real, 5, 5> L = Eigen::Matrix<DNDS::real, 5, 5>::Identity();
    L(0, 0) = .8, L(0, 1) = -.6, L(1, 0) = .6, L(1, 1) = .8;
    {
        auto left = characteristic.forward.Write(), right = characteristic.inverse.Write();
        for (DNDS::index iFace = 0; iFace < metrics.NumFaceProc(); ++iFace)
        {
            Eigen::Map<Eigen::MatrixXd>(left.View()[iFace].data(), 5, 5) = L;
            Eigen::Map<Eigen::MatrixXd>(right.View()[iFace].data(), 5, 5) = L.transpose();
        }
    }
    CHECK_THROWS(plan.DoLimiterWBAP_3Order(coefficients, work, 2, std::cref(characteristic)));
    plan.FinalizeFaceTransform(characteristic);
    C::VariationalReconstruction<2> legacy(mpi, mesh);
    C::VRSettings old(2);
    nlohmann::ordered_json json;
    old.WriteIntoJson(json);
    json["maxOrder"] = 2, json["intOrder"] = 5, json["cacheDiffBase"] = true;
    legacy.parseSettings(json);
    legacy.ConstructMetrics();
    legacy.ConstructBaseAndWeight([](t_index, int order)
                                  { return order == 0 ? 1. : 0.; });
    legacy.ConstructRecCoeff();
    C::tUDof<6> mean;
    C::tURec<6> rec, output, temporary;
    C::tScalarPair indicator;
    legacy.BuildUDof(mean, 6), legacy.BuildURec(rec, 6), legacy.BuildURec(output, 6), legacy.BuildURec(temporary, 6);
    legacy.BuildScalar(indicator);
    mean.setConstant(2);
    for (DNDS::index iCell = 0; iCell < mesh->NumCellProc(); ++iCell)
        indicator(iCell, 0) = 1;
    {
        auto write = coefficients.Owned().Write();
        for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (int v = 0; v < 6; ++v)
                for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                {
                    const auto id = mesh->OriginalEntityId(EntityKind::Cell, iCell);
                    const double value = std::sin(.7 * id + .31 * v + .19 * mode);
                    write.View()(iCell, v * plan.NRecDOF() + mode) = rec[iCell](mode, v) = value;
                }
    }
    auto halo = metrics.BindHalo(coefficients);
    halo.Begin().Wait();
    rec.trans.startPersistentPull(), rec.trans.waitPersistentPull();
    mean.trans.startPersistentPull(), mean.trans.waitPersistentPull();
    using Batch = C::VariationalReconstruction<2>::tLimitBatch<6>;
    C::VariationalReconstruction<2>::tFMEig<6> forward = [L](const auto &a, const auto &b, const tPoint &, const Eigen::Ref<Batch> &input) -> Batch
    {
        CHECK(a.isConstant(2));
        CHECK(b.isConstant(2));
        Batch result = input;
        result.leftCols(5) = input.leftCols(5) * L.transpose();
        return result;
    };
    C::VariationalReconstruction<2>::tFMEig<6> inverse = [L](const auto &, const auto &, const tPoint &, const Eigen::Ref<Batch> &input) -> Batch
    {
        Batch result = input;
        result.leftCols(5) = input.leftCols(5) * L;
        return result;
    };
    legacy.DoLimiterWBAP_3<6>(mean, rec, output, temporary, indicator, true, forward, inverse, true);
    for (int order = 2; order >= 1; --order)
    {
        plan.DoLimiterWBAP_3Order(coefficients, work, order, std::cref(characteristic));
        halo.Begin().Wait();
    }
    auto read = coefficients.Owned().Read();
    for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
        for (int v = 0; v < 6; ++v)
            for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                CHECK(read.View()(iCell, v * plan.NRecDOF() + mode) == doctest::Approx(output[iCell](mode, v)).epsilon(1e-9).scale(1));
    auto bad = characteristic;
    bad.first = 2;
    CHECK_THROWS(plan.DoLimiterWBAP_3Order(coefficients, work, 2, std::cref(bad)));
    {
        auto write = characteristic.inverse.Write();
        if (metrics.NumFaceProc())
            write.View()(0, 0) = 7;
    }
    if (metrics.NumFaceProc())
    {
        CHECK_NOTHROW(plan.FinalizeFaceTransform(characteristic));
        CHECK_THROWS(plan.FinalizeFaceTransform(characteristic, true));
    }
}

TEST_CASE("FV rejects wrong association and invalid shapes before writes")
{
    auto mesh = Mesh();
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    auto other = C::FiniteVolumePlan::ConstructMetrics(mesh);
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics);
    {
        auto read = metrics.AcquireRead();
        const auto view = read.HostView();
        for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            CHECK(view.GetCellVol(iCell) == metrics.GetCellVol(iCell));
            CHECK((view.GetCellCent(iCell) - metrics.GetCellCent(iCell)).norm() == 0);
        }
        for (DNDS::index iFace = 0; iFace < metrics.NumFaceProc(); ++iFace)
            for (DNDS::index i = view.FaceQuadratureBegin(iFace); i < view.FaceQuadratureEnd(iFace); ++i)
            {
                CHECK(view.GetFaceWeight(i) > 0);
                CHECK(view.GetFaceNorm(i).norm() == doctest::Approx(1));
            }
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Geometry));
    }
    auto work = plan.BuildWorkspace(1);
    auto coefficients = plan.BuildCoefficients(1);
    auto boundary = plan.BuildBoundaryValues(1);
    auto wrong = other.BuildField(1);
    CHECK_THROWS(plan.DoReconstructionIter(wrong, coefficients, work, boundary));
    auto malformed = plan.BuildBoundaryValues(2);
    auto mean = metrics.BuildField(1);
    CHECK_THROWS(plan.DoReconstructionIter(mean, coefficients, work, malformed));
    CHECK_THROWS(plan.DoLimiterWBAP_3Order(coefficients, work, 0));
    auto invalid = plan.Settings();
    invalid.maxOrder = 4;
    CHECK_THROWS(C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, invalid));
    // Same-size storage from another plan must not be accepted by row count.
    auto wrongPlan = C::VariationalReconstructionPlan::ConstructRecCoeff(other);
    auto wrongWork = wrongPlan.BuildWorkspace(1);
    auto wrongCoefficients = wrongPlan.BuildCoefficients(1);
    CHECK_THROWS(plan.DoReconstructionIter(mean, wrongCoefficients, wrongWork, boundary));
    {
        auto mutation = mesh->versions.BeginMutation(R::Change::Geometry);
        mutation.Commit();
    }
    CHECK_THROWS(plan.Validate());
}

TEST_CASE("Periodic quad and triangle reconstruction matches relaxed legacy Jacobi and ordered SOR")
{
    for (const char *name : {"IV10_2.cgns", "IV10U_10.cgns"})
        for (bool sor : {false, true})
        {
            CAPTURE(name);
            CAPTURE(sor);
            NativePeriodicImportOptions frames;
            frames.frames.translation[1].map() = tPoint{10, 0, 0};
            frames.frames.translation[2].map() = tPoint{0, 10, 0};
            PartitionOptions partition;
            partition.metisSeed = 42;
            AutoAppendName2ID names;
            auto mesh = ReadNativeMeshFromCGNS(mpi, 2, MeshPath(name), partition, [&](const std::string &zone)
                                               { return names(zone); }, frames);
            mesh->PrepareNativeTopology();
            C::ReconstructionSettings settings;
            settings.SORInstead = sor, settings.jacobiRelax = .73;
            auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, settings.intOrder);
            auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
            auto mean = metrics.BuildField(1);
            auto work = plan.BuildWorkspace(1);
            auto coefficients = plan.BuildCoefficients(1);
            auto boundary = plan.BuildBoundaryValues(1);
            CHECK(plan.NumBoundaryQuadrature() == 0);
            C::VariationalReconstruction<2> legacy(mpi, mesh);
            C::VRSettings old(2);
            nlohmann::ordered_json json;
            old.WriteIntoJson(json);
            json["maxOrder"] = 2, json["intOrder"] = settings.intOrder;
            json["SORInstead"] = sor, json["jacobiRelax"] = settings.jacobiRelax;
            legacy.parseSettings(json);
            legacy.ConstructMetrics();
            legacy.ConstructBaseAndWeight();
            legacy.ConstructRecCoeff();
            legacy.SetPeriodicTransformations();
            C::tUDof<1> oldMean;
            C::tURec<1> oldRec, next;
            legacy.BuildUDof(oldMean, 1), legacy.BuildURec(oldRec, 1), legacy.BuildURec(next, 1);
            oldRec.setConstant(0), next.setConstant(0);
            {
                auto write = mean.Owned().Write();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                    write.View()(iCell, 0) = oldMean[iCell](0) = std::sin(.71 * mesh->OriginalEntityId(EntityKind::Cell, iCell));
            }
            auto halo = metrics.BindHalo(coefficients);
            metrics.BindHalo(mean).Begin().Wait();
            oldMean.trans.startPersistentPull(), oldMean.trans.waitPersistentPull();
            C::VariationalReconstruction<2>::TFBoundary<1> bc = [](const auto &, const auto &, DNDS::index, DNDS::index, int, const tPoint &, const tPoint &, t_index)
            { throw std::runtime_error("fully periodic mesh must never call boundary handler"); return Eigen::Vector<DNDS::real,1>::Zero().eval(); };
            for (int iteration = 0; iteration < 12; ++iteration)
            {
                halo.Begin().Wait();
                oldRec.trans.startPersistentPull(), oldRec.trans.waitPersistentPull();
                plan.DoReconstructionIter(mean, coefficients, work, boundary);
                legacy.DoReconstructionIter<1>(oldRec, next, oldMean, bc, true);
                // putIntoNew=true publishes next for both schedules; legacy
                // internally swaps its SOR working rows at the end of a sweep.
                std::swap(oldRec, next);
                auto read = coefficients.Owned().Read();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                    for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                        CHECK(read.View()(iCell, mode) == doctest::Approx(oldRec[iCell](mode, 0)).epsilon(2e-9).scale(1));
            }
        }
}

TEST_CASE("Three-dimensional FV reproduces cubic cell-average polynomials on tetrahedron and hexahedron with empty ranks")
{
    const auto function = [](const tPoint &p)
    { return 1 + .2 * p[0] - .3 * p[1] + .5 * p[2] + .1 * p[0] * p[1] * p[2] + .07 * p[2] * p[2] * p[2]; };
    for (bool hex : {false, true})
    {
        auto mesh = InterfaceFixture::SingleCell(mpi, hex);
        C::ReconstructionSettings settings;
        settings.maxOrder = 3, settings.intOrder = 6;
        settings.boundaryDerivativeWeights = {1, 0, 0, 0};
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, 6);
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        auto mean = metrics.BuildField(1);
        auto work = plan.BuildWorkspace(1);
        auto coefficients = plan.BuildCoefficients(1);
        auto boundary = plan.BuildBoundaryValues(1);
        {
            auto read = mesh->AcquireRead(MeshReadRequirements::CellFlux());
            const auto view = read.HostView();
            auto write = mean.Owned().Write();
            for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                tSmallCoords coords;
                view.GetCoordsOnCell(iCell, coords);
                Elem::Quadrature quad(view.GetCellElement(iCell), 6);
                double integral = 0;
                quad.Integration(integral, [&](auto &value, int, const tPoint &, const auto &shape)
                                 { value = function(Elem::PPhysicsCoordD01Nj(coords, shape)) * Elem::CellJacobianDet(3, coords, shape); });
                write.View()(iCell, 0) = integral / metrics.GetCellVol(iCell);
            }
        }
        {
            auto write = boundary.values.Write();
            for (DNDS::index i = 0; i < plan.NumBoundaryQuadrature(); ++i)
                write.View()(i, 0) = function(plan.GetBoundaryQuadrature(i).point);
        }
        metrics.BindHalo(mean).Begin().Wait();
        metrics.BindHalo(coefficients).Begin().Wait();
        plan.DoReconstructionIter(mean, coefficients, work, boundary);
        R::Array<DNDS::index> ids(R::RowLayout::Dense(metrics.NumCell(), 1));
        R::Array<double> points(R::RowLayout::Dense(metrics.NumCell(), 3)), result(R::RowLayout::Dense(metrics.NumCell(), 1));
        {
            auto indexWrite = ids.Write();
            auto pointWrite = points.Write();
            for (DNDS::index i = 0; i < metrics.NumCell(); ++i)
            {
                indexWrite.View()(i, 0) = i;
                pointWrite.View()(i, 0) = .2, pointWrite.View()(i, 1) = .1, pointWrite.View()(i, 2) = .3;
            }
        }
        plan.Evaluate(mean, coefficients, ids, points, result);
        auto read = result.Read();
        for (DNDS::index i = 0; i < metrics.NumCell(); ++i)
            CHECK(read.View()(i, 0) == doctest::Approx(function({.2, .1, .3})).epsilon(1e-9));
    }
}

TEST_CASE("SVD removes unconstrained modes and invalid settings fail collectively")
{
    auto mesh = InterfaceFixture::SingleCell(mpi, true);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.boundaryDerivativeWeights = {0, 0, 0, 0};
    auto zero = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    const auto summary = zero.GetInverseDiagnostics();
    DNDS::index localCells = metrics.NumCell(), totalCells = 0;
    MPI::Allreduce(&localCells, &totalCells, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
    CHECK(summary.cells == totalCells);
    CHECK(summary.filteredCells == totalCells);
    CHECK(summary.rankZeroCells == totalCells);
    CHECK(summary.rankCounts[0] == totalCells);
    CHECK(std::accumulate(summary.rankCounts.begin(), summary.rankCounts.end(), DNDS::index(0)) == totalCells);
    CHECK(summary.discardedModes == totalCells * zero.NRecDOF());
    CHECK(std::isinf(summary.maxCondition));
    for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
    {
        const auto cell = zero.GetCellInverseDiagnostics(iCell);
        CHECK(cell.rank == 0);
        CHECK(std::isinf(cell.rawCondition));
    }
    CHECK_THROWS(zero.GetCellInverseDiagnostics(-1));
    auto mean = metrics.BuildField(1);
    auto work = zero.BuildWorkspace(1);
    auto coefficients = zero.BuildCoefficients(1);
    auto boundary = zero.BuildBoundaryValues(1);
    {
        auto values = boundary.values.Write();
        for (DNDS::index iRow = 0; iRow < values.View().RowCount(); ++iRow)
            values.View()(iRow, 0) = 3;
        auto coefficientWrite = coefficients.Owned().Write();
        for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
            std::fill(coefficientWrite.View()[iCell].begin(), coefficientWrite.View()[iCell].end(), 2);
    }
    // Zero is the minimum-norm solution when the entire functional is zero.
    zero.DoReconstructionIter(mean, coefficients, work, boundary);
    auto read = coefficients.Owned().Read();
    for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
        for (auto value : read.View()[iCell])
            CHECK(value == 0);
    settings = C::ReconstructionSettings{};
    if (mpi.rank == 0)
        settings.maxOrder = 4;
    CHECK_THROWS(C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings));
    CHECK_NOTHROW(C::VariationalReconstructionPlan::ConstructRecCoeff(metrics));
}

TEST_CASE("Two-buffer SOR preserves its MPI input and split automatically selects core-first order")
{
    NativePeriodicImportOptions frames;
    frames.frames.translation[1].map() = tPoint{10, 0, 0};
    frames.frames.translation[2].map() = tPoint{0, 10, 0};
    PartitionOptions partition;
    partition.metisSeed = 42;
    AutoAppendName2ID names;
    auto mesh = ReadNativeMeshFromCGNS(mpi, 2, MeshPath("IV10U_10.cgns"), partition, [&](const std::string &zone)
                                       { return names(zone); }, frames);
    mesh->PrepareNativeTopology();
    C::ReconstructionSettings settings;
    settings.SORInstead = true;
    settings.maxOrder = 3;
    settings.jacobiRelax = .73;
    const auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, settings.intOrder);
    const auto split = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    settings.coreFirst = true;
    const auto full = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    const auto regions = EntityWorkPartition::CellNodeCell(*mesh);
    DNDS::index counts[2]{DNDS::index(regions.Core().size()), DNDS::index(regions.HaloDependent().size())};
    MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT64_T, MPI_SUM, mpi.comm);
    CHECK(counts[0] > 0);
    if (mpi.size > 1)
        CHECK(counts[1] > 0);
    for (const auto &shape : {R::RowLayout::Dense(0, 5, 8), R::RowLayout::ColMajor(0, 5, {8, 1}), R::RowLayout::TiledColMajor(0, 5, 32)})
    {
        auto mean = metrics.BuildField(shape);
        const auto coefficientShape = shape.Kind() == R::RowLayoutKind::Table
                                          ? R::RowLayout::Dense(0, 5 * split.NRecDOF())
                                      : shape.Kind() == R::RowLayoutKind::ColMajorTable ? R::RowLayout::ColMajor(0, 5 * split.NRecDOF(), {8, 1})
                                                                                        : R::RowLayout::TiledColMajor(0, 5 * split.NRecDOF(), 32);
        auto a = split.BuildWorkspace(5, coefficientShape), b = full.BuildWorkspace(5, coefficientShape);
        auto coefficientsA = split.BuildCoefficients(5, coefficientShape), coefficientsB = full.BuildCoefficients(5, coefficientShape);
        const auto boundaryA = split.BuildBoundaryValues(5), boundaryB = full.BuildBoundaryValues(5);
        CHECK(split.NumBoundaryQuadrature() == 0);
        {
            auto u = mean.Owned().Write(), x = coefficientsA.Owned().Write(), y = coefficientsB.Owned().Write();
            const auto uv = u.View<R::UniformTableRows>(), xv = x.View<R::UniformTableRows>(), yv = y.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                const auto id = mesh->OriginalEntityId(EntityKind::Cell, iCell);
                for (DNDS::index j = 0; j < 5; ++j)
                    uv(iCell, j) = std::sin(.17 * (id + 1) * (j + 1));
                for (DNDS::index j = 0; j < xv.RowSize(iCell); ++j)
                    xv(iCell, j) = yv(iCell, j) = .03 * std::cos(.11 * (id + 1) + j);
            }
        }
        metrics.BindHalo(mean).Begin().Wait();
        auto haloA = metrics.BindHalo(coefficientsA), haloB = metrics.BindHalo(coefficientsB);
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            haloB.Begin().Wait();
            full.DoReconstructionIter(mean, coefficientsB, b, boundaryB);
            const auto version = coefficientsA.Owned().Version();
            // A real direct transfer retains input reads and ghost writes.
            // Core must neither write that input nor acquire its ghost half.
            auto transfer = haloA.Begin();
            split.DoReconstructionIterTo(mean, coefficientsA, a, boundaryA, C::ReconstructionRegion::Core);
            CHECK(coefficientsA.Owned().Version() == version);
            transfer.Wait();
            split.DoReconstructionIterTo(mean, coefficientsA, a, boundaryA, C::ReconstructionRegion::HaloDependent);
            CHECK(coefficientsA.Owned().Version() == version);
            split.CommitReconstruction(coefficientsA, a);
            auto x = coefficientsA.Owned().Read(), y = coefficientsB.Owned().Read();
            const auto xv = x.View<R::UniformTableRows>(), yv = y.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
                for (DNDS::index j = 0; j < xv.RowSize(iCell); ++j)
                    CHECK(xv(iCell, j) == yv(iCell, j));
        }
        auto alias = a;
        alias.temporary = coefficientsA;
        CHECK_THROWS(split.DoReconstructionIterTo(mean, coefficientsA, alias, boundaryA, C::ReconstructionRegion::Core));
    }
}

int main(int argc, char **argv)
{
    // Native row kernels may create OMP/CPU workers even with controller MPI.
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    DNDS::MPI::Finalize();
    return result;
}
