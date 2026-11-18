// Broad production-process diagnosis on a real SA mesh. Internal timer counters
// are neither read nor used to attribute cost. See OperatorDiagnostic.md.
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Input/StateInput.hpp"
#include "Euler/EulerSolver.hpp"
#include <functional>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <climits>
#include <fstream>
#include <iomanip>
#include "DiagnosticTiming.hpp"

namespace DNDS::Euler::DriverDiagnostic
{
    struct Actions
    {
        std::function<void()> reset, spatial, auxiliary, implicit, composed;
        ArrayDOFV<6> *residual = nullptr;
        ArrayDOFV<1> *pseudoStep = nullptr;
    };
    inline std::function<void(Actions &)> Run;
    inline int levels = 0, corrections = 1;
}
#include "ExtractedEulerDriver.hpp"
// Supplies the production workspace definition for diagnostic-only wall-input
// alignment, without changing the production wall-distance providers.
#include "ExtractedEquationPipeline.cpp"

namespace C = DNDS::Cauchy;
namespace E = DNDS::Euler;
namespace R = DNDS::Runtime;
using I = DNDS::index;

static void RunBatch(E::EulerSolver<E::NS_SA> &solver, C::DomainSettings settings,
                     E::DriverDiagnostic::Actions &legacy, int repetitions, int trials)
{
    const int levels = E::DriverDiagnostic::levels, corrections = E::DriverDiagnostic::corrections;
    auto mesh = solver.getMesh();
    const auto mpi = mesh->getMPI();
    I counts[]{mesh->NumCell(), mesh->NumCell(), mesh->NumCell()};
    MPI_Allreduce(MPI_IN_PLACE, counts, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, counts + 1, 1, MPI_INT64_T, MPI_MIN, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, counts + 2, 1, MPI_INT64_T, MPI_MAX, mpi.comm);
    DNDS_check_throw_info(counts[0] >= 50000 && counts[1] >= 1000,
                          "broad diagnosis requires >=50000 real SA cells and >=1000 owned cells/rank");
    const auto &oldConfig = solver.getConfiguration();
    DNDS_check_throw_info(settings.execution.device == "cpu" && settings.execution.hostThreads == 0 && settings.execution.communicationExecution == "caller",
                          "broad timing scopes require CPU-inline kernels and caller-driven MPI");
    // MPI may progress while inline core kernels run. In that mode the
    // auxiliary exchange interval includes core work: subtracting nested
    // scopes would double-count it. Only the completed batch is a comparable
    // elapsed interval; raw scope observations remain diagnostic evidence.
    const bool exclusiveTiming = !settings.execution.overlap;
    if (!mpi.rank)
        std::cout << "BROAD_TIMING_CONTRACT " << C::Json{{"overlap", settings.execution.overlap}, {"exclusive_phase_decomposition", exclusiveTiming}, {"primary_comparison", "completed-composed-correction-batch"}, {"detail_scopes", "nested-host-observations-not-additive"}}.dump() << std::endl;
    const auto &linear = oldConfig.linearSolverControl;
    DNDS_check_throw_info((linear.jacobiCode == 1 || linear.jacobiCode == 2) && (linear.gmresCode == 0 || linear.gmresCode == 1) &&
                              oldConfig.implicitReconstructionControl.nInternalRecStep == 1 &&
                              !oldConfig.limiterControl.useLimiter && oldConfig.vfvSettings.maxOrder == 3,
                          "broad batch requires a maintained P3 one-VR-sweep SGS/ILU/GMRES case");
    const auto &implicit = settings.implicit;
    // The inputs own every policy. Validate correspondence instead of silently
    // rewriting the native configuration from Euler defaults in this harness.
    DNDS_check_throw_info(implicit.pseudoCFL == oldConfig.implicitCFLControl.CFL && implicit.maxPseudoCFL == implicit.pseudoCFL &&
                              implicit.diagonal == (solver.getEval().settings.useScalarJacobian ? "entries" : "block"),
                          "explicit pseudo-time/diagonal controls differ");
    DNDS_check_throw_info(implicit.ilu.fill == linear.directPrecControl.iluCode && implicit.ilu.ordering == "rcm" &&
                              !implicit.ilu.coreFirst && linear.directPrecControl.orderingCode == 4,
                          "explicit factor controls differ");
    const auto verifyLinear = [&](const auto &old, const auto &actual)
    {
        const auto method = old.gmresCode ? "gmres" : old.jacobiCode == 1 ? "sgs"
                                                                          : "ilu";
        DNDS_check_throw_info(actual.linear.method == method && actual.sgs.exchangeBetweenSweeps == (old.jacobiCode == 1), "explicit linear/middle-exchange controls differ");
        if (old.gmresCode)
        {
            DNDS_check_throw_info(old.gmresCode == 1 && old.jacobiCode == 2 &&
                                      actual.linear.preconditioner == "ilu" && actual.ilu.preconditionerCorrections == 2 * (old.sgsIter + 1) &&
                                      actual.linear.iterations == linear.nGmresSpace * old.nGmresIter && actual.linear.subspace == linear.nGmresSpace &&
                                      actual.linear.relativeTolerance == 0 && actual.linear.absoluteTolerance == 0 && actual.linear.stopping == "residual" &&
                                      (old.gmresScale == 0 || old.gmresScale == 2) && actual.linear.componentScaling == (old.gmresScale == 2 ? "meanState" : "none"),
                                  "explicit GMRES budgets/preconditioner/metric differ");
        }
        else
            DNDS_check_throw_info(actual.linear.stopping == "fixedCount" && actual.linear.componentScaling == "none" &&
                                      actual.linear.iterations == (old.jacobiCode == 1 ? 1 : 2) * (old.sgsIter + 1),
                                  "explicit stationary correction counts differ");
    };
    verifyLinear(linear, implicit);
    const auto &pp = oldConfig.timeMarchControl;
    DNDS_check_throw_info(implicit.positivity.rhsFPPMode == (pp.rhsFPPMode == 0 ? "none" : pp.rhsFPPMode == 2 ? "dTau"
                                                                                                              : "rhs") &&
                              implicit.positivity.rhsFPPRelax == pp.rhsFPPRelax && implicit.positivity.rhsFPPScale == pp.rhsFPPScale &&
                              implicit.positivity.incrementRelax == pp.incrementPPRelax && settings.numerics.reconstructionIterations == 1 &&
                              settings.numerics.positivity == oldConfig.limiterControl.usePPRecLimiter,
                          "explicit PP/VR controls differ");
    const bool nonlinearController = levels || corrections > 1;
    if (nonlinearController)
    {
        DNDS_check_throw_info(implicit.iterations == corrections && implicit.multigrid.cycles == corrections &&
                                  implicit.multigrid.levels == levels && implicit.multigrid.preSmooth == 1 && implicit.multigrid.postSmooth == 0 &&
                                  implicit.multigrid.coarseSmooth == 4 && implicit.relativeTolerance == oldConfig.convergenceControl.rhsThresholdInternal &&
                                  implicit.normOrder == oldConfig.convergenceControl.normOrd && implicit.residualComponents == std::vector<std::string>{"rho"},
                              "explicit nonlinear/hierarchy controls differ");
        for (int level = 1; level <= levels; ++level)
        {
            const auto &coarse = linear.coarseGridLinearSolverControlList.at(std::to_string(level));
            DNDS_check_throw_info((coarse.jacobiCode == 1 || coarse.jacobiCode == 2) && coarse.gmresCode == 0,
                                  "coarse diagnosis requires stationary SGS or ILU");
            verifyLinear(coarse, implicit.ResolveCoarseLevel(level));
        }
    }
    // The borrowed Euler mesh has its evaluator's distance samples, but the
    // native FV prerequisite is the mesh-owned nodal provider. Domain setup
    // normally builds this before constructing its equation operator.
    if (!mesh->HasNodeWallDist())
        mesh->BuildNodeWallDist([&](auto zone)
                                {
            const auto found = settings.boundaries.find(std::to_string(zone));
            return found != settings.boundaries.end() && found->second.kind == "noSlipWall"; });
    // Keep the selected production overlap/check policy from the native input.
    // This driver does not replace the eager engine or bypass native wrappers.
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(6));
    auto initial = C::EquationDofOperations::NewVector(state);
    auto source = C::EquationDofOperations::NewVector(state);
    C::EquationDofOperations::Fill(source, 0.);
    {
        auto write = state.WriteOwned();
        const auto layout = write.ComponentLayout();
        const auto values = layout.View();
        for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            for (int j = 0; j < 6; ++j)
                values(iCell, j) = solver.getU()[iCell][j];
    }
    C::EquationDofOperations::Copy(initial, state);
    auto reconstruction = fv.BuildReconstructionState(state);
    auto work = fv.BuildWorkspace(state);
    double wallDifference = 0;
    {
        const auto read = work->owner->wallDistance->AcquireRead();
        const auto distances = read.HostView();
        auto &oldEval = solver.getEval();
        for (I iCell = 0; iCell < fv.Metrics().NumCellProc(); ++iCell)
        {
            const auto samples = distances.cells[iCell];
            DNDS_check_throw_info(I(samples.size()) == oldEval.dWall[iCell].size(), "wall quadrature shape differs");
            for (I iPoint = 0; iPoint < I(samples.size()); ++iPoint)
            {
                wallDifference = std::max(wallDifference, std::abs(samples[iPoint] - oldEval.dWall[iCell][iPoint]));
                if (!levels)
                    oldEval.dWall[iCell][iPoint] = samples[iPoint];
            }
        }
        for (I iFace = 0; iFace < mesh->NumFaceProc(); ++iFace)
        {
            double sum = 0;
            for (auto value : distances.faces[iFace])
                sum += value;
            if (!levels)
                oldEval.dWallFace[iFace] = sum / double(distances.faces[iFace].size());
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &wallDifference, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    R::EagerEngine engine;
    auto stream = engine.CreateCPUStream();
    C::NonlinearSolver nonlinear(fv, state, reconstruction, work, engine, stream, settings,
                                 DNDS::Algebra::ComponentSelection(6, {0}));
    std::unique_ptr<C::PolynomialMultigrid> multigrid;
    if (levels)
        multigrid = std::make_unique<C::PolynomialMultigrid>(mesh, fv, state, nonlinear, engine, stream, settings);
    const auto reset = [&]
    {
        C::EquationDofOperations::Copy(state, initial);
        fv.ResetReconstruction(*reconstruction, *work);
    };
    const auto composed = [&]
    {
        if (multigrid)
        {
            const auto report = multigrid->Solve(source, 0., 1., 0.);
            DNDS_check_throw_info(report.iterations == corrections, "PMG diagnostic did not perform its requested update count");
        }
        else if (nonlinearController)
        {
            const auto report = nonlinear.Solve(state, source, 0., 1., 0.);
            DNDS_check_throw_info(report.iterations == corrections, "diagnostic did not perform its requested update count");
        }
        else
            nonlinear.CorrectOnce(state, source, 0., 1., 0.);
    };
    auto rate = C::EquationDofOperations::NewVector(state);
    const auto compare = [&](bool spatialOutput)
    {
        const auto read = spatialOutput ? rate.ReadOwned() : state.ReadOwned();
        const auto layout = read.ComponentLayout();
        const auto values = layout.View();
        const auto geometry = fv.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        double sums[18]{};
        for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            for (int j = 0; j < 6; ++j)
            {
                const auto actual = values(iCell, j);
                const auto expected = spatialOutput ? (*legacy.residual)[iCell][j] : solver.getU()[iCell][j];
                DNDS_check_throw_info(std::isfinite(actual) && std::isfinite(expected), "nonfinite broad phase result");
                sums[j] += actual * actual;
                sums[6 + j] += expected * expected;
                sums[12 + j] += DNDS::sqr(actual - expected);
            }
        MPI_Allreduce(MPI_IN_PLACE, sums, 18, MPI_DOUBLE, MPI_SUM, mpi.comm);
        C::Json relative = C::Json::array(), nativeNorm = C::Json::array(), oldNorm = C::Json::array();
        for (int j = 0; j < 6; ++j)
        {
            relative.push_back(std::sqrt(sums[12 + j] / std::max(sums[6 + j], 1e-300)));
            nativeNorm.push_back(std::sqrt(sums[j]));
            oldNorm.push_back(std::sqrt(sums[6 + j]));
        }
        // Optional untimed, rank-local spatial evidence. Keep raw fields under
        // the run directory, not in source control. This observes every owned
        // cell once and cannot change reconstruction, fluxes or the oracle.
        if (spatialOutput)
            if (const auto *prefix = std::getenv("DNDS_DIAGNOSTIC_DUMP"))
            {
                const auto stencilRead = work->owner->stencil.Read();
                const auto stencil = stencilRead.View();
                std::ofstream dump(std::string(prefix) + "-rank" + std::to_string(mpi.rank) + ".csv");
                DNDS_check_throw_info(bool(dump), "cannot open spatial diagnostic dump");
                dump << std::setprecision(17) << "rank,cell,x,y,z,volume,boundary_zone";
                for (int j = 0; j < 6; ++j)
                    dump << ",cauchy_" << j << ",euler_" << j << ",difference_" << j;
                dump << '\n';
                for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                {
                    int boundaryZone = -1;
                    const auto faces = stencil.topology.CellFaces(iCell);
                    for (DNDS::rowsize slot = 0; slot < faces.size(); ++slot)
                    {
                        const auto sample = stencil.GradientSample(iCell, slot);
                        if (sample.other < 0)
                            boundaryZone = sample.zone;
                    }
                    const auto center = metric.GetCellBary(iCell);
                    dump << mpi.rank << ',' << iCell << ',' << center[0] << ',' << center[1] << ',' << center[2]
                         << ',' << metric.GetCellVol(iCell) << ',' << boundaryZone;
                    for (int j = 0; j < 6; ++j)
                    {
                        const auto actual = values(iCell, j), expected = (*legacy.residual)[iCell][j];
                        dump << ',' << actual << ',' << expected << ',' << actual - expected;
                    }
                    dump << '\n';
                }
                DNDS_check_throw_info(bool(dump), "spatial diagnostic dump write failed");
                // The reset starts both one-sweep reconstructions at zero.
                // Record the solver-owned native boundary input beside Euler's
                // actual reconstruction callback evaluated at that same mean.
                // This separates a boundary convention change from RHS flux
                // arithmetic without substituting either solver's values.
                const auto boundaryRead = work->boundary.ReadOwned();
                const auto boundaryLayout = boundaryRead.ComponentLayout();
                const auto boundaryValues = boundaryLayout.View();
                // Use the matching boundary packet ordering with either the
                // face-center or quadrature snapshot of the published mesh.
                const auto pointsRead = stencil.quadratureBoundaryValues ? fv.ReadBoundaryPoints() : fv.ReadBoundaryFaces();
                const auto points = pointsRead.View<R::FixedRows<1>>();
                std::ofstream boundaryDump(std::string(prefix) + "-boundary-rank" + std::to_string(mpi.rank) + ".csv");
                DNDS_check_throw_info(bool(boundaryDump), "cannot open boundary diagnostic dump");
                boundaryDump << std::setprecision(17) << "rank,cell,face,quadrature,x,y,z,zone";
                for (int j = 0; j < 6; ++j)
                    boundaryDump << ",cauchy_" << j << ",euler_" << j;
                boundaryDump << '\n';
                for (I iPoint = 0; iPoint < points.RowCount(); ++iPoint)
                {
                    const auto sample = points(iPoint, 0);
                    const int zone = sample.zone;
                    const DNDS::Geom::tPoint normal(sample.normal[0], sample.normal[1], sample.normal[2]);
                    const auto normalBase = DNDS::Geom::NormBuildLocalBaseV<3>(normal);
                    const DNDS::Geom::tPoint point(sample.point[0], sample.point[1], sample.point[2]);
                    E::EulerEvaluator<E::NS_SA>::TU left = solver.getU()[sample.cell];
                    const int quadrature = int(sample.quadrature);
                    const auto old = solver.getEval().generateBoundaryValue(
                        left, solver.getU()[sample.cell], sample.cell, sample.face, quadrature,
                        normal, normalBase, point, 0., zone, true, 1);
                    boundaryDump << mpi.rank << ',' << sample.cell << ',' << sample.face << ',' << quadrature
                                 << ',' << point[0] << ',' << point[1] << ',' << point[2] << ',' << zone;
                    for (int j = 0; j < 6; ++j)
                        boundaryDump << ',' << boundaryValues(iPoint, j) << ',' << old[j];
                    boundaryDump << '\n';
                }
                DNDS_check_throw_info(bool(boundaryDump), "boundary diagnostic dump write failed");
            }
        return C::Json{{"relative_l2_difference", relative}, {"native_l2", nativeNorm}, {"Euler_l2", oldNorm}};
    };
    // Untimed rank-sized workload metadata for arrival-imbalance diagnosis.
    // Count the actual prepared canonical face/source sets, not a uniform
    // cells-per-rank cost estimate. No counter is added to a numerical loop.
    I sourcePointNumber = 0;
    {
        const auto offsetsRead = work->sourceOffsets.Read();
        const auto offsets = offsetsRead.View<R::FixedRows<1>>();
        if (offsets.RowCount())
            sourcePointNumber = offsets(offsets.RowCount() - 1, 0);
    }
    std::array<I, 10> localWork{fv.Metrics().NumCell(), fv.Metrics().NumCellProc() - fv.Metrics().NumCell(),
                                I(work->owner->partition.Core().size()), I(work->owner->partition.HaloDependent().size()),
                                work->owner->stencil.IncidenceNumber(), work->owner->stencil.stencilFaces.RowCount(), 0, 0, 0,
                                sourcePointNumber};
    {
        const auto geometry = fv.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        const auto stencilRead = work->owner->stencil.Read();
        const auto stencil = stencilRead.View();
        for (I iRow = 0; iRow < stencil.stencilFaces.RowCount(); ++iRow)
        {
            const I iFace = stencil.stencilFaces(iRow, 0);
            const auto sample = stencil.CanonicalFaceSample(iFace, metric.FaceQuadratureBegin(iFace));
            localWork[8] += metric.FaceQuadratureEnd(sample.face) - metric.FaceQuadratureBegin(sample.face);
            if (sample.other == DNDS::UnInitIndex)
            {
                ++localWork[6];
                localWork[7] += work->owner->boundaryConditions.at(sample.condition).kind == C::BoundaryKind::NoSlipWall;
            }
        }
    }
    std::vector<I> rankWork(!mpi.rank ? localWork.size() * mpi.size : 0);
    MPI_Gather(localWork.data(), int(localWork.size()), DNDS::DNDS_MPI_INDEX,
               rankWork.data(), int(localWork.size()), DNDS::DNDS_MPI_INDEX, 0, mpi.comm);
    C::Json resolved = settings;
    if (!mpi.rank)
        std::cout << "BROAD_WORKLOAD " << C::Json{{"cells", counts[0]}, {"np", mpi.size}, {"minimum_owned_cells", counts[1]}, {"maximum_owned_cells", counts[2]}, {"rank_work_columns", {"owned_cells", "ghost_cells", "core_cells", "halo_dependent_cells", "face_incidences", "canonical_faces", "canonical_boundary_faces", "canonical_wall_faces", "canonical_face_quadrature_points", "source_quadrature_points"}}, {"rank_work_rows", rankWork}, {"wall_distance_input", levels ? "actual-production-providers" : "frozen-native-mesh-samples"}, {"raw_wall_provider_difference", wallDifference}, {"native_resolved_settings", resolved}, {"euler_resolved_settings", C::Json(oldConfig)}, {"reconstruction_start", "zero each repetition"}, {"linear_corrections", settings.implicit.linear.iterations}, {"pMG_levels", levels}, {"nonlinear_corrections", corrections}, {"nonlinear_stopping_norm_in_scope", nonlinearController}}.dump() << std::endl;
    // Record the actual owning rank of each local cell ghost. The trace
    // below reports single-node chronology; matching these owner counts
    // to a receiver's wait interval attributes delay to donor ranks
    // without treating the global arrival extremum as a peer identity.
    {
        const auto cells = mesh->Entities(DNDS::Geom::EntityKind::Cell);
        std::vector<int> localOwners(mpi.size, 0), allOwners(mpi.size * mpi.size, 0);
        for (I iGhost = 0; iGhost < cells.NumGhost(); ++iGhost)
        {
            const auto global = cells.LocalToGlobal(cells.NumOwned() + iGhost);
            const auto owner = cells.Owner(global);
            DNDS_check_throw_info(owner >= 0 && owner < mpi.size, "cell ghost owner is outside communicator");
            ++localOwners[size_t(owner)];
        }
        MPI_Allgather(localOwners.data(), mpi.size, MPI_INT, allOwners.data(), mpi.size, MPI_INT, mpi.comm);
        if (!mpi.rank)
        {
            std::vector<C::Json> rows;
            for (int receiver = 0; receiver < mpi.size; ++receiver)
                for (int owner = 0; owner < mpi.size; ++owner)
                    if (allOwners[size_t(receiver * mpi.size + owner)])
                        rows.push_back(C::Json{receiver, owner, allOwners[size_t(receiver * mpi.size + owner)]});
            std::cout << "BROAD_GHOST_OWNERS " << C::Json{{"np", mpi.size}, {"columns", {"receiver", "owner", "ghost_rows"}}, {"rows", rows}}.dump() << std::endl;
        }
    }
    // Independent untimed spatial check before any state correction. This
    // comparison normalizes through the actual production EvaluateRate block.
    reset();
    nonlinear.EvaluateRate(state, rate, 0.);
    legacy.reset();
    legacy.auxiliary();
    legacy.spatial();
    const auto spatialOracle = compare(true);
    // Preserve the failed oracle before aborting; a startup smoke must expose
    // the differing components instead of only its acceptance assertion.
    if (!mpi.rank)
        std::cout << "BROAD_SPATIAL_ORACLE " << spatialOracle.dump() << std::endl;
    if (!levels)
        for (const auto &difference : spatialOracle["relative_l2_difference"])
            DNDS_check_throw_info(difference.get<DNDS::real>() < 1e-9, "production spatial residuals disagree");

    // Untimed positive control for correction-only counters: publish an owned
    // generation with unchanged values, forcing the existing source refresh
    // guard to take its stale branch. Reset below restores the timed fixture.
    {
        auto write = state.WriteOwned();
        const auto flow = write.Block("flow");
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            flow(iCell, 0) = flow(iCell, 0);
    }
    {
        DiagnosticTiming::Sample audit;
        DiagnosticTiming::active = &audit;
        auto auxiliary = fv.BuildAuxiliaryValues({true, true});
        fv.PrepareCorrectionAux(state, *reconstruction, *work, auxiliary, engine, stream, 0.);
        DiagnosticTiming::active = nullptr;
        DNDS_check_throw_info(std::accumulate(audit.correctionSourceCalls.begin(), audit.correctionSourceCalls.end(), DNDS::index(0)) == 1 &&
                                  std::accumulate(audit.correctionSourcePoints.begin(), audit.correctionSourcePoints.end(), DNDS::index(0)) == sourcePointNumber,
                              "correction source audit did not observe the positive control");
    }

    reset();
    composed();
    legacy.reset();
    legacy.composed();
    const auto baseline = compare(false);
    const auto measure = [&](bool native)
    {
        const auto *traceSetting = std::getenv("DNDS_CAUCHY_DIAGNOSTIC_AUX_TRACE");
        const bool traceAuxiliary = native && traceSetting && std::string(traceSetting) == "1";
        std::array<double, 8> total{};
        std::array<double, 33> perLevel{};
        std::array<DNDS::index, 6> sourceRefreshes{};
        constexpr std::array<const char *, DiagnosticTiming::Sample::detailNumber> detailNames{
            "mean-validation", "defect-assembly", "pseudo-time-pp", "correction-local",
            "candidate-acceptance", "correction-statistics", "gradient", "reconstruction",
            "polynomial-pp", "face-rhs", "source-sampling", "volume-source",
            "candidate-validation", "candidate-agreement", "jacobian-setup", "jacobian-freeze",
            "jacobian-source", "jacobian-faces"};
        std::array<double, DiagnosticTiming::Sample::detailNumber * 3> details{};
        for (int repeat = 0; repeat < repetitions; ++repeat)
        {
            if (native)
                reset();
            else
                legacy.reset();
            MPI_Barrier(mpi.comm);
            DiagnosticTiming::Sample observation;
            if (traceAuxiliary)
            {
                observation.traceAuxiliary = true;
                // At most sixteen auxiliary entries per requested finest
                // update across this bounded hierarchy; fail on overflow.
                DNDS_check_throw_info(corrections <= (INT_MAX / 16 - 32), "auxiliary trace capacity overflow");
                observation.auxiliaryEvents.reserve(size_t(16 * (corrections + 32)));
            }
            DiagnosticTiming::active = &observation;
            const double start = MPI_Wtime();
            if (native)
                composed();
            else
                legacy.composed();
            if (native)
                stream.Record().WaitHost();
            const double elapsed = MPI_Wtime() - start;
            DiagnosticTiming::active = nullptr;
            if (traceAuxiliary)
            {
                // Collection occurs after elapsed is frozen. No trace-related
                // collective is placed in the measured production sequence.
                const int width = int(DiagnosticTiming::Sample::AuxiliaryEvent{}.size());
                const int count = int(observation.auxiliaryEvents.size()) * width;
                std::vector<int> counts(!mpi.rank ? mpi.size : 0), offsets(!mpi.rank ? mpi.size : 0);
                MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, mpi.comm);
                int overflow = observation.traceOverflow;
                MPI_Allreduce(MPI_IN_PLACE, &overflow, 1, MPI_INT, MPI_MAX, mpi.comm);
                DNDS_check_throw_info(!overflow, "auxiliary trace exceeded preallocated diagnostic capacity");
                std::vector<int64_t> local;
                local.reserve(size_t(count));
                for (const auto &event : observation.auxiliaryEvents)
                    local.insert(local.end(), event.begin(), event.end());
                std::vector<int64_t> records;
                if (!mpi.rank)
                {
                    I total = 0;
                    for (int rank = 0; rank < mpi.size; ++rank)
                    {
                        DNDS_check_throw_info(total <= INT_MAX - counts[rank], "auxiliary trace gather exceeds MPI range");
                        offsets[rank] = int(total);
                        total += counts[rank];
                    }
                    records.resize(size_t(total));
                }
                MPI_Gatherv(local.data(), count, MPI_INT64_T, records.data(), counts.data(), offsets.data(), MPI_INT64_T, 0, mpi.comm);
                if (!mpi.rank)
                    std::cout << "BROAD_AUXILIARY_TRACE " << C::Json{{"solver", "Cauchy"}, {"repeat", repeat}, {"np", mpi.size}, {"clock", "OS-steady-nanoseconds-single-node-only"}, {"row_width", width}, {"rank_value_counts", counts}, {"columns", {"level", "ordinal", "path", "entered", "launch_begin", "launch_end", "boundary_end", "wait_begin", "wait_end", "compute_begin", "compute_end"}}, {"paths", {"full-evaluation", "correction-only"}}, {"rank_rows", records}}.dump() << std::endl;
            }
            DNDS_check_throw_info(observation.spatialCalls >= 1 && observation.auxiliaryCalls > 0,
                                  "broad production scopes did not observe one complete pipeline");
            const double spatial = exclusiveTiming ? observation.spatialInclusive - (native ? observation.auxiliary : 0.)
                                                   : observation.spatialInclusive;
            const double correction = exclusiveTiming ? elapsed - observation.auxiliary - spatial : 0.;
            DNDS_check_throw_info(spatial >= 0 && correction >= 0, "invalid synchronous timing scopes");
            total[0] += observation.auxiliary;
            total[1] += spatial;
            total[2] += correction;
            total[3] += elapsed;
            const double matrixLinear = std::accumulate(observation.levelMatrixLinear.begin(), observation.levelMatrixLinear.end(), 0.);
            total[4] += matrixLinear;
            total[5] += exclusiveTiming ? correction - matrixLinear : 0.;
            total[6] += std::accumulate(observation.levelMatrix.begin(), observation.levelMatrix.end(), 0.);
            total[7] += std::accumulate(observation.levelLinear.begin(), observation.levelLinear.end(), 0.);
            DNDS_check_throw_info(matrixLinear >= 0 && (!exclusiveTiming || correction + 1e-9 >= matrixLinear), "matrix/linear scopes exceed implicit remainder");
            for (int level = 0; level < 3; ++level)
            {
                sourceRefreshes[level] += observation.correctionSourceCalls[level];
                sourceRefreshes[3 + level] += observation.correctionSourcePoints[level];
                for (int detail = 0; detail < DiagnosticTiming::Sample::detailNumber; ++detail)
                    details[3 * detail + level] += observation.details[detail][level];
                perLevel[2 * level] += observation.levelAuxiliary[level];
                perLevel[2 * level + 1] += observation.levelSpatialInclusive[level] -
                                           (native && exclusiveTiming ? observation.levelAuxiliary[level] : 0.);
                perLevel[6 + 2 * level] += observation.levelAuxiliaryCalls[level];
                perLevel[7 + 2 * level] += observation.levelSpatialCalls[level];
                perLevel[12 + level] += observation.levelMatrixLinear[level];
                perLevel[15 + level] += observation.levelMatrixLinearCalls[level];
                perLevel[18 + level] += observation.levelMatrix[level];
                perLevel[21 + level] += observation.levelLinear[level];
                perLevel[24 + level] += observation.levelAuxiliaryPrepare[level];
                perLevel[27 + level] += observation.levelAuxiliaryExchange[level];
                perLevel[30 + level] += observation.levelAuxiliaryCompute[level];
            }
        }
        for (auto &value : total)
            value /= repetitions;
        for (auto &value : perLevel)
            value /= repetitions;
        for (auto &value : details)
            value /= repetitions;
        std::vector<double> detailRecords(!mpi.rank ? details.size() * mpi.size : 0);
        MPI_Gather(details.data(), int(details.size()), MPI_DOUBLE, detailRecords.data(), int(details.size()), MPI_DOUBLE, 0, mpi.comm);
        std::vector<DNDS::index> sourceRefreshRecords(!mpi.rank ? sourceRefreshes.size() * mpi.size : 0);
        MPI_Gather(sourceRefreshes.data(), int(sourceRefreshes.size()), DNDS::DNDS_MPI_INDEX, sourceRefreshRecords.data(), int(sourceRefreshes.size()), DNDS::DNDS_MPI_INDEX, 0, mpi.comm);
        if (!mpi.rank)
            std::cout << "BROAD_DETAIL_TIMES " << C::Json{{"solver", native ? "Cauchy" : "Euler"}, {"instrumented", native}, {"scope_names", detailNames}, {"levels", {"P3", "P1", "P0"}}, {"rank_rows", detailRecords}, {"source_refresh_columns", {"P3calls", "P1calls", "P0calls", "P3points", "P1points", "P0points"}}, {"source_refresh_rank_rows", sourceRefreshRecords}, {"source_refresh_repetitions", repetitions}, {"nested_scopes", {{"mean-validation", "candidate-validation"}, {"candidate-validation", "candidate-acceptance"}, {"candidate-agreement", "candidate-acceptance"}, {"source-sampling", "volume-source"}}}, {"jacobian_source_scope", "fused-mean-source-and-remaining-eos"}, {"timing", "external-same-production-calls"}}.dump() << std::endl;
        std::array<double, 37> rankRecord{};
        std::copy_n(total.begin(), 4, rankRecord.begin());
        std::copy(perLevel.begin(), perLevel.end(), rankRecord.begin() + 4);
        std::vector<double> rankRecords(!mpi.rank ? 37 * mpi.size : 0);
        MPI_Gather(rankRecord.data(), 37, MPI_DOUBLE, rankRecords.data(), 37, MPI_DOUBLE, 0, mpi.comm);
        if (!mpi.rank && exclusiveTiming)
            std::cout << "BROAD_RANK_TIMES " << C::Json{{"solver", native ? "Cauchy" : "Euler"}, {"columns", {"aux", "reconstructionRHS", "implicit", "total", "P3aux", "P3RHS", "P1aux", "P1RHS", "P0aux", "P0RHS", "P3auxScopes", "P3RHScalls", "P1auxScopes", "P1RHScalls", "P0auxScopes", "P0RHScalls", "P3matrixLinear", "P1matrixLinear", "P0matrixLinear", "P3matrixLinearCalls", "P1matrixLinearCalls", "P0matrixLinearCalls", "P3matrix", "P1matrix", "P0matrix", "P3linear", "P1linear", "P0linear", "P3auxPrepare", "P1auxPrepare", "P0auxPrepare", "P3auxExchange", "P1auxExchange", "P0auxExchange", "P3auxCompute", "P1auxCompute", "P0auxCompute"}}, {"rank_rows", rankRecords}}.dump() << std::endl;
        // These are maxima of local sums. The critical rank can differ across
        // blocks, so do not add reduced blocks to claim a composed interval.
        MPI_Allreduce(MPI_IN_PLACE, total.data(), 8, MPI_DOUBLE, MPI_MAX, mpi.comm);
        return total;
    };
    for (int trial = 0; trial < trials; ++trial)
    {
        std::array<double, 8> nativeSeconds{}, oldSeconds{};
        if (trial % 2)
            oldSeconds = measure(false), nativeSeconds = measure(true);
        else
            nativeSeconds = measure(true), oldSeconds = measure(false);
        const auto outputs = compare(false);
        DNDS_check_throw_info(outputs == baseline, "broad frozen-repeat output changed");
        const std::array<const char *, 8> phases{"auxiliary", "reconstruction-rhs", "implicit-correction", "composed-correction", "matrix-linear", "implicit-other", "matrix", "linear"};
        for (int iPhase = 0; iPhase < 8; ++iPhase)
        {
            // Do not publish an exclusive decomposition for overlapping work.
            // BROAD_* scope records are raw host observations, not additive
            // costs. The complete batch includes all nonlinear synchronization.
            if (!exclusiveTiming && iPhase != 3)
                continue;
            if (!mpi.rank)
                std::cout << "BROAD_OPERATOR " << C::Json{{"phase", phases[iPhase]}, {"trial", trial}, {"repetitions", repetitions}, {"native_seconds", nativeSeconds[iPhase]}, {"euler_seconds", oldSeconds[iPhase]}, {"outputs", outputs}, {"timing", "external-whole-call-max-rank"}, {"preserves_production_call_sequence", true}, {"Euler_ODE_arithmetic", "production-BDF1-Step"}, {"pMG_levels", levels}, {"nonlinear_corrections", corrections}, {"nonlinear_stopping_norm_in_scope", nonlinearController}, {"reset_and_prerequisites_timed", false}, {"internal_counters_used", false}}.dump() << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    try
    {
        DNDS_check_throw_info(argc == 5 || argc == 7, "arguments: Euler-config Cauchy-config repetitions trials [MG-levels nonlinear-corrections]");
        const int repetitions = std::stoi(argv[3]), trials = std::stoi(argv[4]);
        if (argc == 7)
        {
            E::DriverDiagnostic::levels = std::stoi(argv[5]);
            E::DriverDiagnostic::corrections = std::stoi(argv[6]);
        }
        DNDS_check_throw_info(repetitions > 0 && trials > 0, "positive batch counts required");
        DNDS_check_throw_info(E::DriverDiagnostic::levels >= 0 && E::DriverDiagnostic::levels <= 2 &&
                                  E::DriverDiagnostic::corrections > 0 && E::DriverDiagnostic::corrections <= 32,
                              "diagnostic supports zero/one/two MG levels and 1..32 corrections");
        C::Json input;
        DNDS::MPI::LocalStage(mpi, [&]
                              {
            std::ifstream file(argv[2]);
            DNDS_check_throw_info(bool(file), "cannot open broad native input");
            input = C::Json::parse(file, nullptr, true, true); });
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        DNDS_check_throw_info(config.Domains().size() == 1, "broad diagnosis requires one domain");
        auto settings = config.Domains().front().settings;
        {
            const auto *trace = std::getenv("DNDS_CAUCHY_DIAGNOSTIC_AUX_TRACE");
            const std::string mode = trace ? trace : "0";
            int enabled = mode == "1", bounds[2]{enabled, enabled};
            // Agree diagnostic policy before entering solver work; no trace
            // agreement is introduced inside the nonlinear update sequence.
            MPI_Allreduce(MPI_IN_PLACE, bounds, 1, MPI_INT, MPI_MIN, mpi.comm);
            MPI_Allreduce(MPI_IN_PLACE, bounds + 1, 1, MPI_INT, MPI_MAX, mpi.comm);
            DNDS_check_throw_info((mode == "0" || mode == "1") && bounds[0] == bounds[1], "auxiliary trace mode differs across ranks");
        }
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        settings = C::ResolvePhysicalStateInputs(settings, equation, std::nullopt).settings;
        DNDS_check_throw_info(equation.HasRANS() && !equation.HasMixture(), "broad diagnosis requires perfect-gas SA");
        E::EulerSolver<E::NS_SA> solver(mpi);
        const std::string defaults = std::string(argv[1]) + ".generated-default.json";
        solver.ConfigureFromJson(defaults, false);
        solver.ConfigureFromJson(defaults, true, argv[1]);
        solver.ReadMeshAndInitialize();
        E::DriverDiagnostic::Run = [&](auto &actions)
        { RunBatch(solver, settings, actions, repetitions, trials); };
        solver.RunImplicitEuler();
        E::DriverDiagnostic::Run = {};
    }
    catch (const std::exception &error)
    {
        std::cerr << "broad diagnostic rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    return DNDS::MPI::Finalize();
}
