// Standalone extracted-operator diagnosis. Each invocation runs one phase.
// A real maintained SA mesh and frozen means are shared by both paths;
// preparation/halo/VR/PP are excluded unless explicitly selected. The harness
// times complete calls, not production counters. See OperatorDiagnostic.md.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Input/StateInput.hpp"
#include "Euler/EulerSolver.hpp"
#include "ExtractedEulerSA.hpp"
// Diagnostic-only TU extraction: supplies the exact production definitions.
// Static linking therefore does not pull FiniteVolume.cpp.o a second time.
// No private-access macro, extra production API or internal timer is needed.
#include "Cauchy/Discretization/FiniteVolume.cpp"
#include <iostream>

namespace C = DNDS::Cauchy;
namespace E = DNDS::Euler;
namespace R = DNDS::Runtime;
using I = DNDS::index;

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    try
    {
        for (const auto *name : {"DNDS_CAUCHY_PROFILE_PMG", "DNDS_CAUCHY_PROFILE_LINEAR",
                                 "DNDS_CAUCHY_PROFILE_EQUATION_FV", "DNDS_CAUCHY_PROFILE_FV",
                                 "DNDS_EULER_PROFILE_PMG_PHASES"})
            DNDS_check_throw_info(!std::getenv(name), "disable built-in profiling for extracted operator diagnosis");
        DNDS_check_throw_info(argc == 7,
                              "arguments: Euler-config Cauchy-config faces|source|composed|halo degree repetitions trials");
        const std::string operation = argv[3];
        const int degree = std::stoi(argv[4]), repetitions = std::stoi(argv[5]), trials = std::stoi(argv[6]);
        DNDS_check_throw_info(degree == 0 || degree == 1 || degree == 3, "diagnostic degree must be 0, 1 or 3");
        DNDS_check_throw_info(operation == "faces" || operation == "source" || operation == "composed" || operation == "halo", "unsupported SA component operation");
        DNDS_check_throw_info(operation != "halo" || degree == 3, "coefficient-halo diagnosis requires modal P3 reconstruction");
        DNDS_check_throw_info(repetitions > 0 && trials > 0, "positive spatial comparison counts required");
        std::unique_ptr<E::EulerSolver<E::NS_SA>> caseSolver;
        C::DomainSettings settings;
        {
            C::Json input;
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                std::ifstream file(argv[2]);
                DNDS_check_throw_info(bool(file), "cannot open native diagnostic config");
                input = C::Json::parse(file, nullptr, true, true); });
            const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
            DNDS_check_throw_info(config.Domains().size() == 1, "diagnostic requires one domain");
            settings = config.Domains().front().settings;
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                DNDS_check_throw_info(settings.equation.eos == "perfectGas" && settings.equation.rans.model == "SA",
                    "maintained extracted diagnostic currently requires perfect-gas SA");
                const auto equation = C::EquationSystem::Resolve(settings.equation);
                settings = C::ResolvePhysicalStateInputs(settings, equation, std::nullopt).settings; });
            caseSolver = std::make_unique<E::EulerSolver<E::NS_SA>>(mpi);
            // Match the production frontend's generated-default/case merge.
            // This run-local generated file is not maintained solver input.
            const std::string generatedDefaults = std::string(argv[1]) + ".generated-default.json";
            caseSolver->ConfigureFromJson(generatedDefaults, false);
            // Both operators borrow one actual mesh/order. No independent
            // partitioning or nonlinear advance is hidden in a phase timing.
            caseSolver->ConfigureFromJson(generatedDefaults, true, argv[1],
                                          {"/dataIOControl/meshFile", "/dataIOControl/outPltName", "/outputControl/dataOutAtInit"},
                                          {settings.mesh.file, std::string(argv[2]) + ".diagnostic", "false"});
            caseSolver->ReadMeshAndInitialize();
            // ReadMeshAndInitialize allocates U; production initializes it
            // at the start of RunImplicitEuler. Freeze that same initialized
            // state here without starting the nonlinear driver.
            caseSolver->getEval().InitializeUDOF(caseSolver->getU());
            caseSolver->getEval().RepairCellMeanState(caseSolver->getU());
        }
        auto mesh = caseSolver->getMesh();
        I cellCounts[3]{mesh->NumCell(), mesh->NumCell(), mesh->NumCell()};
        MPI_Allreduce(MPI_IN_PLACE, cellCounts, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, cellCounts + 1, 1, MPI_INT64_T, MPI_MIN, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, cellCounts + 2, 1, MPI_INT64_T, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(cellCounts[0] >= 50000 && cellCounts[1] >= 1000,
                              "operator diagnosis requires a real SA mesh with >=50000 cells and >=1000 owned cells per rank");
        if (!mpi.rank)
            std::cout << "EXTRACTED_WORKLOAD " << C::Json{{"cells", cellCounts[0]}, {"np", mpi.size}, {"minimum_owned_cells", cellCounts[1]}, {"maximum_owned_cells", cellCounts[2]}, {"Euler_config", argv[1]}, {"Cauchy_config", argv[2]}}.dump() << std::endl;
        if (!mesh->HasNodeWallDist())
            mesh->BuildNodeWallDist([&](auto zone)
                                    {
                const auto found = settings.boundaries.find(std::to_string(zone));
                return found != settings.boundaries.end() && found->second.kind == "noSlipWall"; });
        const std::string law = settings.equation.viscosityModel;
        settings.numerics.reconstruction = degree == 3 ? "variational" : "gradient";
        settings.numerics.firstOrderConvection = degree == 0;
        settings.numerics.reconstructionIterations = 0;
        settings.numerics.cfv.maxOrder = degree == 3 ? 3 : 1;
        settings.numerics.cfv.intOrder = 5;
        settings.numerics.flux = "Roe_M1";
        settings.numerics.faceFallback = settings.numerics.positivity = false;
        settings.numerics.riemann.useStencil = false;
        settings.numerics.viscousGradient = "polynomial";
        settings.execution.overlap = settings.execution.checkAccess = false;
        settings.implicit.collectiveChecks = false;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(6));
        auto result = C::EquationDofOperations::NewVector(state);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto workspace = fv.BuildWorkspace(state);
        auto reconstructionBinding = workspace->BindReconstruction(*reconstruction);
        R::Array<DNDS::real> meanSourceDiagonal(R::RowLayout::Dense(fv.Metrics().NumCell(), 6));

        using Eval = E::DiagnosticEulerSA;
        auto old = std::make_shared<DNDS::CFV::VariationalReconstruction<2>>(mpi, mesh);
        auto oldSettings = caseSolver->getVFV()->getSettings();
        nlohmann::ordered_json json;
        oldSettings.WriteIntoJson(json);
        json["maxOrder"] = degree == 3 ? 3 : 1;
        json["intOrder"] = 5;
        old->parseSettings(json);
        auto boundary = caseSolver->getBCHandler();
        Eval::InitializeFV(mesh, old, boundary);
        auto physics = caseSolver->getEval().settings;
        physics.ignoreSourceTerm = false;
        physics.rsType = E::Gas::Roe_M1;
        physics.rsFixScale = physics.rsIncFScale = 1;
        physics.useSourceGradFixGG = 0;
        Eval evaluator(mesh, old, boundary, physics);
        // This is an operator-input comparison, not a distance-provider
        // comparison. Preserve the provider discrepancy, then freeze the same
        // actual mesh-derived distance samples for both source/transport calls.
        // Do not modify either production provider or its configuration.
        double distanceDifference = 0;
        {
            const auto read = workspace->owner->wallDistance->AcquireRead();
            const auto distances = read.HostView();
            for (I iCell = 0; iCell < fv.Metrics().NumCellProc(); ++iCell)
            {
                const auto samples = distances.cells[iCell];
                DNDS_check_throw_info(degree == 3 ? I(samples.size()) == evaluator.dWall[iCell].size()
                                                  : samples.size() == 1 && evaluator.dWall[iCell].size() > 0,
                                      "wall source quadrature shape differs");
                for (I iPoint = 0; iPoint < evaluator.dWall[iCell].size(); ++iPoint)
                {
                    // Euler retains its high-order wall buffer even for a
                    // direct-FV source, then reads its mean (ig=-1). Freeze
                    // the Cauchy midpoint into that whole diagnostic row so
                    // both direct calls receive the same one-point input.
                    const auto distance = samples[degree == 3 ? iPoint : 0];
                    distanceDifference = std::max(distanceDifference, std::abs(distance - evaluator.dWall[iCell][iPoint]));
                    evaluator.dWall[iCell][iPoint] = distance;
                }
            }
            for (I iFace = 0; iFace < mesh->NumFaceProc(); ++iFace)
            {
                const auto samples = distances.faces[iFace];
                double sum = 0;
                for (auto value : samples)
                    sum += value;
                evaluator.dWallFace[iFace] = sum / double(samples.size());
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &distanceDifference, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        if (!mpi.rank)
            std::cout << "EXTRACTED_INPUT_ALIGNMENT " << C::Json{{"wall_distance", "frozen-native-mesh-samples"}, {"raw_provider_max_difference", distanceDifference}, {"production_providers_changed", false}}.dump() << std::endl;
        E::ArrayDOFV<6> oldMean, oldResult;
        E::ArrayRECV<6> oldRec;
        E::ArrayDOFV<1> beta, alpha, oldDt;
        E::JacobianDiagBlock<6> source;
        old->BuildUDof(oldMean, 6);
        old->BuildUDof(oldResult, 6);
        old->BuildURec(oldRec, 6);
        old->BuildUDof(beta, 1);
        old->BuildUDof(alpha, 1);
        old->BuildUDof(oldDt, 1);
        evaluator.uGradBuf.setConstant(0.);
        evaluator.uGradBufNoLim.setConstant(0.);
        oldRec.setConstant(0.);
        beta.setConstant(1.);
        alpha.setConstant(1.);
        // Both source calls produce the SA destruction diagonal. The full
        // coupled flow Jacobian belongs to the separate matrix phase.
        source.SetModeAndInit(0, 6, oldMean);
        {
            const auto geometry = fv.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow"), rans = write.Block("rans");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                for (int j = 0; j < 5; ++j)
                    flow(iCell, j) = oldMean[iCell][j] = caseSolver->getU()[iCell][j];
                rans(iCell, 0) = oldMean[iCell][5] = caseSolver->getU()[iCell][5];
            }
        }
        const auto ownedVersion = state.PhysicalField(0).Owned().Version();
        if (operation == "halo")
        {
            {
                auto write = reconstruction->coefficients->PhysicalField(0).Owned().Write();
                const auto values = write.View<R::UniformTableRows>();
                const int modes = fv.PolynomialReconstruction()->NRecDOF();
                for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                    for (int j = 0; j < 6; ++j)
                        for (int mode = 0; mode < modes; ++mode)
                            values(iCell, j * modes + mode) = oldRec[iCell](mode, j) =
                                .01 * (mode + 1) + .02 * j + .001 * mesh->OriginalEntityId(DNDS::Geom::EntityKind::Cell, iCell);
            }
            if (reconstruction->fluxCoefficients)
                C::EquationDofOperations::Copy(*reconstruction->fluxCoefficients, *reconstruction->coefficients);
        }
        auto meanHalo = workspace->owner->BindHalo(state.PhysicalField(0));
        std::optional<decltype(meanHalo)> coefficientHalo;
        if (reconstruction->coefficients)
            coefficientHalo.emplace(workspace->owner->BindHalo(reconstruction->coefficients->PhysicalField(0)));
        const auto haloNative = [&]
        {
            meanHalo.Begin().Wait();
            if (coefficientHalo)
                coefficientHalo->Begin().Wait();
        };
        const auto haloLegacy = [&]
        {
            oldMean.trans.startPersistentPull();
            oldMean.trans.waitPersistentPull();
            oldRec.trans.startPersistentPull();
            oldRec.trans.waitPersistentPull();
        };
        haloNative();
        haloLegacy();
        if (operation == "source" || operation == "composed")
        {
            // Keep the independently frozen centered-source oracle even
            // when the FV RHS uses modal reconstruction.
            if (!workspace->meanGradients)
                workspace->meanGradients.emplace(fv.BuildMeanGradients(state));
            for (auto &field : workspace->meanGradients->fields)
                DNDS::Algebra::Fill(field, 0.);
        }
        if (degree != 3)
        {
            // This phase comparison freezes zero physical gradients, matching
            // Euler's uGradBuf/oldRec above. Direct P0/P1 has no modal arrays;
            // do not resurrect a coefficient placeholder or mutate the means.
            DNDS_check_throw_info(!reconstruction->coefficients && !reconstruction->fluxCoefficients,
                                  "direct reconstruction unexpectedly owns modal coefficients");
            for (auto &field : workspace->directGradients)
                for (auto *array : {&field.Owned(), &field.Ghost()})
                {
                    auto write = array->Write();
                    const auto view = write.View();
                    for (I iRow = 0; iRow < view.RowCount(); ++iRow)
                        for (I j = 0; j < view.RowSize(iRow); ++j)
                            view(iRow, j) = 0;
                }
            workspace->RefreshDirectGradients(state, false, 0);
            workspace->RefreshDirectGradients(state, true, 0);
            if (!mpi.rank)
                std::cout << "EXTRACTED_GRADIENT_INPUT "
                          << C::Json{{"degree", degree}, {"physical_gradients", "frozen zero"}, {"modal_coefficients", "absent"}}.dump() << std::endl;
        }
        auto evaluatorNative = equation.MakePointEvaluator({}, {true, true, true, false, false});
        const uint64_t flags = Eval::RHS_Dont_Update_Integration | Eval::RHS_Dont_Record_Bud_Flux |
                               (degree == 3 ? 0 : Eval::RHS_Direct_2nd_Rec | Eval::RHS_Direct_2nd_Rec_already_have_uGradBufNoLim) |
                               (degree == 0 ? Eval::RHS_Direct_2nd_Rec_1st_Conv : 0);
        const auto phase = operation == "faces" ? Eval::Phase::Faces : operation == "source" ? Eval::Phase::Source
                                                                                             : Eval::Phase::Composed;
        const auto nativeFaces = [&]
        { workspace->ApplyRHS(state, result, DNDS::CFV::ReconstructionRegion::All, evaluatorNative, 0); };
        const auto nativeSource = [&]
        {
            workspace->PrepareVolumeSource(state, std::ref(result), evaluatorNative);
            auto write = meanSourceDiagonal.Write();
            const auto diagonal = write.View<R::FixedRows<6>>();
            for (I iCell = 0; iCell < diagonal.RowCount(); ++iCell)
                for (int j = 0; j < 6; ++j)
                    diagonal(iCell, j) = 0;
            workspace->AddMeanSourceJacobian(state, write, C::DiagonalStorage::Entries, 1, evaluatorNative);
        };
        const auto native = [&]
        {
            if (operation == "halo")
            {
                haloNative();
                return;
            }
            if (operation == "source")
                C::EquationDofOperations::Fill(result, 0);
            else
                nativeFaces();
            if (operation != "faces")
                nativeSource();
            // Use Euler's per-volume output contract for the timed comparison,
            // rather than hiding native's required unit conversion afterwards.
            const auto geometry = fv.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto write = result.WriteOwned();
            const auto layout = write.ComponentLayout();
            const auto values = layout.View();
            std::optional<R::Array<DNDS::real>::WriteLease> diagonalWrite;
            if (operation != "faces")
                diagonalWrite.emplace(meanSourceDiagonal.Write());
            const auto diagonal = diagonalWrite ? diagonalWrite->View<R::FixedRows<6>>() : R::ArrayView<DNDS::real, R::AccessBoundsChecks, R::FixedRows<6>>{};
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                for (int j = 0; j < 6; ++j)
                {
                    values(iCell, j) /= metric.GetCellVol(iCell);
                    if (diagonalWrite)
                        diagonal(iCell, j) /= metric.GetCellVol(iCell);
                }
        };
        const auto legacy = [&]
        {
            if (operation == "halo")
            {
                haloLegacy();
                return;
            }
            evaluator.Apply(phase, oldResult, source, oldMean, oldRec, oldRec, beta, alpha, flags);
        };
        double lastMagnitude = 0, lastDiagonalError = 0, lastDiagonalMagnitude = 0;
        const bool centerSourceJacobian = settings.implicit.sourceJacobianMode == "center";
        const auto compare = [&]
        {
            double error = 0, magnitude = 0, diagonalError = 0, diagonalMagnitude = 0;
            I worstCell = -1;
            int worstComponent = -1;
            double worstActual = 0, worstExpected = 0;
            if (operation == "halo")
            {
                const auto read = state.PhysicalField(0).Ghost().Read();
                const auto values = read.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                    for (int j = 0; j < 6; ++j)
                    {
                        const double expected = oldMean[fv.Metrics().NumCell() + iCell][j];
                        error = std::max(error, std::abs(values(iCell, j) - expected));
                        magnitude = std::max(magnitude, std::abs(expected));
                    }
                const auto coefficientRead = reconstruction->coefficients->PhysicalField(0).Ghost().Read();
                const auto rec = coefficientRead.View<R::UniformTableRows>();
                const int modes = fv.PolynomialReconstruction()->NRecDOF();
                for (I iCell = 0; iCell < rec.RowCount(); ++iCell)
                    for (int j = 0; j < 6; ++j)
                        for (int mode = 0; mode < modes; ++mode)
                        {
                            const double expected = oldRec[fv.Metrics().NumCell() + iCell](mode, j);
                            DNDS_check_throw_info(std::isfinite(rec(iCell, j * modes + mode)) && std::isfinite(expected), "nonfinite coefficient halo");
                            error = std::max(error, std::abs(rec(iCell, j * modes + mode) - expected));
                            magnitude = std::max(magnitude, std::abs(expected));
                        }
            }
            else
            {
                const auto geometry = fv.Metrics().AcquireRead();
                const auto metric = geometry.HostView();
                const auto read = result.ReadOwned();
                // ComponentLayout owns the projection descriptors borrowed by
                // View; retain it through the loop, not just its temporary view.
                const auto layout = read.ComponentLayout();
                const auto values = layout.View();
                for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                    for (int j = 0; j < 6; ++j)
                    {
                        const double actual = values(iCell, j);
                        const double expected = oldResult[iCell][j];
                        DNDS_check_throw_info(std::isfinite(actual) && std::isfinite(expected), "nonfinite extracted phase result");
                        if (std::abs(actual - expected) > error)
                        {
                            worstCell = iCell;
                            worstComponent = j;
                            worstActual = actual;
                            worstExpected = expected;
                        }
                        error = std::max(error, std::abs(actual - expected));
                        magnitude = std::max(magnitude, std::abs(expected));
                    }
                if (operation != "faces" && operation != "halo" && !centerSourceJacobian)
                {
                    // rhsQuadrature intentionally has no one-point legacy
                    // oracle. Its independent numerical oracle is the focused
                    // chemical source-Jacobian UT; this probe still requires a
                    // finite, nonzero published destruction diagonal.
                    const auto destructionRead = meanSourceDiagonal.Read();
                    const auto destruction = destructionRead.View<R::FixedRows<6>>();
                    for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                        for (int j = 0; j < 6; ++j)
                        {
                            const double actual = destruction(iCell, j);
                            DNDS_check_throw_info(std::isfinite(actual), "nonfinite quadrature source destruction diagonal");
                            diagonalMagnitude = std::max(diagonalMagnitude, std::abs(actual));
                        }
                }
                else if (operation != "faces" && operation != "halo")
                {
                    const auto destructionRead = meanSourceDiagonal.Read();
                    const auto destruction = destructionRead.View<R::FixedRows<6>>();
                    Eval::TDiffU gradient(Eval::dim, 6);
                    gradient.setZero();
                    Eval::TJacobianU sourceMatrix(6, 6);
                    for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                    {
                        // Independent legacy point-source entry: ig=-1 uses
                        // its mean wall input. The RHS's quadrature-averaged
                        // destruction is a different approximate Jacobian.
                        const auto expectedDiagonal = evaluator.source(oldMean[iCell], gradient, metric.GetCellCent(iCell), sourceMatrix, iCell, -1, 1);
                        for (int j = 0; j < 6; ++j)
                        {
                            const double actual = destruction(iCell, j);
                            const double expected = expectedDiagonal[j];
                            DNDS_check_throw_info(std::isfinite(actual) && std::isfinite(expected), "nonfinite source destruction diagonal");
                            diagonalError = std::max(diagonalError, std::abs(actual - expected));
                            diagonalMagnitude = std::max(diagonalMagnitude, std::abs(expected));
                        }
                    }
                }
            }
            double global[]{error, magnitude, diagonalError, diagonalMagnitude};
            MPI_Allreduce(MPI_IN_PLACE, global, 4, MPI_DOUBLE, MPI_MAX, mpi.comm);
            lastMagnitude = global[1];
            lastDiagonalError = global[2];
            lastDiagonalMagnitude = global[3];
            if (error == global[0] && error > 1e-10 * std::max(1., global[1]) && worstCell >= 0)
            {
                const auto wallRead = workspace->owner->wallDistance->AcquireRead();
                const auto wall = wallRead.HostView();
                C::Json nativeWall = C::Json::array(), legacyWall = C::Json::array();
                for (auto distance : wall.cells[worstCell])
                    nativeWall.push_back(distance);
                for (auto distance : evaluator.dWall[worstCell])
                    legacyWall.push_back(distance);
                std::cout << "EXTRACTED_CELL_MISMATCH " << C::Json{{"rank", mpi.rank}, {"cell", worstCell}, {"component_or_diagonal", worstComponent}, {"actual", worstActual}, {"expected", worstExpected}, {"native_wall_samples", nativeWall}, {"Euler_wall_samples", legacyWall}}.dump() << std::endl;
            }
            if (!mpi.rank && global[0] > 1e-10 * std::max(1., global[1]))
                std::cout << "EXTRACTED_MISMATCH " << C::Json{{"phase", operation}, {"degree", degree}, {"max_error", global[0]}, {"output_magnitude", global[1]}}.dump() << std::endl;
            DNDS_check_throw_info(global[0] <= 1e-10 * std::max(1., global[1]),
                                  "extracted phase mismatch: error=" + std::to_string(global[0]) +
                                      " magnitude=" + std::to_string(global[1]));
            DNDS_check_throw_info(operation != "source" || global[1] > 1e-8,
                                  "source diagnostic must exercise nonzero SA production/destruction");
            DNDS_check_throw_info(operation == "faces" || operation == "halo" || global[3] > 1e-8,
                                  "source Jacobian diagnostic must exercise a nonzero destruction diagonal");
            if (centerSourceJacobian)
                DNDS_check_throw_info(global[2] <= 1e-10 * std::max(1., global[3]), "mean-state source Jacobian differs from the legacy point-source oracle");
            return global[0];
        };
        native();
        legacy();
        double error = compare();
        // The independently compiled, unextracted Euler entry must agree with
        // the extracted composition. P1/P0 original preparation is deliberately
        // excluded: its limiter/exchange is a separate phase, not a face kernel.
        if (degree == 3 && operation != "halo")
        {
            const auto extracted = oldResult;
            E::ArrayDOFV<6> saved;
            old->BuildUDof(saved, 6);
            evaluator.Apply(Eval::Phase::Composed, oldResult, source, oldMean, oldRec, oldRec, beta, alpha, flags);
            saved = oldResult;
            evaluator.EvaluateRHS(oldResult, source, oldMean, oldRec, oldRec, beta, alpha, false, 0, flags);
            double extractionError = 0;
            for (I iCell = 0; iCell < oldMean.Size(); ++iCell)
                for (int j = 0; j < 6; ++j)
                    extractionError = std::max(extractionError, std::abs(saved[iCell][j] - oldResult[iCell][j]));
            MPI_Allreduce(MPI_IN_PLACE, &extractionError, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(extractionError < 1e-12, "Euler extraction differs from original composed operator");
        }
        I counts[6]{fv.Metrics().NumCell(), fv.Metrics().NumCellProc() - fv.Metrics().NumCell(), 0, 0, 0, 0};
        {
            const auto geometry = fv.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            const auto topologyRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
            const auto topology = topologyRead.HostView();
            const auto offsetsRead = workspace->sourceOffsets.Read();
            const auto offsets = offsetsRead.View<R::FixedRows<1>>();
            const auto stencilRead = workspace->owner->stencil.Read();
            const auto stencil = stencilRead.View();
            for (I iCell = 0; iCell < topology.NumCell(); ++iCell)
            {
                counts[4] += offsets(iCell + 1, 0) - offsets(iCell, 0);
                counts[5] += (degree == 3 ? old->GetCellQuad(iCell) : old->GetCellQuadO1(iCell)).GetNumPoints();
            }
            for (I iFace = 0; iFace < mesh->NumFaceProc(); ++iFace)
                counts[3] += (degree == 3 ? old->GetFaceQuad(iFace) : old->GetFaceQuadO1(iFace)).GetNumPoints();
            for (I iRow = 0; iRow < stencil.stencilFaces.RowCount(); ++iRow)
            {
                const I iFace = stencil.stencilFaces(iRow, 0);
                counts[2] += metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, counts, 6, MPI_INT64_T, MPI_SUM, mpi.comm);
        DNDS_check_throw_info(counts[4] == counts[5], "source quadrature workloads differ");
        const auto measure = [&](const auto &run)
        {
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            for (int repeat = 0; repeat < repetitions; ++repeat)
                run();
            const double local = (MPI_Wtime() - start) / repetitions;
            double maximum = 0;
            MPI_Allreduce(&local, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            return maximum;
        };
        for (int trial = 0; trial < trials; ++trial)
        {
            native();
            legacy(); // Restore phase outputs after composition check.
            error = compare();
            double nativeSeconds, legacySeconds;
            if (trial % 2)
                legacySeconds = measure(legacy), nativeSeconds = measure(native);
            else
                nativeSeconds = measure(native), legacySeconds = measure(legacy);
            error = compare();
            DNDS_check_throw_info(state.PhysicalField(0).Owned().Version() == ownedVersion, "phase changed frozen means");
            if (!mpi.rank)
                std::cout << "EXTRACTED_OPERATOR " << C::Json{{"phase", operation}, {"degree", degree}, {"np", mpi.size}, {"trial", trial}, {"repetitions", repetitions}, {"native_seconds", nativeSeconds}, {"euler_seconds", legacySeconds}, {"max_error", error}, {"output_magnitude", lastMagnitude}, {"fixture", "maintained-SA-mesh-initial-state"}, {"wall_distance_input", "frozen-native-mesh-samples"}, {"raw_wall_provider_difference", distanceDifference}, {"owned_cells", counts[0]}, {"ghost_rows", counts[1]}, {"native_face_point_visits", counts[2]}, {"euler_face_point_visits", counts[3]}, {"native_source_point_visits", counts[4]}, {"euler_source_point_visits", counts[5]}, {"source_diagonal_mode", settings.implicit.sourceJacobianMode}, {"native_source_jacobian_point_visits", operation == "faces" || operation == "halo" ? 0 : centerSourceJacobian ? counts[0]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    : counts[4]},
                                                              {"native_mean_eos_visits", operation == "faces" || operation == "halo" ? 0 : counts[0]},
                                                              {"mean_source_diagonal_error", centerSourceJacobian ? C::Json(lastDiagonalError) : C::Json(nullptr)},
                                                              {"mean_source_diagonal_magnitude", lastDiagonalMagnitude},
                                                              {"source_diagonal_policy", centerSourceJacobian ? "Cauchy mean-state; Euler RHS quadrature" : "Cauchy rhsQuadrature; no legacy one-point oracle"},
                                                              {"mean_width", 6},
                                                              {"coefficients", "zero-frozen"},
                                                              {"gradients", "zero-frozen"},
                                                              {"includes_halo", operation == "halo"},
                                                              {"includes_vr", false},
                                                              {"includes_pp", false},
                                                              {"residual_units", "per-volume"},
                                                              {"native_volume_normalization_timed", operation != "halo"},
                                                              {"timing", "external-whole-call-max-rank"},
                                                              {"viscosity_law", law},
                                                              {"euler_extracted_source_sha256", Eval::SourceSHA256}}
                                                          .dump()
                          << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "extracted diagnostic rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    return DNDS::MPI::Finalize();
}
