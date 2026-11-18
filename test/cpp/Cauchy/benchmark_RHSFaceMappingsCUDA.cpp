// Opt-in frozen spatial benchmark. One MPI rank and one CUDA execution stream.
// The CPU oracle and all GPU candidates read the same prepared state.
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeRHSKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>

namespace C = DNDS::Cauchy;
namespace K = C::CUDA;
namespace R = DNDS::Runtime;
namespace G = DNDS::Geom;
using I = DNDS::index;
using DNDS::real;

namespace
{
    void CheckCUDA(cudaError_t error)
    {
        DNDS_check_throw_info(error == cudaSuccess, cudaGetErrorString(error));
    }

    class EventPair
    {
        cudaEvent_t begin{}, end{};

    public:
        EventPair()
        {
            CheckCUDA(cudaEventCreate(&begin));
            if (auto error = cudaEventCreate(&end); error != cudaSuccess)
            {
                cudaEventDestroy(begin);
                CheckCUDA(error);
            }
        }
        EventPair(const EventPair &) = delete;
        EventPair &operator=(const EventPair &) = delete;
        ~EventPair()
        {
            cudaEventDestroy(begin);
            cudaEventDestroy(end);
        }
        template <class Action>
        double Time(cudaStream_t stream, int iterations, const Action &action)
        {
            CheckCUDA(cudaEventRecord(begin, stream));
            for (int i = 0; i < iterations; ++i)
                action();
            CheckCUDA(cudaEventRecord(end, stream));
            CheckCUDA(cudaEventSynchronize(end));
            float milliseconds{};
            CheckCUDA(cudaEventElapsedTime(&milliseconds, begin, end));
            return double(milliseconds) / iterations;
        }
    };

    using Clock = std::chrono::steady_clock;
    double Milliseconds(Clock::time_point from)
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - from).count();
    }

    R::Array<I> Sequence(I count)
    {
        R::Array<I> ids(R::RowLayout::Dense(count, 1));
        auto write = ids.Write();
        const auto values = write.View<R::FixedRows<1>>();
        for (I i = 0; i < count; ++i)
            values(i, 0) = i;
        return ids;
    }

    double MaxScaledError(const R::Array<real> &expected, const R::Array<real> &actual)
    {
        const auto reads = expected.Read(), checks = actual.Read();
        const auto a = reads.View<R::TableRows>(), b = checks.View<R::TableRows>();
        DNDS_check_throw_info(a.RowCount() == b.RowCount() && a.Width() == b.Width(), "benchmark output shape mismatch");
        double result = 0;
        for (I row = 0; row < a.RowCount(); ++row)
            for (I j = 0; j < a.Width(); ++j)
            {
                DNDS_check_throw_info(std::isfinite(a(row, j)) && std::isfinite(b(row, j)), "nonfinite benchmark flux");
                result = std::max(result, std::abs(a(row, j) - b(row, j)) / std::max(1., std::abs(a(row, j))));
            }
        return result;
    }

    struct Candidate
    {
        std::string name;
        K::RHSKernelControls controls;
        std::optional<K::RHSFacePackages> packages;
        double setupMs = 0;
        std::size_t packageBytes = 0;
        double faceError = 0, cellError = 0;
    };

    void EvaluateFrozenModalFlux(C::FiniteVolume::Workspace &sampling,
                                 const DNDS::CFV::ReconstructionBasisRead &frozenBasis,
                                 const C::EquationDofs &means, const C::EquationDofs &coefficients,
                                 C::EquationDofs &residual, R::Array<real> &faces, bool cached)
    {
        const auto &plan = *sampling.owner;
        const auto metric = plan.metrics.AcquireRead();
        const auto stencil = plan.stencil.Read();
        auto mean = means.ReadWithGhost(), reconstruction = coefficients.ReadWithGhost();
        auto output = residual.WriteOwned();
        auto faceWrite = faces.Write(), force = sampling.boundaryMomentumFlux.Write();
        const C::EquationViewLayout<const real> owned(means, mean), ghost(means, mean, true);
        const C::EquationViewLayout<real> destination(residual, output);
        const auto meansByBlock = C::EquationFVDetail::BlockViews(means, mean);
        const auto coefficientsByBlock = C::EquationFVDetail::BlockViews(coefficients, reconstruction);
        const std::array<I, 2> first{0, 5};
        const std::array<I, 1> vectorStart{1};
        std::optional<C::WallDistanceRead> wall;
        if (plan.wallDistance)
            wall.emplace(plan.wallDistance->AcquireRead());
        C::CPU::EquationRHSRequest request;
        request.stencil = stencil.View();
        request.metric = metric.HostView();
        request.basis = frozenBasis.EvaluationView();
        if (!cached)
        {
            // Substitute only the RHS face points. The basis still uses
            // the cell scales, centers and moments of the frozen VR plan.
            request.basis->geometry.quadrature = request.metric.quadrature;
            request.basis->cached = {};
            request.basis->cachedDerivatives = 0;
        }
        request.conditions = {plan.boundaryConditions.data(), I(plan.boundaryConditions.size())};
        request.meanBlocks = meansByBlock.HostView();
        request.recBlocks = coefficientsByBlock.HostView();
        request.logicalFirst = {first.data(), means.BlockNumber()};
        request.vectorStarts = {vectorStart.data(), 1};
        if (plan.equation.HasRANS())
        {
            request.ransMean = mean.Block("rans");
            request.ransFirst = 5;
        }
        request.faceValues = faceWrite.View<R::UniformTableRows>();
        request.wallFlux = force.View<R::FixedRows<3>>();
        request.wall = wall ? wall->HostView() : C::WallDistanceView{};
        request.thermal = sampling.thermal.View();
        request.frame = plan.motion.At(0);
        request.fluxPolicy = plan.fluxPolicy;
        request.ownedCells = plan.metrics.NumCell();
        request.width = means.ComponentNumber();
        request.flowFirst = 0;
        request.dimension = plan.metrics.Dimension();
        request.modes = request.basis->modes;
        request.viscous = plan.equation.Viscous();
        request.hasWall = bool(wall);
        auto evaluator = plan.equation.MakePointEvaluator({}, {true, true, true});
        const auto result = C::CPU::EvaluateEquationFlux(request, plan.equation, plan.boundaryPoint, plan.flux,
                                                         evaluator, owned.Descriptor(), ghost.Descriptor(),
                                                         destination.Descriptor(), destination.Descriptor(),
                                                         DNDS::CFV::ReconstructionRegion::All);
        C::RequireNumericalSuccess(result.status, "frozen modal CPU reference");
        DNDS_check_throw_info(result.value == 0, "frozen modal CPU reference used fallback");
    }

    void Run(const DNDS::MPIInfo &mpi, const std::string &meshPath, int dimension, int degree, int requestedOrder,
             const std::string &physics, bool viscous, bool periodic, bool cached, int repetitions, int batches)
    {
        const auto meshStart = Clock::now();
        G::PartitionOptions partition;
        std::optional<G::NativePeriodicImportOptions> import;
        if (periodic)
        {
            import.emplace();
            for (int d = 0; d < dimension; ++d)
                import->frames.translation[d + 1].map()[d] = 1;
        }
        G::AutoAppendName2ID names;
        auto mesh = G::ReadNativeMeshFromCGNS(mpi, dimension, meshPath, partition, [&](const std::string &name)
                                              {
                                                const auto id = G::FBC_Name_2_ID_Default(name);
                                                return periodic && id < 0 ? id : names(name); }, import);
        mesh->PrepareNativeTopology();
        const double meshMs = Milliseconds(meshStart);
        C::DomainSettings settings;
        settings.mesh.dimension = dimension;
        settings.execution.overlap = false;
        // The benchmark builds its own candidate packages after the plan is
        // placed. The preparation workspace therefore uses no CTA packages.
        settings.execution.kernels.rhsMapping = "thread";
        settings.numerics.reconstruction = degree ? "variational" : "gradient";
        settings.numerics.cfv.maxOrder = std::max(1, degree);
        // Direct gradients have midpoint face quadrature and still require
        // a valid, independent volume metric rule.
        settings.numerics.cfv.intOrder = degree ? 5 : 3;
        settings.numerics.cfv.cacheDiffBase = cached;
        settings.numerics.cfv.SORInstead = false;
        settings.numerics.cfv.storage.representation = "raw";
        settings.numerics.cfv.storage.faceCoupling = true;
        settings.numerics.reconstructionIterations = 1;
        settings.numerics.reconstructionTolerance = 0;
        settings.numerics.limiter = "none";
        settings.numerics.flux = "Roe_M1";
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.positivity = false;
        settings.numerics.faceFallback = false;
        settings.numerics.riemann.useStencil = false;
        settings.equation.rans.model = physics;
        if (viscous)
        {
            settings.equation.transport = "NewtonianFourier";
            settings.equation.viscosity = .001;
            settings.equation.rans.viscosityReference = .001;
        }
        if (!periodic)
        {
            auto scope = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
            const auto view = scope.HostView();
            for (I cell = 0; cell < view.NumCell(); ++cell)
            {
                const auto faces = view.CellFaces(cell);
                for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(faces.size()); ++slot)
                    if (view.CellFaceOther(cell, faces[slot], slot) == DNDS::UnInitIndex)
                        settings.boundaries[std::to_string(view.GetFaceZone(faces[slot]))].kind = "extrapolate";
            }
        }
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        const int components = 5 + (equation.HasRANS() ? equation.RANS().ComponentNumber() : 0);
        if (equation.HasRANS())
            mesh->BuildNodeWallDist([](auto)
                                    { return false; });

        const auto planStart = Clock::now();
        C::FiniteVolume fv(mesh, settings, equation);
        auto means = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(components));
        auto residual = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(components));
        {
            auto lease = means.CoupledField().Owned().Write();
            const auto values = lease.View<R::TableRows>();
            for (I cell = 0; cell < values.RowCount(); ++cell)
            {
                const auto x = fv.Metrics().GetCellCent(cell);
                const double angle = 2 * DNDS::pi;
                const auto u = equation.Gas().Conservative({1 + .01 * std::sin(angle * x[0]),
                                                            .25 + .04 * std::cos(angle * x[1]),
                                                            .03 * std::sin(angle * x[0]),
                                                            dimension == 3 ? .02 * std::sin(angle * x[2]) : 0., 1.});
                for (int j = 0; j < 5; ++j)
                    values(cell, j) = u[j];
                if (components > 5)
                    values(cell, 5) = .01 + .001 * std::sin(angle * x[1]);
                if (components > 6)
                    values(cell, 6) = .1;
            }
        }
        R::EagerEngine cpu;
        const auto cpuStream = cpu.CreateCPUStream();
        auto reconstruction = fv.BuildReconstructionState(means);
        auto workspace = fv.BuildWorkspace(means);
        fv.PrepareReconstruction(means, *reconstruction, *workspace, cpu, cpuStream);
        std::optional<DNDS::CFV::ReconstructionBasisRead> cpuBasis;
        if (degree)
            cpuBasis.emplace(fv.PolynomialReconstruction()->AcquireBasisRead());
        const auto &hostReconstruction = degree ? fv.FluxReconstruction(*reconstruction, *workspace).CoupledField() : workspace->directGradients.at(0);
        const double cpuSetupMs = Milliseconds(planStart);
        const R::ExecutionContext context{R::Device::CUDA(0), R::CreateCUDAStream(0)};
        const auto device = context.device;
        const auto stream = R::CUDAStream(context);
        std::optional<DNDS::CFV::VariationalReconstructionPlan> residentPolynomial;
        if (degree)
        {
            DNDS::CFV::ReconstructionResidencyOptions options;
            options.sweeps = false;
            options.basis = true;
            residentPolynomial.emplace(fv.PolynomialReconstruction()->Place(device, context, options));
        }
        const auto gpuMean = means.CoupledField().Owned().To(device, context);
        const auto gpuRec = hostReconstruction.Owned().To(device, context);
        const auto meanRead = gpuMean.Read(context), recRead = gpuRec.Read(context);
        K::RHSFields fields;
        fields.mean.owned = meanRead.View<R::UniformTableRows>();
        fields.reconstruction.owned = recRead.View<R::UniformTableRows>();

        const auto runRule = [&](int order)
        {
            const auto ruleStart = Clock::now();
            std::optional<C::FiniteVolume> alternate;
            if (degree && order != 5)
            {
                auto sampling = settings;
                sampling.numerics.cfv.cacheDiffBase = false;
                if (order == 1)
                    sampling.numerics.reconstruction = "constant";
                else
                    sampling.numerics.cfv.intOrder = order;
                alternate.emplace(mesh, sampling, equation);
            }
            const auto &spatial = alternate ? *alternate : fv;
            auto cpuFace = spatial.BuildFaceFluxBuffer(components);
            if (degree)
            {
                auto sampling = workspace;
                std::optional<C::EquationDofs> prototype;
                if (alternate)
                {
                    prototype.emplace(C::EquationDofs::Coupled(equation.Schema(), spatial.Metrics().BuildField(components)));
                    sampling = spatial.BuildWorkspace(*prototype);
                }
                EvaluateFrozenModalFlux(*sampling, *cpuBasis, means, fv.FluxReconstruction(*reconstruction, *workspace),
                                        residual, cpuFace, cached);
            }
            else
                fv.EvaluateFlux(means, residual, *reconstruction, *workspace, DNDS::CFV::ReconstructionRegion::All, 0, DNDS::OptionalRef{cpuFace});
            const double ruleCpuMs = Milliseconds(ruleStart);
            const auto placementStart = Clock::now();
            const auto placed = spatial.Placed(device, context);
            auto prototype = C::EquationDofs::Coupled(equation.Schema(), placed.Metrics().BuildField(components));
            auto placedWork = placed.BuildWorkspace(prototype);
            const auto &data = *placedWork->owner;
            const auto metric = placed.Metrics().AcquireRead(context);
            const auto incidence = data.stencil.Read(context);
            const auto conditions = data.deviceBoundaryConditions.Read(context);
            K::RHSGeometry geometry;
            geometry.stencil = incidence.View();
            geometry.faceOffsets = metric.View().faceOffsets;
            geometry.cells = metric.View().cells;
            geometry.quadrature = metric.View().quadrature;
            geometry.conditions = {conditions.View<C::PerfectGasBoundaryParameters>().data, nullptr,
                                   I(data.deviceBoundaryConditions.size()), 1, 1, nullptr, R::RowLayoutKind::Table, device};
            std::optional<DNDS::CFV::ReconstructionResidentRead> basis;
            if (degree)
            {
                basis.emplace(residentPolynomial->AcquireResidentRead());
                geometry.basis = basis->PointBasisView();
                geometry.hasBasis = true;
            }
            std::optional<C::WallDistanceRead> wall;
            if (data.wallDistance)
            {
                wall.emplace(data.wallDistance->AcquireRead(context));
                const auto v = wall->View();
                geometry.wallMean = {v.cellMean.data, v.cellMean.offsets, v.cellMean.rows, v.cellMean.width,
                                     v.cellMean.stride, v.cellMean.sizes, v.cellMean.kind, v.cellMean.device};
                geometry.wallFaces = {v.faces.data, v.faces.offsets, v.faces.rows, v.faces.width,
                                      v.faces.stride, v.faces.sizes, v.faces.kind, v.faces.device};
                geometry.hasWall = true;
            }
            const K::RHSGeometryPublication publication(geometry, context);
            K::RHSKernelParameters parameters;
            parameters.physics = C::PerfectGasSpatialParameters::FromEquation(equation, dimension);
            parameters.thermal = workspace->thermal.View();
            parameters.Validate(components);
            auto parametersStorage = R::MetadataBuffer<K::RHSKernelParameters>({parameters}).CopyTo(device, context);
            const auto parameterLease = parametersStorage.Read(context);
            const R::ArrayView<const K::RHSKernelParameters, R::AccessBoundsChecks, R::FixedRows<1>> parameterView{
                parameterLease.View<K::RHSKernelParameters>().data, nullptr, 1, 1, 1, nullptr, R::RowLayoutKind::Table, device};

            const I faces = geometry.stencil.stencilFaces.RowCount(), cells = geometry.cells.RowCount();
            const auto faceIds = Sequence(faces).To(device, context), cellIds = Sequence(cells).To(device, context);
            const auto faceIdsLease = faceIds.Read(context), cellIdsLease = cellIds.Read(context);
            const auto selectedFaces = faceIdsLease.View<R::FixedRows<1>>(), selectedCells = cellIdsLease.View<R::FixedRows<1>>();
            const double placeMs = Milliseconds(placementStart);
            std::vector<I> counts(static_cast<std::size_t>(faces));
            {
                const auto ids = data.stencil.stencilFaces.To(R::Device::Host(), context);
                const auto offsets = placed.Metrics().FaceOffsets().To(R::Device::Host(), context);
                const auto a = ids.Read(), b = offsets.Read();
                for (I i = 0; i < faces; ++i)
                {
                    const I face = a.View()(i, 0);
                    counts[std::size_t(i)] = b.View()(face + 1, 0) - b.View()(face, 0);
                }
            }

            K::RHSKernelControls reference;
            reference.components = components;
            reference.dimension = dimension;
            reference.modes = degree ? residentPolynomial->NRecDOF() : 0;
            reference.direct = reference.linear = degree == 0;
            reference.viscous = viscous;
            reference.fluxPolicy = data.fluxPolicy;
            reference.solver = data.flux.Kind();
            reference.frame = data.motion.At(0).KernelParameters();
            reference.reduction = K::RHSFaceReduction::Segmented;
            std::vector<Candidate> candidates;
            std::map<int, std::pair<K::RHSFacePackages, double>> staging;
            auto add = [&](std::string name, K::RHSKernelControls controls)
            {
                Candidate candidate{std::move(name), controls, {}, 0, 0, 0, 0};
                if (controls.mapping == K::RHSFaceMapping::CTASegments)
                {
                    auto prepared = staging.find(controls.blockSize);
                    if (prepared == staging.end())
                    {
                        const auto start = Clock::now();
                        try
                        {
                            auto packages = K::RHSFacePackages::Prepare(faceIds, {counts.data(), faces}, controls, context);
                            prepared = staging.emplace(controls.blockSize, std::make_pair(std::move(packages), Milliseconds(start))).first;
                        }
                        catch (const std::exception &error)
                        {
                            std::cout << "RHS_MAPPING_SKIP " << C::Json{{"order", order}, {"candidate", candidate.name}, {"error", error.what()}}.dump() << '\n';
                            DNDS_check_throw_info(candidate.name != "cta", "production CTA baseline must fit");
                            return;
                        }
                    }
                    candidate.packages.emplace(prepared->second.first);
                    const auto view = candidate.packages->View();
                    candidate.setupMs = prepared->second.second;
                    candidate.packageBytes = std::size_t(view.packages.RowCount()) * sizeof(K::RHSFacePackage) +
                                             std::size_t(view.segments.RowCount()) * sizeof(K::RHSFaceSegment);
                }
                candidates.push_back(std::move(candidate));
            };
            auto serial = reference;
            serial.reduction = K::RHSFaceReduction::Serial;
            auto thread = serial;
            thread.mapping = K::RHSFaceMapping::Thread;
            auto hoist = thread;
            hoist.hoistFaceRows = true;
            if (!degree)
                add("thread", thread);
            else
            {
                for (auto [base, control] : {std::make_pair(std::string("cta"), reference),
                                             std::make_pair(std::string("thread"), thread),
                                             std::make_pair(std::string("thread_hoist"), hoist)})
                    for (int width : {128, 64, 256})
                        for (int cap : {0, publication.MultiprocessorCount() * 2, publication.MultiprocessorCount() * 8})
                        {
                            auto variant = control;
                            variant.blockSize = width;
                            variant.gridBlocks = cap;
                            auto label = base;
                            if (width != 128)
                                label += "_block" + std::to_string(width);
                            if (cap)
                                label += "_grid" + std::to_string(cap);
                            add(std::move(label), variant);
                        }
                add("cta_serial", serial);
            }
            if (const char *filter = std::getenv("DNDS_RHS_MAPPING_PROFILE_ONLY"))
            {
                const auto found = std::find_if(candidates.begin(), candidates.end(),
                                                [&](const Candidate &candidate)
                                                { return candidate.name == filter; });
                DNDS_check_throw_info(found != candidates.end(), "unknown or unavailable RHS candidate");
                if (found != candidates.begin())
                    candidates.erase(std::remove_if(candidates.begin() + 1, candidates.end(),
                                                    [&](const Candidate &candidate)
                                                    { return candidate.name != filter; }),
                                     candidates.end());
                else
                    candidates.erase(candidates.begin() + 1, candidates.end());
            }

            R::Array<real> faceOutput(R::RowLayout::Dense(faces, components), device),
                cellOutput(R::RowLayout::Dense(cells, components), device), wallForce(R::RowLayout::Dense(0, 3), device);
            R::Array<DNDS::NumericalStatus> faceStatus(R::RowLayout::Dense(faces, 1), device),
                cellStatus(R::RowLayout::Dense(cells, 1), device), result(R::RowLayout::Dense(1, 1), device),
                partial(R::RowLayout::Dense(K::StateReductionScratchRows(cells + 1), 1), device);
            R::Array<I> fallback(R::RowLayout::Dense(faces, 1), device);

            auto launch = [&](const Candidate &candidate, int phase)
            {
                auto face = faceOutput.Write(context), cell = cellOutput.Write(context), force = wallForce.Write(context);
                auto fs = faceStatus.Write(context), cs = cellStatus.Write(context), rs = result.Write(context), ps = partial.Write(context);
                auto f = fallback.Write(context);
                const auto faceView = face.View<R::TableRows>();
                const auto status = fs.View<R::FixedRows<1>>();
                const auto cellView = cell.View<R::TableRows>();
                const auto cellStatusView = cs.View<R::FixedRows<1>>();
                const auto resultView = rs.View<R::FixedRows<1>>();
                const auto meansVersion = gpuMean.Version();
                auto runFaces = [&]
                {
                    K::EvaluateFluxFaces(context, publication, parameterView, candidate.controls, fields,
                                         {faceView, faceView, force.View<R::FixedRows<3>>(), status, f.View<R::FixedRows<1>>()},
                                         selectedFaces, false, candidate.packages ? candidate.packages->View() : K::RHSFacePackageView{});
                };
                auto gather = [&]
                {
                    K::GatherFluxCells(context, publication, faceView.ConstView(), faceView.ConstView(), status.ConstView(),
                                       cellView, cellView, cellStatusView, selectedCells, false);
                };
                auto full = [&]
                {
                    K::ClearNumericalStatus(context, resultView);
                    runFaces();
                    gather();
                    K::ReduceNumericalStatus(context, cellStatusView.ConstView(), resultView.ConstView(), ps.View<R::FixedRows<1>>(), resultView);
                };
                EventPair events;
                double duration = phase == 0 ? events.Time(stream, repetitions, runFaces) : phase == 1 ? events.Time(stream, repetitions, gather)
                                                                                                       : events.Time(stream, phase == 3 ? 1 : repetitions, full);
                DNDS_check_throw_info(gpuMean.Version() == meansVersion, "candidate mutated frozen means");
                return duration;
            };

            // Verify every selector exactly once, including non-default grid tails.
            // All outputs belong to this serial benchmark and are overwritten.
            std::optional<R::Array<real>> expectedFace, expectedCell;
            for (auto &candidate : candidates)
            {
                launch(candidate, 3);
                const auto hostFace = faceOutput.To(R::Device::Host()), hostCell = cellOutput.To(R::Device::Host());
                const auto ret = result.To(R::Device::Host());
                const auto counters = fallback.To(R::Device::Host());
                const auto status = ret.Read();
                DNDS_check_throw_info(bool(status.View()(0, 0)), "RHS candidate reported an invalid face/cell");
                const auto checks = counters.Read();
                for (I face = 0; face < faces; ++face)
                    DNDS_check_throw_info(checks.View()(face, 0) == 0, "benchmark input unexpectedly triggered fallback");
                if (!expectedFace)
                {
                    const auto faceError = MaxScaledError(cpuFace, hostFace);
                    const auto cellError = MaxScaledError(residual.CoupledField().Owned(), hostCell);
                    std::cout << "RHS_MAPPING_CPU " << C::Json{{"order", order}, {"face_error", faceError}, {"cell_error", cellError}}.dump() << std::endl;
                    DNDS_check_throw_info(faceError < 2e-8 && cellError < 2e-8, "GPU reference disagrees with CPU OMP1");
                    expectedFace = hostFace;
                    expectedCell = hostCell;
                }
                candidate.faceError = MaxScaledError(*expectedFace, hostFace);
                candidate.cellError = MaxScaledError(*expectedCell, hostCell);
                DNDS_check_throw_info(candidate.faceError < 1e-12 && candidate.cellError < 1e-12, "candidate disagrees with GPU reference");
            }
            const auto [minPoints, maxPoints] = std::minmax_element(counts.begin(), counts.end());
            std::cout << "RHS_MAPPING_SETUP " << C::Json{{"mesh", meshPath}, {"cells", cells}, {"faces", faces}, {"C", components}, {"M", reference.modes}, {"D", dimension}, {"degree", degree}, {"order", order}, {"points_min", *minPoints}, {"points_max", *maxPoints}, {"periodic", periodic}, {"viscous", viscous}, {"cached", cached}, {"mesh_ms", meshMs}, {"cpu_setup_ms", cpuSetupMs}, {"vr_order", settings.numerics.cfv.intOrder}, {"rule_cpu_ms", ruleCpuMs}, {"placement_ms", placeMs}, {"samples", batches}, {"repetitions", repetitions}}.dump() << std::endl;
            for (int batch = 0; batch < batches; ++batch)
                for (std::size_t i = 0; i < candidates.size(); ++i)
                {
                    const auto &candidate = candidates[batch % 2 ? candidates.size() - 1 - i : i];
                    launch(candidate, 3);
                    const double faceMs = launch(candidate, 0), gatherMs = launch(candidate, 1), fullMs = launch(candidate, 2);
                    const auto packages = candidate.packages ? candidate.packages->View() : K::RHSFacePackageView{};
                    const auto block = candidate.controls.blockSize;
                    const I available = candidate.packages ? packages.packages.RowCount() : (faces + block - 1) / block;
                    const I defaultCap = candidate.packages ? publication.MultiprocessorCount() * 4 : 1024;
                    const I cap = candidate.controls.gridBlocks ? candidate.controls.gridBlocks : defaultCap;
                    std::cout << "RHS_MAPPING_SAMPLE " << C::Json{{"order", order}, {"candidate", candidate.name}, {"batch", batch}, {"block", block}, {"grid", std::min(cap, available)}, {"grid_cap", cap}, {"packages", packages.packages.RowCount()}, {"warp_contained_packages", packages.warpContainedPackages}, {"shared_bytes", packages.sharedBytes}, {"package_bytes", candidate.packageBytes}, {"package_ms", candidate.setupMs}, {"face_ms", faceMs}, {"gather_ms", gatherMs}, {"rhs_ms", fullMs}, {"face_error", candidate.faceError}, {"cell_error", candidate.cellError}}.dump() << std::endl;
                }
        };
        if (requestedOrder)
            runRule(requestedOrder);
        else
            for (int order : {1, 3, 5})
                runRule(order);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    try
    {
        DNDS_check_throw_info(mpi.size == 1 && (argc == 10 || argc == 11),
                              "one rank: mesh dimension degree order none|SA|KOmegaSST viscous periodic repetitions batches [cached]");
        CheckCUDA(cudaSetDevice(0));
        const int dimension = std::stoi(argv[2]), degree = std::stoi(argv[3]), order = std::stoi(argv[4]);
        const int repetitions = std::stoi(argv[8]), batches = std::stoi(argv[9]);
        DNDS_check_throw_info((dimension == 2 || dimension == 3) && degree >= 0 && degree <= 3 &&
                                  (degree ? order == 0 || order == 1 || order == 3 || order == 5 : order == 1) && repetitions > 0 && batches > 0,
                              "unsupported dimension, reconstruction, quadrature, or timing work");
        const bool cached = argc == 11 && std::stoi(argv[10]) != 0;
        DNDS_check_throw_info(!cached || (order == 5 && degree), "cached comparison uses the original Q5 face rule");
        Run(mpi, argv[1], dimension, degree, order, argv[5], std::stoi(argv[6]) != 0, std::stoi(argv[7]) != 0, cached,
            repetitions, batches);
    }
    catch (const std::exception &e)
    {
        std::cerr << "RHS mapping benchmark: " << e.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
        return 1;
    }
    MPI_Finalize();
}
