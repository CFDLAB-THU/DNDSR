#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/IO/OutputFields.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <sstream>
#include <iomanip>

namespace DNDS::Cauchy
{
    void FiniteVolume::WriteVisualization(const EquationDofs &state, const ReconstructionState &reconstruction, const Workspace &workspace,
                                          const std::string &prefix, real time, int hdfDeflateLevel,
                                          const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers) const
    {
        PrepareVisualization(state, reconstruction, workspace, time, hdfDeflateLevel, workers).Write(prefix, Metrics().Communication());
    }
    PreparedVisualization FiniteVolume::PrepareVisualization(const EquationDofs &state, const ReconstructionState &reconstruction,
                                                             const Workspace &workspace, real time, int hdfDeflateLevel,
                                                             const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers) const
    {
        using namespace EquationFVDetail;
        const MPIInfo mpi(Metrics().Communicator());
        const auto &equation = _data->equation;
        const int species = equation.HasMixture() ? equation.Mixture().SpeciesNumber() : 0;
        const int independent = std::max(0, species - 1);
        std::vector<std::string> names;
        std::vector<index> first;
        index flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        std::optional<PointEvaluator> evaluator;
        std::optional<ThermalPolicy> thermal;
        Runtime::Array<real> outputFields;
        R::EagerEngine outputEngine;
        R::ExecutionStream outputStream;
        MPI::LocalStage(mpi, [&]
                        {
                            ValidateWorkspace(state, workspace);
                            DNDS_check_throw_info(hdfDeflateLevel >= 0 && hdfDeflateLevel <= 9, "VTKHDF compression level must be in [0,9]");
                            for (const auto &block : state.Schema().blocks)
                            {
                                first.push_back(index(names.size()));
                                if (block.name == "flow")
                                    flowFirst = index(names.size());
                                if (block.name == "species")
                                    speciesFirst = index(names.size());
                                if (block.name == "rans")
                                    ransFirst = index(names.size());
                                for (const auto &component : block.components)
                                    names.push_back(component.name);
                            }
                            for (const auto *name : {"p", "T", "u", "v", "w", "Mach"})
                                names.push_back(name);
                            if (species)
                                for (const auto &name : equation.Mixture().SpeciesNames())
                                    names.push_back("Y:" + name);
                            auto sorted = names;
                            std::sort(sorted.begin(), sorted.end());
                            DNDS_check_throw_info(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "ambiguous equation output field names");
                            if (Device() == R::Device::Host())
                                outputStream = outputEngine.CreateCPUStream();
#ifdef DNDS_USE_CUDA
                            else
                                outputStream = outputEngine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
#else
                            else
                                DNDS_check_throw_info(false, "CUDA visualization is not compiled");
#endif
                        });
        MPI::LocalStage(mpi, [&]
                        {
            auto prepared = OutputFields::Submit(outputEngine, outputStream, state, equation, workers);
            outputEngine.Synchronize();
            outputFields = prepared.CopyToHost(outputStream.Context()); });
        if (Device() != R::Device::Host())
        {
            auto surface = PrepareDeviceSurfaceVisualization(state, reconstruction, workspace, time, outputEngine, outputStream);
            return PreparedVisualization(_data->mesh, std::move(outputFields), std::move(names), std::move(surface), time, hdfDeflateLevel);
        }
        const auto means = state.ReadOwned();
        const auto blocks = BlockViews(state, means);
        PointScratch point(state.ComponentNumber());
        std::ostringstream surface;
        MPI::LocalStage(mpi, [&]
                        {
            const auto &limits = _data->settings.numerics;
            thermal.emplace(equation, limits.densityFloor, limits.temperatureFloor, 0, limits.pressureFloor,
                workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{});
            evaluator.emplace(equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {equation.HasRANS()}));
            surface << "cell_id,zone,x,y,z,nx,ny,nz,weight,rho,p,u,v,w,T";
            if (species) for (const auto &name : equation.Mixture().SpeciesNames()) surface << ",Y:" << name;
            surface << '\n' << std::setprecision(17);
            std::optional<EquationDofs> coefficients;
            std::optional<EquationDofs::ReadScope> rec;
            R::MetadataBuffer<EquationBlockReadView> recBlocks;
            std::optional<CFV::ReconstructionBasisRead> basis;
            if (_data->reconstruction)
            {
                coefficients.emplace(FluxReconstruction(reconstruction, workspace).NumericAlias());
                rec.emplace(coefficients->ReadOwned());
                recBlocks = BlockViews(*coefficients, *rec);
                basis.emplace(_data->reconstruction->AcquireBasisRead());
            }
            auto geometry = Metrics().AcquireRead(); const auto metric = geometry.HostView();
            CFV::Math::SmallMatrix scratch;
            const int modes = _data->reconstruction ? _data->reconstruction->NRecDOF() : 0;
            const auto frame = _data->motion.At(time);
            const auto *ransModel = equation.HasRANS() ? &equation.RANS() : nullptr;
            const auto ransMean = ransModel ? means.Block("rans") : EquationBlockReadView{};
            PointScratch cellMean(state.ComponentNumber());
            std::optional<WallDistanceRead> wallRead;
            if (ransModel && ransModel->ComponentNumber() == 2) wallRead.emplace(_data->wallDistance->AcquireRead());
            const auto walls = wallRead ? wallRead->HostView() : WallDistanceView{};
            std::optional<Runtime::Array<real>::ReadLease> incomingRead;
            InterfaceBoundaryRows incoming;
            const auto boundaryInput = PreparedBoundaryInputs(workspace);
            DNDS_check_throw_info(!HasInterfaces() || boundaryInput,
                                  "surface output needs a ready interface packet");
            if (boundaryInput)
            {
                ValidateBoundaryInputs(*boundaryInput, time);
                incomingRead.emplace((_data->stencil.quadratureBoundaryValues ? boundaryInput->QuadratureValues() : boundaryInput->FaceCenterValues())
                                         .CoupledField()
                                         .Owned()
                                         .Read());
                incoming = incomingRead->View<Runtime::UniformTableRows>();
            }
            const auto stencilRead = _data->stencil.Read();
            const auto stencil = stencilRead.View();
            const auto boundaryRead = _data->boundaryFaces.Read();
            const auto ranges = boundaryRead.View<R::FixedRows<1>>();
            for (index iBoundary = 0; iBoundary < ranges.RowCount(); ++iBoundary)
            {
                const auto range = ranges(iBoundary, 0);
                const index count = stencil.quadratureBoundaryValues ? range.quadratureCount : 1;
                for (index iLocal = 0; iLocal < count; ++iLocal)
                {
                const index iPoint = metric.FaceQuadratureBegin(range.face) + iLocal;
                const auto sample = FiniteVolumeDetail::ExpandSample(stencil.RHSFaceSample(range.face, 0, iPoint),
                    {_data->boundaryConditions.data(), index(_data->boundaryConditions.size())});
                Gather(blocks, sample.cell, point);
                if (basis)
                {
                    const auto b = basis->FaceBasis(sample.face, sample.side, sample.point, 0, scratch);
                    for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                        for (index j = 0; j < blocks[iBlock].ComponentNumber(); ++j)
                            for (int mode = 0; mode < modes; ++mode)
                                point.values[first[iBlock] + j] += recBlocks[iBlock](sample.cell, j * modes + mode) * b(0, mode);
                }
                if (!thermal->View().Admissible(point.values)) Gather(blocks, sample.cell, point);
                if (ransModel)
                    RANSReconstructionView{ransModel->Kind(), ransModel->ComponentNumber()}.RepairPoint(ransMean[sample.cell], point.values.data() + ransFirst);
                if (wallRead) Gather(blocks, sample.cell, cellMean);
                OptionalRef<const std::vector<real>> mean;
                if (wallRead)
                    mean.Emplace(cellMean.values);
                _data->boundaryPoint.Evaluate(
                    point.values, sample.boundary, sample.boundaryPoint,
                    {sample.normal[0], sample.normal[1], sample.normal[2]},
                    {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]},
                    *evaluator, frame, time, BoundaryPointStateKind::FaceTrace,
                    mean, wallRead ? walls.cellMean(sample.cell, 0) : 0,
                    incoming);
                const auto flow = point.Flow(flowFirst);
                const auto properties = evaluator->Evaluate(flow, point.Species(speciesFirst, independent));
                surface << Mesh().OriginalEntityId(Geom::EntityKind::Cell, sample.cell) << ',' << sample.zone;
                for (int d = 0; d < 3; ++d) surface << ',' << sample.ownPoint[d];
                for (int d = 0; d < 3; ++d) surface << ',' << sample.normal[d];
                surface << ',' << metric.GetFaceWeight(sample.point) << ',' << flow[0] << ',' << properties.thermo.pressure;
                for (int d = 0; d < 3; ++d) surface << ',' << flow[d + 1] / flow[0];
                surface << ',' << properties.thermo.temperature;
                for (int k = 0; k < species; ++k) surface << ',' << properties.fractions[k];
                surface << '\n';
                }
            }
            DNDS_check_throw_info(bool(surface), "equation surface output preparation failed"); });
        return PreparedVisualization(_data->mesh, std::move(outputFields), std::move(names), surface.str(), time, hdfDeflateLevel);
    }
}
