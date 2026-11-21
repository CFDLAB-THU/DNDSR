#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/IO/BoundaryOutputPlan.hpp"
#include "Cauchy/IO/OutputFields.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <unordered_map>

namespace DNDS::Cauchy
{
    namespace R = Runtime;

    std::shared_ptr<const BoundaryOutputGeometry> FiniteVolume::Data::PrepareBoundaryOutput() const
    {
        metrics.Validate();
        if (boundaryOutputGeometry)
            return boundaryOutputGeometry;

        auto extracted = mesh->ExtractBoundary();
        const MPIInfo mpi(metrics.Communicator());
        R::Array<BoundaryOutputRow> rows;
        std::unordered_map<index, index> byFace;
        std::vector<R::RowRequest> requests;
        std::vector<int> zones;
        std::vector<int> used(forceZones.size(), 0);
        bool remote = false;
        index ownedFaces = 0;
        MPI::LocalStage(mpi, [&]
                        {
            byFace.reserve(boundaryOutputRows.size());
            for (index row = 0; row < index(boundaryOutputRows.size()); ++row)
                byFace.emplace(boundaryOutputRows[row].face, row);
            const auto volumeState = mesh->CurrentNativeState();
            const auto identifiers = volumeState->ReadOriginalEntityIds(Geom::EntityKind::Bnd);
            DNDS_check_throw_info(identifiers, "volume mesh lacks persistent boundary IDs");
            auto native = volumeState->Snapshot(R::Device::Host()).AcquireRead();
            const auto source = native.HostView();
            const auto faceSpace = volumeState->Entities(Geom::EntityKind::Face);
            ownedFaces = faceSpace.NumOwned();
            std::unordered_map<index, index> sourceRows;
            sourceRows.reserve(mesh->NumBnd());
            for (index iBnd = 0; iBnd < mesh->NumBnd(); ++iBnd)
                sourceRows.emplace(identifiers->View()(iBnd, 0), iBnd);
            const auto extractedState = extracted->CurrentNativeState();
            const auto &provenance = extractedState->Provenance();
            DNDS_check_throw_info(provenance, "boundary output requires extraction ancestry");
            const auto ancestry = provenance->sourceIds.at(Geom::EntityKind::Cell).Read();
            const auto ids = ancestry.View<R::FixedRows<1>>();
            requests.reserve(ids.RowCount());
            zones.reserve(ids.RowCount());
            for (index iRow = 0; iRow < ids.RowCount(); ++iRow)
            {
                const index sourceBnd = sourceRows.at(ids(iRow, 0));
                const auto faces = source.BoundaryFaces(sourceBnd);
                DNDS_check_throw_info(faces.size() == 1 && faces[0] >= 0 && faces[0] < faceSpace.NumLocal(),
                                      "boundary output requires the physical face in the mesh closure");
                const auto global = faceSpace.LocalToGlobal(faces[0]);
                const int owner = faceSpace.Owner(global);
                const int zone = source.GetBndZone(sourceBnd);
                requests.push_back({owner, faceSpace.OwnerLocalRow(global), iRow});
                remote = remote || owner != mpi.rank;
                zones.push_back(zone);
                used.at(forceSlots.at(zone)) = 1;
            } });
        const bool routeRequired = MPI::AllreduceValue<MPI::Reduction::Max>(int(remote), mpi.comm) != 0;
        std::optional<R::RowTransferPlan> route;
        if (routeRequired)
        {
            auto context = metrics.Communication();
            R::RowTransferPlan discovery(context, ownedFaces, extracted->NumCell(), requests);
            std::vector<std::vector<index>> send;
            MPI::LocalStage(mpi, [&]
                            {
                send = discovery.SendRows();
                for (auto &peer : send)
                    for (auto &face : peer)
                        face = byFace.at(face); });
            route = R::RowTransferPlan::Known(context, index(boundaryOutputRows.size()), extracted->NumCell(),
                                              std::move(send), discovery.ReceiveRows());
        }
        R::Array<real> normals;
        MPI::LocalStage(mpi, [&]
                        {
            const index count = route ? index(boundaryOutputRows.size()) : extracted->NumCell();
            rows = R::Array<BoundaryOutputRow>(R::RowLayout::Dense(count, 1));
            normals = R::Array<real>(R::RowLayout::Dense(count, 3));
            auto writeRows = rows.Write();
            auto writeNormals = normals.Write();
            const auto faceOutput = writeRows.View<R::FixedRows<1>>();
            const auto normalOutput = writeNormals.View<R::FixedRows<3>>();
            for (index iRow = 0; iRow < count; ++iRow)
            {
                const auto &row = boundaryOutputRows.at(route ? iRow : byFace.at(requests[iRow].sourceRow));
                DNDS_check_throw_info(row.cell < metrics.NumCell() && (route || row.zone == zones[iRow]),
                                      "boundary output ancestry and face routing disagree");
                faceOutput(iRow, 0) = row;
                for (int d = 0; d < 3; ++d)
                    normalOutput(iRow, d) = row.normal[d] * row.area;
            } });
        MPI::AllreduceInPlace<MPI::Reduction::Max>(MPI::ContiguousView<int>{used.data(), used.size()}, mpi.comm);
        std::vector<std::pair<int, std::string>> names;
        for (index iZone = 0; iZone < index(used.size()); ++iZone)
            if (used[iZone])
            {
                const auto zone = forceZones[iZone];
                const auto begin = names.size();
                for (const auto &[name, id] : settings.mesh.boundaryIds)
                    if (zone == id)
                        names.emplace_back(zone, name);
                if (names.size() == begin)
                    names.emplace_back(zone, "");
            }
        BoundaryOutputGeometry geometry{std::move(extracted), std::move(rows), std::move(route), {}, std::make_shared<const std::vector<int>>(std::move(zones)), std::make_shared<const std::vector<std::pair<int, std::string>>>(std::move(names))};
        geometry.normals = geometry.InFileOrder(std::move(normals), metrics.Communication());
        boundaryOutputGeometry = std::make_shared<const BoundaryOutputGeometry>(std::move(geometry));
        return boundaryOutputGeometry;
    }

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
        R::EagerEngine engine;
        R::ExecutionStream stream;
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
                            if (Device() == R::Device::Host())
                                stream = engine.CreateCPUStream();
#ifdef DNDS_USE_CUDA
                            else
                                stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
#else
                            else
                                DNDS_check_throw_info(false, "CUDA visualization is not compiled");
#endif
                        });
        OutputStaging staging;
        return PackVisualization(state, reconstruction, workspace, time, hdfDeflateLevel, workers, engine, stream, staging);
    }

    PreparedVisualization FiniteVolume::PackVisualization(const EquationDofs &state, const ReconstructionState &reconstruction,
                                                          const Workspace &workspace, real time, int hdfDeflateLevel,
                                                          const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers,
                                                          R::EagerEngine &engine, const R::ExecutionStream &stream, OutputStaging &staging) const
    {
        const MPIInfo mpi(Metrics().Communicator());
        const auto &equation = _data->equation;
        auto boundary = _data->PrepareBoundaryOutput();
        std::vector<std::string> names, fluxNames;
        OutputFields outputFields;
        std::optional<R::Array<real>> boundaryFluxes;
        const bool compatible = workspace.boundaryFacialFluxCurrent && workspace.rhsState &&
                                workspace.rhsState->Current(state, workspace.incoming, time);
        const bool capture = MPI::AllreduceValue<MPI::Reduction::Min>(int(compatible), mpi.comm) != 0;
        R::Array<real> hostFields;
        MPI::LocalStage(mpi, [&]
                        {
            ValidateWorkspace(state, workspace);
            DNDS_check_throw_info(reconstruction.owner == _data, "foreign output reconstruction state");
            DNDS_check_throw_info(hdfDeflateLevel >= 0 && hdfDeflateLevel <= 9, "VTKHDF compression level must be in [0,9]");
            for (const auto &block : state.Schema().blocks)
                for (const auto &component : block.components)
                {
                    names.push_back(component.name);
                    if (capture)
                        fluxNames.push_back("flux_" + component.name + "_facial");
                }
            if (!equation.IsACM())
                for (const auto *name : {"p", "T", "u", "v", "w", "Mach"})
                    names.push_back(name);
            if (equation.HasMixture())
                for (const auto &name : equation.Mixture().SpeciesNames())
                    names.push_back("Y:" + name);
            auto sorted = names;
            std::sort(sorted.begin(), sorted.end());
            DNDS_check_throw_info(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "ambiguous equation output field names");

            // To() completes its placement copy before returning. Stage the
            // immutable row map before launching either numerical producer.
            auto rows = capture ? std::make_optional(boundary->rows.To(Device(), stream.Context())) : std::nullopt;
            outputFields = OutputFields::Submit(engine, stream, state, equation, workers);
            if (rows)
            {
                R::Array<real> packed(R::RowLayout::Dense(rows->RowCount(), state.ComponentNumber() + 3), Device());
                packed.SetName("Cauchy/output/frozenBoundaryFlux", DNDS_ALLOCATION_SITE());
                R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto rows, auto flux, auto momentum, auto quadrature, auto output)
                {
#ifdef DNDS_USE_CUDA
                    if (context.device.space == R::MemorySpace::CUDA)
                    {
                        BoundaryOutputDetail::GatherDeviceFaces(context, rows, flux, momentum, quadrature, output);
                        return;
                    }
#endif
                    for (index iRow = 0; iRow < rows.RowCount(); ++iRow)
                        BoundaryOutputDetail::GatherFace(rows(iRow, 0), flux, momentum, quadrature, output, iRow);
                }, R::ReadAs<R::FixedRows<1>>(*rows), R::ReadAs<R::TableRows>(workspace.faceFlux),
                R::ReadAs<R::FixedRows<3>>(workspace.boundaryMomentumFlux), R::ReadAs<R::TableRows>(Metrics().FaceQuadrature()),
                R::WriteAs<R::TableRows>(packed));
                boundaryFluxes.emplace(std::move(packed));
            }
            // Retire the producer prefix before immediate D2H reads. Both
            // volume and face staging use this same compute stream.
            stream.Record().WaitHost();
            hostFields = outputFields.CopyToHost(stream.Context(), staging);
            if (boundaryFluxes)
                *boundaryFluxes = staging.HostValues(*boundaryFluxes, stream.Context(), "boundaryFlux"); });
        return PreparedVisualization(_data->mesh, std::move(hostFields), std::move(names), std::move(boundary),
                                     std::move(boundaryFluxes), std::move(fluxNames), time, hdfDeflateLevel);
    }
}
