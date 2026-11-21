#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/MeanStateCheck.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "Geom/Interface/BoundaryInterface.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        /// Materialize a requested matching/debug snapshot under its caller's
        /// owner. Ordinary boundary kernels use the compact CFV/face registry;
        /// no persistent duplicate point/normal table belongs to FiniteVolume.
        R::Array<BoundaryPoint> BoundarySnapshot(Geom::UnstructuredMesh &mesh,
                                                 const Detail::BoundaryPointLayout &layout, bool centers)
        {
            const auto meshRead = mesh.AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto topology = meshRead.HostView();
            const auto faceRead = layout.faces.Read();
            const auto faces = faceRead.View<R::FixedRows<1>>();
            R::Array<BoundaryPoint> result(R::RowLayout::Dense(centers ? faces.RowCount() : layout.quadratureRows, 1));
            auto write = result.Write();
            const auto output = write.template View<R::FixedRows<1>>();
            for (index iBoundary = 0; iBoundary < faces.RowCount(); ++iBoundary)
            {
                const auto range = faces(iBoundary, 0);
                if (!centers && !range.quadratureCount)
                    continue;
                const auto element = topology.GetFaceElement(range.face);
                Geom::tSmallCoords coords;
                topology.GetCoordsOnFace(range.face, coords);
                Geom::tPoint meanNormal = Geom::tPoint::Zero(), geometricNormal = Geom::tPoint::Zero();
                const auto normalAt = [&](const auto &shape)
                {
                    const auto jacobian = Geom::Elem::ShapeJacobianCoordD01Nj(coords, shape);
                    return topology.Dimension() == 2 ? Geom::FacialJacobianToNormVec<2>(jacobian) : Geom::FacialJacobianToNormVec<3>(jacobian);
                };
                if (centers)
                {
                    const Geom::Elem::Quadrature measure(element, layout.meanIntegrationOrder);
                    for (int q = 0; q < measure.GetNumPoints(); ++q)
                    {
                        const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, layout.meanIntegrationOrder, q);
                        const auto normal = normalAt(shape);
                        const auto point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                        geometricNormal += normal * measure.GetWeight(q);
                        meanNormal += normal * measure.GetWeight(q) * (layout.axisymmetric ? point[1] : 1.);
                    }
                    if (meanNormal.squaredNorm() == 0)
                        meanNormal = geometricNormal;
                    DNDS_check_throw_info(meanNormal.allFinite() && meanNormal.norm() > 0, "invalid boundary snapshot normal");
                    meanNormal.normalize();
                }
                const int count = centers ? 1 : int(range.quadratureCount);
                for (int q = 0; q < count; ++q)
                {
                    const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, centers ? 1 : layout.integrationOrder, q);
                    const Geom::tPoint point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                    const Geom::tPoint normal = centers ? meanNormal : Geom::tPoint(normalAt(shape).normalized());
                    const index row = centers ? iBoundary : range.quadratureBegin + q;
                    const int zone = topology.GetFaceZone(range.face);
                    output(row, 0) = {topology.FaceCells(range.face)[0], range.face, centers ? -1 : q, zone, {point[0], point[1], point[2]}, {normal[0], normal[1], normal[2]}, range.interface};
                }
            }
            return result;
        }
        R::ExecutionStream PacketStream(R::EagerEngine &engine, R::Device device,
                                        const std::optional<R::ExecutionStream> &requested)
        {
            if (requested)
            {
                DNDS_check_throw_info(engine.Owns(*requested) && requested->Context().device == device, "boundary packet execution lane differs");
                return *requested;
            }
            if (device == R::Device::Host())
                return engine.CreateCPUStream();
#ifdef DNDS_USE_CUDA
            return engine.CreateStream(std::make_shared<R::CUDALauncher>(device.ordinal));
#else
            DNDS_check_throw_info(false, "CUDA boundary packet execution is unavailable");
            return {};
#endif
        }
        EquationDofs BuildPacketRows(const EquationSchema &schema, index rows,
                                     const std::string &name, R::Device device)
        {
            index components = 0;
            for (const auto &block : schema.blocks)
                components += index(block.components.size());
            const Algebra::DofSpace space(size_t(rows), 0, name);
            return EquationDofs::Coupled(
                schema,
                Algebra::DofField<real>(
                    space, Runtime::RowLayout::Dense(0, components), device),
                "equation");
        }
        void Snapshot(const EquationDofs &values, std::vector<uint64_t> &versions,
                      std::vector<uint64_t> &shapes)
        {
            versions.clear();
            shapes.clear();
            versions.reserve(size_t(values.PhysicalFieldNumber()));
            shapes.reserve(size_t(values.PhysicalFieldNumber()));
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
            {
                const auto &array = values.PhysicalField(iField).Owned();
                versions.push_back(array.Version());
                shapes.push_back(array.ShapeVersion());
            }
        }
        bool Current(const EquationDofs &values,
                     const std::vector<uint64_t> &versions,
                     const std::vector<uint64_t> &shapes)
        {
            if (versions.size() != size_t(values.PhysicalFieldNumber()) ||
                shapes.size() != versions.size())
                return false;
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
            {
                const auto &array = values.PhysicalField(iField).Owned();
                if (versions[size_t(iField)] != array.Version() ||
                    shapes[size_t(iField)] != array.ShapeVersion())
                    return false;
            }
            return true;
        }
    }
    Geom::BoundaryPatch FiniteVolume::BuildBoundaryPatch(int zone) const
    {
        std::vector<index> selected;
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
        auto read=_data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto view=read.HostView();
        for(index iBnd=0;iBnd<view.NumBnd();++iBnd)
            if(view.GetBndZone(iBnd)==zone) selected.push_back(iBnd); });
        return Geom::BoundaryPatch::Create(_data->mesh, selected);
    }
    std::vector<BoundaryPoint> FiniteVolume::BoundaryPoints() const
    {
        Metrics().Validate();
        const auto read = ReadBoundaryPoints();
        const auto view = read.View<Runtime::FixedRows<1>>();
        std::vector<BoundaryPoint> result(view.RowCount());
        for (index iRow = 0; iRow < view.RowCount(); ++iRow)
            result[iRow] = view(iRow, 0);
        return result;
    }
    std::vector<BoundaryPoint> FiniteVolume::BoundaryFaces() const
    {
        Metrics().Validate();
        const auto read = ReadBoundaryFaces();
        const auto view = read.View<Runtime::FixedRows<1>>();
        std::vector<BoundaryPoint> result(view.RowCount());
        for (index iRow = 0; iRow < view.RowCount(); ++iRow)
            result[iRow] = view(iRow, 0);
        return result;
    }
    Runtime::Array<BoundaryPoint>::ReadLease FiniteVolume::ReadBoundaryPoints() const
    {
        Metrics().Validate();
        return BoundarySnapshot(*_data->mesh, *_data->boundaryLayout, false).Read();
    }
    Runtime::Array<BoundaryPoint>::ReadLease FiniteVolume::ReadBoundaryFaces() const
    {
        Metrics().Validate();
        return BoundarySnapshot(*_data->mesh, *_data->boundaryLayout, true).Read();
    }
    bool FiniteVolume::HasInterfaces() const { return _data->hasInterfaces; }
    BoundaryInputs FiniteVolume::BuildBoundaryInputs(
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
        std::optional<R::ExecutionStream> stream) const
    {
        Metrics().Validate();
        DNDS_check_throw_info(_data->equation.HasMixture() == bool(workers),
                              "interface packet needs exactly its equation chemistry backend");
        const auto rans = _data->equation.HasRANS()
                              ? RANSStateView{_data->equation.RANS().Kind(), _data->equation.RANS().ComponentNumber()}
                              : RANSStateView{};
        auto admissibility = std::make_shared<const AdmissibilityPolicy>(
            _data->equation, _data->settings.numerics.densityFloor,
            _data->settings.numerics.temperatureFloor,
            _data->settings.numerics.sensibleEnergyFloor,
            _data->settings.numerics.pressureFloor, rans,
            workers ? workers->Acquire()
                    : Physics::Chemistry::ChemicalWorkerPool::Lease{});
        BoundaryInputs input(
            BuildPacketRows(_data->equation.Schema(),
                            _data->boundaryLayout->quadratureRows,
                            "boundary quadrature", Device()),
            BuildPacketRows(_data->equation.Schema(),
                            _data->boundaryLayout->faces.RowCount(),
                            "boundary face centers", Device()),
            _data->equation.PropertyKey(), std::move(admissibility));
        input._layout = _data->boundaryLayout;
        BeginBoundaryUpdate(input, stream);
        return input;
    }
    BoundaryInputs FiniteVolume::CopyBoundaryRows(
        const BoundaryInputs &input, R::EagerEngine &engine, const R::ExecutionStream &lane) const
    {
        ValidateBoundaryInputs(input, input.Time());
        BoundaryInputs frozen(
            EquationDofOperations::NewVector(input._quadrature),
            EquationDofOperations::NewVector(input._faceCenters),
            input._property, input._admissibility);
        frozen._layout = input._layout;
        frozen._consumerLayout = _data->boundaryLayout;
        frozen._publication = std::make_shared<BoundaryInputs::PublicationSlot>();
        EquationDofOperations::Copy(engine, lane, frozen._quadrature, input._quadrature);
        EquationDofOperations::Copy(engine, lane, frozen._faceCenters, input._faceCenters);
        return frozen;
    }
    BoundaryInputs FiniteVolume::FreezeBoundaryInputs(
        const BoundaryInputs &input, std::optional<R::ExecutionStream> stream) const
    {
        auto engine = stream ? stream->LockEngine() : R::EagerEngine{};
        const auto lane = PacketStream(engine, Device(), stream);
        auto frozen = CopyBoundaryRows(input, engine, lane);
        PublishBoundaryInputs(frozen, input.Time(), input.Epoch(),
                              input.InterfaceSources(), lane);
        return frozen;
    }
    void FiniteVolume::ValidateBoundaryLayout(const BoundaryInputs &input) const
    {
        Metrics().Validate();
        DNDS_check_throw_info(input._layout && input._layout->stamp.Valid() &&
                                  (input._layout == _data->boundaryLayout || input._consumerLayout == _data->boundaryLayout),
                              "boundary packet point order or geometry differs");
        DNDS_check_throw_info(input._property == _data->equation.PropertyKey() &&
                                  input._admissibility,
                              "boundary packet physical model differs");
        const auto validate = [&](const EquationDofs &values, index rows)
        {
            values.RequireSchema(_data->equation.Schema());
            DNDS_check_throw_info(values.Device() == Device() &&
                                      values.Space().OwnedRows() == size_t(rows) &&
                                      values.Space().GhostRows() == 0,
                                  "boundary packet requires compatible equation rows and placement");
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                DNDS_check_throw_info(
                    !values.PhysicalField(iField).Owned().Layout().IsVariable(),
                    "boundary packet field needs uniform rows");
        };
        validate(input._quadrature, input._layout->quadratureRows);
        validate(input._faceCenters, input._layout->faces.RowCount());
    }
    void FiniteVolume::BeginBoundaryUpdate(BoundaryInputs &input, std::optional<R::ExecutionStream> stream) const
    {
        ValidateBoundaryLayout(input);
        DNDS_check_throw_info(input._layout == _data->boundaryLayout,
                              "only the provider may begin an aliased boundary update");
        input._publication = std::make_shared<BoundaryInputs::PublicationSlot>();
        if (Device() != R::Device::Host() || stream)
        {
            auto engine = stream ? stream->LockEngine() : R::EagerEngine{};
            const auto lane = PacketStream(engine, Device(), stream);
            EquationDofOperations::Fill(engine, lane, input._quadrature, std::numeric_limits<real>::quiet_NaN());
            EquationDofOperations::Fill(engine, lane, input._faceCenters, std::numeric_limits<real>::quiet_NaN());
            if (!stream)
                engine.Synchronize();
            return;
        }
        for (auto *values : {&input._quadrature, &input._faceCenters})
        {
            auto write = values->WriteOwned();
            const auto layout = write.ComponentLayout();
            const auto rows = layout.View();
            CPU::FillRows(rows, rows.ComponentNumber(), std::numeric_limits<real>::quiet_NaN());
        }
    }
    void FiniteVolume::PublishBoundaryInputs(
        BoundaryInputs &input, real time, StateEpoch epoch,
        std::vector<BoundaryInterfaceSource> sources, std::optional<R::ExecutionStream> stream) const
    {
        ValidateBoundaryLayout(input);
        DNDS_check_throw_info(input._publication && !input.Published() && std::isfinite(time), "boundary publication needs a begun update and finite physical time");
        // PP POLICY -- an interface packet is a frozen complete mean-like point
        // state. Compressible flow/species satisfy the stored-state simplex
        // and clipped EOS/T invariant. ACM accepts any finite relative pressure
        // and velocity. Both families apply their model-specific RANS bounds.
        // Sampling keeps the transported composition intact.
        const auto validate = [&](const EquationDofs &values, bool centers)
        {
            const auto read = values.ReadOwned();
            const auto layout = read.ComponentLayout();
            const auto ranges = input._layout->faces.Read();
            RequireNumericalSuccess(CPU::ValidateInterfacePacketRanges(layout.View(), ranges.View<R::FixedRows<1>>(),
                                                                       centers, input._admissibility->View()),
                                    "ValidateInterfacePacketRows");
        };
        if (Device() == R::Device::Host())
        {
            if (stream)
                stream->Record().WaitHost();
            validate(input._quadrature, false);
            validate(input._faceCenters, true);
        }
        else
        {
            auto engine = stream ? stream->LockEngine() : R::EagerEngine{};
            const auto lane = PacketStream(engine, Device(), stream);
            const auto check = [&](const EquationDofs &values, bool centers, std::shared_ptr<MeanStateCheck> &checker)
            {
                const index rows = index(values.Space().OwnedRows());
                if (!rows)
                    return; // An omitted section has no required rows to validate.
                if (!checker)
                {
                    R::Array<uint8_t> active(R::RowLayout::Dense(rows, 1));
                    {
                        const auto read = input._layout->faces.Read();
                        const auto ranges = read.View<R::FixedRows<1>>();
                        auto write = active.Write();
                        const auto flags = write.View<R::FixedRows<1>>();
                        for (index i = 0; i < rows; ++i)
                            flags(i, 0) = 0;
                        for (index iFace = 0; iFace < ranges.RowCount(); ++iFace)
                        {
                            const auto range = ranges(iFace, 0);
                            if (!range.interface)
                                continue;
                            const index begin = centers ? iFace : range.quadratureBegin;
                            const index end = begin + (centers ? 1 : range.quadratureCount);
                            for (index i = begin; i < end; ++i)
                                flags(i, 0) = 1;
                        }
                    }
                    checker = std::make_shared<MeanStateCheck>(values.Space(), *input._admissibility,
                                                               lane.Context(), active.To(Device(), lane.Context()));
                }
                RequireNumericalSuccess(checker->Evaluate(engine, lane, values), "ValidateInterfacePacketRows");
            };
            check(input._quadrature, false, input._quadratureCheck);
            check(input._faceCenters, true, input._faceCheck);
            // A fully interior rank has no numerical checker to retire its
            // queued copies. Publish the final mutation versions on that lane.
            if (!input._quadrature.Space().OwnedRows() && !input._faceCenters.Space().OwnedRows())
                lane.Record().WaitHost();
        }
        for (const auto &source : sources)
        {
            DNDS_check_throw_info(
                source.donorDomain >= 0 && source.receiverDomain >= 0 &&
                    source.routeGeneration > 0 && source.routeOwner,
                "boundary interface source is incomplete");
            source.donorPose.Validate();
            source.receiverPose.Validate();
        }
        BoundaryInputs::Publication publication{time, epoch};
        publication.sources = std::move(sources);
        Snapshot(input._quadrature, publication.versions, publication.shapes);
        Snapshot(input._faceCenters, publication.faceVersions,
                 publication.faceShapes);
        std::atomic_store(&input._publication->value,
                          std::make_shared<const BoundaryInputs::Publication>(std::move(publication)));
    }
    void FiniteVolume::ValidateBoundaryInputs(const BoundaryInputs &input, real time) const
    {
        const auto read = input._quadrature.ReadOwned({Device(), nullptr});
        const auto faceRead = input._faceCenters.ReadOwned({Device(), nullptr});
        ValidateBoundaryPublication(input, time);
    }
    void FiniteVolume::ValidateBoundaryPublication(const BoundaryInputs &input, real time) const
    {
        ValidateBoundaryLayout(input);
        // Active writers have not advanced their generations yet.
        // The immediate wrapper or queued declared operands must establish
        // read access; this function checks provenance, never readiness.
        const auto publication = input.Published();
        DNDS_check_throw_info(publication && publication->time == time &&
                                  Current(input._quadrature, publication->versions,
                                          publication->shapes) &&
                                  Current(input._faceCenters,
                                          publication->faceVersions,
                                          publication->faceShapes),
                              "boundary packet is unpublished, stale, modified or from another physical stage");
    }
    BoundaryInputs FiniteVolume::AliasBoundaryInputs(const BoundaryInputs &input) const
    {
        Metrics().Validate();
        DNDS_check_throw_info(input._layout && input._layout->stamp.Valid() && input._layout->mesh.lock() == _data->mesh,
                              "boundary alias requires the same live reference mesh");
        const auto read = input._quadrature.ReadOwned({Device(), nullptr});
        const auto faceRead = input._faceCenters.ReadOwned({Device(), nullptr});
        const auto publication = input.Published();
        DNDS_check_throw_info(publication &&
                                  Current(input._quadrature,
                                          publication->versions,
                                          publication->shapes) &&
                                  Current(input._faceCenters,
                                          publication->faceVersions,
                                          publication->faceShapes),
                              "cannot alias an unpublished or modified boundary packet");
        if (input._layout != _data->boundaryLayout)
        {
            const auto source = input._layout->faces.Read(), target = _data->boundaryLayout->faces.Read();
            const auto a = source.View<R::FixedRows<1>>(), b = target.View<R::FixedRows<1>>();
            DNDS_check_throw_info(a.RowCount() == b.RowCount(), "boundary alias face counts differ");
            const bool needsQuadrature = _data->boundaryLayout->quadratureRows != 0;
            if (needsQuadrature)
                DNDS_check_throw_info(input._layout->integrationOrder == _data->boundaryLayout->integrationOrder &&
                                          input._layout->quadratureRows == _data->boundaryLayout->quadratureRows,
                                      "boundary alias lacks the required quadrature section");
            for (index i = 0; i < a.RowCount(); ++i)
            {
                const auto x = a(i, 0), y = b(i, 0);
                DNDS_check_throw_info(x.face == y.face && x.interface == y.interface &&
                                          (!needsQuadrature || (x.quadratureBegin == y.quadratureBegin && x.quadratureCount == y.quadratureCount)),
                                      "boundary alias face order or requested rows differ");
            }
        }
        auto alias = input;
        // Keep the provider's superset alive. A center-only pMG consumer never
        // repacks or destroys quadrature values still needed by a fine reader.
        alias._consumerLayout = _data->boundaryLayout;
        ValidateBoundaryInputs(alias, alias.Time());
        return alias;
    }
}
