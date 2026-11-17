#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/IO/OutputFields.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <sstream>
#include <iomanip>

namespace DNDS::Cauchy
{
    std::string FiniteVolume::PrepareDeviceSurfaceVisualization(
        const EquationDofs &state, const ReconstructionState &reconstruction, const Workspace &workspace,
        real time, R::EagerEngine &engine, const R::ExecutionStream &stream) const
    {
        const MPIInfo mpi(Metrics().Communicator());
        std::ostringstream surface;
        MPI::LocalStage(mpi, [&]
                        {
            DNDS_check_throw_info(Device().space == R::MemorySpace::CUDA && !Equation().HasMixture(),
                                  "resident surface output requires CUDA ACM or perfect-gas physics");
            const index rows = _data->BoundaryValueRows(), components = state.ComponentNumber();
            const Algebra::DofSpace space(size_t(rows), 0, "output/boundaryPoints");
            auto points = EquationDofs::Coupled(state.Schema(),
                Algebra::DofField<real>(space, R::RowLayout::Dense(0, components), Device()), "equation");
            R::Array<NumericalStatus> traceStatus(R::RowLayout::Dense(rows, 1), Device());
            points.CoupledField().Owned().SetName("output/boundaryTraces", DNDS_ALLOCATION_SITE());
            traceStatus.SetName("output/boundaryTraceStatus", DNDS_ALLOCATION_SITE());
            std::optional<EquationDofs> coefficients;
            OptionalRef<const EquationDofs> polynomial;
            if (_data->reconstruction)
            {
                coefficients.emplace(FluxReconstruction(reconstruction, workspace).NumericAlias());
                polynomial.Emplace(*coefficients);
            }
            const auto boundary = PreparedBoundaryInputs(workspace);
            DNDS_check_throw_info(!HasInterfaces() || boundary, "surface output needs a ready interface packet");
            OptionalRef<const R::Array<real>> incoming;
            if (boundary)
            {
                ValidateBoundaryInputs(*boundary, time);
                incoming.Emplace((_data->stencil.quadratureBoundaryValues ? boundary->QuadratureValues() : boundary->FaceCenterValues()).CoupledField().Owned());
            }
            BoundarySubmission request;
            request.time = time;
            request.exterior = false; // FaceTrace, independently of reconstruction's mirrored-target policy.
            request.forcePointFallback = !Equation().IsACM();
            SubmitBoundaryValues(engine, stream, state, polynomial, points.CoupledField().Owned(), traceStatus, request, incoming);
            engine.Synchronize();
            // Explicit I/O boundary. Inspect producer failure before passing
            // its output to the property producer; never consume failed rows.
            const auto hostStatus = traceStatus.To(R::Device::Host(), stream.Context());
            const auto statusRead = hostStatus.Read();
            for (index iPoint = 0; iPoint < rows; ++iPoint)
                RequireNumericalSuccess(statusRead.View()(iPoint, 0), "surface boundary trace");
            auto fields = OutputFields::Submit(engine, stream, points, Equation());
            engine.Synchronize();
            const auto hostFields = fields.CopyToHost(stream.Context());
            const auto hostGeometry = _data->BoundaryTraceSnapshot();
            const auto valuesRead = hostFields.Read();
            const auto geometryRead = hostGeometry.Read();
            const auto values = valuesRead.View<R::TableRows>();
            const auto geometry = geometryRead.View<R::FixedRows<1>>();
            // From here on there is only formatting/file I/O. Numerical
            // boundary evaluation and thermodynamics have completed on device.
            const bool mechanical = Equation().IsACM();
            surface << "cell_id,zone,x,y,z,nx,ny,nz,weight";
            if (mechanical)
                for (const auto &component : Equation().Schema().LogicalComponents())
                    surface << ',' << component.name;
            else
                surface << ",rho,p,u,v,w,T";
            std::optional<R::Array<real>::ReadLease> shearRead;
            if (workspace.boundaryMomentumCurrent && workspace.boundaryWallShear)
            {
                const auto hostShear = workspace.boundaryWallShear->To(R::Device::Host(), stream.Context());
                shearRead.emplace(hostShear.Read());
                surface << ",last_rhs_wall_shear_x,last_rhs_wall_shear_y,last_rhs_wall_shear_z";
            }
            surface << '\n' << std::setprecision(17);
            for (index iPoint = 0; iPoint < rows; ++iPoint)
            {
                const auto &point = geometry(iPoint, 0);
                surface << Mesh().OriginalEntityId(Geom::EntityKind::Cell, point.cell) << ',' << _data->forceZones.at(point.condition);
                for (int d = 0; d < 3; ++d) surface << ',' << point.location[d];
                for (int d = 0; d < 3; ++d) surface << ',' << point.normal[d];
                surface << ',' << point.weight;
                if (mechanical)
                {
                    for (index j = 0; j < components; ++j) surface << ',' << values(iPoint, j);
                    if (shearRead)
                        for (int d = 0; d < 3; ++d) surface << ',' << shearRead->View<R::FixedRows<3>>()(iPoint,d);
                    surface << '\n';
                    continue;
                }
                surface << ',' << values(iPoint, 0) << ',' << values(iPoint, components);
                for (int d = 0; d < 3; ++d) surface << ',' << values(iPoint, components + 2 + d);
                surface << ',' << values(iPoint, components + 1) << '\n';
            }
            DNDS_check_throw_info(bool(surface), "resident surface output preparation failed"); });
        return surface.str();
    }
}
