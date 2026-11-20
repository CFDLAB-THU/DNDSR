#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Input/InitialField.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "DNDS/Core/OMP.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Geom/Quadratures/Quadrature.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        // Only schema-declared vector triples rotate. In particular, flow4 starts
        // with velocity and must never be loaded as a five-component conservative state.
        std::vector<index> InputVectorOffsets(const EquationSchema &schema)
        {
            std::vector<index> offsets;
            index first = 0;
            for (const auto &block : schema.blocks)
            {
                for (index start : block.vectorStarts)
                    offsets.push_back(first + start);
                first += index(block.components.size());
            }
            return offsets;
        }

        template <class Transform>
        void RotateInputVectors(std::vector<real> &values, const std::vector<index> &offsets, Transform transform)
        {
            for (index first : offsets)
            {
                const auto v = transform(Vector3{values[first], values[first + 1], values[first + 2]});
                std::copy(v.begin(), v.end(), values.begin() + first);
            }
        }

        // Setup-only programmable input evaluation belongs at the checked
        // operator boundary. Exprtk and physical-input conversion may reject
        // user data; these exceptions must never escape a noexcept CPU kernel.
        NumericalStatus InitializeProgrammedField(
            Geom::UnstructuredMeshView mesh,
            Algebra::DofComponentView<real> rows, const RigidFrame &frame,
            std::vector<std::unique_ptr<InitialFieldEvaluator>> &evaluators,
            int integrationOrder, const std::vector<index> &vectorOffsets)
        {
            DNDS_check_throw_info(!evaluators.empty(),
                                  "programmed initialization requires a CPU worker");
            const index width = rows.ComponentNumber();
            const index workerNumber = index(evaluators.size());
            std::vector<NumericalStatus> status(
                size_t(workerNumber), NumericalStatus{});
            const index noRow = std::numeric_limits<index>::max();
            std::vector<std::exception_ptr> exceptions(
                size_t(workerNumber), std::exception_ptr{});
            std::vector<index> exceptionRows(size_t(workerNumber), noRow);
            OMP::ForSetupRowsWithThreads(0, workerNumber, int(workerNumber),
                                         [&](index iWorker)
                                         {
                                             const index begin = rows.RowCount() * iWorker / workerNumber;
                                             const index end = rows.RowCount() * (iWorker + 1) / workerNumber;
                                             auto &evaluator = *evaluators[size_t(iWorker)];
                                             std::vector<real> background(size_t(width), real{});
                                             std::vector<real> average(size_t(width), real{});
                                             index iCell = begin;
                                             try
                                             {
                                                 for (; iCell < end; ++iCell)
                                                 {
                                                     for (index j = 0; j < width; ++j)
                                                         background[size_t(j)] = rows(iCell, j);
                                                     if (frame.moving)
                                                         RotateInputVectors(background, vectorOffsets, [&](const Vector3 &v)
                                                                            { return frame.WorldVector(v); });
                                                     std::fill(average.begin(), average.end(), 0);
                                                     Geom::tSmallCoords coords;
                                                     mesh.GetCoordsOnCell(iCell, coords);
                                                     const auto element = mesh.GetCellElement(iCell);
                                                     const Geom::Elem::Quadrature quadrature(element, integrationOrder);
                                                     real volume = 0;
                                                     for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                                                     {
                                                         const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(
                                                             element.type, integrationOrder, iQ);
                                                         const Geom::tPoint point =
                                                             Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                                                         const real jacobian = Geom::Elem::CellJacobianDet(
                                                             mesh.Dimension(), coords, shape);
                                                         if (!(std::isfinite(jacobian) && jacobian > 0))
                                                         {
                                                             status[size_t(iWorker)] = {
                                                                 NumericalError::InvalidState, 1, iCell};
                                                             return;
                                                         }
                                                         const real weight = jacobian * quadrature.GetWeight(iQ);
                                                         const auto value = evaluator.Evaluate(
                                                             frame.WorldPoint({point[0], point[1], point[2]}),
                                                             0, iCell, background);
                                                         for (index j = 0; j < width; ++j)
                                                             average[size_t(j)] += weight * value[size_t(j)];
                                                         volume += weight;
                                                     }
                                                     if (!(std::isfinite(volume) && volume > 0))
                                                     {
                                                         status[size_t(iWorker)] = {
                                                             NumericalError::InvalidState, 2, iCell};
                                                         return;
                                                     }
                                                     for (auto &value : average)
                                                         value /= volume;
                                                     if (frame.moving)
                                                         RotateInputVectors(average, vectorOffsets, [&](const Vector3 &v)
                                                                            { return frame.LocalVector(v); });
                                                     for (index j = 0; j < width; ++j)
                                                         rows(iCell, j) = average[size_t(j)];
                                                 }
                                             }
                                             catch (...)
                                             {
                                                 exceptions[size_t(iWorker)] = std::current_exception();
                                                 exceptionRows[size_t(iWorker)] = iCell;
                                             }
                                         });
            index firstException = noRow;
            index firstExceptionRow = noRow;
            for (index iWorker = 0; iWorker < workerNumber; ++iWorker)
                if (exceptions[size_t(iWorker)] &&
                    exceptionRows[size_t(iWorker)] < firstExceptionRow)
                    firstException = iWorker,
                    firstExceptionRow = exceptionRows[size_t(iWorker)];
            if (firstException != noRow)
                std::rethrow_exception(exceptions[size_t(firstException)]);
            NumericalStatus failure;
            for (const auto &candidate : status)
                if (!candidate &&
                    (failure || candidate.row < failure.row))
                    failure = candidate;
            return failure;
        }
    }
    void FiniteVolume::InitializeConservative(EquationDofs &state, const std::vector<real> &values) const
    {
        Metrics().Communication()->LocalStage([&]
                                              {
            Metrics().Validate();
            state.RequireSchema(_data->equation.Schema());
            _data->equation.ResolveKernelProfile(state.Device());
            index width = 0;
            for (const auto &block : _data->equation.Schema().blocks) width += index(block.components.size());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && index(values.size()) == width &&
                                      std::all_of(values.begin(), values.end(), [](real v) { return std::isfinite(v); }),
                                  "conservative initializer row space or values differ from equation");
            auto local = values;
            const auto frame = _data->motion.At(0);
            if (frame.moving)
                RotateInputVectors(local, InputVectorOffsets(state.Schema()), [&](const Vector3 &v) { return frame.LocalVector(v); });
            auto output = state.WriteOwned();
            index offset = 0;
            for (const auto &block : _data->equation.Schema().blocks)
            {
                const auto rows = output.Block(block.name);
                CPU::FillConstantRows(rows, {local.data() + offset, rows.ComponentNumber()});
                offset += index(block.components.size());
            } });
    }
    void FiniteVolume::InitializeField(EquationDofs &state, const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers) const
    {
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
            if (_data->settings.initialField.Empty()) return;
            DNDS_check_throw_info(_data->settings.initialProfile.kind == "constant", "spatial overrides require a constant background profile");
            Metrics().Validate();
            state.RequireSchema(_data->equation.Schema());
            _data->equation.ResolveKernelProfile(state.Device());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()), "initial field row space differs from mesh");
            const auto geometry = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto mesh = geometry.HostView();
            const auto frame = _data->motion.At(0);
            auto output = state.WriteOwned();
            const auto layout = output.ComponentLayout();
            if (!layout.View().RowCount()) return;
            const int requested = int(std::min<index>(
                OMP::DistributedThreads(), layout.View().RowCount()));
            std::vector<Physics::Chemistry::ChemicalWorkerPool::Lease> leases;
            if (_data->equation.HasMixture())
            {
                DNDS_check_throw_info(workers,
                                      "mixture initial field requires chemistry workers");
                leases = workers->AcquireUpTo(requested);
            }
            const int workerNumber = _data->equation.HasMixture()
                                         ? int(leases.size())
                                         : requested;
            const auto scales = ResolveReferenceScales(_data->settings.equation);
            std::vector<std::unique_ptr<InitialFieldEvaluator>> evaluators;
            evaluators.reserve(size_t(workerNumber));
            for (int iWorker = 0; iWorker < workerNumber; ++iWorker)
                evaluators.push_back(std::make_unique<InitialFieldEvaluator>(
                    _data->settings.initialField, _data->equation, scales,
                    _data->equation.HasMixture()
                        ? std::move(leases[size_t(iWorker)])
                        : Physics::Chemistry::ChemicalWorkerPool::Lease{}));
            RequireNumericalSuccess(
                InitializeProgrammedField(
                    mesh, layout.View(), frame, evaluators,
                    _data->settings.initialField.integrationOrder, InputVectorOffsets(state.Schema())),
                "InitializeProgrammedField"); });
    }
}
