#include "Cauchy/Bindings/Operators_bind.hpp"
#include "Cauchy/Bindings/OperatorsBindInternal.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "DNDS/Runtime/Interop/Array_bind.hpp"
#include <pybind11/stl.h>

namespace DNDS::Cauchy::Bindings
{
    void BindOperatorLinear(py::module_ &m)
    {
        py::enum_<Algebra::MatrixTableLayout>(m, "MatrixTableLayout")
            .value("AoS", Algebra::MatrixTableLayout::AoS)
            .value("SoA", Algebra::MatrixTableLayout::SoA)
            .value("AoSoA32", Algebra::MatrixTableLayout::AoSoA32);
        py::class_<Algebra::MatrixTableOptions>(m, "MatrixTableOptions")
            .def(py::init<>())
            .def_readwrite("layout", &Algebra::MatrixTableOptions::layout)
            .def_readwrite("row_padding_multiple", &Algebra::MatrixTableOptions::rowPaddingMultiple)
            .def_readwrite("row_stride_padding_multiple", &Algebra::MatrixTableOptions::rowStridePaddingMultiple)
            .def_readwrite("allocation_alignment_bytes", &Algebra::MatrixTableOptions::allocationAlignmentBytes);
        py::enum_<LinearIndexStorage>(m, "LinearIndexStorage")
            .value("Int64", LinearIndexStorage::Int64)
            .value("Int32", LinearIndexStorage::Int32);
        py::class_<LinearMatrixOptions>(
            m, "LinearMatrixOptions",
            "Independent native matrix layouts and sparse indices; row_major_blocks selects within-block order without repacking.")
            .def(py::init<>())
            .def_readwrite("diagonal", &LinearMatrixOptions::diagonal)
            .def_readwrite("inverse", &LinearMatrixOptions::inverse)
            .def_readwrite("off_diagonal", &LinearMatrixOptions::offDiagonal)
            .def_readwrite("indices", &LinearMatrixOptions::indices)
            .def_readwrite("row_major_blocks", &LinearMatrixOptions::rowMajorBlocks)
            .def_readwrite("amgx_native_storage", &LinearMatrixOptions::amgxNativeStorage);
        py::enum_<DiagonalStorage>(m, "DiagonalStorage")
            .value("Entries", DiagonalStorage::Entries)
            .value("Block", DiagonalStorage::Block);
        py::enum_<OffDiagonalStorage>(m, "OffDiagonalStorage")
            .value("Assembled", OffDiagonalStorage::Assembled)
            .value("DirectEvaluated", OffDiagonalStorage::DirectEvaluated)
            .value("SemiMatrixFree", OffDiagonalStorage::DirectEvaluated);
        py::enum_<LinearRegion>(m, "LinearRegion")
            .value("All", LinearRegion::All)
            .value("Core", LinearRegion::Core)
            .value("HaloDependent", LinearRegion::HaloDependent);
        py::class_<JacobianPreparation>(m, "JacobianPreparation")
            .def(py::init<>())
            .def_readwrite("physical_mass", &JacobianPreparation::physicalMass)
            .def_readwrite("residual_scale", &JacobianPreparation::residualScale)
            .def_readwrite("roe_dissipation", &JacobianPreparation::roeDissipation)
            .def_readwrite("scalar_boundary_derivative", &JacobianPreparation::scalarBoundaryDerivative)
            .def_readwrite("update_components", &JacobianPreparation::updateComponents);
        py::class_<LinearProblemContext::DofSGSIteration>(
            m, "DofSGSIteration",
            "Staged two-buffer SGS. Join input halo before dependent lower work. Upper reverses order. Finish after both scans; abandonment invalidates unfinished output.")
            .def("lower", &LinearProblemContext::DofSGSIteration::Lower,
                 py::arg_v("region", LinearRegion::All, "LinearRegion.All"), py::call_guard<py::gil_scoped_release>(),
                 Local)
            .def("upper", &LinearProblemContext::DofSGSIteration::Upper,
                 py::arg_v("region", LinearRegion::All, "LinearRegion.All"), py::call_guard<py::gil_scoped_release>(),
                 Local)
            .def("finish", &LinearProblemContext::DofSGSIteration::Finish, py::call_guard<py::gil_scoped_release>(),
                 Local);
        py::class_<LinearProblemContext, std::shared_ptr<LinearProblemContext>>(
            m, "LinearProblem",
            "Frozen approximate matrix. Explicitly prepare coefficients, inverses and factors. Serialize changes with all actions/solves.")
            .def_property_readonly("diagonal",
                                   [](const LinearProblemContext &p)
        {
            return R::PythonArray::Share(p.Diagonal());
        })
            .def_property_readonly("core", &LinearProblemContext::Core)
            .def_property_readonly("row_major_blocks", &LinearProblemContext::RowMajorBlocks)
            .def_property_readonly("halo_dependent", &LinearProblemContext::HaloDependent)
            .def("invert_diagonal", &LinearProblemContext::InvertDiagonal, py::arg("refinement_steps") = 0,
                 py::call_guard<py::gil_scoped_release>(), Local)
            .def("apply",
                 py::overload_cast<const EquationDofs &, EquationDofs &, LinearRegion>(&LinearProblemContext::Apply,
                                                                                       py::const_),
                 py::arg("input"), py::arg("output"), py::arg_v("region", LinearRegion::All, "LinearRegion.All"),
                 py::call_guard<py::gil_scoped_release>(), Local)
            .def("begin_sgs",
                 py::overload_cast<const EquationDofs &, const EquationDofs &, EquationDofs &, EquationDofs &, bool>(
                     &LinearProblemContext::BeginSGS, py::const_),
                 py::arg("rhs"), py::arg("input"), py::arg("forward"), py::arg("output"),
                 py::arg("middle_exchange") = false, py::call_guard<py::gil_scoped_release>(), Local)
            .def(
                "begin_sgs_zero",
                py::overload_cast<const EquationDofs &, EquationDofs &, EquationDofs &, bool>(
                    &LinearProblemContext::BeginSGSZero, py::const_),
                py::arg("rhs"), py::arg("forward"), py::arg("output"), py::arg("middle_exchange") = false,
                "Dedicated statically selected zero-start SGS. No old input allocation or read; exchange forward before Upper only if middle_exchange is enabled.",
                py::call_guard<py::gil_scoped_release>(), Local)
            .def("apply_sgs",
                 py::overload_cast<const EquationDofs &, const EquationDofs &, EquationDofs &, EquationDofs &>(
                     &LinearProblemContext::ApplySGS, py::const_),
                 py::arg("rhs"), py::arg("input"), py::arg("forward"), py::arg("output"),
                 py::call_guard<py::gil_scoped_release>(), Local)
            .def("prepare_ilu", &LinearProblemContext::PrepareILUStructure, py::arg("fill") = 0,
                 py::arg("core_first") = true, py::arg("local_parts") = std::vector<index>{},
                 py::arg("ordering") = "natural", py::call_guard<py::gil_scoped_release>(), Local)
            .def("factorize_ilu", &LinearProblemContext::FactorizeILU, py::call_guard<py::gil_scoped_release>(), Local)
            .def("apply_ilu",
                 py::overload_cast<const EquationDofs &, EquationDofs &, EquationDofs &>(
                     &LinearProblemContext::ApplyILU, py::const_),
                 py::arg("rhs"), py::arg("forward"), py::arg("output"), py::call_guard<py::gil_scoped_release>(),
                 Local);
        py::class_<LinearSolveSettings>(m, "LinearSolveSettings")
            .def(py::init<>())
            .def_readwrite("method", &LinearSolveSettings::method)
            .def_readwrite("preconditioner", &LinearSolveSettings::preconditioner)
            .def_readwrite("stopping", &LinearSolveSettings::stopping)
            .def_readwrite("max_iterations", &LinearSolveSettings::maxIterations)
            .def_readwrite("residual_check_interval", &LinearSolveSettings::residualCheckInterval)
            .def_readwrite("subspace", &LinearSolveSettings::subspace)
            .def_readwrite("sgs_sweeps", &LinearSolveSettings::sgsSweeps)
            .def_readwrite("jacobi_local_sweeps", &LinearSolveSettings::jacobiLocalSweeps)
            .def_readwrite("diagonal_inverse_refinement", &LinearSolveSettings::diagonalInverseRefinement)
            .def_readwrite("cuda_diagonal_inverse", &LinearSolveSettings::cudaDiagonalInverse)
            .def_readwrite("cuda_jacobi_mapping", &LinearSolveSettings::cudaJacobiMapping)
            .def_readwrite("ilu_fill", &LinearSolveSettings::iluFill)
            .def_readwrite("ilu_preconditioner_corrections", &LinearSolveSettings::iluPreconditionerCorrections)
            .def_readwrite("ilu_core_first", &LinearSolveSettings::iluCoreFirst)
            .def_readwrite("ilu_ordering", &LinearSolveSettings::iluOrdering)
            .def_readwrite("exchange_between_sweeps", &LinearSolveSettings::exchangeBetweenSweeps)
            .def_readwrite("relative_tolerance", &LinearSolveSettings::relativeTolerance)
            .def_readwrite("absolute_tolerance", &LinearSolveSettings::absoluteTolerance);
        py::class_<LinearExecution::Streams>(
            m, "LinearExecutionStreams",
            "Domain compute/communication lanes; retain their owning engine through solve.")
            .def(py::init(
                [](Runtime::ExecutionStream compute, Runtime::ExecutionStream communication)
        {
            return LinearExecution::Streams{std::move(compute), std::move(communication)};
        }))
            .def_readwrite("compute", &LinearExecution::Streams::compute)
            .def_readwrite("communication", &LinearExecution::Streams::communication);
        py::class_<LinearExecution>(m, "LinearExecution")
            .def(py::init<>())
            .def_readwrite("overlap", &LinearExecution::overlap)
            .def_readwrite("collective_checks", &LinearExecution::collectiveChecks)
            .def_readwrite("streams", &LinearExecution::streams);
        py::enum_<Linear::LinearStop>(m, "LinearStop")
            .value("Converged", Linear::LinearStop::Converged)
            .value("IterationLimit", Linear::LinearStop::IterationLimit)
            .value("ArnoldiBreakdown", Linear::LinearStop::ArnoldiBreakdown)
            .value("PreconditionerBreakdown", Linear::LinearStop::PreconditionerBreakdown)
            .value("NonFinite", Linear::LinearStop::NonFinite)
            .value("FixedCount", Linear::LinearStop::FixedCount);
        py::class_<Linear::LinearIterationReport>(m, "LinearIterationReport")
            .def_readonly("stop", &Linear::LinearIterationReport::stop)
            .def_readonly("converged", &Linear::LinearIterationReport::converged)
            .def_readonly("usable", &Linear::LinearIterationReport::usable)
            .def_readonly("iterations", &Linear::LinearIterationReport::iterations)
            .def_readonly("initial_true_residual", &Linear::LinearIterationReport::initialTrueResidual)
            .def_readonly("true_residual", &Linear::LinearIterationReport::trueResidual)
            .def_readonly("operator_applications", &Linear::LinearIterationReport::operatorApplications)
            .def_readonly("inner_product_batches", &Linear::LinearIterationReport::innerProductBatches)
            .def_readonly("jacobi_sweeps", &Linear::LinearIterationReport::jacobiSweeps)
            .def_readonly("jacobi_input_exchanges", &Linear::LinearIterationReport::jacobiInputExchanges)
            .def_readonly("preconditioner_applications", &Linear::LinearIterationReport::preconditionerApplications);
        py::class_<DofLinearSolver>(
            m, "DofLinearSolver",
            "Native SGS/left-GMRES/ILU orchestration. Matching collective calls and producer completion are required. Solution aliases reusable scratch; clone before another solve.")
            .def("solve",
                 [](DofLinearSolver &solver, const EquationDofs &rhs, const EquationDofs *initialGuess,
                    const std::vector<real> &componentScales)
        {
            return solver.Solve(
                rhs, initialGuess ? OptionalRef<const EquationDofs>(*initialGuess) : OptionalRef<const EquationDofs>{},
                componentScales);
        },
                 py::arg("rhs"), py::arg("initial_guess") = nullptr, py::arg("component_scales") = std::vector<real>{},
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def_property_readonly("solution", [](const DofLinearSolver &solver)
        {
            return solver.Solution();
        });
    }
}
