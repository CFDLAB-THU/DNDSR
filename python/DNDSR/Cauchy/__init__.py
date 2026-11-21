"""Cauchy CPU solver and finite-volume operators over shared native fields."""

from DNDSR._loader import preload

preload("cauchy")

from ._ext.cauchy_pybind11 import Domain, Simulation, StepResult, InterfaceDiagnostics, NonlinearIterationReport, PhysicalStepLimit, DerivedFields, DerivedQuantity, CheckpointLoadMode
from ._ext.cauchy_pybind11 import (EquationDofs, FiniteVolumeOperators, AuxiliaryRequest,
                                  AuxiliaryValues, MeanGradients, Workspace,
                                  MeanWorkspace, DiagonalStorage, OffDiagonalStorage,
                                  LinearRegion, JacobianPreparation, LinearProblem,
                                  MatrixTableLayout, MatrixTableOptions, LinearIndexStorage, LinearMatrixOptions,
                                  LinearSolveSettings, LinearExecution, LinearIterationReport, LinearStop,
                                  DofLinearSolver, DofSGSIteration,
                                  NonlinearSolveControl, NonlinearStageSolver, NonlinearSession,
                                  PolynomialMultigrid, FASOperation)

__all__ = ["Domain", "Simulation", "StepResult", "InterfaceDiagnostics", "NonlinearIterationReport", "PhysicalStepLimit", "DerivedFields", "DerivedQuantity", "CheckpointLoadMode"]
__all__ += ["EquationDofs", "FiniteVolumeOperators", "AuxiliaryRequest", "AuxiliaryValues",
            "MeanGradients", "Workspace", "MeanWorkspace",
            "DiagonalStorage", "OffDiagonalStorage", "LinearRegion",
            "JacobianPreparation", "LinearProblem", "LinearSolveSettings",
            "MatrixTableLayout", "MatrixTableOptions", "LinearIndexStorage", "LinearMatrixOptions",
            "LinearExecution", "LinearIterationReport", "LinearStop", "DofLinearSolver",
            "DofSGSIteration", "NonlinearSolveControl", "NonlinearStageSolver", "NonlinearSession",
            "PolynomialMultigrid", "FASOperation"]

from .implicit import solve_single_stage, apply_v_cycle

__all__ += ["solve_single_stage", "apply_v_cycle"]
