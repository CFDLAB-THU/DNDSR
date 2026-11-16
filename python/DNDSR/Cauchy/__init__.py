"""Cauchy CPU solver and finite-volume operators over shared native fields."""

from DNDSR._loader import preload

preload("cauchy")

from ._ext.cauchy_pybind11 import Domain, Simulation, StepResult, InterfaceDiagnostics, NonlinearIterationReport, PhysicalStepLimit, DerivedFields, DerivedQuantity
from ._ext.cauchy_pybind11 import (EquationDofs, FiniteVolumeOperators, AuxiliaryRequest,
                                  AuxiliaryValues, MeanGradients, Workspace,
                                  MeanWorkspace, DiagonalStorage, OffDiagonalStorage,
                                  LinearRegion, JacobianPreparation, LinearProblem,
                                  MatrixTableLayout, MatrixTableOptions, LinearIndexStorage, LinearMatrixOptions,
                                  LinearSolveSettings, LinearExecution, LinearIterationReport, LinearStop,
                                  DofLinearSolver, DofSGSIteration,
                                  NonlinearSolveControl, NonlinearStageSolver,
                                  PolynomialMultigrid)

__all__ = ["Domain", "Simulation", "StepResult", "InterfaceDiagnostics", "NonlinearIterationReport", "PhysicalStepLimit", "DerivedFields", "DerivedQuantity"]
__all__ += ["EquationDofs", "FiniteVolumeOperators", "AuxiliaryRequest", "AuxiliaryValues",
            "MeanGradients", "Workspace", "MeanWorkspace",
            "DiagonalStorage", "OffDiagonalStorage", "LinearRegion",
            "JacobianPreparation", "LinearProblem", "LinearSolveSettings",
            "MatrixTableLayout", "MatrixTableOptions", "LinearIndexStorage", "LinearMatrixOptions",
            "LinearExecution", "LinearIterationReport", "LinearStop", "DofLinearSolver",
            "DofSGSIteration", "NonlinearSolveControl", "NonlinearStageSolver",
            "PolynomialMultigrid"]
