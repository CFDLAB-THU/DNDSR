"""BDF and DIRK stage sources with Python-owned history and native Array arithmetic."""
import math
from typing import NamedTuple

from ._ext.cauchy_pybind11 import EquationOperations, _bdf_weights, _dirk_tableau


class StageCoefficients(NamedTuple):
    """Coefficients of resOther - inverse_dt*M*U + rhs_scale*R(U) at time."""
    inverse_dt: float
    rhs_scale: float
    time: float


class BDFEquations:
    """BDF1 or variable-step BDF2. The caller commits history on acceptance."""

    def __init__(self, stream, order):
        if order not in (1, 2):
            raise ValueError("BDF order must be 1 or 2")
        self.order = order
        self.ops = EquationOperations(stream)

    def source(self, out, initial, time, dt, *, previous=None, previous_dt=0., steady=False):
        """Write the frozen owned-row source and return the nonlinear stage coefficients.

        Supply the preceding accepted state and its positive interval together
        for BDF2. Omitting both selects BDF1 startup. Steady BDF1 has zero
        physical mass rate. Output must have independent storage.
        """
        if (previous is None) != (previous_dt == 0.) or (previous is not None and self.order != 2):
            raise ValueError("BDF2 history needs a previous state and its accepted interval")
        a, b, inv_dt, scale, end = _bdf_weights(self.order, time, dt, previous_dt, steady)
        if b:
            self.ops.mass_combination(out, a, initial, b, previous)
        elif a:
            self.ops.mass_copy_scale(out, a, initial)
        else:
            self.ops.fill(out, 0.)
        return StageCoefficients(inv_dt, scale, end)


class DIRKEquations:
    """Serialized trapezoidal/ESDIRK stages on one stream, with caller-owned rates."""

    def __init__(self, stream, method):
        self._a, self._c = _dirk_tableau(method)
        self.ops = EquationOperations(stream)

    @property
    def stage_count(self):
        return len(self._c)

    def stage_source(self, out, initial, rates, time, dt):
        """Prepare stage len(rates) from the completed volume-normalized spatial rates.

        At the explicit first stage, evaluate and save R(initial); out is unused.
        At an implicit stage, out must be independent of initial and every rate.
        The physical mass map applies only to the state and preserves the ACM
        continuity contribution from each previous spatial rate.
        """
        stage = len(rates)
        if stage >= self.stage_count:
            raise ValueError("DIRK has no further stage")
        if not (math.isfinite(dt) and dt > 0 and math.isfinite(1 / dt)
                and math.isfinite(time + dt) and time + dt > time):
            raise ValueError("invalid DIRK timestep")
        row = self._a[stage]
        diagonal = row[stage]
        if diagonal:
            self.ops.mass_copy_scale(out, 1 / dt, initial)
            for weight, rate in zip(row, rates):
                if weight:
                    self.ops.axpy(out, weight, rate)
        return StageCoefficients(1 / dt if diagonal else 0., diagonal, time + dt * self._c[stage])
