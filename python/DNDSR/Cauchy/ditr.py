"""Coupled DITR time-equation assembly on the caller's Runtime stream."""

from ._ext.cauchy_pybind11 import EquationOperations, _ditr_weights


class DITREquations:
    """Assemble frozen stage sources and the I+H stopping residual.

    State arguments are cell means. Rates are volume-normalized spatial rates.
    The caller owns their stage history and orders midpoint correction and rate
    refresh before endpoint correction and rate refresh. Each output is an
    independent EquationDofs allocation. Only state terms use the physical mass
    map M; pressure rates still enter both equations for incompressible ACM.
    """

    def __init__(self, stream, fraction, theta, variant):
        self.c, self.w = _ditr_weights(fraction, theta, variant)
        self.theta = theta
        self.ops = EquationOperations(stream)

    def midpoint_source(self, out, inverse_dt, initial, endpoint, initial_rate, end_rate):
        """Freeze resOther for H + theta*I and return its midpoint rate multiplier."""
        c, w, theta = self.c, self.w, self.theta
        self.ops.mass_combination(out, (c[0] + theta) * inverse_dt, initial,
                                  (c[1] - theta) * inverse_dt, endpoint)
        self.ops.axpy(out, c[2] + theta * w[0], initial_rate)
        self.ops.axpy(out, c[3] + theta * w[2], end_rate)
        return theta * w[1]

    def endpoint_source(self, out, inverse_dt, initial, initial_rate, middle_rate):
        """Freeze resOther for I and return its endpoint rate multiplier."""
        self.ops.mass_state_and_rate(out, inverse_dt, initial, self.w[0], initial_rate)
        self.ops.axpy(out, self.w[1], middle_rate)
        return self.w[2]

    def stopping_residual(self, out, inverse_dt, initial, middle, endpoint,
                          initial_rate, middle_rate, end_rate):
        """Assemble I+H after both corrected rates have been refreshed."""
        c, w = self.c, self.w
        self.ops.mass_combination(out, (1 + c[0]) * inverse_dt, initial,
                                  (c[1] - 1) * inverse_dt, endpoint)
        self.ops.mass_axpy(out, -inverse_dt, middle)
        self.ops.axpy(out, w[0] + c[2], initial_rate)
        self.ops.axpy(out, w[1], middle_rate)
        self.ops.axpy(out, w[2] + c[3], end_rate)
