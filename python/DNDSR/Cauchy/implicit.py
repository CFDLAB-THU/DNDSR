"""Python control of prepared Cauchy nonlinear stages."""


def solve_single_stage(solver, source, inverse_dt, rhs_scale, **stage_options):
    """Advance one single-domain implicit stage using its configured stopping rule.

    All domain ranks enter together. The frozen normalized source contains the
    terms independent of the stage unknown. Observation uses the native domain
    reduction and precedes each correction. The session retains the reconstruction,
    CFL schedule and convergence reference through the last observation.
    """
    session = solver.prepare(source, inverse_dt, rhs_scale, **stage_options)
    while not session.complete:
        session.begin_iteration()
        session.observe()
        terminal = session.local_terminal
        if not terminal:
            session.advance_stage(0)
        session.finish_iteration(terminal)
    return session.report
