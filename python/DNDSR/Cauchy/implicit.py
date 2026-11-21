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


def apply_v_cycle(hierarchy, source, inverse_dt, rhs_scale, **stage_options):
    """Apply one spatial p-FAS correction with a Python-ordered traversal.

    Restriction forms each child's reference and forcing. Prolongation adds the
    child's correction relative to that reference. The hierarchy owns those
    fields, including scratch and the configured smoother on each level.
    Domain ranks must submit identical traversals. This call completes the cycle.
    """
    from ._ext.cauchy_pybind11 import FASOperation as Op

    def visit(coarse_levels):
        if coarse_levels == 0:
            yield Op.CoarseSmooth
            return
        yield Op.PreSmooth
        yield Op.Restrict
        yield from visit(coarse_levels - 1)
        yield Op.Prolong
        yield Op.PostSmooth

    return hierarchy.apply_schedule(list(visit(hierarchy.coarse_level_count)),
                                    source, inverse_dt, rhs_scale, **stage_options)
