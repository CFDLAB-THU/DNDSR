"""Python control of prepared Cauchy nonlinear stages."""


def solve_single_stage(solver, source, inverse_dt, rhs_scale, **stage_options):
    """Advance one single-domain implicit stage using its configured stopping rule.

    All domain ranks enter together. The frozen normalized source contains the
    terms independent of the stage unknown. Observation uses the native domain
    reduction and precedes each correction. The session retains the reconstruction,
    CFL schedule and convergence reference through the last observation.
    """
    session = solver.prepare(source, inverse_dt, rhs_scale, **stage_options)
    _single_stage_iterations(session)
    return session.report


def _single_stage_iterations(session):
    while not session.complete:
        session.begin_iteration()
        session.observe()
        terminal = session.local_terminal
        if not terminal:
            session.advance_stage(0)
        session.finish_iteration(terminal)


def _solve_group(session, policy):
    if policy == "singleStageImplicit":
        _single_stage_iterations(session)
    elif policy == "ditrSequential2Stage":
        while not session.complete:
            session.begin_iteration()
            session.advance_stage(0)
            session.advance_stage(1)
            session.observe()
            session.finish_iteration(session.local_terminal)
    else:
        raise ValueError(f"No Python nonlinear controller for {policy!r}")


def advance_domain(domain, dt):
    """Advance one standalone domain with Python stage and iteration control.

    Enter on every rank. The domain owns trial and accepted storage, the ODE
    history, and normalized residuals. A numerical rejection discards the trial;
    the caller selects any retry timestep. Do not reenter or mutate the domain
    concurrently. Unexpected failures in collective work must terminate the run.
    """
    result = domain._begin_standalone_step(dt)
    while result and domain._has_next_group:
        result = domain._advance_group(_solve_group)
    if result:
        domain.commit_step()
    else:
        domain.discard_step()
    return result


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
