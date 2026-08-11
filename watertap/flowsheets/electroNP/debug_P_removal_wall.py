"""
Targeted debugging script for the electroNP P_removal wall (~P_removal=0.362).

Rather than re-running the full adaptive homotopy sweep (which burns time
re-probing 0.0 -> 0.35 every time), this script:

  1. Builds + initializes the flowsheet exactly as main() does.
  2. Calls run_electroNP_homotopy_sweep() with target set to LAST_KNOWN_GOOD
     (just below the wall) so it stops there instead of continuing to probe
     upward -- this reaches the wall's neighborhood in a handful of solves.
  3. Directly fixes P_removal at TARGET_TEST_POINT (just past the wall) and
     attempts one solve, so you're not waiting through step-halving.
  4. On failure, runs a battery of IDAES diagnostics against the failed
     point:
       - DiagnosticsToolbox.report_structural_issues() /
         report_numerical_issues() -- your existing high-level checks.
       - SVDToolbox (the successor to DegeneracyHunter in idaes-pse 2.12+):
         run_svd_analysis(), display_rank_of_equality_constraints(),
         display_underdetermined_variables_and_constraints() -- this is
         what actually finds the near-singular / degenerate direction in
         the active-constraint Jacobian.
       - DiagnosticsToolbox.compute_infeasibility_explanation() -- computes
         a Minimal Infeasible System (MIS): the smallest set of constraints
         + bounds that are mutually in conflict. This is usually the
         fastest way to pinpoint "which equation is fighting which bound."
       - display_near_parallel_constraints() / display_near_parallel_variables()
         -- flags redundant/near-identical rows, another common degeneracy
         signature.
  5. Prints electroNP split_fraction values so you can see which ones are
     pinned at the 1e-7 pass-through floor when it fails, to correlate with
     whatever the SVD/MIS output implicates.

Note: idaes-pse deprecated the old `DegeneracyHunter` class in 2.12.0 in
favor of `DiagnosticsToolbox.prepare_svd_toolbox()` / the new
`idaes.core.util.diagnostics_tools.svd_toolbox.SVDToolbox`. This script
imports DiagnosticsToolbox from idaes.core.util.model_diagnostics. If
prepare_svd_toolbox()/SVD-related calls fail below, run:
    python -c "from idaes.core.util.model_diagnostics import DiagnosticsToolbox; print([m for m in dir(DiagnosticsToolbox) if not m.startswith('_')])"
to see what's actually available in your installed version, and swap the
calls in run_diagnostics() accordingly.
"""

import pyomo.environ as pyo

from idaes.core.util.model_diagnostics import DiagnosticsToolbox

# BSM2_electroNP_surrogate_initialization_refined_low_PO4_inlet.py must be
# importable (same directory, or on PYTHONPATH).
from BSM2_electroNP_surrogate_initialization_refined_low_PO4_inlet import (
    build_flowsheet,
    set_operating_conditions,
    set_scaling,
    initialize_system,
    solve,
    solve_relaxed_bounds,
    rescale_electroNP_S_PO4,
    rescale_electroNP_and_recycle_P,
    run_electroNP_homotopy_sweep,
)


LAST_KNOWN_GOOD = 0.36  # safely below the ~0.362 wall
TARGET_TEST_POINT = 0.365  # just past the wall -- edit to probe other points


def build_and_converge_to(target, max_attempts=200):
    """Build a fresh flowsheet and homotopy-sweep P_removal from ~0 up to
    target, using the same clean pass-through initialization path as
    main(). Returns the converged model.

    Always use this (rather than perturbing an already-converged model at
    a different P_removal) to reach any new target -- jumping directly
    between two far-apart P_removal values on an already-solved model is
    effectively an oversized homotopy step with a bad warm start, and will
    spuriously report "locally infeasible" even when the model is fine at
    both points individually. Always sweep up from p_start=1e-6.
    """
    m = build_flowsheet(has_electroNP=True)
    set_operating_conditions(m)

    m.fs.FeedWater.conc_mass_comp[0, "S_PO4"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.electroNP.eq_P_removal_surrogate.deactivate()
    m.fs.electroNP.P_removal.fix(1e-6)

    set_scaling(m)
    m, results = initialize_system(m)

    results, homotopy_failed_at, last_good = run_electroNP_homotopy_sweep(
        m,
        target=target,
        p_start=1e-6,
        initial_step=0.05,
        max_step=0.05,
        min_step=1e-4,
        max_attempts=max_attempts,
    )
    if homotopy_failed_at is not None:
        raise RuntimeError(
            f"Failed to reach target={target} -- furthest reached was " f"{last_good}."
        )
    return m


def get_to_last_known_good():
    """Build/initialize the model and homotopy-step up to just below the wall."""
    m = build_and_converge_to(LAST_KNOWN_GOOD)
    print(
        f"\n>>> Reached last-known-good P_removal={LAST_KNOWN_GOOD}. "
        f"Proceeding to probe {TARGET_TEST_POINT}.\n"
    )
    return m


def print_split_fractions(m, label=""):
    print(f"\n--- electroNP split_fraction values/bounds {label} ---")
    for idx, var in m.fs.electroNP.split_fraction.items():
        lb, ub = var.lb, var.ub
        val = pyo.value(var)
        at_lb = lb is not None and abs(val - lb) < 1e-10
        flag = "  <-- AT LOWER BOUND" if at_lb else ""
        print(f"  {idx} = {val:.6g}  (lb={lb}, ub={ub}){flag}")


def check_svd_baseline(p_baseline=0.1):
    """Build a FRESH model and homotopy-sweep cleanly up to a comfortable
    baseline point, well below the wall, then run the same SVD rank check
    used at the failure point.

    Uses build_and_converge_to() rather than perturbing the already-solved
    P_removal=0.36 model, since jumping directly from 0.36 down to 0.1 in
    one solve is an oversized, badly-warm-started step that will falsely
    report "locally infeasible" regardless of whether the model is fine at
    0.1 -- that failure mode would have nothing to do with the actual
    question being asked here.

    Why this matters: 'Number of Singular Values less than 1.0E-06 is 10'
    has come back identical across every homotopy step and every
    rescale/solver-option variant tried so far -- including right up to
    the failure at P_removal~0.365. If the same count of ~10 shows up
    here too, at an easy, comfortably-converged point, that's strong
    evidence this near-singularity is a structural feature of the
    flowsheet's constraint set (e.g. genuinely redundant equations)
    present at every operating point, and NOT what's actually causing the
    wall -- in which case the wall is something else (e.g. a real
    stoichiometric/mass-balance limit, such as the S_A/S_NO3 depletion in
    R3-R6 the last Minimal Infeasible System pointed at) that just happens
    to sit in the same generally poorly-conditioned neighborhood.
    """
    print(
        f"\n================ SVD baseline check at P_removal={p_baseline} "
        "================"
    )
    m_baseline = build_and_converge_to(p_baseline)
    print(f">>> Converged cleanly at baseline P_removal={p_baseline}.")

    dt = DiagnosticsToolbox(m_baseline)
    try:
        svd = dt.prepare_svd_toolbox()
        svd.run_svd_analysis()
        print("\n---- Rank of equality constraints (baseline) ----")
        svd.display_rank_of_equality_constraints()
    except Exception as e:
        print(f"(SVD baseline check raised: {e})")


def probe_failure_point(m, p_try, try_relaxed_bounds_on_failure=True):
    """Fix P_removal at p_try and attempt one direct solve (no step-halving).
    On failure, print diagnostics, then optionally retry with
    solve_relaxed_bounds() to test whether the failure is spurious
    strict-bound infeasibility rather than a genuine one.
    """
    m.fs.electroNP.P_removal.setlb(0)
    m.fs.electroNP.P_removal.setub(1)
    m.fs.electroNP.P_removal.fix(p_try)
    rescale_electroNP_and_recycle_P(m)

    print(f"\n================ Direct probe at P_removal={p_try} ================")
    try:
        results = solve(m)
        pyo.assert_optimal_termination(results)
        print(f">>> Converged at P_removal={p_try}.")
        print_split_fractions(m, label=f"(converged at {p_try})")
        return True
    except Exception as e:
        print(f">>> FAILED at P_removal={p_try}: {e}")
        print_split_fractions(m, label=f"(failed at {p_try})")
        print("\n---- Constraints with Large Residuals (at failed point) ----")
        try:
            DiagnosticsToolbox(m).display_constraints_with_large_residuals()
        except Exception as e2:
            print(f"(display_constraints_with_large_residuals raised: {e2})")

        if not try_relaxed_bounds_on_failure:
            return False

        print(
            f"\n================ Retrying P_removal={p_try} with "
            "solve_relaxed_bounds() ================"
        )
        try:
            results = solve_relaxed_bounds(m)
            pyo.assert_optimal_termination(results)
            print(
                f">>> Converged at P_removal={p_try} WITH RELAXED BOUNDS "
                "-- this confirms the strict-bounds default was the cause, "
                "not a genuine infeasibility."
            )
            print_split_fractions(m, label=f"(converged, relaxed bounds, at {p_try})")
            return True
        except Exception as e3:
            print(
                f">>> Still FAILED at P_removal={p_try} even with relaxed "
                f"bounds: {e3}\n"
                ">>> This suggests the infeasibility is NOT just the strict "
                "bound-enforcement default -- worth digging further into "
                "the SVD/near-parallel-constraint findings below."
            )
            return False


def run_diagnostics(m):
    dt = DiagnosticsToolbox(m)

    print("\n================ DiagnosticsToolbox: structural issues ================")
    dt.report_structural_issues()

    print("\n================ DiagnosticsToolbox: numerical issues ================")
    try:
        dt.report_numerical_issues()
    except Exception as e:
        print(f"(report_numerical_issues raised: {e})")

    print("\n================ Near-parallel constraints/variables ================")
    try:
        dt.display_near_parallel_constraints()
        dt.display_near_parallel_variables()
    except Exception as e:
        print(f"(near-parallel checks raised: {e})")

    print(
        "\n================ SVD analysis (successor to DegeneracyHunter) ================"
    )
    try:
        svd = dt.prepare_svd_toolbox()
        svd.run_svd_analysis()
        print("\n---- Rank of equality constraints ----")
        svd.display_rank_of_equality_constraints()
        print(
            "\n---- Underdetermined variables/constraints (near-singular directions) ----"
        )
        svd.display_underdetermined_variables_and_constraints()
    except Exception as e:
        print(f"(SVD analysis raised: {e})")
        print(
            "If this method doesn't exist in your idaes-pse version, run:\n"
            '  python -c "from idaes.core.util.model_diagnostics import '
            "DiagnosticsToolbox; print([m for m in dir(DiagnosticsToolbox) "
            "if not m.startswith('_')])\"\n"
            "to see the available method names and adjust this script."
        )

    print("\n================ Minimal Infeasible System (MIS) ================")
    try:
        dt.compute_infeasibility_explanation()
    except Exception as e:
        print(f"(compute_infeasibility_explanation raised: {e})")


def bisect_wall(m, p_low, p_high, n_bisections=5):
    """Bisect between a known-good point (p_low) and a known-failing point
    (p_high) to narrow down where the wall actually sits.

    Why this is useful: if the threshold is sharp and repeatable (e.g.
    always fails at ~0.3621, always converges at ~0.3619), that's
    consistent with a real stoichiometric/mass-balance limit being hit
    (a genuine physical boundary of the operating region). If instead the
    pass/fail outcome is inconsistent or jumps around between nearby
    values, that points more toward numerical noise / conditioning issues
    rather than a hard physical wall.
    """
    print(
        f"\n================ Bisecting wall between {p_low} and {p_high} "
        f"================"
    )
    m.fs.electroNP.P_removal.fix(p_low)
    rescale_electroNP_and_recycle_P(m)
    solve(m)

    lo, hi = p_low, p_high
    for i in range(n_bisections):
        mid = (lo + hi) / 2
        print(
            f"\n---- Bisection {i + 1}/{n_bisections}: trying P_removal={mid:.6g} ----"
        )
        m.fs.electroNP.P_removal.setlb(0)
        m.fs.electroNP.P_removal.setub(1)
        m.fs.electroNP.P_removal.fix(mid)
        rescale_electroNP_and_recycle_P(m)
        try:
            results = solve(m)
            pyo.assert_optimal_termination(results)
            print(f"  >>> Converged at {mid:.6g}.")
            lo = mid
        except Exception as e:
            print(f"  >>> Failed at {mid:.6g}: {e}")
            # step back to the last good point so the next bisection
            # attempt has a valid warm start
            m.fs.electroNP.P_removal.fix(lo)
            rescale_electroNP_and_recycle_P(m)
            solve(m)
            hi = mid

    print(
        f"\n>>> Bisection narrowed the wall to between {lo:.6g} (converges) "
        f"and {hi:.6g} (fails)."
    )
    return lo, hi


if __name__ == "__main__":
    m = get_to_last_known_good()

    # Check whether the ~10 near-zero singular values seen at the wall are
    # already present at an easy, comfortably-converged operating point --
    # if so, they're a structural feature of this flowsheet, not specific
    # to being near P_removal~0.362, and the wall itself needs a different
    # explanation. Uses its own fresh model, so this doesn't touch m.
    check_svd_baseline(p_baseline=0.1)

    converged = probe_failure_point(m, TARGET_TEST_POINT)

    # Run diagnostics regardless of outcome -- even a converged point right
    # at the wall may show near-degeneracy building up (e.g. rank deficiency
    # or a near-zero singular value), which foreshadows the failure just
    # past it.
    run_diagnostics(m)

    # Narrow down exactly where the wall sits, to distinguish a sharp
    # repeatable threshold (real physical/stoichiometric limit) from noisy
    # inconsistent pass/fail (numerical conditioning issue). Comment out
    # if you just want the diagnostics above.
    if not converged:
        bisect_wall(m, p_low=LAST_KNOWN_GOOD, p_high=TARGET_TEST_POINT, n_bisections=5)
