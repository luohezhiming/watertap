#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
"""
Simplified 2-reactor validation flowsheet for the UConn WRRF ASM3 model.

Purpose:
    Isolate R1 (anoxic CSTR) and R2 (aerobic AerationTank) from the full
    5-reactor recycle network to test their reaction kinetics in the
    simplest possible topology: no mixers, no splitters, no recycle.

Layout:
    Feed -> R1 (anoxic CSTR) -> R2 (aerobic AerationTank) -> Product

    The feed is fixed directly to the UConn ODE steady-state "Reactor 1
    Inlet" composition (i.e. what R1 actually sees at SS in the full
    model), so R1's local kinetics can be checked against the ODE
    "Mixer 2 In1" (R1 outlet) reference, and R2's kinetics against
    "Mixer 2 In2" (R2 outlet) reference -- keeping in mind that in the
    full flowsheet R2's true inlet is the S1/R5 recycle stream, not R1's
    outlet, so R2's comparison here is a kinetics sanity check under a
    different (but well-defined) inlet condition, not a literal
    reproduction of the full-network SS.

Reactor volumes, KLa, and calibrated kinetic parameters (mu_H, mu_A,
K_NOX, Y_STO_O2, Y_H_NOX) are carried over unchanged from the full
UConn_WRRF_refined.py flowsheet for R1 and R2.
"""

__author__ = "Chenyu Wang, Adam Atia"

import pyomo.environ as pyo
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.models.unit_models import Feed, Product
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core.util.tables import (
    create_stream_table_dataframe,
    stream_table_dataframe_to_string,
)
from idaes.core.util.initialization import propagate_state
from idaes.core.util import DiagnosticsToolbox

from watertap.unit_models import AerationTank, CSTR
from watertap.property_models.unit_specific.activated_sludge.asm3_properties import (
    ASM3ParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.asm3_reactions import (
    ASM3ReactionParameterBlock,
)
from watertap.core.util.initialization import check_solve
from watertap.core.solvers import get_solver
from pyomo.opt import TerminationCondition

_log = idaeslog.getLogger(__name__)


# ---------------------------------------------------------------------------
# UConn ODE steady-state reference data (mg/L unless noted; alkalinity mol/m3)
# ---------------------------------------------------------------------------
# "Reactor 1 Inlet" -- used directly as the feed composition for this model
ODE_R1_IN = {
    "X_H": 237.06,
    "X_STO": 440.80,
    "X_S": 124.89,
    "X_A": 41.40,
    "X_I": 3666.3,
    "X_TSS": 2057.0,
    "S_S": 46.96,
    "S_O": 0.00686,
    "S_NH4": 7.131,
    "S_NOX": 0.4949,
    "S_N2": 0.1149,
    "S_I": 7.387,
    "alkalinity": 0.900,
    "flow_m3_day": 11378.05,
}
# "Mixer 2 In1" -- UConn's true R1 outlet, for comparing R1's local kinetics
ODE_R1_OUT_REF = {
    "X_H": 237.87,
    "X_STO": 441.35,
    "X_S": 103.32,
    "X_A": 41.39,
    "X_I": 3666.3,
    "X_TSS": 2041.9,
    "S_S": 65.83,
    "S_O": 9.406e-6,
    "S_NH4": 7.371,
    "S_NOX": 0.02696,
    "S_N2": 0.5835,
    "S_I": 7.387,
    "alkalinity": 0.9505,
}
# "Mixer 2 In2" -- UConn's true R2 outlet (note: in the full model R2's
# inlet is the S1/R5 recycle stream at 6486 m3/day, NOT R1's outlet -- so
# this reference is informative but not a literal apples-to-apples target
# when R2 is fed R1's outlet directly, as in this simplified flowsheet)
ODE_R2_OUT_REF = {
    "X_H": 226.28,
    "X_STO": 415.71,
    "X_S": 27.87,
    "X_A": 39.67,
    "X_I": 3674.97,
    "X_TSS": 1964.4,
    "S_S": 0.1477,
    "S_O": 8.267,
    "S_NH4": 0.06262,
    "S_NOX": 3.569,
    "S_N2": 0.6106,
    "S_I": 7.387,
    "alkalinity": 0.1755,
}


def build_flowsheet():
    """Build the simplified Feed -> R1 -> R2 -> Product flowsheet."""
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM3ParameterBlock()

    # Same per-reactor calibrated mu_H / mu_A as the full flowsheet
    m.fs.rxn_props_R1 = ASM3ReactionParameterBlock(
        property_package=m.fs.props,
        calibrated_params={"mu_H": 2.7441113149445235, "mu_A": 3.722441313448151},
    )
    m.fs.rxn_props_R2 = ASM3ReactionParameterBlock(
        property_package=m.fs.props,
        calibrated_params={"mu_H": 0.22557810779134918, "mu_A": 2.4028777896132607},
    )

    m.fs.feed = Feed(property_package=m.fs.props)
    m.fs.R1 = CSTR(property_package=m.fs.props, reaction_package=m.fs.rxn_props_R1)
    m.fs.R2 = AerationTank(
        property_package=m.fs.props, reaction_package=m.fs.rxn_props_R2
    )
    m.fs.Treated = Product(property_package=m.fs.props)

    m.fs.feed_to_r1 = Arc(source=m.fs.feed.outlet, destination=m.fs.R1.inlet)
    m.fs.r1_to_r2 = Arc(source=m.fs.R1.outlet, destination=m.fs.R2.inlet)
    m.fs.r2_to_treated = Arc(source=m.fs.R2.outlet, destination=m.fs.Treated.inlet)

    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def set_operating_conditions(m):
    """Fix feed composition (= UConn ODE R1 inlet), reactor volumes, KLa,
    and calibrated kinetic parameters -- all carried over from the full
    UConn_WRRF_refined.py flowsheet for R1 and R2."""

    r = ODE_R1_IN
    flow_m3_s = r["flow_m3_day"] / 86400.0

    m.fs.feed.flow_vol.fix(flow_m3_s * pyo.units.m**3 / pyo.units.s)
    m.fs.feed.temperature.fix(293.15 * pyo.units.K)
    m.fs.feed.pressure.fix(101325 * pyo.units.Pa)
    m.fs.feed.alkalinity.fix(r["alkalinity"] * pyo.units.mol / pyo.units.m**3)

    for species in [
        "X_H",
        "X_STO",
        "X_S",
        "X_A",
        "X_I",
        "X_TSS",
        "S_S",
        "S_O",
        "S_NH4",
        "S_NOX",
        "S_N2",
        "S_I",
    ]:
        m.fs.feed.conc_mass_comp[0, species].fix(
            r[species] * 1e-3 * pyo.units.kg / pyo.units.m**3
        )

    # Reactor volumes (same as full flowsheet)
    m.fs.R1.volume.fix(1135.6 * pyo.units.m**3)
    m.fs.R2.volume.fix(3077.8186727373936 * pyo.units.m**3)

    # R2 aeration: no injection except S_O, KLa=10/hr (same as full flowsheet)
    for j in m.fs.props.component_list:
        if j != "S_O":
            m.fs.R2.injection[:, :, j].fix(0)
    m.fs.R2.KLa.fix(10 / pyo.units.hour)

    # Per-reactor calibrated scalar kinetic parameters (same as full flowsheet)
    m.fs.rxn_props_R1.K_NOX.fix(0.662744537546551e-3)
    m.fs.rxn_props_R1.Y_STO_O2.fix(0.598031273330616)
    m.fs.rxn_props_R1.Y_H_NOX.fix(0.6217434149054809)

    m.fs.rxn_props_R2.K_NOX.fix(0.4946569070893697e-3)
    m.fs.rxn_props_R2.Y_STO_O2.fix(0.6527850814322534)
    m.fs.rxn_props_R2.Y_H_NOX.fix(0.47413046475869364)

    # Touch variable
    m.fs.Treated.conc_mass_comp

    print("DOF = ", degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0


def scale_flowsheet(m):
    """Same scaling scheme as the full flowsheet (including the corrected
    alkalinity scaling factor: actual magnitude is ~0.1-2.3 mol/m3, not
    the previously mistaken ~1e-3 mol/m3)."""
    for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
        name = var.name

        if "flow_vol" in name:
            iscale.set_scaling_factor(var, 10)

        elif "temperature" in name:
            iscale.set_scaling_factor(var, 1e-2)

        elif "pressure" in name:
            iscale.set_scaling_factor(var, 1e-5)

        elif "conc_mass_comp" in name:
            if "S_O" in name:
                iscale.set_scaling_factor(var, 1e2)
            elif "S_N2" in name:
                iscale.set_scaling_factor(var, 1e2)
            elif "S_NOX" in name:
                iscale.set_scaling_factor(var, 1e2)
            elif "S_NH4" in name:
                iscale.set_scaling_factor(var, 1e2)
            elif "S_I" in name:
                iscale.set_scaling_factor(var, 1e2)
            elif "S_S" in name:
                iscale.set_scaling_factor(var, 10)
            elif "X_H" in name:
                iscale.set_scaling_factor(var, 10)
            elif "X_STO" in name:
                iscale.set_scaling_factor(var, 10)
            elif "X_A" in name:
                iscale.set_scaling_factor(var, 10)
            elif "X_I" in name:
                iscale.set_scaling_factor(var, 1)
            elif "X_S" in name:
                iscale.set_scaling_factor(var, 10)
            elif "X_TSS" in name:
                iscale.set_scaling_factor(var, 1)
            else:
                iscale.set_scaling_factor(var, 10)

        elif "alkalinity" in name:
            iscale.set_scaling_factor(var, 1.0)  # actual magnitude ~0.1-2.3 mol/m3

        elif "rate_reaction_extent" in name:
            iscale.set_scaling_factor(var, 1e4)

        elif "rate_reaction_generation" in name:
            iscale.set_scaling_factor(var, 1e3)

        elif "reaction_rate" in name:
            iscale.set_scaling_factor(var, 1e6)

        elif "hydraulic_retention_time" in name:
            iscale.set_scaling_factor(var, 1e-3)

        elif "electricity_consumption" in name:
            iscale.set_scaling_factor(var, 1e-2)

        elif "mass_transfer_term" in name:
            iscale.set_scaling_factor(var, 1e2)

    for R, sf in [(m.fs.R1, 1e-3), (m.fs.R2, 1e-3)]:
        iscale.set_scaling_factor(R.volume, sf)
        iscale.set_scaling_factor(R.control_volume.volume, sf)

    iscale.calculate_scaling_factors(m.fs)


def initialize_flowsheet(m):
    """Straight-line sequential initialization; no recycle to close."""
    _outlvl = idaeslog.WARNING
    m.fs.feed.initialize(outlvl=_outlvl)
    propagate_state(m.fs.feed_to_r1)
    m.fs.R1.initialize(outlvl=_outlvl)
    propagate_state(m.fs.r1_to_r2)
    m.fs.R2.initialize(outlvl=_outlvl)
    propagate_state(m.fs.r2_to_treated)
    m.fs.Treated.initialize(outlvl=_outlvl)


def solve_flowsheet(m):
    solver = get_solver()
    results = solver.solve(m, tee=True)
    check_solve(
        results, checkpoint="simplified R1->R2 solve", logger=_log, fail_flag=False
    )
    return results


def verify_against_ode(m):
    """Compare converged R1.outlet and R2.outlet to UConn ODE reference."""
    species = [
        "X_H",
        "X_STO",
        "X_S",
        "X_A",
        "X_I",
        "X_TSS",
        "S_S",
        "S_O",
        "S_NH4",
        "S_NOX",
        "S_N2",
        "S_I",
    ]

    def _print_block(label, state, ref, note=""):
        print(f"\n--- {label} {note}---")
        print(f"{'Component':<12}{'WaterTAP':>14}{'UConn':>14}{'Diff %':>10}")
        for sp in species:
            wt = pyo.value(state.conc_mass_comp[sp]) * 1e3
            od = ref[sp]
            pdiff = (wt - od) / od * 100 if od != 0 else float("nan")
            print(f"{sp:<12}{wt:>14.4f}{od:>14.4f}{pdiff:>+10.1f}")
        wt_alk = pyo.value(state.alkalinity)
        od_alk = ref["alkalinity"]
        pdiff = (wt_alk - od_alk) / od_alk * 100
        print(f"{'alkalinity':<12}{wt_alk:>14.4f}{od_alk:>14.4f}{pdiff:>+10.1f}")

    print("\n" + "=" * 80)
    print("SIMPLIFIED 2-REACTOR VALIDATION vs UConn ODE SS")
    print("=" * 80)
    _print_block(
        "R1.outlet (-> UConn Mixer 2 In1)",
        m.fs.R1.control_volume.properties_out[0],
        ODE_R1_OUT_REF,
    )
    _print_block(
        "R2.outlet (-> UConn Mixer 2 In2)",
        m.fs.R2.control_volume.properties_out[0],
        ODE_R2_OUT_REF,
        note="[NOTE: R2 here is fed R1's outlet directly, not the true "
        "S1/R5 recycle stream R2 sees in the full model -- treat as a "
        "kinetics sanity check, not a literal SS reproduction] ",
    )


if __name__ == "__main__":
    m = build_flowsheet()
    set_operating_conditions(m)
    scale_flowsheet(m)
    initialize_flowsheet(m)

    dt = DiagnosticsToolbox(m)
    res = solve_flowsheet(m)
    solved = res.solver.termination_condition == TerminationCondition.optimal

    if not solved:
        print("\n--- Post-solve diagnostics ---")
        dt.report_numerical_issues()
        dt.display_constraints_with_large_residuals()
        dt.display_variables_at_or_outside_bounds()
        try:
            dt.compute_infeasibility_explanation()
        except Exception as e:
            print(f"Infeasibility explanation failed: {e}")
    else:
        stream_table = create_stream_table_dataframe(
            {
                "Feed": m.fs.feed.outlet,
                "R1": m.fs.R1.outlet,
                "R2": m.fs.R2.outlet,
                "Treated": m.fs.Treated.inlet,
            },
            time_point=0,
        )
        print(stream_table_dataframe_to_string(stream_table))
        verify_against_ode(m)
