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

Layout:
    Feed -> R1 (anoxic CSTR) -> R2 (aerobic AerationTank) -> Product
    No mixer, splitter, or clarifier -- matches UConn's purpose-built
    2-tank validation case exactly (topology, feed, volumes, and KLa all
    come directly from UConn, not borrowed from the 5-reactor flowsheet).

UConn reference case parameters:
    inlet_flowrate = 1000 m3/day
    R1 volume = 500 m3 (anoxic)
    R2 volume = 500 m3 (aerobic, KLa = 240 /day = 10 /hour)
    Feed COD_total = 416.5 mg/L, split via frac_SI/frac_SS/frac_XI/frac_XS
    Feed also carries X_H = X_A = 100 mg/L directly (no recycle to build up
    biomass in this simplified 2-tank case, so it must be seeded in the feed)
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
# UConn 2-tank reference data (mg/L unless noted; alkalinity mol/m3)
# Source: UConn team's simplified 2-tank simulation (no mixer/splitter/
# clarifier), using the SAME per-reactor calibrated ASM3 parameters as the
# full 5-reactor flowsheet, plus unmodified Gujer (1999) 20C defaults for
# everything else. This replaces the earlier ODE_* reference data, which
# was borrowed from the 5-reactor recycle model and required caveats for
# R2 (fed by a different, recycle-derived stream there).
# ---------------------------------------------------------------------------

# "Reactor 1 Inlet" -- used directly as the feed composition for this model.
# COD_total=416.5 mg/L split via frac_SI=0.034056, frac_SS=0.335454,
# frac_XI=0.181640, frac_XS=0.448850 (sum to 1); X_H, X_A seeded directly
# since there is no recycle in this topology to build up biomass.
UCONN_FEED = {
    "X_H": 100.0,
    "X_STO": 0.0,
    "X_S": 186.9461372932193,
    "X_A": 100.0,
    "X_I": 75.65309077406194,
    "X_TSS": 166.0,
    "S_S": 139.716602774344,
    "S_O": 0.0,
    "S_NH4": 21.0,
    "S_NOX": 0.25,
    "S_N2": 0.0,
    "S_I": 14.184169158374763,
    "alkalinity": 2.3,
    "flow_m3_day": 1000.0,
}
# "Reactor 2 Inlet" == R1 Outlet -- direct comparison target for R1's kinetics
UCONN_R1_OUT_REF = {
    "X_H": 99.93837264455786,
    "X_STO": 2.1395527666631406,
    "X_S": 108.80768290198431,
    "X_A": 99.95393159813625,
    "X_I": 75.6807289467522,
    "X_TSS": 108.60369331451757,
    "S_S": 215.1168412585607,
    "S_O": 0.0,  # UConn reports -4.63e-91, i.e. numerically zero
    "S_NH4": 21.870516960679694,
    "S_NOX": 0.012447746263664145,
    "S_N2": 0.2375522537363357,
    "S_I": 14.18416915837453,
    "alkalinity": 2.379147801029703,
}
# "Reactor 2 Outlet" -- direct comparison target for R2's kinetics. Since
# this is the SAME 2-tank topology (R2 fed directly by R1's outlet, no
# recycle dilution), this is now a legitimate, literal SS reproduction
# target -- no caveat needed, unlike the old 5-reactor-derived reference.
UCONN_R2_OUT_REF = {
    "X_H": 92.09790164046834,
    "X_STO": 130.58378451504018,
    "X_S": 56.361350986423105,
    "X_A": 97.85099650953711,
    "X_I": 78.9035884913647,
    "X_TSS": 139.80356260194856,
    "S_S": 48.78070634547883,
    "S_O": 7.456323044049539,
    "S_NH4": 9.070268808414847,
    "S_NOX": 20.15473323372648,
    "S_N2": 0.6150334781409497,
    "S_I": 14.18416915837211,
    "alkalinity": 0.02610968390615832,
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
    """Fix feed composition (= UConn 2-tank "Reactor 1 Inlet"), reactor
    volumes, KLa, and calibrated kinetic parameters -- all now taken
    directly from UConn's purpose-built 2-tank validation case."""

    r = UCONN_FEED
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

    # Reactor volumes (from UConn's 2-tank case, not borrowed from the
    # 5-reactor flowsheet)
    m.fs.R1.volume.fix(500.0 * pyo.units.m**3)
    m.fs.R2.volume.fix(500.0 * pyo.units.m**3)

    # R2 aeration: no injection except S_O. KLa = 240 /day = 10 /hour
    # (UConn states 240, consistent with a day-basis rate convention
    # matching mu_H/mu_A/etc; equivalent to the 10/hour used previously)
    for j in m.fs.props.component_list:
        if j != "S_O":
            m.fs.R2.injection[:, :, j].fix(0)
    m.fs.R2.KLa.fix(240 / pyo.units.day)

    # Per-reactor calibrated scalar kinetic parameters (unchanged -- UConn's
    # 2-tank email confirms these exact values: KNOX, mu_H, mu_A, Y_STOO2,
    # Y_HNOX per reactor, matching what was already used here and in the
    # full flowsheet)
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
    """Straight-line sequential initialization."""
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
        wt_alk = pyo.value(
            pyo.units.convert(state.alkalinity, to_units=pyo.units.mol / pyo.units.m**3)
        )
        od_alk = ref["alkalinity"]
        pdiff = (wt_alk - od_alk) / od_alk * 100
        print(f"{'alkalinity':<12}{wt_alk:>14.4f}{od_alk:>14.4f}{pdiff:>+10.1f}")

    print("\n" + "=" * 80)
    print("SIMPLIFIED 2-REACTOR VALIDATION vs UConn 2-TANK REFERENCE")
    print("=" * 80)
    _print_block(
        "R1.outlet",
        m.fs.R1.control_volume.properties_out[0],
        UCONN_R1_OUT_REF,
    )
    _print_block(
        "R2.outlet",
        m.fs.R2.control_volume.properties_out[0],
        UCONN_R2_OUT_REF,
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
