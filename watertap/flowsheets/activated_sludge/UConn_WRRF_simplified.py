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

    # Oxygen saturation concentration (S_O_eq) that KLa drives S_O toward.
    # WaterTAP's default is 8.0 mg/L; back-calculating from UConn's own
    # solved S_O outlet (given our reaction kinetics are independently
    # verified correct) implies ~9.03 mg/L, close to the standard
    # clean-water DO saturation at 20C/1atm (commonly cited ~9.08 mg/L).
    # Set explicitly to the standard literature value rather than
    # reverse-fitting UConn's exact number, to keep the correction
    # principled rather than circular.
    m.fs.R2.S_O_eq.set_value(9.08e-3 * pyo.units.kg / pyo.units.m**3)

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
    # Default tol/constr_viol_tol=1e-08 is an ABSOLUTE tolerance. In R1's
    # near-zero-S_NOX anoxic regime, several reaction rates (e.g. R9) have
    # true magnitudes around 1e-12 kg/m3/s -- far below 1e-08. IPOPT can
    # then leave residual noise on the order of its own tolerance inside
    # such a tiny-magnitude constraint and still report "converged", even
    # though that noise is >1000x the true value (confirmed: the model's
    # reported "Constraint violation" exactly equals R9's own equation
    # residual in one such case). Tightening the tolerance forces IPOPT to
    # resolve these tiny rates to physically meaningful precision instead.
    solver.options["tol"] = 1e-12
    solver.options["constr_viol_tol"] = 1e-12
    solver.options["acceptable_constr_viol_tol"] = 1e-12
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


def _arrhenius_value(param_var, params_block, T_kelvin_value):
    """Independently recompute the Arrhenius-adjusted value of a
    temperature-dependent parameter, using the model's OWN index names
    (read dynamically, not hardcoded) and the reactor's actual solved
    temperature -- rather than trusting that T=293.15K exactly zeroes out
    the exponent term algebraically, or assuming a specific index-key
    naming convention that may differ across asm3_reactions.py versions
    (e.g. "ref_temp_1"/"ref_temp_2" vs "10C"/"20C" vs other labels).

        theta = ln(k(low_T_idx) / k(high_T_idx)) / (low_T_val - high_T_val)
        k(T)  = k(high_T_idx) * exp(theta * (T[K] - (high_T_val[C] + 273.15)))

    where low/high are identified by comparing the two INDEX VALUES on
    params_block.ref_temp_1/ref_temp_2 (whichever is smaller is "low").
    """
    import math

    idx_keys = list(param_var.index_set())
    if len(idx_keys) != 2:
        raise ValueError(
            f"Expected a 2-entry temperature-indexed parameter, got keys {idx_keys}"
        )
    # Do NOT match index keys to ref_temp_1/ref_temp_2 by position -- that
    # is unreliable (caused a real bug: it silently returned the 10C value
    # where the 20C value was needed, since index-set iteration order is
    # not guaranteed to match declaration order of ref_temp_1/ref_temp_2).
    # Instead, use the ASM3 physical convention directly: every one of
    # these rate constants (k_H, k_STO, mu_H, mu_A, b_*) is LARGER at 20C
    # than at 10C (biological rates increase with temperature), so we can
    # identify "high" and "low" purely from the parameter's OWN two values,
    # with no dependency on ref_temp_1/ref_temp_2's index-key names at all.
    v0 = pyo.value(param_var[idx_keys[0]])
    v1 = pyo.value(param_var[idx_keys[1]])
    if v0 >= v1:
        key_high, key_low = idx_keys[0], idx_keys[1]
        p_high, p_low = v0, v1
    else:
        key_high, key_low = idx_keys[1], idx_keys[0]
        p_high, p_low = v1, v0

    rt_low = pyo.value(params_block.ref_temp_1)
    rt_high = pyo.value(params_block.ref_temp_2)
    if rt_low > rt_high:
        rt_low, rt_high = rt_high, rt_low

    theta = math.log(p_low / p_high) / (rt_low - rt_high)
    exponent = T_kelvin_value - (rt_high + 273.15)
    return p_high * math.exp(theta * exponent)


def print_kinetic_parameters(m):
    """Print every non-calibrated ASM3 kinetic/stoichiometric parameter,
    side-by-side with UConn's stated Gujer (1999) 20 C defaults. The 5
    explicitly calibrated parameters per reactor (K_NOX, mu_H, mu_A,
    Y_STO_O2, Y_H_NOX) are excluded here since we already set those to
    UConn's calibrated values directly -- this checks everything ELSE, to
    confirm no other parameter silently differs from Gujer 1999.

    For temperature-dependent parameters, the value shown is INDEPENDENTLY
    RECOMPUTED via _arrhenius_value() using the reactor's actual solved
    temperature -- not assumed equal to the raw ref_temp_2 (20C) dict
    entry. A separate check below prints both the recomputed value and the
    raw ref_temp_2 entry side by side, so any discrepancy (e.g. if reactor
    temperature drifted from 293.15K) is directly visible rather than
    assumed away.
    """
    UCONN_DEFAULTS = {
        # name: (UConn value, is_temperature_dependent, native_units_note)
        "k_H": (3.0, True, "day^-1"),
        "K_X": (1.0, False, "-"),
        "k_STO": (5.0, True, "day^-1"),
        "eta_NOX": (0.6, False, "-"),
        "K_O2": (0.2, False, "mg/L"),
        "K_S": (2.0, False, "mg/L"),
        "K_STO": (1.0, False, "-"),
        "K_NH4": (0.01, False, "mg/L"),
        "K_ALK": (0.1, False, "mol/m3"),
        "b_H_O2": (0.2, True, "day^-1"),
        "b_H_NOX": (0.1, True, "day^-1"),
        "b_STO_O2": (0.2, True, "day^-1"),
        "b_STO_NOX": (0.1, True, "day^-1"),
        "K_A_NH4": (1.0, False, "mg/L"),
        "K_A_O2": (0.5, False, "mg/L"),
        "K_A_ALK": (0.5, False, "mol/m3"),
        "b_A_O2": (0.15, True, "day^-1"),
        "b_A_NOX": (0.05, True, "day^-1"),
    }
    # Stoichiometric parameters that live on the property package (shared,
    # not per-reactor)
    UCONN_PROPS_DEFAULTS = {
        "f_SI": 0.0,
        "f_XI": 0.20,
        "i_NSI": 0.01,
        "i_NSS": 0.03,
        "i_NXI": 0.02,
        "i_NXS": 0.04,
        "i_NBM": 0.07,
        "i_SSXI": 0.75,
        "i_SSXS": 0.75,
        "i_SSBM": 0.90,
        "i_SSSTO": 0.60,
    }
    # Reaction-block stoichiometric parameters not shared via property package
    UCONN_RXN_STOICH_DEFAULTS = {
        "Y_A": 0.24,
        "Y_H_O2": 0.63,
        "Y_STO_NOX": 0.80,
    }
    # kg/m3 -> mg/L is *1e3; kmol/m3 -> mol/m3 is *1e3; dimensionless/day^-1: *1
    CONC_UNIT_PARAMS = {"K_O2", "K_S", "K_NH4", "K_A_NH4", "K_A_O2"}
    ALK_UNIT_PARAMS = {"K_ALK", "K_A_ALK"}

    print("\n" + "=" * 78)
    print(
        "NON-CALIBRATED ASM3 PARAMETERS: WaterTAP (at T=293.15K) vs UConn (Gujer 1999, 20C)"
    )
    print("=" * 78)

    for label, rxn, cv in [
        ("R1", m.fs.rxn_props_R1, m.fs.R1.control_volume),
        ("R2", m.fs.rxn_props_R2, m.fs.R2.control_volume),
    ]:
        # Actual solved reactor temperature (K) -- used for the Arrhenius
        # recomputation instead of assuming it's exactly 293.15
        T_actual = pyo.value(
            pyo.units.convert(cv.properties_out[0].temperature, to_units=pyo.units.K)
        )
        print(
            f"\n--- {label} reaction package (actual solved T = {T_actual:.4f} K) ---"
        )
        print(
            f"{'Parameter':<12}{'Arrhenius':>14}{'raw ref_2':>14}{'UConn':>14}{'Diff %':>10}  Units"
        )
        for name, (uconn_val, is_temp_dep, unit_note) in UCONN_DEFAULTS.items():
            var = getattr(rxn, name)
            if is_temp_dep:
                # Independently recompute the actual Arrhenius-adjusted value
                # used in the model, using the reactor's real solved T -- do
                # NOT assume it equals the raw high-temperature dict entry.
                wt_val = _arrhenius_value(var, rxn, T_actual)
                # "raw" shortcut: read whichever index key has the LARGER
                # numeric value (== the 20C entry, per ASM3's convention
                # that rates increase with temperature) -- identified
                # directly from the parameter's own two values, not by
                # matching index-key ORDER against ref_temp_1/ref_temp_2
                # (that positional matching was the bug that caused this
                # to silently return the 10C value instead of 20C).
                idx_keys = list(var.index_set())
                v0 = pyo.value(var[idx_keys[0]])
                v1 = pyo.value(var[idx_keys[1]])
                key_high = idx_keys[0] if v0 >= v1 else idx_keys[1]
                raw_ref2 = pyo.value(var[key_high])
            else:
                wt_val = pyo.value(var)
                raw_ref2 = wt_val
            if name in CONC_UNIT_PARAMS:
                wt_val *= 1e3  # kg/m3 -> mg/L
                raw_ref2 *= 1e3
            elif name in ALK_UNIT_PARAMS:
                wt_val *= 1e3  # kmol/m3 -> mol/m3
                raw_ref2 *= 1e3
            pdiff = (
                (wt_val - uconn_val) / uconn_val * 100
                if uconn_val != 0
                else float("nan")
            )
            print(
                f"{name:<12}{wt_val:>14.5f}{raw_ref2:>14.5f}{uconn_val:>14.5f}{pdiff:>+10.2f}  {unit_note}"
            )

        print(f"\n  Reaction-block stoichiometric parameters ({label}):")
        for name, uconn_val in UCONN_RXN_STOICH_DEFAULTS.items():
            var = getattr(rxn, name)
            wt_val = pyo.value(var)
            pdiff = (
                (wt_val - uconn_val) / uconn_val * 100
                if uconn_val != 0
                else float("nan")
            )
            print(f"  {name:<10}{wt_val:>14.5f}{uconn_val:>14.5f}{pdiff:>+10.2f}")

    print("\n--- Property package (shared across all reactors) ---")
    print(f"{'Parameter':<12}{'WaterTAP':>14}{'UConn':>14}{'Diff %':>10}")
    for name, uconn_val in UCONN_PROPS_DEFAULTS.items():
        var = getattr(m.fs.props, name)
        wt_val = pyo.value(var)
        pdiff = (
            (wt_val - uconn_val) / uconn_val * 100 if uconn_val != 0 else float("nan")
        )
        print(f"{name:<12}{wt_val:>14.5f}{uconn_val:>14.5f}{pdiff:>+10.2f}")

    print("=" * 78)


def diagnose_alkalinity_generation(m):
    """Directly check whether alkalinity's reaction generation term is being
    computed AND applied in the control volume's material balance -- rather
    than continuing to reason about this abstractly. This independently
    recomputes the EXPECTED alkalinity generation (sum over all 12 reactions
    of extent[r] * stoichiometry[r,"S_ALK"]) and compares it against:
      (a) whatever the control volume itself reports as rate_reaction_generation
          for S_ALK (if that attribute exists and is indexed the way expected)
      (b) the ACTUAL observed change in alkalinity across the reactor
          (flow * (alk_out - alk_in)), which is what the state variables
          themselves show happened

    If (a) is zero/missing while the recomputed stoichiometric sum is
    nonzero, that pinpoints a disconnect between the reaction package's
    correctly-computed stoichiometry and the control volume's material
    balance failing to apply it for this specific component.
    """
    for label, R, cv, rxn in [
        ("R1", m.fs.R1, m.fs.R1.control_volume, m.fs.rxn_props_R1),
        ("R2", m.fs.R2, m.fs.R2.control_volume, m.fs.rxn_props_R2),
    ]:
        print(f"\n--- {label}: alkalinity generation diagnostic ---")

        # (a) What the control volume itself reports, if accessible
        try:
            reported_gen = pyo.value(cv.rate_reaction_generation[0, "Liq", "S_ALK"])
            print(
                f"  cv.rate_reaction_generation[.., S_ALK] = {reported_gen:.6e} kg/s"
                f"  (mass basis, via mw_alk=61 baked into z1..z12)"
            )
        except (AttributeError, KeyError) as e:
            print(f"  cv.rate_reaction_generation[.., S_ALK] NOT accessible: {e}")
            reported_gen = None

        # (b) Independently recompute expected generation from reaction
        # extents and the reaction package's own stoichiometry dict
        try:
            recomputed_gen = 0.0
            for r in rxn.rate_reaction_idx:
                stoich_key = (r, "Liq", "S_ALK")
                if stoich_key in rxn.rate_reaction_stoichiometry:
                    coeff = pyo.value(rxn.rate_reaction_stoichiometry[stoich_key])
                    extent = pyo.value(cv.rate_reaction_extent[0, r])
                    recomputed_gen += coeff * extent
            print(
                f"  recomputed sum(extent[r] * stoich[r,S_ALK])  = {recomputed_gen:.6e} kg/s"
            )
        except AttributeError as e:
            print(f"  Could not recompute from extents: {e}")
            recomputed_gen = None

        # (c) Observed actual change in alkalinity across the reactor
        alk_in = pyo.value(
            pyo.units.convert(
                cv.properties_in[0].alkalinity, to_units=pyo.units.mol / pyo.units.m**3
            )
        )
        alk_out = pyo.value(
            pyo.units.convert(
                cv.properties_out[0].alkalinity, to_units=pyo.units.mol / pyo.units.m**3
            )
        )
        flow = pyo.value(cv.properties_in[0].flow_vol)
        # generation is reported on a MASS basis (kg/s, via the fixed x61
        # mw_alk conversion in z1..z12), so convert the molar concentration
        # change to a matching mass-flow rate for a fair comparison:
        # mol/s -> kmol/s (x1e-3) -> kg/s (x61, mw_alk)
        observed_change_rate = flow * (alk_out - alk_in) * 1e-3 * 61.0
        print(f"  alkalinity in={alk_in:.4f}, out={alk_out:.4f} mol/m3")
        print(
            f"  observed flow*(alk_out - alk_in)*mw_alk = {observed_change_rate:.6e} kg/s"
            f"  (should equal generation term if properly coupled)"
        )


def diagnose_reactor_detail(m, label, R, cv):
    """For a given reactor, print:
    (1) every reaction's extent (kg/s) -- to see which of the 12 ASM3
        reactions are actually "active" (nonzero) in this reactor, since
        R1 being anoxic should mean all AEROBIC-pathway reactions (2, 4,
        6, 8, 10, 11) have ~zero extent (their rate is proportional to
        the Monod term S_O/(K_O2+S_O), which is exactly 0 when S_O=0)
    (2) S_O's own generation/consumption term specifically, and whether
        S_O sits at a bound (which would explain a nonzero "anoxic"
        value as a solver/bound artifact rather than a real reaction
        product -- ASM3 has no reaction that PRODUCES oxygen, so any
        nonzero S_O in a zero-aeration reactor must come from either
        the inlet or a bound/tolerance artifact, not net generation)
    """
    print(f"\n--- {label}: reaction extent + S_O detail ---")

    print(f"  {'Reaction':<10}{'extent (kg/s)':>16}")
    for r in sorted(
        m.fs.rxn_props_R1.rate_reaction_idx
        if label == "R1"
        else m.fs.rxn_props_R2.rate_reaction_idx
    ):
        extent = pyo.value(cv.rate_reaction_extent[0, r])
        flag = "  <- active" if abs(extent) > 1e-12 else ""
        print(f"  {r:<10}{extent:>16.6e}{flag}")

    try:
        gen_SO = pyo.value(cv.rate_reaction_generation[0, "Liq", "S_O"])
        print(f"\n  rate_reaction_generation[.., S_O] = {gen_SO:.6e} kg/s")
    except (AttributeError, KeyError) as e:
        print(f"\n  rate_reaction_generation[.., S_O] not accessible: {e}")

    S_O_in = pyo.value(cv.properties_in[0].conc_mass_comp["S_O"]) * 1e3
    S_O_out = pyo.value(cv.properties_out[0].conc_mass_comp["S_O"]) * 1e3
    print(f"  S_O in={S_O_in:.6f} mg/L, out={S_O_out:.6f} mg/L")

    # Check if S_O is sitting at its lower bound (a bound/tolerance
    # artifact would show up as the solved value being suspiciously close
    # to whatever floor the variable's domain/bounds impose)
    var = cv.properties_out[0].conc_mass_comp["S_O"]
    lb = var.lb
    print(f"  S_O outlet variable lower bound = {lb}")
    print(
        f"  S_O outlet value - lower bound  = {pyo.value(var) - (lb if lb is not None else 0):.6e} kg/m3"
    )


def diagnose_storage_reactions(m, label, R, cv, rxn):
    """Break down R2/R3/R8/R9 (aerobic/anoxic storage + storage respiration)
    into their individual Monod-term components, to pin down exactly what
    is driving X_STO's mismatch rather than just looking at net extents.

        R2 (aerobic storage):   k_STO * [S_O/(K_O2+S_O)] * [S_S/(K_S+S_S)] * X_H
        R3 (anoxic storage):    k_STO * eta_NOX * [K_O2/(K_O2+S_O)]
                                 * [S_NOX/(K_NOX+S_NOX)] * [S_S/(K_S+S_S)] * X_H
        R8 (aerobic STO resp):  b_STO_O2 * [S_O/(K_O2+S_O)] * X_STO
        R9 (anoxic STO resp):   b_STO_NOX * [K_O2/(K_O2+S_O)]
                                 * [S_NOX/(K_NOX+S_NOX)] * X_STO
    """
    S_O = pyo.value(cv.properties_out[0].conc_mass_comp["S_O"]) * 1e3  # mg/L
    S_S = pyo.value(cv.properties_out[0].conc_mass_comp["S_S"]) * 1e3  # mg/L
    S_NOX = pyo.value(cv.properties_out[0].conc_mass_comp["S_NOX"]) * 1e3  # mg/L
    X_H = pyo.value(cv.properties_out[0].conc_mass_comp["X_H"]) * 1e3  # mg/L
    X_STO = pyo.value(cv.properties_out[0].conc_mass_comp["X_STO"]) * 1e3  # mg/L

    K_O2 = pyo.value(rxn.K_O2) * 1e3  # mg/L
    K_S = pyo.value(rxn.K_S) * 1e3  # mg/L
    K_NOX = pyo.value(rxn.K_NOX) * 1e3  # mg/L

    mon_O2 = S_O / (K_O2 + S_O)
    mon_O2_inv = K_O2 / (K_O2 + S_O)
    mon_S = S_S / (K_S + S_S)
    mon_NOX = S_NOX / (K_NOX + S_NOX)

    print(f"\n--- {label}: storage-reaction Monod term breakdown ---")
    print(
        f"  State: S_O={S_O:.6f} S_S={S_S:.4f} S_NOX={S_NOX:.4f} X_H={X_H:.4f} X_STO={X_STO:.4f} mg/L"
    )
    print(f"  K_O2={K_O2:.4f} K_S={K_S:.4f} K_NOX={K_NOX:.4f} mg/L")
    print(f"  Monod[S_O/(K_O2+S_O)]     = {mon_O2:.6f}   (gates R2, R8 -- aerobic)")
    print(
        f"  Monod[K_O2/(K_O2+S_O)]    = {mon_O2_inv:.6f}   (gates R3, R9 -- anoxic, O2-inhibition)"
    )
    print(f"  Monod[S_S/(K_S+S_S)]      = {mon_S:.6f}   (gates R2, R3 -- substrate)")
    print(
        f"  Monod[S_NOX/(K_NOX+S_NOX)] = {mon_NOX:.6f}   (gates R3, R9 -- electron acceptor)"
    )

    for r in ["R2", "R3", "R8", "R9"]:
        extent = pyo.value(cv.rate_reaction_extent[0, r])
        print(f"  extent[{r}] = {extent:.6e} kg/s")


def diagnose_R9_rate_directly(m, label, R, cv, rxn):
    """R9's reported extent is ~1792x larger than the textbook Monod formula
    (b_STO_NOX * [K_O2/(K_O2+S_O)] * [S_NOX/(K_NOX+S_NOX)] * X_STO) predicts
    given the actual solved state -- read the RAW reaction_rate Var directly
    (not the extent) and print every individual factor to pin down exactly
    where the model's computation diverges from the textbook formula.
    """
    print(f"\n--- {label}: R9 rate -- raw variable vs hand-formula ---")

    # Raw rate variable itself (kg/m3/s) -- try a few possible attribute
    # paths since the exact location (unit model vs control volume) isn't
    # certain without live access to the installed IDAES/WaterTAP version
    raw_rate = None
    for attr_path, getter in [
        ("R.reactions[0].reaction_rate", lambda: R.reactions[0].reaction_rate["R9"]),
        ("cv.reactions[0].reaction_rate", lambda: cv.reactions[0].reaction_rate["R9"]),
        (
            "R.control_volume.reactions[0].reaction_rate",
            lambda: R.control_volume.reactions[0].reaction_rate["R9"],
        ),
    ]:
        try:
            raw_rate = pyo.value(getter())
            print(f"  {attr_path}['R9'] = {raw_rate:.6e} kg/m3/s")
            break
        except (AttributeError, KeyError) as e:
            print(f"  {attr_path} not accessible: {e}")
    if raw_rate is None:
        print("  Could not locate reaction_rate via any attempted path.")

    extent = pyo.value(cv.rate_reaction_extent[0, "R9"])
    volume = pyo.value(R.volume[0]) if hasattr(R, "volume") else None
    print(f"  cv.rate_reaction_extent[0,'R9'] = {extent:.6e} kg/s")
    if volume is not None:
        print(f"  R.volume = {volume} m3")
        print(
            f"  extent / volume = {extent/volume:.6e} kg/m3/s  (compare to raw rate above)"
        )

    # Recompute the textbook Monod formula from scratch, all factors shown
    b_STO_NOX = (
        pyo.value(rxn.b_STO_NOX["ref_temp_2"])
        if hasattr(rxn.b_STO_NOX, "index_set")
        and len(list(rxn.b_STO_NOX.index_set())) > 1
        else pyo.value(rxn.b_STO_NOX)
    )
    K_O2 = pyo.value(rxn.K_O2)  # kg/m3
    K_NOX = pyo.value(rxn.K_NOX)  # kg/m3
    S_O = pyo.value(cv.properties_out[0].conc_mass_comp["S_O"])  # kg/m3
    S_NOX = pyo.value(cv.properties_out[0].conc_mass_comp["S_NOX"])  # kg/m3
    X_STO = pyo.value(cv.properties_out[0].conc_mass_comp["X_STO"])  # kg/m3

    mon_O2_inv = K_O2 / (K_O2 + S_O)
    mon_NOX = S_NOX / (K_NOX + S_NOX)

    # b_STO_NOX has units day^-1 in the source -- convert to s^-1 for a
    # kg/m3/s rate matching reaction_rate's declared units
    b_STO_NOX_per_s = b_STO_NOX / 86400.0

    hand_rate = b_STO_NOX_per_s * mon_O2_inv * mon_NOX * X_STO
    print(f"\n  Hand-formula factors (all in native kg/m3, s^-1 basis):")
    print(f"  b_STO_NOX = {b_STO_NOX} /day = {b_STO_NOX_per_s:.6e} /s")
    print(f"  K_O2/(K_O2+S_O)     = {mon_O2_inv:.6f}")
    print(f"  S_NOX/(K_NOX+S_NOX) = {mon_NOX:.6f}")
    print(f"  X_STO = {X_STO:.6e} kg/m3")
    print(
        f"  hand_rate = b_STO_NOX_per_s * mon_O2_inv * mon_NOX * X_STO = {hand_rate:.6e} kg/m3/s"
    )
    if raw_rate is not None:
        print(f"  raw_rate / hand_rate = {raw_rate/hand_rate:.2f}x")


def diagnose_R9_internal_state(m, label, R, cv, rxn):
    """The formula for R9 in the source code is confirmed IDENTICAL to the
    hand-derived Monod formula, yet the raw rate is 1792x larger than that
    formula predicts using cv.properties_out[0] values. This means
    b.conc_mass_comp_ref[...] as seen FROM INSIDE the reaction block's own
    constraint must differ from what we're reading externally via
    cv.properties_out[0] -- read the SAME quantities directly from the
    reaction block itself (cv.reactions[0]) to check.
    """
    print(f"\n--- {label}: R9 internal state (read from inside cv.reactions[0]) ---")
    rxn_block = cv.reactions[0]

    try:
        S_O_internal = pyo.value(rxn_block.conc_mass_comp_ref["S_O"])
        S_NOX_internal = pyo.value(rxn_block.conc_mass_comp_ref["S_NOX"])
        X_STO_internal = pyo.value(rxn_block.conc_mass_comp_ref["X_STO"])
        print(
            f"  rxn_block.conc_mass_comp_ref['S_O']   = {S_O_internal:.6e} kg/m3"
            f"  ({S_O_internal*1e3:.6f} mg/L)"
        )
        print(
            f"  rxn_block.conc_mass_comp_ref['S_NOX'] = {S_NOX_internal:.6e} kg/m3"
            f"  ({S_NOX_internal*1e3:.6f} mg/L)"
        )
        print(
            f"  rxn_block.conc_mass_comp_ref['X_STO'] = {X_STO_internal:.6e} kg/m3"
            f"  ({X_STO_internal*1e3:.6f} mg/L)"
        )
    except (AttributeError, KeyError) as e:
        print(f"  Could not read conc_mass_comp_ref directly: {e}")
        S_O_internal = S_NOX_internal = X_STO_internal = None

    print(
        f"\n  For comparison, cv.properties_out[0] (what we've been using externally):"
    )
    S_O_ext = pyo.value(cv.properties_out[0].conc_mass_comp["S_O"])
    S_NOX_ext = pyo.value(cv.properties_out[0].conc_mass_comp["S_NOX"])
    X_STO_ext = pyo.value(cv.properties_out[0].conc_mass_comp["X_STO"])
    print(
        f"  cv.properties_out[0].conc_mass_comp['S_O']   = {S_O_ext:.6e} kg/m3  ({S_O_ext*1e3:.6f} mg/L)"
    )
    print(
        f"  cv.properties_out[0].conc_mass_comp['S_NOX'] = {S_NOX_ext:.6e} kg/m3  ({S_NOX_ext*1e3:.6f} mg/L)"
    )
    print(
        f"  cv.properties_out[0].conc_mass_comp['X_STO'] = {X_STO_ext:.6e} kg/m3  ({X_STO_ext*1e3:.6f} mg/L)"
    )

    # Also check K_O2, K_NOX as seen from the reaction block's own params ref
    try:
        K_O2_internal = pyo.value(rxn_block.params.K_O2)
        K_NOX_internal = pyo.value(rxn_block.params.K_NOX)
        print(
            f"\n  rxn_block.params.K_O2  = {K_O2_internal:.6e} kg/m3  ({K_O2_internal*1e3:.4f} mg/L)"
        )
        print(
            f"  rxn_block.params.K_NOX = {K_NOX_internal:.6e} kg/m3  ({K_NOX_internal*1e3:.4f} mg/L)"
        )
    except AttributeError as e:
        print(f"  Could not read params directly: {e}")

    # Directly recompute the rate using ONLY internally-read values, to
    # confirm whether the discrepancy disappears when using the reaction
    # block's own view of the state (isolating "wrong state" vs "wrong
    # formula/units inside the constraint" as the root cause)
    if S_O_internal is not None:
        b_STO_NOX_per_s = 0.1 / 86400.0
        K_O2_i = pyo.value(rxn_block.params.K_O2)
        K_NOX_i = pyo.value(rxn_block.params.K_NOX)
        mon_O2_inv_i = K_O2_i / (K_O2_i + S_O_internal)
        mon_NOX_i = S_NOX_internal / (K_NOX_i + S_NOX_internal)
        rate_from_internal = b_STO_NOX_per_s * mon_O2_inv_i * mon_NOX_i * X_STO_internal
        raw_rate = pyo.value(rxn_block.reaction_rate["R9"])
        print(
            f"\n  Rate recomputed using ONLY internally-read values = {rate_from_internal:.6e} kg/m3/s"
        )
        print(
            f"  Actual rxn_block.reaction_rate['R9']                = {raw_rate:.6e} kg/m3/s"
        )
        print(f"  ratio = {raw_rate/rate_from_internal:.4f}")


def diagnose_aeration_mass_transfer(m):
    """R2's remaining discrepancies (S_O -14.6%, S_N2 +10.2%) are the only
    two species subject to gas-phase mass transfer in the AerationTank.
    Print whatever mass-transfer mechanism/parameters (KLa, saturation
    concentrations, mass_transfer_term for each species) the model
    actually uses, since we don't have the AerationTank source in this
    session -- extract it directly from the live model instead.
    """
    R2 = m.fs.R2
    print("\n" + "=" * 78)
    print("R2 (AerationTank) mass transfer diagnostic")
    print("=" * 78)

    try:
        kla_val = pyo.value(R2.KLa)
        print(f"\nKLa (fixed) = {kla_val}")
    except (TypeError, KeyError):
        kla_val = pyo.value(R2.KLa[0])
        print(f"\nKLa (fixed) = {kla_val}")

    # Try to find a dissolved-oxygen saturation concentration variable/param
    # under a few plausible names
    for name in [
        "S_O_sat",
        "S_su",
        "S_O_equil",
        "DOsat",
        "S_O_eq",
        "outlet_pressure_gas",
        "P_air",
    ]:
        if hasattr(R2, name):
            var = getattr(R2, name)
            try:
                print(f"  R2.{name} = {pyo.value(var)}")
            except Exception as e:
                print(f"  R2.{name} exists but could not evaluate: {e}")

    # Print the mass_transfer_term for every species, if it exists --
    # this directly shows which species get gas exchange and by how much
    print("\nmass_transfer_term (if present) for every component:")
    for attr_name in ["mass_transfer_term", "mass_transfer"]:
        if hasattr(R2.control_volume, attr_name):
            mt = getattr(R2.control_volume, attr_name)
            try:
                for key in mt:
                    val = pyo.value(mt[key])
                    if abs(val) > 1e-15:
                        print(f"  {attr_name}{key} = {val:.6e}")
            except Exception as e:
                print(f"  Could not iterate {attr_name}: {e}")

    # List ALL variables/params on R2 whose name suggests O2 or N2 transfer,
    # saturation, or gas-phase behavior, to catch anything not anticipated
    # by the specific names tried above
    print(
        "\nAll R2 components with 'sat', 'gas', 'transfer', 'equil', or 'KLa' in the name:"
    )
    for v in R2.component_objects(pyo.Var, descend_into=True):
        vname = v.local_name.lower()
        if any(kw in vname for kw in ["sat", "gas", "transfer", "equil", "kla"]):
            try:
                for idx in v:
                    val = pyo.value(v[idx])
                    print(f"  {v.name}[{idx}] = {val}")
            except Exception:
                try:
                    print(f"  {v.name} = {pyo.value(v)}")
                except Exception:
                    pass


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
        # print_kinetic_parameters(m)
        # diagnose_alkalinity_generation(m)
        # diagnose_reactor_detail(m, "R1", m.fs.R1, m.fs.R1.control_volume)
        # diagnose_reactor_detail(m, "R2", m.fs.R2, m.fs.R2.control_volume)
        # diagnose_storage_reactions(m, "R1", m.fs.R1, m.fs.R1.control_volume, m.fs.rxn_props_R1)
        # diagnose_storage_reactions(m, "R2", m.fs.R2, m.fs.R2.control_volume, m.fs.rxn_props_R2)
        # diagnose_R9_rate_directly(m, "R1", m.fs.R1, m.fs.R1.control_volume, m.fs.rxn_props_R1)
        # diagnose_R9_internal_state(m, "R1", m.fs.R1, m.fs.R1.control_volume, m.fs.rxn_props_R1)
        # diagnose_aeration_mass_transfer(m)
