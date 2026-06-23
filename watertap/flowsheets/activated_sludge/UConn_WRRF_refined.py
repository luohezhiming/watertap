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
Example of activated sludge process model, based on 5 CSTR representation of the University of Connecticut's Water Resource Recovery Facility, conceptualized by Stuber's Process Systems and Operations Research Laboratory.

Layout:
    * 5 reactors
        * R1, R3, R5: anoxic
        * R2 and R4: aerobic
    * 2 Mixers:
        * M1 mixes feed and R5 split fraction, outlet to R1 inlet
        * M2 mixes R1 outlet, R2 outlet, outlet to R3 inlet
    * 1 Splitter:
        * separates R5 outlet into effluent, M1 inlet, and R2 inlet


Unit operations are modeled as follows:

    * Anoxic reactors as standard CSTRs (CSTR)
    * Aerobic reactors as AerationTanks

"""

__author__ = "Adam Atia"

from enum import Enum, auto
import pyomo.environ as pyo
from pyomo.network import Arc, SequentialDecomposition

from idaes.core import FlowsheetBlock, UnitModelCostingBlock, UnitModelBlockData
from idaes.models.unit_models import Feed, Mixer, Separator, Product, MomentumMixingType
from idaes.models.unit_models.separator import SplittingType
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core.util.tables import (
    create_stream_table_dataframe,
    stream_table_dataframe_to_string,
)
from idaes.core.util.initialization import propagate_state

from watertap.unit_models import AerationTank, CSTR
from watertap.unit_models.clarifier import Clarifier, ClarifierScaler
from watertap.property_models.unit_specific.activated_sludge.asm1_properties import (
    ASM1ParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.asm1_reactions import (
    ASM1ReactionParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.asm3_properties import (
    ASM3ParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.asm3_reactions import (
    ASM3ReactionParameterBlock,
)
from watertap.core.util.initialization import check_solve, interval_initializer
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.clarifier import (
    cost_circular_clarifier,
    cost_primary_clarifier,
)
from idaes.core.scaling import set_scaling_factor

from idaes.core.util.model_diagnostics import DegeneracyHunter
from idaes.core.util import DiagnosticsToolbox

# Set up logger
_log = idaeslog.getLogger(__name__)


class ASMModel(auto):
    asm1 = auto()
    asm3 = auto()


def build_flowsheet(asm_model=ASMModel.asm1):
    m = pyo.ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)
    if asm_model == ASMModel.asm1:
        m.fs.props = ASM1ParameterBlock()
        m.fs.rxn_props = ASM1ReactionParameterBlock(property_package=m.fs.props)
    elif asm_model == ASMModel.asm3:
        m.fs.props = ASM3ParameterBlock()
        m.fs.rxn_props_R1 = ASM3ReactionParameterBlock(
            property_package=m.fs.props,
            calibrated_params={"mu_H": 3.8775085806642027, "mu_A": 3.3518116774465154},
        )
        m.fs.rxn_props_R2 = ASM3ReactionParameterBlock(
            property_package=m.fs.props,
            calibrated_params={"mu_H": 0.756835055976454, "mu_A": 1.6539082531538427},
        )
        m.fs.rxn_props_R3 = ASM3ReactionParameterBlock(
            property_package=m.fs.props,
            calibrated_params={"mu_H": 3.9865680124441614, "mu_A": 3.4079349519441977},
        )
        m.fs.rxn_props_R4 = ASM3ReactionParameterBlock(
            property_package=m.fs.props,
            calibrated_params={"mu_H": 5.1722430085814945, "mu_A": 5.061581300506415},
        )
        m.fs.rxn_props_R5 = ASM3ReactionParameterBlock(
            property_package=m.fs.props,
            calibrated_params={"mu_H": 6.1246767396578266, "mu_A": 2.54857697848309},
        )

    # Feed water stream
    m.fs.feed = Feed(property_package=m.fs.props)

    # Mixer for feed water and recycled sludge
    m.fs.M1 = Mixer(
        property_package=m.fs.props,
        inlet_list=["feed", "recycle"],
        momentum_mixing_type=MomentumMixingType.none,
    )
    m.fs.M1.outlet.pressure.fix()
    # m.fs.M1.pressure_equality_constraints[0,2].deactivate()
    # First reactor (anoxic)
    m.fs.R1 = CSTR(property_package=m.fs.props, reaction_package=m.fs.rxn_props_R1)

    # Second reactor (aerobic)
    m.fs.R2 = AerationTank(
        property_package=m.fs.props, reaction_package=m.fs.rxn_props_R2
    )

    # Mixer for R1 and R2 outlets
    m.fs.M2 = Mixer(
        property_package=m.fs.props,
        inlet_list=["R1_outlet", "R2_outlet"],
        momentum_mixing_type=MomentumMixingType.none,
    )
    m.fs.M2.outlet.pressure.fix()

    # Third reactor (anoxic)
    m.fs.R3 = CSTR(property_package=m.fs.props, reaction_package=m.fs.rxn_props_R3)
    # Fourth reactor (aerobic) - CSTR with injection
    m.fs.R4 = AerationTank(
        property_package=m.fs.props, reaction_package=m.fs.rxn_props_R4
    )
    # Fifth reactor (anoxic)
    m.fs.R5 = CSTR(property_package=m.fs.props, reaction_package=m.fs.rxn_props_R5)
    m.fs.S1 = Separator(
        property_package=m.fs.props, outlet_list=["effluent", "M1_inlet", "R2_inlet"]
    )

    m.fs.CL = Clarifier(
        property_package=m.fs.props,
        outlet_list=["underflow", "effluent"],
        split_basis=SplittingType.componentFlow,
    )

    m.fs.S2 = Separator(property_package=m.fs.props, outlet_list=["waste", "recycle"])
    m.fs.outgassing = Separator(
        property_package=m.fs.props,
        outlet_list=["effluent", "gas"],
        split_basis=SplittingType.componentFlow,
    )

    # Mixer for feed water and recycled sludge from clarifier
    m.fs.M3 = Mixer(
        property_package=m.fs.props,
        inlet_list=["feed", "recycle"],
        momentum_mixing_type=MomentumMixingType.none,
    )
    m.fs.M3.outlet.pressure.fix()

    # Product Blocks
    m.fs.Treated = Product(property_package=m.fs.props)
    m.fs.GHG = Product(property_package=m.fs.props)

    # Link units
    m.fs.feed_to_m1 = Arc(source=m.fs.feed.outlet, destination=m.fs.M1.feed)
    m.fs.m1_to_m3 = Arc(source=m.fs.M1.outlet, destination=m.fs.M3.feed)
    m.fs.m3_to_r1 = Arc(source=m.fs.M3.outlet, destination=m.fs.R1.inlet)
    m.fs.r1_to_m2 = Arc(source=m.fs.R1.outlet, destination=m.fs.M2.R1_outlet)
    m.fs.r2_to_m2 = Arc(source=m.fs.R2.outlet, destination=m.fs.M2.R2_outlet)
    m.fs.m2_to_r3 = Arc(source=m.fs.M2.outlet, destination=m.fs.R3.inlet)
    m.fs.r3_to_r4 = Arc(source=m.fs.R3.outlet, destination=m.fs.R4.inlet)
    m.fs.r4_to_r5 = Arc(source=m.fs.R4.outlet, destination=m.fs.R5.inlet)
    m.fs.r5_to_outgas = Arc(source=m.fs.R5.outlet, destination=m.fs.outgassing.inlet)
    m.fs.outgas_to_s1 = Arc(source=m.fs.outgassing.effluent, destination=m.fs.S1.inlet)
    m.fs.outgas_to_GHG = Arc(source=m.fs.outgassing.gas, destination=m.fs.GHG.inlet)
    m.fs.s1_to_m1 = Arc(source=m.fs.S1.M1_inlet, destination=m.fs.M1.recycle)
    m.fs.s1_to_r2 = Arc(source=m.fs.S1.R2_inlet, destination=m.fs.R2.inlet)
    m.fs.s1_to_CL = Arc(source=m.fs.S1.effluent, destination=m.fs.CL.inlet)
    m.fs.CL_to_effluent = Arc(source=m.fs.CL.effluent, destination=m.fs.Treated.inlet)
    m.fs.CL_to_s2 = Arc(source=m.fs.CL.underflow, destination=m.fs.S2.inlet)
    m.fs.s2_to_m3 = Arc(source=m.fs.S2.recycle, destination=m.fs.M3.recycle)

    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def set_asm1_inlet_conditions(m):
    m.fs.feed.flow_vol.fix(18446 * pyo.units.m**3 / pyo.units.day)
    m.fs.feed.temperature.fix(298.15 * pyo.units.K)
    m.fs.feed.pressure.fix(1 * pyo.units.atm)
    m.fs.feed.conc_mass_comp[0, "S_I"].fix(30 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "S_S"].fix(69.5 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "X_I"].fix(51.2 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "X_S"].fix(202.32 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "X_BH"].fix(28.17 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "X_BA"].fix(0 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "X_P"].fix(0 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "S_O"].fix(0 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "S_NO"].fix(0 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "S_NH"].fix(31.56 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "S_ND"].fix(6.95 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "X_ND"].fix(10.59 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.alkalinity.fix(7 * pyo.units.mol / pyo.units.m**3)


def set_asm3_inlet_conditions(m):
    m.fs.feed.flow_vol.fix(92230 * pyo.units.m**3 / pyo.units.day)
    m.fs.feed.temperature.fix(288.15 * pyo.units.K)
    m.fs.feed.pressure.fix(1 * pyo.units.atm)
    m.fs.feed.conc_mass_comp[0, "S_O"].fix(
        0.0333140769653528 * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "S_I"].fix(30 * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "S_S"].fix(
        1.79253833233150 * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "S_NH4"].fix(
        7.47840572528914 * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "S_N2"].fix(
        25.0222401125193 * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "S_NOX"].fix(
        4.49343937121928 * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.alkalinity.fix(4.95892616814772 * pyo.units.mol / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "X_I"].fix(
        1460.88032984731 * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "X_S"].fix(
        239.049918909639 * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "X_H"].fix(
        1624.51533042293 * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "X_STO"].fix(
        316.937373308996 * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "X_A"].fix(
        130.798830163795 * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "X_TSS"].fix(
        3044.89285508125 * pyo.units.g / pyo.units.m**3
    )


def set_operating_conditions(m, asm_model=ASMModel.asm1):
    # Feed Water Conditions
    if asm_model == ASMModel.asm1:
        set_asm1_inlet_conditions(m)
    elif asm_model == ASMModel.asm3:
        set_asm3_inlet_conditions(m)

    # Reactor sizing
    # m.fs.R1.volume.fix(1000 * pyo.units.m**3)
    # m.fs.R2.volume.fix(1000 * pyo.units.m**3)
    # m.fs.R3.volume.fix(1000 * pyo.units.m**3)
    # m.fs.R4.volume.fix(1000 * pyo.units.m**3)
    # m.fs.R5.volume.fix(1000 * pyo.units.m**3)

    # optimal size
    m.fs.R1.volume.fix(1135.6 * pyo.units.m**3)
    m.fs.R2.volume.fix(3189.8202435126686 * pyo.units.m**3)
    m.fs.R3.volume.fix(286.5426495471911 * pyo.units.m**3)
    m.fs.R4.volume.fix(865.306823334365 * pyo.units.m**3)
    m.fs.R5.volume.fix(200.7302836057747 * pyo.units.m**3)

    # Injection rates to Reactors 2 and 4
    for j in m.fs.props.component_list:
        if j != "S_O":
            # All components except S_O have no injection
            m.fs.R2.injection[:, :, j].fix(0)
            m.fs.R4.injection[:, :, j].fix(0)

    m.fs.R2.KLa.fix(10 / pyo.units.hour)
    m.fs.R4.KLa.fix(10 / pyo.units.hour)

    # Per-reactor calibrated scalar parameters
    m.fs.rxn_props_R1.K_NOX.fix(0.6521662103758377e-3)
    m.fs.rxn_props_R1.Y_STO_O2.fix(0.6157572364788948)
    m.fs.rxn_props_R1.Y_H_NOX.fix(0.6304518942319962)

    m.fs.rxn_props_R2.K_NOX.fix(0.4997646425030894e-3)
    m.fs.rxn_props_R2.Y_STO_O2.fix(0.35507602926814985)
    m.fs.rxn_props_R2.Y_H_NOX.fix(0.4719869825604356)

    m.fs.rxn_props_R3.K_NOX.fix(0.4574898084185497e-3)
    m.fs.rxn_props_R3.Y_STO_O2.fix(0.6725156878167102)
    m.fs.rxn_props_R3.Y_H_NOX.fix(0.5016855245491173)

    m.fs.rxn_props_R4.K_NOX.fix(0.4690073247163132e-3)
    m.fs.rxn_props_R4.Y_STO_O2.fix(0.5056406029311666)
    m.fs.rxn_props_R4.Y_H_NOX.fix(0.025005615332414445)

    m.fs.rxn_props_R5.K_NOX.fix(0.3562487847199742e-3)
    m.fs.rxn_props_R5.Y_STO_O2.fix(0.8405542351740866)
    m.fs.rxn_props_R5.Y_H_NOX.fix(0.02873805034642949)

    # Clarifier
    CL_R1 = 0.16522320909698857
    CL_W1 = 0.011936750818302306
    m.fs.CL.split_fraction[0, "underflow", :].fix(CL_R1 + CL_W1)

    # S2
    m.fs.S2.split_fraction[:, "recycle"].fix(CL_R1 / (CL_R1 + CL_W1))

    # S1
    s1_out_factor_1 = 0.3650004473797573
    s1_out_factor_2 = 0.38672345599067465
    # Set fraction of outflow from reactor 5 that recycles to M1 mixer
    m.fs.S1.split_fraction[:, "M1_inlet"].fix(1 - s1_out_factor_1 - s1_out_factor_2)
    # Set fraction of outflow from reactor 5 that goes to effluent
    m.fs.S1.split_fraction[:, "effluent"].fix(s1_out_factor_2)

    # outgassing
    # Fix all components to 1 (no outgassing) except S_O and S_N2
    for j in m.fs.props.component_list:
        if j not in ["S_O", "S_N2"]:
            m.fs.outgassing.split_fraction[0, "gas", j].fix(1e-10)

    # Then fix the outgassing components separately
    m.fs.outgassing.split_fraction[0, "effluent", "S_O"].fix(0.1)
    m.fs.outgassing.split_fraction[0, "effluent", "S_N2"].fix(0.1)

    # Touch variable
    m.fs.Treated.conc_mass_comp

    # Check degrees of freedom
    print("DOF = ", degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0


def scale_flowsheet(m):
    # Apply scaling based on actual variable magnitudes observed from diagnostics
    # Condition number was 6e28 -- key issues were reaction_rate, flow_vol, alkalinity
    for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
        if "flow_vol" in var.name:
            if "gas_state" in var.name or "GHG" in var.name:
                # outgassing gas stream ~ 2e-9 m3/s
                iscale.set_scaling_factor(var, 1e8)
            else:
                # internal flow_vol ~ 4e-5 to 2e-4 m3/s
                iscale.set_scaling_factor(var, 1e2)
        if "temperature" in var.name:
            iscale.set_scaling_factor(var, 1e-1)
        if "pressure" in var.name:
            iscale.set_scaling_factor(var, 1e-5)
        if "conc_mass_comp" in var.name:
            if "gas_state" in var.name or "GHG" in var.name:
                # gas concentrations blow up due to tiny flow
                iscale.set_scaling_factor(var, 1e-9)
            # elif any(s in var.name for s in ["S_O", "S_N2", "X_H", "X_A", "X_STO",
            #                                   "S_NH4", "S_NOX", "S_I"]):
            #     # low-conc species ~ 1e-5 to 1e-6 kg/m3
            #     iscale.set_scaling_factor(var, 1e5)
            else:
                # bulk species (X_S, X_I, X_TSS, S_S) ~ 1e-4 to 2e-4 kg/m3
                iscale.set_scaling_factor(var, 1e2)
        if "alkalinity" in var.name:
            # alkalinity ~ 6e-6 mol/m3 internally (IDAES units)
            iscale.set_scaling_factor(var, 1e5)
        if "rate_reaction_extent" in var.name:
            # extents ~ 1e-10 to 1e-7
            iscale.set_scaling_factor(var, 1e8)
        if "rate_reaction_generation" in var.name:
            # generation ~ 1e-9 to 1e-6
            iscale.set_scaling_factor(var, 1e7)
        if "reaction_rate" in var.name:
            # reaction_rate ~ 1e-10 (biggest scaling problem: was 1e10 Jacobian norm)
            iscale.set_scaling_factor(var, 1e10)
        if "hydraulic_retention_time" in var.name:
            # HRT ~ 1e3 to 4e4 s
            iscale.set_scaling_factor(var, 1e-4)
        if "split_fraction" in var.name:
            # split fractions are O(1)
            iscale.set_scaling_factor(var, 1.0)
        if "electricity_consumption" in var.name:
            iscale.set_scaling_factor(var, 1e-3)
        if "surface_area" in var.name:
            iscale.set_scaling_factor(var, 1e-2)
        if "mass_transfer_term" in var.name:
            # mass transfer ~ 3e-6 to 9e-6
            iscale.set_scaling_factor(var, 1e6)
        if "volume" in var.name and "control_volume" not in var.name:
            # reactor volumes ~ 200-3200 m3
            iscale.set_scaling_factor(var, 1e-3)
    iscale.calculate_scaling_factors(m.fs)


def init_and_propagate(blk, arc=None, source=None, destination=None):
    blk.initialize()
    if arc is not None:
        propagate_state(arc)
    elif source is not None:
        propagate_state(source=source, destination=destination)


def initialize_flowsheet(m):
    # Initialize flowsheet
    # interval_initializer(m)
    m.fs.feed.initialize()
    propagate_state(m.fs.feed_to_m1)
    propagate_state(source=m.fs.feed.outlet, destination=m.fs.M1.recycle)

    m.fs.M1.initialize()
    propagate_state(m.fs.m1_to_m3)
    propagate_state(source=m.fs.M1.outlet, destination=m.fs.M3.recycle)

    m.fs.M3.initialize()
    propagate_state(m.fs.m3_to_r1)

    m.fs.R1.initialize()
    propagate_state(m.fs.r1_to_m2)
    propagate_state(source=m.fs.R1.outlet, destination=m.fs.M2.R2_outlet)

    m.fs.M2.initialize()
    propagate_state(m.fs.m2_to_r3)

    m.fs.R3.initialize()
    propagate_state(m.fs.r3_to_r4)

    m.fs.R4.initialize()
    propagate_state(m.fs.r4_to_r5)

    m.fs.R5.initialize()
    propagate_state(m.fs.r5_to_outgas)

    m.fs.outgassing.initialize()
    propagate_state(m.fs.outgas_to_s1)

    m.fs.S1.initialize()
    propagate_state(m.fs.s1_to_CL)
    propagate_state(m.fs.s1_to_m1)
    propagate_state(m.fs.s1_to_r2)

    # R2
    m.fs.R2.initialize()
    propagate_state(m.fs.r2_to_m2)

    # Clarifier
    m.fs.CL.initialize()
    propagate_state(m.fs.CL_to_s2)

    m.fs.S2.initialize()
    propagate_state(m.fs.s2_to_m3)

    # Reinitialization
    # reinitialize M1 after s1_to_m1
    m.fs.M1.initialize()
    propagate_state(m.fs.m1_to_m3)

    # reinitialize M3 after s2_to_m3
    m.fs.M3.initialize()
    propagate_state(m.fs.m3_to_r1)

    m.fs.R1.initialize()
    propagate_state(m.fs.r1_to_m2)

    # reinitialize M2 after r2_to_m2
    m.fs.M2.initialize()
    propagate_state(m.fs.m2_to_r3)

    m.fs.R3.initialize()
    propagate_state(m.fs.r3_to_r4)

    m.fs.R4.initialize()
    propagate_state(m.fs.r4_to_r5)

    m.fs.R5.initialize()
    propagate_state(m.fs.r5_to_outgas)

    m.fs.outgassing.initialize()
    propagate_state(m.fs.outgas_to_s1)

    m.fs.S1.initialize()
    propagate_state(m.fs.s1_to_CL)
    propagate_state(m.fs.s1_to_m1)
    propagate_state(m.fs.s1_to_r2)

    m.fs.CL.initialize()
    propagate_state(m.fs.CL_to_s2)
    propagate_state(m.fs.CL_to_effluent)

    m.fs.Treated.initialize()

    interval_initializer(m)

    # Apply sequential decomposition - 1 iteration should suffice
    seq = SequentialDecomposition()
    seq.options.select_tear_method = "heuristic"
    seq.options.tear_method = "Direct"
    seq.options.iterLim = 1

    G = seq.create_graph(m)

    # # Uncomment this code to see tear set and initialization order
    # heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")
    # order = seq.calculation_order(G)
    # for o in heuristic_tear_set:
    #     print(o.name)
    # for o in order:
    #     print(o[0].name)

    def function(unit):
        unit.initialize(outlvl=idaeslog.CRITICAL)

    seq.run(m, function)


def add_costing(m):
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyo.units.USD_2020

    # Costing Blocks
    m.fs.R1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R2.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R3.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R4.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R5.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

    # # Initialize electricity consumption values
    # m.fs.R3.electricity_consumption[0].set_value(75)
    # m.fs.R4.electricity_consumption[0].set_value(70)
    # m.fs.R5.electricity_consumption[0].set_value(20)

    # process costing and add system level metrics
    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.S1.effluent.flow_vol[0])
    m.fs.costing.add_LCOW(m.fs.S1.effluent.flow_vol[0])
    m.fs.costing.add_specific_energy_consumption(m.fs.S1.effluent.flow_vol[0])

    set_scaling_factor(m.fs.costing.total_capital_cost, 1e-7)
    set_scaling_factor(m.fs.costing.aggregate_capital_cost, 1e-8)
    set_scaling_factor(m.fs.costing.aggregate_flow_electricity, 1e-1)
    set_scaling_factor(m.fs.costing.aggregate_flow_costs["electricity"], 1e-3)
    set_scaling_factor(m.fs.costing.total_operating_cost, 1e-4)

    for block in m.fs.component_objects(pyo.Block, descend_into=True):
        if isinstance(block, UnitModelBlockData) and hasattr(block, "costing"):
            set_scaling_factor(block.costing.capital_cost, 1e-7)


def display_costing(m):
    print("Levelized cost of water: %.3g $/m3" % pyo.value(m.fs.costing.LCOW))

    print(
        "Total operating cost: %.4g M$/yr"
        % pyo.value(m.fs.costing.total_operating_cost / 1e6)
    )
    print(
        "Total capital cost: %.4g M$" % pyo.value(m.fs.costing.total_capital_cost / 1e6)
    )

    print(
        "Total annualized cost: %.4g M$/yr"
        % pyo.value(m.fs.costing.total_annualized_cost / 1e6)
    )
    print("capital cost R1: %.4g M$" % pyo.value(m.fs.R1.costing.capital_cost / 1e6))
    print("capital cost R2: %.4g M$" % pyo.value(m.fs.R2.costing.capital_cost / 1e6))
    print("capital cost R3: %.4g M$" % pyo.value(m.fs.R3.costing.capital_cost / 1e6))
    print("capital cost R4: %.4g M$" % pyo.value(m.fs.R4.costing.capital_cost / 1e6))
    print("capital cost R5: %.4g M$" % pyo.value(m.fs.R5.costing.capital_cost / 1e6))


def solve_flowsheet(m):
    # Solve overall flowsheet to close recycle loop
    solver = get_solver()
    results = solver.solve(m, tee=True)
    check_solve(results, checkpoint="closing recycle", logger=_log, fail_flag=True)

    return results


def solve_flowsheet_phase1(m):
    """Phase 1 warm-start solve with relaxed tolerances.
    We only need a feasible biomass-rich point, not a tight solution.
    Does not crash on non-optimal exit — the near-feasible point is still
    a good warm start for Phase 2.
    """
    solver = get_solver(
        options={
            "tol": 1e-6,
            "constr_viol_tol": 1e-6,
            "acceptable_tol": 1e-4,
            "acceptable_constr_viol_tol": 1e-4,
            "max_iter": 1500,
        }
    )
    results = solver.solve(m, tee=True)
    # Log but do not crash — a near-feasible point is still useful
    check_solve(results, checkpoint="Phase 1 warm start", logger=_log, fail_flag=False)
    return results


def reset_asm3_inlet_conditions(m, ini_dict):

    for k in ini_dict.keys():
        if k in m.fs.props.solute_set:
            m.fs.feed.conc_mass_comp[0, k].fix(
                ini_dict[k] * pyo.units.g / pyo.units.m**3
            )
        elif k == "alkalinity":
            m.fs.feed.alkalinity[0].fix(ini_dict[k] * pyo.units.mol / pyo.units.m**3)
        elif k == "temperature":
            m.fs.feed.temperature[0].fix(ini_dict[k] + 273.15)

        elif k == "flow_vol":
            m.fs.feed.flow_vol[0].fix(ini_dict[k] * pyo.units.m**3 / pyo.units.day)
        else:
            raise


def set_validation_inlet_conditions(m):
    # Influent conditions from the reference simulation, following Julia sequence:
    # port.S_O   ~ comp[1]
    # port.S_I   ~ frac_SI*comp[2]
    # port.S_S   ~ frac_SS*comp[2]
    # port.X_I   ~ frac_XI*comp[2]
    # port.X_S   ~ frac_XS*comp[2]
    # port.X_STO ~ frac_XSTO*comp[2]
    # port.S_NH  ~ comp[3]
    # port.S_N2  ~ comp[4]
    # port.S_NO  ~ comp[5]
    # port.S_ALK ~ comp[6]
    # port.X_H   ~ comp[7]
    # port.X_A   ~ comp[8]
    # port.X_TS  ~ comp[9]
    comp = [1e-9, 416.5, 21.0, 1e-9, 0.25, 2.3, 1e-9, 1e-9, 166.0]
    frac_SI = 0.017517364307482627
    frac_SS = 0.27354634374988246
    frac_XI = 0.2117186104741312
    frac_XS = 0.49721768146850376
    frac_STO = 1 - frac_SI - frac_SS - frac_XI - frac_XS

    m.fs.feed.flow_vol.fix(3785.42 * pyo.units.m**3 / pyo.units.day)
    m.fs.feed.conc_mass_comp[0, "S_O"].fix(comp[0] * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "S_I"].fix(
        frac_SI * comp[1] * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "S_S"].fix(
        frac_SS * comp[1] * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "X_I"].fix(
        frac_XI * comp[1] * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "X_S"].fix(
        frac_XS * comp[1] * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "X_STO"].fix(
        frac_STO * comp[1] * pyo.units.g / pyo.units.m**3
    )
    m.fs.feed.conc_mass_comp[0, "S_NH4"].fix(comp[2] * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "S_N2"].fix(comp[3] * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "S_NOX"].fix(comp[4] * pyo.units.g / pyo.units.m**3)
    m.fs.feed.alkalinity.fix(comp[5] * pyo.units.mol / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "X_H"].fix(comp[6] * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "X_A"].fix(comp[7] * pyo.units.g / pyo.units.m**3)
    m.fs.feed.conc_mass_comp[0, "X_TSS"].fix(comp[8] * pyo.units.g / pyo.units.m**3)
    m.fs.feed.temperature.fix(293.15 * pyo.units.K)
    m.fs.feed.pressure.fix(1 * pyo.units.atm)


def verify_effluent(m):
    import pandas as pd

    # COD = sum of all COD-bearing components in the effluent (kg/m3 -> mg/L via 1e3)
    cod_components = ["S_I", "S_S", "X_I", "X_S", "X_H", "X_STO", "X_A"]
    COD = (
        sum(pyo.value(m.fs.Treated.conc_mass_comp[0, k]) for k in cod_components) * 1e3
    )

    reference = {
        "COD": 8.720000568688235,
        "S_NH4": 0.40000000408648667,
        "S_NOX": 1.1400000271650608,
    }

    watertap = {
        "COD": COD,
        "S_NH4": pyo.value(m.fs.Treated.conc_mass_comp[0, "S_NH4"]) * 1e3,
        "S_NOX": pyo.value(m.fs.Treated.conc_mass_comp[0, "S_NOX"]) * 1e3,
    }

    df = pd.DataFrame({"watertap": watertap, "reference": reference})
    df["percent_difference"] = (
        (df["watertap"] - df["reference"]) / df["reference"] * 100
    )
    df["percent_difference"] = df["percent_difference"].round(2)
    print("\n=== Effluent Verification ===")
    print(df)


def print_reactor_comparison(m):
    import pandas as pd

    species = [
        ("S_O", "S_O", False),
        ("S_I", "S_I", False),
        ("S_S", "S_S", False),
        ("S_NH", "S_NH4", False),
        ("S_N2", "S_N2", False),
        ("S_NO", "S_NOX", False),
        ("S_ALK", None, True),
        ("X_I", "X_I", False),
        ("X_S", "X_S", False),
        ("X_H", "X_H", False),
        ("X_STO", "X_STO", False),
        ("X_A", "X_A", False),
        ("X_TS", "X_TSS", False),
    ]

    julia_ref = {
        "R1": {
            "S_O": 1.0504047914855494e-5,
            "S_I": 6.0905166358394265,
            "S_S": 92.18304893295047,
            "S_NH": 11.335637771683047,
            "S_N2": 0.7260490980293832,
            "S_NO": 0.039665993815954106,
            "S_ALK": 1.4895832881218996,
            "X_I": 1430.6324322233181,
            "X_S": 146.26493174563302,
            "X_H": 195.12995373117386,
            "X_STO": 36.64357511481302,
            "X_A": 31.991986333907928,
            "X_TS": 595.2356886434673,
        },
        "R2": {
            "S_O": 8.5621958332359,
            "S_I": 6.0905166358394185,
            "S_S": 0.1809661487380628,
            "S_NH": 0.06545899701951757,
            "S_N2": 0.7062638750018604,
            "S_NO": 4.707552793144472,
            "S_ALK": 0.35115003283675633,
            "X_I": 1441.5997386904708,
            "X_S": 27.886967673809004,
            "X_H": 185.102074048344,
            "X_STO": 33.34562438843271,
            "X_A": 30.756291500189107,
            "X_TS": 502.56349537574,
        },
        "R3": {
            "S_O": 0.031520824989993146,
            "S_I": 6.090516635839423,
            "S_S": 52.059299774976715,
            "S_NH": 7.132219765275167,
            "S_N2": 1.7852890787969302,
            "S_NO": 1.0708383332481712,
            "S_ALK": 1.115683977704751,
            "X_I": 1434.7245033937052,
            "X_S": 98.28635921616005,
            "X_H": 192.56863143839658,
            "X_STO": 41.162960796242594,
            "X_A": 31.603161181242456,
            "X_TS": 562.3779949161902,
        },
        "R4": {
            "S_O": 4.864264236364027,
            "S_I": 6.090516635839423,
            "S_S": 7.701030126290658,
            "S_NH": 0.6804389916873439,
            "S_N2": 5.054993357366321,
            "S_NO": 5.4677023856382885,
            "S_ALK": 0.34078077584918415,
            "X_I": 1435.3709320529904,
            "X_S": 85.07130062979549,
            "X_H": 200.09468253727366,
            "X_STO": 44.586021899231376,
            "X_A": 33.10040798104239,
            "X_TS": 563.1264216784342,
        },
        "R5": {
            "S_O": 0.41245899215780024,
            "S_I": 6.090516635839423,
            "S_S": 2.6294839328488124,
            "S_NH": 0.40000000408648667,
            "S_N2": 9.816623212445112,
            "S_NO": 1.1400000271650608,
            "S_ALK": 0.6298710166257822,
            "X_I": 1435.4911512790286,
            "X_S": 82.07535099827011,
            "X_H": 201.69930878387976,
            "X_STO": 34.802026801814144,
            "X_A": 33.15421997351044,
            "X_TS": 556.5918431549485,
        },
    }

    reactor_blocks = [
        ("R1", m.fs.R1.control_volume.properties_out[0]),
        ("R2", m.fs.R2.control_volume.properties_out[0]),
        ("R3", m.fs.R3.control_volume.properties_out[0]),
        ("R4", m.fs.R4.control_volume.properties_out[0]),
        ("R5", m.fs.R5.control_volume.properties_out[0]),
    ]

    for rx_label, props in reactor_blocks:
        rows = {}
        for disp, wt_key, is_alk in species:
            if is_alk:
                wt_val = pyo.value(props.alkalinity) * 1e3
            else:
                wt_val = pyo.value(props.conc_mass_comp[wt_key]) * 1e3
            julia_val = julia_ref[rx_label][disp]
            pct = (
                (wt_val - julia_val) / julia_val * 100
                if julia_val != 0
                else float("nan")
            )
            rows[disp] = {
                "WaterTAP (mg/L)": round(wt_val, 4),
                "Julia (mg/L)": julia_val,
                "% diff": round(pct, 2),
            }
        df_rx = pd.DataFrame(rows).T
        print(f"\n=== Reactor {rx_label} ===")
        print(df_rx.to_string())


if __name__ == "__main__":
    # This method builds and runs a steady state activated sludge
    # flowsheet.
    m = build_flowsheet(asm_model=ASMModel.asm3)
    set_operating_conditions(m, asm_model=ASMModel.asm3)

    # --- Phase 1: biomass-rich warm start to establish recycle concentrations ---
    # Without biomass in recycles, IPOPT finds the trivial zero-reaction solution.
    # We use the validation flow rate so Phase 2 needs no flow adjustment.
    ini2 = {
        "flow_vol": 3785.42,
        "temperature": 20.0,
        "alkalinity": 2.3,
        "S_O": 2.0,
        "S_I": 7.3,
        "S_S": 113.9,
        "S_NH4": 21.0,
        "S_N2": 1e-9,
        "S_NOX": 0.25,
        "X_I": 88.2,
        "X_S": 207.0,
        "X_H": 500.0,  # high biomass to break degeneracy
        "X_STO": 50.0,
        "X_A": 30.0,
        "X_TSS": 166.0,
    }
    reset_asm3_inlet_conditions(m, ini2)
    scale_flowsheet(m)
    initialize_flowsheet(m)
    # scale_flowsheet(m)
    # # Phase 1 solve: relaxed tolerances, establish biomass in recycles
    # solve_flowsheet_phase1(m)

    # --- Phase 2: switch to validation inlet, resolve tightly ---
    set_validation_inlet_conditions(m)
    # scale_flowsheet(m)

    # --- Print Jacobian condition number to diagnose scaling ---
    print("\n--- Scaling diagnostics before solve ---")
    from idaes.core.util.model_statistics import (
        large_residuals_set,
        variables_near_bounds_set,
    )
    import idaes.core.util.scaling as iscale

    jac, nlp = iscale.get_jacobian(m, scaled=True)
    print(f"Jacobian condition number (scaled): {iscale.jacobian_cond(jac=jac):.3e}")
    # Print variables with extreme scaling
    print("\nVariables with extreme Jacobian entries:")
    iscale.report_scaling_issues(m)

    print("---Structural Issues---")
    dt = DiagnosticsToolbox(m)
    dt.report_structural_issues()
    dt.display_potential_evaluation_errors()

    # Always run Jacobian diagnostics -- reveals scaling problems
    print("---Variables with extreme Jacobian entries (scaled)---")
    dt.display_variables_with_extreme_jacobians()
    print("---Constraints with extreme Jacobian entries (scaled)---")
    dt.display_constraints_with_extreme_jacobians()

    try:
        res = solve_flowsheet(m)
    except:
        print("---Numerical Issues post-solve---")
        dt.report_numerical_issues()
        dt.display_constraints_with_large_residuals()
        dt.display_variables_at_or_outside_bounds()

    # --- Stream table (same as original) ---
    stream_table = create_stream_table_dataframe(
        {
            "Feed": m.fs.feed.outlet,
            # "M1": m.fs.M1.outlet,
            "R1": m.fs.R1.outlet,
            # "M2": m.fs.M2.outlet,
            "R2": m.fs.R2.outlet,
            "R3": m.fs.R3.outlet,
            "R4": m.fs.R4.outlet,
            "R5": m.fs.R5.outlet,
            # "S1 to M1": m.fs.S1.M1_inlet,
            # "S1 to R2": m.fs.S1.R2_inlet,
            "Effluent": m.fs.Treated.inlet,
        },
        time_point=0,
    )
    print(stream_table_dataframe_to_string(stream_table))

    # --- Effluent verification ---
    verify_effluent(m)

    # --- Per-reactor comparison table vs Julia (uncomment when flowsheet issues resolved) ---
    print_reactor_comparison(m)
