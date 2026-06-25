#################################################################################
# WaterTAP Copyright (c) 2020-2026, The Regents of the University of California,
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

import pandas as pd
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
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
import os

# Set up logger
_log = idaeslog.getLogger(__name__)

_SURROGATE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "unit_models", "aerator_surrogate"
)
_POWER_SURROGATE_PATH = os.path.join(_SURROGATE_DIR, "aerator_power_surrogate.json")
_OXYGEN_SURROGATE_PATH = os.path.join(_SURROGATE_DIR, "aerator_oxygen_surrogate.json")

_HP_TO_KW = 0.74569987  # 1 HP = 0.74569987 kW
_LB_HR_TO_KG_S = 1.0 / 7936.64  # 1 lb/hr = 1/7936.64 kg/s


class ASMModel(auto):
    asm1 = auto()
    asm3 = auto()


def apply_aerator_surrogate(
    m,
    use_surrogate=True,
    R2_immersion_depth=0.0,
    R2_capacity=80.0,
    R4_immersion_depth=0.0,
    R4_capacity=80.0,
    power_surrogate_path=_POWER_SURROGATE_PATH,
    oxygen_surrogate_path=_OXYGEN_SURROGATE_PATH,
):
    """
    Replace KLa-based aeration constraints on R2 and R4 with polynomial
    surrogates for power draw and oxygen transfer (WesTech Landy-7 data).
    Call after set_operating_conditions().
    """
    if not use_surrogate:
        return

    power_surr = PysmoSurrogate.load_from_file(power_surrogate_path)
    oxygen_surr = PysmoSurrogate.load_from_file(oxygen_surrogate_path)

    if "S_O" in m.fs.props.component_list:
        oxygen_str = "S_O"
    elif "S_O2" in m.fs.props.component_list:
        oxygen_str = "S_O2"
    else:
        raise ValueError(
            "Oxygen component (S_O or S_O2) not found in property package."
        )

    for reactor, name, depth_val, capacity_val in [
        (m.fs.R2, "R2", R2_immersion_depth, R2_capacity),
        (m.fs.R4, "R4", R4_immersion_depth, R4_capacity),
    ]:
        # Deactivate KLa-based constraints
        reactor.eq_mass_transfer.deactivate()
        reactor.eq_electricity_consumption.deactivate()

        # Surrogate input vars (dimensionless)
        # immersion_depth: valid range [-5.12, 5.94] in
        # capacity: valid range [50, 100] % of rated speed (30-60 Hz)
        reactor.immersion_depth = pyo.Var(
            initialize=depth_val,
            bounds=(-5.12, 5.94),
            units=pyo.units.dimensionless,
        )
        reactor.capacity = pyo.Var(
            initialize=capacity_val,
            bounds=(50.0, 100.0),
            units=pyo.units.dimensionless,
        )
        reactor.immersion_depth.fix(depth_val)
        reactor.capacity.fix(capacity_val)

        # Surrogate output vars (dimensionless)
        # Power surrogate output in HP
        reactor.power_surrogate = pyo.Var(
            initialize=50.0,
            bounds=(0, None),
            units=pyo.units.dimensionless,
        )
        # Oxygen surrogate output in lb/hr
        reactor.oxygen_surrogate = pyo.Var(
            initialize=100.0,
            bounds=(0, None),
            units=pyo.units.dimensionless,
        )

        # Wire surrogates via SurrogateBlock
        reactor.surrogate_power_block = SurrogateBlock(concrete=True)
        reactor.surrogate_power_block.build_model(
            power_surr,
            input_vars=[reactor.immersion_depth, reactor.capacity],
            output_vars=[reactor.power_surrogate],
        )

        reactor.surrogate_oxygen_block = SurrogateBlock(concrete=True)
        reactor.surrogate_oxygen_block.build_model(
            oxygen_surr,
            input_vars=[reactor.immersion_depth, reactor.capacity],
            output_vars=[reactor.oxygen_surrogate],
        )

        # Unit conversion constraints
        # Power: HP to kW
        @reactor.Constraint(m.fs.config.time)
        def eq_surrogate_power(b, t):
            return b.electricity_consumption[t] == b.power_surrogate * _HP_TO_KW

        # Oxygen: lb/hr to kg/s
        @reactor.Constraint(m.fs.config.time)
        def eq_surrogate_oxygen(b, t):
            return (
                b.control_volume.mass_transfer_term[t, "Liq", oxygen_str]
                == b.oxygen_surrogate * _LB_HR_TO_KG_S
            )

    print(f"Aerator surrogate applied. DOF = {degrees_of_freedom(m)}")


def add_costing(m):
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyo.units.USD_2020

    # Costing Blocks
    m.fs.R1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R2.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R3.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R4.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R5.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

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


def build_flowsheet(asm_model=ASMModel.asm1):
    m = pyo.ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)
    if asm_model == ASMModel.asm1:
        m.fs.props = ASM1ParameterBlock()
        m.fs.rxn_props = ASM1ReactionParameterBlock(property_package=m.fs.props)
    elif asm_model == ASMModel.asm3:
        m.fs.props = ASM3ParameterBlock()
        m.fs.rxn_props = ASM3ReactionParameterBlock(property_package=m.fs.props)

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
    m.fs.R1 = CSTR(property_package=m.fs.props, reaction_package=m.fs.rxn_props)

    # Second reactor (aerobic)
    m.fs.R2 = AerationTank(property_package=m.fs.props, reaction_package=m.fs.rxn_props)

    # Mixer for R1 and R2 outlets
    m.fs.M2 = Mixer(
        property_package=m.fs.props,
        inlet_list=["R1_outlet", "R2_outlet"],
        momentum_mixing_type=MomentumMixingType.none,
    )
    m.fs.M2.outlet.pressure.fix()

    # Third reactor (anoxic)
    m.fs.R3 = CSTR(property_package=m.fs.props, reaction_package=m.fs.rxn_props)
    # Fourth reactor (aerobic) - CSTR with injection
    m.fs.R4 = AerationTank(property_package=m.fs.props, reaction_package=m.fs.rxn_props)
    # Fifth reactor (anoxic)
    m.fs.R5 = CSTR(property_package=m.fs.props, reaction_package=m.fs.rxn_props)
    m.fs.S1 = Separator(
        property_package=m.fs.props, outlet_list=["effluent", "M1_inlet", "R2_inlet"]
    )

    # Product Blocks
    m.fs.Treated = Product(property_package=m.fs.props)

    # Link units
    m.fs.feed_to_m1 = Arc(source=m.fs.feed.outlet, destination=m.fs.M1.feed)
    m.fs.m1_to_r1 = Arc(source=m.fs.M1.outlet, destination=m.fs.R1.inlet)
    m.fs.r1_to_m2 = Arc(source=m.fs.R1.outlet, destination=m.fs.M2.R1_outlet)
    m.fs.r2_to_m2 = Arc(source=m.fs.R2.outlet, destination=m.fs.M2.R2_outlet)
    m.fs.m2_to_r3 = Arc(source=m.fs.M2.outlet, destination=m.fs.R3.inlet)
    m.fs.r3_to_r4 = Arc(source=m.fs.R3.outlet, destination=m.fs.R4.inlet)
    m.fs.r4_to_r5 = Arc(source=m.fs.R4.outlet, destination=m.fs.R5.inlet)
    m.fs.r5_to_s1 = Arc(source=m.fs.R5.outlet, destination=m.fs.S1.inlet)
    m.fs.s1_to_effluent = Arc(source=m.fs.S1.effluent, destination=m.fs.Treated.inlet)
    m.fs.s1_to_m1 = Arc(source=m.fs.S1.M1_inlet, destination=m.fs.M1.recycle)
    m.fs.s1_to_r2 = Arc(source=m.fs.S1.R2_inlet, destination=m.fs.R2.inlet)

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
    m.fs.R1.volume.fix(1000 * pyo.units.m**3)
    m.fs.R2.volume.fix(1000 * pyo.units.m**3)
    m.fs.R3.volume.fix(1000 * pyo.units.m**3)
    m.fs.R4.volume.fix(1000 * pyo.units.m**3)
    m.fs.R5.volume.fix(1000 * pyo.units.m**3)

    # Injection rates to Reactors 2 and 4
    for j in m.fs.props.component_list:
        if j != "S_O":
            # All components except S_O have no injection
            m.fs.R2.injection[:, :, j].fix(0)
            m.fs.R4.injection[:, :, j].fix(0)

    m.fs.R2.KLa.fix(10 / pyo.units.hour)
    m.fs.R4.KLa.fix(10 / pyo.units.hour)

    # Set fraction of outflow from reactor 5 that recycles to M1 mixer
    m.fs.S1.split_fraction[:, "M1_inlet"].fix(0.1)

    # Set fraction of outflow from reactor 5 that goes to effluent
    m.fs.S1.split_fraction[:, "effluent"].fix(0.4)

    # Check degrees of freedom
    print("DOF = ", degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0


def scale_flowsheet(m):
    # Apply scaling
    for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
        if "flow_vol" in var.name:
            iscale.set_scaling_factor(var, 1e1)
        if "temperature" in var.name:
            iscale.set_scaling_factor(var, 1e-1)
        if "pressure" in var.name:
            iscale.set_scaling_factor(var, 1e-4)
        if "conc_mass_comp" in var.name:
            iscale.set_scaling_factor(var, 1e2)
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
    propagate_state(m.fs.m1_to_r1)

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
    propagate_state(m.fs.r5_to_s1)

    m.fs.S1.initialize()
    propagate_state(m.fs.s1_to_effluent)
    propagate_state(m.fs.s1_to_m1)
    propagate_state(m.fs.s1_to_r2)

    m.fs.R2.initialize()
    propagate_state(m.fs.r2_to_m2)

    m.fs.M1.initialize()
    propagate_state(m.fs.m1_to_r1)

    m.fs.R1.initialize()
    propagate_state(m.fs.r1_to_m2)

    m.fs.M2.initialize()
    propagate_state(m.fs.m2_to_r3)

    m.fs.R3.initialize()
    propagate_state(m.fs.r3_to_r4)

    m.fs.R4.initialize()
    propagate_state(m.fs.r4_to_r5)

    m.fs.R5.initialize()
    propagate_state(m.fs.r5_to_s1)

    m.fs.S1.initialize()
    propagate_state(m.fs.s1_to_effluent)
    propagate_state(m.fs.s1_to_m1)
    propagate_state(m.fs.s1_to_r2)

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
        unit.initialize(outlvl=idaeslog.DEBUG)

    seq.run(m, function)


def solve_flowsheet(m):
    # Solve overall flowsheet to close recycle loop
    solver = get_solver()
    results = solver.solve(m, tee=True)
    check_solve(results, checkpoint="closing recycle", logger=_log, fail_flag=True)

    return results


def validate_results(m):
    # init1 Julia reference solution for R5
    solution_R5 = {
        "S_O": 0.09259299274040665,
        "S_I": 29.999999999999904,
        "S_S": 0.254591951173486,
        "S_NH4": 3.702987579661963,
        "S_N2": 28.7024276740098,
        "S_NOX": 5.148918477118473,
        "alkalinity": 4.642433507324409,
        "X_I": 1462.7153942780158,
        "X_S": 213.59814155397257,
        "X_H": 1630.5032543499726,
        "X_STO": 312.5222537389754,
        "X_A": 131.48604445510662,
        "X_TSS": 3030.538873042037,
        "flowrate": 230575.0,
    }

    my_solution_R5 = {
        k: pyo.value(m.fs.R5.outlet.conc_mass_comp[0, k]) * 1e3
        for k in solution_R5.keys()
        if k != "alkalinity" and k != "flowrate"
    }

    df_R5 = pd.DataFrame({"watertap": my_solution_R5, "julia": solution_R5})
    df_R5.loc["alkalinity", "watertap"] = pyo.value(m.fs.R5.outlet.alkalinity[0]) * 1e3
    df_R5.loc["flowrate", "watertap"] = pyo.value(m.fs.R5.outlet.flow_vol[0]) * 86400
    df_R5["percent_difference"] = (
        (df_R5["watertap"] - df_R5["julia"]) / df_R5["julia"] * 100
    )
    df_R5["percent_difference"] = df_R5["percent_difference"].round(1)

    print(df_R5)


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


if __name__ == "__main__":
    # This method builds and runs a steady state activated sludge
    # flowsheet.
    # m, results = build_flowsheet()
    m = build_flowsheet(asm_model=ASMModel.asm3)
    set_operating_conditions(m, asm_model=ASMModel.asm3)
    ini1 = {
        "S_O": 0.0333140769653528,
        "S_I": 29.9999999999999,
        "S_S": 1.79253833233150,
        "S_NH4": 7.47840572528914,
        "S_N2": 25.0222401125193,
        "S_NOX": 4.49343937121928,
        "alkalinity": 4.95892616814772,
        "X_I": 1460.88032984731,
        "X_S": 239.049918909639,
        "X_H": 1624.51533042293,
        "X_STO": 316.937373308996,
        "X_A": 130.798830163795,
        "X_TSS": 3044.89285508125,
        "flow_vol": 92230,
        "temperature": 15,
    }
    ini2 = {
        "S_O": 2.00000074088136,
        "S_I": 30,
        "S_S": 1.99999995166373,
        "S_NH4": 20.0000000011385,
        "S_N2": 1.30283567480963e-18,
        "S_NOX": 9.07971541001396e-10,
        "alkalinity": 5.00000000001647,
        "X_I": 100.000000001382,
        "X_S": 39.9999999624405,
        "X_H": 100.000000012367,
        "X_STO": 40.0000000397251,
        "X_A": 1.0000000001811,
        "X_TSS": 200.000000007995,
        "flow_vol": 36892,
        "temperature": 14.8581001531874,
    }

    reset_asm3_inlet_conditions(m, ini1)
    scale_flowsheet(m)

    initialize_flowsheet(m)

    scale_flowsheet(m)

    res = solve_flowsheet(m)

    # Surrogate aerator model (set use_surrogate=True to activate)
    # Reference conditions: 54 Hz (90% capacity), +1 inch submergence (WesTech recommendation)
    apply_aerator_surrogate(
        m,
        use_surrogate=False,
        R2_immersion_depth=1.0,
        R2_capacity=90.0,
        R4_immersion_depth=1.0,
        R4_capacity=90.0,
    )

    # Costing
    add_costing(m)
    m.fs.costing.initialize()
    res = solve_flowsheet(m)
    display_costing(m)

    validate_results(m)
