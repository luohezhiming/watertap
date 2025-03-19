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

import pyomo.environ as pyo
from ..util import (
    register_costing_parameter_block,
    make_capital_cost_var,
)


def build_electroNP_cost_param_block(blk):
    # Electrolyzer
    blk.sizing_cost_electrolyzer = pyo.Var(
        initialize=400,
        doc="Electrolyzer sizing cost",
        units=pyo.units.USD_2023 / pyo.units.m**2,
    )

    # Centrifuge
    blk.PEC_centrifuge = pyo.Var(
        initialize=1e5,
        doc="Centrifuge purchase equipment cost",
        units=pyo.units.USD_2013,
    )
    blk.Fd_centrifuge = pyo.Var(
        initialize=16,
        doc="Centrifuge design capacity",
        units=pyo.units.m**3 / pyo.units.hr,
    )
    blk.a_centrifuge = pyo.Var(
        initialize=0.46,
        doc="Centrifuge sizing exponent",
        units=pyo.units.dimensionless,
    )

    # Dryer
    blk.PEC_dryer = pyo.Var(
        initialize=712730,
        doc="Dryer purchase equipment cost",
        units=pyo.units.USD_2015,
    )
    blk.Fd_dryer = pyo.Var(
        initialize=2000,
        doc="Dryer design wet struvite feed",
        units=pyo.units.kg / pyo.units.hr,
    )
    blk.a_dryer = pyo.Var(
        initialize=0.6,
        doc="Dryer sizing exponent",
        units=pyo.units.dimensionless,
    )
    blk.K_dryer = pyo.Var(
        initialize=1360 / 1113,
        doc="Dryer conversion coefficient for design capacity",
        units=pyo.units.kg / pyo.units.m**3,
    )

    # Pump
    blk.PEC_pump = pyo.Var(
        initialize=27000,
        doc="Pump purchase equipment cost",
        units=pyo.units.USD_2015,
    )
    blk.Pd_pump = pyo.Var(
        initialize=59,
        doc="Pump design input power",
        units=pyo.units.kW,
    )
    blk.a_pump = pyo.Var(
        initialize=0.67,
        doc="Pump sizing exponent",
        units=pyo.units.dimensionless,
    )
    blk.K_pump = pyo.Var(
        initialize=109 / 3400,
        doc="Pump conversion coefficient for design capacity",
        units=pyo.units.kWh / pyo.units.m**3,
    )

    # MgCl2
    blk.sizing_cost_MgCl2 = pyo.Var(
        initialize=445,
        doc="MgCl2 sizing cost",
        units=pyo.units.USD_2023 / pyo.units.m**2,
    )

    costing = blk.parent_block()
    # blk.magnesium_chloride_cost = pyo.Param(
    #     mutable=True,
    #     initialize=0.0786,
    #     doc="Magnesium chloride cost",
    #     units=pyo.units.USD_2020 / pyo.units.kg,
    # )
    # costing.register_flow_type("magnesium chloride", blk.magnesium_chloride_cost)

    blk.phosphorus_recovery_value = pyo.Param(
        mutable=True,
        initialize=-0.07,
        doc="Phosphorus recovery value",
        units=pyo.units.USD_2020 / pyo.units.kg,
    )
    costing.register_flow_type("phosphorus salt product", blk.phosphorus_recovery_value)


@register_costing_parameter_block(
    build_rule=build_electroNP_cost_param_block,
    parameter_block_name="electroNP",
)
def cost_electroNP(blk, cost_electricity_flow=True, cost_phosphorus_flow=True):
    """
    ElectroNP costing method
    """
    cost_electroNP_capital(
        blk,
    )

    t0 = blk.flowsheet().time.first()
    if cost_electricity_flow:
        blk.costing_package.cost_flow(
            pyo.units.convert(
                blk.unit_model.electricity[t0],
                to_units=pyo.units.kW,
            ),
            "electricity",
        )

    # if cost_MgCl2_flow:
    #     blk.costing_package.cost_flow(
    #         pyo.units.convert(
    #             blk.unit_model.MgCl2_flowrate[t0],
    #             to_units=pyo.units.kg / pyo.units.hr,
    #         ),
    #         "magnesium chloride",
    #     )

    if cost_phosphorus_flow:
        blk.costing_package.cost_flow(
            pyo.units.convert(
                blk.unit_model.byproduct.flow_vol[t0]
                * blk.unit_model.byproduct.conc_mass_comp[t0, "S_PO4"],
                to_units=pyo.units.kg / pyo.units.hr,
            ),
            "phosphorus salt product",
        )


# def cost_electroNP_capital(blk, HRT, sizing_cost):
#     """
#     Generic function for costing an ElectroNP system.
#     """
#     make_capital_cost_var(blk)
#
#     blk.HRT = pyo.Expression(expr=HRT)
#     blk.sizing_cost = pyo.Expression(expr=sizing_cost)
#
#     flow_in = pyo.units.convert(
#         blk.unit_model.mixed_state[0].flow_vol,
#         to_units=pyo.units.m**3 / pyo.units.hr,
#     )
#
#     blk.costing_package.add_cost_factor(blk, "TIC")
#     blk.capital_cost_constraint = pyo.Constraint(
#         expr=blk.capital_cost
#         == blk.cost_factor
#         * pyo.units.convert(
#             blk.HRT * flow_in * blk.sizing_cost,
#             to_units=blk.costing_package.base_currency,
#         )
#     )


def cost_electroNP_capital(blk):
    """
    Generic function for costing an ElectroNP system.
    """
    make_capital_cost_var(blk)
    cost_blk = blk.costing_package.electroNP
    blk.costing_package.add_cost_factor(blk, "TIC")

    t0 = blk.flowsheet().time.first()
    flow_in = pyo.units.convert(
        blk.unit_model.inlet.flow_vol[t0],
        to_units=pyo.units.m**3 / pyo.units.hr,
    )

    # Electrolyzer
    electrolyzer_cost_expr = blk.cost_factor * pyo.units.convert(
        blk.unit_model.area[t0] * cost_blk.sizing_cost_electrolyzer,
        to_units=blk.costing_package.base_currency,
    )
    blk.electrolyzer_cost = pyo.Expression(expr=electrolyzer_cost_expr)

    # Centrifuge
    centrifuge_cost_expr = blk.cost_factor * pyo.units.convert(
        cost_blk.PEC_centrifuge
        * (flow_in / cost_blk.Fd_centrifuge) ** cost_blk.a_centrifuge,
        to_units=blk.costing_package.base_currency,
    )
    blk.centrifuge_cost = pyo.Expression(expr=centrifuge_cost_expr)

    # Dryer
    dryer_cost_expr = blk.cost_factor * pyo.units.convert(
        cost_blk.PEC_dryer
        * (flow_in * cost_blk.K_dryer / cost_blk.Fd_dryer) ** cost_blk.a_dryer,
        to_units=blk.costing_package.base_currency,
    )
    blk.dryer_cost = pyo.Expression(expr=dryer_cost_expr)

    # Pumps (3 in the system)
    pump_cost_expr = blk.cost_factor * pyo.units.convert(
        3
        * cost_blk.PEC_pump
        * (flow_in * cost_blk.K_pump / cost_blk.Pd_pump) ** cost_blk.a_pump,
        to_units=blk.costing_package.base_currency,
    )
    blk.pump_cost = pyo.Expression(expr=pump_cost_expr)

    # Direct capital costs
    blk.DCC = pyo.Expression(
        expr=blk.electrolyzer_cost
        + blk.centrifuge_cost
        + blk.dryer_cost
        + blk.pump_cost
    )

    # Permits & license
    blk.permits_and_licence_fee = pyo.Expression(expr=0.02 * blk.DCC)

    # Installation
    blk.installation_fee = pyo.Expression(expr=0.12 * blk.DCC)

    # Building
    blk.building_fee = pyo.Expression(expr=0.1 * blk.DCC)

    # Engineering design
    blk.engineering_design_fee = pyo.Expression(expr=0.1 * blk.DCC)

    # Construction & contractor
    blk.construction_and_contractor_fee = pyo.Expression(expr=0.25 * blk.DCC)

    # Contingency
    blk.contingency_fee = pyo.Expression(expr=0.15 * blk.DCC)

    # Indirect capital costs
    blk.IDC = pyo.Expression(
        expr=blk.permits_and_licence_fee
        + blk.installation_fee
        + blk.building_fee
        + blk.engineering_design_fee
        + blk.construction_and_contractor_fee
        + blk.contingency_fee
    )

    # MgCl2
    MgCl2_cost_expr = blk.cost_factor * pyo.units.convert(
        blk.unit_model.area[t0] * cost_blk.sizing_cost_MgCl2,
        to_units=blk.costing_package.base_currency,
    )
    blk.MgCl2_cost = pyo.Expression(expr=MgCl2_cost_expr)

    # Total capital cost
    cap_total = blk.DCC + blk.IDC + blk.MgCl2_cost

    blk.capital_cost_constraint = pyo.Constraint(expr=blk.capital_cost == cap_total)
