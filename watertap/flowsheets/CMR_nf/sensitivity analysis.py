import pyomo.environ as pyo
from pyomo.environ import (
    assert_optimal_termination,
    ConcreteModel,
    Constraint,
    Set,
    Suffix,
    TransformationFactory,
    value,
    Var,
)

from idaes.core import FlowsheetBlock
from idaes.core.scaling import set_scaling_factor
from idaes.core.solvers import get_solver

from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
)
from watertap.unit_models.nanofiltration_ZO import NanofiltrationZO
from idaes.core import UnitModelCostingBlock
from watertap.costing.unit_models.nanofiltration import cost_nanofiltration
from watertap.costing import WaterTAPCosting
from idaes.core.util.scaling import calculate_scaling_factors
from watertap.flowsheets.CMR_nf.CMR_nf_zo import (
    Case,
    build,
    set_scaling,
    initialize_system,
    solve,
    display_performance_metrics,
    display_costing,
)

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

__author__ = "Chenyu Wang"


def CMR_nf_case(case=Case.case1, simplified_routine=False, membrane_cost=15, area=500):
    m = build(case=case, simplified_routine=simplified_routine)
    m.fs.unit.costing.membrane_cost.set_value(membrane_cost)
    m.fs.unit.area.unfix()
    m.fs.unit.area.fix(area)
    set_scaling(m)
    initialize_system(m)
    m, results = solve(m)
    # display_performance_metrics(m)
    # display_costing(m)

    return m, results


def plot_membrane_cost(num):
    membrane_cost_list = np.linspace(10, 90, num)

    capex_list = np.zeros(num)
    capex_list[:] = np.nan

    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = CMR_nf_case(
                case=Case.case1,
                simplified_routine=False,
                membrane_cost=membrane_cost_list[i],
                area=500,
            )
            capex_list[i] = pyo.value(m.fs.costing.total_capital_cost)
            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
        except:
            pass

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(
        membrane_cost_list,
        capex_list / 1000,
        color="tab:red",
        label="_Total capital cost",
    )
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Membrane cost ($/m2)", fontsize=12)
    ax1.set_ylabel("Total capital cost (k$)", fontsize=12)
    # ax1.legend(loc="lower left")
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.yaxis.label.set_color("tab:red")
    ax1.spines["left"].set_color("tab:red")
    ax1.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    ax1b = ax1.twinx()
    ax1b.plot(
        membrane_cost_list,
        LCOW_list,
        color="tab:blue",
        label="_LCOW",
    )
    ax1b.set_ylim(0, 0.08)
    ax1b.set_xlabel("Membrane cost ($/m2)", fontsize=12)
    ax1b.set_ylabel("LCOW ($/m3)", fontsize=12)
    # ax1b.legend(loc="lower left")
    ax1b.tick_params(axis="x", labelsize=12)
    ax1b.tick_params(axis="y", labelsize=12)
    ax1b.yaxis.label.set_color("tab:blue")
    ax1b.spines["right"].set_color("tab:blue")
    ax1b.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    fig1.tight_layout()
    plt.show(block=True)


def plot_area(num):
    area_list = np.linspace(100, 900, num)

    flux_vol_solvent_list = np.zeros(num)
    flux_vol_solvent_list[:] = np.nan

    capex_list = np.zeros(num)
    capex_list[:] = np.nan

    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = CMR_nf_case(
                case=Case.case1,
                simplified_routine=False,
                membrane_cost=15,
                area=area_list[i],
            )
            flux_vol_solvent_list[i] = pyo.value(m.fs.unit.flux_vol_solvent[0, "H2O"])
            capex_list[i] = pyo.value(m.fs.costing.total_capital_cost)
            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
        except:
            pass

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(
        area_list,
        capex_list / 1000,
        color="tab:red",
        label="_Total capital cost",
    )
    ax1.set_ylim(0, 30)
    ax1.set_xlabel("Area (m2)", fontsize=12)
    ax1.set_ylabel("Total capital cost (k$)", fontsize=12)
    # ax1.legend(loc="lower left")
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.yaxis.label.set_color("tab:red")
    ax1.spines["left"].set_color("tab:red")
    ax1.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    ax1b = ax1.twinx()
    ax1b.plot(
        area_list,
        flux_vol_solvent_list,
        color="tab:blue",
        label="_Solvent volumetric flux",
    )
    # ax1b.set_ylim(0, 0.025)
    ax1b.set_xlabel("Area (m2)", fontsize=12)
    ax1b.set_ylabel("Solvent volumetric flux (m/s)", fontsize=12)
    # ax1b.legend(loc="lower left")
    ax1b.tick_params(axis="x", labelsize=12)
    ax1b.tick_params(axis="y", labelsize=12)
    ax1b.yaxis.label.set_color("tab:blue")
    ax1b.spines["right"].set_color("tab:blue")
    ax1b.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    fig1.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    # m, results = CMR_nf_case(case=Case.case1, simplified_routine=False, membrane_cost=15, area=500)
    # plot_membrane_cost(num=15)
    plot_area(num=15)
