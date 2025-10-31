import pyomo.environ as pyo
import numpy as np
from watertap.core.util.initialization import (
    check_solve,
    assert_degrees_of_freedom,
    interval_initializer,
)
from pyomo.environ import (
    ConcreteModel,
    Block,
    Var,
    Constraint,
    value,
    Expression,
    NonNegativeReals,
    units as pyunits,
)
from pyomo.network import Port
from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom, number_total_objectives
from watertap.flowsheets.electroNP.BSM2_electroNP_surrogate_initialization_refined import (
    build_flowsheet,
    set_operating_conditions,
    set_scaling,
    initialize_system,
    solve,
    add_costing,
    setup_optimization,
    objective_fun,
    add_effluent_violations,
)
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from scipy import interpolate


def run_optimization_vary_electricity_cost(
    electricity_cost=0.07,
    has_electroNP=True,
    has_optimization=True,
    objective=objective_fun.LCOW,
):
    m = build_flowsheet(has_electroNP=has_electroNP)
    set_operating_conditions(m)
    set_scaling(m)

    m, results = initialize_system(m)
    add_costing(m)
    m.fs.costing.electricity_cost.unfix()
    m.fs.costing.electricity_cost.fix(electricity_cost)
    m.fs.costing.initialize()
    interval_initializer(m.fs.costing)

    # if has_electroNP is True:
    #     m.fs.electroNP.cathodic_potential.unfix()
    #     m.fs.electroNP.area_volume_ratio.unfix()
    #     m.fs.electroNP.cathodic_potential.fix(-0.96)
    #     m.fs.electroNP.area_volume_ratio.fix(0.1)

    results = solve(m)

    if has_optimization:
        setup_optimization(
            m,
            objective=objective,
            has_effluent_constraints=True,
            reactor_volume_equalities=False,
        )
    results = solve(m)

    return m, results


def plot_electricity_cost_LCOW(num):
    # # 1D plot
    electricity_cost_list = np.linspace(0.05, 0.2, num)

    # P removal
    # P_removal_list = np.zeros(num)
    # P_removal_list[:] = np.nan
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan
    LCOP_list = np.zeros(num)
    LCOP_list[:] = np.nan

    LCOW_no_electroNP_list = np.zeros(num)
    LCOW_no_electroNP_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_optimization_vary_electricity_cost(
                electricity_cost=electricity_cost_list[i],
                has_electroNP=True,
                has_optimization=True,
                objective=objective_fun.LCOW,
            )
            # m2, results = run_with_electricity_cost(
            #     has_electroNP=False, CP=-1.1, electricity_cost=electricity_cost_list[i]
            # )

            # P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            LCOP_list[i] = pyo.value(m.fs.costing.LCOW_P_removal)
            # LCOW_no_electroNP_list[i] = pyo.value(m2.fs.costing.LCOW)
            # LCOP_no_electroNP_list[i] = pyo.value(m2.fs.costing.LCOW_P_removal)
        except:
            pass

    for i in range(0, num):
        try:
            m2, results = run_optimization_vary_electricity_cost(
                electricity_cost=electricity_cost_list[i],
                has_electroNP=False,
                has_optimization=True,
                objective=objective_fun.LCOW,
            )
            # m2, results = run_with_electricity_cost(
            #     has_electroNP=False, CP=-1.1, electricity_cost=electricity_cost_list[i]
            # )

            LCOW_no_electroNP_list[i] = pyo.value(m2.fs.costing.LCOW)
        except:
            pass

    LCOW_list = interp_1d(LCOW_list)
    LCOP_list = interp_1d(LCOP_list)
    LCOW_no_electroNP_list = interp_1d(LCOW_no_electroNP_list)

    LCOW_list = smooth_1d(LCOW_list)
    LCOP_list = smooth_1d(LCOP_list)
    LCOW_no_electroNP_list = smooth_1d(LCOW_no_electroNP_list)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")

    # LCOW
    ax1.plot(
        electricity_cost_list,
        LCOW_list,
        color="tab:blue",
        label="LCOW (BSM2 with electroNP)",
    )
    ax1.plot(
        electricity_cost_list,
        LCOW_no_electroNP_list,
        color="tab:blue",
        linestyle="--",
        label="LCOW (BSM2 without electroNP)",
    )
    ax1.set_xlabel("Electricity Cost ($/kWh (2018))", fontsize=12)
    # ax1.set_ylim([0.65, 0.95])
    ax1.set_ylabel("LCOW ($/m3 (2023))", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)
    ax1.legend(loc="upper center")

    # # LCOP
    # ax1a = ax1.twinx()
    # ax1a.plot(
    #     electricity_cost_list,
    #     LCOP_list,
    #     color="tab:red",
    #     label="LCOP (BSM2 with electroNP)",
    # )
    # # ax1a.set_ylim([44.95, 45.25])
    # ax1a.set_ylabel("LCOP ($/kg (2023))", fontsize=11)
    # ax1a.tick_params(axis="x", labelsize=11)
    # ax1a.tick_params(axis="y", labelsize=11)
    # ax1a.yaxis.label.set_color("tab:red")
    # ax1a.spines["right"].set_color("tab:red")
    # ax1a.tick_params(axis="y", colors="tab:red")
    # plt.locator_params(axis="y", nbins=8)
    # ax1a.legend(loc="lower center")

    plt.show(block=True)


def plot_electricity_cost_LCOP(num):
    # # 1D plot
    electricity_cost_list = np.linspace(0.05, 0.2, num)

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan
    LCOP_list = np.zeros(num)
    LCOP_list[:] = np.nan
    SEC_list = np.zeros(num)
    SEC_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_optimization_vary_electricity_cost(
                electricity_cost=electricity_cost_list[i],
                has_electroNP=True,
                has_optimization=True,
                objective=objective_fun.LCOP,
            )

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            LCOP_list[i] = pyo.value(m.fs.costing.LCOW_P_removal)
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)
            # LCOW_no_electroNP_list[i] = pyo.value(m2.fs.costing.LCOW)
            # LCOP_no_electroNP_list[i] = pyo.value(m2.fs.costing.LCOW_P_removal)
        except:
            pass

    LCOW_list = interp_1d(LCOW_list)
    LCOP_list = interp_1d(LCOP_list)
    P_removal_list = interp_1d(P_removal_list)
    SEC_list = interp_1d(SEC_list)

    LCOW_list = smooth_1d(LCOW_list)
    LCOP_list = smooth_1d(LCOP_list)
    P_removal_list = smooth_1d(P_removal_list)
    SEC_list = smooth_1d(SEC_list)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")

    # LCOP
    ax1.plot(
        electricity_cost_list,
        LCOP_list,
        color="tab:blue",
        label="LCOP",
    )
    ax1.set_xlabel("Electricity Cost ($/kWh (2018))", fontsize=12)
    # ax1.set_ylim([0.65, 0.95])
    ax1.set_ylabel("LCOP ($/m3 (2023))", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)
    ax1.legend(loc="upper center")

    # P recovery
    ax1a = ax1.twinx()
    ax1a.plot(
        electricity_cost_list,
        P_removal_list,
        color="tab:red",
        label="Phosphorus Recovery",
    )
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("Phosphorus Recovery", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:red")
    ax1a.spines["right"].set_color("tab:red")
    ax1a.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)
    ax1a.legend(loc="lower center")

    # SEC
    ax1a = ax1.twinx()
    ax1a.plot(
        electricity_cost_list,
        SEC_list,
        color="tab:red",
        label="Specific energy consumption",
    )
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("SEC (kWh/m3)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:red")
    ax1a.spines["right"].set_color("tab:red")
    ax1a.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)
    ax1a.legend(loc="lower center")

    plt.show(block=True)


def interp_1d(array):
    # Making sequences for interp
    ok = ~np.isnan(array)
    xp = ok.ravel().nonzero()[0]
    fp = array[~np.isnan(array)]
    x = np.isnan(array).ravel().nonzero()[0]

    # Replacing nan values
    array[np.isnan(array)] = np.interp(x, xp, fp)

    return array


def interp_2d(array):
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method="linear")

    return GD1


def smooth_1d(y):
    x = np.arange(len(y))

    # Rolling mean for nearby values (window=3)
    def rolling_mean(arr, window=5):
        pad = window // 2
        padded = np.pad(arr, pad, mode="edge")
        means = np.convolve(padded, np.ones(window) / window, mode="valid")
        return means

    local_mean = rolling_mean(y, window=5)

    # Detect points >20% off local mean
    mask = np.abs(y - local_mean) > 0.2 * local_mean

    # Replace outliers with NaN
    y_clean = y.copy()
    y_clean[mask] = np.nan

    # Interpolate using only nearby good points
    def interpolate_nan(x, y):
        isnan = np.isnan(y)
        return np.interp(x, x[~isnan], y[~isnan])

    y_interp = interpolate_nan(x, y_clean)

    return y_interp


if __name__ == "__main__":
    # plot_electricity_cost_LCOW(num=10)
    plot_electricity_cost_LCOP(num=20)
