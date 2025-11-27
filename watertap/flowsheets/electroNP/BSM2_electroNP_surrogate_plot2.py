import pandas as pd
import pyomo.environ as pyo
import numpy as np
from sympy.abc import epsilon

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
import seaborn as sns


############################################ Auxiliary Functions #######################################################
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


def smooth_1d(arr, threshold=0.2, neighbors=1):
    """
    Detect and interpolate spikes where a point differs from the average
    of nearby neighbors by more than a given relative threshold.

    Parameters:
        arr (array-like): Input array (list or NumPy array).
        threshold (float): Relative threshold (default 0.2 = 20%).
        neighbors (int): Number of neighbors on each side to consider.

    Returns:
        np.ndarray: Array with spikes interpolated.
    """
    arr = np.asarray(arr, dtype=float)
    result = arr.copy()
    n = len(arr)
    if n < 2:
        return result

    for i in range(n):
        # Determine valid neighbor indices
        left_start = max(0, i - neighbors)
        right_end = min(n, i + neighbors + 1)

        # Exclude the current point
        neighbor_vals = np.concatenate((arr[left_start:i], arr[i + 1 : right_end]))
        if len(neighbor_vals) == 0:
            continue

        neighbor_mean = np.mean(neighbor_vals)
        diff_ratio = abs(arr[i] - neighbor_mean) / (
            abs(neighbor_mean) + 1e-8
        )  # avoid div by zero

        # If it's a spike (too different from its local neighborhood)
        if diff_ratio > threshold:
            result[i] = neighbor_mean

    return result


############################################### Optimization ###########################################################
def run_optimization_vary_max(
    COD_max=0.1,
    BOD5_max=0.01,
    TKN_max=0.007,
    TP_max=0.005,
    TSS_max=0.05,
    has_electroNP=True,
    has_optimization=True,
):
    m = build_flowsheet(has_electroNP=has_electroNP)
    set_operating_conditions(m)
    set_scaling(m)

    m, results = initialize_system(m)
    add_costing(m)
    m.fs.costing.initialize()
    interval_initializer(m.fs.costing)

    # if has_electroNP is True:
    #     m.fs.electroNP.cathodic_potential.unfix()
    #     m.fs.electroNP.area_volume_ratio.unfix()
    #     m.fs.electroNP.cathodic_potential.fix(-0.96)
    #     m.fs.electroNP.area_volume_ratio.fix(0.1)

    solve(m)

    if has_optimization:
        setup_optimization_vary_max(
            m,
            COD_max=COD_max,
            BOD5_max=BOD5_max,
            TKN_max=TKN_max,
            TP_max=TP_max,
            TSS_max=TSS_max,
            has_electroNP=has_electroNP,
            objective=objective_fun.LCOW,
        )

    results = solve(m)

    return m, results


def setup_optimization_vary_max(
    m,
    COD_max=0.1,
    BOD5_max=0.01,
    TKN_max=0.007,
    TP_max=0.005,
    TSS_max=0.05,
    has_electroNP=True,
    objective=objective_fun.LCOW,
):
    # Objective function
    if objective == objective_fun.LCOW:
        m.fs.objective = pyo.Objective(expr=m.fs.costing.LCOW)
    elif objective == objective_fun.LCOP:
        m.fs.objective = pyo.Objective(expr=m.fs.costing.LCOW_P_removal)
    else:
        raise TypeError(
            f'objective must be set to "LCOW"  or "LCOP".'
            f" objective was set to {objective}"
        )

    # Decision variables
    if has_electroNP:
        m.fs.electroNP.cathodic_potential.unfix()
        m.fs.electroNP.cathodic_potential.setlb(-1.3)
        m.fs.electroNP.cathodic_potential.setub(-0.8)

        m.fs.electroNP.area_volume_ratio.unfix()
        m.fs.electroNP.area_volume_ratio.setlb(0.065)
        m.fs.electroNP.area_volume_ratio.setub(0.145)

    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].setub(10e-3)

    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].setub(10e-3)

    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].setub(10e-3)

    # # Unfix fraction of outflow from reactor 7 that goes to recycle
    # m.fs.SP1.split_fraction[:, "underflow"].unfix()
    # # m.fs.SP1.split_fraction[:, "underflow"].setlb(0.45)
    # m.fs.SP2.split_fraction[:, "recycle"].unfix()

    add_effluent_violations(m)
    m.fs.COD_max.unfix()
    m.fs.COD_max.fix(COD_max)
    m.fs.BOD5_max.unfix()
    m.fs.BOD5_max.fix(BOD5_max)
    m.fs.TKN_max.unfix()
    m.fs.TKN_max.fix(TKN_max)
    m.fs.total_P_max.unfix()
    m.fs.total_P_max.fix(TP_max)
    m.fs.TSS_max.unfix()
    m.fs.TSS_max.fix(TSS_max)

    # m.fs.eq_total_P_max[0].deactivate()


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


def run_optimization_vary_electricity_cost_phosphorus_revenue(
    electricity_cost=0.07,
    phosphorus_revenue=0.6521,
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
    m.fs.costing.electroNP.phosphorus_recovery_value = -phosphorus_revenue
    m.fs.costing.initialize()
    interval_initializer(m.fs.costing)

    if has_electroNP is True:
        m.fs.electroNP.cathodic_potential.unfix()
        m.fs.electroNP.area_volume_ratio.unfix()
        m.fs.electroNP.cathodic_potential.fix(-0.96)
        m.fs.electroNP.area_volume_ratio.fix(0.1)

    # results = solve(m)

    if has_optimization:
        setup_optimization(
            m,
            objective=objective,
            has_effluent_constraints=True,
            reactor_volume_equalities=False,
        )
    results = solve(m)

    return m, results


def run_optimization_vary_epsilon(
    epsilon=1,
    COD_max=0.1,
    BOD5_max=0.01,
    TKN_max=0.007,
    TP_max=0.005,
    TSS_max=0.05,
    has_electroNP=True,
    has_optimization=True,
):
    m = build_flowsheet(has_electroNP=has_electroNP)
    set_operating_conditions(m)
    set_scaling(m)

    m, results = initialize_system(m)
    add_costing(m)
    m.fs.costing.initialize()
    interval_initializer(m.fs.costing)

    # if has_electroNP is True:
    #     m.fs.electroNP.cathodic_potential.unfix()
    #     m.fs.electroNP.area_volume_ratio.unfix()
    #     m.fs.electroNP.cathodic_potential.fix(-0.96)
    #     m.fs.electroNP.area_volume_ratio.fix(0.1)

    solve(m)

    if has_optimization:
        setup_optimization_vary_max(
            m,
            COD_max=COD_max,
            BOD5_max=BOD5_max,
            TKN_max=TKN_max,
            TP_max=TP_max,
            TSS_max=TSS_max,
            has_electroNP=has_electroNP,
            objective=objective_fun.LCOW,
        )

    m.fs.epsilon = pyo.Var(initialize=epsilon, units=pyo.units.kg / pyo.units.m**3)
    m.fs.epsilon.fix()

    @m.fs.Constraint(m.fs.time)
    def eq_LCOP_max(self, t):
        return m.fs.costing.LCOW_P_removal <= m.fs.epsilon

    results = solve(m)

    return m, results


################################################### plot ###############################################################
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
        label="Case 1",
    )
    ax1.plot(
        electricity_cost_list,
        LCOW_no_electroNP_list,
        color="tab:blue",
        linestyle="--",
        label="Case*",
    )
    ax1.set_xlabel("Electricity Cost ($/kWh (2018))", fontsize=12)
    ax1.set_xlim([0.05, 0.2])
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
    ax1.set_xlim([0.05, 0.2])
    ax1.set_ylabel("LCOP ($/m3 (2023))", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)
    ax1.legend(loc="upper center")

    # # P recovery
    # ax1a = ax1.twinx()
    # ax1a.plot(
    #     electricity_cost_list,
    #     P_removal_list,
    #     color="tab:red",
    #     label="Phosphorus Recovery",
    # )
    # # ax1a.set_ylim([44.95, 45.25])
    # ax1a.set_ylabel("Phosphorus Recovery", fontsize=11)
    # ax1a.tick_params(axis="x", labelsize=11)
    # ax1a.tick_params(axis="y", labelsize=11)
    # ax1a.yaxis.label.set_color("tab:red")
    # ax1a.spines["right"].set_color("tab:red")
    # ax1a.tick_params(axis="y", colors="tab:red")
    # plt.locator_params(axis="y", nbins=8)
    # ax1a.legend(loc="lower center")

    # # SEC
    # ax1a = ax1.twinx()
    # ax1a.plot(
    #     electricity_cost_list,
    #     SEC_list,
    #     color="tab:red",
    #     label="Specific energy consumption",
    # )
    # # ax1a.set_ylim([44.95, 45.25])
    # ax1a.set_ylabel("SEC (kWh/m3)", fontsize=11)
    # ax1a.tick_params(axis="x", labelsize=11)
    # ax1a.tick_params(axis="y", labelsize=11)
    # ax1a.yaxis.label.set_color("tab:red")
    # ax1a.spines["right"].set_color("tab:red")
    # ax1a.tick_params(axis="y", colors="tab:red")
    # plt.locator_params(axis="y", nbins=8)
    # ax1a.legend(loc="lower center")

    plt.show(block=True)


def heatmap_plot_minimize_LCOW(num):
    # 2D plot
    electricity_cost_list = np.linspace(0.05, 0.2, num)
    phosphours_revenue_list = np.linspace(0.6, 1.0, num)

    LCOW_matrix = np.zeros((num, num))
    LCOW_matrix[:] = np.nan
    LCOP_matrix = np.zeros((num, num))
    LCOP_matrix[:] = np.nan
    SEC_matrix = np.zeros((num, num))
    SEC_matrix[:] = np.nan

    for i in range(0, num):
        for j in range(0, num):
            try:
                # # simulation
                m, results = run_optimization_vary_electricity_cost_phosphorus_revenue(
                    electricity_cost=electricity_cost_list[i],
                    phosphorus_revenue=phosphours_revenue_list[j],
                    has_electroNP=True,
                    has_optimization=True,
                    objective=objective_fun.LCOW,
                )

                LCOW_matrix[j, i] = pyo.value(m.fs.costing.LCOW)
                LCOP_matrix[j, i] = pyo.value(m.fs.costing.LCOW_P_removal)
                # SEC_matrix[j, i] = pyo.value(m.fs.costing.specific_energy_consumption)
                # aeration_matrix[j, i] = pyo.value(m.fs.costing.aeration_energy)
            except:
                pass

    LCOW_matrix = interp_2d(LCOW_matrix)
    LCOP_matrix = interp_2d(LCOP_matrix)
    # SEC_matrix = interp_2d(SEC_matrix)
    # aeration_matrix = interp_2d(aeration_matrix)

    electricity_cost_list = [f"{v:.3g}" for v in electricity_cost_list]
    phosphours_revenue_list = [f"{v:.3g}" for v in phosphours_revenue_list]

    # Figure 1 - LCOW (minimize LCOW)
    fig8, ax8 = plt.subplots(figsize=(7, 5))
    # CF = ax8.contourf(electricity_cost_list, phosphours_revenue_list, LCOW_matrix, cmap="GnBu")
    # ax8.set_xlabel("Electricity Cost ($/kWh (2018))", fontsize=12)
    # ax8.set_ylabel("Phosphorus Product Price ($/m3 (2015))", fontsize=12)
    # cbar = fig8.colorbar(CF)
    # cbar.ax.set_ylabel("LCOW ($/m3 (2023))", fontsize=12)

    # HM = ax8.imshow(LCOW_matrix, extent=[electricity_cost_list.min(), electricity_cost_list.max(), phosphours_revenue_list.min(), phosphours_revenue_list.max()],
    #            origin='lower', cmap='viridis', aspect='auto')
    # fig8.colorbar(HM, label='LCOW ($/m3 (2023))')
    # ax8.set_title('LCOW ($/m3 (2023))')
    # ax8.set_xlabel('Electricity Cost ($/kWh (2018))')
    # ax8.set_ylabel('Phosphorus Product Price ($/m3 (2015))')

    data_LCOW_matrix = pd.DataFrame(
        LCOW_matrix, index=phosphours_revenue_list, columns=electricity_cost_list
    )
    sns.heatmap(data_LCOW_matrix, annot=True, fmt=".2f", cmap="viridis", ax=ax8)
    ax8.set_title("LCOW ($/m3 (2023))")
    ax8.set_xlabel("Electricity Cost ($/kWh (2018))")
    ax8.set_ylabel("Phosphorus Product Price ($/m3 (2015))")

    # Figure 2 - LCOP (minimize LCOW)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    data_LCOP_matrix = pd.DataFrame(
        LCOP_matrix, index=phosphours_revenue_list, columns=electricity_cost_list
    )
    sns.heatmap(data_LCOP_matrix, annot=True, fmt=".2f", cmap="viridis", ax=ax2)
    ax2.set_title("LCOP ($/kg (2023))")
    ax2.set_xlabel("Electricity Cost ($/kWh (2018))")
    ax2.set_ylabel("Phosphorus Product Price ($/m3 (2015))")

    plt.show(block=True)


def heatmap_plot_minimize_LCOP(num):
    # 2D plot
    # Minimize LCOP
    electricity_cost_list = np.linspace(0.05, 0.2, num)
    phosphours_revenue_list = np.linspace(0.6, 1.0, num)

    LCOW_matrix = np.zeros((num, num))
    LCOW_matrix[:] = np.nan
    LCOP_matrix = np.zeros((num, num))
    LCOP_matrix[:] = np.nan
    SEC_matrix = np.zeros((num, num))
    SEC_matrix[:] = np.nan

    for i in range(0, num):
        for j in range(0, num):
            try:
                # # simulation
                m, results = run_optimization_vary_electricity_cost_phosphorus_revenue(
                    electricity_cost=electricity_cost_list[i],
                    phosphorus_revenue=phosphours_revenue_list[j],
                    has_electroNP=True,
                    has_optimization=True,
                    objective=objective_fun.LCOP,
                )

                LCOW_matrix[j, i] = pyo.value(m.fs.costing.LCOW)
                LCOP_matrix[j, i] = pyo.value(m.fs.costing.LCOW_P_removal)
                # SEC_matrix[j, i] = pyo.value(m.fs.costing.specific_energy_consumption)
                # aeration_matrix[j, i] = pyo.value(m.fs.costing.aeration_energy)
            except:
                pass

    LCOW_matrix = interp_2d(LCOW_matrix)
    LCOP_matrix = interp_2d(LCOP_matrix)
    # SEC_matrix = interp_2d(SEC_matrix)
    # aeration_matrix = interp_2d(aeration_matrix)

    electricity_cost_list = [f"{v:.3g}" for v in electricity_cost_list]
    phosphours_revenue_list = [f"{v:.3g}" for v in phosphours_revenue_list]

    # Figure 3 - LCOW (minimize LCOP)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    data_LCOW_matrix = pd.DataFrame(
        LCOW_matrix, index=phosphours_revenue_list, columns=electricity_cost_list
    )
    sns.heatmap(data_LCOW_matrix, annot=True, fmt=".2f", cmap="viridis", ax=ax3)
    ax3.set_title("LCOW ($/m3 (2023))")
    ax3.set_xlabel("Electricity Cost ($/kWh (2018))")
    ax3.set_ylabel("Phosphorus Product Price ($/m3 (2015))")

    # Figure 4 - LCOP (minimize LCOP)
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    data_LCOP_matrix = pd.DataFrame(
        LCOP_matrix, index=phosphours_revenue_list, columns=electricity_cost_list
    )
    sns.heatmap(data_LCOP_matrix, annot=True, fmt=".2f", cmap="viridis", ax=ax4)
    ax4.set_title("LCOP ($/kg (2023))")
    ax4.set_xlabel("Electricity Cost ($/kWh (2018))")
    ax4.set_ylabel("Phosphorus Product Price ($/m3 (2015))")

    plt.show(block=True)


def plot_COD_max(num):
    # 1D plot
    COD_max_list = np.linspace(0.0955, 0.0978, num)

    # electroNP flowsheet
    # Cathodic Potential
    CP_list = np.zeros(num)
    CP_list[:] = np.nan

    rAV_list = np.zeros(num)
    rAV_list[:] = np.nan

    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

    LCOP_list = np.zeros(num)
    LCOP_list[:] = np.nan

    # SEC
    SEC_list = np.zeros(num)
    SEC_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_optimization_vary_max(
                COD_max=COD_max_list[i],
                BOD5_max=0.01,
                TKN_max=0.007,
                TP_max=0.005,
                TSS_max=0.05,
                has_electroNP=True,
                has_optimization=True,
            )

            CP_list[i] = pyo.value(m.fs.electroNP.cathodic_potential)
            rAV_list[i] = pyo.value(m.fs.electroNP.area_volume_ratio)

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            LCOP_list[i] = pyo.value(m.fs.costing.LCOW_P_removal)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)

        except:
            pass

    CP_list = interp_1d(CP_list)
    rAV_list = interp_1d(rAV_list)
    LCOW_list = interp_1d(LCOW_list)
    LCOP_list = interp_1d(LCOP_list)
    SEC_list = interp_1d(SEC_list)

    # CP_list = smooth_1d(CP_list, threshold=0.2, neighbors=1)
    # rAV_list = smooth_1d(rAV_list, threshold=0.2, neighbors=1)
    # LCOW_list = smooth_1d(LCOW_list, threshold=0.2, neighbors=1)
    # LCOP_list = smooth_1d(LCOP_list, threshold=0.2, neighbors=1)
    # SEC_list = smooth_1d(SEC_list, threshold=0.2, neighbors=1)

    COD_max_list = 1000 * COD_max_list

    # Figure a - cathodic potential
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    axa.plot(COD_max_list, CP_list, color="k", label="Cathodic Potential")
    # axa.set_xlim([95.62, 97.8])
    axa.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    axa.set_ylabel("Cathodic Potential (V)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    # plt.locator_params(axis="y", nbins=8)

    # Figure b - area volume ratio
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(COD_max_list, rAV_list, color="k", label="Area-Volume Ratio")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axb.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    axb.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)
    # axb.legend(loc="upper right", bbox_to_anchor=(1, 1))
    # plt.locator_params(axis="y", nbins=8)

    # Figure C - LCOW
    figc, axc = plt.subplots(figsize=(9, 5), layout="constrained")
    axc.plot(COD_max_list, LCOW_list, color="k", label="LCOW")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axc.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    axc.set_ylabel("LCOW ($/m3 (2023))", fontsize=11)
    axc.tick_params(axis="x", labelsize=11)
    axc.tick_params(axis="y", labelsize=11)
    # axb.legend(loc="upper right", bbox_to_anchor=(1, 1))
    # plt.locator_params(axis="y", nbins=8)

    # Figure C - LCOP
    figd, axd = plt.subplots(figsize=(9, 5), layout="constrained")
    axd.plot(COD_max_list, LCOP_list, color="k", label="LCOP")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axd.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    axd.set_ylabel("LCOP ($/m3 (2023))", fontsize=11)
    axd.tick_params(axis="x", labelsize=11)
    axd.tick_params(axis="y", labelsize=11)
    # axb.legend(loc="upper right", bbox_to_anchor=(1, 1))
    # plt.locator_params(axis="y", nbins=8)

    plt.show(block=True)


def plot_BOD5_max(num):
    # 1D plot
    BOD5_max_list = np.linspace(0.0058, 0.007, num)

    # electroNP flowsheet
    # Cathodic Potential
    CP_list = np.zeros(num)
    CP_list[:] = np.nan

    rAV_list = np.zeros(num)
    rAV_list[:] = np.nan

    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

    LCOP_list = np.zeros(num)
    LCOP_list[:] = np.nan

    # SEC
    SEC_list = np.zeros(num)
    SEC_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_optimization_vary_max(
                COD_max=0.1,
                BOD5_max=BOD5_max_list[i],
                TKN_max=0.007,
                TP_max=0.005,
                TSS_max=0.05,
                has_electroNP=True,
                has_optimization=True,
            )

            CP_list[i] = pyo.value(m.fs.electroNP.cathodic_potential)
            rAV_list[i] = pyo.value(m.fs.electroNP.area_volume_ratio)

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            LCOP_list[i] = pyo.value(m.fs.costing.LCOW_P_removal)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)

        except:
            pass

    CP_list = interp_1d(CP_list)
    rAV_list = interp_1d(rAV_list)
    LCOW_list = interp_1d(LCOW_list)
    LCOP_list = interp_1d(LCOP_list)
    SEC_list = interp_1d(SEC_list)

    # CP_list = smooth_1d(CP_list, threshold=0.2, neighbors=1)
    # rAV_list = smooth_1d(rAV_list, threshold=0.2, neighbors=1)
    # LCOW_list = smooth_1d(LCOW_list, threshold=0.2, neighbors=1)
    # LCOP_list = smooth_1d(LCOP_list, threshold=0.2, neighbors=1)
    # SEC_list = smooth_1d(SEC_list, threshold=0.2, neighbors=1)

    BOD5_max_list = 1000 * BOD5_max_list

    # Figure a - cathodic potential
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    axa.plot(BOD5_max_list, CP_list, color="k", label="Cathodic Potential")
    # axa.set_xlim([95.62, 97.8])
    axa.set_xlabel("BOD5 Max Concentration (mg/L)", fontsize=11)
    axa.set_ylabel("Cathodic Potential (V)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)

    # Figure b - area volume ratio
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(BOD5_max_list, rAV_list, color="k", label="Area-Volume Ratio")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axb.set_xlabel("BOD5 Max Concentration (mg/L)", fontsize=11)
    axb.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)

    # Figure C - LCOW
    figc, axc = plt.subplots(figsize=(9, 5), layout="constrained")
    axc.plot(BOD5_max_list, LCOW_list, color="k", label="LCOW")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axc.set_xlabel("BOD5 Max Concentration (mg/L)", fontsize=11)
    axc.set_ylabel("LCOW ($/m3 (2023))", fontsize=11)
    axc.tick_params(axis="x", labelsize=11)
    axc.tick_params(axis="y", labelsize=11)

    # Figure C - LCOP
    figd, axd = plt.subplots(figsize=(9, 5), layout="constrained")
    axd.plot(BOD5_max_list, LCOP_list, color="k", label="LCOP")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axd.set_xlabel("BOD5 Max Concentration (mg/L)", fontsize=11)
    axd.set_ylabel("LCOP ($/m3 (2023))", fontsize=11)
    axd.tick_params(axis="x", labelsize=11)
    axd.tick_params(axis="y", labelsize=11)

    plt.show(block=True)


def plot_TKN_max(num):
    # 1D plot
    TKN_max_list = np.linspace(0.0066, 0.0076, num)

    # electroNP flowsheet
    # Cathodic Potential
    CP_list = np.zeros(num)
    CP_list[:] = np.nan

    rAV_list = np.zeros(num)
    rAV_list[:] = np.nan

    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

    LCOP_list = np.zeros(num)
    LCOP_list[:] = np.nan

    # SEC
    SEC_list = np.zeros(num)
    SEC_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_optimization_vary_max(
                COD_max=0.1,
                BOD5_max=0.01,
                TKN_max=TKN_max_list[i],
                TP_max=0.005,
                TSS_max=0.05,
                has_electroNP=True,
                has_optimization=True,
            )

            CP_list[i] = pyo.value(m.fs.electroNP.cathodic_potential)
            rAV_list[i] = pyo.value(m.fs.electroNP.area_volume_ratio)

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            LCOP_list[i] = pyo.value(m.fs.costing.LCOW_P_removal)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)

        except:
            pass

    CP_list = interp_1d(CP_list)
    rAV_list = interp_1d(rAV_list)
    LCOW_list = interp_1d(LCOW_list)
    LCOP_list = interp_1d(LCOP_list)
    SEC_list = interp_1d(SEC_list)

    # CP_list = smooth_1d(CP_list, threshold=0.2, neighbors=1)
    # rAV_list = smooth_1d(rAV_list, threshold=0.2, neighbors=1)
    # LCOW_list = smooth_1d(LCOW_list, threshold=0.2, neighbors=1)
    # LCOP_list = smooth_1d(LCOP_list, threshold=0.2, neighbors=1)
    # SEC_list = smooth_1d(SEC_list, threshold=0.2, neighbors=1)

    TKN_max_list = 1000 * TKN_max_list

    # Figure a - cathodic potential
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    axa.plot(TKN_max_list, CP_list, color="k", label="Cathodic Potential")
    # axa.set_xlim([95.62, 97.8])
    axa.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)
    axa.set_ylabel("Cathodic Potential (V)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)

    # Figure b - area volume ratio
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(TKN_max_list, rAV_list, color="k", label="Area-Volume Ratio")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axb.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)
    axb.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)

    # Figure C - LCOW
    figc, axc = plt.subplots(figsize=(9, 5), layout="constrained")
    axc.plot(TKN_max_list, LCOW_list, color="k", label="LCOW")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axc.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)
    axc.set_ylabel("LCOW ($/m3 (2023))", fontsize=11)
    axc.tick_params(axis="x", labelsize=11)
    axc.tick_params(axis="y", labelsize=11)

    # Figure C - LCOP
    figd, axd = plt.subplots(figsize=(9, 5), layout="constrained")
    axd.plot(TKN_max_list, LCOP_list, color="k", label="LCOP")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axd.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)
    axd.set_ylabel("LCOP ($/m3 (2023))", fontsize=11)
    axd.tick_params(axis="x", labelsize=11)
    axd.tick_params(axis="y", labelsize=11)

    plt.show(block=True)


def plot_TSS_max(num):
    # 1D plot
    TSS_max_list = np.linspace(0.041, 0.046, num)

    # electroNP flowsheet
    # Cathodic Potential
    CP_list = np.zeros(num)
    CP_list[:] = np.nan

    rAV_list = np.zeros(num)
    rAV_list[:] = np.nan

    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

    LCOP_list = np.zeros(num)
    LCOP_list[:] = np.nan

    # SEC
    SEC_list = np.zeros(num)
    SEC_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_optimization_vary_max(
                COD_max=0.1,
                BOD5_max=0.01,
                TKN_max=0.007,
                TP_max=0.005,
                TSS_max=TSS_max_list[i],
                has_electroNP=True,
                has_optimization=True,
            )

            CP_list[i] = pyo.value(m.fs.electroNP.cathodic_potential)
            rAV_list[i] = pyo.value(m.fs.electroNP.area_volume_ratio)

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            LCOP_list[i] = pyo.value(m.fs.costing.LCOW_P_removal)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)

        except:
            pass

    CP_list = interp_1d(CP_list)
    rAV_list = interp_1d(rAV_list)
    LCOW_list = interp_1d(LCOW_list)
    LCOP_list = interp_1d(LCOP_list)
    SEC_list = interp_1d(SEC_list)

    # CP_list = smooth_1d(CP_list, threshold=0.2, neighbors=1)
    # rAV_list = smooth_1d(rAV_list, threshold=0.2, neighbors=1)
    # LCOW_list = smooth_1d(LCOW_list, threshold=0.2, neighbors=1)
    # LCOP_list = smooth_1d(LCOP_list, threshold=0.2, neighbors=1)
    # SEC_list = smooth_1d(SEC_list, threshold=0.2, neighbors=1)

    TSS_max_list = 1000 * TSS_max_list

    # Figure a - cathodic potential
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    axa.plot(TSS_max_list, CP_list, color="k", label="Cathodic Potential")
    # axa.set_xlim([95.62, 97.8])
    axa.set_xlabel("TSS Max Concentration (mg/L)", fontsize=11)
    axa.set_ylabel("Cathodic Potential (V)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)

    # Figure b - area volume ratio
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(TSS_max_list, rAV_list, color="k", label="Area-Volume Ratio")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axb.set_xlabel("TSS Max Concentration (mg/L)", fontsize=11)
    axb.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)

    # Figure C - LCOW
    figc, axc = plt.subplots(figsize=(9, 5), layout="constrained")
    axc.plot(TSS_max_list, LCOW_list, color="k", label="LCOW")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axc.set_xlabel("TSS Max Concentration (mg/L)", fontsize=11)
    axc.set_ylabel("LCOW ($/m3 (2023))", fontsize=11)
    axc.tick_params(axis="x", labelsize=11)
    axc.tick_params(axis="y", labelsize=11)

    # Figure C - LCOP
    figd, axd = plt.subplots(figsize=(9, 5), layout="constrained")
    axd.plot(TSS_max_list, LCOP_list, color="k", label="LCOP")
    # axb.set_ylim([0.86, 0.94])
    # axb.set_xlim([95.62, 97.8])
    axd.set_xlabel("TSS Max Concentration (mg/L)", fontsize=11)
    axd.set_ylabel("LCOP ($/m3 (2023))", fontsize=11)
    axd.tick_params(axis="x", labelsize=11)
    axd.tick_params(axis="y", labelsize=11)

    plt.show(block=True)


def Pareto_front_plot(num):
    epsilon_list = np.linspace(1, 1.4, num)

    pareto_points = []
    for i in range(0, num):
        try:
            m, results = run_optimization_vary_epsilon(
                epsilon=epsilon_list[i],
                COD_max=0.1,
                BOD5_max=0.01,
                TKN_max=0.007,
                TP_max=0.005,
                TSS_max=0.05,
                has_electroNP=True,
                has_optimization=True,
            )

            pareto_points.append(
                [pyo.value(m.fs.costing.LCOW), pyo.value(m.fs.costing.LCOW_P_removal)]
            )
        except:
            pass

    pareto_points = np.array(pareto_points)

    # Figure a
    figa, axa = plt.subplots(figsize=(7, 5), layout="constrained")
    axa.plot(
        pareto_points[:, 0],
        pareto_points[:, 1],
        "ro",
        label="Pareto Front (ε-Constraint)",
    )
    # axa.set_xlim([95.62, 97.8])
    axa.set_xlabel("LCOW ($/m3 (2023))", fontsize=11)
    axa.set_ylabel("LCOP ($/m3 (2023))", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    axa.legend()
    axa.grid(True)

    plt.show(block=True)

    return pareto_points


if __name__ == "__main__":
    plot_electricity_cost_LCOW(num=2)
    # plot_electricity_cost_LCOP(num=10)
    # heatmap_plot_minimize_LCOW(num=5)
    # heatmap_plot_minimize_LCOP(num=5)
    # plot_COD_max(num=19)
    # plot_BOD5_max(num=30)
    # plot_TKN_max(num=15)
    # plot_TSS_max(num=14)

    # Test
    # run_optimization_vary_electricity_cost_phosphorus_revenue(
    #     electricity_cost=0.07,
    #     phosphorus_revenue=0.6521,
    #     has_electroNP=True,
    #     has_optimization=False,
    #     objective=objective_fun.LCOW,
    # )
    #
    # run_optimization_vary_max(
    #     COD_max=0.1,
    #     BOD5_max=0.01,
    #     TKN_max=0.007,
    #     TP_max=0.6,
    #     TSS_max=0.05,
    #     has_electroNP=False,
    #     has_optimization=True,
    # )
    # run_optimization_vary_epsilon(
    #     epsilon=1,
    #     COD_max=0.1,
    #     BOD5_max=0.01,
    #     TKN_max=0.007,
    #     TP_max=0.005,
    #     TSS_max=0.05,
    #     has_electroNP=True,
    #     has_optimization=True,
    # )
    # pareto_points = Pareto_front_plot(num=30)
