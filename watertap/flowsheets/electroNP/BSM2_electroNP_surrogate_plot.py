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


def main(CP=-1.1 * pyo.units.V, r_AV=0.1):
    m = build_flowsheet(has_electroNP=True)
    set_operating_conditions(m)
    # if pyo.value(CP) <= -1.1:
    #     m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    if pyo.value(CP) >= -0.9:
        m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    if pyo.value(r_AV) >= 0.11:
        m.fs.electroNP.area_volume_ratio.fix(pyo.value(r_AV))
    # if pyo.value(CP) >= -0.8 and pyo.value(r_AV) <= 0.09:
    #     m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    #     m.fs.electroNP.area_volume_ratio.fix(pyo.value(r_AV))
    set_scaling(m)
    initialize_system(m)

    m.fs.electroNP.cathodic_potential.unfix()
    m.fs.electroNP.area_volume_ratio.unfix()

    m.fs.electroNP.cathodic_potential.fix(CP)
    m.fs.electroNP.area_volume_ratio.fix(r_AV)

    # results = solve(m)

    add_costing(m)
    m.fs.costing.electroNP_energy_consumption
    m.fs.costing.aeration_energy
    m.fs.costing.initialize()

    interval_initializer(m.fs.costing)

    assert_degrees_of_freedom(m, 0)

    results = solve(m)

    return m, results


def run_with_electricity_cost(CP=-1.1 * pyo.units.V, electricity_cost=0.07):
    m = build_flowsheet(has_electroNP=True)
    set_operating_conditions(m)
    if pyo.value(CP) <= -1.1:
        m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    # # if pyo.value(CP) >= -0.9:
    # #     m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    # if pyo.value(r_AV) >= 0.11:
    #     m.fs.electroNP.area_volume_ratio.fix(pyo.value(r_AV))
    if pyo.value(CP) >= -0.8:
        m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    set_scaling(m)
    initialize_system(m)

    m.fs.electroNP.cathodic_potential.unfix()
    m.fs.electroNP.cathodic_potential.fix(CP)

    # results = solve(m)

    add_costing(m)
    m.fs.costing.electricity_cost.unfix()
    m.fs.costing.electricity_cost.fix(electricity_cost)
    m.fs.costing.electroNP_energy_consumption
    m.fs.costing.aeration_energy
    m.fs.costing.initialize()

    interval_initializer(m.fs.costing)

    assert_degrees_of_freedom(m, 0)

    results = solve(m)

    return m, results


def run_with_KLa(KLa_R5=11, KLa_R6=7, KLa_R7=6):
    m = build_flowsheet(has_electroNP=True)
    set_operating_conditions(m)
    # KLa
    m.fs.R5.KLa = 11
    m.fs.R6.KLa = 7
    m.fs.R7.KLa = 6
    # if pyo.value(CP) <= -1.1:
    #     m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    # # # if pyo.value(CP) >= -0.9:
    # # #     m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    # # if pyo.value(r_AV) >= 0.11:
    # #     m.fs.electroNP.area_volume_ratio.fix(pyo.value(r_AV))
    # if pyo.value(CP) >= -0.8:
    #     m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    set_scaling(m)
    initialize_system(m)

    results = solve(m)

    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R5.KLa.fix(KLa_R5)

    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R6.KLa.fix(KLa_R6)

    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R7.KLa.fix(KLa_R7)

    add_costing(m)
    m.fs.costing.electroNP_energy_consumption
    m.fs.costing.aeration_energy
    m.fs.costing.initialize()

    interval_initializer(m.fs.costing)

    assert_degrees_of_freedom(m, 0)

    results = solve(m)

    return m, results


def run_with_injection(CP=-1.1 * pyo.units.V, r_AV=0.1):
    m = build_flowsheet(has_electroNP=True)
    set_operating_conditions(m)
    # # KLa
    # m.fs.R5.KLa = 11
    # m.fs.R6.KLa = 7
    # m.fs.R7.KLa = 6
    # if pyo.value(CP) <= -1.1:
    #     m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    # # # if pyo.value(CP) >= -0.9:
    # # #     m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    # # if pyo.value(r_AV) >= 0.11:
    # #     m.fs.electroNP.area_volume_ratio.fix(pyo.value(r_AV))
    # if pyo.value(CP) >= -0.8:
    #     m.fs.electroNP.cathodic_potential.fix(pyo.value(CP))
    set_scaling(m)
    initialize_system(m)

    results = solve(m)

    # resolve with oxygen fixed
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R5.injection[0, "Liq", "S_O2"].fix(0.05332)

    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R6.injection[0, "Liq", "S_O2"].fix(0.03025)

    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R7.injection[0, "Liq", "S_O2"].fix(0.02351)

    m.fs.electroNP.cathodic_potential.unfix()
    m.fs.electroNP.area_volume_ratio.unfix()

    m.fs.electroNP.cathodic_potential.fix(CP)
    m.fs.electroNP.area_volume_ratio.fix(r_AV)

    add_costing(m)
    m.fs.costing.electroNP_energy_consumption
    m.fs.costing.aeration_energy
    m.fs.costing.initialize()

    interval_initializer(m.fs.costing)

    assert_degrees_of_freedom(m, 0)

    results = solve(m)

    return m, results


def run_optimization(
    CP=-1.1,
    r_AV=0.1,
    has_electroNP=True,
    has_optimization=True,
    objective=objective_fun.LCOW,
    has_effluent_constraints=True,
):
    m = build_flowsheet(has_electroNP=has_electroNP)
    set_operating_conditions(m)
    set_scaling(m)

    m, results = initialize_system(m)
    add_costing(m)
    m.fs.costing.initialize()
    interval_initializer(m.fs.costing)
    assert_degrees_of_freedom(m, 0)

    # results = solve(m)

    m.fs.electroNP.cathodic_potential.unfix()
    m.fs.electroNP.area_volume_ratio.unfix()

    m.fs.electroNP.cathodic_potential.fix(CP)
    m.fs.electroNP.area_volume_ratio.fix(r_AV)

    if has_optimization:
        setup_aeration(
            m,
            objective=objective,
            has_effluent_constraints=has_effluent_constraints,
            reactor_volume_equalities=False,
        )

    results = solve(m)

    return m, results


def setup_aeration(
    m,
    objective=objective_fun.LCOW,
    has_effluent_constraints=False,
    reactor_volume_equalities=False,
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

    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].setub(1e-2)

    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].setub(1e-2)

    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].setub(1e-2)

    # m.fs.R5.injection[:, :, :].unfix()
    # m.fs.R6.injection[:, :, :].unfix()
    # m.fs.R7.injection[:, :, :].unfix()

    # # Unfix fraction of outflow from reactor 7 that goes to recycle
    # m.fs.SP1.split_fraction[:, "underflow"].unfix()
    # # m.fs.SP1.split_fraction[:, "underflow"].setlb(0.45)
    # m.fs.SP2.split_fraction[:, "recycle"].unfix()

    if has_effluent_constraints:
        add_effluent_violations(m)


def run_optimization_vary_max(
    COD_max=0.1,
    BOD5_max=0.01,
    TKN_max=0.007,
    TP_max=0.005,
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

    if has_electroNP is True:
        m.fs.electroNP.cathodic_potential.unfix()
        m.fs.electroNP.area_volume_ratio.unfix()
        m.fs.electroNP.cathodic_potential.fix(-0.96)
        m.fs.electroNP.area_volume_ratio.fix(0.1)

    # results = solve(m)

    if has_optimization:
        if has_electroNP is True:
            setup_optimization_vary_max(
                m,
                objective=objective_fun.LCOW,
                COD_max=COD_max,
                BOD5_max=BOD5_max,
                TKN_max=TKN_max,
                TP_max=TP_max,
            )
        else:
            setup_optimization_no_electroNP_vary_max(
                m,
                objective=objective_fun.LCOW,
                COD_max=COD_max,
                BOD5_max=BOD5_max,
                TKN_max=TKN_max,
                TP_max=TP_max,
            )

    results = solve(m)

    return m, results


def setup_optimization_vary_max(
    m,
    objective=objective_fun.LCOW,
    COD_max=0.1,
    BOD5_max=0.01,
    TKN_max=0.007,
    TP_max=0.005,
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

    # m.fs.eq_total_P_max[0].deactivate()


def setup_optimization_no_electroNP_vary_max(
    m,
    objective=objective_fun.LCOW,
    COD_max=0.1,
    BOD5_max=0.01,
    TKN_max=0.007,
    TP_max=0.005,
):
    results = solve(m)

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
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].setub(8e-3)

    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].setub(8e-3)

    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].setub(8e-3)

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

    # m.fs.eq_total_P_max[0].deactivate()


def run_optimization_with_aeration_tank_volume(
    aeration_tank_volume=3000,
    has_electroNP=True,
    has_optimization=True,
):
    m = build_flowsheet(has_electroNP=True)
    set_operating_conditions(m)
    # # KLa
    # m.fs.R5.KLa = 11
    # m.fs.R6.KLa = 7
    # m.fs.R7.KLa = 6
    set_scaling(m)
    initialize_system(m)

    m.fs.electroNP.cathodic_potential.unfix()
    m.fs.electroNP.area_volume_ratio.unfix()

    add_costing(m)
    m.fs.costing.electroNP_energy_consumption
    m.fs.costing.aeration_energy
    m.fs.costing.initialize()

    interval_initializer(m.fs.costing)

    # if has_electroNP is True:
    #     m.fs.electroNP.cathodic_potential.unfix()
    #     m.fs.electroNP.area_volume_ratio.unfix()
    #     m.fs.electroNP.cathodic_potential.fix(-0.96)
    #     m.fs.electroNP.area_volume_ratio.fix(0.1)

    results = solve(m)

    m.fs.R5.volume.fix(aeration_tank_volume)
    m.fs.R6.volume.fix(aeration_tank_volume)
    m.fs.R7.volume.fix(aeration_tank_volume)

    if has_optimization:
        setup_optimization(
            m,
            objective=objective_fun.LCOW,
            has_effluent_constraints=True,
            reactor_volume_equalities=False,
        )
        COD_max = (0.099,)
        BOD5_max = (0.0065,)
        TKN_max = (0.007,)
        TP_max = 0.005
        m.fs.COD_max.unfix()
        m.fs.COD_max.fix(COD_max)
        # m.fs.BOD5_max.unfix()
        # m.fs.BOD5_max.fix(BOD5_max)
        # m.fs.TKN_max.unfix()
        # m.fs.TKN_max.fix(TKN_max)
        # m.fs.total_P_max.unfix()
        # m.fs.total_P_max.fix(TP_max)

    results = solve(m)

    return m, results


def plot_CP(num):
    # 1D plot
    CP_list = np.linspace(-1.3, -0.8, num)

    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan
    Ener_electroNP_out = np.zeros(num)
    Ener_electroNP_out[:] = np.nan
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = main(CP=CP_list[i], r_AV=0.10)
            P_out_list[i] = (
                m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
            )

            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            Ener_electroNP_out[i] = pyo.value(m.fs.costing.electroNP_energy_consumption)
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
        except:
            pass

    P_out_list = interp_1d(P_out_list)
    P_removal_list = interp_1d(P_removal_list)
    Ener_electroNP_out = interp_1d(Ener_electroNP_out)
    Ener_aeration_out = interp_1d(Ener_aeration_out)
    SNOX_out_list = interp_1d(SNOX_out_list)

    # Together
    fig1t, ax1t1 = plt.subplots(figsize=(9, 5))
    ax1t1.plot(CP_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    ax1t1.set_ylim([10, 55])
    # # Base case
    # CP_base = -1.1
    # ax1t1.axvline(x=CP_base, color="b", linestyle="--", label="Base case")
    # Optimal
    opt_idx = np.argmin(P_out_list)
    CP_opt = CP_list[opt_idx]
    P_out_opt = P_out_list[opt_idx]
    ax1t1.plot(CP_opt, P_out_opt, marker="o", color="red", label="Optimal")
    ax1t1.annotate(
        f"({round(CP_opt, 2)}, {round(P_out_opt, 2)})",
        (CP_opt, P_out_opt),
        textcoords="offset points",
        xytext=(-76, -10),
    )
    ax1t1.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax1t1.set_ylabel("PO4 Concentration (mg/L)", fontsize=12)
    # ax1t1.legend(loc="lower left")
    ax1t1.tick_params(axis="x", labelsize=12)
    ax1t1.tick_params(axis="y", labelsize=12)
    ax1t1.yaxis.label.set_color("tab:red")
    ax1t1.spines["left"].set_color("tab:red")
    ax1t1.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    ax1t1b = ax1t1.twinx()
    ax1t1b.spines["left"].set_position(("outward", 50))
    ax1t1b.plot(CP_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration")
    ax1t1b.set_ylim([7.7, 7.9])
    # Optimal
    opt_idx = np.argmin(SNOX_out_list)
    CP_opt = CP_list[opt_idx]
    SNOX_out_opt = SNOX_out_list[opt_idx]
    ax1t1b.plot(CP_opt, SNOX_out_opt, marker="o", color="red", label="Optimal")
    ax1t1b.annotate(
        f"({round(CP_opt, 2)}, {round(SNOX_out_opt, 2)})",
        (CP_opt, SNOX_out_opt),
        textcoords="offset points",
        xytext=(6, -16),
    )
    ax1t1b.set_ylabel("SNOx Concentration (mg/L)", fontsize=12)
    ax1t1b.tick_params(axis="x", labelsize=12)
    ax1t1b.tick_params(axis="y", labelsize=12)
    ax1t1b.yaxis.set_ticks_position("left")
    ax1t1b.yaxis.set_label_position("left")
    ax1t1b.yaxis.label.set_color("tab:green")
    ax1t1b.spines["left"].set_color("tab:green")
    ax1t1b.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    ax1t2 = ax1t1.twinx()
    ax1t2.plot(CP_list, P_removal_list, color="tab:blue", label="_Phosphorus Recovery")
    ax1t2.set_ylim([0.87, 0.94])
    # Optimal
    opt_idx = np.argmax(P_removal_list)
    CP_opt = CP_list[opt_idx]
    P_removal_opt = P_removal_list[opt_idx]
    ax1t2.plot(CP_opt, P_removal_opt, marker="o", color="red", label="Optimal")
    ax1t2.annotate(
        f"({round(CP_opt, 2)}, {round(P_removal_opt, 2)})",
        (CP_opt, P_removal_opt),
        textcoords="offset points",
        xytext=(6, 6),
    )
    ax1t2.set_ylabel("Phosphorus Recovery", fontsize=12)
    ax1t2.tick_params(axis="x", labelsize=12)
    ax1t2.tick_params(axis="y", labelsize=12)
    ax1t2.yaxis.label.set_color("tab:blue")
    ax1t2.spines["right"].set_color("tab:blue")
    ax1t2.tick_params(axis="y", colors="tab:blue")

    ax1t3 = ax1t1.twinx()
    ax1t3.spines.right.set_position(("axes", 1.15))
    ax1t3.plot(
        CP_list, Ener_electroNP_out, color="tab:orange", label="_Energy Consumption"
    )
    ax1t3.set_ylim([0, 0.35])
    # Optimal
    opt_idx = np.argmin(Ener_electroNP_out)
    CP_opt = CP_list[opt_idx]
    # m, results = main(CP=CP_opt, r_AV=0.10)
    # Ener_electroNP_out_opt = pyo.value(m.fs.costing.electroNP_energy_consumption)
    Ener_electroNP_out_opt = Ener_electroNP_out[opt_idx]
    ax1t3.plot(CP_opt, Ener_electroNP_out_opt, marker="o", color="red", label="Optimal")
    ax1t3.annotate(
        f"({round(CP_opt, 2)}, {round(Ener_electroNP_out_opt, 4)})",
        (CP_opt, Ener_electroNP_out_opt),
        textcoords="offset points",
        xytext=(-86, -6),
    )
    ax1t3.set_ylabel("Energy Consumption of electron-P (kWh/m3)", fontsize=12)
    ax1t3.tick_params(axis="x", labelsize=12)
    ax1t3.tick_params(axis="y", labelsize=12)
    ax1t3.yaxis.label.set_color("tab:orange")
    ax1t3.spines["right"].set_color("tab:orange")
    ax1t3.tick_params(axis="y", colors="tab:orange")

    ax1t3.spines["left"].set_color("tab:red")
    fig1t.tight_layout()

    plt.show(block=True)


def plot_CP_effluent(num):
    # 1D plot
    CP_list = np.linspace(-1.3, -0.8, num)

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # limits
    TSS_max = 50 * np.ones(num)
    COD_max = 100 * np.ones(num)
    BOD_max = 10 * np.ones(num)
    TKN_max = 7 * np.ones(num)
    TP_max = 5 * np.ones(num)

    for i in range(0, num):
        try:
            m, results = main(CP=CP_list[i], r_AV=0.10)

            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)
            # P_out_list[i] = (
            #     m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
            # )
        except:
            pass

    P_removal_list = interp_1d(P_removal_list)
    TSS_out_list = interp_1d(TSS_out_list)
    COD_out_list = interp_1d(COD_out_list)
    BOD_out_list = interp_1d(BOD_out_list)
    TKN_out_list = interp_1d(TKN_out_list)
    SNOX_out_list = interp_1d(SNOX_out_list)
    P_org_out_list = interp_1d(P_org_out_list)
    P_out_list = interp_1d(P_out_list)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # ax1.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # ax1.legend(loc="lower left")

    # P removal
    ax1.plot(CP_list, P_removal_list, color="k", label="_Phosphorus Recovery")
    # ax1.set_ylim([0.86, 0.94])
    ax1.set_xlabel("Cathodic Potential (V)", fontsize=11)
    ax1.set_ylabel("Phosphorus Recovery", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(CP_list, TSS_out_list, color="tab:blue", label="_TSS Concentration")
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # ax1a.set_ylim([45.15, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(CP_list, COD_out_list, color="tab:orange", label="_COD Concentration")
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # ax1b.set_ylim([96.44, 96.46])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(CP_list, BOD_out_list, color="tab:purple", label="_BOD Concentration")
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    # ax1c.set_ylim([6.066, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax2.plot(CP_list, P_removal_list, color="k", label="_Phosphorus Recovery")
    ax2.set_xlabel("Cathodic Potential (V)", fontsize=11)
    # ax2.set_ylim([0.86, 0.94])
    ax2.set_ylabel("Phosphorus Recovery", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(CP_list, TKN_out_list, color="tab:brown", label="_TKN Concentration")
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # ax1d.set_ylim([6.72, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(CP_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration")
    # ax1e.set_ylim([2.95, 3.15])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # Figure 3
    fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax3.plot(CP_list, P_removal_list, color="k", label="_Phosphorus Recovery")
    # ax3.set_ylim([0.86, 0.94])
    ax3.set_xlabel("Cathodic Potential (V)", fontsize=11)
    ax3.set_ylabel("Phosphorus Recovery", fontsize=11)
    ax3.tick_params(axis="x", labelsize=11)
    ax3.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # organic P
    ax1f = ax3.twinx()
    ax1f.plot(
        CP_list, P_org_out_list, color="tab:pink", label="_Organic P Concentration"
    )
    ax1f.plot(CP_list, TP_max, color="tab:grey", linestyle="--", label="_TP Max")
    # ax1f.set_ylim([5, 5.5])
    ax1f.set_xlabel("Cathodic Potential (V)", fontsize=11)
    ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax1f.tick_params(axis="x", labelsize=11)
    ax1f.tick_params(axis="y", labelsize=11)
    ax1f.yaxis.label.set_color("tab:pink")
    ax1f.spines["right"].set_color("tab:pink")
    ax1f.tick_params(axis="y", colors="tab:pink")
    plt.locator_params(axis="y", nbins=8)

    # PO4
    ax1g = ax3.twinx()
    ax1g.spines.right.set_position(("axes", 1.15))
    ax1g.plot(CP_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    # ax1g.set_ylim([10, 60])
    ax1g.set_xlabel("Cathodic Potential (V)", fontsize=11)
    ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    ax1g.tick_params(axis="x", labelsize=11)
    ax1g.tick_params(axis="y", labelsize=11)
    ax1g.yaxis.label.set_color("tab:red")
    ax1g.spines["right"].set_color("tab:red")
    ax1g.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    # fig1.tight_layout()

    plt.show(block=True)


def plot_rAV(num):
    # 1D plot
    # r_AV_list = np.linspace(0.08, 0.13, num)
    r_AV_list = np.linspace(0.07, 0.14, num)

    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan
    Ener_electroNP_out = np.zeros(num)
    Ener_electroNP_out[:] = np.nan
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = main(CP=-1.1, r_AV=r_AV_list[i])
            P_out_list[i] = (
                m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
            )
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            Ener_electroNP_out[i] = pyo.value(m.fs.costing.electroNP_energy_consumption)
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
        except:
            pass

    P_out_list = interp_1d(P_out_list)
    P_removal_list = interp_1d(P_removal_list)
    Ener_electroNP_out = interp_1d(Ener_electroNP_out)
    Ener_aeration_out = interp_1d(Ener_aeration_out)
    SNOX_out_list = interp_1d(SNOX_out_list)

    # Together
    fig2t, ax2t1 = plt.subplots(figsize=(9, 5))
    ax2t1.plot(r_AV_list, P_out_list, "tab:red", label="_PO4 Concentration")
    ax2t1.set_xlim([0.07, 0.14])
    ax2t1.set_ylim([0, 240])
    # # Base case
    # r_AV_base = 0.1
    # ax2t1.axvline(x=r_AV_base, color="b", linestyle="--", label="Base case")
    # Optimal
    opt_idx = np.argmin(P_out_list)
    r_AV_opt = r_AV_list[opt_idx]
    P_out_opt = P_out_list[opt_idx]
    ax2t1.plot(r_AV_opt, P_out_opt, marker="o", color="red", label="Extremum")
    ax2t1.annotate(
        f"({round(r_AV_opt, 3)}, {round(P_out_opt, 2)})",
        (r_AV_opt, P_out_opt),
        textcoords="offset points",
        xytext=(6, 15),
    )
    ax2t1.set_xlabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    ax2t1.set_ylabel("PO4 Concentration (mg/L)", fontsize=12)
    # ax2t1.legend(loc="lower right")
    ax2t1.tick_params(axis="x", labelsize=12)
    ax2t1.tick_params(axis="y", labelsize=12)
    ax2t1.yaxis.label.set_color("tab:red")
    ax2t1.spines["left"].set_color("tab:red")
    ax2t1.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    ax2t1b = ax2t1.twinx()
    ax2t1b.spines["left"].set_position(("outward", 60))
    ax2t1b.plot(
        r_AV_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration"
    )
    # ax2t1b.set_ylim([2.95, 3.15])
    # Optimal
    opt_idx = np.argmin(SNOX_out_list)
    r_AV_opt = r_AV_list[opt_idx]
    SNOX_out_opt = SNOX_out_list[opt_idx]
    ax2t1b.plot(r_AV_opt, SNOX_out_opt, marker="o", color="red", label="Optimal")
    ax2t1b.annotate(
        f"({round(r_AV_opt, 3)}, {round(SNOX_out_opt, 2)})",
        (r_AV_opt, SNOX_out_opt),
        textcoords="offset points",
        xytext=(6, -10),
    )
    ax2t1b.set_ylabel("SNOx Concentration (mg/L)", fontsize=12)
    ax2t1b.tick_params(axis="x", labelsize=12)
    ax2t1b.tick_params(axis="y", labelsize=12)
    ax2t1b.yaxis.set_ticks_position("left")
    ax2t1b.yaxis.set_label_position("left")
    ax2t1b.yaxis.label.set_color("tab:green")
    ax2t1b.spines["left"].set_color("tab:green")
    ax2t1b.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    ax2t2 = ax2t1.twinx()
    ax2t2.plot(r_AV_list, P_removal_list, "tab:blue", label="_Phosphorus Recovery")
    ax2t2.set_ylim([0.65, 0.95])
    # Optimal
    opt_idx = np.argmax(P_removal_list)
    r_AV_opt = r_AV_list[opt_idx]
    P_removal_opt = P_removal_list[opt_idx]
    ax2t2.plot(r_AV_opt, P_removal_opt, marker="o", color="red", label="Optimal")
    ax2t2.annotate(
        f"({round(r_AV_opt, 3)}, {round(P_removal_opt, 2)})",
        (r_AV_opt, P_removal_opt),
        textcoords="offset points",
        xytext=(3, 6),
    )
    ax2t2.set_ylabel("Phosphorus Recovery", fontsize=12)
    ax2t2.tick_params(axis="x", labelsize=12)
    ax2t2.tick_params(axis="y", labelsize=12)
    ax2t2.yaxis.label.set_color("tab:blue")
    ax2t2.spines["right"].set_color("tab:blue")
    ax2t2.tick_params(axis="y", colors="tab:blue")

    ax2t3 = ax2t1.twinx()
    ax2t3.spines.right.set_position(("axes", 1.15))
    ax2t3.plot(
        r_AV_list, Ener_electroNP_out, color="tab:orange", label="_Energy Consumption"
    )
    ax2t3.set_ylim([0, 0.1])
    # Optimal
    max_idx = np.argmax(Ener_electroNP_out)
    r_AV_max = r_AV_list[max_idx]
    Ener_electroNP_out_max = Ener_electroNP_out[max_idx]
    ax2t3.plot(
        r_AV_max, Ener_electroNP_out_max, marker="o", color="red", label="Optimal"
    )
    ax2t3.annotate(
        f"({round(r_AV_max, 3)}, {round(Ener_electroNP_out_max, 4)})",
        (r_AV_max, Ener_electroNP_out_max),
        textcoords="offset points",
        xytext=(-56, 6),
    )
    # min_idx = np.argmin(Ener_electroNP_out)
    # r_AV_min = r_AV_list[min_idx]
    # Ener_electroNP_out_min = Ener_electroNP_out[min_idx]
    # ax2t3.plot(r_AV_min, Ener_electroNP_out_min, marker="o", color="red", label="Optimal")
    # ax2t3.annotate(
    #     f"({round(r_AV_min, 3)}, {round(Ener_electroNP_out_min, 4)})",
    #     (r_AV_min, Ener_electroNP_out_min),
    #     textcoords="offset points",
    #     xytext=(6, -10),
    # )
    ax2t3.set_ylabel("Energy Consumption of electron-P (kWh/m3)", fontsize=12)
    ax2t3.tick_params(axis="x", labelsize=12)
    ax2t3.tick_params(axis="y", labelsize=12)
    ax2t3.yaxis.label.set_color("tab:orange")
    ax2t3.spines["right"].set_color("tab:orange")
    ax2t3.tick_params(axis="y", colors="tab:orange")

    ax2t3.spines["left"].set_color("tab:red")
    fig2t.tight_layout()
    plt.show(block=True)


def plot_rAV_effluent(num):
    # 1D plot
    r_AV_list = np.linspace(0.07, 0.14, num)

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # limits
    TSS_max = 50 * np.ones(num)
    COD_max = 100 * np.ones(num)
    BOD_max = 10 * np.ones(num)
    TKN_max = 7 * np.ones(num)
    TP_max = 5 * np.ones(num)

    for i in range(0, num):
        try:
            m, results = main(CP=-1.1, r_AV=r_AV_list[i])

            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)
            # P_out_list[i] = (
            #     m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
            # )
        except:
            pass

    P_removal_list = interp_1d(P_removal_list)
    TSS_out_list = interp_1d(TSS_out_list)
    COD_out_list = interp_1d(COD_out_list)
    BOD_out_list = interp_1d(BOD_out_list)
    TKN_out_list = interp_1d(TKN_out_list)
    SNOX_out_list = interp_1d(SNOX_out_list)
    P_org_out_list = interp_1d(P_org_out_list)
    P_out_list = interp_1d(P_out_list)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # ax1.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # ax1.legend(loc="lower left")

    # P removal
    ax1.plot(r_AV_list, P_removal_list, color="k", label="_Phosphorus Recovery")
    ax1.set_xlabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    ax1.set_ylim([0.65, 0.95])
    ax1.set_ylabel("Phosphorus Recovery", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(r_AV_list, TSS_out_list, color="tab:blue", label="_TSS Concentration")
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(r_AV_list, COD_out_list, color="tab:orange", label="_COD Concentration")
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    ax1b.set_ylim([96.41, 96.46])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(r_AV_list, BOD_out_list, color="tab:purple", label="_BOD Concentration")
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    ax1c.set_ylim([6.054, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax2.plot(r_AV_list, P_removal_list, color="k", label="_Phosphorus Recovery")
    ax2.set_ylim([0.65, 0.95])
    ax2.set_xlabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    ax2.set_ylabel("Phosphorus Recovery", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(r_AV_list, TKN_out_list, color="tab:brown", label="_TKN Concentration")
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    ax1d.set_ylim([6.7, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(r_AV_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration")
    ax1e.set_ylim([2.95, 3.5])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # Figure 3
    fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax3.plot(r_AV_list, P_removal_list, color="k", label="_Phosphorus Recovery")
    ax3.set_ylim([0.65, 0.95])
    ax3.set_xlabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    ax3.set_ylabel("Phosphorus Recovery", fontsize=11)
    ax3.tick_params(axis="x", labelsize=11)
    ax3.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # organic P
    ax1f = ax3.twinx()
    ax1f.plot(
        r_AV_list, P_org_out_list, color="tab:pink", label="_Organic P Concentration"
    )
    ax1f.plot(r_AV_list, TP_max, color="tab:grey", linestyle="--", label="_TP Max")
    # ax1f.set_ylim([5, 5.5])
    ax1f.set_xlabel("Area Volume Ratio (cm$^{-1}$)", fontsize=11)
    ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax1f.tick_params(axis="x", labelsize=11)
    ax1f.tick_params(axis="y", labelsize=11)
    ax1f.yaxis.label.set_color("tab:pink")
    ax1f.spines["right"].set_color("tab:pink")
    ax1f.tick_params(axis="y", colors="tab:pink")
    plt.locator_params(axis="y", nbins=8)

    # PO4
    ax1g = ax3.twinx()
    ax1g.spines.right.set_position(("axes", 1.15))
    ax1g.plot(r_AV_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    ax1g.set_ylim([0, 240])
    ax1g.set_xlabel("Area Volume Ratio (cm$^{-1}$)", fontsize=11)
    ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    ax1g.tick_params(axis="x", labelsize=11)
    ax1g.tick_params(axis="y", labelsize=11)
    ax1g.yaxis.label.set_color("tab:red")
    ax1g.spines["right"].set_color("tab:red")
    ax1g.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    # fig1.tight_layout()

    plt.show(block=True)


def contourf_plot(num):
    # # 1D plot
    # CP_list = np.linspace(-1.3, -0.8, num)
    # r_AV_list = np.linspace(0.07, 0.14, num)

    # 2D plot
    CP_list = np.linspace(-1.2, -0.8, num)
    r_AV_list = np.linspace(0.09, 0.12, num)
    # r_AV_list = np.linspace(0.09, 0.13, num)

    P_out_matrix = np.zeros((num, num))
    P_out_matrix[:] = np.nan
    LCOW_matrix = np.zeros((num, num))
    LCOW_matrix[:] = np.nan
    SEC_matrix = np.zeros((num, num))
    SEC_matrix[:] = np.nan
    SEC_electroNP_matrix = np.zeros((num, num))
    SEC_electroNP_matrix[:] = np.nan
    aeration_matrix = np.zeros((num, num))
    aeration_matrix[:] = np.nan
    SEC_electroNP_aeration_matrix = np.zeros((num, num))
    SEC_electroNP_aeration_matrix[:] = np.nan

    for i in range(0, num):
        for j in range(0, num):
            print(f"CP: {CP_list[i]}")
            print(f"rAV: {r_AV_list[j]}")
            try:
                # simulation
                m, results = main(CP=CP_list[i], r_AV=r_AV_list[j])
                # case 2:
                # m, results = run_optimization(
                #     CP=CP_list[i],
                #     r_AV=r_AV_list[j],
                #     has_electroNP=True,
                #     has_optimization=True,
                #     objective=objective_fun.LCOW,
                #     has_effluent_constraints=True,
                # )
                P_out_matrix[j, i] = (
                    m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
                )
                LCOW_matrix[j, i] = pyo.value(m.fs.costing.LCOW)
                SEC_matrix[j, i] = pyo.value(m.fs.costing.specific_energy_consumption)
                SEC_electroNP_matrix[j, i] = pyo.value(
                    m.fs.costing.electroNP_energy_consumption
                ) / pyo.value(m.fs.costing.specific_energy_consumption)
                aeration_matrix[j, i] = pyo.value(m.fs.costing.aeration_energy)
                SEC_electroNP_aeration_matrix[j, i] = pyo.value(
                    m.fs.costing.electroNP_energy_consumption
                ) / pyo.value(m.fs.costing.aeration_energy)
            except:
                pass

    P_out_matrix = interp_2d(P_out_matrix)
    LCOW_matrix = interp_2d(LCOW_matrix)
    SEC_matrix = interp_2d(SEC_matrix)
    SEC_electroNP_matrix = interp_2d(SEC_electroNP_matrix)
    aeration_matrix = interp_2d(aeration_matrix)
    SEC_electroNP_aeration_matrix = interp_2d(SEC_electroNP_aeration_matrix)

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    CF = ax3.contourf(CP_list, r_AV_list, P_out_matrix, cmap="GnBu")
    CP_base = -1.1
    r_AV_base = 0.1
    ax3.plot(CP_base, r_AV_base, marker="o", color="black", markersize=5)
    ax3.annotate(
        f"({CP_base}, {r_AV_base})",
        (CP_base, r_AV_base),
        textcoords="offset points",
        xytext=(6, 6),
    )
    ax3.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax3.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    cbar = fig3.colorbar(CF)
    cbar.ax.set_ylabel("Concentration of PO4 in the treated water (mg/L)", fontsize=12)

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    CF = ax4.contourf(CP_list, r_AV_list, LCOW_matrix, cmap="GnBu")
    ax4.plot(CP_base, r_AV_base, marker="o", color="black", markersize=5)
    ax4.annotate(
        f"({CP_base}, {r_AV_base})",
        (CP_base, r_AV_base),
        textcoords="offset points",
        xytext=(6, 6),
    )
    ax4.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax4.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    cbar = fig4.colorbar(CF)
    cbar.ax.set_ylabel("LCOW ($/m3)", fontsize=12)

    fig5, ax5 = plt.subplots(figsize=(7, 5))
    CF = ax5.contourf(CP_list, r_AV_list, SEC_matrix, cmap="GnBu")
    ax5.plot(CP_base, r_AV_base, marker="o", color="black", markersize=5)
    ax5.annotate(
        f"({CP_base}, {r_AV_base})",
        (CP_base, r_AV_base),
        textcoords="offset points",
        xytext=(6, 6),
    )
    ax5.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax5.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    cbar = fig5.colorbar(CF)
    cbar.ax.set_ylabel("SEC (kWh/m3)", fontsize=12)

    fig6, ax6 = plt.subplots(figsize=(7, 5))
    CF = ax6.contourf(CP_list, r_AV_list, SEC_electroNP_matrix, cmap="GnBu")
    ax6.plot(CP_base, r_AV_base, marker="o", color="black", markersize=5)
    ax6.annotate(
        f"({CP_base}, {r_AV_base})",
        (CP_base, r_AV_base),
        textcoords="offset points",
        xytext=(6, 6),
    )
    ax6.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax6.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    cbar = fig6.colorbar(CF)
    cbar.ax.set_ylabel("ElectroNP SEC / total SEC", fontsize=12)

    fig7, ax7 = plt.subplots(figsize=(7, 5))
    CF = ax7.contourf(CP_list, r_AV_list, aeration_matrix, cmap="GnBu")
    ax7.plot(CP_base, r_AV_base, marker="o", color="black", markersize=5)
    ax7.annotate(
        f"({CP_base}, {r_AV_base})",
        (CP_base, r_AV_base),
        textcoords="offset points",
        xytext=(6, 6),
    )
    ax7.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax7.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    cbar = fig7.colorbar(CF)
    cbar.ax.set_ylabel("Aeration energy (kWh/m3)", fontsize=12)

    fig8, ax8 = plt.subplots(figsize=(7, 5))
    CF = ax8.contourf(CP_list, r_AV_list, SEC_electroNP_aeration_matrix, cmap="GnBu")
    ax8.plot(CP_base, r_AV_base, marker="o", color="black", markersize=5)
    ax8.annotate(
        f"({CP_base}, {r_AV_base})",
        (CP_base, r_AV_base),
        textcoords="offset points",
        xytext=(6, 6),
    )
    ax8.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax8.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    cbar = fig6.colorbar(CF)
    cbar.ax.set_ylabel("ElectroNP SEC / aeration SEC", fontsize=12)

    plt.show(block=True)


def contourf_plot_aeration(num):
    # # 1D plot
    # CP_list = np.linspace(-1.3, -0.8, num)
    # r_AV_list = np.linspace(0.07, 0.14, num)

    # 2D plot
    CP_list = np.linspace(-1.2, -0.8, num)
    # r_AV_list = np.linspace(0.09, 0.12, num)
    r_AV_list = np.linspace(0.09, 0.13, num)

    P_out_matrix = np.zeros((num, num))
    P_out_matrix[:] = np.nan
    LCOW_matrix = np.zeros((num, num))
    LCOW_matrix[:] = np.nan
    SEC_matrix = np.zeros((num, num))
    SEC_matrix[:] = np.nan
    SEC_electroNP_matrix = np.zeros((num, num))
    SEC_electroNP_matrix[:] = np.nan
    aeration_matrix = np.zeros((num, num))
    aeration_matrix[:] = np.nan

    for i in range(0, num):
        for j in range(0, num):
            print(f"CP: {CP_list[i]}")
            print(f"rAV: {r_AV_list[j]}")
            try:
                # simulation
                m, results = run_with_injection(CP=CP_list[i], r_AV=r_AV_list[j])
                # # case 2:
                # m, results = run_optimization(
                #     CP=CP_list[i],
                #     r_AV=r_AV_list[j],
                #     has_electroNP=True,
                #     has_optimization=True,
                #     objective=objective_fun.LCOW,
                #     has_effluent_constraints=True,
                # )
                P_out_matrix[j, i] = (
                    m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
                )
                LCOW_matrix[j, i] = pyo.value(m.fs.costing.LCOW)
                SEC_matrix[j, i] = pyo.value(m.fs.costing.specific_energy_consumption)
                SEC_electroNP_matrix[j, i] = pyo.value(
                    m.fs.costing.electroNP_energy_consumption
                ) / pyo.value(m.fs.costing.specific_energy_consumption)
                aeration_matrix[j, i] = pyo.value(m.fs.costing.aeration_energy)
            except:
                pass

    P_out_matrix = interp_2d(P_out_matrix)
    LCOW_matrix = interp_2d(LCOW_matrix)
    SEC_matrix = interp_2d(SEC_matrix)
    SEC_electroNP_matrix = interp_2d(SEC_electroNP_matrix)
    aeration_matrix = interp_2d(aeration_matrix)

    CP_base = -1.1
    r_AV_base = 0.1

    # fig3, ax3 = plt.subplots(figsize=(7, 5))
    # CF = ax3.contourf(CP_list, r_AV_list, P_out_matrix, cmap="GnBu")

    # ax3.plot(CP_base, r_AV_base, marker="o", color="black", markersize=5)
    # ax3.annotate(
    #     f"({CP_base}, {r_AV_base})",
    #     (CP_base, r_AV_base),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    # ax3.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # ax3.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    # cbar = fig3.colorbar(CF)
    # cbar.ax.set_ylabel("Concentration of PO4 in the treated water (mg/L)", fontsize=12)
    #
    # fig4, ax4 = plt.subplots(figsize=(7, 5))
    # CF = ax4.contourf(CP_list, r_AV_list, LCOW_matrix, cmap="GnBu")
    # ax4.plot(CP_base, r_AV_base, marker="o", color="black", markersize=5)
    # ax4.annotate(
    #     f"({CP_base}, {r_AV_base})",
    #     (CP_base, r_AV_base),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    # ax4.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # ax4.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    # cbar = fig4.colorbar(CF)
    # cbar.ax.set_ylabel("LCOW ($/m3)", fontsize=12)
    #
    # fig5, ax5 = plt.subplots(figsize=(7, 5))
    # CF = ax5.contourf(CP_list, r_AV_list, SEC_matrix, cmap="GnBu")
    # ax5.plot(CP_base, r_AV_base, marker="o", color="black", markersize=5)
    # ax5.annotate(
    #     f"({CP_base}, {r_AV_base})",
    #     (CP_base, r_AV_base),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    # ax5.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # ax5.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    # cbar = fig5.colorbar(CF)
    # cbar.ax.set_ylabel("SEC (kWh/m3)", fontsize=12)
    #
    # fig6, ax6 = plt.subplots(figsize=(7, 5))
    # CF = ax6.contourf(CP_list, r_AV_list, SEC_electroNP_matrix, cmap="GnBu")
    # ax6.plot(CP_base, r_AV_base, marker="o", color="black", markersize=5)
    # ax6.annotate(
    #     f"({CP_base}, {r_AV_base})",
    #     (CP_base, r_AV_base),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    # ax6.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # ax6.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    # cbar = fig6.colorbar(CF)
    # cbar.ax.set_ylabel("ElectroNP SEC / total SEC", fontsize=12)

    fig7, ax7 = plt.subplots(figsize=(7, 5))
    CF = ax7.contourf(CP_list, r_AV_list, aeration_matrix, cmap="GnBu")
    ax7.plot(CP_base, r_AV_base, marker="o", color="black", markersize=5)
    ax7.annotate(
        f"({CP_base}, {r_AV_base})",
        (CP_base, r_AV_base),
        textcoords="offset points",
        xytext=(6, 6),
    )
    ax7.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax7.set_ylabel("Area Volume Ratio (cm$^{-1}$)", fontsize=12)
    cbar = fig7.colorbar(CF)
    cbar.ax.set_ylabel("Aeration energy (kWh/m3)", fontsize=12)

    plt.show(block=True)


def contourf_plot_electricity_cost(num):
    # # 1D plot
    # CP_list = np.linspace(-1.3, -0.8, num)
    # r_AV_list = np.linspace(0.07, 0.14, num)

    # 2D plot
    CP_list = np.linspace(-1.2, -0.8, num)
    electricity_cost_list = np.linspace(0.06, 0.08, num)

    P_out_matrix = np.zeros((num, num))
    P_out_matrix[:] = np.nan
    LCOW_matrix = np.zeros((num, num))
    LCOW_matrix[:] = np.nan
    SEC_matrix = np.zeros((num, num))
    SEC_matrix[:] = np.nan
    SEC_electroNP_matrix = np.zeros((num, num))
    SEC_electroNP_matrix[:] = np.nan
    aeration_matrix = np.zeros((num, num))
    aeration_matrix[:] = np.nan

    for i in range(0, num):
        for j in range(0, num):
            print(f"CP: {CP_list[i]}")
            print(f"electricity cost: {electricity_cost_list[j]}")
            try:
                # # simulation
                m, results = run_with_electricity_cost(
                    CP=CP_list[i], electricity_cost=electricity_cost_list[j]
                )

                P_out_matrix[j, i] = (
                    m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
                )
                LCOW_matrix[j, i] = pyo.value(m.fs.costing.LCOW)
                SEC_matrix[j, i] = pyo.value(m.fs.costing.specific_energy_consumption)
                SEC_electroNP_matrix[j, i] = pyo.value(
                    m.fs.costing.electroNP_energy_consumption
                ) / pyo.value(m.fs.costing.specific_energy_consumption)
                aeration_matrix[j, i] = pyo.value(m.fs.costing.aeration_energy)
            except:
                pass

    P_out_matrix = interp_2d(P_out_matrix)
    LCOW_matrix = interp_2d(LCOW_matrix)
    SEC_matrix = interp_2d(SEC_matrix)
    SEC_electroNP_matrix = interp_2d(SEC_electroNP_matrix)
    aeration_matrix = interp_2d(aeration_matrix)

    fig8, ax8 = plt.subplots(figsize=(7, 5))
    CF = ax8.contourf(CP_list, electricity_cost_list, LCOW_matrix, cmap="GnBu")
    CP_base = -1.1
    electricity_cost_base = 0.07
    ax8.plot(CP_base, electricity_cost_base, marker="o", color="black", markersize=5)
    ax8.annotate(
        f"({CP_base}, {electricity_cost_base})",
        (CP_base, electricity_cost_base),
        textcoords="offset points",
        xytext=(6, 6),
    )
    ax8.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax8.set_ylabel("Electricity Cost ($/kWh (2018))", fontsize=12)
    cbar = fig8.colorbar(CF)
    cbar.ax.set_ylabel("LCOW ($/m3 (2023))", fontsize=12)

    plt.show(block=True)

    return P_out_matrix

    plt.show(block=True)


def plot_aeration_R5(num):
    # 1D plot
    KLa_R5_list = np.linspace(9, 13, num)

    # O2 -- R5
    S_O2_out_R5_list = np.zeros(num)
    S_O2_out_R5_list[:] = np.nan

    # O2 -- R6
    S_O2_out_R6_list = np.zeros(num)
    S_O2_out_R6_list[:] = np.nan

    # O2 -- R7
    S_O2_out_R7_list = np.zeros(num)
    S_O2_out_R7_list[:] = np.nan

    # O2 -- effluent
    S_O2_out_list = np.zeros(num)
    S_O2_out_list[:] = np.nan

    # aeration energy
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # Injection - R5
    Injection_R5_list = np.zeros(num)
    Injection_R5_list[:] = np.nan

    # Injection - R6
    Injection_R6_list = np.zeros(num)
    Injection_R6_list[:] = np.nan

    # Injection - R7
    Injection_R7_list = np.zeros(num)
    Injection_R7_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_with_KLa(KLa_R5=KLa_R5_list[i], KLa_R6=7, KLa_R7=6)

            S_O2_out_R5_list[i] = pyo.value(
                m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R6_list[i] = pyo.value(
                m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R7_list[i] = pyo.value(
                m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3
            )
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)
            Injection_R5_list[i] = pyo.value(m.fs.R5.injection[0, "Liq", "S_O2"])
            Injection_R6_list[i] = pyo.value(m.fs.R6.injection[0, "Liq", "S_O2"])
            Injection_R7_list[i] = pyo.value(m.fs.R7.injection[0, "Liq", "S_O2"])
        except:
            pass

    S_O2_out_R5_list = interp_1d(S_O2_out_R5_list)
    S_O2_out_R6_list = interp_1d(S_O2_out_R6_list)
    S_O2_out_R7_list = interp_1d(S_O2_out_R7_list)
    S_O2_out_list = interp_1d(S_O2_out_list)
    Ener_aeration_out = interp_1d(Ener_aeration_out)
    P_removal_list = interp_1d(P_removal_list)
    TSS_out_list = interp_1d(TSS_out_list)
    COD_out_list = interp_1d(COD_out_list)
    BOD_out_list = interp_1d(BOD_out_list)
    TKN_out_list = interp_1d(TKN_out_list)
    SNOX_out_list = interp_1d(SNOX_out_list)
    P_org_out_list = interp_1d(P_org_out_list)
    P_out_list = interp_1d(P_out_list)
    Injection_R5_list = interp_1d(Injection_R5_list)
    Injection_R6_list = interp_1d(Injection_R6_list)
    Injection_R7_list = interp_1d(Injection_R7_list)

    # Figure a
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # axa.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # axa.legend(loc="lower left")

    # O2 -- R5
    axa.plot(KLa_R5_list, S_O2_out_R5_list, color="k", label="_R5 O2 Concentration")
    # axa.set_ylim([0.86, 0.94])
    axa.set_xlabel("R5 KLa (s$^{-1}$)", fontsize=11)
    axa.set_ylabel("R5 O2 Concentration  (mg/L)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R6
    axa1 = axa.twinx()
    axa1.plot(
        KLa_R5_list, S_O2_out_R6_list, color="tab:blue", label="_R6 O2 Concentration"
    )
    # axa1.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # axa1.set_ylim([45.15, 45.25])
    axa1.set_ylabel("R6 O2 Concentration (mg/L)", fontsize=11)
    axa1.tick_params(axis="x", labelsize=11)
    axa1.tick_params(axis="y", labelsize=11)
    axa1.yaxis.label.set_color("tab:blue")
    axa1.spines["right"].set_color("tab:blue")
    axa1.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R7
    axa2 = axa.twinx()
    axa2.spines.right.set_position(("axes", 1.15))
    axa2.plot(
        KLa_R5_list, S_O2_out_R7_list, color="tab:orange", label="_R7 O2 Concentration"
    )
    # axa2.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # axa2.set_ylim([96.44, 96.46])
    axa2.set_ylabel("R7 O2 Concentration (mg/L)", fontsize=11)
    axa2.tick_params(axis="x", labelsize=11)
    axa2.tick_params(axis="y", labelsize=11)
    axa2.yaxis.label.set_color("tab:orange")
    axa2.spines["right"].set_color("tab:orange")
    axa2.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # Figure b
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(KLa_R5_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    axb.set_xlabel("R5 KLa (s$^{-1}$)", fontsize=11)
    axb.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # aeration energy
    axb1 = axb.twinx()
    axb1.plot(
        KLa_R5_list, Ener_aeration_out, color="tab:brown", label="_Aeration energy"
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axb1.set_ylabel("Aeration energy (kWh/m$^3$)", fontsize=11)
    axb1.tick_params(axis="x", labelsize=11)
    axb1.tick_params(axis="y", labelsize=11)
    axb1.yaxis.label.set_color("tab:brown")
    axb1.spines["right"].set_color("tab:brown")
    axb1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.set_xlabel("R5 KLa (s$^{-1}$)", fontsize=11)

    # O2
    ax1.plot(KLa_R5_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax1.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(KLa_R5_list, TSS_out_list, color="tab:blue", label="_TSS Concentration")
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(KLa_R5_list, COD_out_list, color="tab:orange", label="_COD Concentration")
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    ax1b.set_ylim([96.2, 96.7])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(KLa_R5_list, BOD_out_list, color="tab:purple", label="_BOD Concentration")
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    # ax1c.set_ylim([6.054, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax2.set_xlabel("R5 KLa (s$^{-1}$)", fontsize=12)

    # O2
    ax2.plot(KLa_R5_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax2.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(KLa_R5_list, TKN_out_list, color="tab:brown", label="_TKN Concentration")
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # ax1d.set_ylim([6.7, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(
        KLa_R5_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration"
    )
    # ax1e.set_ylim([2.95, 3.5])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # Figure 3
    fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax3.set_xlabel("R5 KLa (s$^{-1}$)", fontsize=12)

    # O2
    ax3.plot(KLa_R5_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax3.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax3.tick_params(axis="x", labelsize=11)
    ax3.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # organic P
    ax1f = ax3.twinx()
    ax1f.plot(
        KLa_R5_list, P_org_out_list, color="tab:pink", label="_Organic P Concentration"
    )
    # ax1f.plot(KLa_R5_list, TP_max, color="tab:grey", linestyle='--', label='_TP Max')
    ax1f.set_ylim([5.3, 5.52])
    ax1f.set_xlabel("R5 KLa (s$^{-1}$)", fontsize=11)
    ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax1f.tick_params(axis="x", labelsize=11)
    ax1f.tick_params(axis="y", labelsize=11)
    ax1f.yaxis.label.set_color("tab:pink")
    ax1f.spines["right"].set_color("tab:pink")
    ax1f.tick_params(axis="y", colors="tab:pink")
    plt.locator_params(axis="y", nbins=8)

    # PO4
    ax1g = ax3.twinx()
    ax1g.spines.right.set_position(("axes", 1.15))
    ax1g.plot(KLa_R5_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    # ax1g.set_ylim([0, 240])
    ax1g.set_xlabel("R5 KLa (s$^{-1}$)", fontsize=11)
    ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    ax1g.tick_params(axis="x", labelsize=11)
    ax1g.tick_params(axis="y", labelsize=11)
    ax1g.yaxis.label.set_color("tab:red")
    ax1g.spines["right"].set_color("tab:red")
    ax1g.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    # Figure 4
    fig4, ax4 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax4.set_xlabel("R5 KLa (s$^{-1}$)", fontsize=12)

    # R5 Injection
    ax4.plot(KLa_R5_list, Injection_R5_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax4.set_ylabel("R5 Injection (kg/h)", fontsize=11)
    ax4.tick_params(axis="x", labelsize=11)
    ax4.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # R6 Injection
    ax4a = ax4.twinx()
    ax4a.plot(
        KLa_R5_list,
        Injection_R6_list,
        color="tab:blue",
        label="_Organic P Concentration",
    )
    # ax4a.plot(KLa_R5_list, TP_max, color="tab:grey", linestyle='--', label='_TP Max')
    # ax4a.set_ylim([5.3, 5.52])
    ax4a.set_ylabel("R6 Injection (kg/h)", fontsize=11)
    ax4a.tick_params(axis="x", labelsize=11)
    ax4a.tick_params(axis="y", labelsize=11)
    ax4a.yaxis.label.set_color("tab:blue")
    ax4a.spines["right"].set_color("tab:blue")
    ax4a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # R7 Injection
    ax4b = ax4.twinx()
    ax4b.spines.right.set_position(("axes", 1.15))
    ax4b.plot(
        KLa_R5_list, Injection_R7_list, color="tab:red", label="_PO4 Concentration"
    )
    # ax4b.set_ylim([0, 240])
    ax4b.set_ylabel("R7 Injection (kg/h)", fontsize=11)
    ax4b.tick_params(axis="x", labelsize=11)
    ax4b.tick_params(axis="y", labelsize=11)
    ax4b.yaxis.label.set_color("tab:red")
    ax4b.spines["right"].set_color("tab:red")
    ax4b.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    plt.show(block=True)


def plot_aeration_R6(num):
    # 1D plot
    KLa_R6_list = np.linspace(6, 14, num)

    # O2 -- R5
    S_O2_out_R5_list = np.zeros(num)
    S_O2_out_R5_list[:] = np.nan

    # O2 -- R6
    S_O2_out_R6_list = np.zeros(num)
    S_O2_out_R6_list[:] = np.nan

    # O2 -- R7
    S_O2_out_R7_list = np.zeros(num)
    S_O2_out_R7_list[:] = np.nan

    # O2 -- effluent
    S_O2_out_list = np.zeros(num)
    S_O2_out_list[:] = np.nan

    # aeration energy
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # Injection - R5
    Injection_R5_list = np.zeros(num)
    Injection_R5_list[:] = np.nan

    # Injection - R6
    Injection_R6_list = np.zeros(num)
    Injection_R6_list[:] = np.nan

    # Injection - R7
    Injection_R7_list = np.zeros(num)
    Injection_R7_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_with_KLa(KLa_R5=11, KLa_R6=KLa_R6_list[i], KLa_R7=6)

            S_O2_out_R5_list[i] = pyo.value(
                m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R6_list[i] = pyo.value(
                m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R7_list[i] = pyo.value(
                m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3
            )
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)
            Injection_R5_list[i] = pyo.value(m.fs.R5.injection[0, "Liq", "S_O2"])
            Injection_R6_list[i] = pyo.value(m.fs.R6.injection[0, "Liq", "S_O2"])
            Injection_R7_list[i] = pyo.value(m.fs.R7.injection[0, "Liq", "S_O2"])
        except:
            pass

    S_O2_out_R5_list = interp_1d(S_O2_out_R5_list)
    S_O2_out_R6_list = interp_1d(S_O2_out_R6_list)
    S_O2_out_R7_list = interp_1d(S_O2_out_R7_list)
    S_O2_out_list = interp_1d(S_O2_out_list)
    Ener_aeration_out = interp_1d(Ener_aeration_out)
    P_removal_list = interp_1d(P_removal_list)
    TSS_out_list = interp_1d(TSS_out_list)
    COD_out_list = interp_1d(COD_out_list)
    BOD_out_list = interp_1d(BOD_out_list)
    TKN_out_list = interp_1d(TKN_out_list)
    SNOX_out_list = interp_1d(SNOX_out_list)
    P_org_out_list = interp_1d(P_org_out_list)
    P_out_list = interp_1d(P_out_list)
    Injection_R5_list = interp_1d(Injection_R5_list)
    Injection_R6_list = interp_1d(Injection_R6_list)
    Injection_R7_list = interp_1d(Injection_R7_list)

    # Figure 1
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # axa.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # axa.legend(loc="lower left")

    # O2 -- R5
    axa.plot(KLa_R6_list, S_O2_out_R5_list, color="k", label="_R5 O2 Concentration")
    # axa.set_ylim([0.86, 0.94])
    axa.set_xlabel("R6 KLa (s$^{-1}$)", fontsize=11)
    axa.set_ylabel("R5 O2 Concentration  (mg/L)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R6
    axa1 = axa.twinx()
    axa1.plot(
        KLa_R6_list, S_O2_out_R6_list, color="tab:blue", label="_R6 O2 Concentration"
    )
    # axa1.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # axa1.set_ylim([45.15, 45.25])
    axa1.set_ylabel("R6 O2 Concentration (mg/L)", fontsize=11)
    axa1.tick_params(axis="x", labelsize=11)
    axa1.tick_params(axis="y", labelsize=11)
    axa1.yaxis.label.set_color("tab:blue")
    axa1.spines["right"].set_color("tab:blue")
    axa1.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R7
    axa2 = axa.twinx()
    axa2.spines.right.set_position(("axes", 1.15))
    axa2.plot(
        KLa_R6_list, S_O2_out_R7_list, color="tab:orange", label="_R7 O2 Concentration"
    )
    # axa2.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # axa2.set_ylim([96.44, 96.46])
    axa2.set_ylabel("R7 O2 Concentration (mg/L)", fontsize=11)
    axa2.tick_params(axis="x", labelsize=11)
    axa2.tick_params(axis="y", labelsize=11)
    axa2.yaxis.label.set_color("tab:orange")
    axa2.spines["right"].set_color("tab:orange")
    axa2.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(KLa_R6_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    axb.set_xlabel("R6 KLa (s$^{-1}$)", fontsize=11)
    axb.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # aeration energy
    axb1 = axb.twinx()
    axb1.plot(
        KLa_R6_list, Ener_aeration_out, color="tab:brown", label="_Aeration energy"
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axb1.set_ylabel("Aeration energy (kWh/m$^3$)", fontsize=11)
    axb1.tick_params(axis="x", labelsize=11)
    axb1.tick_params(axis="y", labelsize=11)
    axb1.yaxis.label.set_color("tab:brown")
    axb1.spines["right"].set_color("tab:brown")
    axb1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.set_xlabel("R6 KLa (s$^{-1}$)", fontsize=12)

    # O2
    ax1.plot(KLa_R6_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax1.set_xlabel("R6 KLa (s$^{-1}$)", fontsize=11)
    ax1.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(KLa_R6_list, TSS_out_list, color="tab:blue", label="_TSS Concentration")
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(KLa_R6_list, COD_out_list, color="tab:orange", label="_COD Concentration")
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # ax1b.set_ylim([96.41, 96.46])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(KLa_R6_list, BOD_out_list, color="tab:purple", label="_BOD Concentration")
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    # ax1c.set_ylim([6.054, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")

    # O2
    ax2.plot(KLa_R6_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax2.set_xlabel("R6 KLa (s$^{-1}$)", fontsize=11)
    ax2.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(KLa_R6_list, TKN_out_list, color="tab:brown", label="_TKN Concentration")
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # ax1d.set_ylim([6.7, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(
        KLa_R6_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration"
    )
    # ax1e.set_ylim([2.95, 3.5])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # Figure 3
    fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")

    # O2
    ax3.plot(KLa_R6_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax3.set_xlabel("R6 KLa (s$^{-1}$)", fontsize=11)
    ax3.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax3.tick_params(axis="x", labelsize=11)
    ax3.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # organic P
    ax1f = ax3.twinx()
    ax1f.plot(
        KLa_R6_list, P_org_out_list, color="tab:pink", label="_Organic P Concentration"
    )
    # ax1f.plot(KLa_R6_list, TP_max, color="tab:grey", linestyle='--', label='_TP Max')
    # ax1f.set_ylim([5, 5.5])
    ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax1f.tick_params(axis="x", labelsize=11)
    ax1f.tick_params(axis="y", labelsize=11)
    ax1f.yaxis.label.set_color("tab:pink")
    ax1f.spines["right"].set_color("tab:pink")
    ax1f.tick_params(axis="y", colors="tab:pink")
    plt.locator_params(axis="y", nbins=8)

    # PO4
    ax1g = ax3.twinx()
    ax1g.spines.right.set_position(("axes", 1.15))
    ax1g.plot(KLa_R6_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    # ax1g.set_ylim([0, 240])
    ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    ax1g.tick_params(axis="x", labelsize=11)
    ax1g.tick_params(axis="y", labelsize=11)
    ax1g.yaxis.label.set_color("tab:red")
    ax1g.spines["right"].set_color("tab:red")
    ax1g.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    # Figure 4
    fig4, ax4 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax4.set_xlabel("R6 KLa (s$^{-1}$)", fontsize=12)

    # R5 Injection
    ax4.plot(KLa_R6_list, Injection_R5_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax4.set_ylabel("R5 Injection (kg/h)", fontsize=11)
    ax4.tick_params(axis="x", labelsize=11)
    ax4.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # R6 Injection
    ax4a = ax4.twinx()
    ax4a.plot(
        KLa_R6_list,
        Injection_R6_list,
        color="tab:blue",
        label="_Organic P Concentration",
    )
    # ax4a.plot(KLa_R5_list, TP_max, color="tab:grey", linestyle='--', label='_TP Max')
    # ax4a.set_ylim([5.3, 5.52])
    ax4a.set_ylabel("R6 Injection (kg/h)", fontsize=11)
    ax4a.tick_params(axis="x", labelsize=11)
    ax4a.tick_params(axis="y", labelsize=11)
    ax4a.yaxis.label.set_color("tab:blue")
    ax4a.spines["right"].set_color("tab:blue")
    ax4a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # R7 Injection
    ax4b = ax4.twinx()
    ax4b.spines.right.set_position(("axes", 1.15))
    ax4b.plot(
        KLa_R6_list, Injection_R7_list, color="tab:red", label="_PO4 Concentration"
    )
    # ax4b.set_ylim([0, 240])
    ax4b.set_ylabel("R7 Injection (kg/h)", fontsize=11)
    ax4b.tick_params(axis="x", labelsize=11)
    ax4b.tick_params(axis="y", labelsize=11)
    ax4b.yaxis.label.set_color("tab:red")
    ax4b.spines["right"].set_color("tab:red")
    ax4b.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    plt.show(block=True)


def plot_aeration_R7(num):
    # 1D plot
    KLa_R7_list = np.linspace(5, 10, num)

    # O2 -- R5
    S_O2_out_R5_list = np.zeros(num)
    S_O2_out_R5_list[:] = np.nan

    # O2 -- R6
    S_O2_out_R6_list = np.zeros(num)
    S_O2_out_R6_list[:] = np.nan

    # O2 -- R7
    S_O2_out_R7_list = np.zeros(num)
    S_O2_out_R7_list[:] = np.nan

    # O2 -- effluent
    S_O2_out_list = np.zeros(num)
    S_O2_out_list[:] = np.nan

    # aeration energy
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # Injection - R5
    Injection_R5_list = np.zeros(num)
    Injection_R5_list[:] = np.nan

    # Injection - R6
    Injection_R6_list = np.zeros(num)
    Injection_R6_list[:] = np.nan

    # Injection - R7
    Injection_R7_list = np.zeros(num)
    Injection_R7_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_with_KLa(KLa_R5=11, KLa_R6=7, KLa_R7=KLa_R7_list[i])

            S_O2_out_R5_list[i] = pyo.value(
                m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R6_list[i] = pyo.value(
                m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R7_list[i] = pyo.value(
                m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3
            )
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)
            Injection_R5_list[i] = pyo.value(m.fs.R5.injection[0, "Liq", "S_O2"])
            Injection_R6_list[i] = pyo.value(m.fs.R6.injection[0, "Liq", "S_O2"])
            Injection_R7_list[i] = pyo.value(m.fs.R7.injection[0, "Liq", "S_O2"])
        except:
            pass

    S_O2_out_R5_list = interp_1d(S_O2_out_R5_list)
    S_O2_out_R6_list = interp_1d(S_O2_out_R6_list)
    S_O2_out_R7_list = interp_1d(S_O2_out_R7_list)
    S_O2_out_list = interp_1d(S_O2_out_list)
    Ener_aeration_out = interp_1d(Ener_aeration_out)
    P_removal_list = interp_1d(P_removal_list)
    TSS_out_list = interp_1d(TSS_out_list)
    COD_out_list = interp_1d(COD_out_list)
    BOD_out_list = interp_1d(BOD_out_list)
    TKN_out_list = interp_1d(TKN_out_list)
    SNOX_out_list = interp_1d(SNOX_out_list)
    P_org_out_list = interp_1d(P_org_out_list)
    P_out_list = interp_1d(P_out_list)
    Injection_R5_list = interp_1d(Injection_R5_list)
    Injection_R6_list = interp_1d(Injection_R6_list)
    Injection_R7_list = interp_1d(Injection_R7_list)

    # Figure 1
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # axa.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # axa.legend(loc="lower left")

    # O2 -- R5
    axa.plot(KLa_R7_list, S_O2_out_R5_list, color="k", label="_R5 O2 Concentration")
    # axa.set_ylim([0.86, 0.94])
    axa.set_xlabel("R7 KLa (s$^{-1}$)", fontsize=11)
    axa.set_ylabel("R5 O2 Concentration  (mg/L)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R6
    axa1 = axa.twinx()
    axa1.plot(
        KLa_R7_list, S_O2_out_R6_list, color="tab:blue", label="_R6 O2 Concentration"
    )
    # axa1.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # axa1.set_ylim([45.15, 45.25])
    axa1.set_ylabel("R6 O2 Concentration (mg/L)", fontsize=11)
    axa1.tick_params(axis="x", labelsize=11)
    axa1.tick_params(axis="y", labelsize=11)
    axa1.yaxis.label.set_color("tab:blue")
    axa1.spines["right"].set_color("tab:blue")
    axa1.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R7
    axa2 = axa.twinx()
    axa2.spines.right.set_position(("axes", 1.15))
    axa2.plot(
        KLa_R7_list, S_O2_out_R7_list, color="tab:orange", label="_R7 O2 Concentration"
    )
    # axa2.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # axa2.set_ylim([96.44, 96.46])
    axa2.set_ylabel("R7 O2 Concentration (mg/L)", fontsize=11)
    axa2.tick_params(axis="x", labelsize=11)
    axa2.tick_params(axis="y", labelsize=11)
    axa2.yaxis.label.set_color("tab:orange")
    axa2.spines["right"].set_color("tab:orange")
    axa2.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(KLa_R7_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    axb.set_xlabel("R7 KLa (s$^{-1}$)", fontsize=11)
    axb.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # aeration energy
    axb1 = axb.twinx()
    axb1.plot(
        KLa_R7_list, Ener_aeration_out, color="tab:brown", label="_Aeration energy"
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axb1.set_ylabel("Aeration energy (kWh/m$^3$)", fontsize=11)
    axb1.tick_params(axis="x", labelsize=11)
    axb1.tick_params(axis="y", labelsize=11)
    axb1.yaxis.label.set_color("tab:brown")
    axb1.spines["right"].set_color("tab:brown")
    axb1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")

    # O2
    ax1.plot(KLa_R7_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax1.set_xlabel("R7 KLa (s$^{-1}$)", fontsize=11)
    ax1.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(KLa_R7_list, TSS_out_list, color="tab:blue", label="_TSS Concentration")
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(KLa_R7_list, COD_out_list, color="tab:orange", label="_COD Concentration")
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # ax1b.set_ylim([96.41, 96.46])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(KLa_R7_list, BOD_out_list, color="tab:purple", label="_BOD Concentration")
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    # ax1c.set_ylim([6.054, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")

    # O2
    ax2.plot(KLa_R7_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax2.set_xlabel("R7 KLa (s$^{-1}$)", fontsize=11)
    ax2.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(KLa_R7_list, TKN_out_list, color="tab:brown", label="_TKN Concentration")
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # ax1d.set_ylim([6.7, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(
        KLa_R7_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration"
    )
    # ax1e.set_ylim([2.95, 3.5])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # Figure 3
    fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")

    # O2
    ax3.plot(KLa_R7_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax3.set_xlabel("R7 KLa (s$^{-1}$)", fontsize=11)
    ax3.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax3.tick_params(axis="x", labelsize=11)
    ax3.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # organic P
    ax1f = ax3.twinx()
    ax1f.plot(
        KLa_R7_list, P_org_out_list, color="tab:pink", label="_Organic P Concentration"
    )
    # ax1f.plot(KLa_R7_list, TP_max, color="tab:grey", linestyle='--', label='_TP Max')
    # ax1f.set_ylim([5, 5.5])
    ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax1f.tick_params(axis="x", labelsize=11)
    ax1f.tick_params(axis="y", labelsize=11)
    ax1f.yaxis.label.set_color("tab:pink")
    ax1f.spines["right"].set_color("tab:pink")
    ax1f.tick_params(axis="y", colors="tab:pink")
    plt.locator_params(axis="y", nbins=8)

    # PO4
    ax1g = ax3.twinx()
    ax1g.spines.right.set_position(("axes", 1.15))
    ax1g.plot(KLa_R7_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    # ax1g.set_ylim([0, 240])
    ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    ax1g.tick_params(axis="x", labelsize=11)
    ax1g.tick_params(axis="y", labelsize=11)
    ax1g.yaxis.label.set_color("tab:red")
    ax1g.spines["right"].set_color("tab:red")
    ax1g.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    # Figure 4
    fig4, ax4 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax4.set_xlabel("R7 KLa (s$^{-1}$)", fontsize=12)

    # R5 Injection
    ax4.plot(KLa_R7_list, Injection_R5_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax4.set_ylabel("R5 Injection (kg/h)", fontsize=11)
    ax4.tick_params(axis="x", labelsize=11)
    ax4.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # R6 Injection
    ax4a = ax4.twinx()
    ax4a.plot(
        KLa_R7_list,
        Injection_R6_list,
        color="tab:blue",
        label="_Organic P Concentration",
    )
    # ax4a.plot(KLa_R5_list, TP_max, color="tab:grey", linestyle='--', label='_TP Max')
    # ax4a.set_ylim([5.3, 5.52])
    ax4a.set_ylabel("R6 Injection (kg/h)", fontsize=11)
    ax4a.tick_params(axis="x", labelsize=11)
    ax4a.tick_params(axis="y", labelsize=11)
    ax4a.yaxis.label.set_color("tab:blue")
    ax4a.spines["right"].set_color("tab:blue")
    ax4a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # R7 Injection
    ax4b = ax4.twinx()
    ax4b.spines.right.set_position(("axes", 1.15))
    ax4b.plot(
        KLa_R7_list, Injection_R7_list, color="tab:red", label="_PO4 Concentration"
    )
    # ax4b.set_ylim([0, 240])
    ax4b.set_ylabel("R7 Injection (kg/h)", fontsize=11)
    ax4b.tick_params(axis="x", labelsize=11)
    ax4b.tick_params(axis="y", labelsize=11)
    ax4b.yaxis.label.set_color("tab:red")
    ax4b.spines["right"].set_color("tab:red")
    ax4b.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    plt.show(block=True)


def plot_COD_max(num):
    # 1D plot
    COD_max_list = np.linspace(0.0955, 0.0978, num)
    # COD_max_list = np.linspace(0.096, 0.1, num)
    # COD_max_list = np.linspace(0.095, 0.0975, num)

    # No electroNP flowsheet
    m, results = run_optimization_vary_max(
        COD_max=0.1,
        BOD5_max=0.01,
        TKN_max=0.007,
        TP_max=0.68,
        has_electroNP=False,
        has_optimization=True,
    )

    Ne_S_O2_out_R5 = pyo.value(m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3)
    Ne_S_O2_out_R6 = pyo.value(m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3)
    Ne_S_O2_out_R7 = pyo.value(m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3)
    Ne_S_O2_out = pyo.value(m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3)
    Ne_Ener_aeration = pyo.value(m.fs.costing.aeration_energy)
    Ne_TSS_out = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
    Ne_COD_out = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
    Ne_BOD_out = pyo.value(m.fs.Treated.properties[0].BOD5["effluent"] * 1e3)
    Ne_TKN_out = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
    Ne_SNOX_out = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
    Ne_P_org_out = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
    Ne_P_out = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)

    Ne_LCOW = pyo.value(m.fs.costing.LCOW)
    Ne_SEC = pyo.value(m.fs.costing.specific_energy_consumption)

    # O2 -- R5
    Ne_S_O2_out_R5_list = Ne_S_O2_out_R5 * np.ones(num)

    # O2 -- R6
    Ne_S_O2_out_R6_list = Ne_S_O2_out_R6 * np.ones(num)

    # O2 -- R7
    Ne_S_O2_out_R7_list = Ne_S_O2_out_R7 * np.ones(num)

    # O2 -- effluent
    Ne_S_O2_out_list = Ne_S_O2_out * np.ones(num)

    # aeration energy
    Ne_Ener_aeration_out = Ne_Ener_aeration * np.ones(num)

    # TSS
    Ne_TSS_out_list = Ne_TSS_out * np.ones(num)

    # COD
    Ne_COD_out_list = Ne_COD_out * np.ones(num)

    # BOD
    Ne_BOD_out_list = Ne_BOD_out * np.ones(num)

    # TKN
    Ne_TKN_out_list = Ne_TKN_out * np.ones(num)

    # SNOx
    Ne_SNOX_out_list = Ne_SNOX_out * np.ones(num)

    # organic P
    Ne_P_org_out_list = Ne_P_org_out * np.ones(num)

    # PO4 - inorganic P
    Ne_P_out_list = Ne_P_out * np.ones(num)

    # LCOW
    Ne_LCOW_list = Ne_LCOW * np.ones(num)

    # SEC
    Ne_SEC_list = Ne_SEC * np.ones(num)

    # electroNP flowsheet
    # O2 -- R5
    S_O2_out_R5_list = np.zeros(num)
    S_O2_out_R5_list[:] = np.nan

    # O2 -- R6
    S_O2_out_R6_list = np.zeros(num)
    S_O2_out_R6_list[:] = np.nan

    # O2 -- R7
    S_O2_out_R7_list = np.zeros(num)
    S_O2_out_R7_list[:] = np.nan

    # O2 -- effluent
    S_O2_out_list = np.zeros(num)
    S_O2_out_list[:] = np.nan

    # aeration energy
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

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
                has_electroNP=True,
                has_optimization=True,
            )

            S_O2_out_R5_list[i] = pyo.value(
                m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R6_list[i] = pyo.value(
                m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R7_list[i] = pyo.value(
                m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3
            )
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)

        except:
            pass

    S_O2_out_R5_list = interp_1d(S_O2_out_R5_list)
    S_O2_out_R6_list = interp_1d(S_O2_out_R6_list)
    S_O2_out_R7_list = interp_1d(S_O2_out_R7_list)
    S_O2_out_list = interp_1d(S_O2_out_list)
    Ener_aeration_out = interp_1d(Ener_aeration_out)
    P_removal_list = interp_1d(P_removal_list)
    TSS_out_list = interp_1d(TSS_out_list)
    COD_out_list = interp_1d(COD_out_list)
    BOD_out_list = interp_1d(BOD_out_list)
    TKN_out_list = interp_1d(TKN_out_list)
    SNOX_out_list = interp_1d(SNOX_out_list)
    P_org_out_list = interp_1d(P_org_out_list)
    P_out_list = interp_1d(P_out_list)
    LCOW_list = interp_1d(LCOW_list)
    SEC_list = interp_1d(SEC_list)

    COD_max_list = 1000 * COD_max_list

    # Figure a
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # axa.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # axa.legend(loc="lower left")

    # O2 -- R5
    axa.plot(COD_max_list, S_O2_out_R5_list, color="k", label="_R5 O2 Concentration")
    axa.plot(
        COD_max_list,
        Ne_S_O2_out_R5_list,
        color="k",
        linestyle="-.",
        label="_R5 O2 Concentration no electroNP",
    )
    # axa.set_ylim([0.86, 0.94])
    axa.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    axa.set_ylabel("R5 O2 Concentration  (mg/L)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R6
    axa1 = axa.twinx()
    axa1.plot(
        COD_max_list, S_O2_out_R6_list, color="tab:blue", label="_R6 O2 Concentration"
    )
    axa1.plot(
        COD_max_list,
        Ne_S_O2_out_R6_list,
        color="tab:blue",
        linestyle="-.",
        label="_R6 O2 Concentration no electroNP",
    )
    # axa1.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # axa1.set_ylim([45.15, 45.25])
    axa1.set_ylabel("R6 O2 Concentration (mg/L)", fontsize=11)
    axa1.tick_params(axis="x", labelsize=11)
    axa1.tick_params(axis="y", labelsize=11)
    axa1.yaxis.label.set_color("tab:blue")
    axa1.spines["right"].set_color("tab:blue")
    axa1.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R7
    axa2 = axa.twinx()
    axa2.spines.right.set_position(("axes", 1.15))
    axa2.plot(
        COD_max_list, S_O2_out_R7_list, color="tab:orange", label="_R7 O2 Concentration"
    )
    axa2.plot(
        COD_max_list,
        Ne_S_O2_out_R7_list,
        color="tab:orange",
        linestyle="-.",
        label="_R7 O2 Concentration no electroNP",
    )
    # axa2.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # axa2.set_ylim([96.44, 96.46])
    axa2.set_ylabel("R7 O2 Concentration (mg/L)", fontsize=11)
    axa2.tick_params(axis="x", labelsize=11)
    axa2.tick_params(axis="y", labelsize=11)
    axa2.yaxis.label.set_color("tab:orange")
    axa2.spines["right"].set_color("tab:orange")
    axa2.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # Figure b
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(COD_max_list, S_O2_out_list, color="k", label="O2 Concentration")
    axb.plot(
        COD_max_list,
        Ne_S_O2_out_list,
        color="k",
        linestyle="-.",
        label="O2 Concentration (no electroNP)",
    )
    # axb.set_ylim([0.86, 0.94])
    axb.set_xlim([95.62, 97.8])
    axb.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    axb.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)
    axb.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.locator_params(axis="y", nbins=8)

    # aeration energy
    axb1 = axb.twinx()
    axb1.plot(
        COD_max_list, Ener_aeration_out, color="tab:brown", label="Aeration energy"
    )
    axb1.plot(
        COD_max_list,
        Ne_S_O2_out_list,
        color="tab:brown",
        linestyle="-.",
        label="Aeration energy (no electroNP)",
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axb1.set_ylabel("Aeration energy (kWh/m$^3$)", fontsize=11)
    axb1.tick_params(axis="x", labelsize=11)
    axb1.tick_params(axis="y", labelsize=11)
    axb1.yaxis.label.set_color("tab:brown")
    axb1.spines["right"].set_color("tab:brown")
    axb1.tick_params(axis="y", colors="tab:brown")
    axb1.legend(loc="upper right", bbox_to_anchor=(1, 0.85))
    plt.locator_params(axis="y", nbins=8)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)

    # O2
    ax1.plot(COD_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    ax1.plot(
        COD_max_list,
        Ne_S_O2_out_list,
        color="k",
        linestyle="-.",
        label="_O2 Concentration no electroNP",
    )
    # axb.set_ylim([0.86, 0.94])
    ax1.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(COD_max_list, TSS_out_list, color="tab:blue", label="_TSS Concentration")
    ax1a.plot(
        COD_max_list,
        Ne_TSS_out_list,
        color="tab:blue",
        linestyle="-.",
        label="_TSS Concentration no electroNP",
    )
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(
        COD_max_list, COD_out_list, color="tab:orange", label="_COD Concentration"
    )
    ax1b.plot(
        COD_max_list,
        Ne_COD_out_list,
        color="tab:orange",
        linestyle="-.",
        label="_COD Concentration no electroNP",
    )
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # ax1b.set_ylim([96.2, 96.7])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(
        COD_max_list, BOD_out_list, color="tab:purple", label="_BOD Concentration"
    )
    ax1c.plot(
        COD_max_list,
        Ne_BOD_out_list,
        color="tab:purple",
        linestyle="-.",
        label="_BOD Concentration no electroNP",
    )
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    # ax1c.set_ylim([6.054, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax2.set_xlabel("COD Max Concentration (mg/L)", fontsize=12)

    # O2
    ax2.plot(COD_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    ax2.plot(
        COD_max_list,
        Ne_S_O2_out_list,
        color="k",
        linestyle="-.",
        label="_O2 Concentration no electroNP",
    )
    # axb.set_ylim([0.86, 0.94])
    ax2.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(COD_max_list, TKN_out_list, color="tab:brown", label="_TKN Concentration")
    ax1d.plot(
        COD_max_list,
        Ne_TKN_out_list,
        color="tab:brown",
        linestyle="-.",
        label="_TKN Concentration no electroNP",
    )
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # ax1d.set_ylim([6.7, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(
        COD_max_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration"
    )
    ax1e.plot(
        COD_max_list,
        Ne_SNOX_out_list,
        color="tab:green",
        linestyle="-.",
        label="_SNOX Concentration no electroNP",
    )
    # ax1e.set_ylim([2.95, 3.5])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # # Figure 3
    # fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")
    # ax3.set_xlabel("COD Max Concentration (mg/L)", fontsize=12)
    #
    # # O2
    # ax3.plot(COD_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # ax3.plot(
    #     COD_max_list,
    #     Ne_S_O2_out_list,
    #     color="k",
    #     linestyle="-.",
    #     label="_O2 Concentration no electroNP",
    # )
    # # axb.set_ylim([0.86, 0.94])
    # ax3.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    # ax3.tick_params(axis="x", labelsize=11)
    # ax3.tick_params(axis="y", labelsize=11)
    # plt.locator_params(axis="y", nbins=8)
    #
    # # organic P
    # ax1f = ax3.twinx()
    # ax1f.plot(
    #     COD_max_list, P_org_out_list, color="tab:pink", label="_Organic P Concentration"
    # )
    # ax1f.plot(
    #     COD_max_list,
    #     Ne_P_org_out_list,
    #     color="tab:pink",
    #     linestyle="-.",
    #     label="_Organic P no electroNP",
    # )
    # TP_max = 5 * np.ones(num)
    # ax1f.plot(COD_max_list, TP_max, color="tab:grey", linestyle="--", label="_TP Max")
    # # ax1f.set_ylim([5.3, 5.52])
    # ax1f.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    # ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    # ax1f.tick_params(axis="x", labelsize=11)
    # ax1f.tick_params(axis="y", labelsize=11)
    # ax1f.yaxis.label.set_color("tab:pink")
    # ax1f.spines["right"].set_color("tab:pink")
    # ax1f.tick_params(axis="y", colors="tab:pink")
    # ax1f.legend()
    # plt.locator_params(axis="y", nbins=8)
    #
    # # PO4
    # ax1g = ax3.twinx()
    # ax1g.spines.right.set_position(("axes", 1.15))
    # # ax1g = brokenaxes(ylims=((0, 10), (700, 800)), hspace=0.25)
    # ax1g.plot(COD_max_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    # ax1g.plot(
    #     COD_max_list,
    #     Ne_P_out_list,
    #     color="tab:red",
    #     linestyle="-.",
    #     label="_PO4 no electroNP",
    # )
    # # ax1g.set_ylim([0, 240])
    # ax1g.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    # ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    # ax1g.tick_params(axis="x", labelsize=11)
    # ax1g.tick_params(axis="y", labelsize=11)
    # ax1g.yaxis.label.set_color("tab:red")
    # ax1g.spines["right"].set_color("tab:red")
    # ax1g.tick_params(axis="y", colors="tab:red")
    # plt.locator_params(axis="y", nbins=8)

    # Figure 3 - v3
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    plt.gca().axes.get_yaxis().set_visible(False)
    ax3.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)

    bax = brokenaxes(ylims=((0, 1), (675, 676)), hspace=0.15)
    bax.plot(COD_max_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    bax.plot(
        COD_max_list,
        Ne_P_out_list,
        color="tab:red",
        linestyle="-.",
        label="_PO4 no electroNP",
    )
    # bax.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    bax.set_ylabel(
        "PO4 Concentration (mg/L)", fontsize=11, color="tab:red", labelpad=40
    )
    # bax.tick_params(axis="x", labelsize=11)
    bax.tick_params(axis="y", labelsize=11)
    ax3.spines["left"].set_color("tab:red")
    bax.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    ax3i = ax3.twinx()
    ax3i.plot(
        COD_max_list, P_org_out_list, color="tab:pink", label="_Organic P Concentration"
    )
    ax3i.plot(
        COD_max_list,
        Ne_P_org_out_list,
        color="tab:pink",
        linestyle="-.",
        label="_Organic P no electroNP",
    )
    ax3i.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax3i.tick_params(axis="y", labelsize=11)
    ax3i.yaxis.label.set_color("tab:pink")
    ax3i.spines["right"].set_color("tab:pink")
    ax3i.tick_params(axis="y", colors="tab:pink")
    plt.locator_params(axis="y", nbins=8)
    ax3i.spines["left"].set_color("tab:red")

    # Figure i
    figi, axi = plt.subplots(figsize=(9, 5), layout="constrained")
    axi.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    axi.plot(COD_max_list, LCOW_list, color="k", label="LCOW")
    axi.plot(
        COD_max_list,
        Ne_LCOW_list,
        color="k",
        linestyle="-.",
        label="LCOW (no electroNP)",
    )
    axi.set_ylim([0.6, 0.9])
    axi.set_xlim([95.62, 97.8])
    axi.set_ylabel("Levelized Cost of Water (/$/m$^3$ (2023))", fontsize=11)
    axi.tick_params(axis="x", labelsize=11)
    axi.tick_params(axis="y", labelsize=11)
    axi.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.locator_params(axis="y", nbins=8)

    # SEC
    axi1 = axi.twinx()
    axi1.plot(COD_max_list, SEC_list, color="tab:blue", label="SEC")
    axi1.plot(
        COD_max_list,
        Ne_SEC_list,
        color="tab:blue",
        linestyle="-.",
        label="SEC (no electroNP)",
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    axb1.set_ylim([0, 3])
    axi1.set_ylabel("Specific Energy Consumption (kWh/m$^3$)", fontsize=11)
    axi1.tick_params(axis="x", labelsize=11)
    axi1.tick_params(axis="y", labelsize=11)
    axi1.yaxis.label.set_color("tab:blue")
    axi1.spines["right"].set_color("tab:blue")
    axi1.tick_params(axis="y", colors="tab:blue")
    axi1.legend(loc="upper right", bbox_to_anchor=(1, 0.85))
    plt.locator_params(axis="y", nbins=8)

    plt.show(block=True)


def plot_COD_max_no_electroNP(num):
    # 1D plot
    COD_max_list = np.linspace(0.97, 0.1, num)
    # COD_max_list = np.linspace(0.095, 0.0975, num)

    # O2 -- R5
    S_O2_out_R5_list = np.zeros(num)
    S_O2_out_R5_list[:] = np.nan

    # O2 -- R6
    S_O2_out_R6_list = np.zeros(num)
    S_O2_out_R6_list[:] = np.nan

    # O2 -- R7
    S_O2_out_R7_list = np.zeros(num)
    S_O2_out_R7_list[:] = np.nan

    # O2 -- effluent
    S_O2_out_list = np.zeros(num)
    S_O2_out_list[:] = np.nan

    # aeration energy
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

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
                has_electroNP=False,
                has_optimization=True,
            )

            S_O2_out_R5_list[i] = pyo.value(
                m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R6_list[i] = pyo.value(
                m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R7_list[i] = pyo.value(
                m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3
            )
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)

        except:
            pass

    # S_O2_out_R5_list = interp_1d(S_O2_out_R5_list)
    # S_O2_out_R6_list = interp_1d(S_O2_out_R6_list)
    # S_O2_out_R7_list = interp_1d(S_O2_out_R7_list)
    # S_O2_out_list = interp_1d(S_O2_out_list)
    # Ener_aeration_out = interp_1d(Ener_aeration_out)
    # P_removal_list = interp_1d(P_removal_list)
    # TSS_out_list = interp_1d(TSS_out_list)
    # COD_out_list = interp_1d(COD_out_list)
    # BOD_out_list = interp_1d(BOD_out_list)
    # TKN_out_list = interp_1d(TKN_out_list)
    # SNOX_out_list = interp_1d(SNOX_out_list)
    # P_org_out_list = interp_1d(P_org_out_list)
    # P_out_list = interp_1d(P_out_list)
    # LCOW_list = interp_1d(LCOW_list)
    # SEC_list = interp_1d(SEC_list)

    COD_max_list = 1000 * COD_max_list

    # Figure a
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # axa.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # axa.legend(loc="lower left")

    # O2 -- R5
    axa.plot(COD_max_list, S_O2_out_R5_list, color="k", label="_R5 O2 Concentration")
    # axa.set_ylim([0.86, 0.94])
    axa.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    axa.set_ylabel("R5 O2 Concentration  (mg/L)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R6
    axa1 = axa.twinx()
    axa1.plot(
        COD_max_list, S_O2_out_R6_list, color="tab:blue", label="_R6 O2 Concentration"
    )
    # axa1.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # axa1.set_ylim([45.15, 45.25])
    axa1.set_ylabel("R6 O2 Concentration (mg/L)", fontsize=11)
    axa1.tick_params(axis="x", labelsize=11)
    axa1.tick_params(axis="y", labelsize=11)
    axa1.yaxis.label.set_color("tab:blue")
    axa1.spines["right"].set_color("tab:blue")
    axa1.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R7
    axa2 = axa.twinx()
    axa2.spines.right.set_position(("axes", 1.15))
    axa2.plot(
        COD_max_list, S_O2_out_R7_list, color="tab:orange", label="_R7 O2 Concentration"
    )
    # axa2.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # axa2.set_ylim([96.44, 96.46])
    axa2.set_ylabel("R7 O2 Concentration (mg/L)", fontsize=11)
    axa2.tick_params(axis="x", labelsize=11)
    axa2.tick_params(axis="y", labelsize=11)
    axa2.yaxis.label.set_color("tab:orange")
    axa2.spines["right"].set_color("tab:orange")
    axa2.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # Figure b
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(COD_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    axb.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    axb.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # aeration energy
    axb1 = axb.twinx()
    axb1.plot(
        COD_max_list, Ener_aeration_out, color="tab:brown", label="_Aeration energy"
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axb1.set_ylabel("Aeration energy (kWh/m$^3$)", fontsize=11)
    axb1.tick_params(axis="x", labelsize=11)
    axb1.tick_params(axis="y", labelsize=11)
    axb1.yaxis.label.set_color("tab:brown")
    axb1.spines["right"].set_color("tab:brown")
    axb1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)

    # O2
    ax1.plot(COD_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax1.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(COD_max_list, TSS_out_list, color="tab:blue", label="_TSS Concentration")
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(
        COD_max_list, COD_out_list, color="tab:orange", label="_COD Concentration"
    )
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # ax1b.set_ylim([96.2, 96.7])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(
        COD_max_list, BOD_out_list, color="tab:purple", label="_BOD Concentration"
    )
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    # ax1c.set_ylim([6.054, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax2.set_xlabel("COD Max Concentration (mg/L)", fontsize=12)

    # O2
    ax2.plot(COD_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax2.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(COD_max_list, TKN_out_list, color="tab:brown", label="_TKN Concentration")
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # ax1d.set_ylim([6.7, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(
        COD_max_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration"
    )
    # ax1e.set_ylim([2.95, 3.5])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # Figure 3
    fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax3.set_xlabel("COD Max Concentration (mg/L)", fontsize=12)

    # O2
    ax3.plot(COD_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax3.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax3.tick_params(axis="x", labelsize=11)
    ax3.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # organic P
    ax1f = ax3.twinx()
    ax1f.plot(
        COD_max_list, P_org_out_list, color="tab:pink", label="_Organic P Concentration"
    )
    TP_max = 5 * np.ones(num)
    ax1f.plot(COD_max_list, TP_max, color="tab:grey", linestyle="--", label="_TP Max")
    # ax1f.set_ylim([5.3, 5.52])
    ax1f.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax1f.tick_params(axis="x", labelsize=11)
    ax1f.tick_params(axis="y", labelsize=11)
    ax1f.yaxis.label.set_color("tab:pink")
    ax1f.spines["right"].set_color("tab:pink")
    ax1f.tick_params(axis="y", colors="tab:pink")
    ax1f.legend()
    plt.locator_params(axis="y", nbins=8)

    # PO4
    ax1g = ax3.twinx()
    ax1g.spines.right.set_position(("axes", 1.15))
    ax1g.plot(COD_max_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    # ax1g.set_ylim([0, 240])
    ax1g.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    ax1g.tick_params(axis="x", labelsize=11)
    ax1g.tick_params(axis="y", labelsize=11)
    ax1g.yaxis.label.set_color("tab:red")
    ax1g.spines["right"].set_color("tab:red")
    ax1g.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    # Figure i
    figi, axi = plt.subplots(figsize=(9, 5), layout="constrained")
    axi.plot(COD_max_list, LCOW_list, color="k", label="_LCOW")
    # axb.set_ylim([0.86, 0.94])
    axi.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    axi.set_ylabel("Levelized Cost of Water (/$/m$^3$ (2023))", fontsize=11)
    axi.tick_params(axis="x", labelsize=11)
    axi.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # SEC
    axi1 = axi.twinx()
    axi1.plot(COD_max_list, SEC_list, color="tab:brown", label="_SEC")
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axi1.set_ylabel("Specific Energy Consumption (kWh/m$^3$)", fontsize=11)
    axi1.tick_params(axis="x", labelsize=11)
    axi1.tick_params(axis="y", labelsize=11)
    axi1.yaxis.label.set_color("tab:brown")
    axi1.spines["right"].set_color("tab:brown")
    axi1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    plt.show(block=True)


def plot_BOD5_max(num):
    # 1D plot
    BOD5_max_list = np.linspace(0.006, 0.0075, num)
    # BOD5_max_list = np.linspace(0.006, 0.0065, num)

    # No electroNP flowsheet
    m, results = run_optimization_vary_max(
        COD_max=0.1,
        BOD5_max=0.01,
        TKN_max=0.007,
        TP_max=0.68,
        has_electroNP=False,
        has_optimization=True,
    )

    Ne_S_O2_out_R5 = pyo.value(m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3)
    Ne_S_O2_out_R6 = pyo.value(m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3)
    Ne_S_O2_out_R7 = pyo.value(m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3)
    Ne_S_O2_out = pyo.value(m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3)
    Ne_Ener_aeration = pyo.value(m.fs.costing.aeration_energy)
    Ne_TSS_out = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
    Ne_COD_out = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
    Ne_BOD_out = pyo.value(m.fs.Treated.properties[0].BOD5["effluent"] * 1e3)
    Ne_TKN_out = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
    Ne_SNOX_out = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
    Ne_P_org_out = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
    Ne_P_out = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)

    Ne_LCOW = pyo.value(m.fs.costing.LCOW)
    Ne_SEC = pyo.value(m.fs.costing.specific_energy_consumption)

    # O2 -- R5
    Ne_S_O2_out_R5_list = Ne_S_O2_out_R5 * np.ones(num)

    # O2 -- R6
    Ne_S_O2_out_R6_list = Ne_S_O2_out_R6 * np.ones(num)

    # O2 -- R7
    Ne_S_O2_out_R7_list = Ne_S_O2_out_R7 * np.ones(num)

    # O2 -- effluent
    Ne_S_O2_out_list = Ne_S_O2_out * np.ones(num)

    # aeration energy
    Ne_Ener_aeration_out = Ne_Ener_aeration * np.ones(num)

    # TSS
    Ne_TSS_out_list = Ne_TSS_out * np.ones(num)

    # COD
    Ne_COD_out_list = Ne_COD_out * np.ones(num)

    # BOD
    Ne_BOD_out_list = Ne_BOD_out * np.ones(num)

    # TKN
    Ne_TKN_out_list = Ne_TKN_out * np.ones(num)

    # SNOx
    Ne_SNOX_out_list = Ne_SNOX_out * np.ones(num)

    # organic P
    Ne_P_org_out_list = Ne_P_org_out * np.ones(num)

    # PO4 - inorganic P
    Ne_P_out_list = Ne_P_out * np.ones(num)

    # LCOW
    Ne_LCOW_list = Ne_LCOW * np.ones(num)

    # SEC
    Ne_SEC_list = Ne_SEC * np.ones(num)

    # electroNP
    # O2 -- R5
    S_O2_out_R5_list = np.zeros(num)
    S_O2_out_R5_list[:] = np.nan

    # O2 -- R6
    S_O2_out_R6_list = np.zeros(num)
    S_O2_out_R6_list[:] = np.nan

    # O2 -- R7
    S_O2_out_R7_list = np.zeros(num)
    S_O2_out_R7_list[:] = np.nan

    # O2 -- effluent
    S_O2_out_list = np.zeros(num)
    S_O2_out_list[:] = np.nan

    # aeration energy
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

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
                has_electroNP=True,
                has_optimization=True,
            )

            S_O2_out_R5_list[i] = pyo.value(
                m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R6_list[i] = pyo.value(
                m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R7_list[i] = pyo.value(
                m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3
            )
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)

        except:
            pass

    S_O2_out_R5_list = interp_1d(S_O2_out_R5_list)
    S_O2_out_R6_list = interp_1d(S_O2_out_R6_list)
    S_O2_out_R7_list = interp_1d(S_O2_out_R7_list)
    S_O2_out_list = interp_1d(S_O2_out_list)
    Ener_aeration_out = interp_1d(Ener_aeration_out)
    P_removal_list = interp_1d(P_removal_list)
    TSS_out_list = interp_1d(TSS_out_list)
    COD_out_list = interp_1d(COD_out_list)
    BOD_out_list = interp_1d(BOD_out_list)
    TKN_out_list = interp_1d(TKN_out_list)
    SNOX_out_list = interp_1d(SNOX_out_list)
    P_org_out_list = interp_1d(P_org_out_list)
    P_out_list = interp_1d(P_out_list)
    LCOW_list = interp_1d(LCOW_list)
    SEC_list = interp_1d(SEC_list)

    BOD5_max_list = 1000 * BOD5_max_list

    # Figure a
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # axa.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # axa.legend(loc="lower left")

    # O2 -- R5
    axa.plot(BOD5_max_list, S_O2_out_R5_list, color="k", label="_R5 O2 Concentration")
    axa.plot(
        BOD5_max_list,
        Ne_S_O2_out_R5_list,
        color="k",
        linestyle="-.",
        label="_R5 O2 Concentration no electroNP",
    )
    # axa.set_ylim([0.86, 0.94])
    axa.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)
    axa.set_ylabel("R5 O2 Concentration  (mg/L)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R6
    axa1 = axa.twinx()
    axa1.plot(
        BOD5_max_list, S_O2_out_R6_list, color="tab:blue", label="_R6 O2 Concentration"
    )
    axa1.plot(
        BOD5_max_list,
        Ne_S_O2_out_R6_list,
        color="tab:blue",
        linestyle="-.",
        label="_R6 O2 Concentration no electroNP",
    )
    # axa1.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # axa1.set_ylim([45.15, 45.25])
    axa1.set_ylabel("R6 O2 Concentration (mg/L)", fontsize=11)
    axa1.tick_params(axis="x", labelsize=11)
    axa1.tick_params(axis="y", labelsize=11)
    axa1.yaxis.label.set_color("tab:blue")
    axa1.spines["right"].set_color("tab:blue")
    axa1.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R7
    axa2 = axa.twinx()
    axa2.spines.right.set_position(("axes", 1.15))
    axa2.plot(
        BOD5_max_list,
        S_O2_out_R7_list,
        color="tab:orange",
        label="_R7 O2 Concentration",
    )
    axa2.plot(
        BOD5_max_list,
        Ne_S_O2_out_R7_list,
        color="tab:orange",
        linestyle="-.",
        label="_R7 O2 Concentration no electroNP",
    )
    # axa2.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # axa2.set_ylim([96.44, 96.46])
    axa2.set_ylabel("R7 O2 Concentration (mg/L)", fontsize=11)
    axa2.tick_params(axis="x", labelsize=11)
    axa2.tick_params(axis="y", labelsize=11)
    axa2.yaxis.label.set_color("tab:orange")
    axa2.spines["right"].set_color("tab:orange")
    axa2.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # Figure b
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(BOD5_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    axb.plot(
        BOD5_max_list,
        Ne_S_O2_out_list,
        color="k",
        linestyle="-.",
        label="_O2 Concentration no electroNP",
    )
    # axb.set_ylim([0.86, 0.94])
    axb.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)
    axb.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # aeration energy
    axb1 = axb.twinx()
    axb1.plot(
        BOD5_max_list, Ener_aeration_out, color="tab:brown", label="_Aeration energy"
    )
    axb1.plot(
        BOD5_max_list,
        Ne_Ener_aeration_out,
        color="tab:brown",
        linestyle="-.",
        label="_Aeration energy no electroNP",
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axb1.set_ylabel("Aeration energy (kWh/m$^3$)", fontsize=11)
    axb1.tick_params(axis="x", labelsize=11)
    axb1.tick_params(axis="y", labelsize=11)
    axb1.yaxis.label.set_color("tab:brown")
    axb1.spines["right"].set_color("tab:brown")
    axb1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)

    # O2
    ax1.plot(BOD5_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    ax1.plot(
        BOD5_max_list,
        Ne_S_O2_out_list,
        color="k",
        linestyle="-.",
        label="_O2 Concentration no electroNP",
    )
    # axb.set_ylim([0.86, 0.94])
    ax1.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(BOD5_max_list, TSS_out_list, color="tab:blue", label="_TSS Concentration")
    ax1a.plot(
        BOD5_max_list,
        Ne_TSS_out_list,
        color="tab:blue",
        linestyle="-.",
        label="_TSS Concentration no electroNP",
    )
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(
        BOD5_max_list, COD_out_list, color="tab:orange", label="_COD Concentration"
    )
    ax1b.plot(
        BOD5_max_list,
        Ne_COD_out_list,
        color="tab:orange",
        linestyle="-.",
        label="_COD Concentration no electroNP",
    )
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # ax1b.set_ylim([96.2, 96.7])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(
        BOD5_max_list, BOD_out_list, color="tab:purple", label="_BOD Concentration"
    )
    ax1c.plot(
        BOD5_max_list,
        Ne_BOD_out_list,
        color="tab:purple",
        linestyle="-.",
        label="_BOD Concentration no electroNP",
    )
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    # ax1c.set_ylim([6.054, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax2.set_xlabel("BOD Max Concentration (mg/L)", fontsize=12)

    # O2
    ax2.plot(BOD5_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    ax2.plot(
        BOD5_max_list,
        Ne_S_O2_out_list,
        color="k",
        linestyle="-.",
        label="_O2 Concentration no electroNP",
    )
    # axb.set_ylim([0.86, 0.94])
    ax2.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(
        BOD5_max_list, TKN_out_list, color="tab:brown", label="_TKN Concentration"
    )
    ax1d.plot(
        BOD5_max_list,
        Ne_TKN_out_list,
        color="tab:brown",
        linestyle="-.",
        label="_TKN Concentration no electroNP",
    )
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # ax1d.set_ylim([6.7, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(
        BOD5_max_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration"
    )
    ax1e.plot(
        BOD5_max_list,
        Ne_SNOX_out_list,
        color="tab:green",
        linestyle="-.",
        label="_SNOX Concentration no electroNP",
    )
    # ax1e.set_ylim([2.95, 3.5])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # # Figure 3
    # fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")
    # ax3.set_xlabel("BOD Max Concentration (mg/L)", fontsize=12)
    #
    # # O2
    # ax3.plot(BOD5_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # ax3.plot(
    #     BOD5_max_list,
    #     Ne_S_O2_out_list,
    #     color="k",
    #     linestyle="-.",
    #     label="_O2 Concentration no electroNP",
    # )
    # # axb.set_ylim([0.86, 0.94])
    # ax3.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    # ax3.tick_params(axis="x", labelsize=11)
    # ax3.tick_params(axis="y", labelsize=11)
    # plt.locator_params(axis="y", nbins=8)
    #
    # # organic P
    # ax1f = ax3.twinx()
    # ax1f.plot(
    #     BOD5_max_list,
    #     P_org_out_list,
    #     color="tab:pink",
    #     label="_Organic P Concentration",
    # )
    # ax1f.plot(
    #     BOD5_max_list,
    #     Ne_P_org_out_list,
    #     color="tab:pink",
    #     linestyle="-.",
    #     label="_Organic P Concentration no electroNP",
    # )
    # TP_max = 5 * np.ones(num)
    # ax1f.plot(BOD5_max_list, TP_max, color="tab:grey", linestyle="--", label="TP Max")
    # # ax1f.set_ylim([5.3, 5.52])
    # ax1f.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)
    # ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    # ax1f.tick_params(axis="x", labelsize=11)
    # ax1f.tick_params(axis="y", labelsize=11)
    # ax1f.yaxis.label.set_color("tab:pink")
    # ax1f.spines["right"].set_color("tab:pink")
    # ax1f.tick_params(axis="y", colors="tab:pink")
    # ax1f.legend()
    # plt.locator_params(axis="y", nbins=8)
    #
    # # PO4
    # ax1g = ax3.twinx()
    # ax1g.spines.right.set_position(("axes", 1.15))
    # ax1g.plot(BOD5_max_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    # ax1g.plot(
    #     BOD5_max_list,
    #     Ne_P_out_list,
    #     color="tab:red",
    #     linestyle="-.",
    #     label="_PO4 Concentration no electroNP",
    # )
    #
    # # ax1g.set_ylim([0, 240])
    # ax1g.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)
    # ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    # ax1g.tick_params(axis="x", labelsize=11)
    # ax1g.tick_params(axis="y", labelsize=11)
    # ax1g.yaxis.label.set_color("tab:red")
    # ax1g.spines["right"].set_color("tab:red")
    # ax1g.tick_params(axis="y", colors="tab:red")
    # plt.locator_params(axis="y", nbins=8)

    # Figure 3 - v3
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    plt.gca().axes.get_yaxis().set_visible(False)
    ax3.set_xlabel("BOD5 Max Concentration (mg/L)", fontsize=11)

    bax = brokenaxes(ylims=((0, 1), (675, 676)), hspace=0.15)
    bax.plot(BOD5_max_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    bax.plot(
        BOD5_max_list,
        Ne_P_out_list,
        color="tab:red",
        linestyle="-.",
        label="_PO4 no electroNP",
    )
    # bax.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    bax.set_ylabel(
        "PO4 Concentration (mg/L)", fontsize=11, color="tab:red", labelpad=40
    )
    # bax.tick_params(axis="x", labelsize=11)
    bax.tick_params(axis="y", labelsize=11)
    ax3.spines["left"].set_color("tab:red")
    bax.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    ax3i = ax3.twinx()
    ax3i.plot(
        BOD5_max_list,
        P_org_out_list,
        color="tab:pink",
        label="_Organic P Concentration",
    )
    ax3i.plot(
        BOD5_max_list,
        Ne_P_org_out_list,
        color="tab:pink",
        linestyle="-.",
        label="_Organic P no electroNP",
    )
    ax3i.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax3i.tick_params(axis="y", labelsize=11)
    ax3i.yaxis.label.set_color("tab:pink")
    ax3i.spines["right"].set_color("tab:pink")
    ax3i.tick_params(axis="y", colors="tab:pink")
    plt.locator_params(axis="y", nbins=8)
    ax3i.spines["left"].set_color("tab:red")

    # Figure i
    figi, axi = plt.subplots(figsize=(9, 5), layout="constrained")
    axi.plot(BOD5_max_list, LCOW_list, color="k", label="_LCOW")
    axi.plot(
        BOD5_max_list,
        Ne_LCOW_list,
        color="k",
        linestyle="-.",
        label="_LCOW no electroNP",
    )
    # axb.set_ylim([0.86, 0.94])
    axi.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)
    axi.set_ylabel("Levelized Cost of Water (/$/m$^3$ (2023))", fontsize=11)
    axi.tick_params(axis="x", labelsize=11)
    axi.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # SEC
    axi1 = axi.twinx()
    axi1.plot(BOD5_max_list, SEC_list, color="tab:brown", label="_SEC")
    axi1.plot(
        BOD5_max_list,
        Ne_SEC_list,
        color="tab:brown",
        linestyle="-.",
        label="_SEC no electroNP",
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axi1.set_ylabel("Specific Energy Consumption (kWh/m$^3$)", fontsize=11)
    axi1.tick_params(axis="x", labelsize=11)
    axi1.tick_params(axis="y", labelsize=11)
    axi1.yaxis.label.set_color("tab:brown")
    axi1.spines["right"].set_color("tab:brown")
    axi1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    plt.show(block=True)


def plot_BOD5_max_no_electroNP(num):
    # 1D plot
    BOD5_max_list = np.linspace(0.007, 0.01, num)

    # O2 -- R5
    S_O2_out_R5_list = np.zeros(num)
    S_O2_out_R5_list[:] = np.nan

    # O2 -- R6
    S_O2_out_R6_list = np.zeros(num)
    S_O2_out_R6_list[:] = np.nan

    # O2 -- R7
    S_O2_out_R7_list = np.zeros(num)
    S_O2_out_R7_list[:] = np.nan

    # O2 -- effluent
    S_O2_out_list = np.zeros(num)
    S_O2_out_list[:] = np.nan

    # aeration energy
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

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
                has_electroNP=False,
                has_optimization=True,
            )

            S_O2_out_R5_list[i] = pyo.value(
                m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R6_list[i] = pyo.value(
                m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R7_list[i] = pyo.value(
                m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3
            )
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)

        except:
            pass

    # S_O2_out_R5_list = interp_1d(S_O2_out_R5_list)
    # S_O2_out_R6_list = interp_1d(S_O2_out_R6_list)
    # S_O2_out_R7_list = interp_1d(S_O2_out_R7_list)
    # S_O2_out_list = interp_1d(S_O2_out_list)
    # Ener_aeration_out = interp_1d(Ener_aeration_out)
    # P_removal_list = interp_1d(P_removal_list)
    # TSS_out_list = interp_1d(TSS_out_list)
    # COD_out_list = interp_1d(COD_out_list)
    # BOD_out_list = interp_1d(BOD_out_list)
    # TKN_out_list = interp_1d(TKN_out_list)
    # SNOX_out_list = interp_1d(SNOX_out_list)
    # P_org_out_list = interp_1d(P_org_out_list)
    # P_out_list = interp_1d(P_out_list)
    # LCOW_list = interp_1d(LCOW_list)
    # SEC_list = interp_1d(SEC_list)

    BOD5_max_list = 1000 * BOD5_max_list

    # Figure a
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # axa.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # axa.legend(loc="lower left")

    # O2 -- R5
    axa.plot(BOD5_max_list, S_O2_out_R5_list, color="k", label="_R5 O2 Concentration")
    # axa.set_ylim([0.86, 0.94])
    axa.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)
    axa.set_ylabel("R5 O2 Concentration  (mg/L)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R6
    axa1 = axa.twinx()
    axa1.plot(
        BOD5_max_list, S_O2_out_R6_list, color="tab:blue", label="_R6 O2 Concentration"
    )
    # axa1.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # axa1.set_ylim([45.15, 45.25])
    axa1.set_ylabel("R6 O2 Concentration (mg/L)", fontsize=11)
    axa1.tick_params(axis="x", labelsize=11)
    axa1.tick_params(axis="y", labelsize=11)
    axa1.yaxis.label.set_color("tab:blue")
    axa1.spines["right"].set_color("tab:blue")
    axa1.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R7
    axa2 = axa.twinx()
    axa2.spines.right.set_position(("axes", 1.15))
    axa2.plot(
        BOD5_max_list,
        S_O2_out_R7_list,
        color="tab:orange",
        label="_R7 O2 Concentration",
    )
    # axa2.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # axa2.set_ylim([96.44, 96.46])
    axa2.set_ylabel("R7 O2 Concentration (mg/L)", fontsize=11)
    axa2.tick_params(axis="x", labelsize=11)
    axa2.tick_params(axis="y", labelsize=11)
    axa2.yaxis.label.set_color("tab:orange")
    axa2.spines["right"].set_color("tab:orange")
    axa2.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # Figure b
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(BOD5_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    axb.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)
    axb.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # aeration energy
    axb1 = axb.twinx()
    axb1.plot(
        BOD5_max_list, Ener_aeration_out, color="tab:brown", label="_Aeration energy"
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axb1.set_ylabel("Aeration energy (kWh/m$^3$)", fontsize=11)
    axb1.tick_params(axis="x", labelsize=11)
    axb1.tick_params(axis="y", labelsize=11)
    axb1.yaxis.label.set_color("tab:brown")
    axb1.spines["right"].set_color("tab:brown")
    axb1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)

    # O2
    ax1.plot(BOD5_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax1.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(BOD5_max_list, TSS_out_list, color="tab:blue", label="_TSS Concentration")
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(
        BOD5_max_list, COD_out_list, color="tab:orange", label="_COD Concentration"
    )
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # ax1b.set_ylim([96.2, 96.7])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(
        BOD5_max_list, BOD_out_list, color="tab:purple", label="_BOD Concentration"
    )
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    # ax1c.set_ylim([6.054, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax2.set_xlabel("BOD Max Concentration (mg/L)", fontsize=12)

    # O2
    ax2.plot(BOD5_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax2.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(
        BOD5_max_list, TKN_out_list, color="tab:brown", label="_TKN Concentration"
    )
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # ax1d.set_ylim([6.7, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(
        BOD5_max_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration"
    )
    # ax1e.set_ylim([2.95, 3.5])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # Figure 3
    fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax3.set_xlabel("BOD Max Concentration (mg/L)", fontsize=12)

    # O2
    ax3.plot(BOD5_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax3.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax3.tick_params(axis="x", labelsize=11)
    ax3.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # organic P
    ax1f = ax3.twinx()
    ax1f.plot(
        BOD5_max_list,
        P_org_out_list,
        color="tab:pink",
        label="_Organic P Concentration",
    )
    TP_max = 5 * np.ones(num)
    ax1f.plot(BOD5_max_list, TP_max, color="tab:grey", linestyle="--", label="TP Max")
    # ax1f.set_ylim([5.3, 5.52])
    ax1f.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)
    ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax1f.tick_params(axis="x", labelsize=11)
    ax1f.tick_params(axis="y", labelsize=11)
    ax1f.yaxis.label.set_color("tab:pink")
    ax1f.spines["right"].set_color("tab:pink")
    ax1f.tick_params(axis="y", colors="tab:pink")
    ax1f.legend()
    plt.locator_params(axis="y", nbins=8)

    # PO4
    ax1g = ax3.twinx()
    ax1g.spines.right.set_position(("axes", 1.15))
    ax1g.plot(BOD5_max_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    # ax1g.set_ylim([0, 240])
    ax1g.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)
    ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    ax1g.tick_params(axis="x", labelsize=11)
    ax1g.tick_params(axis="y", labelsize=11)
    ax1g.yaxis.label.set_color("tab:red")
    ax1g.spines["right"].set_color("tab:red")
    ax1g.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    # Figure i
    figi, axi = plt.subplots(figsize=(9, 5), layout="constrained")
    axi.plot(BOD5_max_list, LCOW_list, color="k", label="_LCOW")
    # axb.set_ylim([0.86, 0.94])
    axi.set_xlabel("BOD Max Concentration (mg/L)", fontsize=11)
    axi.set_ylabel("Levelized Cost of Water (/$/m$^3$ (2023))", fontsize=11)
    axi.tick_params(axis="x", labelsize=11)
    axi.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # SEC
    axi1 = axi.twinx()
    axi1.plot(BOD5_max_list, SEC_list, color="tab:brown", label="_SEC")
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axi1.set_ylabel("Specific Energy Consumption (kWh/m$^3$)", fontsize=11)
    axi1.tick_params(axis="x", labelsize=11)
    axi1.tick_params(axis="y", labelsize=11)
    axi1.yaxis.label.set_color("tab:brown")
    axi1.spines["right"].set_color("tab:brown")
    axi1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    plt.show(block=True)


def plot_TKN_max(num):
    # 1D plot
    TKN_max_list = np.linspace(0.0066, 0.008, num)

    # No electroNP flowsheet
    m, results = run_optimization_vary_max(
        COD_max=0.1,
        BOD5_max=0.01,
        TKN_max=0.007,
        TP_max=0.68,
        has_electroNP=False,
        has_optimization=True,
    )

    Ne_S_O2_out_R5 = pyo.value(m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3)
    Ne_S_O2_out_R6 = pyo.value(m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3)
    Ne_S_O2_out_R7 = pyo.value(m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3)
    Ne_S_O2_out = pyo.value(m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3)
    Ne_Ener_aeration = pyo.value(m.fs.costing.aeration_energy)
    Ne_TSS_out = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
    Ne_COD_out = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
    Ne_BOD_out = pyo.value(m.fs.Treated.properties[0].BOD5["effluent"] * 1e3)
    Ne_TKN_out = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
    Ne_SNOX_out = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
    Ne_P_org_out = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
    Ne_P_out = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)

    Ne_LCOW = pyo.value(m.fs.costing.LCOW)
    Ne_SEC = pyo.value(m.fs.costing.specific_energy_consumption)

    # O2 -- R5
    Ne_S_O2_out_R5_list = Ne_S_O2_out_R5 * np.ones(num)

    # O2 -- R6
    Ne_S_O2_out_R6_list = Ne_S_O2_out_R6 * np.ones(num)

    # O2 -- R7
    Ne_S_O2_out_R7_list = Ne_S_O2_out_R7 * np.ones(num)

    # O2 -- effluent
    Ne_S_O2_out_list = Ne_S_O2_out * np.ones(num)

    # aeration energy
    Ne_Ener_aeration_out = Ne_Ener_aeration * np.ones(num)

    # TSS
    Ne_TSS_out_list = Ne_TSS_out * np.ones(num)

    # COD
    Ne_COD_out_list = Ne_COD_out * np.ones(num)

    # BOD
    Ne_BOD_out_list = Ne_BOD_out * np.ones(num)

    # TKN
    Ne_TKN_out_list = Ne_TKN_out * np.ones(num)

    # SNOx
    Ne_SNOX_out_list = Ne_SNOX_out * np.ones(num)

    # organic P
    Ne_P_org_out_list = Ne_P_org_out * np.ones(num)

    # PO4 - inorganic P
    Ne_P_out_list = Ne_P_out * np.ones(num)

    # LCOW
    Ne_LCOW_list = Ne_LCOW * np.ones(num)

    # SEC
    Ne_SEC_list = Ne_SEC * np.ones(num)

    # electroNP
    # O2 -- R5
    S_O2_out_R5_list = np.zeros(num)
    S_O2_out_R5_list[:] = np.nan

    # O2 -- R6
    S_O2_out_R6_list = np.zeros(num)
    S_O2_out_R6_list[:] = np.nan

    # O2 -- R7
    S_O2_out_R7_list = np.zeros(num)
    S_O2_out_R7_list[:] = np.nan

    # O2 -- effluent
    S_O2_out_list = np.zeros(num)
    S_O2_out_list[:] = np.nan

    # aeration energy
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

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
                has_electroNP=True,
                has_optimization=True,
            )

            S_O2_out_R5_list[i] = pyo.value(
                m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R6_list[i] = pyo.value(
                m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R7_list[i] = pyo.value(
                m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3
            )
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)

        except:
            pass

    # S_O2_out_R5_list = interp_1d(S_O2_out_R5_list)
    # S_O2_out_R6_list = interp_1d(S_O2_out_R6_list)
    # S_O2_out_R7_list = interp_1d(S_O2_out_R7_list)
    # S_O2_out_list = interp_1d(S_O2_out_list)
    # Ener_aeration_out = interp_1d(Ener_aeration_out)
    # P_removal_list = interp_1d(P_removal_list)
    # TSS_out_list = interp_1d(TSS_out_list)
    # COD_out_list = interp_1d(COD_out_list)
    # BOD_out_list = interp_1d(BOD_out_list)
    # TKN_out_list = interp_1d(TKN_out_list)
    # SNOX_out_list = interp_1d(SNOX_out_list)
    # P_org_out_list = interp_1d(P_org_out_list)
    # P_out_list = interp_1d(P_out_list)
    # LCOW_list = interp_1d(LCOW_list)

    TKN_max_list = 1000 * TKN_max_list

    # Figure a
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # axa.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # axa.legend(loc="lower left")

    # O2 -- R5
    axa.plot(TKN_max_list, S_O2_out_R5_list, color="k", label="_R5 O2 Concentration")
    axa.plot(
        TKN_max_list,
        Ne_S_O2_out_R5_list,
        color="k",
        linestyle="-.",
        label="_R5 O2 Concentration no electroNP",
    )
    # axa.set_ylim([0.86, 0.94])
    axa.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)
    axa.set_ylabel("R5 O2 Concentration  (mg/L)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R6
    axa1 = axa.twinx()
    axa1.plot(
        TKN_max_list, S_O2_out_R6_list, color="tab:blue", label="_R6 O2 Concentration"
    )
    axa1.plot(
        TKN_max_list,
        Ne_S_O2_out_R6_list,
        color="tab:blue",
        linestyle="-.",
        label="_R6 O2 Concentration no electroNP",
    )
    # axa1.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # axa1.set_ylim([45.15, 45.25])
    axa1.set_ylabel("R6 O2 Concentration (mg/L)", fontsize=11)
    axa1.tick_params(axis="x", labelsize=11)
    axa1.tick_params(axis="y", labelsize=11)
    axa1.yaxis.label.set_color("tab:blue")
    axa1.spines["right"].set_color("tab:blue")
    axa1.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R7
    axa2 = axa.twinx()
    axa2.spines.right.set_position(("axes", 1.15))
    axa2.plot(
        TKN_max_list, S_O2_out_R7_list, color="tab:orange", label="_R7 O2 Concentration"
    )
    axa2.plot(
        TKN_max_list,
        Ne_S_O2_out_R7_list,
        color="tab:orange",
        linestyle="-.",
        label="_R7 O2 Concentration no electroNP",
    )
    # axa2.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # axa2.set_ylim([96.44, 96.46])
    axa2.set_ylabel("R7 O2 Concentration (mg/L)", fontsize=11)
    axa2.tick_params(axis="x", labelsize=11)
    axa2.tick_params(axis="y", labelsize=11)
    axa2.yaxis.label.set_color("tab:orange")
    axa2.spines["right"].set_color("tab:orange")
    axa2.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # Figure b
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(TKN_max_list, S_O2_out_list, color="k", label="O2 Concentration")
    axb.plot(
        TKN_max_list,
        Ne_S_O2_out_list,
        color="k",
        linestyle="-.",
        label="O2 Concentration (no electroNP)",
    )
    axb.set_xlim([6.7, 8])
    # axb.set_ylim([0.86, 0.94])
    axb.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)
    axb.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)
    axb.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    plt.locator_params(axis="y", nbins=8)

    # aeration energy
    axb1 = axb.twinx()
    axb1.plot(
        TKN_max_list, Ener_aeration_out, color="tab:brown", label="Aeration energy"
    )
    axb1.plot(
        TKN_max_list,
        Ne_Ener_aeration_out,
        color="tab:brown",
        linestyle="-.",
        label="Aeration energy (no electroNP)",
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axb1.set_ylabel("Aeration energy (kWh/m$^3$)", fontsize=11)
    axb1.tick_params(axis="x", labelsize=11)
    axb1.tick_params(axis="y", labelsize=11)
    axb1.yaxis.label.set_color("tab:brown")
    axb1.spines["right"].set_color("tab:brown")
    axb1.tick_params(axis="y", colors="tab:brown")
    axb1.legend(loc="upper right", bbox_to_anchor=(1, 0.75))
    plt.locator_params(axis="y", nbins=8)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)

    # O2
    ax1.plot(TKN_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    ax1.plot(
        TKN_max_list,
        Ne_S_O2_out_list,
        color="k",
        linestyle="-.",
        label="_O2 Concentration no electroNP",
    )
    # axb.set_ylim([0.86, 0.94])
    ax1.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(TKN_max_list, TSS_out_list, color="tab:blue", label="_TSS Concentration")
    ax1a.plot(
        TKN_max_list,
        Ne_TSS_out_list,
        color="tab:blue",
        linestyle="-.",
        label="_TSS Concentration no electroNP",
    )
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(
        TKN_max_list, COD_out_list, color="tab:orange", label="_COD Concentration"
    )
    ax1b.plot(
        TKN_max_list,
        Ne_COD_out_list,
        color="tab:orange",
        linestyle="-.",
        label="_COD Concentration no electroNP",
    )
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # ax1b.set_ylim([96.2, 96.7])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(
        TKN_max_list, BOD_out_list, color="tab:purple", label="_BOD Concentration"
    )
    ax1c.plot(
        TKN_max_list,
        Ne_BOD_out_list,
        color="tab:purple",
        linestyle="-.",
        label="_BOD Concentration no electroNP",
    )
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    # ax1c.set_ylim([6.054, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax2.set_xlabel("TKN Max Concentration (mg/L)", fontsize=12)

    # O2
    ax2.plot(TKN_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    ax2.plot(
        TKN_max_list,
        Ne_S_O2_out_list,
        color="k",
        linestyle="-.",
        label="_O2 Concentration no electroNP",
    )
    # axb.set_ylim([0.86, 0.94])
    ax2.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(TKN_max_list, TKN_out_list, color="tab:brown", label="_TKN Concentration")
    ax1d.plot(
        TKN_max_list,
        Ne_TKN_out_list,
        color="tab:brown",
        linestyle="-.",
        label="_TKN Concentration no electroNP",
    )
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # ax1d.set_ylim([6.7, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(
        TKN_max_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration"
    )
    ax1e.plot(
        TKN_max_list,
        Ne_SNOX_out_list,
        color="tab:green",
        linestyle="-.",
        label="_SNOX Concentration no electroNP",
    )
    # ax1e.set_ylim([2.95, 3.5])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # # Figure 3
    # fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")
    # ax3.set_xlabel("TKN Max Concentration (mg/L)", fontsize=12)
    #
    # # O2
    # ax3.plot(TKN_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # ax3.plot(
    #     TKN_max_list,
    #     Ne_S_O2_out_list,
    #     color="k",
    #     linestyle="-.",
    #     label="_O2 Concentration no electroNP",
    # )
    # # axb.set_ylim([0.86, 0.94])
    # ax3.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    # ax3.tick_params(axis="x", labelsize=11)
    # ax3.tick_params(axis="y", labelsize=11)
    # plt.locator_params(axis="y", nbins=8)
    #
    # # organic P
    # ax1f = ax3.twinx()
    # ax1f.plot(
    #     TKN_max_list, P_org_out_list, color="tab:pink", label="_Organic P Concentration"
    # )
    # ax1f.plot(
    #     TKN_max_list,
    #     Ne_P_org_out_list,
    #     color="tab:pink",
    #     linestyle="-.",
    #     label="_Organic P Concentration no electroNP",
    # )
    # TP_max = 5 * np.ones(num)
    # ax1f.plot(TKN_max_list, TP_max, color="tab:grey", linestyle="--", label="TP Max")
    # # ax1f.set_ylim([5.3, 5.52])
    # ax1f.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)
    # ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    # ax1f.tick_params(axis="x", labelsize=11)
    # ax1f.tick_params(axis="y", labelsize=11)
    # ax1f.yaxis.label.set_color("tab:pink")
    # ax1f.spines["right"].set_color("tab:pink")
    # ax1f.tick_params(axis="y", colors="tab:pink")
    # ax1f.legend()
    # plt.locator_params(axis="y", nbins=8)
    #
    # # PO4
    # ax1g = ax3.twinx()
    # ax1g.spines.right.set_position(("axes", 1.15))
    # ax1g.plot(TKN_max_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    # ax1g.plot(
    #     TKN_max_list,
    #     Ne_P_out_list,
    #     color="tab:red",
    #     linestyle="-.",
    #     label="_PO4 Concentration no electroNP",
    # )
    # # ax1g.set_ylim([0, 240])
    # ax1g.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)
    # ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    # ax1g.tick_params(axis="x", labelsize=11)
    # ax1g.tick_params(axis="y", labelsize=11)
    # ax1g.yaxis.label.set_color("tab:red")
    # ax1g.spines["right"].set_color("tab:red")
    # ax1g.tick_params(axis="y", colors="tab:red")
    # plt.locator_params(axis="y", nbins=8)

    # Figure 3 - v3
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    plt.gca().axes.get_yaxis().set_visible(False)
    ax3.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)

    bax = brokenaxes(ylims=((0, 1), (675, 676)), hspace=0.15)
    bax.plot(TKN_max_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    bax.plot(
        TKN_max_list,
        Ne_P_out_list,
        color="tab:red",
        linestyle="-.",
        label="_PO4 no electroNP",
    )
    # bax.set_xlabel("COD Max Concentration (mg/L)", fontsize=11)
    bax.set_ylabel(
        "PO4 Concentration (mg/L)", fontsize=11, color="tab:red", labelpad=40
    )
    # bax.tick_params(axis="x", labelsize=11)
    bax.tick_params(axis="y", labelsize=11)
    ax3.spines["left"].set_color("tab:red")
    bax.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    ax3i = ax3.twinx()
    ax3i.plot(
        TKN_max_list, P_org_out_list, color="tab:pink", label="_Organic P Concentration"
    )
    ax3i.plot(
        TKN_max_list,
        Ne_P_org_out_list,
        color="tab:pink",
        linestyle="-.",
        label="_Organic P no electroNP",
    )
    ax3i.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax3i.tick_params(axis="y", labelsize=11)
    ax3i.yaxis.label.set_color("tab:pink")
    ax3i.spines["right"].set_color("tab:pink")
    ax3i.tick_params(axis="y", colors="tab:pink")
    plt.locator_params(axis="y", nbins=8)
    ax3i.spines["left"].set_color("tab:red")

    # Figure i
    figi, axi = plt.subplots(figsize=(9, 5), layout="constrained")
    axi.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)
    axi.plot(TKN_max_list, LCOW_list, color="k", label="LCOW")
    axi.plot(
        TKN_max_list,
        Ne_LCOW_list,
        color="k",
        linestyle="-.",
        label="LCOW (no electroNP)",
    )
    axi.set_xlim([6.7, 8])
    # axb.set_ylim([0.86, 0.94])
    axi.set_xlabel("TKN Max Concentration (mg/L)", fontsize=11)
    axi.set_ylabel("Levelized Cost of Water (/$/m$^3$ (2023))", fontsize=11)
    axi.tick_params(axis="x", labelsize=11)
    axi.tick_params(axis="y", labelsize=11)
    axi.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    plt.locator_params(axis="y", nbins=8)

    # SEC
    axi1 = axi.twinx()
    axi1.plot(TKN_max_list, SEC_list, color="tab:blue", label="SEC")
    axi1.plot(
        TKN_max_list,
        Ne_SEC_list,
        color="tab:blue",
        linestyle="-.",
        label="SEC (no electroNP)",
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axi1.set_ylabel("Specific Energy Consumption (kWh/m$^3$)", fontsize=11)
    axi1.tick_params(axis="x", labelsize=11)
    axi1.tick_params(axis="y", labelsize=11)
    axi1.yaxis.label.set_color("tab:blue")
    axi1.spines["right"].set_color("tab:blue")
    axi1.tick_params(axis="y", colors="tab:blue")
    axi1.legend(loc="upper right", bbox_to_anchor=(1, 0.75))
    plt.locator_params(axis="y", nbins=8)

    plt.show(block=True)


def plot_TP_max(num):
    # 1D plot
    TP_max_list = np.linspace(0.015, 0.05, num)

    # O2 -- R5
    S_O2_out_R5_list = np.zeros(num)
    S_O2_out_R5_list[:] = np.nan

    # O2 -- R6
    S_O2_out_R6_list = np.zeros(num)
    S_O2_out_R6_list[:] = np.nan

    # O2 -- R7
    S_O2_out_R7_list = np.zeros(num)
    S_O2_out_R7_list[:] = np.nan

    # O2 -- effluent
    S_O2_out_list = np.zeros(num)
    S_O2_out_list[:] = np.nan

    # aeration energy
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

    # SEC
    SEC_list = np.zeros(num)
    SEC_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_optimization_vary_max(
                COD_max=0.0975,
                BOD5_max=0.0065,
                TKN_max=0.007,
                TP_max=TP_max_list[i],
                has_electroNP=True,
                has_optimization=True,
            )

            S_O2_out_R5_list[i] = pyo.value(
                m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R6_list[i] = pyo.value(
                m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R7_list[i] = pyo.value(
                m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3
            )
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)

        except:
            pass

    # S_O2_out_R5_list = interp_1d(S_O2_out_R5_list)
    # S_O2_out_R6_list = interp_1d(S_O2_out_R6_list)
    # S_O2_out_R7_list = interp_1d(S_O2_out_R7_list)
    # S_O2_out_list = interp_1d(S_O2_out_list)
    # Ener_aeration_out = interp_1d(Ener_aeration_out)
    # P_removal_list = interp_1d(P_removal_list)
    # TSS_out_list = interp_1d(TSS_out_list)
    # COD_out_list = interp_1d(COD_out_list)
    # BOD_out_list = interp_1d(BOD_out_list)
    # TKN_out_list = interp_1d(TKN_out_list)
    # SNOX_out_list = interp_1d(SNOX_out_list)
    # P_org_out_list = interp_1d(P_org_out_list)
    # P_out_list = interp_1d(P_out_list)
    # LCOW_list = interp_1d(LCOW_list)
    # SEC_list = interp_1d(SEC_list)

    TP_max_list = 1000 * TP_max_list

    # Figure a
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # axa.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # axa.legend(loc="lower left")

    # O2 -- R5
    axa.plot(TP_max_list, S_O2_out_R5_list, color="k", label="_R5 O2 Concentration")
    # axa.set_ylim([0.86, 0.94])
    axa.set_xlabel("Total Phosphorus Max Concentration (mg/L)", fontsize=11)
    axa.set_ylabel("R5 O2 Concentration  (mg/L)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R6
    axa1 = axa.twinx()
    axa1.plot(
        TP_max_list, S_O2_out_R6_list, color="tab:blue", label="_R6 O2 Concentration"
    )
    # axa1.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # axa1.set_ylim([45.15, 45.25])
    axa1.set_ylabel("R6 O2 Concentration (mg/L)", fontsize=11)
    axa1.tick_params(axis="x", labelsize=11)
    axa1.tick_params(axis="y", labelsize=11)
    axa1.yaxis.label.set_color("tab:blue")
    axa1.spines["right"].set_color("tab:blue")
    axa1.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R7
    axa2 = axa.twinx()
    axa2.spines.right.set_position(("axes", 1.15))
    axa2.plot(
        TP_max_list, S_O2_out_R7_list, color="tab:orange", label="_R7 O2 Concentration"
    )
    # axa2.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # axa2.set_ylim([96.44, 96.46])
    axa2.set_ylabel("R7 O2 Concentration (mg/L)", fontsize=11)
    axa2.tick_params(axis="x", labelsize=11)
    axa2.tick_params(axis="y", labelsize=11)
    axa2.yaxis.label.set_color("tab:orange")
    axa2.spines["right"].set_color("tab:orange")
    axa2.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # Figure b
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(TP_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    axb.set_xlabel("Total Phosphorus Max Concentration (mg/L)", fontsize=11)
    axb.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # aeration energy
    axb1 = axb.twinx()
    axb1.plot(
        TP_max_list, Ener_aeration_out, color="tab:brown", label="_Aeration energy"
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axb1.set_ylabel("Aeration energy (kWh/m$^3$)", fontsize=11)
    axb1.tick_params(axis="x", labelsize=11)
    axb1.tick_params(axis="y", labelsize=11)
    axb1.yaxis.label.set_color("tab:brown")
    axb1.spines["right"].set_color("tab:brown")
    axb1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.set_xlabel("Total Phosphorus Max Concentration (mg/L)", fontsize=11)

    # O2
    ax1.plot(TP_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax1.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(TP_max_list, TSS_out_list, color="tab:blue", label="_TSS Concentration")
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(TP_max_list, COD_out_list, color="tab:orange", label="_COD Concentration")
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # ax1b.set_ylim([96.2, 96.7])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(TP_max_list, BOD_out_list, color="tab:purple", label="_BOD Concentration")
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    # ax1c.set_ylim([6.054, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax2.set_xlabel("Total Phosphorus Max Concentration (mg/L)", fontsize=12)

    # O2
    ax2.plot(TP_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax2.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(TP_max_list, TKN_out_list, color="tab:brown", label="_TKN Concentration")
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # ax1d.set_ylim([6.7, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(
        TP_max_list, SNOX_out_list, color="tab:green", label="_SNOX Concentration"
    )
    # ax1e.set_ylim([2.95, 3.5])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # Figure 3
    fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax3.set_xlabel("Total Phosphorus Max Concentration (mg/L)", fontsize=12)

    # O2
    ax3.plot(TP_max_list, S_O2_out_list, color="k", label="_O2 Concentration")
    # axb.set_ylim([0.86, 0.94])
    ax3.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax3.tick_params(axis="x", labelsize=11)
    ax3.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # organic P
    ax1f = ax3.twinx()
    ax1f.plot(
        TP_max_list, P_org_out_list, color="tab:pink", label="_Organic P Concentration"
    )
    # ax1f.plot(TP_max_list, TP_max, color="tab:grey", linestyle='--', label='_TP Max')
    # ax1f.set_ylim([5.3, 5.52])
    ax1f.set_xlabel("Total Phosphorus Max Concentration (mg/L)", fontsize=11)
    ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax1f.tick_params(axis="x", labelsize=11)
    ax1f.tick_params(axis="y", labelsize=11)
    ax1f.yaxis.label.set_color("tab:pink")
    ax1f.spines["right"].set_color("tab:pink")
    ax1f.tick_params(axis="y", colors="tab:pink")
    plt.locator_params(axis="y", nbins=8)

    # PO4
    ax1g = ax3.twinx()
    ax1g.spines.right.set_position(("axes", 1.15))
    ax1g.plot(TP_max_list, P_out_list, color="tab:red", label="_PO4 Concentration")
    # ax1g.set_ylim([0, 240])
    ax1g.set_xlabel("Total Phosphorus Max Concentration (mg/L)", fontsize=11)
    ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    ax1g.tick_params(axis="x", labelsize=11)
    ax1g.tick_params(axis="y", labelsize=11)
    ax1g.yaxis.label.set_color("tab:red")
    ax1g.spines["right"].set_color("tab:red")
    ax1g.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    # Figure i
    figi, axi = plt.subplots(figsize=(9, 5), layout="constrained")
    axi.plot(TP_max_list, LCOW_list, color="k", label="_LCOW")
    # axb.set_ylim([0.86, 0.94])
    axi.set_xlabel("Total Phosphorus Max Concentration (mg/L)", fontsize=11)
    axi.set_ylabel("Levelized Cost of Water (/$/m$^3$ (2023))", fontsize=11)
    axi.tick_params(axis="x", labelsize=11)
    axi.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # SEC
    axi1 = axi.twinx()
    axi1.plot(TP_max_list, SEC_list, color="tab:brown", label="_SEC")
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axi1.set_ylabel("Specific Energy Consumption (kWh/m$^3$)", fontsize=11)
    axi1.tick_params(axis="x", labelsize=11)
    axi1.tick_params(axis="y", labelsize=11)
    axi1.yaxis.label.set_color("tab:brown")
    axi1.spines["right"].set_color("tab:brown")
    axi1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    plt.show(block=True)


def plot_aeration_tank_volume(num):
    # 1D plot
    aeration_tank_volume_list = np.linspace(2900, 3100, num)

    # O2 -- R5
    S_O2_out_R5_list = np.zeros(num)
    S_O2_out_R5_list[:] = np.nan

    # O2 -- R6
    S_O2_out_R6_list = np.zeros(num)
    S_O2_out_R6_list[:] = np.nan

    # O2 -- R7
    S_O2_out_R7_list = np.zeros(num)
    S_O2_out_R7_list[:] = np.nan

    # O2 -- effluent
    S_O2_out_list = np.zeros(num)
    S_O2_out_list[:] = np.nan

    # aeration energy
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan

    # P removal
    P_removal_list = np.zeros(num)
    P_removal_list[:] = np.nan

    # TSS
    TSS_out_list = np.zeros(num)
    TSS_out_list[:] = np.nan

    # COD
    COD_out_list = np.zeros(num)
    COD_out_list[:] = np.nan

    # BOD
    BOD_out_list = np.zeros(num)
    BOD_out_list[:] = np.nan

    # TKN
    TKN_out_list = np.zeros(num)
    TKN_out_list[:] = np.nan

    # SNOx
    SNOX_out_list = np.zeros(num)
    SNOX_out_list[:] = np.nan

    # organic P
    P_org_out_list = np.zeros(num)
    P_org_out_list[:] = np.nan

    # PO4 - inorganic P
    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan

    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

    # SEC
    SEC_list = np.zeros(num)
    SEC_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_optimization_with_aeration_tank_volume(
                aeration_tank_volume=aeration_tank_volume_list[i],
                has_electroNP=True,
                has_optimization=True,
            )

            S_O2_out_R5_list[i] = pyo.value(
                m.fs.R5.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R6_list[i] = pyo.value(
                m.fs.R6.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_R7_list[i] = pyo.value(
                m.fs.R7.outlet.conc_mass_comp[0, "S_O2"] * 1e3
            )
            S_O2_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].conc_mass_comp["S_O2"] * 1e3
            )
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
            P_removal_list[i] = pyo.value(m.fs.electroNP.P_removal)
            TSS_out_list[i] = pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
            COD_out_list[i] = pyo.value(m.fs.Treated.properties[0].COD * 1e3)
            BOD_out_list[i] = pyo.value(
                m.fs.Treated.properties[0].BOD5["effluent"] * 1e3
            )
            TKN_out_list[i] = pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
            SNOX_out_list[i] = pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
            P_org_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
            P_out_list[i] = pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)

        except:
            pass

    # S_O2_out_R5_list = interp_1d(S_O2_out_R5_list)
    # S_O2_out_R6_list = interp_1d(S_O2_out_R6_list)
    # S_O2_out_R7_list = interp_1d(S_O2_out_R7_list)
    # S_O2_out_list = interp_1d(S_O2_out_list)
    # Ener_aeration_out = interp_1d(Ener_aeration_out)
    # P_removal_list = interp_1d(P_removal_list)
    # TSS_out_list = interp_1d(TSS_out_list)
    # COD_out_list = interp_1d(COD_out_list)
    # BOD_out_list = interp_1d(BOD_out_list)
    # TKN_out_list = interp_1d(TKN_out_list)
    # SNOX_out_list = interp_1d(SNOX_out_list)
    # P_org_out_list = interp_1d(P_org_out_list)
    # P_out_list = interp_1d(P_out_list)
    # LCOW_list = interp_1d(LCOW_list)
    # SEC_list = interp_1d(SEC_list)

    # Figure a
    figa, axa = plt.subplots(figsize=(9, 5), layout="constrained")
    # CP_base = -1.1
    # axa.axvline(x=CP_base, color='b', linestyle='--', label="Base case")
    # axa.legend(loc="lower left")

    # O2 -- R5
    axa.plot(
        aeration_tank_volume_list,
        S_O2_out_R5_list,
        color="k",
        label="_R5 O2 Concentration",
    )
    # axa.set_ylim([0.86, 0.94])
    axa.set_xlabel("Aeration Tank Volume (m3)", fontsize=11)
    axa.set_ylabel("R5 O2 Concentration  (mg/L)", fontsize=11)
    axa.tick_params(axis="x", labelsize=11)
    axa.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R6
    axa1 = axa.twinx()
    axa1.plot(
        aeration_tank_volume_list,
        S_O2_out_R6_list,
        color="tab:blue",
        label="_R6 O2 Concentration",
    )
    # axa1.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # axa1.set_ylim([45.15, 45.25])
    axa1.set_ylabel("R6 O2 Concentration (mg/L)", fontsize=11)
    axa1.tick_params(axis="x", labelsize=11)
    axa1.tick_params(axis="y", labelsize=11)
    axa1.yaxis.label.set_color("tab:blue")
    axa1.spines["right"].set_color("tab:blue")
    axa1.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # O2 -- R7
    axa2 = axa.twinx()
    axa2.spines.right.set_position(("axes", 1.15))
    axa2.plot(
        aeration_tank_volume_list,
        S_O2_out_R7_list,
        color="tab:orange",
        label="_R7 O2 Concentration",
    )
    # axa2.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # axa2.set_ylim([96.44, 96.46])
    axa2.set_ylabel("R7 O2 Concentration (mg/L)", fontsize=11)
    axa2.tick_params(axis="x", labelsize=11)
    axa2.tick_params(axis="y", labelsize=11)
    axa2.yaxis.label.set_color("tab:orange")
    axa2.spines["right"].set_color("tab:orange")
    axa2.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # Figure b
    figb, axb = plt.subplots(figsize=(9, 5), layout="constrained")
    axb.plot(
        aeration_tank_volume_list, S_O2_out_list, color="k", label="_O2 Concentration"
    )
    # axb.set_ylim([0.86, 0.94])
    axb.set_xlabel("Aeration Tank Volume (m3)", fontsize=11)
    axb.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    axb.tick_params(axis="x", labelsize=11)
    axb.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # aeration energy
    axb1 = axb.twinx()
    axb1.plot(
        aeration_tank_volume_list,
        Ener_aeration_out,
        color="tab:brown",
        label="_Aeration energy",
    )
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axb1.set_ylabel("Aeration energy (kWh/m$^3$)", fontsize=11)
    axb1.tick_params(axis="x", labelsize=11)
    axb1.tick_params(axis="y", labelsize=11)
    axb1.yaxis.label.set_color("tab:brown")
    axb1.spines["right"].set_color("tab:brown")
    axb1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.set_xlabel("Aeration Tank Volume (m3)", fontsize=11)

    # O2
    ax1.plot(
        aeration_tank_volume_list, S_O2_out_list, color="k", label="_O2 Concentration"
    )
    # axb.set_ylim([0.86, 0.94])
    ax1.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TSS
    ax1a = ax1.twinx()
    ax1a.plot(
        aeration_tank_volume_list,
        TSS_out_list,
        color="tab:blue",
        label="_TSS Concentration",
    )
    # ax1a.plot(CP_list, TSS_max, color="tab:blue", linestyle='--', label='_TSS Max')
    # ax1a.set_ylim([44.95, 45.25])
    ax1a.set_ylabel("TSS Concentration (mg/L)", fontsize=11)
    ax1a.tick_params(axis="x", labelsize=11)
    ax1a.tick_params(axis="y", labelsize=11)
    ax1a.yaxis.label.set_color("tab:blue")
    ax1a.spines["right"].set_color("tab:blue")
    ax1a.tick_params(axis="y", colors="tab:blue")
    plt.locator_params(axis="y", nbins=8)

    # COD
    ax1b = ax1.twinx()
    ax1b.spines.right.set_position(("axes", 1.15))
    ax1b.plot(
        aeration_tank_volume_list,
        COD_out_list,
        color="tab:orange",
        label="_COD Concentration",
    )
    # ax1b.plot(CP_list, COD_max, color="tab:orange", linestyle='--', label='_COD Max')
    # ax1b.set_ylim([96.2, 96.7])
    ax1b.set_ylabel("COD Concentration (mg/L)", fontsize=11)
    ax1b.tick_params(axis="x", labelsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    ax1b.yaxis.label.set_color("tab:orange")
    ax1b.spines["right"].set_color("tab:orange")
    ax1b.tick_params(axis="y", colors="tab:orange")
    plt.locator_params(axis="y", nbins=8)

    # BOD
    ax1c = ax1.twinx()
    ax1c.spines.right.set_position(("axes", 1.35))
    ax1c.plot(
        aeration_tank_volume_list,
        BOD_out_list,
        color="tab:purple",
        label="_BOD Concentration",
    )
    # ax1c.plot(CP_list, BOD_max, color="tab:purple", linestyle='--', label='_BOD Max')
    # ax1c.set_ylim([6.054, 6.074])
    ax1c.set_ylabel("BOD Concentration (mg/L)", fontsize=11)
    ax1c.tick_params(axis="x", labelsize=11)
    ax1c.tick_params(axis="y", labelsize=11)
    ax1c.yaxis.label.set_color("tab:purple")
    ax1c.spines["right"].set_color("tab:purple")
    ax1c.tick_params(axis="y", colors="tab:purple")
    plt.locator_params(axis="y", nbins=8)

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax2.set_xlabel("Aeration Tank Volume (m3)", fontsize=12)

    # O2
    ax2.plot(
        aeration_tank_volume_list, S_O2_out_list, color="k", label="_O2 Concentration"
    )
    # axb.set_ylim([0.86, 0.94])
    ax2.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # TKN
    ax1d = ax2.twinx()
    ax1d.plot(
        aeration_tank_volume_list,
        TKN_out_list,
        color="tab:brown",
        label="_TKN Concentration",
    )
    # ax1d.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # ax1d.set_ylim([6.7, 6.74])
    ax1d.set_ylabel("TKN Concentration (mg/L)", fontsize=11)
    ax1d.tick_params(axis="x", labelsize=11)
    ax1d.tick_params(axis="y", labelsize=11)
    ax1d.yaxis.label.set_color("tab:brown")
    ax1d.spines["right"].set_color("tab:brown")
    ax1d.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    # SNOx
    ax1e = ax2.twinx()
    ax1e.spines.right.set_position(("axes", 1.15))
    ax1e.plot(
        aeration_tank_volume_list,
        SNOX_out_list,
        color="tab:green",
        label="_SNOX Concentration",
    )
    # ax1e.set_ylim([2.95, 3.5])
    ax1e.set_ylabel("SNOx Concentration (mg/L)", fontsize=11)
    ax1e.tick_params(axis="x", labelsize=11)
    ax1e.tick_params(axis="y", labelsize=11)
    ax1e.yaxis.label.set_color("tab:green")
    ax1e.spines["right"].set_color("tab:green")
    ax1e.tick_params(axis="y", colors="tab:green")
    plt.locator_params(axis="y", nbins=8)

    # Figure 3
    fig3, ax3 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax3.set_xlabel("Aeration Tank Volume (m3)", fontsize=12)

    # O2
    ax3.plot(
        aeration_tank_volume_list, S_O2_out_list, color="k", label="_O2 Concentration"
    )
    # axb.set_ylim([0.86, 0.94])
    ax3.set_ylabel("O2 Concentration (mg/L)", fontsize=11)
    ax3.tick_params(axis="x", labelsize=11)
    ax3.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # organic P
    ax1f = ax3.twinx()
    ax1f.plot(
        aeration_tank_volume_list,
        P_org_out_list,
        color="tab:pink",
        label="_Organic P Concentration",
    )
    # ax1f.plot(aeration_tank_volume_list, TP_max, color="tab:grey", linestyle='--', label='_TP Max')
    # ax1f.set_ylim([5.3, 5.52])
    ax1f.set_xlabel("Aeration Tank Volume (m3)", fontsize=11)
    ax1f.set_ylabel("Organic Phosphorus Concentration (mg/L)", fontsize=11)
    ax1f.tick_params(axis="x", labelsize=11)
    ax1f.tick_params(axis="y", labelsize=11)
    ax1f.yaxis.label.set_color("tab:pink")
    ax1f.spines["right"].set_color("tab:pink")
    ax1f.tick_params(axis="y", colors="tab:pink")
    plt.locator_params(axis="y", nbins=8)

    # PO4
    ax1g = ax3.twinx()
    ax1g.spines.right.set_position(("axes", 1.15))
    ax1g.plot(
        aeration_tank_volume_list,
        P_out_list,
        color="tab:red",
        label="_PO4 Concentration",
    )
    # ax1g.set_ylim([0, 240])
    ax1g.set_xlabel("Aeration Tank Volume (m3)", fontsize=11)
    ax1g.set_ylabel("PO4 Concentration (mg/L)", fontsize=11)
    ax1g.tick_params(axis="x", labelsize=11)
    ax1g.tick_params(axis="y", labelsize=11)
    ax1g.yaxis.label.set_color("tab:red")
    ax1g.spines["right"].set_color("tab:red")
    ax1g.tick_params(axis="y", colors="tab:red")
    plt.locator_params(axis="y", nbins=8)

    # Figure i
    figi, axi = plt.subplots(figsize=(9, 5), layout="constrained")
    axi.plot(aeration_tank_volume_list, LCOW_list, color="k", label="_LCOW")
    # axb.set_ylim([0.86, 0.94])
    axi.set_xlabel("Aeration Tank Volume (m3)", fontsize=11)
    axi.set_ylabel("Levelized Cost of Water (/$/m$^3$ (2023))", fontsize=11)
    axi.tick_params(axis="x", labelsize=11)
    axi.tick_params(axis="y", labelsize=11)
    plt.locator_params(axis="y", nbins=8)

    # SEC
    axi1 = axi.twinx()
    axi1.plot(aeration_tank_volume_list, SEC_list, color="tab:brown", label="_SEC")
    # axb1.plot(CP_list, TKN_max, color="tab:brown", linestyle='--', label='_TKN Max')
    # axb1.set_ylim([6.72, 6.74])
    axi1.set_ylabel("Specific Energy Consumption (kWh/m$^3$)", fontsize=11)
    axi1.tick_params(axis="x", labelsize=11)
    axi1.tick_params(axis="y", labelsize=11)
    axi1.yaxis.label.set_color("tab:brown")
    axi1.spines["right"].set_color("tab:brown")
    axi1.tick_params(axis="y", colors="tab:brown")
    plt.locator_params(axis="y", nbins=8)

    plt.show(block=True)


def stackplot_COD_max(num):
    # 1D plot
    COD_max_list = np.linspace(0.0955, 0.0978, num)
    # COD_max_list = np.linspace(0.096, 0.1, num)
    # COD_max_list = np.linspace(0.095, 0.0975, num)

    # No electroNP flowsheet
    m, results = run_optimization_vary_max(
        COD_max=0.1,
        BOD5_max=0.01,
        TKN_max=0.007,
        TP_max=0.68,
        has_electroNP=False,
        has_optimization=True,
    )

    Ne_Ener_aeration = pyo.value(m.fs.costing.aeration_energy)
    Ne_LCOW = pyo.value(m.fs.costing.LCOW)
    Ne_SEC = pyo.value(m.fs.costing.specific_energy_consumption)

    # aeration energy
    Ne_Ener_aeration_list = Ne_Ener_aeration * np.ones(num)

    # LCOW
    Ne_LCOW_list = Ne_LCOW * np.ones(num)

    # SEC
    Ne_SEC_list = Ne_SEC * np.ones(num)

    # electroNP flowsheet
    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

    # SEC
    SEC_list = np.zeros(num)
    SEC_list[:] = np.nan

    # aeration energy
    Ener_aeration_list = np.zeros(num)
    Ener_aeration_list[:] = np.nan

    # electroNP SEC
    SEC_electroNP_list = np.zeros(num)
    SEC_electroNP_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_optimization_vary_max(
                COD_max=COD_max_list[i],
                BOD5_max=0.01,
                TKN_max=0.007,
                TP_max=0.005,
                has_electroNP=True,
                has_optimization=True,
            )

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)
            Ener_aeration_list[i] = pyo.value(m.fs.costing.aeration_energy)
            SEC_electroNP_list[i] = pyo.value(m.fs.costing.electroNP_energy_consumption)
        except:
            pass

    LCOW_list = interp_1d(LCOW_list)
    SEC_list = interp_1d(SEC_list)
    Ener_aeration_list = interp_1d(Ener_aeration_list)
    SEC_electroNP_list = interp_1d(SEC_electroNP_list)

    COD_max_list = 1000 * COD_max_list

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(7, 5), layout="constrained")
    stacked_1 = [a + b for a, b in zip(Ener_aeration_list, SEC_electroNP_list)]
    SEC_other = [a - b for a, b in zip(SEC_list, stacked_1)]
    stacked_cols = [Ener_aeration_list, SEC_electroNP_list, SEC_other]
    labels = ["Aeration energy", "electroN-P", "other"]
    hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    ax1.stackplot(COD_max_list, stacked_cols, labels=labels, hatch=hatches)
    ax1.plot(
        COD_max_list,
        Ne_SEC_list,
        color="k",
        linestyle="-.",
        label="SEC (no electroNP)",
    )
    ax1.set_xlim(95.62, 97.8)
    ax1.set_xlabel("COD Max Concentration (mg/L)", fontsize=14)
    ax1.set_ylabel("SEC (kWh/m3)", fontsize=14)
    ax1.legend()

    plt.show(block=True)


def stackplot_BOD5_max(num):
    # 1D plot
    BOD5_max_list = np.linspace(0.006, 0.0075, num)
    # BOD5_max_list = np.linspace(0.006, 0.0065, num)

    # No electroNP flowsheet
    m, results = run_optimization_vary_max(
        COD_max=0.1,
        BOD5_max=0.01,
        TKN_max=0.007,
        TP_max=0.68,
        has_electroNP=False,
        has_optimization=True,
    )

    Ne_Ener_aeration = pyo.value(m.fs.costing.aeration_energy)
    Ne_LCOW = pyo.value(m.fs.costing.LCOW)
    Ne_SEC = pyo.value(m.fs.costing.specific_energy_consumption)

    # aeration energy
    Ne_Ener_aeration_list = Ne_Ener_aeration * np.ones(num)

    # LCOW
    Ne_LCOW_list = Ne_LCOW * np.ones(num)

    # SEC
    Ne_SEC_list = Ne_SEC * np.ones(num)

    # electroNP flowsheet
    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

    # SEC
    SEC_list = np.zeros(num)
    SEC_list[:] = np.nan

    # aeration energy
    Ener_aeration_list = np.zeros(num)
    Ener_aeration_list[:] = np.nan

    # electroNP SEC
    SEC_electroNP_list = np.zeros(num)
    SEC_electroNP_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_optimization_vary_max(
                COD_max=0.1,
                BOD5_max=BOD5_max_list[i],
                TKN_max=0.007,
                TP_max=0.005,
                has_electroNP=True,
                has_optimization=True,
            )

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)
            Ener_aeration_list[i] = pyo.value(m.fs.costing.aeration_energy)
            SEC_electroNP_list[i] = pyo.value(m.fs.costing.electroNP_energy_consumption)
        except:
            pass

    LCOW_list = interp_1d(LCOW_list)
    SEC_list = interp_1d(SEC_list)
    Ener_aeration_list = interp_1d(Ener_aeration_list)
    SEC_electroNP_list = interp_1d(SEC_electroNP_list)

    BOD5_max_list = 1000 * BOD5_max_list

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(7, 5), layout="constrained")
    stacked_1 = [a + b for a, b in zip(Ener_aeration_list, SEC_electroNP_list)]
    SEC_other = [a - b for a, b in zip(SEC_list, stacked_1)]
    stacked_cols = [Ener_aeration_list, SEC_electroNP_list, SEC_other]
    labels = ["Aeration energy", "electroN-P", "other"]
    hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    ax1.stackplot(BOD5_max_list, stacked_cols, labels=labels, hatch=hatches)
    ax1.plot(
        BOD5_max_list,
        Ne_SEC_list,
        color="k",
        linestyle="-.",
        label="SEC (no electroNP)",
    )
    # ax1.set_xlim(95.62, 97.8)
    ax1.set_xlabel("BOD5 Max Concentration (mg/L)", fontsize=14)
    ax1.set_ylabel("SEC (kWh/m3)", fontsize=14)
    ax1.legend()

    plt.show(block=True)


def stackplot_TKN_max(num):
    # 1D plot
    TKN_max_list = np.linspace(0.0066, 0.0074, num)

    # No electroNP flowsheet
    m, results = run_optimization_vary_max(
        COD_max=0.1,
        BOD5_max=0.01,
        TKN_max=0.007,
        TP_max=0.68,
        has_electroNP=False,
        has_optimization=True,
    )

    Ne_Ener_aeration = pyo.value(m.fs.costing.aeration_energy)
    Ne_LCOW = pyo.value(m.fs.costing.LCOW)
    Ne_SEC = pyo.value(m.fs.costing.specific_energy_consumption)

    # aeration energy
    Ne_Ener_aeration_list = Ne_Ener_aeration * np.ones(num)

    # LCOW
    Ne_LCOW_list = Ne_LCOW * np.ones(num)

    # SEC
    Ne_SEC_list = Ne_SEC * np.ones(num)

    # electroNP flowsheet
    # LCOW
    LCOW_list = np.zeros(num)
    LCOW_list[:] = np.nan

    # SEC
    SEC_list = np.zeros(num)
    SEC_list[:] = np.nan

    # aeration energy
    Ener_aeration_list = np.zeros(num)
    Ener_aeration_list[:] = np.nan

    # electroNP SEC
    SEC_electroNP_list = np.zeros(num)
    SEC_electroNP_list[:] = np.nan

    for i in range(0, num):
        try:
            m, results = run_optimization_vary_max(
                COD_max=0.1,
                BOD5_max=0.01,
                TKN_max=TKN_max_list[i],
                TP_max=0.005,
                has_electroNP=True,
                has_optimization=True,
            )

            LCOW_list[i] = pyo.value(m.fs.costing.LCOW)
            SEC_list[i] = pyo.value(m.fs.costing.specific_energy_consumption)
            Ener_aeration_list[i] = pyo.value(m.fs.costing.aeration_energy)
            SEC_electroNP_list[i] = pyo.value(m.fs.costing.electroNP_energy_consumption)
        except:
            pass

    LCOW_list = interp_1d(LCOW_list)
    SEC_list = interp_1d(SEC_list)
    Ener_aeration_list = interp_1d(Ener_aeration_list)
    SEC_electroNP_list = interp_1d(SEC_electroNP_list)

    TKN_max_list = 1000 * TKN_max_list

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(7, 5), layout="constrained")
    stacked_1 = [a + b for a, b in zip(Ener_aeration_list, SEC_electroNP_list)]
    SEC_other = [a - b for a, b in zip(SEC_list, stacked_1)]
    stacked_cols = [Ener_aeration_list, SEC_electroNP_list, SEC_other]
    labels = ["Aeration energy", "electroN-P", "other"]
    hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    ax1.stackplot(TKN_max_list, stacked_cols, labels=labels, hatch=hatches)
    ax1.plot(
        TKN_max_list,
        Ne_SEC_list,
        color="k",
        linestyle="-.",
        label="SEC (no electroNP)",
    )
    # ax1.set_xlim(6.7, 8)
    ax1.set_xlabel("TKN Max Concentration (mg/L)", fontsize=14)
    ax1.set_ylabel("SEC (kWh/m3)", fontsize=14)
    ax1.legend()

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


if __name__ == "__main__":
    # m, results = main(CP=-1.2 * pyo.units.V, r_AV=0.12)
    # m, results = main(CP=-1.2 * pyo.units.V, r_AV=0.09)
    # m, results = main(CP=-0.8 * pyo.units.V, r_AV=0.12)
    # m, results = main(CP=-0.8 * pyo.units.V, r_AV=0.09)

    # m, results = main(CP=-1.3 * pyo.units.V, r_AV=0.1)

    # m, results = run_optimization(
    #     CP=-1.1,
    #     r_AV=0.1,
    #     has_electroNP=True,
    #     has_optimization=True,
    #     objective=objective_fun.LCOW,
    #     has_effluent_constraints=False,
    # )

    # m, results = run_with_KLa(KLa_R5=13, KLa_R6=7, KLa_R7=6)

    # run_optimization_vary_max(
    #     COD_max=0.096,
    #     has_electroNP=True,
    #     has_optimization=True,
    # )

    # m, results = run_optimization_vary_max(
    #     COD_max=0.1,
    #     BOD5_max=0.01,
    #     TKN_max=0.007,
    #     TP_max=0.68,
    #     has_electroNP=False,
    #     has_optimization=True,
    # )

    # plot_CP(num=30)
    # # plot_CP_effluent(num=25)
    # plot_rAV(num=40)
    # plot_rAV_effluent(num=25)
    # contourf_plot(num=15)
    # contourf_plot_electricity_cost(num=3)
    # contourf_plot_aeration(num=10)

    # plot_aeration_R5(num=15)
    # plot_aeration_R6(num=15)
    # plot_aeration_R7(num=15)

    # plot_COD_max(num=19)
    # plot_BOD5_max(num=30)
    # plot_TKN_max(num=15)
    # plot_TP_max(num=15)
    # plot_aeration_tank_volume(num=15)

    # plot_COD_max_no_electroNP(num=15)
    # plot_BOD5_max_no_electroNP(num=15)

    # stackplot_COD_max(num=10)
    # stackplot_BOD5_max(num=30)
    stackplot_TKN_max(num=19)
