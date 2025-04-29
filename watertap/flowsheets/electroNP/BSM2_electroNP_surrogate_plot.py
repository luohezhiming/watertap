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
)
import matplotlib.pyplot as plt
from scipy import interpolate


def main(CP=-1.05 * pyo.units.V, r_AV=0.105):
    m = build_flowsheet(has_electroNP=True)
    set_operating_conditions(m)
    if CP == -1.2 and r_AV == 0.12:
        m.fs.electroNP.cathodic_potential.fix(-1.2)
        m.fs.electroNP.area_volume_ratio.fix(0.12)
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


def plot(num):
    # 1D plot
    # CP_list = np.linspace(-1.3, -0.8, num)
    # r_AV_list = np.linspace(0.08, 0.13, num)
    # 2D plot
    CP_list = np.linspace(-1.2, -0.8, num)
    r_AV_list = np.linspace(0.09, 0.12, num)

    P_out_list = np.zeros(num)
    P_out_list[:] = np.nan
    Ener_electroNP_out = np.zeros(num)
    Ener_electroNP_out[:] = np.nan
    Ener_aeration_out = np.zeros(num)
    Ener_aeration_out[:] = np.nan

    P_out_matrix = np.zeros((num, num))
    P_out_matrix[:] = np.nan
    LCOW_matrix = np.zeros((num, num))
    LCOW_matrix[:] = np.nan
    SEC_matrix = np.zeros((num, num))
    SEC_matrix[:] = np.nan
    SEC_electroNP_matrix = np.zeros((num, num))
    SEC_electroNP_matrix[:] = np.nan

    for i in range(0, num):
        try:
            m, results = main(CP=CP_list[i], r_AV=0.10)
            P_out_list[i] = (
                m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
            )
            Ener_electroNP_out[i] = pyo.value(m.fs.costing.electroNP_energy_consumption)
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
        except:
            pass

    P_out_list = interp_1d(P_out_list)
    Ener_electroNP_out = interp_1d(Ener_electroNP_out)
    Ener_aeration_out = interp_1d(Ener_aeration_out)

    # fig1, ax1 = plt.subplots(figsize=(7, 5))
    # ax1.plot(CP_list, P_out_list, "b")
    # # ax1.set_xlim([0.02, 0.05])
    # # Base case
    # CP_base = -1.1
    # m, results = main(CP=CP_base, r_AV=0.10)
    # P_out_base = m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
    # ax1.plot(CP_base, P_out_base, marker="o", color="black", label="Base case")
    # ax1.annotate(
    #     f"({CP_base}, {round(P_out_base,2)})",
    #     (CP_base, P_out_base),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    # # Optimal
    # opt_idx = np.argmin(P_out_list)
    # CP_opt = CP_list[opt_idx]
    # m, results = main(CP=CP_opt, r_AV=0.10)
    # P_out_opt = m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
    # ax1.plot(CP_opt, P_out_opt, marker="o", color="red", label="Optimal")
    # ax1.annotate(
    #     f"({round(CP_opt, 2)}, {round(P_out_opt, 2)})",
    #     (CP_opt, P_out_opt),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    #
    # ax1.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # ax1.set_ylabel("Concentration of PO4 (mg/L)", fontsize=12)
    # ax1.legend(
    #     ["Model", "Base case", "Optimal"]
    # )

    # # Ener_CP
    # fig1b, ax1b = plt.subplots(figsize=(7, 5))
    # ax1b.plot(CP_list, Ener_electroNP_out, "r")
    # # ax1b.set_xlim([0.02, 0.05])
    # # Base case
    # CP_base = -1.1
    # m, results = main(CP=CP_base, r_AV=0.10)
    # Ener_electroNP_out_base = pyo.value(m.fs.costing.electroNP_energy_consumption)
    # ax1b.plot(CP_base, Ener_electroNP_out_base, marker="o", color="black", label="Base case")
    # ax1b.annotate(
    #     f"({CP_base}, {round(Ener_electroNP_out_base, 4)})",
    #     (CP_base, Ener_electroNP_out_base),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    # # Optimal
    # opt_idx = np.argmin(Ener_electroNP_out)
    # CP_opt = CP_list[opt_idx]
    # m, results = main(CP=CP_opt, r_AV=0.10)
    # Ener_electroNP_out_opt = pyo.value(m.fs.costing.electroNP_energy_consumption)
    # ax1b.plot(CP_opt, Ener_electroNP_out_opt, marker="o", color="red", label="Optimal")
    # ax1b.annotate(
    #     f"({round(CP_opt, 2)}, {round(Ener_electroNP_out_opt, 4)})",
    #     (CP_opt, Ener_electroNP_out_opt),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    # ax1b.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # ax1b.set_ylabel("Energy Consumption of electron-P (kWh/m3)", fontsize=12)
    # # ax1b.set_title("Low Pressure Side Outlet")
    # ax1b.legend(
    #     ["Model", "Base case", "Optimal"]
    # )

    for i in range(0, num):
        try:
            m, results = main(CP=-1.1, r_AV=r_AV_list[i])
            P_out_list[i] = (
                m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
            )
            Ener_electroNP_out[i] = pyo.value(m.fs.costing.electroNP_energy_consumption)
            Ener_aeration_out[i] = pyo.value(m.fs.costing.aeration_energy)
        except:
            pass

    P_out_list = interp_1d(P_out_list)
    Ener_electroNP_out = interp_1d(Ener_electroNP_out)
    Ener_aeration_out = interp_1d(Ener_aeration_out)

    # fig2, ax2 = plt.subplots(figsize=(7, 5))
    # ax2.plot(r_AV_list, P_out_list, "b")
    # # ax2.set_xlim([0.02, 0.05])
    # # Base case
    # r_AV_base = 0.1
    # m, results = main(CP=-1.1, r_AV=r_AV_base)
    # P_out_base = m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
    # ax2.plot(r_AV_base, P_out_base, marker="o", color="black", label="Base case")
    # ax2.annotate(
    #     f"({r_AV_base}, {round(P_out_base,2)})",
    #     (r_AV_base, P_out_base),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    # # Optimal
    # opt_idx = np.argmin(P_out_list)
    # r_AV_opt = r_AV_list[opt_idx]
    # m, results = main(CP=-1.1, r_AV=r_AV_opt)
    # P_out_opt = m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
    # ax2.plot(r_AV_opt, P_out_opt, marker="o", color="red", label="Optimal")
    # ax2.annotate(
    #     f"({round(r_AV_opt, 2)}, {round(P_out_opt, 2)})",
    #     (r_AV_opt, P_out_opt),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    # ax2.set_xlabel("Area Volume Ratio", fontsize=12)
    # ax2.set_ylabel("Concentration of PO4 (mg/L)", fontsize=12)
    # ax2.legend(
    #     ["Model", "Base case", "Optimal"]
    # )

    # fig2b, ax2b = plt.subplots(figsize=(7, 5))
    # ax2b.plot(r_AV_list, Ener_electroNP_out, "r")
    # # ax2b.plot(r_AV_list, Ener_aeration_out, "b")
    # # ax2b.set_xlim([0.02, 0.05])
    # # Base case
    # r_AV_base = 0.1
    # m, results = main(CP=-1.1, r_AV=r_AV_base)
    # Ener_electroNP_out_base = pyo.value(m.fs.costing.electroNP_energy_consumption)
    # ax2b.plot(r_AV_base, Ener_electroNP_out_base, marker="o", color="black", label="Base case")
    # ax2b.annotate(
    #     f"({r_AV_base}, {round(Ener_electroNP_out_base,4)})",
    #     (r_AV_base, Ener_electroNP_out_base),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    # # Optimal
    # opt_idx = np.argmax(Ener_electroNP_out)
    # r_AV_opt = r_AV_list[opt_idx]
    # m, results = main(CP=-1.1, r_AV=r_AV_opt)
    # Ener_electroNP_out_opt = pyo.value(m.fs.costing.electroNP_energy_consumption)
    # ax2b.plot(r_AV_opt, Ener_electroNP_out_opt, marker="o", color="red", label="Optimal")
    # ax2b.annotate(
    #     f"({round(r_AV_opt, 2)}, {round(Ener_electroNP_out_opt, 4)})",
    #     (r_AV_opt, Ener_electroNP_out_opt),
    #     textcoords="offset points",
    #     xytext=(6, 6),
    # )
    # ax2b.set_xlabel("Area Volume Ratio", fontsize=12)
    # ax2b.set_ylabel("Energy Consumption (kWh/m3) of electron-P", fontsize=12)
    # # ax2b.set_title("Low Pressure Side Outlet")
    # ax2b.legend(
    #     ["Model", "Base case", "Optimal"]
    # )

    for i in range(0, num):
        for j in range(0, num):
            print(f"CP: {CP_list[i]}")
            print(f"rAV: {r_AV_list[j]}")
            try:
                m, results = main(CP=CP_list[i], r_AV=r_AV_list[j])
                P_out_matrix[j, i] = (
                    m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
                )
                LCOW_matrix[j, i] = pyo.value(m.fs.costing.LCOW)
                SEC_matrix[j, i] = pyo.value(m.fs.costing.specific_energy_consumption)
                SEC_electroNP_matrix[j, i] = pyo.value(
                    m.fs.costing.electroNP_energy_consumption
                ) / pyo.value(m.fs.costing.specific_energy_consumption)
            except:
                pass

    P_out_matrix = interp_2d(P_out_matrix)
    LCOW_matrix = interp_2d(LCOW_matrix)
    SEC_matrix = interp_2d(SEC_matrix)
    SEC_electroNP_matrix = interp_2d(SEC_electroNP_matrix)

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
    ax3.set_ylabel("Area Volume Ratio", fontsize=12)
    cbar = fig3.colorbar(CF)
    cbar.ax.set_ylabel("Concentration of PO4 in the treated water (mg/L)")

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
    ax4.set_ylabel("Area Volume Ratio", fontsize=12)
    cbar = fig4.colorbar(CF)
    cbar.ax.set_ylabel("LCOW ($/m3)")

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
    ax5.set_ylabel("Area Volume Ratio", fontsize=12)
    cbar = fig5.colorbar(CF)
    cbar.ax.set_ylabel("SEC (kWh/m3)")

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
    ax6.set_ylabel("Area Volume Ratio", fontsize=12)
    cbar = fig6.colorbar(CF)
    cbar.ax.set_ylabel("ElectroNP SEC / total SEC")

    plt.show(block=True)

    return P_out_matrix


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
    P_out_matrix = plot(num=10)
