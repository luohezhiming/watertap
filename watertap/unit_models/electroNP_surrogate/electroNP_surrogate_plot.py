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
from watertap.unit_models.electroNP_surrogate.electroNP_surrogate_flowsheet import (
    build_flowsheet,
    solve,
)
import tkinter
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def main(CP=-1.05 * pyo.units.V, r_AV=0.105, T=308.15 * pyo.units.K):
    m = build_flowsheet()

    m.fs.unit.cathodic_potential.unfix()
    m.fs.unit.area_volume_ratio.unfix()
    m.fs.unit.inlet.temperature.unfix()

    m.fs.unit.cathodic_potential.fix(CP)
    m.fs.unit.area_volume_ratio.fix(r_AV)
    m.fs.unit.inlet.temperature.fix(T)

    m.fs.unit.initialize(solver="ipopt-watertap")

    results = solve(m)

    return m, results


def plot(num):
    CP_list = np.linspace(-1.3, -0.8, num)
    r_AV_list = np.linspace(0.06, 0.13, num)
    T_list = np.linspace(281.15, 315.15, num)

    P_out_list = np.zeros(num)
    P_out_matrix = np.zeros((num, num))
    P_out_matrix[:] = np.nan

    # LCOW_matrix = np.zeros((num, num))
    # LCOW_matrix[:] = np.nan
    # SEC_matrix = np.zeros((num, num))
    # SEC_matrix[:] = np.nan

    for i in range(0, num):
        m, results = main(CP=CP_list[i], r_AV=0.105, T=308.15 * pyo.units.K)

        P_out_list[i] = m.fs.unit.treated.conc_mass_comp[0, "S_PO4"].value

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(CP_list, P_out_list, "b")
    # ax1.set_xlim([0.02, 0.05])
    ax1.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax1.set_ylabel("Concentration of PO4 (g/L)", fontsize=12)
    # ax1.set_title("Low Pressure Side Outlet")

    for i in range(0, num):
        try:
            m, results = main(CP=-1.05, r_AV=r_AV_list[i], T=308.15 * pyo.units.K)

            P_out_list[i] = m.fs.unit.treated.conc_mass_comp[0, "S_PO4"].value
        except:
            pass

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(r_AV_list, P_out_list, "b")
    # ax2.set_xlim([0.02, 0.05])
    ax2.set_xlabel("Area Volume Ratio", fontsize=12)
    ax2.set_ylabel("Concentration of PO4 (g/L)", fontsize=12)

    for i in range(0, num):
        for j in range(0, num):
            print(f"CP: {CP_list[i]}")
            print(f"rAV: {r_AV_list[j]}")
            try:
                m, results = main(
                    CP=CP_list[i], r_AV=r_AV_list[j], T=308.15 * pyo.units.K
                )
                P_out_list[i] = m.fs.unit.treated.conc_mass_comp[0, "S_PO4"].value
                # LCOW_matrix[i, j] = pyo.value(m.fs.costing.LCOW)
                # SEC_matrix[i, j] = pyo.value(m.fs.costing.specific_energy_consumption)
            except:
                pass

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    CF = ax3.contourf(CP_list, r_AV_list, P_out_matrix, cmap="GnBu")
    ax3.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax3.set_ylabel("Area Volume Ratio", fontsize=12)
    cbar = fig3.colorbar(CF)
    cbar.ax.set_ylabel("Concentration of PO4 in the treated water (g/L)")

    # fig4, ax4 = plt.subplots(figsize=(7, 5))
    # CF = ax4.contourf(CP_list, r_AV_list, LCOW_matrix, cmap="GnBu")
    # ax4.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # ax4.set_ylabel("Area Volume Ratio", fontsize=12)
    # cbar = fig4.colorbar(CF)
    # cbar.ax.set_ylabel('LCOW ($/m3)')

    # fig5, ax5 = plt.subplots(figsize=(7, 5))
    # CF = ax5.contourf(CP_list, r_AV_list, SEC_matrix, cmap="GnBu")
    # ax5.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # ax5.set_ylabel("Area Volume Ratio", fontsize=12)
    # cbar = fig5.colorbar(CF)
    # cbar.ax.set_ylabel('SEC (kWh/m3)')

    # plt.switch_backend('TkAgg')
    plt.show()


if __name__ == "__main__":
    # m, results = main(CP=-1.05 * pyo.units.V, r_AV=0.09)
    plot(num=20)
