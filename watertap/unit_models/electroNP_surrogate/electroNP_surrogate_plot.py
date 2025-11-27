import matplotlib

matplotlib.use("TkAgg")
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

# import tkinter
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
    r_AV_list = np.linspace(0.065, 0.145, num)
    T_list = np.linspace(281.15, 315.15, num)

    P_out_list = np.zeros(num)
    P_out_matrix = np.zeros((num, num))
    P_out_matrix[:] = np.nan

    P_removal_matrix = np.zeros((num, num))
    P_removal_matrix[:] = np.nan

    EI_matrix = np.zeros((num, num))
    EI_matrix[:] = np.nan

    # LCOW_matrix = np.zeros((num, num))
    # LCOW_matrix[:] = np.nan
    # SEC_matrix = np.zeros((num, num))
    # SEC_matrix[:] = np.nan

    # for i in range(0, num):
    #     m, results = main(CP=CP_list[i], r_AV=0.105, T=308.15 * pyo.units.K)
    #
    #     P_out_list[i] = m.fs.unit.treated.conc_mass_comp[0, "S_PO4"].value
    #
    # fig1, ax1 = plt.subplots(figsize=(7, 5))
    # ax1.plot(CP_list, P_out_list, "b")
    # # ax1.set_xlim([0.02, 0.05])
    # ax1.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # ax1.set_ylabel("Concentration of PO4 (g/L)", fontsize=12)
    # # ax1.set_title("Low Pressure Side Outlet")
    #
    # for i in range(0, num):
    #     try:
    #         m, results = main(CP=-1.05, r_AV=r_AV_list[i], T=308.15 * pyo.units.K)
    #
    #         P_out_list[i] = m.fs.unit.treated.conc_mass_comp[0, "S_PO4"].value
    #     except:
    #         pass
    #
    # fig2, ax2 = plt.subplots(figsize=(7, 5))
    # ax2.plot(r_AV_list, P_out_list, "b")
    # # ax2.set_xlim([0.02, 0.05])
    # ax2.set_xlabel("Area Volume Ratio", fontsize=12)
    # ax2.set_ylabel("Concentration of PO4 (g/L)", fontsize=12)

    for i in range(0, num):
        for j in range(0, num):
            print(f"CP: {CP_list[i]}")
            print(f"rAV: {r_AV_list[j]}")
            try:
                m, results = main(
                    CP=CP_list[i], r_AV=r_AV_list[j], T=308.15 * pyo.units.K
                )
                P_removal_matrix[j, i] = m.fs.unit.P_removal.value
                EI_matrix[j, i] = m.fs.unit.energy_electric_flow_mass.value
                # LCOW_matrix[i, j] = pyo.value(m.fs.costing.LCOW)
                # SEC_matrix[i, j] = pyo.value(m.fs.costing.specific_energy_consumption)
            except:
                pass

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    CF = ax3.contourf(CP_list, r_AV_list, P_removal_matrix, cmap="GnBu")
    ax3.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax3.set_ylabel("Area Volume Ratio", fontsize=12)
    cbar = fig3.colorbar(CF)
    cbar.ax.set_ylabel("Phosphorus Recovery")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    CF = ax3.contourf(CP_list, r_AV_list, EI_matrix, cmap="GnBu")
    ax3.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax3.set_ylabel("Area Volume Ratio", fontsize=12)
    cbar = fig3.colorbar(CF)
    cbar.ax.set_ylabel("Electricity Intensity of ElectroN-P (kWh/kg)")

    # plt.switch_backend('TkAgg')
    plt.show()

    PR_max = np.nanmax(P_removal_matrix)
    PR_min = np.nanmin(P_removal_matrix)
    EI_max = np.nanmax(EI_matrix)
    EI_min = np.nanmin(EI_matrix)
    return PR_max, PR_min, EI_max, EI_min


if __name__ == "__main__":
    # m, results = main(CP=-1.05 * pyo.units.V, r_AV=0.09)
    PR_max, PR_min, EI_max, EI_min = plot(num=20)
    print(f"PR max: {PR_max}")
    print(f"PR min: {PR_min}")
    print(f"EI max: {EI_max}")
    print(f"EI min: {EI_min}")
