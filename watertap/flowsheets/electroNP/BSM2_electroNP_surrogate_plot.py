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
from watertap.flowsheets.electroNP.BSM2_electroNP_surrogate import (
    build_flowsheet,
    set_operating_conditions,
    initialize_system,
    solve,
    add_costing,
)
import matplotlib.pyplot as plt


def main(CP=-1.05 * pyo.units.V, r_AV=0.105):
    m = build_flowsheet(has_electroNP=True)
    set_operating_conditions(m)

    m.fs.electroNP.cathodic_potential.unfix()
    m.fs.electroNP.area_volume_ratio.unfix()

    m.fs.electroNP.cathodic_potential.fix(CP)
    m.fs.electroNP.area_volume_ratio.fix(r_AV)

    for mx in m.fs.mixers:
        mx.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 3].deactivate()
    print(f"DOF before initialization: {degrees_of_freedom(m)}")

    initialize_system(m, has_electroNP=True)
    for mx in m.fs.mixers:
        mx.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 3].deactivate()
    print(f"DOF after initialization: {degrees_of_freedom(m)}")

    results = solve(m)

    add_costing(m, has_electroNP=True)
    m.fs.costing.initialize()

    interval_initializer(m.fs.costing)

    assert_degrees_of_freedom(m, 0)

    results = solve(m)

    return m, results


def plot(num):
    CP_list = np.linspace(-1.3, -0.8, num)
    r_AV_list = np.linspace(0.08, 0.11, num)

    P_out_list = np.zeros(num)
    P_out_matrix = np.zeros((num, num))
    P_out_matrix[:] = np.nan
    LCOW_matrix = np.zeros((num, num))
    LCOW_matrix[:] = np.nan
    SEC_matrix = np.zeros((num, num))
    SEC_matrix[:] = np.nan

    for i in range(0, num):
        try:
            m, results = main(CP=CP_list[i], r_AV=0.105)
            P_out_list[i] = (
                m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
            )
        except:
            pass

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(CP_list, P_out_list, "b")
    # ax1.set_xlim([0.02, 0.05])
    ax1.set_xlabel("Cathodic Potential (V)", fontsize=12)
    ax1.set_ylabel("Concentration of PO4 (mg/L)", fontsize=12)
    # ax1.set_title("Low Pressure Side Outlet")

    # for i in range(0, num):
    #     m, results = main(
    #         CP = -1.05,
    #         r_AV = r_AV_list[i]
    #     )
    #
    #
    #
    #     P_out_list[i] = m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
    #
    # fig2, ax2 = plt.subplots(figsize=(7, 5))
    # ax2.plot(r_AV_list, P_out_list, "b")
    # # ax2.set_xlim([0.02, 0.05])
    # ax2.set_xlabel("Area Volume Ratio", fontsize=12)
    # ax2.set_ylabel("Concentration of PO4 (mg/L)", fontsize=12)

    # for i in range(0, num):
    #     for j in range(0, num):
    #         print(f"CP: {CP_list[i]}")
    #         print(f"rAV: {r_AV_list[j]}")
    #         try:
    #             m, results = main(CP=CP_list[i], r_AV=r_AV_list[j])
    #             P_out_matrix[i, j] = m.fs.Treated.properties[0].conc_mass_comp["S_PO4"].value * 1e3
    #             LCOW_matrix[i, j] = pyo.value(m.fs.costing.LCOW)
    #             SEC_matrix[i, j] = pyo.value(m.fs.costing.specific_energy_consumption)
    #         except:
    #             pass
    #
    # # fig3, ax3 = plt.subplots(figsize=(7, 5))
    # # CF = ax3.contourf(CP_list, r_AV_list, P_out_matrix, cmap="GnBu")
    # # ax3.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # # ax3.set_ylabel("Area Volume Ratio", fontsize=12)
    # # cbar = fig3.colorbar(CF)
    # # cbar.ax.set_ylabel('Concentration of PO4 in the treated water (mg/L)')
    #
    # # fig4, ax4 = plt.subplots(figsize=(7, 5))
    # # CF = ax4.contourf(CP_list, r_AV_list, LCOW_matrix, cmap="GnBu")
    # # ax4.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # # ax4.set_ylabel("Area Volume Ratio", fontsize=12)
    # # cbar = fig4.colorbar(CF)
    # # cbar.ax.set_ylabel('LCOW ($/m3)')
    #
    # fig5, ax5 = plt.subplots(figsize=(7, 5))
    # CF = ax5.contourf(CP_list, r_AV_list, SEC_matrix, cmap="GnBu")
    # ax5.set_xlabel("Cathodic Potential (V)", fontsize=12)
    # ax5.set_ylabel("Area Volume Ratio", fontsize=12)
    # cbar = fig5.colorbar(CF)
    # cbar.ax.set_ylabel('SEC (kWh/m3)')

    plt.show()


if __name__ == "__main__":
    # m, results = main(CP=-1.05 * pyo.units.V, r_AV=0.09)
    plot(num=10)
