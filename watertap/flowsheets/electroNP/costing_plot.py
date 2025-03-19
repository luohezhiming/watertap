import pyomo.environ as pyo
import numpy as np
from watertap.core.util.initialization import (
    check_solve,
    assert_degrees_of_freedom,
    interval_initializer,
)
from pyomo.environ import (
    ConcreteModel,
    units,
    Block,
    Var,
    Constraint,
    value,
    Expression,
    NonNegativeReals,
)
from pyomo.network import Port
from idaes.core import FlowsheetBlock
from watertap.unit_models.dewatering import DewateringUnit, ActivatedSludgeModelType
from watertap.property_models.unit_specific.activated_sludge.asm1_properties import (
    ASM1ParameterBlock,
)

from watertap.property_models.unit_specific.activated_sludge.asm2d_properties import (
    ASM2dParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.modified_asm2d_properties import (
    ModifiedASM2dParameterBlock,
)
from pyomo.util.check_units import assert_units_consistent
from watertap.costing import WaterTAPCosting
from idaes.core import UnitModelCostingBlock
from watertap.costing.unit_models.dewatering import (
    cost_centrifuge,
)


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
import matplotlib
from scipy import interpolate

solver = get_solver()


def WaterTAP_centrifuge(F_in):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    m.fs.unit = DewateringUnit(property_package=m.fs.props)

    # m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m ** 3 / units.day)
    m.fs.unit.inlet.flow_vol.fix(F_in)
    m.fs.unit.inlet.temperature.fix(308.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216.2434 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(1442.7882 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
    m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyo.units.USD_2023

    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing, costing_method=cost_centrifuge
    )

    m.fs.costing.cost_process()

    solver.solve(m)

    return m


def New_centrifuge(F_in):

    PEC = 100000 * pyo.units.USD_2023
    F_d = 16 * units.m**3 / units.hr
    a = 0.49
    k = 798.7 / 567.3

    cap = k * PEC * (pyo.units.convert(F_in, to_units=units.m**3 / units.hr) / F_d) ** a
    return cap


def plot(num):
    F_in_low = 10
    F_in_high = 1000
    F_in_list = np.linspace(F_in_low, F_in_high, num)
    cap_WaterTAP_list = np.zeros(num)
    cap_data_list = np.zeros(num)

    for i in range(0, num):
        # try:
        m = WaterTAP_centrifuge(F_in=(F_in_list[i] * units.m**3 / units.hr))
        cap_WaterTAP_list[i] = value(m.fs.unit.costing.capital_cost) / 1e6
        data_cap = New_centrifuge(F_in_list[i] * units.m**3 / units.hr)
        cap_data_list[i] = value(data_cap / 1e6)
        # except:
        #     pass

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    # ax1.plot(F_in_list, cap_WaterTAP_list, "b")
    ax1.plot(F_in_list, cap_data_list, "r")
    # ax1.set_xlim([0.02, 0.05])
    ax1.set_xlabel("Inlet flowrate (m3/hr)", fontsize=12)
    ax1.set_ylabel("capital cost (M$)", fontsize=12)
    ax1.set_title("Centrifuge")
    # ax1.legend(
    #         ["WaterTAP", "data"]
    #     )

    # plt.show()
    # plt.show(block=True)

    return cap_data_list


if __name__ == "__main__":
    # m, results = main(CP=-1.05 * pyo.units.V, r_AV=0.09)
    # F_in = 15.9 * units.m ** 3 / units.hr
    # m = WaterTAP_centrifuge(F_in)
    # cap = New_centrifuge(F_in)
    # print(value(m.fs.unit.costing.capital_cost))
    # print(value(cap))
    cap_data_list = plot(5)
    plt.show(block=True)
