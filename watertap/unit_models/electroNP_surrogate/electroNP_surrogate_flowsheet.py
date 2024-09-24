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

import pytest
import idaes.logger as idaeslog
import pyomo.environ as pyo
from idaes.core.util.tables import (
    create_stream_table_dataframe,
    stream_table_dataframe_to_string,
)
from idaes.core import FlowsheetBlock
from watertap.unit_models.electroNP_surrogate.electroNP_surrogate import ElectroNP

# from watertap.property_models.unit_specific.activated_sludge.simple_modified_asm2d_properties import (
#     SimpleModifiedASM2dParameterBlock,
# )
from watertap.property_models.unit_specific.activated_sludge.modified_asm2d_properties import (
    ModifiedASM2dParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.modified_asm2d_reactions import (
    ModifiedASM2dReactionParameterBlock,
)
from watertap.core.util.initialization import (
    check_solve,
    assert_degrees_of_freedom,
    interval_initializer,
)
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.testing import initialization_tester
from idaes.core.util.scaling import calculate_scaling_factors
from pyomo.util.check_units import assert_units_consistent
from idaes.core import UnitModelCostingBlock
from watertap.costing import WaterTAPCosting
import idaes.core.util.scaling as iscale

from idaes.core.util.model_diagnostics import DegeneracyHunter
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from pyomo.environ import *


def main():
    m = build_flowsheet()

    # results = DiagnosticsToolbox(m)
    # results.report_structural_issues()
    # results.report_numerical_issues()
    # results.display_constraints_with_large_residuals()

    # # Use of Degeneracy Hunter for troubleshooting model.
    # m.obj = pyo.Objective(expr=0)
    # solver = get_solver()
    # solver.options["max_iter"] = 10000
    # results = solver.solve(m, tee=True)
    # dh = DegeneracyHunter(m, solver=pyo.SolverFactory("cbc"))
    # badly_scaled_var_list = iscale.badly_scaled_var_generator(m, large=1e1, small=1e-1)
    # for x in badly_scaled_var_list:
    #     print(f"{x[0].name}\t{x[0].value}\tsf: {iscale.get_scaling_factor(x[0])}")
    # dh.check_residuals(tol=1e-8)

    print("----------------   scaling V0  ----------------")
    badly_scaled_var_list = iscale.badly_scaled_var_generator(m, large=1e1, small=1e-1)
    for x in badly_scaled_var_list:
        print(f"{x[0].name}\t{x[0].value}\tsf: {iscale.get_scaling_factor(x[0])}")

    print("---Structural Issues---")
    dt = DiagnosticsToolbox(m)
    dt.report_structural_issues()
    # dt.display_potential_evaluation_errors()

    # m.fs.unit.initialize(outlvl=idaeslog.INFO_HIGH)
    m.fs.unit.initialize(solver="ipopt-watertap")

    # Costing
    # add_costing(m)
    # m.fs.costing.initialize()

    # Get default solver for testing
    results = solve(m)

    print("---Numerical Issues---")
    dt.report_numerical_issues()
    # dt.compute_infeasibility_explanation()
    # dt.display_variables_at_or_outside_bounds()
    # dt.display_constraints_with_large_residuals()
    # dt.display_variables_with_extreme_jacobians()
    # dt.display_constraints_with_extreme_jacobians()

    return m, results


def build_flowsheet():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = ModifiedASM2dParameterBlock()

    m.fs.unit = ElectroNP(property_package=m.fs.properties)

    EPS = 1e-10

    m.fs.unit.inlet.temperature.fix(308.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.flow_vol.fix(0.0028385 * units.m**3 / units.s)

    m.fs.unit.inlet.conc_mass_comp[0, "S_A"].fix(8.4692 * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_F"].fix(22.083 * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.057262 * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_N2"].fix(EPS * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH4"].fix(2.0103 * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO3"].fix(EPS * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O2"].fix(EPS * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_PO4"].fix(67.379 * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(1.0923 * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(0.74048 * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(1.2161 * units.g / units.liter)

    m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(EPS * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_H"].fix(EPS * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(0.30624 * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(EPS * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(EPS * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(EPS * units.g / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(0.070202 * units.g / units.liter)

    # # Alkalinity was givien in mg/L based on C
    # m.fs.unit.inlet.alkalinity[0].fix(61 / 12 * units.mmol / units.liter)

    # Unit option
    # m.fs.unit.energy_electric_flow_mass.fix(0.044 * units.kWh / units.kg)
    m.fs.unit.magnesium_chloride_dosage.fix(0.388)

    m.fs.unit.cathodic_potential.fix(-1.05 * units.V)
    m.fs.unit.area_volume_ratio.fix(0.105)
    m.fs.unit.settling_time.fix(30 * units.min)

    m.fs.unit.frac_mass_H2O_treated[0].fix(0.9)

    # m.fs.unit.cathodic_potential.fix(-1.05)
    # m.fs.unit.area_volume_ratio.fix(0.105)
    # m.fs.unit.T.fix(25)
    # m.fs.unit.settling_time.fix(30)

    iscale.set_scaling_factor(m.fs.unit.CP_surrogate, 1e0)
    iscale.set_scaling_factor(m.fs.unit.r_AV_surrogate, 1e0)
    iscale.set_scaling_factor(m.fs.unit.T_surrogate, 1e-1)
    iscale.set_scaling_factor(m.fs.unit.t_ss_surrogate, 1e-1)

    def scale_variables(m):
        for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
            if "flow_vol" in var.name:
                iscale.set_scaling_factor(var, 1e0)
            if "temperature" in var.name:
                iscale.set_scaling_factor(var, 1e-2)
            if "pressure" in var.name:
                iscale.set_scaling_factor(var, 1e-5)
            if "conc_mass_comp" in var.name:
                iscale.set_scaling_factor(var, 1e1)

    # m.fs.properties.set_default_scaling("pressure", 1e-5)
    # m.fs.properties.set_default_scaling("temperature", 1e-2)
    # m.fs.properties.set_default_scaling("flow_vol", 1)
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_O2"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_N2"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_NH4"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_NO3"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_PO4"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_F"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_A"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_I"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_I"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_S"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_H"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_PAO"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_PP"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_PHA"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_AUT"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_MeOH"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_MeP"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_TSS"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_K"))
    # m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_Mg"))
    # m.fs.properties.set_default_scaling("alkalinity", 1)

    scale_variables(m)

    calculate_scaling_factors(m)

    return m


def solve(m, solver=None):
    if solver is None:
        solver = get_solver()
    results = solver.solve(m, tee=True)
    pyo.assert_optimal_termination(results)
    return results


def add_costing(m):
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyo.units.USD_2020

    m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)


if __name__ == "__main__":
    m, results = main()
    print(f"P removal: {m.fs.unit.P_removal.value}")
    stream_table = create_stream_table_dataframe(
        {
            "electroNP inlet": m.fs.unit.inlet,
            "electroNP treated": m.fs.unit.treated,
            "electroNP byproduct": m.fs.unit.byproduct,
        },
        time_point=0,
    )
    print(stream_table_dataframe_to_string(stream_table))
