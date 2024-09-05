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
from pyomo.environ import (
    ConcreteModel,
    assert_optimal_termination,
    value,
    units,
)
from idaes.core import FlowsheetBlock
from watertap.unit_models.electroNP_surrogate.electroNP_surrogate import ElectroNP
from watertap.property_models.unit_specific.activated_sludge.simple_modified_asm2d_properties import (
    SimpleModifiedASM2dParameterBlock,
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


def build_flowsheet():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = SimpleModifiedASM2dParameterBlock(
        additional_solute_list=["S_K", "S_Mg"]
    )

    m.fs.unit = ElectroNP(property_package=m.fs.properties)

    EPS = 1e-8

    m.fs.unit.inlet.temperature.fix(298.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.flow_vol.fix(18446 * units.m**3 / units.day)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O2"].fix(10 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_N2"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH4"].fix(16 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO3"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_PO4"].fix(3.6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_F"].fix(30 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_A"].fix(20 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(30 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(25 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(125 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_H"].fix(30 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_MeOH"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_MeP"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_TSS"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(EPS * units.mg / units.liter)

    # Alkalinity was givien in mg/L based on C
    m.fs.unit.inlet.alkalinity[0].fix(61 / 12 * units.mmol / units.liter)

    # Unit option
    # m.fs.unit.energy_electric_flow_mass.fix(0.044 * units.kWh / units.kg)
    m.fs.unit.magnesium_chloride_dosage.fix(0.388)

    m.fs.unit.cathodic_potential.fix(-1.05 * units.V)
    m.fs.unit.area_volume_ratio.fix(0.105)
    m.fs.unit.settling_time.fix(30 * units.min)

    # m.fs.unit.cathodic_potential.fix(-1.05)
    # m.fs.unit.area_volume_ratio.fix(0.105)
    # m.fs.unit.T.fix(25)
    # m.fs.unit.settling_time.fix(30)

    iscale.set_scaling_factor(m.fs.unit.CP_surrogate, 1e0)
    iscale.set_scaling_factor(m.fs.unit.r_AV_surrogate, 1e0)
    iscale.set_scaling_factor(m.fs.unit.T_surrogate, 1e-1)
    iscale.set_scaling_factor(m.fs.unit.t_ss_surrogate, 1e-1)

    m.fs.properties.set_default_scaling("pressure", 1e-5)
    m.fs.properties.set_default_scaling("temperature", 1e-1)
    m.fs.properties.set_default_scaling("flow_vol", 1)
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_O2"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_N2"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_NH4"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_NO3"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_PO4"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_F"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_A"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_I"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_I"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_S"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_H"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_PAO"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_PP"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_PHA"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_AUT"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_MeOH"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_MeP"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("X_TSS"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_K"))
    m.fs.properties.set_default_scaling("conc_mass_comp", 1e1, index=("S_Mg"))
    m.fs.properties.set_default_scaling("alkalinity", 1)

    calculate_scaling_factors(m)

    # results = DiagnosticsToolbox(m)
    # results.report_structural_issues()
    # results.report_numerical_issues()
    # results.display_constraints_with_large_residuals()

    # # Use of Degeneracy Hunter for troubleshooting model.
    # m.obj = pyo.Objective(expr=0)
    # solver = get_solver()
    # solver.options["max_iter"] = 100000
    # results = solver.solve(m, tee=True)
    # dh = DegeneracyHunter(m, solver=pyo.SolverFactory("cbc"))
    # badly_scaled_var_list = iscale.badly_scaled_var_generator(m, large=1e1, small=1e-1)
    # for x in badly_scaled_var_list:
    #     print(f"{x[0].name}\t{x[0].value}\tsf: {iscale.get_scaling_factor(x[0])}")
    # dh.check_residuals(tol=1e-8)

    m.fs.unit.initialize(outlvl=idaeslog.INFO_HIGH)

    # Get default solver for testing
    solver = get_solver()
    results = solver.solve(m, tee=True)

    return m, results


if __name__ == "__main__":
    m, results = build_flowsheet()
