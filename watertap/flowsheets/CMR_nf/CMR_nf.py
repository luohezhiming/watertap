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

import re
import pytest

import pyomo.environ as pyo
from pyomo.environ import (
    assert_optimal_termination,
    ConcreteModel,
    Constraint,
    Set,
    Suffix,
    TransformationFactory,
    value,
    Var,
)
from pyomo.network import Port

from idaes.core import EnergyBalanceType, FlowsheetBlock, MomentumBalanceType
from idaes.core.initialization import InitializationStatus
from idaes.core.scaling import set_scaling_factor
from idaes.core.solvers import get_solver
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.testing import PhysicalParameterTestBlock

from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
)
from watertap.unit_models.nanofiltration_0D import (
    Nanofiltration0D,
    Nanofiltration0DInitializer,
    Nanofiltration0DScaler,
)

__author__ = "Chenyu Wang"


def CMR_nf_case():
    m = build()
    set_scaling(m)
    initialize_system(m)
    m, results = solve(m)
    display_performance_metrics(m)

    return m, results


def build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=[
            "Co_2+",
            "Ca_2+",
            "Cu_2+",
            "Fe_3+",
            "Nd_3+",
            "Ni_2+",
            "Pr_3+",
            "Na_+",
            "Cr_6+",
            "Sn_2+",
            "Zn_2+",
            "Pb_2+",
            "Dy_3+",
        ],
        # diffusivity_data={
        #     ("Liq", "Ca_2+"): 9.2e-10,
        #     ("Liq", "Mg_2+"): 7.06e-10,
        #     ("Liq", "Na_+"): 1.33e-09,
        #     ("Liq", "Cl_-"): 2.03e-09,
        # },
        # mw_data={
        #     "H2O": 0.018,
        #     "Ca_2+": 0.04,
        #     "Mg_2+": 0.024,
        #     "Na_+": 0.023,
        #     "Cl_-": 0.035,
        # },
        # stokes_radius_data={
        #     "Ca_2+": 3.09e-10,
        #     "Mg_2+": 3.47e-10,
        #     "Cl_-": 1.21e-10,
        #     "Na_+": 1.84e-10,
        # },
        charge={
            "Co_2+": 2,
            "Ca_2+": 2,
            "Cu_2+": 2,
            "Fe_3+": 3,
            "Nd_3+": 3,
            "Ni_2+": 2,
            "Pr_3+": 3,
            "Na_+": 1,
            "Cr_6+": 6,
            "Sn_2+": 2,
            "Zn_2+": 2,
            "Pb_2+": 2,
            "Dy_3+": 3,
        },
    )

    m.fs.unit = Nanofiltration0D(
        property_package=m.fs.properties,
        electroneutrality_ion=None,
        has_pressure_change=True,
    )

    # Fix other inlet state variables
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "H2O"].fix(53.6036)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Co_2+"].fix(0.01)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Ca_2+"].fix(0.019)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Cu_2+"].fix(0.001)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Fe_3+"].fix(1.190)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Nd_3+"].fix(0.020)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Ni_2+"].fix(0.012)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Pr_3+"].fix(0.006)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(0.005)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Cr_6+"].fix(0.001)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Sn_2+"].fix(0.001)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Zn_2+"].fix(0.002)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Pb_2+"].fix(0.001)
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Dy_3+"].fix(0.001)
    m.fs.unit.inlet.temperature[0].fix(298.15)
    m.fs.unit.inlet.pressure[0].fix(101325)

    m.fs.unit.recovery_solvent.fix(1)
    # m.fs.unit.rejection_comp.fix()
    m.fs.unit.rejection_comp[0, "Co_2+"].fix(0.98)
    m.fs.unit.rejection_comp[0, "Ca_2+"].fix(0.92)
    m.fs.unit.rejection_comp[0, "Cu_2+"].fix(0.98)
    m.fs.unit.rejection_comp[0, "Fe_3+"].fix(0.98)
    m.fs.unit.rejection_comp[0, "Nd_3+"].fix(0.9945)
    m.fs.unit.rejection_comp[0, "Ni_2+"].fix(0.98)
    m.fs.unit.rejection_comp[0, "Pr_3+"].fix(0.5)
    m.fs.unit.rejection_comp[0, "Na_+"].fix(0.796)
    m.fs.unit.rejection_comp[0, "Cr_6+"].fix(0.93)
    m.fs.unit.rejection_comp[0, "Sn_2+"].fix(0.5)
    m.fs.unit.rejection_comp[0, "Zn_2+"].fix(0.98)
    m.fs.unit.rejection_comp[0, "Pb_2+"].fix(0.99)
    m.fs.unit.rejection_comp[0, "Dy_3+"].fix(0.5)
    m.fs.unit.area.fix(500)
    m.fs.unit.deltaP.fix(0)
    # m.fs.unit.permeate.pressure[0].fix(101325)

    return m


def set_scaling(mcas_case):
    # Scale model
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "H2O"], 1e-1
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Co_2+"], 1e2
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Ca_2+"], 1e2
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Cu_2+"], 1e2
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Fe_3+"], 1e0
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Nd_3+"], 1e2
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Ni_2+"], 1e2
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Pr_3+"], 1e2
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Na_+"], 1e2
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Cr_6+"], 1e2
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Sn_2+"], 1e2
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Zn_2+"], 1e2
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Pb_2+"], 1e2
    )
    set_scaling_factor(
        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Dy_3+"], 1e2
    )

    scaler = Nanofiltration0DScaler()
    scaler.scale_model(mcas_case.fs.unit)


def initialize_system(mcas_case):
    # Initialize system
    initializer = Nanofiltration0DInitializer()
    initializer.initialize(mcas_case.fs.unit)


def solve(m):
    # Solve
    solver = get_solver(
        "ipopt_v2", writer_config={"scale_model": True, "linear_presolve": True}
    )
    results = solver.solve(m)
    assert_optimal_termination(results)

    return m, results


def display_performance_metrics(m):
    f_in = pyo.units.convert(
        m.fs.unit.properties_in[0].flow_vol, to_units=pyo.units.m**3 / pyo.units.hr
    )
    print(f"Influent flow: " f"{pyo.value(f_in):.2f}" f"{pyo.units.get_units(f_in)}")
    f_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].flow_vol,
        to_units=pyo.units.m**3 / pyo.units.hr,
    )
    print(
        f"Permeate flow: " f"{pyo.value(f_permeate):.2f}" f"{pyo.units.get_units(f_in)}"
    )


if __name__ == "__main__":
    m, results = CMR_nf_case()
