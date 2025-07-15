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
    m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "H2O"].fix(1e3)
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
    m.fs.unit.rejection_comp[0, "Pr_3+"].fix(0.95)  # Pr 59; Nd 60
    m.fs.unit.rejection_comp[0, "Na_+"].fix(0.796)
    m.fs.unit.rejection_comp[0, "Cr_6+"].fix(0.93)
    m.fs.unit.rejection_comp[0, "Sn_2+"].fix(0.95)  # Sn 50; Nd 60; Zn 30
    m.fs.unit.rejection_comp[0, "Zn_2+"].fix(0.98)
    m.fs.unit.rejection_comp[0, "Pb_2+"].fix(0.99)
    m.fs.unit.rejection_comp[0, "Dy_3+"].fix(0.95)  # Dy 66; Nd 60
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
    print("---- System Performance Metrics ----")
    # f_in = pyo.units.convert(
    #     m.fs.unit.properties_in[0].flow_vol, to_units=pyo.units.m**3 / pyo.units.hr
    # )
    # print(f"Influent flow: " f"{pyo.value(f_in):.3g}" f"{pyo.units.get_units(f_in)}")
    # f_permeate = pyo.units.convert(
    #     m.fs.unit.properties_permeate[0].flow_vol,
    #     to_units=pyo.units.m**3 / pyo.units.hr,
    # )
    # print(
    #     f"Permeate flow: "
    #     f"{pyo.value(f_permeate):.3g}"
    #     f"{pyo.units.get_units(f_permeate)}"
    # )
    # f_retentate = pyo.units.convert(
    #     m.fs.unit.properties_retentate[0].flow_vol,
    #     to_units=pyo.units.m**3 / pyo.units.hr,
    # )
    # print(
    #     f"Retentate flow: "
    #     f"{pyo.value(f_retentate):.3g}"
    #     f"{pyo.units.get_units(f_retentate)}"
    # )
    water_recovery = m.fs.unit.recovery_vol_phase[0, "Liq"]
    print(f"Volumetric-based recovery: " f"{pyo.value(water_recovery):.3g}")
    Co_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Co_2+"]
    print(f"Co2+ mass rejection: " f"{pyo.value(1-Co_recovery):.3g}")
    Ca_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Ca_2+"]
    print(f"Ca2+ mass rejection: " f"{pyo.value(1 - Ca_recovery):.3g}")
    Cu_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Cu_2+"]
    print(f"Cu2+ mass rejection: " f"{pyo.value(1 - Cu_recovery):.3g}")
    Fe_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Fe_3+"]
    print(f"Fe3+ mass rejection: " f"{pyo.value(1 - Fe_recovery):.3g}")
    Nd_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Nd_3+"]
    print(f"Nd3+ mass rejection: " f"{pyo.value(1 - Nd_recovery):.3g}")
    Ni_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Ni_2+"]
    print(f"Ni2+ mass rejection: " f"{pyo.value(1 - Ni_recovery):.3g}")
    Pr_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Pr_3+"]
    print(f"Pr3+ mass rejection: " f"{pyo.value(1 - Pr_recovery):.3g}")
    Na_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Na_+"]
    print(f"Na+ mass rejection: " f"{pyo.value(1 - Na_recovery):.3g}")
    Cr_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Cr_6+"]
    print(f"Cr6+ mass rejection: " f"{pyo.value(1 - Cr_recovery):.3g}")
    Sn_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Sn_2+"]
    print(f"Sn2+ mass rejection: " f"{pyo.value(1 - Sn_recovery):.3g}")
    Zn_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Zn_2+"]
    print(f"Zn2+ mass rejection: " f"{pyo.value(1 - Zn_recovery):.3g}")
    Pb_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Pb_2+"]
    print(f"Pb2+ mass rejection: " f"{pyo.value(1 - Pb_recovery):.3g}")
    Dy_recovery = m.fs.unit.recovery_mass_phase_comp[0, "Liq", "Dy_3+"]
    print(f"Dy3+ mass rejection: " f"{pyo.value(1 - Dy_recovery):.3g}")

    print("\n---- Feed Metrics ----")
    f_in = pyo.units.convert(
        m.fs.unit.properties_in[0].flow_vol, to_units=pyo.units.m**3 / pyo.units.hr
    )
    print(f"Influent flow: " f"{pyo.value(f_in):.3g}" f"{pyo.units.get_units(f_in)}")
    f_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].flow_vol,
        to_units=pyo.units.m**3 / pyo.units.hr,
    )
    Co_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Co_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Co2+ feed mass concentration: "
        f"{pyo.value(Co_in):.3g}"
        f"{pyo.units.get_units(Co_in)}"
    )
    Ca_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Ca_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Ca2+ feed mass concentration: "
        f"{pyo.value(Ca_in):.3g}"
        f"{pyo.units.get_units(Ca_in)}"
    )
    Cu_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Cu_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Cu2+ feed mass concentration: "
        f"{pyo.value(Cu_in):.3g}"
        f"{pyo.units.get_units(Cu_in)}"
    )
    Fe_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Fe_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Fe3+ feed mass concentration: "
        f"{pyo.value(Fe_in):.3g}"
        f"{pyo.units.get_units(Fe_in)}"
    )
    Nd_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Nd_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Nd3+ feed mass concentration: "
        f"{pyo.value(Nd_in):.3g}"
        f"{pyo.units.get_units(Nd_in)}"
    )
    Ni_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Ni_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Ni2+ feed mass concentration: "
        f"{pyo.value(Ni_in):.3g}"
        f"{pyo.units.get_units(Ni_in)}"
    )
    Pr_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Pr_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Pr3+ feed mass concentration: "
        f"{pyo.value(Pr_in):.3g}"
        f"{pyo.units.get_units(Pr_in)}"
    )
    Na_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Na_+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Na+ feed mass concentration: "
        f"{pyo.value(Na_in):.3g}"
        f"{pyo.units.get_units(Na_in)}"
    )
    Cr_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Cr_6+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Cr6+ feed mass concentration: "
        f"{pyo.value(Cr_in):.3g}"
        f"{pyo.units.get_units(Cr_in)}"
    )
    Sn_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Sn_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Sn2+ feed mass concentration: "
        f"{pyo.value(Sn_in):.3g}"
        f"{pyo.units.get_units(Sn_in)}"
    )
    Zn_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Zn_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Zn2+ feed mass concentration: "
        f"{pyo.value(Zn_in):.3g}"
        f"{pyo.units.get_units(Zn_in)}"
    )
    Pb_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Pb_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Pb2+ feed mass concentration: "
        f"{pyo.value(Pb_in):.3g}"
        f"{pyo.units.get_units(Pb_in)}"
    )
    Dy_in = pyo.units.convert(
        m.fs.unit.properties_in[0].conc_mass_phase_comp["Liq", "Dy_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Dy3+ feed mass concentration: "
        f"{pyo.value(Dy_in):.3g}"
        f"{pyo.units.get_units(Dy_in)}"
    )

    print("\n---- Permeate Metrics ----")
    f_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].flow_vol,
        to_units=pyo.units.m**3 / pyo.units.hr,
    )
    print(
        f"Permeate flow: "
        f"{pyo.value(f_permeate):.3g}"
        f"{pyo.units.get_units(f_permeate)}"
    )
    Co_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Co_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Co2+ permeate mass concentration: "
        f"{pyo.value(Co_permeate):.3g}"
        f"{pyo.units.get_units(Co_permeate)}"
    )
    Ca_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Ca_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Ca2+ permeate mass concentration: "
        f"{pyo.value(Ca_permeate):.3g}"
        f"{pyo.units.get_units(Ca_permeate)}"
    )
    Cu_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Cu_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Cu2+ permeate mass concentration: "
        f"{pyo.value(Cu_permeate):.3g}"
        f"{pyo.units.get_units(Cu_permeate)}"
    )
    Fe_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Fe_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Fe3+ permeate mass concentration: "
        f"{pyo.value(Fe_permeate):.3g}"
        f"{pyo.units.get_units(Fe_permeate)}"
    )
    Nd_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Nd_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Nd3+ permeate mass concentration: "
        f"{pyo.value(Nd_permeate):.3g}"
        f"{pyo.units.get_units(Nd_permeate)}"
    )
    Ni_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Ni_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Ni2+ permeate mass concentration: "
        f"{pyo.value(Ni_permeate):.3g}"
        f"{pyo.units.get_units(Ni_permeate)}"
    )
    Pr_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Pr_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Pr3+ permeate mass concentration: "
        f"{pyo.value(Pr_permeate):.3g}"
        f"{pyo.units.get_units(Pr_permeate)}"
    )
    Na_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Na_+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Na+ permeate mass concentration: "
        f"{pyo.value(Na_permeate):.3g}"
        f"{pyo.units.get_units(Na_permeate)}"
    )
    Cr_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Cr_6+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Cr6+ permeate mass concentration: "
        f"{pyo.value(Cr_permeate):.3g}"
        f"{pyo.units.get_units(Cr_permeate)}"
    )
    Sn_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Sn_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Sn2+ permeate mass concentration: "
        f"{pyo.value(Sn_permeate):.3g}"
        f"{pyo.units.get_units(Sn_permeate)}"
    )
    Zn_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Zn_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Zn2+ permeate mass concentration: "
        f"{pyo.value(Zn_permeate):.3g}"
        f"{pyo.units.get_units(Zn_permeate)}"
    )
    Pb_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Pb_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Pb2+ permeate mass concentration: "
        f"{pyo.value(Pb_permeate):.3g}"
        f"{pyo.units.get_units(Pb_permeate)}"
    )
    Dy_permeate = pyo.units.convert(
        m.fs.unit.properties_permeate[0].conc_mass_phase_comp["Liq", "Dy_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Dy3+ permeate mass concentration: "
        f"{pyo.value(Dy_permeate):.3g}"
        f"{pyo.units.get_units(Dy_permeate)}"
    )


if __name__ == "__main__":
    m, results = CMR_nf_case()
