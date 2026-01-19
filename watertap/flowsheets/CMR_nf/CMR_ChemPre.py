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
    Expression,
)

from idaes.core import FlowsheetBlock
from idaes.core.scaling import set_scaling_factor

# from idaes.core.solvers import get_solver
from watertap.core.solvers import get_solver

from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
)
from watertap.unit_models.nanofiltration_ZO import NanofiltrationZO
from idaes.core import UnitModelCostingBlock
from watertap.costing.unit_models.nanofiltration import cost_nanofiltration
from watertap.costing import WaterTAPCosting
from idaes.core.util.scaling import calculate_scaling_factors
from idaes.core.util.misc import StrEnum
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice
from watertap.unit_models.stoichiometric_reactor import StoichiometricReactor
from pyomo.network import Arc
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
)
import idaes.core.util.scaling as iscale

__author__ = "Chenyu Wang"


class Case(StrEnum):
    case1 = "case1"
    case2 = "case2"


def CMR_nf_case(case=Case.case1, simplified_routine=False):
    m = build(case=case)
    set_scaling(m)
    initialize_system(m)
    m, results = solve(m)
    m.fs.NF.area.unfix()
    m.fs.NF.flux_vol_solvent[0, "H2O"].fix(
        10 * pyo.units.L / (pyo.units.m**2 * pyo.units.hr)
    )
    m, results = solve(m)
    display_performance_metrics(m)
    display_costing(m)

    return m, results


def build(case):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.costing = WaterTAPCosting()

    m.fs.properties = MCASParameterBlock(
        solute_list=[
            "Ca_2+",
            "Co_2+",
            "Cu_2+",
            "Fe_3+",
            "Mg_2+",
            "Mn_2+",
            "Na_+",
            "Ni_2+",
            "Zn_2+",
            "Dy_3+",
            "Gd_3+",
            "Nd_3+",
            "Pr_3+",
            "B_3+",
            "Si_4+",
            "Cr_6+",
            "SO4_2-",
            "H_+",
        ],
        # diffusivity_data={
        #     ("Liq", "Ca_2+"): 9.2e-10,
        #     ("Liq", "Mg_2+"): 7.06e-10,
        #     ("Liq", "Na_+"): 1.33e-09,
        #     ("Liq", "SO4_2-"): 2.03e-09,
        # },
        mw_data={
            "H2O": 18.015e-3,
            "Ca_2+": 40.08e-3,
            "Co_2+": 58.9332e-3,
            "Cu_2+": 63.546e-3,
            "Fe_3+": 55.847e-3,
            "Mg_2+": 24.312e-3,
            "Mn_2+": 54.938e-3,
            "Na_+": 22.9898e-3,
            "Ni_2+": 58.71e-3,
            "Zn_2+": 65.3699e-3,
            "Dy_3+": 162.5e-3,
            "Gd_3+": 157.25e-3,
            "Nd_3+": 144.24e-3,
            "Pr_3+": 140.908e-3,
            "B_3+": 10.81e-3,
            "Si_4+": 28.09e-3,
            "Cr_6+": 51.996e-3,
            "SO4_2-": 96.0616e-3,
            "H_+": 1.008e-3,
        },
        # stokes_radius_data={
        #     "Ca_2+": 3.09e-10,
        #     "Mg_2+": 3.47e-10,
        #     "SO4_2-": 1.21e-10,
        #     "Na_+": 1.84e-10,
        # },
        charge={
            "Ca_2+": 2,
            "Co_2+": 2,
            "Cu_2+": 2,
            "Fe_3+": 3,
            "Mg_2+": 2,
            "Mn_2+": 2,
            "Na_+": 1,
            "Ni_2+": 2,
            "Zn_2+": 2,
            "Dy_3+": 3,
            "Gd_3+": 3,
            "Nd_3+": 3,
            "Pr_3+": 3,
            "B_3+": 3,
            "Si_4+": 4,
            "Cr_6+": 6,
            "SO4_2-": -2,
            "H_+": 1,
        },
    )

    # Define precipitatnts
    precipitants = {
        "CoFe2O4(s)": {
            "mw": 234.6252 * pyo.units.g / pyo.units.mol,
            "precipitation_stoichiometric": {
                "Co_2+": 1,
                "Fe_3+": 2,
                # "H2O": 4,
                # "H_+": -8,
            },
        },
        "Fe2O3(s)": {
            "mw": 159.6925 * pyo.units.g / pyo.units.mol,
            "precipitation_stoichiometric": {
                "Fe_3+": 2,
                # "H2O": 3,
                # "H_+": -6
            },
        },
        "Zn4(OH)6SO4(s)": {
            "mw": 459.6674 * pyo.units.g / pyo.units.mol,
            "precipitation_stoichiometric": {
                "Zn_2+": 4,
                # "H2O": 6,
                # "H_+": -6,
                "SO4_2-": 1,
            },
        },
    }

    m.fs.ChemPre = StoichiometricReactor(
        property_package=m.fs.properties,
        precipitate=precipitants,
    )

    # the reactor us assumed performance model
    # # MINTEQ original
    # Conc_mol_Co_precipitate = 1.7941e-03 * (pyo.units.mol / pyo.units.L)
    # Conc_mass_Co_precipitate = (
    #     Conc_mol_Co_precipitate * m.fs.ChemPre.mw_precipitate["CoFe2O4(s)"]
    # )
    #
    # Conc_mol_Fe_precipitate = 1.5448e-01 * (pyo.units.mol / pyo.units.L)
    # Conc_mass_Fe_precipitate = (
    #     Conc_mol_Fe_precipitate * m.fs.ChemPre.mw_precipitate["Fe2O3(s)"]
    # )
    #
    # Conc_mol_Zn_precipitate = 1.3398e-03 * (pyo.units.mol / pyo.units.L)
    # Conc_mass_Zn_precipitate = (
    #     Conc_mol_Zn_precipitate * m.fs.ChemPre.mw_precipitate["Zn4(OH)6SO4(s)"]
    # )

    # MINTEQ adjustment
    Conc_mol_Co_precipitate = 1.827e-03 * (pyo.units.mol / pyo.units.L)
    Conc_mass_Co_precipitate = (
        Conc_mol_Co_precipitate * m.fs.ChemPre.mw_precipitate["CoFe2O4(s)"]
    )

    Conc_mol_Fe_precipitate = 1.573e-01 * (pyo.units.mol / pyo.units.L)
    Conc_mass_Fe_precipitate = (
        Conc_mol_Fe_precipitate * m.fs.ChemPre.mw_precipitate["Fe2O3(s)"]
    )

    Conc_mol_Zn_precipitate = 1.364e-03 * (pyo.units.mol / pyo.units.L)
    Conc_mass_Zn_precipitate = (
        Conc_mol_Zn_precipitate * m.fs.ChemPre.mw_precipitate["Zn4(OH)6SO4(s)"]
    )

    m.fs.ChemPre.conc_mass_precipitate["CoFe2O4(s)"].fix(Conc_mass_Co_precipitate)
    m.fs.ChemPre.conc_mass_precipitate["Fe2O3(s)"].fix(Conc_mass_Fe_precipitate)
    m.fs.ChemPre.conc_mass_precipitate["Zn4(OH)6SO4(s)"].fix(Conc_mass_Zn_precipitate)
    m.fs.ChemPre.waste_mass_frac_precipitate.fix(1)

    m.fs.ChemPre.inlet.pressure[0].fix(101325)
    m.fs.ChemPre.inlet.temperature[0].fix(273.15 + 20)

    # set_scaling(m)

    # Fix other inlet state variables
    # fully specify system
    # Case 1
    if case is Case.case1:
        m.fs.ChemPre.precipitation_reactor.properties_in.calculate_state(
            var_args={
                ("flow_vol_phase", "Liq"): 0.0000281653,
                ("conc_mass_phase_comp", ("Liq", "Co_2+")): value(
                    105.732e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Ca_2+")): value(
                    11.3742e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Cu_2+")): value(
                    1e-9
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Fe_3+")): value(
                    17454.9915e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Nd_3+")): value(
                    521.0505e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Ni_2+")): value(
                    15.9399e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Pr_3+")): value(
                    134.4078e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Na_+")): value(
                    6.3279e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Cr_6+")): value(
                    1e-9
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Zn_2+")): value(
                    668.3544e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Dy_3+")): value(
                    0.801e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "B_3+")): value(
                    72.0099e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Gd_3+")): value(
                    2.6433e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Mn_2+")): value(
                    40.2903e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Mg_2+")): value(
                    93.6369e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Si_4+")): value(
                    20.1852e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "SO4_2-")): value(
                    1762.2e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "H_+")): value(
                    1e-9
                ),  # feed mass concentration
            },  # volumetric feed flowrate [-]
            hold_state=True,  # fixes the calculated component mass flow rates
        )
    elif case is Case.case2:
        m.fs.ChemPre.precipitation_reactor.properties_in.calculate_state(
            var_args={
                ("flow_vol_phase", "Liq"): 0.0000281653,
                ("conc_mass_phase_comp", ("Liq", "Co_2+")): value(
                    7.503e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Ca_2+")): value(
                    0
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Cu_2+")): value(
                    72.2871e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Fe_3+")): value(
                    1248.983e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Nd_3+")): value(
                    1.9065e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Ni_2+")): value(
                    1.7753e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Pr_3+")): value(
                    0.4264e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Na_+")): value(
                    3.3907e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Cr_6+")): value(
                    0.0287e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Zn_2+")): value(
                    223.2409e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Dy_3+")): value(
                    0.0738e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "B_3+")): value(
                    0
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Gd_3+")): value(
                    0.2214e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Mn_2+")): value(
                    0.3895e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "Si_4+")): value(
                    1.1808e-3
                ),  # feed mass concentration
                ("conc_mass_phase_comp", ("Liq", "SO4_2-")): value(
                    10000e-3
                ),  # feed mass concentration
            },  # volumetric feed flowrate [-]
            hold_state=True,  # fixes the calculated component mass flow rates
        )

    # touch variables
    m.fs.ChemPre.precipitation_reactor.properties_out[0].flow_vol
    m.fs.ChemPre.precipitation_reactor.properties_out[0].conc_mass_phase_comp
    m.fs.ChemPre.separator.treated_state[0].flow_vol
    m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp

    # m.fs.costing.cost_process()
    # m.fs.costing.add_annual_water_production(m.fs.NF.properties_permeate[0].flow_vol)
    # m.fs.costing.add_LCOW(m.fs.NF.properties_permeate[0].flow_vol)

    # Expressions
    # @self.Expression(m.fs.properties.solute_set, doc="S_ac concentration (kgCOD/m3) step 1")
    # def metal_rejection(b, j):
    #     return (
    #             b.properties_in[0].flow_mass_phase_comp["Liq", j]
    #             / b.properties_out[0].flow_mass_phase_comp["Liq", j]
    #     )

    # m.fs.ChemPre.metal_rejection = Var(
    #     m.fs.properties.solute_set,
    #     initialize=5e-1,
    #     bounds=(0, 1),
    #     units=pyo.units.dimensionless,
    #     doc="Metal rejection ratio",
    # )

    # def metal_rejection_rule(j):
    #     return (m.fs.ChemPre.precipitation_reactor.properties_out[0].flow_mass_phase_comp["Liq", j]
    #             / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp["Liq", j])
    #
    # m.fs.ChemPre.metal_rejection = Expression(
    #     m.fs.ChemPre.config.property_package.component_list,
    #     rule=metal_rejection_rule
    # )

    assert degrees_of_freedom(m) == 0

    return m


def set_scaling(m):
    def scale_variables(m):
        for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
            if "flow_vol" in var.name:
                iscale.set_scaling_factor(var, 1e5)
            if "flow_vol_phase" in var.name:
                iscale.set_scaling_factor(var, 1e5)
            if "temperature" in var.name:
                iscale.set_scaling_factor(var, 1e-2)
            if "pressure" in var.name:
                iscale.set_scaling_factor(var, 1e-5)
            if "flow_mol_phase_comp" in var.name:
                iscale.set_scaling_factor(var, 1e2)
            if "conc_mass_phase_comp" in var.name:
                iscale.set_scaling_factor(var, 1e2)

    # # Scale model
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "H2O"], 1e5)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Co_2+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Ca_2+"], 1e2)
    # # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Cu_2+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Fe_3+"], 1e0)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Nd_3+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Ni_2+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Pr_3+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Na_+"], 1e2)
    # # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Cr_6+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Zn_2+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Dy_3+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "B_3+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Gd_3+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Mn_2+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Mg_2+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Si_4+"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "SO4_2-"], 1e2)
    # set_scaling_factor(m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "H_+"], 1e2)

    scale_variables(m)

    set_scaling_factor(m.fs.ChemPre.flow_mass_precipitate["CoFe2O4(s)"], 1e5)
    set_scaling_factor(m.fs.ChemPre.flow_mass_precipitate["Fe2O3(s)"], 1e4)
    set_scaling_factor(m.fs.ChemPre.flow_mass_precipitate["Zn4(OH)6SO4(s)"], 1e5)
    calculate_scaling_factors(m)


def initialize_system(m):
    # Initialize system
    m.fs.ChemPre.initialize()
    # try:
    #     m.fs.ChemPre.initialize()
    # except:
    #     pass
    # m.fs.costing.initialize()


def solve(m):
    # Solve
    solver = get_solver()
    # solver = get_solver(
    #     "ipopt_v2", writer_config={"scale_model": True, "linear_presolve": True}
    # )
    results = solver.solve(m)
    assert_optimal_termination(results)

    return m, results


def display_performance_metrics(m):
    print("\n---- Feed Metrics ----")
    f_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_vol,
        to_units=pyo.units.gal / pyo.units.day,
    )
    print(f"Influent flow: " f"{pyo.value(f_in):.3g}" f"{pyo.units.get_units(f_in)}")

    Co_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Co_2+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Co2+ feed mass concentration: "
        f"{pyo.value(Co_in):.3g}"
        f"{pyo.units.get_units(Co_in)}"
    )
    Ca_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Ca_2+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Ca2+ feed mass concentration: "
        f"{pyo.value(Ca_in):.3g}"
        f"{pyo.units.get_units(Ca_in)}"
    )
    # Cu_in = pyo.units.convert(
    #     m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
    #         "Liq", "Cu_2+"
    #     ],
    #     to_units=pyo.units.mg / pyo.units.L,
    # )
    # print(
    #     f"Cu2+ feed mass concentration: "
    #     f"{pyo.value(Cu_in):.3g}"
    #     f"{pyo.units.get_units(Cu_in)}"
    # )
    Fe_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Fe_3+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Fe3+ feed mass concentration: "
        f"{pyo.value(Fe_in):.3g}"
        f"{pyo.units.get_units(Fe_in)}"
    )
    Nd_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Nd_3+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Nd3+ feed mass concentration: "
        f"{pyo.value(Nd_in):.3g}"
        f"{pyo.units.get_units(Nd_in)}"
    )
    Ni_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Ni_2+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Ni2+ feed mass concentration: "
        f"{pyo.value(Ni_in):.3g}"
        f"{pyo.units.get_units(Ni_in)}"
    )
    Pr_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Pr_3+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Pr3+ feed mass concentration: "
        f"{pyo.value(Pr_in):.3g}"
        f"{pyo.units.get_units(Pr_in)}"
    )
    Na_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Na_+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Na+ feed mass concentration: "
        f"{pyo.value(Na_in):.3g}"
        f"{pyo.units.get_units(Na_in)}"
    )
    # Cr_in = pyo.units.convert(
    #     m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
    #         "Liq", "Cr_6+"
    #     ],
    #     to_units=pyo.units.mg / pyo.units.L,
    # )
    # print(
    #     f"Cr6+ feed mass concentration: "
    #     f"{pyo.value(Cr_in):.3g}"
    #     f"{pyo.units.get_units(Cr_in)}"
    # )
    Zn_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Zn_2+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Zn2+ feed mass concentration: "
        f"{pyo.value(Zn_in):.3g}"
        f"{pyo.units.get_units(Zn_in)}"
    )
    Dy_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Dy_3+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Dy3+ feed mass concentration: "
        f"{pyo.value(Dy_in):.3g}"
        f"{pyo.units.get_units(Dy_in)}"
    )
    B_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "B_3+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"B3+ feed mass concentration: "
        f"{pyo.value(B_in):.3g}"
        f"{pyo.units.get_units(B_in)}"
    )
    Gd_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Gd_3+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Gd3+ feed mass concentration: "
        f"{pyo.value(Gd_in):.3g}"
        f"{pyo.units.get_units(Gd_in)}"
    )
    Mn_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Mn_2+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Mn2+ feed mass concentration: "
        f"{pyo.value(Mn_in):.3g}"
        f"{pyo.units.get_units(Mn_in)}"
    )
    Mg_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Mg_2+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Mg2+ feed mass concentration: "
        f"{pyo.value(Mg_in):.3g}"
        f"{pyo.units.get_units(Mg_in)}"
    )
    Si_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "Si_4+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Si4+ feed mass concentration: "
        f"{pyo.value(Si_in):.3g}"
        f"{pyo.units.get_units(Si_in)}"
    )
    SO4_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "SO4_2-"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"SO4_2- feed mass concentration: "
        f"{pyo.value(SO4_in):.3g}"
        f"{pyo.units.get_units(SO4_in)}"
    )
    H_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].conc_mass_phase_comp[
            "Liq", "H_+"
        ],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"H_+ feed mass concentration: "
        f"{pyo.value(H_in):.3g}"
        f"{pyo.units.get_units(H_in)}"
    )

    print("\n---- Outlet Metrics ----")
    f_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].flow_vol,
        to_units=pyo.units.gal / pyo.units.day,
    )
    print(
        f"treated flow: "
        f"{pyo.value(f_treated):.3g}"
        f"{pyo.units.get_units(f_treated)}"
    )
    Co_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Co_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Co2+ treated mass concentration: "
        f"{pyo.value(Co_treated):.3g}"
        f"{pyo.units.get_units(Co_treated)}"
    )
    Ca_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Ca_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Ca2+ treated mass concentration: "
        f"{pyo.value(Ca_treated):.3g}"
        f"{pyo.units.get_units(Ca_treated)}"
    )
    # Cu_treated = pyo.units.convert(
    #     m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Cu_2+"],
    #     to_units=pyo.units.mg / pyo.units.L,
    # )
    # print(
    #     f"Cu2+ treated mass concentration: "
    #     f"{pyo.value(Cu_treated):.3g}"
    #     f"{pyo.units.get_units(Cu_treated)}"
    # )
    Fe_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Fe_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Fe3+ treated mass concentration: "
        f"{pyo.value(Fe_treated):.3g}"
        f"{pyo.units.get_units(Fe_treated)}"
    )
    Nd_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Nd_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Nd3+ treated mass concentration: "
        f"{pyo.value(Nd_treated):.3g}"
        f"{pyo.units.get_units(Nd_treated)}"
    )
    Ni_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Ni_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Ni2+ treated mass concentration: "
        f"{pyo.value(Ni_treated):.3g}"
        f"{pyo.units.get_units(Ni_treated)}"
    )
    Pr_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Pr_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Pr3+ treated mass concentration: "
        f"{pyo.value(Pr_treated):.3g}"
        f"{pyo.units.get_units(Pr_treated)}"
    )
    Na_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Na_+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Na+ treated mass concentration: "
        f"{pyo.value(Na_treated):.3g}"
        f"{pyo.units.get_units(Na_treated)}"
    )
    # Cr_treated = pyo.units.convert(
    #     m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Cr_6+"],
    #     to_units=pyo.units.mg / pyo.units.L,
    # )
    # print(
    #     f"Cr6+ treated mass concentration: "
    #     f"{pyo.value(Cr_treated):.3g}"
    #     f"{pyo.units.get_units(Cr_treated)}"
    # )
    Zn_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Zn_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Zn2+ treated mass concentration: "
        f"{pyo.value(Zn_treated):.3g}"
        f"{pyo.units.get_units(Zn_treated)}"
    )
    Dy_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Dy_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Dy3+ treated mass concentration: "
        f"{pyo.value(Dy_treated):.3g}"
        f"{pyo.units.get_units(Dy_treated)}"
    )
    B_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "B_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"B3+ treated mass concentration: "
        f"{pyo.value(B_treated):.3g}"
        f"{pyo.units.get_units(B_treated)}"
    )
    Gd_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Gd_3+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Gd3+ treated mass concentration: "
        f"{pyo.value(Gd_treated):.3g}"
        f"{pyo.units.get_units(Gd_treated)}"
    )
    Mn_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Mn_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Mn2+ treated mass concentration: "
        f"{pyo.value(Mn_treated):.3g}"
        f"{pyo.units.get_units(Mn_treated)}"
    )
    Mg_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Mg_2+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Mg2+ treated mass concentration: "
        f"{pyo.value(Mg_treated):.3g}"
        f"{pyo.units.get_units(Mg_treated)}"
    )
    Si_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "Si_4+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"Si4+ treated mass concentration: "
        f"{pyo.value(Si_treated):.3g}"
        f"{pyo.units.get_units(Si_treated)}"
    )
    SO4_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "SO4_2-"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"SO4_2- treated mass concentration: "
        f"{pyo.value(SO4_treated):.3g}"
        f"{pyo.units.get_units(SO4_treated)}"
    )
    H_treated = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].conc_mass_phase_comp["Liq", "H_+"],
        to_units=pyo.units.mg / pyo.units.L,
    )
    print(
        f"H_+ treated mass concentration: "
        f"{pyo.value(H_treated):.3g}"
        f"{pyo.units.get_units(H_treated)}"
    )

    print("\n---- System Performance Metrics ----")
    f_in = pyo.units.convert(
        m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_vol,
        to_units=pyo.units.m**3 / pyo.units.hr,
    )
    f_out = pyo.units.convert(
        m.fs.ChemPre.separator.treated_state[0].flow_vol,
        to_units=pyo.units.m**3 / pyo.units.hr,
    )
    water_recovery = f_out / f_in
    # water_recovery = (
    #     m.fs.ChemPre.outlet.flow_mol_phase_comp[0, "Liq", "H2O"]
    #     / m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "H2O"]
    # )
    print(f"Volumetric-based recovery: " f"{pyo.value(water_recovery):.8g}")
    Co_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Co_2+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Co_2+"
        ]
    )
    # Co_recovery = (
    #     m.fs.ChemPre.outlet.flow_mol_phase_comp[0, "Liq", "Co_2+"]
    #     / m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Co_2+"]
    # )
    print(f"Co2+ mass rejection: " f"{pyo.value(1 - Co_recovery):.3g}")
    Ca_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Ca_2+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Ca_2+"
        ]
    )
    # Ca_recovery = (
    #     m.fs.ChemPre.outlet.flow_mol_phase_comp[0, "Liq", "Ca_2+"]
    #     / m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Ca_2+"]
    # )
    print(f"Ca2+ mass rejection: " f"{pyo.value(1 - Ca_recovery):.3g}")
    # Cu_recovery = (
    #     m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Cu_2+"]
    #     / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
    #         "Liq", "Cu_2+"
    #     ]
    # )
    # # Cu_recovery = (
    # #     m.fs.ChemPre.outlet.flow_mol_phase_comp[0, "Liq", "Cu_2+"]
    # #     / m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Cu_2+"]
    # # )
    # print(f"Cu2+ mass rejection: " f"{pyo.value(1 - Cu_recovery):.3g}")
    Fe_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Fe_3+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Fe_3+"
        ]
    )
    # Fe_recovery = (
    #     m.fs.ChemPre.outlet.flow_mol_phase_comp[0, "Liq", "Fe_3+"]
    #     / m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "Fe_3+"]
    # )
    print(f"Fe3+ mass rejection: " f"{pyo.value(1 - Fe_recovery):.3g}")
    Nd_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Nd_3+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Nd_3+"
        ]
    )
    print(f"Nd3+ mass rejection: " f"{pyo.value(1 - Nd_recovery):.3g}")
    Ni_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Ni_2+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Ni_2+"
        ]
    )
    print(f"Ni2+ mass rejection: " f"{pyo.value(1 - Ni_recovery):.3g}")
    Pr_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Pr_3+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Pr_3+"
        ]
    )
    print(f"Pr3+ mass rejection: " f"{pyo.value(1 - Pr_recovery):.3g}")
    Na_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Na_+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Na_+"
        ]
    )
    print(f"Na+ mass rejection: " f"{pyo.value(1 - Na_recovery):.3g}")
    # Cr_recovery = (
    #     m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Cr_6+"]
    #     / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
    #         "Liq", "Cr_6+"
    #     ]
    # )
    # print(f"Cr6+ mass rejection: " f"{pyo.value(1 - Cr_recovery):.3g}")
    Zn_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Zn_2+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Zn_2+"
        ]
    )
    print(f"Zn2+ mass rejection: " f"{pyo.value(1 - Zn_recovery):.3g}")
    Dy_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Dy_3+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Dy_3+"
        ]
    )
    print(f"Dy3+ mass rejection: " f"{pyo.value(1 - Dy_recovery):.3g}")
    B_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "B_3+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "B_3+"
        ]
    )
    print(f"B3+ mass rejection: " f"{pyo.value(1 - B_recovery):.3g}")
    Gd_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Gd_3+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Gd_3+"
        ]
    )
    print(f"Gd3+ mass rejection: " f"{pyo.value(1 - Gd_recovery):.3g}")
    Mn_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Mn_2+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Mn_2+"
        ]
    )
    print(f"Mn2+ mass rejection: " f"{pyo.value(1 - Mn_recovery):.3g}")
    Mg_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Mg_2+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Mg_2+"
        ]
    )
    print(f"Mg2+ mass rejection: " f"{pyo.value(1 - Mg_recovery):.3g}")
    Si_recovery = (
        m.fs.ChemPre.separator.treated_state[0].flow_mass_phase_comp["Liq", "Si_4+"]
        / m.fs.ChemPre.precipitation_reactor.properties_in[0].flow_mass_phase_comp[
            "Liq", "Si_4+"
        ]
    )
    print(f"Si4+ mass rejection: " f"{pyo.value(1 - Si_recovery):.3g}")
    SO4_recovery = (
        m.fs.ChemPre.outlet.flow_mol_phase_comp[0, "Liq", "SO4_2-"]
        / m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "SO4_2-"]
    )
    print(f"SO4_2- mass rejection: " f"{pyo.value(1 - SO4_recovery):.3g}")
    H_recovery = (
        m.fs.ChemPre.outlet.flow_mol_phase_comp[0, "Liq", "H_+"]
        / m.fs.ChemPre.inlet.flow_mol_phase_comp[0, "Liq", "H_+"]
    )
    print(f"H_+ mass rejection: " f"{pyo.value(1 - H_recovery):.3g}")

    # Display Precipitate
    print("\n---- Precipitate Metrics ----")
    CoFe2O4_precipitate = m.fs.ChemPre.flow_mass_precipitate["CoFe2O4(s)"]
    print(f"CoFe2O4 precipitate: " f"{pyo.value(CoFe2O4_precipitate):.3g}")
    Fe2O3_precipitate = m.fs.ChemPre.flow_mass_precipitate["Fe2O3(s)"]
    print(f"Fe2O3 precipitate: " f"{pyo.value(Fe2O3_precipitate):.3g}")
    Zn4OH6SO4_precipitate = m.fs.ChemPre.flow_mass_precipitate["Zn4(OH)6SO4(s)"]
    print(f"Zn4(OH)6SO4 precipitate: " f"{pyo.value(Zn4OH6SO4_precipitate):.3g}")


def display_costing(m):
    print("\n---- Cost Metrics ----")
    print("Levelized cost of water: %.3g $/m3" % pyo.value(m.fs.costing.LCOW))

    print(
        "Total operating cost: %.3g $/yr" % pyo.value(m.fs.costing.total_operating_cost)
    )
    print("Total capital cost: %.3g $" % pyo.value(m.fs.costing.total_capital_cost))

    print(
        "Total annualized cost: %.3g $/yr"
        % pyo.value(m.fs.costing.total_annualized_cost)
    )
    print("Capital cost pump: %.3g $" % pyo.value(m.fs.P1.costing.capital_cost))
    print(
        "Capital cost nanofiltration: %.3g $" % pyo.value(m.fs.NF.costing.capital_cost)
    )


if __name__ == "__main__":
    # m, results = CMR_nf_case(case=Case.case1, simplified_routine=False)
    m = build(case=Case.case1)
    set_scaling(m)
    m, results = solve(m)
    display_performance_metrics(m)
