#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
"""
Flowsheet example full Water Resource Recovery Facility
(WRRF; a.k.a., wastewater treatment plant) with ASM2d and ADM1 with P extension.

The flowsheet follows the same formulation as benchmark simulation model no.2 (BSM2)
but comprises different specifications for default values than BSM2.
"""

# Some more information about this module
__author__ = "Chenyu Wang, Adam Atia, Alejandro Garciadiego, Marcus Holly"

import pyomo.environ as pyo
from pyomo.network import Arc, SequentialDecomposition

from idaes.core import (
    FlowsheetBlock,
    UnitModelCostingBlock,
    UnitModelBlockData,
)
from idaes.models.unit_models import (
    CSTR,
    Feed,
    Separator,
    Product,
    Mixer,
    PressureChanger,
)
from idaes.models.unit_models.separator import SplittingType
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core.util.tables import (
    create_stream_table_dataframe,
    stream_table_dataframe_to_string,
)
from watertap.unit_models.cstr_injection import CSTR_Injection
from watertap.unit_models.clarifier import Clarifier
from watertap.property_models.unit_specific.anaerobic_digestion.modified_adm1_properties import (
    ModifiedADM1ParameterBlock,
)
from watertap.property_models.unit_specific.anaerobic_digestion.adm1_properties_vapor import (
    ADM1_vaporParameterBlock,
)
from watertap.property_models.unit_specific.anaerobic_digestion.modified_adm1_reactions import (
    ModifiedADM1ReactionParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.modified_asm2d_properties import (
    ModifiedASM2dParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.modified_asm2d_reactions import (
    ModifiedASM2dReactionParameterBlock,
)
from watertap.unit_models.translators.translator_adm1_asm2d import (
    Translator_ADM1_ASM2D,
)
from idaes.models.unit_models.mixer import MomentumMixingType
from watertap.unit_models.translators.translator_asm2d_adm1 import (
    Translator_ASM2d_ADM1,
)
from watertap.unit_models.anaerobic_digester import AD
from watertap.unit_models.dewatering import (
    DewateringUnit,
    ActivatedSludgeModelType as dewater_type,
)
from watertap.unit_models.thickener import (
    Thickener,
    ActivatedSludgeModelType as thickener_type,
)
from watertap.core.util.initialization import (
    check_solve,
    assert_degrees_of_freedom,
    interval_initializer,
)
from watertap.unit_models.electroNP_surrogate.electroNP_surrogate import ElectroNP

from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.clarifier import (
    cost_circular_clarifier,
    cost_primary_clarifier,
)

from pyomo.environ import *
from watertap.unit_models.aeration_tank import AerationTank, ElectricityConsumption
from watertap.costing.unit_models.pump import cost_pump, PumpType

from idaes.core.util.model_diagnostics import DegeneracyHunter
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.scaling.custom_scaler_base import (
    CustomScalerBase,
    ConstraintScalingScheme,
)
from idaes.core.scaling.autoscaling import AutoScaler
import numpy as np
from idaes.core.util.misc import StrEnum

# Set up logger
_log = idaeslog.getLogger(__name__)

import logging

logging.getLogger("idaes.core.util.scaling").setLevel(logging.ERROR)


class objective_fun(StrEnum):
    LCOW = "LCOW"
    LCOP = "LCOP"


def multi_run(
    has_electroNP=True,
    objective=objective_fun.LCOW,
    has_effluent_constraints=False,
    num=5,
):
    CP_list = np.linspace(-1.3, -0.8, num)
    r_AV_list = np.linspace(0.065, 0.145, num)

    # m = build_flowsheet(has_electroNP=has_electroNP)
    # set_operating_conditions(m)
    # set_scaling(m)
    # m, results = initialize_system(m)
    # add_costing(m)
    # m.fs.costing.initialize()
    # interval_initializer(m.fs.costing)
    # assert_degrees_of_freedom(m, 0)
    # results = solve(m)
    # # setup_optimization(m, objective=objective, has_effluent_constraints=has_effluent_constraints,
    # #                    reactor_volume_equalities=False)
    # # results = solve(m)
    # # m_set = [m]
    # # obj_set = [pyo.value(m.fs.objective)]

    m_set = []
    obj_set = []
    CP_set_opt = []
    r_AV_opt = []

    for i in range(0, num):
        for j in range(0, num):
            m = build_flowsheet(has_electroNP=has_electroNP)
            set_operating_conditions(m)
            # m.fs.electroNP.cathodic_potential.unfix()
            # m.fs.electroNP.area_volume_ratio.unfix()
            # m.fs.electroNP.cathodic_potential.fix(CP_list[i])
            # m.fs.electroNP.area_volume_ratio.fix(r_AV_list[j])
            set_scaling(m)
            try:
                m, results = initialize_system(m)
                add_costing(m)
                m.fs.costing.initialize()
                interval_initializer(m.fs.costing)
                m.fs.electroNP.cathodic_potential.unfix()
                m.fs.electroNP.area_volume_ratio.unfix()
                m.fs.electroNP.cathodic_potential.fix(CP_list[i])
                m.fs.electroNP.area_volume_ratio.fix(r_AV_list[j])
                # results = solve(m)
                setup_optimization(
                    m,
                    objective=objective,
                    has_effluent_constraints=has_effluent_constraints,
                    reactor_volume_equalities=False,
                )
                results = solve(m)
                m_set.append(m)
                obj_set.append(pyo.value(m.fs.objective))
                CP_set_opt.append(CP_list[i])
                r_AV_opt.append(r_AV_list[j])
            except:
                pass

    min_value = min(obj_set)
    min_idx = obj_set.index(min_value)

    cp_opt = CP_set_opt[min_idx]
    r_AV_opt = r_AV_opt[min_idx]

    # set_operating_conditions(m)
    # m.fs.electroNP.cathodic_potential.unfix()
    # m.fs.electroNP.area_volume_ratio.unfix()
    # m.fs.electroNP.cathodic_potential.fix(cp_opt)
    # m.fs.electroNP.area_volume_ratio.fix(r_AV_opt)
    # set_scaling(m)
    # setup_optimization(m, objective=objective, has_effluent_constraints=has_effluent_constraints,
    #                    reactor_volume_equalities=False)
    # results = solve(m)

    m_min = m_set[min_idx]

    display_design(m_min)

    display_performance_metrics(m_min)
    display_TP_table(m_min)
    display_costing(m_min)

    return m_min, obj_set, m_set, cp_opt, r_AV_opt


def main(
    has_electroNP=False,
    has_optimization=False,
    objective=objective_fun.LCOW,
    has_effluent_constraints=False,
):
    m = build_flowsheet(has_electroNP=has_electroNP)
    set_operating_conditions(m)
    set_scaling(m)

    # print("\n================ Badly Scaled Vars AFTER set_scaling ================")
    # badly_scaled_var_list = iscale.badly_scaled_var_generator(m, large=1e1, small=1e-1)
    # for x in badly_scaled_var_list:
    #     print(f"{x[0].name}\t{x[0].value}\tsf: {iscale.get_scaling_factor(x[0])}")

    m, results = initialize_system(m)

    add_costing(m)

    m.fs.costing.initialize()

    interval_initializer(m.fs.costing)
    assert_degrees_of_freedom(m, 0)

    # dt = DiagnosticsToolbox(m)
    # print("\n================ Structural Issues ================")
    # dt.report_structural_issues()
    # dt.display_potential_evaluation_errors()

    # print("\n================ Numerical Issues AFTER initialization ================")
    # dt.report_numerical_issues()
    # dt.display_variables_with_extreme_jacobians()
    # dt.display_constraints_with_extreme_jacobians()

    # if m.fs.has_electroNP is True:
    #     # m.fs.electroNP.eq_P_removal_surrogate.deactivate()
    #     # m.fs.electroNP.P_removal.fix(0.95)
    #     m.fs.electroNP.cathodic_potential.fix(-1.1 * pyo.units.V)
    #     m.fs.electroNP.area_volume_ratio.fix(0.1)
    #     m.fs.electroNP.settling_time.fix(30 * pyo.units.min)
    #     m.fs.electroNP.magnesium_chloride_dosage.fix(0.388)
    #     m.fs.electroNP.frac_mass_H2O_treated[0].fix(0.9)
    #     for p_removal_step in homotopy_steps:
    #         m.fs.electroNP.P_removal.fix(p_removal_step)
    #         print(
    #             f"\n================ Homotopy step: P_removal={p_removal_step} "
    #             f"(frac_mass_H2O_treated held at 0.95) ================"
    #         )
    #         def print_step_diagnostics():
    #             S_IP_into_AD = pyo.value(
    #                 m.fs.translator_asm2d_adm1.properties_out[0].conc_mass_comp[
    #                     "S_IP"
    #                 ]
    #             )
    #             K_S_IP_mass = pyo.value(m.fs.rxn_props_ADM1.K_S_IP) * 31
    #             print(
    #                 f"S_IP into AD: {S_IP_into_AD:.6g} kg/m3  "
    #                 f"(K_S_IP ~ {K_S_IP_mass:.6g} kg/m3, ratio "
    #                 f"{S_IP_into_AD / K_S_IP_mass:.3g})"
    #             )
    #             K_MAX = pyo.value(m.fs.rxn_props_ASM2D.K_MAX)
    #             for name, block in (
    #                 ("R1", m.fs.R1),
    #                 ("R3", m.fs.R3),
    #                 ("R5", m.fs.R5),
    #                 ("R7", m.fs.R7),
    #             ):
    #                 X_PP = pyo.value(
    #                     block.control_volume.properties_out[0].conc_mass_comp[
    #                         "X_PP"
    #                     ]
    #                 )
    #                 X_PAO = pyo.value(
    #                     block.control_volume.properties_out[0].conc_mass_comp[
    #                         "X_PAO"
    #                     ]
    #                 )
    #                 margin = K_MAX * X_PAO - X_PP
    #                 print(
    #                     f"  {name} outlet: X_PP={X_PP:.4g}, X_PAO={X_PAO:.4g}, "
    #                     f"K_MAX*X_PAO={K_MAX * X_PAO:.4g}, margin={margin:.4g} "
    #                     f"({100 * margin / (K_MAX * X_PAO):.1f}% of capacity "
    #                     "remaining)"
    #                 )
    #             display_stream_table(m)
    #
    #         try:
    #             results = solve(m)
    #         except Exception:
    #             print(
    #                 f"Homotopy step P_removal={p_removal_step} FAILED. "
    #                 "Leaving value fixed at this step and constraint "
    #                 "deactivated for inspection -- NOT reverting."
    #             )
    #
    #             print(
    #                 "\n================ Constraints with Large Residuals "
    #                 "(at failed step) ================"
    #             )
    #             dt_fail = DiagnosticsToolbox(m)
    #             dt_fail.display_constraints_with_large_residuals()
    #             print(
    #                 "\n================ Variables At or Outside Bounds "
    #                 "(at failed step) ================"
    #             )
    #             dt_fail.display_variables_at_or_outside_bounds()
    #
    #             print(
    #                 "\n================ TP table / diagnostics AT FAILED "
    #                 "(near-feasible) POINT ================"
    #             )
    #             print_step_diagnostics()
    #             raise
    #         print_step_diagnostics()
    #
    # else:
    #     try:
    #         results = solve(m)
    #     except Exception:
    #         print("\n================ Constraints with Large Residuals ================")
    #         dt_fail = DiagnosticsToolbox(m)
    #         dt_fail.display_constraints_with_large_residuals()
    #         print("\n================ Variables At or Outside Bounds ================")
    #         dt_fail.display_variables_at_or_outside_bounds()
    #         raise

    if has_optimization:
        setup_optimization(
            m,
            objective=objective,
            has_effluent_constraints=has_effluent_constraints,
            reactor_volume_equalities=False,
        )

    results = solve(m)
    pyo.assert_optimal_termination(results)

    check_solve(
        results,
        checkpoint="re-solve with controls in place",
        logger=_log,
        fail_flag=True,
    )

    # print("\n================ Numerical Issues AFTER solve ================")
    # dt.report_numerical_issues()
    # dt.display_variables_with_extreme_jacobians()
    # dt.display_constraints_with_extreme_jacobians()

    # display_TP_table(m)

    display_design(m)

    display_performance_metrics(m)

    display_costing(m)

    return m, results


def build_flowsheet(has_electroNP=False):
    m = pyo.ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.has_electroNP = has_electroNP

    # Properties
    m.fs.props_ASM2D = ModifiedASM2dParameterBlock()
    m.fs.rxn_props_ASM2D = ModifiedASM2dReactionParameterBlock(
        property_package=m.fs.props_ASM2D
    )
    m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
    m.fs.props_vap_ADM1 = ADM1_vaporParameterBlock()
    m.fs.rxn_props_ADM1 = ModifiedADM1ReactionParameterBlock(
        property_package=m.fs.props_ADM1
    )

    # Feed water stream
    m.fs.FeedWater = Feed(property_package=m.fs.props_ASM2D)

    # ====================================================================
    # Primary Clarifier
    m.fs.CL = Clarifier(
        property_package=m.fs.props_ASM2D,
        outlet_list=["underflow", "effluent"],
        split_basis=SplittingType.componentFlow,
    )

    # ======================================================================
    # Activated Sludge Process
    # Mixer for feed water and recycled sludge
    m.fs.MX1 = Mixer(
        property_package=m.fs.props_ASM2D,
        inlet_list=["feed_water", "recycle"],
        momentum_mixing_type=MomentumMixingType.equality,
    )
    # First reactor (anaerobic) - standard CSTR
    m.fs.R1 = CSTR(
        property_package=m.fs.props_ASM2D, reaction_package=m.fs.rxn_props_ASM2D
    )
    # Second reactor (anaerobic) - standard CSTR
    m.fs.R2 = CSTR(
        property_package=m.fs.props_ASM2D, reaction_package=m.fs.rxn_props_ASM2D
    )
    # Third reactor (anoxic) - standard CSTR
    m.fs.R3 = CSTR(
        property_package=m.fs.props_ASM2D, reaction_package=m.fs.rxn_props_ASM2D
    )
    # Fourth reactor (anoxic) - standard CSTR
    m.fs.R4 = CSTR(
        property_package=m.fs.props_ASM2D, reaction_package=m.fs.rxn_props_ASM2D
    )
    # Fifth reactor (aerobic) - CSTR with injection
    m.fs.R5 = AerationTank(
        property_package=m.fs.props_ASM2D,
        reaction_package=m.fs.rxn_props_ASM2D,
        electricity_consumption=ElectricityConsumption.calculated,
    )
    # Sixth reactor (aerobic) - CSTR with injection
    m.fs.R6 = AerationTank(
        property_package=m.fs.props_ASM2D,
        reaction_package=m.fs.rxn_props_ASM2D,
        electricity_consumption=ElectricityConsumption.calculated,
    )
    # Seventh reactor (aerobic) - CSTR with injection
    m.fs.R7 = AerationTank(
        property_package=m.fs.props_ASM2D,
        reaction_package=m.fs.rxn_props_ASM2D,
        electricity_consumption=ElectricityConsumption.calculated,
    )
    m.fs.SP1 = Separator(
        property_package=m.fs.props_ASM2D, outlet_list=["underflow", "overflow"]
    )
    # Secondary Clarifier
    # TODO: Replace with more detailed model when available
    m.fs.CL2 = Clarifier(
        property_package=m.fs.props_ASM2D,
        outlet_list=["underflow", "effluent"],
        split_basis=SplittingType.componentFlow,
    )
    # Mixing sludge recycle and R5 underflow
    m.fs.MX2 = Mixer(
        property_package=m.fs.props_ASM2D,
        inlet_list=["reactor", "clarifier"],
        momentum_mixing_type=MomentumMixingType.equality,
    )
    # Sludge separator
    m.fs.SP2 = Separator(
        property_package=m.fs.props_ASM2D, outlet_list=["waste", "recycle"]
    )
    # Recycle pressure changer - use a simple isothermal unit for now
    m.fs.P1 = PressureChanger(property_package=m.fs.props_ASM2D)

    # ======================================================================
    # Thickener
    m.fs.thickener = Thickener(
        property_package=m.fs.props_ASM2D,
        activated_sludge_model=thickener_type.modified_ASM2D,
    )
    # Mixing feed and recycle streams from thickener and dewatering unit
    m.fs.MX3 = Mixer(
        property_package=m.fs.props_ASM2D,
        inlet_list=["feed_water", "recycle1", "recycle2"],
        momentum_mixing_type=MomentumMixingType.equality,
    )
    # Mixing sludge from thickener and primary clarifier
    m.fs.MX4 = Mixer(
        property_package=m.fs.props_ASM2D,
        inlet_list=["thickener", "clarifier"],
        momentum_mixing_type=MomentumMixingType.equality,
    )

    # ======================================================================
    # Anaerobic digester section
    # ASM2d-ADM1 translator
    m.fs.translator_asm2d_adm1 = Translator_ASM2d_ADM1(
        inlet_property_package=m.fs.props_ASM2D,
        outlet_property_package=m.fs.props_ADM1,
        inlet_reaction_package=m.fs.rxn_props_ASM2D,
        outlet_reaction_package=m.fs.rxn_props_ADM1,
        has_phase_equilibrium=False,
        outlet_state_defined=True,
        bio_P=False,
    )

    # Anaerobic digester
    m.fs.AD = AD(
        liquid_property_package=m.fs.props_ADM1,
        vapor_property_package=m.fs.props_vap_ADM1,
        reaction_package=m.fs.rxn_props_ADM1,
        has_heat_transfer=True,
        has_pressure_change=False,
    )

    # ADM1-ASM2d translator
    m.fs.translator_adm1_asm2d = Translator_ADM1_ASM2D(
        inlet_property_package=m.fs.props_ADM1,
        outlet_property_package=m.fs.props_ASM2D,
        inlet_reaction_package=m.fs.rxn_props_ADM1,
        outlet_reaction_package=m.fs.rxn_props_ASM2D,
        has_phase_equilibrium=False,
        outlet_state_defined=True,
    )

    # ======================================================================
    # Dewatering Unit
    m.fs.dewater = DewateringUnit(
        property_package=m.fs.props_ASM2D,
        activated_sludge_model=dewater_type.modified_ASM2D,
    )

    # ======================================================================
    # ElectroN-P
    if has_electroNP is True:
        m.fs.electroNP = ElectroNP(property_package=m.fs.props_ASM2D)

    # ======================================================================
    # Product Blocks
    m.fs.Treated = Product(property_package=m.fs.props_ASM2D)
    m.fs.Sludge = Product(property_package=m.fs.props_ASM2D)
    # Mixers
    m.fs.mixers = (m.fs.MX1, m.fs.MX2, m.fs.MX4)

    # ======================================================================
    # Link units related to ASM section
    m.fs.stream2 = Arc(source=m.fs.MX1.outlet, destination=m.fs.R1.inlet)
    m.fs.stream3 = Arc(source=m.fs.R1.outlet, destination=m.fs.R2.inlet)
    m.fs.stream4 = Arc(source=m.fs.R2.outlet, destination=m.fs.MX2.reactor)
    m.fs.stream5 = Arc(source=m.fs.MX2.outlet, destination=m.fs.R3.inlet)
    m.fs.stream6 = Arc(source=m.fs.R3.outlet, destination=m.fs.R4.inlet)
    m.fs.stream7 = Arc(source=m.fs.R4.outlet, destination=m.fs.R5.inlet)
    m.fs.stream8 = Arc(source=m.fs.R5.outlet, destination=m.fs.R6.inlet)
    m.fs.stream9 = Arc(source=m.fs.R6.outlet, destination=m.fs.R7.inlet)
    m.fs.stream10 = Arc(source=m.fs.R7.outlet, destination=m.fs.SP1.inlet)
    m.fs.stream11 = Arc(source=m.fs.SP1.overflow, destination=m.fs.CL2.inlet)
    m.fs.stream12 = Arc(source=m.fs.SP1.underflow, destination=m.fs.MX2.clarifier)
    m.fs.stream13 = Arc(source=m.fs.CL2.effluent, destination=m.fs.Treated.inlet)
    m.fs.stream14 = Arc(source=m.fs.CL2.underflow, destination=m.fs.SP2.inlet)
    m.fs.stream15 = Arc(source=m.fs.SP2.recycle, destination=m.fs.P1.inlet)
    m.fs.stream16 = Arc(source=m.fs.P1.outlet, destination=m.fs.MX1.recycle)

    # Link units related to AD section
    m.fs.stream_AD_translator = Arc(
        source=m.fs.AD.liquid_outlet, destination=m.fs.translator_adm1_asm2d.inlet
    )
    m.fs.stream_SP_thickener = Arc(
        source=m.fs.SP2.waste, destination=m.fs.thickener.inlet
    )
    m.fs.stream3adm = Arc(
        source=m.fs.thickener.underflow, destination=m.fs.MX4.thickener
    )
    m.fs.stream7adm = Arc(source=m.fs.thickener.overflow, destination=m.fs.MX3.recycle2)
    m.fs.stream9adm = Arc(source=m.fs.CL.underflow, destination=m.fs.MX4.clarifier)
    m.fs.stream_translator_dewater = Arc(
        source=m.fs.translator_adm1_asm2d.outlet, destination=m.fs.dewater.inlet
    )
    m.fs.stream1a = Arc(source=m.fs.FeedWater.outlet, destination=m.fs.MX3.feed_water)
    m.fs.stream1b = Arc(source=m.fs.MX3.outlet, destination=m.fs.CL.inlet)
    m.fs.stream1c = Arc(source=m.fs.CL.effluent, destination=m.fs.MX1.feed_water)
    m.fs.stream_dewater_sludge = Arc(
        source=m.fs.dewater.underflow, destination=m.fs.Sludge.inlet
    )
    if has_electroNP is True:
        m.fs.stream_dewater_electroNP = Arc(
            source=m.fs.dewater.overflow, destination=m.fs.electroNP.inlet
        )
        m.fs.stream_electroNP_mixer = Arc(
            source=m.fs.electroNP.treated, destination=m.fs.MX3.recycle1
        )
    else:
        m.fs.stream_dewater_mixer = Arc(
            source=m.fs.dewater.overflow, destination=m.fs.MX3.recycle1
        )
    m.fs.stream10adm = Arc(
        source=m.fs.MX4.outlet, destination=m.fs.translator_asm2d_adm1.inlet
    )
    m.fs.stream_translator_AD = Arc(
        source=m.fs.translator_asm2d_adm1.outlet, destination=m.fs.AD.inlet
    )

    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # Oxygen concentration in reactors 3 and 4 is governed by mass transfer
    # Add additional parameter and constraints
    # m.fs.R5.KLa = pyo.Var(
    #     initialize=240,
    #     units=pyo.units.hour**-1,
    #     doc="Lumped mass transfer coefficient for oxygen",
    # )
    # m.fs.R6.KLa = pyo.Var(
    #     initialize=240,
    #     units=pyo.units.hour**-1,
    #     doc="Lumped mass transfer coefficient for oxygen",
    # )
    # m.fs.R7.KLa = pyo.Var(
    #     initialize=84,
    #     units=pyo.units.hour**-1,
    #     doc="Lumped mass transfer coefficient for oxygen",
    # )
    # m.fs.S_O_eq = pyo.Param(
    #     default=8e-3,
    #     units=pyo.units.kg / pyo.units.m**3,
    #     mutable=True,
    #     doc="Dissolved oxygen concentration at equilibrium",
    # )

    # @m.fs.R5.Constraint(m.fs.time, doc="Mass transfer constraint for R3")
    # def mass_transfer_R5(self, t):
    #     return pyo.units.convert(
    #         m.fs.R5.injection[t, "Liq", "S_O2"], to_units=pyo.units.kg / pyo.units.hour
    #     ) == (
    #         m.fs.R5.KLa
    #         * m.fs.R5.volume[t]
    #         * (m.fs.S_O_eq - m.fs.R5.outlet.conc_mass_comp[t, "S_O2"])
    #     )
    #
    # @m.fs.R6.Constraint(m.fs.time, doc="Mass transfer constraint for R4")
    # def mass_transfer_R6(self, t):
    #     return pyo.units.convert(
    #         m.fs.R6.injection[t, "Liq", "S_O2"], to_units=pyo.units.kg / pyo.units.hour
    #     ) == (
    #         m.fs.R6.KLa
    #         * m.fs.R6.volume[t]
    #         * (m.fs.S_O_eq - m.fs.R6.outlet.conc_mass_comp[t, "S_O2"])
    #     )
    #
    # @m.fs.R7.Constraint(m.fs.time, doc="Mass transfer constraint for R4")
    # def mass_transfer_R7(self, t):
    #     return pyo.units.convert(
    #         m.fs.R7.injection[t, "Liq", "S_O2"], to_units=pyo.units.kg / pyo.units.hour
    #     ) == (
    #         m.fs.R7.KLa
    #         * m.fs.R7.volume[t]
    #         * (m.fs.S_O_eq - m.fs.R7.outlet.conc_mass_comp[t, "S_O2"])
    #     )

    return m


def set_operating_conditions(m):
    # Feed Water Conditions
    # print(f"DOF before feed: {degrees_of_freedom(m)}")
    m.fs.FeedWater.flow_vol.fix(20935.15 * pyo.units.m**3 / pyo.units.day)
    m.fs.FeedWater.temperature.fix(308.15 * pyo.units.K)
    m.fs.FeedWater.pressure.fix(1 * pyo.units.atm)
    m.fs.FeedWater.conc_mass_comp[0, "S_O2"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_F"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_A"].fix(70 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_NH4"].fix(26.6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_NO3"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    # m.fs.FeedWater.conc_mass_comp[0, "S_PO4"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_PO4"].fix(15 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_I"].fix(57.45 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_N2"].fix(25.19 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_I"].fix(84 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_S"].fix(94.1 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_H"].fix(370 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_PAO"].fix(
        51.5262 * pyo.units.g / pyo.units.m**3
    )
    # m.fs.FeedWater.conc_mass_comp[0, "X_PAO"].fix(500 * pyo.units.g / pyo.units.m ** 3)
    m.fs.FeedWater.conc_mass_comp[0, "X_PP"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    # m.fs.FeedWater.conc_mass_comp[0, "X_PP"].fix(10 * pyo.units.g / pyo.units.m ** 3)
    m.fs.FeedWater.conc_mass_comp[0, "X_PHA"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_AUT"].fix(1e-6 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_IC"].fix(5.652 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_K"].fix(374.6925 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_Mg"].fix(20 * pyo.units.g / pyo.units.m**3)

    # Primary Clarifier
    # TODO: Update primary clarifier once more detailed model available
    m.fs.CL.split_fraction[0, "effluent", "H2O"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_A"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_F"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_I"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_N2"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_NH4"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_NO3"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_O2"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_PO4"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_IC"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_K"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_Mg"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "X_AUT"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_H"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_I"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_PAO"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_PHA"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_PP"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_S"].fix(0.5192)

    # Reactor sizing
    m.fs.R1.volume.fix(1000 * pyo.units.m**3)
    m.fs.R2.volume.fix(1000 * pyo.units.m**3)
    m.fs.R3.volume.fix(1500 * pyo.units.m**3)
    m.fs.R4.volume.fix(1500 * pyo.units.m**3)
    m.fs.R5.volume.fix(3000 * pyo.units.m**3)
    m.fs.R6.volume.fix(3000 * pyo.units.m**3)
    m.fs.R7.volume.fix(3000 * pyo.units.m**3)

    # Injection rates to Reactions 5, 6 and 7
    for j in m.fs.props_ASM2D.component_list:
        if j != "S_O2":
            # All components except S_O have no injection
            m.fs.R5.injection[:, :, j].fix(0)
            m.fs.R6.injection[:, :, j].fix(0)
            m.fs.R7.injection[:, :, j].fix(0)
    # Then set injections rates for O2
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].fix(1.91e-3)
    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].fix(2.60e-3)
    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].fix(3.20e-3)

    # KLa
    # m.fs.R5.KLa = 11
    # m.fs.R6.KLa = 7
    # m.fs.R7.KLa = 6

    # Set fraction of outflow from reactor 7 that goes to recycle
    m.fs.SP1.split_fraction[:, "underflow"].fix(0.60)

    # Secondary Clarifier
    # TODO: Update once more detailed model available
    m.fs.CL2.split_fraction[0, "effluent", "H2O"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_A"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_F"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_I"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_N2"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_NH4"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_NO3"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_O2"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_PO4"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_IC"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_K"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "S_Mg"].fix(0.48956)
    m.fs.CL2.split_fraction[0, "effluent", "X_AUT"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_H"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_I"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_PAO"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_PHA"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_PP"].fix(0.00187)
    m.fs.CL2.split_fraction[0, "effluent", "X_S"].fix(0.00187)

    m.fs.CL2.surface_area.fix(1500 * pyo.units.m**2)

    # Sludge purge separator
    m.fs.SP2.split_fraction[:, "recycle"].fix(0.985)

    # Outlet pressure from recycle pump
    m.fs.P1.outlet.pressure.fix(101325)

    # AD
    m.fs.AD.volume_liquid.fix(3400)
    m.fs.AD.volume_vapor.fix(300)
    m.fs.AD.liquid_outlet.temperature.fix(308.15)

    # Dewatering Unit - fix either HRT or volume.
    m.fs.dewater.hydraulic_retention_time.fix(1800 * pyo.units.s)

    # Thickener unit
    m.fs.thickener.hydraulic_retention_time.fix(86400 * pyo.units.s)
    m.fs.thickener.diameter.fix(10 * pyo.units.m)

    # ElectroNP
    if m.fs.has_electroNP is True:
        # m.fs.electroNP.energy_electric_flow_mass.fix(
        #     0.044 * pyo.units.kWh / pyo.units.kg
        # )
        m.fs.electroNP.cathodic_potential.fix(-1.1 * pyo.units.V)
        m.fs.electroNP.area_volume_ratio.fix(0.1 * pyo.units.cm**-1)
        m.fs.electroNP.settling_time.fix(30 * pyo.units.min)
        # m.fs.electroNP.magnesium_chloride_dosage.fix(0.388)
        # m.fs.electroNP.P_removal = 0.95
        # m.fs.electroNP.N_removal = 0.3
        m.fs.electroNP.frac_mass_H2O_treated[0].fix(0.9)
        # m.fs.electroNP.area[0].fix(5)
        m.fs.electroNP.HRT.fix(0.5 * pyo.units.hr)

        # These electroNP-specific variables were entirely unscaled (only
        # inlet.flow_vol had a factor below) -- with has_electroNP=True the
        # model gains 117 vars/constraints from this block, and the
        # "closing recycle" solve now diverges explosively in the first
        # ~17 iterations (inf_du: 3e3 -> 5.6e11) before falling into
        # restoration and ending at genuine local infeasibility. This is
        # the classic signature of newly-introduced unscaled variables.
        #   HRT / settling_time fixed at 1800 s (0.5 hr / 30 min) -- same
        #   order as the reactor HRTs already scaled elsewhere at 1e-3.
        #   cathodic_potential ~ -1.1 V, area_volume_ratio ~ 0.1 cm^-1,
        #   frac_mass_H2O_treated ~ 0.9 (dimensionless fraction) -- all
        #   O(1e-1 - 1e0), moderate factors below.
        iscale.set_scaling_factor(m.fs.electroNP.cathodic_potential, 1e0)
        iscale.set_scaling_factor(m.fs.electroNP.area_volume_ratio, 1e1)
        iscale.set_scaling_factor(m.fs.electroNP.settling_time, 1e-3)
        iscale.set_scaling_factor(m.fs.electroNP.HRT, 1e-3)
        iscale.set_scaling_factor(m.fs.electroNP.frac_mass_H2O_treated, 1e0)

        # T_surrogate, t_ss_surrogate, P_removal_surrogate, EI_surrogate are
        # PySMO RBF surrogate OUTPUT variables. The latest diagnostics run
        # (has_electroNP=True) now shows their real values with sf:None --
        # entirely unscaled, since these names don't match any pattern in
        # scale_variables():
        #   T_surrogate = 25, t_ss_surrogate = 30, P_removal_surrogate = 90,
        #   EI_surrogate = 0.044
        # The "closing recycle" solve fails at exactly this stage (inf_du
        # explodes 3e3 -> 5.6e11 within the first ~17 iterations before
        # ending in genuine local infeasibility), consistent with these
        # four unscaled variables being the last piece of the electroNP
        # block without a scaling factor. Now grounded in real values
        # rather than guessed.
        iscale.set_scaling_factor(m.fs.electroNP.T_surrogate, 1e-1)
        iscale.set_scaling_factor(m.fs.electroNP.t_ss_surrogate, 1e-1)
        iscale.set_scaling_factor(m.fs.electroNP.P_removal_surrogate, 1e-2)
        iscale.set_scaling_factor(m.fs.electroNP.EI_surrogate, 1e2)

    # Expressions
    m.fs.water_recovery = Expression(
        expr=(m.fs.Treated.flow_vol[0] / m.fs.FeedWater.flow_vol[0])
    )
    if m.fs.has_electroNP is True:
        m.fs.phosphorus_recovery = Expression(expr=(m.fs.electroNP.P_removal))


def set_scaling(m):
    def scale_variables(m):
        for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
            if "flow_vol" in var.name:
                iscale.set_scaling_factor(var, 1e1)
            if "temperature" in var.name:
                iscale.set_scaling_factor(var, 1e-2)
            if "pressure" in var.name:
                iscale.set_scaling_factor(var, 1e-5)
                # # for plotting
                # iscale.set_scaling_factor(var, 1e-3)
            # if "pressure_sat" in var.name:
            #     iscale.set_scaling_factor(var, 1e-3)
            # if "pressure_sat[S_h2]" in var.name:
            #     iscale.set_scaling_factor(var, 1e-2)
            if "conc_mass_comp" in var.name:
                iscale.set_scaling_factor(var, 1e2)
            # if "conc_mass_comp[S_h2]" in var.name:
            #     iscale.set_scaling_factor(var, 1e5)
            # if "conc_mass_comp[S_ch4]" in var.name:
            #     iscale.set_scaling_factor(var, 1e0)

    # scaling factor for ASM reactors
    for unit in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        block = getattr(m.fs, unit)
        iscale.set_scaling_factor(
            block.control_volume.reactions[0.0].rate_expression, 1e3
        )
        # rate_reaction_extent was previously unscaled (default sf=1), which
        # is the source of the bulk of "Missing scaling factor" warnings and
        # is why rate_reaction_extent[0.0,R18] shows up as an extreme
        # Jacobian column (~1.9E+04) unchanged across every diagnostic run so
        # far. It is ~ rate_expression * volume, so start at the same order
        # of magnitude as rate_expression; re-check diagnostics afterward --
        # R18 specifically may need its own override if it's still extreme.
        iscale.set_scaling_factor(block.control_volume.rate_reaction_extent, 1e3)
        iscale.set_scaling_factor(block.cstr_performance_eqn, 1e3)
        iscale.set_scaling_factor(
            block.control_volume.rate_reaction_stoichiometry_constraint, 1e3
        )
        iscale.set_scaling_factor(block.control_volume.material_balances, 1e3)
        # control_volume.volume had no scaling factor at all -- BSM2
        # reactor volumes are typically O(1e3) m3, so sf ~1e-3 brings the
        # scaled value to O(1); adjust if the next diagnostic run still
        # flags it.
        iscale.set_scaling_factor(block.control_volume.volume, 1e-3)

    # HRT for R5-R7 is ~1460 s and had no scaling factor at all (flagged as
    # badly scaled with sf: None in diagnostics) -> fix with a direct factor.
    m.fs.aerobic_reactors = (m.fs.R5, m.fs.R6, m.fs.R7)
    for R in m.fs.aerobic_reactors:
        iscale.set_scaling_factor(R.hydraulic_retention_time[0], 1e-3)

    # scaling factor for low flowrate units
    if m.fs.has_electroNP is True:
        m.fs.low_flowrate = (
            m.fs.thickener,
            m.fs.translator_asm2d_adm1,
            m.fs.AD,
            m.fs.translator_adm1_asm2d,
            m.fs.dewater,
            m.fs.electroNP,
            m.fs.MX4,
            m.fs.Sludge,
        )
    else:
        m.fs.low_flowrate = (
            m.fs.thickener,
            m.fs.translator_asm2d_adm1,
            m.fs.AD,
            m.fs.translator_adm1_asm2d,
            m.fs.dewater,
            m.fs.MX4,
            m.fs.Sludge,
        )

    # for unit in m.fs.low_flowrate:
    #     for var in unit.component_data_objects(pyo.Var, descend_into=True):
    #         if "flow_vol" in var.name:
    #             iscale.set_scaling_factor(var, 1e3)

    # # scaling factor of AD
    iscale.set_scaling_factor(m.fs.AD.volume_AD[0.0], 1e-3)
    iscale.set_scaling_factor(m.fs.AD.KH_h2[0.0], 1e4)
    iscale.set_scaling_factor(m.fs.AD.liquid_phase.reactions[0.0].pKW, 1e-1)
    iscale.set_scaling_factor(m.fs.AD.liquid_phase.reactions[0.0].S_H, 1e8)
    iscale.set_scaling_factor(m.fs.AD.liquid_phase.reactions[0.0].conc_mol_Mg, 1e5)
    iscale.set_scaling_factor(m.fs.AD.liquid_phase.reactions[0.0].conc_mol_K, 1e2)
    iscale.set_scaling_factor(m.fs.AD.vapor_phase[0.0].pressure_sat["H2O"], 1e-3)
    iscale.set_scaling_factor(m.fs.AD.vapor_phase[0.0].pressure_sat["S_h2"], 1e0)

    # scaling factor of electroNP
    if m.fs.has_electroNP is True:
        iscale.set_scaling_factor(m.fs.electroNP.inlet.flow_vol[0], 1e3)
        # iscale.set_scaling_factor(m.fs.electroNP.inlet.conc_mass_comp[0, "S_PO4"], 1e-2)

        # iscale.set_scaling_factor(m.fs.electroNP.T_surrogate, 1e-1)
        # iscale.set_scaling_factor(m.fs.electroNP.t_ss_surrogate, 1e-1)
        # iscale.set_scaling_factor(m.fs.electroNP.P_removal_surrogate, 1e-2)
        # iscale.set_scaling_factor(m.fs.electroNP.EI_surrogate, 1e1)

    # scaling factor of other units
    iscale.set_scaling_factor(m.fs.CL.surface_area, 1e-3)
    iscale.set_scaling_factor(m.fs.CL2.surface_area, 1e-3)
    iscale.set_scaling_factor(m.fs.dewater.volume[0.0], 1e-3)
    iscale.set_scaling_factor(m.fs.thickener.volume[0.0], 1e-3)
    # P1.control_volume.work had no scaling factor at all -- recycle pump
    # work on a low-pressure-rise stream like this is typically small
    # relative to the aeration/costing-scale quantities elsewhere in the
    # model; sf=1e-2 is a starting point, adjust if flagged in diagnostics.
    iscale.set_scaling_factor(m.fs.P1.control_volume.work, 1e-2)

    # Apply scaling
    scale_variables(m)

    csb = CustomScalerBase()
    auto = AutoScaler()

    # scaling factor of variables with extreme Jacobian
    auto.scale_variables_by_magnitude(m.fs.dewater.mixed_state[0.0].flow_vol)
    auto.scale_variables_by_magnitude(m.fs.AD.liquid_phase.properties_in[0.0].flow_vol)

    # R24 (Lysis of X_PP) was previously scaled with
    # auto.scale_variables_by_magnitude(...) / scale_constraint_by_nominal_value(...),
    # both of which are evaluated HERE in set_scaling(), i.e. BEFORE
    # initialize_system(m) runs. At this point conc_mass_comp["X_PP"] still
    # holds its un-initialized placeholder value (~0), so inverseMaximum
    # scaling divided by a near-zero coefficient and produced an enormous
    # scaling factor. Diagnostics confirmed this: rate_expression[R24] and
    # conc_mass_comp[X_PP] both showed Jacobian norms of 6.75E+09 -- five
    # orders of magnitude worse than the next-worst entry (~7.3E+08).
    #
    # Fix: use fixed, physically-derived scaling factors instead of a
    # value-dependent scheme.
    #   b_PP ~ 0.2 /day = 0.2/86400 /s ~ 2.3e-6 /s
    #   X_PP (ADM1, kg P/m3, post 1/31 fix) ~ O(0.01-1)
    #   => rate_expression[R24] = b_PP * X_PP ~ O(1e-8 - 1e-6) kg/m3/s
    #   => scaling factor (1/magnitude) ~ 1e7
    iscale.set_scaling_factor(
        m.fs.AD.liquid_phase.reactions[0.0].reaction_rate["R24"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.AD.liquid_phase.reactions[0.0].rate_expression["R24"], 1e7
    )
    csb.scale_constraint_by_nominal_value(
        m.fs.AD.AD_retention_time[0.0],
        scheme=ConstraintScalingScheme.inverseMaximum,
        overwrite=True,
    )

    csb.scale_constraint_by_nominal_value(
        m.fs.MX1.enthalpy_mixing_equations[0.0],
        scheme=ConstraintScalingScheme.inverseMaximum,
        overwrite=True,
    )
    csb.scale_constraint_by_nominal_value(
        m.fs.MX2.enthalpy_mixing_equations[0.0],
        scheme=ConstraintScalingScheme.inverseMaximum,
        overwrite=True,
    )
    csb.scale_constraint_by_nominal_value(
        m.fs.MX3.enthalpy_mixing_equations[0.0],
        scheme=ConstraintScalingScheme.inverseMaximum,
        overwrite=True,
    )
    csb.scale_constraint_by_nominal_value(
        m.fs.MX4.enthalpy_mixing_equations[0.0],
        scheme=ConstraintScalingScheme.inverseMaximum,
        overwrite=True,
    )
    csb.scale_constraint_by_nominal_value(
        m.fs.AD.liquid_phase.reactions[0.0].pH_calc,
        scheme=ConstraintScalingScheme.inverseMaximum,
        overwrite=True,
    )

    # Scaling adjustments for Pi["X_PP"] = 1/31 change
    # X_PP in ADM1 is now treated as kg P/m3 (like S_IP), so the
    # SPO4_output constraint and related translator constraints have
    # smaller magnitudes and need rescaling
    csb.scale_constraint_by_nominal_value(
        m.fs.translator_adm1_asm2d.SPO4_output[0],
        scheme=ConstraintScalingScheme.inverseMaximum,
        overwrite=True,
    )
    csb.scale_constraint_by_nominal_value(
        m.fs.translator_adm1_asm2d.SK_output[0],
        scheme=ConstraintScalingScheme.inverseMaximum,
        overwrite=True,
    )
    csb.scale_constraint_by_nominal_value(
        m.fs.translator_adm1_asm2d.SMg_output[0],
        scheme=ConstraintScalingScheme.inverseMaximum,
        overwrite=True,
    )
    # X_PP in ADM1 properties — now comparable to S_IP magnitude
    for props in [
        m.fs.AD.liquid_phase.properties_in[0],
        m.fs.AD.liquid_phase.properties_out[0],
        m.fs.translator_asm2d_adm1.properties_out[0],
        m.fs.translator_adm1_asm2d.properties_in[0],
    ]:
        iscale.set_scaling_factor(props.conc_mass_comp["X_PP"], 1e1)
        iscale.set_scaling_factor(props.conc_mass_comp["S_IP"], 1e0)

    # The blanket "conc_mass_comp -> sf 1e2" rule in scale_variables() spans
    # ~9 orders of magnitude on the ADM1 side (S_ch4 ~1e-9 to X_I ~13) and is
    # the real source of the "badly scaled vars" list -- almost every entry
    # in it is one of these four property blocks. Override per-species with
    # factors tuned to their typical magnitude in this flowsheet.
    adm1_conc_sf = {
        "S_su": 1e0,
        "S_aa": 1e0,
        "S_fa": 1e0,
        "S_va": 1e1,
        "S_bu": 1e1,
        "S_pro": 1e1,
        "S_ac": 1e1,
        "S_h2": 1e6,
        "S_ch4": 1e2,
        "S_IC": 1e0,
        "S_IN": 1e0,
        "X_ch": 1e1,
        "X_pr": 1e1,
        "X_li": 1e1,
        "X_su": 1e6,
        "X_aa": 1e1,
        "X_fa": 1e2,
        "X_c4": 1e2,
        "X_pro": 1e2,
        "X_ac": 1e2,
        "X_h2": 1e1,
        "X_I": 1e0,
        "X_PHA": 1e0,
        "X_PAO": 1e1,
        "S_K": 1e1,
        "S_Mg": 1e1,
    }
    for props in [
        m.fs.AD.liquid_phase.properties_in[0],
        m.fs.AD.liquid_phase.properties_out[0],
        m.fs.translator_asm2d_adm1.properties_out[0],
        m.fs.translator_adm1_asm2d.properties_in[0],
    ]:
        for comp, sf in adm1_conc_sf.items():
            if comp in props.conc_mass_comp:
                iscale.set_scaling_factor(props.conc_mass_comp[comp], sf)

    iscale.calculate_scaling_factors(m)


def initialize_system(m):
    # Deactivate extra constraints
    for mx in m.fs.mixers:
        mx.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 3].deactivate()
    # print(f"DOF before initialization: {degrees_of_freedom(m)}")

    # Initialize flowsheet
    # Apply sequential decomposition - 1 iteration should suffice
    seq = SequentialDecomposition()
    seq.options.tear_method = "Direct"
    # Reverted: iterLim=5 caused seq.run() itself to fail (a Mixer's
    # initialize_build hit a hard solver error), earlier and worse than
    # every prior run in this session, all of which got past seq.run fine
    # and only failed later at the "closing recycle" solve. Back to 1.
    seq.options.iterLim = 1
    # seq.options.tear_set = [m.fs.stream5, m.fs.stream10adm]
    seq.options.tear_set = [m.fs.stream2, m.fs.stream5, m.fs.stream10adm]

    # G = seq.create_graph(m)
    # # Uncomment this code to see tear set and initialization order
    # order = seq.calculation_order(G)
    # print("Initialization Order")
    # for o in order:
    #     print(o[0].name)

    if m.fs.has_electroNP is True:
        # with HRT specified
        # tear_guesses0 = {
        #     "flow_vol": {0: 0.495},
        #     "conc_mass_comp": {
        #         (0, "S_A"): 0.08,
        #         (0, "S_F"): 0.13,
        #         (0, "S_I"): 0.057,
        #         (0, "S_N2"): 0.035,
        #         (0, "S_NH4"): 0.02,
        #         (0, "S_NO3"): 0.0015,
        #         (0, "S_O2"): 0.0016,
        #         (0, "S_PO4"): 0.05,
        #         (0, "S_K"): 0.37,
        #         (0, "S_Mg"): 0.02,
        #         (0, "S_IC"): 0.09,
        #         (0, "X_AUT"): 0.07,
        #         (0, "X_H"): 3.5,
        #         (0, "X_I"): 3.1,
        #         (0, "X_PAO"): 3.4,
        #         (0, "X_PHA"): 0.002,
        #         (0, "X_PP"): 1.1,
        #         (0, "X_S"): 0.08,
        #     },
        #     "temperature": {0: 308.15},
        #     "pressure": {0: 101325},
        # }
        #
        # tear_guesses = {
        #     "flow_vol": {0: 1.235},
        #     "conc_mass_comp": {
        #         (0, "S_A"): 0.0006,
        #         (0, "S_F"): 0.0004,
        #         (0, "S_I"): 0.057,
        #         (0, "S_N2"): 0.04,
        #         (0, "S_NH4"): 0.006,
        #         (0, "S_NO3"): 0.002,
        #         (0, "S_O2"): 0.0019,
        #         (0, "S_PO4"): 0.04,
        #         (0, "S_K"): 0.37,
        #         (0, "S_Mg"): 0.020,
        #         (0, "S_IC"): 0.13,
        #         (0, "X_AUT"): 0.074,
        #         (0, "X_H"): 3.4,
        #         (0, "X_I"): 3.1,
        #         (0, "X_PAO"): 3.4,
        #         (0, "X_PHA"): 0.088,
        #         (0, "X_PP"): 1.14,
        #         (0, "X_S"): 0.057,
        #     },
        #     "temperature": {0: 308.15},
        #     "pressure": {0: 101325},
        # }
        #
        # tear_guesses2 = {
        #     "flow_vol": {0: 0.003},
        #     "conc_mass_comp": {
        #         (0, "S_A"): 0.1,
        #         (0, "S_F"): 0.15,
        #         (0, "S_I"): 0.057,
        #         (0, "S_N2"): 0.033,
        #         (0, "S_NH4"): 0.025,
        #         (0, "S_NO3"): 0.0013,
        #         (0, "S_O2"): 0.0013,
        #         (0, "S_PO4"): 0.05,
        #         (0, "S_K"): 0.37,
        #         (0, "S_Mg"): 0.024,
        #         (0, "S_IC"): 0.075,
        #         (0, "X_AUT"): 0.19,
        #         (0, "X_H"): 23,
        #         (0, "X_I"): 11,
        #         (0, "X_PAO"): 10,
        #         (0, "X_PHA"): 0.005,
        #         (0, "X_PP"): 2.8,
        #         (0, "X_S"): 3.9,
        #     },
        #     "temperature": {0: 308.15},
        #     "pressure": {0: 101325},
        # }

        # Bootstrapped from the near-feasible point (same one as before --
        # Using the EXACT SAME tear guesses as has_electroNP=False. Combined
        # with the pass-through homotopy (P_removal~0, frac_mass_H2O_treated
        # ~1), the system should be nearly mathematically identical to
        # has_electroNP=False, which we know converges cleanly with these
        # exact values. If this still fails at the very first homotopy
        # step, that rules out tear guesses as a factor too and points to
        # something structural (e.g. AD's own initialize_build(), which
        # has failed with its own "locally infeasible" warning in every
        # has_electroNP=True run this session, independent of anything
        # we've changed downstream of it).
        tear_guesses0 = {
            "flow_vol": {0: 0.495},
            "conc_mass_comp": {
                (0, "S_A"): 0.08,
                (0, "S_F"): 0.13,
                (0, "S_I"): 0.057,
                (0, "S_N2"): 0.05,
                (0, "S_NH4"): 0.025,
                (0, "S_NO3"): 0.005,
                (0, "S_O2"): 0.0016,
                (0, "S_PO4"): 0.65,
                (0, "S_K"): 0.37,
                (0, "S_Mg"): 0.02,
                (0, "S_IC"): 0.085,
                (0, "X_AUT"): 0.18,
                (0, "X_H"): 3.7,
                (0, "X_I"): 3.2,
                (0, "X_PAO"): 2.8,
                (0, "X_PHA"): 0.0011,
                (0, "X_PP"): 0.92,
                (0, "X_S"): 0.08,
            },
            "temperature": {0: 308.15},
            "pressure": {0: 101325},
        }

        tear_guesses = {
            "flow_vol": {0: 1.237},
            "conc_mass_comp": {
                (0, "S_A"): 0.0008,
                (0, "S_F"): 0.0004,
                (0, "S_I"): 0.057,
                (0, "S_N2"): 0.06,
                (0, "S_NH4"): 0.01,
                (0, "S_NO3"): 0.006,
                (0, "S_O2"): 0.0019,
                (0, "S_PO4"): 0.64,
                (0, "S_K"): 0.37,
                (0, "S_Mg"): 0.020,
                (0, "S_IC"): 0.13,
                (0, "X_AUT"): 0.18,
                (0, "X_H"): 3.7,
                (0, "X_I"): 3.2,
                (0, "X_PAO"): 2.8,
                (0, "X_PHA"): 0.076,
                (0, "X_PP"): 0.93,
                (0, "X_S"): 0.057,
            },
            "temperature": {0: 308.15},
            "pressure": {0: 101325},
        }

        tear_guesses2 = {
            "flow_vol": {0: 0.003},
            "conc_mass_comp": {
                (0, "S_A"): 0.1,
                (0, "S_F"): 0.15,
                (0, "S_I"): 0.057,
                (0, "S_N2"): 0.04,
                (0, "S_NH4"): 0.03,
                (0, "S_NO3"): 0.004,
                (0, "S_O2"): 0.0013,
                (0, "S_PO4"): 0.65,
                (0, "S_K"): 0.38,
                (0, "S_Mg"): 0.024,
                (0, "S_IC"): 0.07,
                (0, "X_AUT"): 0.47,
                (0, "X_H"): 24,
                (0, "X_I"): 12,
                (0, "X_PAO"): 9.2,
                (0, "X_PHA"): 0.0028,
                (0, "X_PP"): 2.4,
                (0, "X_S"): 4.0,
            },
            "temperature": {0: 308.15},
            "pressure": {0: 101325},
        }

    else:
        tear_guesses0 = {
            "flow_vol": {0: 0.495},
            "conc_mass_comp": {
                (0, "S_A"): 0.08,
                (0, "S_F"): 0.13,
                (0, "S_I"): 0.057,
                (0, "S_N2"): 0.05,
                (0, "S_NH4"): 0.025,
                (0, "S_NO3"): 0.005,
                (0, "S_O2"): 0.0016,
                (0, "S_PO4"): 0.65,
                (0, "S_K"): 0.37,
                (0, "S_Mg"): 0.02,
                (0, "S_IC"): 0.085,
                (0, "X_AUT"): 0.18,
                (0, "X_H"): 3.7,
                (0, "X_I"): 3.2,
                (0, "X_PAO"): 2.8,
                (0, "X_PHA"): 0.0011,
                (0, "X_PP"): 0.92,
                (0, "X_S"): 0.08,
            },
            "temperature": {0: 308.15},
            "pressure": {0: 101325},
        }

        tear_guesses = {
            "flow_vol": {0: 1.237},
            "conc_mass_comp": {
                (0, "S_A"): 0.0008,
                (0, "S_F"): 0.0004,
                (0, "S_I"): 0.057,
                (0, "S_N2"): 0.06,
                (0, "S_NH4"): 0.01,
                (0, "S_NO3"): 0.006,
                (0, "S_O2"): 0.0019,
                (0, "S_PO4"): 0.64,
                (0, "S_K"): 0.37,
                (0, "S_Mg"): 0.020,
                (0, "S_IC"): 0.13,
                (0, "X_AUT"): 0.18,
                (0, "X_H"): 3.7,
                (0, "X_I"): 3.2,
                (0, "X_PAO"): 2.8,
                (0, "X_PHA"): 0.076,
                (0, "X_PP"): 0.93,
                (0, "X_S"): 0.057,
            },
            "temperature": {0: 308.15},
            "pressure": {0: 101325},
        }

        tear_guesses2 = {
            "flow_vol": {0: 0.003},
            "conc_mass_comp": {
                (0, "S_A"): 0.1,
                (0, "S_F"): 0.15,
                (0, "S_I"): 0.057,
                (0, "S_N2"): 0.04,
                (0, "S_NH4"): 0.03,
                (0, "S_NO3"): 0.004,
                (0, "S_O2"): 0.0013,
                (0, "S_PO4"): 0.65,
                (0, "S_K"): 0.38,
                (0, "S_Mg"): 0.024,
                (0, "S_IC"): 0.07,
                (0, "X_AUT"): 0.47,
                (0, "X_H"): 24,
                (0, "X_I"): 12,
                (0, "X_PAO"): 9.2,
                (0, "X_PHA"): 0.0028,
                (0, "X_PP"): 2.4,
                (0, "X_S"): 4.0,
            },
            "temperature": {0: 308.15},
            "pressure": {0: 101325},
        }

    # Pass the tear_guess to the SD tool
    # seq.set_guesses_for(m.fs.CL.inlet, tear_guesses_CL)
    seq.set_guesses_for(m.fs.R1.inlet, tear_guesses0)
    seq.set_guesses_for(m.fs.R3.inlet, tear_guesses)
    seq.set_guesses_for(m.fs.translator_asm2d_adm1.inlet, tear_guesses2)

    def function(unit):
        # unit.initialize(outlvl=idaeslog.INFO, solver="ipopt-watertap")
        unit.initialize(solver="ipopt-watertap", outlvl=idaeslog.CRITICAL)

    results = seq.run(m, function)

    # # Print actual tear-stream values after seq.run so we can build new,
    # # grounded tear_guesses for has_electroNP=True (the current guesses
    # # predate the Pi["X_PP"] fix, translator fix, and all scaling changes
    # # made this session). These reflect what the real unit models/
    # # constraints produce given the current guess as a starting point --
    # # not another blind estimate.
    # if m.fs.has_electroNP is True:
    #     for name, port in (
    #         ("R1.inlet", m.fs.R1.inlet),
    #         ("R3.inlet", m.fs.R3.inlet),
    #         ("translator_asm2d_adm1.inlet", m.fs.translator_asm2d_adm1.inlet),
    #     ):
    #         print(f"\n---- {name} after seq.run ----")
    #         print(f"flow_vol: {pyo.value(port.flow_vol[0])}")
    #         print(f"temperature: {pyo.value(port.temperature[0])}")
    #         print(f"pressure: {pyo.value(port.pressure[0])}")
    #         for (t, j), v in port.conc_mass_comp.items():
    #             print(f'(0, "{j}"): {pyo.value(v)},')

    # Deactivate extra constraints
    for mx in m.fs.mixers:
        mx.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 2].deactivate()
    m.fs.MX3.pressure_equality_constraints[0.0, 3].deactivate()
    # print(f"DOF before initialization: {degrees_of_freedom(m)}")

    return m, results


def solve(m, solver=None):
    if solver is None:
        solver = get_solver()
    results = solver.solve(m, tee=True)
    check_solve(results, checkpoint="closing recycle", logger=_log, fail_flag=True)
    pyo.assert_optimal_termination(results)
    return results


def add_costing(m):
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyo.units.USD_2023

    # Costing Blocks
    m.fs.R1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R2.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R3.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R4.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R5.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R6.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R7.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.CL.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_primary_clarifier,
    )

    m.fs.CL2.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_circular_clarifier,
    )
    # m.fs.P1.costing = UnitModelCostingBlock(
    #     flowsheet_costing_block=m.fs.costing,
    #     costing_method=cost_pump,
    #     costing_method_arguments={"pump_type": PumpType.low_pressure},
    # )

    m.fs.AD.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.dewater.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.thickener.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    if m.fs.has_electroNP is True:
        m.fs.electroNP.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing
        )

    # TODO: Leaving out mixer costs; consider including later

    # process costing and add system level metrics
    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.Treated.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.FeedWater.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.FeedWater.properties[0].flow_vol)

    # Set scaling:
    iscale.set_scaling_factor(m.fs.costing.total_capital_cost, 1e-6)

    for block in m.fs.component_objects(pyo.Block, descend_into=True):
        if isinstance(block, UnitModelBlockData) and hasattr(block, "costing"):
            iscale.set_scaling_factor(block.costing.capital_cost, 1e-6)

    # scaling constraints
    csb = CustomScalerBase()
    csb.scale_constraint_by_nominal_value(
        m.fs.AD.costing.capital_cost_constraint,
        scheme=ConstraintScalingScheme.inverseMaximum,
        overwrite=True,
    )
    csb.scale_constraint_by_nominal_value(
        m.fs.dewater.costing.capital_cost_constraint,
        scheme=ConstraintScalingScheme.inverseMaximum,
        overwrite=True,
    )

    # Expression
    if m.fs.has_electroNP is True:
        m.fs.costing.LCOW_P_removal = Expression(
            expr=(
                m.fs.costing.total_capital_cost * m.fs.costing.capital_recovery_factor
                + m.fs.costing.total_operating_cost
            )
            / (
                pyo.units.convert(
                    # m.fs.electroNP.byproduct.flow_vol[0]
                    # * m.fs.electroNP.byproduct.conc_mass_comp[0, "S_PO4"],
                    m.fs.electroNP.inlet.flow_vol[0]
                    * m.fs.electroNP.inlet.conc_mass_comp[0, "S_PO4"]
                    * m.fs.electroNP.P_removal,
                    to_units=pyo.units.kg / m.fs.costing.base_period,
                )
                * m.fs.costing.utilization_factor
            )
        )

        m.fs.costing.specific_energy_consumption_P_removal = Expression(
            expr=(
                m.fs.costing.aggregate_flow_electricity
                / pyo.units.convert(
                    m.fs.electroNP.byproduct.flow_vol[0]
                    * m.fs.electroNP.byproduct.conc_mass_comp[0, "S_PO4"],
                    to_units=pyo.units.kg / pyo.units.hr,
                )
            )
        )

        m.fs.costing.electroNP_energy_consumption_side_stream = Expression(
            expr=(
                m.fs.electroNP.electricity[0]
                / pyo.units.convert(
                    m.fs.electroNP.properties_in[0].flow_vol,
                    to_units=pyo.units.m**3 / pyo.units.hr,
                )
            )
        )

        m.fs.costing.electroNP_energy_consumption = Expression(
            expr=(
                m.fs.electroNP.electricity[0]
                / pyo.units.convert(
                    m.fs.FeedWater.properties[0].flow_vol,
                    to_units=pyo.units.m**3 / pyo.units.hr,
                )
            )
        )

        m.fs.costing.electrode_energy_consumption = Expression(
            expr=(
                (
                    m.fs.electroNP.energy_electric_flow_mass
                    * pyo.units.convert(
                        m.fs.electroNP.properties_byproduct[0].get_material_flow_terms(
                            "Liq", "S_PO4"
                        ),
                        to_units=pyo.units.kg / pyo.units.hour,
                    )
                )
                / pyo.units.convert(
                    m.fs.FeedWater.properties[0].flow_vol,
                    to_units=pyo.units.m**3 / pyo.units.hr,
                )
            )
        )

        m.fs.costing.dryer_energy_consumption = Expression(
            expr=(
                (
                    m.fs.electroNP.ratio_electricity_intensity_dryer
                    * m.fs.electroNP.energy_electric_flow_mass
                    * pyo.units.convert(
                        m.fs.electroNP.properties_byproduct[0].get_material_flow_terms(
                            "Liq", "S_PO4"
                        ),
                        to_units=pyo.units.kg / pyo.units.hour,
                    )
                )
                / pyo.units.convert(
                    m.fs.FeedWater.properties[0].flow_vol,
                    to_units=pyo.units.m**3 / pyo.units.hr,
                )
            )
        )

        m.fs.costing.centrifuge_energy_consumption = Expression(
            expr=(
                (
                    m.fs.electroNP.electricity_intensity_centrifuge
                    * pyo.units.convert(
                        m.fs.electroNP.properties_in[0].flow_vol,
                        to_units=pyo.units.m**3 / pyo.units.hour,
                    )
                )
                / pyo.units.convert(
                    m.fs.FeedWater.properties[0].flow_vol,
                    to_units=pyo.units.m**3 / pyo.units.hr,
                )
            )
        )

        m.fs.costing.electroNP_pump_energy_consumption = Expression(
            expr=(
                (
                    3
                    * m.fs.electroNP.electricity_intensity_pump
                    * pyo.units.convert(
                        m.fs.electroNP.properties_in[0].flow_vol,
                        to_units=pyo.units.m**3 / pyo.units.hour,
                    )
                )
                / pyo.units.convert(
                    m.fs.FeedWater.properties[0].flow_vol,
                    to_units=pyo.units.m**3 / pyo.units.hr,
                )
            )
        )

    m.fs.costing.aeration_energy = Expression(
        expr=(
            (
                m.fs.R5.electricity_consumption[0]
                + m.fs.R6.electricity_consumption[0]
                + m.fs.R7.electricity_consumption[0]
            )
            / pyo.units.convert(
                m.fs.FeedWater.properties[0].flow_vol,
                to_units=pyo.units.m**3 / pyo.units.hr,
            )
        )
    )


def setup_optimization(
    m,
    objective=objective_fun.LCOW,
    has_effluent_constraints=False,
    reactor_volume_equalities=False,
):
    # Objective function
    if objective == objective_fun.LCOW:
        m.fs.objective = pyo.Objective(expr=m.fs.costing.LCOW)
    elif objective == objective_fun.LCOP:
        m.fs.objective = pyo.Objective(expr=m.fs.costing.LCOW_P_removal)
    else:
        raise TypeError(
            f'objective must be set to "LCOW"  or "LCOP".'
            f" objective was set to {objective}"
        )

    # Decision variables
    if m.fs.has_electroNP is True:
        m.fs.electroNP.cathodic_potential.unfix()
        m.fs.electroNP.cathodic_potential.setlb(-1.3)
        m.fs.electroNP.cathodic_potential.setub(-0.8)

        m.fs.electroNP.area_volume_ratio.unfix()
        m.fs.electroNP.area_volume_ratio.setlb(0.065)
        m.fs.electroNP.area_volume_ratio.setub(0.145)

    # for i in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
    #     reactor = getattr(m.fs, i)
    #     reactor.volume.unfix()
    #     reactor.volume.setlb(1)
    #     reactor.volume.setub(5000)
    # if reactor_volume_equalities:
    #     add_reactor_volume_equalities(m)

    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R5.outlet.conc_mass_comp[:, "S_O2"].setub(8e-3)

    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R6.outlet.conc_mass_comp[:, "S_O2"].setub(8e-3)

    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].unfix()
    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].setlb(0)
    m.fs.R7.outlet.conc_mass_comp[:, "S_O2"].setub(8e-3)

    # m.fs.R5.injection[:, :, :].unfix()
    # m.fs.R6.injection[:, :, :].unfix()
    # m.fs.R7.injection[:, :, :].unfix()

    # # Unfix fraction of outflow from reactor 7 that goes to recycle
    # m.fs.SP1.split_fraction[:, "underflow"].unfix()
    # # m.fs.SP1.split_fraction[:, "underflow"].setlb(0.45)
    # m.fs.SP2.split_fraction[:, "recycle"].unfix()

    if has_effluent_constraints:
        add_effluent_violations(m)
        # if m.fs.has_electroNP is False:
        #     m.fs.total_P_max.unfix()
        #     m.fs.total_P_max.fix(0.6)


def add_reactor_volume_equalities(m):
    # TODO: These constraints were applied for initial optimization of AS reactor volumes; otherwise, volumes drive towards lower bound. Revisit
    @m.fs.Constraint(m.fs.time)
    def Vol_1(self, t):
        return m.fs.R1.volume[0] == m.fs.R2.volume[0]

    @m.fs.Constraint(m.fs.time)
    def Vol_2(self, t):
        return m.fs.R3.volume[0] == m.fs.R4.volume[0]

    @m.fs.Constraint(m.fs.time)
    def Vol_3(self, t):
        return m.fs.R5.volume[0] == m.fs.R6.volume[0]

    @m.fs.Constraint(m.fs.time)
    def Vol_4(self, t):
        return m.fs.R7.volume[0] >= m.fs.R6.volume[0] * 0.5


def add_effluent_violations(m):
    # TODO: Revisit the max effluent concentration values

    # Max value taken from Flores-Alsina Excel 0.03 - modified to 0.05
    m.fs.TSS_max = pyo.Var(initialize=0.05, units=pyo.units.kg / pyo.units.m**3)
    m.fs.TSS_max.fix()

    @m.fs.Constraint(m.fs.time)
    def eq_TSS_max(self, t):
        return m.fs.Treated.properties[t].TSS <= m.fs.TSS_max

    # Max value carried over from BSM2
    m.fs.COD_max = pyo.Var(initialize=0.1, units=pyo.units.kg / pyo.units.m**3)
    m.fs.COD_max.fix()

    @m.fs.Constraint(m.fs.time)
    def eq_COD_max(self, t):
        return m.fs.Treated.properties[t].COD <= m.fs.COD_max

    # Max value taken from Flores-Alsina Excel 0.004 - modified to 0.007
    m.fs.TKN_max = pyo.Var(initialize=0.007, units=pyo.units.kg / pyo.units.m**3)
    m.fs.TKN_max.fix()

    @m.fs.Constraint(m.fs.time)
    def eq_TKN_max(self, t):
        return m.fs.Treated.properties[t].TKN <= m.fs.TKN_max

    # Max value carried over from BSM2
    m.fs.BOD5_max = pyo.Var(initialize=0.01, units=pyo.units.kg / pyo.units.m**3)
    m.fs.BOD5_max.fix()

    @m.fs.Constraint(m.fs.time)
    def eq_BOD5_max(self, t):
        return m.fs.Treated.properties[t].BOD5["effluent"] <= m.fs.BOD5_max

    # Max value taken from Flores-Alsina Excel 0.002 - modified to 0.005
    m.fs.total_P_max = pyo.Var(initialize=0.005, units=pyo.units.kg / pyo.units.m**3)
    m.fs.total_P_max.fix()

    @m.fs.Constraint(m.fs.time)
    def eq_total_P_max(self, t):
        return (
            m.fs.Treated.properties[0].SP_organic
            + m.fs.Treated.properties[0].SP_inorganic
            <= m.fs.total_P_max
        )


def display_costing(m):
    print("\n--- Costing Metrics ---")
    print("Levelized cost of water: %.3f $/m3" % pyo.value(m.fs.costing.LCOW))
    if m.fs.has_electroNP is True:
        print(
            "Levelized cost of phosphorus removal: %.3f $/kg"
            % pyo.value(m.fs.costing.LCOW_P_removal)
        )
    print(
        "Total annualized cost: %.3f M$/yr"
        % pyo.value(m.fs.costing.total_annualized_cost / 1e6)
    )

    print(
        "\nTotal capital cost: %.3f M$"
        % pyo.value(m.fs.costing.total_capital_cost / 1e6)
    )

    # print("capital cost R1: %.3f M$" % pyo.value(m.fs.R1.costing.capital_cost / 1e6))
    # print("capital cost R2: %.3f M$" % pyo.value(m.fs.R2.costing.capital_cost / 1e6))
    # print("capital cost R3: %.3f M$" % pyo.value(m.fs.R3.costing.capital_cost / 1e6))
    # print("capital cost R4: %.3f M$" % pyo.value(m.fs.R4.costing.capital_cost / 1e6))
    # print("capital cost R5: %.3f M$" % pyo.value(m.fs.R5.costing.capital_cost / 1e6))
    # print("capital cost R6: %.3f M$" % pyo.value(m.fs.R6.costing.capital_cost / 1e6))
    # print("capital cost R7: %.3f M$" % pyo.value(m.fs.R7.costing.capital_cost / 1e6))
    # print(
    #     "capital cost activated sludge reactors: %.3f M$"
    #     % pyo.value(
    #         (
    #             m.fs.R1.costing.capital_cost
    #             + m.fs.R2.costing.capital_cost
    #             + m.fs.R3.costing.capital_cost
    #             + m.fs.R4.costing.capital_cost
    #             + +m.fs.R5.costing.capital_cost
    #             + m.fs.R6.costing.capital_cost
    #             + m.fs.R7.costing.capital_cost
    #         )
    #         / 1e6
    #     )
    # )
    # print(
    #     "capital cost primary clarifier: %.3f M$"
    #     % pyo.value(m.fs.CL.costing.capital_cost / 1e6)
    # )
    # print(
    #     "capital cost secondary clarifier: %.3f M$"
    #     % pyo.value(m.fs.CL2.costing.capital_cost / 1e6)
    # )
    # print("capital cost AD: %.3f M$" % pyo.value(m.fs.AD.costing.capital_cost / 1e6))
    # print(
    #     "capital cost dewatering Unit: %.3f M$"
    #     % pyo.value(m.fs.dewater.costing.capital_cost / 1e6)
    # )
    # print(
    #     "capital cost thickener unit: %.3f M$"
    #     % pyo.value(m.fs.thickener.costing.capital_cost / 1e6)
    # )
    if m.fs.has_electroNP is True:
        print(
            "capital cost electroNP unit: %.3f M$"
            % pyo.value(m.fs.electroNP.costing.capital_cost / 1e6)
        )

    print(
        "\nTotal operating cost: %.3f M$/yr"
        % pyo.value(m.fs.costing.total_operating_cost / 1e6)
    )
    print(
        "Total fixed operating cost: %.3f M$/yr"
        % pyo.value(m.fs.costing.total_fixed_operating_cost / 1e6)
    )
    if m.fs.has_electroNP is True:
        print(
            "Total variable operating cost: %.3f M$/yr"
            % pyo.value(
                (
                    m.fs.costing.total_variable_operating_cost
                    - m.fs.costing.aggregate_flow_costs["phosphorus salt product"]
                    * m.fs.costing.utilization_factor
                )
                / 1e6
            )
        )
        print(
            "Revenue: %.3f M$/yr"
            % pyo.value(
                -m.fs.costing.aggregate_flow_costs["phosphorus salt product"]
                * m.fs.costing.utilization_factor
                / 1e6
            )
        )
    else:
        print(
            "Total variable operating cost: %.3f M$/yr"
            % pyo.value(m.fs.costing.total_variable_operating_cost / 1e6)
        )

    print(
        "Electricity cost: %.3f M$/yr"
        % pyo.value(
            (
                m.fs.costing.aggregate_flow_costs["electricity"]
                * m.fs.costing.utilization_factor
            )
            / 1e6
        )
    )


def _TP_conc(m, props):
    """Return total phosphorus [kg P / m3] for any ModifiedASM2d StateBlockData."""
    p = m.fs.props_ASM2D
    c = props.conc_mass_comp
    return (
        c["S_PO4"]
        + p.i_PSI * c["S_I"]
        + p.i_PSF * c["S_F"]
        + p.i_PXI * c["X_I"]
        + p.i_PXS * c["X_S"]
        + p.i_PBM * (c["X_H"] + c["X_PAO"] + c["X_AUT"])
        + c["X_PP"]
    )


def _TP_conc_adm1(m, props):
    """Return total phosphorus [kg P / m3] for any ModifiedADM1 StateBlockData.

    TP = S_IP  (already kg P/m3)
       + sum over all particulates with Pi defined: X_comp * Pi[comp] * mw_p
    Mirrors the same logic as _TP_conc for ASM2d.
    """
    p = m.fs.rxn_props_ADM1
    c = props.conc_mass_comp
    mw_p = 31  # kg/kmol — same as defined locally in the reaction package
    return (
        c["S_IP"]
        + p.Pi["S_I"] * mw_p * c["S_I"]
        + p.Pi["X_li"] * mw_p * c["X_li"]
        + p.Pi["X_su"] * mw_p * c["X_su"]
        + p.Pi["X_aa"] * mw_p * c["X_aa"]
        + p.Pi["X_fa"] * mw_p * c["X_fa"]
        + p.Pi["X_c4"] * mw_p * c["X_c4"]
        + p.Pi["X_pro"] * mw_p * c["X_pro"]
        + p.Pi["X_ac"] * mw_p * c["X_ac"]
        + p.Pi["X_h2"] * mw_p * c["X_h2"]
        + p.Pi["X_I"] * mw_p * c["X_I"]
        + p.Pi["X_PP"] * mw_p * c["X_PP"]
        + p.Pi["X_PAO"] * mw_p * c["X_PAO"]
    )


def display_stream_table(m):
    """Print the full per-species stream table (flow, concentrations,
    temperature, pressure) across the main process streams -- same table
    shown at the end of __main__, but callable anywhere (including inside
    the homotopy failure handler) to see the actual near-feasible state."""
    if m.fs.has_electroNP is False:
        stream_table = create_stream_table_dataframe(
            {
                "Feed": m.fs.FeedWater.outlet,
                "R1": m.fs.R1.outlet,
                "R2": m.fs.R2.outlet,
                "R3": m.fs.R3.outlet,
                "R4": m.fs.R4.outlet,
                "R5": m.fs.R5.outlet,
                "R6": m.fs.R6.outlet,
                "R7": m.fs.R7.outlet,
                "ASM-ADM inlet": m.fs.translator_asm2d_adm1.inlet,
                "ADM-ASM outlet": m.fs.translator_adm1_asm2d.outlet,
                "Treated water": m.fs.Treated.inlet,
            },
            time_point=0,
        )
    else:
        stream_table = create_stream_table_dataframe(
            {
                "Feed": m.fs.FeedWater.outlet,
                # "R1": m.fs.R1.outlet,
                # "R2": m.fs.R2.outlet,
                # "R3": m.fs.R3.outlet,
                # "R4": m.fs.R4.outlet,
                # "R5": m.fs.R5.outlet,
                # "R6": m.fs.R6.outlet,
                # "R7": m.fs.R7.outlet,
                "ASM-ADM translator inlet": m.fs.translator_asm2d_adm1.inlet,
                "ADM-ASM translator outlet": m.fs.translator_adm1_asm2d.outlet,
                "electroNP inlet": m.fs.electroNP.inlet,
                "electroNP treated": m.fs.electroNP.treated,
                # "electroNP byproduct": m.fs.electroNP.byproduct,
                "Treated water": m.fs.Treated.inlet,
                # "Sludge": m.fs.Sludge.inlet,
            },
            time_point=0,
        )
    print(stream_table_dataframe_to_string(stream_table))


def display_TP_table(m):
    streams = {
        "Feed": m.fs.FeedWater.properties[0],
        "CL effluent": m.fs.CL.effluent_state[0],
        "CL underflow": m.fs.CL.underflow_state[0],
        "MX1 outlet": m.fs.MX1.mixed_state[0],
        "R1 outlet": m.fs.R1.control_volume.properties_out[0],
        "R2 outlet": m.fs.R2.control_volume.properties_out[0],
        "R3 outlet": m.fs.R3.control_volume.properties_out[0],
        "R4 outlet": m.fs.R4.control_volume.properties_out[0],
        "R5 outlet": m.fs.R5.control_volume.properties_out[0],
        "R6 outlet": m.fs.R6.control_volume.properties_out[0],
        "R7 outlet": m.fs.R7.control_volume.properties_out[0],
        "CL2 effluent": m.fs.CL2.effluent_state[0],
        "CL2 underflow": m.fs.CL2.underflow_state[0],
        "Thickener underflow": m.fs.thickener.underflow_state[0],
        "Thickener overflow": m.fs.thickener.overflow_state[0],
        "Dewater underflow": m.fs.dewater.underflow_state[0],
        "Dewater overflow": m.fs.dewater.overflow_state[0],
        "Sludge": m.fs.Sludge.properties[0],
        "Treated": m.fs.Treated.properties[0],
    }
    if m.fs.has_electroNP:
        streams["ElectroNP inlet"] = m.fs.electroNP.properties_in[0]
        streams["ElectroNP treated"] = m.fs.electroNP.properties_treated[0]

    # ADM1 streams (MX4 → trans_asm2d_adm1 → AD → trans_adm1_asm2d → dewater)
    adm1_streams = {
        "Trans ASM2d-ADM1 inlet": (
            m.fs.translator_asm2d_adm1.properties_in[0],
            "asm2d",
        ),
        "Trans ASM2d-ADM1 outlet": (
            m.fs.translator_asm2d_adm1.properties_out[0],
            "adm1",
        ),
        "AD outlet": (m.fs.AD.liquid_phase.properties_out[0], "adm1"),
        "Trans ADM1-ASM2d inlet": (m.fs.translator_adm1_asm2d.properties_in[0], "adm1"),
        "Trans ADM1-ASM2d outlet": (
            m.fs.translator_adm1_asm2d.properties_out[0],
            "asm2d",
        ),
    }

    print("\n--- Total Phosphorus by Stream ---")
    print(f"{'Stream':<26}  {'TP (mg/L)':>12}  {'TP flow (g P/s)':>16}")
    print("-" * 60)
    for name, props in streams.items():
        tp_conc = pyo.value(_TP_conc(m, props))
        tp_flow = pyo.value(_TP_conc(m, props) * props.flow_vol)
        print(f"{name:<26}  {tp_conc * 1e3:>12.2f}  {tp_flow * 1e3:>16.4f}")

    print("-" * 60)
    print(f"  {'--- Sludge digestion loop (MX4 → AD → dewater) ---'}")
    print("-" * 60)
    for name, (props, kind) in adm1_streams.items():
        if kind == "asm2d":
            tp_conc = pyo.value(_TP_conc(m, props))
            tp_flow = pyo.value(_TP_conc(m, props) * props.flow_vol)
        else:
            tp_conc = pyo.value(_TP_conc_adm1(m, props))
            tp_flow = pyo.value(_TP_conc_adm1(m, props) * props.flow_vol)
        print(f"{name:<26}  {tp_conc * 1e3:>12.2f}  {tp_flow * 1e3:>16.4f}")


def display_performance_metrics(m):
    print("\n--- Influent Metrics ---")
    Q_in = pyo.units.convert(
        m.fs.FeedWater.flow_vol[0], to_units=pyo.units.gallon / pyo.units.day
    )
    print("Influent flow: %.2f MGD" % pyo.value(Q_in / 1e6))
    print(
        "Feed TSS concentration: %.1f mg/L"
        % pyo.value(m.fs.FeedWater.properties[0].TSS * 1e3)
    )
    print(
        "Feed COD concentration: %.1f mg/L"
        % pyo.value(m.fs.FeedWater.properties[0].COD * 1e3)
    )
    print(
        "BOD5 concentration: %.1f mg/L"
        % pyo.value(m.fs.FeedWater.properties[0].BOD5["raw"] * 1e3)
    )
    print(
        "TKN concentration: %.1f mg/L"
        % pyo.value(m.fs.FeedWater.properties[0].TKN * 1e3)
    )
    print(
        "SNOX concentration: %.1f mg/L"
        % pyo.value(m.fs.FeedWater.properties[0].SNOX * 1e3)
    )
    print(
        "Organic phosphorus concentration: %.1f mg/L"
        % pyo.value(m.fs.FeedWater.properties[0].SP_organic * 1e3)
    )
    print(
        "Inorganic phosphorus concentration: %.1f mg/L"
        % pyo.value(m.fs.FeedWater.properties[0].SP_inorganic * 1e3)
    )
    print(
        "Total phosphorus (TP) concentration: %.2f mg/L"
        % (pyo.value(_TP_conc(m, m.fs.FeedWater.properties[0])) * 1e3)
    )
    print("\n--- Effluent Metrics ---")
    Q_out = pyo.units.convert(
        m.fs.Treated.flow_vol[0], to_units=pyo.units.gallon / pyo.units.day
    )
    print("Effluent flow: %.2f MGD" % pyo.value(Q_out / 1e6))
    print(
        "TSS concentration: %.1f mg/L" % pyo.value(m.fs.Treated.properties[0].TSS * 1e3)
    )
    print(
        "COD concentration: %.1f mg/L" % pyo.value(m.fs.Treated.properties[0].COD * 1e3)
    )
    print(
        "BOD5 concentration: %.1f mg/L"
        % pyo.value(m.fs.Treated.properties[0].BOD5["effluent"] * 1e3)
    )
    print(
        "TKN concentration: %.1f mg/L" % pyo.value(m.fs.Treated.properties[0].TKN * 1e3)
    )
    print(
        "SNOX concentration: %.1f mg/L"
        % pyo.value(m.fs.Treated.properties[0].SNOX * 1e3)
    )
    print(
        "Organic phosphorus concentration: %.1f mg/L"
        % pyo.value(m.fs.Treated.properties[0].SP_organic * 1e3)
    )
    print(
        "Inorganic phosphorus concentration: %.1f mg/L"
        % pyo.value(m.fs.Treated.properties[0].SP_inorganic * 1e3)
    )
    print(
        "Total phosphorus (TP) concentration: %.2f mg/L"
        % (pyo.value(_TP_conc(m, m.fs.Treated.properties[0])) * 1e3)
    )

    # print("Inlet total nitrogen concentration: %.1f mg/L" % pyo.value(m.fs.TN_in * 1e3))
    # print(
    #     "Treated total nitrogen concentration: %.1f mg/L"
    #     % pyo.value(m.fs.TN_treated * 1e3)
    # )

    print("\n--- Performance Metrics ---")
    print("Water recovery: %.3f" % pyo.value(m.fs.water_recovery))
    if m.fs.has_electroNP is True:
        print("Phosphorus recovery: %.3f" % pyo.value(m.fs.phosphorus_recovery))
        print(
            "Recovered phosphorus mass: %.3f kg/hr"
            % pyo.value(
                pyo.units.convert(
                    (
                        m.fs.electroNP.inlet.flow_vol[0]
                        * m.fs.electroNP.inlet.conc_mass_comp[0, "S_PO4"]
                        * m.fs.electroNP.P_removal
                    ),
                    to_units=pyo.units.kg / pyo.units.hr,
                )
            )
        )

    print("\n--- Energy Metrics ---")
    print(
        "SEC with respect to influent flowrate: %.3f kWh/m3"
        % pyo.value(m.fs.costing.specific_energy_consumption)
    )
    if m.fs.has_electroNP is True:
        print(
            "SEC with respect to phosphorus removal: %.3f kWh/kg"
            % pyo.value(m.fs.costing.specific_energy_consumption_P_removal)
        )
        print(
            "ElectroNP energy consumption: %.3g kWh/m3"
            % pyo.value(m.fs.costing.electroNP_energy_consumption)
        )
        # print(
        #     "ElectroNP energy consumption side stream: %.3g kWh/m3"
        #     % pyo.value(m.fs.costing.electroNP_energy_consumption_side_stream)
        # )
        # print(
        #     "Electrode energy consumption: %.3g kWh/m3"
        #     % pyo.value(m.fs.costing.electrode_energy_consumption)
        # )
        # print(
        #     "Dryer energy consumption: %.3g kWh/m3"
        #     % pyo.value(m.fs.costing.dryer_energy_consumption)
        # )
        # print(
        #     "Centrifuge energy consumption: %.3g kWh/m3"
        #     % pyo.value(m.fs.costing.centrifuge_energy_consumption)
        # )
        # print(
        #     "ElectroNP pumps energy consumption: %.3g kWh/m3"
        #     % pyo.value(m.fs.costing.electroNP_pump_energy_consumption)
        # )
    print("Aeration energy: %.3f kWh/m3" % pyo.value(m.fs.costing.aeration_energy))

    # print(
    #     "electricity consumption R5",
    #     pyo.value(m.fs.R5.electricity_consumption[0]),
    #     pyo.units.get_units(m.fs.R5.electricity_consumption[0]),
    # )
    # print(
    #     "electricity consumption R6",
    #     pyo.value(m.fs.R6.electricity_consumption[0]),
    #     pyo.units.get_units(m.fs.R6.electricity_consumption[0]),
    # )
    # print(
    #     "electricity consumption R7",
    #     pyo.value(m.fs.R7.electricity_consumption[0]),
    #     pyo.units.get_units(m.fs.R7.electricity_consumption[0]),
    # )
    # print(
    #     "electricity consumption primary clarifier",
    #     pyo.value(m.fs.CL.electricity_consumption[0]),
    #     pyo.units.get_units(m.fs.CL.electricity_consumption[0]),
    # )
    # print(
    #     "electricity consumption secondary clarifier",
    #     pyo.value(m.fs.CL2.electricity_consumption[0]),
    #     pyo.units.get_units(m.fs.CL2.electricity_consumption[0]),
    # )
    # print(
    #     "electricity consumption AD",
    #     pyo.value(m.fs.AD.electricity_consumption[0]),
    #     pyo.units.get_units(m.fs.AD.electricity_consumption[0]),
    # )
    # print(
    #     "electricity consumption dewatering Unit",
    #     pyo.value(m.fs.dewater.electricity_consumption[0]),
    #     pyo.units.get_units(m.fs.dewater.electricity_consumption[0]),
    # )
    # print(
    #     "electricity consumption thickening Unit",
    #     pyo.value(m.fs.thickener.electricity_consumption[0]),
    #     pyo.units.get_units(m.fs.thickener.electricity_consumption[0]),
    # )
    # print(
    #     "flow into R3",
    #     pyo.value(m.fs.R3.control_volume.properties_in[0].flow_vol),
    #     pyo.units.get_units(m.fs.R3.control_volume.properties_in[0].flow_vol),
    # )
    # print(
    #     "flow into RADM",
    #     pyo.value(m.fs.AD.liquid_phase.properties_in[0].flow_vol),
    #     pyo.units.get_units(m.fs.AD.liquid_phase.properties_in[0].flow_vol),
    # )


def display_design(m):
    print("\n--- decision variables ---")
    if m.fs.has_electroNP is True:
        print(
            "Cathodic potential: %.4g V" % pyo.value(m.fs.electroNP.cathodic_potential)
        )
        print(
            "Area volume ratio: %.4g cm-1" % pyo.value(m.fs.electroNP.area_volume_ratio)
        )


if __name__ == "__main__":
    # This method builds and runs a steady state activated sludge flowsheet.
    m, results = main(
        has_electroNP=False,
        has_optimization=False,
        objective=objective_fun.LCOW,
        has_effluent_constraints=True,
    )

    if m.fs.has_electroNP is False:
        stream_table = create_stream_table_dataframe(
            {
                "Feed": m.fs.FeedWater.outlet,
                # "R1 inlet": m.fs.R1.inlet,
                # "R3 inlet": m.fs.R3.inlet,
                # "ASM-ADM translator inlet": m.fs.translator_asm2d_adm1.inlet,
                # "R1": m.fs.R1.outlet,
                # "R2": m.fs.R2.outlet,
                # "R3": m.fs.R3.outlet,
                # "R4": m.fs.R4.outlet,
                # "R5": m.fs.R5.outlet,
                # "R6": m.fs.R6.outlet,
                # "R7": m.fs.R7.outlet,
                # "thickener outlet": m.fs.thickener.underflow,
                "ASM-ADM inlet": m.fs.translator_asm2d_adm1.inlet,
                "ADM-ASM outlet": m.fs.translator_adm1_asm2d.outlet,
                "dewater outlet": m.fs.dewater.overflow,
                "Treated water": m.fs.Treated.inlet,
                # "Sludge": m.fs.Sludge.inlet,
            },
            time_point=0,
        )
    else:
        stream_table = create_stream_table_dataframe(
            {
                "Feed": m.fs.FeedWater.outlet,
                # "CL inlet": m.fs.CL.inlet,
                # "R1 inlet": m.fs.R1.inlet,
                # "R3 inlet": m.fs.R3.inlet,
                # "ASM-ADM translator inlet": m.fs.translator_asm2d_adm1.inlet,
                # "R1": m.fs.R1.outlet,
                # "R2": m.fs.R2.outlet,
                # "R3": m.fs.R3.outlet,
                # "R4": m.fs.R4.outlet,
                # "R5": m.fs.R5.outlet,
                # "R6": m.fs.R6.outlet,
                # "R7": m.fs.R7.outlet,
                # # "thickener inlet": m.fs.thickener.inlet,
                # "thickener outlet": m.fs.thickener.underflow,
                "ASM-ADM translator inlet": m.fs.translator_asm2d_adm1.inlet,
                "ADM-ASM translator outlet": m.fs.translator_adm1_asm2d.outlet,
                # "dewater outlet": m.fs.dewater.overflow,
                "electroNP inlet": m.fs.electroNP.inlet,
                "electroNP treated": m.fs.electroNP.treated,
                "electroNP byproduct": m.fs.electroNP.byproduct,
                "Treated water": m.fs.Treated.inlet,
                # "Sludge": m.fs.Sludge.inlet,
                # "MX1": m.fs.MX1.outlet,
                # "MX2": m.fs.MX2.outlet,
                # "MX3": m.fs.MX3.outlet,
                # "MX4": m.fs.MX4.outlet,
            },
            time_point=0,
        )
    print(stream_table_dataframe_to_string(stream_table))

    # m_min, obj_set, m_set, cp_opt, r_AV_opt = multi_run(
    #     has_electroNP=True,
    #     objective=objective_fun.LCOW,
    #     has_effluent_constraints=True,
    #     num=10,
    # )
    # stream_table = create_stream_table_dataframe(
    #     {
    #         "Feed": m_min.fs.FeedWater.outlet,
    #         "CL inlet": m_min.fs.CL.inlet,
    #         # "R1 inlet": m_min.fs.R1.inlet,
    #         # "R3 inlet": m_min.fs.R3.inlet,
    #         # "ASM-ADM translator inlet": m.fs.translator_asm2d_adm1.inlet,
    #         "R1": m_min.fs.R1.outlet,
    #         "R2": m_min.fs.R2.outlet,
    #         "R3": m_min.fs.R3.outlet,
    #         "R4": m_min.fs.R4.outlet,
    #         "R5": m_min.fs.R5.outlet,
    #         "R6": m_min.fs.R6.outlet,
    #         "R7": m_min.fs.R7.outlet,
    #         # "thickener inlet": m_min.fs.thickener.inlet,
    #         "thickener outlet": m_min.fs.thickener.underflow,
    #         "ASM-ADM translator inlet": m_min.fs.translator_asm2d_adm1.inlet,
    #         "ADM-ASM translator outlet": m_min.fs.translator_adm1_asm2d.outlet,
    #         "dewater outlet": m_min.fs.dewater.overflow,
    #         "electroNP inlet": m_min.fs.electroNP.inlet,
    #         "electroNP treated": m_min.fs.electroNP.treated,
    #         # "electroNP byproduct": m_min.fs.electroNP.byproduct,
    #         # "electroNP byproduct": m_min.fs.electroNP.byproduct,
    #         "Treated water": m_min.fs.Treated.inlet,
    #         "Sludge": m_min.fs.Sludge.inlet,
    #         # "MX1": m_min.fs.MX1.outlet,
    #         # "MX2": m_min.fs.MX2.outlet,
    #         # "MX3": m_min.fs.MX3.outlet,
    #         # "MX4": m_min.fs.MX4.outlet,
    #     },
    #     time_point=0,
    # )
    #
    # print(stream_table_dataframe_to_string(stream_table))
