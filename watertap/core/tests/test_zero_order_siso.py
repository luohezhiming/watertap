###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################
"""
Tests for zero-order SISO unit model
"""
import pytest

from idaes.core import declare_process_block_class, FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.testing import initialization_tester
from idaes.core.solvers import get_solver
import idaes.core.util.scaling as iscale
from pyomo.environ import (
    check_optimal_termination,
    ConcreteModel,
    Constraint,
    value,
    Var,
)
from pyomo.network import Port
from pyomo.util.check_units import assert_units_consistent


from watertap.core import WaterParameterBlock, WaterStateBlock, ZeroOrderBaseData
from watertap.core.zero_order_siso import (
    build_siso,
    initialize_siso,
    calculate_scaling_factors_siso,
    _get_Q_siso,
)

solver = get_solver()


@declare_process_block_class("DerivedSISO")
class DerivedSISOData(ZeroOrderBaseData):
    def build(self):
        super().build()

        build_siso(self)


class TestSISO:
    @pytest.fixture(scope="module")
    def model(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(default={"dynamic": False})

        m.fs.water_props = WaterParameterBlock(default={"solute_list": ["A", "B", "C"]})

        m.fs.unit = DerivedSISO(default={"property_package": m.fs.water_props})

        m.fs.unit.inlet.flow_mass_comp[0, "H2O"].fix(1000)
        m.fs.unit.inlet.flow_mass_comp[0, "A"].fix(10)
        m.fs.unit.inlet.flow_mass_comp[0, "B"].fix(20)
        m.fs.unit.inlet.flow_mass_comp[0, "C"].fix(30)

        m.fs.unit.recovery_frac_mass_H2O.fix(1)
        m.fs.unit.removal_frac_mass_solute[0, "A"].fix(0.1)
        m.fs.unit.removal_frac_mass_solute[0, "B"].fix(0.2)
        m.fs.unit.removal_frac_mass_solute[0, "C"].fix(0.3)

        return m

    @pytest.mark.unit
    def test_private_attributes(self, model):
        assert model.fs.unit._tech_type is None
        assert model.fs.unit._has_recovery_removal is True
        assert model.fs.unit._fixed_perf_vars == []
        assert model.fs.unit._initialize is initialize_siso
        assert model.fs.unit._scaling is calculate_scaling_factors_siso
        assert model.fs.unit._get_Q is _get_Q_siso
        assert model.fs.unit._stream_table_dict == {
            "Inlet": model.fs.unit.inlet,
            "Treated": model.fs.unit.treated,
        }
        assert model.fs.unit._perf_var_dict == {
            "Solute Removal": model.fs.unit.removal_frac_mass_solute
        }

    @pytest.mark.unit
    def test_build(self, model):
        assert isinstance(model.fs.unit.properties_in, WaterStateBlock)
        assert isinstance(model.fs.unit.properties_treated, WaterStateBlock)

        assert isinstance(model.fs.unit.inlet, Port)
        assert isinstance(model.fs.unit.treated, Port)

        assert isinstance(model.fs.unit.recovery_frac_mass_H2O, Var)
        assert isinstance(model.fs.unit.removal_frac_mass_solute, Var)
        assert len(model.fs.unit.removal_frac_mass_solute) == 3

        assert isinstance(model.fs.unit.water_recovery_equation, Constraint)
        assert len(model.fs.unit.water_recovery_equation) == 1
        assert isinstance(model.fs.unit.solute_treated_equation, Constraint)
        assert len(model.fs.unit.solute_treated_equation) == 3

    @pytest.mark.unit
    def test_degrees_of_freedom(self, model):
        from idaes.core.util.model_statistics import unfixed_variables_set

        [print(i) for i in unfixed_variables_set(model)]
        assert degrees_of_freedom(model) == 0

    @pytest.mark.component
    def test_unit_consistency(self, model):
        assert_units_consistent(model)

    @pytest.mark.component
    def test_scaling(self, model):
        iscale.calculate_scaling_factors(model)

        assert (
            iscale.get_constraint_transform_applied_scaling_factor(
                model.fs.unit.water_recovery_equation[0]
            )
            == 1e5
        )
        assert (
            iscale.get_constraint_transform_applied_scaling_factor(
                model.fs.unit.solute_treated_equation[0, "A"]
            )
            == 1e5
        )
        assert (
            iscale.get_constraint_transform_applied_scaling_factor(
                model.fs.unit.solute_treated_equation[0, "B"]
            )
            == 1e5
        )
        assert (
            iscale.get_constraint_transform_applied_scaling_factor(
                model.fs.unit.solute_treated_equation[0, "C"]
            )
            == 1e5
        )

    @pytest.mark.component
    def test_initialization(self, model):
        initialization_tester(model)

    @pytest.mark.component
    def test_solve(self, model):
        results = solver.solve(model)

        # Check for optimal solution
        assert check_optimal_termination(results)

    @pytest.mark.component
    def test_solution(self, model):
        assert pytest.approx(1000, rel=1e-5) == value(
            model.fs.unit.treated.flow_mass_comp[0, "H2O"]
        )

        assert pytest.approx(9, rel=1e-5) == value(
            model.fs.unit.treated.flow_mass_comp[0, "A"]
        )
        assert pytest.approx(16, rel=1e-5) == value(
            model.fs.unit.treated.flow_mass_comp[0, "B"]
        )
        assert pytest.approx(21, rel=1e-5) == value(
            model.fs.unit.treated.flow_mass_comp[0, "C"]
        )

    @pytest.mark.component
    def test_report(self, model):
        model.fs.unit.report()
