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
Tests for zero-order well field model
"""
import pytest


from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Param,
    Block,
    value,
    Var,
    assert_optimal_termination,
)
from pyomo.util.check_units import assert_units_consistent

from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.testing import initialization_tester
from idaes.core import UnitModelCostingBlock

from watertap.unit_models.zero_order import WellFieldZO
from watertap.core.wt_database import Database
from watertap.core.zero_order_properties import WaterParameterBlock
from watertap.core.zero_order_costing import ZeroOrderCosting

solver = get_solver()


class TestWellFieldZO:
    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()
        m.db = Database()

        m.fs = FlowsheetBlock(default={"dynamic": False})
        m.fs.params = WaterParameterBlock(
            default={"solute_list": ["toc", "nitrate", "sulfate", "bar", "crux"]}
        )

        m.fs.unit = WellFieldZO(
            default={"property_package": m.fs.params, "database": m.db}
        )

        m.fs.unit.inlet.flow_mass_comp[0, "H2O"].fix(120)
        m.fs.unit.inlet.flow_mass_comp[0, "toc"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "nitrate"].fix(2)
        m.fs.unit.inlet.flow_mass_comp[0, "sulfate"].fix(0.3)
        m.fs.unit.inlet.flow_mass_comp[0, "bar"].fix(40)
        m.fs.unit.inlet.flow_mass_comp[0, "crux"].fix(0.0005)

        return m

    @pytest.mark.unit
    def test_build(self, model):
        assert model.fs.unit.config.database is model.db
        assert model.fs.unit._tech_type == "well_field"
        assert isinstance(model.fs.unit.electricity, Var)
        assert isinstance(model.fs.unit.electricity_consumption, Constraint)
        assert isinstance(model.fs.unit.lift_height, Param)
        assert isinstance(model.fs.unit.eta_pump, Param)
        assert isinstance(model.fs.unit.eta_motor, Param)
        assert isinstance(model.fs.unit.pipe_distance, Var)
        assert isinstance(model.fs.unit.pipe_diameter, Var)

    @pytest.mark.component
    def test_load_parameters(self, model):
        data = model.db.get_unit_operation_parameters("well_field")

        model.fs.unit.load_parameters_from_database()

        assert model.fs.unit.pipe_distance[0].fixed
        assert model.fs.unit.pipe_distance[0].value == data["pipe_distance"]["value"]
        assert model.fs.unit.pipe_diameter[0].fixed
        assert model.fs.unit.pipe_diameter[0].value == data["pipe_diameter"]["value"]

    @pytest.mark.component
    def test_degrees_of_freedom(self, model):
        assert degrees_of_freedom(model.fs.unit) == 0

    @pytest.mark.component
    def test_unit_consistency(self, model):
        assert_units_consistent(model.fs.unit)

    @pytest.mark.component
    def test_initialize(self, model):
        initialization_tester(model)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, model):
        results = solver.solve(model)

        # Check for optimal solution
        assert_optimal_termination(results)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, model):
        for t, j in model.fs.unit.inlet.flow_mass_comp:
            assert pytest.approx(
                value(model.fs.unit.inlet.flow_mass_comp[t, j]), rel=1e-5
            ) == value(model.fs.unit.outlet.flow_mass_comp[t, j])

        assert pytest.approx(60.174085, rel=1e-5) == value(model.fs.unit.electricity[0])

        # assert (pytest.approx(1.665893, rel=1e-5) ==
        # value(model.fs.unit.costing.capital_cost))

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_conservation(self, model):
        for j in model.fs.params.component_list:
            assert 1e-6 >= abs(
                value(
                    model.fs.unit.inlet.flow_mass_comp[0, j]
                    - model.fs.unit.outlet.flow_mass_comp[0, j]
                )
            )

    @pytest.mark.component
    def test_report(self, model):

        model.fs.unit.report()


class TestWellFieldZOsubtype:
    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()
        m.db = Database()

        m.fs = FlowsheetBlock(default={"dynamic": False})
        m.fs.params = WaterParameterBlock(
            default={"solute_list": ["toc", "nitrate", "sulfate", "bar", "crux"]}
        )

        m.fs.unit = WellFieldZO(
            default={
                "property_package": m.fs.params,
                "database": m.db,
                "process_subtype": "emwd",
            }
        )

        m.fs.unit.inlet.flow_mass_comp[0, "H2O"].fix(120)
        m.fs.unit.inlet.flow_mass_comp[0, "toc"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "nitrate"].fix(2)
        m.fs.unit.inlet.flow_mass_comp[0, "sulfate"].fix(0.3)
        m.fs.unit.inlet.flow_mass_comp[0, "bar"].fix(40)
        m.fs.unit.inlet.flow_mass_comp[0, "crux"].fix(0.0005)

        return m

    @pytest.mark.unit
    def test_build(self, model):
        assert model.fs.unit.config.database is model.db
        assert model.fs.unit._tech_type == "well_field"
        assert isinstance(model.fs.unit.electricity, Var)
        assert isinstance(model.fs.unit.electricity_consumption, Constraint)
        assert isinstance(model.fs.unit.lift_height, Param)
        assert isinstance(model.fs.unit.eta_pump, Param)
        assert isinstance(model.fs.unit.eta_motor, Param)
        assert isinstance(model.fs.unit.pipe_distance, Var)
        assert isinstance(model.fs.unit.pipe_diameter, Var)

    @pytest.mark.component
    def test_load_parameters(self, model):
        data = model.db.get_unit_operation_parameters("well_field", subtype="emwd")

        model.fs.unit.load_parameters_from_database()

        assert model.fs.unit.pipe_distance[0].fixed
        assert model.fs.unit.pipe_distance[0].value == data["pipe_distance"]["value"]
        assert model.fs.unit.pipe_diameter[0].fixed
        assert model.fs.unit.pipe_diameter[0].value == data["pipe_diameter"]["value"]

    @pytest.mark.component
    def test_degrees_of_freedom(self, model):
        assert degrees_of_freedom(model.fs.unit) == 0

    @pytest.mark.component
    def test_unit_consistency(self, model):
        assert_units_consistent(model.fs.unit)

    @pytest.mark.component
    def test_initialize(self, model):
        initialization_tester(model)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, model):
        results = solver.solve(model)

        # Check for optimal solution
        assert_optimal_termination(results)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, model):
        for t, j in model.fs.unit.inlet.flow_mass_comp:
            assert pytest.approx(
                value(model.fs.unit.inlet.flow_mass_comp[t, j]), rel=1e-5
            ) == value(model.fs.unit.outlet.flow_mass_comp[t, j])

        assert pytest.approx(60.174085, rel=1e-5) == value(model.fs.unit.electricity[0])

        # assert (pytest.approx(1.665893, rel=1e-5) ==
        # value(model.fs.unit.costing.capital_cost))

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_conservation(self, model):
        for j in model.fs.params.component_list:
            assert 1e-6 >= abs(
                value(
                    model.fs.unit.inlet.flow_mass_comp[0, j]
                    - model.fs.unit.outlet.flow_mass_comp[0, j]
                )
            )

    @pytest.mark.component
    def test_report(self, model):

        model.fs.unit.report()


db = Database()
params = db._get_technology("well_field")


@pytest.mark.parametrize("subtype", [k for k in params.keys()])
def test_costing(subtype):
    m = ConcreteModel()
    m.db = Database()

    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.params = WaterParameterBlock(
        default={"solute_list": ["toc", "nitrate", "sulfate", "bar", "crux"]}
    )
    m.fs.costing = ZeroOrderCosting()
    m.fs.unit = WellFieldZO(
        default={
            "property_package": m.fs.params,
            "database": m.db,
            "process_subtype": subtype,
        }
    )

    m.fs.unit.inlet.flow_mass_comp[0, "H2O"].fix(120)
    m.fs.unit.inlet.flow_mass_comp[0, "toc"].fix(1)
    m.fs.unit.inlet.flow_mass_comp[0, "nitrate"].fix(2)
    m.fs.unit.inlet.flow_mass_comp[0, "sulfate"].fix(0.3)
    m.fs.unit.inlet.flow_mass_comp[0, "bar"].fix(40)
    m.fs.unit.inlet.flow_mass_comp[0, "crux"].fix(0.0005)

    m.fs.unit.load_parameters_from_database()

    assert degrees_of_freedom(m.fs.unit) == 0
    m.fs.unit.costing = UnitModelCostingBlock(
        default={"flowsheet_costing_block": m.fs.costing}
    )
    assert_units_consistent(m.fs)
    assert degrees_of_freedom(m.fs.unit) == 0
    initialization_tester(m)
    _ = solver.solve(m)

    assert isinstance(m.fs.unit.costing.capital_cost_constraint, Constraint)

    if subtype == "default":
        assert pytest.approx(1.665893, rel=1e-5) == value(
            m.fs.unit.costing.capital_cost
        )
    if subtype == "emwd":
        assert pytest.approx(47.921893, rel=1e-5) == value(
            m.fs.unit.costing.capital_cost
        )

    assert m.fs.unit.electricity[0] in m.fs.costing._registered_flows["electricity"]
