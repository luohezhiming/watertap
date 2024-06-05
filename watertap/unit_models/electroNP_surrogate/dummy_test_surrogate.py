# Import Pyomo libraries
from pyomo.environ import (
    Var,
    Param,
    Suffix,
    NonNegativeReals,
    NegativeReals,
    units as pyunits,
)
from idaes.models.unit_models.separator import SeparatorData, SplittingType

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
)

from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.misc import add_object_reference
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from watertap.costing.unit_models.electroNP import cost_electroNP

from watertap.core.solvers import get_solver
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate

from pyomo.environ import (
    ConcreteModel,
    assert_optimal_termination,
    value,
    units,
)
from idaes.core import FlowsheetBlock
import pyomo.environ as pyo
from idaes.core.util.model_diagnostics import DegeneracyHunter
from idaes.core.util.model_diagnostics import DiagnosticsToolbox


def build_flowsheet():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.cathodic_potential = Var(
        initialize=-1.05,
        domain=NegativeReals,
        units=pyunits.dimensionless,
        bounds=(-1.3, -0.8),
        doc="Cathodic potential",
    )

    m.fs.area_volume_ratio = Var(
        initialize=0.105,
        domain=NonNegativeReals,
        units=pyunits.dimensionless,
        bounds=(0, 1),
        doc="Area-volume ratio",
    )

    m.fs.T = Var(
        initialize=25,
        domain=NonNegativeReals,
        units=pyunits.dimensionless,
        bounds=(0, None),
        doc="Temperature",
    )

    m.fs.P_removal = Var(
        initialize=0.9,
        domain=NonNegativeReals,
        units=pyunits.dimensionless,
        bounds=(0, 1),
        doc="phosphorus removal fraction on a mass basis",
    )

    m.fs.N_removal = Var(
        initialize=0.3,
        domain=NonNegativeReals,
        units=pyunits.dimensionless,
        bounds=(0, 1),
        doc="Nitrogen removal fraction on a mass basis",
    )

    m.fs.settling_time = Var(
        initialize=30,
        domain=NonNegativeReals,
        units=pyunits.dimensionless,
        bounds=(0, None),
        doc="Settling time for electroN-P process",
    )

    # CP = pyunits.convert((m.fs.cathodic_potential / (1 * pyunits.V)), to_units=pyunits.dimensionless)
    # r_AV = pyunits.convert(m.fs.area_volume_ratio, to_units=pyunits.dimensionless)
    # T = pyunits.convert((m.fs.properties_in[0].temperature / (1 * pyunits.K)), to_units=pyunits.dimensionless)
    # t_ss = pyunits.convert((m.fs.settling_time / (1 * pyunits.min)), to_units=pyunits.dimensionless)
    inputs = [
        m.fs.cathodic_potential,
        m.fs.area_volume_ratio,
        m.fs.T,
        m.fs.settling_time,
    ]
    outputs = [m.fs.P_removal]
    m.fs.surrogate = SurrogateBlock(concrete=True)
    PR_surrogate = PysmoSurrogate.load_from_file(
        r"D:\Keylogic\WaterTap_Chenyu\watertap\watertap\unit_models\electroNP_surrogate\pysmo_RBF_PR_surrogate.json"
    )
    m.fs.surrogate.build_model(PR_surrogate, input_vars=inputs, output_vars=outputs)

    m.fs.EI_surrogate = Var(
        initialize=0.044,
        domain=NonNegativeReals,
        units=pyunits.dimensionless,
        bounds=(0, None),
        doc="Electricity intensity with respect to phosphorus removal",
    )

    inputs = [
        m.fs.CP_surrogate,
        m.fs.r_AV_surrogate,
        m.fs.T_surrogate,
        m.fs.t_ss_surrogate,
    ]
    outputs = [m.fs.EI_surrogate]
    m.fs.surrogate_EI = SurrogateBlock(concrete=True)
    EI_surrogate = PysmoSurrogate.load_from_file("pysmo_RBF_Ener_surrogate.json")
    m.fs.surrogate_EI.build_model(EI_surrogate, input_vars=inputs, output_vars=outputs)

    # set operating conditions
    m.fs.cathodic_potential.fix(-1.05)
    m.fs.area_volume_ratio.fix(0.105)
    m.fs.T.fix(25)
    m.fs.settling_time.fix(30)

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

    # Get default solver for testing
    solver = get_solver()
    results = solver.solve(m, tee=True)

    return m, results


if __name__ == "__main__":
    m, results = build_flowsheet()
