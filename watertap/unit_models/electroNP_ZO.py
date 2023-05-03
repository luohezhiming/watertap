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

# Import Pyomo libraries
from pyomo.environ import (
    Var,
    check_optimal_termination,
    Param,
    Suffix,
    NonNegativeReals,
    Reference,
    units as pyunits,
)
from pyomo.common.config import Bool, ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
    MaterialFlowBasis,
)
from idaes.core.solvers import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.exceptions import ConfigurationError, InitializationError
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from watertap.core import ControlVolume0DBlock, InitializationMixin

__author__ = "Chenyu Wang"

_log = idaeslog.getLogger(__name__)


@declare_process_block_class("ElectroNPZO")
class ElectroNPZOData(InitializationMixin, UnitModelBlockData):
    """
    Zero order electrochemical nutrient removal (ElectroNP) model based on specified water flux and ion rejection.
    """

    CONFIG = ConfigBlock()

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False. NF units do not support dynamic
    behavior.""",
        ),
    )
    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False. NF units do not have defined volume, thus
    this must be False.""",
        ),
    )
    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}""",
        ),
    )

    def _process_config(self):
        if len(self.config.property_package.solvent_set) > 1:
            raise ConfigurationError(
                "ElectroNP model only supports one solvent component,"
                "the provided property package has specified {} solvent components".format(
                    len(self.config.property_package.solvent_set)
                )
            )

        if len(self.config.property_package.solvent_set) == 0:
            raise ConfigurationError(
                "The ElectroNP model was expecting a solvent and did not receive it."
            )

        if (
            len(self.config.property_package.solute_set) == 0
            and len(self.config.property_package.ion_set) == 0
        ):
            raise ConfigurationError(
                "The ElectroNP model was expecting at least one solute or ion and did not receive any."
            )

    def build(self):
        # Call UnitModel.build to setup dynamics
        super().build()

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        units_meta = self.config.property_package.get_metadata().get_derived_units

        # Check configs for errors
        self._process_config()

        # Create state blocks for inlet and outlets
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["defined_state"] = True

        self.properties_in = self.config.property_package.build_state_block(
            self.flowsheet().time, doc="Material properties at inlet", **tmp_dict
        )

        tmp_dict_2 = dict(**tmp_dict)
        tmp_dict_2["defined_state"] = False

        self.properties_treated = self.config.property_package.build_state_block(
            self.flowsheet().time,
            doc="Material properties of treated water",
            **tmp_dict_2,
        )
        self.properties_byproduct = self.config.property_package.build_state_block(
            self.flowsheet().time,
            doc="Material properties of byproduct stream",
            **tmp_dict_2,
        )

        # Create Ports
        self.add_port("inlet", self.properties_in, doc="Inlet port")
        self.add_port(
            "treated", self.properties_treated, doc="Treated water outlet port"
        )
        self.add_port(
            "byproduct", self.properties_byproduct, doc="Byproduct outlet port"
        )

        # Add isothermal constraints
        @self.Constraint(
            self.flowsheet().config.time,
            doc="Isothermal assumption for treated flow",
        )
        def eq_treated_isothermal(b, t):
            return b.properties_in[t].temperature == b.properties_treated[t].temperature

        @self.Constraint(
            self.flowsheet().config.time,
            doc="Isothermal assumption for byproduct flow",
        )
        def eq_byproduct_isothermal(b, t):
            return (
                b.properties_in[t].temperature == b.properties_byproduct[t].temperature
            )

        # Add performance variables
        self.recovery_frac_mass_H2O = Var(
            self.flowsheet().time,
            initialize=0.8,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0.0, 1.0000001),
            doc="Mass recovery fraction of water in the treated stream",
        )
        self.removal_frac_mass_comp = Var(
            self.flowsheet().time,
            self.config.property_package.solute_set,
            domain=NonNegativeReals,
            initialize=0.01,
            units=pyunits.dimensionless,
            doc="Solute removal fraction on a mass basis",
        )

        # Add performance constraints
        # Water recovery
        @self.Constraint(self.flowsheet().time, doc="Water recovery equation")
        def water_recovery_equation(b, t):
            return b.recovery_frac_mass_H2O[t] * b.properties_in[
                t
            ].get_material_flow_terms("Liq", "H2O") == b.properties_treated[
                t
            ].get_material_flow_terms(
                "Liq", "H2O"
            )

        # Flow balance
        @self.Constraint(self.flowsheet().time, doc="Overall flow balance")
        def water_balance(b, t):
            return b.properties_in[t].get_material_flow_terms(
                "Liq", "H2O"
            ) == b.properties_treated[t].get_material_flow_terms(
                "Liq", "H2O"
            ) + b.properties_byproduct[
                t
            ].get_material_flow_terms(
                "Liq", "H2O"
            )

        # default water recovery
        @self.Constraint(self.flowsheet().time, doc="Default water recovery equation")
        def default_water_recovery_equation(b, t):
            return b.recovery_frac_mass_H2O[t] == 1

        # Solute removal
        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.solute_set,
            doc="Solute removal equations",
        )
        def solute_removal_equation(b, t, p, j):
            return b.removal_frac_mass_comp[t, j] * b.properties_in[
                t
            ].get_material_flow_terms(p, j) == b.properties_byproduct[
                t
            ].get_material_flow_terms(
                p, j
            )

        # Solute concentration of treated stream
        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.solute_set,
            doc="Constraint for solute concentration in treated " "stream.",
        )
        def solute_treated_equation(b, t, p, j):
            return (1 - b.removal_frac_mass_comp[t, j]) * b.properties_in[
                t
            ].get_material_flow_terms(p, j) == b.properties_treated[
                t
            ].get_material_flow_terms(
                p, j
            )

        # Default solute concentration
        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.solute_set,
            doc="Default solute removal equations",
        )
        def default_solute_removal_equation(b, t, j):
            if hasattr(self.config.property_package.solute_set, "S_PO4"):
                return b.removal_frac_mass_comp[t, "S_PO4"] == 0.98
            elif hasattr(self.config.property_package.solute_set, "S_NH4"):
                return b.removal_frac_mass_comp[t, "S_NH4"] == 0.3
            else:
                return b.removal_frac_mass_comp[t, j] == 0

        self._stream_table_dict = {
            "Inlet": self.inlet,
            "Treated": self.treated,
            "Byproduct": self.byproduct,
        }

        self.electricity = Var(
            self.flowsheet().time,
            units=pyunits.kW,
            bounds=(0, None),
            doc="Electricity consumption of unit",
        )

        self.energy_electric_flow_mass = Var(
            units=pyunits.kWh / pyunits.kg,
            doc="Electricity intensity with respect to phosphorus removal",
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="Constraint for electricity consumption based on phosphorus removal",
        )
        def electricity_consumption(b, t):
            return b.electricity[t] == (
                b.energy_electric_flow_mass
                * pyunits.convert(
                    b.properties_treated[t].get_material_flow_terms("Liq", "S_PO4"),
                    to_units=pyunits.kg / pyunits.hour,
                )
            )

        self.magnesium_chloride_dosage = Var(
            units=pyunits.dimensionless,
            bounds=(0, None),
            doc="Dosage of magnesium chloride per treated phosphorus",
        )

        self.MgCl2_flowrate = Var(
            self.flowsheet().time,
            units=pyunits.kg / pyunits.hr,
            bounds=(0, None),
            doc="Magnesium chloride flowrate",
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="Constraint for magnesium chloride demand based on phosphorus removal.",
        )
        def MgCl2_demand(b, t):
            return b.MgCl2_flowrate[t] == (
                b.magnesium_chloride_dosage
                * pyunits.convert(
                    b.properties_treated[t].get_material_flow_terms("Liq", "S_PO4"),
                    to_units=pyunits.kg / pyunits.hour,
                )
            )

    def initialize(
        self, state_args=None, outlvl=idaeslog.NOTSET, solver=None, optarg=None
    ):
        """
        Initialization routine for single inlet-double outlet unit models.

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                           package(s) to provide an initial state for
                           initialization (see documentation of the specific
                           property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None, use
                     default solver options)
            solver : str indicating which solver to use during
                     initialization (default = None, use default IDAES solver)

        Returns:
            None
        """
        if optarg is None:
            optarg = {}

        # Set solver options
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        solver_obj = get_solver(solver, optarg)

        # Get initial guesses for inlet if none provided
        if state_args is None:
            state_args = {}
            state_dict = self.properties_in[
                self.flowsheet().time.first()
            ].define_port_members()

            for k in state_dict.keys():
                if state_dict[k].is_indexed():
                    state_args[k] = {}
                    for m in state_dict[k].keys():
                        state_args[k][m] = state_dict[k][m].value
                else:
                    state_args[k] = state_dict[k].value

        # ---------------------------------------------------------------------
        # Initialize control volume block
        flags = self.properties_in.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=True,
        )
        self.properties_treated.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=False,
        )
        self.properties_byproduct.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=False,
        )

        init_log.info_high("Initialization Step 1 Complete.")

        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            results = solver_obj.solve(self, tee=slc.tee)

        init_log.info_high(
            "Initialization Step 2 {}.".format(idaeslog.condition(results))
        )

        # ---------------------------------------------------------------------
        # Release Inlet state
        self.properties_in.release_state(flags, outlvl)

        init_log.info("Initialization Complete: {}".format(idaeslog.condition(results)))

        if not check_optimal_termination(results):
            raise InitializationError(
                f"{self.name} failed to initialize successfully. Please check "
                f"the output logs for more information."
            )

    def _get_performance_contents(self, time_point=0):
        var_dict = {}
        var_dict["Water Recovery"] = self.recovery_frac_mass_H2O[time_point]
        for j in self.config.property_package.solute_set:
            var_dict[f"Solute Removal {j}"] = self.removal_frac_mass_comp[time_point, j]
        var_dict["Electricity Demand"] = self.electricity[time_point]
        var_dict["Electricity Intensity"] = self.energy_electric_flow_mass
        var_dict[
            "Dosage of magnesium chloride per treated phosphorus"
        ] = self.magnesium_chloride_dosage
        var_dict["Magnesium Chloride Demand"] = self.MgCl2_flowrate[time_point]
        return {"vars": var_dict}

    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {
                "Inlet": self.inlet,
                "Treated": self.treated,
                "Byproduct": self.byproduct,
            },
            time_point=time_point,
        )

    def calculate_scaling_factors(self):
        # Get default scale factors and do calculations from base classes
        for t, v in self.water_recovery_equation.items():
            iscale.constraint_scaling_transform(
                v,
                iscale.get_scaling_factor(
                    self.properties_in[t].get_material_flow_terms("Liq", "H2O"),
                    default=1,
                    warning=True,
                    hint=" for water recovery",
                ),
            )

        for t, v in self.water_balance.items():
            iscale.constraint_scaling_transform(
                v,
                iscale.get_scaling_factor(
                    self.properties_in[t].get_material_flow_terms("Liq", "H2O"),
                    default=1,
                    warning=False,
                ),
            )  # would just be a duplicate of above

        for (t, p, j), v in self.solute_removal_equation.items():
            iscale.constraint_scaling_transform(
                v,
                iscale.get_scaling_factor(
                    self.properties_in[t].get_material_flow_terms(p, j),
                    default=1,
                    warning=True,
                    hint=" for solute removal",
                ),
            )

        for (t, p, j), v in self.solute_treated_equation.items():
            iscale.constraint_scaling_transform(
                v,
                iscale.get_scaling_factor(
                    self.properties_in[t].get_material_flow_terms(p, j),
                    default=1,
                    warning=False,
                ),
            )  # would just be a duplicate of above
