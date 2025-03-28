"""
Initial property package for pure water system (vapor or liquid)
"""

# Import Python libraries
import idaes.logger as idaeslog

# Import Pyomo libraries
from pyomo.environ import (
    Constraint,
    Expression,
    Reals,
    NonNegativeReals,
    Var,
    Param,
    Suffix,
    value,
    log,
    log10,
    exp,
    check_optimal_termination,
    Set,
)
from pyomo.environ import units as pyunits

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialFlowBasis,
    PhysicalParameterBlock,
    StateBlockData,
    StateBlock,
    MaterialBalanceType,
    EnergyBalanceType,
)
from idaes.core.base.components import Component, Solute, Solvent
from idaes.core.base.phases import LiquidPhase, VaporPhase
from idaes.core.util.constants import Constants
from idaes.core.util.initialization import (
    fix_state_vars,
    revert_state_vars,
    solve_indexed_blocks,
)
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.exceptions import (
    ConfigurationError,
    InitializationError,
    PropertyPackageError,
)
import idaes.core.util.scaling as iscale

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("WaterParameterBlock")
class WaterParameterData(PhysicalParameterBlock):
    """Parameter block for a water property package."""

    CONFIG = PhysicalParameterBlock.CONFIG()

    def build(self):
        """
        Callable method for Block construction.
        """
        super(WaterParameterData, self).build()

        self._state_block_class = WaterStateBlock

        # components
        self.H2O = Component()

        # phases
        self.Liq = LiquidPhase()
        self.Vap = VaporPhase()

        """ References
        This package was developed from the following references:

        - K.G.Nayar, M.H.Sharqawy, L.D.Banchik, and J.H.Lienhard V, "Thermophysical properties of seawater: A review and
        new correlations that include pressure dependence,"Desalination, Vol.390, pp.1 - 24, 2016.
        doi: 10.1016/j.desal.2016.02.024(preprint)

        - Mostafa H.Sharqawy, John H.Lienhard V, and Syed M.Zubair, "Thermophysical properties of seawater: A review of 
        existing correlations and data,"Desalination and Water Treatment, Vol.16, pp.354 - 380, April 2010.
        (2017 corrections provided at http://web.mit.edu/seawater)
        """

        # Parameters
        # molecular weight
        self.mw_comp = Param(
            self.component_list,
            mutable=False,
            initialize=18.01528e-3,
            units=pyunits.kg / pyunits.mol,
            doc="Molecular weight",
        )

        # Liq mass density parameters, eq. 8 in Sharqawy et al. (2010)
        dens_units = pyunits.kg / pyunits.m**3
        t_inv_units = pyunits.K**-1
        s_inv_units = pyunits.kg / pyunits.g

        self.dens_mass_param_A1 = Var(
            within=Reals,
            initialize=9.999e2,
            units=dens_units,
            doc="Mass density parameter A1",
        )
        self.dens_mass_param_A2 = Var(
            within=Reals,
            initialize=2.034e-2,
            units=dens_units * t_inv_units,
            doc="Mass density parameter A2",
        )
        self.dens_mass_param_A3 = Var(
            within=Reals,
            initialize=-6.162e-3,
            units=dens_units * t_inv_units**2,
            doc="Mass density parameter A3",
        )
        self.dens_mass_param_A4 = Var(
            within=Reals,
            initialize=2.261e-5,
            units=dens_units * t_inv_units**3,
            doc="Mass density parameter A4",
        )
        self.dens_mass_param_A5 = Var(
            within=Reals,
            initialize=-4.657e-8,
            units=dens_units * t_inv_units**4,
            doc="Mass density parameter A5",
        )

        # Vap mass density parameters (approximating using ideal gas)
        self.dens_mass_param_mw = Var(
            within=Reals,
            initialize=18.01528e-3,
            units=pyunits.kg / pyunits.mol,
            doc="Mass density parameter molecular weight",
        )
        self.dens_mass_param_R = Var(
            within=Reals,
            initialize=8.31462618,
            units=pyunits.J / pyunits.mol / pyunits.K,
            doc="Mass density parameter universal gas constant",
        )

        # vapor pressure parameters,  eq. 5 and 6 in Nayar et al.(2016)
        self.pressure_sat_param_psatw_A1 = Var(
            within=Reals,
            initialize=-5.8002206e3,
            units=pyunits.K,
            doc="Vapor pressure of pure water parameter A1",
        )
        self.pressure_sat_param_psatw_A2 = Var(
            within=Reals,
            initialize=1.3914993,
            units=pyunits.dimensionless,
            doc="Vapor pressure of pure water parameter A2",
        )
        self.pressure_sat_param_psatw_A3 = Var(
            within=Reals,
            initialize=-4.8640239e-2,
            units=t_inv_units,
            doc="Vapor pressure of pure water parameter A3",
        )
        self.pressure_sat_param_psatw_A4 = Var(
            within=Reals,
            initialize=4.1764768e-5,
            units=t_inv_units**2,
            doc="Vapor pressure of pure water parameter A4",
        )
        self.pressure_sat_param_psatw_A5 = Var(
            within=Reals,
            initialize=-1.4452093e-8,
            units=t_inv_units**3,
            doc="Vapor pressure of pure water parameter A5",
        )
        self.pressure_sat_param_psatw_A6 = Var(
            within=Reals,
            initialize=6.5459673,
            units=pyunits.dimensionless,
            doc="Vapor pressure of pure water parameter A6",
        )

        # specific enthalpy parameters, eq. 55 and 43 in Sharqawy et al. (2010)
        enth_mass_units = pyunits.J / pyunits.kg

        self.enth_mass_param_A1 = Var(
            within=Reals,
            initialize=141.355,
            units=enth_mass_units,
            doc="Specific enthalpy parameter A1",
        )
        self.enth_mass_param_A2 = Var(
            within=Reals,
            initialize=4202.07,
            units=enth_mass_units * t_inv_units,
            doc="Specific enthalpy parameter A2",
        )
        self.enth_mass_param_A3 = Var(
            within=Reals,
            initialize=-0.535,
            units=enth_mass_units * t_inv_units**2,
            doc="Specific enthalpy parameter A3",
        )
        self.enth_mass_param_A4 = Var(
            within=Reals,
            initialize=0.004,
            units=enth_mass_units * t_inv_units**3,
            doc="Specific enthalpy parameter A4",
        )
        # self.enth_mass_param_B1 = Var(
        #     within=Reals, initialize=-2.348e4, units=enth_mass_units,
        #     doc='Specific enthalpy parameter B1')
        # self.enth_mass_param_B2 = Var(
        #     within=Reals, initialize=3.152e5, units=enth_mass_units,
        #     doc='Specific enthalpy parameter B2')
        # self.enth_mass_param_B3 = Var(
        #     within=Reals, initialize=2.803e6, units=enth_mass_units,
        #     doc='Specific enthalpy parameter B3')
        # self.enth_mass_param_B4 = Var(
        #     within=Reals, initialize=-1.446e7, units=enth_mass_units,
        #     doc='Specific enthalpy parameter B4')
        # self.enth_mass_param_B5 = Var(
        #     within=Reals, initialize=7.826e3, units=enth_mass_units * t_inv_units,
        #     doc='Specific enthalpy parameter B5')
        # self.enth_mass_param_B6 = Var(
        #     within=Reals, initialize=-4.417e1, units=enth_mass_units * t_inv_units**2,
        #     doc='Specific enthalpy parameter B6')
        # self.enth_mass_param_B7 = Var(
        #     within=Reals, initialize=2.139e-1, units=enth_mass_units * t_inv_units**3,
        #     doc='Specific enthalpy parameter B7')
        # self.enth_mass_param_B8 = Var(
        #     within=Reals, initialize=-1.991e4, units=enth_mass_units * t_inv_units,
        #     doc='Specific enthalpy parameter B8')
        # self.enth_mass_param_B9 = Var(
        #     within=Reals, initialize=2.778e4, units=enth_mass_units * t_inv_units,
        #     doc='Specific enthalpy parameter B9')
        # self.enth_mass_param_B10 = Var(
        #     within=Reals, initialize=9.728e1, units=enth_mass_units * t_inv_units**2,
        #     doc='Specific enthalpy parameter B10')

        # specific heat parameters from eq (9) in Sharqawy et al. (2010)
        cp_units = pyunits.J / (pyunits.kg * pyunits.K)
        self.cp_phase_param_A1 = Var(
            within=Reals,
            initialize=5.328,
            units=cp_units,
            doc="Specific heat of seawater parameter A1",
        )
        # self.cp_phase_param_A2 = Var(
        #     within=Reals, initialize=-9.76e-2, units=cp_units * s_inv_units,
        #     doc='Specific heat of seawater parameter A2')
        # self.cp_phase_param_A3 = Var(
        #     within=Reals, initialize=4.04e-4, units=cp_units * s_inv_units**2,
        #     doc='Specific heat of seawater parameter A3')
        self.cp_phase_param_B1 = Var(
            within=Reals,
            initialize=-6.913e-3,
            units=cp_units * t_inv_units,
            doc="Specific heat of seawater parameter B1",
        )
        # self.cp_phase_param_B2 = Var(
        #     within=Reals, initialize=7.351e-4, units=cp_units * s_inv_units * t_inv_units,
        #     doc='Specific heat of seawater parameter B2')
        # self.cp_phase_param_B3 = Var(
        #     within=Reals, initialize=-3.15e-6, units=cp_units * s_inv_units**2 * t_inv_units,
        #     doc='Specific heat of seawater parameter B3')
        self.cp_phase_param_C1 = Var(
            within=Reals,
            initialize=9.6e-6,
            units=cp_units * t_inv_units**2,
            doc="Specific heat of seawater parameter C1",
        )
        # self.cp_phase_param_C2 = Var(
        #     within=Reals, initialize=-1.927e-6, units=cp_units * s_inv_units * t_inv_units**2,
        #     doc='Specific heat of seawater parameter C2')
        # self.cp_phase_param_C3 = Var(
        #     within=Reals, initialize=8.23e-9, units=cp_units * s_inv_units**2 * t_inv_units**2,
        #     doc='Specific heat of seawater parameter C3')
        self.cp_phase_param_D1 = Var(
            within=Reals,
            initialize=2.5e-9,
            units=cp_units * t_inv_units**3,
            doc="Specific heat of seawater parameter D1",
        )
        # self.cp_phase_param_D2 = Var(
        #     within=Reals, initialize=1.666e-9, units=cp_units * s_inv_units * t_inv_units**3,
        #     doc='Specific heat of seawater parameter D2')
        # self.cp_phase_param_D3 = Var(
        #     within=Reals, initialize=-7.125e-12, units=cp_units * s_inv_units**2 * t_inv_units**3,
        #     doc='Specific heat of seawater parameter D3')

        # Specific heat parameters for Cp vapor from NIST Webbook
        # Chase, M.W., Jr., NIST-JANAF Themochemical Tables, Fourth Edition, J. Phys. Chem. Ref. Data, Monograph 9, 1998, 1-1951
        self.cp_vap_param_A = Var(
            within=Reals,
            initialize=30.09200 / 18.01528e-3,
            units=cp_units,
            doc="Specific heat of water vapor parameter A",
        )
        self.cp_vap_param_B = Var(
            within=Reals,
            initialize=6.832514 / 18.01528e-3,
            units=cp_units * t_inv_units,
            doc="Specific heat of water vapor parameter B",
        )
        self.cp_vap_param_C = Var(
            within=Reals,
            initialize=6.793435 / 18.01528e-3,
            units=cp_units * t_inv_units**2,
            doc="Specific heat of water vapor parameter C",
        )
        self.cp_vap_param_D = Var(
            within=Reals,
            initialize=-2.534480 / 18.01528e-3,
            units=cp_units * t_inv_units**3,
            doc="Specific heat of water vapor parameter D",
        )
        self.cp_vap_param_E = Var(
            within=Reals,
            initialize=0.082139 / 18.01528e-3,
            units=cp_units * t_inv_units**-2,
            doc="Specific heat of water vapor parameter E",
        )

        # latent heat of pure water parameters from eq. 54 in Sharqawy et al. (2010)
        self.dh_vap_w_param_0 = Var(
            within=Reals,
            initialize=2.501e6,
            units=enth_mass_units,
            doc="Latent heat of pure water parameter 0",
        )
        self.dh_vap_w_param_1 = Var(
            within=Reals,
            initialize=-2.369e3,
            units=cp_units,
            doc="Latent heat of pure water parameter 1",
        )
        self.dh_vap_w_param_2 = Var(
            within=Reals,
            initialize=2.678e-1,
            units=enth_mass_units * t_inv_units**2,
            doc="Latent heat of pure water parameter 2",
        )
        self.dh_vap_w_param_3 = Var(
            within=Reals,
            initialize=-8.103e-3,
            units=enth_mass_units * t_inv_units**3,
            doc="Latent heat of pure water parameter 3",
        )
        self.dh_vap_w_param_4 = Var(
            within=Reals,
            initialize=-2.079e-5,
            units=enth_mass_units * t_inv_units**4,
            doc="Latent heat of pure water parameter 4",
        )

        # traditional parameters are the only Vars currently on the block and should be fixed
        for v in self.component_objects(Var):
            v.fix()

        # ---default scaling---
        self.set_default_scaling("temperature", 1e-2)
        self.set_default_scaling("pressure", 1e-5)
        self.set_default_scaling("dens_mass_phase", 1e-3, index="Liq")
        self.set_default_scaling("dens_mass_phase", 1, index="Vap")
        # self.set_default_scaling('dens_mass_solvent', 1e-3)
        self.set_default_scaling("enth_mass_phase", 1e-5, index="Liq")
        self.set_default_scaling("enth_mass_phase", 1e-6, index="Vap")
        self.set_default_scaling("pressure_sat", 1e-5)
        self.set_default_scaling("cp_phase", 1e-3, index="Liq")
        self.set_default_scaling("cp_phase", 1e-3, index="Vap")
        self.set_default_scaling("dh_vap", 1e-6)

    @classmethod
    def define_metadata(cls, obj):
        """Define properties supported and units."""
        obj.add_properties(
            {
                "flow_mass_phase_comp": {"method": None},
                "temperature": {"method": None},
                "pressure": {"method": None},
                "dens_mass_phase": {"method": "_dens_mass_phase"},
                "flow_vol_phase": {"method": "_flow_vol_phase"},
                "flow_vol": {"method": "_flow_vol"},
                "flow_mol_phase_comp": {"method": "_flow_mol_phase_comp"},
                "mole_frac_phase_comp": {"method": "_mole_frac_phase_comp"},
                "pressure_sat": {"method": "_pressure_sat"},
                "enth_mass_phase": {"method": "_enth_mass_phase"},
                "enth_flow_phase": {"method": "_enth_flow_phase"},
                "cp_phase": {"method": "_cp_phase"},
                "dh_vap": {"method": "_dh_vap"},
            }
        )

        obj.add_default_units(
            {
                "time": pyunits.s,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )


class _WaterStateBlock(StateBlock):
    """
    This Class contains methods which should be applied to Property Blocks as a
    whole, rather than individual elements of indexed Property Blocks.
    """

    def initialize(
        self,
        state_args=None,
        state_vars_fixed=False,
        hold_state=False,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        Initialization routine for property package.
        Keyword Arguments:
            state_args : Dictionary with initial guesses for the state vars
                         chosen. Note that if this method is triggered
                         through the control volume, and if initial guesses
                         were not provided at the unit model level, the
                         control volume passes the inlet values as initial
                         guess.The keys for the state_args dictionary are:

                         flow_mass_phase_comp : value at which to initialize
                                               phase component flows
                         pressure : value at which to initialize pressure
                         temperature : value at which to initialize temperature
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default={})
            state_vars_fixed: Flag to denote if state vars have already been
                              fixed.
                              - True - states have already been fixed by the
                                       control volume 1D. Control volume 0D
                                       does not fix the state vars, so will
                                       be False if this state block is used
                                       with 0D blocks.
                             - False - states have not been fixed. The state
                                       block will deal with fixing/unfixing.
            solver : Solver object to use during initialization if None is provided
                     it will use the default solver for IDAES (default = None)
            hold_state : flag indicating whether the initialization routine
                         should unfix any state variables fixed during
                         initialization (default=False).
                         - True - states variables are not unfixed, and
                                 a dict of returned containing flags for
                                 which states were fixed during
                                 initialization.
                        - False - state variables are unfixed after
                                 initialization by calling the
                                 release_state method
        Returns:
            If hold_states is True, returns a dict containing flags for
            which states were fixed during initialization.
        """
        # Get loggers
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="properties")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="properties")

        # Set solver and options
        opt = get_solver(solver, optarg)

        # Fix state variables
        flags = fix_state_vars(self, state_args)
        # Check when the state vars are fixed already result in dof 0
        for k in self.keys():
            dof = degrees_of_freedom(self[k])
            if dof != 0:
                raise PropertyPackageError(
                    "State vars fixed but degrees of "
                    "freedom for state block is not "
                    "zero during initialization."
                )

        # ---------------------------------------------------------------------
        # Initialize properties
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            results = solve_indexed_blocks(opt, [self], tee=slc.tee)
        init_log.info(
            "Property initialization: {}.".format(idaeslog.condition(results))
        )

        if not check_optimal_termination(results):
            raise InitializationError(
                f"{self.name} failed to initialize successfully. Please check "
                f"the output logs for more information."
            )

        # ---------------------------------------------------------------------
        # If input block, return flags, else release state
        if state_vars_fixed is False:
            if hold_state is True:
                return flags
            else:
                self.release_state(flags)

    def release_state(self, flags, outlvl=idaeslog.NOTSET):
        """
        Method to release state variables fixed during initialisation.

        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state=True.
            outlvl : sets output level of of logging
        """
        # Unfix state variables
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="properties")
        revert_state_vars(self, flags)
        init_log.info("{} State Released.".format(self.name))

    def calculate_state(
        self,
        var_args=None,
        hold_state=False,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        Solves state blocks given a set of variables and their values. These variables can
        be state variables or properties. This method is typically used before
        initialization to solve for state variables because non-state variables (i.e. properties)
        cannot be fixed in initialization routines.

        Keyword Arguments:
            var_args : dictionary with variables and their values, they can be state variables or properties
                       {(VAR_NAME, INDEX): VALUE}
            hold_state : flag indicating whether all of the state variables should be fixed after calculate state.
                         True - State variables will be fixed.
                         False - State variables will remain unfixed, unless already fixed.
            outlvl : idaes logger object that sets output level of solve call (default=idaeslog.NOTSET)
            solver : solver name string if None is provided the default solver
                     for IDAES will be used (default = None)
            optarg : solver options dictionary object (default={})

        Returns:
            results object from state block solve
        """
        # Get logger
        solve_log = idaeslog.getSolveLogger(self.name, level=outlvl, tag="properties")

        # Initialize at current state values (not user provided)
        self.initialize(solver=solver, optarg=optarg, outlvl=outlvl)

        # Set solver and options
        opt = get_solver(solver, optarg)

        # Fix variables and check degrees of freedom
        flags = (
            {}
        )  # dictionary noting which variables were fixed and their previous state
        for k in self.keys():
            sb = self[k]
            for (v_name, ind), val in var_args.items():
                var = getattr(sb, v_name)
                if iscale.get_scaling_factor(var[ind]) is None:
                    _log.warning(
                        "While using the calculate_state method on {sb_name}, variable {v_name} "
                        "was provided as an argument in var_args, but it does not have a scaling "
                        "factor. This suggests that the calculate_scaling_factor method has not been "
                        "used or the variable was created on demand after the scaling factors were "
                        "calculated. It is recommended to touch all relevant variables (i.e. call "
                        "them or set an initial value) before using the calculate_scaling_factor "
                        "method.".format(v_name=v_name, sb_name=sb.name)
                    )
                if var[ind].is_fixed():
                    flags[(k, v_name, ind)] = True
                    if value(var[ind]) != val:
                        raise ConfigurationError(
                            "While using the calculate_state method on {sb_name}, {v_name} was "
                            "fixed to a value {val}, but it was already fixed to value {val_2}. "
                            "Unfix the variable before calling the calculate_state "
                            "method or update var_args."
                            "".format(
                                sb_name=sb.name,
                                v_name=var.name,
                                val=val,
                                val_2=value(var[ind]),
                            )
                        )
                else:
                    flags[(k, v_name, ind)] = False
                    var[ind].fix(val)

            if degrees_of_freedom(sb) != 0:
                raise RuntimeError(
                    "While using the calculate_state method on {sb_name}, the degrees "
                    "of freedom were {dof}, but 0 is required. Check var_args and ensure "
                    "the correct fixed variables are provided."
                    "".format(sb_name=sb.name, dof=degrees_of_freedom(sb))
                )

        # Solve
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            results = solve_indexed_blocks(opt, [self], tee=slc.tee)
            solve_log.info_high(
                "Calculate state: {}.".format(idaeslog.condition(results))
            )

        if not check_optimal_termination(results):
            _log.warning(
                "While using the calculate_state method on {sb_name}, the solver failed "
                "to converge to an optimal solution. This suggests that the user provided "
                "infeasible inputs, or that the model is poorly scaled, poorly initialized, "
                "or degenerate."
            )

        # unfix all variables fixed with var_args
        for (k, v_name, ind), previously_fixed in flags.items():
            if not previously_fixed:
                var = getattr(self[k], v_name)
                var[ind].unfix()

        # fix state variables if hold_state
        if hold_state:
            fix_state_vars(self)

        return results


@declare_process_block_class("WaterStateBlock", block_class=_WaterStateBlock)
class WaterStateBlockData(StateBlockData):
    """A water property package."""

    def build(self):
        """Callable method for Block construction."""
        super().build()

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Add state variables
        self.flow_mass_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=0.5,
            bounds=(1e-10, None),
            domain=NonNegativeReals,
            units=pyunits.kg / pyunits.s,
            doc="Mass flow rate",
        )

        self.temperature = Var(
            initialize=298.15,
            bounds=(273.15, 1000),
            domain=NonNegativeReals,
            units=pyunits.K,
            doc="Temperature",
        )

        self.pressure = Var(
            initialize=101325,
            bounds=(1e3, 5e7),
            domain=NonNegativeReals,
            units=pyunits.Pa,
            doc="Pressure",
        )

    # -----------------------------------------------------------------------------
    # Property Methods

    def _dens_mass_phase(self):
        self.dens_mass_phase = Var(
            self.params.phase_list,
            initialize={"Liq": 1e3, "Vap": 1},
            bounds=(1e-3, 1e6),
            units=pyunits.kg * pyunits.m**-3,
            doc="Mass density of seawater",
        )

        def rule_dens_mass_phase(b, phase):  # density, eq. 8 in Sharqawy
            t = b.temperature - 273.15 * pyunits.K
            # s = b.mass_frac_phase_comp['Liq', 'TDS']
            if phase == "Liq":
                dens_mass = (
                    b.params.dens_mass_param_A1
                    + b.params.dens_mass_param_A2 * t
                    + b.params.dens_mass_param_A3 * t**2
                    + b.params.dens_mass_param_A4 * t**3
                    + b.params.dens_mass_param_A5 * t**4
                )
                return b.dens_mass_phase["Liq"] == dens_mass
            else:  # phase == 'Vap'
                dens_mass = (
                    b.params.dens_mass_param_mw
                    * b.pressure
                    / b.params.dens_mass_param_R
                    / b.temperature
                )
                return b.dens_mass_phase["Vap"] == dens_mass

        self.eq_dens_mass_phase = Constraint(
            self.params.phase_list, rule=rule_dens_mass_phase
        )

    def _flow_vol_phase(self):
        self.flow_vol_phase = Var(
            self.params.phase_list,
            initialize=1,
            bounds=(None, None),
            units=pyunits.m**3 / pyunits.s,
            doc="Volumetric flow rate",
        )

        def rule_flow_vol_phase(b, p):
            return (
                b.flow_vol_phase[p]
                == b.flow_mass_phase_comp[p, "H2O"] / b.dens_mass_phase[p]
            )

        self.eq_flow_vol_phase = Constraint(
            self.params.phase_list, rule=rule_flow_vol_phase
        )

    def _flow_vol(self):
        def rule_flow_vol(b):
            return sum(b.flow_vol_phase[p] for p in b.params.phase_list)

        self.flow_vol = Expression(rule=rule_flow_vol)

    def _flow_mol_phase_comp(self):
        self.flow_mol_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=100,
            bounds=(None, None),
            units=pyunits.mol / pyunits.s,
            doc="Molar flowrate",
        )

        def rule_flow_mol_phase_comp(b, p, j):
            return (
                b.flow_mol_phase_comp[p, j]
                == b.flow_mass_phase_comp[p, j] / b.params.mw_comp[j]
            )

        self.eq_flow_mol_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_flow_mol_phase_comp,
        )

    def _mole_frac_phase_comp(self):
        self.mole_frac_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=0.1,
            bounds=(1e-8, None),
            units=pyunits.dimensionless,
            doc="Mole fraction",
        )

        def rule_mole_frac_phase_comp(b, p, j):
            return b.mole_frac_phase_comp[p, j] == b.flow_mol_phase_comp[p, j] / sum(
                b.flow_mol_phase_comp[p, j] for p in b.params.phase_list
            )

        self.eq_mole_frac_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_mole_frac_phase_comp,
        )

    def _enth_mass_phase(self):
        self.enth_mass_phase = Var(
            self.params.phase_list,  # ['Liq','Vap']
            initialize=1e6,
            bounds=(1, 1e9),
            units=pyunits.J * pyunits.kg**-1,
            doc="Specific enthalpy",
        )

        def rule_enth_mass_phase(
            b, phase
        ):  # specific enthalpy, eq. 55 and 43 in Sharqawy
            t = (
                b.temperature - 273.15 * pyunits.K
            )  # temperature in degC, but pyunits in K
            h_w = (
                b.params.enth_mass_param_A1
                + b.params.enth_mass_param_A2 * t
                + b.params.enth_mass_param_A3 * t**2
                + b.params.enth_mass_param_A4 * t**3
            )

            if phase == "Liq":
                return b.enth_mass_phase["Liq"] == h_w
            else:  # phase == 'Vap'
                # dh_vap_w = b.params.dh_vap_w_param_0 + b.params.dh_vap_w_param_1 * t + b.params.dh_vap_w_param_2 * t ** 2 \
                #           + b.params.dh_vap_w_param_3 * t ** 3 + b.params.dh_vap_w_param_4 * t ** 4
                # h_vap = h_w + dh_vap_w
                return b.enth_mass_phase["Vap"] == h_w + b.dh_vap

        self.eq_enth_mass_phase = Constraint(
            self.params.phase_list, rule=rule_enth_mass_phase
        )

    def _pressure_sat(self):
        self.pressure_sat = Var(
            initialize=1e3,
            bounds=(1, 1e8),
            units=pyunits.Pa,
            doc="Saturation vapor pressure",
        )

        def rule_pressure_sat(b):  # vapor pressure, eq. 5 and 6 in Nayar et al.(2016)
            t = b.temperature
            psatw = (
                exp(
                    b.params.pressure_sat_param_psatw_A1 * t**-1
                    + b.params.pressure_sat_param_psatw_A2
                    + b.params.pressure_sat_param_psatw_A3 * t
                    + b.params.pressure_sat_param_psatw_A4 * t**2
                    + b.params.pressure_sat_param_psatw_A5 * t**3
                    + b.params.pressure_sat_param_psatw_A6 * log(t / pyunits.K)
                )
                * pyunits.Pa
            )
            return b.pressure_sat == psatw

        self.eq_pressure_sat = Constraint(rule=rule_pressure_sat)

    def _enth_flow_phase(self):
        # enthalpy flow variable
        self.enth_flow_phase = Var(
            self.params.phase_list,  # ['Liq','Vap']
            initialize=1e6,
            bounds=(None, None),
            units=pyunits.J / pyunits.s,
            doc="Enthalpy flow",
        )

        def rule_enth_flow_phase(b, p):  # enthalpy flow [J/s]
            return (
                b.enth_flow_phase[p]
                == b.flow_mass_phase_comp[p, "H2O"] * b.enth_mass_phase[p]
            )

        self.eq_enth_flow_phase = Constraint(
            self.params.phase_list, rule=rule_enth_flow_phase
        )

    def _cp_phase(self):
        self.cp_phase = Var(
            self.params.phase_list,
            initialize=4e3,
            bounds=(1e-8, 1e8),
            units=pyunits.J / pyunits.kg / pyunits.K,
            doc="Specific heat capacity",
        )

        def rule_cp_phase(b, phase):
            if phase == "Liq":
                # specific heat, eq. 9 in Sharqawy et al. (2010)
                # Convert T90 to T68, eq. 4 in Sharqawy et al. (2010); primary reference from Rusby (1991)
                t = (b.temperature - 0.00025 * 273.15 * pyunits.K) / (1 - 0.00025)
                # s = b.mass_frac_phase_comp['Liq', 'TDS'] * 1000 * pyunits.g / pyunits.kg
                A = (
                    b.params.cp_phase_param_A1
                )  # + b.params.cp_phase_param_A2 * s + b.params.cp_phase_param_A3 * s ** 2
                B = (
                    b.params.cp_phase_param_B1
                )  # + b.params.cp_phase_param_B2 * s + b.params.cp_phase_param_B3 * s ** 2
                C = (
                    b.params.cp_phase_param_C1
                )  # + b.params.cp_phase_param_C2 * s + b.params.cp_phase_param_C3 * s ** 2
                D = (
                    b.params.cp_phase_param_D1
                )  # + b.params.cp_phase_param_D2 * s + b.params.cp_phase_param_D3 * s ** 2
                return b.cp_phase["Liq"] == (A + B * t + C * t**2 + D * t**3) * 1000
            else:  # phase == 'Vap'
                t = b.temperature / 1000
                return (
                    b.cp_phase["Vap"]
                    == b.params.cp_vap_param_A
                    + b.params.cp_vap_param_B * t
                    + b.params.cp_vap_param_C * t**2
                    + b.params.cp_vap_param_D * t**3
                    + b.params.cp_vap_param_E / t**2
                )

        self.eq_cp_phase = Constraint(self.params.phase_list, rule=rule_cp_phase)

    def _dh_vap(self):
        self.dh_vap = Var(
            initialize=2.4e6,
            bounds=(1, 1e9),
            units=pyunits.J / pyunits.kg,
            doc="Latent heat of vaporization",
        )

        def rule_dh_vap(
            b,
        ):  # latent heat of seawater from eq. 37 and eq. 55 in Sharqawy et al. (2010)
            t = b.temperature - 273.15 * pyunits.K
            return (
                b.dh_vap
                == b.params.dh_vap_w_param_0
                + b.params.dh_vap_w_param_1 * t
                + b.params.dh_vap_w_param_2 * t**2
                + b.params.dh_vap_w_param_3 * t**3
                + b.params.dh_vap_w_param_4 * t**4
            )

        self.eq_dh_vap = Constraint(rule=rule_dh_vap)

    # General Methods
    # NOTE: For scaling in the control volume to work properly, these methods must
    # return a pyomo Var or Expression

    def get_material_flow_terms(self, p, j):
        """Create material flow terms for control volume."""
        return self.flow_mass_phase_comp[p, j]

    def get_enthalpy_flow_terms(self, p):
        """Create enthalpy flow terms."""
        return self.enth_flow_phase[p]

    # TODO: make property package compatible with dynamics
    # def get_material_density_terms(self, p, j):
    #     """Create material density terms."""

    # def get_enthalpy_density_terms(self, p):
    #     """Create enthalpy density terms."""

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal

    def get_material_flow_basis(b):
        return MaterialFlowBasis.mass

    def define_state_vars(self):
        """Define state vars."""
        return {
            "flow_mass_phase_comp": self.flow_mass_phase_comp,
            "temperature": self.temperature,
            "pressure": self.pressure,
        }

    # -----------------------------------------------------------------------------
    # Scaling methods
    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        # setting scaling factors for variables

        # default scaling factors have already been set with
        # idaes.core.property_base.calculate_scaling_factors()
        # for the following variables: flow_mass_phase_comp, pressure,
        # temperature, dens_mass_phase, visc_d_phase, osm_coeff, and enth_mass_phase

        # these variables should have user input
        if iscale.get_scaling_factor(self.flow_mass_phase_comp["Liq", "H2O"]) is None:
            sf = iscale.get_scaling_factor(
                self.flow_mass_phase_comp["Liq", "H2O"], default=1e0, warning=True
            )
            iscale.set_scaling_factor(self.flow_mass_phase_comp["Liq", "H2O"], sf)

        if iscale.get_scaling_factor(self.flow_mass_phase_comp["Vap", "H2O"]) is None:
            sf = iscale.get_scaling_factor(
                self.flow_mass_phase_comp["Vap", "H2O"], default=1e0, warning=True
            )
            iscale.set_scaling_factor(self.flow_mass_phase_comp["Vap", "H2O"], sf)

        # scaling factors for parameters
        for j, v in self.params.mw_comp.items():
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(self.params.mw_comp, 1e2)

        # these variables do not typically require user input,
        # will not override if the user does provide the scaling factor
        if self.is_property_constructed("flow_vol_phase"):
            for p in self.params.phase_list:
                if iscale.get_scaling_factor(self.flow_vol_phase[p]) is None:
                    sf = iscale.get_scaling_factor(
                        self.flow_mass_phase_comp[p, "H2O"]
                    ) / iscale.get_scaling_factor(self.dens_mass_phase[p])
                    iscale.set_scaling_factor(self.flow_vol_phase[p], sf)

        if self.is_property_constructed("flow_vol"):
            if iscale.get_scaling_factor(self.flow_vol) is None:
                sf_liq = iscale.get_scaling_factor(self.flow_vol_phase["Liq"])
                sf_vap = iscale.get_scaling_factor(self.flow_vol_phase["Vap"])
                sf = min(sf_liq, sf_vap)
                iscale.set_scaling_factor(self.flow_vol, sf)

        if self.is_property_constructed("flow_mol_phase_comp"):
            for p in self.params.phase_list:
                if (
                    iscale.get_scaling_factor(self.flow_mol_phase_comp[p, "H2O"])
                    is None
                ):
                    sf = iscale.get_scaling_factor(self.flow_mass_phase_comp[p, "H2O"])
                    sf /= iscale.get_scaling_factor(self.params.mw_comp["H2O"])
                    iscale.set_scaling_factor(self.flow_mol_phase_comp[p, "H2O"], sf)

        if self.is_property_constructed("mole_frac_phase_comp"):
            sf_flow_mol_liq = iscale.get_scaling_factor(
                self.flow_mol_phase_comp["Liq", "H2O"]
            )
            sf_flow_mol_vap = iscale.get_scaling_factor(
                self.flow_mol_phase_comp["Vap", "H2O"]
            )
            sf_flow_mol = min(sf_flow_mol_liq, sf_flow_mol_vap)
            for p in self.params.phase_list:
                if (
                    iscale.get_scaling_factor(self.mole_frac_phase_comp[p, "H2O"])
                    is None
                ):
                    sf = (
                        iscale.get_scaling_factor(self.flow_mol_phase_comp[p, "H2O"])
                        / sf_flow_mol
                    )
                    print("iter:", p, sf)
                    iscale.set_scaling_factor(self.mole_frac_phase_comp[p, "H2O"], sf)

        if self.is_property_constructed("enth_flow_phase"):
            for p in self.params.phase_list:
                if iscale.get_scaling_factor(self.enth_flow_phase[p]) is None:
                    sf = iscale.get_scaling_factor(self.flow_mass_phase_comp[p, "H2O"])
                    sf *= iscale.get_scaling_factor(self.enth_mass_phase[p])
                    iscale.set_scaling_factor(self.enth_flow_phase[p], sf)

        # transforming constraints
        for metadata_dic in self.params.get_metadata().properties.values():
            var_str = metadata_dic["name"]
            if metadata_dic["method"] is not None and self.is_property_constructed(
                var_str
            ):
                var = getattr(self, var_str)
                if isinstance(var, Expression):
                    continue  # properties that are expressions do not have constraints
                con = getattr(self, "eq_" + var_str)
                for ind in con.keys():
                    sf = iscale.get_scaling_factor(var[ind], default=1, warning=True)
                    iscale.constraint_scaling_transform(con[ind], sf)
