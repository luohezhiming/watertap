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

from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
import os


__author__ = "Chenyu Wang"

_log = idaeslog.getLogger(__name__)


@declare_process_block_class("ElectroNP")
class ElectroNPdata(SeparatorData):
    """
    Zero order electrochemical nutrient removal (ElectroNP) model based on specified removal efficiencies for nitrogen and phosphorus.
    """

    CONFIG = SeparatorData.CONFIG()
    CONFIG.outlet_list = ["treated", "byproduct"]
    CONFIG.split_basis = SplittingType.componentFlow

    def build(self):
        # Call UnitModel.build to set up dynamics
        super(ElectroNPdata, self).build()

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

        if "treated" and "byproduct" not in self.config.outlet_list:
            raise ConfigurationError(
                "{} encountered unrecognised "
                "outlet_list. This should not "
                "occur - please use treated "
                "and byproduct.".format(self.name)
            )

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        units_meta = self.config.property_package.get_metadata().get_derived_units

        add_object_reference(self, "properties_in", self.mixed_state)
        add_object_reference(self, "properties_treated", self.treated_state)
        add_object_reference(self, "properties_byproduct", self.byproduct_state)

        # Add performance variables
        # NOTE: the mass fraction of H2O to treated stream is estimated from P recovered in the byproduct (struvite)
        self.frac_mass_H2O_treated = Var(
            self.flowsheet().time,
            initialize=0.8777,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0.0, 1),
            doc="Mass recovery fraction of water in the treated stream",
        )
        self.frac_mass_H2O_treated.fix()

        add_object_reference(self, "removal_frac_mass_comp", self.split_fraction)

        ############################################## var with unit ##############################################
        self.cathodic_potential = Var(
            initialize=-1.05,
            domain=NegativeReals,
            units=pyunits.V,
            bounds=(-1.3, -0.8),
            doc="Cathodic potential",
        )

        self.area_volume_ratio = Var(
            initialize=0.105,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0, 1),
            doc="Area-volume ratio",
        )

        self.P_removal = Var(
            initialize=0.9,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0, 1),
            doc="phosphorus removal fraction on a mass basis",
        )

        self.N_removal = Var(
            initialize=0.3,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0, 1),
            doc="Nitrogen removal fraction on a mass basis",
        )

        self.settling_time = Var(
            initialize=30,
            domain=NonNegativeReals,
            units=pyunits.min,
            bounds=(0, None),
            doc="Settling time for electroN-P process",
        )

        self.CP_surrogate = Var(
            initialize=-1.05,
            domain=NegativeReals,
            units=pyunits.dimensionless,
            bounds=(-1.3, -0.8),
            doc="Cathodic potential",
        )

        @self.Constraint(
            doc="Constraint for convert cathodic potential to dimensionless form",
        )
        def eq_CP_surrogate(b):
            return b.CP_surrogate == pyunits.convert(
                b.cathodic_potential / (1 * pyunits.V), to_units=pyunits.dimensionless
            )

        self.r_AV_surrogate = Var(
            initialize=0.105,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0, 1),
            doc="Area-volume ratio",
        )

        @self.Constraint(
            doc="Constraint for convert area-volume ratio to dimensionless form",
        )
        def eq_r_AV_surrogate(b):
            return b.r_AV_surrogate == pyunits.convert(
                b.area_volume_ratio, to_units=pyunits.dimensionless
            )

        self.T_surrogate = Var(
            initialize=25,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0, None),
            doc="Temperature",
        )

        @self.Constraint(
            doc="Constraint for convert temperature to dimensionless form",
        )
        def eq_T_surrogate(b):
            return b.T_surrogate == pyunits.convert(
                b.inlet.temperature[0] / (1 * pyunits.K) - 273.15,
                to_units=pyunits.dimensionless,
            )

        self.t_ss_surrogate = Var(
            initialize=30,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0, None),
            doc="Settling time for electroN-P process",
        )

        @self.Constraint(
            doc="Constraint for convert settling time to dimensionless form",
        )
        def eq_t_ss_surrogate(b):
            return b.t_ss_surrogate == pyunits.convert(
                b.settling_time / (1 * pyunits.min), to_units=pyunits.dimensionless
            )

        self.P_removal_surrogate = Var(
            initialize=90,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0, 100),
            doc="Percentage of phosphorus removal on a mass basis",
        )

        @self.Constraint(
            doc="Constraint for convert P removal to dimensionless form",
        )
        def eq_P_removal_surrogate(b):
            return b.P_removal_surrogate == pyunits.convert(
                (b.P_removal * 100), to_units=pyunits.dimensionless
            )

        inputs = [
            self.CP_surrogate,
            self.r_AV_surrogate,
            self.T_surrogate,
            self.t_ss_surrogate,
        ]
        outputs = [self.P_removal_surrogate]
        self.surrogate_PR = SurrogateBlock(concrete=True)
        PR_source_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pysmo_RBF_PR_surrogate.json",
        )
        # PR_surrogate = PysmoSurrogate.load_from_file(
        #     r"D:\Keylogic\WaterTap_Chenyu\watertap\watertap\unit_models\electroNP_surrogate\pysmo_RBF_PR_surrogate.json")
        # PR_surrogate = PysmoSurrogate.load_from_file("pysmo_RBF_PR_surrogate.json")
        PR_surrogate = PysmoSurrogate.load_from_file(PR_source_file)
        self.surrogate_PR.build_model(
            PR_surrogate, input_vars=inputs, output_vars=outputs
        )

        ############################################## dimensionless var ##############################################
        # self.cathodic_potential = Var(
        #     initialize=-1.05,
        #     domain=NegativeReals,
        #     units=pyunits.dimensionless,
        #     bounds=(-1.3, -0.8),
        #     doc="Cathodic potential",
        # )
        #
        # self.area_volume_ratio = Var(
        #     initialize=0.105,
        #     domain=NonNegativeReals,
        #     units=pyunits.dimensionless,
        #     bounds=(0, 1),
        #     doc="Area-volume ratio",
        # )
        #
        # self.T = Var(
        #     initialize=25,
        #     domain=NonNegativeReals,
        #     units=pyunits.dimensionless,
        #     bounds=(0, None),
        #     doc="Temperature",
        # )
        #
        # self.P_removal = Var(
        #     initialize=0.9,
        #     domain=NonNegativeReals,
        #     units=pyunits.dimensionless,
        #     bounds=(0, 1),
        #     doc="phosphorus removal fraction on a mass basis",
        # )
        #
        # self.N_removal = Var(
        #     initialize=0.3,
        #     domain=NonNegativeReals,
        #     units=pyunits.dimensionless,
        #     bounds=(0, 1),
        #     doc="Nitrogen removal fraction on a mass basis",
        # )
        #
        # self.settling_time = Var(
        #     initialize=30,
        #     domain=NonNegativeReals,
        #     units=pyunits.dimensionless,
        #     bounds=(0, None),
        #     doc="Settling time for electroN-P process",
        # )
        #
        # # CP = pyunits.convert((self.cathodic_potential / (1 * pyunits.V)), to_units=pyunits.dimensionless)
        # # r_AV = pyunits.convert(self.area_volume_ratio, to_units=pyunits.dimensionless)
        # # T = pyunits.convert((self.properties_in[0].temperature / (1 * pyunits.K)), to_units=pyunits.dimensionless)
        # # t_ss = pyunits.convert((self.settling_time / (1 * pyunits.min)), to_units=pyunits.dimensionless)
        # inputs = [self.cathodic_potential, self.area_volume_ratio, self.T, self.settling_time]
        # outputs = [self.P_removal]
        # self.surrogate = SurrogateBlock(concrete=True)
        # PR_surrogate = PysmoSurrogate.load_from_file(
        #     r"D:\Keylogic\WaterTap_Chenyu\watertap\watertap\unit_models\electroNP_surrogate\pysmo_RBF_PR_surrogate.json")
        # self.surrogate.build_model(PR_surrogate, input_vars=inputs, output_vars=outputs)

        # @self.Constraint(
        #     self.flowsheet().time,
        #     doc="Constraint for phosphorus removal",
        # )
        # def eq_P_removal(b, t):
        #     CP = pyunits.convert((b.cathodic_potential/(1 * pyunits.V)), to_units=pyunits.dimensionless)
        #     r_AV = pyunits.convert(b.area_volume_ratio, to_units=pyunits.dimensionless)
        #     T = pyunits.convert((b.properties_in[t].temperature/ (1 * pyunits.K)), to_units=pyunits.dimensionless)
        #     t_ss = pyunits.convert((b.settling_time/ (1 * pyunits.min)), to_units=pyunits.dimensionless)
        #     inputs = [CP, r_AV, T, t_ss]
        #     outputs = [b.P_removal]
        #     b.surrogate = SurrogateBlock(concrete=True)
        #     PR_surrogate = PysmoSurrogate.load_from_file(r"D:\Keylogic\WaterTap_Chenyu\watertap\watertap\unit_models\electroNP_surrogate\pysmo_RBF_PR_surrogate.json")
        #     return b.surrogate.build_model(PR_surrogate, input_vars=inputs, output_vars=outputs)

        # @self.Constraint(
        #     doc="Constraint for phosphorus removal",
        # )
        # def eq_P_removal(b):
        #     inputs = [b.cathodic_potential, b.area_volume_ratio, b.T, b.settling_time]
        #     outputs = [b.P_removal]
        #     b.surrogate = SurrogateBlock(concrete=True)
        #     PR_surrogate = PysmoSurrogate.load_from_file(
        #         r"D:\Keylogic\WaterTap_Chenyu\watertap\watertap\unit_models\electroNP_surrogate\pysmo_RBF_PR_surrogate.json")
        #     b.surrogate.build_model(PR_surrogate, input_vars=inputs, output_vars=outputs)

        @self.Constraint(
            doc="Constraint for nitrogen removal",
        )
        def eq_N_removal(b):
            return b.N_removal == 0.3 * b.P_removal

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.component_list,
            doc="soluble fraction",
        )
        def split_components(blk, t, i):
            if i == "H2O":
                return (
                    blk.removal_frac_mass_comp[t, "byproduct", i]
                    == 1 - blk.frac_mass_H2O_treated[t]
                )
            elif i == "S_PO4":
                return blk.removal_frac_mass_comp[t, "byproduct", i] == blk.P_removal
            elif i == "S_NH4":
                return blk.removal_frac_mass_comp[t, "byproduct", i] == blk.N_removal
            else:
                return blk.removal_frac_mass_comp[t, "byproduct", i] == 1e-7

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

        self.EI_surrogate = Var(
            initialize=0.044,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            bounds=(0, None),
            doc="Electricity intensity with respect to phosphorus removal",
        )

        @self.Constraint(
            doc="Constraint for convert energy intensity to dimensionless form",
        )
        def eq_EI_surrogate(b):
            return b.EI_surrogate == pyunits.convert(
                b.energy_electric_flow_mass / (1 * pyunits.kWh / pyunits.kg),
                to_units=pyunits.dimensionless,
            )

        inputs = [
            self.CP_surrogate,
            self.r_AV_surrogate,
            self.T_surrogate,
            self.t_ss_surrogate,
        ]
        outputs = [self.EI_surrogate]
        self.surrogate_EI = SurrogateBlock(concrete=True)
        EI_source_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pysmo_RBF_Ener_surrogate.json",
        )
        EI_surrogate = PysmoSurrogate.load_from_file(EI_source_file)
        # EI_surrogate = PysmoSurrogate.load_from_file(
        #     r"D:\Keylogic\WaterTap_Chenyu\watertap\watertap\unit_models\electroNP_surrogate\pysmo_RBF_Ener_surrogate.json")
        self.surrogate_EI.build_model(
            EI_surrogate, input_vars=inputs, output_vars=outputs
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="Constraint for electricity consumption based on phosphorus removal",
        )
        def electricity_consumption(b, t):
            return b.electricity[t] == (
                b.energy_electric_flow_mass
                * pyunits.convert(
                    b.properties_byproduct[t].get_material_flow_terms("Liq", "S_PO4"),
                    to_units=pyunits.kg / pyunits.hour,
                )
            )

        self.magnesium_chloride_dosage = Var(
            units=pyunits.dimensionless,
            bounds=(0, None),
            doc="Dosage of magnesium chloride per phosphorus removal",
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
                    b.properties_byproduct[t].get_material_flow_terms("Liq", "S_PO4"),
                    to_units=pyunits.kg / pyunits.hour,
                )
            )

    def _get_performance_contents(self, time_point=0):
        var_dict = {}
        var_dict["Mass fraction of H2O in treated stream"] = self.frac_mass_H2O_treated[
            time_point
        ]
        for j in self.config.property_package.solute_set:
            var_dict[f"Solute Removal {j}"] = self.removal_frac_mass_comp[
                time_point, "byproduct", j
            ]
        var_dict["Electricity Demand"] = self.electricity[time_point]
        var_dict["Electricity Intensity"] = self.energy_electric_flow_mass
        var_dict["Dosage of magnesium chloride per treated phosphorus"] = (
            self.magnesium_chloride_dosage
        )
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
        super().calculate_scaling_factors()

        iscale.set_scaling_factor(self.frac_mass_H2O_treated, 1)

        if iscale.get_scaling_factor(self.energy_electric_flow_mass) is None:
            sf = iscale.get_scaling_factor(
                self.energy_electric_flow_mass, default=1e-3, warning=True
            )
            iscale.set_scaling_factor(self.energy_electric_flow_mass, sf)

        if iscale.get_scaling_factor(self.magnesium_chloride_dosage) is None:
            sf = iscale.get_scaling_factor(
                self.magnesium_chloride_dosage, default=1e0, warning=True
            )
            iscale.set_scaling_factor(self.magnesium_chloride_dosage, sf)

        for (t, i, j), v in self.removal_frac_mass_comp.items():
            if i == "treated":
                for i in self.config.outlet_list:
                    if j == "S_PO4":
                        sf = 1
                    elif j == "S_NH4":
                        sf = 1
                    else:
                        sf = 1
            iscale.set_scaling_factor(v, sf)

        for (t, i, j), v in self.removal_frac_mass_comp.items():
            if i == "byproduct":
                for i in self.config.outlet_list:
                    if j == "S_PO4":
                        sf = 1
                    elif j == "S_NH4":
                        sf = 1
                    else:
                        sf = 1e6
            iscale.set_scaling_factor(v, sf)

        for t, v in self.electricity.items():
            sf = (
                iscale.get_scaling_factor(self.energy_electric_flow_mass)
                * iscale.get_scaling_factor(self.inlet.flow_vol[t])
                * iscale.get_scaling_factor(self.inlet.conc_mass_comp[t, "S_PO4"])
            )
            iscale.set_scaling_factor(v, sf)

        for t, v in self.MgCl2_flowrate.items():
            sf = (
                iscale.get_scaling_factor(self.magnesium_chloride_dosage)
                * iscale.get_scaling_factor(self.inlet.flow_vol[t])
                * iscale.get_scaling_factor(self.inlet.conc_mass_comp[t, "S_PO4"])
            )
            iscale.set_scaling_factor(v, sf)

    @property
    def default_costing_method(self):
        return cost_electroNP
