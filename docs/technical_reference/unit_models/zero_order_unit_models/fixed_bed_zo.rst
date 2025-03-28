Fixed Bed (ZO)
==============

Model Type
----------
This unit model is formulated as a single-input, single-output model form.
See documentation for :ref:`single-input, single-output Helper Methods<siso_methods>`.

Electricity Consumption
-----------------------
Electricity consumption is calculated using the constant_intensity helper function.
See documentation for :ref:`Helper Methods for Electricity Demand<electricity_methods>`.

Costing Method
--------------
Costing is calculated using the cost_fixed_bed method in the zero-order costing package.
See documentation for the :ref:`zero-order costing package<zero_order_costing>`.

Additional Variables
--------------------

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Dosing rate of acetic acid", "acetic_acid_dose", "kg/m**3"
   "Dosing rate of phosphoric acid", "phosphoric_acid_dose", "kg/m**3"
   "Dosing rate of ferric chloride", "ferric_chloride_dose", "kg/m**3"
   "Consumption rate of acetic acid", "acetic_acid_demand", "kg/hr"
   "Consumption rate of phosphoric acid", "phosphoric_acid_demand", "kg/hr"
   "Consumption rate of ferric chloride", "ferric_chloride_demand", "kg/hr"
   "Replacement rate for activated carbon", "activated_carbon_demand", "kg/hr"
   "Pre-exponential factor for activated carbon demand", "activated_carbon_parameter_a", "kg/m**3"
   "Exponential factor for activated carbon demand", "activated_carbon_parameter_b", "None"
   "Replacement rate for sand", "sand_demand", "kg/hr"
   "Pre-exponential factor for sand demand", "sand_parameter_a", "kg/m**3"
   "Exponential factor for sand demand", "sand_parameter_b", "None"
   "Replacement rate for anthracite", "anthracite_demand", "kg/hr"
   "Pre-exponential factor for anthracite demand", "anthracite_parameter_a", "kg/m**3"
   "Exponential factor for anthracite demand", "anthracite_parameter_b", "None"
   "Replacement rate for cationic polymer", "cationic_polymer_demand", "kg/hr"
   "Pre-exponential factor for cationic polymer demand", "cationic_polymer_parameter_a", "kg/m**3"
   "Exponential factor for cationic polymer demand", "cationic_polymer_parameter_b", "None"

Additional Constraints
----------------------

.. csv-table::
   :header: "Description", "Constraint Name"

   "Acetic acid demand constraint", "acetic_acid_demand_equation"
   "Phosphoric acid demand constraint", "phosphoric_acid_demand_equation"
   "Acetic acid demand constraint", "ferric_chloride_demand_equation"
   "Activated carbon demand constraint", "activated_carbon_demand_equation"
   "Sand demand constraint", "sand_demand_equation"
   "Anthracite demand constraint", "anthracite_demand_equation"
   "Cationic Polymer demand constraint", "cationic_polymer_demand_equation"

.. index::
   pair: watertap.unit_models.zero_order.fixed_bed_zo;fixed_bed_zo

.. currentmodule:: watertap.unit_models.zero_order.fixed_bed_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.fixed_bed_zo
    :members:
    :noindex:
