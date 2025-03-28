Granular Activated Carbon (ZO)
==============================

Model Type
----------
This unit model is formulated as a single-input, double-output model form.
See documentation for :ref:`single-input, double-output Helper Methods<sido_methods>`.

Electricity Consumption
-----------------------
The constraint used to calculate energy consumption is described in the Additional Constraints section below. More details can be found in the unit model class.

Costing Method
--------------
Costing is calculated using the cost_gac method in the zero-order costing package.
See documentation for the :ref:`zero-order costing package<zero_order_costing>`.

Additional Variables
--------------------

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Empty bed contact time of unit", "empty_bed_contact_time", "hr"
   "Electricity consumption of unit", "electricity", "kW"
   "Parameter for calculating electricity based on empty bed contact time", "electricity_intensity_parameter", "kW/m**3"
   "Electricity intensity with respect to inlet flowrate of unit", "energy_electric_flow_vol_inlet", "kWh/m**3"
   "Replacement rate of activated carbon", "activated_carbon_replacement", "kg/m**3"
   "Demand for activated carbon", "activated_carbon_demand", "kg/hr"

Additional Constraints
----------------------

.. csv-table::
   :header: "Description", "Constraint Name"

   "Electricity intensity based on empty bed contact time.", "electricity_intensity_constraint"
   "Constraint for electricity consumption based on feed flowrate.", "electricity_consumption"
   "Constraint for activated carbon consumption.", "activated_carbon_equation"

.. index::
   pair: watertap.unit_models.zero_order.gac_zo;gac_zo

.. currentmodule:: watertap.unit_models.zero_order.gac_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.gac_zo
    :members:
    :noindex:
