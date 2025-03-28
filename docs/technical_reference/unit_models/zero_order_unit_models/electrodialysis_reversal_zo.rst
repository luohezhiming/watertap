Electrodialysis Reversal (ZO)
=============================

Model Type
----------
This unit model is formulated as a single-input, double-output model form.
See documentation for :ref:`single-input, double-output Helper Methods<sido_methods>`.

Electricity Consumption
-----------------------
The constraint used to calculate energy consumption is described in the Additional Constraints section below. More details can be found in the unit model class.

Costing Method
--------------
Costing is calculated using the cost_power_law_flow method in the zero-order costing package.
See documentation for the :ref:`zero-order costing package<zero_order_costing>`.

Additional Variables
--------------------

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Constant 1 in electricity intensity equation", "elec_coeff_1", "kWh/m**3"
   "Constant 2 in electricity intensity equation", "elec_coeff_2", "kWh*l/m**3/mg"
   "Power consumption of brine concentrator", "electricity", "kW"
   "Specific energy consumption with respect to feed flowrate", "electricity_intensity", "kWh/m**3"

Additional Constraints
----------------------

.. csv-table::
   :header: "Description", "Constraint Name"

   "Electricity intensity constraint", "electricity_intensity_constraint"
   "Power consumption constraint", "electricity_constraint"

.. index::
   pair: watertap.unit_models.zero_order.electrodialysis_reversal_zo;electrodialysis_reversal_zo

.. currentmodule:: watertap.unit_models.zero_order.electrodialysis_reversal_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.electrodialysis_reversal_zo
    :members:
    :noindex:
