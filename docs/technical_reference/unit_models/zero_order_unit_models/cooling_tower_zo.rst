Cooling Tower  (ZO)
===================

Model Type
----------
This unit model is formulated as a single-input, double-output model form.
See documentation for :ref:`single-input, double-output Helper Methods<sido_methods>`.

Electricity Consumption
-----------------------
Electricity consumption is calculated using the constant_intensity helper function.
See documentation for :ref:`Helper Methods for Electricity Demand<electricity_methods>`.

Costing Method
--------------
Costing is calculated using the cost_power_law_flow method in the zero-order costing package.
See documentation for the :ref:`zero-order costing package<zero_order_costing>`.

Additional Variables
--------------------

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Cycles of concentration", "cycles", "None"
   "Flowrate of blowdown", "blowdown", "m**3/hr"

Additional Constraints
----------------------

.. csv-table::
   :header: "Description", "Constraint Name"

   "Blowdown constraint", "blowdown_constraint"

.. index::
   pair: watertap.unit_models.zero_order.cooling_tower_zo;cooling_tower_zo

.. currentmodule:: watertap.unit_models.zero_order.cooling_tower_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.cooling_tower_zo
    :members:
    :noindex:
