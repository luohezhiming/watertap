Well Field (ZO)
===============

Model Type
----------
This unit model is formulated as a pass-through model form.
See documentation for :ref:`pass-through Helper Methods<pt_methods>`.

Electricity Consumption
-----------------------
Electricity consumption is calculated using the pump_electricity helper function.
See documentation for :ref:`Helper Methods for Electricity Demand<electricity_methods>`.

Costing Method
--------------
Costing is calculated using the cost_well_field method in the zero-order costing package.
See documentation for the :ref:`zero-order costing package<zero_order_costing>`.

Additional Variables
--------------------

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Piping distance", "pipe_distance", "mi"
   "Pipe diameter", "pipe_diameter", "in"

.. index::
   pair: watertap.unit_models.zero_order.well_field_zo;well_field_zo

.. currentmodule:: watertap.unit_models.zero_order.well_field_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.well_field_zo
    :members:
    :noindex:
