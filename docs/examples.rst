.. _examples:

Examples
========

The worked examples live in the tutorials.  Each one is a complete run: a
config to copy, the command to launch it, and the plots it produces.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Tutorial
     - What it shows
   * - :doc:`tutorial_1_greenfield`
     - A greenfield build from nothing, run twice — once against a fixed
       demand, once against a product price.  Start here.
   * - :doc:`tutorial_2_brownfield`
     - Existing plant kept in the system, with sunk and residual capital
       costs, plus ramp and minimum-load constraints.
   * - :doc:`tutorial_2b_brownfield_heat`
     - The heat side: district heating off-take, thermal storage, and how the
       heat circuits connect.
   * - :doc:`tutorial_3_rolling_horizon`
     - Dispatch only, on fixed capacities, compared against perfect foresight.
   * - :doc:`tutorial_4_stochastic`
     - One investment decision across several price years, and the value of
       perfect information (EVPI).

For the model itself rather than a run, see :doc:`design`, :doc:`model_approach` and
:doc:`technologies`.  For reading results, see :doc:`guide_outputs` and
:doc:`guide_economic_analysis`.
