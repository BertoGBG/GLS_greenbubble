.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tutorial-2b-brownfield-heat:

Tutorial 2b — Heat Network Integration
=======================================

This tutorial extends the brownfield plant from :ref:`tutorial-2-brownfield`
by putting **heat at the centre of the analysis**. The key question: when a
large electrolyser produces substantial waste heat, what is the best way to
collect, store and sell it?

The setup adds three heat-side technologies to the existing plant:

* **Heat pump** — upgrades low-temperature electrolyser waste heat (30–60 °C)
  to district-heating supply temperature (≈ 70–90 °C).
* **District-heating water-tank storage (TES DH)** — a thermal buffer that
  decouples heat production from DH demand, allowing the plant to exploit cheap
  electricity hours without oversizing the DH connection.
* **DH connection** — sells heat to the district-heating grid at a fixed price
  of 100 €/MWh.

The electrolysis plant is fixed at **100 MW** (a larger unit than Tutorial 2) to
generate the waste heat that makes the heat-network economics interesting.

.. contents:: On this page
   :local:
   :depth: 1

---

1 · Run it
----------

.. code-block:: bash

   cp tutorials/2_brownfield_heat/config.yaml   config/config.yaml
   cp tutorials/2_brownfield_heat/n_config.yaml config/n_config.yaml
   snakemake --cores 4
   python tutorials/2_brownfield_heat/plot_heat_network.py   # topology + DH demand figures

Key config choices (:download:`config.yaml <../tutorials/2_brownfield_heat/config.yaml>`,
:download:`n_config.yaml <../tutorials/2_brownfield_heat/n_config.yaml>`):

.. code-block:: yaml

   targets:
     driver: demand
     demand_H2: 580000   # MWh/y — large hydrogen demand drives a big electrolyser
     demand_CH4: 262000  # MWh/y biomethane
     demand_meoh: 0

   n_flags:
     meoh: false         # methanol synthesis switched off

   options:
     DH:
       enable: true
       price: 100        # €/MWh paid for heat delivered to DH
       peak capacity: 80 # MW DH system peak

The biogas, wind, solar and electrolysis units are all existing brownfield
assets. Only biogas upgrading, the heat pump and the TES are allowed to expand.

---

2 · The heat-network topology
------------------------------

The ``plot_heat_network.py`` script generates a topology diagram of the heat
subsystem, showing all buses with "Heat" or "DH" in their name and their
one-hop neighbourhood:

.. figure:: /_static/tutorials/tut2_heat_heat_subsystem.svg
   :width: 95%

   Heat-network topology. Electrolyser waste heat (low-temperature, LT) flows
   through the heat pump to reach district-heating temperature (DH bus), then
   to the DH grid or to the TES buffer. The biomass and NG boilers provide
   backup medium-temperature heat (Heat MT).

The DH demand profile is temperature-driven: it peaks in winter and drops to a
small sanitary-water base load in summer.

.. figure:: /_static/tutorials/tut2_heat_DH_load_profile.png
   :width: 95%

   DH demand time series (top) and load-duration curve (bottom). Peak demand
   ≈ 50 MW, annual supply ≈ 222 GWh/y.

---

3 · Interpret the results
-------------------------

.. figure:: /_static/tutorials/tut2_heat_Opt_capacities_SP_vs_WS.png
   :width: 95%

   Optimal investments. The electrolyser (100 MW, fixed EXI), wind (52 MW) and
   solar (30 MW) are brownfield. The model adds **biogas upgrading 29.9 MW**,
   **battery 51.1 MW / 102.3 MWh** and crucially **TES DH 27 MW / 405 MWh**
   (15 h of thermal storage at peak power).

.. admonition:: Key heat results
   :class: important

   - **TES DH: 27 MW / 405 MWh.** The thermal storage is sized to shift about
     half a day of DH supply. Its capacity factor is low (CF ≈ 0.05) — it acts
     as a **peak-shaving buffer** rather than a baseload asset, absorbing excess
     heat during high-renewable hours and releasing it when the heat pump is idle.
   - **Heat pump CF ≈ 0.72.** The heat pump runs intensively during cheap
     electricity hours, converting low-temperature electrolyser waste heat into
     district-heating supply. It is the key link between the electricity and heat
     networks.
   - **Battery 51.1 MW / 102.3 MWh.** A much larger battery than Tutorial 2
     (2.7 MW) because the 100 MW fixed electrolyser creates large and variable
     electricity demand spikes that the battery smooths against the renewable
     profile.
   - **District heating revenue ≈ €20.9 M/y** — the largest single revenue
     stream after biomethane, reflecting the high DH price (100 €/MWh) and the
     222 GWh/y annual supply.
   - The high H₂ shadow price (340 €/MWh at H₂ delivery) reflects the cost
     of a large fixed electrolysis investment amortised over the demand volume.

.. figure:: /_static/tutorials/tut2_heat_CF_operation_by_scenario.png
   :width: 95%

   Per-component utilization duration curves. The electrolyser runs near
   constantly (CF 0.97), the heat pump follows with CF 0.72, and the TES DH
   smooths the seasonal mismatch between heat production and demand.

.. figure:: /_static/tutorials/tut2_heat_shd_prices_mean_bar.png
   :width: 80%

   Energy-weighted mean shadow prices and annual throughput at internal carrier
   buses. The electricity bus (El3) clears at ≈ 140 €/MWh (high, reflecting
   the large fixed electrolysis CAPEX). Heat DH ≈ 64 €/MWh, Heat MT ≈ 66 €/MWh.
   CO₂ distribution has a negative shadow price (surplus CO₂ not fully absorbed
   by biomethanation).

.. figure:: /_static/tutorials/tut2_heat_TSC_by_carrier.png
   :width: 95%

   Total system cost by carrier. The dominant cost is the fixed electrolysis
   CAPEX + large H₂ demand. District-heating revenue (≈ €20.9 M/y) partially
   offsets the heat-side investments.

---

What you learned
----------------

- How electrolyser **waste heat** flows through a heat pump into a DH network.
- Sizing a **thermal energy storage (TES)** as a peak-shaving buffer — low CF
  but high value for decoupling heat production from the seasonal DH profile.
- How heat-side investments (pump, TES, DH connection) interact with electricity
  storage (battery) when the RE profile drives both.
- The impact of a high DH off-take price on overall plant economics.

.. seealso::

   :ref:`tutorial-2-brownfield` · :ref:`guide-outputs` · :ref:`config-economics`
