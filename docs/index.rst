.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

GreenBubble Documentation
==========================

**GreenBubble is an open-source techno-economic optimisation model for Power-to-X
industrial clusters** — co-located plants that share electricity, hydrogen, CO₂,
biomethane, methanol and heat infrastructure. It co-optimises **capacity
expansion** and **hourly dispatch** together, across a full year at 1-hour
resolution, in a single linear programme.

.. image:: front_image.png
   :alt: GreenBubble network diagram
   :width: 100%
   :align: center

The question it exists to answer is what the shared infrastructure is worth: how
much cheaper it is to build these plants next to each other and let them trade
directly than to build each one alone. Everything in the model follows from
that — the carriers are modelled separately so a by-product of one plant can be
a feedstock of the next, heat is three temperature circuits rather than one, and
the connections between plants can be switched off to measure what they were
contributing.

The model is built on `PyPSA <https://docs.pypsa.org/latest/>`_, and was
developed around `GreenLab Skive <https://www.greenlab.dk>`_, an
agricultural-industrial park in Denmark integrating biogas, electrolysis,
methanation and methanol synthesis. The methodology is described in:

   *Optimizing hydrogen and e-methanol production through Power-to-X integration
   in biogas plants*, Energy Conversion and Management, 2024.
   `doi:10.1016/j.enconman.2024.119175 <https://doi.org/10.1016/j.enconman.2024.119175>`_

Where to start
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - If you want to…
     - Go to
   * - **See what the model is**
     - :doc:`model_anatomy` — an interactive diagram of the whole structure.
       Switch parts of the site on and off and watch what depends on what. The
       quickest way to understand the model without reading anything.
   * - **Understand how it works**
     - :doc:`design` for what it inherits from PyPSA, then
       :doc:`model_approach` for the parts that are specific to GreenBubble —
       the system boundary, the agents, process integration.
   * - **Run it**
     - :doc:`installation`, then :doc:`tutorial_1_greenfield` — one complete
       run, solved twice: once against a fixed demand, once against a product
       price.
   * - **Look something up**
     - :doc:`configuration` for every setting, :doc:`technologies` for every
       technology, and :doc:`guide_outputs` for reading the results.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   introduction
   installation
   guide_snakemake

.. toctree::
   :maxdepth: 1
   :caption: Model Description

   design
   model_approach
   model_anatomy
   technologies
   economics
   methods

.. toctree::
   :maxdepth: 1
   :caption: Configuration Reference

   configuration
   wildcards

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial_1_greenfield
   tutorial_2_brownfield
   tutorial_2b_brownfield_heat
   tutorial_3_rolling_horizon
   tutorial_4_stochastic

.. toctree::
   :maxdepth: 1
   :caption: How-to Guides

   guide_demands
   guide_process_streams
   guide_stochastic
   guide_temporal_resolution
   guide_rolling_horizon
   guide_outputs
   guide_economic_analysis
   examples

.. toctree::
   :maxdepth: 1
   :caption: Developer Reference

   workflow
   rules
   guide_new_technology
   implementation

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: References

   references
