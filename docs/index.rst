.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

GreenBubble Documentation
==========================

**GreenBubble is an open-source techno-economic optimisation model for Power-to-X
industrial clusters.** These are co-located plants that share electricity,
hydrogen, CO₂, biomethane, methanol and heat infrastructure. The model optimises
capacity expansion and hourly dispatch together, over a full year at 1-hour
resolution, in a single linear programme.

.. image:: front_image.png
   :alt: GreenBubble network diagram
   :width: 100%
   :align: center

The model answers one question: what is the shared infrastructure worth? That is,
how much cheaper is it to build these plants together and let them trade directly
than to build each one alone?

Three design choices follow from that question. Each carrier is modelled
separately, so a by-product of one plant can be a feedstock of the next. Heat is
divided into three temperature circuits instead of one, so waste heat can only
serve demands it is hot enough to reach. And the connections between plants can
be switched off, which is how the model measures what they contribute.

The model is built on `PyPSA <https://docs.pypsa.org/latest/>`_. It was developed
around `GreenLab Skive <https://www.greenlab.dk>`_, an agricultural-industrial
park in Denmark that integrates biogas, electrolysis, methanation and methanol
synthesis. The methodology is described in:

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
     - :doc:`model_anatomy`. An interactive diagram of the whole structure.
       Switch parts of the site on and off to see what depends on what.
   * - **Understand how it works**
     - :doc:`design` covers what the model inherits from PyPSA.
       :doc:`model_approach` covers what is specific to GreenBubble: the system
       boundary, the agents, and process integration.
   * - **Run it**
     - :doc:`installation`, then :doc:`tutorial_1_greenfield`. One complete run,
       solved twice: once against a fixed demand, once against a product price.
   * - **Look something up**
     - :doc:`configuration` for every setting, :doc:`technologies` for every
       technology, :doc:`guide_outputs` for reading the results.

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
