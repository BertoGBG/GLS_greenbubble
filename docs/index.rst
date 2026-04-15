.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

GreenBubble Documentation
==========================

**GreenBubble** is an open techno-economic optimisation model for Power-to-X industrial
clusters, built on the `PyPSA <https://pypsa.readthedocs.io>`_ framework.
It is inspired by the `GreenLab Skive <https://www.greenlab.dk>`_ industrial park in Denmark.

The model performs simultaneous capacity expansion and hourly dispatch optimisation across
multi-energy networks (electricity, hydrogen, CO₂, heat, biomethane, methanol) with
support for stochastic scenarios and Shapley value cost allocation.

.. image:: front_image.png
   :alt: GreenBubble network diagram
   :width: 100%
   :align: center

----

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   workflow

.. toctree::
   :maxdepth: 1
   :caption: Configuration

   config/config
   config/n_config
   config/n_options
   config/plots_config

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/scripts

----

If you use GreenBubble in academic work, please cite:

   *Optimizing hydrogen and e-methanol production through Power-to-X integration in
   biogas plants*, DOI: https://doi.org/10.1016/j.enconman.2024.119175
