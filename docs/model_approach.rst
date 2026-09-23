.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _model-approach:

Model Approach
==============

:doc:`design` covers what GreenBubble inherits from PyPSA. This section covers
the choices that make it GreenBubble: how the site is bounded, how it is divided
into agents, what connects them, and how far the physics is taken.

.. toctree::
   :maxdepth: 1

   model_approach/multi_energy
   model_approach/system_boundary
   model_approach/agents
   model_approach/grid_interface
   model_approach/constraints
   model_approach/brownfield
   model_approach/process_integration

----

In brief
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - What it answers
   * - :doc:`model_approach/multi_energy`
     - Which carriers exist, and why heat is three circuits rather than one.
   * - :doc:`model_approach/system_boundary`
     - What is inside the bubble, what it trades with, and why the interfaces
       carry no capital cost.
   * - :doc:`model_approach/agents`
     - The ``n_flags`` agents, the competing technologies inside each, and why
       ``symbiosis`` is what turns plants into a hub.
   * - :doc:`model_approach/grid_interface`
     - How import and export share one connection capacity and are charged once.
   * - :doc:`model_approach/constraints`
     - The four constraints GreenBubble adds on top of PyPSA, with what each one
       prevents.
   * - :doc:`model_approach/brownfield`
     - Greenfield and brownfield, set per technology, and how an existing asset
       is charged.
   * - :doc:`model_approach/process_integration`
     - How far the physics goes: declared stream states, the heat circuits, and
       the compressor duties computed before the solve.

If you would rather see the structure than read about it, :doc:`model_anatomy`
is the same material as a diagram you can operate.
