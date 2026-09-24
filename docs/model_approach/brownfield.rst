.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _greenfield-brownfield:

Greenfield and brownfield
=========================

The two terms describe what the optimiser may assume already exists.

A **greenfield** optimisation starts from an empty site. Every capacity is a free
variable starting at zero, and the result is the cost-optimal system as if nothing
had been built before. This is the useful question for a feasibility study, or
when comparing technologies on equal terms, because no technology starts with an
advantage.

A **brownfield** optimisation starts from a site that already exists. Some
capacities are fixed rather than chosen, and the optimiser decides what to add
around them. The result answers a different question: not what the best system
would be, but what the best next investment is. The two answers can differ
considerably, because an existing asset changes what is worth building beside
it.

Why brownfield matters here
---------------------------

GreenBubble is used to study industrial clusters, and real clusters are rarely
empty. GreenLab Skive had a biogas plant before it had an electrolyser. Such a
site does not ask what it would build from scratch. It asks **what to expand or
retrofit next**, and answering that requires the existing plant to be in the
model on its real terms.

Two things follow, and both matter for reading results:

**An existing asset can be a whole plant or a single technology.** Existing
capacity is set per ``n_config`` entry, so an existing biogas plant, boiler or
compressor can each be represented. A site can be part existing and part
greenfield in any combination; nothing forces the whole cluster into one mode.

**An existing asset carries only the finance still outstanding.** This is what
distinguishes a brownfield model from simply fixing a capacity. A plant built
fifteen years ago and fully paid off should compete on its operating cost alone,
because its capital is already spent. Charging that capital again would make the
model prefer to demolish and rebuild. A plant half way through its loan should
carry half. The fraction still owed is ``remaining_investment_fraction``, and it
allows an old asset and a new one to be compared in the same objective.

Whether an asset already exists is therefore set **per technology** in
``n_config``, not globally. Three parameters control it:

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 42

   * - ``initial capacity``
     - ``expansion``
     - ``rif``
     - Meaning
   * - 0
     - ``true``
     - 0
     - Pure greenfield: build from nothing.
   * - > 0
     - ``false``
     - 0
     - Existing asset, fully depreciated. Sunk cost, no annual charge.
   * - > 0
     - ``false``
     - > 0
     - Existing asset still being paid for. Charged ``rif`` of its original
       investment.
   * - > 0
     - ``true``
     - any
     - Existing capacity plus the option to expand.

.. figure:: /_static/model/brownfield_timeline.svg
   :width: 100%
   :alt: A time axis showing construction year, the catalogue year the investment is read from, and the rif-scaled annual charge

   The investment cost is read at the asset's **own** construction year, then
   scaled by ``rif``. A new build of the same technology is a separate component
   charged at the investment year.

Existing capacity is added as a separate component with an ``EXI_`` prefix, so
it can be told apart from new build in every result. Its capital cost is looked
up at its own ``construction_year`` rather than at the investment year, which
matters for assets built when the technology cost was different. The economics
are in :doc:`/economics`.
