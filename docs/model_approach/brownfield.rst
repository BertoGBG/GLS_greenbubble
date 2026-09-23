.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _greenfield-brownfield:

Greenfield and brownfield
=========================

The two words describe what the optimiser is allowed to assume already exists.

A **greenfield** optimisation starts from an empty site. Every capacity is a free
variable from zero, and the answer is the cost-optimal system as if nothing had
ever been built. It is the right question for a feasibility study, or for
comparing technologies on equal terms, because no incumbent gets a head start.

A **brownfield** optimisation starts from a site that is already there. Some
capacities are given, not chosen, and the optimiser decides what to add around
them. The answer is no longer "what is the best system" but "what is the best
next investment, given this" — and the two can differ sharply, because an
existing asset changes what is worth building next to it.

Why brownfield matters here
---------------------------

GreenBubble exists to study industrial clusters, and real clusters are almost
never empty. GreenLab Skive had a biogas plant before it had an electrolyser.
The practical question such a site asks is not what it would build from scratch,
but **what to expand or retrofit next** — and answering that requires the
existing plant to be in the model, competing on its real terms.

Two things follow, and both matter for reading results:

**An existing asset can be a whole plant or a single technology.** The
granularity is per ``n_config`` entry, so an existing biogas plant, an existing
boiler, or an existing compressor are all expressible, and a site can be
part-existing and part-greenfield in any combination. Nothing forces the whole
cluster into one mode.

**An existing asset carries only the finance still outstanding.** This is the
point that distinguishes a brownfield model from simply pinning a capacity. A
plant built fifteen years ago and fully paid off should compete on its operating
cost alone — its capital is sunk, and charging it again would make the model
prefer to demolish and rebuild. A plant half-way through its loan should carry
half. That fraction is ``remaining_investment_fraction``, and it is what lets an
old asset and a new one be compared honestly in the same objective.

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
