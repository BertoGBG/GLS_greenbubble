.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tech-meoh:

Methanol
========

Agent: ``n_flags.meoh``. Consumes hydrogen, CO₂ and biogas from the shared buses and delivers methanol to the product store. Requires ``electrolysis``, ``biogas`` and ``symbiosis``. The optional synthesis/distillation split is described in :doc:`/meoh_split`.

CO₂ hydrogenation to methanol (``methanolisation``): CO₂ + H₂ → MeOH + H₂O.
Modelled as a multi-input ``Link`` with ramp limits (8 %/h) and minimum
load (15 %).  Waste heat is recovered to the medium-temperature heat bus.

The water is not a side reaction that better catalysis could avoid: CO₂ carries
two oxygens and methanol contains one, so the spare oxygen must leave with
hydrogen.  Equivalently, the CO₂ route is the CO route plus reverse water-gas
shift, and RWGS both consumes 45 % of the CO exotherm and creates the water —
the modest reactor duty and the wet crude are the same phenomenon.

Optionally this link is **split** into ``methanol synthesis`` + ``methanol
distillation`` with a crude-methanol tank between them, so the reactor and the
column can run at different times.  Enable with ``options['meoh split']`` in
``n_config``; see :doc:`/meoh_split`.

**Methanol from biogas** (``methanol from biogas``) is an *alternative* to
``methanolisation``, not an addition to it: both produce into the same methanol
collection bus, and the optimiser picks. Instead of feeding separated CO₂ and
hydrogen to a synthesis reactor, raw biogas is reformed with oxygen in an
autothermal reformer and the resulting syngas is hydrogenated.

The carbon comes from the biogas itself, so the route needs **2.5x less
hydrogen** per MWh of methanol (0.449 MWh_H2/MWh_MeOH against 1.138) — but it
burns biomethane that could have been sold instead. Which one wins is exactly
the trade-off the model is set up to answer.

Costs and coefficients come from DEA sheet 97, "Methanol from biogas and
hydrogen", and are stored per MWh of hydrogen:

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Quantity
     - Value
     - Unit
   * - Methanol output
     - 2.229
     - MWh_MeOH / MWh_H2
   * - Biogas input
     - 1.6516
     - MWh_biogas / MWh_H2
   * - Electricity input
     - 0.1371
     - MWh_e / MWh_H2
   * - Heat input (MT)
     - 0.0617
     - MWh_th / MWh_H2
   * - Investment
     - 6423.8
     - EUR / kW_H2

Two streams are deliberately **not** wired. Oxygen: the reformer needs
0.1729 t_O2/MWh_H2 while the electrolysis feeding it co-produces 0.2381 t, so
oxygen never binds on a site with its own electrolyser and is treated as free.
Water output is left unwired for the same reason as in ``methanolisation``.

This route also has **no feed compressors at all**, unlike ``methanolisation``.
Two independent reasons agree.

*Physically*, there is nothing to compress at the battery limit. The pilot this
route is based on reforms biogas at ambient pressure and compresses the
**syngas** afterwards, once, with the hydrogen already mixed in [FRITSCH]_:

   "we designed a system for simple, atmospheric conversion of biogas to SynGas
   in a single-stage ATR-unit ... SynGas is cooled and compressed for subsequent
   introduction into the MeOH-synthesis loop"

The 20 bar quoted for that plant is the synthesis **loop** pressure, downstream
of both feeds — and it is a pilot-scale ceiling rather than an optimum, set by
the EU Pressure Equipment Directive on the basis of system gas volume.
Industrial methanol loops run at 50-100 bar.

*In the accounting*, DEA agrees: sheet 97 bundles "compressors prior to reformer
and methanol reactor" into its CAPEX and its electricity, while sheet 98
excludes feed compression (hence ``electricity-input-no-compression`` and the
separate compressor components on the other route). Both plants pay for
compression; only the bookkeeping differs. A GreenBubble compressor here would
charge it twice.

Because this plant creates no compressed bus of its own, it has no ``processes:``
entry in ``p_config`` — see :doc:`/guide_process_streams` for why that is the
correct form rather than an omission.

.. note::

   **Known simplification.** Hydrogen is drawn from the shared 30 bar header
   while the reference plant doses it at ambient, so this route pays for
   compression it does not physically need. Left in place deliberately:
   correcting it needs a low-pressure hydrogen tap, and it would only make the
   route cheaper.

.. note::

   DEA costed the **oxygen-fired autothermal (tri-reforming)** configuration,
   which is what is implemented. The electrically heated **eSMR** (bi-reforming)
   variant described in the same DEA chapter has no cost data and is not
   implemented.

.. [FRITSCH] C. Fritsch, J. Blankenstein, B. Bender, J. Dornseiffer, M. Haep and
   K. Ooms, *Design, implementation and piloting of an integrated hydrogen- and
   oxygen-added process for conversion of biogas to methanol*, Sustainable Energy
   & Fuels, 2025. https://doi.org/10.1039/d5se00691k
