.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _guide-temporal-resolution:

Temporal Resolution
===================

By default GreenBubble runs at **1-hour resolution** over a full year
(8 760 snapshots).  For exploratory runs, sensitivity studies, or when using
the open-source HiGHS solver, reducing the time resolution can cut solve times
significantly with acceptable accuracy trade-offs.

---

How it works
------------

When ``clustering.temporal.resolution`` is set to a pandas offset string
(e.g. ``"4h"``), :func:`scripts.helpers.resample_network` is called at the end
of :func:`scripts.prepare_network.build_network` before the network is written
to disk.  All time-varying data are resampled in one step:

- **Prices, capacity factors, demands** → arithmetic mean over the interval.
- **Store bounds** (``e_max_pu``, ``e_min_pu``) → ``min`` / ``max`` over the
  interval (conservative: the tightest constraint within the period is kept).
- **Snapshot weightings** → sum over the interval, so multiplying by weights
  still gives correct annual energy totals.

PyPSA automatically scales investment and operational costs by snapshot
weightings, so the LP objective remains consistent regardless of resolution.

---

Quick start
-----------

Add to ``config/config.yaml``::

   clustering:
     temporal:
       resolution: "4h"

Then run as normal::

   snakemake -n          # verify 2 190 snapshots appear in the plan
   snakemake --cores 4

Outputs land in the same folder as a full-resolution run — the folder name does
not encode the resolution, so use a distinct ``run_name`` if you want to keep
both results::

   clustering:
     temporal:
       resolution: "4h"
   run_name: "4h_test"

---

Choosing a resolution
---------------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - resolution
     - snapshots
     - guidance
   * - ``false``
     - 8 760
     - Full accuracy.  Required for final results and publication runs.
   * - ``"4h"``
     - 2 190
     - Good balance — captures most diurnal variability; ~4× faster.
       Recommended for iterative scenario development.
   * - ``"8h"``
     - 1 095
     - Suitable for large parameter sweeps.  Some peak-shaving behaviour
       is smoothed out.
   * - ``"24h"``
     - 365
     - Fast screening only.  Daily averaging loses all intra-day dynamics;
       storage dispatch results are not reliable at this resolution.

.. note::
   Sub-hourly strings (e.g. ``"30min"``) raise ``NotImplementedError``.
   Upsampling requires interpolating input data that is not yet implemented.

---

Limitations
-----------

**Stochastic mode**

Temporal resampling and stochastic optimisation can be used together, but a
``UserWarning`` is issued at runtime.  Each scenario year is resampled
independently, so correlations between scenarios at the sub-period level are
lost.  For stochastic runs where scenario co-movement matters, use full
1-hour resolution.

**Rolling horizon mode**

When ``rolling_horizon.enabled: true``, the temporal resolution setting is
ignored (a ``UserWarning`` is issued).  The RH solver loads a pre-existing
fixed-capacity network via ``network_path`` and slides windows over its full
hourly time series internally — resampling at the build stage would have no
effect.

A valid combined workflow is:

1. Run capacity expansion with ``resolution: "4h"`` to obtain approximate
   optimal capacities quickly.
2. Save the resulting OPT network.
3. Point ``rolling_horizon.network_path`` at that network and run RH at full
   1-hour resolution for detailed dispatch analysis.

---

Future work
-----------

Time-series aggregation via `tsam <https://tsam.readthedocs.io/>`_ (typical
periods or contiguous segments) is planned as a second phase.  tsam can
preserve chronological order (required for cyclic store constraints) through
contiguous segmentation, and typically gives better accuracy than uniform
resampling at the same number of segments.  When implemented it will be
available through the same ``clustering.temporal`` config block.

See also
--------

- :ref:`config-clustering` — full config reference
- :ref:`guide-rolling-horizon` — dispatch-only runs on a fixed network
- :func:`scripts.helpers.resample_network` — API reference
