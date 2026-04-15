.. _workflow:

Workflow
========

GreenBubble is orchestrated by `Snakemake <https://snakemake.readthedocs.io>`_.
Each step is a rule defined under ``rules/`` and the full DAG is assembled by ``Snakefile``.

Running the workflow
--------------------

::

   snakemake -j4

``-j4`` allows up to 4 jobs in parallel (useful for multi-year preprocessing).
Use ``-j1`` for sequential execution.

To force re-execution of a specific step::

   snakemake -j1 --forcerun preprocess_inputs

DAG overview
------------

.. code-block:: text

   retrieve_tech_data        ← downloads technology cost CSV
         │
   preprocess_inputs         ← downloads & preprocesses market data (one job per year)
   (one per scenario year)
         │
   prepare_inputs            ← assembles all timeseries into a single inputs dict
         │
   build_network             ← constructs PyPSA network; adds stochastic scenarios
         │
   solve_network             ← runs capacity expansion + dispatch optimisation
         │
   plot_results              ← exports figures and result tables


Rules
-----

``rules/retrieve.smk``
   - **retrieve_tech_data** — fetches ``costs_{year_EU}.csv`` from the
     `technology-data <https://github.com/BertoGBG/technology-data>`_ repository.
   - **preprocess_inputs** — downloads electricity prices, CO₂ intensities, NG prices,
     renewable capacity factors, and district heating demand for each scenario year.
     Parameterised by ``{year}`` wildcard; runs in parallel when ``-j > 1``.

``rules/build.smk``
   - **prepare_costs** — builds the ``tech_costs`` DataFrame from the cost CSV.
   - **prepare_inputs** — loads all preprocessed CSV files and assembles the
     ``inputs_dict`` used by the network builder.
   - **build_network** — constructs the PyPSA network with all active technologies
     (controlled by ``n_flags``). Adds stochastic scenario links if
     ``stochastic.stochastic: true``.

``rules/solve.smk``
   - **solve_network** — runs the linear programme via Linopy. Solver and profile
     are set in :ref:`config-yaml` under ``optimization``.

``rules/plot.smk``
   - **plot_results** — exports dispatch plots, capacity tables, and shadow prices.

Stochastic mode
---------------

When ``stochastic.stochastic: true`` in :ref:`config-yaml`:

- ``PREPROCESS_YEARS`` is expanded to include all keys in ``stochastic.scenarios``.
- ``preprocess_inputs`` runs once per scenario year (in parallel).
- ``build_network`` couples the scenarios into a single stochastic LP.
- If ``stochastic.EVPI: true``, deterministic runs per scenario are also executed
  to compute the Expected Value of Perfect Information.

Outputs
-------

Results are written to ``outputs/single_analysis/{network_name}/``:

- ``networks/`` — pre- and post-optimisation PyPSA ``.nc`` files
- ``plots/`` — dispatch figures, capacity bar charts, shadow price tables
