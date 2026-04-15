.. _installation:

Installation
============

Requirements
------------

- Python 3.11
- conda or mamba
- A linear solver: **Gurobi** (recommended) or **HiGHS** (open-source, no licence needed)


Get the code
------------

Clone the repository::

   git clone https://github.com/BertoGBG/GLS_greenbubble.git
   cd GLS_greenbubble


Create the environment
----------------------

The recommended way to install all dependencies is via **conda-lock**, which provides
fully reproducible environments across platforms.

**1. Add conda-forge and enable strict priority** (once per machine)::

   conda config --add channels conda-forge
   conda config --set channel_priority strict

**2. Install conda-lock** (once)::

   conda install -n base -c conda-forge conda-lock

**3. Create the environment for your platform**::

   # macOS Apple Silicon
   conda-lock install -n greenbubble-pypsa107 --platform osx-arm64 envs/locks/conda-lock-osx-arm64.yml

   # macOS Intel
   conda-lock install -n greenbubble-pypsa107 --platform osx-64 envs/locks/conda-lock-osx-64.yml

   # Linux
   conda-lock install -n greenbubble-pypsa107 --platform linux-64 envs/locks/conda-lock-linux-64.yml

   # Windows
   conda-lock install -n greenbubble-pypsa107 --platform win-64 envs/locks/conda-lock-win-64.yml

**4. Activate**::

   conda activate greenbubble-pypsa107


Solver setup
------------

**Gurobi** (recommended for large problems)

   Gurobi requires a valid licence. Free academic licences are available at
   https://www.gurobi.com/academia/academic-program-and-licenses/.
   Once installed, set ``optimization.solver: 'gurobi'`` in ``config/config.yaml``.

**HiGHS** (open-source, no licence needed)

   HiGHS is included in the conda environment. Set ``optimization.solver: 'highs'``
   in ``config/config.yaml`` to use it. Suitable for smaller or exploratory runs.


Running the model
-----------------

**Option A — Snakemake workflow** (recommended)::

   # Preview the execution plan without running
   snakemake -n

   # Run the full workflow with 4 parallel jobs
   snakemake -j4

   # Force re-run of a specific rule
   snakemake -j1 --forcerun preprocess_inputs

See :ref:`rules` for a description of each step.

**Option B — standalone script**::

   python greenbubble_main.py

This runs the full pipeline in a single Python process (no Snakemake required).
Useful for quick tests or debugging individual steps.


Updating input data
-------------------

Preprocessed input data (electricity prices, capacity factors, etc.) is downloaded
automatically by Snakemake the first time you run the workflow.
To refresh the data for a specific year::

   snakemake -j1 --forcerun preprocess_inputs

To re-download all years (stochastic mode)::

   rm -rf data/Inputs_20*/
   snakemake -j4
