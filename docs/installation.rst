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

Each platform has its own environment file in ``envs/``. It pins PyPSA and the
solver interfaces and sets minimum versions for the rest.

**1. Add conda-forge and enable strict channel priority** (once per machine)::

   conda config --add channels conda-forge
   conda config --set channel_priority strict

**2. Create the environment from the file for your platform**::

   # macOS Apple Silicon
   conda env create -f envs/environment-osx-arm64.yaml

   # macOS Intel
   conda env create -f envs/environment-osx-64.yaml

   # Linux
   conda env create -f envs/environment-linux-64.yaml

   # Windows
   conda env create -f envs/environment-win-64.yaml

.. warning:: **For Windows users: enable long path support**

   Snakemake encodes output file paths as long filenames in its metadata directory. Windows
   enforces a 260-character path limit by default, which can cause ``[WinError 3]`` errors
   during workflow execution. In that case you must enable long path support before running.

   **Option A — PowerShell (run as Administrator):**

   .. code-block:: powershell

      Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
          -Name "LongPathsEnabled" -Value 1

   **Option B — Registry editor:**

   Navigate to ``HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem``
   and set ``LongPathsEnabled`` to ``1``.

   Restart your machine after applying either option. Alternatively, store the
   metadata in a single database file (see :ref:`installation-troubleshooting`).

**3. Activate**::

   conda activate greenbubble-pypsa107


Solver setup
------------

**Gurobi** (recommended for large problems)

   Gurobi requires a valid licence. Free academic licences are available at
   https://www.gurobi.com/academia/academic-program-and-licenses/.
   Once installed, set ``optimization.solver: 'gurobi'`` in ``config/config.yaml`` (see :ref:`guide-snakemake`).

**HiGHS** (open-source, no licence needed)

   HiGHS is included in the conda environment and is the default solver
   (``optimization.solver: 'highs'``). The tutorials use it and each solves in
   minutes. It is suited to smaller or exploratory runs; full-year hourly runs
   can take hours.


Running the model
-----------------

Run with `Snakemake <https://snakemake.readthedocs.io/en/stable/>`_::

   # Preview the execution plan without running
   snakemake -n

   # Run the full workflow with 4 parallel jobs
   snakemake -j4

   # Force re-run of a specific rule
   snakemake -j1 --forcerun preprocess_inputs

See :ref:`rules` for a description of each step.


Updating input data
-------------------

Preprocessed input data (electricity prices, capacity factors, etc.) is downloaded
automatically by Snakemake the first time you run the workflow.
To refresh the data for a specific year::

   snakemake -j1 --forcerun preprocess_inputs

To re-download all years (stochastic mode)::

   rm -rf data/Inputs_20*/
   snakemake -j4


.. _installation-troubleshooting:

Troubleshooting
---------------

Snakemake reads the workflow profile ``profiles/default/config.yaml`` on every
run. It keeps Snakemake's defaults so that the workflow behaves the same on
every platform. The two cases below need extra settings on some machines. Put
them in a **personal profile** outside the repository, so they never reach git.

Create the folder ``~/.config/snakemake/greenbubble/`` with a ``config.yaml``
holding the settings you need. Then point Snakemake at it in your shell start-up
file (``~/.zshrc`` or ``~/.bashrc``)::

   export SNAKEMAKE_PROFILE=~/.config/snakemake/greenbubble

Snakemake then applies both profiles. The run log starts with
``Using profiles ... and workflow specific profile profiles/default``.

**"Bad CPU type in executable" on Apple Silicon (M1-M5 Macs)**

Every real run stops at ``Select jobs to execute...`` with an error ending in
``pulp/solverdir/cbc/osx/i64/cbc``. A dry run (``snakemake -n``) works, because
it does not schedule jobs.

Snakemake chooses which jobs to run in parallel by solving a small MILP through
PuLP. PuLP's bundled CBC binary is built for Intel Macs. Without Rosetta it
cannot run on Apple Silicon. Use HiGHS for the scheduler instead. It is already
installed with the environment:

.. code-block:: yaml

   # ~/.config/snakemake/greenbubble/config.yaml
   scheduler-ilp-solver: HiGHS

Alternatively, install Rosetta (``softwareupdate --install-rosetta``) or pass
``--scheduler greedy`` on the command line.

**The repository is inside OneDrive, Dropbox or another synced folder**

Snakemake stores one metadata file per output in ``.snakemake/metadata/``. Each
file is named after the base64-encoded output path. GreenBubble's output names
are long (see :ref:`wildcards`), so these names can exceed the path limit of the
sync client, and syncing stops.

Store the metadata in a single SQLite database instead:

.. code-block:: yaml

   # ~/.config/snakemake/greenbubble/config.yaml
   persistence-backend: db
   persistence-backend-db-url: sqlite:///.snakemake/metadata.db

Snakemake still reruns a rule when its code, params or inputs change. Avoid this
setting on cluster filesystems such as NFS, where SQLite file locking is
unreliable. As a last resort, ``drop-metadata: true`` stops writing metadata
altogether. Reruns are then decided by file times only, so changes to code or
params need ``--forcerun``.
