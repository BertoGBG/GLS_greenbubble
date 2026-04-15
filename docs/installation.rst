.. _installation:

Installation
============

Clone the repository::

   git clone https://github.com/BertoGBG/GLS_greenbubble.git
   cd GLS_greenbubble

Create the conda environment (choose your platform)::

   conda-lock install -n greenbubble-pypsa107 --platform osx-arm64 envs/locks/conda-lock-osx-arm64.yml
   conda-lock install -n greenbubble-pypsa107 --platform linux-64  envs/locks/conda-lock-linux-64.yml

Activate::

   conda activate greenbubble-pypsa107

Solver
------

The model requires either **Gurobi** (recommended) or **HiGHS** (open-source).
Set ``optimization.solver`` in :ref:`config-yaml` accordingly.

For Gurobi, a valid licence must be available (academic licences are free).
HiGHS requires no licence and can be selected with ``solver: 'highs'``.
