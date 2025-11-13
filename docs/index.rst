.. SPDX-FileCopyrightText: Contributors to GreenBubble
..
.. SPDX-License-Identifier: CC-BY-4.0

====================================================================================
GreenBubble: An open techno-economic optimisation model for industrial clusters
====================================================================================

GreenLab Skive – GreenBubble Model
----------------------------------

The **GreenBubble** model is an open techno-economic optimisation tool inspired by 
the **GreenLab Skive** industrial hub. It is developed using the 
`PyPSA <https://github.com/PyPSA/pypsa>`_ framework and performs capacity expansion 
and dispatch optimisation for a Power-to-X (PtX) industrial park driven by hydrogen 
and methanol demands.

The model is used in the paper:

*Optimizing hydrogen and e-methanol production through Power-to-X integration in 
biogas plants*  
DOI: https://doi.org/10.1016/j.enconman.2024.119175

.. image:: _static/front_image.png
   :alt: GreenBubble Diagram
   :width: 700px
   :align: center


Installation
------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/BertoGBG/GLS_greenbubble.git

The recommended way to install dependencies is via **conda**.  
Install `Miniconda <https://docs.anaconda.com/miniconda/>`_ or ensure conda is already
available on your system.  
For installation instructions, refer to the
`conda installation guide <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

**1. Add conda-forge channel and enable strict priority**

.. code-block:: bash

   conda config --add channels conda-forge
   conda config --set channel_priority strict

**2. Install conda-lock (only once)**

.. code-block:: bash

   conda install -n base -c conda-forge conda-lock

**3. Create the environment (choose your platform)**

.. code-block:: bash

   conda-lock install -n greenbubble_gls --platform linux-64 envs/locks/conda-lock-linux-64.yml
   conda-lock install -n greenbubble_gls --platform osx-64 envs/locks/conda-lock-osx-64.yml
   conda-lock install -n greenbubble_gls --platform osx-arm64 envs/locks/conda-lock-osx-arm64.yml
   conda-lock install -n greenbubble_gls --platform win-64 envs/locks/conda-lock-win-64.yml

**4. Activate the environment**

.. code-block:: bash

   conda activate greenbubble_gls


Reference
---------

If you use GreenBubble in academic work, please cite:

DOI: https://doi.org/10.1016/j.enconman.2024.119175


What the Model Can Do
---------------------

GreenBubble is an open modelling framework for **optimising industrial energy systems** 
based on agricultural and biomass-driven setups.

The model:

- uses the `PyPSA framework <https://pypsa.readthedocs.io/en/stable/>`_
- performs **simultaneous capacity expansion and operational dispatch**
- covers **1-year simulations** with hourly resolution
- supports both **greenfield** and **brownfield** optimisation
- includes long-term economic optimisation and internal market shadow prices
- models multi-energy networks (electricity, heat, hydrogen, CO₂)

Internal networks include:

- hydrogen (incl. compression)
- CO₂ (incl. compression & liquefaction)
- electricity
- heat (three temperature levels)


Technologies and Processes
--------------------------

**Hydrogen production**
- Alkaline electrolysis

**Methane production**
- Biogas upgrading
- Biomethanation of biogas (with H₂)
- Biomethanation of CO₂ (with H₂)
- Catalytic methanation of biogas
- Catalytic methanation of CO₂

**Methanol production**
- CO₂ hydrogenation
- eSMR + methanol synthesis *(coming soon)*

**Renewable electricity**
- Onshore wind
- Solar PV

**Storage technologies**
- Lithium-ion batteries
- H₂ in steel vessels
- CO₂ liquefaction and storage
- CO₂ in pressurised cylinders
- Hot water storage (district heating style)
- Concrete-based thermal energy storage

**Biomass handling**
- Hot air belt drying
- Dewatering of digestate fibres


External Markets (Exogenous Assumptions)
----------------------------------------

The optimisation behaves as a **price taker** in external markets.  
Key exogenous inputs include:

- CO₂ taxes on fossil emissions  
- Electricity prices, emission intensities, TSO/DSO tariffs  
- Natural gas prices  
- District heating prices  
- Biomass pellet & chip prices  
- Digestible biomass (manure)  


Workflow
--------

**1) Configuration**

Three configuration files are stored in ``../config/``:

a) ``config.yaml`` — main optimisation settings  
   - H₂, methanol, CH₄ demands  
   - which technologies may be included (``n_flags``)

b) ``n_config.yaml`` — greenfield/brownfield settings  
   - initial installed capacity  
   - expansion limits  
   - technical constraints (ramps, minimum loads, etc.)

c) ``n_options.yaml`` — external market settings  
   - biomass purchase  
   - biochar credit sales  
   - enabling/disabling technologies  

----

**2) Running the model**

From the terminal:

.. code-block:: bash

   python greenbubble_main.py

----

**3) Preprocessing**

Data for the Skive system is pre-downloaded in ``../data/``.  
Configured via ``../scripts/parameters.py``.

Includes:

- electricity spot prices  
- CO₂ emission intensities  
- natural gas (NG) prices  
- DK1 electricity demand  
- renewable capacity factors (wind, solar)  
- NG demand profile  
- district heating demand profile  

----

**4) GreenLab Skive specific data**

Retrieved from:

``GreenLab_Input_file.xlsx``

----

**5) Techno-economic database**

General database:  
`technology-data <https://technology-data.readthedocs.io/en/latest/>`_

----

**6) Technology exceptions**

Defined via ``../scripts/technology_inputs.py``:

- compressors  
- biomass drying (semi-empirical)  

Results of the Single Optimisation
----------------------------------

The optimisation returns:

- optimal capacities for all technologies  
- dispatch time series (hourly)  
- shadow prices for all material & energy carriers  

Example results appear in:

``../outputs/single_analysis/``

Each run creates a dedicated folder (name based on ``n_flags`` and config settings)  
containing:

- ``/plot`` — figures & tables  
- ``/networks`` — PyPSA networks (pre- and post-optimisation)  
- full configuration snapshots  

