# GreenLab skive - greenbubble model

GLS greenbubble is a open model of the PtX section of the GreenLab Skive industrial hub, developped in [PyPSA](https://github.com/PyPSA/pypsa). This model allows for capacity and dispatch optimization of the GreenLab Skive Power-to-X site for driven by demands for H2 and Methanol and it is used in the in the paper "Optimizing hydrogen and e-methanol production through Power-to-X integration in biogas plants" https://doi.org/10.1016/j.enconman.2024.119175.

<img width="1184" alt="Screenshot 2025-02-19 at 12 31 24" src="https://github.com/user-attachments/assets/5f6ee063-35cb-4a9e-b6d0-26efd2ed2069" />



**Installation**
Clone this repository to your destination folder:

% git clone https://github.com/BertoGBG/GLS_greenbubble.git

Create the virtual environment from environment.yaml
We recommend using the package manager and environment management system conda to install python dependencies. Install [miniconda](https://docs.anaconda.com/miniconda/), which is a mini version of [anaconda](https://www.anaconda.com)
 that includes only conda and its dependencies or make sure conda is already installed on your system. For instructions for your operating system follow the conda [installation guide] (https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
To create the virtual environment for each platform:

Add the conda-forge channel and enable strict priority
.../greenbubble % conda config --add channels conda-forge
.../greenbubble % conda config --set channel_priority strict

Install conda-lock (once)
.../greenbubble % conda install -n base -c conda-forge conda-lock

Create environment for your OS:
.../greenbubble % conda-lock install -n greenbubble_gls --platform linux-64 envs/locks/conda-lock-linux-64.yml
.../greenbubble % conda-lock install -n greenbubble_gls --platform osx-64 envs/locks/conda-lock-osx-64.yml
.../greenbubble % conda-lock install -n greenbubble_gls --platform osx-arm64 envs/locks/conda-lock-osx-arm64.yml
.../greenbubble % conda-lock install -n greenbubble_gls --platform win-64 envs/locks/conda-lock-win-64.yml

Activate environment:
.../greenbubble % conda activate greenbubble_gls


**Reference**
please cite as: https://doi.org/10.1016/j.enconman.2024.119175


**What can the model do**
GreenBubble is an open model for optimization of industrial energy system based on agricultural setups. The model is based on PyPSA framework https://pypsa.readthedocs.io/en/stable/ and can simultaneously optimize the capacity of the plants in the industrial hub and they operation, over 1 year time horizon with time resolution up to 1h.
The capacity expansion can be green-field or brown-field with optimization is based on long-term economic equilibrium, and shadow prices for internal eergy and material flow are considered valid for the internal market.
The optimization also includes the internal hydrogen (inc. compression), CO2 (inc. comnpression), electricity and heat (3 temperature levels) networks .

**Technolgy and processes:**
In the current the only PtX products available are: hydrogen (for grid and/or internal use), methanol and biomethane. The Energy inputs are biomass (digestible and solid biomass) and renewable energy (onshore wind and solar).  
Electricity can be sold as a product, but the sales are constrained proportionally to the internal demand.

1) Hydrogen production:
   - Alkaline electrolysis
   
2) Methane production:
   - Biogas + upgrading
   - Biomethannation of biogas (with H2)
   - Biomethanation of CO2 (with H2)
   - Catalytic methanation of biogas (with H2)
   - Catalytic methanation of CO2 (with H2)
   
3) Methanol production:
   - CO2 hydrogenation 
   - eSMR with methanol synthesis (available soon)

4) Renewable electricity 
   - On-shore wind 
   - solar PV

5) Storage technologies:
   - Lithium-ion battieries
   - H2 in steel vessels
   - CO2 liquefaction and storage
   - CO2 pressurized in cylinders
   - Heat at water tanks (as for district heating)
   - Heat in concrete based Thermal Energy Storage
   
6) Biomass handling:
   - Biomass drying in hot air belt dryer
   - Dewatering of digestate fibers

**External markets (exogenous assumptions) :**
The optimization behaves as a price taker with respect to external markets, hence prices and availability of external resources are exogenously set.
these include: 
- CO2 tax on fossil emission
- Electricity prices and emission intensities + TSO and DSO tariffs
- Natural gas prices
- District heating Price 
- Biomass pellets
- Biomass chips
- Digestible biomass (manure) 


**Workflow:**

1) Configuration:
    in ../config/ are present three files for configuration
    a) config.yaml : main config file with all the optimization paramaters as demands for H2, MeOH, CH4 and which plants can be part of the solution (in n_flags).
    b) n_config.yaml : green/brown field config file. Default is all green field, but each technology can initialized with an existing capacity and expansion limited.
                        all constrains relative to a specific technolgy are set here (e.g. ramp up/down limits and min load)
    c) n_options.yaml : config for options relative to external markets. e.g. enable biomass purchase, sales of biochar credits etc...

2) Run the model:
   - from terminal run:  greenbubble_main.py  

3) preprocessing:
   Data packages for Skive are pre-downloaded in ../data/ . See ../scripts/paramaters.py for inputs to the pre-processing.
   - electricity spot prices
   - CO2 emission intensities
   - NG prices
   - Electricity demand profile (DK_1)
   - Capacity factors for wind and solar
   - NG demand profile (if used to geenrate an H2 demand profile)
   - DH demand profile in Skive
   - 
4) GLS specific data are retrive from the file: GreenLab_Input_file.xlxs

5) general database for techno-economic data of various technolgies: [technology-data](https://technology-data.readthedocs.io/en/latest/)

6) exceptions to the technology-data are set via ../scripts/technology_inputs.py in particoular: compressors and biomass drying are based on physics (semi-empirical for dryer) 

**Results of the single optimization**
The optimized network returns the optimal capacties for all the components in the model and their dispatch with one hour resolution and the shadow prices for each material and energy flows in the behihd-the-meter market.
Example results are stored within: ../outputs/single_analysis/
Each optimization run creates a folder based on n_flags and 'run name' set in config.yaml. Thsi folder contains two subfolders, /plot (for graphicals and table) and /networks for the pre- adn post- networks and the full configuration of each run 





  
