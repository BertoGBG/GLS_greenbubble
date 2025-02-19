# GreenLab skive - greenbubble model

GLS greenbubble is a open model of the PtX section of the GreenLab Skive industrial hub, developped in [PyPSA](https://github.com/PyPSA/pypsa). This model allows for capacity and dispatch optimization of the GreenLab Skive Power-to-X site for driven by demands for H2 and Methanol and it is used in the in the paper "Optimizing hydrogen and e-methanol production through Power-to-X integration in biogas plants" https://doi.org/10.1016/j.enconman.2024.119175.

<img width="1184" alt="Screenshot 2025-02-19 at 12 31 24" src="https://github.com/user-attachments/assets/5f6ee063-35cb-4a9e-b6d0-26efd2ed2069" />



**Installation**
Clone this repository to your destination folder:

% git clone https://github.com/BertoGBG/GLS_greenbubble.git

Create the virtual environment from environment.yaml
We recommend using the package manager and environment management system conda to install python dependencies. Install [miniconda](https://docs.anaconda.com/miniconda/), which is a mini version of [anaconda](https://www.anaconda.com)
 that includes only conda and its dependencies or make sure conda is already installed on your system. For instructions for your operating system follow the conda [installation guide] (https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
To create the virtual environment: 

.../greenbubble % conda env create -f environment.yaml

.../greenbubble % conda activate technology-data



**Reference**
please cite as: https://doi.org/10.1016/j.enconman.2024.119175



**What can the model do**
The model calcualtes the optimal capacities and operation of the plants and components forming the PtX hub. The modle optimizes all teh content of the "GreenBubble" in the figure, while the external energy systems is modelled as exogenously set inputs. Production cost for H2 and MeOH are estimated as well as prices for internally traded commodities.


**Structure of the Repository:**
The three main files to run a single optimization are:
- main.py
- functions.py
- paramaters.py

The other files were used inthe paper to produce results of various sensitivity analysis.

_main.py_ (not executable from terminal) : 
   - set the main paramaters in the analysis
   - sets the pypsa netowrk configuration options
   - execute the workflow (load input data, create pypsa network, optimize network, plot basic results)

_paramaters.py_ : contains all the other paramaters and assumption used in the model. 
   - pre-processign paramaters:

_functions.py_ : contains all functions developped for this model (and extra plotting functions) 



**Workflow:**
1) Set main analysis paramaters (in main.py):
   - annual H2 demand 
   - annual MeOH demand (% of max demand compatible with biogas production)
   - CO2 tax for fossil emission
   - max electricity sellable to external grids (as % of consumed in the production of RFNBOs)
   - En_price_year (set in paramaters.py) select the historical year the for surrounding energy system and RE

2) preprocessing of input data (if preprocess_flag = True , in main.py ):
   download  data from varous sources (see paramaters.py) to generate exogenously set the variables in the model:
   - electricity spot prices
   - CO2 emission intensities
   - NG prices
   - Electricity demand profile (DK_1)
   - Capacity factors for wind and solar
   - NG demand profile (if used to geenrate an H2 demand profile)
   - H2 and MeOH demand (set from main.py)
   - DH demand profile in Skive
     
   All data are saved as .csv in /data once downloaded.  Pre-download input data for years 2019-2024 are provided and do not need to be downloaded again. To download other input data enter your [Renewable](https://www.renewables.ninja) and [entsoe]                    (https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html) API tokens in
paramaters.py and set  preprocess_flag = True in main.py 

3) create PyPSA network
   The _n_flags_ dictionary contains binary variables which allows plants in to be part of the PtX hub  solution: 
   example:
   n_flags = {'SkiveBiogas': True,         # VARIABLE: Biomethane plant                    
              'central_heat': True,        # VARIABLE:Heat generation  (including pyrolysis for bichar)     
              'renewables': True,          # VARIABLE:Onshore wind and solar      
              'electrolyzer': True,        # VARIABLE:H2 production 
              'meoh': True,                # VARIABLE:MeOH synthesys
              'symbiosis_net': True,       # VARIABLE:Trading of energy and material flows withinthe park
              'DH': True,                  # VARIABLE:connection to district heating network 
              'bioChar' : True,            # VARIABLE: if biochar credits have value or not
              'print': True,               # OPTION: saves svg of network before optimization
              'export': False}             # OPTION: saves network before optimization

4) retrive techno-economic
   GLS specific data are retrive from the file: GreenLab_Input_file.xlxs
   general database for techno-economic data of varous technolgies: [technology-data](https://technology-data.readthedocs.io/en/latest/)
 
5) solve network
   - build linopy model
   - solve network
     we sugget to use [gurobi](gurobi.com) as a solver and we have successfully tested the free solver [highspy](https://pypi.org/project/highspy/)

**Results of the single optimization**
The optimized network returns the optimal capacties for all the components in the model and their dispatch with one hour resolution and the shadow prices for each material and energy flows in the behihd-the-meter market.
Example results are stored within the output folder


**Analysis Options**
In addition to the aforementioned configuration options, several other parameters can be modfied within in paramaters.py
examples are: 
- discout rate (var: discout_rate) : for discoutning of investment cost sin the model. default = 7%
- H2 and MeOH demannd (frequency of delivery): can me set with difference frequency on the delivery: weekly, monthly, and annual (base case). default n=1 (delivery at the end of the year)
- H2 demand (annual profile): for H2 with weekly or monthly delivery, the amount for each delivery can follow the profile of the NG demand in DK (set H2_profile_flag = True). default = 'False'
- CO2 tax on the reference year prices (var: CO2_cost_ref_year): it is automatically applied to energy prices for reference energy year. default = 0
- prices for other commodities traded with external energy systems: DH, pellets (note 'biomass' refers to biogas feedstock)
- taxes on electricity (TSO an DSO): default values from DK! and Skive in 2022
- techno-economic inputs for technolgies not presents in technology-data





  
