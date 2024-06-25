# GreenLab skive - greenbubble model

GLS greenbubble is a open model of the PtX section of the GreenLab Skive industrial hub, developped in the PyPSA framework https://github.com/PyPSA/pypsa. This model allows for capacity and dispatch optimization of the GreenLab Skive Power-to-X site for driven by demands for H2 and Methanol and it is used in the in the paper "Optimizing hydrogen and e-methanol production through Power-to-X integration in biogas plants" https://arxiv.org/abs/2406.00442.

![PYPSA schematics_V4_colors](https://github.com/BertoGBG/GLS_greenbubble/assets/99412005/61a5d328-c28b-4b25-b129-1396315c3d0e)

**Structure of the repository:**
1) input data from open sources, located within the folder "data". references are available in the paper or in the end of the file "parameters"
2) input data for technology cost and performance parameters: "Technology-data-master". Which is a branch from https://github.com/PyPSA/technology-data.
3) file with all the input paramaters: "parameters_GL_paper_V3" 
4) all functions for the model formulaiton in the file: "functions_GL_paper"
5) main file for running a single optimization analysis: "main_GL_paper_V3"
   this file also includes plots for a single optimization analysis
6) main file for running sensitivity analysis (multiple optimization runs): "main_GL_paper_analysis"
7) file generating plots for sensitivity analysis: "plots_sensitivity_analysis_GL_paper_V3"
8) Folders with results: output/single_analysis and output/sensitivity_analysis
9) requirements to run this project in the file requirements.txt

**Indipendent variables for single optimization**
The following variable are set indipendendlty in a signle optimization run in the file "main_GL_paper_V3":


The _n_flags_ dictionary contains binary variables which allows for enabling part of the PtX hub in the solution: 
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
           
In the section _sensitivity analysis parameters_ other variables are specificed:
* Production of H2  for injection to the H2 grid: set at full load hours equivalent for a plant of 100MWe capacity
* Target production of MeOH: set as fraction of CO2 form biogas upgrading recovered tpo MeOH (msize of Biogas plant is fixed)
* Maximum renewable electricity (RE) which can be sold to the external grid: set as fraction of the electricity consumed by the PtX plants
* CO2 tax for fossil emissions

The exogenously set price at the interface between the PtX hub and the external energy systems are set for a specific year which is defined in the file "paramaters_GL_paper_V3":
* Year for exogenously set energy prices: 2019 (Low) or 2022 (high)

**Results of the single optimization**
The optimized network returns the optimal capacties for all the components in the model and their dispatch with one hour resolution and the shadow prices for each material and energy flows in the behihd-the-meter market.
Example results are stored within the output folder

**Indipendent variables for sensitivity analysis**
In the file main_GL_paper_analaysis the following variables are changed: 
* Production of H2
* Production of MeOH
* Max RE sold to the grid
* CO2 tax
* District heating
* biochar credits

The results for the optimization runs are not available in this repository due to size constraints, hence they must be generated locally. 


For visualization of model it is suggested to install the package pypsatopo 
from: https://github.com/ricnogfer/pypsatopo
via # pip install pypsatopo

  
