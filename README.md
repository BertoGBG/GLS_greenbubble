# GLS_greenbubble

This repository includes the model GLS_greenbubble used in the paper "Power-to-X integration in biogas plants: industrial symbiosis and cost optimal production of hydrogen and e-fuels" .

The model allows for capacity and dispatch optimization of the GreenLab Skive Power-to-X site based on demands for H2 and Methanol.

Other key paramaters are: CO2 tax, reference year for energy prices. The agents/plants allowed in the network can be selected and include:
1) Biogas plant
2) Electrolysis
3) Methanol plant
4) Centralized heating technologies
5) Symbiosys net
6) Distrct heating

**Structure of the repository:**

data: folder with input data
outputs: folder where results are stored. This has subfolders:
        - single_analysis:
        - sensitivty_analysis:
        
main_GL_paper_v3.py: main python script doing a single optimzation. Includes plots for single optimization
parameters_GL_papar_V3.py: file with all parameters inputs. Set here the folder for saving results
main_GL_paper_analysis.py: file with script for senstivity analysis over: CO2 tax, MeOH demand, H2 demand, netwrok strcture
plots_GL_paper_V3.py: script for plotting of the results of the sensitivity analysis

See the requirements to run this project in the file requirements.txt
