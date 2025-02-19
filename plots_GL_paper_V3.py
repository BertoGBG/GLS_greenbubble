''' Plots function for GL papers - sensitivity analysis '''

from functions_GL_paper_V3 import *

GL_inputs, GL_eff, Elspotprices, CO2_emiss_El, bioCH4_prod, CF_wind, CF_solar, NG_price_year, Methanol_demand_max, NG_demand_DK, El_demand_DK1, DH_external_demand = load_input_data()

'''Full list of sensitivity analysis parameters'''
# CO2_cost_list = [0, 150, 250]  # €/t (CO2_cost)
# H2_demand_list = [0, 4000]  # flh at 100 MW (flh_H2)
MeOH_rec_list = [0.7, 0.8, 0.85, 0.90, 0.95]  # % of CO2 from (f_max_MeOH_y_demand)
# DH_flag_list = [True, False]  # true false
# En_year_price = [2019, 2022]

# folder with saved optimized networks
data_folder_Opt = 'outputs/sensitivity_analysis/Results_OptNetworkV2/'

# DECIDE WAT TO IMPORT : Select the datatset space by indicating all the indipended variables accepted
dataset_flags = {
            'CO2_cost': [0, 150, 250],                  # input variable
            'fC_MeOH': [0.8, 0.85, 0.9, 0.95, 0.99],    # input variable
            'd_H2': [0, 4000],                          # input variable
            'En_year_price': [2019, 2022],              # input variable
            'DH': [True]                       # output variable
}

data_folder = 'outputs/sensitivity_analysis/Results_OptNetworkV2/'  # folder with all results
name_files= []
for f in os.listdir(data_folder):
    if f.endswith('.nc'):
        # check if CO2 cost in the file is among the selected ones
        m = re.search('CO2c(\d+)', f)
        m_co2 = int(m.group(1))

        # check if H2 demand in the file is among the selected ones
        m = re.search('H2d(\d+)', f)
        m_h2 = int(m.group(1)) * 1000 / p.H2_output

        # check if fC_MeOH in the file is among the selected ones
        m = re.search('MeOHd(\d+)', f)
        MeOH_y_d = int(m.group(1)) * 1000  # demand in GWh/y
        bioCH4_y_d = bioCH4_prod.values.sum()
        CO2_MeOH_plant = 1 / GL_eff.at['Methanol', 'Methanol plant']  # bus0 = CO2, bus1 = Methanol
        bioCH4_CO2plant = GL_eff.at['bioCH4', 'SkiveBiogas'] / GL_eff.at['CO2 pure', 'SkiveBiogas']
        fC_MeOH_value = round((MeOH_y_d * CO2_MeOH_plant) * bioCH4_CO2plant / bioCH4_y_d, 2)
        fC_MeOH = min(MeOH_rec_list, key=lambda x: abs(x - fC_MeOH_value))

        # check if DH in the file is among the selected ones
        cond_DH = np.sign(f.find('DH')+1) in dataset_flags['DH']

        # check if En_year in the file is among the selected ones
        y_list= [str(y) for y in  dataset_flags['En_year_price']]
        cond_En_y = any(ele in f for ele in y_list)

        if (m_co2 in dataset_flags['CO2_cost']) and (m_h2 in dataset_flags['d_H2']) and (fC_MeOH in dataset_flags['fC_MeOH']) and cond_DH and cond_En_y:
            name_files.append(f)

# ref network for agents
single_analysis_folder = 'outputs/single_analysis/'
with open(single_analysis_folder+'network_comp_allocation.pkl', 'rb') as f:
    network_comp_allocation = pkl.load(f)
agent_list=[key for key in network_comp_allocation]

# DECIDE WHAT TO PLOT :Set True to the variables in the results
results_flags = {
            'CO2_cost': True,           # input variable -> for categorization
            'fC_MeOH': True,            # input variable -> for categorization
            'd_H2': True,               # input variable -> for categorization
            'En_year_price': True,      # input variable -> for categorization
            'DH': True,                 # input variable -> for categorization
            'DH_y': True,               # output variable
            'mu_H2': True,              # output variable
            'mu_MeOH': True,            # output variable
            'mu_el_GLS': True,          # output variable
            'mu_heat_MT': True,         # output variable
            'mu_heat_DH':True,          # output variable
            'mu_heat_LT': True,         # output variable
            'mu_CO2': True,             # output variable
            'tot_sys_cost': True,       # output variable
            'tot_cap_cost' : True,      # output variable
            'tot_mar_cost' : True,      # output variable
}
# add agents list to the list of resutls
agent_dict1=dict((ag +'_cc',True) for ag in agent_list)
agent_dict2=dict((ag +'_mc',True) for ag in agent_list)
results_flags.update(agent_dict1)
results_flags.update(agent_dict2)

results_units = {
            'CO2_cost': '(€/t)',            # input variable -> for categorization
            'fC_MeOH': '(C_MeOH/C_sep)',    # input variable -> for categorization
            'd_H2': '(GWh/y)',              # input variable -> for categorization
            'En_year_price': '',            # input variable -> for categorization
            'DH': '(-)',                    # input variable -> for categorization
            'DH_y': '(GWh/y)',              # output variable
            'mu_H2': '(€/kg)',              # output variable
            'mu_MeOH': '(€/kg)',            # output variable
            'mu_el_GLS': '(€/MWh)',         # output variable
            'mu_heat_MT': '(€/MWh)',        # output variable
            'mu_heat_DH': '(€/MWh)',        # output variable
            'mu_heat_LT': '(€/MWh)',        # output variable
            'mu_CO2': '(€/t)',              # output variable
            'tot_sys_cost': '(€/y)',        # output variable
            'tot_cap_cost': '(€/y)',        # output variable
            'tot_mar_cost': '(€/y)',        # output variable
}
agent_dict3=dict((ag+'_cc','(€/y)') for ag in agent_list)
agent_dict4=dict((ag+'_mc','(€/y)') for ag in agent_list)
results_units.update(agent_dict3)
results_units.update(agent_dict4)

# define Results Data Frame
results_columns=[]
for key in results_flags:
    if results_flags[key]:
        results_columns.append(key)

df_results= pd.DataFrame(0, index=name_files, columns=results_columns)

# load all results
for name in name_files:
    # import network
    n_name= 'n_'+'name' # network name
    n_name = pypsa.Network(os.path.join(data_folder, name))

    # Retrive info from Network or network name
    # Independent variables
    m = re.search('CO2c(\d+)', name)
    df_results.at[name, 'CO2_cost'] = int(m.group(1))

    # Retrive Fraction of CO2 from biogas plant recovered to MeOH
    MeOH_y_d = n_name.loads_t.p_set['Methanol'].sum()
    bioCH4_y_d = n_name.loads_t.p_set['bioCH4'].sum()
    CO2_MeOH_plant = 1/n_name.links.efficiency['Methanol plant'] # bus0 = CO2, bus1 = Methanol
    bioCH4_CO2plant = n_name.links.efficiency['SkiveBiogas'] / n_name.links.efficiency2['SkiveBiogas'] # bus0 = biomass, bus1= bioCH4, bus2=CO2
    fC_MeOH = round((MeOH_y_d * CO2_MeOH_plant) * bioCH4_CO2plant / bioCH4_y_d, 2)
    df_results.at[name,'fC_MeOH']=fC_MeOH

    m = re.search('H2d(\d+)', name)
    df_results.at[name,'d_H2']= int(m.group(1))

    if '2019' in name:
        df_results.at[name,'En_year_price'] = 2019
    elif '2022' in name:
        df_results.at[name, 'En_year_price'] = 2022


    if 'DH' in name:
        df_results.at[name, 'DH'] = 1
    else:
        df_results.at[name, 'DH'] = 0

    # Output variables
    # DH y production
    if 'DH' in name:
        df_results.at[name, 'DH_y'] = int(n_name.links_t.p0['DH GL_to_DH grid'].sum() // 1000)
    else:
        df_results.at[name, 'DH_y'] = 0

    # H2 prod cost
    m = re.search('H2d(\d+)', name)
    d_H2 = int(m.group(1))
    if d_H2 ==0:
        df_results.at[name, 'mu_H2'] = np.mean(n_name.buses_t.marginal_price['H2_meoh']) * p.lhv_h2 /1000
    else:
        df_results.at[name, 'mu_H2'] = np.mean(n_name.buses_t.marginal_price['H2 delivery']) * p.lhv_h2 /1000

    # MeOH prod cost
    if results_flags['mu_MeOH']:
        df_results.at[name, 'mu_MeOH'] = np.mean(n_name.buses_t.marginal_price['Methanol']) * p.lhv_meoh

    # El GL cost
    el_GL_bus= 'El_meoh'
    df_results.at[name, 'mu_el_GLS'] = np.mean(n_name.buses_t.marginal_price['El_meoh'])

    # Heat MT GL cost
    if 'Heat MT' in n_name.buses.index.values:
        df_results.at[name, 'mu_heat_MT'] = np.mean(n_name.buses_t.marginal_price['Heat MT'])
    else:
        df_results.at[name, 'mu_heat_MT'] = 0

    # Heat DH GL cost
    if 'Heat DH' in n_name.buses.index.values:
        df_results.at[name, 'mu_heat_DH'] = np.mean(n_name.buses_t.marginal_price['Heat DH'])
    else:
        df_results.at[name, 'mu_heat_DH'] = 0

    # Heat LT GL cost
    if 'Heat DH' in n_name.buses.index.values:
        df_results.at[name, 'mu_heat_LT'] = np.mean(n_name.buses_t.marginal_price['Heat LT'])
    else:
        df_results.at[name, 'mu_heat_LT'] = 0

    # CO2 price GL sold by Biogas plant (LP)
    df_results.at[name, 'mu_CO2'] = np.mean(n_name.buses_t.marginal_price['CO2 sep'])

    # total system , capital and marginal cost
    tot_cc, tot_mc, tot_sc = get_system_cost(n_name)
    df_results.at[name, 'tot_sys_cost'] = np.sum(tot_sc)
    df_results.at[name, 'tot_cap_cost'] = np.sum(tot_cc)
    df_results.at[name, 'tot_mar_cost'] = np.sum(tot_mc)

    # total capital and marginal costs, by agents
    cc_tot_agent, mc_tot_agent = get_total_marginal_capital_cost_agents(n_name, network_comp_allocation, False)
    for key in cc_tot_agent:
        df_results.at[name, key + '_cc'] = cc_tot_agent[key]
        df_results.at[name, key + '_mc'] = mc_tot_agent[key]


''' PLOTS SECTION'''
folder_plot= 'outputs/sensitivity_analysis/plots'

# plot Pair Plot - shadow prices
category1= 'd_H2'
vars_list= ['mu_H2', 'mu_MeOH', 'fC_MeOH', 'mu_el_GLS', 'mu_heat_MT', 'tot_sys_cost' ]
g= sns.pairplot(df_results, hue=category1, diag_kind="hist", vars= vars_list, palette='muted', corner=True)
size_v= df_results['CO2_cost']
g.map_offdiag(sns.scatterplot, size=size_v)
g.fig.suptitle('shadow prices (€/unit)')
g.savefig(folder_plot + 'pailplot_shadow_prices.png')

# plot Pair Plot - Fixed costs by agent

category1= 'd_H2'
agent_list1 = list(set(agent_list) - {'external_grids', 'SkiveBiogas'})
vars_list=[key + '_cc' for key in agent_list1]
g= sns.pairplot(df_results/1e06, hue=category1, diag_kind="hist", vars= vars_list, palette='muted', corner= True)
size_v= df_results['CO2_cost']
g.map_offdiag(sns.scatterplot, size=size_v)
g.fig.suptitle('capital costs per agent (M€/y)')
g.savefig(folder_plot + 'pairplot_agents_cc.png')


# Electrolyzer cost section
# DECIDE WAT TO IMPORT : Select the datatset space by indicating all the indipended variables accepted
dataset_flags = {
            'CO2_cost': [0, 150, 250],                  # input variable
            'fC_MeOH': [0.8, 0.85, 0.9, 0.95, 0.99],    # input variable
            'd_H2': [0, 4000],                          # input variable
            'En_year_price': [2019, 2022],              # input variable
            'DH': [False]                       # output variable
}

data_folder = 'outputs/sensitivity_analysis/electrolyzer_cost/'  # folder with all results
name_files= []
for f in os.listdir(data_folder):
    if f.endswith('.nc'):
        # check if CO2 cost in the file is among the selected ones
        m = re.search('CO2c(\d+)', f)
        m_co2 = int(m.group(1))

        # check if H2 demand in the file is among the selected ones
        m = re.search('H2d(\d+)', f)
        m_h2 = int(m.group(1)) * 1000 / p.H2_output

        # check if fC_MeOH in the file is among the selected ones
        m = re.search('MeOHd(\d+)', f)
        MeOH_y_d = int(m.group(1)) * 1000  # demand in GWh/y
        bioCH4_y_d = bioCH4_prod.values.sum()
        CO2_MeOH_plant = 1 / GL_eff.at['Methanol', 'Methanol plant']  # bus0 = CO2, bus1 = Methanol
        bioCH4_CO2plant = GL_eff.at['bioCH4', 'SkiveBiogas'] / GL_eff.at['CO2 pure', 'SkiveBiogas']
        fC_MeOH_value = round((MeOH_y_d * CO2_MeOH_plant) * bioCH4_CO2plant / bioCH4_y_d, 2)
        fC_MeOH = min(MeOH_rec_list, key=lambda x: abs(x - fC_MeOH_value))

        # check if DH in the file is among the selected ones
        cond_DH = np.sign(f.find('DH')+1) in dataset_flags['DH']

        # check if En_year in the file is among the selected ones
        y_list= [str(y) for y in  dataset_flags['En_year_price']]
        cond_En_y = any(ele in f for ele in y_list)

        # electrolyzer cost multiplier
        m = re.search('H2cost_(\d+)', f)
        m_h2_c = int(m.group(1))

        if (m_co2 in dataset_flags['CO2_cost']) and (m_h2 in dataset_flags['d_H2']) and (fC_MeOH in dataset_flags['fC_MeOH']) and cond_DH and cond_En_y:
            name_files.append(f)

# ref network for agents
single_analysis_folder = 'outputs/single_analysis/'
with open(single_analysis_folder+'network_comp_allocation.pkl', 'rb') as f:
    network_comp_allocation = pkl.load(f)
agent_list=[key for key in network_comp_allocation]

# DECIDE WHAT TO PLOT :Set True to the variables in the results
results_flags = {
            'CO2_cost': True,           # input variable -> for categorization
            'fC_MeOH': True,            # input variable -> for categorization
            'd_H2': True,               # input variable -> for categorization
            'En_year_price': True,      # input variable -> for categorization
            'DH': True,                 # input variable -> for categorization
            'DH_y': True,               # output variable
            'mu_H2': True,              # output variable
            'mu_MeOH': True,            # output variable
            'mu_el_GLS': True,          # output variable
            'mu_heat_MT': True,         # output variable
            'mu_heat_DH':True,          # output variable
            'mu_heat_LT': True,         # output variable
            'mu_CO2': True,             # output variable
            'tot_sys_cost': True,       # output variable
            'tot_cap_cost' : True,      # output variable
            'tot_mar_cost' : True,      # output variable
            'electrolyzer cost m' : True, # input variable
}
# add agents list to the list of resutls
agent_dict1=dict((ag +'_cc',True) for ag in agent_list)
agent_dict2=dict((ag +'_mc',True) for ag in agent_list)
results_flags.update(agent_dict1)
results_flags.update(agent_dict2)

results_units = {
            'CO2_cost': '(€/t)',            # input variable -> for categorization
            'fC_MeOH': '(C_MeOH/C_sep)',    # input variable -> for categorization
            'd_H2': '(GWh/y)',              # input variable -> for categorization
            'En_year_price': '',            # input variable -> for categorization
            'DH': '(-)',                    # input variable -> for categorization
            'DH_y': '(GWh/y)',              # output variable
            'mu_H2': '(€/kg)',              # output variable
            'mu_MeOH': '(€/kg)',            # output variable
            'mu_el_GLS': '(€/MWh)',         # output variable
            'mu_heat_MT': '(€/MWh)',        # output variable
            'mu_heat_DH': '(€/MWh)',        # output variable
            'mu_heat_LT': '(€/MWh)',        # output variable
            'mu_CO2': '(€/t)',              # output variable
            'tot_sys_cost': '(€/y)',        # output variable
            'tot_cap_cost': '(€/y)',        # output variable
            'tot_mar_cost': '(€/y)',        # output variable
            'electrolyzer cost m' : '(-)',
}
agent_dict3=dict((ag+'_cc','(€/y)') for ag in agent_list)
agent_dict4=dict((ag+'_mc','(€/y)') for ag in agent_list)
results_units.update(agent_dict3)
results_units.update(agent_dict4)

# define Results Data Frame
results_columns=[]
for key in results_flags:
    if results_flags[key]:
        results_columns.append(key)

df_results= pd.DataFrame(0, index=name_files, columns=results_columns)

# load all results
for name in name_files:
    # import network
    n_name= 'n_'+'name' # network name
    n_name = pypsa.Network(os.path.join(data_folder, name))

    # Retrive info from Network or network name
    # Independent variables
    m = re.search('CO2c(\d+)', name)
    df_results.at[name, 'CO2_cost'] = int(m.group(1))

    # Retrive Fraction of CO2 from biogas plant recovered to MeOH
    MeOH_y_d = n_name.loads_t.p_set['Methanol'].sum()
    bioCH4_y_d = n_name.loads_t.p_set['bioCH4'].sum()
    CO2_MeOH_plant = 1/n_name.links.efficiency['Methanol plant'] # bus0 = CO2, bus1 = Methanol
    bioCH4_CO2plant = n_name.links.efficiency['SkiveBiogas'] / n_name.links.efficiency2['SkiveBiogas'] # bus0 = biomass, bus1= bioCH4, bus2=CO2
    fC_MeOH = round((MeOH_y_d * CO2_MeOH_plant) * bioCH4_CO2plant / bioCH4_y_d, 2)
    df_results.at[name,'fC_MeOH']=fC_MeOH

    m = re.search('H2d(\d+)', name)
    df_results.at[name,'d_H2']= int(m.group(1))

    if '2019' in name:
        df_results.at[name,'En_year_price'] = 2019
    elif '2022' in name:
        df_results.at[name, 'En_year_price'] = 2022


    if 'DH' in name:
        df_results.at[name, 'DH'] = 1
    else:
        df_results.at[name, 'DH'] = 0

    # Output variables
    # DH y production
    if 'DH' in name:
        df_results.at[name, 'DH_y'] = int(n_name.links_t.p0['DH GL_to_DH grid'].sum() // 1000)
    else:
        df_results.at[name, 'DH_y'] = 0

    # H2 prod cost
    m = re.search('H2d(\d+)', name)
    d_H2 = int(m.group(1))
    if d_H2 ==0:
        df_results.at[name, 'mu_H2'] = np.mean(n_name.buses_t.marginal_price['H2_meoh']) * p.lhv_h2 /1000
    else:
        df_results.at[name, 'mu_H2'] = np.mean(n_name.buses_t.marginal_price['H2 delivery']) * p.lhv_h2 /1000

    # MeOH prod cost
    if results_flags['mu_MeOH']:
        df_results.at[name, 'mu_MeOH'] = np.mean(n_name.buses_t.marginal_price['Methanol']) * p.lhv_meoh

    # El GL cost
    el_GL_bus= 'El2 bus'
    df_results.at[name, 'mu_el_GLS'] = np.mean(n_name.buses_t.marginal_price['El2 bus'])

    # Heat MT GL cost
    if 'Heat MT' in n_name.buses.index.values:
        df_results.at[name, 'mu_heat_MT'] = np.mean(n_name.buses_t.marginal_price['Heat MT'])
    else:
        df_results.at[name, 'mu_heat_MT'] = 0

    # Heat DH GL cost
    if 'Heat DH' in n_name.buses.index.values:
        df_results.at[name, 'mu_heat_DH'] = np.mean(n_name.buses_t.marginal_price['Heat DH'])
    else:
        df_results.at[name, 'mu_heat_DH'] = 0

    # Heat LT GL cost
    if 'Heat DH' in n_name.buses.index.values:
        df_results.at[name, 'mu_heat_LT'] = np.mean(n_name.buses_t.marginal_price['Heat LT'])
    else:
        df_results.at[name, 'mu_heat_LT'] = 0

    # CO2 price GL sold by Biogas plant (LP)
    df_results.at[name, 'mu_CO2'] = np.mean(n_name.buses_t.marginal_price['CO2 sep'])

    # total system , capital and marginal cost
    tot_cc, tot_mc, tot_sc = get_system_cost(n_name)
    df_results.at[name, 'tot_sys_cost'] = np.sum(tot_sc)
    df_results.at[name, 'tot_cap_cost'] = np.sum(tot_cc)
    df_results.at[name, 'tot_mar_cost'] = np.sum(tot_mc)

    # total capital and marginal costs, by agents
    cc_tot_agent, mc_tot_agent = get_total_marginal_capital_cost_agents(n_name, network_comp_allocation, False)
    for key in cc_tot_agent:
        df_results.at[name, key + '_cc'] = cc_tot_agent[key]
        df_results.at[name, key + '_mc'] = mc_tot_agent[key]

    # electrolyzer cost multiplier

    m = re.search('H2cost_(\d+)', name)
    df_results.at[name, 'electrolyzer cost m'] = int(m.group(1))


# ---------------- PLOT SECTION--------------------------------
# definition of reference case
ref_CO2_cost = 150
ref_fC_MeOH = 0.90
ref_DH = 0
ref_bioChar = 0
ref_el_DK1_sale_el_RFNBO = 0.5

# general plot parameters
fnt_sz = 14  # font size for publications

''' PLOT 1 - LCO H2 and MEOH'''

# select variables in df_resutls

x_var = ['electrolyzer cost m']

y_var1 = ['mu_MeOH',
          'mu_H2']

var_list = x_var + y_var1

# build the df for plots:
df_plot_1 = df_results.query('d_H2==0 & DH ==@ref_DH & bioChar ==@ref_bioChar & fC_MeOH == @ref_fC_MeOH')[var_list].copy()
df_plot_1.columns = [results_plot[v] for v in var_list]

df_plot_2 = df_results.query('d_H2==272 & DH ==@ref_DH & bioChar ==@ref_bioChar & fC_MeOH == @ref_fC_MeOH')[var_list].copy()
df_plot_2.columns = [results_plot[v] for v in var_list]

# Plot
y_label = '(€/MWh)'

# H2 and MeOH

c_v1 = ['#a559aa', '#59a89c']  # , '#f0c571']

fig, axes = plt.subplots(1, 2)
# plot H2 grid results
for i in range(len(x_var)):
    for j in range(len(y_var1)):
        sns.lineplot(data=df_plot_2, x=results_plot[x_var[i]], y=results_plot[y_var1[j]], ax=axes[i], marker='o',
                     linestyle='solid', label=results_plot[y_var1[j]] + ' ' + 'H2 Grid', color=c_v1[j], ci=100)
        sns.lineplot(data=df_plot_1, x=results_plot[x_var[i]], y=results_plot[y_var1[j]], ax=axes[i], marker='D',
                     linestyle='dashed', label=results_plot[y_var1[j]] + ' ' + 'MeOH only', color=c_v1[j], ci=100)
        axes[i].set_ylabel(y_label, fontsize=fnt_sz)
        axes[i].set_xlabel(axes[i].get_xlabel(), fontsize=fnt_sz)
        axes[i].tick_params(axis='both', which='major', labelsize=fnt_sz)
        #        axes[i].get_legend().set_visible(False)
        axes[i].axhline(0, c='black', lw=0.5)
        axes[i].set_ylim(20, 170)
axes[0].axvline(0.9, c='black', lw=0.3)
axes[1].axvline(0.5, c='black', lw=0.3)

axes[1].get_legend().set_visible(False)

plt.suptitle('Year average production cost', fontsize=fnt_sz)
sns.despine()
plt.subplots_adjust(wspace=0.4)
plt.show()
plt.tight_layout()
