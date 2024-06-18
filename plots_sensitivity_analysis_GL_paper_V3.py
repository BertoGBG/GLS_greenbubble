from functions_GL_paper_V3 import *

# ------------- LOAD OF RESULTS FROM SENSITIVITY ANALYSIS----------------------------------

''' Folder with saved results '''
data_folder = 'outputs/sensitivity_analysis/Results_V2/'  # folder with all results

'''Retrieve agent list from existing network_comp_allocation file'''
# CHECK that a network_comp_allocation file is present in the folder with sensitivity analysis.
with open(data_folder + 'network_comp_allocation.pkl', 'rb') as f:
    network_comp_allocation = pkl.load(f)

agent_list = [key for key in network_comp_allocation]

capacity_list = ['solar', 'onshorewind', 'Electrolyzer', 'El3_to_El2', 'El3_to_DK1', 'Methanol plant', 'SkyClean',
                 'Heat pump', 'CO2 compressor', 'H2 compressor', 'battery', 'Heat DH storage',
                 'Concrete Heat MT storage', 'H2 HP', 'CO2 pure HP', 'CO2 Liq', 'El boiler', 'NG boiler extra',
                 'Pellets boiler']

''' Variables to import'''
results_flags = {
    'CO2_cost': True,  # input parameter
    'fC_MeOH': True,  # input parameter
    'd_H2': True,  # input parameter
    'En_year_price': True,  # input parameter
    'DH': True,  # input parameter
    'el_DK1_sale_el_RFNBO': True,  # input parameter
    'bioChar': True,  # input parameter
    'DH_y': True,  # output variable
    'RE_y': True,  # output variable
    'MeOH_y': True,  # output variable
    'mu_H2': True,  # output variable
    'mu_MeOH': True,  # output variable
    'mu_el_GLS': True,  # output variable
    'mu_heat_MT': True,  # output variable
    'mu_heat_DH': True,  # output variable
    'mu_heat_LT': True,  # output variable
    'mu_CO2': True,  # output variable
    'mu_bioCH4': True,  # output variable
    'H2_sales': True,  # output variable
    'MeOH_sales': True,  # output variable
    'RE_sales': True,  # output variable
    'DH_sales': True,  # output variable
    'BECS_sales': True,  # output variable
    'bioCH4_sales': True,  # output variable
    'tot_sys_cost': True,  # output variable
    'tot_cap_cost': True,  # output variable
    'tot_mar_cost': True,  # output variable
}
# add marginal and capital cost per agent
agent_dict1 = dict((key + '_cc', True) for key in network_comp_allocation)
agent_dict2 = dict((key + '_mc', True) for key in network_comp_allocation)
results_flags.update(agent_dict1)
results_flags.update(agent_dict2)

''' Create df_results '''
'''Range of parameters to import'''
dataset_flags = {
    'CO2_cost': p.CO2_cost_list,
    'd_H2': p.H2_demand_list,
    'fC_MeOH': p.MeOH_rec_list,
    'DH': p.DH_flag_list,
    'bioChar': p.bioCh_credits_list,
    'el_DK1_sale_el_RFNBO': p.el_DK1_sale_el_RFNBO_list,
    'En_year_price': [2019, 2022]
}

df_results, results_plot = results_df_plot_build(data_folder, dataset_flags, results_flags, network_comp_allocation,
                                                 capacity_list)
# to save DF if needed
# df_results.to_pickle(data_folder+'df_results')
# df_results = pd.read_pickle(data_folder+'df_results')

# import other network independent data
_, GL_eff, _, _, _, _, _, _, _, _, _, _ = load_input_data()
tech_costs = prepare_costs(p.cost_file, p.USD_to_EUR, p.discount_rate, 1, p.lifetime)

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

x_var = ['fC_MeOH',
         'el_DK1_sale_el_RFNBO']

y_var1 = ['mu_MeOH',
          'mu_H2']

"""y_var2 = ['mu_heat_MT',
          'mu_el_GLS']"""

var_list = x_var + y_var1  # + y_var2

# build the df for plots:
df_plot_1 = df_results.query('d_H2==0 & DH ==@ref_DH & bioChar ==@ref_bioChar')[var_list].copy()
df_plot_1.columns = [results_plot[v] for v in var_list]

df_plot_2 = df_results.query('d_H2==272 & DH ==@ref_DH & bioChar ==@ref_bioChar')[var_list].copy()
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

""" PLOT 2 - violin plot for shadow prices"""

bus_dict_v = {'mu_H2': 'H2_meoh',  # output variable
              'mu_MeOH': 'Methanol',  # output variable
              'mu_heat_MT': 'Heat MT',  # output variable
              'mu_heat_DH': 'Heat DH',  # output variable
              'mu_heat_LT': 'Heat LT',  # output variable
              'mu_CO2': 'CO2 sep',  # output variable
              'mu_bioCH4': 'bioCH4',  # output variable
              'mu_el_GLS': 'El2 bus'  # output variable
              }

grid_dict_units = {
    'el_spotprice': 'El. grid sale' + '\n' + '(€/MWh)',
    'el_price_w_tariff_CO2': 'El. grid purchase' + '\n' + '(€/MWh)',
    'NG_price_wCO2': 'NG grid purchase' + '\n' + '(€/MWh)'}

# prepare DF for violin plot
results_sp = {}
for h2 in df_results['d_H2'].unique():
    for y in df_results['En_year_price'].unique():
        name = str(np.squeeze(df_results.query('d_H2==@h2 & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
                                               '==@ref_DH  & bioChar ==@ref_bioChar & '
                                               'el_DK1_sale_el_RFNBO==@ref_el_DK1_sale_el_RFNBO & En_year_price== '
                                               '@y').index.values))
        # results DF
        df_results_sp = pd.DataFrame()
        # import network
        n = pypsa.Network(os.path.join(data_folder, name))
        # read all marginal prices in bus_list
        for key in bus_dict_v:
            if key == 'mu_H2' and h2 > 0:  # adjust H2 bus if grid connected
                b = 'H2 delivery'
            else:
                b = bus_dict_v[key]

            if b in n.buses.index.values:
                df_results_sp[key] = n.buses_t.marginal_price[b]

        # add grid prices
        inputs_dict_sc1 = create_local_input_dict(y)
        inputs_dict_sc1['CO2 cost'] = ref_CO2_cost
        inputs_dict_sc1['GL_eff'] = GL_eff
        en_market_prices = en_market_prices_w_CO2(inputs_dict_sc1, tech_costs)
        df_results_sp['el_spotprice'] = -en_market_prices['el_grid_sell_price'].values
        df_results_sp['el_price_w_tariff_CO2'] = en_market_prices['el_grid_price'].values
        df_results_sp['NG_price_wCO2'] = en_market_prices['NG_grid_price'].values
        df_results_sp.set_index(n.snapshots, inplace=True)

        # change column names for plotting
        cols_name = {}
        for c in df_results_sp.columns.values:
            if c in results_plot.keys():
                cols_name[c] = results_plot[c]
            elif c in grid_dict_units.keys():
                cols_name[c] = grid_dict_units[c]
        df_results_sp.rename(columns=cols_name, inplace=True)

        # save df in dict
        results_sp[name] = df_results_sp

# violin plots
df_plot_v = results_sp[
    (str(np.squeeze(df_results.query('d_H2==272 & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
                                     '==@ref_DH  & bioChar ==@ref_bioChar & '
                                     'el_DK1_sale_el_RFNBO==@ref_el_DK1_sale_el_RFNBO & En_year_price== '
                                     '2019').index.values)))].copy()
df_plot_v3 = results_sp[
    (str(np.squeeze(df_results.query('d_H2==0 & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
                                     '==@ref_DH  & bioChar ==@ref_bioChar & '
                                     'el_DK1_sale_el_RFNBO==@ref_el_DK1_sale_el_RFNBO & En_year_price== '
                                     '2019').index.values)))].copy()
df_plot_v2 = results_sp[
    (str(np.squeeze(df_results.query('d_H2==272 & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
                                     '==@ref_DH  & bioChar ==@ref_bioChar & '
                                     'el_DK1_sale_el_RFNBO==@ref_el_DK1_sale_el_RFNBO & En_year_price== '
                                     '2022').index.values)))].copy()

df_plot_v4 = results_sp[
    (str(np.squeeze(df_results.query('d_H2==0 & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
                                     '==@ref_DH  & bioChar ==@ref_bioChar & '
                                     'el_DK1_sale_el_RFNBO==@ref_el_DK1_sale_el_RFNBO & En_year_price== '
                                     '2022').index.values)))].copy()

fig, ax = plt.subplots(2, 1)
v1 = ax[0].violinplot(df_plot_v, points=100, positions=np.arange(0, len(df_plot_v.columns)),
                      showmeans=True, showextrema=False, showmedians=False)
ax[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for b in v1['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_color('navy')
v1['cmeans'].set_color('navy')

v2 = ax[0].violinplot(df_plot_v2, points=100, positions=np.arange(0, len(df_plot_v2.columns)),
                      showmeans=True, showextrema=False, showmedians=False)
for b in v2['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_color('firebrick')
v2['cmeans'].set_color('firebrick')

plt_ttl = 'H2 to Grid'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax[0].text(0.05, 0.95, plt_ttl, transform=ax[0].transAxes, fontsize=fnt_sz - 2, verticalalignment='top', bbox=props)
ax[0].legend([v1['bodies'][0], v2['bodies'][0]], ['2019', '2022'])  # , title= lgn_ttl)
ax[0].set_xticks([])
ax[0].tick_params(axis='both', which='major', labelsize=fnt_sz)
ax[0].set_ylim(-100, 1100)
sns.despine()

v3 = ax[1].violinplot(df_plot_v3, points=100, positions=np.arange(0, len(df_plot_v3.columns)),
                      showmeans=True, showextrema=False, showmedians=False)
ax[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

for b in v3['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_color('navy')
v3['cmeans'].set_color('navy')

v4 = ax[1].violinplot(df_plot_v4, points=100, positions=np.arange(0, len(df_plot_v4.columns)),
                      showmeans=True, showextrema=False, showmedians=False)
for b in v4['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_color('firebrick')
v4['cmeans'].set_color('firebrick')

plt_ttl = 'H2 to MeOH only'
ax[1].legend([v3['bodies'][0], v4['bodies'][0]], ['2019', '2022'])  # , title= lgn_ttl)
x_ticks_plot = list(df_plot_v.columns.values)
ax[1].set_xticks(range(0, len(x_ticks_plot)), x_ticks_plot, rotation=45, fontsize=fnt_sz)
ax[1].tick_params(axis='both', which='major', labelsize=fnt_sz)
ax[1].text(0.05, 0.95, plt_ttl, transform=ax[1].transAxes, fontsize=fnt_sz - 2, verticalalignment='top', bbox=props)
ax[1].set_ylim(-100, 1100)
sns.despine()
fig.tight_layout()

""" PLOT 3  Bar plot system costs """

df_query = df_results.query('CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
                            '==@ref_DH  & bioChar ==@ref_bioChar & '
                            'el_DK1_sale_el_RFNBO==@ref_el_DK1_sale_el_RFNBO').copy()

SC_list = []
for idx in df_query.index:
    if df_query.loc[idx, 'd_H2'] == 0:
        str1 = 'H2 to MeOH only'
    else:
        str1 = 'H2 to Grid'
    if df_query.loc[idx, 'En_year_price'] == 2019:
        str2 = '2019'
    else:
        str2 = '2022'
    SC_list.append(str1 + '\n' + str2)

df_plot = pd.DataFrame()
for ag in agent_list:
    df_plot[ag + '_cc'] = df_query[ag + '_cc'] / 1e6
df_plot['tot_mar_cost'] = df_query['tot_mar_cost'] / 1e6
df_plot['symbiosis_net_cc'] = (df_plot['symbiosis_net_cc'] + df_plot['SkiveBiogas_cc']) / 1e6
df_plot.drop(columns=['external_grids_cc', 'DH_cc', 'SkiveBiogas_cc'], inplace=True)

col_plot_list = ['renewables cap. cost', 'electrolyzer cap. cost', 'MeOH synthesis cap. cost',
                 'central heating cap. cost', 'other cap. cost', 'total marginal cost']
df_plot.columns = col_plot_list

col_plot_list.insert(0, 'total system cost')
my_colors = ['lightskyblue', 'palegreen', 'darkorchid', 'orange', 'pink', 'lightsalmon']

df_plot.set_axis(SC_list, inplace=True)

# include sales of H2 and MeOH at shadow price
PtX_sales = (df_query['d_H2'] * df_query['mu_H2'] + df_query['MeOH_y'] * df_query['mu_MeOH']) * 1e3 / 1e6
PtX_sales.set_axis(SC_list, inplace=True)
tot_sys_cost_w_PtX = df_plot.sum(axis=1) - PtX_sales

col_plot_list.insert(0, 'total system cost w/ PtX sales')

df_plot.plot(kind='bar', stacked=True, color=my_colors)  # colormap='Paired')

plt.scatter(df_plot.sum(axis=1).index.values, tot_sys_cost_w_PtX, marker='D', color='black')
plt.scatter(df_plot.sum(axis=1).index.values, df_plot.sum(axis=1), marker='X', color='black')
plt.xticks(rotation=45)
plt.ylabel('(M€/y)', fontsize=fnt_sz)
plt.yticks(fontsize=fnt_sz)
plt.legend(col_plot_list)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

for idx in df_plot.sum(axis=1).index.values:
    y_min = df_plot.sum(axis=1)[idx]
    y_max = tot_sys_cost_w_PtX[idx]
    plt.vlines(x=idx, ymin=y_min, ymax=y_max, color='black', linestyle='dashed', linewidth=0.3)

sns.despine()
plt.tight_layout()

""" PLOT 4 capacities correlations - REF MeOH Only """

x_var = ['En_year_price',
         'el_DK1_sale_el_RFNBO',
         'CO2_cost',
         'bioChar',
         'fC_MeOH']

y_var = [c for c in capacity_list]
y_var2 = ['tot_sys_cost', 'tot_mar_cost']

var_list = x_var + y_var + y_var2

# plot var name
var_plot_new_list = {'onshorewind': 'wind' + '\n' + '(MW el)',
                     'solar': 'solar' + '\n' + '(MW el)',
                     'El. connections': 'Electric.' + '\n' + 'connections' + '\n' + '(MW el)',
                     'Methanol plant': 'Methanol' + '\n' + 'plant' + '\n' + '(MW meoh)',
                     'Compressors': 'Compress.' + '\n' + '(CO2 & H2)' + '\n' + '(MW el)',
                     'H2 HP': 'H2 ' + '\n' + 'storage' + '\n' + '(MWh H2)',
                     'battery': 'Battery' + '\n' + '(MWh el)',
                     'Electrolyzer': 'Electrolyzer' + '\n' + '(MW el)',
                     'CO2 storage': 'CO2' + '\n' + 'storage' + '\n' + '(t)',
                     'SkyClean': 'SkyClean' + '\n' + '(MW th)',
                     'El boiler': 'El boiler' + '\n' + '(MW el)',
                     'NG boiler extra': 'NG boiler extra' + '\n' + '(MW th)',
                     'Heat storage': 'Heat storage' + '\n' + '(MWh th)',
                     'Heat pump': 'Heat pump' + '\n' + '(MW th)',
                     'Pellets boiler': 'Pellets' + '\n' + 'boiler' + '\n' + '(MW th)',
                     'Renewables': 'Renewables' + '\n' + '(MW el)'}

# data for MeOH only -  short version -> paper
# df_plot_1 = df_results.query('d_H2==0 & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
#                             '==@ref_DH  & bioChar ==@ref_bioChar'
#                             )[var_list].copy()

# data for MeOH only - long verision -> appendix
df_plot_1 = df_results.query('d_H2==0 & CO2_cost==@ref_CO2_cost & fC_MeOH <=0.95 & DH '
                             '==@ref_DH  & bioChar ==@ref_bioChar'
                             )[var_list].copy()

# aggregation - MeOH Only
df_plot_1['El. connections'] = df_plot_1['El3_to_El2'] + df_plot_1['El3_to_DK1']
df_plot_1['Compressors'] = df_plot_1['CO2 compressor'] + df_plot_1['H2 compressor']
df_plot_1['CO2 storage'] = df_plot_1['CO2 pure HP'] + df_plot_1['CO2 Liq']
df_plot_1['Heat storage'] = df_plot_1['Heat DH storage'] + df_plot_1['Concrete Heat MT storage']
df_plot_1['Renewables'] = df_plot_1['onshorewind'] + df_plot_1['solar']
df_plot_1['tot_sys_cost'] = df_plot_1['tot_sys_cost'] / 1e6  # adjustment for plotting

# M€/y for cost
results_plot['tot_sys_cost'] = 'tot\nsystem\ncost\n(M€/y)'
results_plot['tot_mar_cost'] = 'tot\nmarginal\ncost\n(M€/y)'
results_plot['tot_cap_cost'] = 'tot\ncapital\ncost\n(M€/y)'

df_plot_1.rename(columns={'el_DK1_sale_el_RFNBO': results_plot['el_DK1_sale_el_RFNBO'],
                          'En_year_price': results_plot['En_year_price'],
                          'CO2_cost': results_plot['CO2_cost'],
                          'fC_MeOH': results_plot['fC_MeOH'],
                          'tot_sys_cost': results_plot['tot_sys_cost'],
                          'tot_mar_cost': results_plot['tot_mar_cost'],
                          },
                 inplace=True)

df_plot_1.rename(columns=var_plot_new_list,
                 inplace=True)

# plot variable - short version
var_plot_list = ['Renewables', 'Electrolyzer', 'Methanol plant', 'H2 HP', 'battery', 'CO2 storage',
                 results_plot['tot_mar_cost']]

# for full table:
var_plot_new_list2 = var_plot_new_list.copy()
for k in ['Renewables', 'Heat pump', 'Heat storage', 'NG boiler extra', 'SkyClean']:
    var_plot_new_list2.pop(k, None)
var_plot_list = [key for key in var_plot_new_list2] + ['tot_sys_cost']

var_plot_1 = []
for v in var_plot_list:
    if v in var_plot_new_list:
        var_plot_1.append(var_plot_new_list[v])
    elif v in results_plot:
        var_plot_1.append(results_plot[v])

b_adjust_kde = 0.8
trsh = 0.07
g1 = sns.PairGrid(df_plot_1, vars=var_plot_1,
                  hue=results_plot['el_DK1_sale_el_RFNBO'],
                  palette='muted', diag_sharey=False)
g1.map_diag(sns.kdeplot, fill=True, common_norm=False, warn_singular=False)
g1.map_lower(sns.scatterplot, s=70, marker='^',
             size=df_plot_1[results_plot['fC_MeOH']]),
#             style= df_plot_1[results_plot['CO2_cost']])
g1.map_lower(sns.kdeplot,
             hue=df_plot_1[results_plot['En_year_price']],
             levels=1, thresh=trsh, palette='coolwarm', bw_adjust=b_adjust_kde, common_norm=False,
             linewidths=0.9, warn_singular=False)
g1.map_upper(reg_coef, hue=None)
g1.add_legend(title='max RE to grid')
# sns.set_context("paper", rc={"axes.labelsize":10})
# plt.suptitle('correlations between PtX capacities' + '\n' + 'H2 production: MeOH only')
plt.show()
plt.tight_layout()

""" PLOT 5 capacities correlations - Ref H2 grid """

# data for H2 grid - short version -> paper
# df_plot_2 = df_results.query('d_H2==272 & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
#                             '==@ref_DH  & bioChar ==@ref_bioChar'
#                             )[var_list].copy()[var_list].copy()

# H2 grid case - long version - appendix
df_plot_2 = df_results.query('d_H2==272 & CO2_cost==@ref_CO2_cost & fC_MeOH <=0.95 & DH '
                             '==@ref_DH  & bioChar ==@ref_bioChar'
                             )[var_list].copy()

# aggregation  H2 grid
df_plot_2['El. connections'] = df_plot_2['El3_to_El2'] + df_plot_2['El3_to_DK1']
df_plot_2['Compressors'] = df_plot_2['CO2 compressor'] + df_plot_2['H2 compressor']
df_plot_2['CO2 storage'] = df_plot_2['CO2 pure HP'] + df_plot_2['CO2 Liq']
df_plot_2['Heat storage'] = df_plot_2['Heat DH storage'] + df_plot_2['Concrete Heat MT storage']
df_plot_2['tot_sys_cost'] = df_plot_2['tot_sys_cost'] / 1e6

# M€/y for cost
results_plot['tot_sys_cost'] = 'tot\nsystem\ncost\n(M€/y)'
results_plot['tot_mar_cost'] = 'tot\nmarginal\ncost\n(M€/y)'
results_plot['tot_cap_cost'] = 'tot\ncapital\ncost\n(M€/y)'

df_plot_2.rename(columns={'el_DK1_sale_el_RFNBO': results_plot['el_DK1_sale_el_RFNBO'],
                          'En_year_price': results_plot['En_year_price'],
                          'CO2_cost': results_plot['CO2_cost'],
                          'fC_MeOH': results_plot['fC_MeOH'],
                          'tot_sys_cost': results_plot['tot_sys_cost'],
                          'tot_mar_cost': results_plot['tot_mar_cost'],
                          },
                 inplace=True)

df_plot_2.rename(columns=var_plot_new_list,
                 inplace=True)

# short version -> paper
# var_plot_list_2 = ['onshorewind', 'solar', 'Electrolyzer', 'Methanol plant', 'battery', 'tot_sys_cost']

# for long version -> appendix :
var_plot_new_list2 = var_plot_new_list.copy()
for k in ['Renewables', 'Heat pump', 'H2 HP', 'SkyClean', 'El boiler', 'NG boiler extra', 'Heat storage']:
    var_plot_new_list2.pop(k, None)
var_plot_list_2 = [key for key in var_plot_new_list2] + ['tot_sys_cost']

var_plot_2 = []
for v in var_plot_list_2:
    if v in var_plot_new_list2:
        var_plot_2.append(var_plot_new_list[v])
    elif v in results_plot:
        var_plot_2.append(results_plot[v])

# adjustment for plotting KDE
df_plot_2['Compress.' + '\n' + '(CO2 & H2)' + '\n' + '(MW el)'] = df_plot_2[
                                                                      'Compress.' + '\n' + '(CO2 & H2)' + '\n' + '(MW el)'] + np.random.normal(
    loc=0,
    scale=0.0000001,
    size=len(
        df_plot_2[
            'Compress.' + '\n' + '(CO2 & H2)' + '\n' + '(MW el)']))
df_plot_2['Pellets' + '\n' + 'boiler' + '\n' + '(MW th)'] = df_plot_2[
                                                                'Pellets' + '\n' + 'boiler' + '\n' + '(MW th)'] + np.random.normal(
    loc=0, scale=0.0000001,
    size=len(df_plot_2[
                 'Pellets' + '\n' + 'boiler' + '\n' + '(MW th)']))
b_adjust_kde = 0.9
trsh = 0.07

g2 = sns.PairGrid(df_plot_2, vars=var_plot_2,
                  hue=results_plot['el_DK1_sale_el_RFNBO'],
                  palette='muted', diag_sharey=False)
g2.map_diag(sns.kdeplot, fill=True, common_norm=False, warn_singular=False)
g2.map_lower(sns.scatterplot, s=70,
             size=df_plot_2[results_plot['fC_MeOH']]),
# style= df_plot_2[results_plot['CO2_cost']] )
g2.map_lower(sns.kdeplot,
             hue=df_plot_2[results_plot['En_year_price']],
             levels=1, thresh=trsh, palette='coolwarm', bw_adjust=b_adjust_kde, common_norm=False,
             linewidths=0.9, warn_singular=False)
g2.map_upper(reg_coef, hue=None)
g2.add_legend(title='max RE to grid')
# plt.suptitle('correlations between PtX capacities' + '\n' + 'H2 production: Grid')


""" PLOT 6 -  CORRELATIONS BETWEEN SHADOW PRICES -  MeOH only """

x_var = ['En_year_price',
         'el_DK1_sale_el_RFNBO',
         'CO2_cost',
         'bioChar',
         'd_H2',
         'fC_MeOH']

y_var = ['mu_H2',
         'mu_MeOH',
         'mu_el_GLS',
         'mu_bioCH4',
         'mu_heat_MT',
         'mu_CO2']

var_list = x_var + y_var

# short version -> paper
# df_plot_1= df_results.query('d_H2==0 & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
#                             '==@ref_DH  & bioChar ==@ref_bioChar'
#                             )[var_list].copy()
# long version -> appendix
df_plot_1 = df_results.query('d_H2==0 & CO2_cost==@ref_CO2_cost & fC_MeOH <=0.95 & DH '
                             '==@ref_DH  & bioChar ==@ref_bioChar'
                             )[var_list].copy()

df_plot_1.columns = [results_plot[v] for v in var_list]

# add little variance for plotting KDE
df_plot_1[results_plot['mu_heat_MT']] = df_plot_1[results_plot['mu_heat_MT']] + np.random.normal(loc=0, scale=0.0000001,
                                                                                                 size=len(df_plot_1[
                                                                                                              results_plot[
                                                                                                                  'mu_heat_MT']]))

g1 = sns.PairGrid(df_plot_1, vars=[results_plot[v] for v in y_var], hue=results_plot['el_DK1_sale_el_RFNBO'],
                  palette='muted', diag_sharey=False)
g1.map_diag(sns.kdeplot, fill=True, common_norm=False)
g1.map_lower(sns.scatterplot, s=70, marker='^', size=df_plot_1[results_plot['fC_MeOH']])
g1.map_lower(sns.kdeplot,
             hue=df_plot_1[results_plot['En_year_price']],
             levels=1, thresh=trsh, palette='coolwarm', bw_adjust=b_adjust_kde, common_norm=False,
             linewidths=0.9, warn_singular=False)
g1.map_upper(reg_coef, hue=None)
g1.add_legend(title='max RE to grid')

"""PLOT 7 -  CORRELATIONS BETWEEN SHADOW PRICES -  H2 grid """

# short version -> paper
# df_plot_2= df_results.query('d_H2==272 & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
#                             '==@ref_DH  & bioChar ==@ref_bioChar'
#                             )[var_list].copy()

# full version -> appendix
df_plot_2 = df_results.query('d_H2==272 & CO2_cost==@ref_CO2_cost & fC_MeOH <=0.95 & DH '
                             '==@ref_DH  & bioChar ==@ref_bioChar'
                             )[var_list].copy()

df_plot_2.columns = [results_plot[v] for v in var_list]

# add little variance for plotting KDE
df_plot_2[results_plot['mu_heat_MT']] = df_plot_2[results_plot['mu_heat_MT']] + np.random.normal(loc=0, scale=0.0000001,
                                                                                                 size=len(df_plot_2[
                                                                                                              results_plot[
                                                                                                                  'mu_heat_MT']]))

g2 = sns.PairGrid(df_plot_2, vars=[results_plot[v] for v in y_var], hue=results_plot['el_DK1_sale_el_RFNBO'],
                  palette='muted', diag_sharey=False)
g2.map_diag(sns.kdeplot, fill=True, common_norm=False)
g2.map_lower(sns.scatterplot, s=70, size=df_plot_2[results_plot['fC_MeOH']])
g2.map_lower(sns.kdeplot,
             hue=df_plot_2[results_plot['En_year_price']],
             levels=1, thresh=trsh, palette='coolwarm', bw_adjust=b_adjust_kde, common_norm=False,
             linewidths=0.9, warn_singular=False)
g2.map_upper(reg_coef, hue=None)
g2.add_legend(title='max RE to grid')

"""PLOT 8 -  impact of DH and Biochar on optimal capacities """

x_var = ['En_year_price',
         'el_DK1_sale_el_RFNBO',
         'CO2_cost',
         'bioChar',
         'fC_MeOH',
         'DH',
         'd_H2']

y_var = [c for c in capacity_list]

y_var3 = ['mu_MeOH',
          'mu_H2',
          'mu_bioCH4',
          'mu_heat_MT',
          'mu_heat_DH',
          'mu_heat_LT',
          'mu_el_GLS',
          'mu_CO2', 'tot_sys_cost',
          'tot_mar_cost']

var_list = x_var + y_var + y_var3

df_plot_1 = df_results.query('el_DK1_sale_el_RFNBO==@ref_el_DK1_sale_el_RFNBO & '
                             'fC_MeOH ==@ref_fC_MeOH ')[var_list].copy()

# aggregation - H2 grid Only
df_plot_1['El. connections'] = df_plot_1['El3_to_El2'] + df_plot_1['El3_to_DK1']
df_plot_1['Compressors'] = df_plot_1['CO2 compressor'] + df_plot_1['H2 compressor']
df_plot_1['CO2 storage'] = (df_plot_1['CO2 pure HP'] + df_plot_1['CO2 Liq']) / 10
df_plot_1['Heat storage'] = df_plot_1['Heat DH storage'] + df_plot_1['Concrete Heat MT storage']
df_plot_1['Renewables'] = df_plot_1['onshorewind'] + df_plot_1['solar']
df_plot_1['tot_sys_cost'] = df_plot_1['tot_sys_cost'] / 1e6
df_plot_1['tot_mar_cost'] = df_plot_1['tot_mar_cost'] / 1e6

# M€/y for cost
results_plot['tot_sys_cost'] = 'tot system cost\n(M€/y)'
results_plot['tot_mar_cost'] = 'tot marginal cost\n(M€/y)'
results_plot['tot_cap_cost'] = 'tot capital cost\n(M€/y)'

df_plot_1.rename(columns={'el_DK1_sale_el_RFNBO': results_plot['el_DK1_sale_el_RFNBO'],
                          'En_year_price': results_plot['En_year_price'],
                          'CO2_cost': results_plot['CO2_cost'],
                          'fC_MeOH': results_plot['fC_MeOH'],
                          'DH': results_plot['DH'],
                          'bioChar': results_plot['bioChar'],
                          'd_H2': results_plot['d_H2']
                          },
                 inplace=True)

var_plot_new_list1 = var_plot_new_list.copy()
var_plot_new_list1['CO2 storage'] = 'CO2 storage' + '\n' + '(10e4 kg)'

df_plot_1.rename(columns=var_plot_new_list1,
                 inplace=True)

# select some columns from df
var_plot_list = ['onshorewind', 'solar', 'El. connections', 'Electrolyzer', 'Methanol plant', 'H2 HP',
                 'battery', 'CO2 storage', 'El boiler', 'SkyClean', 'Pellets boiler', 'Heat pump']

# var_plot_list_2 = ['tot_sys_cost', 'DH', 'bioChar', 'd_H2']
var_bar_list1 = [var_plot_new_list1[v] for v in var_plot_list]
# var_bar_list2 = [results_plot[v] for v in var_plot_list_2]

# build DF for plot
n_obs = len(df_plot_1.index.values)

cap_v = []
dh_v = []
year_v = []
biochar_v = []
CO2_v = []
d_H2_v = []

shd_v = []
dh_v2 = []
year_v2 = []
biochar_v2 = []
CO2_v2 = []
d_H2_v2 = []

for ob in range(n_obs):
    data = [df_plot_1.loc[df_plot_1.index.values[ob], c] for c in var_bar_list1]
    data1 = [df_plot_1.loc[df_plot_1.index.values[ob], results_plot['DH']]]
    data2 = [df_plot_1.loc[df_plot_1.index.values[ob], results_plot['En_year_price']]]
    data3 = [df_plot_1.loc[df_plot_1.index.values[ob], results_plot['bioChar']]]
    data4 = [df_plot_1.loc[df_plot_1.index.values[ob], results_plot['CO2_cost']]]
    data5 = [df_plot_1.loc[df_plot_1.index.values[ob], results_plot['d_H2']]]
    data6 = [df_plot_1.loc[df_plot_1.index.values[ob], results_plot['d_H2']]]
    data7 = [df_plot_1.loc[df_plot_1.index.values[ob], results_plot['d_H2']]]

    cap_v = cap_v + data
    dh_v = dh_v + data1 * len(data)
    year_v = year_v + data2 * len(data)
    biochar_v = biochar_v + data3 * len(data)
    CO2_v = CO2_v + data4 * len(data)
    d_H2_v = d_H2_v + data5 * len(data)

    data_shd = [df_plot_1.loc[df_plot_1.index.values[ob], v] for v in y_var3 + []]
    shd_v = shd_v + data_shd
    dh_v2 = dh_v2 + data1 * len(data_shd)
    year_v2 = year_v2 + data2 * len(data_shd)
    biochar_v2 = biochar_v2 + data3 * len(data_shd)
    CO2_v2 = CO2_v2 + data4 * len(data_shd)
    d_H2_v2 = d_H2_v2 + data5 * len(data_shd)

    del data
    del data1
    del data2
    del data3
    del data4
    del data5
    del data_shd

df_plt_1 = pd.DataFrame()
df_plt_1['component'] = var_bar_list1 * n_obs
df_plt_1['capacity'] = cap_v
df_plt_1['DH'] = dh_v
df_plt_1['Energy year'] = year_v
df_plt_1['bioChar'] = biochar_v
df_plt_1['CO2_cost'] = CO2_v
df_plt_1['d_H2'] = d_H2_v

df_plt_2 = pd.DataFrame()
df_plt_2['shd'] = [results_plot[v] for v in y_var3] * n_obs
df_plt_2['shd val'] = shd_v
df_plt_2['DH'] = dh_v2
df_plt_2['Energy year'] = year_v2
df_plt_2['bioChar'] = biochar_v2
df_plt_2['CO2_cost'] = CO2_v2
df_plt_2['d_H2'] = d_H2_v2

# build plot - H2 grid only
data_2 = df_plt_1.query('d_H2 == 272 & DH==0 ').copy()
data_3 = df_plt_1.query('d_H2 == 272 & DH==1 ').copy()

# build plot - H2 to MeOH only
data_4 = df_plt_1.query('d_H2 == 0 & DH==0 ').copy()
data_5 = df_plt_1.query('d_H2 == 0 & DH==1 ').copy()

# Shadow prices
# build plot - H2 grid only
data_6 = df_plt_2.query('d_H2 == 272 & DH==0 ').copy()
data_7 = df_plt_2.query('d_H2 == 272 & DH==1 ').copy()

# build plot - H2 to MeOH only
data_8 = df_plt_2.query('d_H2 == 0 & DH==0 ').copy()
data_9 = df_plt_2.query('d_H2 == 0 & DH==1 ').copy()

fig, ax = plt.subplots(1, 2)
bottom_plot = sns.barplot(ax=ax[0], data=data_2, x='capacity', y='component', hue='Energy year',
                          palette='coolwarm')  # no DH
top_plot = sns.barplot(ax=ax[0], data=data_3, x='capacity', y='component', hue='Energy year', palette='seismic',
                       alpha=0.2)  # DH
top_plot = sns.barplot(ax=ax[0], data=data_3, x='capacity', y='component', hue='Energy year', palette='seismic',
                       edgecolor='black', fill=False)  # DH
ax[0].axvline(0, c='black', lw=0.3)
ax[0].grid(visible=True, axis='x')
ax[0].set_title(label='H2 to Grid')
ax[0].set(ylabel=None)
ax[0].set(xlabel=None)
ax[0].get_legend().set_visible(False)

bottom_plot = sns.barplot(ax=ax[1], data=data_4, x='capacity', y='component', hue='Energy year',
                          palette='coolwarm')  # no DH
top_plot = sns.barplot(ax=ax[1], data=data_5, x='capacity', y='component', hue='Energy year', palette='seismic',
                       alpha=0.2)  # DH
top_plot = sns.barplot(ax=ax[1], data=data_5, x='capacity', y='component', hue='Energy year', palette='seismic',
                       edgecolor='black', fill=False)  # DH
ax[1].set_title(label='H2 to MeOH')
ax[1].set(ylabel=None)
ax[1].set(xlabel=None)
ax[1].set_xlim(0, ax[0].get_xlim()[1])
ax[1].set_yticks([])
ax[1].axvline(0, c='black', lw=0.3)
ax[1].grid(visible=True, axis='x')

plt.legend(title='bioChar credits', loc='lower right', labels=['0-250 (€/tCO2)'])
plt.subplots_adjust(hspace=0.4)
plt.suptitle('Sensitivity of hub capacities to: \n DH and biochar credits with CO2 tax (0-250€/t) ', fontsize=fnt_sz)
plt.show()
plt.tight_layout()

"""PLOT 9  -  impact of DH and Biochar on optimal capacities """

fig, ax = plt.subplots(1, 2)
bottom_plot = sns.barplot(ax=ax[0], data=data_6, x='shd val', y='shd', hue='Energy year', palette='coolwarm')  # no DH
top_plot = sns.barplot(ax=ax[0], data=data_7, x='shd val', y='shd', hue='Energy year', palette='seismic',
                       alpha=0.2)  # DH
top_plot = sns.barplot(ax=ax[0], data=data_7, x='shd val', y='shd', hue='Energy year', palette='seismic',
                       edgecolor='black', fill=False)  # DH
ax[0].set_title(label='H2 to Grid')
ax[0].set(ylabel=None)
ax[0].set(xlabel=None)
ax[0].set_xlim(-80, 150)
sns.move_legend(ax[0], "lower right")
ax[0].get_legend().set_visible(False)
ax[0].axvline(0, c='black', lw=0.3)
ax[0].grid(visible=True, axis='x')

bottom_plot = sns.barplot(ax=ax[1], data=data_8, x='shd val', y='shd', hue='Energy year', palette='coolwarm')  # no DH
top_plot = sns.barplot(ax=ax[1], data=data_9, x='shd val', y='shd', hue='Energy year', palette='seismic',
                       alpha=0.2)  # DH
top_plot = sns.barplot(ax=ax[1], data=data_9, x='shd val', y='shd', hue='Energy year', palette='seismic',
                       edgecolor='black', fill=False)  # DH

ax[1].set_title(label='H2 to MeOH')
ax[1].set(ylabel=None)
ax[1].set(xlabel=None)
ax[1].set_xlim(-80, 150)
ax[1].set_yticks([])
ax[1].axvline(0, c='black', lw=0.3)
ax[1].grid(visible=True, axis='x')

plt.legend(title='bioChar credits', loc='lower right', labels=['0-250 (€/tCO2)'])
plt.subplots_adjust(hspace=0.4)
plt.suptitle('Sensitivity of hub yearly average shadow prices to : \n DH and biochar credits with CO2 tax (0-250€/t) ',
             fontsize=fnt_sz)
plt.show()

plt.tight_layout()

""" PLOT 10 - SINGLE OPTIMIZATION:  Heat maps reference cases"""

# H2 grid
h2_hm = 272
y_hm = 2019
name = str(np.squeeze(df_results.query('d_H2==@h2_hm & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
                                       '==1  & bioChar ==@ref_bioChar & '
                                       'el_DK1_sale_el_RFNBO==@ref_el_DK1_sale_el_RFNBO & En_year_price== '
                                       '@y_hm').index.values))

# import network
n_opt1 = pypsa.Network(os.path.join(data_folder, name))

key_comp_dict = {'generators': ['onshorewind', 'solar'],
                 'links': ['El3_to_DK1', 'Electrolyzer', 'Methanol plant', 'Heat pump'],
                 'stores': ['Water tank DH storage', ]}

heat_map_CF(n_opt1, key_comp_dict)

# MeOH standalone
h2_hm = 0
y_hm = 2019
name2 = str(np.squeeze(df_results.query('d_H2==@h2_hm & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
                                        '==1  & bioChar ==@ref_bioChar & '
                                        'el_DK1_sale_el_RFNBO==@ref_el_DK1_sale_el_RFNBO & En_year_price== '
                                        '@y_hm').index.values))

# import network
n_opt2 = pypsa.Network(os.path.join(data_folder, name2))

key_comp_dict = {'generators': ['onshorewind', 'solar'],
                 'links': ['El3_to_DK1', 'Electrolyzer', 'Methanol plant', 'Heat pump', 'CO2 compressor',
                           'H2 compressor'],
                 'stores': ['Water tank DH storage', 'H2 HP', 'CO2 Liq', 'battery']}

heat_map_CF(n_opt2, key_comp_dict)

""" PLOT 11  - INPUT DATA :duration curves for prices etc"""
# 2019
h2_hm = 272
y_hm = 2019
name = str(np.squeeze(df_results.query('d_H2==@h2_hm & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
                                       '==1  & bioChar ==@ref_bioChar & '
                                       'el_DK1_sale_el_RFNBO==@ref_el_DK1_sale_el_RFNBO & En_year_price== '
                                       '@y_hm').index.values))

# import network
n_opt1 = pypsa.Network(os.path.join(data_folder, name))
inputs_dict_sc1 = create_local_input_dict(2019)
inputs_dict_sc1['CO2 cost'] = ref_CO2_cost
inputs_dict_sc1['GL_eff'] = GL_eff
# 2022
h2_hm = 272
y_hm = 2022
name = str(np.squeeze(df_results.query('d_H2==@h2_hm & CO2_cost==@ref_CO2_cost & fC_MeOH ==@ref_fC_MeOH & DH '
                                       '==1  & bioChar ==@ref_bioChar & '
                                       'el_DK1_sale_el_RFNBO==@ref_el_DK1_sale_el_RFNBO & En_year_price== '
                                       '@y_hm').index.values))

# import network
n_opt2 = pypsa.Network(os.path.join(data_folder, name))
inputs_dict_sc2 = create_local_input_dict(2022)
inputs_dict_sc2['CO2 cost'] = ref_CO2_cost
inputs_dict_sc2['GL_eff'] = GL_eff

# plot time series and load duraiton curves
plot_El_Heat_prices(n_opt1, inputs_dict_sc1, tech_costs)
plot_El_Heat_prices(n_opt2, inputs_dict_sc2, tech_costs)

# plot Time series and Load duration curve for external El demand (maxRE saels)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(p.hours_in_period, n_opt1.loads_t.p_set['Grid Load'])  # , label='DK1 price + tariff + CO2 tax')

ax1.set_ylabel('MW')
ax1.grid(True)
ax1.set_title('Maximum RE sales')
ax1.tick_params(axis='x', rotation=45)

plot_duration_curve(ax2, pd.DataFrame(n_opt1.loads_t.p_set['Grid Load']), 'Grid Load')
ax2.set_ylabel('MW')
ax2.set_xlabel('h/y')
ax2.grid(True)
ax2.set_title('Max RE sales duration curve')
