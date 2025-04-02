import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np
import sys

path1 = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/runningwindow_cindex_lag0y_Onset_Binary_Global_NINO3_square4_leq80_ratio0.60.csv'
path2 = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/runningwindow_cindex_lag0y_Onset_Binary_Global_NINO3_square4_geq80_ratio0.60.csv'
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)


import pandas as pd
import matplotlib.pyplot as plt

# Example DataFrame (replace this with your actual data)
# df = pd.read_csv('your_data.csv')

# Calculate the lower and upper error margins
# lower_errors = df1['Estimate'] - df1['Q5']
# upper_errors = df1['Q95'] - df1['Estimate']



# # Create a plot with error bars
# plt.figure(figsize=(5, 4))
# plt.axhspan(0.00048, 0.00238, color='gray', linewidth=0, alpha=0.2, label='95% credible interval (all years model)') # full model credible interval
# plt.axhline(0.00144, color='grey', linestyle='-', linewidth=1.5, label='Posterior mean (all years model)') # full model posterior mean
# plt.errorbar(df1['window_start_year'], df1['Estimate'],
#              yerr=[lower_errors, upper_errors],
#              fmt='o', capsize=5, capthick=0, linestyle='None',
#              color='red',
#              label='Posterior mean w/ 90% credible interval')
# plt.axhline(0, color='black', linestyle='-', linewidth=1)
# plt.ylim(-0.0004,0.0045)
# plt.xlabel('Window Start Year')
# plt.ylabel('Estimate')
# plt.title('Running regression (window = 44 years)')
# plt.grid(False)
# plt.legend(loc=0, frameon=False)  # This adds the legend to the plot
# plt.show()




import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
dfs = [df1, df2]
titles = [
    'Weakly teleconnected\ngrid boxes',
    'Strongly teleconnected\ngrid boxes'
]
colors = ['blue', 'red']
post_mean_full  = [+0.00011, 0.00188]
post_lower_full = [-0.00036, 0.00080]
post_upper_full = [+0.00058, 0.00297]

for i, ax in enumerate(axs):
    df = dfs[i]
    
    # Example errors — you would replace these with your actual data
    lower_errors = df['Estimate'] - df['Q5']  # replace 'Lower' with your lower CI column
    upper_errors = df['Q95'] - df['Estimate']  # replace 'Upper' with your upper CI column

    ax.axhspan(post_lower_full[i], post_upper_full[i], color='gray', linewidth=0, alpha=0.2,
               label='95% CI (all years model)' if i == 0 else "")
    ax.errorbar(df['window_start_year'], df['Estimate'],
                yerr=[lower_errors, upper_errors],
                fmt='o', capsize=2, capthick=0, linestyle='None',
                color=colors[i], ms=3, linewidth=1,
                label='Post. mean w/ 90% CI' if i == 0 else "")
    ax.axhline(post_mean_full[i], color='grey', linestyle='-', linewidth=1.5,
               label='Post. mean (all years model)' if i == 0 else "")
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylim(-0.001, 0.004)
    ax.set_xlabel('Window Start Year')
    ax.set_title(titles[i], fontsize=10)
    ax.grid(False)
    if i == 0:
        ax.set_ylabel('ENSO effect on ACR (per s.d.)')
        ax.legend(loc=0, frameon=False, fontsize=9)
plt.suptitle('ENSO effect over time, global grid4\nRunning regressions (N = 45)', fontsize=12)
plt.tight_layout()
plt.show()


