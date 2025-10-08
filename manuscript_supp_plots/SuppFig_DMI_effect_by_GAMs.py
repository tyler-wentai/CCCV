import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
import cmocean
from matplotlib.ticker import MultipleLocator

print('\n\nSTART ---------------------\n')

path_ci1 = "/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Count_Global_NINO3type2_square4.csv"
path_ci2 = "/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Count_Global_NINO3type2_square4.csv"
path_ci3 = "/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/cindex_lag0y_Onset_Binary_Global_DMI_square4.csv"

path1_h = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results/Onset_Binary_GlobalState_DMItype2_cindex_lag0y_responsive_ci90_ACR_gam.csv'
path2_h = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results/Onset_Binary_GlobalState_DMItype2_cindex_lag1y_responsive_ci90_ACR_gam.csv'

path1_l = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results/Onset_Binary_GlobalState_DMItype2_cindex_lag0y_unresponsive_ci90_ACR_gam.csv'
path2_l = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results/Onset_Binary_GlobalState_DMItype2_cindex_lag1y_unresponsive_ci90_ACR_gam.csv'

df_c1 = pd.read_csv(path_ci1)
df_c2 = pd.read_csv(path_ci2)
df_c3 = pd.read_csv(path_ci3)

df1_h = pd.read_csv(path1_h)
df2_h = pd.read_csv(path2_h)

df1_l = pd.read_csv(path1_l)
df2_l = pd.read_csv(path2_l)

index_closest1_h = df1_h['cindex_lag0y'].abs().idxmin()
index_closest2_h = df2_h['cindex_lag1y'].abs().idxmin()
val1_h = df1_h['estimate__'].iloc[index_closest1_h]
val2_h = df2_h['estimate__'].iloc[index_closest2_h]


df1_h['estimate__']     = 100*(df1_h['estimate__'])
df1_h['upper__']        = 100*(df1_h['upper__'])
df1_h['lower__']        = 100*(df1_h['lower__'])

df2_h['estimate__']     = 100*(df2_h['estimate__'])
df2_h['upper__']        = 100*(df2_h['upper__'])
df2_h['lower__']        = 100*(df2_h['lower__'])


index_closest1_l = df1_l['cindex_lag0y'].abs().idxmin()
index_closest2_l = df2_l['cindex_lag1y'].abs().idxmin()
val1_l = df1_l['estimate__'].iloc[index_closest1_l]
val2_l = df2_l['estimate__'].iloc[index_closest2_l]

df1_l['estimate__']     = 100*(df1_l['estimate__'])
df1_l['upper__']        = 100*(df1_l['upper__'])
df1_l['lower__']        = 100*(df1_l['lower__'])

df2_l['estimate__']     = 100*(df2_l['estimate__'])
df2_l['upper__']        = 100*(df2_l['upper__'])
df2_l['lower__']        = 100*(df2_l['lower__'])





cmap = sns.diverging_palette(220, 20, as_cmap=True)
num_colors = 5
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]





import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple

# assume df_ci, df2_l, df2_h are your first plot inputs
# and df_ci2, df2_l2, df2_h2 your second
# also assume `colors` is defined

fig, (ax1, ax3) = plt.subplots(1, 2,
                               figsize=(9.5, 3.75),
                               constrained_layout=True)

# -- Panel 1 (top) --
ax2 = ax1.twinx()
ax1.set_title('IOD & Conflict, State', fontsize=12)

# hist + lines
sns.histplot(x=df_c3['cindex_lag0y'], color='gainsboro',
             stat='proportion', bins=12, alpha=1.0, zorder=3,
             ax=ax1)
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df1_l,
             color='dimgray', ax=ax2)
ax2.fill_between(df1_l['cindex_lag0y'],
                 df1_l['lower__'], df1_l['upper__'],
                 color='dimgray',
                 alpha=0.45, edgecolor=None)
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df1_h,
             color='red', ax=ax2)
ax2.fill_between(df1_h['cindex_lag0y'],
                 df1_h['lower__'], df1_h['upper__'],
                 color='red',
                 alpha=0.25, edgecolor=None)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_locator(MultipleLocator(4))


# swap ticks & labels
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position("right")
ax1.set_yticks([0, 0.20, 0.40])
ax1.set_yticklabels([0, 0.20, 0.40], fontsize=10)
ax1.set_ylabel("Obs. proportion", ha='right', fontsize=10)
ax1.set_xlabel(r"DMI$_t$ ($^{\circ}$C)", fontsize=12)
ax1.tick_params(axis='y', direction='in')
ax1.yaxis.set_label_coords(1.1, 0.30)

ax2.yaxis.tick_left()
ax2.yaxis.set_label_position("left")
ax2.tick_params(axis='y', direction='in')
ax2.set_ylabel('ACR (%)', fontsize=12)

# background spans
ax1.axvspan(+0.4, +2.5, color=colors[4], linewidth=0, alpha=0.20, zorder=0)
ax1.axvspan(-0.4, +0.4, color=colors[2], linewidth=0, alpha=0.00, zorder=0)
ax1.axvspan(-2.5, -0.4, color=colors[0], linewidth=0, alpha=0.20, zorder=0)

# annotations
ax1.text(-0.04, 1.02, 'a', transform=ax1.transAxes, ha="center", va="center",
         fontsize=14, bbox=dict(boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',          # box fill color
            edgecolor='black',          # box edge color
            linewidth=0.5))             # edge line width

line_w = mlines.Line2D([], [], color='dimgray')
patch_w = mpatches.Patch(alpha=0.35, color='dimgray')
line_s = mlines.Line2D([], [], color='red')
patch_s = mpatches.Patch(alpha=0.25, color='red')
handles = [(line_s, patch_s), (line_w, patch_w)]
labels  = ['Strongly IOD teleconnected', 'Not strongly IOD teleconnected']
ax2.legend(handles=handles, labels=labels,
           handler_map={tuple: HandlerTuple(ndivide=1)},
           loc=[0.20,0.65], fontsize=10,
           title=r'', frameon=False,
           title_fontsize=10)

plt.text(0.0, 16.0, 'Neutral',    fontsize=10, ha='center')
plt.text(0.5, 16.0, '+IOD',       fontsize=10, ha='left')
plt.text(-0.5,16.0, '-IOD',       fontsize=10, ha='right')

ax1.set_ylim(0, 2.5)
ax2.set_ylim(0, 18.0)
ax2.set_xlim(-1.10, 1.10)

ax2.grid(
    linestyle='--',   # dashed lines
    linewidth=1.0,     # thinner lines
    alpha=0.1,          # a bit transparent
    color='black',
    zorder=1
)


# -- Panel 2 --
ax4 = ax3.twinx()
ax3.set_title('IOD & Conflict, State', fontsize=12)

sns.histplot(x=df_c3['cindex_lag1y'], color='gainsboro',
             stat='proportion', bins=12, alpha=1.0, zorder=3,
             ax=ax3)
sns.lineplot(x='cindex_lag1y', y='estimate__', data=df2_l,
             color='dimgray', ax=ax4)
ax4.fill_between(df2_l['cindex_lag1y'],
                 df2_l['lower__'], df2_l['upper__'],
                 color='dimgray',
                 alpha=0.45, edgecolor=None)
sns.lineplot(x='cindex_lag1y', y='estimate__', data=df2_h,
             color='red', ax=ax4)
ax4.fill_between(df2_h['cindex_lag1y'],
                 df2_h['lower__'], df2_h['upper__'],
                 color='red',
                 alpha=0.25, edgecolor=None)
ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
ax3.set_yticks([0, 0.20, 0.40])
ax3.set_yticklabels([0, 0.20, 0.40], fontsize=10)
ax3.set_ylabel("Obs. proportion", ha='right', fontsize=10)
ax3.set_xlabel(r"DMI$_{t-1}$ ($^{\circ}$C)", fontsize=12)
ax3.tick_params(axis='y', direction='in')
ax3.yaxis.set_label_coords(1.1, 0.30)

ax4.yaxis.tick_left()
ax4.yaxis.set_label_position("left")
ax4.tick_params(axis='y', direction='in')
ax4.set_ylabel('ACR (%)', fontsize=12)
from matplotlib.ticker import MultipleLocator
ax4.yaxis.set_major_locator(MultipleLocator(4))

ax3.axvspan(+0.4, +2.5, color=colors[4], linewidth=0, alpha=0.20, zorder=0)
ax3.axvspan(-0.4, +0.4, color=colors[2], linewidth=0, alpha=0.00, zorder=0)
ax3.axvspan(-2.5, -0.4, color=colors[0], linewidth=0, alpha=0.20, zorder=0)

ax3.text(-0.04, 1.02, 'b', transform=ax3.transAxes, ha="center", va="center",
         fontsize=14, bbox=dict(boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',          # box fill color
            edgecolor='black',          # box edge color
            linewidth=0.5))             # edge line width

line_w = mlines.Line2D([], [], color='dimgray')
patch_w = mpatches.Patch(alpha=0.35, color='dimgray')
line_s = mlines.Line2D([], [], color='red')
patch_s = mpatches.Patch(alpha=0.25, color='red')
handles = [(line_s, patch_s), (line_w, patch_w)]
labels  = ['Strongly IOD teleconnected', 'Not strongly IOD teleconnected']
ax4.legend(handles=handles, labels=labels,
           handler_map={tuple: HandlerTuple(ndivide=1)},
           loc=[0.20,0.65], fontsize=10,
           title=r'', frameon=False,
           title_fontsize=10)

plt.text(0.0, 16.0, 'Neutral',    fontsize=10, ha='center')
plt.text(0.5, 16.0, '+IOD',       fontsize=10, ha='left')
plt.text(-0.5,16.0, '-IOD',       fontsize=10, ha='right')

ax3.set_ylim(0, 2.5)
ax4.set_ylim(0, 18.0)
ax4.set_xlim(-1.10, 1.10)

ax4.grid(
    linestyle='--',   # dashed lines
    linewidth=1.0,     # thinner lines
    alpha=0.1,          # a bit transparent
    color='black',
    zorder=1
)

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_DMI_effect_by_GAMs.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

