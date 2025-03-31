import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys

print('\n\nSTART ---------------------\n')


# path1 = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE90pct_INDEX_lag0y_Onset_Binary_Global_ANI_country_low.csv'
# path2 = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE90pct_INDEX_lag0y_Onset_Binary_Global_ANI_country_high.csv'

# df1 = pd.read_csv(path1)
# df2 = pd.read_csv(path2)

# index_closest1 = df1['INDEX_lag0y'].abs().idxmin()
# index_closest2 = df2['INDEX_lag0y'].abs().idxmin()
# val1 = df1['estimate__'].iloc[index_closest1]
# val2 = df2['estimate__'].iloc[index_closest2]


# df1['estimate__'] = df1['estimate__'] - val1
# df1['upper__'] = df1['upper__'] - val1
# df1['lower__'] = df1['lower__'] - val1

# df2['estimate__'] = df2['estimate__'] - val2
# df2['upper__'] = df2['upper__'] - val2
# df2['lower__'] = df2['lower__'] - val2

# plt.plot(df1['INDEX_lag0y'], df1['estimate__'], color='blue', label='weakly-teleconnected countries')
# plt.fill_between(df1['INDEX_lag0y'], df1['lower__'], df1['upper__'], color='blue', alpha=0.15)

# plt.plot(df2['INDEX_lag0y'], df2['estimate__'], color='red', label='highly-teleconnected countries')
# plt.fill_between(df2['INDEX_lag0y'], df2['lower__'], df2['upper__'], color='tomato', alpha=0.20)

# plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
# plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

# plt.title('ACR and AN')
# plt.xlabel('ANI (s.d.)')
# plt.ylabel(r'Change in ACR (%) from neutral phase baseline')
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# plt.legend(frameon=False)

# plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/Onset_ANI_results.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()



#######################################
#######################################

import seaborn as sns
cmap = sns.diverging_palette(220, 20, as_cmap=True)
num_colors = 7
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]

val_min = 0
val_max = 0

path1_l = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Binary_Global_DMI_square4_low90.csv'
path2_l = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Binary_Global_DMI_square4_low95.csv'
path3_l = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Binary_Global_DMI_square4_low99.csv'

path1_h = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Binary_Global_DMI_square4_veryhigh95_quad.csv'
path2_h = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Binary_Global_DMI_square4_veryhigh95_quad.csv'
path3_h = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Binary_Global_DMI_square4_veryhigh95_quad.csv'

df1_l = pd.read_csv(path1_l)
df2_l = pd.read_csv(path2_l)
df3_l = pd.read_csv(path3_l)

df1_h = pd.read_csv(path1_h)
df2_h = pd.read_csv(path2_h)
df3_h = pd.read_csv(path3_h)

index_closest1_l = df1_l['cindex_lag0y'].abs().idxmin()
index_closest2_l = df2_l['cindex_lag0y'].abs().idxmin()
# index_closest3 = df3['cindex_lag0y'].abs().idxmin()
val1_l = df1_l['estimate__'].iloc[index_closest1_l]
val2_l = df2_l['estimate__'].iloc[index_closest2_l]
# val3 = df3['estimate__'].iloc[index_closest3]

index_closest1_h = df1_h['cindex_lag0y'].abs().idxmin()
index_closest2_h = df2_h['cindex_lag0y'].abs().idxmin()
# index_closest3 = df3['cindex_lag0y'].abs().idxmin()
val1_h = df1_h['estimate__'].iloc[index_closest1_h]
val2_h = df2_h['estimate__'].iloc[index_closest2_h]
# val3 = df3['estimate__'].iloc[index_closest3]


df1_l['estimate__'] = df1_l['estimate__'] - val1_l
df1_l['upper__'] = df1_l['upper__'] - val1_l
df1_l['lower__'] = df1_l['lower__'] - val1_l

df2_l['estimate__'] = df2_l['estimate__'] - val2_l
df2_l['upper__'] = df2_l['upper__'] - val2_l
df2_l['lower__'] = df2_l['lower__'] - val2_l

# df3['estimate__'] = df3['estimate__'] - val3
# df3['upper__'] = df3['upper__'] - val3
# df3['lower__'] = df3['lower__'] - val3

df1_h['estimate__'] = df1_h['estimate__'] - val1_h
df1_h['upper__'] = df1_h['upper__'] - val1_h
df1_h['lower__'] = df1_h['lower__'] - val1_h

df2_h['estimate__'] = df2_h['estimate__'] - val2_h
df2_h['upper__'] = df2_h['upper__'] - val2_h
df2_h['lower__'] = df2_h['lower__'] - val2_h

# df3['estimate__'] = df3['estimate__'] - val3
# df3['upper__'] = df3['upper__'] - val3
# df3['lower__'] = df3['lower__'] - val3

# maroon, navy
plt.figure(figsize=(5, 4))
# plt.plot(df1_l['cindex_lag0y'], df1_l['estimate__'], color='navy', label='Weakly, posterior mean', linewidth=2.0)
# plt.fill_between(df1_l['cindex_lag0y'], df1_l['lower__'], df1_l['upper__'], color='b', alpha=0.30, edgecolor=None, label='likely (90% CI)')
# plt.fill_between(df2_l['cindex_lag0y'], df2_l['lower__'], df2_l['upper__'], color='b', alpha=0.20, edgecolor=None)#, label='very likely (95% CI)')
# plt.fill_between(df3['cindex_lag0y'], df3['lower__'], df3['upper__'], color='r', alpha=0.15, edgecolor=None, label='virtually certain (99% CI)')

plt.plot(df1_h['cindex_lag0y'], df1_h['estimate__'], color='maroon', label='Strongly, posterior mean', linewidth=2.0)
# plt.fill_between(df1_h['cindex_lag0y'], df1_h['lower__'], df1_h['upper__'], color='r', alpha=0.30, edgecolor=None, label='likely (90% CI)')
plt.fill_between(df2_h['cindex_lag0y'], df2_h['lower__'], df2_h['upper__'], color='r', alpha=0.20, edgecolor=None)#, label='very likely (95% CI)')
# plt.fill_between(df3['cindex_lag0y'], df3['lower__'], df3['upper__'], color='r', alpha=0.15, edgecolor=None, label='virtually certain (99% CI)')

# import matplotlib.lines as mlines
# import matplotlib.patches as mpatches
# gray_line_high = mlines.Line2D([], [], color='0.4', linewidth=2.0, label='posterior mean')
# gray_patch_high = mpatches.Patch(facecolor='0.7', edgecolor='0.5', alpha=0.33, label='95% CI')

# # Combine the handles in a list; adjust the list order to control the legend ordering
# handles = [gray_line_high, gray_patch_high]

# plt.legend(handles=handles, loc=2, frameon=False)


plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, zorder=0)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, zorder=0)

plt.title('ACR for very strongly teleconnected\n Global grid boxes (N=70)')
plt.xlabel('Dipole Mode Index (s.d.)', fontsize=12)
plt.ylabel(r'$\Delta$ ACR from neutral phase', fontsize=12)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.legend(loc=2, frameon=False)

# plt.axvspan(+2.5, +3.5, color=colors[6], alpha=0.25, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(+1.5, +2.5, color=colors[5], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(+0.5, +1.5, color=colors[4], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(-0.5, +0.5, color=colors[3], alpha=0.00, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(-1.5, -0.5, color=colors[2], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(-2.5, -1.5, color=colors[1], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(-2.5, -3.5, color=colors[0], alpha=0.25, edgecolor='none', linewidth=0.0, zorder=0)

plt.text(0.0, -0.0055, 'Neutral', fontsize=9, color='k', horizontalalignment='center')
plt.text(-1., -0.0055, 'Moderate', fontsize=9, color='k', horizontalalignment='center')
plt.text(+1., -0.0055, 'Moderate', fontsize=9, color='k', horizontalalignment='center')
plt.text(-2., -0.0055, 'Strong', fontsize=9, color='k', horizontalalignment='center')
plt.text(+2., -0.0055, 'Strong', fontsize=9, color='k', horizontalalignment='center')
# plt.text(+3., -0.0055, 'Very\nStrong', fontsize=9, color='k', horizontalalignment='center')

plt.xlim(-3.0,2.5)
plt.ylim(-0.003,0.01)

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/Onset_DMI_Global_verystrongly_square4.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
sys.exit()


#######################################
#######################################
# import seaborn as sns
# cmap = sns.diverging_palette(220, 20, as_cmap=True)
# num_colors = 5
# levels = np.linspace(0, 1, num_colors)
# colors = [cmap(level) for level in levels]
# print(colors)

# path_A = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE90pct_INDEX_lag0y_Onset_Binary_Global_NINO3_country_high90.csv'
# path_B = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE90pct_INDEX_lag0y_Onset_Binary_Global_DMI_country_high90.csv'
# path_C = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE90pct_INDEX_lag0y_Onset_Binary_Global_ANI_country_high90.csv'

# df_A = pd.read_csv(path_A)
# df_B = pd.read_csv(path_B)
# df_C = pd.read_csv(path_C)

# index_closest_A = df_A['INDEX_lag0y'].abs().idxmin()
# index_closest_B = df_B['INDEX_lag0y'].abs().idxmin()
# index_closest_C = df_C['INDEX_lag0y'].abs().idxmin()
# val_A = df_A['estimate__'].iloc[index_closest_A]
# val_B = df_B['estimate__'].iloc[index_closest_B]
# val_C = df_C['estimate__'].iloc[index_closest_C]

# df_A['estimate__'] = df_A['estimate__'] - val_A
# df_A['upper__']    = df_A['upper__'] - val_A
# df_A['lower__']    = df_A['lower__'] - val_A

# df_B['estimate__'] = df_B['estimate__'] - val_B
# df_B['upper__']    = df_B['upper__'] - val_B
# df_B['lower__']    = df_B['lower__'] - val_B

# df_C['estimate__'] = df_C['estimate__'] - val_C
# df_C['upper__']    = df_C['upper__'] - val_C
# df_C['lower__']    = df_C['lower__'] - val_C


# Ap2_estimate = df_A.loc[(df_A['INDEX_lag0y'] - (+2)).abs().idxmin(), 'estimate__']
# Bp2_estimate = df_B.loc[(df_B['INDEX_lag0y'] - (+2)).abs().idxmin(), 'estimate__']
# Cp2_estimate = df_C.loc[(df_C['INDEX_lag0y'] - (+2)).abs().idxmin(), 'estimate__']

# Ap1_estimate = df_A.loc[(df_A['INDEX_lag0y'] - (+1)).abs().idxmin(), 'estimate__']
# Bp1_estimate = df_B.loc[(df_B['INDEX_lag0y'] - (+1)).abs().idxmin(), 'estimate__']
# Cp1_estimate = df_C.loc[(df_C['INDEX_lag0y'] - (+1)).abs().idxmin(), 'estimate__']

# A0_estimate = df_A.loc[(df_A['INDEX_lag0y'] - (+0)).abs().idxmin(), 'estimate__']
# B0_estimate = df_B.loc[(df_B['INDEX_lag0y'] - (+0)).abs().idxmin(), 'estimate__']
# C0_estimate = df_C.loc[(df_C['INDEX_lag0y'] - (+0)).abs().idxmin(), 'estimate__']

# Am1_estimate = df_A.loc[(df_A['INDEX_lag0y'] - (-1)).abs().idxmin(), 'estimate__']
# Bm1_estimate = df_B.loc[(df_B['INDEX_lag0y'] - (-1)).abs().idxmin(), 'estimate__']
# Cm1_estimate = df_C.loc[(df_C['INDEX_lag0y'] - (-1)).abs().idxmin(), 'estimate__']

# Am2_estimate = df_A.loc[(df_A['INDEX_lag0y'] - (-2)).abs().idxmin(), 'estimate__']
# Bm2_estimate = df_B.loc[(df_B['INDEX_lag0y'] - (-2)).abs().idxmin(), 'estimate__']
# Cm2_estimate = df_C.loc[(df_C['INDEX_lag0y'] - (-2)).abs().idxmin(), 'estimate__']

# Ap2_lower = df_A.loc[(df_A['INDEX_lag0y'] - (+2)).abs().idxmin(), 'lower__']
# Bp2_lower = df_B.loc[(df_B['INDEX_lag0y'] - (+2)).abs().idxmin(), 'lower__']
# Cp2_lower = df_C.loc[(df_C['INDEX_lag0y'] - (+2)).abs().idxmin(), 'lower__']

# Ap1_lower = df_A.loc[(df_A['INDEX_lag0y'] - (+1)).abs().idxmin(), 'lower__']
# Bp1_lower = df_B.loc[(df_B['INDEX_lag0y'] - (+1)).abs().idxmin(), 'lower__']
# Cp1_lower = df_C.loc[(df_C['INDEX_lag0y'] - (+1)).abs().idxmin(), 'lower__']

# A0_lower = df_A.loc[(df_A['INDEX_lag0y'] - (+0)).abs().idxmin(), 'lower__']
# B0_lower = df_B.loc[(df_B['INDEX_lag0y'] - (+0)).abs().idxmin(), 'lower__']
# C0_lower = df_C.loc[(df_C['INDEX_lag0y'] - (+0)).abs().idxmin(), 'lower__']

# Am1_lower = df_A.loc[(df_A['INDEX_lag0y'] - (-1)).abs().idxmin(), 'lower__']
# Bm1_lower = df_B.loc[(df_B['INDEX_lag0y'] - (-1)).abs().idxmin(), 'lower__']
# Cm1_lower = df_C.loc[(df_C['INDEX_lag0y'] - (-1)).abs().idxmin(), 'lower__']

# Am2_lower = df_A.loc[(df_A['INDEX_lag0y'] - (-2)).abs().idxmin(), 'lower__']
# Bm2_lower = df_B.loc[(df_B['INDEX_lag0y'] - (-2)).abs().idxmin(), 'lower__']
# Cm2_lower = df_C.loc[(df_C['INDEX_lag0y'] - (-2)).abs().idxmin(), 'lower__']

# Ap2_upper = df_A.loc[(df_A['INDEX_lag0y'] - (+2)).abs().idxmin(), 'upper__']
# Bp2_upper = df_B.loc[(df_B['INDEX_lag0y'] - (+2)).abs().idxmin(), 'upper__']
# Cp2_upper = df_C.loc[(df_C['INDEX_lag0y'] - (+2)).abs().idxmin(), 'upper__']

# Ap1_upper = df_A.loc[(df_A['INDEX_lag0y'] - (+1)).abs().idxmin(), 'upper__']
# Bp1_upper = df_B.loc[(df_B['INDEX_lag0y'] - (+1)).abs().idxmin(), 'upper__']
# Cp1_upper = df_C.loc[(df_C['INDEX_lag0y'] - (+1)).abs().idxmin(), 'upper__']

# A0_upper = df_A.loc[(df_A['INDEX_lag0y'] - (+0)).abs().idxmin(), 'upper__']
# B0_upper = df_B.loc[(df_B['INDEX_lag0y'] - (+0)).abs().idxmin(), 'upper__']
# C0_upper = df_C.loc[(df_C['INDEX_lag0y'] - (+0)).abs().idxmin(), 'upper__']

# Am1_upper = df_A.loc[(df_A['INDEX_lag0y'] - (-1)).abs().idxmin(), 'upper__']
# Bm1_upper = df_B.loc[(df_B['INDEX_lag0y'] - (-1)).abs().idxmin(), 'upper__']
# Cm1_upper = df_C.loc[(df_C['INDEX_lag0y'] - (-1)).abs().idxmin(), 'upper__']

# Am2_upper = df_A.loc[(df_A['INDEX_lag0y'] - (-2)).abs().idxmin(), 'upper__']
# Bm2_upper = df_B.loc[(df_B['INDEX_lag0y'] - (-2)).abs().idxmin(), 'upper__']
# Cm2_upper = df_C.loc[(df_C['INDEX_lag0y'] - (-2)).abs().idxmin(), 'upper__']



# A_est = [Am2_estimate, Am1_estimate, A0_estimate, Ap1_estimate, Ap2_estimate]
# B_est = [Bm2_estimate, Bm1_estimate, B0_estimate, Bp1_estimate, Bp2_estimate]
# C_est = [Cm2_estimate, Cm1_estimate, C0_estimate, Cp1_estimate, Cp2_estimate]

# xA = [-2.1,-1.1,-0.1,0.9,1.90]
# xB = [-2,-1,0,1,2]
# xC = [-1.9,-0.9,0.1,1.1,2.1]

# col1 = 'gray'
# col2 = 'black'
# col3 = 'red'

# plt.scatter(xA, A_est, color=col1, linewidth=2.0, marker='o', zorder=2, label='ENSO')
# plt.scatter(xB, B_est, color=col2, linewidth=2.0, marker='v', zorder=2, label='IOD')
# plt.scatter(xC, C_est, color=col3, linewidth=2.0, marker='s', zorder=2, label='A. Nino')

# plt.plot(xA, A_est, color=col1, linewidth=2.0, marker='o', zorder=0, alpha=0.1)
# plt.plot(xB, B_est, color=col2, linewidth=2.0, marker='v', zorder=0, alpha=0.1)
# plt.plot(xC, C_est, color=col3, linewidth=2.0, marker='s', zorder=0, alpha=0.1)

# plt.plot([xA[0],xA[0]], [Am2_lower, Am2_upper], color=col1, linewidth=2.0, zorder=1)
# plt.plot([xA[1],xA[1]], [Am1_lower, Am1_upper], color=col1, linewidth=2.0, zorder=1)
# plt.plot([xA[2],xA[2]], [A0_lower, A0_upper], color=col1, linewidth=2.0, zorder=1)
# plt.plot([xA[3],xA[3]], [Ap1_lower, Ap1_upper], color=col1, linewidth=2.0, zorder=1)
# plt.plot([xA[4],xA[4]], [Ap2_lower, Ap2_upper], color=col1, linewidth=2.0, zorder=1)

# plt.plot([xB[0],xB[0]], [Bm2_lower, Bm2_upper], color=col2, linewidth=2.0, zorder=1)
# plt.plot([xB[1],xB[1]], [Bm1_lower, Bm1_upper], color=col2, linewidth=2.0, zorder=1)
# plt.plot([xB[2],xB[2]], [B0_lower, B0_upper], color=col2, linewidth=2.0, zorder=1)
# plt.plot([xB[3],xB[3]], [Bp1_lower, Bp1_upper], color=col2, linewidth=2.0, zorder=1)
# plt.plot([xB[4],xB[4]], [Bp2_lower, Bp2_upper], color=col2, linewidth=2.0, zorder=1)

# plt.plot([xC[0],xC[0]], [Cm2_lower, Cm2_upper], color=col3, linewidth=2.0, zorder=1)
# plt.plot([xC[1],xC[1]], [Cm1_lower, Cm1_upper], color=col3, linewidth=2.0, zorder=1)
# plt.plot([xC[2],xC[2]], [C0_lower, C0_upper], color=col3, linewidth=2.0, zorder=1)
# plt.plot([xC[3],xC[3]], [Cp1_lower, Cp1_upper], color=col3, linewidth=2.0, zorder=1)
# plt.plot([xC[4],xC[4]], [Cp2_lower, Cp2_upper], color=col3, linewidth=2.0, zorder=1)

# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, zorder=0)
# plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, zorder=0)
# plt.legend(loc=2, frameon=False)
# plt.xlabel('Mode index (s.d.)', fontsize=12)
# plt.ylabel(r'Change in ACR from neutral phase baseline', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)

# plt.axvspan(+1.5, +2.5, color=colors[4], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
# plt.axvspan(+0.5, +1.5, color=colors[3], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)
# plt.axvspan(-0.5, +0.5, color=colors[2], alpha=0.00, edgecolor='none', linewidth=0.0, zorder=0)
# plt.axvspan(-1.5, -0.5, color=colors[1], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)
# plt.axvspan(-2.5, -1.5, color=colors[0], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)

# plt.text(0.0, -0.02, 'Neutral', fontsize=10, color='k', horizontalalignment='center')
# plt.text(-1., -0.02, 'Moderate', fontsize=10, color='k', horizontalalignment='center')
# plt.text(+1., -0.02, 'Moderate', fontsize=10, color='k', horizontalalignment='center')
# plt.text(-2., -0.02, 'Strong', fontsize=10, color='k', horizontalalignment='center')
# plt.text(+2., -0.02, 'Strong', fontsize=10, color='k', horizontalalignment='center')

# plt.text(+1., +0.055, r'$90\%$ credible interval', fontsize=10, color='k', horizontalalignment='center')

# plt.xlim(-2.5,2.5)
# plt.title('Climate mode effects on conflict onset\nfor strongly-teleconnected countries', horizontalalignment='center')

# plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/panel_datasets/results/Onset_intermode_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()




#######################################
#######################################

import seaborn as sns
cmap = sns.diverging_palette(220, 20, as_cmap=True)
num_colors = 5
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]
print(colors)

path_A = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Binary_Global_NINO3_square4_low95.csv'
path_B = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Binary_Global_NINO3_square4_mod95.csv'
path_C = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Binary_Global_NINO3_square4_high95.csv'

df_A = pd.read_csv(path_A)
df_B = pd.read_csv(path_B)
df_C = pd.read_csv(path_C)

index_closest_A = df_A['cindex_lag0y'].abs().idxmin()
index_closest_B = df_B['cindex_lag0y'].abs().idxmin()
index_closest_C = df_C['cindex_lag0y'].abs().idxmin()
val_A = df_A['estimate__'].iloc[index_closest_A]
val_B = df_B['estimate__'].iloc[index_closest_B]
val_C = df_C['estimate__'].iloc[index_closest_C]

df_A['estimate__'] = df_A['estimate__'] - val_A
df_A['upper__']    = df_A['upper__'] - val_A
df_A['lower__']    = df_A['lower__'] - val_A

df_B['estimate__'] = df_B['estimate__'] - val_B
df_B['upper__']    = df_B['upper__'] - val_B
df_B['lower__']    = df_B['lower__'] - val_B

df_C['estimate__'] = df_C['estimate__'] - val_C
df_C['upper__']    = df_C['upper__'] - val_C
df_C['lower__']    = df_C['lower__'] - val_C


Ap2_estimate = df_A.loc[(df_A['cindex_lag0y'] - (+2)).abs().idxmin(), 'estimate__']
Bp2_estimate = df_B.loc[(df_B['cindex_lag0y'] - (+2)).abs().idxmin(), 'estimate__']
Cp2_estimate = df_C.loc[(df_C['cindex_lag0y'] - (+2)).abs().idxmin(), 'estimate__']

Ap1_estimate = df_A.loc[(df_A['cindex_lag0y'] - (+1)).abs().idxmin(), 'estimate__']
Bp1_estimate = df_B.loc[(df_B['cindex_lag0y'] - (+1)).abs().idxmin(), 'estimate__']
Cp1_estimate = df_C.loc[(df_C['cindex_lag0y'] - (+1)).abs().idxmin(), 'estimate__']

A0_estimate = df_A.loc[(df_A['cindex_lag0y'] - (+0)).abs().idxmin(), 'estimate__']
B0_estimate = df_B.loc[(df_B['cindex_lag0y'] - (+0)).abs().idxmin(), 'estimate__']
C0_estimate = df_C.loc[(df_C['cindex_lag0y'] - (+0)).abs().idxmin(), 'estimate__']

Am1_estimate = df_A.loc[(df_A['cindex_lag0y'] - (-1)).abs().idxmin(), 'estimate__']
Bm1_estimate = df_B.loc[(df_B['cindex_lag0y'] - (-1)).abs().idxmin(), 'estimate__']
Cm1_estimate = df_C.loc[(df_C['cindex_lag0y'] - (-1)).abs().idxmin(), 'estimate__']

Am2_estimate = df_A.loc[(df_A['cindex_lag0y'] - (-2)).abs().idxmin(), 'estimate__']
Bm2_estimate = df_B.loc[(df_B['cindex_lag0y'] - (-2)).abs().idxmin(), 'estimate__']
Cm2_estimate = df_C.loc[(df_C['cindex_lag0y'] - (-2)).abs().idxmin(), 'estimate__']

Ap2_lower = df_A.loc[(df_A['cindex_lag0y'] - (+2)).abs().idxmin(), 'lower__']
Bp2_lower = df_B.loc[(df_B['cindex_lag0y'] - (+2)).abs().idxmin(), 'lower__']
Cp2_lower = df_C.loc[(df_C['cindex_lag0y'] - (+2)).abs().idxmin(), 'lower__']

Ap1_lower = df_A.loc[(df_A['cindex_lag0y'] - (+1)).abs().idxmin(), 'lower__']
Bp1_lower = df_B.loc[(df_B['cindex_lag0y'] - (+1)).abs().idxmin(), 'lower__']
Cp1_lower = df_C.loc[(df_C['cindex_lag0y'] - (+1)).abs().idxmin(), 'lower__']

A0_lower = df_A.loc[(df_A['cindex_lag0y'] - (+0)).abs().idxmin(), 'lower__']
B0_lower = df_B.loc[(df_B['cindex_lag0y'] - (+0)).abs().idxmin(), 'lower__']
C0_lower = df_C.loc[(df_C['cindex_lag0y'] - (+0)).abs().idxmin(), 'lower__']

Am1_lower = df_A.loc[(df_A['cindex_lag0y'] - (-1)).abs().idxmin(), 'lower__']
Bm1_lower = df_B.loc[(df_B['cindex_lag0y'] - (-1)).abs().idxmin(), 'lower__']
Cm1_lower = df_C.loc[(df_C['cindex_lag0y'] - (-1)).abs().idxmin(), 'lower__']

Am2_lower = df_A.loc[(df_A['cindex_lag0y'] - (-2)).abs().idxmin(), 'lower__']
Bm2_lower = df_B.loc[(df_B['cindex_lag0y'] - (-2)).abs().idxmin(), 'lower__']
Cm2_lower = df_C.loc[(df_C['cindex_lag0y'] - (-2)).abs().idxmin(), 'lower__']

Ap2_upper = df_A.loc[(df_A['cindex_lag0y'] - (+2)).abs().idxmin(), 'upper__']
Bp2_upper = df_B.loc[(df_B['cindex_lag0y'] - (+2)).abs().idxmin(), 'upper__']
Cp2_upper = df_C.loc[(df_C['cindex_lag0y'] - (+2)).abs().idxmin(), 'upper__']

Ap1_upper = df_A.loc[(df_A['cindex_lag0y'] - (+1)).abs().idxmin(), 'upper__']
Bp1_upper = df_B.loc[(df_B['cindex_lag0y'] - (+1)).abs().idxmin(), 'upper__']
Cp1_upper = df_C.loc[(df_C['cindex_lag0y'] - (+1)).abs().idxmin(), 'upper__']

A0_upper = df_A.loc[(df_A['cindex_lag0y'] - (+0)).abs().idxmin(), 'upper__']
B0_upper = df_B.loc[(df_B['cindex_lag0y'] - (+0)).abs().idxmin(), 'upper__']
C0_upper = df_C.loc[(df_C['cindex_lag0y'] - (+0)).abs().idxmin(), 'upper__']

Am1_upper = df_A.loc[(df_A['cindex_lag0y'] - (-1)).abs().idxmin(), 'upper__']
Bm1_upper = df_B.loc[(df_B['cindex_lag0y'] - (-1)).abs().idxmin(), 'upper__']
Cm1_upper = df_C.loc[(df_C['cindex_lag0y'] - (-1)).abs().idxmin(), 'upper__']

Am2_upper = df_A.loc[(df_A['cindex_lag0y'] - (-2)).abs().idxmin(), 'upper__']
Bm2_upper = df_B.loc[(df_B['cindex_lag0y'] - (-2)).abs().idxmin(), 'upper__']
Cm2_upper = df_C.loc[(df_C['cindex_lag0y'] - (-2)).abs().idxmin(), 'upper__']



A_est = [Am2_estimate, Am1_estimate, A0_estimate, Ap1_estimate, Ap2_estimate]
B_est = [Bm2_estimate, Bm1_estimate, B0_estimate, Bp1_estimate, Bp2_estimate]
C_est = [Cm2_estimate, Cm1_estimate, C0_estimate, Cp1_estimate, Cp2_estimate]

xA = [-2.1,-1.1,-0.1,0.9,1.90]
xB = [-2,-1,0,1,2]
xC = [-1.9,-0.9,0.1,1.1,2.1]

col1 = 'green'
col2 = 'orange'
col3 = 'red'

plt.figure(figsize=(5, 4))

plt.scatter(xA, A_est, color=col1, linewidth=2.0, marker='o', zorder=2, label='Weak')
plt.scatter(xB, B_est, color=col2, linewidth=2.0, marker='v', zorder=2, label='Moderate')
plt.scatter(xC, C_est, color=col3, linewidth=2.0, marker='s', zorder=2, label='Strong')

plt.plot(xA, A_est, color=col1, linewidth=2.0, marker='o', zorder=0, alpha=0.1)
plt.plot(xB, B_est, color=col2, linewidth=2.0, marker='v', zorder=0, alpha=0.1)
plt.plot(xC, C_est, color=col3, linewidth=2.0, marker='s', zorder=0, alpha=0.1)

plt.plot([xA[0],xA[0]], [Am2_lower, Am2_upper], color=col1, linewidth=2.0, zorder=1)
plt.plot([xA[1],xA[1]], [Am1_lower, Am1_upper], color=col1, linewidth=2.0, zorder=1)
plt.plot([xA[2],xA[2]], [A0_lower, A0_upper], color=col1, linewidth=2.0, zorder=1)
plt.plot([xA[3],xA[3]], [Ap1_lower, Ap1_upper], color=col1, linewidth=2.0, zorder=1)
plt.plot([xA[4],xA[4]], [Ap2_lower, Ap2_upper], color=col1, linewidth=2.0, zorder=1)

plt.plot([xB[0],xB[0]], [Bm2_lower, Bm2_upper], color=col2, linewidth=2.0, zorder=1)
plt.plot([xB[1],xB[1]], [Bm1_lower, Bm1_upper], color=col2, linewidth=2.0, zorder=1)
plt.plot([xB[2],xB[2]], [B0_lower, B0_upper], color=col2, linewidth=2.0, zorder=1)
plt.plot([xB[3],xB[3]], [Bp1_lower, Bp1_upper], color=col2, linewidth=2.0, zorder=1)
plt.plot([xB[4],xB[4]], [Bp2_lower, Bp2_upper], color=col2, linewidth=2.0, zorder=1)

plt.plot([xC[0],xC[0]], [Cm2_lower, Cm2_upper], color=col3, linewidth=2.0, zorder=1)
plt.plot([xC[1],xC[1]], [Cm1_lower, Cm1_upper], color=col3, linewidth=2.0, zorder=1)
plt.plot([xC[2],xC[2]], [C0_lower, C0_upper], color=col3, linewidth=2.0, zorder=1)
plt.plot([xC[3],xC[3]], [Cp1_lower, Cp1_upper], color=col3, linewidth=2.0, zorder=1)
plt.plot([xC[4],xC[4]], [Cp2_lower, Cp2_upper], color=col3, linewidth=2.0, zorder=1)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, zorder=0)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, zorder=0)
plt.legend(loc=2, frameon=False, title=r'Teleconnection $\Psi$')
plt.xlabel('NINO3 Index (s.d.)', fontsize=12)
plt.ylabel(r'$\Delta$ ACR from neutral phase', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.axvspan(+1.5, +2.5, color=colors[4], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(+0.5, +1.5, color=colors[3], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(-0.5, +0.5, color=colors[2], alpha=0.00, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(-1.5, -0.5, color=colors[1], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(-2.5, -1.5, color=colors[0], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)

plt.text(0.0, -0.0046, 'Neutral', fontsize=10, color='k', horizontalalignment='center')
plt.text(-1., -0.0046, 'Moderate', fontsize=10, color='k', horizontalalignment='center')
plt.text(+1., -0.0046, 'Moderate', fontsize=10, color='k', horizontalalignment='center')
plt.text(-2., -0.0046, 'Strong', fontsize=10, color='k', horizontalalignment='center')
plt.text(+2., -0.0046, 'Strong', fontsize=10, color='k', horizontalalignment='center')

plt.text(+1., +0.004, r'$95\%$ credible intervals', fontsize=8, color='k', horizontalalignment='center')

plt.xlim(-2.5,2.5)
plt.title('ENSO effects on conflict onset\nfor global grid boxes', horizontalalignment='center')

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/panel_datasets/results/Onset_Global_NINO3_square4.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()