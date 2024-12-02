import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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



#############

path1 = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_INDEX_lag0y_Onset_Binary_Asia_ANI_country_high66.csv'
path2 = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_INDEX_lag0y_Onset_Binary_Asia_ANI_country_high90.csv'
path3 = '/Users/tylerbagwell/Desktop/panel_datasets/results/CE_INDEX_lag0y_Onset_Binary_Asia_ANI_country_high99.csv'

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)

index_closest1 = df1['INDEX_lag0y'].abs().idxmin()
index_closest2 = df2['INDEX_lag0y'].abs().idxmin()
index_closest3 = df3['INDEX_lag0y'].abs().idxmin()
val1 = df1['estimate__'].iloc[index_closest1]
val2 = df2['estimate__'].iloc[index_closest2]
val3 = df3['estimate__'].iloc[index_closest3]


df1['estimate__'] = df1['estimate__'] - val1
df1['upper__'] = df1['upper__'] - val1
df1['lower__'] = df1['lower__'] - val1

df2['estimate__'] = df2['estimate__'] - val2
df2['upper__'] = df2['upper__'] - val2
df2['lower__'] = df2['lower__'] - val2

df3['estimate__'] = df3['estimate__'] - val3
df3['upper__'] = df3['upper__'] - val3
df3['lower__'] = df3['lower__'] - val3

# maroon, navy
plt.plot(df1['INDEX_lag0y'], df1['estimate__'], color='maroon', label='posterior estimate', linewidth=2.0)
plt.fill_between(df1['INDEX_lag0y'], df1['lower__'], df1['upper__'], color='r', alpha=0.30, edgecolor=None, label='likely (66% CI)')
plt.fill_between(df2['INDEX_lag0y'], df2['lower__'], df2['upper__'], color='r', alpha=0.22, edgecolor=None, label='very likely (90% CI)')
plt.fill_between(df3['INDEX_lag0y'], df3['lower__'], df3['upper__'], color='r', alpha=0.15, edgecolor=None, label='virtually certain (99% CI)')

plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, zorder=0)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, zorder=0)

plt.title('ACR of STRONGLY teleconnected countries in ASIA with AN (N=70)')
plt.xlabel('ANI (s.d.)')
plt.ylabel(r'Change in ACR from neutral phase baseline')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.legend(loc=0, frameon=False)

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/Onset_ANI_Asia_high.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()