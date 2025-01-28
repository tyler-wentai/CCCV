import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

print('\n\nSTART ---------------------\n')

# Dataset: UCDP/PRIO Armed Conflict Dataset version 24.1
# URL: https://ucdp.uu.se/downloads/index.html#armedconflict


conflict_data_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/UcdpPrioConflict_v24_1.csv'
df = pd.read_csv(conflict_data_path)

n_conflicts = df["conflict_id"].nunique()
print("...There are", n_conflicts, "unique conflicts.")
print("......", df.shape)

# 1. ADD ADDITIONAL COLUMNS
df["onset_lat"]             = None
df["onset_lon"]             = None
df["onset_loc"]             = None
df["onset_loc_prec"]        = None
df["onset_loc_sovereignty"] = None
df["common_name"]           = None
df["coder"]                 = None
df["coder_remarks"]         = None

# 2. REMOVE CONFLICT-YEARS WHERE THERE IS AN OBSERVATION BEFORE YEAR OF start_date2
df["start_date2"] = pd.to_datetime(df["start_date2"])
df = df[df["year"] >= df["start_date2"].dt.year]

# 3. SORT DATA BY CONFLICT AND ACTIVE YEAR
df = df.sort_values(['conflict_id', 'year'])

# 4. COMPUTE THE DIFFERENCE BETWEEN SEQUENTIAL ACTIVE YEARS
df["year_diff"] = (
    df.groupby("conflict_id")["year"]
      .diff()
      .fillna(-10)   # First year of the conflict will be coded -1
      .astype(int)
)

# 5. REMOVE ALL OBSERVATIONS WHERE THE DIFFERENCE B/W SEQUENTIAL ACTIVE YEARS FOR A CONFLICT IS 1
df_final = df[df['year_diff'] != 1]
df_final.reset_index(drop=True, inplace=True)

# 6. COMPUTE onsetX X=2,3,...9 VARIABLE COLUMNS
for i in range(2, 10):
    df_final[f"onset{i}"] = np.where(
        (df_final["year_diff"] >= i) | (df_final["year_diff"] == -10),
        1,
        0
    )
df_final["primary_onset"] = np.where((df_final["year_diff"] == -10),1,0)

# 7. SAVE FILE
df_final.to_csv('/Users/tylerbagwell/Desktop/UcdpPrioRice_GeoArmedConflictOnset_v1.csv', index=False)

print(df_final)



###### PLOT: conflictdiffyearshist.png
# plt.figure(figsize=(5.25, 4))
# sns.histplot(df['year_diff'], binwidth=1, color='tomato')
# # plt.ylim(0, (df[df['year_diff'] == -1].shape[0] + 10))
# plt.xlabel('Number of years b/w sequential active years for all conflicts')
# plt.title('UcdpPrioConflict_v24_1 (N=2686)')
# plt.text(+15, 1900, "conflict-years corresponding\nto onset coded as -10")
# # plt.savefig('/Users/tylerbagwell/Desktop/cccv_data/prelim_plots/conflictdiffyearshist.png', dpi=300)
# plt.show()

###### PLOT: onseticountsbar.png
# onset_counts = []
# for i in range(2, 10):
#     col_name = f"onset{i}"
#     onset_sum = df_final[col_name].sum()
#     onset_counts.append(onset_sum)
#     print(f"{col_name} sum: {onset_sum}")
# onset_counts.append(df_final["primary_onset"].sum())

# xx = list(range(2, 10))
# xx.append(99)

# from brokenaxes import brokenaxes
# plt.figure(figsize=(5.25, 4))
# bax = brokenaxes(xlims=((1,10), (98,100)), hspace=0.05)
# bax.bar(xx, onset_counts, color='tomato')
# bax.set_xlabel('X')
# bax.set_ylabel('Count')
# bax.set_title('OnsetX (UcdpPrioConflict_v24_1)')
# bax.axs[0].set_xticks([2,3,4,5,6,7,8,9])
# bax.axs[1].set_xticks([99])
# plt.savefig('/Users/tylerbagwell/Desktop/cccv_data/prelim_plots/onseticountsbar.png', dpi=300)
# plt.show()

