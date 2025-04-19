import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from calc_annual_index import *

print('\n\nSTART ---------------------\n')


########### NINO3

# start_year  = 1950
# end_year    = 2024 # need to add one more year

# cindex1 = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
#                        start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                        end_date=datetime(end_year, 12, 1, 0, 0, 0))

# cindex1.index = pd.to_datetime(cindex1.index)
# cindex1['Year']  = cindex1.index.year
# cindex1['Month'] = cindex1.index.month

# cindex1_monthly = cindex1.pivot(index='Year', columns='Month', values='ANOM')
# for m in range(1, 8):                       # months 1‑4
#     cindex1_monthly[int(f'{m+12}')] = cindex1_monthly[m].shift(-1)
# cindex1_monthly = cindex1_monthly.iloc[:-1]

# print(cindex1_monthly)

# month_mapping = {
#     1: r'Jan$_t$', 2: r'Feb$_t$', 3: r'Mar$_t$', 4: r'Apr$_t$',
#     5: r'May$_t$', 6: r'Jun$_t$', 7: r'Jul$_t$', 8: r'Aug$_t$',
#     9: r'Sep$_t$', 10: r'Oct$_t$', 11: r'Nov$_t$', 12: r'Dec$_t$',
#     13: r'Jan$_{t+1}$', 14: r'Feb$_{t+1}$', 15: r'Mar$_{t+1}$', 16: r'Apr$_{t+1}$',
#     17: r'May$_{t+1}$', 18: r'Jun$_{t+1}$', 19: r'Jul$_{t+1}$'
# }

# cindex1_corr = cindex1_monthly.corr()
# cindex1_corr.rename(index=month_mapping, columns=month_mapping, inplace=True)
# print(len(cindex1_monthly))
# avg_abs1 = cindex1_monthly.abs().sum() / len(cindex1_monthly)

# print(avg_abs1)


# df = avg_abs1.reset_index().rename(columns={"index": "Month", 0: "Value"})
# df = df[df["Month"].between(1, 12)]
# df["MonthName"] = df["Month"].map(month_mapping)

# # PLOTTING
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10), 
#                                gridspec_kw={"height_ratios": [1, 0.15],
#                                               "hspace": -0.025})

# plt.suptitle('NINO3', fontsize=11, y=0.925)

# sns.heatmap(
#     cindex1_corr,
#     annot=True, fmt=".2f",
#     cmap="PuOr", vmin=-1, vmax=1, center=0,
#     linewidths=0.5, linecolor="white", square=True,
#     annot_kws={"size": 6},
#     cbar_kws={
#         "orientation": "horizontal",
#         "shrink": 0.7,
#         "pad": 0.11    # distance between heatmap and colorbar
#     },
#     ax=ax1
# )
# ax1.set_title("Monthly Correlations (1950-2023)", fontsize=10)
# ax1.set_xlabel("", fontsize=10)
# ax1.set_ylabel("", fontsize=10)

# sns.set_theme(style="darkgrid")
# sns.lineplot(
#     data=df, x="MonthName", y="Value",
#     marker="o", linewidth=2, color="k",
#     ax=ax2
# )

# ax2.set_title("(1950-2023)", fontsize=10)
# ax2.set(
#     xlabel="Month",
#     ylabel=r"Avg. abs. NINO3 ($^\circ$C)",
# )
# ax2.grid(
#     True,                # enable grid
#     which="major",        # both major and minor ticks
#     linestyle="--",      # dashed grid lines
#     linewidth=0.5        # thinner line width
# )

# ax2.tick_params(axis="x", rotation=0)  # if you want horizontal labels
# ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0, fontsize=10)
# ax2.set_ylim(0.3,1.0)

# plt.tight_layout()
# plt.savefig("/Users/tylerbagwell/Desktop/cccv_data/pub_plots/NINO3_monthlycorrs_and_avgval.png", dpi=300, bbox_inches='tight')
# plt.show()

# sys.exit()




########### DMI

# start_year  = 1950
# end_year    = 2024 # need to add one more year

# cindex1 = prepare_DMI(file_path='data/NOAA_DMI_data.txt',
#                        start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                        end_date=datetime(end_year, 12, 1, 0, 0, 0))

# cindex1.index = pd.to_datetime(cindex1.index)
# cindex1['Year']  = cindex1.index.year
# cindex1['Month'] = cindex1.index.month

# cindex1_monthly = cindex1.pivot(index='Year', columns='Month', values='ANOM')
# for m in range(1, 8):                       # months 1‑4
#     cindex1_monthly[int(f'{m+12}')] = cindex1_monthly[m].shift(-1)
# cindex1_monthly = cindex1_monthly.iloc[:-1]

# print(cindex1_monthly)

# month_mapping = {
#     1: r'Jan$_t$', 2: r'Feb$_t$', 3: r'Mar$_t$', 4: r'Apr$_t$',
#     5: r'May$_t$', 6: r'Jun$_t$', 7: r'Jul$_t$', 8: r'Aug$_t$',
#     9: r'Sep$_t$', 10: r'Oct$_t$', 11: r'Nov$_t$', 12: r'Dec$_t$',
#     13: r'Jan$_{t+1}$', 14: r'Feb$_{t+1}$', 15: r'Mar$_{t+1}$', 16: r'Apr$_{t+1}$',
#     17: r'May$_{t+1}$', 18: r'Jun$_{t+1}$', 19: r'Jul$_{t+1}$'
# }

# cindex1_corr = cindex1_monthly.corr()
# cindex1_corr.rename(index=month_mapping, columns=month_mapping, inplace=True)
# print(len(cindex1_monthly))
# avg_abs1 = cindex1_monthly.abs().sum() / len(cindex1_monthly)

# print(avg_abs1)


# df = avg_abs1.reset_index().rename(columns={"index": "Month", 0: "Value"})
# df = df[df["Month"].between(1, 12)]
# df["MonthName"] = df["Month"].map(month_mapping)

# # PLOTTING
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10), 
#                                gridspec_kw={"height_ratios": [1, 0.15],
#                                               "hspace": -0.025})

# plt.suptitle('DMI', fontsize=11, y=0.925)

# sns.heatmap(
#     cindex1_corr,
#     annot=True, fmt=".2f",
#     cmap="PuOr", vmin=-1, vmax=1, center=0,
#     linewidths=0.5, linecolor="white", square=True,
#     annot_kws={"size": 6},
#     cbar_kws={
#         "orientation": "horizontal",
#         "shrink": 0.7,
#         "pad": 0.11    # distance between heatmap and colorbar
#     },
#     ax=ax1
# )
# ax1.set_title("Monthly Correlations (1950-2023)", fontsize=10)
# ax1.set_xlabel("", fontsize=10)
# ax1.set_ylabel("", fontsize=10)

# sns.set_theme(style="darkgrid")
# sns.lineplot(
#     data=df, x="MonthName", y="Value",
#     marker="o", linewidth=2, color="k",
#     ax=ax2
# )

# ax2.set_title("(1950-2023)", fontsize=10)
# ax2.set(
#     xlabel="Month",
#     ylabel=r"Avg. abs. DMI ($^\circ$C)",
# )
# ax2.grid(
#     True,                # enable grid
#     which="major",        # both major and minor ticks
#     linestyle="--",      # dashed grid lines
#     linewidth=0.5        # thinner line width
# )

# ax2.tick_params(axis="x", rotation=0)  # if you want horizontal labels
# ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0, fontsize=10)
# ax2.set_ylim(0.0,0.5)

# plt.tight_layout()
# plt.savefig("/Users/tylerbagwell/Desktop/cccv_data/pub_plots/DMI_monthlycorrs_and_avgval.png", dpi=300, bbox_inches='tight')
# plt.show()

# sys.exit()



########### ANI

start_year  = 1950
end_year    = 2023 # need to add one more year

cindex1 = prepare_ANI(file_path='data/Atlantic_NINO.csv',
                       start_date=datetime(start_year, 1, 1, 0, 0, 0),
                       end_date=datetime(end_year, 12, 1, 0, 0, 0))

cindex1.index = pd.to_datetime(cindex1.index)
cindex1['Year']  = cindex1.index.year
cindex1['Month'] = cindex1.index.month

cindex1_monthly = cindex1.pivot(index='Year', columns='Month', values='ANOM')
for m in range(1, 8):                       # months 1‑4
    cindex1_monthly[int(f'{m+12}')] = cindex1_monthly[m].shift(-1)
cindex1_monthly = cindex1_monthly.iloc[:-1]

print(cindex1_monthly)

month_mapping = {
    1: r'Jan$_t$', 2: r'Feb$_t$', 3: r'Mar$_t$', 4: r'Apr$_t$',
    5: r'May$_t$', 6: r'Jun$_t$', 7: r'Jul$_t$', 8: r'Aug$_t$',
    9: r'Sep$_t$', 10: r'Oct$_t$', 11: r'Nov$_t$', 12: r'Dec$_t$',
    13: r'Jan$_{t+1}$', 14: r'Feb$_{t+1}$', 15: r'Mar$_{t+1}$', 16: r'Apr$_{t+1}$',
    17: r'May$_{t+1}$', 18: r'Jun$_{t+1}$', 19: r'Jul$_{t+1}$'
}

cindex1_corr = cindex1_monthly.corr()
cindex1_corr.rename(index=month_mapping, columns=month_mapping, inplace=True)
print(len(cindex1_monthly))
avg_abs1 = cindex1_monthly.abs().sum() / len(cindex1_monthly)

print(avg_abs1)


df = avg_abs1.reset_index().rename(columns={"index": "Month", 0: "Value"})
df = df[df["Month"].between(1, 12)]
df["MonthName"] = df["Month"].map(month_mapping)

# PLOTTING
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10), 
                               gridspec_kw={"height_ratios": [1, 0.15],
                                              "hspace": -0.025})

plt.suptitle('ANI', fontsize=11, y=0.925)

sns.heatmap(
    cindex1_corr,
    annot=True, fmt=".2f",
    cmap="PuOr", vmin=-1, vmax=1, center=0,
    linewidths=0.5, linecolor="white", square=True,
    annot_kws={"size": 6},
    cbar_kws={
        "orientation": "horizontal",
        "shrink": 0.7,
        "pad": 0.11    # distance between heatmap and colorbar
    },
    ax=ax1
)
ax1.set_title("Monthly Correlations (1950-2023)", fontsize=10)
ax1.set_xlabel("", fontsize=10)
ax1.set_ylabel("", fontsize=10)

sns.set_theme(style="darkgrid")
sns.lineplot(
    data=df, x="MonthName", y="Value",
    marker="o", linewidth=2, color="k",
    ax=ax2
)

ax2.set_title("(1950-2023)", fontsize=10)
ax2.set(
    xlabel="Month",
    ylabel=r"Avg. abs. ANI ($^\circ$C)",
)
ax2.grid(
    True,                # enable grid
    which="major",        # both major and minor ticks
    linestyle="--",      # dashed grid lines
    linewidth=0.5        # thinner line width
)

ax2.tick_params(axis="x", rotation=0)  # if you want horizontal labels
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0, fontsize=10)
ax2.set_ylim(0.1,0.6)

plt.tight_layout()
plt.savefig("/Users/tylerbagwell/Desktop/cccv_data/pub_plots/ANI_monthlycorrs_and_avgval.png", dpi=300, bbox_inches='tight')
plt.show()

sys.exit()