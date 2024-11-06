import pandas as pd
import sys

print('\n\nSTART ---------------------\n')

# Dataset: UCDP/PRIO Armed Conflict Dataset version 24.1
# URL: https://ucdp.uu.se/downloads/index.html#armedconflict


conflict_data_path = '/Users/tylerbagwell/Desktop/UcdpPrioConflict_v24_1.csv'
df = pd.read_csv(conflict_data_path)
print(df.shape)

df['start_date2'] = pd.to_datetime(df['start_date2'])
df = df[df['start_date2'].dt.year >= 1950]

onsets_df = df.drop_duplicates(subset=['conflict_id', 'start_date2'], keep='first')
onsets_df.reset_index(drop=True, inplace=True)
print(onsets_df.shape)

onsets_df.to_csv('/Users/tylerbagwell/Desktop/UcdpPrioConflict_v24_1_ONSETS_ONLY.csv', index=False)
