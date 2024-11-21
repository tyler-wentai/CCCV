import pandas as pd
import sys

print('\n\nSTART ---------------------\n')

# Dataset: UCDP Georeferenced Event Dataset Codebook Version 24.1
# URL: https://ucdp.uu.se/downloads/index.html#ged_global

### Some column metadata:
# sida_a (string):      The name of Side A in the dyad. In state-based conflicts always a government. In one-sided violence always the perpetrating party.
# sida_b (string):      The name of Side B in the dyad. In state-based always the rebel movement or rivalling government. In one-sided violence always “civilians”.
# where_prec (integer): The precision with which the coordinates and location assigned to the event reflects the location of the actual event.
# best:                 The best (most likely) estimate of total fatalities resulting from an event.

conflict_data_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/GEDEvent_v24_1.csv'
df_conflict = pd.read_csv(conflict_data_path)

print(df_conflict.shape)

# step 1: remove conflicts between two or more governments, i.e., inter-state or state-on-state conflict
# df_cleaned = df_conflict[~(df_conflict['side_a'].str.startswith('Government') & df_conflict['side_b'].str.startswith('Government'))]
df_cleaned = df_conflict#[df_conflict['active_year']==1]
df_cleaned = df_cleaned[df_cleaned['type_of_violence']==1]


# step 2: remove conflicts where there is large uncertainty in the event geo-location:
df_cleaned = df_cleaned[df_cleaned['where_prec'] <= 4]

# # step 3a: remove extremely violent events with death counts above 5
# df_cleaned = df_cleaned[df_cleaned['best'] <= 5]
# # step 3b: remove extremely violent events with death counts above 5
df_cleaned = df_cleaned[df_cleaned['best'] >= 1]


print(df_cleaned.shape)
# df_cleaned.to_csv('/Users/tylerbagwell/Desktop/GEDEvent_v24_1_CLEANED.csv', index=False)
df_cleaned.to_csv('/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/GEDEvent_v24_1_CLEANED.csv', index=False)