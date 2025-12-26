import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

print('\n\nSTART ---------------------\n')

# Dataset: GeoArmedConflictOnset_v1.csv


conflict_data_path = 'data/conflict_data/GeoArmedConflictOnset_v2.csv'
df = pd.read_csv(conflict_data_path)

# 1. REMOVE CONFLICT-YEARS WHERE THERE IS AN OBSERVATION BEFORE 1950
df_cleaned = df.loc[df['year'] >= 1950]
nrow0 = df_cleaned.shape[0]
print("...There are", nrow0, "observations in the dataset.")

# 2. REMOVE CONFLICT-YEARS WHERE THERE IS NO LOCATION DATA (LOCATOIN DATA COULD NOT BE FOUND)
df_cleaned = df_cleaned.dropna(subset=['onset_loc']).loc[df_cleaned['onset_loc'] != '']

# 3. ONLY INCLUDE ONSETX OBSERVATIONS
df_cleaned = df_cleaned.loc[df_cleaned['onset2'] == 1]

# 4. ONLY INCLUDE INTRASTATE CONFLICTS (OPTIONAL)
df_cleaned = df_cleaned.loc[df_cleaned['type_of_conflict'].isin([3, 4])]


nrow1 = df_cleaned.shape[0]
print("...There are", nrow1, "observations in the dataset after removing conflicts with unfound loc data.")
print("Total removed:", nrow0 - nrow1)
print("Percent removed:", round(((nrow0 - nrow1) / nrow0) * 100, 3), "%")

# 4. SAVE THE CLEANED DATA
df_cleaned.to_csv('data/conflict_data/GeoArmedConflictOnset_v2_CLEANED_instrastateONLY.csv', index=False)

