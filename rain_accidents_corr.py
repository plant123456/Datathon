import pandas as pd

df = pd.read_csv('dataset/Crashes/crash_info_general.csv',
                 usecols=['CRN', 'ROAD_CONDITION', 'WEATHER1', 'FATAL_COUNT', 'INJURY_COUNT'])

# Rainy days
df_rain = df.query('WEATHER1 == 7').query('ROAD_CONDITION == 9').reset_index(drop=True)
df_rain.dropna()

# Clear days
df_clear = df.query('WEATHER1 == 3').query('ROAD_CONDITION == 1').reset_index(drop=True)
df_clear.dropna()

# Assign arbitrary values for rainy and clear days with 1 and 0 respectively
df_rain = df_rain.assign(condition=1)
df_clear = df_clear.assign(condition=0)

# Concat the two dataframes together to calculate correlation
df_concat = pd.concat([df_rain, df_clear], axis=0).reset_index(drop=True)

print(df_concat.corr())