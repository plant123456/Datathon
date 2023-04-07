import pandas as pd

# Initialises the dataframe for crashes
crashes = pd.read_csv('dataset/Crashes/crash_info_general.csv',
                 usecols=['CRN', 'ILLUMINATION'])
crashes.dropna(inplace=True)
crashes.reset_index(drop=True)

# Initialises dataframe for people involved in crashes
people = pd.read_csv('dataset/Crashes/crash_info_people.csv', usecols=['CRN', 'DVR_PED_CONDITION', 'PERSON_TYPE'])
people.dropna(inplace=True)
people.reset_index(drop=True)

# Merges the two dataframes
merged_df = pd.merge(crashes, people, on='CRN', how='inner')

# Filters for only crashes by drivers who were not under the influence of anything (crash caused by road)
df_crashes = merged_df.query('DVR_PED_CONDITION == 1').query('PERSON_TYPE == 1').reset_index(drop=True)

# Crashes which occured due to poor lighting
df_dark_no_lights = df_crashes.query('ILLUMINATION == 2').reset_index(drop=True)

# Crashes which occured not due to poor lighting
df_dark_lights = df_crashes.query('ILLUMINATION == 3').reset_index(drop=True)

# Calculates the amount of crashes occuring in rainy and clear weather
crashes_no_streetlights = len(df_dark_no_lights)
crashes_with_streetlights = len(df_dark_lights)
print("Number of crashes due to absence of streetlights: ", crashes_no_streetlights)
print("Number of crashes not due to absence of streetlights: ", crashes_with_streetlights)