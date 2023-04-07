import pandas as pd

# Initialises the dataframe for crashes
crashes = pd.read_csv('dataset/Crashes/crash_info_general.csv',
                 usecols=['CRN', 'ROAD_CONDITION', 'WEATHER1'])
crashes.dropna(inplace=True)
crashes.reset_index(drop=True)

# Initialises the dataframe for weather
df_weather = pd.read_csv('dataset/Traffic, Investigations _ Other/hourly_weather_philadelphia.csv',
                         usecols=['datetime', 'precipitation'])
df_weather.dropna(inplace=True)
df_weather.reset_index(drop=True)

# Initialises dataframe for people involved in crashes
people = pd.read_csv('dataset/Crashes/crash_info_people.csv', usecols=['CRN', 'DVR_PED_CONDITION', 'PERSON_TYPE'])
people.dropna(inplace=True)
people.reset_index(drop=True)

# Merges the two dataframes
merged_df = pd.merge(crashes, people, on='CRN', how='inner')

# Filters for only crashes by drivers who were not under the influence of anything (crash caused by road)
df_crashes = merged_df.query('DVR_PED_CONDITION == 1').query('PERSON_TYPE == 1').reset_index(drop=True)

# Crashes which occured in rainy weather
df_rain = df_crashes.query('WEATHER1 == 7').query('ROAD_CONDITION == 9').reset_index(drop=True)

# Crashes which occured in clear weather
df_clear = df_crashes.query('WEATHER1 == 3').query('ROAD_CONDITION == 1').reset_index(drop=True)

# Calculates the amount of crashes occuring in rainy and clear weather
crashes_rain = len(df_rain)
crashes_clear = len(df_clear)
print("Number of crashes in rain: ", crashes_rain)
print("Number of crashes in clear: ", crashes_clear)

# Hours of rainy weather
df_raincount = df_weather.query('precipitation != 0').reset_index(drop=True)

# Hours of clear weather
df_clearcount = df_weather.query('precipitation == 0').reset_index(drop=True)

# Calculates the amount of hours of rainy and clear weather
rainy_hours = len(df_raincount)
clear_hours = len(df_clearcount)
print("Number of rainy hours: ", rainy_hours)
print("Number of clear hours: ", clear_hours)

# Calculates the ratio of crashes per hour
print("Number of crashes per day in rain: ", 24 * (crashes_rain / rainy_hours))
print("Number of crashes per day in clear: ", 24 * (crashes_clear / clear_hours))