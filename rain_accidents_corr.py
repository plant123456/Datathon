import pandas as pd

# Initialises the dataframe for crashes
df_crashes = pd.read_csv('dataset/Crashes/crash_info_general.csv',
                 usecols=['CRN', 'ROAD_CONDITION', 'WEATHER1', 'FATAL_COUNT', 'INJURY_COUNT'])
df_crashes.dropna(inplace=True)
df_crashes.reset_index(drop=True)

# Initialises the dataframe for weather
df_weather = pd.read_csv('dataset/Traffic, Investigations _ Other/hourly_weather_philadelphia.csv',
                         usecols=['datetime', 'precipitation'])
df_weather.dropna(inplace=True)
df_weather.reset_index(drop=True)

# Crashes which occured in rainy weather
df_rain = df_crashes.query('WEATHER1 == 7').query('ROAD_CONDITION == 9').reset_index(drop=True)

# Crashes which occured in clear weather
df_clear = df_crashes.query('WEATHER1 == 3').query('ROAD_CONDITION == 1').reset_index(drop=True)

# Hours of rainy weather
df_raincount = df_weather.query('precipitation != 0').reset_index(drop=True)

# Hours of clear weather
df_clearcount = df_weather.query('precipitation == 0').reset_index(drop=True)

# Calculates the amount of hours of rainy and clear weather
rainy_hours = len(df_raincount)
clear_hours = len(df_clearcount)
print("Number of rainy hours: ", rainy_hours)
print("Number of clear hours: ", clear_hours)

# Calculates the amount of crashes occuring in rainy and clear weather
crashes_rain = len(df_rain)
crashes_clear = len(df_clear)
print("Number of crashes in rain: ", crashes_rain)
print("Number of crashes in clear: ", crashes_clear)

# Calculates the ratio of crashes per hour
print("Number of crashes per hour in rain: ", (crashes_rain / rainy_hours))
print("Number of crashes per hour in clear: ", (crashes_clear / clear_hours))