import pandas as pd
import plotly.express as px

# Initialises dataframe containing latitude and longtitude for visualisation
location = pd.read_csv('dataset/Crashes/crash_info_general.csv',
                       usecols=['CRN', 'DEC_LAT', 'DEC_LONG'])
location.dropna(inplace=True)
location.reset_index(drop=True)

# Initialises the dataframe for crashes caused by tree shrubs
crashes = pd.read_csv('dataset/Crashes/crash_info_flag_variables.csv',
                      usecols=['CRN', 'FATAL_OR_SUSP_SERIOUS_INJ', 'HIT_TREE_SHRUB'])
crashes.dropna(inplace=True)
crashes.reset_index(drop=True)

# Merges the two dataframes
merged_df = pd.merge(location, crashes, on='CRN', how='inner')

# Filters for only crashes by drivers who were not under the influence of anything (crash caused by road)
merged_df = merged_df.query('FATAL_OR_SUSP_SERIOUS_INJ == 1').query('HIT_TREE_SHRUB == 1').reset_index(drop=True)

# Create the heatmap using Plotly
fig = px.density_mapbox(merged_df, lat='DEC_LAT', lon='DEC_LONG', z='FATAL_OR_SUSP_SERIOUS_INJ', radius=10,
                        center=dict(lat=39.9526, lon=-75.1652), zoom=10,
                        mapbox_style='carto-positron', title='Crash Density Map')

# Display the heatmap
fig.show()