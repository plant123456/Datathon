import pandas as pd
import folium
from folium.plugins import HeatMap

# Initialises dataframe for crashes that occured
crashes = pd.read_csv('dataset/Crashes/crash_info_general.csv', usecols=['CRN', 'DEC_LAT', 'DEC_LONG', 'ILLUMINATION', 'CRASH_YEAR'])
crashes.dropna(inplace=True)
crashes.reset_index(drop=True)

# Initialises dataframe for people involved in crashes
people = pd.read_csv('dataset/Crashes/crash_info_people.csv', usecols=['CRN', 'DVR_PED_CONDITION', 'PERSON_TYPE'])
people.dropna(inplace=True)
people.reset_index(drop=True)

# Merges the two dataframes
merged_df = pd.merge(crashes, people, on='CRN', how='inner')

# Filters for only crashes by drivers who were not under the influence of anything (crash caused by road)
merged_df = merged_df.query('DVR_PED_CONDITION == 1').query('PERSON_TYPE == 1').query('ILLUMINATION == 2').query('CRASH_YEAR >= 2020').reset_index(drop=True)

# Create a map of Philadelphia
map_philadelphia = folium.Map(location=[39.9526, -75.1652], zoom_start=12, tiles='Stamen Terrain')

# Create a heatmap layer
heatmap_layer = folium.FeatureGroup(name='Heatmap')

# Add the heatmap layer to the map
map_philadelphia.add_child(heatmap_layer)

# Create a list of points
points = merged_df[['DEC_LAT', 'DEC_LONG']].values.tolist()

# Add the points to the heatmap layer
heatmap_layer.add_child(HeatMap(points, radius=15, gradient={0.2: 'white', 0.4: 'yellow', 0.6: 'orange', 1: 'red'}))

# Create a layer control
layer_control = folium.LayerControl()

# Add the layer control to the map
map_philadelphia.add_child(layer_control)

# Save the map as an HTML file
map_philadelphia.save('heatmap_crashes_dark_after2020.html')