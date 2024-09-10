import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx


# node_data = pd.read_csv('node.csv')

node_data=data = {
    "lat": [59.9333637, 60.3693258, 59.2327725, 60.998248, 60.600397],
    "lon": [10.5040742, 8.54187, 11.2946315, 10.511856, 11.224594],
}

df = pd.DataFrame(node_data)
print(df)
# Define coordinates

# Filter for nodes in Norway (denoted by 'NO' in the area column)
#norway_nodes = df[df['area'] == 'NO']

# Extract relevant columns
#norway_coordinates = norway_nodes[['lat', 'lon']]

# Display the extracted coordinates
#norway_coordinates.head()

import folium

# Create a base map centered around the average coordinates
m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=8)

# Plot each point on the map
for idx, row in df.iterrows():
    folium.Marker([row['lat'], row['lon']]).add_to(m)

# Save the map as an HTML file or display it in a Jupyter notebook
m.save('map.html')



"""
# Create a GeoDataFrame from the data
geometry = [Point(xy) for xy in zip(node_data['lon'], node_data['lat'])]
gdf = gpd.GeoDataFrame(node_data, geometry=geometry)

# Set the CRS (assuming WGS84, EPSG:4326)
gdf.set_crs(epsg=4326, inplace=True)

# Optionally reproject to the CRS used by contextily (EPSG:3857)
gdf = gdf.to_crs(epsg=3857)

# Create a plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the points
gdf.plot(ax=ax, color='red', markersize=50)

# Add a basemap using contextily
ctx.add_basemap(ax, crs=gdf.crs.to_string())

# Save the plot as an SVG file (vector graphic)
plt.savefig('map_with_basemap.svg', format='svg')

plt.show()
"""