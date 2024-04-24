import streamlit as st
from streamlit_folium import st_folium
import folium


st.set_page_config(layout="wide", )
st.title("Interactive Streamlit Map")

full_screen_style = """
<style>
body {
    margin: 0;
    padding: 0;
    overflow: hidden;
}
</style>
"""
st.markdown(full_screen_style, unsafe_allow_html=True)

# latitude = 31.1471  # Replace with default latitude
# longitude = 75.3412  # Replace with default longitude
# zoom_level = st.slider("Zoom Level:", 1, 18, 13)  # Set default zoom

# m = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)

# # Add a marker (optional)
# folium.Marker([latitude, longitude]).add_to(m)

# st_data = st_folium(m, width=725)
import geopandas as gpd

# Load your shapefile (assuming 'map_data' folder and 'aoi_punjab.shp' filename)
gdf = gpd.read_file('punjabaoi/aoi_punjab.shp')

# Calculate average centroid coordinates
center_x = 31.1471
center_y = 75.3412

# Create the Folium map
m = folium.Map(location=[center_x, center_y], zoom_start=7)  # Adjust zoom level if needed

# Add a GeoJSON layer
folium.GeoJson(data=gdf.to_json(), name='My Shapefile').add_to(m)

# Display the map with desired width
st_folium(m, width=1400, height=1000)

st.sidebar.write("This is a sidebar")

