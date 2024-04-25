import streamlit as st
from streamlit_folium import st_folium
import folium
from utils import *

st.set_page_config(layout="wide")
st.title("Interactive Streamlit Map")

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

last_week_dates = get_last_seven_days()
next_week_dates = get_next_seven_days()

st.sidebar.markdown("## Options")
# selecting date
selected_week = st.sidebar.radio("Select a date", ("Last Week", "Next Week"))

# Code based on selected week
if selected_week == "Last Week":
  # Use last_week_dates for further processing
    with st.sidebar:
        st.write("You selected Last Week")
        date = st.selectbox("Select a date", last_week_dates)
    try:
        df = load_db()
        date_data = get_date_data(df, date)
        # adding a condition here
        if len(date_data) > 0:
            # selecting two columns 
            date_data = date_data[['date', 'AQI']]
            st.markdown('## Hourly AQI for selected date')
            # creating a bar chart
            st.bar_chart(date_data.set_index('date'))
        else: 
            st.write('No data available for the selected date')
    except:
        st.write("No data available for the selected date")
  
elif selected_week == "Next Week":
    # Use next_week_dates for further processing
    with st.sidebar:
        st.write("You selected Next Week")
        date = st.selectbox("Select a date", next_week_dates)


  

