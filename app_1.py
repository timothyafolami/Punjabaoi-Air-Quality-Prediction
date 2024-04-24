import streamlit as st
from streamlit_folium import st_folium
import folium

st.title("Interactive Streamlit Map")

latitude = 31.1471  # Replace with default latitude
longitude = 75.3412  # Replace with default longitude
zoom_level = st.slider("Zoom Level:", 1, 18, 13)  # Set default zoom

m = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)

# Add a marker (optional)
folium.Marker([latitude, longitude]).add_to(m)

st_data = st_folium(m, width=725)

