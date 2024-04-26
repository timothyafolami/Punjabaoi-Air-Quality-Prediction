import streamlit as st
from streamlit_folium import st_folium
import folium
import datetime
from datetime import date, timedelta
from utils import *
from tools import *
import geopandas as gpd
st.set_page_config(layout="wide")
st.title("AQI in Punjab, Pakistan")
# prep for today info
today_forecast = predict_today()
@st.cache_data
def get_current_hour():
    return datetime.datetime.now().hour

# function to get the current date
@st.cache_data
def get_current_date():
    return datetime.datetime.now().date()

date_of_today = get_current_date()
hour_of_day = get_current_hour()
aqi_value = today_forecast[hour_of_day]
# add these values at the top left cover of the screen in a column
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("## Today's Date")
    st.markdown(date_of_today)
with col2:
    st.markdown("## Current Hour")
    st.markdown(hour_of_day)
with col3:
    st.markdown("## Current AQI")
    st.markdown(aqi_value)


# Load your shapefile (assuming 'map_data' folder and 'aoi_punjab.shp' filename)
gdf = gpd.read_file('punjabaoi/aoi_punjab.shp')

# Calculate average centroid coordinates
center_x = 31.1471
center_y = 75.3412

# Create the Folium map
m = folium.Map(location=[center_x, center_y], zoom_start=8)

# Add a GeoJSON layer
folium.GeoJson(data=gdf.to_json(), name='My Shapefile').add_to(m)

# Display the map with desired width
st_folium(m, width=1400, height=800)

########################################################
# Todays forecast
st.markdown("## Todays Forecast")
hrs = [j for j in range(0, 24)]
# getting today's forecast
today_forecast = predict_today()
# placing the forecast in a dataframe
today_forecast = pd.DataFrame(today_forecast)
today_forecast.index = hrs
# displaying the forecast
st.bar_chart(today_forecast)

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
        date_data = get_date_data(date)
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

################################################
# making a plot of the past week data
st.markdown("## Past Week Data")
# getting the data
past_week_data = past_week_model_data()
past_week_data_ = past_week_data['AQI'] 
# getting the forecast
forecast = predict_last_seven_days()
# plotting the data and forecast together
# using a dataframe to add themtogether
d1 = pd.DataFrame()
d1['past_week'] = past_week_data_
d1['forecast'] = forecast
d1.index = past_week_data['date']
st.line_chart(d1)

#########################################################

# Next week data forecasting
st.markdown("## Next Week Forecast")

nxt_hrs = next_week_dates_()

next_week_forecast = predict_next_seven_days()
# in a dataframe
next_week_forecast = pd.DataFrame(next_week_forecast)
next_week_forecast.index = nxt_hrs['date']
# plotting the forecast
st.line_chart(next_week_forecast)


  

