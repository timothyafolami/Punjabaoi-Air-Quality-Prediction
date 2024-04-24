import streamlit as st
import geopandas as gpd
from shapely.geometry import box, Point
from contextily import Place
import contextily as cx
import numpy as np
from matplotlib import pyplot as plt
import rasterio
from rasterio.plot import show as rioshow

plt.rcParams["figure.dpi"] = 70 # lower image size

madrid = Place("Punjab")
ax = madrid.plot()

# showing on streamlit
st.plot(ax)

# st.title("Air Quality Index Prediction")

# st.write("Please enter the following details to predict the Air Quality Index")