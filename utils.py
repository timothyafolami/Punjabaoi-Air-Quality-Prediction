import datetime
from datetime import date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import streamlit as st



# creating a function to get current date
def get_today_date():
    now = datetime.datetime.now()

    current_date = now.strftime("%Y-%m-%d")

    return current_date

@st.cache_data
def load_db():
  db = pd.read_csv('clean.csv')
  return db


def get_last_seven_days():
  """Returns a list of dates for the last seven days, including today."""
  today = date.today()
  last_week = today - timedelta(days = 7)
  dates = [last_week + timedelta(days=i) for i in range(7)]
  string_format = "%Y-%m-%d"
  # Convert dates to strings using list comprehension
  next_week_strings = [date.strftime(string_format) for date in dates]
  return next_week_strings


def get_next_seven_days():
  """Returns a list of dates for the next seven days, starting from tomorrow."""
  today = date.today()
  next_week = today + timedelta(days = 1)
  dates = [next_week + timedelta(days=i) for i in range(7)]
  string_format = "%Y-%m-%d"

  # Convert dates to strings using list comprehension
  last_week_strings = [date.strftime(string_format) for date in dates]
  return last_week_strings

def create_hourly_increments(date):

    return pd.date_range(date, periods=24, freq='h')
    
def get_date_data(date):
    # load the database
    df = load_db()
    df['date'] = pd.to_datetime(df['date'])
    # convert date to datetime
    date_dt = pd.to_datetime(date)
    # getting the hourly increments for the selected date
    hourly_df = pd.DataFrame({'date': create_hourly_increments(date)})
    data_ = df[df['date'].isin(hourly_df['date'].astype(str).values.tolist())]
    return data_


@st.cache_data
def load_db():
  db = pd.read_csv('clean.csv')
  return db

def today():
    return date.today()

def next_week_dates_():
    next = today() + timedelta(days=7)
    dates = [next + timedelta(days=i) for i in range(7)]
    string_format = "%Y-%m-%d"
    next_week_strings = [date.strftime(string_format) for date in dates]
    days_dt = [pd.to_datetime(date) for date in next_week_strings]

    hourly_dataframes = []
    for date in days_dt:
        hourly_df = pd.DataFrame({'date': create_hourly_increments(date)})
        hourly_dataframes.append(hourly_df)

    all_hours_df = pd.concat(hourly_dataframes, ignore_index=True)
    return all_hours_df 


def past_week_model_data():
    # loading the database
    data = load_db()
    last_week = today() - timedelta(days = 7)
    dates = [last_week + timedelta(days=i) for i in range(7)]
    string_format = "%Y-%m-%d"
    # Convert dates to strings using list comprehension
    next_week_strings = [date.strftime(string_format) for date in dates]
    days_dt = [pd.to_datetime(date) for date in next_week_strings]

    # Create a list of DataFrames, each containing hourly increments for a date
    hourly_dataframes = []
    for date in days_dt:
        hourly_df = pd.DataFrame({'date': create_hourly_increments(date)}) 
        hourly_dataframes.append(hourly_df)

    # Concatenate all DataFrames into a single DataFrame
    all_hours_df = pd.concat(hourly_dataframes, ignore_index=True)  
    dates_ = all_hours_df['date'].astype(str).values.tolist()
    # now gwtting from the data
    data_ = data[data['date'].isin(all_hours_df['date'].astype(str).values.tolist())]
    # return the AQI column
    out = data_['AQI'].values
    # extending by adding zeros to make it 168
    out = np.append(out, np.zeros(168 - len(out)))

    # creating a dataframe with the date and the AQI
    out = pd.DataFrame({'date': dates_, 'AQI': out})
    return out