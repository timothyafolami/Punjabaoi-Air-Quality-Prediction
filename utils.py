import datetime
from datetime import date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn



# creating a function to get current date
def get_today_date():
    now = datetime.datetime.now()

    current_date = now.strftime("%Y-%m-%d")

    return current_date


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
  return pd.date_range(start=date, periods=24, freq='H', end = date)


def get_date_data(df, date):
    df['date'] = pd.to_datetime(df['date'])
    # convert date to datetime
    date_dt = pd.to_datetime(date)
    # getting the hourly increments for the selected date
    hourly_df = pd.DataFrame({'date': create_hourly_increments(date)})
    data_ = df[df['date'].isin(all_hours_df['date'].astype(str).values.tolist())]
    return data_


def prepare_data(df, n_steps):
  df = dc(df)
  df['date'] = pd.to_datetime(df['date'])

  df.set_index('date', inplace=True)

  for i in range(1, n_steps+1):
    df[f"AQI(t-{i})"] = df['AQI'].shift(i)

  df.dropna(inplace=True)

  return df

@st.cache_data
  def load_db():
    db = pd.read_csv('clean.csv')
    return db