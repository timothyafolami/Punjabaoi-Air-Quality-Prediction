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
import pandas as pd


def get_data_for_dates(df, date_list):

  # Check if 'date' column exists
  if 'date' not in df.columns:
    raise ValueError("Dataframe does not contain a 'date' column")

  # Convert date column to datetime format (assuming it's not already)
  df['date'] = pd.to_datetime(df['date'])

  # Set the date column as the index for efficient lookup
  df.set_index('date', inplace=True)

  # Try to select data for the specified dates
  try:
    return df.loc[date_list]
  except KeyError:  # Handle cases where some dates might not be present
    print(f"Warning: Data not found for all dates in {date_list}")
    return df.loc[[date for date in date_list if date in df.index]]  # Return available data


def get_data_before_last_week(df):

  # Check if 'date' column exists
  if 'date' not in df.columns:
    raise ValueError("Dataframe does not contain a 'date' column")

  # Convert date column to datetime format (assuming it's not already)
  df['date'] = pd.to_datetime(df['date'])

  # Check if DataFrame is empty
  if df.empty:
    return None

  # Get the minimum date and calculate the starting date for the last seven days before
  min_date = df['date'].min()
  last_week_start = min_date - pd.Timedelta(days=7)

  # Set the date column as the index for efficient lookup
  df.set_index('date', inplace=True)

  # Select data for the last seven days before the minimum date
  return df.loc[last_week_start:min_date-pd.Timedelta(days=1)]


# Sample DataFrame (assuming you have your data)
data = {'date': ['2024-04-20', '2024-04-21', '2024-04-22', '2024-04-23', '2024-04-24'],
        'value': [10, 15, 20, 25, 30]}
df = pd.DataFrame(data)

# Example usage
date_list = [pd.to_datetime('2024-04-22'), pd.to_datetime('2024-04-24')]
filtered_data = get_data_for_dates(df.copy(), date_list)  # Copy to avoid modifying original df
print(filtered_data)

last_week_data = get_data_before_last_week(df.copy())  # Copy to avoid modifying original df
print(last_week_data)



class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers

    self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_model():
    model = LSTM(1, 4, 1)
    model.to(device)
    # Load the state dictionaries
    model.load_state_dict(torch.load("lstm_model.pt"))
    return model

def prepare_data(df, n_steps):
  df = dc(df)
  df['date'] = pd.to_datetime(df['date'])

  df.set_index('date', inplace=True)

  for i in range(1, n_steps+1):
    df[f"AQI(t-{i})"] = df['AQI'].shift(i)

  df.dropna(inplace=True)

  return df