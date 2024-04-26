import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from copy import deepcopy as dc
import datetime
from datetime import date, timedelta
import pandas as pd
import numpy as np
import joblib


class TimeSeriesDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i], self.y[i]

# LSTM Model
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
lookback = 7
batch_size = 16

def prepare_data(df, n_steps):
  df = dc(df)
  df['date'] = pd.to_datetime(df['date'])

  df.set_index('date', inplace=True)

  for i in range(1, n_steps+1):
    df[f"AQI(t-{i})"] = df['AQI'].shift(i)

  df.dropna(inplace=True)

  return df

# data for the model prediction
def model_data(df):
    df['date'] = pd.to_datetime(df['date'])
    shifted_df = prepare_data(df, lookback)
    shifted_df_as_np = shifted_df.to_numpy()
    # loading the scaler
    scaler = joblib.load('scaler.pkl')
    # scaling the data
    scaled_data = scaler.transform(shifted_df_as_np)
    # preparing the data for the model
    X = scaled_data[:, 1:]
    y = scaled_data[:, 0]
    # flipping X
    X = dc(np.flip(X, axis=1))
    # reshaping X and y
    X_ = X.reshape((-1, lookback, 1))
    y_ = y.reshape((-1, 1))
    # converting to tensor
    X_ = torch.tensor(X_).float().to(device)
    y_ = torch.tensor(y_).float().to(device)
    # now using the dataset class
    dataset = TimeSeriesDataset(X_, y_)
    # creating a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, X_, y_

# model loading function
def load_model():
    model = torch.load('new_1_lstm_model.pt')
    for param in model.lstm.parameters():
        param.requires_grad = False
    return model


def predict(dataloader):
    model = load_model()
    model.eval()
    predictions = []
    y_ = []
    with torch.no_grad():
        for X, y in dataloader:
            y_pred = model(X)
            predictions.append(y_pred)
            y_.append(y)
    return predictions, y_

# making future predictions
def make_future_predictions(df, future_steps):
    dataloader, X_, y_ = model_data(df)
    model = load_model()
    model.eval()
    with torch.no_grad():
        for i in range(future_steps):
            y_pred = model(X_[-1].reshape(1, lookback, 1))
            X_ = torch.cat((X_, y_pred.reshape(1, 1, 1)), dim=1)
    return X_, y_

def predict_today(n=24):
    # loading the database
    data = pd.read_csv('clean.csv')
    # getting today's date
    today_date = date.today()
    today_date = pd.to_datetime(today_date)
    # get last date in the data
    last_date = data['date'].iloc[-1]
    # converting both to time and getting the total hours between them
    last_date = pd.to_datetime(last_date)
    hours_diff = (today_date - last_date).total_seconds() / 3600
    aqi_data = data['AQI'].values.tolist()
    model = load_model()
    model.eval()
    data_scaler = joblib.load('data_scaler.pkl')
    target_Scaler = joblib.load('target_scaler.pkl')
    lookback = 7
    # ensureing we are just collecting the last 6 values
    current_data = aqi_data[-lookback:]
    predictions = []
    for _ in range(int(hours_diff)):
        # Ensure current_data has the required length
        if len(current_data) != lookback:
            raise ValueError(f"Current data must have length {lookback}")

        # Prepare data for the model (scaling and reshaping)
        current_data_np = np.array([current_data])  # Assuming data is a list
        scaled_data = data_scaler.transform(current_data_np)
        X = scaled_data.reshape((1, lookback, 1))
        X_ = torch.tensor(X).float().to(device)

        # Make prediction
        with torch.no_grad():
            y_pred = model(X_)

        # Inverse scaling of the predicted value (assuming single value prediction)
        predicted_value = target_scaler.inverse_transform(y_pred.cpu().detach().numpy())[0][0]

        # Update current_data for the next iteration
        current_data.append(predicted_value)
        current_data = current_data[-lookback:]  # Keep only the last lookback values

        # Append the prediction to the final list
        predictions.append(predicted_value)


    return predictions[-24:]
data_scaler = joblib.load('data_scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')

# predicting for thr last seven days
def predict_last_seven_days():
    # getting the data
    data = pd.read_csv('clean.csv')
    # converting date to the right format
    data['date'] = pd.to_datetime(data['date'])
    # today's date
    today_date = date.today()
    # last seven days
    last_seven_days = today_date - timedelta(days=7)
    # converting to datetime
    today_date = pd.to_datetime(today_date)
    last_seven_days = pd.to_datetime(last_seven_days)
    # caalculating the hours difference
    hours_diff = (today_date - last_seven_days).days * 24
    # the prev day before the last seven days data
    previous_day_hrs = hours_diff + 24 
    # getting the data
    aqi_data = data['AQI'].values.tolist()[-previous_day_hrs:-hours_diff]
    # loading model
    model = load_model()
    model.eval()
    # loading scalers
    data_scaler = joblib.load('data_scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')
    # lookback
    lookback = 7
    # current data
    current_data = aqi_data[-lookback:]
    # predictions
    predictions = []
    # loop to make predictions
    for _ in range(hours_diff):
        if len(current_data) != lookback:
            raise ValueError(f"Current data must have length {lookback}")

        current_data_np = np.array([current_data])
        scaled_data = data_scaler.transform(current_data_np)
        X = scaled_data.reshape((1, lookback, 1))
        X_ = torch.tensor(X).float().to(device)

        with torch.no_grad():
            y_pred = model(X_)

        predicted_value = target_scaler.inverse_transform(y_pred.cpu().detach().numpy())[0][0]

        current_data.append(predicted_value)
        current_data = current_data[-lookback:]

        predictions.append(predicted_value)

    return predictions[-24*7:]

# next seven days prediction
def predict_next_seven_days():
    # loading the database
    data = pd.read_csv('clean.csv')
    # getting today's date
    today_date = date.today()
    today_date = pd.to_datetime(today_date)
    # get last date in the data
    last_date = data['date'].iloc[-1]
    # converting both to time and getting the total hours between them
    last_date = pd.to_datetime(last_date)
    hours_diff = (today_date - last_date).total_seconds() / 3600
    # adding 168 to the hours diff
    hours_diff += 168
    aqi_data = data['AQI'].values.tolist()
    model = load_model()
    model.eval()
    data_scaler = joblib.load('data_scaler.pkl')
    target_Scaler = joblib.load('target_scaler.pkl')
    lookback = 7
    # ensureing we are just collecting the last 6 values
    current_data = aqi_data[-lookback:]
    predictions = []
    for _ in range(int(hours_diff)):
        # Ensure current_data has the required length
        if len(current_data) != lookback:
            raise ValueError(f"Current data must have length {lookback}")

        # Prepare data for the model (scaling and reshaping)
        current_data_np = np.array([current_data])  # Assuming data is a list
        scaled_data = data_scaler.transform(current_data_np)
        X = scaled_data.reshape((1, lookback, 1))
        X_ = torch.tensor(X).float().to(device)

        # Make prediction
        with torch.no_grad():
            y_pred = model(X_)

        # Inverse scaling of the predicted value (assuming single value prediction)
        predicted_value = target_scaler.inverse_transform(y_pred.cpu().detach().numpy())[0][0]

        # Update current_data for the next iteration
        current_data.append(predicted_value)
        current_data = current_data[-lookback:]  # Keep only the last lookback values

        # Append the prediction to the final list
        predictions.append(predicted_value)

    return predictions[-168:]