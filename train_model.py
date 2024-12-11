import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch historical data
def fetch_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Prepare data
def prepare_data(data, look_back=90):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

# Build the LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(60, return_sequences=True, activation='relu'),
        Dropout(0.3),
        LSTM(80, activation='relu'),
        Dropout(0.4),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and save the model
def train_model():
    stock = 'RELIANCE.NS'
    start = '2015-01-01'
    end = '2024-12-31'

    data = fetch_data(stock, start, end)
    X, y, scaler = prepare_data(data)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=50, batch_size=32)
    model.save('Stock_Market_Prediction_Model.keras')
    return model, scaler

if __name__ == "__main__":
    model, scaler = train_model()
    print("Model trained and saved.")
