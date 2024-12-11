import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime

# Load pre-trained model or create a new one if it doesn't exist
@st.cache_resource
def load_or_create_model():
    try:
        model = load_model('Stock_Market_Prediction_Model.keras')
    except:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(90, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Fetch historical data
@st.cache_data
def fetch_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data


# Prepare data for LSTM model
def prepare_data(data, look_back=90):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler


# Train the model
@st.cache_resource
def train_model(_model, X, y):
    _model.fit(X, y, batch_size=32, epochs=100, verbose=0)
    _model.save('Stock_Market_Prediction_Model.keras')
    return _model


# Predict next day's prices
def predict_prices(data, scaler, _model, look_back=90):
    last_data = data[-look_back:]['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(last_data)
    X = np.array([scaled_data])
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    prediction = _model.predict(X)
    predicted_close = scaler.inverse_transform(prediction)[0][0]
    
    last_close = data['Close'].iloc[-1]
    predicted_open = float(last_close * (1 + np.random.uniform(-0.01, 0.001)))
    predicted_high = float(max(predicted_close, predicted_open) * 1.002)
    predicted_low = float(min(predicted_close, predicted_open) * 0.98)

    return predicted_open, predicted_high, predicted_low, predicted_close


# Calculate accuracy metrics
def calculate_accuracy(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2


# Streamlit App
def main():
    st.title("Stock Price Prediction Analysis App")

    # User inputs
    stock = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
    start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.date.today())

    if st.button("Analyze"):
        try:
            # Load or create the model
            model = load_or_create_model()

            # Fetch data
            data = fetch_data(stock, start_date, end_date)
            if data.empty:
                st.error("No data available for the selected stock.")
                return

            # Prepare data
            X, y, scaler = prepare_data(data)

            # Train model
            model = train_model(model, X, y)

            # Predict next day prices
            predicted_open, predicted_high, predicted_low, predicted_close = predict_prices(data, scaler, model)

            # Calculate accuracy
            y_pred = model.predict(X)
            y_pred = scaler.inverse_transform(y_pred)
            y_true = scaler.inverse_transform(y.reshape(-1, 1))
            mse, mae, r2 = calculate_accuracy(y_true, y_pred)

            # Display predictions in table format
            st.subheader("Predicted Prices for Tomorrow")
            prediction_data = {
                "Metric": ["Predicted Open Price", "Predicted High Price", "Predicted Low Price", "Predicted Close Price"],
                "Value": [f"${predicted_open:.2f}", f"${predicted_high:.2f}", f"${predicted_low:.2f}", f"${predicted_close:.2f}"]
            }
            st.table(pd.DataFrame(prediction_data))

            # Display accuracy metrics in table format
            st.subheader("Model Accuracy Metrics")
            accuracy_data = {
                "Metric": ["Mean Squared Error", "Mean Absolute Error", "R-squared Score"],
                "Value": [f"{mse:.4f}", f"{mae:.4f}", f"{r2:.4f}"]
            }
            st.table(pd.DataFrame(accuracy_data))

            # Visualize historical and predicted prices
            st.subheader("Historical vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index[-100:], data['Close'].tail(100), label="Historical Close Prices", color='blue', linestyle='-')
            ax.axhline(y=predicted_close, color='orange', linestyle='--', label=f"Predicted Close Price: ${predicted_close:.2f}")
            ax.axhline(y=predicted_open, color='green', linestyle='--', label=f"Predicted Open Price: ${predicted_open:.2f}")
            ax.legend()
            ax.set_title(f"{stock} Price Prediction (Last 100 Days)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            st.pyplot(fig)

            # Moving Average Analysis
            st.subheader("Moving Average Analysis (Last 100 Days)")
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['MA100'] = data['Close'].rolling(window=100).mean()

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index[-100:], data['Close'].tail(100), label='Close Price', color='blue')
            ax.plot(data.index[-100:], data['MA20'].tail(100), label='20-Day MA', color='red')
            ax.plot(data.index[-100:], data['MA50'].tail(100), label='50-Day MA', color='green')
            ax.plot(data.index[-100:], data['MA100'].tail(100), label='100-Day MA', color='orange')
            ax.set_title(f"{stock} Moving Average Analysis")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # Display raw data
            st.subheader("Raw Stock Data (Last 100 Days)")
            st.dataframe(data.tail(100))

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
