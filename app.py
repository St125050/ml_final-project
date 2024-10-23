import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the trained model
model = load_model('stock_price_model.h5')

# Load stock data (or allow users to upload files)
@st.cache
def load_data():
    df1 = pd.read_csv('AAPL.csv')
    df2 = pd.read_csv('GOOGL.csv')
    df3 = pd.read_csv('MSFT.csv')
    df4 = pd.read_csv('NVDA(2).csv')
    df1['stock'] = 'AAPL'
    df2['stock'] = 'GOOGL'
    df3['stock'] = 'MSFT'
    df4['stock'] = 'NVDA'
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    return df

# Load and preprocess data
df = load_data()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

scaler = MinMaxScaler()
df[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']])

# Create sequences for LSTM
def create_sequences(data, time_steps=60):
    sequences = []
    for i in range(len(data) - time_steps):
        seq = data[i:i + time_steps]
        sequences.append(seq)
    return np.array(sequences)

# Streamlit user interface
st.title("Stock Price Prediction App")
st.write("Select a stock to predict its price:")

stocks = df['stock'].unique()
selected_stock = st.selectbox("Stock", stocks)

# Prepare data for selected stock
stock_data = df[df['stock'] == selected_stock]
stock_features = stock_data[['Open', 'High', 'Low', 'Close']].values
sequences = create_sequences(stock_features)

# Make prediction on the last sequence
if st.button("Predict"):
    last_sequence = sequences[-1]
    last_sequence = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
    predicted_prices = model.predict(last_sequence)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Display results
    st.write(f"Predicted Prices: {predicted_prices.flatten()}")
    recommendation = "Buy" if predicted_prices[0, 3] > predicted_prices[0, 0] else "Sell"
    st.write(f"Recommendation: {recommendation}")
