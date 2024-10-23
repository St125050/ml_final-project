import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Load CSV files
df1 = pd.read_csv('/content/AAPL.csv')
df2 = pd.read_csv('/content/GOOGL.csv')
df3 = pd.read_csv('/content/MSFT.csv')
df4 = pd.read_csv('/content/NVDA(2).csv')

# Add a column for stock names
df1['stock'] = 'AAPL'
df2['stock'] = 'GOOGL'
df3['stock'] = 'MSFT'
df4['stock'] = 'NVDA'

# Concatenate DataFrames
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Map stock names to integers
df['stock'] = df['stock'].map({'AAPL': 0, 'GOOGL': 1, 'MSFT': 2, 'NVDA': 3})

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort DataFrame by date
df = df.sort_values(by='Date')

# Select relevant features
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'stock']]

# Normalize features
scaler = MinMaxScaler()
df[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']])

# Create sequences for LSTM
def create_sequences(data, time_steps=60):
    sequences = []
    labels = []
    for i in range(len(data) - time_steps):
        seq = data[i:i + time_steps]
        label = data[i + time_steps]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Prepare data for each stock
stock_data = {}
for stock in df['stock'].unique():
    stock_df = df[df['stock'] == stock]
    stock_features = stock_df[['Open', 'High', 'Low', 'Close']].values
    sequences, labels = create_sequences(stock_features)
    stock_data[stock] = (sequences, labels)

# Concatenate all stock sequences and labels
X = np.concatenate([data[0] for data in stock_data.values()])
y = np.concatenate([data[1] for data in stock_data.values()])

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(4))  # Predicting Open, High, Low, Close

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss (MSE): {loss}')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate additional metrics
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Convert to percentage
r2 = r2_score(y_test, y_pred)

# Custom "accuracy" metric: percentage of predictions within a certain percentage of the actual values
def custom_accuracy(y_true, y_pred, threshold=0.05):
    # Count how many predictions fall within the threshold percentage
    within_threshold = np.abs(y_true - y_pred) / y_true < threshold
    return np.mean(within_threshold) * 100  # Convert to percentage

accuracy = custom_accuracy(y_test, y_pred)

# Print metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
print(f'R-squared (R²): {r2:.2f}')
print(f'Custom Accuracy: {accuracy:.2f}%')

# Save the model for future use
model.save('stock_price_model.h5')

# Function to predict stock prices
def predict_stock(stock_id, last_sequence):
    last_sequence = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
    prediction = model.predict(last_sequence)
    return scaler.inverse_transform(prediction)  # Inverse scaling to get actual prices

# Function to provide buy/sell recommendation
def buy_sell_recommendation(predicted_prices):
    open_price, high_price, low_price, close_price = predicted_prices[0]
    if close_price > open_price:
        return "Buy"
    else:
        return "Sell"

# Example prediction
stock_id = 0  # AAPL
last_sequence = X_test[-1]  # For demonstration
predicted_prices = predict_stock(stock_id, last_sequence)
recommendation = buy_sell_recommendation(predicted_prices)
print(f"Predicted Prices: {predicted_prices}")
print(f"Recommendation: {recommendation}")
