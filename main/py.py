# File path: quant_ai.py

import os
import openai
import alpaca_trade_api as tradeapi
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Setup API keys
openai.api_key = os.getenv('sk-proj-TKT63VOvUpok4aWhtaHIT3BlbkFJ56vu0L3nbVem3kE5TSU1')
alpaca_api_key = os.getenv('PKFX1PMFTXQZ831CSCQH')
alpaca_secret_key = os.getenv('zDycTjhlc3FIMoRAgdaYLqcbXiTXBrR2RVKJCSH1')

# Initialize Alpaca API
alpaca = tradeapi.REST(alpaca_api_key, alpaca_secret_key, base_url='https://paper-api.alpaca.markets/v2')

# Function to fetch historical stock data from Alpaca
def fetch_alpaca_data(symbol, start, end):
    barset = alpaca.get_barset(symbol, 'day', start=start, end=end)
    df = barset[symbol].df
    return df

# Function to fetch historical stock data from Yahoo Finance
def fetch_yahoo_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

# Fetch and preprocess data
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'

alpaca_data = fetch_alpaca_data(symbol, start_date, end_date)
yahoo_data = fetch_yahoo_data(symbol, start_date, end_date)

# Merge datasets
data = pd.concat([alpaca_data, yahoo_data], axis=1)
data = data.dropna()

# Feature extraction and normalization
X = data[['open', 'high', 'low', 'volume']].values
y = data['close'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define a simple neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('quant_model.h5')

# Function to generate insights using ChatGPT
def generate_insights(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Fetch real-time data and make predictions
def predict_and_generate_insights(symbol):
    new_data = fetch_yahoo_data(symbol, start='2023-01-02', end='2023-01-03')
    new_data_scaled = scaler.transform(new_data[['Open', 'High', 'Low', 'Volume']].values)
    
    prediction = model.predict(new_data_scaled)
    prompt = f"The predicted closing price for {symbol} is {prediction[0][0]:.2f}. Can you provide some investment insights?"
    insights = generate_insights(prompt)
    return prediction[0][0], insights

# Example usage
predicted_price, investment_insights = predict_and_generate_insights('AAPL')
print(f"Predicted Price: {predicted_price}")
print(f"Investment Insights: {investment_insights}")
