import alpaca_trade_api as tradeapi
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import requests

# Alpaca API credentials
API_KEY = 'PKFX1PMFTXQZ831CSCQH'
API_SECRET = 'zDycTjhlc3FIMoRAgdaYLqcbXiTXBrR2RVKJCSH1'
BASE_URL = 'https://paper-api.alpaca.markets/v2'

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def get_spy_tickers():
    url = "https://api.nasdaq.com/api/quote/SPY/holdings"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check if the request was successful
        try:
            data = response.json()
            holdings = data['data']['holdings']
            tickers = [holding['ticker'] for holding in holdings]
            return tickers
        except ValueError:
            print("Error decoding JSON response")
            return []
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return []

def fetch_data(symbol, start_date, end_date):
    barset = api.get_barset(symbol, 'day', start=start_date, end=end_date)
    df = barset[symbol].df
    df['symbol'] = symbol
    return df

def create_features(data):
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    data['Momentum'] = data['close'] - data['close'].shift(4)
    data.dropna(inplace=True)
    return data

def normalize(data):
    return (data - data.mean()) / data.std()

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def make_prediction(model, data):
    predictions = model.predict(data)
    return predictions

def place_order(symbol, qty, side, type, time_in_force):
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=type,
        time_in_force=time_in_force,
    )

# Fetch data for all SPY stocks
tickers = get_spy_tickers()
if not tickers:
    print("Failed to fetch SPY tickers. Exiting...")
    exit()

all_data = pd.DataFrame()

for ticker in tickers:
    try:
        data = fetch_data(ticker, '2022-01-01', '2022-12-31')
        data = create_features(data)
        data = normalize(data)
        all_data = all_data.append(data)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Prepare data for training
X = all_data[['SMA_20', 'SMA_50', 'Momentum']].values
y = all_data['close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = build_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Model Mean Absolute Error: {mae}')

# Fetch new data for prediction and trading
new_data = pd.DataFrame()

for ticker in tickers:
    try:
        data = fetch_data(ticker, '2023-01-01', '2023-01-31')
        data = create_features(data)
        data = normalize(data)
        new_data = new_data.append(data)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

X_new = new_data[['SMA_20', 'SMA_50', 'Momentum']].values
predictions = make_prediction(model, X_new)

# Implement a basic trading strategy
portfolio = {}
initial_capital = 100000  # Initial capital in dollars

for i, ticker in enumerate(new_data['symbol'].unique()):
    pred = predictions[i]
    current_price = new_data[new_data['symbol'] == ticker]['close'].iloc[-1]
    if pred > current_price:  # Buy signal
        qty = int(initial_capital / (len(tickers) * current_price))  # Equal distribution
        place_order(ticker, qty, 'buy', 'market', 'gtc')
        portfolio[ticker] = qty
    elif ticker in portfolio and pred < current_price:  # Sell signal
        qty = portfolio[ticker]
        place_order(ticker, qty, 'sell', 'market', 'gtc')
        del portfolio[ticker]

print("Portfolio:", portfolio)
