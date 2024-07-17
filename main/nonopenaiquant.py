import alpaca_trade_api as tradeapi
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Alpaca API credentials
API_KEY = 'PKFX1PMFTXQZ831CSCQH'
API_SECRET = 'zDycTjhlc3FIMoRAgdaYLqcbXiTXBrR2RVKJCSH1'

# Initializing Alpaca with REST API
alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets/v2')

def fetch_spy_data():
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1d")
    return hist

def fetch_spy_data_via_alpaca():
    barset = alpaca.get_bars('SPY', TimeFrame.Day, limit=1)
    spy_bar = barset[0]  # Access the first element of the list
    
    data = {
        'time': spy_bar.t,
        'open': spy_bar.o,
        'high': spy_bar.h,
        'low': spy_bar.l,
        'close': spy_bar.c,
        'volume': spy_bar.v
    }
    return data

def fetch_alpaca_and_yahoo_data():
    yahoo_data = fetch_spy_data()
    alpaca_data = fetch_spy_data_via_alpaca()
    
    return {
        "Yahoo Finance": yahoo_data,
        "Alpaca": alpaca_data
    }

def fetch_news_data():
    news = alpaca.get_news("SPY", limit=10)
    articles = [article.headline + " " + article.summary for article in news]
    return " ".join(articles)

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return "Positive"
    elif sentiment['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def predict_price(data):
    # Prepare the data for regression
    yahoo_data = data['Yahoo Finance']
    close_prices = yahoo_data['Close'].values.reshape(-1, 1)
    dates = np.arange(len(close_prices)).reshape(-1, 1)  # Simple integer sequence for dates
    
    # Create and train the model
    model = LinearRegression()
    model.fit(dates, close_prices)
    
    # Predict future prices
    future_dates = np.arange(len(close_prices), len(close_prices) + 7).reshape(-1, 1)
    predicted_prices = model.predict(future_dates)
    
    prediction = {f"{i+1}d": price[0] for i, price in enumerate(predicted_prices)}
    return prediction

combined_data = fetch_alpaca_and_yahoo_data()
news_data = fetch_news_data()
sentiment = analyze_sentiment(news_data)
prediction = predict_price(combined_data)

print(f"combined_data: {combined_data}")
print(f"News Sentiment: {sentiment}")
print(f"Prediction: {prediction}")
