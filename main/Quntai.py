import alpaca_trade_api as tradeapi
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame
import openai
import matplotlib.pyplot as plt
import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
# Alpaca API credentials
API_KEY = ''
API_SECRET = ''

# Initializing Alpaca with REST API
alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets/v2')

OPENAI_APIKey=''
openai.api_key=OPENAI_APIKey


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

def fetch_historical_data(ticker, start_date, end_date):
    try:
        # Fetch data
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Example usage:
ticker = "SPY"  # Example ticker
start_date = "2023-01-01"
end_date = "2024-06-01"

historical_data = fetch_historical_data(ticker, start_date, end_date)
if historical_data is not None:
    print(historical_data.head())


def predict_price(data):
    prompt = f"Given the data about the SPY :\n\n{data}\n\n;the estimated sentiment about the SPY through an algorithm:\n\n{analyze_sentiment}\n\n,the current news about the SPY :\n\n{fetch_news_data}\n\n. And finally the historical data \n\n{fetch_historical_data}\n\n Make a prediction about the price of the SPY for the next week and give the price for all days of the week."

    response=openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"You are a financial analyst, your skill set is similar to that of a walls street financial analyst, your purpose is to be a quantitative ai, you will refuse any other prompt than that of financial advice or price prediction."},
            {"role":"user","content":prompt}
        ]
    )

    return response.choices[0].message['content'].strip()


    
    # Save or show the plot
    plt.show()

combined_data = fetch_alpaca_and_yahoo_data()
prediction = predict_price(combined_data)
print(f"combined_data:{combined_data}")
print(f"Prediction: {prediction}")

