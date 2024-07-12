import alpaca_trade_api as tradeapi
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame
import openai
import matplotlib.pyplot as plt
import datetime
# Alpaca API credentials
API_KEY = 'PKFX1PMFTXQZ831CSCQH'
API_SECRET = 'zDycTjhlc3FIMoRAgdaYLqcbXiTXBrR2RVKJCSH1'

# Initializing Alpaca with REST API
alpaca = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets/v2')

OPENAI_APIKey='sk-proj-QssDhyq5Nh38TTIpTozET3BlbkFJgeSd42wSVo7Rg8fwqwqP'
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
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Analyze the sentiment of the following text and return it as 'Positive', 'Neutral', or 'Negative':\n\n{text}",
        max_tokens=60
    )
    return response.choices[0].text.strip()
def predict_price(data):
    prompt = f"Given the data about the SPY :\n\n{data}\n\nMake a prediction about the price of the SPY for the next week and give the price for different time intervals. Ex: 1d, 2d, 3d."

    response=openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"Your are a financial analysist, your skill set is similar to that of a walls street financial anlaysist, your purpose is to be a quantitative ai, you will refuse anyother prompt than that of financial advice or price prediction."},
            {"role":"user","content":prompt}
        ]
    )

    return response.choices[0].message['content'].strip()
    return response.choices[0].message['content'].strip()
def plot_data(data, prediction):
    # Extract necessary data for plotting
    yahoo_close = data['Yahoo Finance']['Close']
    alpaca_close = data['Alpaca']['close']  # Adjust this to match your data structure
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(yahoo_close.index, yahoo_close, label='Yahoo Finance Close', marker='o')
    plt.plot(alpaca_close['time'], alpaca_close['close'], label='Alpaca Close', marker='o')
    
    # Add prediction point to the plot
    # Example: Assuming prediction is a single value for simplicity
    # Replace with actual prediction logic
    plt.axvline(x=yahoo_close.index[-1], color='r', linestyle='--', label='Prediction')
    
    plt.title('SPY Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save or show the plot
    plt.show()

combined_data = fetch_alpaca_and_yahoo_data()
prediction = predict_price(combined_data)
print(f"combined_data:{combined_data}")
print(f"Prediction: {prediction}")

plot_data(combined_data,prediction)
