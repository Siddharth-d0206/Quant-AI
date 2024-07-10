import alpaca_trade_api as tradeapi
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame

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

# Example usage
combined_data = fetch_alpaca_and_yahoo_data()
print(combined_data)
