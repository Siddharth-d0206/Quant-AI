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

    plot_data(combined_data,prediction)