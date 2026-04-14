import yfinance as yf
import pandas as pd
import numpy as np

# -----------------------------
# Predict next-day return
# -----------------------------
def predict_next_return(df):
    """
    Predict next-day return using percent change.
    Always returns a list the SAME LENGTH as df.
    """

    # Force Close to be a Series, even if yfinance returns multiple columns
    close = df['Close']

    # If Close is a DataFrame, take the first column
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Now compute returns safely
    returns = close.pct_change().astype(float)

    # Replace first NaN with 0
    returns.iloc[0] = 0

    return returns.tolist()



# -----------------------------
# Backtest a single stock
# -----------------------------
def backtest_stock(df):
    """
    Runs a simple backtest:
    - Predict next-day return
    - Buy if predicted return > 0
    - Hold cash otherwise
    """

    # Generate predictions (same length as df)
    df['Predicted'] = predict_next_return(df)

    # Strategy: buy if predicted return > 0
    df['Signal'] = df['Predicted'].apply(lambda x: 1 if x > 0 else 0)

    # Actual returns
    df['ActualReturn'] = df['Close'].pct_change().fillna(0)

    # Strategy returns
    df['StrategyReturn'] = df['Signal'] * df['ActualReturn']

    # Cumulative return
    cumulative_return = (1 + df['StrategyReturn']).prod() - 1

    return cumulative_return


# -----------------------------
# Backtest a portfolio of tickers
# -----------------------------
def backtest_portfolio(tickers, period="1y"):
    results = {}

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, period=period)

        if df.empty:
            print(f"⚠️ No data for {ticker}, skipping.")
            continue

        cr = backtest_stock(df)
        results[ticker] = cr

    return results


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

    print("Running portfolio backtest...\n")
    results = backtest_portfolio(tickers)

    print("\n===== RESULTS =====")
    for t, r in results.items():
        print(f"{t}: {r:.2%}")
