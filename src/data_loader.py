# src/data_loader.py
import yfinance as yf

def get_classic_pair_train(ticker_y, ticker_x, start='2006-01-01', end = '2010-01-01'):
    print(f"Downloading data: {ticker_y} vs {ticker_x} ({start} to {end})...")
    tickers = [ticker_y, ticker_x]
    df = yf.download(tickers, start=start, end=end)['Close']
    df.columns = ["price_y", "price_x"]
    df = df.dropna()
    return df

def get_classic_pair_test(ticker_y, ticker_x, start='2016-01-01', end='2020-01-01'):
    print(f"Downloading data: {ticker_y} vs {ticker_x} ({start} to {end})...")
    tickers = [ticker_y, ticker_x]
    df = yf.download(tickers, start=start, end=end)['Close']
    df.columns = ["price_y", "price_x"]
    df = df.dropna()
    return df