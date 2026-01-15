# src/screening.py
import yfinance as yf
import pandas as pd
import itertools
from statsmodels.tsa.stattools import coint

def download_screening_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Close']
    return data.dropna(axis=1)

def test_cointegration(df, p_cutoff=0.05):
    keys = df.columns
    pairs = list(itertools.combinations(keys, 2))
    results = []

    print(f"Screening {len(pairs)} pairs...")
    
    for y_name, x_name in pairs:
        y = df[y_name]
        x = df[x_name]
        
        # Engel-Granger Test
        score, p_value, _ = coint(y, x)
        
        if p_value < p_cutoff:
            results.append({
                'Y': y_name,
                'X': x_name,
                'p_value': p_value
            })
    res_df = pd.DataFrame(results)

    if not res_df.empty:
        res_df = res_df.sort_values('p_value').reset_index(drop = True)

    return res_df

if __name__ == "__main__":
    tickers = ['NEM', 'GOLD', 'AEM', 'KGC', 'AU', 'HMY']
    data = download_screening_data(tickers, '2015-01-01', '2023-01-01')
    results = test_cointegration(data)
    if not results.empty:
        print("\n--- Top Cointegrated Pairs ---")
        print(results.head())
    else:
        print("⚠️ No cointegrated pairs found at the current p_cutoff.")
