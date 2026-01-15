# src/backtesting.py
def calculate_pnl(strategy_df, transaction_cost_pct=0.0005):
    df = strategy_df.copy()
    y_ret = df['price_y'].pct_change()
    x_ret = df['price_x'].pct_change()

    df['strategy_return'] = (df['position'].shift(1) * (y_ret - df['hedge_ratio'] * x_ret)).fillna(0)
    trades = df['position'].diff().abs().fillna(0)
    cost_drag = trades * transaction_cost_pct
    
    df['strategy_return_net'] = df['strategy_return'] - cost_drag
    df['equity_curve'] = (1 + df['strategy_return_net']).cumprod()
    return df

def calculate_metrics(df):
    total_days = (df.index[-1] - df.index[0]).days
    years = total_days / 365.25
    
    # Annualized Return
    total_return = df['equity_curve'].iloc[-1] - 1
    cagr = (1 + total_return) ** (1 / years) - 1
    
    # Sharpe Ratio (Annualized)
    daily_mean = df['strategy_return_net'].mean()
    daily_std = df['strategy_return_net'].std()
    sharpe = (daily_mean / daily_std) * (252 ** 0.5)

    # Drawdown
    cum_max = df['equity_curve'].cummax()
    drawdown = (df['equity_curve'] - cum_max) / cum_max
    max_dd = drawdown.min()
    
    # Trades
    trades = df['position'].diff().abs().sum() / 2
    
    return {
        "Total Trades": int(trades),
        "Annualized Return": cagr,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }

def analyze_performance(df, strategy_name):
    metrics = calculate_metrics(df)
    
    print(f"--- Performance: {strategy_name} ---")
    print(f"Total Trades:      {metrics['Total Trades']}")
    print(f"Annualized Return: {metrics['Annualized Return']:.2%}")
    print(f"Sharpe Ratio:      {metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:      {metrics['Max Drawdown']:.2%}")
    