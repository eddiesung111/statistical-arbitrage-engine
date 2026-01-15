# optimize_kalman.py
import pandas as pd
import itertools
from src import (
    get_classic_pair_train, get_classic_pair_test,
    KalmanPairsTrader, calculate_pnl, calculate_metrics, analyze_performance
)

def run_strategy_pipeline(strategy_name, train_data, test_data, mode, ve, delta, entry):
    warmup_window = 90
    trader = KalmanPairsTrader(delta=delta, ve=ve, entry=entry)
    
    if mode == 'train':
        signals_full = trader.calculate_signals(train_data)
        
    elif mode == 'test':
        signals_full = trader.calculate_signals(test_data)
    
    signals_df = signals_full.iloc[warmup_window:]

    results_df = calculate_pnl(signals_df)
    metrics = calculate_metrics(results_df)
    analyze_performance(results_df, strategy_name)
    return metrics


def main():
    # --- CONFIGURATION ---
    TICKER_Y = 'EWC'
    TICKER_X = 'EWA'
    
    # Dates
    TRAIN_START = '2005-09-01' # 3 Months before for WARMUP Logic.
    TRAIN_END   = '2010-01-01'
    TEST_START  = '2015-09-01'
    TEST_END    = '2020-10-01'

    print("Loading Data...")
    df_train = get_classic_pair_train(TICKER_Y, TICKER_X, start=TRAIN_START, end = TRAIN_END)
    df_test = get_classic_pair_test(TICKER_Y, TICKER_X, start=TEST_START, end = TEST_END)

    # --- PARAMETER GRID ---
    ve_choices = [1e-1, 1e-2, 1e-3]
    delta_choices = [1e-4, 1e-5]
    entry_choices = [1.0, 1.5, 2.0]

    strategy_name = "Kalman Filter"
    tuning_results = []

    print(f"\n--- GRID SEARCH: {strategy_name} (Training Data) ---")
    print(f"{'Delta':<10} | {'Ve':<10} | {'Entry':<6} \n")
    print("-" * 75)

    for delta, ve, entry in itertools.product(delta_choices, ve_choices, entry_choices):
        
        metrics = run_strategy_pipeline(
            strategy_name, 
            train_data=df_train, 
            test_data=df_test,
            mode='train', 
            ve=ve, delta=delta, entry=entry
        )
        
        print(f"{delta:<10} | {ve:<10} | {entry:<6}\n")
        
        tuning_results.append({
            'delta': delta, 've': ve, 'entry': entry,
            'trades': metrics['Total Trades'],
            'return': metrics["Annualized Return"],
            'sharpe': metrics[ "Sharpe Ratio"],
            'max_dd': metrics['Max Drawdown']
        })

    df_results = pd.DataFrame(tuning_results)
    best_params = df_results.sort_values(by='sharpe', ascending=False).iloc[0]

    print("\n" + "="*60)
    print("🏆 BEST CONFIGURATION (Based on Training Sharpe)")
    print(f"Delta: {best_params['delta']} | Ve: {best_params['ve']} | Entry: {best_params['entry']} | Sharpe: {best_params['sharpe']:.2f}")
    print("="*60)

    print(f"\n🚀 RUNNING VERIFICATION ON TEST DATA...")
    run_strategy_pipeline(
        strategy_name,
        train_data=df_train, 
        test_data=df_test,
        mode='test',
        ve=best_params['ve'],
        delta=best_params['delta'],
        entry=best_params['entry']
    )

if __name__ == "__main__":
    main()
