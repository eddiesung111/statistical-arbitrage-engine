# main,py
from src import (
    get_classic_pair_train, get_classic_pair_test,
    OLSTrader, RollingOLSTrader, KalmanPairsTrader,
    calculate_pnl, analyze_performance,
    plot_diagnostic, plot_strategy_comparison
)

def run_full_analysis():
    warmup_window = 90

    # --- 1. CONFIGURATION ---
    TICKER_Y = 'EWC'
    TICKER_X = 'EWA'
    
    # Dates
    TRAIN_START = '2005-09-01' # 3 Months before for WARMUP Logic.
    TRAIN_END   = '2010-01-01'
    TEST_START  = '2015-09-01'
    TEST_END    = '2020-10-01'

    strategies = {
        "Static OLS":  OLSTrader(window=90, entry=1.0, exit=0.1),
        "Rolling OLS": RollingOLSTrader(window=90, entry=1.0, exit=0.1),
        "Kalman Filter": KalmanPairsTrader(delta=1e-5, ve=1e-2, entry=1.0, exit=0.1)
    }


    print("--- Loading Data ---")
    df_train_raw = get_classic_pair_train(TICKER_Y, TICKER_X, start=TRAIN_START, end = TRAIN_END)
    df_test_raw = get_classic_pair_test(TICKER_Y, TICKER_X, start=TEST_START, end = TEST_END)

    if df_train_raw.empty or df_test_raw.empty:
        print("CRITICAL ERROR: No data found. Stopping.")
        return
    # Containers for results
    results_train = {}
    results_test = {}


    # IN-SAMPLE (TRAINING)
    print("\n" + "="*40)
    print(f"PHASE 1: IN-SAMPLE (TRAINING) {TRAIN_START} to {TRAIN_END}")
    print("="*40)

    for name, strategy in strategies.items():
        df_train = df_train_raw.copy()
        print(f"\nRunning {name}...")
        
        # Fit model if required (Static OLS needs this)
        if hasattr(strategy, 'fit'):
            strategy.fit(df_train)
            
        signals_train_full = strategy.calculate_signals(df_train)
        signals_train = signals_train_full.iloc[warmup_window:]

        res = calculate_pnl(signals_train)
        results_train[name] = res

        analyze_performance(res, f"{name} train")
        plot_diagnostic(res, f"{name} train")

        # Zoom in Graph
        # plot_diagnostic(res.iloc[200:400], f"{name} snapshot")


    print("\nGenerating Comparison Plots...")
    plot_strategy_comparison(results_train, "Train")
    print("\n" + "="*40)


    # OUT-OF-SAMPLE (TESTING)
    print(f"PHASE 2: OUT-OF-SAMPLE (TESTING) {TEST_START} to {TEST_END}")
    print("="*40)
    
    for name, strategy in strategies.items():
        df_test = df_test_raw.copy()
        print(f"\nRunning {name}...")
        
        signals_test_full = strategy.calculate_signals(df_test)
        signals_test = signals_test_full.iloc[warmup_window:]
        
        res = calculate_pnl(signals_test)
        results_test[name] = res
        
        analyze_performance(res, f"{name} test")
        plot_diagnostic(res, f"{name} test")


    print("Generating Comparison Plots...")
    plot_strategy_comparison(results_test, "Test")


    print("Done.")

if __name__ == "__main__":
    run_full_analysis()