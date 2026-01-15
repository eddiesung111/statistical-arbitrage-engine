# src/__init__.py

# 1. Strategies
from .strategies import OLSTrader, RollingOLSTrader, KalmanPairsTrader

# 2. Data
from .data_loader import get_classic_pair_train, get_classic_pair_test

# 3. Backtesting
from .backtesting import calculate_pnl, calculate_metrics, analyze_performance

# 4. Visualization
from .visualization import plot_diagnostic, plot_strategy_comparison
