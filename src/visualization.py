# src/visualization.py
import matplotlib.pyplot as plt

def plot_diagnostic(df, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Panel 1: Predicted Price vs True Price
    ax = axes[0]
    ax.plot(df.index, df['price_y'], label='Asset Y (Target)', color='blue', alpha=0.6)
    
    fair_value = df['intercept'] + df['hedge_ratio'] * df['price_x']
    ax.plot(df.index, fair_value, label='Predicted Y (Fair Value)', color='orange', linestyle='--', alpha=0.8)

    ax.set_title(f"{title}: Predicted Price vs True Price")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Z-Score (Spread Deviation)
    ax = axes[1]
    # Rolling OLS and OLS
    if 'z_score' in df.columns:
        ax.plot(df.index, df['z_score'], label='Z-Score', color='purple', linewidth=1)
        ax.axhline(0, color='black', alpha=0.5)
        ax.axhline(1.0, color='green', linestyle='--', alpha=0.5)
        ax.axhline(-1.0, color='green', linestyle='--', alpha=0.5)
        ax.set_title("Standard Z-Score (Rolling Std)")
        ax.set_ylabel("σ")

    # Kalman Filter
    else:
        ax.plot(df.index, df['spread'], label='Spread (Error)', color='blue', linewidth=1)

        upper = 1.0 * df['std_spread']
        lower = -1.0 * df['std_spread']
        
        ax.plot(df.index, upper, color='green', linestyle='--', alpha=0.8, linewidth=1)
        ax.plot(df.index, lower, color='green', linestyle='--', alpha=0.8, linewidth=1)
        ax.fill_between(df.index, upper, lower, color='green', alpha=0.1)
        
        ax.set_title("Spread & Dynamic Uncertainty (Q)")
        ax.set_ylabel("Error")

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
        

    # Panel 3: Equity Curve
    ax = axes[2]
    ax.plot(df.index, df['equity_curve'], label='Equity', color='green', linewidth=1.5)
    ax.axhline(1.0, color='black', linewidth=1)
    
    ax.set_title("Strategy Equity Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    file_name = f"results/{title.replace(' ', '_').lower()}.png"
    plt.savefig(file_name)
    print(f"Plot saved to {file_name}\n")
    plt.show()


def plot_strategy_comparison(results_dict, title_suffix):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Equity Curves
    ax = axes[0]
    for name, df in results_dict.items():
        final_ret = (df['equity_curve'].iloc[-1] - 1) * 100
        ax.plot(df.index, df['equity_curve'], label=f"{name} ({final_ret:.1f}%)", linewidth=1.5)
    
    ax.axhline(1.0, color='black', linewidth=1, linestyle='--')
    ax.set_title(f"Equity Curve Comparison ({title_suffix})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Hedge Ratios (Beta)
    ax = axes[1]
    for name, df in results_dict.items():
        if 'hedge_ratio' in df.columns:
            ax.plot(df.index, df['hedge_ratio'], label=name, linewidth=1)
            
    ax.set_title("Hedge Ratio (Beta) Stability")
    ax.set_ylabel("Beta")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"results/Comparison_{title_suffix.lower()}.png"
    plt.savefig(filename)
    print(f"Saved Comparison: {filename}")
    plt.show()