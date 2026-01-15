# src/strategies.py
import numpy as np
import pandas as pd
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm    

class OLSTrader:
    def __init__(self, window=90, entry=1.0, exit=0.1):
        self.window = window
        self.entry = entry
        self.exit = exit

        self.hedge_ratio = None
        self.intercept = None
    
    def fit(self, df):
        y = df['price_y']
        x = sm.add_constant(df['price_x'])

        model = sm.OLS(y, x).fit()
        self.hedge_ratio = model.params['price_x']
        self.intercept = model.params['const']

        print(f"OLS Trained | Beta: {self.hedge_ratio:.4f}")

    def calculate_signals(self, df):
        if self.hedge_ratio is None:
            raise ValueError("Model is not fitted yet! Run .fit(train_df) first.")
        
        df['hedge_ratio'] = self.hedge_ratio
        df['intercept'] = self.intercept
        df['spread'] = df['price_y'] - (df['intercept'] + df['hedge_ratio'] * df['price_x'])

        df['mean_spread'] = df['spread'].rolling(window=self.window).mean()
        df['std_spread'] = df['spread'].rolling(window=self.window).std()
        df['z_score'] = df['spread']  / df['std_spread']

        df['signal'] = 0
        df.loc[df['z_score'] < -self.entry, 'signal'] = 1
        df.loc[df['z_score'] > self.entry, 'signal'] = -1

        df['position'] = df['signal'].replace(0, np.nan).ffill()
        df.loc[abs(df['z_score']) < self.exit, 'position'] = 0
        df['position'] = df['position'].fillna(0)

        return df

class RollingOLSTrader:
    def __init__(self, window=90, entry=1.0, exit=0.1):
        self.window = window
        self.entry = entry
        self.exit = exit
    
    def calculate_signals(self, df):
        y = df['price_y']
        x = sm.add_constant(df['price_x'])
        model = RollingOLS(y, x, window = self.window).fit()

        df['hedge_ratio'] = model.params['price_x']
        df['intercept'] = model.params['const']
        df['spread'] = df['price_y'] - (df['intercept'].shift(1) + df['hedge_ratio'].shift(1) * df['price_x'])

        df['std_spread'] = df['spread'].rolling(window=self.window).std()
        df['z_score'] = df['spread'] / df['std_spread']

        df['signal'] = 0
        df.loc[df['z_score'] < -self.entry, 'signal'] = 1
        df.loc[df['z_score'] > self.entry, 'signal'] = -1

        df['position'] = df['signal'].replace(0, np.nan).ffill()
        df.loc[abs(df['z_score']) < self.exit, 'position'] = 0
        df['position'] = df['position'].fillna(0)

        return df
    

class KalmanPairsTrader:
    def __init__(self, delta = 1e-5, ve = 1e-2, entry = 1.0, exit = 0.1):
        self.delta = delta
        self.Ve = ve # Measurement Noise
        self.entry = entry
        self.exit = exit

    
    def calculate_signals(self, df):
        # initialize values
        y = df['price_y'].values
        x = df['price_x'].values
        n = len(x)

        beta_t = np.zeros(2)
        R_t = np.zeros((2, 2))

        e_history = np.zeros(n) # Prediction spread
        Q_history = np.zeros(n) # Total Variance of the prediction spread
        beta_history = np.zeros((n, 2)) # state variable

        Vw = (self.delta / (1 - self.delta)) * np.eye(2)
        

        for t in range(n):
            x_t = np.array([1.0, x[t]])

            # Prediction [Hidden State]
            beta_pred = beta_t
            R_pred = R_t + Vw

            # Measurement Perdiction [Observation State]
            y_pred = np.dot(x_t, beta_pred)
            Q_t = x_t.T @ R_pred @ x_t + self.Ve # Q calculation: (1,2) @ (2,2) @ (2,1) -> Scalar


            # State Update
            e_t = y[t] - y_pred
            K = (R_pred @ x_t)/Q_t # K calculation: (2,2) @ (2,1) -> (2,1)
            beta_t = beta_pred + K * e_t
            R_t = R_pred - np.outer(K, x_t) @ R_pred # R calculation [Outer]: (2,) @ (2,) -> (2,2)

            # History Memo
            Q_history[t] = Q_t
            e_history[t] = e_t
            beta_history[t] = beta_t


        df['intercept'] = beta_history[:, 0]
        df['hedge_ratio'] = beta_history[:, 1]
        df['spread'] = e_history
        df['std_spread'] = np.sqrt(Q_history)

        
        df['signal'] = 0
        df.loc[df['spread'] < - self.entry * df['std_spread'], 'signal'] = 1
        df.loc[df['spread'] > self.entry * df['std_spread'], 'signal'] = -1
        
        
        df['position'] = df['signal'].replace(0, np.nan).ffill()

        df.loc[abs(df['spread'] ) < self.exit * df['std_spread'], 'position'] = 0
        df['position'] = df['position'].fillna(0)

        return df