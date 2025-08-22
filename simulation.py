import numpy as np
import pandas as pd

def compute_returns(price_df: pd.DataFrame):
    df = price_df.sort_values(['ticker','date']).copy()
    df['ret'] = df.groupby('ticker')['close'].pct_change()
    df = df.dropna(subset=['ret'])
    return df

def portfolio_mc_var(returns_df: pd.DataFrame, weights_df: pd.DataFrame, shocks: dict, horizon_days=5, n_sims=5000, seed=42):
    # returns_df: columns [date, ticker, ret]
    np.random.seed(seed)
    tickers = list(weights_df['ticker'])
    weights = weights_df.set_index('ticker')['weight'].reindex(tickers).values
    # historical mean/vol per ticker
    stats = returns_df.groupby('ticker')['ret'].agg(['mean','std']).reindex(tickers)
    mu = stats['mean'].fillna(0).values
    sigma = stats['std'].replace(0,np.nan).fillna(stats['std'].mean()).values
    # covariance (simplified to diagonal for demo)
    cov = np.diag(sigma**2)
    # build shock vector (percentage drop applied day 0)
    shock_vec = np.array([shocks.get(t, 0.0) for t in tickers])  # negative numbers
    # simulate horizon
    port_rets = []
    for _ in range(n_sims):
        # day 0 shock
        day0 = shock_vec
        # subsequent days random normals
        rand = np.random.normal(mu, sigma, size=(horizon_days, len(tickers)))
        path = np.vstack([day0, rand])
        # aggregate to portfolio daily returns then cumulative
        daily_port = path @ weights
        cum = (1 + daily_port).prod() - 1
        port_rets.append(cum)
    port_rets = np.array(port_rets)
    var_95 = np.quantile(port_rets, 0.05)
    es_95 = port_rets[port_rets<=var_95].mean()
    return {'VaR_95': float(var_95), 'ES_95': float(es_95), 'distribution': port_rets}
