import pandas as pd

def load_portfolio(path='data/sample_portfolio.csv'):
    return pd.read_csv(path)

def load_prices(path='data/sample_prices.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def load_news(path='data/sample_news.csv'):
    return pd.read_csv(path)

def load_sector_sensitivity(path='data/sector_sensitivity.csv'):
    return pd.read_csv(path)
