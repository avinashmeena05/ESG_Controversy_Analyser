import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.data_ingest import load_news, load_portfolio, load_prices, load_sector_sensitivity
from src.shock_score import compute_esg_events, attach_shock_score
from src.simulation import compute_returns, portfolio_mc_var

st.set_page_config(page_title='ESG & Controversy Impact Analyzer', layout='wide')

st.title('🛰️ ESG & Controversy Impact Analyzer')
st.caption('Detect controversies, score ESG shocks, and simulate portfolio risk. (Demo with synthetic data)')

# Sidebar
model_path = Path('models/esg_text_clf.joblib')
if not model_path.exists():
    st.warning('Model not found. Please run: `python -m src.nlp_model --train_path data/sample_news.csv --out_dir models`')
else:
    clf = joblib.load(model_path)

news_df = load_news()
portfolio_df = load_portfolio()
prices_df = load_prices()
sector_sens_df = load_sector_sensitivity()

st.sidebar.header('Portfolio')
st.sidebar.dataframe(portfolio_df)

if model_path.exists():
    st.subheader('1) Detect ESG Controversies')
    events = compute_esg_events(clf, news_df[['ticker','text']])
    st.dataframe(events)

    st.subheader('2) Compute Shock Scores')
    shocks_df = attach_shock_score(events, portfolio_df, sector_sens_df)
    st.dataframe(shocks_df[['ticker','category','severity','sector','shock_pct']])

    # Build per-ticker max shock (worst current article per ticker)
    per_ticker_shock = shocks_df.groupby('ticker')['shock_pct'].min().to_dict()  # min since negative
    st.write('**Applied per-ticker shock (one-day %):**', per_ticker_shock)

    st.subheader('3) Portfolio Simulation (Monte Carlo)')
    returns_df = compute_returns(prices_df)
    res = portfolio_mc_var(returns_df, portfolio_df[['ticker','weight']], per_ticker_shock, horizon_days=5, n_sims=4000)
    st.metric('VaR (95%) over 5d', f"{res['VaR_95']*100:.2f}%")
    st.metric('Expected Shortfall (95%)', f"{res['ES_95']*100:.2f}%")
    st.caption('Negative values indicate losses.')
