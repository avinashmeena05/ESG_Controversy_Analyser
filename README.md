# AI-Powered ESG & Controversy Impact Analyzer

A recruiter-ready, end-to-end project that detects ESG controversies from unstructured text, assigns an **ESG Shock Score**, and simulates **portfolio impact** under controversy scenarios.

## Features
- **NLP pipeline** (TF–IDF + linear model) classifies E / S / G controversies
- **Severity scoring** using a transparent keyword lexicon
- **Shock Score** combines severity with sector-level historical sensitivity
- **Portfolio simulation** (Monte Carlo + VaR-style downside view)
- **Streamlit dashboard** for interactive exploration

## Project Structure
```
esg_controversy_analyzer/
├── app/
│   └── dashboard_app.py
├── data/
│   ├── sample_news.csv
│   ├── sample_portfolio.csv
│   ├── sample_prices.csv
│   └── sector_sensitivity.csv
├── models/                 # trained model + vectorizer stored here
├── notebooks/              # (optional) your EDA notebooks
├── src/
│   ├── data_ingest.py
│   ├── nlp_model.py
│   ├── severity_lexicon.py
│   ├── shock_score.py
│   └── simulation.py
├── requirements.txt
└── README.md
```

## How it works

### 1) NLP Controversy Classifier
- Uses TF–IDF features + LinearSVC to classify text into one of **E**, **S**, **G**, or **NONE**.
- Ships with a small toy dataset; replace `data/sample_news.csv` with your own labeled data to improve.

### 2) Severity Scoring (Transparent)
- A small keyword lexicon assigns severity points by category (e.g., `spill`, `fraud`, `bribery`, `child labor`).
- Final severity ∈ [0, 100].

### 3) ESG Shock Score
```
ShockScore = severity_scaled * sector_drop_avg(category, sector)
```
Where `sector_drop_avg` is a historical average one-day move (in %) when similar controversies occurred (synthetic here; replace with real estimates).

### 4) Portfolio Impact Simulation
- Takes your portfolio weights and price history.
- Applies **category-specific shocks** and simulates returns (Monte Carlo) to estimate downside (VaR-like).

## Extend with Real Data
- **News**: GDELT, NewsAPI
- **Filings**: SEC EDGAR (10-K/8-K)
- **NGO Reports**: PDF parsing (use `pdfminer.six`), then feed into classifier
- **Prices**: `yfinance`

## License
MIT
