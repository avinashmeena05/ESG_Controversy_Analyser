# AI-Powered ESG & Controversy Impact Analyzer

This project is an end-to-end ESG analytics tool that detects environmental, social, and governance (ESG) controversies from unstructured text, assigns an ESG Shock Score, and simulates the potential portfolio impact under controversy scenarios.

Built with NLP, risk modeling, and interactive visualization, it helps investors, analysts, and consultants assess ESG risks in real time and quantify their effect on portfolio downside.

## Features
- **NLP pipeline** (TF–IDF + linear model) classifies E / S / G controversies
- **Severity scoring** using a transparent keyword lexicon
- **Shock Score** combines severity with sector-level historical sensitivity
- **Portfolio simulation** (Monte Carlo + VaR-style downside view)
- **Streamlit dashboard** for interactive exploration
