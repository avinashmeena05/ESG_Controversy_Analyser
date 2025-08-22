import pandas as pd
from .severity_lexicon import score_severity

def compute_esg_events(clf, texts_df: pd.DataFrame):
    # texts_df columns: ticker, text
    preds = clf.predict(texts_df['text'])
    texts_df = texts_df.copy()
    texts_df['category'] = preds
    texts_df['severity'] = [
        score_severity(t, c) if c in ['E','S','G'] else 0
        for t,c in zip(texts_df['text'], preds)
    ]
    return texts_df

def attach_shock_score(events_df: pd.DataFrame, portfolio_df: pd.DataFrame, sector_sens_df: pd.DataFrame):
    df = events_df.merge(portfolio_df[['ticker','sector']], on='ticker', how='left')
    sens = sector_sens_df.copy()
    df = df.merge(sens, on='sector', how='left')
    def shock(row):
        if row['category'] not in ['E','S','G']: 
            return 0.0
        col = f"{row['category']}_drop_avg"
        drop = row.get(col, 0.0)  # in % terms
        sev = row['severity'] / 100.0  # 0..1
        return sev * float(drop)  # negative number expected
    df['shock_pct'] = df.apply(shock, axis=1)
    return df
