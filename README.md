# NBA Hall of Fame Predictor

A random forest classifier that predicts NBA Hall of Fame induction probability based on career statistics and accolades.

**Live app:** [Coming soon on Streamlit Cloud]

## Overview

- **380 players** in training set (157 HOF inductees + 223 non-HOF)
- **25 active candidates** predicted (retired after 2020)
- **19 features:** career averages, shooting percentages, win shares, All-Star/All-NBA/All-Defense selections
- **94.7% accuracy** (10-fold stratified cross-validation)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Update data

Scrapes fresh data from [Basketball Reference](https://www.basketball-reference.com/):

```bash
python update_data.py
```

## Files

- `app.py` — Streamlit app with prediction, comparison, and visualization pages
- `data.csv` — Training dataset (HOF members + retired non-HOF candidates)
- `potentials.csv` — Current candidates not yet inducted
- `update_data.py` — Scraper to refresh all data from Basketball Reference
- `requirements.txt` — Python dependencies

## History

Originally built in R/Shiny at Carleton College (2020). Rebuilt in Python/Streamlit (2026) with expanded data through 2026.

[keelp2.github.io](https://keelp2.github.io)