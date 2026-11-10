# Football Player Market Value Predictor

Predicts a player's Transfermarkt-style market value from the 12 months of stats before the valuation date.

## Run
1. `pip install -r requirements.txt`
2. `python src/download_data.py`   (or put the Kaggle CSVs in data/raw/)
3. `python src/build_dataset.py`
4. `python src/train.py`
5. `streamlit run src/app.py`

## Design choices
- Labels: public Transfermarkt valuations (CC0 dataset) - football-data.org / API-Football have no value labels.
- No leakage: features come only from the 365 days before each valuation.
- Target is log(value); metrics are reported back in euros.
- Chronological split: train on the past, test on the most recent year.
- `src/api_football.py` is optional, for fetching live stats to feed the app.

## v3: future value forecasting
- Target = the player's market value ~12 months after the reference date (HORIZON_DAYS in src/config.py).
- Features = last-12-month stats, career totals, age/position/league, current + earlier valuations.
- `with_current_value` predicts the growth from today's value; `stats_only` is the fallback.
- Purged chronological split; benchmark "Naive (no change)" in models/metrics.csv - a forecast is only useful if it beats it.
- Direction accuracy = how often the model correctly predicts that a value rises vs falls.
