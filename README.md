# JSE Sentiment

Honours research project (ITRPA4-34): does sentiment from South African financial news improve stock movement prediction for JSE-listed companies?

## Setup

```
conda env create -f environment.yml
conda activate jse-sentiment
```

`environment.yml` lists what the project needs. `environment.lock.yml` records the exact versions that were installed, so results can be reproduced.

## Layout

| Folder | Contents |
| --- | --- |
| `config/` | Tickers, date range, model parameters |
| `data/raw/` | Untouched downloads |
| `data/interim/` | Cleaned and sentiment-scored text |
| `data/processed/` | Final modelling dataset |
| `logs/` | Collection logs, including blocked or unavailable sources |
| `notebooks/` | Exploration and dissertation figures |
| `sql/` | BigQuery queries |
| `src/` | Pipeline code |
| `app/` | Streamlit prototype |
