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

## Pipeline

Run from the project folder, in this order:

```
python src\collect_prices.py
python src\collect_gdelt_news.py
python src\clean_news.py
python src\score_news.py
python src\build_features.py
python src\train_models.py
python src\train_models.py --sensitivity
python src\train_models.py --news-days-only
python src\robustness.py
```

## Prototype

```
streamlit run app\dashboard.py
```

The dashboard reads the files the pipeline writes to `data/` and `reports/`; it doesn't train anything itself. Pages:

| Page | Shows |
| --- | --- |
| Overview | The answer to the research question, computed from the significance tests |
| Model comparison | Test AUC for every model and feature set, AUC change with intervals, DeLong and McNemar results, ROC curves and confusion matrices |
| Sentiment and returns | Panel regressions and returns by sentiment third |
| Company explorer | Share price, rolling sentiment and headline volume for one company, with its headlines |
| Headlines and scorers | How VADER, FinBERT, Loughran-McDonald and GDELT tone agree and disagree |
| Test-period predictions | Day-by-day out-of-sample predictions for one company |
| Score a headline | Runs the scorers on any headline you type |
| Data and method | Pipeline, cleaning steps, features, run metadata, ethics and limitations |
