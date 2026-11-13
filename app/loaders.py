import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common import ROOT, load_settings  # noqa: E402

DATA = ROOT / "data"
REPORTS = ROOT / "reports"

# The --sensitivity run repeats the main target with every feature set, so it is a superset of the plain run
SAMPLE_SUFFIXES = {"all_days": ["_sensitivity", ""], "news_days": ["_newsdays"]}
BOOTSTRAP_SAMPLES = {"all_days": "all_days", "news_days": "news_days_only"}


@st.cache_data(show_spinner=False)
def _read(path: str, modified: float) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def read(path: Path) -> pd.DataFrame | None:
    # The modified time is part of the cache key, so re-running a pipeline step refreshes the app
    return _read(str(path), path.stat().st_mtime) if path.exists() else None


def require(frame, script: str, flags: str = ""):
    if frame is None:
        st.info(f"This page needs output that isn't there yet. From the project folder run "
                f"`python src\\{script}{' ' + flags if flags else ''}`, then refresh the page.")
        st.stop()
    return frame


@st.cache_data(show_spinner=False)
def _settings(modified: float) -> dict:
    return load_settings()


def settings() -> dict:
    return _settings((ROOT / "config" / "settings.yaml").stat().st_mtime)


def companies() -> pd.DataFrame:
    return pd.DataFrame(settings()["companies"])[["ticker", "name", "sector"]]


def company_names() -> dict[str, str]:
    return dict(companies()[["ticker", "name"]].itertuples(index=False))


def news() -> pd.DataFrame | None:
    frame = read(DATA / "interim" / "news_scored.parquet")
    if frame is None:
        return None
    timezone = settings()["market"]["timezone"]
    labels = frame[["finbert_positive", "finbert_negative", "finbert_neutral"]].idxmax(axis=1)
    return frame.assign(
        published_local=frame["published_at"].dt.tz_convert(timezone).dt.tz_localize(None),
        finbert_label=labels.str.removeprefix("finbert_"),
        company=frame["ticker"].map(company_names()),
    )


def dataset() -> pd.DataFrame | None:
    frame = read(DATA / "processed" / "model_dataset.parquet")
    if frame is not None:
        # yfinance quotes JSE shares in cents
        frame["close_rand"] = frame["close"] / 100
    return frame


def _first_existing(stem: str, folder: Path, suffix: str, sample: str) -> pd.DataFrame | None:
    for variant in SAMPLE_SUFFIXES[sample]:
        frame = read(folder / f"{stem}{variant}{suffix}")
        if frame is not None:
            return frame
    return None


def results(sample: str) -> pd.DataFrame | None:
    return _first_existing("model_results", REPORTS, ".csv", sample)


def tests(sample: str) -> pd.DataFrame | None:
    return _first_existing("significance_tests", REPORTS, ".csv", sample)


def predictions(sample: str) -> pd.DataFrame | None:
    return _first_existing("test_predictions", DATA / "processed", ".parquet", sample)


def bootstrap(sample: str) -> pd.DataFrame | None:
    frame = read(REPORTS / "bootstrap_comparisons.csv")
    if frame is None:
        return None
    return frame[frame["sample"] == BOOTSTRAP_SAMPLES[sample]]


def report(name: str) -> pd.DataFrame | None:
    return read(REPORTS / name)


def report_json(name: str) -> dict | None:
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
