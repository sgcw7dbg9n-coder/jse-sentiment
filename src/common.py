from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "collection_log.csv"
LOG_COLUMNS = ["run_at", "source", "ticker", "rows", "first_date", "last_date", "status", "message"]


def load_settings() -> dict:
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_log(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=LOG_COLUMNS).to_csv(
        LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False
    )


def is_sa_domain(domains: pd.Series, settings: dict) -> pd.Series:
    gdelt = settings["gdelt"]
    return domains.str.endswith(tuple(gdelt["sa_domain_suffixes"])) | domains.isin(gdelt["sa_extra_domains"])