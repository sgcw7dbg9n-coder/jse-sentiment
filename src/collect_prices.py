import logging

import pandas as pd
import yfinance as yf

from common import ROOT, append_log, load_settings, utc_now

PRICES_PATH = ROOT / "data" / "raw" / "prices.csv"
PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

log = logging.getLogger("collect_prices")


def fetch_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    history = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
    if history.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    # tz_localize(None) keeps the Johannesburg trading date; tz_convert(None) would
    # shift midnight SAST back to 22:00 UTC on the previous day
    history.index = history.index.tz_localize(None).normalize()
    history = history.rename(columns=lambda c: c.lower().replace(" ", "_"))
    return history.rename_axis("date").reset_index()[PRICE_COLUMNS]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    start, end = settings["prices"]["start"], settings["prices"]["end"]

    targets = {c["ticker"]: c["name"] for c in settings["companies"]}
    proxy = settings["market_proxy"]
    targets[proxy["ticker"]] = proxy["name"]

    run_at = utc_now()
    frames, log_rows = [], []

    for symbol, name in targets.items():
        entry = {"run_at": run_at, "source": "yfinance", "ticker": symbol, "rows": 0, "status": "ok", "message": ""}
        try:
            prices = fetch_history(symbol, start, end)
        except Exception as exc:  # yfinance surfaces HTTP, JSON and parsing failures with no common base class
            entry.update(status="error", message=str(exc)[:300])
            log.error("%s: %s", symbol, exc)
            log_rows.append(entry)
            continue

        if prices.empty:
            entry.update(status="empty", message="no rows returned; ticker may be delisted or renamed")
            log.warning("%s: no rows returned", symbol)
        else:
            prices.insert(0, "ticker", symbol)
            prices.insert(1, "company", name)
            frames.append(prices)
            entry.update(rows=len(prices), first_date=prices["date"].min().date(),
                         last_date=prices["date"].max().date())
            log.info("%s: %d rows, %s to %s", symbol, entry["rows"], entry["first_date"], entry["last_date"])
        log_rows.append(entry)

    append_log(log_rows)

    if not frames:
        raise SystemExit("No price data downloaded; see logs/collection_log.csv")

    combined = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PRICES_PATH, index=False)

    failed = [r["ticker"] for r in log_rows if r["status"] != "ok"]
    log.info("Saved %d rows for %d tickers to %s", len(combined), combined["ticker"].nunique(),
             PRICES_PATH.relative_to(ROOT))
    if failed:
        log.warning("Missing tickers: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
