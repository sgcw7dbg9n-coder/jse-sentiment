import argparse
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from common import ROOT, append_log, load_settings, utc_now

OUTPUT_PATH = ROOT / "data" / "raw" / "sens_announcements.parquet"
INDEX_PATH = "/communication/sens-announcements/Index/{issuer_id}/_/{start}/{end}/{page}/"
RETRY_STATUSES = {429, 500, 502, 504}
MAX_RETRIES = 2

log = logging.getLogger("collect_sens")


class UnexpectedPage(Exception):
    pass


def parse_page(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    form = soup.select_one("form#sa-form")
    if form is None:
        raise UnexpectedPage("SENS search form not found; the site may have shown a bot check or changed its layout")

    def form_value(selector: str) -> str:
        field = form.select_one(selector)
        return field.get("value", "") if field else ""

    issuer_value = form_value("input#sa-issuer-id")

    rows = []
    for row in soup.select("div.sa-list div.sa-row"):
        headline = row.select_one("a.sa-headline")
        roles = row.select_one("span.sa-roles")
        rows.append({
            "sens_id": row.select_one("span.sa-ref").get_text(strip=True),
            "title": headline.get_text(" ", strip=True),
            "published_local": row.select_one("span.sa-time").get_text(strip=True),
            "issuers": "; ".join(pill.get_text(strip=True) for pill in row.select("span.sa-issuer-pill")),
            "roles": roles.get_text(" ", strip=True) if roles else "",
            "pdf_url": headline.get("href", ""),
        })

    page_info = soup.select_one("span.sa-page-info")
    match = re.search(r"Page\s+(\d+)\s+of\s+(\d+)", page_info.get_text()) if page_info else None
    return {
        "issuer_id": int(issuer_value) if issuer_value.isdigit() else None,
        "start": form_value("input#sa-from"),
        "end": form_value("input#sa-to"),
        "page": int(match.group(1)) if match else 1,
        "total_pages": int(match.group(2)) if match else 1,
        "rows": rows,
    }


def fetch(session: requests.Session, url: str, delay: float) -> str:
    for attempt in range(MAX_RETRIES + 1):
        time.sleep(delay if attempt == 0 else delay * 10 * attempt)
        response = session.get(url, timeout=30)
        if response.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES:
            response.raise_for_status()
            return response.text
        log.warning("HTTP %d for %s; waiting before retry", response.status_code, url)


def collect_online(settings: dict) -> list[pd.DataFrame]:
    sens = settings["sens"]
    start, end = settings["study_period"]["start"], settings["study_period"]["end"]
    session = requests.Session()
    session.headers["User-Agent"] = sens["user_agent"]

    run_at = utc_now()
    frames, log_rows = [], []
    try:
        for company in settings["companies"]:
            for issuer_id in company["sens_ids"]:
                rows, page, total_pages = [], 1, 1
                while page <= total_pages:
                    url = sens["base_url"] + INDEX_PATH.format(issuer_id=issuer_id, start=start, end=end, page=page)
                    parsed = parse_page(fetch(session, url, sens["delay_seconds"]))
                    rows.extend(parsed["rows"])
                    total_pages = parsed["total_pages"]
                    page += 1

                log.info("%s issuer %d: %d announcements over %d pages", company["ticker"], issuer_id, len(rows), total_pages)
                log_rows.append({"run_at": run_at, "source": "jse_sens", "ticker": company["ticker"], "rows": len(rows),
                                 "status": "ok" if rows else "empty", "message": f"issuer {issuer_id}"})
                if rows:
                    frames.append(pd.DataFrame(rows).assign(ticker=company["ticker"], issuer_id=issuer_id))
    except (requests.RequestException, UnexpectedPage) as exc:
        log_rows.append({"run_at": run_at, "source": "jse_sens", "ticker": "ALL", "rows": 0,
                         "status": "error", "message": str(exc)[:300]})
        append_log(log_rows)
        raise SystemExit(f"Stopped without bypassing anything: {exc}\n"
                         "If the site is showing a bot check, save the result pages from your browser "
                         "and run: python src\\collect_sens.py --from-html <folder>")

    append_log(log_rows)
    return frames


def collect_offline(settings: dict, folder: Path) -> list[pd.DataFrame]:
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")
    paths = sorted(folder.glob("*.htm*"))
    if not paths:
        raise SystemExit(f"No saved .html pages in {folder}")

    id_to_ticker = {issuer_id: c["ticker"] for c in settings["companies"] for issuer_id in c["sens_ids"]}
    study = (settings["study_period"]["start"], settings["study_period"]["end"])
    frames, pages_seen, totals = [], {}, {}

    for path in paths:
        try:
            parsed = parse_page(path.read_text(encoding="utf-8"))
        except UnexpectedPage:
            log.warning("%s: not a SENS results page; skipped", path.name)
            continue
        issuer_id = parsed["issuer_id"]
        ticker = id_to_ticker.get(issuer_id)
        if ticker is None:
            log.warning("%s: issuer %s is not one of the selected companies; skipped", path.name, issuer_id)
            continue
        if (parsed["start"], parsed["end"]) != study:
            log.warning("%s: saved with dates %s to %s, expected %s to %s; page numbers may not line up",
                        path.name, parsed["start"], parsed["end"], *study)

        pages_seen.setdefault(issuer_id, set()).add(parsed["page"])
        totals[issuer_id] = parsed["total_pages"]
        if parsed["rows"]:
            frames.append(pd.DataFrame(parsed["rows"]).assign(ticker=ticker, issuer_id=issuer_id))

    for company in settings["companies"]:
        for issuer_id in company["sens_ids"]:
            if issuer_id not in pages_seen:
                log.warning("%s issuer %d: no saved pages (fine only if it has no announcements)", company["ticker"], issuer_id)
                continue
            missing = sorted(set(range(1, totals[issuer_id] + 1)) - pages_seen[issuer_id])
            status = f"missing pages {missing}" if missing else "complete"
            log.info("%s issuer %d: %d of %d pages, %s", company["ticker"], issuer_id,
                     len(pages_seen[issuer_id]), totals[issuer_id], status)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect JSE SENS announcement headlines for the selected companies")
    parser.add_argument("--from-html", type=Path, help="read saved result pages from this folder instead of the website")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    frames = collect_offline(settings, args.from_html) if args.from_html else collect_online(settings)
    if not frames:
        raise SystemExit("No announcements collected")

    sens = pd.concat(frames, ignore_index=True)
    # The portal shows Johannesburg time; store UTC to match the news data
    sens["published_at"] = (
        pd.to_datetime(sens.pop("published_local"), format="%Y/%m/%d %H:%M")
        .dt.tz_localize(settings["market"]["timezone"])
        .dt.tz_convert("UTC")
    )
    sens = (
        sens.drop_duplicates(subset=["ticker", "sens_id"])
        .sort_values(["ticker", "published_at"])
        .reset_index(drop=True)[["ticker", "issuer_id", "sens_id", "published_at", "title", "issuers", "roles", "pdf_url"]]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sens.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved %d announcements to %s", len(sens), OUTPUT_PATH.relative_to(ROOT))

    summary = sens.groupby("ticker").agg(
        announcements=("sens_id", "size"), first=("published_at", "min"), last=("published_at", "max")
    )
    print()
    print(summary.to_string())


if __name__ == "__main__":
    main()