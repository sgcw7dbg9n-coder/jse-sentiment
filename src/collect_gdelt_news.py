import logging
import os

import pandas as pd
import pydata_google_auth
from dotenv import load_dotenv
from google.cloud import bigquery

from common import ROOT, append_log, is_sa_domain, load_settings, utc_now

SCOPES = ["https://www.googleapis.com/auth/bigquery"]
TOP_DOMAINS = 40

log = logging.getLogger("collect_gdelt_news")


def job_config(settings: dict, *, dry_run: bool = False, max_bytes: int | None = None) -> bigquery.QueryJobConfig:
    companies = settings["companies"]
    patterns = [c["pattern"] for c in companies]
    params = [
        bigquery.ScalarQueryParameter("start_date", "STRING", settings["study_period"]["start"]),
        bigquery.ScalarQueryParameter("end_date", "STRING", settings["study_period"]["end"]),
        bigquery.ArrayQueryParameter("tickers", "STRING", [c["ticker"] for c in companies]),
        bigquery.ArrayQueryParameter("patterns", "STRING", patterns),
        bigquery.ScalarQueryParameter("any_pattern", "STRING", "|".join(f"(?:{p})" for p in patterns)),
    ]
    # A cached result reports 0 bytes, so the dry run must bypass the cache to give a real estimate
    config = bigquery.QueryJobConfig(query_parameters=params, dry_run=dry_run, use_query_cache=not dry_run)
    # Assigning None sends the literal string "None" to the API, so only set a real limit
    if max_bytes is not None:
        config.maximum_bytes_billed = max_bytes
    return config


def domain_report(articles: pd.DataFrame, settings: dict) -> pd.DataFrame:
    report = articles.groupby("domain").agg(headlines=("url", "size"), companies=("ticker", "nunique"))
    report["sa_outlet"] = is_sa_domain(report.index.to_series(), settings).map({True: "yes", False: ""})
    return report.sort_values("headlines", ascending=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise SystemExit("Set GCP_PROJECT_ID in .env to your personal Google Cloud project ID")

    settings = load_settings()
    gdelt = settings["gdelt"]
    sql = (ROOT / gdelt["query"]).read_text(encoding="utf-8")
    output_path = ROOT / gdelt["output"]
    max_gb = gdelt["max_gb"]

    credentials = pydata_google_auth.get_user_credentials(SCOPES)
    client = bigquery.Client(project=project_id, credentials=credentials)

    estimate_gb = client.query(sql, job_config=job_config(settings, dry_run=True)).total_bytes_processed / 1e9
    log.info("Query will scan %.1f GB (limit %s GB)", estimate_gb, max_gb)
    if estimate_gb > max_gb:
        raise SystemExit(f"Estimated scan of {estimate_gb:.1f} GB is over gdelt.max_gb; query not run")

    run_at = utc_now()
    try:
        job = client.query(sql, job_config=job_config(settings, max_bytes=int(max_gb * 1e9)))
        raw = job.to_dataframe(create_bqstorage_client=False)
    except Exception as exc:
        append_log([{"run_at": run_at, "source": "gdelt_gkg", "ticker": "ALL", "rows": 0,
                     "status": "error", "message": str(exc)[:300]}])
        raise

    articles = (
        raw.sort_values("published_at")
        .drop_duplicates(subset=["url", "ticker"])
        .reset_index(drop=True)
    )
    duplicates = len(raw) - len(articles)
    log.info("Fetched %d rows, %d after removing %d duplicate url/ticker pairs", len(raw), len(articles), duplicates)

    if articles.empty:
        append_log([{"run_at": run_at, "source": "gdelt_gkg", "ticker": "ALL", "rows": 0,
                     "status": "empty", "message": "query returned no matching headlines"}])
        raise SystemExit("No matching headlines returned; check patterns in settings.yaml")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(output_path, index=False)

    append_log([{
        "run_at": run_at, "source": "gdelt_gkg", "ticker": "ALL", "rows": len(articles),
        "first_date": articles["published_at"].min().date(), "last_date": articles["published_at"].max().date(),
        "status": "ok", "message": f"{job.total_bytes_processed / 1e9:.1f} GB scanned; {duplicates} duplicates removed",
    }])

    domains = domain_report(articles, settings)
    report_path = ROOT / "reports" / "gdelt_domains.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    domains.to_csv(report_path)
    sa = is_sa_domain(articles["domain"], settings)

    log.info("Saved %d headlines to %s", len(articles), output_path.relative_to(ROOT))
    print(f"\nSA outlets: {sa.sum()} headlines from {articles.loc[sa, 'domain'].nunique()} outlets | "
          f"other outlets: {(~sa).sum()} headlines from {articles.loc[~sa, 'domain'].nunique()} outlets")
    print(f"\nTop {TOP_DOMAINS} outlets:")
    print(domains.head(TOP_DOMAINS).to_string())
    print("\nHeadlines per company (SA outlets / all outlets):")
    print(pd.DataFrame({"sa": articles[sa].groupby("ticker").size(), "all": articles.groupby("ticker").size()})
          .fillna(0).astype(int).sort_values("sa", ascending=False).to_string())


if __name__ == "__main__":
    main()