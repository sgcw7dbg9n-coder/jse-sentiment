import json
import logging
import re
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
import transformers
from tqdm import tqdm
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from common import ROOT, load_settings, utc_now

NEWS_PATH = ROOT / "data" / "interim" / "news_clean.parquet"
OUTPUT_PATH = ROOT / "data" / "interim" / "news_scored.parquet"
METADATA_PATH = ROOT / "reports" / "scoring_metadata.json"
TOKEN = re.compile(r"[A-Z]+")
CHUNK = 256

log = logging.getLogger("score_news")


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def vader_scores(titles: pd.Series) -> pd.Series:
    analyzer = SentimentIntensityAnalyzer()
    return titles.map(lambda title: analyzer.polarity_scores(title)["compound"]).rename("vader_score")


def finbert_scores(titles: pd.Series, model_name: str, batch_size: int) -> tuple[pd.DataFrame, str]:
    classifier = pipeline("text-classification", model=model_name, top_k=None, truncation=True)
    texts = titles.tolist()
    outputs = []
    for start in tqdm(range(0, len(texts), CHUNK), desc="FinBERT", unit="chunk"):
        outputs.extend(classifier(texts[start:start + CHUNK], batch_size=batch_size))

    probs = pd.DataFrame([{item["label"]: item["score"] for item in result} for result in outputs], index=titles.index)
    probs = probs.rename(columns=lambda label: f"finbert_{label.lower()}")
    probs["finbert_score"] = probs["finbert_positive"] - probs["finbert_negative"]
    revision = getattr(classifier.model.config, "_commit_hash", None) or "unknown"
    return probs, revision


def load_lm_dictionary(path) -> tuple[set[str], set[str]]:
    if not path.exists():
        raise SystemExit(
            f"Loughran-McDonald dictionary not found at {path.relative_to(ROOT)}\n"
            "Download the CSV from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ into data\\external\\"
        )
    # keep_default_na=False stops genuine dictionary words such as NULL and NA being read as missing
    lm = pd.read_csv(path, usecols=["Word", "Positive", "Negative"], keep_default_na=False)
    words = lm["Word"].astype(str).str.upper()
    # The category columns hold the year a word was added; zero or negative means not (or no longer) in the list
    return set(words[lm["Positive"] > 0]), set(words[lm["Negative"] > 0])


def lm_scores(titles: pd.Series, positive: set[str], negative: set[str]) -> pd.DataFrame:
    tokens = titles.map(lambda title: TOKEN.findall(title.upper()))
    pos = tokens.map(lambda words: sum(word in positive for word in words))
    neg = tokens.map(lambda words: sum(word in negative for word in words))
    hits = pos + neg
    return pd.DataFrame({
        "lm_positive": pos,
        "lm_negative": neg,
        "lm_score": ((pos - neg) / hits.where(hits > 0)).fillna(0.0),
    })


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    config = settings["sentiment"]

    news = pd.read_parquet(NEWS_PATH)
    positive, negative = load_lm_dictionary(ROOT / config["lm_dictionary"])
    log.info("Scoring %d headlines", len(news))

    finbert, revision = finbert_scores(news["title"], config["finbert_model"], config["batch_size"])
    scored = pd.concat([news, vader_scores(news["title"]), finbert, lm_scores(news["title"], positive, negative)], axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUTPUT_PATH, index=False)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps({
        "scored_at": utc_now(),
        "headlines": len(scored),
        "finbert_model": config["finbert_model"],
        "finbert_revision": revision,
        "transformers": transformers.__version__,
        "vadersentiment": package_version("vaderSentiment"),
        "lm_dictionary": ROOT.joinpath(config["lm_dictionary"]).name,
        "lm_positive_words": len(positive),
        "lm_negative_words": len(negative),
    }, indent=2), encoding="utf-8")
    log.info("Saved scores to %s", OUTPUT_PATH.relative_to(ROOT))

    score_columns = ["vader_score", "finbert_score", "lm_score", "gdelt_tone"]
    finbert_label = finbert[["finbert_positive", "finbert_negative", "finbert_neutral"]].idxmax(axis=1).str.replace("finbert_", "")

    print("\nFinBERT labels:")
    print(finbert_label.value_counts(normalize=True).map("{:.1%}".format).to_string())
    print(f"\nHeadlines with any Loughran-McDonald word: {(scored['lm_positive'] + scored['lm_negative'] > 0).mean():.1%}")
    print(f"Headlines VADER rates exactly neutral: {(scored['vader_score'] == 0).mean():.1%}")
    print("\nSummary:")
    print(scored[score_columns].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
    print("\nSpearman correlation between scorers:")
    print(scored[score_columns].corr(method="spearman").round(2).to_string())

    ranked = scored.sort_values("finbert_score")
    for label, rows in (("Most negative (FinBERT)", ranked.head(5)), ("Most positive (FinBERT)", ranked.tail(5)[::-1])):
        print(f"\n{label}:")
        for row in rows.itertuples():
            print(f"  {row.finbert_score:+.2f} vader {row.vader_score:+.2f} lm {row.lm_score:+.2f}  {row.ticker}  {row.title[:90]}")


if __name__ == "__main__":
    main()
