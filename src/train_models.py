import argparse
import json
import logging
import re

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from common import ROOT, load_settings, utc_now

DATASET_PATH = ROOT / "data" / "processed" / "model_dataset.parquet"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "test_predictions.parquet"
REPORTS = ROOT / "reports"

log = logging.getLogger("train_models")


def build_feature_sets(settings: dict) -> dict[str, list[str]]:
    scores = settings["features"]["scores"]
    windows = settings["features"]["rolling_windows"]
    baseline = settings["modelling"]["baseline_features"]
    common = ["has_news", "news_count"] + [f"news_count_{w}d" for w in windows]

    def score_columns(score: str) -> list[str]:
        columns = [f"{score}_mean"] + [f"{score}_mean_{w}d" for w in windows]
        return columns + (["finbert_pos_share", "finbert_neg_share"] if score == "finbert_score" else [])

    sets = {"baseline": baseline, "all_sentiment": baseline + common + [c for s in scores for c in score_columns(s)]}
    sets.update({score: baseline + common + score_columns(score) for score in scores})
    return sets


def build_models(settings: dict) -> dict[str, tuple[Pipeline, dict]]:
    seed = settings["modelling"]["random_state"]
    estimators = {
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
        "random_forest": Pipeline([("model", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1))]),
        "svm_rbf": Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", random_state=seed))]),
    }
    grids = settings["modelling"]["grids"]
    return {name: (estimator, {f"model__{k}": v for k, v in grids[name].items()}) for name, estimator in estimators.items()}


def horizon_of(target: str) -> int:
    return int(re.search(r"_(\d+)d$", target).group(1))


def date_splits(dates: np.ndarray, n_splits: int, gap: int) -> list[tuple[np.ndarray, np.ndarray]]:
    # Expanding-window folds over whole trading days, so one day's rows never sit on both sides of a split.
    # The last `gap` training days are dropped because their targets look into the validation block.
    unique = np.sort(np.unique(dates))
    block = len(unique) // (n_splits + 1)
    splits = []
    for k in range(1, n_splits + 1):
        train_days = unique[: k * block - gap]
        val_days = unique[k * block: (k + 1) * block] if k < n_splits else unique[k * block:]
        splits.append((np.flatnonzero(np.isin(dates, train_days)), np.flatnonzero(np.isin(dates, val_days))))
    return splits


def midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(len(x))
    result[order] = ranks
    return result


def delong_test(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float, float, float]:
    """DeLong et al. (1988) test for two correlated ROC AUCs, using the fast algorithm of Sun and Xu (2014)."""
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    n = len(y) - m
    preds = np.vstack([scores_a, scores_b])[:, order]

    tx = np.array([midrank(p[:m]) for p in preds])
    ty = np.array([midrank(p[m:]) for p in preds])
    tz = np.array([midrank(p) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n

    variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[1] - aucs[0]) / np.sqrt(variance) if variance > 0 else 0.0
    return aucs[0], aucs[1], z, 2 * norm.sf(abs(z))


def mcnemar_test(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float, int, int]:
    correct_a, correct_b = pred_a == y, pred_b == y
    only_a, only_b = int(np.sum(correct_a & ~correct_b)), int(np.sum(~correct_a & correct_b))
    table = [[int(np.sum(correct_a & correct_b)), only_a], [only_b, int(np.sum(~correct_a & ~correct_b))]]
    # The exact binomial version is the safer choice when few predictions disagree
    result = mcnemar(table, exact=(only_a + only_b) < 25, correction=True)
    return float(result.statistic), float(result.pvalue), only_a, only_b


def run_target(data: pd.DataFrame, target: str, set_names: list[str], settings: dict) -> tuple[list, list, list]:
    cfg = settings["modelling"]
    feature_sets = build_feature_sets(settings)
    horizon = horizon_of(target)
    train_end = pd.Timestamp(cfg["train_end"])

    data = data.dropna(subset=[target])
    sessions = np.sort(data["date"].unique())
    train_sessions = sessions[sessions <= train_end]
    # Purge the final training days whose targets reach past train_end into the test period
    train = data[data["date"].isin(train_sessions[:-horizon])].reset_index(drop=True)
    test = data[data["date"] > train_end].reset_index(drop=True)
    y_train, y_test = train[target].astype(int).to_numpy(), test[target].astype(int).to_numpy()
    splits = date_splits(train["date"].to_numpy(), cfg["cv_splits"], horizon)

    results, predictions = [], []
    for set_name in set_names:
        columns = feature_sets[set_name]
        for model_name, (estimator, grid) in build_models(settings).items():
            search = GridSearchCV(estimator, grid, cv=splits, scoring="roc_auc", n_jobs=-1, refit=True)
            search.fit(train[columns], y_train)
            model = search.best_estimator_
            scores = (model.predict_proba(test[columns])[:, 1] if hasattr(model, "predict_proba")
                      else model.decision_function(test[columns]))
            pred = model.predict(test[columns])
            tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

            results.append({
                "target": target, "feature_set": set_name, "model": model_name, "features": len(columns),
                "train_rows": len(train), "test_rows": len(test), "cv_auc": search.best_score_,
                "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, scores), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "best_params": json.dumps({k.removeprefix("model__"): v for k, v in search.best_params_.items()}),
            })
            predictions.append(test[["date", "ticker"]].assign(
                target=target, feature_set=set_name, model=model_name, y_true=y_test, score=scores, pred=pred))
            log.info("%s | %-14s | %-19s | test AUC %.3f  accuracy %.3f", target, set_name, model_name,
                     results[-1]["roc_auc"], results[-1]["accuracy"])

    majority = int(y_train.mean() >= 0.5)
    results.append({"target": target, "feature_set": "-", "model": "majority_class", "train_rows": len(train),
                    "test_rows": len(test), "accuracy": float(np.mean(y_test == majority)), "roc_auc": 0.5})

    tests = []
    preds = pd.concat(predictions, ignore_index=True)
    for model_name in build_models(settings):
        base = preds[(preds["model"] == model_name) & (preds["feature_set"] == "baseline")]
        for set_name in [s for s in set_names if s != "baseline"]:
            enhanced = preds[(preds["model"] == model_name) & (preds["feature_set"] == set_name)]
            auc_base, auc_enh, z, p_auc = delong_test(y_test, base["score"].to_numpy(), enhanced["score"].to_numpy())
            stat, p_acc, only_base, only_enh = mcnemar_test(y_test, base["pred"].to_numpy(), enhanced["pred"].to_numpy())
            tests.append({"target": target, "model": model_name, "feature_set": set_name,
                          "auc_baseline": auc_base, "auc_enhanced": auc_enh, "auc_diff": auc_enh - auc_base,
                          "delong_z": z, "delong_p": p_auc, "mcnemar_stat": stat, "mcnemar_p": p_acc,
                          "only_baseline_correct": only_base, "only_enhanced_correct": only_enh})

    # Holm correction across every comparison for this target controls the chance of a false "significant"
    tests_frame = pd.DataFrame(tests)
    for column in ("delong_p", "mcnemar_p"):
        tests_frame[f"{column}_holm"] = multipletests(tests_frame[column], method="holm")[1]
    return results, tests_frame.to_dict("records"), predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and sentiment-enhanced models")
    parser.add_argument("--sensitivity", action="store_true",
                        help="also run the extra horizons and market-relative targets (baseline vs all_sentiment)")
    parser.add_argument("--news-days-only", action="store_true",
                        help="exploratory: train and test only on company-days that have at least one headline")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    cfg = settings["modelling"]
    data = pd.read_parquet(DATASET_PATH)
    if args.news_days_only:
        data = data[data["has_news"] == 1]

    runs = [(cfg["main_target"], cfg["feature_sets"])]
    if args.sensitivity:
        runs += [(target, ["baseline", "all_sentiment"]) for target in cfg["sensitivity_targets"]]

    results, tests, predictions = [], [], []
    for target, set_names in runs:
        r, t, p = run_target(data, target, set_names, settings)
        results += r
        tests += t
        predictions += p

    REPORTS.mkdir(parents=True, exist_ok=True)
    suffix = ("_sensitivity" if args.sensitivity else "") + ("_newsdays" if args.news_days_only else "")
    results_frame, tests_frame = pd.DataFrame(results), pd.DataFrame(tests)
    results_frame.to_csv(REPORTS / f"model_results{suffix}.csv", index=False)
    tests_frame.to_csv(REPORTS / f"significance_tests{suffix}.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        PREDICTIONS_PATH.with_stem(f"test_predictions{suffix}"), index=False)
    (REPORTS / f"modelling_run{suffix}.json").write_text(json.dumps({
        "run_at": utc_now(), "train_end": cfg["train_end"], "cv_splits": cfg["cv_splits"],
        "random_state": cfg["random_state"], "targets": [t for t, _ in runs],
        "news_days_only": args.news_days_only,
    }, indent=2), encoding="utf-8")

    main_results = results_frame[results_frame["target"] == cfg["main_target"]]
    print(f"\nTest ROC-AUC, {cfg['main_target']} ({int(main_results['test_rows'].iloc[0])} test rows):")
    print(main_results[main_results["model"] != "majority_class"]
          .pivot(index="feature_set", columns="model", values="roc_auc").round(3).to_string())
    majority = main_results[main_results["model"] == "majority_class"].iloc[0]
    print(f"\nMajority-class accuracy (always predict the training majority): {majority['accuracy']:.3f}")
    print("\nTest accuracy:")
    print(main_results[main_results["model"] != "majority_class"]
          .pivot(index="feature_set", columns="model", values="accuracy").round(3).to_string())

    print("\nSentiment-enhanced vs baseline (Holm-adjusted p-values):")
    view = tests_frame[["target", "model", "feature_set", "auc_diff", "delong_p_holm", "mcnemar_p_holm"]]
    print(view.round({"auc_diff": 3, "delong_p_holm": 3, "mcnemar_p_holm": 3}).to_string(index=False))


if __name__ == "__main__":
    main()