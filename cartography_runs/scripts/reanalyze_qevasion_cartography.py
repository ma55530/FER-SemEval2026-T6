#!/usr/bin/env python3
"""Reanalyse QEvasion cartography outputs with a variability-first category rule.

This script does not rerun training. It consumes completed files in
output/results and writes corrected plots, tables, and qualitative examples to
output/reanalysis.
"""

from __future__ import annotations

import ast
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from textwrap import shorten
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional local dependency
    scipy_stats = None


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "output"
RESULTS_DIR = OUTPUT_ROOT / "results"
REANALYSIS_DIR = OUTPUT_ROOT / "reanalysis"
RE_RESULTS = REANALYSIS_DIR / "results"
RE_PLOTS = REANALYSIS_DIR / "plots"
RE_EXAMPLES = REANALYSIS_DIR / "examples"

REGION_ORDER = ["easy", "ambiguous", "hard"]
REGION_COLORS = {
    "easy": "#2878b5",
    "ambiguous": "#f28e2b",
    "hard": "#c83e4d",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold-mode",
        choices=["natural", "median"],
        default="natural",
        help=(
            "natural uses one-dimensional Otsu/natural-break thresholds and does not "
            "force fixed region proportions; median reproduces the older 50/25/25 split."
        ),
    )
    parser.add_argument("--var-threshold", type=float, default=None, help="Optional explicit variability threshold.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional explicit confidence threshold inside the low-variability subset.",
    )
    return parser


def ensure_dirs() -> None:
    for path in [REANALYSIS_DIR, RE_RESULTS, RE_PLOTS, RE_EXAMPLES]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, tuple):
                return list(parsed)
            return [parsed]
        except Exception:
            continue
    return [text]


def vote_label(item: Any) -> str | None:
    if item is None:
        return None
    if isinstance(item, str):
        text = item.strip()
        return text or None
    if isinstance(item, dict):
        for key in [
            "evasion",
            "evasion_label",
            "label",
            "answer_type",
            "category",
            "accepted_evasion_label",
        ]:
            if key in item and item[key] is not None:
                text = str(item[key]).strip()
                if text:
                    return text
    text = str(item).strip()
    return text or None


def shannon_entropy(labels: list[str]) -> float:
    labels = [label for label in labels if label]
    if not labels:
        return float("nan")
    counts = np.array(list(Counter(labels).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def bool_series(series: pd.Series) -> pd.Series:
    if str(series.dtype) == "bool":
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(["true", "1", "yes", "y"])


def pct(n: float, d: float) -> float:
    if d == 0 or pd.isna(d):
        return 0.0
    return float(100.0 * n / d)


def natural_break_threshold(values: pd.Series, bins: int = 256) -> float:
    """Return a one-dimensional Otsu threshold.

    This chooses the split that maximizes between-group separation. It is still
    data-informed, but unlike quantiles it does not prescribe how many examples
    must fall into either side.
    """
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan")
    if float(arr.min()) == float(arr.max()):
        return float(arr.min())
    counts, edges = np.histogram(arr, bins=bins, range=(float(arr.min()), float(arr.max())))
    centers = (edges[:-1] + edges[1:]) / 2
    total = counts.sum()
    if total == 0:
        return float(np.nanmedian(arr))
    weight1 = np.cumsum(counts)
    weight2 = total - weight1
    weighted_sum = np.cumsum(counts * centers)
    total_mean = float((counts * centers).sum() / total)
    mean1 = weighted_sum / np.maximum(weight1, 1)
    mean2 = (total_mean * total - weighted_sum) / np.maximum(weight2, 1)
    between = weight1 * weight2 * (mean1 - mean2) ** 2
    between[(weight1 == 0) | (weight2 == 0)] = 0
    return float(centers[int(np.argmax(between))])


def compact_text(value: Any, width: int = 260) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return shorten(text, width=width, placeholder="...")


def load_stability() -> tuple[pd.DataFrame, pd.DataFrame]:
    stability = read_csv(RESULTS_DIR / "qevasion_constrained_stability.csv")
    metadata_path = RESULTS_DIR / "qevasion_train_metadata.csv"
    if metadata_path.exists():
        metadata = read_csv(metadata_path)
        add_cols = [
            "idx",
            "annotator_votes",
            "accepted_evasion_labels",
            "annotator1",
            "annotator2",
            "annotator3",
        ]
        add_cols = [col for col in add_cols if col in metadata.columns]
        missing_cols = [col for col in add_cols if col == "idx" or col not in stability.columns]
        if len(missing_cols) > 1:
            stability = stability.merge(metadata[missing_cols], on="idx", how="left")
    else:
        metadata = pd.DataFrame()
    return stability, metadata


def threshold_pair(
    df: pd.DataFrame,
    confidence_col: str,
    variability_col: str,
    mode: str,
    explicit_var_threshold: float | None = None,
    explicit_confidence_threshold: float | None = None,
) -> tuple[float, float]:
    if explicit_var_threshold is not None:
        var_threshold = float(explicit_var_threshold)
    elif mode == "median":
        var_threshold = float(df[variability_col].median())
    else:
        var_threshold = natural_break_threshold(df[variability_col])

    low_variability = df[variability_col] <= var_threshold
    if explicit_confidence_threshold is not None:
        confidence_threshold = float(explicit_confidence_threshold)
    elif mode == "median":
        confidence_threshold = float(df.loc[low_variability, confidence_col].median())
    else:
        confidence_threshold = natural_break_threshold(df.loc[low_variability, confidence_col])
    return var_threshold, confidence_threshold


def add_corrected_regions(
    df: pd.DataFrame,
    threshold_mode: str = "natural",
    explicit_var_threshold: float | None = None,
    explicit_confidence_threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    var_threshold, confidence_threshold = threshold_pair(
        out,
        "conf_mean",
        "var_mean",
        threshold_mode,
        explicit_var_threshold,
        explicit_confidence_threshold,
    )
    low_variability = out["var_mean"] <= var_threshold

    out["variability_percentile"] = out["var_mean"].rank(method="average", pct=True)
    out["low_variability_for_split"] = low_variability
    out["corrected_region"] = np.where(
        out["var_mean"] > var_threshold,
        "ambiguous",
        np.where(out["conf_mean"] >= confidence_threshold, "easy", "hard"),
    )
    out["category"] = out["corrected_region"]
    out["corrected_region_rule"] = (
        f"ambiguous if var_mean > {threshold_mode} threshold ({var_threshold:.6f}); "
        f"otherwise easy if conf_mean >= low-var {threshold_mode} threshold "
        f"({confidence_threshold:.6f}), else hard"
    )

    seed_thresholds: dict[str, dict[str, float]] = {}
    seed_region_cols: list[str] = []
    for col in out.columns:
        match = re.fullmatch(r"conf_seed(\d+)", col)
        if not match:
            continue
        seed = match.group(1)
        var_col = f"var_seed{seed}"
        if var_col not in out.columns:
            continue
        var_thr, conf_thr = threshold_pair(out, col, var_col, threshold_mode)
        low_seed = out[var_col] <= var_thr
        seed_col = f"corrected_region_seed{seed}"
        out[seed_col] = np.where(
            out[var_col] > var_thr,
            "ambiguous",
            np.where(out[col] >= conf_thr, "easy", "hard"),
        )
        seed_thresholds[seed] = {
            "var_threshold": var_thr,
            "confidence_threshold_low_variability": conf_thr,
        }
        seed_region_cols.append(seed_col)

    if seed_region_cols:
        patterns: list[str] = []
        majority: list[str] = []
        agreement_counts: list[int] = []
        for _, row in out[seed_region_cols].iterrows():
            values = [str(row[col]) for col in seed_region_cols]
            counts = Counter(values)
            top_region, top_count = counts.most_common(1)[0]
            patterns.append(",".join(values))
            majority.append(top_region)
            agreement_counts.append(top_count)
        out["corrected_region_pattern"] = patterns
        out["corrected_majority_seed_region"] = majority
        out["corrected_region_agreement_count"] = agreement_counts
        out["corrected_region_agreement"] = np.array(agreement_counts, dtype=float) / len(seed_region_cols)
        out["corrected_is_stable"] = out["corrected_region_agreement_count"] == len(seed_region_cols)
        out["corrected_is_chronically_unstable"] = out["corrected_region_agreement"] <= 0.4

    config = {
        "region_rule": f"variability-first {threshold_mode} split",
        "threshold_mode": threshold_mode,
        "var_threshold": var_threshold,
        "confidence_threshold_low_variability": confidence_threshold,
        "seed_thresholds": seed_thresholds,
        "n_examples": int(len(out)),
    }
    return out, config


def extract_annotator_entropy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    if "annotator_votes" not in out.columns:
        out["annotator_votes"] = "[]"
    if "accepted_evasion_labels" not in out.columns:
        out["accepted_evasion_labels"] = "[]"

    vote_lengths: list[int] = []
    entropies: list[float] = []
    accepted_counts: list[int] = []
    accepted_entropies: list[float] = []

    for _, row in out.iterrows():
        votes = parse_list(row.get("annotator_votes", "[]"))
        labels = [vote_label(item) for item in votes]
        labels = [label for label in labels if label]
        vote_lengths.append(len(labels))
        entropies.append(shannon_entropy(labels))

        accepted = parse_list(row.get("accepted_evasion_labels", "[]"))
        accepted_labels = [vote_label(item) for item in accepted]
        accepted_labels = [label for label in accepted_labels if label]
        accepted_counts.append(len(accepted_labels))
        accepted_entropies.append(shannon_entropy(accepted_labels))

    out["annotator_vote_count"] = vote_lengths
    out["annotator_entropy"] = entropies
    out["accepted_evasion_label_count"] = accepted_counts
    out["accepted_evasion_entropy"] = accepted_entropies

    usable_votes = int(np.sum(np.array(vote_lengths) > 0))
    usable_entropy = int(pd.Series(entropies).notna().sum())
    accepted_multi = int(np.sum(np.array(accepted_counts) > 1))
    info = {
        "annotator_vote_rows": usable_votes,
        "annotator_entropy_rows": usable_entropy,
        "accepted_multi_label_rows": accepted_multi,
        "available": usable_entropy > 0 or accepted_multi > 0,
        "note": (
            "No annotator entropy is available in the exported files."
            if usable_entropy == 0 and accepted_multi == 0
            else "Annotator entropy was computed from available vote or accepted-label fields."
        ),
    }
    return out, info


def add_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for path in sorted(RESULTS_DIR.glob("qevasion_constrained_train_predictions_seed*.csv")):
        match = re.search(r"seed(\d+)", path.name)
        if not match:
            continue
        seed = match.group(1)
        pred = read_csv(path)
        keep = [
            "idx",
            "pred_pair",
            "pred_clarity",
            "pred_evasion",
            "prob_true_pair",
            "prob_true_pair_raw",
            "pred_prob_clarity",
            "pred_prob_evasion",
            "is_correct_pair",
            "failure_type",
        ]
        keep = [col for col in keep if col in pred.columns]
        pred = pred[keep].copy()
        if {"pred_prob_clarity", "pred_prob_evasion"}.issubset(pred.columns):
            pred[f"pred_joint_conf_seed{seed}"] = (
                pd.to_numeric(pred["pred_prob_clarity"], errors="coerce")
                * pd.to_numeric(pred["pred_prob_evasion"], errors="coerce")
            )
        rename = {col: f"{col}_seed{seed}" for col in pred.columns if col != "idx"}
        pred = pred.rename(columns=rename)
        out = out.merge(pred, on="idx", how="left")
    return out


def forgetfulness_from_series(value: Any) -> float:
    values = parse_list(value)
    try:
        ints = [int(bool(int(v))) for v in values]
    except Exception:
        return float("nan")
    if not ints:
        return float("nan")
    return float(sum(1 for i in range(1, len(ints)) if ints[i - 1] == 1 and ints[i] == 0))


def dimension_seed_cartography(path: Path, dimension: str) -> pd.DataFrame:
    seed_match = re.search(r"seed(\d+)", path.name)
    seed = seed_match.group(1) if seed_match else "unknown"
    df = read_csv(path)
    if dimension == "evasion":
        conf_col = "evasion_prob_true_mean"
        var_col = "evasion_prob_true_std"
        correctness_col = "correct_evasion_mean"
        series_col = "correct_evasion_series"
    elif dimension == "clarity":
        conf_col = "clarity_prob_true_mean"
        var_col = "clarity_prob_true_std"
        correctness_col = "correct_clarity_mean"
        series_col = "correct_clarity_series"
    else:
        raise ValueError(f"Unknown dimension: {dimension}")
    required = ["idx", conf_col, var_col, correctness_col, series_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing dimension cartography columns: {missing}")
    out = df[["idx", conf_col, var_col, correctness_col, series_col]].copy()
    out = out.rename(
        columns={
            conf_col: f"{dimension}_conf_seed{seed}",
            var_col: f"{dimension}_var_seed{seed}",
            correctness_col: f"{dimension}_correctness_seed{seed}",
            series_col: f"{dimension}_correct_series_seed{seed}",
        }
    )
    out[f"{dimension}_forgetfulness_seed{seed}"] = out[f"{dimension}_correct_series_seed{seed}"].apply(
        forgetfulness_from_series
    )
    out = out.drop(columns=[f"{dimension}_correct_series_seed{seed}"])
    return out


def build_dimension_cartography(
    metadata: pd.DataFrame,
    dimension: str,
    threshold_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seed_files = sorted(RESULTS_DIR.glob("qevasion_constrained_carto_seed*.csv"))
    if not seed_files:
        raise FileNotFoundError("No constrained cartography seed files found.")

    out = metadata.copy()
    for path in seed_files:
        out = out.merge(dimension_seed_cartography(path, dimension), on="idx", how="left")

    conf_cols = sorted([c for c in out.columns if re.fullmatch(fr"{dimension}_conf_seed\d+", c)])
    var_cols = sorted([c for c in out.columns if re.fullmatch(fr"{dimension}_var_seed\d+", c)])
    corr_cols = sorted([c for c in out.columns if re.fullmatch(fr"{dimension}_correctness_seed\d+", c)])
    forget_cols = sorted([c for c in out.columns if re.fullmatch(fr"{dimension}_forgetfulness_seed\d+", c)])
    out[f"{dimension}_conf_mean"] = out[conf_cols].mean(axis=1)
    out[f"{dimension}_conf_std"] = out[conf_cols].std(axis=1)
    out[f"{dimension}_var_mean"] = out[var_cols].mean(axis=1)
    out[f"{dimension}_var_std"] = out[var_cols].std(axis=1)
    out[f"{dimension}_correctness_mean"] = out[corr_cols].mean(axis=1)
    out[f"{dimension}_correctness_std"] = out[corr_cols].std(axis=1)
    out[f"{dimension}_forgetfulness_mean"] = out[forget_cols].mean(axis=1)
    out[f"{dimension}_forgetfulness_std"] = out[forget_cols].std(axis=1)

    temp = pd.DataFrame(
        {
            "conf_mean": out[f"{dimension}_conf_mean"],
            "var_mean": out[f"{dimension}_var_mean"],
        }
    )
    var_threshold, confidence_threshold = threshold_pair(temp, "conf_mean", "var_mean", threshold_mode)
    out[f"{dimension}_region"] = np.where(
        out[f"{dimension}_var_mean"] > var_threshold,
        "ambiguous",
        np.where(out[f"{dimension}_conf_mean"] >= confidence_threshold, "easy", "hard"),
    )
    out[f"{dimension}_region_rule"] = (
        f"ambiguous if {dimension}_var_mean > {threshold_mode} threshold ({var_threshold:.6f}); "
        f"otherwise easy if {dimension}_conf_mean >= low-var {threshold_mode} threshold "
        f"({confidence_threshold:.6f}), else hard"
    )

    seed_thresholds: dict[str, dict[str, float]] = {}
    seed_region_cols: list[str] = []
    for conf_col in conf_cols:
        seed = re.search(r"seed(\d+)", conf_col).group(1)  # type: ignore[union-attr]
        var_col = f"{dimension}_var_seed{seed}"
        temp_seed = pd.DataFrame({"conf_seed": out[conf_col], "var_seed": out[var_col]})
        var_thr, conf_thr = threshold_pair(temp_seed, "conf_seed", "var_seed", threshold_mode)
        region_col = f"{dimension}_region_seed{seed}"
        out[region_col] = np.where(
            out[var_col] > var_thr,
            "ambiguous",
            np.where(out[conf_col] >= conf_thr, "easy", "hard"),
        )
        seed_region_cols.append(region_col)
        seed_thresholds[seed] = {
            "var_threshold": var_thr,
            "confidence_threshold_low_variability": conf_thr,
        }

    if seed_region_cols:
        patterns: list[str] = []
        majority: list[str] = []
        agreements: list[int] = []
        for _, row in out[seed_region_cols].iterrows():
            values = [str(row[col]) for col in seed_region_cols]
            counts = Counter(values)
            top, n = counts.most_common(1)[0]
            patterns.append(",".join(values))
            majority.append(top)
            agreements.append(n)
        out[f"{dimension}_region_pattern"] = patterns
        out[f"{dimension}_majority_seed_region"] = majority
        out[f"{dimension}_region_agreement_count"] = agreements
        out[f"{dimension}_region_agreement"] = np.array(agreements, dtype=float) / len(seed_region_cols)
        out[f"{dimension}_is_stable"] = out[f"{dimension}_region_agreement_count"] == len(seed_region_cols)

    config = {
        "dimension": dimension,
        "region_rule": f"{dimension}-only variability-first {threshold_mode} split",
        "threshold_mode": threshold_mode,
        "var_threshold": var_threshold,
        "confidence_threshold_low_variability": confidence_threshold,
        "seed_thresholds": seed_thresholds,
        "n_examples": int(len(out)),
        "seed_files": [path.name for path in seed_files],
    }
    return out, config


def long_crosstab(df: pd.DataFrame, label_col: str, region_col: str = "corrected_region") -> pd.DataFrame:
    table = pd.crosstab(df[region_col], df[label_col]).reindex(REGION_ORDER).fillna(0).astype(int)
    total = table.values.sum()
    label_totals = table.sum(axis=0)
    rows: list[dict[str, Any]] = []
    for region in table.index:
        region_total = int(table.loc[region].sum())
        for label in table.columns:
            count = int(table.loc[region, label])
            rows.append(
                {
                    "region": region,
                    "label": label,
                    "count": count,
                    "region_pct": pct(count, region_total),
                    "label_pct": pct(count, float(label_totals[label])),
                    "overall_pct": pct(count, float(total)),
                    "region_total": region_total,
                    "label_total": int(label_totals[label]),
                }
            )
    return pd.DataFrame(rows)


def enrichment_table(df: pd.DataFrame, label_col: str, region_col: str = "corrected_region") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    region_rates = df[region_col].value_counts(normalize=True).to_dict()
    for label, group in df.groupby(label_col, dropna=False):
        row: dict[str, Any] = {"label": label, "n": int(len(group))}
        for region in REGION_ORDER:
            rate = float((group[region_col] == region).mean())
            base = float(region_rates.get(region, 0.0))
            row[f"{region}_rate"] = rate
            row[f"{region}_enrichment"] = rate / base if base > 0 else float("nan")
            row[f"{region}_count"] = int((group[region_col] == region).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["hard_enrichment", "n"], ascending=[False, False])


def aggregate_by_region(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "conf_mean",
        "var_mean",
        "correctness_mean",
        "forgetfulness_mean",
        "conf_std",
        "var_std",
        "question_word_count",
        "answer_word_count",
        "subquestion_word_count",
        "annotator_entropy",
        "accepted_evasion_label_count",
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    rows: list[dict[str, Any]] = []
    for region, group in df.groupby("corrected_region"):
        row: dict[str, Any] = {"region": region, "n": int(len(group)), "pct": pct(len(group), len(df))}
        for col in numeric_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            row[f"{col}_median"] = float(vals.median()) if vals.notna().any() else float("nan")
            row[f"{col}_std"] = float(vals.std()) if vals.notna().sum() > 1 else float("nan")
        for col in ["multiple_questions", "affirmative_questions", "inaudible", "is_stable", "is_chronically_unstable", "corrected_is_stable", "corrected_is_chronically_unstable"]:
            if col in group.columns:
                row[f"{col}_rate"] = float(bool_series(group[col]).mean())
        if "corrected_region_agreement" in group.columns:
            row["corrected_region_agreement_mean"] = float(pd.to_numeric(group["corrected_region_agreement"], errors="coerce").mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out["region"] = pd.Categorical(out["region"], categories=REGION_ORDER, ordered=True)
    return out.sort_values("region").reset_index(drop=True)


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "conf_mean",
        "var_mean",
        "correctness_mean",
        "forgetfulness_mean",
        "question_word_count",
        "answer_word_count",
        "subquestion_word_count",
        "annotator_entropy",
        "accepted_evasion_label_count",
    ]
    candidates = [col for col in candidates if col in df.columns]
    rows: list[dict[str, Any]] = []
    for a in candidates:
        for b in candidates:
            if a >= b:
                continue
            left = pd.to_numeric(df[a], errors="coerce")
            right = pd.to_numeric(df[b], errors="coerce")
            valid = left.notna() & right.notna()
            if valid.sum() < 3:
                continue
            pearson = float(left[valid].corr(right[valid], method="pearson"))
            spearman = float(left[valid].corr(right[valid], method="spearman"))
            rows.append({"metric_a": a, "metric_b": b, "n": int(valid.sum()), "pearson": pearson, "spearman": spearman})
    return pd.DataFrame(rows)


def load_prediction_files() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pattern = re.compile(r"qevasion_(constrained|independent|joint)_(train|test)_predictions_seed(\d+)\.csv")
    for path in sorted(RESULTS_DIR.glob("qevasion_*_predictions_seed*.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        model, split, seed = match.groups()
        df = read_csv(path)
        df["model"] = model
        df["split_name"] = split
        df["seed"] = int(seed)
        if {"pred_prob_clarity", "pred_prob_evasion"}.issubset(df.columns):
            df["pred_joint_conf"] = (
                pd.to_numeric(df["pred_prob_clarity"], errors="coerce")
                * pd.to_numeric(df["pred_prob_evasion"], errors="coerce")
            )
        elif "pred_prob_pair" in df.columns:
            df["pred_joint_conf"] = pd.to_numeric(df["pred_prob_pair"], errors="coerce")
        else:
            df["pred_joint_conf"] = np.nan
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def prediction_summaries(predictions: pd.DataFrame, stability: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if predictions.empty:
        return {}

    out = predictions.copy()
    if "idx" in out.columns:
        region_map = stability[["idx", "corrected_region"]].copy()
        out = out.merge(region_map, on="idx", how="left")
        out.loc[out["split_name"] == "test", "corrected_region"] = np.nan

    for col in ["illegal_pair", "is_correct_pair", "is_correct_clarity", "is_correct_evasion", "is_evasion_compatible"]:
        if col in out.columns:
            out[col] = bool_series(out[col])

    grouped = []
    for keys, group in out.groupby(["model", "split_name", "seed"], dropna=False):
        model, split, seed = keys
        row = {"model": model, "split": split, "seed": int(seed), "n": int(len(group))}
        for col in ["is_correct_clarity", "is_correct_evasion", "is_correct_pair", "is_evasion_compatible"]:
            if col in group.columns:
                row[f"{col}_rate"] = float(group[col].mean())
        if "illegal_pair" in group.columns:
            row["illegal_pair_count"] = int(group["illegal_pair"].sum())
            row["illegal_pair_rate"] = float(group["illegal_pair"].mean())
        grouped.append(row)
    summary = pd.DataFrame(grouped).sort_values(["model", "split", "seed"])

    label_rows: list[dict[str, Any]] = []
    for label_col in ["clarity_label", "evasion_label", "gold_pair", "failure_type", "corrected_region"]:
        if label_col not in out.columns:
            continue
        data = out.dropna(subset=[label_col])
        for keys, group in data.groupby(["model", "split_name", "seed", label_col], dropna=False):
            model, split, seed, label = keys
            row = {
                "model": model,
                "split": split,
                "seed": int(seed),
                "dimension": label_col,
                "label": label,
                "n": int(len(group)),
            }
            if "is_correct_pair" in group.columns:
                row["pair_accuracy"] = float(group["is_correct_pair"].mean())
            if "is_correct_clarity" in group.columns:
                row["clarity_accuracy"] = float(group["is_correct_clarity"].mean())
            if "is_correct_evasion" in group.columns:
                row["evasion_accuracy"] = float(group["is_correct_evasion"].mean())
            if "illegal_pair" in group.columns:
                row["illegal_pair_rate"] = float(group["illegal_pair"].mean())
            label_rows.append(row)
    by_label = pd.DataFrame(label_rows)

    illegal = out[out.get("illegal_pair", False) == True].copy()  # noqa: E712
    if not illegal.empty:
        top_illegal = (
            illegal.groupby(["model", "split_name", "seed", "pred_clarity", "pred_evasion", "pred_pair"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "model", "split_name"], ascending=[False, True, True])
        )
    else:
        top_illegal = pd.DataFrame(columns=["model", "split_name", "seed", "pred_clarity", "pred_evasion", "pred_pair", "count"])

    high_conf_wrong = out[(out.get("is_correct_pair", True) == False) & out["pred_joint_conf"].notna()].copy()  # noqa: E712
    if not high_conf_wrong.empty:
        high_conf_wrong = high_conf_wrong.sort_values("pred_joint_conf", ascending=False)
        keep = [
            "model",
            "split_name",
            "seed",
            "idx",
            "corrected_region",
            "gold_pair",
            "pred_pair",
            "clarity_label",
            "evasion_label",
            "pred_clarity",
            "pred_evasion",
            "failure_type",
            "pred_joint_conf",
            "prob_true_pair",
            "interview_question",
            "interview_answer",
            "question",
        ]
        keep = [col for col in keep if col in high_conf_wrong.columns]
        high_conf_wrong = high_conf_wrong[keep].head(250)

    return {
        "all_predictions_with_corrected_region": out,
        "prediction_summary": summary,
        "prediction_by_label": by_label,
        "illegal_top_pairs": top_illegal,
        "high_confidence_wrong": high_conf_wrong,
    }


def plot_cartography(df: pd.DataFrame, config: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=160)
    for region in REGION_ORDER:
        group = df[df["corrected_region"] == region]
        ax.scatter(
            group["var_mean"],
            group["conf_mean"],
            s=28,
            alpha=0.72,
            color=REGION_COLORS[region],
            edgecolors="white",
            linewidths=0.25,
            label=f"{region} (n={len(group)})",
        )
    ax.axvline(config["var_threshold"], color="#333333", linestyle="--", linewidth=1.2, label="variability threshold")
    ax.axhline(
        config["confidence_threshold_low_variability"],
        color="#666666",
        linestyle=":",
        linewidth=1.2,
        label="confidence threshold within low-variability subset",
    )
    ax.set_title("QEvasion Cartography With Corrected Variability-First Categories", pad=14)
    ax.set_xlabel("Variability (mean over constrained seeds)")
    ax.set_ylabel("Gold-pair confidence (mean over constrained seeds)")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.text(
        0.02,
        0.98,
        "Rule: ambiguous = above natural variability break; remaining examples split by confidence",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_stacked(table: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    table = table.reindex(REGION_ORDER).fillna(0)
    table = table.loc[:, table.sum(axis=0).sort_values(ascending=False).index]
    pct_table = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(12.5, 7.4), dpi=160)
    bottom = np.zeros(len(pct_table))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(pct_table.columns), 1)))
    x = np.arange(len(pct_table.index))
    for i, col in enumerate(pct_table.columns):
        values = pct_table[col].values
        ax.bar(x, values, bottom=bottom, label=col, color=colors[i], edgecolor="white", linewidth=0.4)
        for j, value in enumerate(values):
            if value >= 7:
                ax.text(j, bottom[j] + value / 2, f"{value:.0f}%", ha="center", va="center", fontsize=7, color="white")
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={int(table.loc[r].sum())})" for r in pct_table.index])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.set_title(title, pad=14)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_illegal_pairs(summary: pd.DataFrame, top_illegal: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=160, gridspec_kw={"width_ratios": [1.05, 1.35]})
    ax = axes[0]
    if summary.empty:
        ax.text(0.5, 0.5, "No prediction files found", ha="center", va="center")
        ax.axis("off")
    else:
        plot_df = (
            summary.groupby(["model", "split"], dropna=False)
            .agg(n=("n", "sum"), illegal_pair_count=("illegal_pair_count", "sum"))
            .reset_index()
        )
        plot_df["illegal_pair_rate"] = plot_df["illegal_pair_count"] / plot_df["n"].replace(0, np.nan)
        order = ["constrained", "independent", "joint"]
        split_order = ["train", "test"]
        plot_df["model_order"] = plot_df["model"].map({name: i for i, name in enumerate(order)}).fillna(99)
        plot_df["split_order"] = plot_df["split"].map({name: i for i, name in enumerate(split_order)}).fillna(99)
        plot_df = plot_df.sort_values(["model_order", "split_order"])
        plot_df["label"] = plot_df["model"] + "\n" + plot_df["split"]
        colors = ["#c83e4d" if rate > 0 else "#2878b5" for rate in plot_df["illegal_pair_rate"]]
        ax.bar(np.arange(len(plot_df)), plot_df["illegal_pair_rate"] * 100, color=colors)
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(plot_df["label"], fontsize=8)
        ax.set_ylabel("Illegal predicted pair rate (%)")
        ax.set_title("Illegal Pair Rate By Decoder And Split")
        ax.grid(axis="y", alpha=0.25)
        for i, (_, row) in enumerate(plot_df.iterrows()):
            if row["illegal_pair_rate"] > 0:
                ax.text(
                    i,
                    row["illegal_pair_rate"] * 100 + 0.4,
                    f"{int(row['illegal_pair_count'])}/{int(row['n'])}",
                    ha="center",
                    fontsize=8,
                )

    ax2 = axes[1]
    if top_illegal.empty:
        ax2.text(0.5, 0.5, "No illegal pairs found", ha="center", va="center")
        ax2.axis("off")
    else:
        top = top_illegal.copy()
        top["pair_label"] = top["pred_clarity"].astype(str) + " | " + top["pred_evasion"].astype(str)
        top = (
            top.groupby("pair_label", dropna=False)["count"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .sort_values()
        )
        ax2.barh(np.arange(len(top)), top.values, color="#c83e4d")
        ax2.set_yticks(np.arange(len(top)))
        ax2.set_yticklabels(top.index, fontsize=8)
        ax2.set_xlabel("Illegal predictions (all independent files)")
        ax2.set_title("Most Frequent Illegal Predicted Pairs")
        ax2.grid(axis="x", alpha=0.25)
        for i, value in enumerate(top.values):
            ax2.text(value + 0.3, i, str(int(value)), va="center", fontsize=8)

    fig.suptitle("QEvasion Illegal Pair Analysis", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_entropy(df: pd.DataFrame, info: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    if not info["available"] or df["annotator_entropy"].notna().sum() == 0:
        ax.text(
            0.5,
            0.55,
            "Annotator entropy unavailable in exported QEvasion files",
            ha="center",
            va="center",
            fontsize=13,
        )
        ax.text(
            0.5,
            0.42,
            "annotator_votes are empty and accepted_evasion_labels contains one label per example",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.axis("off")
    else:
        data = [df.loc[df["corrected_region"] == region, "annotator_entropy"].dropna() for region in REGION_ORDER]
        ax.boxplot(data, labels=REGION_ORDER, patch_artist=True)
        ax.set_ylabel("Annotator entropy")
        ax.set_title("Annotator Entropy By Corrected Category")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_metadata(df: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("question_word_count", "Question length"),
        ("answer_word_count", "Answer length"),
        ("subquestion_word_count", "Subquestion length"),
        ("multiple_questions", "Multiple questions rate"),
        ("affirmative_questions", "Affirmative question rate"),
        ("inaudible", "Inaudible rate"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=160)
    for ax, (col, title) in zip(axes.ravel(), metrics):
        if col not in df.columns:
            ax.axis("off")
            continue
        vals = []
        for region in REGION_ORDER:
            group = df[df["corrected_region"] == region]
            if group.empty:
                vals.append(float("nan"))
            elif col in ["multiple_questions", "affirmative_questions", "inaudible"]:
                vals.append(float(bool_series(group[col]).mean()) * 100)
            else:
                vals.append(float(pd.to_numeric(group[col], errors="coerce").median()))
        ax.bar(REGION_ORDER, vals, color=[REGION_COLORS[r] for r in REGION_ORDER])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if col in ["multiple_questions", "affirmative_questions", "inaudible"]:
            ax.set_ylabel("%")
        else:
            ax.set_ylabel("Median words")
    fig.suptitle("Metadata Signals By Corrected Category", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_stability(df: pd.DataFrame, path: Path) -> None:
    if "corrected_majority_seed_region" not in df.columns:
        return
    table = pd.crosstab(df["corrected_region"], df["corrected_majority_seed_region"]).reindex(index=REGION_ORDER, columns=REGION_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, 6), dpi=160)
    im = ax.imshow(table.values, cmap="Blues")
    ax.set_xticks(np.arange(len(REGION_ORDER)))
    ax.set_yticks(np.arange(len(REGION_ORDER)))
    ax.set_xticklabels(REGION_ORDER)
    ax.set_yticklabels(REGION_ORDER)
    ax.set_xlabel("Corrected majority region across per-seed splits")
    ax.set_ylabel("Aggregate corrected region")
    ax.set_title("Corrected Region Agreement Check")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, int(table.values[i, j]), ha="center", va="center", color="#111111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def region_difference_table(df: pd.DataFrame, label_col: str, dimension: str) -> pd.DataFrame:
    base_rates = df["corrected_region"].value_counts(normalize=True).reindex(REGION_ORDER).fillna(0.0)
    rows: list[dict[str, Any]] = []
    for label, group in df.groupby(label_col, dropna=False):
        row: dict[str, Any] = {
            "dimension": dimension,
            "label": label,
            "n": int(len(group)),
            "label_share": float(len(group) / max(1, len(df))),
            "conf_mean": float(pd.to_numeric(group["conf_mean"], errors="coerce").mean()),
            "var_mean": float(pd.to_numeric(group["var_mean"], errors="coerce").mean()),
            "correctness_mean": float(pd.to_numeric(group["correctness_mean"], errors="coerce").mean()),
            "forgetfulness_mean": float(pd.to_numeric(group["forgetfulness_mean"], errors="coerce").mean()),
        }
        diffs = []
        for region in REGION_ORDER:
            rate = float((group["corrected_region"] == region).mean())
            diff = rate - float(base_rates[region])
            row[f"{region}_rate"] = rate
            row[f"{region}_diff_pp"] = diff * 100.0
            row[f"{region}_enrichment"] = rate / float(base_rates[region]) if base_rates[region] > 0 else float("nan")
            diffs.append(abs(diff))
        row["max_abs_diff_pp"] = max(diffs) * 100.0 if diffs else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["max_abs_diff_pp", "n"], ascending=[False, False])


def association_metrics(df: pd.DataFrame, label_col: str, dimension: str) -> dict[str, Any]:
    table = pd.crosstab(df[label_col], df["corrected_region"]).reindex(columns=REGION_ORDER).fillna(0.0)
    obs = table.to_numpy(dtype=float)
    n = float(obs.sum())
    if n <= 0:
        return {"dimension": dimension, "n": 0}
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.nansum(np.where(expected > 0, (obs - expected) ** 2 / expected, 0.0))
    denom = n * max(1.0, min(obs.shape[0] - 1, obs.shape[1] - 1))
    cramer_v = float(math.sqrt(chi / denom)) if denom > 0 else float("nan")

    p = obs / n
    pi = p.sum(axis=1, keepdims=True)
    pj = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((p > 0) & (pi @ pj > 0), p / (pi @ pj), 1.0)
        mi = float(np.nansum(np.where(p > 0, p * np.log2(ratio), 0.0)))
        hx = float(-np.nansum(np.where(pi > 0, pi * np.log2(pi), 0.0)))
        hy = float(-np.nansum(np.where(pj > 0, pj * np.log2(pj), 0.0)))
    nmi = mi / math.sqrt(hx * hy) if hx > 0 and hy > 0 else float("nan")
    diff = region_difference_table(df, label_col, dimension)
    return {
        "dimension": dimension,
        "n": int(n),
        "n_labels": int(table.shape[0]),
        "cramers_v": cramer_v,
        "mutual_information_bits": mi,
        "normalized_mutual_information": float(nmi),
        "max_abs_diff_pp": float(diff["max_abs_diff_pp"].max()) if not diff.empty else 0.0,
        "mean_label_max_abs_diff_pp": float(diff["max_abs_diff_pp"].mean()) if not diff.empty else 0.0,
        "weighted_label_max_abs_diff_pp": float((diff["max_abs_diff_pp"] * diff["label_share"]).sum()) if not diff.empty else 0.0,
    }


def plot_label_cartography(df: pd.DataFrame, label_col: str, title: str, path: Path) -> None:
    labels = list(df[label_col].dropna().astype(str).value_counts().index)
    cmap = plt.cm.tab20 if len(labels) > 3 else plt.cm.Set2
    colors = cmap(np.linspace(0, 1, max(1, len(labels))))
    fig, ax = plt.subplots(figsize=(11, 7.4), dpi=160)
    for color, label in zip(colors, labels):
        group = df[df[label_col].astype(str) == label]
        ax.scatter(
            group["var_mean"],
            group["conf_mean"],
            s=24,
            alpha=0.68,
            color=color,
            edgecolors="white",
            linewidths=0.2,
            label=f"{label} (n={len(group)})",
        )
    ax.set_xlabel("Variability (mean over constrained seeds)")
    ax.set_ylabel("Gold-pair confidence (mean over constrained seeds)")
    ax.set_title(title, pad=14)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_region_difference(diff: pd.DataFrame, title: str, path: Path, top_n: int | None = None) -> None:
    plot_df = diff.copy()
    if top_n:
        plot_df = plot_df.head(top_n)
    plot_df = plot_df.sort_values("max_abs_diff_pp", ascending=True)
    fig, ax = plt.subplots(figsize=(11.5, max(5.5, 0.42 * len(plot_df) + 2)), dpi=160)
    y = np.arange(len(plot_df))
    offsets = {"easy": -0.24, "ambiguous": 0.0, "hard": 0.24}
    for region in REGION_ORDER:
        ax.barh(
            y + offsets[region],
            plot_df[f"{region}_diff_pp"],
            height=0.22,
            color=REGION_COLORS[region],
            label=region,
        )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{label} (n={n})" for label, n in zip(plot_df["label"], plot_df["n"])], fontsize=8)
    ax.set_xlabel("Difference from overall category rate (percentage points)")
    ax.set_title(title, pad=14)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_label(pred_by_label: pd.DataFrame, dimension: str, path: Path) -> None:
    label_col = f"{dimension}_label"
    score_col = "evasion_accuracy" if dimension == "evasion" else "clarity_accuracy"
    title = f"Held-Out {dimension.title()} Accuracy By {dimension.title()} Label"
    if pred_by_label.empty or label_col not in set(pred_by_label.get("dimension", [])):
        return
    data = pred_by_label[
        (pred_by_label["model"] == "constrained")
        & (pred_by_label["split"] == "test")
        & (pred_by_label["dimension"] == label_col)
    ].copy()
    if data.empty or score_col not in data.columns:
        return
    agg = (
        data.groupby("label", dropna=False)
        .agg(n=("n", "sum"), mean_accuracy=(score_col, "mean"), std_accuracy=(score_col, "std"))
        .reset_index()
        .sort_values("mean_accuracy")
    )
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.4 * len(agg) + 2)), dpi=160)
    colors = plt.cm.RdYlGn(np.clip(agg["mean_accuracy"].to_numpy(dtype=float), 0, 1))
    ax.barh(np.arange(len(agg)), agg["mean_accuracy"] * 100, color=colors)
    ax.set_yticks(np.arange(len(agg)))
    ax.set_yticklabels([f"{label} (n={n})" for label, n in zip(agg["label"], agg["n"])], fontsize=8)
    ax.set_xlabel("Mean accuracy across constrained seeds (%)")
    ax.set_title(title, pad=14)
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(agg["mean_accuracy"] * 100):
        ax.text(value + 1, i, f"{value:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_comparison(metrics: pd.DataFrame, path: Path) -> None:
    if metrics.empty:
        return
    display_metrics = [
        ("cramers_v", "Cramer's V"),
        ("normalized_mutual_information", "NMI"),
        ("weighted_label_max_abs_diff_pp", "Weighted max diff (pp)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=160)
    for ax, (col, label) in zip(axes, display_metrics):
        vals = metrics.set_index("dimension")[col].reindex(["evasion", "clarity"]).fillna(0)
        ax.bar(vals.index, vals.values, color=["#f28e2b", "#2878b5"])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(vals.values):
            fmt = f"{value:.1f}" if "pp" in label else f"{value:.3f}"
            ax.text(i, value, fmt, ha="center", va="bottom", fontsize=9)
    fig.suptitle("How Strongly Each Label Dimension Separates Corrected Cartography Categories", y=1.05)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def dimension_region_difference_table(
    df: pd.DataFrame,
    label_col: str,
    region_col: str,
    metric_prefix: str,
    dimension: str,
) -> pd.DataFrame:
    base_rates = df[region_col].value_counts(normalize=True).reindex(REGION_ORDER).fillna(0.0)
    rows: list[dict[str, Any]] = []
    for label, group in df.groupby(label_col, dropna=False):
        row: dict[str, Any] = {
            "dimension": dimension,
            "label": label,
            "n": int(len(group)),
            "label_share": float(len(group) / max(1, len(df))),
            "confidence_mean": float(pd.to_numeric(group[f"{metric_prefix}_conf_mean"], errors="coerce").mean()),
            "variability_mean": float(pd.to_numeric(group[f"{metric_prefix}_var_mean"], errors="coerce").mean()),
            "correctness_mean": float(pd.to_numeric(group[f"{metric_prefix}_correctness_mean"], errors="coerce").mean()),
            "forgetfulness_mean": float(pd.to_numeric(group[f"{metric_prefix}_forgetfulness_mean"], errors="coerce").mean()),
        }
        diffs = []
        for region in REGION_ORDER:
            rate = float((group[region_col] == region).mean())
            diff = rate - float(base_rates[region])
            row[f"{region}_rate"] = rate
            row[f"{region}_diff_pp"] = diff * 100.0
            row[f"{region}_enrichment"] = rate / float(base_rates[region]) if base_rates[region] > 0 else float("nan")
            diffs.append(abs(diff))
        row["max_abs_diff_pp"] = max(diffs) * 100.0 if diffs else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["max_abs_diff_pp", "n"], ascending=[False, False])


def dimension_association_metrics(df: pd.DataFrame, label_col: str, region_col: str, dimension: str) -> dict[str, Any]:
    table = pd.crosstab(df[label_col], df[region_col]).reindex(columns=REGION_ORDER).fillna(0.0)
    obs = table.to_numpy(dtype=float)
    n = float(obs.sum())
    if n <= 0:
        return {"dimension": dimension, "n": 0}
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.nansum(np.where(expected > 0, (obs - expected) ** 2 / expected, 0.0))
    denom = n * max(1.0, min(obs.shape[0] - 1, obs.shape[1] - 1))
    cramer_v = float(math.sqrt(chi / denom)) if denom > 0 else float("nan")
    p = obs / n
    pi = p.sum(axis=1, keepdims=True)
    pj = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((p > 0) & (pi @ pj > 0), p / (pi @ pj), 1.0)
        mi = float(np.nansum(np.where(p > 0, p * np.log2(ratio), 0.0)))
        hx = float(-np.nansum(np.where(pi > 0, pi * np.log2(pi), 0.0)))
        hy = float(-np.nansum(np.where(pj > 0, pj * np.log2(pj), 0.0)))
    nmi = mi / math.sqrt(hx * hy) if hx > 0 and hy > 0 else float("nan")
    metric_prefix = "evasion" if dimension == "evasion" else "clarity"
    diff = dimension_region_difference_table(df, label_col, region_col, metric_prefix, dimension)
    return {
        "dimension": dimension,
        "n": int(n),
        "n_labels": int(table.shape[0]),
        "cramers_v": cramer_v,
        "mutual_information_bits": mi,
        "normalized_mutual_information": float(nmi),
        "max_abs_diff_pp": float(diff["max_abs_diff_pp"].max()) if not diff.empty else 0.0,
        "mean_label_max_abs_diff_pp": float(diff["max_abs_diff_pp"].mean()) if not diff.empty else 0.0,
        "weighted_label_max_abs_diff_pp": float((diff["max_abs_diff_pp"] * diff["label_share"]).sum()) if not diff.empty else 0.0,
    }


def plot_dimension_cartography_regions(df: pd.DataFrame, dimension: str, config: dict[str, Any], path: Path) -> None:
    x_col = f"{dimension}_var_mean"
    y_col = f"{dimension}_conf_mean"
    fig, ax = plt.subplots(figsize=(10.8, 7.4), dpi=160)
    correctness_col = f"{dimension}_correctness_mean"
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        c=df[correctness_col],
        s=28,
        alpha=0.78,
        cmap="viridis",
        vmin=0,
        vmax=1,
        edgecolors="none",
    )
    ax.set_xlabel("Variability")
    ax.set_ylabel("Confidence")
    ax.grid(alpha=0.18, linewidth=0.5)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correctness")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_label_cartography(df: pd.DataFrame, dimension: str, label_col: str, title: str, path: Path) -> None:
    x_col = f"{dimension}_var_mean"
    y_col = f"{dimension}_conf_mean"
    labels = list(df[label_col].dropna().astype(str).value_counts().index)
    cmap = plt.cm.tab20 if len(labels) > 3 else plt.cm.Set2
    colors = cmap(np.linspace(0, 1, max(1, len(labels))))
    fig, ax = plt.subplots(figsize=(11, 7.4), dpi=160)
    for color, label in zip(colors, labels):
        group = df[df[label_col].astype(str) == label]
        ax.scatter(
            group[x_col],
            group[y_col],
            s=24,
            alpha=0.68,
            color=color,
            edgecolors="white",
            linewidths=0.2,
            label=f"{label} (n={len(group)})",
        )
    ax.set_xlabel(f"{dimension.title()} variability over epochs/seeds")
    ax.set_ylabel(f"{dimension.title()} confidence in the gold label")
    ax.set_title(title, pad=14)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_distribution(df: pd.DataFrame, label_col: str, region_col: str, title: str, path: Path) -> None:
    table = pd.crosstab(df[region_col], df[label_col]).reindex(REGION_ORDER).fillna(0).astype(int)
    plot_stacked(table, title, "Within-category share (%)", path)


def plot_dimension_difference(diff: pd.DataFrame, title: str, path: Path) -> None:
    plot_region_difference(diff, title, path)


def write_dimension_qualitative_analysis(
    evasion_df: pd.DataFrame,
    clarity_df: pd.DataFrame,
    evasion_diff: pd.DataFrame,
    clarity_diff: pd.DataFrame,
    evasion_config: dict[str, Any],
    clarity_config: dict[str, Any],
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Dimension-Specific Cartography Qualitative Analysis")
    lines.append("")
    lines.append("## What Changed")
    lines.append("- This analysis no longer uses joint-pair confidence/variability as a proxy.")
    lines.append("- Evasion categories use `evasion_prob_true_*` and `correct_evasion_*` only.")
    lines.append("- Clarity categories use `clarity_prob_true_*` and `correct_clarity_*` only.")
    lines.append("")
    for name, df, cfg in [("evasion", evasion_df, evasion_config), ("clarity", clarity_df, clarity_config)]:
        counts = df[f"{name}_region"].value_counts().reindex(REGION_ORDER).fillna(0).astype(int)
        lines.append(f"## {name.title()} Cartography Split")
        lines.append(
            f"- Thresholds: variability `{cfg['var_threshold']:.6f}`, "
            f"confidence `{cfg['confidence_threshold_low_variability']:.6f}`."
        )
        lines.append("- Counts: " + ", ".join(f"{r}={int(counts[r])} ({pct(int(counts[r]), len(df)):.1f}%)" for r in REGION_ORDER))
        profile = df.groupby(f"{name}_region")[
            [f"{name}_conf_mean", f"{name}_var_mean", f"{name}_correctness_mean", f"{name}_forgetfulness_mean"]
        ].median()
        for region in REGION_ORDER:
            if region in profile.index:
                row = profile.loc[region]
                lines.append(
                    f"  - {region}: median confidence={row[f'{name}_conf_mean']:.3f}, "
                    f"variability={row[f'{name}_var_mean']:.3f}, correctness={row[f'{name}_correctness_mean']:.3f}, "
                    f"forgetfulness={row[f'{name}_forgetfulness_mean']:.3f}."
                )
        lines.append("")

    lines.append("## Evasion Label Effects Under Evasion-Only Cartography")
    for row in evasion_diff.head(9).itertuples(index=False):
        lines.append(
            f"- {row.label}: easy={row.easy_rate * 100:.1f}%, ambiguous={row.ambiguous_rate * 100:.1f}%, "
            f"hard={row.hard_rate * 100:.1f}%, max shift={row.max_abs_diff_pp:.1f} pp."
        )
    lines.append("")
    lines.append("## Clarity Label Effects Under Clarity-Only Cartography")
    for row in clarity_diff.head(3).itertuples(index=False):
        lines.append(
            f"- {row.label}: easy={row.easy_rate * 100:.1f}%, ambiguous={row.ambiguous_rate * 100:.1f}%, "
            f"hard={row.hard_rate * 100:.1f}%, max shift={row.max_abs_diff_pp:.1f} pp."
        )
    lines.append("")

    def emit_examples(title: str, df: pd.DataFrame, dimension: str, region: str, n: int = 8) -> None:
        lines.append(f"## {title}")
        subset = df[df[f"{dimension}_region"] == region].copy()
        if region == "ambiguous":
            subset = subset.sort_values([f"{dimension}_var_mean", f"{dimension}_correctness_mean"], ascending=[False, True])
        elif region == "hard":
            subset = subset.sort_values([f"{dimension}_correctness_mean", f"{dimension}_conf_mean"], ascending=[True, True])
        else:
            subset = subset.sort_values([f"{dimension}_correctness_mean", f"{dimension}_conf_mean"], ascending=[False, False])
        for i, row in enumerate(subset.head(n).itertuples(index=False), start=1):
            s = pd.Series(row._asdict())
            lines.append(
                f"{i}. idx={s.get('idx')} | gold={s.get('clarity_label')} | {s.get('evasion_label')} | "
                f"conf={float(s.get(f'{dimension}_conf_mean', float('nan'))):.3f} | "
                f"var={float(s.get(f'{dimension}_var_mean', float('nan'))):.3f} | "
                f"correctness={float(s.get(f'{dimension}_correctness_mean', float('nan'))):.3f}"
            )
            q = compact_text(s.get("question", ""), 220)
            a = compact_text(s.get("interview_answer", ""), 320)
            if q:
                lines.append(f"   - Subquestion: {q}")
            if a:
                lines.append(f"   - Answer: {a}")
        lines.append("")

    emit_examples("Evasion-Hard Examples", evasion_df, "evasion", "hard")
    emit_examples("Evasion-Ambiguous Examples", evasion_df, "evasion", "ambiguous")
    emit_examples("Clarity-Hard Examples", clarity_df, "clarity", "hard")
    emit_examples("Clarity-Ambiguous Examples", clarity_df, "clarity", "ambiguous")
    lines.append("## Training Implication")
    lines.append("- The new model sweep should optimize evasion macro F1 from constrained multitask predictions, because evasion difficulty is separable from clarity difficulty and the constrained decoder previously prevented illegal label pairs.")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_qualitative_analysis(
    df: pd.DataFrame,
    evasion_diff: pd.DataFrame,
    clarity_diff: pd.DataFrame,
    dimension_metrics: pd.DataFrame,
    examples: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Final QEvasion Qualitative Analysis")
    lines.append("")
    lines.append("## Natural-Break Cartography Split")
    lines.append(
        f"- Ambiguous threshold: `var_mean > {summary['config']['var_threshold']:.6f}` "
        f"({summary['config']['threshold_mode']})."
    )
    lines.append(
        f"- Easy/Hard threshold among non-ambiguous examples: "
        f"`conf_mean >= {summary['config']['confidence_threshold_low_variability']:.6f}`."
    )
    counts = df["corrected_region"].value_counts().reindex(REGION_ORDER).fillna(0).astype(int)
    lines.append("- Counts: " + ", ".join(f"{r}={int(counts[r])} ({pct(int(counts[r]), len(df)):.1f}%)" for r in REGION_ORDER))
    lines.append("")

    if not dimension_metrics.empty:
        lines.append("## Evasion vs Clarity Separation")
        for row in dimension_metrics.itertuples(index=False):
            lines.append(
                f"- {row.dimension}: Cramer's V={row.cramers_v:.3f}, "
                f"NMI={row.normalized_mutual_information:.3f}, "
                f"weighted max category difference={row.weighted_label_max_abs_diff_pp:.1f} pp."
            )
        lines.append("")

    lines.append("## Evasion-Specific Findings")
    for row in evasion_diff.head(9).itertuples(index=False):
        lines.append(
            f"- {row.label}: n={row.n}, hard={row.hard_rate * 100:.1f}% "
            f"({row.hard_diff_pp:+.1f} pp), ambiguous={row.ambiguous_rate * 100:.1f}% "
            f"({row.ambiguous_diff_pp:+.1f} pp), easy={row.easy_rate * 100:.1f}% "
            f"({row.easy_diff_pp:+.1f} pp)."
        )
    lines.append("")

    lines.append("## Clarity-Specific Findings")
    for row in clarity_diff.head(3).itertuples(index=False):
        lines.append(
            f"- {row.label}: n={row.n}, hard={row.hard_rate * 100:.1f}% "
            f"({row.hard_diff_pp:+.1f} pp), ambiguous={row.ambiguous_rate * 100:.1f}% "
            f"({row.ambiguous_diff_pp:+.1f} pp), easy={row.easy_rate * 100:.1f}% "
            f"({row.easy_diff_pp:+.1f} pp)."
        )
    lines.append("")

    def emit_group(title: str, frame: pd.DataFrame, n: int = 10) -> None:
        lines.append(f"## {title}")
        if frame.empty:
            lines.append("- No examples exported.")
            lines.append("")
            return
        for i, row in enumerate(frame.head(n).itertuples(index=False), start=1):
            s = pd.Series(row._asdict())
            lines.append(
                f"{i}. idx={s.get('idx')} | category={s.get('corrected_region')} | "
                f"gold={s.get('gold_pair')} | seed42_pred={s.get('pred_pair_seed42', '')} | "
                f"conf={float(s.get('conf_mean', float('nan'))):.3f} | "
                f"var={float(s.get('var_mean', float('nan'))):.3f} | "
                f"correctness={float(s.get('correctness_mean', float('nan'))):.3f}"
            )
            q = compact_text(s.get("question", ""), 240)
            iq = compact_text(s.get("interview_question", ""), 220)
            ans = compact_text(s.get("interview_answer", ""), 340)
            if q:
                lines.append(f"   - Subquestion: {q}")
            if iq and iq != q:
                lines.append(f"   - Full question: {iq}")
            if ans:
                lines.append(f"   - Answer: {ans}")
            lines.append(f"   - Qualitative read: {example_reason(s)}")
        lines.append("")

    emit_group("Hard Actual Examples", examples.get("corrected_hard_examples", pd.DataFrame()))
    emit_group("Ambiguous Actual Examples", examples.get("corrected_ambiguous_examples", pd.DataFrame()))
    emit_group("Easy Actual Examples", examples.get("corrected_easy_examples", pd.DataFrame()))

    lines.append("## Training Implications For The New Evasion Sweep")
    lines.append("- Optimize evasion directly rather than pair accuracy: constrained decoding fixed illegal outputs but evasion remains the bottleneck.")
    lines.append("- Use class-balanced/focal losses because rare labels and pragmatic labels separate strongly by corrected difficulty.")
    lines.append("- Oversample hard and ambiguous examples, especially Dodging, Implicit, Deflection, and Partial/half-answer style cases.")
    lines.append("- Preserve answer/question markers and metadata flags in the input template because long answers, multiple questions, and terse clarification turns drive many failures.")
    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_stat_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "conf_mean",
        "var_mean",
        "correctness_mean",
        "forgetfulness_mean",
        "question_word_count",
        "answer_word_count",
        "subquestion_word_count",
        "annotator_entropy",
    ]
    for metric in metrics:
        if metric not in df.columns:
            continue
        groups = [pd.to_numeric(df.loc[df["corrected_region"] == region, metric], errors="coerce").dropna() for region in REGION_ORDER]
        groups = [g for g in groups if len(g) > 1]
        row: dict[str, Any] = {"metric": metric, "n_groups": len(groups)}
        if len(groups) >= 2 and scipy_stats is not None:
            try:
                stat, p_value = scipy_stats.kruskal(*groups)
                row["kruskal_statistic"] = float(stat)
                row["p_value"] = float(p_value)
            except Exception:
                row["kruskal_statistic"] = float("nan")
                row["p_value"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def example_reason(row: pd.Series) -> str:
    region = row.get("corrected_region", "")
    evasion = str(row.get("evasion_label", ""))
    clarity = str(row.get("clarity_label", ""))
    bits: list[str] = []
    if region == "easy":
        bits.append("low variability and comparatively higher gold-pair confidence inside the low-variability subset")
    elif region == "hard":
        bits.append("low variability but consistently low gold-pair confidence")
    else:
        bits.append("high epoch/seed variability, so the model does not settle on one mapping")
    if evasion in {"Dodging", "Deflection", "Implicit", "Partial/half-answer"}:
        bits.append(f"the evasion label is pragmatic rather than surface-explicit ({evasion})")
    if evasion in {"Claims ignorance", "Declining to answer", "Clarification"}:
        bits.append(f"the label depends on a specific non-answer subtype ({evasion})")
    if clarity == "Clear Reply":
        bits.append("clarity says it is a reply, which can conflict with subtle evasion labels")
    if row.get("multiple_questions", False) is True or str(row.get("multiple_questions", "")).lower() == "true":
        bits.append("the source question contains multiple questions")
    try:
        if float(row.get("answer_word_count", 0)) >= 300:
            bits.append("the answer is long enough to dilute the relevant evidence")
    except Exception:
        pass
    try:
        if float(row.get("forgetfulness_mean", 0)) >= 0.6:
            bits.append("forgetfulness is high")
    except Exception:
        pass
    return "; ".join(bits) + "."


def select_examples(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    cols = [
        "idx",
        "corrected_region",
        "category",
        "clarity_label",
        "evasion_label",
        "gold_pair",
        "conf_mean",
        "var_mean",
        "correctness_mean",
        "forgetfulness_mean",
        "corrected_region_agreement",
        "corrected_region_pattern",
        "region_pattern",
        "multiple_questions",
        "affirmative_questions",
        "inaudible",
        "question_word_count",
        "answer_word_count",
        "subquestion_word_count",
        "pred_pair_seed42",
        "pred_clarity_seed42",
        "pred_evasion_seed42",
        "prob_true_pair_seed42",
        "failure_type_seed42",
        "interview_question",
        "interview_answer",
        "question",
    ]
    cols = [col for col in cols if col in df.columns]

    examples: dict[str, pd.DataFrame] = {}
    easy = df[df["corrected_region"] == "easy"].sort_values(
        ["conf_mean", "correctness_mean", "var_mean"], ascending=[False, False, True]
    )
    hard = df[df["corrected_region"] == "hard"].sort_values(
        ["correctness_mean", "conf_mean", "var_mean"], ascending=[True, True, True]
    )
    ambiguous = df[df["corrected_region"] == "ambiguous"].sort_values(
        ["var_mean", "forgetfulness_mean", "conf_mean"], ascending=[False, False, False]
    )
    examples["corrected_easy_examples"] = easy[cols].head(250)
    examples["corrected_hard_examples"] = hard[cols].head(250)
    examples["corrected_ambiguous_examples"] = ambiguous[cols].head(250)

    balanced_frames: list[pd.DataFrame] = []
    for (region, label), group in df.groupby(["corrected_region", "evasion_label"]):
        rank_cols = ["var_mean", "conf_mean"]
        asc = [False, False] if region == "ambiguous" else [True, region == "hard"]
        subset = group.sort_values(rank_cols, ascending=asc).head(20)
        balanced_frames.append(subset[cols])
    examples["balanced_region_evasion_examples"] = pd.concat(balanced_frames, ignore_index=True) if balanced_frames else pd.DataFrame(columns=cols)

    if "corrected_region_agreement" in df.columns:
        unstable = df.sort_values(["corrected_region_agreement", "var_mean"], ascending=[True, False])
        examples["chronically_unstable_corrected_examples"] = unstable[cols].head(250)

    var_threshold = float(config["var_threshold"])
    conf_threshold = float(config["confidence_threshold_low_variability"])
    borderline = df.assign(
        var_distance=(df["var_mean"] - var_threshold).abs(),
    )
    low = borderline[borderline["var_mean"] <= var_threshold].copy()
    if not low.empty:
        low["conf_distance"] = (low["conf_mean"] - conf_threshold).abs()
        border = pd.concat(
            [
                borderline.sort_values("var_distance").head(100),
                low.sort_values("conf_distance").head(100),
            ],
            ignore_index=True,
        ).drop_duplicates("idx")
        examples["borderline_corrected_examples"] = border[cols].head(200)
    return examples


def write_qualitative_markdown(
    df: pd.DataFrame,
    examples: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    entropy_info: dict[str, Any],
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# QEvasion Corrected Cartography Qualitative Deep Dive")
    lines.append("")
    lines.append("## Corrected Category Rule")
    lines.append(
        f"- Ambiguous: examples above the {summary['config']['threshold_mode']} variability threshold "
        f"({summary['config']['var_threshold']:.6f})."
    )
    lines.append(
        f"- Easy/Hard: the remaining low-variability examples are split by the "
        f"{summary['config']['threshold_mode']} gold-pair confidence threshold "
        f"({summary['config']['confidence_threshold_low_variability']:.6f})."
    )
    lines.append("")

    counts = pd.Series(summary["region_counts"])
    total = int(counts.sum())
    lines.append("## Headline Counts")
    for region in REGION_ORDER:
        count = int(counts.get(region, 0))
        lines.append(f"- {region}: {count} / {total} ({pct(count, total):.1f}%)")
    lines.append("")

    region_stats = df.groupby("corrected_region")[["conf_mean", "var_mean", "correctness_mean", "forgetfulness_mean", "answer_word_count"]].median()
    lines.append("## Category Profiles")
    for region in REGION_ORDER:
        if region not in region_stats.index:
            continue
        row = region_stats.loc[region]
        lines.append(
            f"- {region}: median confidence {row['conf_mean']:.3f}, variability {row['var_mean']:.3f}, "
            f"correctness {row['correctness_mean']:.3f}, forgetfulness {row['forgetfulness_mean']:.3f}, "
            f"answer length {row['answer_word_count']:.0f} words."
        )
    lines.append("")

    lines.append("## Label Patterns")
    ev_enrich = enrichment_table(df, "evasion_label")
    hard_top = ev_enrich.sort_values(["hard_enrichment", "n"], ascending=[False, False]).head(5)
    amb_top = ev_enrich.sort_values(["ambiguous_enrichment", "n"], ascending=[False, False]).head(5)
    lines.append("- Hard-enriched evasion labels: " + "; ".join(f"{r.label} ({r.hard_enrichment:.2f}x)" for r in hard_top.itertuples()))
    lines.append("- Ambiguous-enriched evasion labels: " + "; ".join(f"{r.label} ({r.ambiguous_enrichment:.2f}x)" for r in amb_top.itertuples()))
    clarity = long_crosstab(df, "clarity_label")
    clarity_notes = []
    for label, group in clarity.groupby("label"):
        top = group.sort_values("label_pct", ascending=False).iloc[0]
        clarity_notes.append(f"{label}: mostly {top['region']} ({top['label_pct']:.1f}%)")
    lines.append("- Clarity distribution: " + "; ".join(clarity_notes))
    lines.append("")

    lines.append("## Annotator Entropy")
    lines.append(f"- {entropy_info['note']}")
    lines.append(
        f"- Rows with usable annotator votes: {entropy_info['annotator_vote_rows']}; "
        f"rows with multiple accepted evasion labels: {entropy_info['accepted_multi_label_rows']}."
    )
    lines.append("")

    def emit_examples(title: str, frame: pd.DataFrame, n: int = 12) -> None:
        lines.append(f"## {title}")
        if frame.empty:
            lines.append("- No examples available.")
            lines.append("")
            return
        for i, row in enumerate(frame.head(n).itertuples(index=False), start=1):
            s = pd.Series(row._asdict())
            q = compact_text(s.get("question", ""), 220)
            iq = compact_text(s.get("interview_question", ""), 220)
            a = compact_text(s.get("interview_answer", ""), 320)
            pred = s.get("pred_pair_seed42", "")
            lines.append(
                f"{i}. idx={s.get('idx')} | gold={s.get('gold_pair')} | seed42_pred={pred} | "
                f"conf={float(s.get('conf_mean', float('nan'))):.3f} | "
                f"var={float(s.get('var_mean', float('nan'))):.3f} | "
                f"corr={float(s.get('correctness_mean', float('nan'))):.3f}"
            )
            if q:
                lines.append(f"   - Subquestion: {q}")
            if iq and iq != q:
                lines.append(f"   - Interview question: {iq}")
            if a:
                lines.append(f"   - Answer: {a}")
            lines.append(f"   - Reading: {example_reason(s)}")
        lines.append("")

    emit_examples("Easy Examples: Low Variability, Higher Relative Confidence", examples.get("corrected_easy_examples", pd.DataFrame()))
    emit_examples("Hard Examples: Low Variability, Low Confidence", examples.get("corrected_hard_examples", pd.DataFrame()))
    emit_examples("Ambiguous Examples: High Variability", examples.get("corrected_ambiguous_examples", pd.DataFrame()))
    emit_examples("Unstable / Borderline Examples", examples.get("chronically_unstable_corrected_examples", pd.DataFrame()), n=10)

    lines.append("## Concrete Failure Takeaways")
    lines.append("- Hard examples are not merely noisy. They are often consistently low-confidence, which points to systematic mismatch between answer surface form and the fine-grained evasion label.")
    lines.append("- Ambiguous examples are model-state sensitive. Their labels tend to move with seed or epoch because the answer contains both answer-like and non-answer-like cues.")
    lines.append("- The constrained decoder removes illegal label pairs, but evasion accuracy remains the bottleneck; enforcing legality fixes ontology consistency, not semantic confusion.")
    lines.append("- Clear Reply examples can still be difficult when the evasion label is implicit, deflective, or partial, because clarity and evasion encode different judgment axes.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def report_markdown(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    pred_summary: pd.DataFrame,
    illegal_top: pd.DataFrame,
    entropy_info: dict[str, Any],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# QEvasion Corrected Cartography Reanalysis Report")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- Corrected stability CSV: `{(RE_RESULTS / 'qevasion_corrected_stability.csv').as_posix()}`")
    lines.append(f"- Corrected plots: `{RE_PLOTS.as_posix()}`")
    lines.append(f"- Qualitative examples: `{RE_EXAMPLES.as_posix()}`")
    lines.append("")
    lines.append("## Corrected Category Rule")
    lines.append(
        f"Ambiguous examples are above the {summary['config']['threshold_mode']} variability threshold "
        f"(`var_mean > {summary['config']['var_threshold']:.6f}`). "
        f"The remaining examples are split into easy/hard at confidence "
        f"`{summary['config']['confidence_threshold_low_variability']:.6f}`."
    )
    lines.append("")
    lines.append("## Main Counts")
    total = int(sum(summary["region_counts"].values()))
    for region in REGION_ORDER:
        count = int(summary["region_counts"].get(region, 0))
        lines.append(f"- {region}: {count} ({pct(count, total):.1f}%)")
    lines.append("")

    region_profile = aggregate_by_region(df)
    lines.append("## Region Profiles")
    for row in region_profile.itertuples(index=False):
        lines.append(
            f"- {row.region}: n={row.n}, confidence mean={row.conf_mean_mean:.3f}, "
            f"variability mean={row.var_mean_mean:.3f}, correctness mean={row.correctness_mean_mean:.3f}, "
            f"forgetfulness mean={row.forgetfulness_mean_mean:.3f}, "
            f"answer length median={row.answer_word_count_median:.0f}."
        )
    lines.append("")

    ev = enrichment_table(df, "evasion_label")
    lines.append("## Evasion Enrichment")
    for row in ev.sort_values("hard_enrichment", ascending=False).head(8).itertuples(index=False):
        lines.append(
            f"- {row.label}: hard {row.hard_rate * 100:.1f}% ({row.hard_enrichment:.2f}x), "
            f"ambiguous {row.ambiguous_rate * 100:.1f}% ({row.ambiguous_enrichment:.2f}x), n={row.n}."
        )
    lines.append("")

    lines.append("## Model And Illegal Pair Summary")
    if not pred_summary.empty:
        constrained_test = pred_summary[(pred_summary["model"] == "constrained") & (pred_summary["split"] == "test")]
        if not constrained_test.empty:
            lines.append(
                f"- Constrained test pair accuracy mean across seeds: "
                f"{constrained_test['is_correct_pair_rate'].mean() * 100:.1f}% "
                f"(range {constrained_test['is_correct_pair_rate'].min() * 100:.1f}-{constrained_test['is_correct_pair_rate'].max() * 100:.1f}%)."
            )
        independent = pred_summary[(pred_summary["model"] == "independent") & (pred_summary["split"] == "test")]
        if not independent.empty:
            row = independent.iloc[0]
            lines.append(
                f"- Independent seed {int(row.seed)} test illegal pair rate: "
                f"{row.illegal_pair_rate * 100:.1f}% ({int(row.illegal_pair_count)} / {int(row.n)})."
            )
        if not illegal_top.empty:
            top = illegal_top.groupby(["pred_clarity", "pred_evasion"], dropna=False)["count"].sum().sort_values(ascending=False).head(5)
            lines.append("- Most common illegal predictions: " + "; ".join(f"{a} | {b}: {int(c)}" for (a, b), c in top.items()))
    lines.append("")

    lines.append("## Annotator Entropy")
    lines.append(f"- {entropy_info['note']}")
    lines.append("")

    lines.append("## Reviewer-Facing Takeaways")
    lines.append("- The corrected category split makes ambiguity explicitly about high variability, without forcing fixed category proportions.")
    lines.append("- Hard-to-learn samples correlate with particular labels, especially non-explicit and pragmatic evasion classes; the exported enrichment tables quantify this directly.")
    lines.append("- The qualitative exports include concrete question-answer snippets for easy, hard, ambiguous, unstable, borderline, and high-confidence-wrong examples.")
    lines.append("- Illegal-pair analysis shows the value of constrained decoding separately from semantic accuracy: legality improves to zero illegal pairs, but fine-grained evasion remains hard.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dirs()

    stability, _metadata = load_stability()
    stability, config = add_corrected_regions(
        stability,
        threshold_mode=args.threshold_mode,
        explicit_var_threshold=args.var_threshold,
        explicit_confidence_threshold=args.confidence_threshold,
    )
    stability, entropy_info = extract_annotator_entropy(stability)
    stability = add_prediction_columns(stability)
    evasion_dim, evasion_dim_config = build_dimension_cartography(stability, "evasion", args.threshold_mode)
    clarity_dim, clarity_dim_config = build_dimension_cartography(stability, "clarity", args.threshold_mode)

    write_csv(stability, RE_RESULTS / "qevasion_corrected_stability.csv")
    write_csv(evasion_dim, RE_RESULTS / "qevasion_evasion_dimension_cartography.csv")
    write_csv(clarity_dim, RE_RESULTS / "qevasion_clarity_dimension_cartography.csv")

    region_counts = stability["corrected_region"].value_counts().reindex(REGION_ORDER).fillna(0).astype(int)
    region_counts_df = pd.DataFrame(
        {
            "region": region_counts.index,
            "count": region_counts.values,
            "pct": [pct(v, len(stability)) for v in region_counts.values],
        }
    )
    write_csv(region_counts_df, RE_RESULTS / "qevasion_corrected_region_counts.csv")

    region_profile = aggregate_by_region(stability)
    write_csv(region_profile, RE_RESULTS / "qevasion_corrected_region_profile.csv")
    write_csv(correlation_table(stability), RE_RESULTS / "qevasion_corrected_correlations.csv")
    write_csv(maybe_stat_tests(stability), RE_RESULTS / "qevasion_corrected_region_stat_tests.csv")

    for label_col, stem in [
        ("clarity_label", "clarity"),
        ("evasion_label", "evasion"),
        ("gold_pair", "pair"),
    ]:
        long = long_crosstab(stability, label_col)
        write_csv(long, RE_RESULTS / f"qevasion_corrected_region_by_{stem}.csv")
        write_csv(enrichment_table(stability, label_col), RE_RESULTS / f"qevasion_corrected_{stem}_enrichment.csv")

    evasion_diff = region_difference_table(stability, "evasion_label", "evasion")
    clarity_diff = region_difference_table(stability, "clarity_label", "clarity")
    dimension_metrics = pd.DataFrame(
        [
            association_metrics(stability, "evasion_label", "evasion"),
            association_metrics(stability, "clarity_label", "clarity"),
        ]
    )
    write_csv(evasion_diff, RE_RESULTS / "qevasion_final_evasion_category_differences.csv")
    write_csv(clarity_diff, RE_RESULTS / "qevasion_final_clarity_category_differences.csv")
    write_csv(dimension_metrics, RE_RESULTS / "qevasion_final_dimension_association_comparison.csv")

    evasion_dim_counts = evasion_dim["evasion_region"].value_counts().reindex(REGION_ORDER).fillna(0).astype(int)
    clarity_dim_counts = clarity_dim["clarity_region"].value_counts().reindex(REGION_ORDER).fillna(0).astype(int)
    write_csv(
        pd.DataFrame({"region": evasion_dim_counts.index, "count": evasion_dim_counts.values, "pct": [pct(v, len(evasion_dim)) for v in evasion_dim_counts.values]}),
        RE_RESULTS / "qevasion_evasion_dimension_region_counts.csv",
    )
    write_csv(
        pd.DataFrame({"region": clarity_dim_counts.index, "count": clarity_dim_counts.values, "pct": [pct(v, len(clarity_dim)) for v in clarity_dim_counts.values]}),
        RE_RESULTS / "qevasion_clarity_dimension_region_counts.csv",
    )
    evasion_dimension_diff = dimension_region_difference_table(
        evasion_dim,
        "evasion_label",
        "evasion_region",
        "evasion",
        "evasion",
    )
    clarity_dimension_diff = dimension_region_difference_table(
        clarity_dim,
        "clarity_label",
        "clarity_region",
        "clarity",
        "clarity",
    )
    dimension_only_metrics = pd.DataFrame(
        [
            dimension_association_metrics(evasion_dim, "evasion_label", "evasion_region", "evasion"),
            dimension_association_metrics(clarity_dim, "clarity_label", "clarity_region", "clarity"),
        ]
    )
    write_csv(evasion_dimension_diff, RE_RESULTS / "qevasion_evasion_dimension_category_differences.csv")
    write_csv(clarity_dimension_diff, RE_RESULTS / "qevasion_clarity_dimension_category_differences.csv")
    write_csv(dimension_only_metrics, RE_RESULTS / "qevasion_dimension_specific_association_comparison.csv")

    metadata_cols = [
        "corrected_region",
        "multiple_questions",
        "affirmative_questions",
        "inaudible",
        "question_word_count",
        "answer_word_count",
        "subquestion_word_count",
        "president",
    ]
    metadata_cols = [col for col in metadata_cols if col in stability.columns]
    write_csv(stability[metadata_cols], RE_RESULTS / "qevasion_corrected_metadata_with_region.csv")

    predictions = load_prediction_files()
    pred_outputs = prediction_summaries(predictions, stability)
    for name, frame in pred_outputs.items():
        write_csv(frame, RE_RESULTS / f"qevasion_corrected_{name}.csv")

    examples = select_examples(stability, config)
    for name, frame in examples.items():
        write_csv(frame, RE_EXAMPLES / f"{name}.csv")

    if "high_confidence_wrong" in pred_outputs and not pred_outputs["high_confidence_wrong"].empty:
        write_csv(pred_outputs["high_confidence_wrong"], RE_EXAMPLES / "high_confidence_wrong_all_models.csv")

    plot_cartography(stability, config, RE_PLOTS / "corrected_cartography.png")
    clarity_table = pd.crosstab(stability["corrected_region"], stability["clarity_label"])
    evasion_table = pd.crosstab(stability["corrected_region"], stability["evasion_label"])
    plot_stacked(clarity_table, "Clarity Label Composition By Corrected Category", "Within-category share (%)", RE_PLOTS / "corrected_region_by_clarity.png")
    plot_stacked(evasion_table, "Evasion Label Composition By Corrected Category", "Within-category share (%)", RE_PLOTS / "corrected_region_by_evasion.png")
    plot_illegal_pairs(
        pred_outputs.get("prediction_summary", pd.DataFrame()),
        pred_outputs.get("illegal_top_pairs", pd.DataFrame()),
        RE_PLOTS / "illegal_pair_rates.png",
    )
    plot_entropy(stability, entropy_info, RE_PLOTS / "annotator_entropy_by_corrected_region.png")
    plot_metadata(stability, RE_PLOTS / "corrected_metadata_by_region.png")
    plot_stability(stability, RE_PLOTS / "corrected_region_stability_heatmap.png")
    plot_dimension_cartography_regions(
        evasion_dim,
        "evasion",
        evasion_dim_config,
        RE_PLOTS / "final_evasion_cartography.png",
    )
    plot_dimension_cartography_regions(
        clarity_dim,
        "clarity",
        clarity_dim_config,
        RE_PLOTS / "final_clarity_cartography.png",
    )
    plot_dimension_cartography_regions(
        evasion_dim,
        "evasion",
        evasion_dim_config,
        RE_PLOTS / "final_evasion_cartography_by_label.png",
    )
    plot_dimension_cartography_regions(
        clarity_dim,
        "clarity",
        clarity_dim_config,
        RE_PLOTS / "final_clarity_cartography_by_label.png",
    )
    plot_dimension_distribution(
        evasion_dim,
        "evasion_label",
        "evasion_region",
        "Evasion Labels Inside Evasion-Only Cartography Categories",
        RE_PLOTS / "final_evasion_category_distribution.png",
    )
    plot_dimension_distribution(
        clarity_dim,
        "clarity_label",
        "clarity_region",
        "Clarity Labels Inside Clarity-Only Cartography Categories",
        RE_PLOTS / "final_clarity_category_distribution.png",
    )
    plot_dimension_difference(
        evasion_dimension_diff,
        "Evasion-Only Cartography: Evasion Label Category Differences",
        RE_PLOTS / "final_evasion_category_differences.png",
    )
    plot_dimension_difference(
        clarity_dimension_diff,
        "Clarity-Only Cartography: Clarity Label Category Differences",
        RE_PLOTS / "final_clarity_category_differences.png",
    )
    plot_dimension_comparison(
        dimension_only_metrics,
        RE_PLOTS / "final_evasion_vs_clarity_association.png",
    )
    plot_label_cartography(
        stability,
        "evasion_label",
        "Joint-Pair Cartography Colored Only By Evasion Label",
        RE_PLOTS / "pair_metric_evasion_cartography_by_label.png",
    )
    plot_label_cartography(
        stability,
        "clarity_label",
        "Joint-Pair Cartography Colored Only By Clarity Label",
        RE_PLOTS / "pair_metric_clarity_cartography_by_label.png",
    )
    plot_stacked(
        evasion_table,
        "Evasion-Only Category Composition",
        "Within-category evasion share (%)",
        RE_PLOTS / "pair_metric_evasion_category_distribution.png",
    )
    plot_stacked(
        clarity_table,
        "Clarity-Only Category Composition",
        "Within-category clarity share (%)",
        RE_PLOTS / "pair_metric_clarity_category_distribution.png",
    )
    plot_region_difference(
        evasion_diff,
        "Evasion Labels: Category Differences From Overall Rates",
        RE_PLOTS / "pair_metric_evasion_category_differences.png",
    )
    plot_region_difference(
        clarity_diff,
        "Clarity Labels: Category Differences From Overall Rates",
        RE_PLOTS / "pair_metric_clarity_category_differences.png",
    )
    plot_dimension_comparison(
        dimension_metrics,
        RE_PLOTS / "pair_metric_evasion_vs_clarity_association.png",
    )
    if "prediction_by_label" in pred_outputs:
        plot_accuracy_by_label(
            pred_outputs["prediction_by_label"],
            "evasion",
            RE_PLOTS / "final_evasion_heldout_accuracy_by_label.png",
        )
        plot_accuracy_by_label(
            pred_outputs["prediction_by_label"],
            "clarity",
            RE_PLOTS / "final_clarity_heldout_accuracy_by_label.png",
        )

    summary: dict[str, Any] = {
        "config": config,
        "dimension_configs": {
            "evasion": evasion_dim_config,
            "clarity": clarity_dim_config,
        },
        "region_counts": region_counts.to_dict(),
        "dimension_region_counts": {
            "evasion": evasion_dim_counts.to_dict(),
            "clarity": clarity_dim_counts.to_dict(),
        },
        "entropy": entropy_info,
        "outputs": {
            "results": str(RE_RESULTS),
            "plots": str(RE_PLOTS),
            "examples": str(RE_EXAMPLES),
        },
    }
    if "prediction_summary" in pred_outputs:
        pred_summary = pred_outputs["prediction_summary"]
        illegal_summary = pred_summary[["model", "split", "seed", "n", "illegal_pair_count", "illegal_pair_rate"]].to_dict("records")
        summary["illegal_pairs"] = illegal_summary
        constrained_test = pred_summary[(pred_summary["model"] == "constrained") & (pred_summary["split"] == "test")]
        if not constrained_test.empty:
            summary["constrained_test_pair_accuracy_mean"] = float(constrained_test["is_correct_pair_rate"].mean())
            summary["constrained_test_pair_accuracy_min"] = float(constrained_test["is_correct_pair_rate"].min())
            summary["constrained_test_pair_accuracy_max"] = float(constrained_test["is_correct_pair_rate"].max())

    ev_enrich = enrichment_table(stability, "evasion_label")
    summary["top_hard_evasion_enrichment"] = ev_enrich.sort_values("hard_enrichment", ascending=False).head(10).to_dict("records")
    summary["top_ambiguous_evasion_enrichment"] = ev_enrich.sort_values("ambiguous_enrichment", ascending=False).head(10).to_dict("records")
    summary["dimension_association"] = dimension_metrics.to_dict("records")
    summary["dimension_specific_association"] = dimension_only_metrics.to_dict("records")

    write_json(summary, RE_RESULTS / "qevasion_corrected_reanalysis_summary.json")
    write_qualitative_markdown(
        stability,
        examples,
        summary,
        entropy_info,
        RE_EXAMPLES / "qualitative_deep_dive.md",
    )
    report_markdown(
        stability,
        read_csv(RESULTS_DIR / "qevasion_all_metrics.csv") if (RESULTS_DIR / "qevasion_all_metrics.csv").exists() else pd.DataFrame(),
        pred_outputs.get("prediction_summary", pd.DataFrame()),
        pred_outputs.get("illegal_top_pairs", pd.DataFrame()),
        entropy_info,
        summary,
        REANALYSIS_DIR / "REANALYSIS_REPORT.md",
    )
    write_final_qualitative_analysis(
        stability,
        evasion_diff,
        clarity_diff,
        dimension_metrics,
        examples,
        summary,
        RE_EXAMPLES / "final_qualitative_analysis.md",
    )
    write_dimension_qualitative_analysis(
        evasion_dim,
        clarity_dim,
        evasion_dimension_diff,
        clarity_dimension_diff,
        evasion_dim_config,
        clarity_dim_config,
        RE_EXAMPLES / "dimension_specific_qualitative_analysis.md",
    )

    print("Corrected QEvasion cartography reanalysis complete.")
    print(f"Results: {RE_RESULTS}")
    print(f"Plots: {RE_PLOTS}")
    print(f"Examples: {RE_EXAMPLES}")
    print("Region counts:", region_counts.to_dict())
    if not entropy_info["available"]:
        print("Annotator entropy unavailable:", entropy_info["note"])


if __name__ == "__main__":
    main()
