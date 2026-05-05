#!/usr/bin/env python3
"""Generate paper-ready figures for QEvasion clarity cartography."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "output" / "clarity_cartography" / "results"
DEFAULT_LATEX_DIR = ROOT / "latex"
REGION_ORDER = ["easy", "ambiguous", "hard"]
REGION_DISPLAY = {"easy": "Easy", "ambiguous": "Ambig.", "hard": "Hard"}
REGION_COLORS = {"easy": "#3B82BD", "ambiguous": "#FDB847", "hard": "#D94F4F"}
VARIANT_PREFIX = {
    "regular": "qevasion_clarity_regular",
    "hierarchical": "qevasion_clarity_hierarchical",
}
DISPLAY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
LABEL_DISPLAY = {
    "Clear Non-Reply": "Non-Reply",
    "Ambivalent": "Ambivalent",
    "Clear Reply": "Reply",
}
LABEL_ORDER = ["Clear Non-Reply", "Ambivalent", "Clear Reply"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--latex-dir", type=Path, default=DEFAULT_LATEX_DIR)
    parser.add_argument("--variant", choices=["regular", "hierarchical", "both"], default="both")
    parser.add_argument(
        "--paper-alias",
        choices=["regular", "hierarchical"],
        default=None,
        help=(
            "Also copy the selected variant to canonical paper names "
            "qevasion_clarity_cartography_no_title.png and "
            "qevasion_clarity_region_distribution_no_title.png."
        ),
    )
    return parser.parse_args()


def read_stability(input_dir: Path, variant: str) -> pd.DataFrame:
    prefix = VARIANT_PREFIX[variant]
    path = input_dir / f"{prefix}_stability.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run extract_results.sh first, or pass --input-dir.")
    df = pd.read_csv(path)
    if "majority_region" not in df.columns and "region" in df.columns:
        df["majority_region"] = df["region"]
    for col in ["conf_mean", "var_mean", "correctness_mean"]:
        if col not in df.columns:
            fallback = {"conf_mean": "confidence", "var_mean": "variability", "correctness_mean": "correctness"}[col]
            if fallback in df.columns:
                df[col] = df[fallback]
    required = {"conf_mean", "var_mean", "correctness_mean", "majority_region", "clarity_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def plot_cartography(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 7.4), dpi=160)
    scatter = ax.scatter(
        pd.to_numeric(df["var_mean"], errors="coerce"),
        pd.to_numeric(df["conf_mean"], errors="coerce"),
        c=pd.to_numeric(df["correctness_mean"], errors="coerce"),
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


def plot_region_distribution(df: pd.DataFrame, path: Path) -> None:
    table = pd.crosstab(df["clarity_label"], df["majority_region"])
    labels = [label for label in LABEL_ORDER if label in table.index]
    labels += [label for label in table.index if label not in labels]
    table = table.reindex(index=labels, columns=REGION_ORDER).fillna(0).astype(int)
    pct_table = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0

    fig, ax = plt.subplots(figsize=(8.7, 2.65), dpi=160)
    left = np.zeros(len(pct_table))
    y = np.arange(len(pct_table.index))
    for region in REGION_ORDER:
        values = pct_table[region].to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            height=0.66,
            label=REGION_DISPLAY[region],
            color=REGION_COLORS[region],
            edgecolor="white",
            linewidth=1.35,
        )
        for j, value in enumerate(values):
            if value >= 7:
                text_color = "white" if region in {"easy", "hard"} else "#111111"
                ax.text(
                    left[j] + value / 2,
                    j,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=text_color,
                )
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([LABEL_DISPLAY.get(label, str(label)) for label in pct_table.index], fontsize=16)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("%", x=0.985, fontsize=13)
    ax.tick_params(axis="x", labelsize=12, colors="#444444")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.03),
        ncol=3,
        frameon=False,
        fontsize=13,
        handlelength=1.0,
        handleheight=0.9,
        columnspacing=1.4,
        handletextpad=0.35,
        borderaxespad=0.0,
        prop={"weight": "bold", "size": 13},
    )
    fig.tight_layout(pad=0.45)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_variant(input_dir: Path, latex_dir: Path, variant: str) -> dict[str, Path]:
    latex_dir.mkdir(parents=True, exist_ok=True)
    df = read_stability(input_dir, variant)
    stem = VARIANT_PREFIX[variant]
    carto_path = latex_dir / f"{stem}_cartography_no_title.png"
    dist_path = latex_dir / f"{stem}_region_distribution_no_title.png"
    plot_cartography(df, carto_path)
    plot_region_distribution(df, dist_path)
    return {"cartography": carto_path, "distribution": dist_path}


def main() -> None:
    args = parse_args()
    variants = ["regular", "hierarchical"] if args.variant == "both" else [args.variant]
    generated: dict[str, dict[str, Path]] = {}
    for variant in variants:
        generated[variant] = generate_variant(args.input_dir, args.latex_dir, variant)

    if args.paper_alias:
        if args.paper_alias not in generated:
            generated[args.paper_alias] = generate_variant(args.input_dir, args.latex_dir, args.paper_alias)
        shutil.copyfile(
            generated[args.paper_alias]["cartography"],
            args.latex_dir / "qevasion_clarity_cartography_no_title.png",
        )
        shutil.copyfile(
            generated[args.paper_alias]["distribution"],
            args.latex_dir / "qevasion_clarity_region_distribution_no_title.png",
        )

    for variant, paths in generated.items():
        print(f"{variant}:")
        print(f"  cartography:  {paths['cartography']}")
        print(f"  distribution: {paths['distribution']}")
    if args.paper_alias:
        print(f"paper aliases updated from: {args.paper_alias}")


if __name__ == "__main__":
    main()
