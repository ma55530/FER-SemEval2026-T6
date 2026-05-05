#!/usr/bin/env python3
"""SemEval/QEvasion multitask dataset cartography.

Trains the QEvasion multitask variants, records per-example training dynamics,
and exports cartography, stability, breakdown, plot, and qualitative example
artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tarfile
import time
import traceback
import warnings
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


REGION_NAMES = ["easy", "ambiguous", "hard"]
REGION_COLORS = {"easy": "#2ca02c", "ambiguous": "#ff7f0e", "hard": "#d62728"}
DEFAULT_SEEDS = [42, 123, 456, 789, 1337]
ABLATION_SEEDS = [42, 123, 456, 789, 1337]
CLARITY_MAPPING = {"Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2}
INV_CLARITY_MAPPING = {v: k for k, v in CLARITY_MAPPING.items()}
ANNOTATOR_COLS = ["annotator1", "annotator2", "annotator3"]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_clarity(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text == "Ambivalent Reply":
        return "Ambivalent"
    return text


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def compact_text(text: str, max_chars: int = 360) -> str:
    text = " ".join(safe_str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def balanced_weights(labels: np.ndarray, n_classes: int, boost_idx: int | None = None) -> torch.Tensor:
    labels = labels[(labels >= 0) & (labels != -100)]
    counts = np.bincount(labels.astype(int), minlength=n_classes).astype(np.float64)
    total = counts.sum()
    weights = np.ones(n_classes, dtype=np.float32)
    if total > 0:
        weights = (total / (n_classes * np.maximum(counts, 1))).astype(np.float32)
        weights[counts == 0] = 0.0
    if boost_idx is not None and 0 <= boost_idx < n_classes and weights[boost_idx] > 0:
        weights[boost_idx] *= 2.5
    return torch.tensor(weights, dtype=torch.float32)


def parse_seed_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


@dataclass
class RunConfig:
    dataset_name: str
    model_name: str
    output_dir: Path
    seeds: list[int]
    ablation_seeds: list[int]
    epochs: int
    batch_size: int
    eval_batch_size: int
    grad_accumulation: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    max_length: int
    clarity_loss_weight: float
    freeze_embeddings: bool
    mixed_precision: bool
    gradient_checkpointing: bool
    dry_run: bool
    subset_train: int | None
    subset_test: int | None
    num_workers: int
    run_ablations: bool
    large_one_run: bool
    export_example_limit: int
    resume: bool
    save_checkpoints: bool
    save_epoch_predictions: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="ailsntua/QEvasion")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./output"))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--ablation-seeds", default=",".join(map(str, ABLATION_SEEDS)))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--clarity-loss-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-freeze-embeddings", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--subset-train", type=int, default=None)
    parser.add_argument("--subset-test", type=int, default=None)
    parser.add_argument("--large-one-run", action="store_true")
    parser.add_argument("--export-example-limit", type=int, default=80)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--no-epoch-predictions", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    seeds = parse_seed_list(args.seeds, DEFAULT_SEEDS)
    ablation_seeds = parse_seed_list(args.ablation_seeds, ABLATION_SEEDS)
    epochs = args.epochs
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    grad_accumulation = args.grad_accumulation
    model_name = args.model_name
    output_dir = Path(args.output_dir)
    gradient_checkpointing = args.gradient_checkpointing

    if args.dry_run:
        seeds = [seeds[0]]
        ablation_seeds = [seeds[0]]
        epochs = 1
        batch_size = min(batch_size, 4)
        eval_batch_size = min(eval_batch_size, 8)
        grad_accumulation = 1
        args.subset_train = args.subset_train or 48
        args.subset_test = args.subset_test or 24

    if args.large_one_run:
        model_name = args.model_name
        seeds = [seeds[0]]
        ablation_seeds = []
        batch_size = min(batch_size, 2)
        eval_batch_size = min(eval_batch_size, 4)
        grad_accumulation = max(grad_accumulation, 16)
        gradient_checkpointing = True
        output_dir = output_dir / "deberta_v3_large_one_run"

    return RunConfig(
        dataset_name=args.dataset_name,
        model_name=model_name,
        output_dir=output_dir,
        seeds=seeds,
        ablation_seeds=ablation_seeds,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        grad_accumulation=grad_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_length,
        clarity_loss_weight=args.clarity_loss_weight,
        freeze_embeddings=not args.no_freeze_embeddings,
        mixed_precision=not args.no_mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
        dry_run=args.dry_run,
        subset_train=args.subset_train,
        subset_test=args.subset_test,
        num_workers=args.num_workers,
        run_ablations=not args.skip_ablations and not args.large_one_run,
        large_one_run=args.large_one_run,
        export_example_limit=args.export_example_limit,
        resume=not args.no_resume,
        save_checkpoints=not args.no_checkpoints,
        save_epoch_predictions=not args.no_epoch_predictions,
    )


class QEDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer: AutoTokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["interview_question"],
            ex["interview_answer"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels_clarity": torch.tensor(ex["labels_clarity"], dtype=torch.long),
            "labels_evasion": torch.tensor(ex["labels_evasion"], dtype=torch.long),
            "labels_joint": torch.tensor(ex["labels_joint"], dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


class IndependentMultitaskModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_evasion: int,
        clarity_weights: torch.Tensor,
        evasion_weights: torch.Tensor,
        clarity_loss_weight: float,
        gradient_checkpointing: bool = False,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if freeze_embeddings and hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        dropout_prob = getattr(self.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.cls_clarity = nn.Linear(self.config.hidden_size, 3)
        self.cls_evasion = nn.Linear(self.config.hidden_size, n_evasion)
        self.loss_c = nn.CrossEntropyLoss(weight=clarity_weights, ignore_index=-1)
        self.loss_e = nn.CrossEntropyLoss(weight=evasion_weights, ignore_index=-100)
        self.clarity_loss_weight = clarity_loss_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels_clarity: torch.Tensor | None = None,
        labels_evasion: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        logits_c = self.cls_clarity(pooled)
        logits_e = self.cls_evasion(pooled)
        loss = None
        if labels_clarity is not None and labels_evasion is not None:
            loss = self.loss_e(logits_e, labels_evasion) + self.clarity_loss_weight * self.loss_c(
                logits_c, labels_clarity
            )
        return {"loss": loss, "logits_clarity": logits_c, "logits_evasion": logits_e}


class JointPairModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_joint: int,
        joint_weights: torch.Tensor,
        gradient_checkpointing: bool = False,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if freeze_embeddings and hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        dropout_prob = getattr(self.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, n_joint)
        self.loss_fct = nn.CrossEntropyLoss(weight=joint_weights, ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels_joint: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        logits = self.classifier(pooled)
        loss = self.loss_fct(logits, labels_joint) if labels_joint is not None else None
        return {"loss": loss, "logits_joint": logits}


class CartographyTracker:
    def __init__(self, name: str):
        self.name = name
        self.epoch_probs: dict[int, list[float]] = defaultdict(list)
        self.epoch_preds: dict[int, list[str]] = defaultdict(list)
        self.epoch_correct: dict[int, list[int]] = defaultdict(list)
        self.epoch_losses: dict[int, list[float]] = defaultdict(list)
        self.gold_labels: dict[int, str] = {}
        self.extra_values: dict[str, dict[int, list[Any]]] = defaultdict(lambda: defaultdict(list))

    def record(
        self,
        idx: int,
        prob_true: float,
        pred: str,
        gold: str,
        correct: bool,
        loss_val: float | None = None,
        **extra: Any,
    ) -> None:
        self.epoch_probs[idx].append(float(prob_true))
        self.epoch_preds[idx].append(str(pred))
        self.epoch_correct[idx].append(int(bool(correct)))
        if loss_val is not None:
            self.epoch_losses[idx].append(float(loss_val))
        self.gold_labels[idx] = str(gold)
        for key, value in extra.items():
            if isinstance(value, np.generic):
                value = value.item()
            self.extra_values[key][idx].append(value)

    def compute_metrics(self) -> pd.DataFrame:
        recs = []
        for idx in sorted(self.epoch_probs):
            probs = np.asarray(self.epoch_probs[idx], dtype=np.float64)
            correct = self.epoch_correct[idx]
            losses = self.epoch_losses.get(idx, [])
            rec: dict[str, Any] = {
                "idx": idx,
                "gold_pair": self.gold_labels[idx],
                "confidence": float(probs.mean()),
                "variability": float(probs.std(ddof=0)),
                "correctness": float(np.mean(correct)) if correct else 0.0,
                "forgetfulness": int(
                    sum(1 for j in range(1, len(correct)) if correct[j - 1] == 1 and correct[j] == 0)
                ),
                "first_confidence": float(probs[0]),
                "last_confidence": float(probs[-1]),
                "confidence_delta": float(probs[-1] - probs[0]),
                "epoch_probs": json_dumps([float(x) for x in probs.tolist()]),
                "epoch_preds": json_dumps(self.epoch_preds[idx]),
                "epoch_correct": json_dumps([int(x) for x in correct]),
            }
            if losses:
                rec["loss_mean"] = float(np.mean(losses))
                rec["loss_last"] = float(losses[-1])
                rec["epoch_losses"] = json_dumps([float(x) for x in losses])
            for key, values_by_idx in self.extra_values.items():
                values = values_by_idx.get(idx, [])
                if not values:
                    continue
                rec[f"{key}_series"] = json_dumps(values)
                rec[f"{key}_last"] = values[-1]
                if all(isinstance(v, (int, float, np.integer, np.floating, bool)) for v in values):
                    arr = np.asarray(values, dtype=np.float64)
                    rec[f"{key}_mean"] = float(arr.mean())
                    rec[f"{key}_std"] = float(arr.std(ddof=0))
            recs.append(rec)
        return pd.DataFrame(recs)

    def to_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "epoch_probs": {int(k): list(v) for k, v in self.epoch_probs.items()},
            "epoch_preds": {int(k): list(v) for k, v in self.epoch_preds.items()},
            "epoch_correct": {int(k): list(v) for k, v in self.epoch_correct.items()},
            "epoch_losses": {int(k): list(v) for k, v in self.epoch_losses.items()},
            "gold_labels": {int(k): v for k, v in self.gold_labels.items()},
            "extra_values": {
                key: {int(idx): list(values) for idx, values in values_by_idx.items()}
                for key, values_by_idx in self.extra_values.items()
            },
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "CartographyTracker":
        tracker = cls(state.get("name", "tracker"))
        for key, values in state.get("epoch_probs", {}).items():
            tracker.epoch_probs[int(key)] = list(values)
        for key, values in state.get("epoch_preds", {}).items():
            tracker.epoch_preds[int(key)] = list(values)
        for key, values in state.get("epoch_correct", {}).items():
            tracker.epoch_correct[int(key)] = list(values)
        for key, values in state.get("epoch_losses", {}).items():
            tracker.epoch_losses[int(key)] = list(values)
        tracker.gold_labels = {int(k): str(v) for k, v in state.get("gold_labels", {}).items()}
        for extra_key, values_by_idx in state.get("extra_values", {}).items():
            for idx, values in values_by_idx.items():
                tracker.extra_values[extra_key][int(idx)] = list(values)
        return tracker


def get_train_evasion_label(example: dict[str, Any], unique_evasion: set[str]) -> str | None:
    evasion = safe_str(example.get("evasion_label")).strip()
    if evasion in unique_evasion:
        return evasion
    votes = []
    for col in ANNOTATOR_COLS:
        value = safe_str(example.get(col)).strip()
        if value in unique_evasion:
            votes.append(value)
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


def collect_label_space(dataset: Any) -> tuple[list[str], list[tuple[str, str]]]:
    all_evasion: list[str] = []
    for split in ["train", "test"]:
        for row in dataset[split]:
            evasion = safe_str(row.get("evasion_label")).strip()
            if evasion:
                all_evasion.append(evasion)
            for col in ANNOTATOR_COLS:
                value = safe_str(row.get(col)).strip()
                if value:
                    all_evasion.append(value)
    unique_evasion = sorted(set(all_evasion))
    evasion_set = set(unique_evasion)
    legal_pairs: set[tuple[str, str]] = set()
    for row in dataset["train"]:
        clarity = normalize_clarity(row.get("clarity_label"))
        evasion = get_train_evasion_label(row, evasion_set)
        if clarity in CLARITY_MAPPING and evasion:
            legal_pairs.add((clarity, evasion))
    return unique_evasion, sorted(legal_pairs)


def prepare_examples(
    dataset: Any,
    split: str,
    evasion_encoder: LabelEncoder,
    joint_encoder: LabelEncoder,
    legal_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    unique_evasion = set(evasion_encoder.classes_)
    examples = []
    for local_idx, row in enumerate(dataset[split]):
        clarity = normalize_clarity(row.get("clarity_label"))
        evasion = get_train_evasion_label(row, unique_evasion)
        c_idx = CLARITY_MAPPING.get(clarity, -1)
        e_idx = int(evasion_encoder.transform([evasion])[0]) if evasion in unique_evasion else -100
        joint_label = -100
        pair = None
        if clarity in CLARITY_MAPPING and evasion in unique_evasion and (clarity, evasion) in legal_pairs:
            pair = f"{clarity} | {evasion}"
            joint_label = int(joint_encoder.transform([pair])[0])

        annotator_votes = []
        for col in ANNOTATOR_COLS:
            value = safe_str(row.get(col)).strip()
            if value:
                annotator_votes.append(value)
        accepted_evasion = []
        primary_evasion = safe_str(row.get("evasion_label")).strip()
        if primary_evasion:
            accepted_evasion.append(primary_evasion)
        accepted_evasion.extend(v for v in annotator_votes if v and v not in accepted_evasion)
        if evasion and evasion not in accepted_evasion:
            accepted_evasion.append(evasion)

        question = safe_str(row.get("interview_question"))
        answer = safe_str(row.get("interview_answer"))
        subquestion = safe_str(row.get("question"))
        out = {
            "idx": local_idx,
            "split": split,
            "dataset_index": row.get("index", local_idx),
            "title": safe_str(row.get("title")),
            "date": safe_str(row.get("date")),
            "president": safe_str(row.get("president")),
            "url": safe_str(row.get("url")),
            "question_order": row.get("question_order", ""),
            "interview_question": question,
            "interview_answer": answer,
            "question": subquestion,
            "gpt3.5_summary": safe_str(row.get("gpt3.5_summary")),
            "gpt3.5_prediction": safe_str(row.get("gpt3.5_prediction")),
            "clarity_label": clarity or "",
            "evasion_label": evasion or "",
            "gold_pair": pair or "",
            "labels_clarity": c_idx,
            "labels_evasion": e_idx,
            "labels_joint": joint_label,
            "annotator_votes": json_dumps(annotator_votes),
            "accepted_evasion_labels": json_dumps(accepted_evasion),
            "inaudible": bool(row.get("inaudible", False)),
            "multiple_questions": bool(row.get("multiple_questions", False)),
            "affirmative_questions": bool(row.get("affirmative_questions", False)),
            "question_word_count": len(question.split()),
            "answer_word_count": len(answer.split()),
            "subquestion_word_count": len(subquestion.split()),
        }
        for col in ANNOTATOR_COLS:
            out[col] = safe_str(row.get(col))
        examples.append(out)
    return examples


def stratified_subset(examples: list[dict[str, Any]], n: int | None, seed: int = 42) -> list[dict[str, Any]]:
    if n is None or n >= len(examples):
        return examples
    rng = random.Random(seed)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_pair[ex["gold_pair"] or "missing"].append(ex)
    per_group = max(1, n // max(1, len(by_pair)))
    selected = []
    leftovers = []
    for group in by_pair.values():
        group = list(group)
        rng.shuffle(group)
        selected.extend(group[:per_group])
        leftovers.extend(group[per_group:])
    rng.shuffle(leftovers)
    selected.extend(leftovers[: max(0, n - len(selected))])
    selected = selected[:n]
    for new_idx, ex in enumerate(selected):
        ex = dict(ex)
        ex["idx"] = new_idx
        selected[new_idx] = ex
    return selected


def examples_metadata(examples: list[dict[str, Any]]) -> pd.DataFrame:
    cols = [
        "idx",
        "split",
        "dataset_index",
        "title",
        "date",
        "president",
        "url",
        "question_order",
        "interview_question",
        "interview_answer",
        "question",
        "clarity_label",
        "evasion_label",
        "gold_pair",
        "annotator_votes",
        "accepted_evasion_labels",
        "inaudible",
        "multiple_questions",
        "affirmative_questions",
        "question_word_count",
        "answer_word_count",
        "subquestion_word_count",
    ] + ANNOTATOR_COLS
    return pd.DataFrame([{c: ex.get(c, "") for c in cols} for ex in examples])


def attach_metadata(carto_df: pd.DataFrame, examples: list[dict[str, Any]]) -> pd.DataFrame:
    meta = examples_metadata(examples)
    duplicate_cols = [c for c in meta.columns if c in carto_df.columns and c != "idx"]
    if duplicate_cols:
        meta = meta.drop(columns=duplicate_cols)
    return carto_df.merge(meta, on="idx", how="left")


def ranked_percentile_regions(confidence: pd.Series) -> np.ndarray:
    n = len(confidence)
    if n < 3:
        return np.asarray(["ambiguous"] * n, dtype=object)
    regions = np.asarray(["ambiguous"] * n, dtype=object)
    order = np.argsort(confidence.to_numpy(dtype=np.float64), kind="mergesort")
    third = max(1, n // 3)
    regions[order[:third]] = "hard"
    regions[order[-third:]] = "easy"
    return regions


def assign_regions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vt = float(df["variability"].median()) if len(df) else 0.0
    df["region_classic"] = np.select(
        [
            (df["confidence"] >= 0.5) & (df["variability"] <= vt),
            (df["confidence"] < 0.5) & (df["variability"] <= vt),
            df["variability"] > vt,
        ],
        ["easy", "hard", "ambiguous"],
        default="ambiguous",
    )
    df["region_percentile"] = ranked_percentile_regions(df["confidence"])
    classic_n_regions = int(pd.Series(df["region_classic"]).nunique())
    percentile_n_regions = int(pd.Series(df["region_percentile"]).nunique())
    if len(df) >= 3 and classic_n_regions < 3 and percentile_n_regions >= 3:
        df["region"] = df["region_percentile"]
        df["region_assignment"] = "percentile_fallback"
    else:
        df["region"] = df["region_classic"]
        df["region_assignment"] = "classic"
    return df


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = ["input_ids", "attention_mask", "token_type_ids"]
    return {k: batch[k] for k in keys if k in batch}


def preferred_amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    cfg: RunConfig,
    model_kind: str,
) -> float:
    model.train()
    amp_dtype = preferred_amp_dtype(device) if cfg.mixed_precision else None
    amp_enabled = amp_dtype is not None
    has_fp16_trainable_params = any(
        p.requires_grad and p.dtype == torch.float16 for p in model.parameters()
    )
    use_scaler = amp_enabled and amp_dtype == torch.float16 and not has_fp16_trainable_params
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = batch_to_device(batch, device)
        labels = {
            "labels_clarity": batch["labels_clarity"],
            "labels_evasion": batch["labels_evasion"],
            "labels_joint": batch["labels_joint"],
        }
        amp_context = (
            torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype)
            if amp_enabled
            else nullcontext()
        )
        with amp_context:
            if model_kind == "joint":
                out = model(**model_inputs(batch), labels_joint=labels["labels_joint"])
            else:
                out = model(
                    **model_inputs(batch),
                    labels_clarity=labels["labels_clarity"],
                    labels_evasion=labels["labels_evasion"],
                )
            loss = out["loss"] / cfg.grad_accumulation
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % cfg.grad_accumulation == 0 or step == len(loader):
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu().item()) * cfg.grad_accumulation
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_independent(
    model: IndependentMultitaskModel,
    loader: DataLoader,
    device: torch.device,
    tracker_constrained: CartographyTracker | None,
    tracker_independent: CartographyTracker | None,
    examples: list[dict[str, Any]],
    evasion_encoder: LabelEncoder,
    legal_pairs: set[tuple[str, str]],
    constraint_mask: torch.Tensor,
    split_name: str,
) -> dict[str, float]:
    model.eval()
    correct_joint_constrained = 0
    correct_joint_independent = 0
    total = 0
    illegal_independent = 0
    mask = constraint_mask.to(device)

    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(**model_inputs(batch))
        logits_c = out["logits_clarity"]
        logits_e = out["logits_evasion"]
        probs_c = F.softmax(logits_c, dim=-1)
        probs_e = F.softmax(logits_e, dim=-1)
        pred_c_ind = probs_c.argmax(dim=-1)
        pred_e_ind = probs_e.argmax(dim=-1)

        joint_raw = probs_c.unsqueeze(2) * probs_e.unsqueeze(1)
        masked = joint_raw * mask.unsqueeze(0)
        denom = masked.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        masked_norm = masked / denom
        flat = masked.view(masked.shape[0], -1)
        pred_flat = flat.argmax(dim=-1)
        pred_c_cons = pred_flat // len(evasion_encoder.classes_)
        pred_e_cons = pred_flat % len(evasion_encoder.classes_)

        labels_c = batch["labels_clarity"]
        labels_e = batch["labels_evasion"]
        ce_c = F.cross_entropy(logits_c, labels_c, reduction="none", ignore_index=-1)
        ce_e = F.cross_entropy(logits_e, labels_e, reduction="none", ignore_index=-100)
        per_loss = ce_e + 0.5 * ce_c

        for i in range(labels_c.shape[0]):
            idx = int(batch["idx"][i].detach().cpu().item())
            ex = examples[idx]
            gold_c_idx = int(labels_c[i].detach().cpu().item())
            gold_e_idx = int(labels_e[i].detach().cpu().item())
            if gold_c_idx < 0 or gold_e_idx < 0:
                continue
            gold_c = INV_CLARITY_MAPPING[gold_c_idx]
            gold_e = str(evasion_encoder.inverse_transform([gold_e_idx])[0])
            gold_pair = f"{gold_c} | {gold_e}"

            p_c_true = float(probs_c[i, gold_c_idx].detach().cpu().item())
            p_e_true = float(probs_e[i, gold_e_idx].detach().cpu().item())
            p_joint_raw = float(joint_raw[i, gold_c_idx, gold_e_idx].detach().cpu().item())
            p_joint_norm = float(masked_norm[i, gold_c_idx, gold_e_idx].detach().cpu().item())
            loss_val = float(per_loss[i].detach().cpu().item())

            pc = int(pred_c_cons[i].detach().cpu().item())
            pe = int(pred_e_cons[i].detach().cpu().item())
            pred_c_name = INV_CLARITY_MAPPING[pc]
            pred_e_name = str(evasion_encoder.inverse_transform([pe])[0])
            pred_pair = f"{pred_c_name} | {pred_e_name}"
            correct_c = pred_c_name == gold_c
            correct_e = pred_e_name == gold_e
            correct_joint = correct_c and correct_e
            correct_joint_constrained += int(correct_joint)

            if tracker_constrained is not None:
                tracker_constrained.record(
                    idx=idx,
                    prob_true=p_joint_norm,
                    pred=pred_pair,
                    gold=gold_pair,
                    correct=correct_joint,
                    loss_val=loss_val,
                    clarity_prob_true=p_c_true,
                    evasion_prob_true=p_e_true,
                    joint_prob_raw=p_joint_raw,
                    joint_prob_norm=p_joint_norm,
                    pred_clarity=pred_c_name,
                    pred_evasion=pred_e_name,
                    pred_pair=pred_pair,
                    correct_clarity=int(correct_c),
                    correct_evasion=int(correct_e),
                    illegal_pair=0,
                )

            pc_i = int(pred_c_ind[i].detach().cpu().item())
            pe_i = int(pred_e_ind[i].detach().cpu().item())
            pred_c_ind_name = INV_CLARITY_MAPPING[pc_i]
            pred_e_ind_name = str(evasion_encoder.inverse_transform([pe_i])[0])
            pred_pair_ind = f"{pred_c_ind_name} | {pred_e_ind_name}"
            illegal = (pred_c_ind_name, pred_e_ind_name) not in legal_pairs
            illegal_independent += int(illegal)
            correct_ind = pred_c_ind_name == gold_c and pred_e_ind_name == gold_e
            correct_joint_independent += int(correct_ind)

            if tracker_independent is not None:
                tracker_independent.record(
                    idx=idx,
                    prob_true=p_joint_raw,
                    pred=pred_pair_ind,
                    gold=gold_pair,
                    correct=correct_ind,
                    loss_val=loss_val,
                    clarity_prob_true=p_c_true,
                    evasion_prob_true=p_e_true,
                    joint_prob_raw=p_joint_raw,
                    pred_clarity=pred_c_ind_name,
                    pred_evasion=pred_e_ind_name,
                    pred_pair=pred_pair_ind,
                    correct_clarity=int(pred_c_ind_name == gold_c),
                    correct_evasion=int(pred_e_ind_name == gold_e),
                    illegal_pair=int(illegal),
                )
            total += 1

    return {
        f"{split_name}_joint_acc_constrained": correct_joint_constrained / max(1, total),
        f"{split_name}_joint_acc_independent": correct_joint_independent / max(1, total),
        f"{split_name}_illegal_rate_independent": illegal_independent / max(1, total),
    }


@torch.no_grad()
def evaluate_joint(
    model: JointPairModel,
    loader: DataLoader,
    device: torch.device,
    tracker: CartographyTracker | None,
    examples: list[dict[str, Any]],
    joint_encoder: LabelEncoder,
    split_name: str,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(**model_inputs(batch))
        logits = out["logits_joint"]
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        labels_j = batch["labels_joint"]
        ce = F.cross_entropy(logits, labels_j, reduction="none", ignore_index=-100)
        for i in range(labels_j.shape[0]):
            idx = int(batch["idx"][i].detach().cpu().item())
            gold_j = int(labels_j[i].detach().cpu().item())
            if gold_j < 0:
                continue
            gold_pair = str(joint_encoder.inverse_transform([gold_j])[0])
            pred_j = int(preds[i].detach().cpu().item())
            pred_pair = str(joint_encoder.inverse_transform([pred_j])[0])
            gold_c, gold_e = gold_pair.split(" | ", 1)
            pred_c, pred_e = pred_pair.split(" | ", 1)
            p_true = float(probs[i, gold_j].detach().cpu().item())
            correct_joint = pred_j == gold_j
            correct += int(correct_joint)
            total += 1
            if tracker is not None:
                tracker.record(
                    idx=idx,
                    prob_true=p_true,
                    pred=pred_pair,
                    gold=gold_pair,
                    correct=correct_joint,
                    loss_val=float(ce[i].detach().cpu().item()),
                    joint_prob_norm=p_true,
                    pred_clarity=pred_c,
                    pred_evasion=pred_e,
                    pred_pair=pred_pair,
                    correct_clarity=int(pred_c == gold_c),
                    correct_evasion=int(pred_e == gold_e),
                    illegal_pair=0,
                )
    return {f"{split_name}_joint_acc_joint": correct / max(1, total)}


@torch.no_grad()
def collect_independent_predictions(
    model: IndependentMultitaskModel,
    loader: DataLoader,
    device: torch.device,
    examples: list[dict[str, Any]],
    evasion_encoder: LabelEncoder,
    legal_pairs: set[tuple[str, str]],
    constraint_mask: torch.Tensor,
    decode_mode: str,
) -> pd.DataFrame:
    model.eval()
    rows = []
    mask = constraint_mask.to(device)
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(**model_inputs(batch))
        probs_c = F.softmax(out["logits_clarity"], dim=-1)
        probs_e = F.softmax(out["logits_evasion"], dim=-1)
        if decode_mode == "constrained":
            joint_raw = probs_c.unsqueeze(2) * probs_e.unsqueeze(1)
            masked = joint_raw * mask.unsqueeze(0)
            denom = masked.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            masked_norm = masked / denom
            pred_flat = masked.view(masked.shape[0], -1).argmax(dim=-1)
            pred_c = pred_flat // len(evasion_encoder.classes_)
            pred_e = pred_flat % len(evasion_encoder.classes_)
        else:
            joint_raw = probs_c.unsqueeze(2) * probs_e.unsqueeze(1)
            masked_norm = None
            pred_c = probs_c.argmax(dim=-1)
            pred_e = probs_e.argmax(dim=-1)

        for i in range(batch["idx"].shape[0]):
            idx = int(batch["idx"][i].detach().cpu().item())
            ex = examples[idx]
            gold_c = ex["clarity_label"]
            gold_e = ex["evasion_label"]
            gold_pair = ex["gold_pair"]
            pc = int(pred_c[i].detach().cpu().item())
            pe = int(pred_e[i].detach().cpu().item())
            pred_c_name = INV_CLARITY_MAPPING[pc]
            pred_e_name = str(evasion_encoder.inverse_transform([pe])[0])
            pred_pair = f"{pred_c_name} | {pred_e_name}"
            illegal = (pred_c_name, pred_e_name) not in legal_pairs
            accepted_evasion = json.loads(ex["accepted_evasion_labels"]) if ex["accepted_evasion_labels"] else []
            c_gold_idx = ex["labels_clarity"]
            e_gold_idx = ex["labels_evasion"]
            p_c = float(probs_c[i, c_gold_idx].detach().cpu().item()) if c_gold_idx >= 0 else np.nan
            p_e = float(probs_e[i, e_gold_idx].detach().cpu().item()) if e_gold_idx >= 0 else np.nan
            p_joint_raw = (
                float(joint_raw[i, c_gold_idx, e_gold_idx].detach().cpu().item())
                if c_gold_idx >= 0 and e_gold_idx >= 0
                else np.nan
            )
            p_joint = p_joint_raw
            if decode_mode == "constrained" and masked_norm is not None and c_gold_idx >= 0 and e_gold_idx >= 0:
                p_joint = float(masked_norm[i, c_gold_idx, e_gold_idx].detach().cpu().item())
            row = {
                **{k: ex.get(k, "") for k in prediction_metadata_cols()},
                "decode_mode": decode_mode,
                "pred_clarity": pred_c_name,
                "pred_evasion": pred_e_name,
                "pred_pair": pred_pair,
                "prob_true_clarity": p_c,
                "prob_true_evasion": p_e,
                "prob_true_pair": p_joint,
                "prob_true_pair_raw": p_joint_raw,
                "pred_prob_clarity": float(probs_c[i, pc].detach().cpu().item()),
                "pred_prob_evasion": float(probs_e[i, pe].detach().cpu().item()),
                "is_correct_clarity": pred_c_name == gold_c,
                "is_correct_evasion": pred_e_name == gold_e,
                "is_correct_pair": pred_pair == gold_pair,
                "is_evasion_compatible": pred_e_name in accepted_evasion,
                "illegal_pair": bool(illegal),
            }
            row["failure_type"] = classify_failure(row)
            rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def collect_joint_predictions(
    model: JointPairModel,
    loader: DataLoader,
    device: torch.device,
    examples: list[dict[str, Any]],
    joint_encoder: LabelEncoder,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(**model_inputs(batch))
        probs = F.softmax(out["logits_joint"], dim=-1)
        preds = probs.argmax(dim=-1)
        for i in range(batch["idx"].shape[0]):
            idx = int(batch["idx"][i].detach().cpu().item())
            ex = examples[idx]
            gold_pair = ex["gold_pair"]
            gold_j = ex["labels_joint"]
            pred_j = int(preds[i].detach().cpu().item())
            pred_pair = str(joint_encoder.inverse_transform([pred_j])[0])
            pred_c, pred_e = pred_pair.split(" | ", 1)
            accepted_evasion = json.loads(ex["accepted_evasion_labels"]) if ex["accepted_evasion_labels"] else []
            p_true = float(probs[i, gold_j].detach().cpu().item()) if gold_j >= 0 else np.nan
            row = {
                **{k: ex.get(k, "") for k in prediction_metadata_cols()},
                "decode_mode": "joint",
                "pred_clarity": pred_c,
                "pred_evasion": pred_e,
                "pred_pair": pred_pair,
                "prob_true_pair": p_true,
                "pred_prob_pair": float(probs[i, pred_j].detach().cpu().item()),
                "is_correct_clarity": pred_c == ex["clarity_label"],
                "is_correct_evasion": pred_e == ex["evasion_label"],
                "is_correct_pair": pred_pair == gold_pair,
                "is_evasion_compatible": pred_e in accepted_evasion,
                "illegal_pair": False,
            }
            row["failure_type"] = classify_failure(row)
            rows.append(row)
    return pd.DataFrame(rows)


def prediction_metadata_cols() -> list[str]:
    return [
        "idx",
        "split",
        "dataset_index",
        "title",
        "date",
        "president",
        "url",
        "question_order",
        "interview_question",
        "interview_answer",
        "question",
        "clarity_label",
        "evasion_label",
        "gold_pair",
        "annotator_votes",
        "accepted_evasion_labels",
        "inaudible",
        "multiple_questions",
        "affirmative_questions",
        "question_word_count",
        "answer_word_count",
        "subquestion_word_count",
    ] + ANNOTATOR_COLS


def classify_failure(row: dict[str, Any]) -> str:
    if row.get("illegal_pair"):
        return "illegal_pair"
    c_ok = bool(row.get("is_correct_clarity"))
    e_ok = bool(row.get("is_correct_evasion"))
    if c_ok and e_ok:
        return "correct"
    if c_ok and not e_ok:
        return "evasion_wrong"
    if not c_ok and e_ok:
        return "clarity_wrong"
    return "both_wrong"


def metric_row(pred_df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if pred_df.empty:
        return {"prefix": prefix, "n": 0}
    y_c = pred_df["clarity_label"].astype(str)
    p_c = pred_df["pred_clarity"].astype(str)
    y_e = pred_df["evasion_label"].astype(str)
    p_e = pred_df["pred_evasion"].astype(str)
    y_pair = pred_df["gold_pair"].astype(str)
    p_pair = pred_df["pred_pair"].astype(str)
    return {
        "prefix": prefix,
        "n": int(len(pred_df)),
        "clarity_accuracy": float(accuracy_score(y_c, p_c)),
        "clarity_macro_f1": float(f1_score(y_c, p_c, average="macro", zero_division=0)),
        "evasion_accuracy": float(accuracy_score(y_e, p_e)),
        "evasion_macro_f1": float(f1_score(y_e, p_e, average="macro", zero_division=0)),
        "pair_accuracy": float(accuracy_score(y_pair, p_pair)),
        "pair_macro_f1": float(f1_score(y_pair, p_pair, average="macro", zero_division=0)),
        "evasion_compatible_accuracy": float(pred_df["is_evasion_compatible"].mean()),
        "illegal_pair_rate": float(pred_df["illegal_pair"].mean()),
    }


def export_error_breakdowns(pred_df: pd.DataFrame, prefix: str, res_dir: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    if pred_df.empty:
        return summaries

    metric = metric_row(pred_df, prefix)
    summaries["metrics"] = metric
    pd.DataFrame([metric]).to_csv(res_dir / f"{prefix}_metrics.csv", index=False)

    for label_col, name in [
        ("clarity_label", "clarity"),
        ("evasion_label", "evasion"),
        ("gold_pair", "pair"),
        ("failure_type", "failure_type"),
    ]:
        tab = (
            pred_df.groupby(label_col)
            .agg(
                n=("idx", "count"),
                pair_acc=("is_correct_pair", "mean"),
                clarity_acc=("is_correct_clarity", "mean"),
                evasion_acc=("is_correct_evasion", "mean"),
                compatible_acc=("is_evasion_compatible", "mean"),
                illegal_rate=("illegal_pair", "mean"),
                mean_prob_true=("prob_true_pair", "mean"),
            )
            .reset_index()
            .sort_values("n", ascending=False)
        )
        tab.to_csv(res_dir / f"{prefix}_error_by_{name}.csv", index=False)
        summaries[f"by_{name}"] = tab.to_dict(orient="records")

    return summaries


def trainable_steps(loader: DataLoader, epochs: int, grad_accumulation: int) -> int:
    return max(1, math.ceil(len(loader) / grad_accumulation) * epochs)


def make_optimizer_scheduler(
    model: nn.Module, loader: DataLoader, cfg: RunConfig
) -> tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps = trainable_steps(loader, cfg.epochs, cfg.grad_accumulation)
    warmup = int(steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, steps)
    return optimizer, scheduler


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json_dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def atomic_torch_save(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    tracker_constrained: CartographyTracker | None = None,
    tracker_independent: CartographyTracker | None = None,
    tracker_joint: CartographyTracker | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "tracker_constrained": tracker_constrained.to_state() if tracker_constrained is not None else None,
            "tracker_independent": tracker_independent.to_state() if tracker_independent is not None else None,
            "tracker_joint": tracker_joint.to_state() if tracker_joint is not None else None,
            "metadata": metadata or {},
            "saved_at": now(),
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    expected_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if expected_metadata:
        metadata = ckpt.get("metadata", {})
        for key, expected_value in expected_metadata.items():
            if metadata.get(key) != expected_value:
                return None
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if ckpt.get("scheduler_state") is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def latest_complete_epoch_from_tracker(tracker: CartographyTracker, n_examples: int) -> int:
    if not tracker.epoch_probs:
        return 0
    lengths = [len(tracker.epoch_probs.get(i, [])) for i in range(n_examples)]
    return int(min(lengths)) if lengths else 0


def carto_output_matches(path: Path, expected_n: int, expected_epochs: int) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, usecols=lambda c: c in {"idx", "epoch_probs"})
        if len(df) != expected_n:
            return False
        if "epoch_probs" not in df.columns:
            return False
        lengths = df["epoch_probs"].map(lambda x: len(json.loads(x)) if isinstance(x, str) else 0)
        return bool((lengths >= expected_epochs).all())
    except Exception:
        return False


def prediction_output_matches(path: Path, expected_n: int) -> bool:
    if not path.exists():
        return False
    try:
        return len(pd.read_csv(path, usecols=["idx"])) == expected_n
    except Exception:
        return False


def export_epoch_artifacts(
    tracker: CartographyTracker,
    examples: list[dict[str, Any]],
    prefix: str,
    seed: int,
    epoch: int,
    partial_dir: Path,
) -> pd.DataFrame:
    df = tracker.compute_metrics()
    df = assign_regions(df)
    df = attach_metadata(df, examples)
    atomic_to_csv(df, partial_dir / f"{prefix}_carto_seed{seed}_epoch{epoch:02d}.csv")
    atomic_to_csv(df, partial_dir / f"{prefix}_carto_seed{seed}_latest.csv")
    return df


def append_epoch_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)


def write_crash_report(output_dir: Path, exc: BaseException) -> None:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "failed_at": now(),
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
    }
    atomic_json_dump(report, logs_dir / "crash_report.json")


def entropy_from_votes(votes_json: str) -> float:
    try:
        votes = json.loads(votes_json)
    except Exception:
        votes = []
    votes = [v for v in votes if v]
    if not votes:
        return 0.0
    counts = Counter(votes)
    total = sum(counts.values())
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))


def export_dataset_audit(train_examples: list[dict[str, Any]], test_examples: list[dict[str, Any]], res_dir: Path) -> None:
    for split, examples in [("train", train_examples), ("test", test_examples)]:
        df = examples_metadata(examples)
        df["annotator_entropy"] = df["annotator_votes"].map(entropy_from_votes)
        df["accepted_evasion_n"] = df["accepted_evasion_labels"].map(lambda x: len(json.loads(x)) if x else 0)

        for col, name in [
            ("clarity_label", "clarity"),
            ("evasion_label", "evasion"),
            ("gold_pair", "pair"),
            ("president", "president"),
        ]:
            tab = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="n")
            tab["pct"] = 100.0 * tab["n"] / max(1, len(df))
            atomic_to_csv(tab, res_dir / f"qevasion_{split}_{name}_distribution.csv")

        meta = (
            df.groupby(["clarity_label", "evasion_label"], dropna=False)
            .agg(
                n=("idx", "count"),
                question_words_mean=("question_word_count", "mean"),
                answer_words_mean=("answer_word_count", "mean"),
                multiple_questions_pct=("multiple_questions", "mean"),
                affirmative_questions_pct=("affirmative_questions", "mean"),
                inaudible_pct=("inaudible", "mean"),
                annotator_entropy_mean=("annotator_entropy", "mean"),
                accepted_evasion_n_mean=("accepted_evasion_n", "mean"),
            )
            .reset_index()
        )
        for col in ["multiple_questions_pct", "affirmative_questions_pct", "inaudible_pct"]:
            meta[col] = meta[col] * 100.0
        atomic_to_csv(meta, res_dir / f"qevasion_{split}_label_pair_metadata_audit.csv")


def build_stability_df(seed_dfs: dict[int, pd.DataFrame], seeds: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    ref = seed_dfs[seeds[0]].sort_values("idx").reset_index(drop=True)
    n = len(ref)
    conf_matrix = np.zeros((n, len(seeds)))
    var_matrix = np.zeros((n, len(seeds)))
    correct_matrix = np.zeros((n, len(seeds)))
    forget_matrix = np.zeros((n, len(seeds)))
    region_lists = []

    for si, seed in enumerate(seeds):
        df_s = seed_dfs[seed].sort_values("idx").reset_index(drop=True)
        conf_matrix[:, si] = df_s["confidence"].values
        var_matrix[:, si] = df_s["variability"].values
        correct_matrix[:, si] = df_s["correctness"].values
        forget_matrix[:, si] = df_s["forgetfulness"].values
        region_lists.append(df_s["region"].astype(str).values)

    data: dict[str, Any] = {
        "idx": ref["idx"].values,
        "gold_pair": ref["gold_pair"].values,
        "clarity_label": ref["clarity_label"].values,
        "evasion_label": ref["evasion_label"].values,
        "interview_question": ref["interview_question"].values,
        "interview_answer": ref["interview_answer"].values,
        "question": ref["question"].values,
        "president": ref["president"].values,
        "inaudible": ref["inaudible"].values,
        "multiple_questions": ref["multiple_questions"].values,
        "affirmative_questions": ref["affirmative_questions"].values,
        "question_word_count": ref["question_word_count"].values,
        "answer_word_count": ref["answer_word_count"].values,
        "subquestion_word_count": ref["subquestion_word_count"].values,
        "conf_mean": conf_matrix.mean(axis=1),
        "conf_std": conf_matrix.std(axis=1),
        "var_mean": var_matrix.mean(axis=1),
        "var_std": var_matrix.std(axis=1),
        "correctness_mean": correct_matrix.mean(axis=1),
        "forgetfulness_mean": forget_matrix.mean(axis=1),
    }
    for si, seed in enumerate(seeds):
        data[f"conf_seed{seed}"] = conf_matrix[:, si]
        data[f"var_seed{seed}"] = var_matrix[:, si]
        data[f"correctness_seed{seed}"] = correct_matrix[:, si]
        data[f"forgetfulness_seed{seed}"] = forget_matrix[:, si]
        data[f"region_seed{seed}"] = region_lists[si]
        data[f"region_classic_seed{seed}"] = seed_dfs[seed].sort_values("idx")["region_classic"].values
        data[f"region_percentile_seed{seed}"] = seed_dfs[seed].sort_values("idx")["region_percentile"].values
        data[f"region_assignment_seed{seed}"] = seed_dfs[seed].sort_values("idx")["region_assignment"].values

    region_counts = np.zeros((n, len(REGION_NAMES)))
    for si in range(len(seeds)):
        for ri, region in enumerate(REGION_NAMES):
            region_counts[:, ri] += (region_lists[si] == region).astype(int)
    majority_idx = region_counts.argmax(axis=1)
    max_agreement = region_counts.max(axis=1)
    data["majority_region"] = np.asarray(REGION_NAMES)[majority_idx]
    data["region_agreement_count"] = max_agreement.astype(int)
    data["region_agreement"] = max_agreement / max(1, len(seeds))
    data["is_stable"] = max_agreement == len(seeds)
    data["is_chronically_unstable"] = max_agreement <= 3 if len(seeds) >= 5 else max_agreement < len(seeds)
    data["region_pattern"] = [",".join(region_lists[si][i] for si in range(len(seeds))) for i in range(n)]

    stats = compute_cross_seed_stats(conf_matrix, var_matrix, region_lists, seeds)
    assignment_modes = []
    for seed in seeds:
        mode_counts = seed_dfs[seed]["region_assignment"].value_counts().to_dict()
        assignment_modes.append({"seed": seed, "mode_counts": {str(k): int(v) for k, v in mode_counts.items()}})
    stats["region_assignment_modes"] = assignment_modes
    return pd.DataFrame(data), stats


def compute_cross_seed_stats(
    conf_matrix: np.ndarray, var_matrix: np.ndarray, region_lists: list[np.ndarray], seeds: list[int]
) -> dict[str, Any]:
    n_seeds = len(seeds)
    stats: dict[str, Any] = {
        "n_seeds": n_seeds,
        "seeds": seeds,
        "n_examples": int(conf_matrix.shape[0]),
    }
    conf_corrs = []
    var_corrs = []
    tau_vals = []
    if n_seeds >= 2:
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                conf_corrs.append(float(np.corrcoef(conf_matrix[:, i], conf_matrix[:, j])[0, 1]))
                var_corrs.append(float(np.corrcoef(var_matrix[:, i], var_matrix[:, j])[0, 1]))
                tau, _ = scipy_stats.kendalltau(conf_matrix[:, i], conf_matrix[:, j])
                tau_vals.append(float(tau))
    stats["conf_r_all"] = conf_corrs
    stats["var_r_all"] = var_corrs
    stats["kendall_tau_all"] = tau_vals
    stats["conf_r_mean"] = float(np.nanmean(conf_corrs)) if conf_corrs else None
    stats["var_r_mean"] = float(np.nanmean(var_corrs)) if var_corrs else None
    stats["kendall_tau_mean"] = float(np.nanmean(tau_vals)) if tau_vals else None

    n_ex = conf_matrix.shape[0]
    region_counts = np.zeros((n_ex, len(REGION_NAMES)))
    for si in range(n_seeds):
        for ri, region in enumerate(REGION_NAMES):
            region_counts[:, ri] += (region_lists[si] == region).astype(int)
    max_agreement = region_counts.max(axis=1)
    stable_count = int((max_agreement == n_seeds).sum())
    chronically_unstable = int((max_agreement <= 3).sum()) if n_seeds >= 5 else int((max_agreement < n_seeds).sum())

    trans_counts = np.zeros((len(REGION_NAMES), len(REGION_NAMES)))
    if n_seeds >= 2:
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                for r_from_i, r_from in enumerate(REGION_NAMES):
                    for r_to_i, r_to in enumerate(REGION_NAMES):
                        trans_counts[r_from_i, r_to_i] += np.sum(
                            (region_lists[i] == r_from) & (region_lists[j] == r_to)
                        )
    trans_probs = trans_counts / (trans_counts.sum(axis=1, keepdims=True) + 1e-12)
    stats.update(
        {
            "n_stable": stable_count,
            "pct_stable": float(100.0 * stable_count / max(1, n_ex)),
            "n_chronically_unstable": chronically_unstable,
            "pct_chronically_unstable": float(100.0 * chronically_unstable / max(1, n_ex)),
            "transition_counts": trans_counts.tolist(),
            "transition_matrix": trans_probs.tolist(),
        }
    )
    return stats


def export_stability(stability_df: pd.DataFrame, prefix: str, res_dir: Path) -> None:
    stability_df.to_csv(res_dir / f"{prefix}_stability.csv", index=False)
    stability_df[stability_df["is_stable"]].to_csv(res_dir / f"{prefix}_stable_examples.csv", index=False)
    stability_df[stability_df["is_chronically_unstable"]].to_csv(
        res_dir / f"{prefix}_unstable_examples.csv", index=False
    )


def crosstab_counts_and_pct(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    counts = pd.crosstab(df[row], df[col]).reindex(REGION_NAMES, fill_value=0)
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    out = counts.add_suffix("_n").join(pct.add_suffix("_pct"))
    out.insert(0, row, out.index)
    return out.reset_index(drop=True)


def export_reviewer_breakdowns(stability_df: pd.DataFrame, prefix: str, res_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = len(stability_df)
    region_counts = (
        stability_df["majority_region"]
        .value_counts()
        .reindex(REGION_NAMES, fill_value=0)
        .rename_axis("region")
        .reset_index(name="n")
    )
    region_counts["pct"] = 100.0 * region_counts["n"] / max(1, n)
    region_counts.to_csv(res_dir / f"{prefix}_region_counts.csv", index=False)
    out["region_counts"] = region_counts.to_dict(orient="records")

    for col, label in [("clarity_label", "clarity"), ("evasion_label", "evasion"), ("gold_pair", "pair")]:
        tab = crosstab_counts_and_pct(stability_df, "majority_region", col)
        tab.to_csv(res_dir / f"{prefix}_region_by_{label}.csv", index=False)
        out[f"region_by_{label}"] = tab.to_dict(orient="records")

        reverse = pd.crosstab(stability_df[col], stability_df["majority_region"]).reindex(
            columns=REGION_NAMES, fill_value=0
        )
        reverse["total"] = reverse.sum(axis=1)
        for region in REGION_NAMES:
            reverse[f"{region}_pct_within_label"] = 100.0 * reverse[region] / reverse["total"].replace(0, np.nan)
        reverse = reverse.fillna(0).reset_index()
        reverse.to_csv(res_dir / f"{prefix}_{label}_to_region_distribution.csv", index=False)

    hard = stability_df[stability_df["majority_region"] == "hard"]
    overall_counts = stability_df["evasion_label"].value_counts()
    hard_counts = hard["evasion_label"].value_counts()
    rows = []
    for label in sorted(overall_counts.index):
        overall_pct = overall_counts[label] / max(1, len(stability_df))
        hard_pct = hard_counts.get(label, 0) / max(1, len(hard))
        rows.append(
            {
                "evasion_label": label,
                "overall_n": int(overall_counts[label]),
                "hard_n": int(hard_counts.get(label, 0)),
                "overall_pct": float(100.0 * overall_pct),
                "hard_pct": float(100.0 * hard_pct),
                "hard_enrichment_ratio": float(hard_pct / overall_pct) if overall_pct > 0 else np.nan,
            }
        )
    enrichment = pd.DataFrame(rows).sort_values("hard_enrichment_ratio", ascending=False)
    enrichment.to_csv(res_dir / f"{prefix}_hard_evasion_enrichment.csv", index=False)
    out["hard_evasion_enrichment"] = enrichment.to_dict(orient="records")

    meta_rows = []
    for region, grp in stability_df.groupby("majority_region"):
        meta_rows.append(
            {
                "region": region,
                "n": int(len(grp)),
                "multiple_questions_pct": float(100.0 * grp["multiple_questions"].mean()),
                "affirmative_questions_pct": float(100.0 * grp["affirmative_questions"].mean()),
                "inaudible_pct": float(100.0 * grp["inaudible"].mean()),
                "question_words_mean": float(grp["question_word_count"].mean()),
                "answer_words_mean": float(grp["answer_word_count"].mean()),
                "subquestion_words_mean": float(grp["subquestion_word_count"].mean()),
                "confidence_mean": float(grp["conf_mean"].mean()),
                "variability_mean": float(grp["var_mean"].mean()),
            }
        )
    meta_df = pd.DataFrame(meta_rows).sort_values("region")
    meta_df.to_csv(res_dir / f"{prefix}_metadata_by_region.csv", index=False)
    out["metadata_by_region"] = meta_df.to_dict(orient="records")
    return out


def select_balanced_by_group(df: pd.DataFrame, group_col: str, n_per_group: int, sort_cols: list[str]) -> pd.DataFrame:
    parts = []
    for _, grp in df.groupby(group_col, dropna=False):
        grp_sorted = grp.sort_values(sort_cols, ascending=[False if c.endswith("_mean") else True for c in sort_cols])
        parts.append(grp_sorted.head(n_per_group))
    if not parts:
        return df.head(0)
    return pd.concat(parts, ignore_index=True)


def export_qualitative_examples(
    stability_df: pd.DataFrame,
    train_pred_seed: pd.DataFrame,
    test_pred_seed: pd.DataFrame,
    prefix: str,
    examples_dir: Path,
    limit: int,
) -> dict[str, Any]:
    examples_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Any] = {}

    def save(name: str, df: pd.DataFrame) -> pd.DataFrame:
        readable = df.copy()
        for text_col in ["interview_question", "interview_answer", "question"]:
            if text_col in readable.columns:
                readable[text_col] = readable[text_col].map(lambda x: compact_text(str(x), 500))
        path = examples_dir / f"{prefix}_{name}.csv"
        readable.to_csv(path, index=False)
        exported[name] = {"path": str(path), "n": int(len(readable))}
        return readable

    base_cols = [
        "idx",
        "majority_region",
        "region_agreement",
        "clarity_label",
        "evasion_label",
        "gold_pair",
        "conf_mean",
        "conf_std",
        "var_mean",
        "correctness_mean",
        "forgetfulness_mean",
        "interview_question",
        "interview_answer",
        "question",
        "multiple_questions",
        "affirmative_questions",
        "inaudible",
    ]
    base_cols = [c for c in base_cols if c in stability_df.columns]

    by_region_parts = []
    for region in REGION_NAMES:
        grp = stability_df[stability_df["majority_region"] == region].copy()
        if region == "hard":
            grp = grp.sort_values(["conf_mean", "correctness_mean", "var_mean"], ascending=[True, True, False])
        elif region == "ambiguous":
            grp = grp.sort_values(["var_mean", "conf_std"], ascending=[False, False])
        else:
            grp = grp.sort_values(["conf_mean", "correctness_mean"], ascending=[False, False])
        by_region_parts.append(grp.head(limit))
    save("examples_by_region", pd.concat(by_region_parts, ignore_index=True)[base_cols])

    hard = stability_df[stability_df["majority_region"] == "hard"].copy()
    save(
        "hard_examples_by_evasion",
        hard.sort_values(["evasion_label", "conf_mean", "correctness_mean"]).groupby("evasion_label").head(limit)[
            base_cols
        ],
    )
    save(
        "ambiguous_high_variability",
        stability_df[stability_df["majority_region"] == "ambiguous"]
        .sort_values(["var_mean", "conf_std"], ascending=False)
        .head(limit * 3)[base_cols],
    )
    save(
        "chronically_unstable",
        stability_df[stability_df["is_chronically_unstable"]]
        .sort_values(["region_agreement", "conf_std"], ascending=[True, False])
        .head(limit * 3)[base_cols],
    )

    pred_cols = [
        "idx",
        "split",
        "clarity_label",
        "evasion_label",
        "gold_pair",
        "pred_clarity",
        "pred_evasion",
        "pred_pair",
        "prob_true_pair",
        "failure_type",
        "interview_question",
        "interview_answer",
        "question",
    ]
    pred_cols = [c for c in pred_cols if c in train_pred_seed.columns]
    wrong_train = train_pred_seed[~train_pred_seed["is_correct_pair"]].copy()
    save(
        "train_high_confidence_wrong",
        wrong_train.sort_values("prob_true_pair", ascending=False).head(limit * 2)[pred_cols],
    )
    wrong_test = test_pred_seed[~test_pred_seed["is_correct_pair"]].copy()
    save(
        "test_high_confidence_wrong",
        wrong_test.sort_values("prob_true_pair", ascending=False).head(limit * 2)[pred_cols],
    )
    confusion = wrong_test.sort_values("prob_true_pair", ascending=False).groupby(["gold_pair", "pred_pair"]).head(3)
    save("test_confusion_examples", confusion.head(limit * 4)[pred_cols])

    json_records = []
    for _, row in pd.concat(
        [
            hard.sort_values(["conf_mean", "var_mean"], ascending=[True, False]).head(limit),
            stability_df[stability_df["majority_region"] == "ambiguous"]
            .sort_values("var_mean", ascending=False)
            .head(limit),
        ]
    ).iterrows():
        json_records.append(
            {
                "idx": int(row["idx"]),
                "region": row["majority_region"],
                "region_agreement": float(row["region_agreement"]),
                "gold_clarity": row["clarity_label"],
                "gold_evasion": row["evasion_label"],
                "gold_pair": row["gold_pair"],
                "confidence": float(row["conf_mean"]),
                "variability": float(row["var_mean"]),
                "question": compact_text(row.get("interview_question", ""), 900),
                "answer": compact_text(row.get("interview_answer", ""), 1200),
                "subquestion": compact_text(row.get("question", ""), 500),
            }
        )
    json_path = examples_dir / f"{prefix}_human_readable_examples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2, ensure_ascii=False)
    exported["human_readable_examples_json"] = {"path": str(json_path), "n": len(json_records)}

    md_path = examples_dir / f"{prefix}_qualitative_examples.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# QEvasion Qualitative Cartography Examples\n\n")
        for rec in json_records[: min(30, len(json_records))]:
            f.write(f"## Example {rec['idx']} - {rec['region']}\n\n")
            f.write(f"- Gold: {rec['gold_pair']}\n")
            f.write(f"- Confidence: {rec['confidence']:.4f}\n")
            f.write(f"- Variability: {rec['variability']:.4f}\n")
            f.write(f"- Agreement: {rec['region_agreement']:.2f}\n\n")
            f.write(f"**Question:** {rec['question']}\n\n")
            f.write(f"**Answer:** {rec['answer']}\n\n")
            if rec["subquestion"]:
                f.write(f"**Subquestion:** {rec['subquestion']}\n\n")
    exported["qualitative_examples_markdown"] = {"path": str(md_path), "n": min(30, len(json_records))}
    return exported


def plot_cartography(stability_df: pd.DataFrame, prefix: str, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for region in REGION_NAMES:
        mask = stability_df["majority_region"] == region
        ax.scatter(
            stability_df.loc[mask, "var_mean"],
            stability_df.loc[mask, "conf_mean"],
            s=18,
            alpha=0.45,
            color=REGION_COLORS[region],
            label=f"{region} (n={int(mask.sum())})",
        )
    ax.set_xlabel("Variability (mean across seeds)")
    ax.set_ylabel("Confidence (mean across seeds)")
    ax.set_title("QEvasion Cartography - Constrained Multitask")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    fig.savefig(plot_dir / f"{prefix}_cartography.png")
    plt.close(fig)


def plot_transition_heatmap(stats: dict[str, Any], prefix: str, plot_dir: Path) -> None:
    matrix = np.asarray(stats.get("transition_matrix", np.zeros((3, 3))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=REGION_NAMES,
        yticklabels=REGION_NAMES,
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("To region")
    ax.set_ylabel("From region")
    ax.set_title("Cross-seed Region Transitions")
    fig.savefig(plot_dir / f"{prefix}_region_transitions.png")
    plt.close(fig)


def plot_region_bars(stability_df: pd.DataFrame, prefix: str, plot_dir: Path) -> None:
    for col, label in [("clarity_label", "clarity"), ("evasion_label", "evasion")]:
        tab = pd.crosstab(stability_df["majority_region"], stability_df[col]).reindex(REGION_NAMES, fill_value=0)
        tab_pct = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        ax = tab_pct.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab20")
        ax.set_ylabel("Within-region share")
        ax.set_title(f"Region by {label} label")
        ax.legend(title=label, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{prefix}_region_by_{label}.png")
        plt.close(fig)


def plot_hard_enrichment(enrichment_csv: Path, prefix: str, plot_dir: Path) -> None:
    if not enrichment_csv.exists():
        return
    df = pd.read_csv(enrichment_csv).sort_values("hard_enrichment_ratio", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["evasion_label"], df["hard_enrichment_ratio"], color="#d62728", alpha=0.8)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Hard-region enrichment ratio")
    ax.set_title("Which Evasion Labels Are Overrepresented Among Hard Examples")
    fig.savefig(plot_dir / f"{prefix}_hard_evasion_enrichment.png")
    plt.close(fig)


def plot_confusion(pred_df: pd.DataFrame, prefix: str, plot_dir: Path) -> None:
    specs = [
        ("clarity_label", "pred_clarity", "clarity"),
        ("evasion_label", "pred_evasion", "evasion"),
        ("gold_pair", "pred_pair", "pair"),
    ]
    for true_col, pred_col, label in specs:
        labels = sorted(set(pred_df[true_col].dropna().astype(str)) | set(pred_df[pred_col].dropna().astype(str)))
        if not labels:
            continue
        cm = confusion_matrix(pred_df[true_col].astype(str), pred_df[pred_col].astype(str), labels=labels)
        size = max(6, min(14, 0.5 * len(labels) + 4))
        fig, ax = plt.subplots(figsize=(size, size))
        sns.heatmap(cm, annot=len(labels) <= 12, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(f"{label.capitalize()} confusion")
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
        fig.savefig(plot_dir / f"{prefix}_confusion_{label}.png")
        plt.close(fig)


def parse_series(value: Any) -> list[float]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [float(x) for x in parsed]
        except Exception:
            return []
    return []


def plot_epoch_snapshots(carto_df: pd.DataFrame, prefix: str, plot_dir: Path) -> None:
    if carto_df.empty or "epoch_probs" not in carto_df.columns:
        return
    series = carto_df["epoch_probs"].map(parse_series)
    n_epochs = max((len(x) for x in series), default=0)
    if n_epochs == 0:
        return
    show = sorted(set([1, min(2, n_epochs), max(1, n_epochs // 2), n_epochs]))
    fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 4.5), squeeze=False)
    for ax, epoch in zip(axes[0], show):
        rows = []
        for i, vals in enumerate(series):
            vals = vals[:epoch]
            if not vals:
                continue
            rows.append(
                {
                    "confidence": float(np.mean(vals)),
                    "variability": float(np.std(vals)),
                    "region": carto_df.iloc[i]["region"],
                }
            )
        snap = pd.DataFrame(rows)
        for region in REGION_NAMES:
            mask = snap["region"] == region
            ax.scatter(
                snap.loc[mask, "variability"],
                snap.loc[mask, "confidence"],
                s=12,
                alpha=0.4,
                color=REGION_COLORS[region],
                label=region,
            )
        ax.set_title(f"Epoch {epoch}")
        ax.set_xlabel("Variability")
        ax.set_ylabel("Confidence")
        ax.set_ylim(-0.03, 1.03)
    axes[0, 0].legend()
    fig.suptitle("Per-epoch Cartography Snapshots")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{prefix}_epoch_snapshots.png")
    plt.close(fig)


def plot_confidence_trajectories(carto_df: pd.DataFrame, prefix: str, plot_dir: Path, n_show: int = 60) -> None:
    if carto_df.empty or "epoch_probs" not in carto_df.columns:
        return
    sample = []
    for region in REGION_NAMES:
        grp = carto_df[carto_df["region"] == region].sort_values("variability", ascending=False).head(
            max(1, n_show // 3)
        )
        sample.append(grp)
    df = pd.concat(sample, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        vals = parse_series(row["epoch_probs"])
        if vals:
            ax.plot(range(1, len(vals) + 1), vals, alpha=0.35, color=REGION_COLORS.get(row["region"], "gray"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Probability assigned to gold pair")
    ax.set_title("Representative Confidence Trajectories")
    fig.savefig(plot_dir / f"{prefix}_confidence_trajectories.png")
    plt.close(fig)


def plot_forgetfulness(carto_df: pd.DataFrame, prefix: str, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    max_forget = int(carto_df["forgetfulness"].max()) if len(carto_df) else 0
    bins = np.arange(0, max_forget + 2) - 0.5
    ax.hist(carto_df["forgetfulness"], bins=bins, color="#9467bd", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Forgetfulness")
    ax.set_ylabel("Examples")
    ax.set_title("Forgetfulness Distribution")
    fig.savefig(plot_dir / f"{prefix}_forgetfulness.png")
    plt.close(fig)


def plot_metadata(stability_df: pd.DataFrame, prefix: str, plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bool_cols = ["multiple_questions", "affirmative_questions", "inaudible"]
    bool_rows = []
    for region, grp in stability_df.groupby("majority_region"):
        for col in bool_cols:
            bool_rows.append({"region": region, "feature": col, "pct": 100.0 * grp[col].mean()})
    bool_df = pd.DataFrame(bool_rows)
    sns.barplot(data=bool_df, x="region", y="pct", hue="feature", order=REGION_NAMES, ax=axes[0])
    axes[0].set_ylabel("% examples")
    axes[0].set_title("Metadata Flags by Region")

    len_rows = []
    for region, grp in stability_df.groupby("majority_region"):
        for col, label in [("question_word_count", "question"), ("answer_word_count", "answer")]:
            len_rows.append({"region": region, "text": label, "mean_words": grp[col].mean()})
    len_df = pd.DataFrame(len_rows)
    sns.barplot(data=len_df, x="region", y="mean_words", hue="text", order=REGION_NAMES, ax=axes[1])
    axes[1].set_ylabel("Mean words")
    axes[1].set_title("Length by Region")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{prefix}_metadata_vs_difficulty.png")
    plt.close(fig)


def export_plots(
    stability_df: pd.DataFrame,
    stats: dict[str, Any],
    seed42_carto: pd.DataFrame,
    seed42_test_pred: pd.DataFrame,
    prefix: str,
    res_dir: Path,
    plot_dir: Path,
) -> None:
    plot_cartography(stability_df, prefix, plot_dir)
    plot_transition_heatmap(stats, prefix, plot_dir)
    plot_region_bars(stability_df, prefix, plot_dir)
    plot_hard_enrichment(res_dir / f"{prefix}_hard_evasion_enrichment.csv", prefix, plot_dir)
    plot_confusion(seed42_test_pred, f"{prefix}_test_seed42", plot_dir)
    plot_epoch_snapshots(seed42_carto, f"{prefix}_seed42", plot_dir)
    plot_confidence_trajectories(seed42_carto, f"{prefix}_seed42", plot_dir)
    plot_forgetfulness(seed42_carto, f"{prefix}_seed42", plot_dir)
    plot_metadata(stability_df, prefix, plot_dir)


def save_label_maps(
    res_dir: Path,
    evasion_encoder: LabelEncoder,
    joint_encoder: LabelEncoder,
    legal_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    label_maps = {
        "clarity_mapping": CLARITY_MAPPING,
        "evasion_labels": {int(i): str(label) for i, label in enumerate(evasion_encoder.classes_)},
        "joint_labels": {int(i): str(label) for i, label in enumerate(joint_encoder.classes_)},
        "legal_pairs": [{"clarity": c, "evasion": e, "pair": f"{c} | {e}"} for c, e in legal_pairs],
    }
    with open(res_dir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2, ensure_ascii=False)
    return label_maps


def write_run_manifest(cfg: RunConfig, res_dir: Path, device: torch.device) -> None:
    manifest = {
        "started_at": now(),
        "config": {
            "dataset_name": cfg.dataset_name,
            "model_name": cfg.model_name,
            "seeds": cfg.seeds,
            "ablation_seeds": cfg.ablation_seeds,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "grad_accumulation": cfg.grad_accumulation,
            "learning_rate": cfg.learning_rate,
            "max_length": cfg.max_length,
            "dry_run": cfg.dry_run,
            "large_one_run": cfg.large_one_run,
            "resume": cfg.resume,
            "save_checkpoints": cfg.save_checkpoints,
            "save_epoch_predictions": cfg.save_epoch_predictions,
        },
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        manifest["gpu"] = {"name": props.name, "total_memory_gb": props.total_memory / 1e9}
    with open(res_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)

    res_dir = cfg.output_dir / "results"
    plot_dir = cfg.output_dir / "plots"
    examples_dir = cfg.output_dir / "examples"
    logs_dir = cfg.output_dir / "logs"
    partial_dir = cfg.output_dir / "partial"
    checkpoint_dir = cfg.output_dir / "checkpoints"
    for d in [res_dir, plot_dir, examples_dir, logs_dir, partial_dir, checkpoint_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    write_run_manifest(cfg, res_dir, device)

    t0 = time.time()
    print("=" * 80, flush=True)
    print("SEMEVAL/QEVASION CARTOGRAPHY", flush=True)
    print("=" * 80, flush=True)
    print(f"Time: {now()}", flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)", flush=True)
    print(f"Model: {cfg.model_name}", flush=True)
    print(f"Output: {cfg.output_dir}", flush=True)

    print("\nLoading dataset and tokenizer...", flush=True)
    raw_dataset = load_dataset(cfg.dataset_name)
    unique_evasion, legal_pairs_list = collect_label_space(raw_dataset)
    legal_pairs = set(legal_pairs_list)
    evasion_encoder = LabelEncoder().fit(unique_evasion)
    joint_labels = [f"{c} | {e}" for c, e in legal_pairs_list]
    joint_encoder = LabelEncoder().fit(joint_labels)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_examples = prepare_examples(raw_dataset, "train", evasion_encoder, joint_encoder, legal_pairs)
    test_examples = prepare_examples(raw_dataset, "test", evasion_encoder, joint_encoder, legal_pairs)
    train_examples = stratified_subset(train_examples, cfg.subset_train, seed=42)
    test_examples = stratified_subset(test_examples, cfg.subset_test, seed=42)
    print(f"Train examples: {len(train_examples)}", flush=True)
    print(f"Test examples: {len(test_examples)}", flush=True)
    print(f"Clarity labels: {CLARITY_MAPPING}", flush=True)
    print(f"Evasion labels: {list(evasion_encoder.classes_)}", flush=True)
    print(f"Legal pairs: {len(legal_pairs_list)}", flush=True)

    label_maps = save_label_maps(res_dir, evasion_encoder, joint_encoder, legal_pairs_list)
    atomic_to_csv(examples_metadata(train_examples), res_dir / "qevasion_train_metadata.csv")
    atomic_to_csv(examples_metadata(test_examples), res_dir / "qevasion_test_metadata.csv")
    export_dataset_audit(train_examples, test_examples, res_dir)

    y_c = np.asarray([ex["labels_clarity"] for ex in train_examples])
    y_e = np.asarray([ex["labels_evasion"] for ex in train_examples])
    y_j = np.asarray([ex["labels_joint"] for ex in train_examples])
    partial_idx = None
    if "Partial/half-answer" in set(evasion_encoder.classes_):
        partial_idx = int(evasion_encoder.transform(["Partial/half-answer"])[0])
    clarity_weights = balanced_weights(y_c, 3).to(device)
    evasion_weights = balanced_weights(y_e, len(evasion_encoder.classes_), partial_idx).to(device)
    joint_weights = balanced_weights(y_j, len(joint_encoder.classes_)).to(device)
    for i, label in enumerate(joint_encoder.classes_):
        if "Partial/half-answer" in label:
            joint_weights[i] *= 2.5

    constraint_mask = torch.zeros((3, len(evasion_encoder.classes_)), dtype=torch.float32)
    for c, e in legal_pairs_list:
        constraint_mask[CLARITY_MAPPING[c], int(evasion_encoder.transform([e])[0])] = 1.0

    train_ds = QEDataset(train_examples, tokenizer, cfg.max_length)
    test_ds = QEDataset(test_examples, tokenizer, cfg.max_length)
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_eval_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    constrained_seed_dfs: dict[int, pd.DataFrame] = {}
    independent_seed_dfs: dict[int, pd.DataFrame] = {}
    metrics_rows: list[dict[str, Any]] = []
    seed42_train_pred = pd.DataFrame()
    seed42_test_pred = pd.DataFrame()

    print("\nTraining independent multitask models for constrained cartography...", flush=True)
    for seed in cfg.seeds:
        set_seed(seed)
        free_gpu()
        print(f"\n[{now()}] Seed {seed} independent-head training", flush=True)
        final_carto_path = res_dir / f"qevasion_constrained_carto_seed{seed}.csv"
        final_train_pred_path = res_dir / f"qevasion_constrained_train_predictions_seed{seed}.csv"
        final_test_pred_path = res_dir / f"qevasion_constrained_test_predictions_seed{seed}.csv"
        if (
            cfg.resume
            and carto_output_matches(final_carto_path, len(train_examples), cfg.epochs)
            and prediction_output_matches(final_train_pred_path, len(train_examples))
            and prediction_output_matches(final_test_pred_path, len(test_examples))
        ):
            print(f"  Found complete constrained seed {seed}; loading existing outputs.", flush=True)
            constrained_seed_dfs[seed] = pd.read_csv(final_carto_path)
            train_pred_con_existing = pd.read_csv(final_train_pred_path)
            test_pred_con_existing = pd.read_csv(final_test_pred_path)
            metrics_rows.append(metric_row(train_pred_con_existing, f"constrained_train_seed{seed}"))
            metrics_rows.append(metric_row(test_pred_con_existing, f"constrained_test_seed{seed}"))
            if seed == cfg.seeds[0]:
                seed42_train_pred = train_pred_con_existing
                seed42_test_pred = test_pred_con_existing
            continue

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        model = IndependentMultitaskModel(
            cfg.model_name,
            len(evasion_encoder.classes_),
            clarity_weights,
            evasion_weights,
            cfg.clarity_loss_weight,
            cfg.gradient_checkpointing,
            cfg.freeze_embeddings,
        ).to(device)
        model.float()
        optimizer, scheduler = make_optimizer_scheduler(model, train_loader, cfg)
        tracker_constrained = CartographyTracker(f"constrained_seed{seed}")
        tracker_independent = CartographyTracker(f"independent_seed{seed}") if seed in cfg.ablation_seeds else None
        start_epoch = 1
        ckpt_path = checkpoint_dir / f"independent_seed{seed}.pt"
        independent_ckpt_metadata = {
            "kind": "independent",
            "seed": seed,
            "n_train_examples": len(train_examples),
            "epochs": cfg.epochs,
            "model_name": cfg.model_name,
            "max_length": cfg.max_length,
        }
        if cfg.resume and ckpt_path.exists():
            try:
                ckpt = load_training_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    device,
                    expected_metadata=independent_ckpt_metadata,
                )
                if ckpt is not None:
                    if ckpt.get("tracker_constrained") is not None:
                        tracker_constrained = CartographyTracker.from_state(ckpt["tracker_constrained"])
                    if tracker_independent is not None and ckpt.get("tracker_independent") is not None:
                        tracker_independent = CartographyTracker.from_state(ckpt["tracker_independent"])
                    start_epoch = int(ckpt.get("epoch", 0)) + 1
                    print(f"  Resumed independent seed {seed} from epoch {start_epoch - 1}.", flush=True)
                else:
                    print(f"  Ignoring incompatible checkpoint for independent seed {seed}.", flush=True)
            except Exception as exc:
                print(f"  WARNING: failed to load checkpoint {ckpt_path}: {exc}", flush=True)

        for epoch in range(start_epoch, cfg.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, scheduler, device, cfg, model_kind="independent")
            train_metrics = evaluate_independent(
                model,
                train_eval_loader,
                device,
                tracker_constrained,
                tracker_independent,
                train_examples,
                evasion_encoder,
                legal_pairs,
                constraint_mask,
                "train",
            )
            print(
                f"  seed={seed} epoch={epoch}/{cfg.epochs} loss={loss:.4f} "
                f"train_joint_acc_constrained={train_metrics['train_joint_acc_constrained']:.3f} "
                f"elapsed={(time.time() - t0) / 60:.1f}m",
                flush=True,
            )
            partial_constrained_df = export_epoch_artifacts(
                tracker_constrained,
                train_examples,
                "qevasion_constrained",
                seed,
                epoch,
                partial_dir,
            )
            if tracker_independent is not None:
                export_epoch_artifacts(
                    tracker_independent,
                    train_examples,
                    "qevasion_independent",
                    seed,
                    epoch,
                    partial_dir,
                )
            epoch_metric_row = {
                "model": "constrained",
                "seed": seed,
                "epoch": epoch,
                "train_loss": loss,
                **train_metrics,
                "n_regions": int(partial_constrained_df["region"].nunique()),
                "region_assignment": str(partial_constrained_df["region_assignment"].mode().iloc[0])
                if "region_assignment" in partial_constrained_df
                else "",
                "timestamp": now(),
            }
            if cfg.save_epoch_predictions:
                epoch_train_pred = collect_independent_predictions(
                    model,
                    train_eval_loader,
                    device,
                    train_examples,
                    evasion_encoder,
                    legal_pairs,
                    constraint_mask,
                    "constrained",
                )
                epoch_test_pred = collect_independent_predictions(
                    model,
                    test_eval_loader,
                    device,
                    test_examples,
                    evasion_encoder,
                    legal_pairs,
                    constraint_mask,
                    "constrained",
                )
                atomic_to_csv(
                    epoch_train_pred,
                    partial_dir / f"qevasion_constrained_train_predictions_seed{seed}_epoch{epoch:02d}.csv",
                )
                atomic_to_csv(
                    epoch_test_pred,
                    partial_dir / f"qevasion_constrained_test_predictions_seed{seed}_epoch{epoch:02d}.csv",
                )
                epoch_metric_row.update(metric_row(epoch_train_pred, f"constrained_train_seed{seed}_epoch{epoch}"))
            append_epoch_metrics(partial_dir / "qevasion_epoch_metrics.csv", epoch_metric_row)
            if cfg.save_checkpoints:
                save_training_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    tracker_constrained=tracker_constrained,
                    tracker_independent=tracker_independent,
                    metadata=independent_ckpt_metadata,
                )

        constrained_df = tracker_constrained.compute_metrics()
        constrained_df = assign_regions(constrained_df)
        constrained_df = attach_metadata(constrained_df, train_examples)
        atomic_to_csv(constrained_df, final_carto_path)
        constrained_seed_dfs[seed] = constrained_df

        if tracker_independent is not None:
            independent_df = tracker_independent.compute_metrics()
            independent_df = assign_regions(independent_df)
            independent_df = attach_metadata(independent_df, train_examples)
            atomic_to_csv(independent_df, res_dir / f"qevasion_independent_carto_seed{seed}.csv")
            independent_seed_dfs[seed] = independent_df

        train_pred_con = collect_independent_predictions(
            model, train_eval_loader, device, train_examples, evasion_encoder, legal_pairs, constraint_mask, "constrained"
        )
        test_pred_con = collect_independent_predictions(
            model, test_eval_loader, device, test_examples, evasion_encoder, legal_pairs, constraint_mask, "constrained"
        )
        atomic_to_csv(train_pred_con, final_train_pred_path)
        atomic_to_csv(test_pred_con, final_test_pred_path)
        metrics_rows.append(metric_row(train_pred_con, f"constrained_train_seed{seed}"))
        metrics_rows.append(metric_row(test_pred_con, f"constrained_test_seed{seed}"))
        export_error_breakdowns(test_pred_con, f"qevasion_constrained_test_seed{seed}", res_dir)

        if seed in cfg.ablation_seeds:
            train_pred_ind = collect_independent_predictions(
                model,
                train_eval_loader,
                device,
                train_examples,
                evasion_encoder,
                legal_pairs,
                constraint_mask,
                "independent",
            )
            test_pred_ind = collect_independent_predictions(
                model,
                test_eval_loader,
                device,
                test_examples,
                evasion_encoder,
                legal_pairs,
                constraint_mask,
                "independent",
            )
            atomic_to_csv(train_pred_ind, res_dir / f"qevasion_independent_train_predictions_seed{seed}.csv")
            atomic_to_csv(test_pred_ind, res_dir / f"qevasion_independent_test_predictions_seed{seed}.csv")
            metrics_rows.append(metric_row(train_pred_ind, f"independent_train_seed{seed}"))
            metrics_rows.append(metric_row(test_pred_ind, f"independent_test_seed{seed}"))
            export_error_breakdowns(test_pred_ind, f"qevasion_independent_test_seed{seed}", res_dir)

        if seed == cfg.seeds[0]:
            seed42_train_pred = train_pred_con
            seed42_test_pred = test_pred_con

        del model, optimizer, scheduler
        free_gpu()

    joint_seed_dfs: dict[int, pd.DataFrame] = {}
    if cfg.run_ablations:
        print("\nTraining joint-pair ablation models...", flush=True)
        for seed in cfg.ablation_seeds:
            set_seed(seed)
            free_gpu()
            print(f"\n[{now()}] Seed {seed} joint-pair ablation", flush=True)
            final_joint_carto_path = res_dir / f"qevasion_joint_carto_seed{seed}.csv"
            final_joint_train_pred_path = res_dir / f"qevasion_joint_train_predictions_seed{seed}.csv"
            final_joint_test_pred_path = res_dir / f"qevasion_joint_test_predictions_seed{seed}.csv"
            if (
                cfg.resume
                and carto_output_matches(final_joint_carto_path, len(train_examples), cfg.epochs)
                and prediction_output_matches(final_joint_train_pred_path, len(train_examples))
                and prediction_output_matches(final_joint_test_pred_path, len(test_examples))
            ):
                print(f"  Found complete joint seed {seed}; loading existing outputs.", flush=True)
                joint_seed_dfs[seed] = pd.read_csv(final_joint_carto_path)
                train_pred_joint_existing = pd.read_csv(final_joint_train_pred_path)
                test_pred_joint_existing = pd.read_csv(final_joint_test_pred_path)
                metrics_rows.append(metric_row(train_pred_joint_existing, f"joint_train_seed{seed}"))
                metrics_rows.append(metric_row(test_pred_joint_existing, f"joint_test_seed{seed}"))
                continue

            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=device.type == "cuda",
            )
            model = JointPairModel(
                cfg.model_name,
                len(joint_encoder.classes_),
                joint_weights,
                cfg.gradient_checkpointing,
                cfg.freeze_embeddings,
            ).to(device)
            model.float()
            optimizer, scheduler = make_optimizer_scheduler(model, train_loader, cfg)
            tracker = CartographyTracker(f"joint_seed{seed}")
            start_epoch = 1
            ckpt_path = checkpoint_dir / f"joint_seed{seed}.pt"
            joint_ckpt_metadata = {
                "kind": "joint",
                "seed": seed,
                "n_train_examples": len(train_examples),
                "epochs": cfg.epochs,
                "model_name": cfg.model_name,
                "max_length": cfg.max_length,
            }
            if cfg.resume and ckpt_path.exists():
                try:
                    ckpt = load_training_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        scheduler,
                        device,
                        expected_metadata=joint_ckpt_metadata,
                    )
                    if ckpt is not None and ckpt.get("tracker_joint") is not None:
                        tracker = CartographyTracker.from_state(ckpt["tracker_joint"])
                        start_epoch = int(ckpt.get("epoch", 0)) + 1
                        print(f"  Resumed joint seed {seed} from epoch {start_epoch - 1}.", flush=True)
                    elif ckpt is None:
                        print(f"  Ignoring incompatible checkpoint for joint seed {seed}.", flush=True)
                except Exception as exc:
                    print(f"  WARNING: failed to load checkpoint {ckpt_path}: {exc}", flush=True)

            for epoch in range(start_epoch, cfg.epochs + 1):
                loss = train_epoch(model, train_loader, optimizer, scheduler, device, cfg, model_kind="joint")
                train_metrics = evaluate_joint(
                    model, train_eval_loader, device, tracker, train_examples, joint_encoder, "train"
                )
                print(
                    f"  seed={seed} epoch={epoch}/{cfg.epochs} loss={loss:.4f} "
                    f"train_joint_acc={train_metrics['train_joint_acc_joint']:.3f}",
                    flush=True,
                )
                partial_joint_df = export_epoch_artifacts(
                    tracker,
                    train_examples,
                    "qevasion_joint",
                    seed,
                    epoch,
                    partial_dir,
                )
                epoch_metric_row = {
                    "model": "joint",
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": loss,
                    **train_metrics,
                    "n_regions": int(partial_joint_df["region"].nunique()),
                    "region_assignment": str(partial_joint_df["region_assignment"].mode().iloc[0])
                    if "region_assignment" in partial_joint_df
                    else "",
                    "timestamp": now(),
                }
                if cfg.save_epoch_predictions:
                    epoch_train_pred = collect_joint_predictions(
                        model, train_eval_loader, device, train_examples, joint_encoder
                    )
                    epoch_test_pred = collect_joint_predictions(
                        model, test_eval_loader, device, test_examples, joint_encoder
                    )
                    atomic_to_csv(
                        epoch_train_pred,
                        partial_dir / f"qevasion_joint_train_predictions_seed{seed}_epoch{epoch:02d}.csv",
                    )
                    atomic_to_csv(
                        epoch_test_pred,
                        partial_dir / f"qevasion_joint_test_predictions_seed{seed}_epoch{epoch:02d}.csv",
                    )
                    epoch_metric_row.update(metric_row(epoch_test_pred, f"joint_test_seed{seed}_epoch{epoch}"))
                append_epoch_metrics(partial_dir / "qevasion_epoch_metrics.csv", epoch_metric_row)
                if cfg.save_checkpoints:
                    save_training_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        tracker_joint=tracker,
                        metadata=joint_ckpt_metadata,
                    )
            joint_df = tracker.compute_metrics()
            joint_df = assign_regions(joint_df)
            joint_df = attach_metadata(joint_df, train_examples)
            atomic_to_csv(joint_df, final_joint_carto_path)
            joint_seed_dfs[seed] = joint_df
            train_pred_joint = collect_joint_predictions(model, train_eval_loader, device, train_examples, joint_encoder)
            test_pred_joint = collect_joint_predictions(model, test_eval_loader, device, test_examples, joint_encoder)
            atomic_to_csv(train_pred_joint, final_joint_train_pred_path)
            atomic_to_csv(test_pred_joint, final_joint_test_pred_path)
            metrics_rows.append(metric_row(train_pred_joint, f"joint_train_seed{seed}"))
            metrics_rows.append(metric_row(test_pred_joint, f"joint_test_seed{seed}"))
            export_error_breakdowns(test_pred_joint, f"qevasion_joint_test_seed{seed}", res_dir)
            del model, optimizer, scheduler
            free_gpu()

    print("\nComputing cross-seed stability and reviewer analyses...", flush=True)
    available_constrained_seeds = [seed for seed in cfg.seeds if seed in constrained_seed_dfs]
    if not available_constrained_seeds:
        raise RuntimeError("No constrained cartography seed outputs are available for stability analysis.")
    if len(available_constrained_seeds) < len(cfg.seeds):
        print(
            f"  WARNING: stability will use completed seeds only: {available_constrained_seeds}",
            flush=True,
        )
    stability_df, stability_stats = build_stability_df(constrained_seed_dfs, available_constrained_seeds)
    export_stability(stability_df, "qevasion_constrained", res_dir)
    reviewer_breakdowns = export_reviewer_breakdowns(stability_df, "qevasion_constrained", res_dir)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(res_dir / "qevasion_all_metrics.csv", index=False)

    examples_exported = export_qualitative_examples(
        stability_df,
        seed42_train_pred,
        seed42_test_pred,
        "qevasion_constrained",
        examples_dir,
        cfg.export_example_limit,
    )

    seed_ref = available_constrained_seeds[0]
    export_plots(
        stability_df,
        stability_stats,
        constrained_seed_dfs[seed_ref],
        seed42_test_pred,
        "qevasion_constrained",
        res_dir,
        plot_dir,
    )

    summary = {
        "completed_at": now(),
        "elapsed_seconds": time.time() - t0,
        "dataset": cfg.dataset_name,
        "model": cfg.model_name,
        "n_train_examples": len(train_examples),
        "n_test_examples": len(test_examples),
        "seeds": cfg.seeds,
        "seeds_completed": available_constrained_seeds,
        "ablation_seeds": cfg.ablation_seeds if cfg.run_ablations else [],
        "epochs": cfg.epochs,
        "dry_run": cfg.dry_run,
        "large_one_run": cfg.large_one_run,
        "label_maps": label_maps,
        "stability": stability_stats,
        "reviewer_breakdowns": reviewer_breakdowns,
        "metrics": metrics_df.to_dict(orient="records"),
        "examples_exported": examples_exported,
        "headline_stats": {
            "easy_n": int((stability_df["majority_region"] == "easy").sum()),
            "ambiguous_n": int((stability_df["majority_region"] == "ambiguous").sum()),
            "hard_n": int((stability_df["majority_region"] == "hard").sum()),
            "region_assignment_modes": stability_stats.get("region_assignment_modes", []),
            "stable_pct": stability_stats["pct_stable"],
            "chronically_unstable_pct": stability_stats["pct_chronically_unstable"],
            "hard_region_top_evasion": reviewer_breakdowns.get("hard_evasion_enrichment", [{}])[0].get(
                "evasion_label"
            )
            if reviewer_breakdowns.get("hard_evasion_enrichment")
            else None,
        },
    }
    with open(res_dir / "summary_semeval_cartography.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80, flush=True)
    print("SEMEVAL CARTOGRAPHY COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Elapsed: {(time.time() - t0) / 3600:.2f}h", flush=True)
    print(f"Results: {res_dir}", flush=True)
    print(f"Plots: {plot_dir}", flush=True)
    print(f"Examples: {examples_dir}", flush=True)
    print(
        "Region counts: "
        f"easy={summary['headline_stats']['easy_n']} "
        f"ambiguous={summary['headline_stats']['ambiguous_n']} "
        f"hard={summary['headline_stats']['hard_n']}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        fallback_out = Path(os.environ.get("OUTPUT_DIR", "./output"))
        try:
            write_crash_report(fallback_out, exc)
        finally:
            print("\nSEMEVAL CARTOGRAPHY FAILED", flush=True)
            print(f"Crash report: {fallback_out / 'logs' / 'crash_report.json'}", flush=True)
        raise
