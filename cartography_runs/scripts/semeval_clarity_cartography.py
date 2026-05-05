#!/usr/bin/env python3
"""SemEval/QEvasion clarity cartography.

Runs two clarity-only cartography variants:

1. regular: one 3-way clarity classifier trained on all clarity labels at once.
2. hierarchical: the two-stage setup from Hijerarhijski_0.6.ipynb:
   binary Clear Non-Reply vs {Clear Reply, Ambivalent}, followed by a fine
   Clear Reply vs Ambivalent classifier.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from semeval_cartography import (
    CLARITY_MAPPING,
    INV_CLARITY_MAPPING,
    REGION_COLORS,
    REGION_NAMES,
    CartographyTracker,
    assign_regions,
    atomic_json_dump,
    atomic_to_csv,
    compact_text,
    compute_cross_seed_stats,
    normalize_clarity,
    now,
    parse_seed_list,
    safe_str,
    set_seed,
    write_crash_report,
)


warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

DEFAULT_SEEDS = [42, 123, 456, 789, 1337]


@dataclass
class RunConfig:
    dataset_name: str
    model_name: str
    output_dir: Path
    seeds: list[int]
    seed: int
    epochs: int
    batch_size: int
    eval_batch_size: int
    grad_accumulation: int
    learning_rate: float
    fine_learning_rate: float
    regular_learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    max_length: int
    mixed_precision: bool
    gradient_checkpointing: bool
    freeze_embeddings: bool
    num_workers: int
    subset_train: int | None
    subset_test: int | None
    dry_run: bool
    run_regular: bool
    run_hierarchical: bool
    hierarchical_threshold: float
    balanced_regular_loss: bool
    save_epoch_predictions: bool
    export_example_limit: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="ailsntua/QEvasion")
    parser.add_argument("--model-name", default="FacebookAI/roberta-base")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./output") + "/clarity_cartography")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--seed", type=int, default=None, help="Run a single seed; kept for compatibility.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--fine-learning-rate", type=float, default=1e-5)
    parser.add_argument("--regular-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-train", type=int, default=None)
    parser.add_argument("--subset-test", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-regular", action="store_true")
    parser.add_argument("--skip-hierarchical", action="store_true")
    parser.add_argument("--hierarchical-threshold", type=float, default=0.5)
    parser.add_argument("--balanced-regular-loss", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--save-epoch-predictions", action="store_true")
    parser.add_argument("--export-example-limit", type=int, default=60)
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    seeds = [args.seed] if args.seed is not None else parse_seed_list(args.seeds, DEFAULT_SEEDS)
    epochs = args.epochs
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    grad_accumulation = args.grad_accumulation
    subset_train = args.subset_train
    subset_test = args.subset_test

    if args.dry_run:
        seeds = [seeds[0]]
        epochs = 1
        batch_size = min(batch_size, 4)
        eval_batch_size = min(eval_batch_size, 8)
        grad_accumulation = 1
        subset_train = subset_train or 64
        subset_test = subset_test or 32

    return RunConfig(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        output_dir=Path(args.output_dir),
        seeds=seeds,
        seed=seeds[0],
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        grad_accumulation=grad_accumulation,
        learning_rate=args.learning_rate,
        fine_learning_rate=args.fine_learning_rate,
        regular_learning_rate=args.regular_learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_length,
        mixed_precision=not args.no_mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_embeddings=args.freeze_embeddings,
        num_workers=args.num_workers,
        subset_train=subset_train,
        subset_test=subset_test,
        dry_run=args.dry_run,
        run_regular=not args.skip_regular,
        run_hierarchical=not args.skip_hierarchical,
        hierarchical_threshold=args.hierarchical_threshold,
        balanced_regular_loss=args.balanced_regular_loss,
        save_epoch_predictions=args.save_epoch_predictions,
        export_example_limit=args.export_example_limit,
    )


class ClarityDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer: AutoTokenizer, max_length: int, label_key: str):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex[self.label_key], dtype=torch.long),
            "idx": torch.tensor(ex["idx"], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


def prepare_examples(raw_dataset: Any, split: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in raw_dataset[split]:
        clarity = normalize_clarity(row.get("clarity_label"))
        if clarity not in CLARITY_MAPPING:
            continue
        label = CLARITY_MAPPING[clarity]
        question = safe_str(row.get("interview_question"))
        answer = safe_str(row.get("interview_answer"))
        subquestion = safe_str(row.get("question"))
        text = f"Question: {question} Answer: {answer}"
        examples.append(
            {
                "idx": len(examples),
                "split": split,
                "dataset_index": row.get("index", len(examples)),
                "title": safe_str(row.get("title")),
                "date": safe_str(row.get("date")),
                "president": safe_str(row.get("president")),
                "url": safe_str(row.get("url")),
                "question_order": row.get("question_order", ""),
                "interview_question": question,
                "interview_answer": answer,
                "question": subquestion,
                "text": text,
                "clarity_label": clarity,
                "gold_label": clarity,
                "label": label,
                "binary_label": 0 if label in (0, 1) else 1,
                "fine_label": label if label in (0, 1) else -100,
                "inaudible": bool(row.get("inaudible", False)),
                "multiple_questions": bool(row.get("multiple_questions", False)),
                "affirmative_questions": bool(row.get("affirmative_questions", False)),
                "question_word_count": len(question.split()),
                "answer_word_count": len(answer.split()),
                "subquestion_word_count": len(subquestion.split()),
            }
        )
    return examples


def stratified_subset(examples: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    if n is None or n >= len(examples):
        return examples
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for ex in examples:
        by_label.setdefault(ex["clarity_label"], []).append(ex)
    per_label = max(1, n // max(1, len(by_label)))
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for label_examples in by_label.values():
        order = rng.permutation(len(label_examples))
        shuffled = [label_examples[int(i)] for i in order]
        selected.extend(shuffled[:per_label])
        leftovers.extend(shuffled[per_label:])
    if len(selected) < n and leftovers:
        order = rng.permutation(len(leftovers))
        selected.extend([leftovers[int(i)] for i in order[: n - len(selected)]])
    selected = selected[:n]
    return reindex_examples(selected)


def reindex_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for idx, ex in enumerate(examples):
        copied = dict(ex)
        copied["idx"] = idx
        out.append(copied)
    return out


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
        "gold_label",
        "label",
        "binary_label",
        "fine_label",
        "inaudible",
        "multiple_questions",
        "affirmative_questions",
        "question_word_count",
        "answer_word_count",
        "subquestion_word_count",
    ]
    return pd.DataFrame([{c: ex.get(c, "") for c in cols} for ex in examples])


def attach_metadata(carto_df: pd.DataFrame, examples: list[dict[str, Any]]) -> pd.DataFrame:
    meta = examples_metadata(examples)
    duplicate_cols = [c for c in meta.columns if c in carto_df.columns and c != "idx"]
    if duplicate_cols:
        meta = meta.drop(columns=duplicate_cols)
    return carto_df.merge(meta, on="idx", how="left")


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


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_prepare_model(model: torch.nn.Module, cfg: RunConfig) -> torch.nn.Module:
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if cfg.freeze_embeddings:
        base_model = getattr(model, "base_model", None)
        embeddings = getattr(base_model, "embeddings", None) if base_model is not None else None
        if embeddings is not None:
            for param in embeddings.parameters():
                param.requires_grad = False
    return model


def make_model(cfg: RunConfig, num_labels: int, device: torch.device) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=num_labels)
    model = maybe_prepare_model(model, cfg)
    return model.to(device)


def trainable_steps(loader: DataLoader, epochs: int, grad_accumulation: int) -> int:
    return max(1, math.ceil(len(loader) / grad_accumulation) * epochs)


def make_optimizer_scheduler(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: RunConfig,
    learning_rate: float,
) -> tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)
    steps = trainable_steps(loader, cfg.epochs, cfg.grad_accumulation)
    warmup = int(steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, steps)
    return optimizer, scheduler


def balanced_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=n_classes).astype(np.float32)
    total = float(counts.sum())
    weights = np.ones(n_classes, dtype=np.float32)
    if total > 0:
        weights = total / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    cfg: RunConfig,
    class_weights: torch.Tensor | None = None,
) -> float:
    model.train()
    amp_dtype = preferred_amp_dtype(device) if cfg.mixed_precision else None
    amp_enabled = amp_dtype is not None
    has_fp16_trainable_params = any(p.requires_grad and p.dtype == torch.float16 for p in model.parameters())
    use_scaler = amp_enabled and amp_dtype == torch.float16 and not has_fp16_trainable_params
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    weights = class_weights.to(device) if class_weights is not None else None
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        batch = batch_to_device(batch, device)
        labels = batch["labels"]
        amp_context = torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype) if amp_enabled else nullcontext()
        with amp_context:
            logits = model(**model_inputs(batch)).logits
            loss = F.cross_entropy(logits, labels, weight=weights) / cfg.grad_accumulation

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
def predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n = len(loader.dataset)
    n_labels = int(model.config.num_labels)
    probs_all = np.zeros((n, n_labels), dtype=np.float32)
    labels_all = np.zeros(n, dtype=np.int64)
    amp_dtype = preferred_amp_dtype(device) if cfg.mixed_precision else None
    amp_enabled = amp_dtype is not None

    for batch in loader:
        batch = batch_to_device(batch, device)
        if amp_enabled:
            with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                logits = model(**model_inputs(batch)).logits
        else:
            logits = model(**model_inputs(batch)).logits
        probs = F.softmax(logits.float(), dim=-1).detach().cpu().numpy()
        idx = batch["idx"].detach().cpu().numpy().astype(int)
        probs_all[idx] = probs
        labels_all[idx] = batch["labels"].detach().cpu().numpy().astype(int)

    return probs_all, labels_all


def make_loader(
    examples: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    cfg: RunConfig,
    label_key: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = ClarityDataset(examples, tokenizer, cfg.max_length, label_key)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def label_names(labels: np.ndarray) -> list[str]:
    return [INV_CLARITY_MAPPING[int(x)] for x in labels]


def metric_row(labels: np.ndarray, preds: np.ndarray, prefix: str) -> dict[str, Any]:
    return {
        "prefix": prefix,
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }


def build_carto(
    prefix: str,
    seed: int,
    epoch_probs: list[np.ndarray],
    labels: np.ndarray,
    examples: list[dict[str, Any]],
    epoch_preds: list[np.ndarray] | None = None,
    extras: list[dict[str, np.ndarray]] | None = None,
) -> pd.DataFrame:
    tracker = CartographyTracker(f"{prefix}_seed{seed}")
    for epoch_idx, probs in enumerate(epoch_probs):
        preds = epoch_preds[epoch_idx] if epoch_preds is not None else probs.argmax(axis=1)
        epoch_extra = extras[epoch_idx] if extras is not None else {}
        for idx, gold_idx in enumerate(labels.astype(int)):
            pred_idx = int(preds[idx])
            extra_kwargs = {}
            for key, values in epoch_extra.items():
                value = values[idx]
                if isinstance(value, np.generic):
                    value = value.item()
                extra_kwargs[key] = value
            tracker.record(
                idx=idx,
                prob_true=float(probs[idx, gold_idx]),
                pred=INV_CLARITY_MAPPING[pred_idx],
                gold=INV_CLARITY_MAPPING[int(gold_idx)],
                correct=pred_idx == int(gold_idx),
                **extra_kwargs,
            )
    df = tracker.compute_metrics().rename(columns={"gold_pair": "gold_label"})
    df = assign_regions(df)
    df["seed"] = seed
    df["cartography_variant"] = prefix
    df = attach_metadata(df, examples)
    return df


def one_seed_stability(carto_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = carto_df.copy()
    df["conf_mean"] = df["confidence"]
    df["conf_std"] = 0.0
    df["var_mean"] = df["variability"]
    df["var_std"] = 0.0
    df["correctness_mean"] = df["correctness"]
    df["forgetfulness_mean"] = df["forgetfulness"]
    df[f"conf_seed{seed}"] = df["confidence"]
    df[f"var_seed{seed}"] = df["variability"]
    df[f"correctness_seed{seed}"] = df["correctness"]
    df[f"forgetfulness_seed{seed}"] = df["forgetfulness"]
    df[f"region_seed{seed}"] = df["region"]
    df[f"region_classic_seed{seed}"] = df["region_classic"]
    df[f"region_percentile_seed{seed}"] = df["region_percentile"]
    df[f"region_assignment_seed{seed}"] = df["region_assignment"]
    df["majority_region"] = df["region"]
    df["region_agreement_count"] = 1
    df["region_agreement"] = 1.0
    df["is_stable"] = True
    df["is_chronically_unstable"] = False
    df["region_pattern"] = df["region"]
    return df


def build_multi_seed_stability(seed_dfs: dict[int, pd.DataFrame], seeds: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    ref = seed_dfs[seeds[0]].sort_values("idx").reset_index(drop=True)
    n = len(ref)
    conf_matrix = np.zeros((n, len(seeds)))
    var_matrix = np.zeros((n, len(seeds)))
    correct_matrix = np.zeros((n, len(seeds)))
    forget_matrix = np.zeros((n, len(seeds)))
    region_lists: list[np.ndarray] = []

    for si, seed in enumerate(seeds):
        df_s = seed_dfs[seed].sort_values("idx").reset_index(drop=True)
        conf_matrix[:, si] = df_s["confidence"].values
        var_matrix[:, si] = df_s["variability"].values
        correct_matrix[:, si] = df_s["correctness"].values
        forget_matrix[:, si] = df_s["forgetfulness"].values
        region_lists.append(df_s["region"].astype(str).values)

    data: dict[str, Any] = {}
    for col in [
        "idx",
        "gold_label",
        "clarity_label",
        "interview_question",
        "interview_answer",
        "question",
        "president",
        "inaudible",
        "multiple_questions",
        "affirmative_questions",
        "question_word_count",
        "answer_word_count",
        "subquestion_word_count",
    ]:
        if col in ref.columns:
            data[col] = ref[col].values

    data.update(
        {
            "conf_mean": conf_matrix.mean(axis=1),
            "conf_std": conf_matrix.std(axis=1),
            "var_mean": var_matrix.mean(axis=1),
            "var_std": var_matrix.std(axis=1),
            "correctness_mean": correct_matrix.mean(axis=1),
            "forgetfulness_mean": forget_matrix.mean(axis=1),
        }
    )
    for si, seed in enumerate(seeds):
        df_s = seed_dfs[seed].sort_values("idx").reset_index(drop=True)
        data[f"conf_seed{seed}"] = conf_matrix[:, si]
        data[f"var_seed{seed}"] = var_matrix[:, si]
        data[f"correctness_seed{seed}"] = correct_matrix[:, si]
        data[f"forgetfulness_seed{seed}"] = forget_matrix[:, si]
        data[f"region_seed{seed}"] = region_lists[si]
        data[f"region_classic_seed{seed}"] = df_s["region_classic"].values
        data[f"region_percentile_seed{seed}"] = df_s["region_percentile"].values
        data[f"region_assignment_seed{seed}"] = df_s["region_assignment"].values

    region_counts = np.zeros((n, len(REGION_NAMES)))
    for regions in region_lists:
        for ri, region in enumerate(REGION_NAMES):
            region_counts[:, ri] += (regions == region).astype(int)
    majority_idx = region_counts.argmax(axis=1)
    max_agreement = region_counts.max(axis=1)
    data["majority_region"] = np.asarray(REGION_NAMES)[majority_idx]
    data["region_agreement_count"] = max_agreement.astype(int)
    data["region_agreement"] = max_agreement / max(1, len(seeds))
    data["is_stable"] = max_agreement == len(seeds)
    data["is_chronically_unstable"] = max_agreement <= 3 if len(seeds) >= 5 else max_agreement < len(seeds)
    data["region_pattern"] = [",".join(region_lists[si][i] for si in range(len(seeds))) for i in range(n)]

    stats = compute_cross_seed_stats(conf_matrix, var_matrix, region_lists, seeds)
    stats["region_assignment_modes"] = [
        {
            "seed": seed,
            "mode_counts": {str(k): int(v) for k, v in seed_dfs[seed]["region_assignment"].value_counts().to_dict().items()},
        }
        for seed in seeds
    ]
    return pd.DataFrame(data), stats


def predictions_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    examples: list[dict[str, Any]],
    prefix: str,
    preds: np.ndarray | None = None,
    extra_cols: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    pred_idx = preds if preds is not None else probs.argmax(axis=1)
    rows = []
    for idx, ex in enumerate(examples):
        gold = int(labels[idx])
        pred = int(pred_idx[idx])
        row = {
            "idx": idx,
            "split": ex["split"],
            "dataset_index": ex["dataset_index"],
            "variant": prefix,
            "clarity_label": ex["clarity_label"],
            "gold_label": INV_CLARITY_MAPPING[gold],
            "pred_label": INV_CLARITY_MAPPING[pred],
            "prob_true": float(probs[idx, gold]),
            "pred_prob": float(probs[idx, pred]),
            "is_correct": bool(pred == gold),
            "president": ex.get("president", ""),
            "interview_question": ex.get("interview_question", ""),
            "interview_answer": ex.get("interview_answer", ""),
            "question": ex.get("question", ""),
            "multiple_questions": ex.get("multiple_questions", False),
            "affirmative_questions": ex.get("affirmative_questions", False),
            "inaudible": ex.get("inaudible", False),
        }
        if extra_cols:
            for key, values in extra_cols.items():
                value = values[idx]
                row[key] = value.item() if isinstance(value, np.generic) else value
        rows.append(row)
    return pd.DataFrame(rows)


def export_breakdowns(stability_df: pd.DataFrame, pred_df: pd.DataFrame, prefix: str, res_dir: Path) -> dict[str, Any]:
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
    atomic_to_csv(region_counts, res_dir / f"{prefix}_region_counts.csv")
    out["region_counts"] = region_counts.to_dict(orient="records")

    region_by_clarity = pd.crosstab(stability_df["majority_region"], stability_df["clarity_label"]).reindex(
        REGION_NAMES,
        fill_value=0,
    )
    region_by_clarity_pct = region_by_clarity.div(
        region_by_clarity.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0) * 100.0
    region_by_clarity_out = region_by_clarity.add_suffix("_n").join(region_by_clarity_pct.add_suffix("_pct"))
    region_by_clarity_out.insert(0, "majority_region", region_by_clarity_out.index)
    region_by_clarity_out = region_by_clarity_out.reset_index(drop=True)
    atomic_to_csv(region_by_clarity_out, res_dir / f"{prefix}_region_by_clarity.csv")
    out["region_by_clarity"] = region_by_clarity_out.to_dict(orient="records")

    reverse = pd.crosstab(stability_df["clarity_label"], stability_df["majority_region"]).reindex(
        columns=REGION_NAMES,
        fill_value=0,
    )
    reverse["total"] = reverse.sum(axis=1)
    for region in REGION_NAMES:
        reverse[f"{region}_pct_within_label"] = 100.0 * reverse[region] / reverse["total"].replace(0, np.nan)
    reverse = reverse.fillna(0).reset_index()
    atomic_to_csv(reverse, res_dir / f"{prefix}_clarity_to_region_distribution.csv")

    meta_rows = []
    for region, grp in stability_df.groupby("majority_region"):
        meta_rows.append(
            {
                "region": region,
                "n": int(len(grp)),
                "confidence_mean": float(grp["conf_mean"].mean()),
                "variability_mean": float(grp["var_mean"].mean()),
                "correctness_mean": float(grp["correctness_mean"].mean()),
                "forgetfulness_mean": float(grp["forgetfulness_mean"].mean()),
                "question_words_mean": float(grp["question_word_count"].mean()),
                "answer_words_mean": float(grp["answer_word_count"].mean()),
                "multiple_questions_pct": float(100.0 * grp["multiple_questions"].mean()),
                "affirmative_questions_pct": float(100.0 * grp["affirmative_questions"].mean()),
                "inaudible_pct": float(100.0 * grp["inaudible"].mean()),
            }
        )
    meta = pd.DataFrame(meta_rows).sort_values("region")
    atomic_to_csv(meta, res_dir / f"{prefix}_metadata_by_region.csv")
    out["metadata_by_region"] = meta.to_dict(orient="records")

    if not pred_df.empty:
        by_label = (
            pred_df.groupby("clarity_label")
            .agg(
                n=("idx", "count"),
                accuracy=("is_correct", "mean"),
                prob_true_mean=("prob_true", "mean"),
                pred_prob_mean=("pred_prob", "mean"),
            )
            .reset_index()
            .sort_values("n", ascending=False)
        )
        atomic_to_csv(by_label, res_dir / f"{prefix}_prediction_by_clarity.csv")
        out["prediction_by_clarity"] = by_label.to_dict(orient="records")

    return out


def export_examples(stability_df: pd.DataFrame, pred_df: pd.DataFrame, prefix: str, examples_dir: Path, limit: int) -> dict[str, Any]:
    examples_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Any] = {}
    base_cols = [
        "idx",
        "majority_region",
        "clarity_label",
        "conf_mean",
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

    parts = []
    for region in REGION_NAMES:
        grp = stability_df[stability_df["majority_region"] == region].copy()
        if region == "hard":
            grp = grp.sort_values(["conf_mean", "correctness_mean", "var_mean"], ascending=[True, True, False])
        elif region == "ambiguous":
            grp = grp.sort_values(["var_mean", "conf_mean"], ascending=[False, True])
        else:
            grp = grp.sort_values(["conf_mean", "correctness_mean"], ascending=[False, False])
        parts.append(grp.head(limit))
    by_region = pd.concat(parts, ignore_index=True) if parts else stability_df.head(0)
    for text_col in ["interview_question", "interview_answer", "question"]:
        if text_col in by_region.columns:
            by_region[text_col] = by_region[text_col].map(lambda x: compact_text(str(x), 500))
    atomic_to_csv(by_region[base_cols], examples_dir / f"{prefix}_examples_by_region.csv")
    exported["examples_by_region"] = int(len(by_region))

    if not pred_df.empty:
        wrong = pred_df[~pred_df["is_correct"]].copy()
        for text_col in ["interview_question", "interview_answer", "question"]:
            if text_col in wrong.columns:
                wrong[text_col] = wrong[text_col].map(lambda x: compact_text(str(x), 500))
        cols = [
            "idx",
            "clarity_label",
            "pred_label",
            "prob_true",
            "pred_prob",
            "interview_question",
            "interview_answer",
            "question",
        ]
        cols = [c for c in cols if c in wrong.columns]
        high_conf_wrong = wrong.sort_values("pred_prob", ascending=False).head(limit * 2)
        atomic_to_csv(high_conf_wrong[cols], examples_dir / f"{prefix}_high_confidence_wrong.csv")
        exported["high_confidence_wrong"] = int(len(high_conf_wrong))
    return exported


def plot_cartography(stability_df: pd.DataFrame, prefix: str, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for region in REGION_NAMES:
        mask = stability_df["majority_region"] == region
        ax.scatter(
            stability_df.loc[mask, "var_mean"],
            stability_df.loc[mask, "conf_mean"],
            s=18,
            alpha=0.5,
            color=REGION_COLORS[region],
            label=f"{region} (n={int(mask.sum())})",
        )
    ax.set_xlabel("Variability")
    ax.set_ylabel("Confidence")
    ax.set_title(prefix.replace("_", " ").title())
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    fig.savefig(plot_dir / f"{prefix}_cartography.png")
    plt.close(fig)


def plot_confusion(pred_df: pd.DataFrame, prefix: str, plot_dir: Path) -> None:
    if pred_df.empty:
        return
    labels = [INV_CLARITY_MAPPING[i] for i in range(3)]
    cm = confusion_matrix(pred_df["gold_label"], pred_df["pred_label"], labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(prefix.replace("_", " ").title())
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{prefix}_confusion.png")
    plt.close(fig)


def export_variant_outputs(
    prefix: str,
    seed: int,
    carto_df: pd.DataFrame,
    train_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    res_dir: Path,
    plot_dir: Path,
    examples_dir: Path,
    export_example_limit: int,
) -> dict[str, Any]:
    stability_df = one_seed_stability(carto_df, seed)
    seed_prefix = f"{prefix}_seed{seed}"
    carto_path = res_dir / f"{prefix}_carto_seed{seed}.csv"
    stability_path = res_dir / f"{prefix}_stability_seed{seed}.csv"
    train_pred_path = res_dir / f"{prefix}_train_predictions_seed{seed}.csv"
    test_pred_path = res_dir / f"{prefix}_test_predictions_seed{seed}.csv"

    atomic_to_csv(carto_df, carto_path)
    atomic_to_csv(stability_df, stability_path)
    atomic_to_csv(stability_df[stability_df["is_stable"]], res_dir / f"{prefix}_stable_examples_seed{seed}.csv")
    atomic_to_csv(stability_df[stability_df["is_chronically_unstable"]], res_dir / f"{prefix}_unstable_examples_seed{seed}.csv")
    atomic_to_csv(train_pred, train_pred_path)
    atomic_to_csv(test_pred, test_pred_path)

    breakdowns = export_breakdowns(stability_df, test_pred, seed_prefix, res_dir)
    examples = export_examples(stability_df, test_pred, seed_prefix, examples_dir, export_example_limit)
    plot_cartography(stability_df, seed_prefix, plot_dir)
    plot_confusion(test_pred, f"{prefix}_test_seed{seed}", plot_dir)
    return {
        "carto_csv": str(carto_path),
        "stability_csv": str(stability_path),
        "train_predictions_csv": str(train_pred_path),
        "test_predictions_csv": str(test_pred_path),
        "breakdowns": breakdowns,
        "examples": examples,
    }


def export_multi_seed_outputs(
    prefix: str,
    seed_dfs: dict[int, pd.DataFrame],
    seeds: list[int],
    pred_df: pd.DataFrame,
    res_dir: Path,
    plot_dir: Path,
    examples_dir: Path,
    export_example_limit: int,
) -> dict[str, Any]:
    stability_df, stats = build_multi_seed_stability(seed_dfs, seeds)
    stability_path = res_dir / f"{prefix}_stability.csv"
    atomic_to_csv(stability_df, stability_path)
    atomic_to_csv(stability_df[stability_df["is_stable"]], res_dir / f"{prefix}_stable_examples.csv")
    atomic_to_csv(stability_df[stability_df["is_chronically_unstable"]], res_dir / f"{prefix}_unstable_examples.csv")
    breakdowns = export_breakdowns(stability_df, pred_df, prefix, res_dir)
    examples = export_examples(stability_df, pred_df, prefix, examples_dir, export_example_limit)
    plot_cartography(stability_df, prefix, plot_dir)
    atomic_json_dump(stats, res_dir / f"{prefix}_stability_summary.json")
    return {
        "variant": prefix,
        "seeds": seeds,
        "stability_csv": str(stability_path),
        "stable_examples_csv": str(res_dir / f"{prefix}_stable_examples.csv"),
        "unstable_examples_csv": str(res_dir / f"{prefix}_unstable_examples.csv"),
        "stability_summary_json": str(res_dir / f"{prefix}_stability_summary.json"),
        "breakdowns": breakdowns,
        "examples": examples,
    }


def write_partial_carto(
    prefix: str,
    seed: int,
    epoch: int,
    epoch_probs: list[np.ndarray],
    labels: np.ndarray,
    examples: list[dict[str, Any]],
    partial_dir: Path,
    epoch_preds: list[np.ndarray] | None = None,
    extras: list[dict[str, np.ndarray]] | None = None,
) -> pd.DataFrame:
    df = build_carto(prefix, seed, epoch_probs, labels, examples, epoch_preds=epoch_preds, extras=extras)
    atomic_to_csv(df, partial_dir / f"{prefix}_carto_seed{seed}_epoch{epoch:02d}.csv")
    atomic_to_csv(df, partial_dir / f"{prefix}_carto_seed{seed}_latest.csv")
    return df


def append_epoch_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def run_regular(
    cfg: RunConfig,
    tokenizer: AutoTokenizer,
    train_examples: list[dict[str, Any]],
    test_examples: list[dict[str, Any]],
    device: torch.device,
    dirs: dict[str, Path],
) -> dict[str, Any]:
    prefix = "qevasion_clarity_regular"
    print(f"\n[{now()}] Regular 3-way clarity cartography", flush=True)
    labels_train = np.asarray([ex["label"] for ex in train_examples], dtype=np.int64)
    labels_test = np.asarray([ex["label"] for ex in test_examples], dtype=np.int64)
    train_loader = make_loader(train_examples, tokenizer, cfg, "label", cfg.batch_size, True, cfg.seed)
    train_eval_loader = make_loader(train_examples, tokenizer, cfg, "label", cfg.eval_batch_size, False, cfg.seed)
    test_eval_loader = make_loader(test_examples, tokenizer, cfg, "label", cfg.eval_batch_size, False, cfg.seed)

    model = make_model(cfg, 3, device)
    optimizer, scheduler = make_optimizer_scheduler(model, train_loader, cfg, cfg.regular_learning_rate)
    class_weights = balanced_weights(labels_train, 3) if cfg.balanced_regular_loss else None
    epoch_probs_train: list[np.ndarray] = []
    epoch_probs_test: list[np.ndarray] = []
    metrics_rows: list[dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device, cfg, class_weights)
        train_probs, _ = predict_probs(model, train_eval_loader, device, cfg)
        test_probs, _ = predict_probs(model, test_eval_loader, device, cfg)
        epoch_probs_train.append(train_probs)
        epoch_probs_test.append(test_probs)
        train_preds = train_probs.argmax(axis=1)
        test_preds = test_probs.argmax(axis=1)
        train_metrics = metric_row(labels_train, train_preds, f"{prefix}_train_seed{cfg.seed}_epoch{epoch}")
        test_metrics = metric_row(labels_test, test_preds, f"{prefix}_test_seed{cfg.seed}_epoch{epoch}")
        metrics_rows.extend([train_metrics, test_metrics])
        partial_df = write_partial_carto(
            prefix,
            cfg.seed,
            epoch,
            epoch_probs_train,
            labels_train,
            train_examples,
            dirs["partial"],
        )
        append_epoch_metrics(
            dirs["partial"] / "qevasion_clarity_epoch_metrics.csv",
            {
                "variant": prefix,
                "seed": cfg.seed,
                "epoch": epoch,
                "train_loss": loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "n_regions": int(partial_df["region"].nunique()),
                "region_assignment": str(partial_df["region_assignment"].mode().iloc[0]),
                "timestamp": now(),
            },
        )
        if cfg.save_epoch_predictions:
            atomic_to_csv(
                predictions_from_probs(train_probs, labels_train, train_examples, prefix),
                dirs["partial"] / f"{prefix}_train_predictions_seed{cfg.seed}_epoch{epoch:02d}.csv",
            )
            atomic_to_csv(
                predictions_from_probs(test_probs, labels_test, test_examples, prefix),
                dirs["partial"] / f"{prefix}_test_predictions_seed{cfg.seed}_epoch{epoch:02d}.csv",
            )
        print(
            f"  epoch={epoch}/{cfg.epochs} loss={loss:.4f} "
            f"train_f1={train_metrics['macro_f1']:.3f} test_f1={test_metrics['macro_f1']:.3f}",
            flush=True,
        )

    carto_df = build_carto(prefix, cfg.seed, epoch_probs_train, labels_train, train_examples)
    train_pred = predictions_from_probs(epoch_probs_train[-1], labels_train, train_examples, prefix)
    test_pred = predictions_from_probs(epoch_probs_test[-1], labels_test, test_examples, prefix)
    output_info = export_variant_outputs(
        prefix,
        cfg.seed,
        carto_df,
        train_pred,
        test_pred,
        dirs["results"],
        dirs["plots"],
        dirs["examples"],
        cfg.export_example_limit,
    )
    del model, optimizer, scheduler
    free_gpu()
    return {
        "variant": prefix,
        "seed": cfg.seed,
        "metrics": metrics_rows,
        "final_train_metrics": metric_row(labels_train, train_pred["pred_label"].map(CLARITY_MAPPING).to_numpy(), f"{prefix}_train_seed{cfg.seed}"),
        "final_test_metrics": metric_row(labels_test, test_pred["pred_label"].map(CLARITY_MAPPING).to_numpy(), f"{prefix}_test_seed{cfg.seed}"),
        "outputs": output_info,
    }


def combine_hierarchical_probs(binary_probs: np.ndarray, fine_probs: np.ndarray) -> np.ndarray:
    combined = np.zeros((binary_probs.shape[0], 3), dtype=np.float32)
    combined[:, 0] = binary_probs[:, 0] * fine_probs[:, 0]
    combined[:, 1] = binary_probs[:, 0] * fine_probs[:, 1]
    combined[:, 2] = binary_probs[:, 1]
    return combined


def hierarchical_predictions(binary_probs: np.ndarray, fine_probs: np.ndarray, threshold: float) -> np.ndarray:
    fine_pred = fine_probs.argmax(axis=1).astype(np.int64)
    return np.where(binary_probs[:, 1] >= threshold, 2, fine_pred).astype(np.int64)


def hierarchical_extras(binary_probs: np.ndarray, fine_probs: np.ndarray, preds: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "binary_prob_reply_or_ambivalent": binary_probs[:, 0],
        "binary_prob_clear_non_reply": binary_probs[:, 1],
        "fine_prob_clear_reply": fine_probs[:, 0],
        "fine_prob_ambivalent": fine_probs[:, 1],
        "hierarchical_pred_id": preds,
    }


def run_hierarchical(
    cfg: RunConfig,
    tokenizer: AutoTokenizer,
    train_examples: list[dict[str, Any]],
    test_examples: list[dict[str, Any]],
    device: torch.device,
    dirs: dict[str, Path],
) -> dict[str, Any]:
    prefix = "qevasion_clarity_hierarchical"
    print(f"\n[{now()}] Hierarchical clarity cartography", flush=True)
    labels_train = np.asarray([ex["label"] for ex in train_examples], dtype=np.int64)
    labels_test = np.asarray([ex["label"] for ex in test_examples], dtype=np.int64)
    binary_labels_train = np.asarray([ex["binary_label"] for ex in train_examples], dtype=np.int64)
    binary_labels_test = np.asarray([ex["binary_label"] for ex in test_examples], dtype=np.int64)
    fine_train_examples = reindex_examples([ex for ex in train_examples if ex["fine_label"] in (0, 1)])

    all_train_eval_loader = make_loader(train_examples, tokenizer, cfg, "label", cfg.eval_batch_size, False, cfg.seed)
    all_test_eval_loader = make_loader(test_examples, tokenizer, cfg, "label", cfg.eval_batch_size, False, cfg.seed)

    binary_train_loader = make_loader(train_examples, tokenizer, cfg, "binary_label", cfg.batch_size, True, cfg.seed)
    binary_model = make_model(cfg, 2, device)
    binary_optimizer, binary_scheduler = make_optimizer_scheduler(binary_model, binary_train_loader, cfg, cfg.learning_rate)
    binary_train_epoch_probs: list[np.ndarray] = []
    binary_test_epoch_probs: list[np.ndarray] = []
    metrics_rows: list[dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(binary_model, binary_train_loader, binary_optimizer, binary_scheduler, device, cfg)
        train_probs, _ = predict_probs(binary_model, all_train_eval_loader, device, cfg)
        test_probs, _ = predict_probs(binary_model, all_test_eval_loader, device, cfg)
        binary_train_epoch_probs.append(train_probs)
        binary_test_epoch_probs.append(test_probs)
        train_metrics = metric_row(binary_labels_train, train_probs.argmax(axis=1), f"{prefix}_binary_train_seed{cfg.seed}_epoch{epoch}")
        test_metrics = metric_row(binary_labels_test, test_probs.argmax(axis=1), f"{prefix}_binary_test_seed{cfg.seed}_epoch{epoch}")
        metrics_rows.extend([train_metrics, test_metrics])
        print(
            f"  binary epoch={epoch}/{cfg.epochs} loss={loss:.4f} "
            f"train_f1={train_metrics['macro_f1']:.3f} test_f1={test_metrics['macro_f1']:.3f}",
            flush=True,
        )

    del binary_model, binary_optimizer, binary_scheduler
    free_gpu()

    fine_train_loader = make_loader(fine_train_examples, tokenizer, cfg, "fine_label", cfg.batch_size, True, cfg.seed + 1)
    fine_model = make_model(cfg, 2, device)
    fine_optimizer, fine_scheduler = make_optimizer_scheduler(fine_model, fine_train_loader, cfg, cfg.fine_learning_rate)
    fine_train_epoch_probs: list[np.ndarray] = []
    fine_test_epoch_probs: list[np.ndarray] = []
    combined_train_epoch_probs: list[np.ndarray] = []
    combined_test_epoch_probs: list[np.ndarray] = []
    combined_train_epoch_preds: list[np.ndarray] = []
    combined_test_epoch_preds: list[np.ndarray] = []
    combined_epoch_extras: list[dict[str, np.ndarray]] = []

    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(fine_model, fine_train_loader, fine_optimizer, fine_scheduler, device, cfg)
        fine_train_probs, _ = predict_probs(fine_model, all_train_eval_loader, device, cfg)
        fine_test_probs, _ = predict_probs(fine_model, all_test_eval_loader, device, cfg)
        fine_train_epoch_probs.append(fine_train_probs)
        fine_test_epoch_probs.append(fine_test_probs)

        combined_train = combine_hierarchical_probs(binary_train_epoch_probs[epoch - 1], fine_train_probs)
        combined_test = combine_hierarchical_probs(binary_test_epoch_probs[epoch - 1], fine_test_probs)
        pred_train = hierarchical_predictions(binary_train_epoch_probs[epoch - 1], fine_train_probs, cfg.hierarchical_threshold)
        pred_test = hierarchical_predictions(binary_test_epoch_probs[epoch - 1], fine_test_probs, cfg.hierarchical_threshold)
        combined_train_epoch_probs.append(combined_train)
        combined_test_epoch_probs.append(combined_test)
        combined_train_epoch_preds.append(pred_train)
        combined_test_epoch_preds.append(pred_test)
        combined_epoch_extras.append(hierarchical_extras(binary_train_epoch_probs[epoch - 1], fine_train_probs, pred_train))

        train_metrics = metric_row(labels_train, pred_train, f"{prefix}_train_seed{cfg.seed}_epoch{epoch}")
        test_metrics = metric_row(labels_test, pred_test, f"{prefix}_test_seed{cfg.seed}_epoch{epoch}")
        fine_mask_train = labels_train != 2
        fine_mask_test = labels_test != 2
        fine_train_metrics = metric_row(
            labels_train[fine_mask_train],
            fine_train_probs[fine_mask_train].argmax(axis=1),
            f"{prefix}_fine_train_seed{cfg.seed}_epoch{epoch}",
        )
        fine_test_metrics = metric_row(
            labels_test[fine_mask_test],
            fine_test_probs[fine_mask_test].argmax(axis=1),
            f"{prefix}_fine_test_seed{cfg.seed}_epoch{epoch}",
        )
        metrics_rows.extend([train_metrics, test_metrics, fine_train_metrics, fine_test_metrics])

        partial_df = write_partial_carto(
            prefix,
            cfg.seed,
            epoch,
            combined_train_epoch_probs,
            labels_train,
            train_examples,
            dirs["partial"],
            epoch_preds=combined_train_epoch_preds,
            extras=combined_epoch_extras,
        )
        append_epoch_metrics(
            dirs["partial"] / "qevasion_clarity_epoch_metrics.csv",
            {
                "variant": prefix,
                "seed": cfg.seed,
                "epoch": epoch,
                "train_loss": loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "fine_test_macro_f1": fine_test_metrics["macro_f1"],
                "hierarchical_threshold": cfg.hierarchical_threshold,
                "n_regions": int(partial_df["region"].nunique()),
                "region_assignment": str(partial_df["region_assignment"].mode().iloc[0]),
                "timestamp": now(),
            },
        )
        if cfg.save_epoch_predictions:
            atomic_to_csv(
                predictions_from_probs(
                    combined_train,
                    labels_train,
                    train_examples,
                    prefix,
                    preds=pred_train,
                    extra_cols=hierarchical_extras(binary_train_epoch_probs[epoch - 1], fine_train_probs, pred_train),
                ),
                dirs["partial"] / f"{prefix}_train_predictions_seed{cfg.seed}_epoch{epoch:02d}.csv",
            )
            atomic_to_csv(
                predictions_from_probs(
                    combined_test,
                    labels_test,
                    test_examples,
                    prefix,
                    preds=pred_test,
                    extra_cols=hierarchical_extras(binary_test_epoch_probs[epoch - 1], fine_test_probs, pred_test),
                ),
                dirs["partial"] / f"{prefix}_test_predictions_seed{cfg.seed}_epoch{epoch:02d}.csv",
            )
        print(
            f"  fine/combined epoch={epoch}/{cfg.epochs} loss={loss:.4f} "
            f"train_f1={train_metrics['macro_f1']:.3f} test_f1={test_metrics['macro_f1']:.3f}",
            flush=True,
        )

    carto_df = build_carto(
        prefix,
        cfg.seed,
        combined_train_epoch_probs,
        labels_train,
        train_examples,
        epoch_preds=combined_train_epoch_preds,
        extras=combined_epoch_extras,
    )
    train_pred = predictions_from_probs(
        combined_train_epoch_probs[-1],
        labels_train,
        train_examples,
        prefix,
        preds=combined_train_epoch_preds[-1],
        extra_cols=hierarchical_extras(
            binary_train_epoch_probs[-1],
            fine_train_epoch_probs[-1],
            combined_train_epoch_preds[-1],
        ),
    )
    test_pred = predictions_from_probs(
        combined_test_epoch_probs[-1],
        labels_test,
        test_examples,
        prefix,
        preds=combined_test_epoch_preds[-1],
        extra_cols=hierarchical_extras(
            binary_test_epoch_probs[-1],
            fine_test_epoch_probs[-1],
            combined_test_epoch_preds[-1],
        ),
    )
    output_info = export_variant_outputs(
        prefix,
        cfg.seed,
        carto_df,
        train_pred,
        test_pred,
        dirs["results"],
        dirs["plots"],
        dirs["examples"],
        cfg.export_example_limit,
    )
    del fine_model, fine_optimizer, fine_scheduler
    free_gpu()
    return {
        "variant": prefix,
        "seed": cfg.seed,
        "hierarchical_threshold": cfg.hierarchical_threshold,
        "metrics": metrics_rows,
        "final_train_metrics": metric_row(labels_train, train_pred["pred_label"].map(CLARITY_MAPPING).to_numpy(), f"{prefix}_train_seed{cfg.seed}"),
        "final_test_metrics": metric_row(labels_test, test_pred["pred_label"].map(CLARITY_MAPPING).to_numpy(), f"{prefix}_test_seed{cfg.seed}"),
        "outputs": output_info,
    }


def write_run_manifest(cfg: RunConfig, dirs: dict[str, Path], device: torch.device) -> None:
    manifest: dict[str, Any] = {
        "created_at": now(),
        "script": "semeval_clarity_cartography.py",
        "source_notebook": "Hijerarhijski_0.6.ipynb",
        "config": {
            "dataset_name": cfg.dataset_name,
            "model_name": cfg.model_name,
            "seeds": cfg.seeds,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "grad_accumulation": cfg.grad_accumulation,
            "learning_rate": cfg.learning_rate,
            "fine_learning_rate": cfg.fine_learning_rate,
            "regular_learning_rate": cfg.regular_learning_rate,
            "max_length": cfg.max_length,
            "mixed_precision": cfg.mixed_precision,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "freeze_embeddings": cfg.freeze_embeddings,
            "dry_run": cfg.dry_run,
            "run_regular": cfg.run_regular,
            "run_hierarchical": cfg.run_hierarchical,
            "hierarchical_threshold": cfg.hierarchical_threshold,
            "balanced_regular_loss": cfg.balanced_regular_loss,
            "subset_train": cfg.subset_train,
            "subset_test": cfg.subset_test,
        },
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        manifest["gpu"] = {"name": props.name, "total_memory_gb": props.total_memory / 1e9}
    atomic_json_dump(manifest, dirs["results"] / "run_manifest_clarity_cartography.json")


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)
    dirs = {
        "results": cfg.output_dir / "results",
        "plots": cfg.output_dir / "plots",
        "examples": cfg.output_dir / "examples",
        "logs": cfg.output_dir / "logs",
        "partial": cfg.output_dir / "partial",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    base_seed = cfg.seeds[0]
    set_seed(base_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    write_run_manifest(cfg, dirs, device)

    t0 = time.time()
    print("=" * 80, flush=True)
    print("SEMEVAL/QEVASION CLARITY CARTOGRAPHY", flush=True)
    print("=" * 80, flush=True)
    print(f"Time: {now()}", flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)", flush=True)
    print(f"Model: {cfg.model_name}", flush=True)
    print(f"Output: {cfg.output_dir}", flush=True)
    print(f"Seeds: {cfg.seeds}", flush=True)

    print("\nLoading dataset and tokenizer...", flush=True)
    raw_dataset = load_dataset(cfg.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_examples = prepare_examples(raw_dataset, "train")
    test_examples = prepare_examples(raw_dataset, "test")
    train_examples = stratified_subset(train_examples, cfg.subset_train, base_seed)
    test_examples = stratified_subset(test_examples, cfg.subset_test, base_seed + 1)
    atomic_to_csv(examples_metadata(train_examples), dirs["results"] / "qevasion_clarity_train_metadata.csv")
    atomic_to_csv(examples_metadata(test_examples), dirs["results"] / "qevasion_clarity_test_metadata.csv")
    print(f"Train examples: {len(train_examples)}", flush=True)
    print(f"Test examples: {len(test_examples)}", flush=True)
    print(f"Clarity labels: {CLARITY_MAPPING}", flush=True)

    variant_summaries: list[dict[str, Any]] = []
    summaries_by_variant: dict[str, list[dict[str, Any]]] = {}
    for seed in cfg.seeds:
        seed_cfg = replace(cfg, seed=seed)
        set_seed(seed)
        print("\n" + "-" * 80, flush=True)
        print(f"Seed {seed}", flush=True)
        if seed_cfg.run_regular:
            summary = run_regular(seed_cfg, tokenizer, train_examples, test_examples, device, dirs)
            variant_summaries.append(summary)
            summaries_by_variant.setdefault(summary["variant"], []).append(summary)
        if seed_cfg.run_hierarchical:
            summary = run_hierarchical(seed_cfg, tokenizer, train_examples, test_examples, device, dirs)
            variant_summaries.append(summary)
            summaries_by_variant.setdefault(summary["variant"], []).append(summary)

    aggregate_outputs: list[dict[str, Any]] = []
    for variant, summaries in summaries_by_variant.items():
        seed_dfs: dict[int, pd.DataFrame] = {}
        completed_seeds: list[int] = []
        for item in summaries:
            seed = int(item["seed"])
            carto_path = Path(item["outputs"]["carto_csv"])
            if carto_path.exists():
                seed_dfs[seed] = pd.read_csv(carto_path)
                completed_seeds.append(seed)
        if not completed_seeds:
            continue
        pred_path = Path(summaries[0]["outputs"]["test_predictions_csv"])
        pred_df = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
        aggregate_outputs.append(
            export_multi_seed_outputs(
                variant,
                seed_dfs,
                completed_seeds,
                pred_df,
                dirs["results"],
                dirs["plots"],
                dirs["examples"],
                cfg.export_example_limit,
            )
        )

    all_metrics = []
    for summary in variant_summaries:
        all_metrics.extend(summary.get("metrics", []))
        all_metrics.append(summary.get("final_train_metrics", {}))
        all_metrics.append(summary.get("final_test_metrics", {}))
    metrics_df = pd.DataFrame([row for row in all_metrics if row])
    atomic_to_csv(metrics_df, dirs["results"] / "qevasion_clarity_all_metrics.csv")

    summary = {
        "completed_at": now(),
        "elapsed_seconds": time.time() - t0,
        "dataset": cfg.dataset_name,
        "model": cfg.model_name,
        "seeds": cfg.seeds,
        "seeds_completed": sorted({int(item["seed"]) for item in variant_summaries}),
        "epochs": cfg.epochs,
        "n_train_examples": len(train_examples),
        "n_test_examples": len(test_examples),
        "source_notebook": "Hijerarhijski_0.6.ipynb",
        "clarity_mapping": CLARITY_MAPPING,
        "aggregate_outputs": aggregate_outputs,
        "variants": variant_summaries,
    }
    atomic_json_dump(summary, dirs["results"] / "summary_semeval_clarity_cartography.json")

    print("\n" + "=" * 80, flush=True)
    print("CLARITY CARTOGRAPHY COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Elapsed: {(time.time() - t0) / 3600:.2f}h", flush=True)
    print(f"Results: {dirs['results']}", flush=True)
    print(f"Plots: {dirs['plots']}", flush=True)
    print(f"Examples: {dirs['examples']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        fallback_out = Path(os.environ.get("OUTPUT_DIR", "./output")) / "clarity_cartography"
        try:
            write_crash_report(fallback_out, exc)
        finally:
            print("\nCLARITY CARTOGRAPHY FAILED", flush=True)
            print(f"Crash report: {fallback_out / 'logs' / 'crash_report.json'}", flush=True)
        raise
