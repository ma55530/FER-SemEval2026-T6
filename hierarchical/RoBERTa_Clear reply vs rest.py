import os
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

os.environ["WANDB_DISABLED"] = "true"
dataset = load_dataset("ailsntua/QEvasion")

def prepare_labels(row):
    mapping = {"Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2}
    label = mapping[row["clarity_label"]]
    return {
        "label": label,
        "binary_label": 0 if label == 0 else 1
    }

dataset = dataset.map(prepare_labels)
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

def tokenize_function(example):
    texts = [
        "Question: " + q + " Answer: " + a
        for q, a in zip(example["interview_question"], example["interview_answer"])
    ]
    return tokenizer(texts, padding="max_length", max_length=512, truncation=True)

tokenized_ds = dataset.map(tokenize_function, batched=True)
tokenized_ds.set_format("torch")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics_binary(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, average="macro")
    }

class WeightedTrainer(Trainer):
    def __init__(self, weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

results_binary = {"f1": [], "accuracy": [], "precision": [], "recall": []}
results_cascade = {"f1": [], "accuracy": [], "precision": [], "recall": []}

for run in range(5):
    current_seed = 42 + run
    set_seed(current_seed)
    print(f"\n{'='*50}\nRUN {run + 1}/5\n{'='*50}")

    binary_train = tokenized_ds["train"].map(lambda x: {"labels": x["binary_label"]})
    binary_test  = tokenized_ds["test"].map(lambda x: {"labels": x["binary_label"]})

    b_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(binary_train["labels"])),
        dtype=torch.float
    )

    binary_model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/roberta-base", num_labels=2
    )

    trainer_b = WeightedTrainer(
        model=binary_model,
        weights=b_weights,
        train_dataset=binary_train,
        eval_dataset=binary_test,
        compute_metrics=compute_metrics_binary,
        args=TrainingArguments(
            output_dir=f"output_b_run{run}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=True,
            report_to="none"
        )
    )
    trainer_b.train()
    binary_model = trainer_b.model
    binary_model.eval()
    binary_model.to(device)

    binary_preds = []
    binary_true = list(tokenized_ds["test"]["binary_label"])

    for item in tokenized_ds["test"]:
        with torch.no_grad():
            inputs = {k: v.unsqueeze(0).to(device)
                      for k, v in item.items()
                      if k in ["input_ids", "attention_mask"]}
            pred = torch.argmax(binary_model(**inputs).logits, dim=-1).item()
            binary_preds.append(pred)

    results_binary["f1"].append(f1_score(binary_true, binary_preds, average="macro"))
    results_binary["accuracy"].append(accuracy_score(binary_true, binary_preds))
    results_binary["precision"].append(precision_score(binary_true, binary_preds, average="macro", zero_division=0))
    results_binary["recall"].append(recall_score(binary_true, binary_preds, average="macro", zero_division=0))

    # label==1 (Ambivalent) -> 0,  label==2 (Clear Non-Reply) -> 1
    fine_train_data = tokenized_ds["train"].filter(lambda x: x["label"] != 0)
    fine_test_data  = tokenized_ds["test"].filter(lambda x: x["label"] != 0)

    fine_train = fine_train_data.map(lambda x: {"labels": 0 if x["label"] == 1 else 1})
    fine_test  = fine_test_data.map(lambda x: {"labels": 0 if x["label"] == 1 else 1})

    f_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(fine_train["labels"])),
        dtype=torch.float
    )

    fine_model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/roberta-base", num_labels=2
    )

    trainer_f = WeightedTrainer(
        model=fine_model,
        weights=f_weights,
        train_dataset=fine_train,
        eval_dataset=fine_test,
        compute_metrics=compute_metrics_binary,
        args=TrainingArguments(
            output_dir=f"output_f_run{run}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=True,
            report_to="none"
        )
    )
    trainer_f.train()

    fine_model = trainer_f.model
    fine_model.eval()
    fine_model.to(device)

    cascade_preds = []
    true_labels = list(tokenized_ds["test"]["label"])

    for item in tokenized_ds["test"]:
        with torch.no_grad():
            inputs = {k: v.unsqueeze(0).to(device)
                      for k, v in item.items()
                      if k in ["input_ids", "attention_mask"]}

            b_pred = torch.argmax(binary_model(**inputs).logits, dim=-1).item()

            if b_pred == 0:
                cascade_preds.append(0)  # Clear Reply
            else:
                f_pred = torch.argmax(fine_model(**inputs).logits, dim=-1).item()
                # f_pred==0 -> Ambivalent (1), f_pred==1 -> Clear Non-Reply (2)
                cascade_preds.append(1 if f_pred == 0 else 2)

    results_cascade["f1"].append(f1_score(true_labels, cascade_preds, average="macro"))
    results_cascade["accuracy"].append(accuracy_score(true_labels, cascade_preds))
    results_cascade["precision"].append(precision_score(true_labels, cascade_preds, average="macro", zero_division=0))
    results_cascade["recall"].append(recall_score(true_labels, cascade_preds, average="macro", zero_division=0))

print(f"\nRun {run+1} — Binary (Clear Reply vs Rest):")
print(classification_report(binary_true, binary_preds,
                                target_names=["Clear Reply", "Rest"], zero_division=0))
print(f"Run {run+1} — Cascade (3-class):")
print(classification_report(true_labels, cascade_preds,
                                target_names=["Clear Reply", "Ambivalent", "Clear Non-Reply"],
                                zero_division=0))

for task_name, results in [("Clear Reply vs Rest (Binary)", results_binary),
                            ("Cascade 3-class classifier", results_cascade)]:
    print(f"\n{'='*75}")
    print(f"FINAL RESULTS — {task_name}, RoBERTa-base, 5 runs")
    print(f"{'='*75}")
    print(f"{'Metric':<15} {'Min':<12} {'Max':<12} {'Avg ± Std':<25}")
    print(f"{'-'*75}")
    for metric_name in ["f1", "accuracy", "precision", "recall"]:
        vals = results[metric_name]
        avg, std = np.mean(vals), np.std(vals)
        display = "F1 Score" if metric_name == "f1" else metric_name.capitalize()
        print(f"{display:<15} {np.min(vals):<12.4f} {np.max(vals):<12.4f} {avg:.4f} ± {std:.4f}")
    print(f"{'='*75}")
