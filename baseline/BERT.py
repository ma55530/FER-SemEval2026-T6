from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, Trainer, set_seed, TrainingArguments
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = load_dataset("ailsntua/QEvasion")
print(dataset)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
print(dataset["train"][0])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(example):
    questions = example["interview_question"]
    answers = example["interview_answer"]
    combined = [q + ", Answer: " + a for q, a in zip(questions, answers)]
    return tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=256,
    )

tokenized_train = train_dataset.map(preprocess, batched=True, batch_size=32)
tokenized_test = test_dataset.map(preprocess, batched=True, batch_size=32)


tokenized_train.set_format("torch", columns=["input_ids", "attention_mask"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask"])

label_map = {
    "Clear Reply": 0,
    "Ambivalent": 1,
    "Clear Non-Reply": 2
}

def map_labels(example):
    example["labels"] = label_map[example["clarity_label"]]
    return example

tokenized_train = tokenized_train.map(map_labels)
tokenized_test = tokenized_test.map(map_labels)

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_name = "bert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)
model.to(device)

device_id = 0 if torch.cuda.is_available() else -1
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device_id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    cm = confusion_matrix(labels, preds).tolist()

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted"),
        "confusion_matrix": cm
    }

training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

NUM_RUNS = 5

metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

for run in range(NUM_RUNS):
    seed = 42 + run
    set_seed(seed)
    torch.manual_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    )
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()

    metrics["accuracy"].append(eval_results["eval_accuracy"])
    metrics["precision"].append(eval_results["eval_precision"])
    metrics["recall"].append(eval_results["eval_recall"])
    metrics["f1"].append(eval_results["eval_f1"])

    print(
        f"Run {run + 1}: "
        f"Acc={eval_results['eval_accuracy']:.4f}, "
        f"Prec={eval_results['eval_precision']:.4f}, "
        f"Rec={eval_results['eval_recall']:.4f}, "
        f"F1={eval_results['eval_f1']:.4f}"
    )

print("\nFINAL RESULTS (5 starts)")
print("=" * 60)

for name, values in metrics.items():
    values = np.array(values)
    print(f"{name.capitalize()}:")
    print("  Scores:", [f"{v:.4f}" for v in values])
    print(f"  Avg ± Std: {values.mean():.4f} ± {values.std():.4f}")
    if name == "f1":
        print(f"  Min: {values.min():.4f}, Max: {values.max():.4f}")
    print()