import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
from seqeval.metrics import classification_report, f1_score
import inspect
import transformers

print(transformers.__version__)
print(transformers.__file__)
print(TrainingArguments)
print(inspect.signature(TrainingArguments.__init__))

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "bert-base-cased"
MAX_LENGTH = 128

label_list = [
    "O",
    "B-SEVERITY", "I-SEVERITY",
    "B-INCIDENT_TYPE", "I-INCIDENT_TYPE",
    "B-APPLICATION", "I-APPLICATION",
    "B-SYSTEM_COMPONENT", "I-SYSTEM_COMPONENT",
    "B-ERROR_CODE", "I-ERROR_CODE",
    "B-IMPACT", "I-IMPACT",
    "B-ENVIRONMENT", "I-ENVIRONMENT",
    "B-REGION", "I-REGION",
    "B-TIME", "I-TIME",
    "B-DATE", "I-DATE",
    "B-TICKET_ID", "I-TICKET_ID",
    "B-TEAM", "I-TEAM",
    "B-ROOT_CAUSE", "I-ROOT_CAUSE"
]

label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for l, i in label_to_id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -----------------------------
# Load JSONL
# -----------------------------
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

# -----------------------------
# Convert char spans → token labels
# -----------------------------
def tokenize_and_align(example):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True
    )

    offsets = encoding["offset_mapping"]
    labels = ["O"] * len(offsets)

    # Assign labels based on span containment
    for ent in example["entities"]:
        start = ent["start"]
        end = ent["end"]
        label = ent["label"]
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue

            if tok_start >= start and tok_end <= end:
                prefix = "B-" if tok_start == start else "I-"
                tag = f"{prefix}{label}"

                if tag not in label_to_id:
                    raise ValueError(f"Unknown label: {tag}")

                labels[i] = tag

    # Convert to IDs
    label_ids = []
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            label_ids.append(-100)
        else:
            label_ids.append(label_to_id[labels[i]])

    encoding["labels"] = label_ids
    encoding.pop("offset_mapping")

    return encoding

# -----------------------------
# Dataset Prep
# -----------------------------
def prepare_dataset(path):
    raw = load_jsonl(path)

    # 🔥 FIX: normalize entities
    for row in raw:
        row["entities"] = [
            {"start": e[0], "end": e[1], "label": e[2]}
            for e in row["entities"]
        ]

    dataset = Dataset.from_list(raw)

    dataset = dataset.map(
        tokenize_and_align,
        remove_columns=dataset.column_names
    )

    return dataset.train_test_split(test_size=0.1)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred, lab in zip(predictions, labels):
        cur_preds = []
        cur_labels = []

        for p_, l_ in zip(pred, lab):
            if l_ != -100:
                cur_preds.append(id_to_label[p_])
                cur_labels.append(id_to_label[l_])

        true_predictions.append(cur_preds)
        true_labels.append(cur_labels)

    return {
        "f1": f1_score(true_labels, true_predictions)
    }

# -----------------------------
# Load Dataset
# -----------------------------
dataset = prepare_dataset("custom_incidents.jsonl")

# -----------------------------
# Model
# -----------------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id_to_label,
    label2id=label_to_id
)

# -----------------------------
# Training Args
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50
)

# -----------------------------
# Data Collator
# -----------------------------
data_collator = DataCollatorForTokenClassification(tokenizer)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# -----------------------------
# Train
# -----------------------------
trainer.train()
trainer.save_model("./results/final_model")
tokenizer.save_pretrained("./results/final_model")
