import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import f1_score

MODEL_NAME = "bert-base-cased"

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
# Load BIO dataset
# -----------------------------
def load_bio(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

# -----------------------------
# Convert to HF format
# -----------------------------
def prepare_dataset(path):
    data = load_bio(path)

    input_ids = []
    labels = []

    for row in data:
        tokens = row["tokens"]
        bio_labels = row["labels"]

        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=128
        )

        label_ids = []

        for i in range(len(enc["input_ids"])):
            if i < len(bio_labels):
                label_ids.append(label_to_id[bio_labels[i]])
            else:
                label_ids.append(-100)

        enc["labels"] = label_ids

        input_ids.append(enc)

    return Dataset.from_list(input_ids).train_test_split(test_size=0.1)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds = []
    true_labels = []

    for pred, lab in zip(predictions, labels):
        cur_preds = []
        cur_labels = []

        for p_, l_ in zip(pred, lab):
            if l_ != -100:
                cur_preds.append(id_to_label[p_])
                cur_labels.append(id_to_label[l_])

        true_preds.append(cur_preds)
        true_labels.append(cur_labels)

    return {"f1": f1_score(true_labels, true_preds)}

# -----------------------------
# Load dataset
# -----------------------------
dataset = prepare_dataset("bio_dataset.jsonl")

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
# Training
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./results/final_model")
tokenizer.save_pretrained("./results/final_model")
