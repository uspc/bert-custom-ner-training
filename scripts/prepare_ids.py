import json
from datasets import Dataset
from transformers import AutoTokenizer

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_bio(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def convert_to_ids(row):
    tokens = row["tokens"]
    bio_labels = row["labels"]

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=False
    )

    word_ids = enc.word_ids()  

    label_ids = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label = bio_labels[word_idx]

            # If same word split into multiple tokens → convert B- to I-
            if word_idx == previous_word_idx:
                if label.startswith("B-"):
                    label = "I-" + label[2:]

            label_ids.append(label_to_id[label])
        previous_word_idx = word_idx

    enc["labels"] = label_ids
    return enc


def prepare_and_save(input_path, output_dir):
    data = load_bio(input_path)

    dataset = Dataset.from_list(data)
    dataset = dataset.map(convert_to_ids)

    dataset = dataset.train_test_split(test_size=0.1)

    # ✅ Save dataset
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    prepare_and_save("bio_dataset.jsonl", "processed_dataset")
