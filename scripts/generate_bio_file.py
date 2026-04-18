import json
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-cased"
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def convert_to_bio(example):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True
    )

    offsets = encoding["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    labels = ["O"] * len(tokens)

    for ent in example["entities"]:
        start = ent[0]
        end = ent[1]
        label = ent[2]

        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue

            if tok_start >= start and tok_end <= end:
                prefix = "B-" if tok_start == start else "I-"
                labels[i] = f"{prefix}{label}"

    return {
        "text": example["text"],
        "tokens": tokens,
        "labels": labels
    }

def generate_bio_file(input_path, output_path):
    data = load_jsonl(input_path)

    with open(output_path, "w") as f:
        for row in data:
            bio = convert_to_bio(row)
            f.write(json.dumps(bio) + "\n")

if __name__ == "__main__":
    generate_bio_file("custom_incidents.jsonl", "bio_dataset.jsonl")
