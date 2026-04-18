import json
from transformers import pipeline, AutoTokenizer

MODEL_PATH = "./results/final_model"

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

ner_pipeline = pipeline(
    "token-classification",
    model=MODEL_PATH,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

def run_inference(samples, n=5):
    for i, sample in enumerate(samples[:n]):
        print(f"\n--- Incident {i+1} ---")
        print("TEXT:", sample["text"])

        preds = ner_pipeline(sample["text"])

        print("\nExtracted Entities:")
        for p in preds:
            print(f"{p['entity_group']:20} | {p['word']:30} | score={p['score']:.2f}")

raw_data = load_jsonl("custom_incidents.jsonl")
run_inference(raw_data, n=10)
