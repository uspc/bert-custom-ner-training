from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

MODEL_PATH = "./results/final_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

id_to_label = model.config.id2label


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    results = []

    for token, pred in zip(tokens, predictions):
        label = id_to_label[pred]

        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            results.append((token, label))

    return results


if __name__ == "__main__":
    text = "High failure in Oracle CRM authentication service with error HTTP_500"

    preds = predict(text)

    print("\n=== PREDICTIONS ===")
    for token, label in preds:
        print(f"{token:15} → {label}")
