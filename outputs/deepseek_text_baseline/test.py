import json
from evaluate import load
import numpy as np

# ---------- CONFIG ----------
RESULTS_PATH = "/mnt/data/home/natyke582/Deep_Learning_2/outputs/deepseek_text_baseline/results.json"
MODEL_TYPE = "distilbert-base-uncased"  # швидко і стабільно
LANG = "en"

# ---------- FILTER ----------
BAD_PHRASES = [
    "i can't see",
    "i cannot see",
    "i don't have access",
    "cannot answer",
    "i am not sure",
    "i don't know"
]

def is_valid_prediction(pred):
    pred = pred.lower()
    return not any(bad in pred for bad in BAD_PHRASES)

# ---------- LOAD DATA ----------
with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

predictions = []
references = []

for item in data:
    pred = item["prediction"].strip()
    gt = item["gt_answer"].strip()

    if is_valid_prediction(pred):
        predictions.append(pred)
        references.append(gt)

print(f"Total samples: {len(data)}")
print(f"Valid for BERTScore: {len(predictions)}")

# ---------- BERTSCORE ----------
bertscore = load("bertscore")

results = bertscore.compute(
    predictions=predictions,
    references=references,
    model_type=MODEL_TYPE,
    lang=LANG,
    rescale_with_baseline=True
)

# ---------- REPORT ----------
precision = np.mean(results["precision"])
recall = np.mean(results["recall"])
f1 = np.mean(results["f1"])

print("\n=== BERTScore (DeepSeek) ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
