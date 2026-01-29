import json
from evaluate import load
import numpy as np

# ===== CONFIG =====
RESULTS_JSON = "/mnt/data/home/natyke582/Deep_Learning_2/outputs/internvl3_5_eval/results.json"
BERT_MODEL = "distilbert-base-uncased"
LANG = "en"
# ==================

# 1️⃣ Load JSON
with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2️⃣ Extract predictions & references
predictions = []
references = []

for item in data:
    pred = item["prediction"]
    ref = item["gt_answer"]

    # Ensure strings
    predictions.append(str(pred))
    references.append(str(ref))

print(f"Loaded {len(predictions)} samples")

# 3️⃣ Load BERTScore
bertscore = load("bertscore")

# 4️⃣ Compute BERTScore
results = bertscore.compute(
    predictions=predictions,
    references=references,
    model_type=BERT_MODEL,
    rescale_with_baseline=True,
    lang=LANG
)

# 5️⃣ Aggregate metrics
precision = np.mean(results["precision"])
recall = np.mean(results["recall"])
f1 = np.mean(results["f1"])

# 6️⃣ Print results
print("\n===== BERTScore Results =====")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
