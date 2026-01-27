import os
import time
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ===== CONFIG =====
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
CACHE_DIR = "/mnt/data/natyke582/hf"

DATASET_DIR = "dataset"
CSV_FILE = "data_eval.csv"

OUTPUT_DIR = "outputs/deepseek_text_baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        cache_dir=CACHE_DIR
    )
    model.eval()

    print("Model loaded:", model.config._name_or_path)

    # ===== LOAD CSV =====
    df = pd.read_csv(os.path.join(DATASET_DIR, CSV_FILE))
    print(f"Loaded {len(df)} samples")

    results = []
    MAX_SAMPLES = 200
    # ===== TEXT-ONLY LOOP =====
    for idx, row in df.head(MAX_SAMPLES).iterrows():
        question = row["question"]

        prompt = (
            "Answer the following question as best as you can.\n\n"
            f"Question: {question}\nAnswer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64
            )
        latency = time.time() - start

        answer = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        results.append({
            "index": idx,
            "question": question,
            "prediction": answer,
            "gt_answer": row.get("answer", None),
            "latency_sec": latency
        })

        if idx % 10 == 0:
            print(f"[{idx}/{len(df)}] processed")

    # ===== SAVE =====
    out_file = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved results to:", out_file)


if __name__ == "__main__":
    main()
