import os
import time
import json
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from pathlib import Path
import sys
import re

# ================= CONFIG =================
MODEL_ID = "stepfun-ai/Step3-VL-10B"

DATASET_DIR = "dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
CSV_FILE = "data_eval.csv"

OUTPUT_DIR = "outputs/step3_vl_10b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SAMPLES = 200        # set to None for full dataset
MAX_NEW_TOKENS = 64
DTYPE = torch.bfloat16
# =========================================

def _patch_step3_processor_for_py39():
    """
    The Step3-VL processor uses Python 3.10 union syntax (list[int] | None).
    On Python 3.9 this raises TypeError during dynamic import.
    This helper rewrites the cached module to Optional[...] so the import works.
    """
    if sys.version_info >= (3, 10):
        return

    candidate_roots = []
    # Honor HF_HOME and TRANSFORMERS_CACHE first, then common defaults and paths seen in tracebacks.
    for env_var in ("HF_HOME", "TRANSFORMERS_CACHE"):
        if os.environ.get(env_var):
            candidate_roots.append(Path(os.environ[env_var]))
    candidate_roots.extend([
        Path.home() / ".cache" / "huggingface",
        Path("/mnt/data/natyke582/hf"),            # path from traceback
        Path("/mnt/data/home/natyke582/.cache/huggingface"),
    ])

    patched = False
    for root in candidate_roots:
        if not root.exists():
            continue
        processing_files = list(root.rglob("processing_step3.py"))
        for path in processing_files:
            try:
                text = path.read_text()
            except OSError:
                continue
            # Skip if already patched
            if "Optional[" in text:
                patched = True
                break

            new_text = text
            # Replace union syntax with Optional[...] for common list/int cases
            new_text = re.sub(r"list\[int\]\s*\|\s*None", "Optional[list[int]]", new_text)
            new_text = re.sub(r"List\[int\]\s*\|\s*None", "Optional[List[int]]", new_text)

            if new_text != text:
                # Ensure Optional is imported
                if re.search(r"from typing import", new_text):
                    def add_optional(match):
                        imports = match.group(1)
                        if "Optional" not in imports:
                            return f"from typing import Optional, {imports}"
                        return match.group(0)
                    new_text = re.sub(r"from typing import ([^\n]+)", add_optional, new_text, count=1)
                elif "import Optional" not in new_text:
                    new_text = "from typing import Optional\n" + new_text

                try:
                    path.write_text(new_text)
                    print(f"Patched Step3 processor for py39 compatibility: {path}")
                    patched = True
                    break
                except OSError:
                    continue
        if patched:
            break

    if not patched:
        print("Warning: processing_step3.py not patched (file not found).")


def main():
    print("Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True)
    except TypeError as e:
        if "unsupported operand type(s) for |" in str(e):
            _patch_step3_processor_for_py39()
            # ensure old failed module isn't cached
            for mod_name in list(sys.modules.keys()):
                if "processing_step3" in mod_name:
                    sys.modules.pop(mod_name, None)
            processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                force_download=False  # use freshly patched cache
            )
        else:
            raise

    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).eval()

    print("Model loaded successfully.")

    # ===== LOAD DATASET =====
    df = pd.read_csv(os.path.join(DATASET_DIR, CSV_FILE))

    if MAX_SAMPLES:
        df = df.head(MAX_SAMPLES)

    print(f"Loaded {len(df)} samples")

    results = []

    # ===== EVALUATION LOOP =====
    for idx, row in df.iterrows():
        image_path = os.path.join(IMAGE_DIR, row["image_id"] + ".png")
        image = Image.open(image_path).convert("RGB")

        question = row["question"]
        gt_answer = row["answer"]

        inputs = processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(model.device)

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )
        latency = time.time() - start

        prediction = processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        results.append({
            "index": int(idx),
            "image": row["image_id"],
            "question": question,
            "prediction": prediction,
            "gt_answer": gt_answer,
            "latency_sec": latency
        })

        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(df)}] processed")

    # ===== SAVE RESULTS =====
    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Evaluation finished.")
    print("Results saved to:", out_path)


if __name__ == "__main__":
    main()
