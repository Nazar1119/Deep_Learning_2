import os
import time
import json
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ================= CONFIG =================
MODEL_PATH = "/mnt/data/natyke582/models/InternVL3_5-30B-A3B"

DATASET_DIR = "dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
CSV_FILE = "data_eval.csv"

OUTPUT_DIR = "outputs/internvl3_5_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SAMPLES = 1000
MAX_NEW_TOKENS = 64
DTYPE = torch.bfloat16
IMAGE_SIZE = 448
MAX_IMAGE_TILES = 12
RUN_OLLAMA_SNIPPET = False
# =========================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    """Preprocessing pipeline aligned with InternVL3.5 quick-start."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=IMAGE_SIZE, use_thumbnail=True):
    """Tile the image so longer sides keep more detail (same logic as InternVL)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image(image_file: str, input_size: int = IMAGE_SIZE, max_num: int = MAX_IMAGE_TILES):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(tile) for tile in tiles]
    return torch.stack(pixel_values)


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )

    # 🔥 REQUIRED: register image token
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<image>"]}
        )

    print("Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=DTYPE,
        trust_remote_code=True
    ).eval()

    model.resize_token_embeddings(len(tokenizer))
    print("Model loaded.")

    # ===== LOAD DATA =====
    df = pd.read_csv(os.path.join(DATASET_DIR, CSV_FILE))
    if MAX_SAMPLES:
        df = df.head(MAX_SAMPLES)
    eval_df = df

    results = []

    # ===== INFERENCE LOOP =====
    for idx, row in df.iterrows():
        image_path = os.path.join(IMAGE_DIR, row["image_id"] + ".png")
        pixel_values = load_image(image_path)
        pixel_values = pixel_values.to(dtype=DTYPE)
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()

        question = row["question"]
        # HARD guarantee it's a string
        if not isinstance(question, str):
            question = str(question)

        gt_answer = row["answer"]

        start = time.time()
        with torch.no_grad():
            prediction = model.chat(
                tokenizer,
                pixel_values,      # ✅ tensor of image patches
                question,          # ✅ text prompt
                {
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "do_sample": False
                }
            )

        latency = time.time() - start

        results.append({
            "index": int(idx),
            "image_id": row["image_id"],
            "question": question,
            "prediction": prediction,
            "gt_answer": gt_answer,
            "latency_sec": round(latency, 3)
        })

        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(df)}] done")

    # ===== SAVE =====
    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved results to:", out_path)

    if RUN_OLLAMA_SNIPPET:
        from tqdm import tqdm
        from pathlib import Path
        from modules.llm_providers.views import ollama as provider
        from modules.llm_providers.models.Answer import ImageAnswer
        from modules.llm_providers.models.ollama import OllamaOptions

        predictions, references = list(), list()
        data_gen = eval_df.iterrows()

        for idx, row in tqdm(data_gen, desc="LLM answering", total=len(eval_df)):

            answer = provider.answer(
                query = ImageAnswer(
                    query=row['question'],
                    paths=[Path('../../data/dataset-1/images').resolve() / f"{row['image_id']}.png"],
                    other_dict=[{
                        'role': 'system',
                        'content': "\n".join([
                            "# Role",
                            "You are an AI assitant, that answer on user question for provided image",
                            "",
                            "# Instructions",
                            "1. Answer short and clear.",
                            f"2. Answer in {len(row['answer'].split(' '))} word, that exactly answer on user question",
                            "",
                        ])
                    }]
                ),
                model="llava:13b",
                options=OllamaOptions(
                    temperature=0
                )
            )
            
            predictions.append(answer.answer)
            references.append(row['answer'])


if __name__ == "__main__":
    main()
