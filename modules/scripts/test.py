import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
CACHE_DIR = "/mnt/data/natyke582/hf"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=CACHE_DIR
)

print("Model loaded successfully")

prompt = "Explain what a large language model is in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

print("OUTPUT:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
