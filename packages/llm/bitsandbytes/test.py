#!/usr/bin/env python3
import gc
import torch

def clear_memory():
    """Free GPU and CPU memory before running the model."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        gc.collect()
        print("✅ Memory cleared")
    except Exception as e:
        print(f"⚠️ Memory cleanup failed: {e}")

# ---- Memory Cleanup Before Model Load ----
clear_memory()

import bitsandbytes
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from threading import Thread

print("transformers version:", transformers.__version__)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"loading {model_name} with bitsandbytes (8-bit)")

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",                 # <-- important (not "cuda")
    quantization_config=bnb_config,    # <-- v5 way
    trust_remote_code=True,
    torch_dtype=torch.float16,         # good default on GPU
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextIteratorStreamer(tokenizer)

prompt = [{"role": "user", "content": "Can I get a recipe for French Onion soup?"}]

if hasattr(tokenizer, "apply_chat_template"):
    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
).to(model.device)
else:
    inputs = tokenizer(
        "Once upon a time, in a land far far away, ",
        return_tensors="pt",
    ).input_ids.to(model.device)

Thread(target=lambda: model.generate(**inputs, max_new_tokens=64, streamer=streamer)).start()

for text in streamer:
    print(text, end="", flush=True)

print(f"\n\ndone testing bitsandbytes with {model_name}")
