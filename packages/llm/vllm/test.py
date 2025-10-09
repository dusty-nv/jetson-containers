#!/usr/bin/env python3
import gc
import torch

print('testing vLLM...')
from vllm import LLM
import xgrammar

def clear_memory():
    """Free GPU and CPU memory before launching the model."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        gc.collect()
        print("✅ Memory cleared")
    except Exception as e:
        print(f"⚠️ Memory cleanup failed: {e}")

def run_inference():
    clear_memory()  # <-- Clean before creating the model
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.5,
        enforce_eager=True
    )

    prompts = [
        "Ahoy! How many helicopters can a human eat in one sitting?",
        "Avast! What's the future of AI?",
    ]

    outputs = llm.generate(prompts)
    for output in outputs:
        print(output.outputs[0].text)

if __name__ == "__main__":
    run_inference()
    print(xgrammar)

print('vLLM OK')
