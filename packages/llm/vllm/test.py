#!/usr/bin/env python3

print('testing vLLM...')
from vllm import LLM
import xgrammar

def run_inference():
    llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.5)

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
