#!/usr/bin/env python3
import sys
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams

def run_gguf_inference(model_path: str):
    """
    Loads the GGUF LLaMA model at model_path into vLLM, then runs two toy prompts
    with a simple “pirate-style” system message.
    """
    PROMPT_TEMPLATE = "<|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    system_message = "You are a friendly chatbot who always responds in the style of a pirate."

    prompts = [
        "How many helicopters can a human eat in one sitting?",
        "What's the future of AI?",
    ]
    chat_prompts = [
        PROMPT_TEMPLATE.format(system_message=system_message, prompt=p)
        for p in prompts
    ]

    sampling_params = SamplingParams(temperature=0, max_tokens=128)

    # Initialize vLLM with 50% GPU memory.
    try:
        llm = LLM(
            model=model_path,
            tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            gpu_memory_utilization=0.5,
        )
    except Exception as e:
        print(f"Error initializing vLLM: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        outputs = llm.generate(chat_prompts, sampling_params)
    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr)
        sys.exit(1)

    for output in outputs:
        # Show the raw prompt and the pirate-style reply.
        print("=== Prompt ===")
        print(output.prompt)
        print("--- Pirate Reply ---")
        print(output.outputs[0].text)
        print()

if __name__ == "__main__":
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    filename = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

    try:
        model_path = hf_hub_download(repo_id, filename=filename)
    except Exception as e:
        print(f"Failed to download model {filename} from {repo_id}: {e}", file=sys.stderr)
        sys.exit(1)

    run_gguf_inference(model_path)
    print("vLLM GGUF OK")