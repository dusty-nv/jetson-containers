#!/usr/bin/env python3
print('testing SGLang...')
import argparse
import time
import sglang as sgl


def build_cli() -> argparse.Namespace:
    """Parse command‑line arguments."""
    cli = argparse.ArgumentParser(
        prog="sglang_demo",
        description="Interactive test harness for SGLang‑based models",
    )
    cli.add_argument(
        "-m",
        "--model",
        default="Qwen/Qwen3-4B-Base",
        help="Model identifier or local path",
    )
    cli.add_argument(
        "-g",
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to allocate for data parallelism",
    )
    cli.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    cli.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top‑p (nucleus) sampling parameter",
    )
    cli.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Upper limit on newly generated tokens",
    )
    return cli.parse_args()


def run_generation(engine: sgl.Engine, batch, params):
    """Generate text for a batch of prompts and return plain responses."""
    outputs = engine.generate(batch, params)
    return [out["text"] for out in outputs]


def tiny_benchmark():
    """Return a list of (prompt, expected_answer) tuples for sanity checks."""
    return [
        ("Complete this sentence: The capital of France is", "Paris"),
        ("What is 2+2?", "4"),
        ("Translate 'hello' to Spanish:", "hola"),
        ("Name a primary color:", ["red", "blue", "yellow"]),
    ]


def check_answer(reply: str, gold):
    """Evaluate whether *gold* appears in *reply* (case‑insensitive)."""
    if isinstance(gold, (list, tuple)):
        return any(item.lower() in reply.lower() for item in gold)
    return gold.lower() in reply.lower()


def main() -> None:
    args = build_cli()

    engine = sgl.Engine(
        model_path=args.model,
        dp_size=args.gpus,
        tp_size=1,
        device="cuda",
        context_length=1000,
    )

    prompts_gold = tiny_benchmark()
    prompts = [p for p, _ in prompts_gold]

    sampling_cfg = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_tokens,
    }

    start = time.time()
    responses = run_generation(engine, prompts, sampling_cfg)
    elapsed = time.time() - start

    correct = 0
    for idx, (prompt, gold) in enumerate(prompts_gold, start=1):
        reply = responses[idx - 1]
        passed = check_answer(reply, gold)
        correct += int(passed)

        print(f"Prompt {idx}: {prompt}")
        print(f"Reply   : {reply}")
        print(f"Expected: {gold}")
        print(f"Pass    : {passed}")
        print("-" * 60)

    print(f"Accuracy : {correct}/{len(prompts)} ({correct / len(prompts):.2%})")
    print(f"Latency  : {elapsed:.3f} seconds")

    engine.shutdown()


if __name__ == "__main__":
    main()
    print("SGLang demo completed.")

print('SGLang  OK\n')

