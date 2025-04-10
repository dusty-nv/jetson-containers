#!/usr/bin/env python3
import argparse
import ollama

DEFAULT_MODEL = "smollm2:135m-instruct-q2_K"
DEFAULT_PROMPT = "Test"

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL)
parser.add_argument('-p', '--prompt', type=str, default=DEFAULT_PROMPT)
args = parser.parse_args()

print(args)

ollama.pull(args.model)
ollama.chat(model=args.model, messages=[{"role":"user", "content":args.prompt}])
