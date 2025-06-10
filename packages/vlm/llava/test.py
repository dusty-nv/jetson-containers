#!/usr/bin/env python3
import argparse

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

parser = argparse.ArgumentParser()

parser.add_argument('--model-path', type=str, default='SaffalPoosh/llava-llama-2-7B-merged')
parser.add_argument('--model-base', type=str, default=None)

args = parser.parse_args()
print(args)

model = load_pretrained_model(
    model_path=args.model_path,
    model_base=args.model_base,
    model_name=get_model_name_from_path(args.model_path)
)

print(model)
