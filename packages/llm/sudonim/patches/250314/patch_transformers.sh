#!/usr/bin/env bash
set -ex

gemma_path="/opt/venv/lib/python3.12/site-packages/transformers/models/gemma3"
gemma_file="modeling_gemma3.py"

uv pip install --force-reinstall transformers-*.whl
patch $gemma_path/$gemma_file $gemma_file.diff

uv pip install -r requirements.txt
