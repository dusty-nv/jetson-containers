#!/usr/bin/env bash

echo "Testing wyoming-whisper..."

python3 -c 'import wyoming_faster_whisper; print(wyoming_faster_whisper.__version__);'

echo "wyoming-whisper OK"
