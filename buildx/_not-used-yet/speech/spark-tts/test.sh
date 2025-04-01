#!/usr/bin/env bash

cd /opt/spark-tts

python3 inference.py --text "Hi, I'm Spark-TTS" --gender male --pitch moderate --speed moderate