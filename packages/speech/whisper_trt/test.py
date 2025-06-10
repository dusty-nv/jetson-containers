#!/usr/bin/env python3
# https://github.com/NVIDIA-AI-IOT/whisper_trt
from whisper_trt import load_trt_model

MODEL="tiny.en"
AUDIO="/data/audio/dusty.wav"

print(f"Loading whisper {MODEL} with TensorRT (this could take a few minutes)")

model = load_trt_model("tiny.en", verbose=True)  # base.en small.en

print(f"Transcribing {AUDIO}")

result = model.transcribe("/data/audio/dusty.wav") # or pass numpy array

print(result['text'])
