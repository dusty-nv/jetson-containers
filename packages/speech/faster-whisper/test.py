from faster_whisper import WhisperModel

model = "small.en"
# model = WhisperModel(model, device="cuda", compute_type="float16")       # Run on GPU with FP16
model = WhisperModel(model, device="cuda", compute_type="int8_float16")  # or run on GPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")      # or run on CPU with INT8

segments, info = model.transcribe("/data/audio/dusty.wav", beam_size=5)
print(f"Detected language '{info.language}' with probability {info.language_probability}")

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))