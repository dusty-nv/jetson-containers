# Speaches

Speaches supports VAD (Voice Activity Detection), STT (Speech-to-Text) and TTS (Text-to-Speech).

## What is VAD and why it's needed

Voice Activity Detection (VAD) is used to detect the presence or absence of human speech in audio streams.
It is important when processing speech because by identifying when someone is actually speaking, you prevent unnecessary processing of silence or background noise, reducing computational overhead.

## Important Notes on TTS Implementation

At this time of adding support to Speaches, Kokoro uses `onnxruntime` and `kokoro-onnx` for TTS.

This backend is currently slower than `kokoro-tts:fastapi`, which also implements the OpenAI protocol, and can be used separately.


## Build

```bash
LSB_RELEASE=24.04 jetson-containers build speaches
```

or

```bash
LSB_RELEASE=24.04 jetson-containers build --build-args=PIP_RETRIES:10,PIP_TIMEOUT:60 speaches
```

## Contributing

If you need TTS endpoint support, please open an issue and tag @OriNachum

