# Speaches

Speaches supports VAD (Voice Activity Detection), STT (Speech-to-Text) and TTS (Text-to-Speech) with GPU acceleration through CTranslate2.

## Key Features

- **GPU-accelerated inference** using custom-built CTranslate2 with CUDA 12.8 and cuDNN 9 support
- **OpenAI-compatible API** for easy integration
- **Whisper support** for speech-to-text through CTranslate2
- **TTS support** with various voice models

## Build Information

This container builds CTranslate2 from source with:
- CUDA 12.8 support
- cuDNN 9 for optimized inference
- OpenBLAS for CPU operations
- GNU OpenMP for parallelization

The speaches server is installed using `uv` with all extras for complete functionality.

## Build

```bash
jetson-containers build speaches
```

To specify a custom tag:
```bash
jetson-containers build --name speaches:cuda12.8-py3.12 speaches
```


## Run

Run the container with:
```bash
jetson-containers run $(autotag speaches)
```

Or directly with Docker:
```bash
docker run -it --rm --network=host --runtime nvidia $(autotag speaches)
```

The speaches API will be available at:
```
http://localhost:8000/
```

## Model Downloads

By default, no models will be downloaded. Run the script below to download the default models:

```bash
jetson-containers run $(autotag speaches) /workspace/speaches/download-models.sh
```

This will download:
- `Systran/faster-whisper-large-v3` - For speech-to-text
- `speaches-ai/Kokoro-82M-v1.0-ONNX-fp16` - For text-to-speech

To download additional models, you can either edit the script above by adding the models you want, or run the following from within the container:
```bash
uvx speaches-cli model download model_name
```

To see supported models, run the following from the container:

```bash
uvx speaches-cli registry ls --task automatic-speech-recognition
uvx speaches-cli registry ls --task text-to-speech # For TTS
```





## API Endpoints

- `/v1/audio/transcriptions` - Speech-to-text (Whisper compatible)
- `/v1/audio/speech` - Text-to-speech
- `/v1/audio/speech/voices` - List available TTS voices



## Notes

- The container uses a Python virtual environment managed by `uv`
- CTranslate2 is built from source for optimal Jetson performance
- The server runs with Uvicorn in production mode

## Important Notes on TTS Implementation

At this time of adding support to Speaches, Kokoro uses `onnxruntime` and `kokoro-onnx` for TTS.

This backend is currently slower than `kokoro-tts:fastapi`, which also implements the OpenAI protocol, and can be used separately.

## Contributing

If you need TTS endpoint support, please open an issue and tag @OriNachum

