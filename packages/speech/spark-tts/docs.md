# SparkTTS

SparkTTS is a high-quality text-to-speech synthesis model that provides natural-sounding speech generation. This container includes the SparkTTS model running optimized for Jetson devices, offering both standard TTS functionality and zero-shot voice cloning capabilities.

## System Requirements

- **Memory**: Requires at least 5GB of available RAM

## Features

- Natural-sounding speech synthesis
- Adjustable pitch and speed
- Gender selection for standard TTS
- Zero-shot voice cloning from audio samples

## Usage Examples

When using `jetson-containers run`, the generated audio files are automatically saved in the `jetson-containers/data/audio/tts/spark-tts/` directory on your host system, and models are cached in `jetson-containers/data/models/huggingface/`.

### Standard Text-to-Speech (CLI)

Generate speech from text with customizable parameters:

```bash
jetson-containers run $(autotag spark-tts) \
    --pitch "moderate" \
    --speed "moderate" \
    --gender "female" \
    --text "The quick brown fox jumps over the lazy dog"
```

Available options:

* `--pitch`: "very_low", "low", "moderate", "high", "very_high"
* `--speed`: "very_low", "low", "moderate", "high", "very_high"
* `--gender`: "female", "male"

### Zero-shot Voice Cloning (CLI) 

Clone a voice from a sample audio file (note: the audio file must be accessible inside the container, put it in the jetson-containers/data directory):

```bash
jetson-containers run $(autotag spark-tts) \
    --prompt_speech_path "/data/audio/sample.wav" \
    --prompt_text "This is a sample prompt text that matches the audio sample..." \
    --speed "moderate" \
    --text "Hi, this is a test of voice cloning with Spark TTS!"
```

## Output Location

When using `jetson-containers run`, the following directories are automatically mounted and accessible:

* Audio output: `jetson-containers/data/audio/tts/spark-tts/`
* Model cache: `jetson-containers/data/models/huggingface/`

The generated audio files will be saved with timestamped filenames like `20250325230742.wav`.


## Model Source

This container uses the SparkTTS model from Hugging Face:
[Spark-TTS by SparkAudio](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)