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

### Standard Text-to-Speech

Generate speech from text with customizable parameters:

```bash
python3 /opt/spark-tts/inference.py \
    --pitch "moderate" \
    --speed "moderate" \
    --gender "female" \
    --text "The Quick brown fox jumps over the lazy dog"
```

Available options:
- `--pitch`: "very_low", "low", "moderate", "high", "very_high"
- `--speed`: "very_slow", "slow", "moderate", "high", "very_high"
- `--gender`: "female", "male"

### Zero-shot Voice Cloning

Clone a voice from a sample audio file:

```bash
python3 /opt/spark-tts/inference.py \
    --prompt_speech_path "/data/audio/dusty.wav" \
    --prompt_text "Hi, this is Dusty. Check, 1, 2, 3. What's the weather going to be tomorrow in Pittsburg? Today is Wendsday, tomorrow is Thursday. I would like to order a large peperroni pizza. Is it going to be cloudy tomorrow?" \
    --speed "very_high" \
    --text "Hi, this is Dusty. I have a quick announcement: SparkTTS is now running smoothly on Jetson! See you down the next rabbit hole!"
```

For best results with voice cloning:
1. Use a clear reference audio with minimal background noise
2. Provide accurate transcription of the reference audio in `--prompt_text`
3. The longer the reference audio (15-30 seconds), the better the voice cloning quality

## Model Source

This container uses the SparkTTS model from Hugging Face:
[Spark-TTS by SparkAudio](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)