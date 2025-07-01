# Chatterbox TTS

Chatterbox TTS is an advanced text-to-speech engine that offers high-quality voice synthesis with emotion control and voice cloning capabilities. It's based on the [Chatterbox project by Resemble AI](https://github.com/resemble-ai/chatterbox).

## Overview

Chatterbox TTS is a more resource-intensive model compared to alternatives like Piper and Kokoro, offering enhanced synthesis quality and additional features. The added capabilities come at the cost of higher computational requirements.

## Features

- **Standard Voice Generation**: Generate natural-sounding speech from text
- **Emotional Voice Synthesis**: Control the emotional tone of generated speech
- **Voice Cloning**: Clone voices from audio samples for personalized speech output

## Hardware Compatibility

| Device | Status |
|--------|--------|
| Jetson AGX | ✅ Tested |
| Jetson Nano | 🔄 Testing in progress |
| Other Jetson devices | 📝 To be tested |

## System Requirements

- CUDA 12.9+
- Ubuntu 24.04 or compatible OS
- Sufficient GPU memory (tested on Jetson AGX)

## Performance

- **Claimed inference time:** ~200ms per generation
- **Actual benchmarks:** Pending detailed testing across Jetson devices

## Usage

### Building the container

When building the container, use:
```bash
CUDA_VERSION=12.9 LSB_RELEASE=24.04 jetson-containers build --name=... chatterbox-tts 
```


### Running the Container

```bash
docker run -it --rm --runtime nvidia dustynv/chatterbox-tts:r36.2.0
```

### Basic Text-to-Speech Generation

```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Load model to GPU
model = ChatterboxTTS.from_pretrained(device="cuda")

# Generate speech
text = "Hello, I am Chatterbox TTS running on NVIDIA Jetson!"
wav = model.generate(text)
ta.save("output.wav", wav, model.sr)
```

### Voice Cloning

```python
# Load reference audio for voice cloning
reference_audio = "path/to/reference.wav"
wav = model.generate(text, reference_audio_path=reference_audio)
ta.save("cloned_voice.wav", wav, model.sr)
```

## ⚠️ Development Status: Work in Progress ⚠️

This package is currently under active development. Planned improvements include:

- Proper Docker container lifecycle management
- Comprehensive test suite
- Extended device compatibility testing
- Performance optimizations for Jetson devices
- Expanded documentation with more usage examples

## Troubleshooting

If you encounter issues with GPU memory or performance:
- Ensure you have sufficient GPU memory available
- Consider reducing batch size or sequence length for larger inputs
- Check that you're using the appropriate CUDA version

## Additional Resources

- [Chatterbox GitHub Repository](https://github.com/resemble-ai/chatterbox)
- [Jetson Containers Documentation](https://github.com/dusty-nv/jetson-containers)

