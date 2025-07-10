# Speech Packages

This directory contains speech processing packages for audio-to-text (STT/ASR), text-to-speech (TTS), and voice processing capabilities optimized for NVIDIA Jetson platforms.

## Package Categories

### Speech-to-Text (STT/ASR)
Packages that convert spoken audio into text:
- **whisper** variants - OpenAI's multilingual speech recognition
- **faster-whisper** - Optimized implementations using CTranslate2
- **riva-client** - NVIDIA's production-grade speech services

### Text-to-Speech (TTS)
Packages that generate speech from text:
- **piper-tts** - Fast, lightweight neural TTS
- **xtts** - Advanced TTS with voice cloning capabilities
- **kokoro-tts** - Multiple implementation variants
- Traditional engines like **espeak** for lightweight needs

### Speech Processing/Utilities
Supporting packages for voice processing:
- **audiocraft** - Meta's audio generation framework
- **voicecraft** - Voice manipulation tools
- Interface layers like **speech-dispatcher**

## When to Add New Packages Here

Add a package to the speech category when it:
- Processes audio input for speech recognition
- Generates speech audio from text
- Provides voice processing, synthesis, or analysis capabilities
- Enables speech-related AI/ML workflows
- Integrates speech capabilities into applications

## Package Selection Guidelines

### For Speech-to-Text Tasks

**Choose based on your requirements:**
- **Accuracy vs Speed**: `whisper` for best accuracy, `faster-whisper` for speed
- **Hardware optimization**: `whisper_trt` for TensorRT acceleration
- **Production deployment**: `riva-client` for enterprise features
- **Extended features**: `whisperx` for alignment and diarization

### For Text-to-Speech Tasks

**Consider these factors:**
- **Quality vs Speed**: Higher quality models are slower
- **Voice cloning**: Choose `xtts` if you need voice cloning
- **Resource constraints**: `piper-tts` for lightweight deployment
- **Language support**: Check package docs for supported languages
- **Real-time needs**: Look for packages with streaming support

### For Integration

**Look for packages with:**
- Compatible audio formats (WAV, MP3, etc.)
- Appropriate compute precision (FP16, INT8)
- Required dependencies already in your stack
- API compatibility with your application

## Package Structure Requirements

Each speech package must include:

### Required Files
- `Dockerfile` - Container build instructions with metadata header
- `README.md` - Documentation following the standard template

### Dockerfile Metadata
```dockerfile
#---
# name: your-speech-package
# group: audio
# depends: [pytorch, numpy]  # Common dependencies
# requires: '>=34.1.0'       # L4T version requirement
# test: test.py              # Test script
# docs: docs.md              # Additional documentation
#---
```

### Optional but Recommended
- `test.py` - Validation script to verify functionality
- `config.py` - Package-specific configuration
- Example scripts demonstrating usage
- Model download/setup scripts

## Common Dependencies

Speech packages typically depend on:
- **ML Frameworks**: pytorch, tensorflow, onnxruntime
- **Audio Processing**: torchaudio, librosa, soundfile
- **Optimization**: tensorrt, ctranslate2
- **Utilities**: numpy, transformers, huggingface-hub

## Performance Considerations

When selecting packages, consider:
- **GPU Memory**: Speech models can be memory-intensive
- **Latency Requirements**: Real-time vs batch processing
- **Compute Type**: FP16 usually offers best speed/quality balance
- **Model Size**: Larger models need more memory but offer better quality

## Testing Speech Packages

Before using in production:
1. Check supported audio formats and sample rates
2. Verify language support for your use case
3. Test with representative audio samples
4. Measure latency and throughput
5. Monitor GPU memory usage

## Integration Tips

- Most packages expose Python APIs for easy integration
- Check for streaming support if processing long audio
- Consider audio preprocessing requirements
- Some packages support batching for efficiency
- Look for packages with similar dependency stacks to minimize image size

Remember to check individual package READMEs for specific usage instructions, supported models, and configuration options.