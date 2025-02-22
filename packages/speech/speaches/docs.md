# Speaches

Speaches supports VAD (Voice Activity Detection), STT (Speech-to-Text) and TTS (Text-to-Speech).

## What is VAD and why it's needed

Voice Activity Detection (VAD) is used to detect the presence or absence of human speech in audio streams.  
It is important when processing speech because by identifying when someone is actually speaking, you prevent unnecessary processing of silence or background noise, reducing computational overhead.

## Important Notes on TTS Implementation

At this time of adding support to Speaches, TTS uses ONNX for KokoroTTS.  
**Do not use it** - for OpenAI API TTS, use `kokoro-tts:fastapi` or `kokoro-tts:hf` / `kokoro-tts:onnx` instead.  

Since we already have KokoroTTS on onnx support (`kokoro-tts:onnx`), it was not supported here and using TTS Endpoints will default to CPU.  
Adding support for these endpoints is TBD, but not currently prioritized.  

## Contributing

If you need TTS endpoint support, please open an issue and tag @OriNachum

