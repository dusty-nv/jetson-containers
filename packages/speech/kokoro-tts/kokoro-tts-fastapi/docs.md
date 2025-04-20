# KokoroTTS FastAPI

This Dockerfile sets up a text-to-speech (TTS) service based on the Kokoro TTS engine using FastAPI (https://github.com/remsky/Kokoro-FastAPI)

The server will automatically be started at `0.0.0.0:8880` when you run the container:

```
docker run -it --rm -p 8880:8880 dustynv/kokoro-tts:fastapi-r36.4.0
```

Included is a built-in web UI ([`http://0.0.0.0:8880/web`](http://0.0.0.0:8880/web)) that converts text to speech using the Kokoro TTS model with GPU acceleration.  

> [!WARNING]  
> There appears to be an issue with audio playback in the web UI on Firefox - Chrome or Chromium is recommended. 

This is the default CMD executed at container startup:

```
uvicorn api.src.main:app --reload --host 0.0.0.0 --port 8880
```