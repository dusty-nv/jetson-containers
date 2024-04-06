#!/usr/bin/env python3

import time
import wave
from pathlib import Path
from typing import Any, Dict

from piper import PiperVoice
from piper.download import ensure_voice_exists, find_voice, get_voices

MODEL="/en_US-lessac-high.onnx"
CONFIG="/en_US-lessac-high.onnx.json"
DATA_DIR="/"
SPEAKER=0
LENGTH_SCALE=None
NOISE_SCALE=None
NOISE_W=None
SENTENCE_SILENCE=0.0
TEXT_EN="""A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky.
It takes the form of a multi-colored circular arc.
Rainbows caused by sunlight always appear in the section of sky directly opposite the Sun.
With tenure, Suzie’d have all the more leisure for yachting, but her publications are no good.
Shaw, those twelve beige hooks are joined if I patch a young, gooey mouth.
Are those shy Eurasian footwear, cowboy chaps, or jolly earthmoving headgear?
The beige hue on the waters of the loch impressed all, including the French queen, before she heard that symphony again, just as young Arthur wanted.
"""
DEVNULL="/dev/null"

def main() -> None:
    model = MODEL
    config = CONFIG
    model_path = Path(MODEL)
    if not model_path.exists():
        # Load voice info
        voices_info = get_voices(DATA_DIR, update_voices=False)

        # Resolve aliases for backwards compatibility with old voice names
        aliases_info: Dict[str, Any] = {}
        for voice_info in voices_info.values():
            for voice_alias in voice_info.get("aliases", []):
                aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

        voices_info.update(aliases_info)
        ensure_voice_exists(MODEL, DATA_DIR, DATA_DIR, voices_info)
        model, config = find_voice(MODEL, DATA_DIR)

    # Load voice
    voice = PiperVoice.load(model, config_path=config, use_cuda=True)
    synthesize_args = {
        "speaker_id": SPEAKER,
        "length_scale": LENGTH_SCALE,
        "noise_scale": NOISE_SCALE,
        "noise_w": NOISE_W,
        "sentence_silence": SENTENCE_SILENCE,
    }

    with wave.open(DEVNULL, "wb") as wav_file:
        start = time.time()
        voice.synthesize(TEXT_EN, wav_file, **synthesize_args)
        end = time.time()

        inference_duration = end - start

        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        audio_duration = frames / float(rate)

    print(f"Inference duration: {inference_duration:.3f}s")
    print(f"Audio duration: {audio_duration:.3f}s")
    print(f"Real time factor: {inference_duration/audio_duration:.3f}")
    print("Piper TTS: OK")


if __name__ == "__main__":
    main()
