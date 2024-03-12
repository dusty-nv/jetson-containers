#!/usr/bin/env python3
import os
import torch
import pprint

from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

#print(TTS().list_models())

model="tts_models/multilingual/multi-dataset/xtts_v2"  # "tts_models/multilingual/multi-dataset/xtts_v1.1"
speaker_wav="/data/audio/tts/xtts_v2_claribel_dervla_0.wav"
speaker="Claribel Dervla"
language='en'

print(f"Loading TTS model {model}")

tts = TTS(model).to(device)

print(dir(tts.synthesizer.tts_model.speaker_manager))
print(tts.synthesizer.tts_model.speaker_manager)

print(f"\nMulti-speaker:  {tts.is_multi_speaker}")

if tts.is_multi_speaker:
    print(f"\nSpeakers:  {tts.synthesizer.tts_model.speaker_manager.name_to_id}")
    
print(f"\nLanguages:  {tts.synthesizer.tts_model.language_manager.name_to_id}")

# Text to speech to a file
prompts = [
    "Hello world!", 
    "How are you today?", 
    "The weather is 53 degrees out and rainy.", 
    "What is your favorite food to eat for breakfast?", 
    "Mine is a French toast with maple syrup and sausages.", 
    "I'm a big fan of Sunday brunch and taking an early afternoon nap afterwards, of course."
]
    
if tts.is_multi_speaker:
    prompts = [' '.join(prompts)] + prompts

for prompt_idx, prompt in enumerate(prompts):
    wav = f"/data/audio/tts/{os.path.basename(model)}_offline_{speaker.lower().replace(' ', '_')}_{prompt_idx}.wav"
    print(f'\ngenerating "{prompt}"  speaker="{speaker}"  lang="{language}"  wav="{wav}"\n')
    if tts.is_multi_speaker:
        tts.tts_to_file(text=prompt, speaker=speaker, language=language, file_path=wav)
    else:
        tts.tts_to_file(text=prompt, speaker_wav=speaker_wav, language=language, file_path=wav)
