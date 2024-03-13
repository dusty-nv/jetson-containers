import os
import time
import torch
import pprint
import logging
import torchaudio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from TTS.api import TTS

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.DEBUG)

model_dir="/data/models/tts/tts_models--multilingual--multi-dataset--xtts_v2"
model_name=model_dir.split('--')[-1]

speaker='Sofia Hellen'
language='en'
device = "cuda" if torch.cuda.is_available() else "cpu"

logging.info(f"loading TTS model {model_dir}")

config = XttsConfig()
config.load_json(f"{model_dir}/config.json")

logging.info(f"TTS model config:\n{pprint.pformat(config, indent=1)}")

model = Xtts.init_from_config(config)

model.load_checkpoint(
    config, 
    checkpoint_dir=model_dir, 
    speaker_file_path=f"{model_dir}/speakers_xtts.pth",
    use_tensorrt=True
)

model.cuda()

speaker_manager = model.speaker_manager

gpt_cond_latent, speaker_embedding = speaker_manager.speakers[speaker].values()

gpt_cond_latent.to(device)
speaker_embedding.to(device)

prompts = [
    "Hello there, how are you today?", 
    "The weather is 76 degrees out and sunny.", 
    "Your first meeting is in an hour downtown, with normal traffic.",
    "Can I interest you in anything quick for breakfast?",
]

prompts = [' '.join(prompts)] + prompts

for prompt_idx, prompt in enumerate(prompts):
    wav_path = f"/data/audio/tts/{model_name}_streaming_{speaker.lower().replace(' ', '_')}_{prompt_idx}.wav"
    logging.info(f'\nstreaming "{prompt}"  speaker="{speaker}"  lang="{language}"  wav="{wav_path}"\n')
    
    time_begin = time.perf_counter()
    time_last = time_begin
    
    chunks = model.inference_stream(
        prompt,
        language,
        gpt_cond_latent,
        speaker_embedding,
        enable_text_splitting=False, #True,
        overlap_len=128,
        #stream_chunk_size=20,
        do_sample=False,
        speed=0.9,
    )

    wav_chunks = []
    wav_length = 0
    
    for i, chunk in enumerate(chunks):
        time_curr = time.perf_counter()
        if i == 0:
            logging.info(f"Time to first chunk: {time_curr - time_begin}")
        logging.info(f"Received chunk {i} of audio length {chunk.shape[-1]}  ({time_curr-time_last:.3f} seconds since last, RTFX={(chunk.shape[-1]/24000)/(time_curr-time_last):.4f})")
        time_last = time_curr
        wav_chunks.append(chunk)
        wav_length += chunk.shape[-1]
        
    time_elapsed = time.perf_counter() - time_begin
    logging.info(f"streamed {wav_length/24000:.3f} seconds of audio ({wav_length} samples at 24KHz) in {time_elapsed:.3f} seconds (RTFX={(wav_length/24000)/time_elapsed:.4f})")
        
    wav = torch.cat(wav_chunks, dim=0).to(dtype=torch.float32)
    torchaudio.save(wav_path, wav.squeeze().unsqueeze(0).cpu(), 24000)