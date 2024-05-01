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

long_prompt = """French onion soup is a classic and delicious dish that is easy to make at home. Here's a simple recipe for French onion soup that you can try:

Ingredients:

* 1 onion, 1/4 cup, chopped
* 2 tablespoons butter
* 1/4 cup white wine (optional)
* 4 cups beef broth
* 2 tablespoons tomato paste
* 1 teaspoon dried thyme
* 1/2 teaspoon dried oregano
* 1/2 teaspoon salt
* 1/4 teaspoon black pepper
* 2 tablespoons all-purpose flour
* 2 tablespoons butter
* 1/2 cup grated Gruyère cheese
* 1/2 cup grated Swiss cheese
* 1/4 cup chopped fresh parsley

Instructions:

1. Heat 2 tablespoons of butter in a large saucepan over medium heat.
2. Add the chopped onion and cook until it is softened and translucent, about 5 minutes.
3. Add the white wine (if using) and 4 cups of beef broth to the saucepan. Bring to a boil, then reduce the heat to low and let it simmer for 10 minutes.
4. In a small bowl, mix the tomato paste, thyme, oregano, salt, and pepper.
5. Stir the tomato paste mixture into the broth and let it simmer for 5 more minutes.
6. In a small bowl, mix the flour and 2 tablespoons of butter.
7. Stir the flour mixture into the broth and let it cook for 1-2 minutes, or until the soup thickens.
8. Stir in the Gruyère and Swiss cheese, and let it melt and thicken the soup.
9. Taste and adjust the seasoning as needed.
10. Serve the French onion soup hot, garnished with chopped fresh parsley and a side of crusty bread or a salad.

Enjoy your"""

#prompts = [long_prompt[:500]]

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
        #overlap_len=128,
        #stream_chunk_size=20,
        #do_sample=False,
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