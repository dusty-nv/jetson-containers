from kokoro import KPipeline
from IPython import get_ipython
from IPython.display import display, Audio
import torchaudio
import torch
import time

# Big thanks for Shakh and for Kokoro_onnx Dockerfile

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Torch Device: {device}")

if torch.cuda.is_available():
    print(f"CUDA Device:  {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}\n")

# Initialize pipeline
pipeline = KPipeline(lang_code='a', device='cuda')
print("Pipeline initialized with lang_code='a'")

# Sample text
text = '''
Kokoro is an open-weight TTS model with 82 million parameters. 
Despite its lightweight architecture, it delivers comparable quality to larger 
models while being significantly faster and more cost-efficient. 
With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.
'''

# Track generation progress
voice='af_heart'
pipeline.load_voice(voice=voice, delimiter=r'\.')
print(f'Loaded voice {voice}')

generator = pipeline(
    text, 
    voice=voice,
    speed=1, 
    split_pattern=r'\.'
)

print("Starting audio generation (per sentence)...")

start_time = time.time()
current_time = start_time
audio_phrases = []

for i, (gs, ps, audio) in enumerate(generator):
    last_time = current_time
    current_time = time.time()
        
    # First phrase
    if i == 0:
        print(f"First phrase:  '{gs}' [{current_time - last_time:.2f}s]")
    # Second phrase
    elif i == 1:
        print(f"Second phrase: '{gs}' [{current_time - last_time:.2f}s]")
    # Third phrase
    elif i == 2:
        print(f"Third phrase:  '{gs}' [{current_time - last_time:.2f}s]\n")
    
    # Save audio file
    audio_phrases.append(audio)

    # Display audio (if in Jupyter notebook)
    if i == 0 and get_ipython() is not None:  # Autoplay first segment only
        display(Audio(data=audio.numpy(), rate=24000, autoplay=True))

# Final stats
end_time = time.time()

print(f"Total generation time:   {end_time - start_time:.2f} seconds")
print(f"Total phrases generated: {len(audio_phrases)}")
print(f"Phrases per second:      {len(audio_phrases) / (end_time - start_time):.2f}")

def save_audio(chunks, path, sample_rate=24000):
    samples = torch.cat(audio_phrases, axis=0).unsqueeze(0).float()
    print(f"\nSaving {len(samples[0])} samples ({len(samples[0]) / sample_rate:.2f} sec) to {path} ({sample_rate} KHz)\n")
    torchaudio.save(path, samples, sample_rate=sample_rate)

save_audio(audio_phrases, f'kokoro-{voice}-phrases.wav')

# Generate audio (per word)
print("Changing to word-level delimiter...")
pipeline.load_voice(voice=voice, delimiter=r'\ ')

generator = pipeline(
    text, 
    voice=voice,
    speed=1, 
    split_pattern=r'\ '
)

print("Starting audio generation (per word)...")

start_time = time.time()
current_time = start_time
audio_words = []

for i, (gs, ps, audio) in enumerate(generator):
    last_time = current_time
    current_time = time.time()
        
    # First phrase
    if i == 0:
        print(f"First word:  '{gs}' [{current_time - last_time:.2f}s]")
    # Second phrase
    elif i == 1:
        print(f"Second word: '{gs}' [{current_time - last_time:.2f}s]")
    # Third phrase
    elif i == 2:
        print(f"Third word:  '{gs}' [{current_time - last_time:.2f}s]\n")

    # Display audio (if in Jupyter notebook)
    if i == 0 and get_ipython() is not None:  # Autoplay first segment only
        display(Audio(data=audio.numpy(), rate=24000, autoplay=True))
    
    audio_words.append(audio)

# Final stats
end_time = time.time()

print(f"Total generation time: {end_time - start_time:.2f} seconds")
print(f"Total words generated: {len(audio_words)}")
print(f"Words per second:      {len(audio_words) / (end_time - start_time):.2f}")

save_audio(audio_words, f'kokoro-{voice}-words.wav')