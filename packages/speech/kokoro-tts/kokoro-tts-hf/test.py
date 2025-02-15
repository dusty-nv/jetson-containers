from kokoro import KPipeline
from IPython.display import display, Audio
import torchaudio
import torch
import time

# Big thanks for Shakh and for Kokoro_onnx Dockerfile

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Initialize pipeline
pipeline = KPipeline(lang_code='a', device='cuda')
print("Pipeline initialized with lang_code='a'")

# Sample text
text = '''
a. Kokoro is an open-weight TTS model with 82 million parameters. 
Despite its lightweight architecture, it delivers comparable quality to larger 
models while being significantly faster and more cost-efficient. 
With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.
'''

# Track generation progress
start_time = None
token_count = 0

pipeline.load_voice(voice='af_heart', delimiter=r'\.')
print('loaded voice and delimiter')
# Generate audio with logging
generator = pipeline(
    text, 
    voice='af_heart',
    speed=1, 
    split_pattern=r'\.'
)

print("Starting audio generation...")

start_time = time.time()
current_time = start_time
for i, (gs, ps, audio) in enumerate(generator):
    last_time = current_time
    current_time = time.time()
        
    # First token
    if i == 0:
        print(f"First token: '{gs}' [{current_time - last_time:.2f}s]")
    # Second token
    elif i == 1:
        print(f"Second token: '{gs}' [{current_time - last_time:.2f}s]")
    # Third token
    elif i == 2:
        print(f"Third token: '{gs}' [{current_time - last_time:.2f}s]")
    
    # Save audio file
    torchaudio.save(
        f'output_{i}.wav',
        audio.unsqueeze(0).float(),
        sample_rate=24000
    )
    
    # Display audio (if in Jupyter notebook)
    if i == 0:  # Autoplay first segment only
        display(Audio(data=audio.numpy(), rate=24000, autoplay=True))
    
    token_count = i + 1

# Final stats
end_time = time.time()
print(f"Total generation time: {end_time - start_time:.2f} seconds")
print(f"Total tokens generated: {token_count}")
print(f"Tokens per second: {token_count / (end_time - start_time):.2f}")
print("Audio generation completed")


print("Now per word")

pipeline.load_voice(voice='af_heart', delimiter=r'\ ')
print('loaded voice and delimiter')

# Track generation progress
start_time = None
token_count = 0

# Generate audio with logging
generator = pipeline(
    text, 
    voice='af_heart',
    speed=1, 
    split_pattern=r'\ '
)

print("Starting audio generation...")

start_time = time.time()
current_time = start_time
for i, (gs, ps, audio) in enumerate(generator):
    last_time = current_time
    current_time = time.time()
        
    # First token
    if i == 0:
        print(f"First token: '{gs}' [{current_time - last_time:.2f}s]")
    # Second token
    elif i == 1:
        print(f"Second token: '{gs}' [{current_time - last_time:.2f}s]")
    # Third token
    elif i == 2:
        print(f"Third token: '{gs}' [{current_time - last_time:.2f}s]")
    
    # Save audio file
    torchaudio.save(
        f'output_{i}.wav',
        audio.unsqueeze(0).float(),
        sample_rate=24000
    )
    
    # Display audio (if in Jupyter notebook)
    if i == 0:  # Autoplay first segment only
        display(Audio(data=audio.numpy(), rate=24000, autoplay=True))
    
    token_count = i + 1

# Final stats
end_time = time.time()
print(f"Total generation time: {end_time - start_time:.2f} seconds")
print(f"Total tokens generated: {token_count}")
print(f"Tokens per second: {token_count / (end_time - start_time):.2f}")
print("Audio generation completed")