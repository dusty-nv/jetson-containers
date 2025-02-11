from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import time

# Big thanks for Shakh and for Kokoro_onnx Dockerfile

# Initialize pipeline
pipeline = KPipeline(lang_code='a')
print("Pipeline initialized with lang_code='a'")

# Sample text
text = '''
Kokoro is an open-weight TTS model with 82 million parameters. 
Despite its lightweight architecture, it delivers comparable quality to larger 
models while being significantly faster and more cost-efficient. 
With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.
'''

# Track generation progress
start_time = None
token_count = 0

# Generate audio with logging
generator = pipeline(
    text, 
    voice='af_heart',
    speed=1, 
    split_pattern=r' '
)

print("Starting audio generation...")

for i, (gs, ps, audio) in enumerate(generator):
    current_time = time.time()
    
    # First token
    if i == 0:
        start_time = current_time
        print(f"First token: '{gs}'")
    # Second token
    elif i == 1:
        print(f"Second token: '{gs}'")
    # Third token
    elif i == 2:
        print(f"Third token: '{gs}'")
    
    # Save audio file
    sf.write(f'output_{i}.wav', audio, 24000)
    
    # Display audio (if in Jupyter notebook)
    if i == 0:  # Autoplay first segment only
        display(Audio(data=audio, rate=24000, autoplay=True))
    
    token_count = i + 1

# Final stats
end_time = time.time()
print(f"Final token: '{gs}'")
print(f"Total generation time: {end_time - start_time:.2f} seconds")
print(f"Total tokens generated: {token_count}")
print("Audio generation completed")