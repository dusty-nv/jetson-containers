# Silero VAD

[Silero VAD](https://github.com/snakers4/silero-vad) is a pre-trained model for Voice Activity Detection (VAD) that can be used to detect speech in audio streams. This container provides a ready-to-use environment for voice activity detection applications on Jetson devices.

## Overview

Silero VAD is a lightweight model that accurately detects speech in audio streams, which is useful for:

- Audio preprocessing in speech recognition systems
- Automated transcription pipelines
- Voice-controlled applications
- Meeting recording and analytics
- Voice assistant applications

This container comes with all necessary dependencies pre-installed, making it an ideal building block for Python applications that require voice recording and speech detection capabilities.

## Dependencies

The container is built with the following dependencies:

- CUDA support via the Jetson base image
- PyTorch and torchaudio
- ONNX Runtime
- portaudio19-dev for audio device access
- Python packages:
  - soundfile: for reading and writing audio files
  - sounddevice: for recording and playing back audio
  - silero-vad: the voice activity detection model

## Usage

### Import Options

There are two main ways to import and use Silero VAD:

#### 1. Using PyTorch Hub (original method)

```python
import torch

# Load the VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

# Get functions for inference
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
```

#### 2. Using the silero_vad Package (simplified method)

```python
# Import the model loader directly from the package
from silero_vad import load_silero_vad

# Load the model with a simpler interface
vad_model = load_silero_vad()  # Set force_reload=True if needed
```

### Basic Voice Activity Detection

```python
import torch
import numpy as np
import sounddevice as sd
from silero_vad import get_speech_timestamps, save_audio

# Load the VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

# Get functions for inference
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Record audio (16kHz mono is recommended for the model)
sample_rate = 16000
recording_duration = 5  # seconds
print("Recording...")
audio = sd.rec(int(recording_duration * sample_rate), 
               samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
print("Recording complete.")

# Convert to the format expected by the model
audio = audio.reshape(-1)

# Get timestamps of speech segments
speech_timestamps = get_speech_timestamps(audio, model, threshold=0.5, sampling_rate=sample_rate)

print(f"Found {len(speech_timestamps)} speech segments")
for i, ts in enumerate(speech_timestamps):
    print(f"Speech segment {i+1}: {ts['start']/sample_rate:.2f}s to {ts['end']/sample_rate:.2f}s")
```

### Real-time Voice Activity Detection

```python
import torch
import sounddevice as sd
import numpy as np
from silero_vad import get_speech_timestamps, save_audio

# Load the VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Initialize VAD iterator
vad_iterator = VADIterator(model)

# Settings
sample_rate = 16000
window_size_samples = 1536  # Number of samples per chunk (96ms for 16kHz)
is_speech = False

# Callback function for real-time audio processing
def audio_callback(indata, frames, time, status):
    global is_speech
    
    if status:
        print(status)
        
    # Process audio chunk
    audio_chunk = indata[:, 0]  # Use first channel if stereo
    speech_dict = vad_iterator(audio_chunk, return_seconds=True)
    
    # Check if speech was detected
    if speech_dict:
        if speech_dict['speech']:
            if not is_speech:
                is_speech = True
                print("Speech detected!")
        else:
            if is_speech:
                is_speech = False
                print("Speech ended.")

# Start real-time audio stream
with sd.InputStream(samplerate=sample_rate,
                    blocksize=window_size_samples,
                    channels=1,
                    callback=audio_callback):
    print("Real-time VAD started. Press Ctrl+C to stop.")
    try:
        # Run indefinitely
        while True:
            sd.sleep(100)  # Small sleep to reduce CPU usage
    except KeyboardInterrupt:
        print("Stopping...")
        vad_iterator.reset_states()  # Reset VAD states
```

### Voice Assistant Example

Below is a more comprehensive example using Silero VAD in a voice assistant application:

```python
import sounddevice as sd
import numpy as np
import torch
import time
import os
import tempfile
import scipy.io.wavfile as wavfile
import requests

# Load Silero VAD
from silero_vad import load_silero_vad

# Audio Settings
SAMPLE_RATE = 16000  # Sample rate expected by Silero VAD
CHANNELS = 1
BLOCK_SIZE = 512     # Smaller chunks for faster VAD response
DTYPE = 'int16'      # Data type for recording

# VAD Settings
VAD_THRESHOLD = 0.5  # VAD confidence threshold
MIN_SILENCE_DURATION_MS = 1000  # How long silence indicates end of speech
SPEECH_PAD_MS = 300  # Add slight padding around detected speech
VAD_EVERY_N_CHUNKS = 3  # Process VAD every N chunks to reduce CPU load

# Calculate VAD timing in chunks
ms_per_chunk = (BLOCK_SIZE / SAMPLE_RATE) * 1000
num_silent_chunks_needed = int(MIN_SILENCE_DURATION_MS / ms_per_chunk)
PRESPEECH_BUFFER_SECONDS = 2  # Audio to keep before speech detection
max_prespeech_chunks = int(PRESPEECH_BUFFER_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

# Load VAD model using the simplified interface
print("Loading Silero VAD model...")
vad_model = load_silero_vad()
print("Silero VAD model loaded successfully.")

def main_loop():
    print("\nStarting voice assistant loop. Press Ctrl+C to exit.")
    
    # State variables for VAD processing
    audio_buffer = []
    triggered = False
    silent_chunks = 0
    prespeech_buffer = []
    
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                               blocksize=BLOCK_SIZE,
                               channels=CHANNELS,
                               dtype=DTYPE) as stream:
            
            print("Listening...")
            chunk_counter = 0
            
            while True:
                # Read audio chunk
                audio_chunk_raw, overflowed = stream.read(BLOCK_SIZE)
                if overflowed:
                    print("Warning: Input overflowed!")
                
                # Convert raw bytes to numpy array
                audio_chunk_np = np.frombuffer(audio_chunk_raw, dtype=np.int16)
                if audio_chunk_np.size == 0:
                    continue
                
                # Maintain pre-speech buffer
                prespeech_buffer.append(audio_chunk_np)
                if len(prespeech_buffer) > max_prespeech_chunks:
                    prespeech_buffer.pop(0)
                
                # If speech is triggered, always add chunk to buffer
                if triggered:
                    audio_buffer.append(audio_chunk_np)
                
                # Process VAD only every N chunks to reduce CPU load
                chunk_counter = (chunk_counter + 1) % VAD_EVERY_N_CHUNKS
                if chunk_counter != 0:
                    continue
                
                # Normalize to [-1.0, 1.0] for VAD model
                audio_chunk_tensor = torch.from_numpy(audio_chunk_np).float() / 32768.0
                
                # Run VAD prediction
                speech_prob = vad_model(audio_chunk_tensor, SAMPLE_RATE).item()
                
                if speech_prob >= VAD_THRESHOLD:
                    silent_chunks = 0  # Reset silence counter
                    if not triggered:
                        print("Speech started...")
                        triggered = True
                        # Add pre-speech buffer to capture context
                        audio_buffer.extend(prespeech_buffer)
                        prespeech_buffer.clear()
                
                elif triggered:  # Was speech, now silence
                    silent_chunks += 1
                    
                    if silent_chunks >= num_silent_chunks_needed:
                        print(f"Speech ended after {silent_chunks * ms_per_chunk:.0f}ms silence.")
                        
                        # Process the captured audio
                        if audio_buffer:
                            full_audio = np.concatenate(audio_buffer)
                            print(f"Captured audio: {len(full_audio)/SAMPLE_RATE:.2f} seconds")
                            
                            # Here you would:
                            # 1. Save audio to WAV file
                            # 2. Send to speech recognition
                            # 3. Process recognized text
                            # 4. Generate and play response
                            
                            # Reset for next utterance
                            audio_buffer = []
                            triggered = False
                            silent_chunks = 0
                            print("\nListening...")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main_loop()
```

This example demonstrates:
- Using the simplified `silero_vad` package import
- Processing audio in small chunks for real-time response
- Maintaining a pre-speech buffer to capture context before speech starts
- Reducing CPU load by running VAD predictions only every N chunks
- Detecting both speech start and end based on VAD confidence and silence duration

## Additional Resources

- [Silero VAD GitHub Repository](https://github.com/snakers4/silero-vad)
- [Silero VAD Paper](https://arxiv.org/abs/2106.09624)
- [PyTorch Hub Model](https://pytorch.org/hub/snakers4_silero-vad_vad/)
- [silero-vad PyPI Package](https://pypi.org/project/silero-vad/)

## License

The Silero VAD model is distributed under the Open Data Commons Attribution License (ODC-By).