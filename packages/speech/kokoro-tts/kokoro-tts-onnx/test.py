import soundfile as sf
from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession, SessionOptions, get_available_providers
import onnxruntime
import os
import time

DEFAULT_PROMPT="""
Kokoro is an open-weight TTS model with 82 million parameters. 
Despite its lightweight architecture, it delivers comparable quality to larger 
models while being significantly faster and more cost-efficient. 
With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.
"""

def create_session():
    providers = get_available_providers()
    print(f"Available ONNX Runtime providers: {providers}")

    # Define session options
    sess_options = SessionOptions()
    cpu_count = os.cpu_count()
    print(f"Setting threads to CPU cores count: {cpu_count}")
    sess_options.intra_op_num_threads = cpu_count

    # Iterate through providers to create a valid session
    for provider in providers:
        try:
            print(f"Trying provider: {provider}")
            session = InferenceSession(
                "/opt/kokoro-onnx/examples/kokoro-v1.0.onnx", providers=[provider], sess_options=sess_options
            )
            print(f"Session created successfully with provider: {provider}")
            return session
        except Exception as e:
            print(f"Failed to create session with provider {provider}: {e}")

    # If no provider works, raise an error
    raise RuntimeError("Failed to create an ONNX Runtime session with any available provider.")

# Create session and initialize Kokoro model
session = create_session()
kokoro = Kokoro.from_session(session, "/opt/kokoro-onnx/examples/voices-v1.0.bin")

# Warm-up iteration
print("Performing warm-up inference...")
kokoro.create("Warm-up text", voice="af_sarah", speed=1.0, lang="en-us")
print("Warm-up complete.")

# Measure inference time
start_time = time.time()
samples, sample_rate = kokoro.create(
    DEFAULT_PROMPT, voice="af_sarah", speed=1.0, lang="en-us"
)
inference_time = time.time() - start_time

# Calculate real-time factor (RTF)
audio_duration = len(samples) / sample_rate
rtf = inference_time / audio_duration

# Write audio output and print timings
sf.write("audio.wav", samples, sample_rate)
print(f"Created audio.wav")
print(f"Inference time: {inference_time:.4f} seconds")
print(f"Audio duration: {audio_duration:.4f} seconds")
print(f"Real-Time Factor (RTF): {rtf:.4f}")
