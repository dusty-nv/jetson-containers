import os
import requests
import json
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
#from dotenv import load_dotenv

# Load environment variables from .env file
#load_dotenv()

# Ensure we're running from the correct directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != SCRIPT_DIR:
    print(f"Changing working directory to {SCRIPT_DIR}")
    os.chdir(SCRIPT_DIR)

def check_model_files():
    """Check if required model files exist"""
    model_dir = "/data/models/huggingface/models--hexgrad--Kokoro-82M/snapshots/main"
    required_files = ["kokoro-v0_19.onnx", "voices.bin"]

    print(f"\nChecking model files in {model_dir}:")
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        print(f"  {file}: {'✓' if exists else '✗'} ({size} bytes)")

        if not exists:
            raise FileNotFoundError(f"Required model file not found: {file_path}")
        if size == 0:
            raise ValueError(f"Model file is empty: {file_path}")

def create_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4, 8, 16 seconds between retries
        status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def wait_for_server(api_base_url, timeout=30):
    """Wait for the server to become available"""
    session = create_session()
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = session.get(f"{api_base_url}/v1/audio/speech/voices", verify=False, timeout=5)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print("Waiting for server to start...")
            time.sleep(2)
    raise TimeoutError("Server did not become available within the timeout period")

def transcribe_file(file_path, model, language, response_format, temperature):
    """
    Transcribe an audio file using the HTTP endpoint.
    Supported file types include wav, mp3, webm, and other types supported by the OpenAI API.

    Args:
        file_path (str): Path to the audio file
        model (str): Model name
        language (str): Language code
        response_format (str): Response format
        temperature (str): Temperature setting

    Returns:
        dict: The transcription response
    """
    # Use direct IP address for container-to-container communication
    api_base_url = "http://0.0.0.0:8000"  # Use the host's exposed port
    # You may need to replace with the actual IP of the host or container if this doesn't work
    # api_base_url = "http://172.17.0.2:8000"  # Replace with container IP if needed

    # Wait for server to be ready
    wait_for_server(api_base_url)

    # Create a session with retry logic
    session = create_session()

    # Open the file in binary mode
    with open(file_path, "rb") as audio_file:
        files = {"file": audio_file}
        data = {
            "model": model,
            "language": language,
            "response_format": response_format,
            "temperature": temperature
        }

        # No auth headers for internal communication
        headers = {}

        endpoint = f"{api_base_url}/v1/audio/transcriptions"
        print(f"Sending request to: {endpoint}")

        # Disable SSL verification for internal communication
        start_time = time.time()
        response = session.post(endpoint, headers=headers, files=files, data=data, verify=False)
        end_time = time.time()

        # Check if the request was successful
        print(f"Response status code: {response.status_code}")
        response.raise_for_status()

        transcription_time = end_time - start_time
        print(f"Transcription completed in {transcription_time:.2f} seconds")

        # Parse the response based on response format
        if response_format == "json":
            return response.json()
        else:
            return response.text

def generate_speech(text, model, voice, output_path):
    """
    Generate speech from text using the TTS endpoint.

    Args:
        text (str): Text to convert to speech
        model (str): Model name
        voice (str): Voice to use
        output_path (str): Path to save the generated audio file

    Returns:
        float: Time taken to generate the speech in seconds
    """
    api_base_url = "http://0.0.0.0:8000"

    # Wait for server to be ready
    wait_for_server(api_base_url)

    # Create a session with retry logic
    session = create_session()

    # First, let's check available voices
    voices_endpoint = f"{api_base_url}/v1/audio/speech/voices"
    print(f"Checking available voices at: {voices_endpoint}")
    voices_response = session.get(voices_endpoint, verify=False)
    print(f"Available voices: {voices_response.text}")

    # Parse the voice ID from the full voice string
    voice_id = voice.split('/')[-1] if '/' in voice else voice

    data = {
        "model": model,
        "input": text,
        "voice": voice_id,  # Use just the voice ID part
        "response_format": "wav"
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "audio/wav"
    }

    print(f"Generating speech for text: {text}")
    print(f"Using model: {model}, voice: {voice_id}")
    print(f"Request data: {json.dumps(data, indent=2)}")

    start_time = time.time()
    response = session.post(f"{api_base_url}/v1/audio/speech", headers=headers, json=data, verify=False)
    end_time = time.time()

    print(f"Response status code: {response.status_code}")
    if response.status_code != 200:
        print(f"Error response: {response.text}")
        print(f"Response headers: {response.headers}")
    response.raise_for_status()

    # Save the audio file
    with open(output_path, "wb") as f:
        f.write(response.content)

    generation_time = end_time - start_time
    print(f"Speech generation completed in {generation_time:.2f} seconds")
    print(f"Audio saved to: {output_path}")

    return generation_time

def main():
    # First check if model files exist
    check_model_files()

    # Test TTS first
    tts_text = "Hello, this is a test of the text-to-speech system. How are you today?"
    tts_model = "hexgrad/Kokoro-82M"
    tts_voice = "af"  # Using just the voice ID part

    # Test ASR next with independent WAV file
    asr_file = "/data/audio/dusty.wav" # mounted under jetson-containers/data

    # Create a descriptive filename with parameters
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"speaches_kokoro82m_voice{tts_voice}_{timestamp}.wav"
    tts_output = f"/data/audio/tts/{filename}"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(tts_output), exist_ok=True)

    try:
        print("\n=== Testing Text-to-Speech ===")
        tts_time = generate_speech(tts_text, tts_model, tts_voice, tts_output)
        print(f"TTS Generation Time: {tts_time:.2f} seconds")

        # Now test ASR with the generated file
        print("\n=== Testing Automatic Speech Recognition ===")
        result = transcribe_file(asr_file, "guillaumekln/faster-whisper-tiny", "en", "json", "0")
        print("Transcription completed successfully")

        if isinstance(result, dict) and "text" in result:
            print("\nTranscribed Text:")
            print(result["text"])
        else:
            print("\nTranscription Result:")
            print(result)

    except Exception as e:
        print(f"Error during testing: {type(e)} - {e}")
        import traceback
        print(f"Full error traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
