import os
import requests
import json
#from dotenv import load_dotenv

# Load environment variables from .env file
#load_dotenv()

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

    # No API key needed for internal container communication

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
        response = requests.post(endpoint, headers=headers, files=files, data=data, verify=False)

        # Check if the request was successful
        print(f"Response status code: {response.status_code}")
        response.raise_for_status()

        # Parse the response based on response format
        if response_format == "json":
            return response.json()
        else:
            return response.text

def main():
    # File path - adjusted to where you copied the file
    file_path = "/data/audio/dusty.wav"  # mounted under jetson-containers/data

    # Default parameters
    model = "guillaumekln/faster-whisper-tiny"
    language = "en"  # English, change to the language of your audio
    response_format = "json"  # 'json', 'text', 'srt', 'verbose_json', 'vtt'
    temperature = "0"  # Lower values are more focused

    try:
        print(f"Transcribing file: {file_path}")
        result = transcribe_file(file_path, model, language, response_format, temperature)
        print("Transcription completed successfully")

        if isinstance(result, dict) and "text" in result:
            print("\nTranscribed Text:")
            print(result["text"])
        else:
            print("\nTranscription Result:")
            print(result)

    except Exception as e:
        print(f"Error during transcription: {type(e)} - {e}")

if __name__ == "__main__":
    main()
