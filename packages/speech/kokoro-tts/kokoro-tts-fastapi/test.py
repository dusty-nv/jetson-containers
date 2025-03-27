import requests
import time
import json
import os
import hashlib
from jetson_containers import handle_binary_request_or_fail

def test_tts_api():
    """Simple test for the TTS API's essential endpoints"""
    base_url = "http://localhost:8880"
    print("🧪 Testing TTS API at", base_url)
    print("\nWaiting for server to start for 20 seconds...")
    time.sleep(20)

    # Test essential endpoints
    endpoints = {
        "web_interface": "/web",
        "api_docs": "/docs",
        "models": "/v1/models",
        "voices": "/v1/audio/voices"
    }

    all_passed = True
    model_to_use = None
    voice_to_use = None

    # Step 1: Test basic endpoints
    print("\n1️⃣ TESTING BASIC ENDPOINTS")
    print(f"{'Endpoint':<15} {'Status':<8} {'Time (ms)':<12} {'Result'}")
    print("-" * 50)

    for name, path in endpoints.items():
        url = base_url + path
        try:
            start_time = time.time()
            response_bytes = handle_binary_request_or_fail(url, timeout=10)
            duration_ms = (time.time() - start_time) * 1000

            status = "✅ Pass"
            response_text = response_bytes.decode('utf-8', errors='replace')
            response_json = None

            # Try to parse JSON response if applicable
            try:
                response_json = json.loads(response_text)
            except json.JSONDecodeError:
                pass

            # Extract models if available
            if name == "models" and response_json:
                tts_models = [m["id"] for m in response_json.get("data", [])
                            if m["id"].startswith("tts-")]
                if tts_models:
                    model_to_use = tts_models[0]
                    print(f"   Found models: {', '.join(tts_models)}")

            # Extract voices if available
            if name == "voices" and response_json:
                available_voices = response_json.get("voices", [])
                if available_voices:
                    voice_to_use = available_voices[0]
                    print(f"   Found voices (first 5): {', '.join(available_voices[:5])}")

            print(f"{name:<15} {'200':<8} {duration_ms:.2f}ms      {status}")

        except Exception as e:
            print(f"{name:<15} {'ERR':<8} {'--':<12} ❌ Failed: {str(e)[:40]}")
            all_passed = False

    # Step 2: Test speech synthesis
    if all_passed and model_to_use and voice_to_use:
        print("\n2️⃣ TESTING SPEECH SYNTHESIS")

        # Configure defaults if we couldn't get them
        if not model_to_use:
            model_to_use = "tts-1"
        if not voice_to_use:
            voice_to_use = "alloy"

        # Extract language code if needed
        lang_code = None
        voice_name = voice_to_use
        if "_" in voice_to_use:
            lang_code, voice_name = voice_to_use.split("_", 1)
            print(f"Using language: {lang_code}, voice: {voice_name}")

        # Prepare test data
        test_text = "This is a test of the speech synthesis API. We are checking if everything works correctly."
        word_count = len(test_text.split())
        char_count = len(test_text)

        request_data = {
            "model": model_to_use,
            "input": test_text,
            "voice": voice_name,
            "response_format": "mp3"
        }

        # Add language code if present
        if lang_code:
            request_data["lang_code"] = lang_code

        print(f"Generating speech with model: {model_to_use}, voice: {voice_name}")
        start_time = time.time()

        try:
            response = requests.post(
                f"{base_url}/v1/audio/speech",
                json=request_data,
                headers={"Accept": "audio/mpeg"},
                timeout=15
            )

            duration = time.time() - start_time
            rtf = duration / (word_count / 150)  # Real-time factor (150 words per minute)
            char_per_sec = char_count / duration if duration > 0 else 0

            if response.status_code == 200 and len(response.content) > 1000:
                # Save the audio file
                filename = f'kokoro-{voice_name}-fastapi.mp3'

                with open(filename, "wb") as f:
                    f.write(response.content)

                file_size = len(response.content)
                print(f"\n✅ Speech synthesis successful!")
                print(f"Output saved to: {filename} ({file_size} bytes)")
                print(f"\nPERFORMANCE METRICS:")
                print(f"Processing time: {duration:.3f}s")
                print(f"Characters: {char_count}")
                print(f"Words: {word_count}")
                print(f"Speed: {char_per_sec:.2f} chars/sec")
                print(f"Real-time factor (RTF): {rtf:.3f} (lower is better, <1 is faster than real-time)")

                if rtf < 1:
                    print(f"✓ Generation is faster than real-time (good)")
                else:
                    print(f"⚠️ Generation is slower than real-time")

            else:
                print(f"❌ Speech synthesis failed with status {response.status_code}")
                if response.headers.get('content-type') == 'application/json':
                    print(f"Error: {response.json()}")
                else:
                    print(f"Response: {response.text[:100]}...")
                all_passed = False

        except Exception as e:
            print(f"❌ Speech synthesis request failed: {e}")
            all_passed = False
    else:
        print("\n⚠️ Skipping speech synthesis test due to previous failures or missing data")

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed. See log for details.")

    return all_passed

if __name__ == "__main__":
    success = test_tts_api()
    exit(0 if success else 1)