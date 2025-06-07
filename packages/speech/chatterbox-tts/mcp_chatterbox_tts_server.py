from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import base64
import io
import soundfile as sf # For handling audio data

# MCP Server specific imports
from mcp.server.fastmcp import FastMCP
from mcp.model.domain import Message, Role, ToolCall, ToolResponseMessage

# Chatterbox specific imports
# Assuming chatterbox is installed and accessible in PYTHONPATH
# The base Dockerfile for chatterbox-tts should handle this.
try:
    from chatterbox.inference import TextToSpeech
    from chatterbox.utils.config import Config
    from chatterbox.utils.device import get_device
    CHATTERBOX_AVAILABLE = True
    print("Chatterbox library imported successfully.")
except ImportError as e:
    CHATTERBOX_AVAILABLE = False
    print(f"Warning: Chatterbox library not found or import error: {e}. MCP server will run with dummy responses for TTS.")

# --- MCP Server Setup ---
mcp_server = FastMCP(
    "Chatterbox TTS MCP Server",
    "Provides Text-to-Speech generation using Resemble AI's Chatterbox model."
)

# --- Model Loading ---
tts_model = None
if CHATTERBOX_AVAILABLE:
    try:
        # Determine config path. Default to a path inside /opt/chatterbox-tts
        # This path should align with where chatterbox.yaml is within the cloned repo.
        config_path = os.environ.get("CHATTERBOX_CONFIG_PATH", "/opt/chatterbox-tts/config/chatterbox.yaml")
        
        if not os.path.exists(config_path):
            # Fallback for local testing if /opt/chatterbox-tts/config/chatterbox.yaml doesn't exist
            # Try relative path if running from within a cloned chatterbox repo structure
            alt_config_path = "config/chatterbox.yaml"
            if os.path.exists(alt_config_path):
                config_path = alt_config_path
            else:
                # If still not found, try to locate it within /opt/chatterbox-tts
                found_configs = []
                if os.path.exists("/opt/chatterbox-tts"):
                    for root, _, files in os.walk("/opt/chatterbox-tts"):
                        if "chatterbox.yaml" in files:
                            found_configs.append(os.path.join(root, "chatterbox.yaml"))
                if found_configs:
                    config_path = found_configs[0] # Use the first one found
                    print(f"Auto-detected chatterbox.yaml at: {config_path}")
                else:
                    print(f"Error: Chatterbox config file 'chatterbox.yaml' not found at {config_path} or default locations.")
                    CHATTERBOX_AVAILABLE = False # Treat as unavailable if config is missing

        if CHATTERBOX_AVAILABLE and os.path.exists(config_path):
            print(f"Loading Chatterbox config from: {config_path}")
            config = Config(config_path=config_path)
            device = get_device()
            tts_model = TextToSpeech(config.model, device=device)
            print("Chatterbox TTS model loaded successfully.")
        elif CHATTERBOX_AVAILABLE: # Config path existed but model loading failed or config_path was not found
            print(f"Chatterbox config file not found at the specified or default paths. Model not loaded.")
            CHATTERBOX_AVAILABLE = False

    except Exception as e:
        print(f"Error loading Chatterbox TTS model: {e}")
        CHATTERBOX_AVAILABLE = False
else:
    print("Chatterbox library not available, TTS model not loaded.")

# --- MCP Tool Definition ---
class TTSRequest(BaseModel):
    text: str
    # Potentially add other parameters like speaker_id, voice_preset, etc.
    # based on what chatterbox supports and what you want to expose.
    # Example: speaker_id: str | None = None

class TTSResponse(BaseModel):
    audio_base64: str | None = None
    content_type: str = "audio/wav" # Default content type
    error: str | None = None
    text_generated_for: str

@mcp_server.tool(name="generate_speech", description="Generates speech from text using Chatterbox TTS.")
def generate_speech_tool(text: str) -> TTSResponse:
    """Generates audio from the input text using the Chatterbox TTS model.
    Args:
        text: The text to synthesize into speech.
    Returns:
        A TTSResponse object containing the base64 encoded audio and content type, or an error message.
    """
    if not CHATTERBOX_AVAILABLE or tts_model is None:
        return TTSResponse(error="TTS model is not available.", text_generated_for=text)

    try:
        print(f"MCP Tool: Received TTS request for: '{text}'")
        
        # Synthesize audio. Chatterbox's `synthesize` method might save to a file or return raw data.
        # We need to get the audio data as bytes.
        # The `inference.py` script in chatterbox repo uses `tts.synthesize(args.text, args.output_path, ...)`
        # which implies it writes to a file. We need to adapt this.
        # Let's assume tts_model.synthesize can return audio data directly or we can make it do so.
        # If it must write to a file, we'll use a temporary in-memory buffer.

        # Option 1: If model can return bytes (ideal)
        # audio_data_np, sample_rate = tts_model.synthesize_to_numpy(text) # Fictional method, check chatterbox docs
        # audio_bytes = audio_data_np.tobytes()
        
        # Option 2: If model writes to file path, use BytesIO
        # This is more likely based on typical TTS libraries.
        buffer = io.BytesIO()
        # The chatterbox `TextToSpeech.synthesize` method takes `output_path`.
        # It also takes `sample_rate`, `voice_preset`. We might need to expose these.
        # For now, using defaults implied by the loaded model config.
        # The method seems to return the output path. We need to ensure it writes to our buffer.
        # This might require a small modification to chatterbox or a wrapper if it strictly needs a file path.
        
        # Let's assume a hypothetical `synthesize_to_buffer` or adapt.
        # For now, we'll simulate it by trying to get raw audio output.
        # The `inference.py` uses `sf.write(output_path, audio_data, sample_rate)`. 
        # This means `tts.synthesize` likely returns (audio_data, sample_rate).

        # Based on `chatterbox/inference.py` from resemble-ai/chatterbox:
        # `audio_data, sr = model.synthesize(text=text, voice_preset=voice_preset)`
        # We need to know the default or how to set `voice_preset`.
        # Let's assume a default voice_preset or None is acceptable for now.
        voice_preset = None # Or get from request if added to TTSRequest
        audio_data, sample_rate = tts_model.synthesize(text=text, voice_preset=voice_preset)
        
        # Now write this to our BytesIO buffer as a WAV file
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"MCP Tool: Successfully generated audio for: '{text}'")
        return TTSResponse(
            audio_base64=audio_base64,
            content_type="audio/wav",
            text_generated_for=text
        )

    except Exception as e:
        print(f"Error during TTS generation for MCP tool: {e}")
        import traceback
        traceback.print_exc()
        return TTSResponse(error=str(e), text_generated_for=text)

# --- MCP Prompt (Optional, but good for direct interaction) ---
# This allows an AI to "prompt" the MCP server like a user would prompt an LLM.
@mcp_server.prompt("chatterbox_tts")
def tts_prompt(messages: list[Message]) -> list[Message]:
    """Handles a direct prompt to the Chatterbox TTS MCP server.
    Expects the last user message to contain the text to synthesize.
    Responds with a tool call to `generate_speech_tool`.
    """
    if not messages:
        return [Message(role=Role.ASSISTANT, content="No input provided.")]

    last_user_message = next((m for m in reversed(messages) if m.role == Role.USER), None)

    if not last_user_message or not last_user_message.content:
        return [Message(role=Role.ASSISTANT, content="No text found in the last user message.")]

    text_to_synthesize = last_user_message.content
    print(f"MCP Prompt: Received request to synthesize: '{text_to_synthesize}'")

    # Create a tool call message to invoke our TTS tool
    tool_call_id = f"call_tts_{os.urandom(4).hex()}"
    tool_call_message = Message(
        role=Role.ASSISTANT,
        tool_calls=[
            ToolCall(
                id=tool_call_id,
                name="generate_speech",
                arguments={
                    "text": text_to_synthesize
                }
            )
        ]
    )
    return [tool_call_message]

# --- Main Application Runner (FastAPI part) ---
app = mcp_server.create_app() # This creates the FastAPI app from FastMCP

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Uvicorn arguments for running the app directly.
    # In a container, you might use `CMD ["python", "/opt/mcp_server/mcp_chatterbox_tts_server.py"]`
    # and then uvicorn is called here.
    # Or `CMD ["uvicorn", "mcp_chatterbox_tts_server:app", "--host", "0.0.0.0", "--port", "8080"]`
    # if the file is named mcp_chatterbox_tts_server.py
    print(f"Starting Chatterbox TTS MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
