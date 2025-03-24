import logging
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, BeforeValidator, Field, model_validator

from speaches import kokoro_utils, piper_utils
from speaches.api_types import Voice
from speaches.audio import convert_audio_format
from speaches.dependencies import KokoroModelManagerDependency, PiperModelManagerDependency
from speaches.model_aliases import ModelId

DEFAULT_MODEL_ID = kokoro_utils.MODEL_ID
# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-response_format
DEFAULT_RESPONSE_FORMAT = "mp3"
DEFAULT_VOICE_ID = "af"  # TODO: make configurable

# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-voice
# https://platform.openai.com/docs/guides/text-to-speech/voice-options
OPENAI_SUPPORTED_SPEECH_VOICE_NAMES = ("alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse")

# https://platform.openai.com/docs/guides/text-to-speech/supported-output-formats
ResponseFormat = Literal["mp3", "flac", "wav", "pcm"]
SUPPORTED_RESPONSE_FORMATS = ("mp3", "flac", "wav", "pcm")
UNSUPORTED_RESPONSE_FORMATS = ("opus", "aac")

MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000


logger = logging.getLogger(__name__)

router = APIRouter(tags=["speech-to-text"])


def handle_openai_supported_voices(voice_id: str) -> str:
    if voice_id in OPENAI_SUPPORTED_SPEECH_VOICE_NAMES:
        logger.warning(f"{voice_id} is not a valid voice id. Using '{DEFAULT_VOICE_ID}' instead.")
        return DEFAULT_VOICE_ID
    return voice_id


VoiceId = Annotated[str, BeforeValidator(handle_openai_supported_voices)]  # TODO: description and examples


class CreateSpeechRequestBody(BaseModel):
    model: ModelId = DEFAULT_MODEL_ID
    input: str = Field(
        ...,
        description="The text to generate audio for. ",
        examples=[
            "A rainbow is an optical phenomenon caused by refraction, internal reflection and dispersion of light in water droplets resulting in a continuous spectrum of light appearing in the sky. The rainbow takes the form of a multicoloured circular arc. Rainbows caused by sunlight always appear in the section of sky directly opposite the Sun. Rainbows can be caused by many forms of airborne water. These include not only rain, but also mist, spray, and airborne dew."
        ],
    )
    voice: VoiceId = DEFAULT_VOICE_ID
    """
For 'rhasspy/piper-voices' voices the last part of the voice name is the quality (x_low, low, medium, high).
Each quality has a different default sample rate:
- x_low: 16000 Hz
- low: 16000 Hz
- medium: 22050 Hz
- high: 22050 Hz
    """
    language: kokoro_utils.Language | None = None
    """
    Only used for 'hexgrad/Kokoro-82M' models. The language of the text to generate audio for.
    """
    response_format: ResponseFormat = Field(
        DEFAULT_RESPONSE_FORMAT,
        description=f"The format to audio in. Supported formats are {', '.join(SUPPORTED_RESPONSE_FORMATS)}. {', '.join(UNSUPORTED_RESPONSE_FORMATS)} are not supported",
        examples=list(SUPPORTED_RESPONSE_FORMATS),
    )
    # https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-voice
    speed: float = Field(1.0)
    """The speed of the generated audio. 1.0 is the default.
    For 'hexgrad/Kokoro-82M' models, the speed can be set to 0.5 to 2.0.
    For 'rhasspy/piper-voices' models, the speed can be set to 0.25 to 4.0.
    """
    sample_rate: int | None = Field(None, ge=MIN_SAMPLE_RATE, le=MAX_SAMPLE_RATE)
    """Desired sample rate to convert the generated audio to. If not provided, the model's default sample rate will be used.
    For 'hexgrad/Kokoro-82M' models, the default sample rate is 24000 Hz.
    For 'rhasspy/piper-voices' models, the sample differs based on the voice quality (see `voice`).
    """

    @model_validator(mode="after")
    def verify_voice_is_valid(self) -> "CreateSpeechRequestBody":
        if self.model == kokoro_utils.MODEL_ID:
            assert self.voice in kokoro_utils.list_kokoro_voice_names()
        elif self.model == piper_utils.MODEL_ID:
            assert self.voice in piper_utils.read_piper_voices_config()
        return self

    @model_validator(mode="after")
    def validate_speed(self) -> "CreateSpeechRequestBody":
        if self.model == kokoro_utils.MODEL_ID:
            assert 0.5 <= self.speed <= 2.0
        if self.model == piper_utils.MODEL_ID:
            assert 0.25 <= self.speed <= 4.0
        return self


# https://platform.openai.com/docs/api-reference/audio/createSpeech
@router.post("/v1/audio/speech")
async def synthesize(
    piper_model_manager: PiperModelManagerDependency,
    kokoro_model_manager: KokoroModelManagerDependency,
    body: CreateSpeechRequestBody,
) -> StreamingResponse:
    if body.model == kokoro_utils.MODEL_ID:
        # TODO: download the `voices.bin` file
        with kokoro_model_manager.load_model(body.voice) as tts:
            audio_generator = kokoro_utils.generate_audio(
                tts,
                body.input,
                body.voice,
                language=body.language or "en-us",
                speed=body.speed,
                sample_rate=body.sample_rate,
            )
            if body.response_format != "pcm":
                audio_generator = (
                    convert_audio_format(
                        audio_bytes, body.sample_rate or kokoro_utils.SAMPLE_RATE, body.response_format
                    )
                    async for audio_bytes in audio_generator
                )
            return StreamingResponse(audio_generator, media_type=f"audio/{body.response_format}")
    elif body.model == piper_utils.MODEL_ID:
        with piper_model_manager.load_model(body.voice) as piper_tts:
            # TODO: async generator
            audio_generator = piper_utils.generate_audio(
                piper_tts, body.input, speed=body.speed, sample_rate=body.sample_rate
            )
            if body.response_format != "pcm":
                audio_generator = (
                    convert_audio_format(
                        audio_bytes, body.sample_rate or piper_tts.config.sample_rate, body.response_format
                    )
                    for audio_bytes in audio_generator
                )
            return StreamingResponse(audio_generator, media_type=f"audio/{body.response_format}")
    raise HTTPException(status_code=404, detail=f"Model '{body.model}' not found")


@router.get("/v1/audio/speech/voices")
def list_voices(model_id: ModelId | None = None) -> list[Voice]:
    voices: list[Voice] = []
    if model_id == kokoro_utils.MODEL_ID or model_id is None:
        voices.extend(list(kokoro_utils.list_kokoro_voices()))
    elif model_id == piper_utils.MODEL_ID or model_id is None:
        voices.extend(list(piper_utils.list_piper_voices()))

    return voices
