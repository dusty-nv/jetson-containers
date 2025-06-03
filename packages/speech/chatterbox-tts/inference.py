import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
reference_audio_path = "path/to/reference_audio.wav"  # Specify the path to the audio prompt
wav = model.generate(text, reference_audio_path=reference_audio_path)
ta.save("test-2.wav", wav, model.sr)
