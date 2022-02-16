
print('testing torchaudio...')
import torchaudio
print('torchaudio version: ' + str(torchaudio.__version__))

torchaudio.set_audio_backend("sox_io")
print("found audio backend 'sox_io' OK")

torchaudio.set_audio_backend("soundfile")
print("found audio backend 'soundfile' OK")

print('torchaudio OK\n')
