#!/usr/bin/env python3
#
# Test of streaming ASR transcription on a live microphone:
#
#    python3 -m local_llm.test.asr --verbose \
#        --asr riva \
#        --sample-rate-hz 44100 \
#        --audio-input-device 25
#
# The sample rate should be set to one that the audio output device supports (like 16000, 44100,
# 48000, ect).  This command will list the connected audio devices available:
#
#    python3 -m local_llm.test.asr --list-audio-devices
#
from local_llm.plugins import AutoASR, PrintStream
from local_llm.utils import ArgParser

args = ArgParser(extras=['asr', 'audio_input', 'log']).parse_args()
asr = AutoASR.from_pretrained(**vars(args))

asr.add(PrintStream(partial=False, prefix='## ', color='green'), AutoASR.OutputFinal)
asr.add(PrintStream(partial=False, prefix='>> ', color='blue'), AutoASR.OutputPartial)

asr.start().join()
