#!/usr/bin/env python3
#
# Feed ASR transcript into streaming TTS, your words are played back to you in another voice.
#
#    python3 -m local_llm.test.asr_tts_loopback
#        --asr riva \
#        --tts xtts \
#        --sample-rate-hz 44100 \
#        --audio-input-device 25
#
# The sample rate should be set to one that the audio output device supports (like 16000, 44100,
# 48000, ect).  This command will list the connected audio devices available:
#
#    python3 -m local_llm.test.asr_tts_loopback --list-audio-devices
#
import time

from local_llm.plugins import AutoASR, AutoTTS, AudioOutputDevice, AudioOutputFile, PrintStream
from local_llm.utils import ArgParser

args = ArgParser(extras=['asr', 'tts', 'audio_input', 'audio_output', 'log']).parse_args()

asr = AutoASR.from_pretrained(**vars(args))
tts = AutoTTS.from_pretrained(**vars(args))

asr.add(PrintStream(partial=False, prefix='## ', color='green'), AutoASR.OutputFinal)
asr.add(PrintStream(partial=False, prefix='>> ', color='blue'), AutoASR.OutputPartial)

asr.add(tts, AutoASR.OutputFinal)

if args.audio_output_device is not None:
    tts.add(AudioOutputDevice(**vars(args)))

if args.audio_output_file is not None:
    tts.add(AudioOutputFile(**vars(args)))
    
asr.start()

def print_help():
    print(f"\nSpeak into the mic, or enter these commands:\n")
    print(f"  /voices            List the voice names")
    print(f"  /voice Voice Name  Change the voice (current='{tts.voice}')")
    print(f"  /languages         List the languages")
    print(f"  /language en-US    Set the language code (current='{tts.language}')")
    print(f"  /rate 1.0          Set the speaker rate (current={tts.rate:.2f})")
    print(f"  /buffer none       Disable input buffering (current='{','.join(tts.buffering)}')")
    print(f"  /help or /?        Print the help text\n")  
    print(f"Press Ctrl+C to exit.\n")

time.sleep(2.5)
print_help()

while True:
    try:
        text = input()
        cmd = text.lower()
        
        if cmd.startswith('/voices'):
            print(tts.voices)
        elif cmd.startswith('/voice'):
            tts.voice = text[6:].strip()
        elif cmd.startswith('/languages'):
            print(tts.languages)
        elif cmd.startswith('/language'):
            tts.language = text[9:].strip()
        elif cmd.startswith('/rate'):
            tts.rate = float(cmd.split(' ')[1])
        elif cmd.startswith('/buffer'):
            tts.buffering = text[7:].strip()
        elif cmd.startswith('/h') or cmd.startswith('/?'):
            print_help()
    except Exception as error:
        print(f"\nError: {error}")
        print_help()
