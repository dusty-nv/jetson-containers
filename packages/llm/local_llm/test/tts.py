#!/usr/bin/env python3
#
# Interactively test streaming TTS models, with support for live output and writing to wav files.
# See the print_help() function below for a description of the commands you can enter this program.
# Here is an example of starting it with XTTS model and sound output:
#
#    python3 -m local_llm.test.tts --verbose \
#	    --tts xtts \
#	    --voice 'Damien Black' \
#	    --sample-rate-hz 44100 \
#       --audio-output-device 25 \
#	    --audio-output-file /data/audio/tts/test.wav
#
# The sample rate should be set to one that the audio output device supports (like 16000, 44100,
# 48000, ect).  This command will list the connected audio devices available:
#
#    python3 -m local_llm.test.tts --list-audio-devices
#
# The TTS output is automatically resampled to match the sampling rate of the audio device.
#
import sys
import termcolor

from local_llm.utils import ArgParser, KeyboardInterrupt
from local_llm.plugins import AutoTTS, UserPrompt, AudioOutputDevice, AudioOutputFile, Callback

args = ArgParser(extras=['tts', 'audio_output', 'prompt', 'log']).parse_args()

def print_prompt():
    termcolor.cprint('\n>> ', 'blue', end='', flush=True)

def print_help():
    print(f"Enter text to synthesize, or one of these commands:\n")
    print(f"  /defaults          Generate a default test sequence")
    print(f"  /voices            List the voice names")
    print(f"  /voice Voice Name  Change the voice (current='{tts.voice}')")
    print(f"  /languages         List the languages")
    print(f"  /language en-US    Set the language code (current='{tts.language}')")
    print(f"  /rate 1.0          Set the speaker rate (current={tts.rate:.2f})")
    print(f"  /buffer none       Disable input buffering (current='{','.join(tts.buffering)}')")
    print(f"  /interrupt or /i   Interrupt/mute the TTS output")
    print(f"  /help or /?        Print the help text")  
    print(f"  /quit or /exit     Exit the program\n")
    print(f"Press Ctrl+C to interrupt output, and Ctrl+C twice to exit.")
    print_prompt()

def commands(text):
    try:
        cmd = text.lower().strip()
        if cmd.startswith('/default'):
            tts("Hello there, how are you today? ")
            tts("The weather is 76 degrees out and sunny. ")
            tts("Your first meeting is in an hour downtown, with normal traffic. ")
            tts("Can I interest you in anything quick for breakfast?")
        elif cmd.startswith('/voices'):
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
        elif cmd.startswith('/i'):
            on_interrupt()
        elif cmd.startswith('/quit') or cmd.startswith('/exit'):
            sys.exit(0)
        elif cmd.startswith('/h') or cmd.startswith('/?'):
            print_help()
        elif len(cmd.strip()) == 0:
            pass
        else:
            return text  # send to TTS
    except Exception as error:
        print(f"\nError: {error}")
    print_prompt()
 
def on_interrupt():
    tts.interrupt()
    print_prompt()

tts = AutoTTS.from_pretrained(**vars(args))

interrupt = KeyboardInterrupt(callback=on_interrupt)

if args.audio_output_device is not None:
    tts.add(AudioOutputDevice(**vars(args)))

if args.audio_output_file is not None:
    tts.add(AudioOutputFile(**vars(args)))

prompt = UserPrompt(interactive=True, **vars(args)).add(
    Callback(commands).add(tts)
)

print_help()
prompt.start().join()
