#!/usr/bin/env python3
import sys

from local_llm.utils import ArgParser, KeyboardInterrupt
from local_llm.plugins import TTS, UserPrompt, AudioOutputDevice, AudioOutputFile, Callback

from termcolor import cprint

args = ArgParser(extras=['tts', 'audio_output', 'prompt', 'log']).parse_args()

def print_prompt():
    cprint('\n>> ', 'blue', end='', flush=True)

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
    print(f"Press Ctrl+C to interrupt output, and Ctrl+C twice to exit")
    print_prompt()

def commands(text):
    try:
        cmd = text.lower().strip()
        if cmd.startswith('/default'):
            tts("Hello there, how are you today?")
            tts("The weather is 76 degrees out and sunny.")
            tts("Your first meeting is in an hour downtown, with normal traffic.")
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

interrupt = KeyboardInterrupt(callback=on_interrupt)

tts = TTS(**vars(args))

if args.audio_output_device is not None:
    tts.add(AudioOutputDevice(**vars(args)))

if args.audio_output_file is not None:
    tts.add(AudioOutputFile(**vars(args)))

prompt = UserPrompt(interactive=True, **vars(args)).add(
    Callback(commands).add(tts)
)

print_help()
prompt.start().join()
