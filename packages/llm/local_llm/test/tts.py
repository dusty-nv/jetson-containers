#!/usr/bin/env python3
from local_llm.utils import ArgParser
from local_llm.plugins import TTS, UserPrompt, AudioOutputDevice, AudioOutputFile, Callback

from termcolor import cprint

args = ArgParser(extras=['tts', 'audio_output', 'prompt', 'log']).parse_args()

def print_prompt():
    cprint('\n>> ', 'blue', end='', flush=True)

def print_help():
    print("Enter text to synthesize, or one of these commands:\n")
    print("  /defaults          Generate a default test sequence")
    print("  /voices            List the voice names")
    print("  /voice Voice Name  Change the voice")
    print("  /languages         List the languages")
    print("  /language en-US    Set the language code")
    print("  /rate 1.0          Set the speaker rate")
    print("  /help or ?         Print the help text")  
    print_prompt()

def commands(text):
    try:
        cmd = text.lower()
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
        elif cmd == "help" or cmd == "?":
            print_help()
        elif len(cmd.strip()) == 0:
            pass
        else:
            return text  # send to TTS
    except Exception as error:
        print(f"\nError: {error}")
    print_prompt()
    
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
