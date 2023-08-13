#!/usr/bin/env python3
# python3 chat.py --input-device=24 --output-device=24 --sample-rate-hz=48000
import sys
import time
import pprint
import argparse
import threading

import riva.client
import riva.client.audio_io

from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters

from asr import ASR
from tts import TTS
from llm import LLM


def parse_args():
    """
    Parse command-line arguments for configuring the chatbot.
    """
    default_device_info = riva.client.audio_io.get_default_input_device_info()
    default_device_index = None if default_device_info is None else default_device_info['index']
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # audio I/O
    parser.add_argument("--list-devices", action="store_true", help="List output audio devices indices.")
    parser.add_argument("--input-device", type=int, default=default_device_index, help="An input audio device to use.")
    parser.add_argument("--output-device", type=int, help="Output device to use.")
    parser.add_argument("--sample-rate-hz", type=int, default=44100, help="Number of audio frames per second in synthesized audio.")
    parser.add_argument("--audio-chunk", type=int, default=1600, help="A maximum number of frames in a audio chunk sent to server.")
    parser.add_argument("--audio-channels", type=int, default=1, help="The number of audio channels to use")
    
    # ASR/TTS settings
    parser.add_argument("--voice", type=str, default='English-US.Female-1', help="A voice name to use for TTS")
    parser.add_argument("--no-punctuation", action='store_true', help="Disable ASR automatic punctuation")
    
    # LLM settings
    parser.add_argument("--llm-server", type=str, default='0.0.0.0', help="hostname of the LLM server (text-generation-webui)")
    parser.add_argument("--llm-api-port", type=int, default=5000, help="port of the blocking API on the LLM server")
    parser.add_argument("--llm-streaming-port", type=int, default=5005, help="port of the streaming websocket API on the LLM server")
    
    parser = add_asr_config_argparse_parameters(parser, profanity_filter=True)
    parser = add_connection_argparse_parameters(parser)
    
    args = parser.parse_args()

    args.automatic_punctuation = not args.no_punctuation
    args.verbatim_transcripts = not args.no_verbatim_transcripts
    
    print(args)
    return args
    
 
class Chatbot(threading.Thread):
    """
    LLM-based chatbot with streaming ASR/TTS 
    """
    def __init__(self, args, **kwargs):
        super(Chatbot, self).__init__()
        
        self.args = args
        self.auth = riva.client.Auth(args.ssl_cert, args.use_ssl, args.server)
        
        self.asr = ASR(self.auth, callback=self.on_asr_transcript, **vars(args))
        self.tts = TTS(self.auth, **vars(args))
        self.llm = LLM(**vars(args))
        
        self.asr_history = ""
        
    def on_asr_transcript(self, result):
        """
        Recieve new ASR responses
        """
        transcript = result.alternatives[0].transcript.strip()
        
        if result.is_final:
            print(f"## {transcript} ({result.alternatives[0].confidence})")
            self.tts.generate(transcript)
            self.llm.generate(transcript)
        else:
            if transcript != self.asr_history:
                print(f">> {transcript}")
                
        self.asr_history = transcript
        
    def run(self):
        self.asr.start()
        self.tts.start()
        self.llm.start()
        
        while True:
            time.sleep(1.0)
            
    
if __name__ == '__main__':
    args = parse_args()
     
    if args.list_devices:
        riva.client.audio_io.list_output_devices()
        sys.exit(0)
    
    chatbot = Chatbot(args)
    chatbot.start()
    