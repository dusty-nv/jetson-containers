#!/usr/bin/env python3
# python3 chat.py --input-device=24 --output-device=24 --sample-rate-hz=48000
import sys
import time
import pprint
import signal
import argparse
import readline
import threading
import subprocess
import numpy as np

import riva.client
import riva.client.audio_io

from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters

from asr import ASR
from tts import TTS
from llm import LLM

from audio import AudioMixer


def parse_args():
    """
    Parse command-line arguments for configuring the chatbot.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # audio I/O
    parser.add_argument("--list-devices", action="store_true", help="List output audio devices indices.")
    parser.add_argument("--input-device", type=int, default=None, help="An input audio device to use.")
    parser.add_argument("--output-device", type=int, default=None, help="Output device to use.")
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
    parser.add_argument("--max-new-tokens", type=int, default=256, help="maximum number of new tokens for the LLM to generate for a reply")
    
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
        
        self.audio_mixer = AudioMixer(**vars(args))

        self.asr = ASR(self.auth, callback=self.on_asr_transcript, **vars(args)) if args.input_device is not None else None
        self.tts = TTS(self.auth, **vars(args)) if self.audio_mixer.opened else None
        self.llm = LLM(**vars(args))
        
        self.asr_history = ""
        self.tts_history = ""
        self.llm_history = {'internal': [], 'visible': []}
        
        self.tts_track = None
        
        #self.interrupt_flag = False
 
    def mute(self):
        """
        Mutes output audio and cancels any text or TTS requests being generated for it
        """
        if self.llm:
            self.mute()
            
        if self.tts:
            self.tts.mute()
            
        if self.tts_track:
            self.tts_track['status'] = 'muted'
            
    def on_asr_transcript(self, result):
        """
        Recieve new ASR responses
        """
        transcript = result.alternatives[0].transcript.strip()
        
        if result.is_final:
            confidence = result.alternatives[0].confidence
            print(f"## {transcript} ({confidence})")
            if confidence > -2.0: #len(transcript.split(' ')) > 1:
                self.llm.generate_chat(transcript, self.llm_history, max_new_tokens=self.args.max_new_tokens, callback=self.on_llm_reply)
        else:
            if transcript != self.asr_history:
                print(f">> {transcript}")   
                self.asr_history = transcript
                if len(self.asr_history.split(' ')) > 1:
                    self.mute()
                        
    def on_llm_reply(self, response, request, end):
        """
        Recieve replies from the LLM
        """  
        if not end:
            if request['type'] == 'completion':
                print(response, end='')
                sys.stdout.flush()
            elif request['type'] == 'chat':
                current_length = request.get('current_length', 0)
                msg = response['visible'][-1][1][current_length:]
                request['current_length'] = current_length + len(msg)
                self.llm_history = response
                self.send_tts(msg)
                print(msg, end='')
                sys.stdout.flush()
        else:
            self.send_tts(end=True)
            print("\n")
    
    def on_tts_audio(self, audio, request):
        """
        Recieve audio output from the TTS
        """
        if not self.tts_track or self.tts_track['status'] == 'done':
            self.tts_track = self.audio_mixer.play(audio)
        else:
            self.tts_track['samples'] = np.append(self.tts_track['samples'], audio)

    def send_tts(self, msg=None, end=False):
        """ 
        Buffer and dispatch text to the TTS service 
        """
        if not self.tts:
            return
            
        if msg:
            self.tts_history += msg

        txt = None
        
        if end:
            txt = self.tts_history
            self.tts_history = ""
        elif self.tts.needs_text():
            if False: #len(self.tts_history.split(' ')) > 5:
                txt = self.tts_history
                self.tts_history = ""
            else:
                punctuation = ['.', ',', '!', '?']
                idx = max([self.tts_history.rfind(x) for x in punctuation])
                if idx >= 0:
                    txt = self.tts_history[:idx+1]
                    self.tts_history = self.tts_history[idx+1:]

        if txt:
            self.tts.generate(txt, callback=self.on_tts_audio)
            
            
    #def interrupt():
    #    if self.interrupt_flag:
    #        sys.exit()
    #    self.interrupt_flag = True
        
    def run(self):
        if self.asr:
            self.asr.start()
            
        if self.tts:
            self.tts.start()

        if self.llm:
            self.llm.start()
        
        time.sleep(0.5)
        
        while True:
            if not self.asr: #or self.interrupt_flag:
                #self.interrupt_flag = False
                sys.stdout.write(">> PROMPT: ")
                prompt = input()
                print('PROMPT => ', prompt)
                request = self.llm.generate_chat(prompt, self.llm_history, max_new_tokens=self.args.max_new_tokens, callback=self.on_llm_reply)
                request['events']['end'].wait()
            else:
                time.sleep(1.0)
 

if __name__ == '__main__':
    args = parse_args()
     
    if args.list_devices:
        riva.client.audio_io.list_output_devices()
        sys.exit(0)
    
    chatbot = Chatbot(args)
    chatbot.start()
    
    #def sigint_handler(signum, frame):
    #    chatbot.interrupt()
