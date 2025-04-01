#!/usr/bin/env python3
# python3 chat.py --input-device=24 --output-device=24 --sample-rate-hz=48000
import os
import sys
import time
import pprint
import signal
import argparse
import termcolor
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
from webserver import Webserver
from tegrastats import Tegrastats


def parse_args():
    """
    Parse command-line arguments for configuring the chatbot.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # audio I/O
    parser.add_argument("--list-devices", action="store_true", help="List output audio devices indices.")
    parser.add_argument("--input-device", type=int, default=None, help="An input audio device to use.")
    parser.add_argument("--output-device", type=int, default=None, help="Output device to use.")
    parser.add_argument("--sample-rate-hz", type=int, default=48000, help="Number of audio frames per second in synthesized audio.")
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
    
    # Webserver settings
    parser.add_argument("--web-server", type=str, default='0.0.0.0', help="interface to bind to for hosting the chat webserver")
    parser.add_argument("--web-port", type=int, default=8050, help="port used for webserver HTTP/HTTPS")
    parser.add_argument("--ssl-key", default=os.getenv('SSL_KEY'), type=str, help="path to PEM-encoded SSL/TLS key file for enabling HTTPS")
    
    # verbose/debug logging
    parser.add_argument("--log-level", type=str, default='info', choices=['info', 'verbose', 'debug'], help="logging level")
    parser.add_argument("--verbose", action='store_true', help="enable verbose logging")
    parser.add_argument("--debug", action='store_true', help="enable debug logging (extra verbose)")
    
    parser = add_asr_config_argparse_parameters(parser, profanity_filter=True)
    parser = add_connection_argparse_parameters(parser)
    
    args = parser.parse_args()

    args.automatic_punctuation = not args.no_punctuation
    args.verbatim_transcripts = not args.no_verbatim_transcripts
    
    # setup logging level
    if args.debug:
        args.verbose = True
        args.log_level = 'debug'
    elif args.verbose:
        args.log_level = 'verbose'
        
    if args.log_level == 'info':
        args.log_level = 0
    elif args.log_level == 'verbose':
        args.log_level = 1
    elif args.log_level == 'debug':
        args.log_level = 2
        
    if not args.ssl_cert:
        args.ssl_cert = os.getenv('SSL_CERT')
        
    print(args)
    return args
    
 
class Chatbot(threading.Thread):
    """
    LLM-based chatbot with streaming ASR/TTS.
    This class essentially routes different requests/responses between the services.
    """
    def __init__(self, args, **kwargs):
        super(Chatbot, self).__init__()
        
        self.args = args
        self.auth = riva.client.Auth(uri=args.server) # args.ssl_cert, args.use_ssl
        
        self.asr = ASR(self.auth, callback=self.on_asr_transcript, **vars(args)) #if args.input_device is not None else None
        self.tts = TTS(self.auth, **vars(args))
        self.llm = LLM(**vars(args))
        
        self.webserver = Webserver(msg_callback=self.on_websocket_msg, **vars(args))
        self.tegrastats = Tegrastats(callback=self.on_tegrastats, **vars(args))
        self.audio_mixer = AudioMixer(callback=self.on_mixed_audio, **vars(args))

        self.asr_history = ""
        self.tts_history = ""
        self.llm_history = LLM.create_new_history()
        
        self.llm_timer = None
        self.tts_track = None
        self.log_level = args.log_level
        
        self.last_sigint = 0.0
        
        #signal.signal(signal.SIGINT, self.on_sigint)

    def mute(self):
        """
        Interrupt the bot by muting audio and cancelling LLM/TTS requests
        """
        print("-- muting chatbot output")
        
        if self.llm:
            self.llm.mute()
            
        if self.tts:
            self.tts.mute()
            
        if self.tts_track:
            self.tts_track['status'] = 'done'

    def on_sigint(self, signum, frame):
        """
        Ctrl+C handler - if done once, mutes the LLM/TTS
        If done twice in succession, exits the program
        """
        curr_time = time.perf_counter()
        time_diff = curr_time - self.last_sigint
        self.last_sigint = curr_time
        
        if time_diff > 2.0:
            print("-- Ctrl+C:  muting chatbot")
            self.mute()
        else:
            while True:
                print("-- Ctrl+C:  exiting...")
                sys.exit(0)
                time.sleep(0.5)
    
    def on_websocket_msg(self, msg, type, timestamp):
        """
        Recieve websocket message from client
        """
        if type == 0:  # JSON
            if 'chat_history_reset' in msg:
                self.llm_history = LLM.create_new_history()
                self.webserver.send_chat_history(self.llm_history['internal'])
            if 'client_state' in msg:
                if msg['client_state'] == 'connected':
                    threading.Timer(1.0, lambda: self.webserver.send_chat_history(self.llm_history['internal'])).start()
            if 'tts_voice' in msg:
                self.tts.voice = msg['tts_voice']
        elif type == 1:  # text (chat input)
            self.on_llm_prompt(msg)
        elif type == 2:  # web audio (mic)
            self.asr.process(msg)
    
    def on_tegrastats(self, stats):
        """
        Recieve update system stats
        """
        self.webserver.send_message({'tegrastats': stats['summary']})
        
        if self.log_level == 0:
            print(f"-- tegrastats:  {stats['summary']}")
            
    def on_mixed_audio(self, audio, silent):
        """
        Send mixed-down audio from the TTS/ect to web client
        """
        if not silent:
            self.webserver.send_message(audio, type=2)
            
    def on_asr_transcript(self, result):
        """
        Recieve new ASR responses
        """
        transcript = result.alternatives[0].transcript.strip()
        
        if result.is_final:
            confidence = result.alternatives[0].confidence
            termcolor.cprint(f"## {transcript} ({confidence})", "green")
            if confidence > -2.0: #len(transcript.split(' ')) > 1:
                self.asr_history = transcript
                self.on_llm_prompt(transcript)
            else:
                self.webserver.send_chat_history(self.llm_history['internal']) # drop the rejected ASR from the client
        else:
            if transcript != self.asr_history:
                self.asr_history = transcript
                termcolor.cprint(f">> {transcript}", "green") 
                
                if len(self.asr_history.split(' ')) >= 3:
                    self.mute()
                    
                web_history = LLM.add_prompt_history(self.llm_history, transcript) # show streaming ASR on the client
                self.webserver.send_chat_history(web_history)
                
                threading.Timer(1.5, self.on_asr_waiting, args=[self.asr_history]).start()
    
    def on_asr_waiting(self, transcript):
        """
        If the ASR partial transcript has stagnated and not "gone final", then it was probably a misque and hsould be dropped
        """
        if self.asr_history == transcript:  # if the partial transcript hasn't changed, probably a misrecognized sound or echo
            self.asr_history = ""
            self.webserver.send_chat_history(self.llm_history['internal']) # drop the rejected ASR from the client

    def on_llm_prompt(self, prompt):
        """
        Send the LLM the next chat message) from the user
        """
        self.mute()  # interrupt any ongoing bot output
        self.audio_mixer.play(tone={'note': 'C', 'duration': 0.25, 'attack': 0.05, 'amplitude': 0.5})
        self.llm.generate_chat(prompt, self.llm_history, max_new_tokens=self.args.max_new_tokens, callback=self.on_llm_reply)

        web_history = LLM.add_prompt_history(self.llm_history, prompt) # show prompt on the client before reply is ready
        self.webserver.send_chat_history(web_history)
        
        self.llm_timer = threading.Timer(1.0, self.on_llm_waiting, args=[web_history]) # add a "..." chat bubble
        self.llm_timer.start()
        
    def on_llm_waiting(self, history):
        """
        Called when the user is waiting for an LLM reply to provide a "..." update
        """
        history[-1].append("...")
        self.webserver.send_chat_history(history)
        self.llm_timer = None
                        
    def on_llm_reply(self, response, request, end):
        """
        Recieve replies from the LLM
        """  
        if self.llm_timer is not None:
            self.llm_timer.cancel()
            self.llm_timer = None
            
        if not end:
            if request['type'] == 'completion':
                termcolor.cprint(f"<< {response}", "blue")
                #print(response, end='')
                #sys.stdout.flush()
            elif request['type'] == 'chat':
                current_length = request.get('current_length', 0)
                msg = response['internal'][-1][1][current_length:]
                request['current_length'] = current_length + len(msg)
                self.llm_history = response
                self.send_tts(msg)
                self.webserver.send_chat_history(response['internal'])
                termcolor.cprint(f"<< {response['internal'][-1][1]}", "blue")
                #print(msg, end='')
                #sys.stdout.flush()
        else:
            self.send_tts(end=True)
            #print("\n")
        
    def on_tts_audio(self, audio, request):
        """
        Recieve audio output from the TTS
        """
        if not self.tts_track or self.tts_track['status'] != 'playing':
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
            if self.log_level > 0:
                print(f"-- TTS:  '{txt}'")
            self.tts.generate(txt, callback=self.on_tts_audio)

    def run(self):
        """
        Chatbot thread main()
        """
        self.tegrastats.start()
        
        self.asr.start()
        self.tts.start()
        self.llm.start()
        
        self.audio_mixer.start()
        self.webserver.start()

        time.sleep(0.5)
        
        num_msgs = 0
        num_users = 2
        note_step = -8
        
        while True:
            if not self.asr: #or self.interrupt_flag:
                #self.interrupt_flag = False
                sys.stdout.write(">> PROMPT: ")
                prompt = sys.stdin.readline().strip() #input()  # https://stackoverflow.com/a/74325860
                print('PROMPT => ', prompt)
                request = self.llm.generate_chat(prompt, self.llm_history, max_new_tokens=self.args.max_new_tokens, callback=self.on_llm_reply)
                request['event'].wait()
            else:
                time.sleep(1.0)
                
                """
                self.audio_mixer.play(tone={
                    'frequency': 440 * 2** ((1 + note_step) / 12),
                    'duration': 0.25,
                    'vibrato_frequency': 2.5, 
                    'vibrato_variance': 5
                })

                note_step += 1
                
                if note_step > 8:
                    note_step = -8
                    self.audio_mixer.play(wav="/opt/riva/python-clients/data/examples/en-US_AntiBERTa_for_word_boosting_testing.wav") # en-US_sample.wav  en-US_percent.wav  en-US_AntiBERTa_for_word_boosting_testing.wav
                
                time.sleep(1.5)
                """
                
                #self.webserver.send_message("abc".encode('utf-8'), 1);
                """
                self.webserver.send_message({
                    'id': num_msgs,
                    'type': 'message',
                    'text': f"This is message {num_msgs}",
                    'user': (num_msgs % num_users)
                })
                
                time.sleep(1.0)
                
                self.webserver.send_message({
                    'id': num_msgs,
                    'type': 'message',
                    'text': f"This is message {num_msgs} (updated)",
                    'user': (num_msgs % num_users)
                })
                
                num_msgs += 1
                time.sleep(0.5)
                """
                
if __name__ == '__main__':
    args = parse_args()
     
    if args.list_devices:
        riva.client.audio_io.list_output_devices()
        sys.exit(0)
    
    chatbot = Chatbot(args)
    chatbot.run() #start()
