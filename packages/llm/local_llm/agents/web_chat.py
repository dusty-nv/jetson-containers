#!/usr/bin/env python3
import os
import ssl
import flask
import queue
import pprint
import logging
import threading

from local_llm.utils import ArgParser
from local_llm.web import WebServer

from .voice_chat import VoiceChat

from websockets.sync.server import serve as websocket_serve

class WebChat(VoiceChat):
    """
    Adds webserver to ASR/TTS voice chat agent.
    """
    def __init__(self, **kwargs):
        """
        Parameters:
        
          web_host (str) -- network interface to bind to (0.0.0.0 for all)
          web_port (int) -- port to serve HTTP/HTTPS webpages on
          ws_port (int) -- port to use for websocket communication
          ssl_cert (str) -- path to PEM-encoded SSL/TLS cert file for enabling HTTPS
          ssl_key (str) -- path to PEM-encoded SSL/TLS cert key for enabling HTTPS
        """
        super().__init__(**kwargs)

        self.server = WebServer(msg_callback=self.on_message, **kwargs)
        
    def on_message(self, payload, type, timestamp):
        if type == 0:  # JSON
            if 'chat_history_reset' in msg:
                self.llm.chat_history.reset()
                self.send_chat_history(self.llm.chat_history)
            if 'client_state' in msg:
                if msg['client_state'] == 'connected':
                    threading.Timer(1.0, lambda: self.send_chat_history(self.llm.chat_history)).start()
            if 'tts_voice' in msg:
                self.tts.voice = msg['tts_voice']
        elif type == 1:  # text (chat input)
            self.prompt(msg)
        elif type == 2:  # web audio (mic)
            self.asr(msg)
            
    def send_chat_history(self, history):
        history = copy.deepcopy(history)
        
        def translate_web(text):
            text = text.replace('\n', '<br/>')
            return text
            
        for n in range(len(history)):
            for m in range(len(history[n])):
                history[n][m] = translate_web(history[n][m])
                
        #print("-- sending chat history", history)
        self.server.send_message({'chat_history': history})
 
    def start(self):
        super().start()
        self.server.start()
        return self


if __name__ == "__main__":
    from local_llm.utils import ArgParser

    parser = ArgParser(extras=ArgParser.Defaults+['asr', 'tts', 'audio_output', 'web'])
    args = parser.parse_args()
    
    agent = WebChat(**vars(args)).run() 
    