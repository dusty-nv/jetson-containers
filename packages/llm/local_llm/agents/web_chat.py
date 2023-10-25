#!/usr/bin/env python3
import os
import logging
import threading

from local_llm.utils import ArgParser
from local_llm.web import WebServer

from .voice_chat import VoiceChat


class WebChat(VoiceChat):
    """
    Adds webserver to ASR/TTS voice chat agent.
    """
    MESSAGE_AUDIO = 2
    MESSAGE_IMAGE = 3
    
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

        self.llm.add(self.on_llm, threaded=True)
        
        self.server = WebServer(msg_callback=self.on_message, **kwargs)
        
    def on_message(self, msg, type, timestamp):
        if type == WebServer.MESSAGE_JSON:
            if 'chat_history_reset' in msg:
                self.llm.chat_history.reset()
                self.send_chat_history(self.llm.chat_history)
            if 'client_state' in msg:
                if msg['client_state'] == 'connected':
                    threading.Timer(1.0, lambda: self.send_chat_history(self.llm.chat_history)).start()
            if 'tts_voice' in msg:
                self.tts.voice = msg['tts_voice']
        elif type == WebServer.MESSAGE_TEXT:  # chat input
            self.prompt(msg)
        elif type == WebChat.MESSAGE_AUDIO:  # web audio (mic)
            self.asr(msg)
     
    def on_llm(self, text):
        self.send_chat_history(self.llm.chat_history)
        
    def on_tts(self, audio):
        self.server.send_message(audio, type=WebChat.MESSAGE_AUDIO)
        
    def send_chat_history(self, history):
        # TODO convert images to filenames
        # TODO sanitize text for HTML
        history = history.to_list()
        
        #def translate_web(text):
        #    text = text.replace('\n', '<br/>')
        #    return text
            
        #for n in range(len(history)):
        #    for m in range(len(history[n])):
        #        history[n][m] = translate_web(history[n][m])
                
        #logging.debug(f"sending chat history {history}")
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
    