#!/usr/bin/env python3
import os
import logging
import threading
import numpy as np

from local_llm.web import WebServer
from local_llm.utils import ArgParser
from local_llm.plugins import RivaASR

from .voice_chat import VoiceChat


class WebChat(VoiceChat):
    """
    Adds webserver to ASR/TTS voice chat agent.
    """
    def __init__(self, **kwargs):
        """
        Parameters:
        
          upload_dir (str) -- the path to save files uploaded from the client
          
        See VoiceChat and WebServer for inherited arguments.
        """
        super().__init__(**kwargs)

        self.asr.add(self.on_asr_partial, RivaASR.OutputPartial)
        #self.asr.add(self.on_asr_final, RivaASR.OutputFinal)
        
        self.llm.add(self.on_llm_reply)
        self.tts_output.add(self.on_tts_samples)
        
        self.server = WebServer(msg_callback=self.on_message, **kwargs)
        
    def on_message(self, msg, msg_type=0, metadata='', **kwargs):
        if msg_type == WebServer.MESSAGE_JSON:
            if 'chat_history_reset' in msg:
                self.llm.chat_history.reset()
                self.send_chat_history()
            if 'client_state' in msg:
                if msg['client_state'] == 'connected':
                    threading.Timer(1.0, lambda: self.send_chat_history()).start()
            if 'tts_voice' in msg:
                self.tts.voice = msg['tts_voice']
        elif msg_type == WebServer.MESSAGE_TEXT:  # chat input
            self.prompt(msg.strip('"'))
        elif msg_type == WebServer.MESSAGE_AUDIO:  # web audio (mic)
            self.asr(msg)
        elif msg_type == WebServer.MESSAGE_IMAGE:
            logging.info(f"recieved {metadata} image message {msg.size} -> {msg.filename}")
            self.llm.chat_history.reset()
            self.llm.chat_history.append(role='user', image=msg)
            self.send_chat_history()
        else:
            logging.warning(f"WebChat agent ignoring websocket message with unknown type={msg_type}")
    
    def on_asr_partial(self, text):
        self.send_chat_history()
        threading.Timer(1.5, self.on_asr_waiting, args=[text]).start()
        
    def on_asr_waiting(self, transcript):
        if self.asr_history == transcript:  # if the partial transcript hasn't changed, probably a misrecognized sound or echo
            logging.warning(f"ASR partial transcript has stagnated, dropping from chat ({self.asr_history})")
            self.asr_history = None
            self.send_chat_history() # drop the rejected ASR from the client

    def on_llm_reply(self, text):
        self.send_chat_history()
        
    def on_tts_samples(self, audio):
        self.server.send_message(audio, type=WebServer.MESSAGE_AUDIO)
        
    def send_chat_history(self, history=None):
        # TODO convert images to filenames
        # TODO sanitize text for HTML
        if history is None:
            history = self.llm.chat_history
            
        history = history.to_list()
        
        if self.asr_history:
            history.append({'role': 'user', 'text': self.asr_history})
            
        def web_text(text):
            text = text.strip()
            text = text.strip('\n')
            text = text.replace('\n', '<br/>')
            text = text.replace('<s>', '')
            text = text.replace('</s>', '')
            return text
          
        def web_image(image):
            if not isinstance(image, str):
                if not hasattr(image, 'filename'):
                    return None
                image = image.filename
            return os.path.join(self.server.upload_route, os.path.basename(image))
            
        for entry in history:
            if 'text' in entry:
                entry['text'] = web_text(entry['text'])
            if 'image' in entry:
                entry['image'] = web_image(entry['image'])
                
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
    