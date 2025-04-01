#!/usr/bin/env python3
import time
import queue
import pprint
import threading

import riva.client
import riva.client.audio_io

import numpy as np


class TTS(threading.Thread):
    """
    Streaming TTS service
    """
    def __init__(self, auth, language_code='en-US', voice='English-US.Female-1', sample_rate_hz=44100, **kwargs): 
          
        super(TTS, self).__init__()
        
        self.queue = queue.Queue()
        self.voice = voice
        self.muted = False
        
        self.language_code = language_code
        self.sample_rate = sample_rate_hz
        self.request_count = 0
        self.needs_text_by = 0.0

        self.tts_service = riva.client.SpeechSynthesisService(auth)

    def generate(self, text, voice=None, callback=None):
        """
        Generate an asynchronous request to synthesize speech from the given text.
        The voice can be changed for each request if one is provided (otherwise the default will be used)
        If the callback function is provided, it will be called as the audio chunks are streamed in.
        This function returns the request that was queued.
        """
        request = {
            'id': self.request_count,
            'text': text.strip(),
            'voice': voice if voice else self.voice,
            'callback': callback
        }
        
        self.request_count += 1
        self.queue.put(request)
        return request

    def mute(self):
        """
        Mutes the TTS until another request comes in
        """
        self.muted = True
     
    def needs_text(self):
        """
        Returns true if the TTS needs text to keep the audio flowing.
        """
        return (time.perf_counter() > self.needs_text_by)
        
    def run(self):
        print(f"-- running TTS service ({self.language_code}, {self.voice})")
        
        while True:
            request = self.queue.get()
            self.muted = False
            
            #print(f"-- TTS:  '{request['text']}'")
            
            responses = self.tts_service.synthesize_online(
                request['text'], request['voice'], self.language_code, sample_rate_hz=self.sample_rate
            )
            
            num_samples = 0

            for response in responses:
                if self.muted:
                    print(f"-- TTS muted, exiting request early:  {request['text']}")
                    break
                    
                samples = np.frombuffer(response.audio, dtype=np.int16)

                current_time = time.perf_counter()
                if current_time > self.needs_text_by:
                    self.needs_text_by = current_time
                self.needs_text_by += len(samples) / self.sample_rate
                
                if request['callback'] is not None:
                    request['callback'](samples, request)
            