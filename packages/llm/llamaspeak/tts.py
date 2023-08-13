#!/usr/bin/env python3
import queue
import pprint
import threading

import riva.client
import riva.client.audio_io


class TTS(threading.Thread):
    """
    Streaming TTS service
    """
    def __init__(self, auth, output_device=0, sample_rate_hz=44100, audio_channels=1, 
                 language_code='en-US', voice='English-US.Female-1', **kwargs):
                 
        super(TTS, self).__init__()
        
        self.queue = queue.Queue()
        self.voice = voice
        
        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz
        self.request_count = 0
        
        self.tts_service = riva.client.SpeechSynthesisService(auth)
        
        self.output_stream = riva.client.audio_io.SoundCallBack(
            output_device, nchannels=audio_channels, sampwidth=2, framerate=sample_rate_hz
        ).__enter__()

    def generate(self, text, voice=None, callback=None):
        """
        Generate an asynchronous request to synthesize speech from the given text.
        The voice can be changed for each request if one is provided (otherwise the default will be used)
        If the callback function is provided, it will be called as the audio chunks are streamed in.
        This function returns the request that was queued.
        """
        request = {
            'id': self.request_count,
            'text': text,
            'voice': voice if voice else self.voice,
            'callback': callback
        }
        
        self.request_count += 1
        self.queue.put(request)
        return request
        
    def run(self):
        print(f"-- running TTS service ({self.language_code}, {self.voice})")
        
        while True:
            request = self.queue.get()
            
            print(f"-- TTS:  {request['text']}")
            
            responses = self.tts_service.synthesize_online(
                request['text'], request['voice'], self.language_code, sample_rate_hz=self.sample_rate_hz
            )

            for response in responses:
                self.output_stream(response.audio)
                
                if request['callback'] is not None:
                    request['callback'](response, request)
            