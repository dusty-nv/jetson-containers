#!/usr/bin/env python3
import time
import queue
import logging
import numpy as np

import riva.client
import riva.client.audio_io

from .auto_tts import AutoTTS


class RivaTTS(AutoTTS):
    """
    Streaming TTS service using NVIDIA Riva
    https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html
    
    You need to have the Riva server running first:
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64
    
    Inputs:  words to speak (str)
    Output:  audio samples (np.ndarray, int16)
    """
    def __init__(self, riva_server='localhost:50051', 
                 voice='English-US.Female-1', language_code='en-US', sample_rate_hz=48000, 
                 voice_rate='default', voice_pitch='default', voice_volume='default',
                 **kwargs):
        """
        The available voices are from:
          https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html#voices
        
        rate, pitch, and volume are dynamic SSML tags from:
          https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/tts-basics-customize-ssml.html#customizing-rate-pitch-and-volume-with-the-prosody-tag
        """
        super().__init__(**kwargs)
        
        self.server = riva_server
        self.auth = riva.client.Auth(uri=riva_server)
        self.tts_service = riva.client.SpeechSynthesisService(self.auth)
        
        self.voice = voice   # these voice settings be changed at runtime
        self.rate = voice_rate
        self.pitch = voice_pitch
        self.volume = voice_volume

        self.language = language_code
        self.sample_rate = sample_rate_hz
        
        # find out how to query these for non-English models
        self.voices = [
            "English-US.Female-1",
            "English-US.Male-1"
        ]
        
        self.languages = ["en-US"]
        
        self.process("This is a test of Riva text to speech.", flush=True)
    
    def process(self, text, **kwargs):
        """
        Inputs text, outputs stream of audio samples (np.ndarray, np.int16)
        
        The input text is buffered by punctuation/phrases as it sounds better,
        and filtered for emojis/ect, and has SSML tags applied (if enabled) 
        """
        if len(self.outputs[0]) == 0:
            #logging.debug(f"TTS has no output connections, skipping generation")
            return
            
        text = self.buffer_text(text)
        text = self.filter_text(text)
        text = self.apply_ssml(text)
        
        if not text or self.interrupted:
            return
            
        logging.debug(f"generating TTS for '{text}'")

        responses = self.tts_service.synthesize_online(
            text, self.voice, self.language, sample_rate_hz=self.sample_rate
        )

        for response in responses:
            if self.interrupted:
                logging.debug(f"TTS interrupted, terminating request early:  {text}")
                break
                
            samples = np.frombuffer(response.audio, dtype=np.int16)
            #logging.debug(f"TTS outputting {len(samples)} audio samples")
            self.output(samples)
            
        #logging.debug(f"done with TTS request '{text}'")
