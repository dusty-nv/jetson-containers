#!/usr/bin/env python3
import time
import queue
import logging
import numpy as np

import riva.client
import riva.client.audio_io

from local_llm import Plugin


class RivaTTS(Plugin):
    """
    Streaming TTS service using NVIDIA Riva
    https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html
    
    You need to have the Riva server running first:
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64
    
    Inputs:  words to speak (str)
    Output:  audio samples (np.ndarray, int16)
    """
    def __init__(self, riva_server='localhost:50051', 
                 voice='English-US.Female-1', language_code='en-US', 
                 sample_rate_hz=48000, **kwargs):
        """
        The available voices are from:
        https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html#voices
        """
        super().__init__(**kwargs)
        
        self.server = riva_server
        self.auth = riva.client.Auth(uri=riva_server)
        self.tts_service = riva.client.SpeechSynthesisService(self.auth)
        
        self.voice = voice   # can be changed mid-stream
        self.muted = False   # will supress TTS outputs
        
        self.language_code = language_code
        self.sample_rate = sample_rate_hz
        self.request_count = 0
        self.needs_text_by = 0.0
    
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
        """
        TTS sounds better on several words or a sentence at a time,
        so buffer incoming text until it is needed (based on the
        playback time of the previous outgoing audio samples)
        """
        while True:
            text = ''

            while True:  # accumulate text until its needed to prevent audio gap-out
                time_to_go = self.needs_text_by - time.perf_counter()

                if time_to_go < 0.01:
                    break
                    
                try:
                    text += self.input_queue.get(timeout=time_to_go-0.009)
                except queue.Empty:
                    break
            
            if len(text) == 0:  # there can be extended times of no speaking
                self.input_event.wait()
                self.input_event.clear()
                text += self.input_queue.get()

            logging.debug(f"running TTS request '{text}'")
            
            self.muted = False
            
            responses = self.tts_service.synthesize_online(
                text, self.voice, self.language_code, sample_rate_hz=self.sample_rate
            )
            
            num_samples = 0

            for response in responses:
                if self.muted:
                    logging.debug(f"-- TTS muted, exiting request early:  {text}")
                    break
                    
                samples = np.frombuffer(response.audio, dtype=np.int16)

                current_time = time.perf_counter()
                if current_time > self.needs_text_by:
                    self.needs_text_by = current_time
                self.needs_text_by += len(samples) / self.sample_rate
                
                self.output(samples)
                
                
if __name__ == "__main__":
    from local_llm.utils import ArgParser
    from local_llm.plugins import UserPrompt, AudioOutputDevice, AudioOutputFile
    
    from termcolor import cprint
    
    args = ArgParser(extras=['tts', 'audio_output', 'prompt', 'log']).parse_args()
    
    def print_prompt():
        cprint('>> PROMPT: ', 'blue', end='', flush=True)
            
    #def on_audio(samples, **kwargs):
    #    logging.info(f"recieved TTS audio samples {type(samples)}  shape={samples.shape}  dtype={samples.dtype}")
    #    print_prompt()
        
    tts = RivaTTS(**vars(args))
    
    if args.audio_output_device is not None:
        tts.add(AudioOutputDevice(**vars(args)))

    if args.audio_output_file is not None:
        tts.add(AudioOutputFile(**vars(args)))
 
    prompt = UserPrompt(interactive=True, **vars(args)).add(tts)

    print_prompt()
    prompt.start().join()
