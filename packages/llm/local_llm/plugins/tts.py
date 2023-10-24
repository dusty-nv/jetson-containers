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
                 voice='English-US.Female-1', language_code='en-US', sample_rate_hz=48000, 
                 voice_rate='default', voice_pitch='default', voice_volume='default',
                 #voice_min_words=4, 
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
        
        self.voice = voice   # can be changed mid-stream
        self.muted = False   # will supress TTS outputs
        
        self.rate = voice_rate
        self.pitch = voice_pitch
        self.volume = voice_volume
        #self.min_words = voice_min_words
        
        self.language_code = language_code
        self.sample_rate = sample_rate_hz
        self.request_count = 0
        self.needs_text_by = 0.0
        self.text_buffer = ''
    
    def process(self, text, **kwargs):
        """
        Inputs text, outputs stream of audio samples (np.ndarray, np.int16)
        
        The input text is buffered by punctuation/phrases as it sounds better,
        and filtered for emojis/ect, and has SSML tags applied (if enabled) 
        """
        text = self.buffer_text(text)
        text = self.filter_text(text)
        text = self.apply_ssml(text)
        
        if not text:
            return
            
        logging.debug(f"generating TTS for '{text}'")
        
        self.muted = False
        
        responses = self.tts_service.synthesize_online(
            text, self.voice, self.language_code, sample_rate_hz=self.sample_rate
        )

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
            
        #logging.debug(f"done with TTS request '{text}'")
        
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
     
    def buffer_text(self, text):
        """
        Wait for punctuation to occur because that sounds better
        """
        self.text_buffer += text
            
        # always submit on EOS
        if '</s>' in self.text_buffer:
            text = self.text_buffer
            self.text_buffer = ''
            return text
              
        # look for punctuation
        punc_pos = -1

        for punc in ('. ', ', ', '! ', '? ', ': ', '\n'):  # the space after has fewer non-sentence uses
            punc_pos = max(self.text_buffer.rfind(punc), punc_pos)
                
        if punc_pos < 0:
            return None
            
        # see if input is needed to prevent a gap-out
        timeout = self.needs_text_by - time.perf_counter() - 0.05  # TODO make this RTFX factor adjustable
        
        if timeout > 0:
            return None   # we can keep accumulating text
            
        # return the latest phrase/sentence
        text = self.text_buffer[:punc_pos+1]
        
        if len(self.text_buffer) > punc_pos + 1:  # save characters after for next request
            self.text_buffer = self.text_buffer[punc_pos+1:]
        else:
            self.text_buffer = ''
            
        return text

        """
        # accumulate text until its needed to prevent audio gap-out
        while True:
            timeout = self.needs_text_by - time.perf_counter() - 0.1  # TODO make this RTFX factor adjustable

            try:
                text += self.input_queue.get(timeout=timeout if timeout > 0 else None)
                while not self.input_queue.empty():  # pull any additional input without waiting
                    text += self.input_queue.get(block=False)
            except queue.Empty:
                pass
                
            # make sure there are at least N words (or EOS)
            if '</s>' in text:
                break
            elif timeout <= 0 and len(text.strip().split(' ')) >= 4:
                break
        """
        
    def filter_text(self, text):
        """
        Santize inputs (TODO remove emojis, *giggles*, ect)
        """
        if not text:
            return None
            
        # text = text.strip()
        text = text.replace('</s>', '')
        text = text.replace('\n', ' ')
        #text = text.replace('  ', ' ')
        
        return text
    
    def apply_ssml(self, text):
        """
        Apply SSML tags to text (if enabled)
        """
        if not text:
            return None
            
        if self.rate != 'default' or self.pitch != 'default' or self.volume != 'default':
            text = f"<speak><prosody rate='{self.rate}' pitch='{self.pitch}' volume='{self.volume}'>{text}</prosody></speak>"  
            
        return text
    
    '''
    def run(self):
        """
        TTS sounds better on several words or a sentence at a time,
        so buffer incoming text until it is needed (based on the
        playback time of the previous outgoing audio samples)
        """
        while True:
            text = self.buffer_text()
            text = self.filter_text(text)
                
            if not text:
                continue
                
            text = self.apply_ssml(text)
            self.process(text)
    '''
    
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
