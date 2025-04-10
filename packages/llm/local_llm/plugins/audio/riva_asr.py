#!/usr/bin/env python3
import time
import queue
import threading
import logging

import riva.client
import riva.client.audio_io

from .auto_asr import AutoASR
from local_llm.utils import audio_silent


class RivaASR(AutoASR):
    """
    Streaming ASR service using NVIDIA Riva
    https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html
    
    You need to have the Riva server running first:
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64
    
    Inputs:  incoming audio samples coming from another audio plugin
             RivaASR can also open an audio device connected to this machine
             
    Output:  two channels, the first for word-by-word 'partial' transcript strings
             the second is for the full/final sentences
    """
    def __init__(self, riva_server='localhost:50051',
                 language_code='en-US', sample_rate_hz=48000, 
                 asr_confidence=-2.5, asr_silence=-1.0, asr_chunk=1600,
                 automatic_punctuation=True, inverse_text_normalization=False, 
                 profanity_filter=False, boosted_lm_words=None, boosted_lm_score=4.0, 
                 audio_input_device=None, audio_input_channels=1, **kwargs):
        """
        Parameters:
        
          riva_server (str) -- URL of the Riva GRPC server that should be running
          audio_input (int) -- audio input device number for locally-connected microphone
          sample_rate_hz (int) -- sample rate of any incoming audio or device (typically 16000, 44100, 48000)
          audio_chunk (int) -- the audio input buffer length (in samples) to use for input devices
          audio_input_channels (int) -- 1 for mono, 2 for stereo
          inverse_text_normalization (bool) -- https://developer.nvidia.com/blog/text-normalization-and-inverse-text-normalization-with-nvidia-nemo/
        """
        super().__init__(output_channels=2, **kwargs)
        
        self.server = riva_server
        self.auth = riva.client.Auth(uri=riva_server)

        self.audio_queue = AudioQueue(self)
        self.audio_chunk = asr_chunk
        self.input_device = audio_input_device
        self.language_code = language_code
        self.sample_rate = sample_rate_hz
        self.confidence_threshold = asr_confidence
        self.silence_threshold = asr_silence
        self.keep_alive_timeout = 5  # requests timeout after 1000 seconds
        
        self.asr_service = riva.client.ASRService(self.auth)
        
        self.asr_config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=language_code,
                max_alternatives=1,
                profanity_filter=profanity_filter,
                enable_automatic_punctuation=automatic_punctuation,
                verbatim_transcripts=not inverse_text_normalization,
                sample_rate_hertz=sample_rate_hz,
                audio_channel_count=audio_input_channels,
            ),
            interim_results=True,
        )
        
        riva.client.add_word_boosting_to_config(self.asr_config, boosted_lm_words, boosted_lm_score)

    def run(self):
        if self.input_device is not None:
            self.mic_thread = threading.Thread(target=self.run_mic, daemon=True)
            self.mic_thread.start()
    
        self.generate(self.audio_queue)
        
    def run_mic(self):
        logging.info(f"opening audio input device ({self.input_device})")
        self.generate(riva.client.audio_io.MicrophoneStream(
            self.sample_rate,
            self.audio_chunk,
            device=self.input_device,
        ))
        
    def generate(self, audio_generator):
        with audio_generator:
            responses = self.asr_service.streaming_response_generator(
                audio_chunks=audio_generator, streaming_config=self.asr_config
            )
        
            for response in responses:
                if not response.results:
                    continue

                for result in response.results:
                    transcript = result.alternatives[0].transcript.strip()
                    if result.is_final:
                        score = result.alternatives[0].confidence
                        if score >= self.confidence_threshold:
                            logging.debug(f"submitting ASR transcript (confidence={score:.3f}) -> '{transcript}'")
                            self.output(self.add_punctuation(transcript), AutoASR.OutputFinal)
                        else:
                            logging.warning(f"dropping ASR transcript (confidence={score:.3f} < {self.confidence_threshold:.3f}) -> '{transcript}'")
                    else:
                        self.output(transcript, AutoASR.OutputPartial)
        

class AudioQueue:
    """
    Implement same context manager/iterator interfaces as Riva's MicrophoneStream
    for ingesting ASR audio samples from external sources via the plugin's input queue.
    """
    def __init__(self, asr):
        self.asr = asr

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        pass
        
    def __next__(self) -> bytes:
        data = []
        size = 0
        chunk_size = self.asr.audio_chunk * 2  # 2 bytes per int16 sample
        time_begin = time.perf_counter()
        
        while size <= chunk_size:  
            try:
                input, _ = self.asr.input_queue.get(timeout=self.asr.keep_alive_timeout) 
            except queue.Empty:
                logging.debug(f"sending ASR keep-alive silence (idle for {self.asr.keep_alive_timeout} seconds)")
                return bytes(chunk_size)

            if audio_silent(input, self.asr.silence_threshold):
                if time.perf_counter() - time_begin >= self.asr.keep_alive_timeout:
                    return bytes(chunk_size)
                else: # drop the previous audio, so it doesn't get included later
                    data = []
                    size = 0
                    continue
                    
            data.append(input)
            size += len(data[-1])
        
        """
        while True:
            try:
                data.append(self.queue.get(block=False))
            except queue.Empty:
                break
        """

        return b''.join(data)
    
    def __iter__(self):
        return self

 
if __name__ == "__main__":
    from local_llm.utils import ArgParser
    from local_llm.plugins import PrintStream
    
    from termcolor import cprint
    
    args = ArgParser(extras=['asr', 'log']).parse_args()
    
    def print_prompt():
        cprint('>> PROMPT: ', 'blue', end='', flush=True)
            
    #def on_audio(samples, **kwargs):
    #    logging.info(f"recieved TTS audio samples {type(samples)}  shape={samples.shape}  dtype={samples.dtype}")
    #    print_prompt()
        
    asr = RivaASR(**vars(args))
    
    asr.add(PrintStream(partial=False, prefix='## ', color='green'), AutoASR.OutputFinal)
    asr.add(PrintStream(partial=False, prefix='>> ', color='blue'), AutoASR.OutputPartial)
    
    asr.start()
    asr.join()
    