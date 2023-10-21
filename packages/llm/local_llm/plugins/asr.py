#!/usr/bin/env python3
import logging

import riva.client
import riva.client.audio_io

from local_llm import Plugin


class RivaASR(Plugin):
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
    OutputFinal=0    # output full transcripts (channel 0)
    OutputPartial=1  # output partial transcripts (channel 1)
    
    def __init__(self, riva_server='localhost:50051',
                 audio_input=None, sample_rate_hz=48000,
                 audio_chunk=1600, audio_input_channels=1,
                 automatic_punctuation=True, verbatim_transcripts=True, 
                 profanity_filter=False, language_code='en-US', 
                 boosted_lm_words=None, boosted_lm_score=4.0, **kwargs):
        """
        Parameters:
        
          riva_server (str) -- URL of the Riva GRPC server that should be running
          audio_input (str) -- audio input device number for locally-connected microphone
          sample_rate_hz (int) -- sample rate of any incoming audio or device (typically 16000, 44100, 48000)
          audio_chunk (int) -- the audio input buffer length (in samples) to use for input devices
          audio_input_channels (int) -- 1 for mono, 2 for stereo
        """
        super().__init__(plugin_outputs=2, **kwargs)
        
        self.server = riva_server
        self.auth = riva.client.Auth(uri=riva_server)
        
        self.audio_queue = AudioQueue(self.input_queue, audio_chunk)
        self.audio_chunk = audio_chunk
        self.input_device = audio_input
        self.language_code = language_code
        self.sample_rate = sample_rate_hz

        self.asr_service = riva.client.ASRService(auth)
        
        self.asr_config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=language_code,
                max_alternatives=1,
                profanity_filter=profanity_filter,
                enable_automatic_punctuation=automatic_punctuation,
                verbatim_transcripts=verbatim_transcripts,
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
                    self.output(result)  # TODO partial -> channel 0, final -> channel 1
        
        
class AudioQueue:
    """
    Implement same context manager/iterator interfaces as Riva's MicrophoneStream
    """
    def __init__(self, queue, audio_chunk=1600):
        self.queue = queue
        self.audio_chunk = audio_chunk

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        pass
        
    def __next__(self) -> bytes:
        data = []
        size = 0
        
        while size <= self.audio_chunk * 2:
            data.append(self.queue.get())
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
            