#!/usr/bin/env python3
import time
import queue
import threading

import riva.client
import riva.client.audio_io


class ASR(threading.Thread):
    """
    Streaming ASR service, either from microphone or web audio (or other samples from process_audio())
    """
    def __init__(self, auth, input_device=None, sample_rate_hz=44100, audio_chunk=1600, audio_channels=1, 
                 automatic_punctuation=True, verbatim_transcripts=True, profanity_filter=False, 
                 language_code='en-US', boosted_lm_words=None, boosted_lm_score=4.0, callback=None, **kwargs):
                 
        super(ASR, self).__init__()
        
        self.queue = AudioQueue()
        self.callback = callback
        self.audio_chunk = audio_chunk
        self.input_device = input_device
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
                audio_channel_count=audio_channels,
            ),
            interim_results=True,
        )
        
        riva.client.add_word_boosting_to_config(self.asr_config, boosted_lm_words, boosted_lm_score)

    def process(self, samples):
        self.queue.put(samples)

    def generate(self, audio_generator):
        with audio_generator:
            responses = self.asr_service.streaming_response_generator(
                audio_chunks=audio_generator, streaming_config=self.asr_config
            )
        
            for response in responses:
                if not response.results:
                    continue

                for result in response.results:
                    if self.callback is not None:
                        self.callback(result)
      
    def run_mic(self):
        print(f"-- opening audio input device ({self.input_device})")
        self.generate(riva.client.audio_io.MicrophoneStream(
            self.sample_rate,
            self.audio_chunk,
            device=self.input_device,
        ))
        
    def run(self):
        print(f"-- running ASR service ({self.language_code})")
        
        if self.input_device is not None:
            self.mic_thread = threading.Thread(target=self.run_mic, daemon=True)
            self.mic_thread.start()

        self.generate(self.queue)
               
                        
class AudioQueue:
    """
    Implement same context manager/iterator interfaces as MicrophoneStream (for ASR.process_audio())
    """
    def __init__(self, audio_chunk=1600):
        self.queue = queue.Queue()
        self.audio_chunk = audio_chunk

    def put(self, samples):
        self.queue.put(samples)
        
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
        