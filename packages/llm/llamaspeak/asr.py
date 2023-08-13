#!/usr/bin/env python3
import pprint
import threading

import riva.client
import riva.client.audio_io


class ASR(threading.Thread):
    """
    Streaming ASR service
    """
    def __init__(self, auth, input_device=0, sample_rate_hz=44100, audio_chunk=1600, audio_channels=1, 
                 automatic_punctuation=True, verbatim_transcripts=True, profanity_filter=False, 
                 language_code='en-US', boosted_lm_words=None, boosted_lm_score=4.0, callback=None, **kwargs):
                 
        super(ASR, self).__init__()
        
        self.callback=callback
        self.language_code = language_code
        
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

        self.mic_stream = riva.client.audio_io.MicrophoneStream(
            sample_rate_hz,
            audio_chunk,
            device=input_device,
        ).__enter__()
        
        self.responses = self.asr_service.streaming_response_generator(
            audio_chunks=self.mic_stream, streaming_config=self.asr_config
        )

    def run(self):
        print(f"-- running ASR service ({self.language_code})")
        
        for response in self.responses:
            if not response.results:
                continue

            for result in response.results:
                if self.callback is not None:
                    self.callback(result)