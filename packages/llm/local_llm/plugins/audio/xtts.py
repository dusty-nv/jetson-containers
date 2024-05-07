#!/usr/bin/env python3
import os
import glob
import time
import queue
import pprint
import logging

import torch
import torchaudio
import numpy as np

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from .auto_tts import AutoTTS
from local_llm.utils import download_model, convert_audio


class XTTS(AutoTTS):
    """
    Streaming TTS service using XTTS model with HiFiGAN decoder in TensorRT.
    
    https://huggingface.co/coqui/XTTS-v2
    https://github.com/coqui-ai/TTS

    Inputs:  words to speak (str)
    Output:  audio samples (np.ndarray, int16)
    
    You can get the list of voices with tts.voices, and list of languages with tts.languages
    The speed can be set with tts.rate (1.0 = normal). The default voice is '...' with rate 1.0
    """
    def __init__(self, model='coqui/XTTS-v2', voice='Sofia Hellen', language_code='en', 
                 sample_rate_hz=24000, voice_rate=1.0, use_tensorrt=True, **kwargs):
        """
        Load XTTS model and set default options (many of which can be changed at runtime)
        """
        super().__init__(**kwargs)
        
        if os.path.isdir(model):
            self.model_path = model
            self.model_name = model
        else:
            t = model.lower()
            if XTTS.is_model(model):
                model = 'coqui/XTTS-v2'
            self.model_path = download_model(model)
            self.model_name = model
            
        self.config_path = os.path.join(self.model_path, 'config.json')
        logging.info(f"loading XTTS model config from {self.config_path}")
        
        self.config = XttsConfig()
        self.config.load_json(self.config_path)
        
        logging.debug(f"XTTS model config for {self.model_name}\n{pprint.pformat(self.config, indent=1)}")
        logging.info(f"loading XTTS model from {self.model_path}")
        
        self.model = Xtts.init_from_config(self.config)
        
        self.model.load_checkpoint(
            self.config, 
            checkpoint_dir=self.model_path, 
            speaker_file_path=os.path.join(self.model_path, 'speakers_xtts.pth'),
            use_tensorrt=use_tensorrt,
        )
        
        self.model.cuda()

        # get supported voices and languages
        self.speaker_manager = self.model.speaker_manager

        self.voices = list(self.speaker_manager.speakers.keys())
        self.languages = self.model.language_manager.language_names
        
        logging.info(f"XTTS voices for {self.model_name}:\n{self.voices}")
        logging.info(f"XTTS languages for {self.model_name}:\n{self.languages}")
        
        # clone custom voices from wav files
        self.cloned_voices = {}
        clones = glob.glob("/data/audio/tts/voices/*.wav")
        
        for clone in clones:
            self.cloned_voices[os.path.basename(clone)] = self.clone(clone)
            self.voices.append(os.path.basename(clone))
            
        try:
            self.voice = voice
        except Exception as error:
            logging.error(f"Error loading voice '{voice}' with {self.model_name} ({error})")
            self.voice = self.voices[0]
            
        self.language = language_code
        self.rate = voice_rate
        self.sample_rate = sample_rate_hz
        self.model_sample_rate = self.config.model_args.output_sample_rate
        
        if self.sample_rate != self.model_sample_rate:
            self.resampler = torchaudio.transforms.Resample(self.model_sample_rate, self.sample_rate).cuda()
        else:
            self.resampler = None
            
        logging.debug(f"running TTS model warm-up for {self.model_name}")
        self.process("This is a test of the text to speech.")
    
    @property
    def voice(self):
        return self._voice
        
    @voice.setter
    def voice(self, voice):
        if os.path.isfile(voice):
            self.gpt_cond_latent, self.speaker_embedding = self.clone(voice)
        else:
            if voice in self.speaker_manager.speakers:
                self.gpt_cond_latent, self.speaker_embedding = self.speaker_manager.speakers[voice].values()
            else:
                if voice in self.cloned_voices:
                    self.gpt_cond_latent, self.speaker_embedding = self.cloned_voices[voice]
                else:
                    raise ValueError(f"'{voice}' was not in the supported list of voices for {self.model_name}\n{self.voices}")
 
        self._voice = voice
    
    @property
    def language(self):
        return self._language
        
    @language.setter
    def language(self, language):
        language = language.lower().split('-')[0]  # drop the country code (e.g. 'en-US')
        if language not in self.languages:
            raise ValueError(f"'{language}' was not in the supported list of languages for {self.model_name}\n{self.languages}")
        self._language = language
       
    @staticmethod
    def is_model(model):
        if os.path.isdir(model):
            return 'xtts' in model.lower()
        model_names = ['xtts', 'xtts2', 'xtts-v2', 'xtts_v2', 'coqui/xtts-v2']
        return any([x == model.lower() for x in model_names])
      
    def clone(self, audio):
        """
        Clone the speaker's voice in the wav file or audio samples, return the speaker embeddings.
        """
        logging.info(f"{self.model_name} cloning voice from {audio}")
        return self.model.get_conditioning_latents(
            audio_path=audio,
            max_ref_length=3600,
            gpt_cond_len=3600,
            gpt_cond_chunk_len=6,
            sound_norm_refs=False,
        )
        
    def process(self, text, **kwargs):
        """
        Inputs text, outputs stream of audio samples (np.ndarray, np.int16)
        
        The input text is buffered by punctuation/phrases as it sounds better,
        and filtered for emojis/ect, and has SSML tags applied (if enabled) 
        """
        text = self.buffer_text(text)    
        text = self.filter_text(text)

        if not text or self.interrupted:
            logging.debug(f"TTS {self.model_name} waiting for more input text (buffering={self.buffering} interrupted={self.interrupted})")
            return
            
        logging.debug(f"generating TTS with {self.model_name} for '{text}'")

        time_begin = time.perf_counter()
        num_samples = 0
        
        stream = self.model.inference_stream(
            text,
            self.language,
            self.gpt_cond_latent,
            self.speaker_embedding,
            enable_text_splitting=True,
            #overlap_len=128,
            #stream_chunk_size=20,
            do_sample=True,
            speed=self.rate,
        )

        for samples in stream:
            if self.interrupted:
                logging.debug(f"TTS interrupted, terminating request early:  {text}")
                return
                
            if self.resampler:
                samples = self.resampler(samples)
                
            samples = convert_audio(samples, dtype=torch.int16)
            samples = samples.detach().cpu().numpy()
            
            num_samples += len(samples)
            self.output(samples)

        time_elapsed = time.perf_counter() - time_begin
        logging.debug(f"finished TTS request, streamed {num_samples} samples at {self.sample_rate/1000:.1f}KHz - {num_samples/self.sample_rate:.2f} sec of audio in {time_elapsed:.2f} sec (RTFX={num_samples/self.sample_rate/time_elapsed:.4f})")
