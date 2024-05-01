#!/usr/bin/env python3
import os
import re
import json
import pprint
import logging
import inflect

import torch
import torchaudio
import numpy as np

from .auto_tts import AutoTTS
from local_llm.utils import ONNXRuntimeModel, download_model, convert_audio


class FastPitchTTS(AutoTTS):
    """
    Streaming TTS service using FastPitch-HiFiGAN model with ONNXRuntime.
    This is a single-speaker model with a female voice in English.
    
    Inputs:  words to speak (str)
    Output:  audio samples (np.ndarray, int16)
    """
    def __init__(self, model='/data/models/tts/fastpitch_hifigan', sample_rate_hz=22050, use_tensorrt=True, **kwargs):
        """
        Load XTTS model and set default options (many of which can be changed at runtime)
        """
        super().__init__(**kwargs)
        
        self.model_dir = model
        self.model_name = os.path.basename(self.model_dir)
        self.config_path = os.path.join(self.model_dir, 'fastpitch_hifigan.json')
        
        logging.info(f"loading {self.model_name} TTS model config from {self.config_path}")
        
        with open(self.config_path) as config_file:
            self.config = json.load(config_file)
            
        logging.info(f"{self.model_name} model config:\n{pprint.pformat(self.config, indent=2)}")
        
        onnx_provider = ['CUDAExecutionProvider'] #'TensorRTExecutionProvider'
        
        self.generator = ONNXRuntimeModel(os.path.join(self.model_dir, self.config['generator']['model_path']), provider=onnx_provider, **kwargs)
        self.vocoder = ONNXRuntimeModel(os.path.join(self.model_dir, self.config['vocoder']['model_path']), provider=onnx_provider, **kwargs)
        
        self.voices = ['female']
        self.languages = ['en']
        self.symbol_to_id = {s: i for i, s in enumerate(self.get_symbols())}
        
        self.rate = 1.0
        self.sample_rate = sample_rate_hz
        self.model_sample_rate = self.config['vocoder']['sample_rate']
        
        if self.sample_rate != self.model_sample_rate:
            self.resampler = torchaudio.transforms.Resample(self.model_sample_rate, self.sample_rate).cuda()
        else:
            self.resampler = None
 
        logging.debug(f"running TTS model warm-up for {self.model_name}")
        self.process("This is a test of the Fast Pitch text to speech.")
    
    @property
    def voice(self):
        return self.voices[0]
        
    @voice.setter
    def voice(self, voice):
        logging.warning(f"{self.model_name} is not a multi-speaker TTS model, ignoring voice setting")

    @property
    def language(self):
        return self._language
        
    @language.setter
    def language(self, language):
        logging.warning(f"{self.model_name} is not a multi-language TTS model, ignoring language setting")
       
    @staticmethod
    def is_model(model):
        return 'fastpitch' in model.lower()
        
    def get_symbols(self):
        """
        Return a list of all the accepted character symbols / embeddings
        """
        _arpabet = [
          'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
          'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
          'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
          'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
          'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
          'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
          'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
        ]
        _arpabet = ['@' + s for s in _arpabet]
        _pad = '_'
        _punctuation = '!\'(),.:;? '
        _special = '-'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
        return symbols
        
    def process(self, text, **kwargs):
        """
        Inputs text, outputs stream of audio samples (np.ndarray, np.int16)
        
        The input text is buffered by punctuation/phrases as it sounds better,
        and filtered for emojis/ect, and has SSML tags applied (if enabled) 
        """
        #text = self.buffer_text(text)
        text = self.filter_text(text, numbers_to_words=True)

        if not text or self.interrupted:
            return
            
        logging.debug(f"generating TTS with {self.model_name} for '{text}'")
        
        pad_symbol = ' '
        min_length = 6
        
        if text[-1].isalnum():      # end with punctuation, otherwise audio is cut-off
            text += pad_symbol
          
        if len(text) < min_length:  # WAR for cuDNN error on JetPack <= 4.5.x
            text = text.ljust(min_length, pad_symbol)
            
        # convert chars to symbol embeddings
        encoded_text = [self.symbol_to_id[s] for s in text.lower() if s in self.symbol_to_id]
        encoded_text = np.expand_dims(np.array(encoded_text, dtype=np.int64), axis=0)
        
        # generate MEL spectrogram + audio
        mels = self.generator.execute(encoded_text)[0]
        audio = self.vocoder.execute(mels).squeeze()
        
        if self.resampler:
            audio = torch.from_numpy(audio).cuda()
            audio = self.resampler(audio)
            audio = audio.detach().cpu().numpy()
            
        audio = convert_audio(audio, np.int16)
        
        return audio
        