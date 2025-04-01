#!/usr/bin/env python3
import re
import time
import inflect
import logging

from local_llm import Plugin, StopTokens

    
class AutoTTS(Plugin):
    """
    Base class for TTS model plugins, supporting streaming, interruption/muting,
    text buffering for gapless audio, injection of SSML speaker tags for pitch/rate,
    text filtering (removing of emojis and number-to-text conversion), ect.
    
    It's designed for streaming out chunks of audio as they are generated,
    while recieving a stream of incoming text (usually word-by-word from LLM)
    """
    def __init__(self, tts_buffering='punctuation', **kwargs):
        super().__init__(**kwargs)
        
        self.request_count = 0
        self.needs_text_by = 0.0
        self.text_buffer = ''
        self.buffering = tts_buffering
        
        self.number_regex = None
        self.number_inflect = None
    
    @staticmethod
    def from_pretrained(tts=None, **kwargs):
        """
        Factory function for automatically creating different types of TTS models.
        The `tts` param should either be 'riva' or 'xtts' (or name/path of XTTS model)
        The kwargs are forwarded to the TTS plugin implementing the model.
        """
        from local_llm.plugins import RivaTTS, FastPitchTTS, XTTS
        
        if not tts or tts.lower() == 'none' or tts.lower().startswith('disable'):
            return None
            
        if FastPitchTTS.is_model(tts):
            return FastPitchTTS(**{**kwargs, 'model': tts})
        elif XTTS.is_model(tts):
            return XTTS(**{**kwargs, 'model': tts})
        elif tts.lower() == 'riva':
            return RivaTTS(**kwargs)
        else:
            raise ValueError(f"TTS model type should be either Riva or XTTS ({tts})")
        
    @property
    def buffering(self):
        return self._buffering
        
    @buffering.setter
    def buffering(self, mode):
        if isinstance(mode, str):
            if mode.lower() == 'none':
                self._buffering = []
            else:
                self._buffering = mode.split(',')
        else:
            if mode:
                self._buffering = mode
            else:
                self._buffering = []

    def buffer_text(self, text):
        """
        Wait for punctuation to occur before generating the TTS, and accumulate
        as much text as possible until audio is needed, because it sounds better.
        
        The buffering methods this function uses can be controlled by setting the
        tts.buffering property, either to 'none', 'punctuation', 'time', or some
        comma-separated combination like 'punctuation,time' (which applies both)
        
        Punctuation-based buffering waits for delimiters like .,!?: to occur in the
        stream of input text (which do not proceed another alphanumeric character).
        This is done because the TTS sounds worse if it doesn't have phrases to work on.
        
        Time-based buffering accumulates as much text as possible before it's predicted
        for the audio to gap-out, based on the prior RTFX performance and samples generated.
        """
        self.text_buffer += text
            
        # always submit on EOS
        if any([stop_token in self.text_buffer for stop_token in StopTokens]):
            text = self.text_buffer
            self.text_buffer = ''
            return text
              
        # see if input is needed to prevent a gap-out
        if 'time' in self.buffering:    
            timeout = self.needs_text_by - time.perf_counter() - 0.05  # TODO make this RTFX factor adjustable
            if timeout > 0:
                return None   # we can keep accumulating text
                
        # look for punctuation
        if 'punctuation' in self.buffering:
            punc_pos = -1

            #for punc in ('. ', ', ', '! ', '? ', ': ', '\n'):  # the space after has fewer non-sentence uses
            for punc in ('.', ',', '!', '?', ':', '\n'):  # the space after has fewer non-sentence uses
                punc_pos = max(self.text_buffer.rfind(punc), punc_pos)
                
            if punc_pos < 0:
                #if len(self.text_buffer.split(' ')) > 6:
                #    punc_pos = len(self.text_buffer) - 1
                #else:
                return None
               
            # for commas, make sure there are at least a handful of proceeding words
            if len(self.text_buffer[:punc_pos].split(' ')) < 4: #and self.text_buffer[punc_pos] == ',':
                return None
                
            # make sure that the character following the punctuation isn't alphanumeric
            # (it could be the punctuation is part of a contraction or acronym)
            if punc_pos < len(self.text_buffer) - 1:  
                if self.text_buffer[punc_pos+1].isalnum():
                    return None
                #elif self.text_buffer[punc_pos+1] == ' ':
                #    punc_pos = punc_pos + 1
 
            # return the latest phrase/sentence
            text = self.text_buffer[:punc_pos+1]
            
            if len(self.text_buffer) > punc_pos+1:  # save characters after for next request
                self.text_buffer = self.text_buffer[punc_pos+1:]
            else:
                self.text_buffer = ''
        else:
            text = self.text_buffer
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
        
    def filter_text(self, text, numbers_to_words=False):
        """
        Santize inputs (TODO remove emojis, *giggles*, ect)
        """
        if not text:
            return None
            
        # text = text.strip()
        for stop_token in StopTokens:
            text = text.replace(stop_token, '')
            
        #text = text.replace('</s>', '')
        text = text.replace('\n', ' ')
        text = text.replace('...', ' ')        
        text = self.filter_chars(text)

        if numbers_to_words:
            text = self.numbers_to_words(text)
            
        if len(text.strip()) == 0:
            return None
            
        return text
    
    def filter_chars(self, text):
        """
        Filter out non-alphanumeric and non-punctuation characters
        """
        def filter_char(input):
            for idx, char in enumerate(input):
                if char.isalnum() or any([char == x for x in ('.', ',', '?', '!', ':', ';', '-', "'", '"', ' ', '/', '+', '-', '*')]):
                    continue
                else:
                    return input.replace(char, ' ')
            return input
        
        while True:
            filtered = filter_char(text)
            if filtered == text:
                return text
            else:
                text = filtered
                continue
                
    def numbers_to_words(self, text):
        """
        Convert instances of numbers to words in the text.
        For example:  "The answer is 42" -> "The answer is forty two."
        """
        if self.number_regex is None:
            self.number_regex = re.compile(r'\d+(?:,\d+)?')  # https://stackoverflow.com/a/16321189
            self.number_inflect = inflect.engine()
            
        number_tokens = self.number_regex.findall(text)
        
        for number_token in number_tokens:
            # TODO test/handle floating-point numbers
            word_text = self.number_inflect.number_to_words(number_token)              
            num_begin = text.index(number_token)

            # insert the words back at the old location
            text = text[:num_begin] + word_text + text[num_begin + len(number_token):]
            
        return text
        
    def apply_ssml(self, text):
        """
        Apply SSML tags to text (if enabled)
        """
        if not text:
            return None
            
        use_ssml = False
        
        if isinstance(self.rate, str) and self.rate != 'default':
            use_ssml = True
        elif isinstance(self.rate, float) and self.rate != 1.0:
            use_ssml = True
        elif self.pitch != 'default' or self.volume != 'volume':
            use_ssml = True
            
        if use_ssml:
            text = f"<speak><prosody rate='{self.rate}' pitch='{self.pitch}' volume='{self.volume}'>{text}</prosody></speak>"  
            
        return text
        
    def interrupt(self, **kwargs):
        """
        Mute TTS output (typically when the user interrupts the bot)
        """
        super().interrupt(**kwargs)
        
        self.needs_text_by = 0
        self.text_buffer = ''
        
    def needs_text(self):
        """
        Returns true if the TTS needs text to keep the audio flowing.
        """
        return (time.perf_counter() > self.needs_text_by)

    def output(self, output, channel=0, **kwargs):
        """
        Override the default data output call for tracking of 
        """
        if output is None:
            return
        current_time = time.perf_counter()
        if current_time > self.needs_text_by:
            self.needs_text_by = current_time
        self.needs_text_by += len(output) / self.sample_rate
        super().output(output, channel=channel, **kwargs)

