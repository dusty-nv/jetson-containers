#!/usr/bin/env python3
import time
import logging

from local_llm import Plugin, StopTokens


def TTS(tts_model=None, **kwargs):
    """
    Factory function for automatically creating different types of TTS plugins.
    The model should either be 'riva' or 'xtts' (or name/path of XTTS model)
    The kwargs are forwarded to the TTS plugin implementing the model.
    """
    from local_llm.plugins import RivaTTS, XTTS
    
    if not tts_model or tts_model.lower() == 'none':
        return None
        
    if XTTS.is_xtts_model(tts_model):
        return XTTS(model=tts_model, **kwargs)
    elif tts_model.lower() == 'riva':
        return RivaTTS(**kwargs)
    else:
        raise ValueError(f"TTS model type should be either Riva or XTTS ({tts_model})")
       
       
class TTSPlugin(Plugin):
    """
    Base class for streaming TTS plugins, providing interruption/muting,
    text buffering for gapless audio, support for SSML speaker tags,
    and audio conversion. TODO: standard interfaces for voice/language/rates
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.request_count = 0
        self.needs_text_by = 0.0
        self.text_buffer = ''

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

    def buffer_text(self, text):
        """
        Wait for punctuation to occur before generating the TTS, and accumulate
        as much text as possible until audio is needed, because it sounds better.
        """
        self.text_buffer += text
            
        # always submit on EOS
        if any([stop_token in self.text_buffer for stop_token in StopTokens]):
            text = self.text_buffer
            self.text_buffer = ''
            return text
              
        # look for punctuation
        punc_pos = -1

        for punc in ('. ', ', ', '! ', '? ', ': ', '\n'):  # the space after has fewer non-sentence uses
            punc_pos = max(self.text_buffer.rfind(punc), punc_pos)
                
        if punc_pos < 0:
            #if len(self.text_buffer.split(' ')) > 6:
            #    punc_pos = len(self.text_buffer) - 1
            #else:
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
        
        if len(text.strip()) == 0:
            return None
            
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

