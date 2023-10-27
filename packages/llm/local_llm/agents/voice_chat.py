#!/usr/bin/env python3
from local_llm import Agent, Pipeline

from local_llm.plugins import (
    UserPrompt, ChatQuery, PrintStream, 
    RivaASR, RivaTTS, RateLimit,
    AudioOutputDevice, AudioOutputFile
)

from local_llm.utils import ArgParser, print_table

from termcolor import cprint


class VoiceChat(Agent):
    """
    Uses ASR + TTS to chat with LLM
    """
    def __init__(self, **kwargs):
        super().__init__()

        # ASR
        self.asr = RivaASR(**kwargs)
    
        self.asr.add(PrintStream(partial=False, prefix='## ', color='blue'), RivaASR.OutputFinal)
        self.asr.add(PrintStream(partial=False, prefix='>> ', color='magenta'), RivaASR.OutputPartial)
        
        self.asr.add(self.asr_partial, RivaASR.OutputPartial) # pause output when user is speaking
        self.asr.add(self.asr_final, RivaASR.OutputFinal)     # clear queues on final ASR transcript
        
        self.asr_history = None
        
        # LLM
        self.llm = ChatQuery(**kwargs)
    
        self.llm.add(PrintStream(color='green', relay=True).add(self.on_eos))
        self.asr.add(self.llm, RivaASR.OutputFinal)  # submit ASR sentences to the LLM
        
        # TTS
        self.tts = RivaTTS(**kwargs)
        self.tts_output = RateLimit(kwargs['sample_rate_hz'], chunk=9600) # slow down TTS to realtime and be able to pause it
        
        self.tts.add(self.tts_output)
        self.llm.add(self.tts, ChatQuery.OutputWords)
        
        # Audio Output
        self.audio_output_device = kwargs.get('audio_output_device')
        self.audio_output_file = kwargs.get('audio_output_file')
        
        if self.audio_output_device is not None:
            self.audio_output_device = AudioOutputDevice(**kwargs)
            self.tts_output.add(self.audio_output_device)
        
        if self.audio_output_file is not None:
            self.audio_output_file = AudioOutputFile(**kwargs)
            self.tts_output.add(self.audio_output_file)
        
        # text prompts from web UI or CLI
        self.prompt = UserPrompt(interactive=True, **kwargs)
        self.prompt.add(self.llm)
        
        self.pipeline = [self.prompt, self.asr]

    def asr_partial(self, text):
        self.asr_history = text
        self.tts_output.pause(1.0)

    def asr_final(self, text):
        self.asr_history = None
        
        self.llm.interrupt()
        self.tts.interrupt()
        
        self.tts_output.interrupt()
        self.tts_output.unpause()
            
    def on_eos(self, text):
        if text.endswith('</s>'):
            print_table(self.llm.model.stats)
            #self.print_input_prompt()

    """
    def print_input_prompt(self):
        if self.interactive:
            cprint('>> PROMPT: ', 'blue', end='', flush=True)
    """ 
        
if __name__ == "__main__":
    from local_llm.utils import ArgParser

    parser = ArgParser(extras=ArgParser.Defaults+['asr', 'tts', 'audio_output'])
    args = parser.parse_args()
    
    agent = VoiceChat(**vars(args)).run() 
    