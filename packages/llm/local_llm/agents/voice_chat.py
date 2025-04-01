#!/usr/bin/env python3
from local_llm import Agent, Pipeline
from local_llm.utils import ArgParser, print_table

from local_llm.plugins import (
    UserPrompt, ChatQuery, PrintStream, 
    AutoASR, AutoTTS, RateLimit, ProcessProxy, 
    AudioOutputDevice, AudioOutputFile
)


class VoiceChat(Agent):
    """
    Uses ASR + TTS to chat with LLM
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # LLM
        self.llm = ProcessProxy('ChatQuery', **kwargs)  #ChatQuery(**kwargs) # # 
        self.llm.add(PrintStream(color='green'))
        
        # ASR
        self.asr = AutoASR.from_pretrained(**kwargs)
    
        if self.asr:
            self.asr.add(PrintStream(partial=False, prefix='## ', color='blue'), AutoASR.OutputFinal)
            self.asr.add(PrintStream(partial=False, prefix='>> ', color='magenta'), AutoASR.OutputPartial)
            
            self.asr.add(self.asr_partial, AutoASR.OutputPartial) # pause output when user is speaking
            self.asr.add(self.asr_final, AutoASR.OutputFinal)     # clear queues on final ASR transcript
            self.asr.add(self.llm, AutoASR.OutputFinal)  # runs after asr_final() and any interruptions occur
            
            self.asr_history = None  # store the partial ASR transcript

        # TTS
        self.tts = AutoTTS.from_pretrained(**kwargs)
        
        if self.tts:
            self.tts_output = RateLimit(kwargs['sample_rate_hz'], chunk=9600) # slow down TTS to realtime and be able to pause it
            self.tts.add(self.tts_output)
            self.llm.add(self.tts, ChatQuery.OutputWords)

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
        
        # setup pipeline with two entry nodes
        self.pipeline = [self.prompt]

        if self.asr:
            self.pipeline.append(self.asr)
            
    def asr_partial(self, text):
        self.asr_history = text
        if len(text.split(' ')) < 2:
            return
        if self.tts:
            self.tts_output.pause(1.0)

    def asr_final(self, text):
        self.asr_history = None
        self.on_interrupt()
        
    def on_interrupt(self):
        self.llm.interrupt(recursive=False)
        if self.tts:
            self.tts.interrupt(recursive=False)
            self.tts_output.interrupt(block=False, recursive=False) # might be paused/asleep
 
if __name__ == "__main__":
    parser = ArgParser(extras=ArgParser.Defaults+['asr', 'tts', 'audio_output'])
    args = parser.parse_args()
    
    agent = VoiceChat(**vars(args)).run() 
    