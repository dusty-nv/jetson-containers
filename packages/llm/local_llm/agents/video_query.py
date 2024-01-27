#!/usr/bin/env python3
import time
import threading

from local_llm import Agent

from local_llm.plugins import VideoSource, VideoOutput, ChatQuery, PrintStream, ProcessProxy
from local_llm.utils import ArgParser, print_table

from termcolor import cprint
from jetson_utils import cudaFont, cudaMemcpy, cudaToNumpy, cudaDeviceSynchronize


class VideoQuery(Agent):
    """
    Perpetual always-on closed-loop visual agent that applies prompts to a video stream.
    """
    def __init__(self, model="liuhaotian/llava-v1.5-13b", **kwargs):
        super().__init__()

        # load model in another process for smooth streaming
        self.llm = ProcessProxy((lambda **kwargs: ChatQuery(model, drop_inputs=True, **kwargs)), **kwargs)
        self.llm.add(PrintStream(color='green', relay=True).add(self.on_eos))
        self.llm.start()

        # test / warm-up query
        self.warmup = True
        self.last_text = ""
        self.eos = False
        
        self.llm("What is 2+2?")
        
        while self.warmup:
            time.sleep(0.25)
            
        # create video streams    
        self.video_source = VideoSource(**kwargs)
        self.video_output = VideoOutput(**kwargs)
        
        self.video_source.add(self.on_video, threaded=False)
        self.video_output.start()
        
        self.font = cudaFont()

        # setup prompts
        self.prompt = 0
        self.prompts = kwargs.get('prompt')
        
        if not self.prompts:
            self.prompts = [
                'Describe the image concisely.', 
                'How many fingers is the person holding up?',
                'What does the text in the image say?',
                'There is a question asked in the image.  What is the answer?',
            ]
        
        self.keyboard_thread = threading.Thread(target=self.poll_keyboard)
        self.keyboard_thread.start()
        
        # entry node
        self.pipeline = [self.video_source]
   
    def on_video(self, image):
        np_image = cudaToNumpy(image)
        cudaDeviceSynchronize()
        
        self.llm([
            'reset',
            np_image,
            self.prompts[self.prompt],
        ])
        
        text = self.last_text.replace('\n', '').replace('</s>', '').strip()

        if text:
            self.font.OverlayText(image, text=text, x=5, y=42, color=self.font.White, background=self.font.Gray40)

        self.font.OverlayText(image, text=self.prompts[self.prompt], x=5, y=5, color=(120,215,21), background=self.font.Gray40)
        self.video_output(image)
            
    def on_eos(self, text):
        if self.eos:
            self.last_text = text  # new query response
            self.eos = False
        elif not self.warmup:  # don't view warmup response
            self.last_text = self.last_text + text

        if text.endswith('</s>'):
            #print_table(self.llm.model.stats)
            self.warmup = False
            self.eos = True
     
    def poll_keyboard(self):
        while True:
            try:
                key = input().strip() #getch.getch()
                
                if key == 'd' or key == 'l':
                    self.prompt = (self.prompt + 1) % len(self.prompts)
                elif key == 'a' or key == 'j':
                    self.prompt = self.prompt - 1
                    if self.prompt < 0:
                        self.prompt = len(self.prompts) - 1
                
                num = int(key)
                
                if num > 0 and num <= len(self.prompts):
                    self.prompt = num - 1
            except Exception as err:
                continue
                
if __name__ == "__main__":
    parser = ArgParser(extras=ArgParser.Defaults+['video_input', 'video_output'])
    args = parser.parse_args()
    agent = VideoQuery(**vars(args)).run() 
    