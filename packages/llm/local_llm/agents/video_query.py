#!/usr/bin/env python3
import os
import time
import json
import torch
import pprint
import logging
import threading

from datetime import datetime
from termcolor import cprint

from local_llm import Agent, StopTokens

from local_llm.web import WebServer
from local_llm.plugins import VideoSource, VideoOutput, ChatQuery, PrintStream, ProcessProxy, EventFilter, NanoDB
from local_llm.utils import ArgParser, print_table

from jetson_utils import cudaFont, cudaMemcpy, cudaToNumpy, cudaDeviceSynchronize, saveImage


class VideoQuery(Agent):
    """
    Perpetual always-on closed-loop visual agent that applies prompts to a video stream.
    """
    def __init__(self, model="liuhaotian/llava-v1.5-13b", nanodb=None, **kwargs):
        super().__init__()

        # load model in another process for smooth streaming
        self.llm = ProcessProxy('ChatQuery', model=model, drop_inputs=True, **kwargs) #ProcessProxy((lambda **kwargs: ChatQuery(model, drop_inputs=True, **kwargs)), **kwargs)
        self.llm.add(PrintStream(color='green', relay=True).add(self.on_text))
        self.llm.start()

        # test / warm-up query
        self.warmup = True
        self.text = ""
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
        
        self.pause_video = False
        self.pause_image = None
        self.last_image = None
        self.tag_image = None

        self.pipeline = [self.video_source]
        
        # setup prompts
        self.prompt_history = kwargs.get('prompt')
        
        if not self.prompt_history:
            self.prompt_history = [
                'Describe the image concisely.', 
                'How many fingers is the person holding up?',
                'What does the text in the image say?',
                'There is a question asked in the image.  What is the answer?',
                'Is there a person in the image?  Answer yes or no.',
            ]
        
        self.prompt = self.prompt_history[0]
        
        self.last_prompt = None
        self.auto_refresh = True
        self.auto_refresh_db = True
        
        self.keyboard_prompt = 0
        self.keyboard_thread = threading.Thread(target=self.poll_keyboard)
        self.keyboard_thread.start()

        # nanoDB
        if nanodb:
            self.db = NanoDB(
                path=nanodb, 
                model=None, # disable DB's model because VLM's CLIP is used 
                reserve=kwargs.get('nanodb_reserve'), 
                k=8, drop_inputs=True,
            ).start().add(self.on_search)
            self.llm.add(self.on_image_embedding, channel=ChatQuery.OutputImageEmbedding)
        else:
            self.db = None
            
        # webserver
        mounts = {
            scan : f"/images/{n}" 
            for n, scan in enumerate(self.db.scans)
        } if self.db else {}
        
        mounts['/data/datasets/uploads'] = '/images/uploads'
        
        self.server = WebServer(
            msg_callback=self.on_websocket, 
            index='video_query.html', 
            send_webrtc=False, 
            title="LIVE LLAVA", 
            model=os.path.basename(model),
            mounts=mounts,
            nanodb=nanodb,
            **kwargs
        )
        
        # event filters
        self.events = EventFilter(server=self.server)
   
    def on_video(self, image):
        if self.pause_video:
            if not self.pause_image:
                self.pause_image = cudaMemcpy(image)
            image = cudaMemcpy(self.pause_image)
        
        if self.auto_refresh or self.prompt != self.last_prompt:
            np_image = cudaToNumpy(image)
            cudaDeviceSynchronize()
            self.llm(['/reset', np_image, self.prompt])
            self.last_prompt = self.prompt
            if self.db:
                self.last_image = cudaMemcpy(image)

        text = self.text.replace('\n', '').replace('</s>', '').strip()

        if text:
            self.font.OverlayText(image, text=text, x=5, y=42, color=self.font.White, background=self.font.Gray40)

        self.font.OverlayText(image, text=self.prompt, x=5, y=5, color=(120,215,21), background=self.font.Gray40)

        self.video_output(image)
   
    def on_text(self, text):
        if self.eos:
            self.text = text  # new query response
            self.eos = False
        elif not self.warmup:  # don't view warmup response
            self.text = self.text + text

        if text.endswith(tuple(StopTokens + ['###'])):
            self.print_stats()
            
            if not self.warmup:
                self.events(self.text, prompt=self.prompt)
                
            self.warmup = False
            self.eos = True

    def on_image_embedding(self, embedding):
        if self.tag_image and self.last_image:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"/data/datasets/uploads/{timestamp}.jpg"
            metadata = dict(path=filename, time=timestamp, tags=self.tag_image)
            self.tag_image = None 
            
            def save_image(filename, image, embedding, metadata):
                saveImage(filename, image)
                self.db(embedding, add=True, metadata=metadata)
                logging.info(f"added incoming image to database with tags '{self.tag_image}' ({filename})")
        
            threading.Thread(target=save_image, args=(filename, self.last_image, embedding, metadata)).start()
            
        if self.auto_refresh_db:
            self.db(embedding)
        
    def on_search(self, results):
        html = []
        
        for result in results:
            path = result['metadata']['path']
            for root, mount in self.server.mounts.items():
                if root in path:
                    html.append(dict(
                        image=path.replace(root, mount), 
                        similarity=f"{result['similarity']*100:.1f}%",
                        metadata=json.dumps(result['metadata'], indent=2).replace('"', '&quot;')
                    ))
                    
        if html:
            self.server.send_message({'search_results': html})
            
        cprint(f"nanodb search results:\n{pprint.pformat(results, indent=2)}", color='blue')
        
    def on_websocket(self, msg, msg_type=0, metadata='', **kwargs):
        if msg_type == WebServer.MESSAGE_JSON:
            #print(f'\n\n###############\n# WEBSOCKET JSON MESSAGE\n#############\n{msg}')
            if 'prompt' in msg:
                self.prompt = msg['prompt']
                self.prompt_history.append(self.prompt)
            elif 'pause_video' in msg:
                self.pause_video = msg['pause_video']
                self.pause_image = None
                logging.info(f"{'pausing' if self.pause_video else 'resuming'} processing of incoming video stream")
            elif 'auto_refresh' in msg:
                self.auto_refresh = msg['auto_refresh']
                logging.info(f"{'enabling' if self.auto_refresh else 'disabling'} auto-refresh of model output with prior query")
            elif 'auto_refresh_db' in msg:
                self.auto_refresh_db = msg['auto_refresh_db']
                logging.info(f"{'enabling' if self.auto_refresh_db else 'disabling'} auto-refresh of vector database search results")
            elif 'save_db' in msg:
                if self.db:
                    self.db.db.save()
            elif 'tag_image' in msg:
                self.tag_image = msg['tag_image']
   
    def poll_keyboard(self):
        while True:
            try:
                key = input().strip() #getch.getch()
                
                if key == 'd' or key == 'l':
                    self.keyboard_prompt = (self.keyboard_prompt + 1) % len(self.prompt_history)
                    self.prompt = self.prompt_history[self.keyboard_prompt]
                elif key == 'a' or key == 'j':
                    self.keyboard_prompt = self.keyboard_prompt - 1
                    if self.keyboard_prompt < 0:
                        self.keyboard_prompt = len(self.prompt_history) - 1
                    self.prompt = self.prompt_history[self.keyboard_prompt]
                    
                num = int(key)
                
                if num > 0 and num <= len(self.prompt_history):
                    self.keyboard_prompt = num - 1
                    self.prompt = self.prompt_history[self.keyboard_prompt]
                    
            except Exception as err:
                continue
     
    def print_stats(self):
        #print_table(self.llm.model.stats)
        curr_time = time.perf_counter()
            
        if not hasattr(self, 'start_time'):
            self.start_time = curr_time
        else:
            frame_time = curr_time - self.start_time
            self.start_time = curr_time
            refresh_str = f"{1.0 / frame_time:.2f} FPS ({frame_time*1000:.1f} ms)"
            self.server.send_message({'refresh_rate': refresh_str})
            logging.info(f"refresh rate:  {refresh_str}")

    def start(self):
        super().start()
        self.server.start()
        return self
        
if __name__ == "__main__":
    parser = ArgParser(extras=ArgParser.Defaults+['video_input', 'video_output', 'web', 'nanodb'])
    args = parser.parse_args()
    agent = VideoQuery(**vars(args)).run() 
    