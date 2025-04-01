#!/usr/bin/env python3
import time
import logging
import traceback

import torch
import numpy as np

from local_llm import Plugin
from local_llm.utils import cuda_image

from jetson_utils import videoSource, videoOutput, cudaDeviceSynchronize, cudaToNumpy


class VideoSource(Plugin):
    """
    Captures or loads a video/camera stream or sequence of images
    https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md
    """
    def __init__(self, video_input='/dev/video0', 
                 video_input_width=None, video_input_height=None, 
                 video_input_codec=None, video_input_framerate=None, 
                 video_input_save=None, return_tensors='cuda', **kwargs):
        """
        Parameters:
        
          input (str) -- path to video file, directory of images, or stream URL
          input_width (int) -- the disired width in pixels (default uses stream's resolution)
          input_height (int) -- the disired height in pixels (default uses stream's resolution)
          input_codec (str) -- force a particular codec ('h264', 'h265', 'vp8', 'vp9', 'mjpeg', ect)
          return_tensors (str) -- the object datatype of the image to output ('np', 'pt', 'cuda')
        """
        super().__init__(**kwargs)
        
        options = {}
        
        if video_input_width:
            options['width'] = video_input_width
            
        if video_input_height:
            options['height'] = video_input_height
            
        if video_input_codec:
            options['codec'] = video_input_codec
 
        if video_input_framerate:
            options['framerate'] = video_input_framerate
            
        if video_input_save:
            options['save'] = video_input_save
        
        self.stream = videoSource(video_input, options=options)
        self.options = options
        self.resource = video_input  # self.stream.GetOptions().resource['string']
        self.return_tensors = return_tensors
        
    def capture(self):
        """
        Capture images from the video source as long as it's streaming
        """
        retries = 0
        
        while retries < 8:
            image = self.stream.Capture(format='rgb8', timeout=2500)

            if image is None:
                logging.warning(f"video source {self.resource} timed out during capture, re-trying...")
                retries = retries + 1
                continue
            else:
                retries = 0

            if self.return_tensors == 'pt':
                image = torch.as_tensor(image, device='cuda')
            elif self.return_tensors == 'np':
                image = cudaToNumpy(image)
                cudaDeviceSynchronize()
            elif self.return_tensors != 'cuda':
                raise ValueError(f"return_tensors should be 'np', 'pt', or 'cuda' (was '{self.return_tensors}')")
                
            self.output(image)             
    
    def reconnect(self):
        """
        Attempt to re-open the stream if the connection fails
        """
        while True:
            try:
                if self.stream is not None:
                    self.stream.Close()
                    self.stream = None
            except Exception as error:
                logging.error(f"Exception occurred closing video source \"{self.resource}\"\n\n{''.join(traceback.format_exception(error))}")

            try:
                self.stream = videoSource(self.resource, options=self.options)
                return
            except Exception as error:
                logging.error(f"Failed to create video source \"{self.resource}\"\n\n{''.join(traceback.format_exception(error))}")
                traceback.print_exception(error)
                time.sleep(2.5)
            
    def run(self):
        """
        Run capture continuously and attempt to handle disconnections
        """
        while True:
            try:
                self.capture()
            except Exception as error:
                logging.error(f"Exception occurred during video source capture of \"{self.resource}\"\n\n{''.join(traceback.format_exception(error))}")
                
            logging.error(f"Re-initializing video source \"{self.resource}\"")
            self.reconnect()
                
                    
                    
class VideoOutput(Plugin):
    """
    Saves images to a compressed video or directory of individual images, the display, or a network stream.
    https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md
    """
    def __init__(self, video_output=None, video_output_codec=None, video_output_bitrate=None, video_output_save=None, **kwargs):
        """
        Parameters:
        
          input (str) -- path to video file, directory of images, or stream URL
          output_codec (str) -- force a particular codec ('h264', 'h265', 'vp8', 'vp9', 'mjpeg', ect)
          output_bitrate (int) -- the desired bitrate in bits per second (default is 4 Mbps)
        """
        super().__init__(**kwargs)
        
        options = {}

        if video_output_codec:
            options['codec'] = video_output_codec
            
        if video_output_bitrate:
            options['bitrate'] = video_output_bitrate

        if video_output_save:
            options['save'] = video_output_save
            
        if video_output is None:
            video_output = ''
            
        args = None if 'display://' in video_output else ['--headless']
        
        self.stream = videoOutput(video_output, options=options, argv=args)
        self.resource = video_output
        
    def process(self, input, **kwargs):
        """
        Input should be a jetson_utils.cudaImage, np.ndarray, torch.Tensor, or have __cuda_array_interface__
        """
        self.stream.Render(cuda_image(input))
            