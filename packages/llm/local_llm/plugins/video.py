#!/usr/bin/env python3
import torch
import numpy as np

from local_llm import Plugin
from local_llm.utils import cuda_image

from jetson_utils import videoSource, videoOutput, cudaMemcpy


class VideoSource(Plugin):
    """
    Captures or loads a video/camera stream or sequence of images
    https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md
    """
    def __init__(self, video_input='/dev/video0', 
                 video_input_width=None, video_input_height=None, 
                 video_input_codec=None, video_input_framerate=None, 
                 **kwargs):
        """
        Parameters:
        
          input (str) -- path to video file, directory of images, or stream URL
          input_width (int) -- the disired width in pixels (default uses stream's resolution)
          input_height (int) -- the disired height in pixels (default uses stream's resolution)
          input_codec (str) -- force a particular codec ('h264', 'h265', 'vp8', 'vp9', 'mjpeg', ect)
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
            
        self.stream = videoSource(video_input, options=options)
        self.resource = video_input
        
    def run(self):
        """
        Capture images from the video source as long as it's streaming
        """
        while True:
            image = self.stream.Capture(format='rgb8', timeout=-1)
            
            if image is None:
                continue
                
            self.output(cudaMemcpy(image))            
            
        
class VideoOutput(Plugin):
    """
    Saves images to a compressed video or directory of individual images, the display, or a network stream.
    https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md
    """
    def __init__(self, video_output=None, video_output_codec=None, video_output_bitrate=None, **kwargs):
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

        self.stream = videoOutput(video_output, options=options)
        self.resource = output
        
    def process(self, input, **kwargs):
        """
        Input should be a jetson_utils.cudaImage, np.ndarray, torch.Tensor, or have __cuda_array_interface__
        """
        self.stream.Render(cuda_image(input))
            