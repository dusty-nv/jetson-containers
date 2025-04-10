#!/usr/bin/env python3
import logging

from local_llm import Agent

from local_llm.plugins import VideoSource, VideoOutput
from local_llm.utils import ArgParser


class VideoStream(Agent):
    """
    Relay, view, or test a video stream.  Use the --video-input and --video-output arguments
    to set the video source and output protocols used from:
    
      https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md
      
    For example, this will capture a V4L2 camera and serve it via WebRTC with H.264 encoding:
    
      python3 -m local_llm.agents.video_stream \
        --video-input /dev/video0 \
        --video-output webrtc://@:8554/output
      
    It's also used as a basic test of video streaming before using more complex agents that rely on it.
    """
    def __init__(self, video_input=None, video_output=None, **kwargs):
        super().__init__()

        self.video_source = VideoSource(video_input, **kwargs)
        self.video_output = VideoOutput(video_output, **kwargs)
        
        self.video_source.add(self.on_video, threaded=False)
        self.video_source.add(self.video_output)
        
        self.pipeline = [self.video_source]
        
    def on_video(self, image):
        logging.debug(f"captured {image.width}x{image.height} frame from {self.video_source.resource}")

         
if __name__ == "__main__":
    parser = ArgParser(extras=['video_input', 'video_output', 'log'])
    args = parser.parse_args()
    
    agent = VideoStream(**vars(args)).run() 