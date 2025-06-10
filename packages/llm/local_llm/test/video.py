#!/usr/bin/env python3
#
# Relay, view, or test a video stream.  Use the --video-input and --video-output arguments
# to set the video source and output protocols used from:
#    
#      https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md
#      
# For example, this will capture a V4L2 camera and serve it via WebRTC with H.264 encoding:
#    
#      python3 -m local_llm.test.video \
#        --video-input /dev/video0 \
#        --video-output webrtc://@:8554/output
#      
# It's also used as a basic test of video streaming before using more complex agents that rely on it.
#
import logging

from local_llm.plugins import VideoSource, VideoOutput
from local_llm.utils import ArgParser

args = ArgParser(extras=['video_input', 'video_output', 'log']).parse_args()

def on_video(image):
    num_frames = video_source.stream.GetFrameCount()
    if num_frames % 25 == 0:
        logging.info(f'captured {num_frames} frames ({image.width}x{image.height}) from {video_source.resource}')
        
video_source = VideoSource(**vars(args))
video_output = VideoOutput(**vars(args))

video_source.add(on_video, threaded=False)
video_source.add(video_output)

video_source.start().join()
