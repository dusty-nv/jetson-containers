#!/usr/bin/env python3
import time
import logging

from local_llm import Plugin

# RateLimit(48000*2, chunk=6000)

class RateLimit(Plugin):
    """
    Rate limiter plugin with the ability to pause/resume from the queue.
    
      video_limiter = RateLimit(30)  # 30 FPS
      audio_limiter = RateLimit(48000, chunk=4800)  
      
    It can also chunk indexable outputs into smaller amounts of data at a time.
    """
    def __init__(self, rate=None, chunk=None, **kwargs):
        """
        Parameters:
        
          rate (int) -- The number of items per second that can be transmitted
          chunk (int) -- for indexable inputs, the maximum number of items 
                         that can be in each output message (if None, no chunking)
        """
        super().__init__(**kwargs)
        
        self.rate = rate
        self.chunk = chunk
        self.paused = -1
        
    def process(self, input, **kwargs):
        """
        First, wait for any pauses that were requested in the output.
        If chunking enabled, chunk the input down until it's gone.
        Then wait as necessary to maintain the requested output rate.
        """
        while True:
            if self.interrupted:
                #logging.debug(f"RateLimit interrupted (input={len(input)})")
                return
            
            pause_duration = self.pause_duration()
            
            if pause_duration > 0:
                #logging.debug(f"RateLimit pausing for {pause_duration} seconds (input={len(input)})")
                time.sleep(pause_duration)
                continue
               
            if self.chunk > 0:
                #logging.debug(f"RateLimit chunk {len(input)}  {self.chunk}  {time.perf_counter()}")
                if len(input) > self.chunk:
                    self.output(input[:self.chunk])
                    input = input[self.chunk:]
                    time.sleep(self.chunk/self.rate*0.95)
                    new=False
                    continue
                else:
                    self.output(input)
                    time.sleep(len(input)/self.rate*0.95)
                    return
            else:
                self.output(input)
                if self.rate > 0:
                    time.sleep(1.0/self.rate)
                return
            
    def pause(self, duration=None, until=None):
        """
        Pause audio playback for `duration` number of seconds, or until the end time.
        
        If `duration` is 0, it will be paused indefinitely until unpaused.
        If `duration` is negative, it will be unpaused.
        
        If already paused, the pause will be extended if it exceeds the current duration.
        """
        current_time = time.perf_counter()
        
        if duration is None and until is None:
            raise ValueError("either 'duration' or 'until' need to be specified")
            
        if duration is not None:
            if duration <= 0:
                self.paused = duration  # disable/infinite pausing
            else:
                until = current_time + duration
         
        if until is not None:
            if until > self.paused and self.paused != 0:
                self.paused = until
                logging.debug(f"RateLimit - pausing output for {until-current_time} seconds")

    def unpause(self):
        """
        Unpause audio playback
        """
        self.pause(-1.0)
        
    def is_paused(self):
        """
        Returns true if playback is currently paused.
        """
        return self.pause_duration() > 0
      
    def pause_duration(self):
        """
        Returns the time to go if still paused (or zero if unpaused)
        """
        if self.paused < 0:
            return 0

        if self.paused == 0:
            return float('inf')
            
        current_time = time.perf_counter()
        
        if current_time >= self.paused:
            self.paused = -1
            return 0
            
        return self.paused - current_time
