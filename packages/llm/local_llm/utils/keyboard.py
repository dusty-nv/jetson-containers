#!/usr/bin/env python3
import sys
import time
import signal
import logging

class KeyboardInterrupt():
    """
    Ctrl+C handler - if done once, sets the interrupt flag and optionally calls a callback.
    If done twice in succession, exits when 'timeout' is set to a positive number of seconds.
    """
    def __init__(self, callback=None, timeout=2.0):
        self.timeout = -1.0 if timeout is None else timeout
        self.callback = callback
        self.interrupted = False
        self.last_interrupt = 0.0
        signal.signal(signal.SIGINT, self.on_interrupt)
    
    def on_interrupt(self, signum, frame):
        curr_time = time.perf_counter()
        time_diff = curr_time - self.last_interrupt
        self.last_interrupt = curr_time
        
        if self.timeout <= 0 or time_diff > 2.0:
            logging.warning("Ctrl+C:  interrupting output")
            self.interrupted = True
            if self.callback is not None:
                self.callback()
        else:
            while True:
                try:
                    logging.warning("Ctrl+C:  exiting...")
                    sys.exit(0)
                    time.sleep(0.5)
                except Exception as error:
                    logging.error(f"Exception exiting the program:\n{error}")
        
    def __bool__(self):
        return self.interrupted
        
    def reset(self):
        self.interrupted = False