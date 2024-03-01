#!/usr/bin/env python3
import sys
import time
import signal
import logging

class KeyboardInterrupt():
    """
    Ctrl+C handler - if done once, interrupts the LLM
    If done twice in succession, exits the program
    """
    def __init__(self):
        self.interrupted = False
        self.last_interrupt = 0.0
        signal.signal(signal.SIGINT, self.on_interrupt)
    
    def on_interrupt(self, signum, frame):
        curr_time = time.perf_counter()
        time_diff = curr_time - self.last_interrupt
        self.last_interrupt = curr_time
        
        if time_diff > 2.0:
            logging.warning("Ctrl+C:  interrupting chatbot")
            self.interrupted = True
        else:
            while True:
                logging.warning("Ctrl+C:  exiting...")
                sys.exit(0)
                time.sleep(0.5)
        
    def __bool__(self):
        return self.interrupted
        
    def reset(self):
        self.interrupted = False