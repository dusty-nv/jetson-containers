#!/usr/bin/env python3
import time

from local_llm import Agent, Pipeline

from local_llm.plugins import PrintStream, ProcessProxy
from local_llm.utils import ArgParser


class MultiprocessTest(Agent):
    """
    This is a test of the ProcessProxy plugin for running pipelines and plugins in their own processes.
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.pipeline = Pipeline([
            ProcessProxy((lambda **kwargs: PrintStream(**kwargs)), color='green', relay=True, **kwargs),
            PrintStream(color='blue', **kwargs),
        ])
        
         
if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_args()
    
    agent = MultiprocessTest(**vars(args)).start()
    
    while True:
        agent("INSERT MESSAGE HERE")
        time.sleep(1.0)