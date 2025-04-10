#!/usr/bin/env python3
import collections
import termcolor
import logging

from local_llm import Plugin, ChatHistory

class PrintStream(Plugin):
    """
    Output plugin that prints chatbot responses to stdout.
    """
    def __init__(self, partial=True, prefix=None, color='green', **kwargs):
        """
        Parameters:
          partial (bool) -- if true, print token-by-token (otherwise, end with newline)
          prefix (str) -- text to print out before incoming text
          color (str) -- the color to print the output stream (or None for no colors)
        """
        super().__init__(**kwargs)
        
        self.partial = partial
        self.prefix = prefix
        self.color = color
        self.last_length = 0
        
    def process(self, input, **kwargs):
        """
        Expects to recieve a string, ChatHistory, or StreamingResponse iterator,
        and prints the incoming token stream to stdout.
        """
        if isinstance(input, str):
            self.print(input)
        elif isinstance(input, ChatHistory):  # latest chat history entry
            entry = input[-1]
            if entry.role != 'bot':
                logging.warning(f"PrintStream plugin recieved chat entry with role={entry.role}")
            if self.partial:
                self.print(entry.text[self.last_length:])
                self.last_length = len(entry.text)
        elif isinstance(input, collections.abc.Iterable):  # StreamingResponse from LocalLM.generate()
            for token in output:
                self.print(token)
        else:
            raise TypeError(f"PrintStream plugin expects inputs of type str, ChatHistory, or StreamingResponse (was {type(input)})")

    def print(self, text):
        eos = text.endswith('</s>')
        text = text.replace('</s>', '')
        
        if self.prefix:
            text = self.prefix + text
            
        if self.color:
            text = termcolor.colored(text, self.color)
            
        if self.partial and not eos:
            print(text, end='', flush=True)
        else:
            print(text)