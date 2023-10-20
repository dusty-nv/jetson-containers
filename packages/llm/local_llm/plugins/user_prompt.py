#!/usr/bin/env python3
import sys
import threading

from termcolor import cprint

from local_llm import Plugin
from local_llm.utils import load_prompts

#
# TODO  ImageLoader in another plugin?
# ChatAgent that manages the ChatHistory, outputs embeddings to LLMQuery
# (or ChatPlugin, ChatManager, ChatDialog - and save ChatAgent for the overall pipeline)
#
# DefaultChatPrompts and DefaultCompletionPrompts globals defined up above
#
# create_plugin()  create_agent()
#
# each agent should kinda be it's own main() cause has different args??
#
# Create Agent which also inherits Plugin and makes pipeline management easier
# Change __main__ to have --agent arg   chang
# OK - make agents/chat.py with ChatAgent and __main__ , creates/manages the ChatHistory
#
class UserPrompt(Plugin):
    """
    Source plugin that reads text prompts, either interactively from stdin,
    or from a .txt or .json file.  It will also forward/open any text inputs.
    Outputs a string of the prompt text (or list of strings if multiple prompts).
    """
    def __init__(self, prompt=None, interactive=False, prefix=None, **kwargs):
        """
        Parameters:
        
          prompt (str) -- optional initial prompt or path to txt/json file
          interactive (bool) -- read user input from stdin (todo file descriptors)
          prefix (str) -- when in interactive mode, the prompt for user input
        """
        super().__init__(**kwargs)
        
        self.prefix = prefix
        self.interactive = interactive
        
        if self.interactive:
            self.user_thread = threading.Thread(target=self.read_user, daemon=True)
            self.user_thread.start()
            
    def process(self, input, **kwargs):
        if not isinstance(input, str):
            raise TypeError(f"AutoPrompt input should be of type str (was {type(input)})")
    
        if input.lower().endswith(('.txt', '.json')):
            input = load_prompts(input) #' '.join(load_prompts(input))
            
        return input
    
    def read_user(self):
        while True:
            if self.prefix:
                cprint(self.prefix, 'blue', end='', flush=True)
            self.output(sys.stdin.readline().strip())
            