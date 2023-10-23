#!/usr/bin/env python3
from local_llm import Agent, Pipeline, ChatTemplates

from local_llm.plugins import UserPrompt, ChatQuery, PrintStream
from local_llm.utils import ArgParser, print_table

from termcolor import cprint


class ChatAgent(Agent):
    """
    Agent for two-turn multimodal chat.
    """
    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", interactive=True, **kwargs):
        super().__init__()
        
        """
        # Equivalent to:
        self.pipeline = UserPrompt(interactive=interactive, **kwargs).add(
            LLMQuery(model, **kwargs).add(
            PrintStream(relay=True).add(self.on_eos)     
        ))
        """
        self.pipeline = Pipeline([
            UserPrompt(interactive=interactive, **kwargs),
            ChatQuery(model, **kwargs),
            PrintStream(relay=True),
            self.on_eos    
        ])
        
        self.model = self.pipeline[0].find(ChatQuery).model
        self.interactive = interactive
        
        self.print_input_prompt()

    def on_eos(self, input):
        if input.endswith('</s>'):
            print_table(self.model.stats)
            self.print_input_prompt()

    def print_input_prompt(self):
        if self.interactive:
            cprint('>> PROMPT: ', 'blue', end='', flush=True)
        
        
if __name__ == "__main__":
    from local_llm.utils import ArgParser

    parser = ArgParser()
    parser.add_argument("-it", "--interactive", action="store_true", help="enable interactive user input from the terminal")
    args = parser.parse_args()
    
    agent = ChatAgent(**vars(args)).run() 