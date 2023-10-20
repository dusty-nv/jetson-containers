#!/usr/bin/env python3
import argparse
import logging

from .log import LogFormatter


class ArgParser(argparse.ArgumentParser):
    """
    Adds selectable extra args that are commonly used by this project
    """
    def __init__(self, extras=['model', 'chat', 'generation', 'log'], **kwargs):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs)
        
        if 'model' in extras:
            self.add_argument("--model", type=str, required=True, 
                help="path to the model, or repository on HuggingFace Hub")
            self.add_argument("--quant", type=str, default=None, 
                help="path to the quantized weights (AWQ uses this)")
            self.add_argument("--api", type=str, default=None, choices=['auto_gptq', 'awq', 'hf', 'mlc'], 
                help="specify the API to use (otherwise inferred)")
            self.add_argument("--vision-model", type=str, default=None, 
                help="for VLMs, manually select the CLIP vision model to use (e.g. openai/clip-vit-large-patch14-336 for higher-res)")

        if 'chat' in extras:
            self.add_argument("--prompt", action='append', nargs='*', 
                help="add a prompt (can be prompt text or path to .txt, .json, or image file)")
            self.add_argument("--system-prompt", type=str, default=None, 
                help="override the system prompt instruction")
            self.add_argument("--chat-template", type=str, default=None, #choices=list(ChatTemplates.keys()), 
                help="manually select the chat template ('llama-2', 'llava-v1', 'vicuna-v1')")

        if 'generation' in extras:
            self.add_argument("--max-new-tokens", type=int, default=128, 
                help="the maximum number of new tokens to generate, in addition to the prompt")
            self.add_argument("--min-new-tokens", type=int, default=-1,
                help="force the model to generate a minimum number of output tokens")
            self.add_argument("--do-sample", action="store_true",
                help="enable output token sampling using temperature and top_p")
            self.add_argument("--temperature", type=float, default=0.7,
                help="token sampling temperature setting when --do-sample is used")
            self.add_argument("--top-p", type=float, default=0.95,
                help="token sampling top_p setting when --do-sample is used")
            self.add_argument("--repetition-penalty", type=float, default=1.0,
                help="the parameter for repetition penalty. 1.0 means no penalty")

        if 'log' in extras:
            self.add_argument("--log-level", type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'], 
                help="the logging level to stdout")
            self.add_argument("--debug", "--verbose", action="store_true", 
                help="set the logging level to debug/verbose mode")
                
    def parse_args(self, **kwargs):
        """
        Override for parse_args() that does some additional configuration
        """
        args = super().parse_args(**kwargs)
        
        if hasattr(args, 'prompt'):
            args.prompt = ArgParser.parse_prompt_args(args.prompt)
        
        if hasattr(args, 'log_level'):
            if args.debug:
                args.log_level = "debug"
            LogFormatter.config(level=args.log_level)
            
        logging.debug(f"{args}")
        return args
        
    @staticmethod
    def parse_prompt_args(prompts, chat=True):
        """
        Parse prompt command-line argument and return list of prompts
        It's assumed that the argparse argument was created like this:
        
          `parser.add_argument('--prompt', action='append', nargs='*')`
          
        If the prompt text is 'default', then default chat prompts will
        be assigned if chat=True (otherwise default completion prompts)
        """
        if prompts is None:
            return None
            
        prompts = [x[0] for x in prompts]
        
        if prompts[0].lower() == 'default' or prompts[0].lower() == 'defaults':
            prompts = DefaultChatPrompts if chat else DefaultCompletionPrompts
            
        return prompts
        