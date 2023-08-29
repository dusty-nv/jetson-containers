#!/usr/bin/env python3
#
# Streaming LLM service for text completion and chat (using oobabooga text-generation-webui API)
#
# To test text completion, load a model in the text-generation-webui server and run the following: 
#
#   python3 llm.py --max-new-tokens=128 \
#      --prompt "Once upon a time," \
#      --prompt "In a galaxy far, far away" \
#      --prompt "A good recipe for pasta is"
#
# To test multi-turn chat, load a chat model in the text-generation-webui server and run the following:
#
#   python3 llm.py --max-new-tokens=128 --chat \
#      --prompt "What's for dinner tonight?" \
#      --prompt "My, that does sound delightful!" \
#      --prompt "It sounds like you're quite the chef!  How long have you been cooking for?"
#
import sys
import json
import queue
import pprint
import asyncio
import requests
import threading

from websockets.sync.client import connect as websocket_connect


class LLM(threading.Thread):
    """
    Streaming LLM service using text-generation-webui API
    """
    def __init__(self, llm_server='0.0.0.0', llm_api_port=5000, llm_streaming_port=5005, log_level=0, **kwargs):
                 
        super(LLM, self).__init__(daemon=True)  # stop thread on main() exit
        
        self.queue = queue.Queue()
        self.log_level = log_level
        
        self.server = llm_server
        self.blocking_port = llm_api_port
        self.streaming_port = llm_streaming_port
        
        self.request_count = 0
        self.muted = False
        
        pprint.pprint(self.model_list())
        pprint.pprint(self.model_info())   
        
        model_name = self.model_name().lower()
        
        # find default chat template based on the model
        self.instruction_template = None
        
        if any(x in model_name for x in ['llama2', 'llama_2', 'llama-2']):
            self.instruction_template = 'Llama-v2'
        elif 'vicuna' in model_name:
            self.instruction_template = 'Vicuna-v1.1'
    
    def model_info(self):
        """
        Returns info about the model currently loaded on the server.
        """
        return self.model_api({'action': 'info'})['result']
        
    def model_name(self):
        """
        Return the list of models available on the server.
        """
        return self.model_info()['model_name']
        
    def model_list(self):
        """
        Return the list of models available on the server.
        """
        return self.model_api({'action': 'list'})['result']

    def model_api(self, request):
        """
        Call the text-generation-webui model API with one of these requests:
        
           {'action': 'info'}
           {'action': 'list'}
           
        See model_list() and model_info() for using these requests.
        """
        return requests.post(f'http://{self.server}:{self.blocking_port}/api/v1/model', json=request).json()
    
    @staticmethod
    def create_new_history():
        """
        Create a new/blank history dict
        """
        return {'internal': [], 'visible': []}
    
    @staticmethod
    def add_prompt_history(history, prompt, type='internal'):
        """
        Return a new history list that includes the given prompt (but no reply yet)
        This only operates on the history list specified by type ('internal' or 'visible')
        """
        history = history[type].copy()
        
        if len(history) > 0 and len(history[-1]) == 1:
            history[-1][0] = prompt
        else:
            history.append([prompt])
            
        return history
        
    def generate(self, prompt, callback=None, **kwargs):
        """
        Generate an asynchronous text completion request to run on the LLM server.
        You can set optional parameters for the request through the kwargs (e.g. max_new_tokens=50)
        If the callback function is provided, it will be called as the generated tokens are streamed in.
        This function returns the request that was queued.
        """
        params = {
            'prompt': prompt,
            'max_new_tokens': 250,
            'auto_max_new_tokens': False,

            # Generation params. If 'preset' is set to different than 'None', the values
            # in presets/preset-name.yaml are used instead of the individual numbers.
            'preset': 'None',
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.1,
            'typical_p': 1,
            'epsilon_cutoff': 0,  # In units of 1e-4
            'eta_cutoff': 0,  # In units of 1e-4
            'tfs': 1,
            'top_a': 0,
            'repetition_penalty': 1.18,
            'repetition_penalty_range': 0,
            'top_k': 40,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,
            'guidance_scale': 1,
            'negative_prompt': '',

            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'skip_special_tokens': True,
            'stopping_strings': []
        }
        
        params.update(kwargs)
        
        request = {
            'id': self.request_count,
            'type': 'completion',
            'params': params,
            'output': '',
            'callback': callback,
            'event': threading.Event(),
        }
        
        self.request_count += 1
        self.enqueue(request, clear=True)
        
        return request
    
    def generate_chat(self, user_input, history, callback=None, **kwargs):
        """
        Generate an asynchronous chat request to run on the LLM server.
        You can set optional parameters for the request through the kwargs (e.g. max_new_tokens=50)
        If the callback function is provided, it will be called as the generated tokens are streamed in.
        This function returns the request that was queued.
        """
        if history is None:
            history = LLM.create_new_history()
            
        params = {
            'user_input': user_input,
            'max_new_tokens': 250,
            'auto_max_new_tokens': False,
            'history': history,
            'mode': 'instruct',  # Valid options: 'chat', 'chat-instruct', 'instruct'
            'character': 'Example',
            #'instruction_template': 'Llama-v2',  # Will get autodetected if unset (see below)
            'your_name': 'You',
            # 'name1': 'name of user', # Optional
            # 'name2': 'name of character', # Optional
            # 'context': 'character context', # Optional
            # 'greeting': 'greeting', # Optional
            # 'name1_instruct': 'You', # Optional
            # 'name2_instruct': 'Assistant', # Optional
            # 'context_instruct': 'context_instruct', # Optional
            # 'turn_template': 'turn_template', # Optional
            'regenerate': False,
            '_continue': False,
            'stop_at_newline': False,
            'chat_generation_attempts': 1,
            'chat_instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',

            # Generation params. If 'preset' is set to different than 'None', the values
            # in presets/preset-name.yaml are used instead of the individual numbers.
            'preset': 'None',
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.1,
            'typical_p': 1,
            'epsilon_cutoff': 0,  # In units of 1e-4
            'eta_cutoff': 0,  # In units of 1e-4
            'tfs': 1,
            'top_a': 0,
            'repetition_penalty': 1.18,
            'repetition_penalty_range': 0,
            'top_k': 40,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,
            'guidance_scale': 1,
            'negative_prompt': '',

            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'skip_special_tokens': True,
            'stopping_strings': []
        }
        
        params.update(kwargs)
        
        if 'instruction_template' not in params and self.instruction_template:
            params['instruction_template'] = self.instruction_template
            
        request = {
            'id': self.request_count,
            'type': 'chat',
            'params': params,
            'output': {},
            'callback': callback,
            'event': threading.Event()
        }
        
        self.request_count += 1
        self.enqueue(request, clear=True)
        
        return request
        
    def enqueue(self, request, clear=False):
        if clear:
            while not self.queue.empty():
                self.queue.get()        
        self.queue.put(request)
    
    def mute(self):
        self.muted = True
        
    def run(self):
        print(f"-- running LLM service ({self.model_name()})")
        
        while True:
            request = self.queue.get()
            self.muted = False
            
            if self.log_level > 0:
                print("-- LLM:")
                pprint.pprint(request)
            
            if request['type'] == 'completion':
                url = f"ws://{self.server}:{self.streaming_port}/api/v1/stream"
            elif request['type'] == 'chat':
                url = f"ws://{self.server}:{self.streaming_port}/api/v1/chat-stream"
                
            with websocket_connect(url) as websocket:
                websocket.send(json.dumps(request['params']))
                
                while True:
                    incoming_data = websocket.recv()
                    incoming_data = json.loads(incoming_data)

                    if incoming_data['event'] == 'text_stream':
                        key = 'history' if request['type'] == 'chat' else 'text'
                        if request['type'] == 'chat':
                            response = incoming_data['history']
                            request['output'] = response
                        elif request['type'] == 'completion':
                            response = incoming_data['text']
                            request['output'] += response
                        if request['callback'] is not None:
                            request['callback'](response, request=request, end=False)
                            
                    if incoming_data['event'] == 'stream_end' or self.muted:
                        if request['callback'] is not None:
                            request['callback'](None, request=request, end=True)
                        request['event'].set()
                        break


if __name__ == '__main__':
    import argparse
    from termcolor import cprint
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--llm-server", type=str, default='0.0.0.0', help="hostname of the LLM server (text-generation-webui)")
    parser.add_argument("--llm-api-port", type=int, default=5000, help="port of the blocking API on the LLM server")
    parser.add_argument("--llm-streaming-port", type=int, default=5005, help="port of the streaming websocket API on the LLM server")
    parser.add_argument("--max-new-tokens", type=int, default=250, help="the maximum number of new tokens for the LLM to generate")
    parser.add_argument("--prompt", action='append', nargs='*')
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if not args.prompt:
        if args.chat:
            args.prompt = ["Please give me a step-by-step guide on how to plant a tree in my backyard."]
        else:
            args.prompt = ["Once upon a time,"]
    else:
        args.prompt = [x[0] for x in args.prompt]
        
    print(args)
    
    llm = LLM(**vars(args))
    llm.start()
    
    def on_llm_reply(response, request, end):
        if not end:
            if request['type'] == 'completion':
                print(response, end='')
                sys.stdout.flush()
            elif request['type'] == 'chat':
                current_length = request.get('current_length', 0)
                msg = response['visible'][-1][1][current_length:]
                request['current_length'] = current_length + len(msg)
                print(msg, end='')
                sys.stdout.flush()
        else:
            print("\n")

    if args.chat:
        history = {'internal': [], 'visible': []}
        
        for i, prompt in enumerate(args.prompt):
            cprint(prompt, 'blue')
            request = llm.generate_chat(prompt, history, max_new_tokens=args.max_new_tokens, _continue=False, callback=on_llm_reply)
            request['event'].wait()
            history = request['output']
    else:
        for prompt in args.prompt:
            cprint(prompt, 'blue')
            request = llm.generate(prompt, max_new_tokens=args.max_new_tokens, callback=on_llm_reply)
            request['event'].wait()
    