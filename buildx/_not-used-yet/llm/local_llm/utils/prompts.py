#!/usr/bin/env python3
import os
import json
import logging

# https://modal.com/docs/guide/ex/vllm_inference
DefaultChatPrompts = [
    "What is the weather forecast today?",
    "What is the fable involving a fox and grapes?",
    "What's a good recipe for making tabouli?",
    "How do I allocate memory in C?",
    "Implement a Python function to compute the Fibonacci numbers.",
    "What is the product of 9 and 8?",
    "Is Pluto really a planet or not?",
    "When was the Hoover Dam built?",
    "What's a training plan to run a marathon?",
    "If a train travels 120 miles in 2 hours, what is its average speed?",
]

DefaultCompletionPrompts = [
    "Once upon a time,",
    "A great place to live is",
    "In a world where dreams are shared,",
    "The weather forecast today is",
    "Large language models are",
    "Space exploration is exciting",
    "The history of the Hoover Dam is",
    "San Fransisco is a city in",
    "To train for running a marathon,",
    "A recipe for making tabouli is"
]


def load_prompts(prompts):
    """
    Load prompts from a list of txt or json files
    (or if these are strings, just return the strings)
    """
    if prompts is None:
        return None
        
    if isinstance(prompts, str):
        prompts = [prompts]
        
    prompt_list = []
    
    for prompt in prompts:
        ext = os.path.splitext(prompt)[1]
        
        if ext == '.json':
            logging.info(f"loading prompts from {prompt}")
            with open(prompt) as file:
                json_prompts = json.load(file)
            for json_prompt in json_prompts:
                if isinstance(json_prompt, dict):
                    prompt_list.append(json_prompt['text'])
                elif isinstance(json_prompt, str):
                    prompt_list.append(json_prompt)
                else:
                    raise TypeError(f"{type(json_prompt)}")
        elif ext == '.txt':
            with open(prompt) as file:
                prompt_list.append(file.read())
        else:
            prompt_list.append(prompt)
            
    return prompt_list
    