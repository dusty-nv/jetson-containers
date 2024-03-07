#!/usr/bin/env python3
from ..utils import AttributeDict


# TODO: revisit bot trailing templates, and if \n are necessary (they were for open_llama)
#       add proper generation template instead of pre-pending it to the user template
ChatTemplates = {
    # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    'llama-2': {
        'system_prompt': "Answer the questions.",
        'system': '<s>[INST] <<SYS>>\n${MESSAGE}\n<</SYS>>\n\n',
        'first': '${MESSAGE} [/INST]',
        'user': '<s>[INST] ${MESSAGE} [/INST]',
        'bot': ' ${MESSAGE}'  # llama-2 output already ends in </s>
    },
    
    # https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
    'tiny-llama': {
        'system_prompt': "You are a friendly chatbot who always gives helpful answers to the user's questions.",
        'system': "<|system|>\n${MESSAGE}</s>\n",
        'user': "<|user|>\n${MESSAGE}</s>\n<|assistant|>\n",
        'bot': "${MESSAGE}",  # model output already ends in </s>
    },
    
    # https://huggingface.co/princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT
    'sheared-llama': {
        'system_prompt': "You are a helpful assistant. Write a response that appropriately completes the request.",
        'system': "${MESSAGE}\n\n",
        'user': "### Input:\n${MESSAGE}\n\n### Response:",
        'bot': "${MESSAGE}",
    },
    
    # https://huggingface.co/openlm-research/open_llama_3b_v2
    'open-llama': {
        'user': "Q: ${MESSAGE}\nA:",
        'bot': "${MESSAGE}\n",
        'stop': ["</s>", '\n', 'Q:'],  # open_llama only really outputs 1 line, and spits gibberish after
    },
    
    # https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
    'vicuna-v0': {
        'system_prompt': "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
        'system': '${MESSAGE}\n\n',
        'user': '### Human: ${MESSAGE}\n',
        'bot': '### Assistant: ${MESSAGE}\n',
    },
    
    'vicuna-v1': {
        'system_prompt': "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
        'system': '${MESSAGE}\n\n',
        'user': 'USER: ${MESSAGE}\n',
        'bot': 'ASSISTANT: ${MESSAGE}\n', # TODO: does output already end in </s> ?
    },
    
    # https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/ai-services/openai/includes/chat-markup-language.md#working-with-chat-markup-language-chatml
    # TODO:  if bot template has prefix prompt before $MESSAGE, prepend generation prompt to the chat
    #        make a add_reply() function that makes an empty bot placehold ChatMessage 
    #        make a generate() function that adds the bot reply, calls LLM
    'chat-ml': {
        'system_prompt': "You are a helpful AI assistant.",
        'system': "<|im_start|>system\n${MESSAGE}<|im_end|>\n",
        'user': "<|im_start|>user\n${MESSAGE}<|im_end|>\n<|im_start|>assistant\n",
        'bot': "${MESSAGE}\n",  # <|im_end|> is after $MESSAGE, but is already included in bot output
    },
    
    # https://github.com/NousResearch/Obsidian/blob/e09c51d88d74657f442a898e3c4607a5b961f0b3/llava/llava/conversation.py#L385
    'nous-obsidian': {
        'system_prompt': "You are a helpful AI assistant.",
        'system': "<|im_start|>system\n${MESSAGE}\n###\n",
        'user': "<|im_start|>user\n${MESSAGE}\n###\n<|im_start|>assistant\n",
        'bot': "${MESSAGE}\n",  # ### is after $MESSAGE, but is already included in bot output
        'stop': ["###", '<|im_end|>'],
    },
    
    # https://ollama.com/library/stablelm-zephyr:latest
    'stablelm-zephyr': {
        'system_prompt': "You are a helpful AI assistant.",
        'system': "<|system|>\n${MESSAGE}<|endoftext|>\n",
        'user': "<|user|>\n${MESSAGE}<|endoftext|>\n<|assistant|>\n",
        'bot': "${MESSAGE}\n",  # <|endoftext|> is after $MESSAGE, but is already included in bot output
    },
    
    # https://huggingface.co/microsoft/phi-2
    'phi-2-chat': {
        'user': "Alice: ${MESSAGE}\nBob: ",
        'bot': "${MESSAGE}\n",
    },
    
    'phi-2-instruct': {
        'user': "Instruct: ${MESSAGE}\nOutput: ",
        'bot': "${MESSAGE}\n",
    },
    
    # https://huggingface.co/google/gemma-2b-it
    'gemma': {
        'first': "<bos><start_of_turn>user\n${MESSAGE}<end_of_turn>\n<start_of_turn>model\n",
        'user': "<end_of_turn>\n<start_of_turn>user\n${MESSAGE}<end_of_turn>\n<start_of_turn>model\n",
        'bot': "${MESSAGE}",
    },
    
    'bunny': {
        'user': 'USER: ${MESSAGE}\n',
        'bot': 'ASSISTANT: ${MESSAGE}\n', # TODO: does output already end in </s> ?
    },
}

ChatTemplates['llava-v0'] = ChatTemplates['vicuna-v0']
ChatTemplates['llava-v1'] = ChatTemplates['vicuna-v1']

ChatTemplates['llava-llama-2'] = ChatTemplates['llama-2'].copy()
ChatTemplates['llava-llama-2'].update({
    'system_prompt': "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
})

StopTokens = ['</s>', '<|endoftext|>', '<|im_end|>', '<eos>']

for key in ChatTemplates:
    ChatTemplates[key] = AttributeDict(name=key, **ChatTemplates[key])


def ChatTemplate(model):
    """
    Attempt to automatically determine the chat template from the model name/type.
    Either returns one of the ChatTemplate dicts from above, or None if undetermined.
    """
    if not isinstance(model, str):
        model = model.config.name.lower()

    if 'stablelm' in model and 'zephyr' in model:
        chat_template = 'stablelm-zephyr'
    elif 'obsidian-3b' in model:
        chat_template = 'nous-obsidian'
    elif 'phi' in model:
        chat_template = 'phi-2-instruct'
    elif 'gemma' in model:
        chat_template = 'gemma'
    elif 'tinyllama' in model:
        chat_template = 'tiny-llama'
    elif 'sheared-llama' in model:
        chat_template = 'sheared-llama'
    elif 'open_llama' in model:
        chat_template = 'open-llama'
    elif 'vila' in model:
        chat_template = 'vicuna-v1'
    elif 'llama-2' in model:
        if 'llava' in model:
            chat_template = 'llava-llama-2'
        else:
            chat_template = 'llama-2'
    elif 'vicuna' in model:
        if 'v1' in model:
            chat_template = 'vicuna-v1'
        else:
            chat_template = 'vicuna-v0'
    elif 'llava' in model:
        if 'v1' in model:
            chat_template = 'llava-v1'
        else:
            chat_template = 'llava-v0'
    else:
        return None
        
    return AttributeDict(ChatTemplates[chat_template])  # return a copy in case user edits it
    