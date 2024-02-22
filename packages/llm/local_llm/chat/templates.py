#!/usr/bin/env python3
from ..utils import AttributeDict


ChatTemplates = {
    # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    'llama-2': {
        'system_prompt': "Answer the questions.",
        'system': '<s>[INST] <<SYS>>\n${MESSAGE}\n<</SYS>>\n\n',
        'first': '${MESSAGE} [/INST]',
        'user': '<s>[INST] ${MESSAGE} [/INST]',
        'bot': ' ${MESSAGE}'  # llama-2 output already ends in </s>
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
        'bot': 'ASSISTANT: ${MESSAGE}</s>\n', # TODO: does output already end in </s> ?
    },
    
    # https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/ai-services/openai/includes/chat-markup-language.md#working-with-chat-markup-language-chatml
    'chat-ml': {
        'system_prompt': "You are a helpful AI assistant.",
        'system': "<|im_start|>system\n${MESSAGE}<|im_end|>\n",
        'user': "<|im_start|>user\n${MESSAGE}<|im_end|>\n",
        'bot': "<|im_start|>user\n${MESSAGE}<|im_end|>\n",
    }
}

ChatTemplates['llava-v0'] = ChatTemplates['vicuna-v0']
ChatTemplates['llava-v1'] = ChatTemplates['vicuna-v1']

ChatTemplates['llava-llama-2'] = ChatTemplates['llama-2'].copy()
ChatTemplates['llava-llama-2'].update({
    'system_prompt': "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
})

for key in ChatTemplates:
    ChatTemplates[key] = AttributeDict(name=key, **ChatTemplates[key])


def ChatTemplate(model):
    """
    Attempt to automatically determine the chat template from the model name/type.
    Either returns one of the ChatTemplate dicts from above, or None if undetermined.
    """
    if not isinstance(model, str):
        model = model.config.name.lower()
    
    if 'llama-2' in model:
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
    elif 'zephyr' in model:
        chat_template = 'chat-ml'
    else:
        return None
        
    return AttributeDict(ChatTemplates[chat_template])  # return a copy in case user edits it
    