#!/usr/bin/env python3
import os
import inspect
import numpy as np

from .clip import CLIPModel
from .utils import AttrDict, replace_text, print_table

chat_templates = {
    'llama-2': {
        'system_prompt': "Answer the questions.",
        'system': '[INST] <<SYS>>\n${SYSTEM_PROMPT}\n<</SYS>>\n\n',
        'first': '${MESSAGE} [/INST]',
        'user': '<s>[INST] ${MESSAGE} [/INST]',
        'bot': ' ${MESSAGE} </s>'
        #'turn': '${USER_MESSAGE} [/INST] ${BOT_MESSAGE} </s><s>[INST] ', 
    },
    
    'llava-2': {
        'system_prompt': "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
        'system': '[INST] <<SYS>>\n${SYSTEM_PROMPT}\n<</SYS>>\n\n',
        'first': '${MESSAGE} [/INST]',
        'user': '<s>[INST] ${MESSAGE} [/INST]',
        'bot': ' ${MESSAGE} </s>'
    }  
}

for key in chat_templates:
    chat_templates[key] = AttrDict(chat_templates[key])
    

class ChatHistory():
    """
    Unstructured chat that can be a mix of images/text from multiple roles.
    
    It can be indexed like a list of chat entries, where each entry contains
    different media like text, images or video, sound, ect.  
    
    Each of these types has a different embedding function (e.g. for converting 
    text to tokens and it's word embedding, or extracting image features with CLIP
    and performing their embedding). From these, it assembles the embedding for
    the entire chat as input to the LLM.
    
    It uses templating to add the required special tokens as defined by different
    model architectures.  In typicaly 2-turn chat, there are 'user' and 'bot' roles
    defined, but arbitrary roles can be added, each with their own template.
    The system prompt can also be configured through the chat template.
    """
    def __init__(self, model, template='llama-2'):
        self.model = model
        self.template = chat_templates[template]
        self.template_name = template
        self.entries = []
        self.entry_handlers = {}
        
        self.register_embedding('text', self.embed_text)
        self.register_embedding('image', self.embed_image)
        
    def add_entry(self, role='user', text=None, image=None):
        self.entries.append(self.create_entry(role, text, image))
        return self.entries[-1]

    def create_entry(self, role='user', text=None, image=None):
        entry = AttrDict(role=role)

        if text is not None:
            entry.text = text
            
        if image is not None:
            entry.image = image
            
        return entry
        
    def __getitem__(self, entry):
        return self.entries[entry]
    
    def embed_text(self, text, template):
        """
        Get the text embedding after applying the template for 'user', 'bot', ect.
        """
        return self.model.embed_text(
            replace_text(role_template, {'${MESSAGE}': entry.text})
        )
    
    def embed_image(self, image):
        """
        Given an image, extract features and perfom the image embedding.
        This uses the CLIP encoder and a linear projection layer that
        maps it into the embedding space the model expects.
        """
        clip = CLIPModel.from_pretrained()
        embedding = clip.embed_image(image)
        print_table(clip.stats)
        
    def embed_chat(self):
        """
        Assemble the embedding of the entire chat.
        """
        system_prompt = replace_text(
            self.template['system'],
            {'${SYSTEM_PROMPT}': self.template.system_prompt}
        )
        
        print(f"system_prompt:\n```\n{system_prompt}```")
        embeddings = [self.model.embed_text(system_prompt, use_cache=True)]
        print('system_embedding', system_embedding.shape, system_embedding.dtype)
        
        # add all the chat entries
        for i, entry in enumerate(self.entries):
            for key in entry.copy():
                if key in ('role'):
                    continue
                    
                if (i == 0) and 'first' in self.template:
                    role_template = self.template['first']
                else:
                    if entry.role not in self.template:
                        raise RuntimeError(f"chat template {self.template_name} didn't have an entry for role={entry.role}")
                    role_template = self.template[entry.role]

                if '_embedding' not in key and key + '_embedding' not in entry:
                    if self.embedding_functions[key].uses_template:
                        embedding = self.embedding_functions[key].func(entry[key], role)
                    else:
                        embedding = self.embedding_functions[key].func(entry[key])
                    entry[key + '_embedding'] = embedding  # cache the embedding
                elif isinstance(entry[key], np.ndarray):
                    embedding = entry[key]
                else:
                    embedding = entry[key + '_embedding']
                    
                embeddings.append(embedding)

        return np.concatenate(embeddings, axis=1)

    def register_embedding(self, type, func):
        params = inspect.signature(func).parameters
        self.entry_handlers[type] = AttrDict(
            func=func,
            uses_template='role_template' in params
        )
        