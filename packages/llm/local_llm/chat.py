#!/usr/bin/env python3
import os
import inspect
import numpy as np

from .clip import CLIPModel
from .utils import AttrDict, replace_text, print_table, ImageExtensions

chat_templates = {
    'llama-2': {
        'system_prompt': "Answer the questions.",
        'system': '<s>[INST] <<SYS>>\n${MESSAGE}\n<</SYS>>\n\n',
        'first': '${MESSAGE} [/INST]',
        'user': '<s>[INST] ${MESSAGE} [/INST]',
        'bot': ' ${MESSAGE}'
        #'turn': '${USER_MESSAGE} [/INST] ${BOT_MESSAGE} </s><s>[INST] ', 
    },
    
    'llava-2': {
        'system_prompt': "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
        'system': '<s>[INST] <<SYS>>\n${MESSAGE}\n<</SYS>>\n\n',
        'first': '${MESSAGE} [/INST]',
        'user': '<s>[INST] ${MESSAGE} [/INST]',
        'bot': ' ${MESSAGE}'
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
        self.embedding_functions = {}
        
        self.register_embedding('text', self.embed_text)
        self.register_embedding('dict', self.embed_dict)
        self.register_embedding('image', self.embed_image)
    
        self.reset()
 
    def add_entry(self, role='user', text=None, image=None, **kwargs):
        """
        Add a chat entry consisting of text, image, ect.  Other inputs
        can be specified in kwargs that have registered embedding types.
        Role is the turn template to apply, typically 'user' or 'bot'.
        """
        self.entries.append(self.create_entry(role, text, image, **kwargs))
        return self.entries[-1]

    def create_entry(self, role='user', text=None, image=None, **kwargs):
        """
        Create a chat entry consisting of text, image, ect.  Other inputs
        can be specified in kwargs that have registered embedding types.
        Role is the turn template to apply, typically 'user' or 'bot'.
        """    
        return AttrDict(role=role, text=text, image=image, **kwargs)

    def reset(self, add_system_prompt=True):
        """
        Reset the chat history, and optionally add the system prompt to the new chat.
        """
        self.entries = []
        self.kv_cache = None
        if add_system_prompt:
            self.add_entry(role='system', text=self.template['system_prompt'])
        
    def __getitem__(self, entry):
        return self.entries[entry]
    
    def embed(self, input, type=None, template=None):
        """
        Get the embedding for a general input (text, image, ect)
        The embedding type will attempt to be determined automatically
        if it isn't explicitly specified. Paths that end in image extensions
        are assumed to be an image, otherwise strings are treated as text.
        """
        if not type:
            type = self.embedding_type(input)
            
        if type not in self.embedding_functions:
            raise ValueError(f"type '{type}' did not have an embedding registered")
        
        print('embedding_type', type, input)
        
        if self.embedding_functions[type].uses_template:
            return self.embedding_functions[type].func(input, template)
        else:
            return self.embedding_functions[type].func(input)
        
    def embed_text(self, text, template):
        """
        Get the text embedding after applying the template for 'user', 'bot', ect.
        """
        text = replace_text(template, {'${MESSAGE}': text})
        print(f"```{text}```")
        return self.model.embed_text(text)
    
    def embed_dict(self, dict, template):
        """
        Get the embedding of a chat entry dict that can contain multiple embedding types.
        """
        embeddings = []
        
        for key, value in dict.items():
            if value is None:
                continue
            if key in self.embedding_functions:
                embeddings.append(self.embed(value, type=key, template=template))
                
        if len(embeddings) == 0:
            raise ValueError("dict did not contain any entries with valid embedding types")
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            return np.concatenate(embeddings, axis=1)
                
    def embed_image(self, image):
        """
        Given an image, extract features and perfom the image embedding.
        This uses the CLIP encoder and a linear projection layer that
        maps it into the embedding space the model expects.
        """
        clip = CLIPModel.from_pretrained()
        embedding = clip.embed_image(image)
        print_table(clip.stats)
        
    def embed_chat(self, use_cache=True):
        """
        Assemble the embedding of the entire chat.
        If cached, only the new embeddings will be returned.
        Otherwise, the entire chat history will be returned.
        Returns the embedding and its token position in the history.
        
        TODO image templating in user input
        """
        embeddings = []
        position = 0
        
        for i, entry in enumerate(self.entries):
            for key in entry.copy():
                if key not in self.embedding_functions or entry[key] is None:
                    continue

                if 'first' in self.template and i == (1 if self.entries[0].role == 'system' else 0):
                    role_template = self.template['first']  # if the first non-system message has a different template
                else:
                    if entry.role not in self.template:
                        raise RuntimeError(f"chat template {self.template_name} didn't have an entry for role={entry.role}")
                    role_template = self.template[entry.role]

                embed_key = key + '_embedding'

                if use_cache:
                    if embed_key not in entry:
                        entry[embed_key] = self.embed(entry[key], type=key, template=role_template)
                        if entry.role != 'bot':  # bot outputs are already included in kv_cache
                            embeddings.append(entry[embed_key])
                            use_cache = False  # all entries after this need to be included
                        else:
                            position += entry[embed_key].shape[1]
                    else:
                        position += entry[embed_key].shape[1]
                else:
                    if embed_key not in entry:
                        entry[embed_key] = self.embed(entry[key], type=key, template=role_template)
                    embeddings.append(entry[embed_key])
                
                """
                if embed_key not in entry:
                    embedding = self.embed(entry, type=key, template=role_template)
                    entry[embed_key] = embedding
                    if use_cache and entry.role != 'bot': # bot outputs are already included in kv_cache
                        embeddings.append(embedding)
                        use_cache = False
                elif use_cache:
                    position += entry[embed_key].shape[1]
                else:
                    embeddings.append(entry[embed_key])
                """
                
        return np.concatenate(embeddings, axis=1), position

    def register_embedding(self, type, func):
        params = inspect.signature(func).parameters
        self.embedding_functions[type] = AttrDict(
            func=func,
            uses_template=len(params) > 1 #'role_template' in params
        )
        
    def embedding_type(self, input):
        if isinstance(input, str):
            ext = os.path.splitext(input)[1].lower()
            if ext in ImageExtensions:
                return 'image'
            elif len(ext) > 0:
                raise ValueError(f"-- file {str} has unsupported extension for embeddings")
            else:
                return "text" 
        elif isinstance(input, PIL.Image.Image):
            return 'image'
        elif isinstance(input, list) and len(input) > 0 and isinstance(input[0], str):
            return 'text'
        else:
            raise ValueError(f"couldn't find type of embedding for {type(input)}, please specify the 'type' argument")
            